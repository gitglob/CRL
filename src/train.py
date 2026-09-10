from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .utils.config import parser, resolve, run_directory, task_order
from .utils.envs import action_count, make_task, observation_size, task_fingerprint
from .utils.evaluate import evaluate, evaluate_all
from .utils.io import WandbLogger, fingerprint, save_config, save_json, versions
from .utils.metrics import area_under_curve

STREAMS = ("env", "exploration", "replay_sampling", "cbp_reinit", "weight_init", "memory")


def streams(seed):
    """Independent generators per purpose, so adding CBP cannot shift the env or exploration draws."""
    return dict(zip(STREAMS, np.random.SeedSequence(seed).spawn(len(STREAMS))))


def blocks_for(config, arm, tasks):
    if arm == "scratch":
        return list(tasks)
    if arm == "multitask":
        return ["multitask"]
    return task_order(config)


def run_signature(config, arm, seed, blocks):
    return fingerprint({"arm": arm, "seed": seed, "blocks": blocks, "physics": task_fingerprint(config),
                        "train": config["train"], "replay": config["replay"], "cbp": config["cbp"],
                        "eval": config["eval"], "env": config["env"], "order": config["benchmark"]["order"]})


def train(config, arm, seed, out, tasks=None, overwrite=False):
    from .dqn import DQNAgent, epsilon

    out = Path(out)
    order = config["benchmark"]["order"]
    blocks = blocks_for(config, arm, tasks or order)
    signature = run_signature(config, arm, seed, blocks)
    if (out / "metrics.json").exists() and not overwrite:
        metrics = json.loads((out / "metrics.json").read_text())
        if metrics["signature"] != signature or not (out / "model.pt").exists():
            raise ValueError(f"Existing run incompatible or incomplete: {out}; use --overwrite")
        print(f"[train] reuse {arm}/{'-'.join(blocks)}/seed{seed}", flush=True)
        return metrics
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").unlink(missing_ok=True)
    save_config(out / "config.yaml", config)

    settings = config["train"]
    per_block = settings["timesteps"] * (len(order) if arm == "multitask" else 1)
    keys = streams(seed)
    agent = DQNAgent(config, arm, seed, observation_size(config), action_count(config), keys)
    logger = WandbLogger(config, f"{arm}-seed{seed}" + (f"-{blocks[0]}" if arm == "scratch" else ""))

    matrix, curve, jumpstart, diagnostics, episodes = [], [], [], [], []
    total_steps, eval_steps, train_s, eval_s = 0, 0, 0.0, 0.0
    matrix_episodes = config["eval"]["matrix_episodes"]

    timer = time.perf_counter()
    initial = evaluate_all(agent, config, matrix_episodes)
    eval_s += time.perf_counter() - timer
    eval_steps += sum(report["eval_env_steps"] for report in initial.values())
    matrix.append({"block": None, "task": None, "env_steps": 0, "evaluations": initial})

    for block, name in enumerate(blocks):
        pool = order if arm == "multitask" else [name]
        envs = {}
        for task in pool:
            env = make_task(config, task)
            # Seed once; re-seeding every episode would replay one identical initial state.
            env.reset(seed=int(keys["env"].generate_state(1)[0]))
            envs[task] = env
        agent.begin_task(name)

        timer = time.perf_counter()
        report = evaluate(agent, config, pool[0] if arm != "multitask" else order[0], matrix_episodes) if arm != "multitask" else None
        eval_s += time.perf_counter() - timer
        if report is not None:
            eval_steps += report["eval_env_steps"]
            jumpstart.append({"block": block, "task": name, "env_steps": total_steps, "normalized": report["normalized"], "return_mean": report["return_mean"]})

        def measure(block_steps):
            nonlocal eval_s, eval_steps
            timer = time.perf_counter()
            if arm == "multitask":
                reports = evaluate_all(agent, config, config["eval"]["episodes"])
                normalized = float(np.mean([r["normalized"] for r in reports.values()]))
                measured = {"normalized": normalized, "return_mean": float(np.mean([r["return_mean"] for r in reports.values()])), "eval_env_steps": sum(r["eval_env_steps"] for r in reports.values())}
            else:
                measured = evaluate(agent, config, name, config["eval"]["episodes"])
            eval_s += time.perf_counter() - timer
            eval_steps += measured["eval_env_steps"]
            statistics = agent.statistics()
            curve.append({"block": block, "task": name, "env_steps": total_steps, "block_steps": block_steps, "normalized": measured["normalized"], "return_mean": measured["return_mean"]})
            diagnostics.append({"block": block, "env_steps": total_steps, **statistics})
            save_json(out / "progress.json", {"env_steps": total_steps, "block": block, "task": name, "budget": per_block * len(blocks), **{k: measured[k] for k in ("normalized", "return_mean")}, **statistics})
            logger.log({"env_steps": total_steps, "normalized": measured["normalized"], "return_mean": measured["return_mean"], **statistics})
            print(f"[train {arm}/{name}/seed{seed}] {block_steps:,}/{per_block:,} normalized={measured['normalized']:.2f}", flush=True)

        rng = np.random.default_rng(keys["env"])
        task = pool[0] if arm != "multitask" else pool[int(rng.integers(0, len(pool)))]
        state, _ = envs[task].reset()
        episode_return, episode_steps = 0.0, 0
        block_steps, next_eval = 0, 0
        timer = time.perf_counter()
        while block_steps < per_block:
            if block_steps >= next_eval:
                train_s += time.perf_counter() - timer
                measure(block_steps)
                next_eval = (block_steps // config["eval"]["every"] + 1) * config["eval"]["every"]
                timer = time.perf_counter()
            action = agent.act(state, epsilon(settings, block_steps, per_block))
            next_state, reward, terminated, truncated, _ = envs[task].step(action)
            agent.observe(state, action, float(reward), next_state, terminated)
            episode_return += float(reward)
            episode_steps += 1
            block_steps += 1
            total_steps += 1
            state = next_state
            if agent.ready() and block_steps % settings["train_every"] == 0:
                agent.optimize()
            if terminated or truncated:
                episodes.append({"env_steps": total_steps, "task": task, "return": episode_return})
                task = pool[0] if arm != "multitask" else pool[int(rng.integers(0, len(pool)))]
                state, _ = envs[task].reset()
                episode_return, episode_steps = 0.0, 0
        train_s += time.perf_counter() - timer
        measure(per_block)

        timer = time.perf_counter()
        evaluations = evaluate_all(agent, config, matrix_episodes)
        eval_s += time.perf_counter() - timer
        eval_steps += sum(report["eval_env_steps"] for report in evaluations.values())
        matrix.append({"block": block, "task": name, "env_steps": total_steps, "evaluations": evaluations})
        for env in envs.values():
            env.close()

    auc = {}
    for block, name in enumerate(blocks):
        if name not in auc:
            auc[name] = area_under_curve(curve, block)

    agent.save(out / "model.pt", {"signature": signature, "arm": arm, "seed": seed, "blocks": blocks})
    save_json(out / "checkpoint.json", {"signature": signature, "arm": arm, "seed": seed, "blocks": blocks, "model": "model.pt"})
    metrics = {"signature": signature, "arm": arm, "seed": seed, "blocks": blocks, "order": order,
               "physics": task_fingerprint(config), "budget_per_block": per_block, "env_steps": total_steps,
               "eval_env_steps": eval_steps, "updates": agent.updates, "train_seconds": train_s, "eval_seconds": eval_s,
               "matrix": matrix, "learning_curve": curve, "jumpstart": jumpstart, "auc": auc,
               "plasticity": diagnostics, "train_episodes": episodes, "versions": versions()}
    save_json(out / "metrics.json", metrics)
    return metrics


def main():
    args = parser("Train one DQN over a CartPole task sequence").parse_args()
    config = resolve(args)
    arm = config["benchmark"]["arms"][0]
    seed = config["benchmark"]["seeds"][0]
    tasks = [args.task] if args.task else None
    root = args.out or config["output"]["root"]
    out = run_directory(root, arm, seed, args.task if arm == "scratch" else None)
    train(config, arm, seed, out, tasks, args.overwrite)


if __name__ == "__main__":
    main()
