import math
import time
from pathlib import Path

import numpy as np
import torch

from .io import append_csv, save_config, save_json
from .envs import fixed_observations, make_env, seed_for, task_names
from ..clear.learner import ActorCritic

EPISODE_COLUMNS = ("episode", "block", "task", "start_env_steps", "env_steps", "return")
EVALUATION_COLUMNS = ("block", "scope", "env_steps", "train_episodes", "block_task", "task", "episode", "return", "eval_steps")
BLOCK_COLUMNS = ("block", "task", "start_env_steps", "env_steps", "seconds", "auc")
PROBE_COLUMNS = ("phase", "task", "scope", "env_steps", "episode", "return")
PROBE_SUMMARY_COLUMNS = ("phase", "task", "status", "auc", "train_env_steps")
RUN_FIELDS = ("schema", "arm", "suite", "seed", "blocks", "completed_blocks", "env_steps", "attempted_env_steps", "planned_steps", "status", "error", "seconds", "updates", "transitions_per_second", "gpu_peak_bytes", "probe_checkpoint_blocks")


class BudgetExpired(RuntimeError):
    pass


def evaluation_rows(block, scope, env_steps, train_episodes, block_task, scores):
    """One row per evaluation episode: the group mean is a groupby away, so it is not stored."""
    return [{"block": block, "scope": scope, "env_steps": env_steps, "train_episodes": train_episodes, "block_task": block_task, "task": task, "episode": index, "return": value, "eval_steps": score["env_steps"]}
            for task, score in scores.items() for index, value in enumerate(score["episodes"], start=1)]


def probe_rows(phase, results):
    rows, summary = [], []
    for task, payload in results.items():
        summary.append({"phase": phase, "task": task, "status": payload["status"], "auc": payload["auc"], "train_env_steps": payload["train_env_steps"]})
        for point in payload["curve"]:
            rows.extend({"phase": phase, "task": task, "scope": "eval", "env_steps": point["steps"], "episode": index, "return": value}
                        for index, value in enumerate(point.get("episodes", [point["return"]]), start=1))
        rows.extend({"phase": phase, "task": task, "scope": "train", "env_steps": entry["env_steps"], "episode": number, "return": entry["return"]}
                    for number, entry in enumerate(payload["episode_log"], start=1))
    return rows, summary


def save_probes(out, phase, results):
    rows, summary = probe_rows(phase, results)
    append_csv(out / "probes.csv", PROBE_COLUMNS, rows)
    append_csv(out / "probe_summary.csv", PROBE_SUMMARY_COLUMNS, summary)


def save_run(out, metrics):
    save_json(out / "run.json", {key: metrics[key] for key in RUN_FIELDS if key in metrics})


def check_time(deadline):
    if time.time() >= deadline:
        raise BudgetExpired("Experiment phase deadline reached")


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class Collector:
    def __init__(self, agent, tasks, purpose=17):
        self.agent = agent
        width = agent.config["train"]["num_envs"]
        width = math.lcm(width, len(tasks)) if len(tasks) > 1 else width
        self.tasks = [tasks[i % len(tasks)] for i in range(width)]
        self.envs = [make_env(agent.suite, task) for task in self.tasks]
        self.states = [env.reset(seed=seed_for(task, purpose + i))[0] for i, (env, task) in enumerate(zip(self.envs, self.tasks))]
        self.returns = np.zeros(width)
        self.starts = np.zeros(width, dtype=np.int64)
        self.steps = 0
        self.episodes = []

    def collect(self, deadline):
        check_time(deadline)
        rows = []
        for _ in range(self.agent.config["train"]["unroll_length"]):
            observation = np.stack(self.states)
            actions, logits, values = self.agent.act(observation)
            next_states, rewards, terminals, boundaries = [], [], [], []
            self.steps += len(self.envs)
            for slot, env in enumerate(self.envs):
                state, reward, terminated, truncated, _ = env.step(actions[slot])
                next_states.append(state.copy())
                rewards.append(reward)
                terminals.append(terminated)
                boundaries.append(terminated or truncated)
                self.returns[slot] += reward
                if terminated or truncated:
                    self.episodes.append({"task": self.tasks[slot], "return": float(self.returns[slot]), "start_env_steps": int(self.starts[slot]), "env_steps": self.steps})
                    self.returns[slot] = 0
                    self.starts[slot] = self.steps
                    state, _ = env.reset()
                self.states[slot] = state
            rows.append({"observation": observation, "next_observation": np.stack(next_states), "action": actions.astype(np.int64), "reward": np.asarray(rewards, dtype=np.float32), "terminated": np.asarray(terminals, dtype=bool), "boundary": np.asarray(boundaries, dtype=bool), "behavior_logits": logits, "behavior_value": values})
        return {key: np.stack([row[key] for row in rows]) for key in rows[0]}

    def close(self):
        for env in self.envs:
            env.close()


def learn_rollout(agent, data):
    width = agent.config["train"]["batch_unrolls"]
    for start in range(0, data["action"].shape[1], width):
        agent.optimize({key: value[:, start:start + width] for key, value in data.items()})


@torch.no_grad()
def evaluate(agent, tasks, episodes, deadline=float("inf"), random_policy=False, moment=0):
    scores = {}
    for task in tasks:
        check_time(deadline)
        envs = [make_env(agent.suite, task) for _ in range(episodes)]
        try:
            # Seeds follow the evaluation point: fresh episodes each time, still reproducible.
            states = [env.reset(seed=seed_for(task, 500 + moment + i))[0] for i, env in enumerate(envs)]
            totals = np.zeros(episodes)
            active = np.ones(episodes, dtype=bool)
            steps = 0
            rng = np.random.default_rng(seed_for(task, 77 + moment))
            generator = torch.Generator(device=agent.device).manual_seed(seed_for(task, 77 + moment))
            while active.any():
                check_time(deadline)
                actions = rng.integers(agent.actions, size=episodes) if random_policy else agent.act(states, rng=generator)[0]
                for i, env in enumerate(envs):
                    if not active[i]:
                        continue
                    state, reward, terminated, truncated, _ = env.step(actions[i])
                    states[i] = state
                    totals[i] += reward
                    steps += 1
                    active[i] = not (terminated or truncated)
            scores[task] = {"return": float(totals.mean()), "episodes": totals.tolist(), "env_steps": steps}
        finally:
            for env in envs:
                env.close()
    return scores


def auc(points):
    if len(points) < 2:
        return None
    steps = np.asarray([point["steps"] for point in points])
    if np.any(np.diff(steps) <= 0):
        raise ValueError("AUC step coordinates must be strictly increasing")
    return float(np.trapezoid([point["return"] for point in points], steps) / (steps[-1] - steps[0]))


def probe_tasks(source, deadline, label, results=None):
    results = {} if results is None else results
    config = source.config
    for task in task_names(source.suite, probes=True):
        check_time(deadline)
        agent = source.probe_learner()
        collector = Collector(agent, [task], purpose=313)
        points = []
        budget = config["train"]["probe_steps"]
        interval = max(1, min(config["eval"]["period"], budget))
        next_measure = 0
        complete = False
        try:
            while collector.steps <= budget:
                if collector.steps >= next_measure or collector.steps == budget:
                    report = evaluate(agent, [task], config["eval"]["episodes"], deadline, moment=collector.steps)[task]
                    points.append({"steps": collector.steps, **report})
                    next_measure = collector.steps + interval
                if collector.steps == budget:
                    complete = True
                    break
                learn_rollout(agent, collector.collect(deadline))
            results[task] = {"status": "complete", "curve": points, "auc": auc(points), "train_env_steps": collector.steps, "episode_log": list(collector.episodes)}
        finally:
            collector.close()
            if not complete:
                results[task] = {"status": "incomplete", "curve": points, "auc": None, "train_env_steps": collector.steps, "episode_log": list(collector.episodes)}
        print(f"[probe {source.arm}/{label}/{task}] auc={results[task]['auc']:.2f}", flush=True)
    return results


def run_job(job):
    config, suite, arm = job["config"], job["suite"], job["arm"]
    deadline = job["deadline"]
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(out / "config.yaml", config)
    checkpoints = Path(job.get("checkpoints", out / "checkpoints"))
    agent = ActorCritic(config, suite, arm)
    if str(agent.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    tasks = task_names(suite)
    blocks = job.get("blocks", tasks * config["cycles"])
    observations = fixed_observations(suite, config["eval"]["fixed_observations"])
    metrics = {"schema": 1, "arm": arm, "suite": suite, "seed": 0, "config": config, "blocks": blocks, "completed_blocks": 0, "env_steps": 0, "attempted_env_steps": 0, "matrix": [], "curves": [], "episode_log": [], "diagnostics": [], "probes": {}, "probe_checkpoint_blocks": {}, "status": "running", "planned_steps": len(blocks) * job.get("block_steps", config["train"]["block_steps"])}
    started = time.time()
    model_path = out / "model.pt"
    collector = None
    diagnostic_columns = None
    try:
        check_time(deadline)
        metrics["matrix"].append({"block": -1, "task": None, "steps": 0, "scores": evaluate(agent, tasks, config["eval"]["episodes"], deadline)})
        append_csv(out / "evaluations.csv", EVALUATION_COLUMNS, evaluation_rows(-1, "boundary", 0, 0, None, metrics["matrix"][0]["scores"]))
        save_run(out, metrics)
        atomic_checkpoint(model_path, agent.checkpoint())
        checkpoints.mkdir(parents=True, exist_ok=True)
        atomic_checkpoint(checkpoints / "block_0000.pt", {"weights": agent.weights(), "updates": 0})
        if job.get("probes", False):
            metrics["probe_checkpoint_blocks"]["initial"] = 0
            if job.get("initial_probes"):
                metrics["probes"]["initial"] = job["initial_probes"]
            else:
                probe_tasks(agent, deadline, "initial", metrics["probes"].setdefault("initial", {}))
            save_probes(out, "initial", metrics["probes"]["initial"])
        for index, task in enumerate(blocks):
            check_time(deadline)
            pool = tasks if task == "multitask" else [task]
            steps_before = metrics["env_steps"]
            logged_before = len(metrics["episode_log"])
            collector = Collector(agent, pool)
            quantum = config["train"]["unroll_length"] * len(collector.envs)
            budget = job.get("block_steps", config["train"]["block_steps"])
            budget = max(quantum, budget // quantum * quantum)
            period = config["eval"]["period"]
            initial = metrics["matrix"][-1]["scores"]
            points = [{"steps": 0, "episodes": logged_before, "return": float(np.mean([initial[name]["return"] for name in pool])), "scores": {name: initial[name] for name in pool}}]
            # The grid follows global steps, so every arm is measured on the same absolute schedule.
            next_measure = (steps_before // period + 1) * period
            train_start = time.time()
            while collector.steps < budget:
                data = collector.collect(deadline)
                learn_rollout(agent, data)
                if steps_before + collector.steps >= next_measure and collector.steps < budget:
                    # Every task, not just the trained one: retention must be measured.
                    scores = evaluate(agent, tasks, config["eval"]["episodes"], deadline, moment=steps_before + collector.steps)
                    points.append({"steps": collector.steps, "episodes": logged_before + len(collector.episodes), "return": float(np.mean([scores[name]["return"] for name in pool])), "scores": scores})
                    while next_measure <= steps_before + collector.steps:
                        next_measure += period
            scores = evaluate(agent, tasks, config["eval"]["episodes"], deadline, moment=steps_before + collector.steps)
            points.append({"steps": collector.steps, "episodes": logged_before + len(collector.episodes), "return": float(np.mean([scores[name]["return"] for name in pool])), "scores": {name: scores[name] for name in pool}})
            metrics["env_steps"] += collector.steps
            metrics["attempted_env_steps"] = metrics["env_steps"]
            metrics["completed_blocks"] = index + 1
            metrics["curves"].append({"block": index, "task": task, "points": points, "auc": auc(points), "steps": collector.steps, "seconds": time.time() - train_start})
            metrics["episode_log"].extend({"block": index, "task": entry["task"], "start_env_steps": steps_before + entry["start_env_steps"], "env_steps": steps_before + entry["env_steps"], "return": entry["return"]} for entry in collector.episodes)
            metrics["matrix"].append({"block": index, "task": task, "steps": metrics["env_steps"], "scores": scores})
            metrics["diagnostics"].append({"block": index, "steps": metrics["env_steps"], **agent.diagnostics(observations)})
            collector.close()
            collector = None
            atomic_checkpoint(model_path, agent.checkpoint())
            atomic_checkpoint(checkpoints / f"block_{index + 1:04d}.pt", {"weights": agent.weights(), "updates": agent.updates})
            metrics["seconds"] = time.time() - started
            append_csv(out / "episodes.csv", EPISODE_COLUMNS, [{"episode": number, **entry} for number, entry in enumerate(metrics["episode_log"][logged_before:], start=logged_before + 1)])
            # First and last points repeat neighbouring boundaries, so log interiors only.
            rows = evaluation_rows(index, "boundary", metrics["env_steps"], len(metrics["episode_log"]), task, scores)
            for point in points[1:-1]:
                rows.extend(evaluation_rows(index, "period", steps_before + point["steps"], point["episodes"], task, point["scores"]))
            append_csv(out / "evaluations.csv", EVALUATION_COLUMNS, rows)
            append_csv(out / "blocks.csv", BLOCK_COLUMNS, [{"block": index, "task": task, "start_env_steps": steps_before, "env_steps": metrics["env_steps"], "seconds": time.time() - train_start, "auc": auc(points)}])
            diagnostic_columns = diagnostic_columns or sorted(metrics["diagnostics"][-1])
            append_csv(out / "diagnostics.csv", diagnostic_columns, [metrics["diagnostics"][-1]])
            save_run(out, metrics)
            if not job.get("quiet", False):
                print(f"[train {arm}/{task}] block={index + 1}/{len(blocks)} steps={metrics['env_steps']:,} score={points[-1]['return']:.1f} loss={agent.last.get('loss', 0):.3f}", flush=True)
            if job.get("probes", False) and index + 1 == max(1, len(blocks) // 2):
                metrics["probe_checkpoint_blocks"]["midpoint"] = index + 1
                probe_tasks(agent, deadline, "midpoint", metrics["probes"].setdefault("midpoint", {}))
                save_probes(out, "midpoint", metrics["probes"]["midpoint"])
        if job.get("probes", False):
            metrics["probe_checkpoint_blocks"]["final"] = len(blocks)
            probe_tasks(agent, deadline, "final", metrics["probes"].setdefault("final", {}))
            save_probes(out, "final", metrics["probes"]["final"])
        metrics["status"] = "complete"
    except BudgetExpired:
        metrics["status"] = "budget_exhausted"
        if collector:
            metrics["attempted_env_steps"] = metrics["env_steps"] + collector.steps
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if collector:
            collector.close()
        metrics["seconds"] = time.time() - started
        metrics["updates"] = metrics["diagnostics"][-1]["updates"] if metrics["diagnostics"] else 0
        metrics["transitions_per_second"] = metrics["attempted_env_steps"] / max(metrics["seconds"], 1e-9)
        metrics["gpu_peak_bytes"] = torch.cuda.max_memory_allocated() if str(agent.device).startswith("cuda") else 0
        save_run(out, metrics)
    return {"out": str(out), "status": metrics["status"], "completed_blocks": metrics["completed_blocks"], "seconds": metrics["seconds"], "transitions_per_second": metrics["transitions_per_second"], "gpu_peak_bytes": metrics["gpu_peak_bytes"]}
