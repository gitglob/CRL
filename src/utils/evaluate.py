from __future__ import annotations

import numpy as np
import torch

from .envs import make_tasks, verify_task


def episode_seeds(config, task, count):
    """Common random numbers: every arm and matrix cell replays the same episodes."""
    index = list(config["tasks"]).index(task)
    return [int(s) for s in np.random.SeedSequence([7, index]).generate_state(count)]


@torch.no_grad()
def evaluate(agent, config, task, episodes):
    """Greedy on private envs, drawing no randomness, so any checkpoint re-evaluates identically."""
    seeds = episode_seeds(config, task, episodes)
    width = min(episodes, config["train"]["num_envs"])
    envs = make_tasks(config, task, width)
    for env in envs:
        verify_task(env, config, task)
    records, assigned = [None] * episodes, list(range(width))
    states = [env.reset(seed=seeds[slot])[0] for slot, env in enumerate(envs)]
    totals, lengths = [0.0] * width, [0] * width
    steps, pending, live = 0, width, width
    while live:
        actions = agent.act_batch(states)
        for slot in range(width):
            if assigned[slot] < 0:
                continue
            state, reward, terminated, truncated, _ = envs[slot].step(actions[slot])
            totals[slot] += float(reward)
            lengths[slot] += 1
            steps += 1
            states[slot] = state
            if not (terminated or truncated):
                continue
            records[assigned[slot]] = {"steps": lengths[slot], "return": totals[slot]}
            if pending < episodes:
                assigned[slot], pending = pending, pending + 1
                states[slot] = envs[slot].reset(seed=seeds[assigned[slot]])[0]
                totals[slot], lengths[slot] = 0.0, 0
            else:
                # Retired, but its last observation still pads the batch to a constant width.
                assigned[slot], live = -1, live - 1
    for env in envs:
        env.close()
    returns = [r["return"] for r in records]
    return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)), "normalized": float(np.mean(returns)) / config["env"]["max_return"], "episode_length": float(np.mean([r["steps"] for r in records])), "eval_env_steps": steps, "episodes": records}


def evaluate_all(agent, config, episodes):
    """One row of the task-performance matrix."""
    return {task: evaluate(agent, config, task, episodes) for task in config["benchmark"]["order"]}
