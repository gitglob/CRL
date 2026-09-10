from __future__ import annotations

import numpy as np
import torch

from .envs import make_task, verify_task


def episode_seeds(config, task, count):
    """Common random numbers: every arm, seed and matrix cell replays the same episodes.

    Cross-arm differences then become paired comparisons, which is free variance reduction
    on a return distribution as bimodal as CartPole's.
    """
    index = list(config["tasks"]).index(task)
    return [int(s) for s in np.random.SeedSequence([7, index]).generate_state(count)]


@torch.no_grad()
def evaluate(agent, config, task, episodes):
    """Greedy, on a private environment, never touching training state."""
    env = make_task(config, task)
    verify_task(env, config, task)
    records, steps = [], 0
    for seed in episode_seeds(config, task, episodes):
        state, _ = env.reset(seed=seed)
        total, length = 0.0, 0
        while True:
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            total += float(reward)
            length += 1
            if terminated or truncated:
                records.append({"steps": length, "return": total})
                steps += length
                break
    env.close()
    returns = [r["return"] for r in records]
    return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)), "normalized": float(np.mean(returns)) / config["env"]["max_return"], "episode_length": float(np.mean([r["steps"] for r in records])), "eval_env_steps": steps, "episodes": records}


def evaluate_all(agent, config, episodes):
    """One row of the task-performance matrix."""
    return {task: evaluate(agent, config, task, episodes) for task in config["benchmark"]["order"]}
