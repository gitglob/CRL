from __future__ import annotations

import numpy as np


def area_under_curve(curve, block):
    """Riemann average of one block's greedy evaluations, warmup included: FT shows there."""
    points = [row["normalized"] for row in curve if row["block"] == block]
    return float(np.mean(points)) if points else None


def final_row(matrix):
    return matrix[-1]["evaluations"]


def average_performance(matrix, order):
    return float(np.mean([final_row(matrix)[task]["normalized"] for task in order]))


def forgetting(matrix, order):
    """F_i drops from the score just after learning task i to the final one; F_{N-1} is always 0."""
    after = {}
    for row in matrix:
        # The multitask arm's single block is not an evaluated task, so forgetting is undefined.
        if row["block"] is not None and row["task"] in row["evaluations"]:
            after[row["task"]] = row["evaluations"][row["task"]]["normalized"]
    final = final_row(matrix)
    per_task = {task: after[task] - final[task]["normalized"] for task in order if task in after}
    values = [per_task[task] for task in order if task in per_task]
    return {"per_task": per_task, "mean": float(np.mean(values)) if values else None, "mean_excluding_last": float(np.mean(values[:-1])) if len(values) > 1 else None}


def forward_transfer(auc, scratch_auc, order):
    """FT_i against a fresh agent's AUC; the raw difference rides along since 1 - AUC varies."""
    per_task, deltas = {}, {}
    for task in order:
        mine, base = auc.get(task), scratch_auc.get(task)
        if mine is None or base is None:
            continue
        deltas[task] = mine - base
        per_task[task] = (mine - base) / (1 - base) if base < 1 else None
    values = [v for task in order for v in [per_task.get(task)] if v is not None]
    later = [per_task[task] for task in order[1:] if per_task.get(task) is not None]
    return {"per_task": per_task, "delta_auc": deltas, "mean": float(np.mean(values)) if values else None, "mean_excluding_first": float(np.mean(later)) if later else None}


def zero_shot(jumpstart, order):
    """Score on a task the instant before learning it; task 1 is excluded as nothing precedes it."""
    scores = {}
    for row in jumpstart:
        scores.setdefault(row["task"], row["normalized"])
    values = [scores[task] for task in order[1:] if task in scores]
    return {"per_task": scores, "mean_excluding_first": float(np.mean(values)) if values else None}


def bootstrap(values, resamples=10000, level=0.95, seed=11):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "low": None, "high": None, "n": 0}
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return {"mean": float(array[0]), "low": float(array[0]), "high": float(array[0]), "n": 1}
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    tail = (1 - level) / 2
    return {"mean": float(array.mean()), "low": float(np.quantile(draws, tail)), "high": float(np.quantile(draws, 1 - tail)), "n": len(array)}


def summarize(run, scratch_auc, order):
    """Every headline number for one run, given the shared scratch baseline."""
    matrix = run["matrix"]
    return {
        "average_performance": average_performance(matrix, order),
        "forgetting": forgetting(matrix, order),
        "forward_transfer": forward_transfer(run["auc"], scratch_auc, order),
        "zero_shot": zero_shot(run["jumpstart"], order),
    }
