from __future__ import annotations

import numpy as np
import torch

FIELDS = [("state", np.float32), ("action", np.int64), ("reward", np.float32), ("next_state", np.float32), ("terminated", bool)]


def _columns(capacity, obs_size):
    return {key: np.empty((capacity, obs_size) if key in ("state", "next_state") else capacity, dtype=dtype) for key, dtype in FIELDS}


def _batch(data, indices, device):
    return {key: torch.from_numpy(np.asarray(values[indices])).to(device) for key, values in data.items()}


class Replay:
    """Fixed-size ring buffer holding the transitions of the task being learned right now."""

    def __init__(self, capacity, obs_size):
        self.capacity, self.obs_size = capacity, obs_size
        self.data = _columns(capacity, obs_size)
        self.size, self.total = 0, 0

    def add(self, state, action, reward, next_state, terminated):
        slot = self.total % self.capacity
        for key, value in zip(self.data, (state, action, reward, next_state, terminated)):
            self.data[key][slot] = value
        self.total += 1
        self.size = min(self.total, self.capacity)

    def sample(self, rng, count, device="cpu"):
        indices = rng.integers(0, self.size, size=count)
        return _batch(self.data, indices, device)

    def clear(self):
        """Every arm starts a task with an empty current buffer, so persistent replay is the only difference."""
        self.size, self.total = 0, 0


class PersistentMemory:
    """Bounded cross-task memory: an equal share of every task seen so far.

    Filled by reservoir sampling *during* each task rather than snapshotted at the
    boundary, so the retained states cover the whole learning trajectory and not just
    the narrow tube visited by the near-greedy policy at the end.
    """

    def __init__(self, capacity, obs_size):
        self.capacity, self.obs_size = capacity, obs_size
        self.tasks = {}

    @property
    def size(self):
        return sum(task["size"] for task in self.tasks.values())

    def past_size(self, current):
        """Only earlier tasks count as rehearsal; the current one is already in the FIFO."""
        return sum(task["size"] for name, task in self.tasks.items() if name != current)

    def budget(self, extra=0):
        return max(1, self.capacity // max(1, len(self.tasks) + extra))

    def begin(self, rng, name):
        """Open a slot for a new task and shrink the older ones to make room."""
        if name not in self.tasks:
            self.tasks[name] = {"data": _columns(self.capacity, self.obs_size), "size": 0, "seen": 0}
            self.rebalance(rng)

    def absorb(self, rng, name, state, action, reward, next_state, terminated):
        task = self.tasks[name]
        limit = self.budget()
        record = (state, action, reward, next_state, terminated)
        if task["size"] < limit:
            slot = task["size"]
            task["size"] += 1
        else:
            # Reservoir sampling keeps a uniform sample of everything the task ever saw.
            slot = int(rng.integers(0, task["seen"] + 1))
            if slot >= limit:
                task["seen"] += 1
                return
        for key, value in zip(task["data"], record):
            task["data"][key][slot] = value
        task["seen"] += 1

    def rebalance(self, rng):
        limit = self.budget()
        for task in self.tasks.values():
            if task["size"] > limit:
                keep = rng.choice(task["size"], size=limit, replace=False)
                for key, values in task["data"].items():
                    values[:limit] = values[keep]
                task["size"] = limit

    def sample(self, rng, count, device="cpu", exclude=None):
        """Uniform over tasks first, so an old task is not crowded out by a newer one."""
        names = [name for name, task in self.tasks.items() if task["size"] > 0 and name != exclude]
        picks = rng.integers(0, len(names), size=count)
        chunks = []
        for index, name in enumerate(names):
            wanted = int((picks == index).sum())
            if wanted:
                task = self.tasks[name]
                chunks.append(_batch(task["data"], rng.integers(0, task["size"], size=wanted), device))
        return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
