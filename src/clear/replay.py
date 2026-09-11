import numpy as np


class UnrollReservoir:
    """A task-agnostic uniform reservoir of complete, immutable behavior-policy unrolls."""

    def __init__(self, capacity, unroll_length, seed=0):
        self.capacity = capacity // unroll_length
        self.unroll_length = unroll_length
        self.items = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, batch):
        for column in range(batch["action"].shape[1]):
            self.seen += 1
            slot = len(self.items) if len(self.items) < self.capacity else int(self.rng.integers(self.seen))
            if slot >= self.capacity:
                continue
            item = {key: value[:, column].copy() for key, value in batch.items()}
            if slot == len(self.items):
                self.items.append(item)
            else:
                self.items[slot] = item

    def sample(self, count, rng):
        picks = rng.integers(len(self.items), size=count)
        return {key: np.stack([self.items[i][key] for i in picks], axis=1) for key in self.items[0]}

    def state_dict(self):
        return {"items": self.items, "seen": self.seen, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.items, self.seen = state["items"], state["seen"]
        self.rng.bit_generator.state = state["rng"]


def mix_batch(fresh, memory, ratio, rng):
    count = fresh["action"].shape[1]
    replay_count = int(count * ratio) if memory is not None and memory.items else 0
    if replay_count == 0:
        return fresh, np.zeros(count, dtype=bool)
    selected = rng.choice(count, count - replay_count, replace=False)
    replay = memory.sample(replay_count, rng)
    batch = {key: np.concatenate((value[:, selected], replay[key]), axis=1) for key, value in fresh.items()}
    mask = np.arange(count) >= count - replay_count
    return batch, mask
