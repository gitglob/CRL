import zlib

import gymnasium as gym
import numpy as np

MINATAR_TASKS = ("breakout", "space_invaders", "freeway")
MINATAR_PROBES = ("asterix",)


def task_names(suite, probes=False):
    if suite != "minatar":
        raise ValueError(f"Unknown suite: {suite}")
    return list(MINATAR_PROBES if probes else MINATAR_TASKS)


def seed_for(task, purpose=0):
    return int(np.random.SeedSequence([0, zlib.crc32(task.encode()), purpose]).generate_state(1)[0])


class PaddedMinAtar(gym.Wrapper):
    """All games expose the full six-action set and ten padded binary channels."""

    def __init__(self, task, render_mode=None):
        import minatar.gym

        core = minatar.gym.BaseEnv(task, render_mode=render_mode, use_minimal_action_set=False)
        super().__init__(gym.wrappers.TimeLimit(core, max_episode_steps=2500))
        self.observation_space = gym.spaces.Box(0, 1, shape=(1000,), dtype=np.uint8)
        if self.action_space.n != 6 or self.env.observation_space.shape[-1] > 10:
            raise RuntimeError("Unexpected MinAtar interface")

    def observation(self, state):
        output = np.zeros((10, 10, 10), dtype=np.uint8)
        output[:, :, :state.shape[-1]] = state
        return output.flatten()

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return self.observation(state), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(int(action))
        return self.observation(state), reward, terminated, truncated, info


def make_env(suite, task, render_mode=None):
    if suite != "minatar":
        raise ValueError(f"Unknown suite: {suite}")
    return PaddedMinAtar(task, render_mode)


def dimensions(suite):
    return (1000, 6)


def fixed_observations(suite, count):
    rng = np.random.default_rng(829)
    observations = []
    tasks = task_names(suite)
    for index, task in enumerate(tasks):
        env = make_env(suite, task)
        state, _ = env.reset(seed=seed_for(task, 829))
        for _ in range(count // len(tasks) + (index < count % len(tasks))):
            observations.append(state.copy())
            state, _, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            if terminated or truncated:
                state, _ = env.reset()
        env.close()
    return np.stack(observations)
