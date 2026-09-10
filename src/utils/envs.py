from __future__ import annotations

import gymnasium as gym

from .config import PHYSICS
from .io import fingerprint


def make_task(config, name, render_mode=None):
    """The only way to build an environment. Wrappers do not forward attribute writes."""
    spec = config["tasks"][name]
    # disable_env_checker: PassiveEnvChecker costs per-step time we pay ten million times.
    env = gym.make("CartPole-v1", max_episode_steps=config["env"]["max_steps"], render_mode=render_mode, disable_env_checker=True)
    core = env.unwrapped
    for key in PHYSICS:
        if not hasattr(core, key):
            raise RuntimeError(f"gymnasium CartPoleEnv has no attribute {key}; the physics override would be silent")
        setattr(core, key, float(spec[key]))
    # Both are cached in CartPoleEnv.__init__, so setting length or the masses alone lies.
    core.total_mass = core.masspole + core.masscart
    core.polemass_length = core.masspole * core.length
    if getattr(core, "sutton_barto_reward", False):
        raise RuntimeError("CartPole must use the +1 per step reward; normalization assumes it")
    verify_task(env, config, name)
    return env


def make_tasks(config, name, count):
    """One env per slot; SyncVectorEnv's autoreset would corrupt the terminating transition."""
    return [make_task(config, name) for _ in range(count)]


def verify_task(env, config, name):
    """Cheap enough to call inside evaluation loops, and the failure it catches is silent."""
    spec = config["tasks"][name]
    core = env.unwrapped
    for key in PHYSICS:
        if getattr(core, key) != float(spec[key]):
            raise RuntimeError(f"Task {name}: {key} is {getattr(core, key)}, expected {spec[key]}")
    if core.total_mass != core.masspole + core.masscart or core.polemass_length != core.masspole * core.length:
        raise RuntimeError(f"Task {name}: derived masses were not recomputed")


def observation_size(config):
    env = make_task(config, config["benchmark"]["order"][0])
    size = int(env.observation_space.shape[0])
    env.close()
    return size


def action_count(config):
    env = make_task(config, config["benchmark"]["order"][0])
    count = int(env.action_space.n)
    env.close()
    return count


def task_fingerprint(config):
    return fingerprint({name: {key: float(config["tasks"][name][key]) for key in PHYSICS} for name in config["benchmark"]["order"]})
