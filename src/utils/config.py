import argparse
from copy import deepcopy
from pathlib import Path

import yaml


ARMS = ("finetune", "cbp", "replay", "clear", "clear_cbp", "scratch", "multitask")
MAIN_ARMS = ARMS[:5]


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path="config/study.yaml", seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Config inheritance cycle")
    config = yaml.safe_load(path.read_text())
    parent = config.pop("inherits", None)
    if parent:
        config = merge(load_config(path.parent / parent, (*seen, path)), config)
    return config


def validate(config):
    if config["seed"] != 0:
        raise ValueError("The showcase uses exactly one seed: 0")
    if config["suite"] not in ("auto", "cartpole", "minatar"):
        raise ValueError("Unknown suite")
    if config["max_seconds"] <= 0 or config["max_seconds"] > 3600:
        raise ValueError("max_seconds must be in (0, 3600]")
    arms = config.get("arms", MAIN_ARMS)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("arms must be a nonempty list of unique supported arms")
    if any(arm in arms for arm in ("scratch", "multitask")) and len(arms) != 1:
        raise ValueError("Scratch and multitask are standalone references; select each separately")
    train = config["train"]
    if train["unroll_length"] <= 0 or train["batch_unrolls"] < 2 or train["num_envs"] <= 0 or train["num_envs"] % train["batch_unrolls"]:
        raise ValueError("num_envs must be a multiple of batch_unrolls")
    quantum = train["unroll_length"] * train["num_envs"]
    if train["block_steps"] <= 0 or train["probe_steps"] <= 0 or train["block_steps"] % quantum or train["probe_steps"] % quantum:
        raise ValueError("Block and probe budgets must contain complete rollouts")
    if not 0 <= config["clear"]["replay_ratio"] < 1:
        raise ValueError("replay_ratio must be in [0, 1)")
    if config["clear"]["capacity"] < train["unroll_length"]:
        raise ValueError("Replay capacity must hold at least one complete unroll")
    if not 0 < train["gamma"] <= 1 or train["learning_rate"] <= 0:
        raise ValueError("Invalid learning settings")
    if config["cycles"] < 1 or config["eval"]["episodes"] < 1:
        raise ValueError("Cycles and evaluation episodes must be positive")
    if any(config["eval"][key] < 1 for key in ("points_per_block", "fixed_observations", "clip_stride")):
        raise ValueError("Evaluation intervals and observation counts must be positive")
    if not 0 <= config["cbp"]["replacement_rate"] < 1 or config["cbp"]["maturity_threshold"] < 0:
        raise ValueError("Invalid CBP settings")
    if not 0 < config["cbp"]["decay_rate"] < 1:
        raise ValueError("Invalid CBP decay")
    return config


def parser():
    result = argparse.ArgumentParser(description="Run the timed CLEAR and CBP showcase")
    result.add_argument("--config", default="config/study.yaml")
    result.add_argument("--suite", choices=("auto", "cartpole", "minatar"))
    result.add_argument("--max-seconds", type=float)
    result.add_argument("--out", type=Path)
    result.add_argument("--workers", type=int, choices=(1, 2, 4))
    result.add_argument("--arm", choices=ARMS)
    result.add_argument("--overwrite", action="store_true")
    return result


def resolve(args):
    config = deepcopy(load_config(args.config))
    for name in ("suite", "max_seconds", "workers"):
        if getattr(args, name, None) is not None:
            config[name] = getattr(args, name)
    if args.out:
        config["output"] = str(args.out)
    if args.arm:
        config["arms"] = [args.arm]
    return validate(config)


def save_config(path, config):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))
