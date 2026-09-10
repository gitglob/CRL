from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

ARMS = ("scratch", "finetune", "replay", "cbp", "replay_cbp", "multitask")
PHYSICS = ("gravity", "masscart", "masspole", "length", "force_mag")


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path="config/base.yaml", seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Config inheritance cycle")
    config = yaml.safe_load(path.read_text())
    parent = config.pop("inherits", None)
    if parent:
        config = merge(load_config(path.parent / parent, (*seen, path)), config)
    return config


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--seed", type=int, help="Override the seed list with a single seed")
    p.add_argument("--arm", choices=ARMS, help="Restrict the study to one arm")
    p.add_argument("--task", help="Scratch runs only: restrict to one task")
    p.add_argument("--timesteps", type=int, help="Override the per-task training budget")
    p.add_argument("--workers", type=int, default=1, help="Run independent runs in parallel processes")
    p.add_argument("--out", type=Path)
    p.add_argument("--wandb", action="store_true", help="Opt in to W&B; local logging always remains enabled")
    p.add_argument("--overwrite", action="store_true", help="Replace existing artifacts for requested runs")
    return p


def resolve(args):
    c = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        c["benchmark"]["seeds"] = [args.seed]
    if getattr(args, "arm", None) is not None:
        c["benchmark"]["arms"] = [args.arm]
    if getattr(args, "timesteps", None) is not None:
        c["train"]["timesteps"] = args.timesteps
    if getattr(args, "out", None) is not None:
        c["output"]["root"] = str(args.out)
    if getattr(args, "wandb", False):
        c["wandb"]["enabled"] = True
    validate(c)
    if getattr(args, "task", None) is not None and args.task not in c["tasks"]:
        raise ValueError(f"Unknown task: {args.task}")
    return c


def validate(c):
    arms = c["benchmark"]["arms"]
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError(f"benchmark.arms must be distinct names from {list(ARMS)}")
    order = c["benchmark"]["order"]
    if not order or len(set(order)) != len(order) or any(t not in c["tasks"] for t in order):
        raise ValueError("benchmark.order must select distinct known tasks")
    for name in order:
        missing = [key for key in PHYSICS if key not in c["tasks"][name]]
        if missing:
            raise ValueError(f"Task {name} is missing physics parameters: {missing}")
    seeds = c["benchmark"]["seeds"]
    # Exactly one: run directories no longer carry the seed, so a second would overwrite it.
    if len(seeds) != 1:
        raise ValueError("benchmark.seeds must hold exactly one seed")
    if c["benchmark"]["cycles"] < 1:
        raise ValueError("benchmark.cycles must be at least one")
    for section, key in [("train", "timesteps"), ("train", "batch_size"), ("train", "buffer_capacity"), ("train", "train_every"), ("train", "num_envs"), ("env", "max_steps"), ("eval", "every"), ("eval", "episodes"), ("eval", "matrix_episodes"), ("replay", "capacity")]:
        if c[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    # block_steps strides by num_envs, so anything it does not divide drifts off the eval grid.
    if c["eval"]["every"] % c["train"]["num_envs"] or c["train"]["timesteps"] % c["train"]["num_envs"]:
        raise ValueError("train.num_envs must divide both train.timesteps and eval.every")
    if c["train"]["learning_starts"] < 0 or c["train"]["learning_starts"] >= c["train"]["timesteps"]:
        raise ValueError("train.learning_starts must be nonnegative and smaller than the budget")
    if not 0 < c["train"]["gamma"] <= 1:
        raise ValueError("train.gamma must be in (0, 1]")
    if c["train"]["learning_rate"] <= 0 or c["train"]["gradient_clip"] <= 0:
        raise ValueError("Learning rate and gradient clip must be positive")
    if not 0 < c["train"]["target_tau"] <= 1:
        raise ValueError("train.target_tau must be in (0, 1]")
    if c["train"]["l2_lambda"] < 0:
        raise ValueError("train.l2_lambda must be nonnegative")
    if not c["train"]["hidden_sizes"] or any(not isinstance(w, int) or w <= 0 for w in c["train"]["hidden_sizes"]):
        raise ValueError("train.hidden_sizes must be positive integers")
    if not 0 <= c["train"]["epsilon_end"] <= c["train"]["epsilon_start"] <= 1:
        raise ValueError("Invalid epsilon schedule")
    if not 0 < c["train"]["epsilon_decay_fraction"] <= 1:
        raise ValueError("Invalid epsilon decay fraction")
    if c["train"]["device"] not in ("auto", "cuda") and not (c["train"]["device"].startswith("cuda:") and c["train"]["device"][5:].isdigit()):
        raise ValueError("train.device must be auto or a cuda device; the CPU is not an option")
    if not 0 <= c["replay"]["ratio"] < 1:
        raise ValueError("replay.ratio must be in [0, 1)")
    if not 0 <= c["cbp"]["replacement_rate"] < 1:
        raise ValueError("cbp.replacement_rate must be in [0, 1)")
    if not 0 < c["cbp"]["decay_rate"] < 1:
        raise ValueError("cbp.decay_rate must be in (0, 1)")
    if c["cbp"]["maturity_threshold"] < 0:
        raise ValueError("cbp.maturity_threshold must be nonnegative")
    if c["env"]["max_return"] <= 0:
        raise ValueError("env.max_return must be positive")


def task_order(config):
    """The full block sequence, repeated once per cycle."""
    return list(config["benchmark"]["order"]) * config["benchmark"]["cycles"]


def run_directory(root, arm, task=None):
    """One seed per study, so the seed is not part of the path."""
    root = Path(root)
    return root / arm / task if task is not None else root / arm
