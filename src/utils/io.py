from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def versions():
    recorded = {}
    for name in ["gymnasium", "numpy", "PyYAML", "matplotlib", "torch"]:
        try:
            recorded[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            recorded[name] = None
    return {"python": platform.python_version(), **recorded}


def save_config(path, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False))


class WandbLogger:
    """Optional external logging must never stop the local experiment."""

    def __init__(self, config, name):
        self.run = None
        if config["wandb"]["enabled"]:
            try:
                import wandb
                self.run = wandb.init(project=config["wandb"]["project"], name=name, config=config)
            except Exception as exc:
                print(f"[wandb] disabled: {exc}", flush=True)

    def log(self, values):
        if self.run is not None:
            try:
                self.run.log(values)
            except Exception as exc:
                print(f"[wandb] logging disabled: {exc}", flush=True)
                self.run = None
