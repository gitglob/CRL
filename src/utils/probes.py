import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import torch

from .config import PROBE_TRAIN_KEYS, load_config, validate
from .io import read_run, save_config, save_json
from .runtime import probe_tasks, save_probes
from ..clear.learner import ActorCritic

PHASES = ("initial", "midpoint", "final")


def run_probes(run, out, settings, phases=PHASES, overwrite=False):
    run, out = Path(run).resolve(), Path(out).resolve()
    if run == out or run in out.parents or out in run.parents:
        raise ValueError("Probe output must be separate from the source run")
    if not phases or len(set(phases)) != len(phases) or set(phases) - set(PHASES):
        raise ValueError("Select unique initial, midpoint or final phases")
    checkpoints, configs = {}, {}
    for phase in phases:
        path = run / "checkpoints" / f"{phase}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("kind") != "probe_weights" or saved.get("schema") != 1 or saved["phase"] != phase:
            raise ValueError(f"Incompatible phase checkpoint: {path}")
        config = deepcopy(saved["config"])
        config["device"] = settings["device"]
        config["train"]["probe_steps"] = settings["train"]["probe_steps"]
        config["probe_train"] = {key: settings.get("probe_train", {}).get(key, settings["train"][key]) for key in PROBE_TRAIN_KEYS}
        configs[phase] = validate(config)
        checkpoints[phase] = (path, saved)
    first = checkpoints[phases[0]][1]
    if any(saved["arm"] != first["arm"] or saved["suite"] != first["suite"] for _, saved in checkpoints.values()):
        raise ValueError("Phase checkpoints must belong to the same arm and suite")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise ValueError("Probe output exists; use --overwrite to archive it")
        archive = Path("tmp") / f"{out.name}_previous_{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        out.rename(archive)
    out.mkdir(parents=True, exist_ok=True)
    save_config(out / "config.yaml", configs[phases[0]])
    metadata = {"schema": 1, "kind": "isolated_probes", "arm": first["arm"], "suite": first["suite"],
                "status": "running", "started_at": time.time(), "source_checkpoints": {},
                "probe_checkpoint_blocks": {phase: saved["completed_blocks"] for phase, (_, saved) in checkpoints.items()}}
    for phase, (path, saved) in checkpoints.items():
        metadata["source_checkpoints"][phase] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                "completed_blocks": saved["completed_blocks"], "env_steps": saved["env_steps"]}
    save_json(out / "run.json", metadata)
    try:
        for phase, (_, saved) in checkpoints.items():
            source = ActorCritic(configs[phase], saved["suite"], "finetune")
            source.load_weights(saved["weights"])
            results = {}
            try:
                probe_tasks(source, float("inf"), phase, results)
            finally:
                save_probes(out, phase, results)
        metadata["status"] = "complete"
    except BaseException:
        metadata["status"] = "incomplete"
        raise
    finally:
        metadata["seconds"] = time.time() - metadata["started_at"]
        save_json(out / "run.json", metadata)
    from .report import probe_figure

    probe_figure(out, {first["arm"]: read_run(out)}, first["suite"])
    print(f"Probe results: {out}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train Asterix from saved phase weights without main-task training")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--phase", action="append", choices=PHASES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_probes(args.run, args.out, load_config(args.config), args.phase or PHASES, args.overwrite)
