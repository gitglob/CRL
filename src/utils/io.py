from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
from pathlib import Path

import yaml

import numpy as np


def append_csv(path, columns, rows):
    """Append logged rows, writing the header only when the file is first created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", restval="", lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerows(rows)


def save_csv(path, columns, rows):
    path = Path(path)
    if path.exists():
        path.unlink()
    append_csv(path, columns, rows)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _number(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


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


def group_scores(rows):
    """Collapse tidy per-episode evaluation rows back into {task: {return, episodes, env_steps}}."""
    scores = {}
    for row in rows:
        score = scores.setdefault(row["task"], {"episodes": [], "env_steps": row["eval_steps"]})
        score["episodes"].append(row["return"])
    for score in scores.values():
        score["return"] = float(np.mean(score["episodes"]))
    return {task: {"return": score["return"], "episodes": score["episodes"], "env_steps": score["env_steps"]} for task, score in scores.items()}


def read_run(directory):
    """Reassemble one arm's run: metadata from run.json, every logged series from its CSV."""
    directory = Path(directory)
    if not (directory / "run.json").exists():
        if not (directory / "metrics.json").exists():
            return {}
        legacy = json.loads((directory / "metrics.json").read_text())
        for entry in legacy.get("episode_log", []):
            entry.setdefault("env_steps", entry.get("steps"))
        legacy["evaluations"] = [{"block": row["block"], "scope": "boundary", "env_steps": row["steps"], "block_task": row["task"], "task": task, "episode": index, "return": value, "eval_steps": score.get("env_steps"), "train_episodes": None}
                                 for row in legacy.get("matrix", []) for task, score in row["scores"].items() for index, value in enumerate(score.get("episodes", []), start=1)]
        legacy["plasticity"] = []
        legacy.pop("plasticity_metrics", None)
        return legacy
    run = json.loads((directory / "run.json").read_text())
    config = directory / "config.yaml"
    run["config"] = yaml.safe_load(config.read_text()) if config.exists() else {}
    run["episode_log"] = read_csv(directory / "episodes.csv")
    run["diagnostics"] = read_csv(directory / "diagnostics.csv")
    run["plasticity"] = read_csv(directory / "plasticity.csv") if run.get("plasticity_metrics", {}).get("schema") == 1 else []
    run["block_log"] = read_csv(directory / "blocks.csv")
    evaluations = read_csv(directory / "evaluations.csv")
    run["evaluations"] = evaluations
    boundaries = {}
    for row in evaluations:
        if row["scope"] == "boundary":
            boundaries.setdefault(row["block"], []).append(row)
    run["matrix"] = [{"block": block, "task": rows[0]["block_task"], "steps": rows[0]["env_steps"], "scores": group_scores(rows)} for block, rows in sorted(boundaries.items())]
    within = {}
    for row in evaluations:
        if row["scope"] == "period":
            within.setdefault((row["block"], row["env_steps"]), []).append(row)
    points = {}
    for (block, steps), rows in sorted(within.items()):
        scores = group_scores(rows)
        points.setdefault(block, []).append({"steps": steps, "return": float(np.mean([score["return"] for score in scores.values()])), "scores": scores})
    run["curves"] = [{**entry, "points": points.get(entry["block"], [])} for entry in run["block_log"]]
    run["probes"] = {}
    settings_path = directory / "probe_settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    probe_rows = read_csv(directory / "probes.csv")
    for row in read_csv(directory / "probe_summary.csv"):
        phase = run["probes"].setdefault(row["phase"], {})
        mine = [entry for entry in probe_rows if entry["phase"] == row["phase"] and entry["task"] == row["task"]]
        phase[row["task"]] = {"status": row["status"], "train_env_steps": row["train_env_steps"],
                              "updates": row.get("updates"), "train_settings": settings.get(row["phase"], {}).get(row["task"], {}),
                              "episode_log": [entry for entry in mine if entry["scope"] == "train"]}
    return run
