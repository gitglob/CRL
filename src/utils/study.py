from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import parser, resolve, run_directory
from .envs import action_count, observation_size
from .evaluate import evaluate_all
from .io import save_config, save_json


def jobs(config, root, overwrite):
    planned = []
    for seed in config["benchmark"]["seeds"]:
        for arm in config["benchmark"]["arms"]:
            if arm == "scratch":
                for task in config["benchmark"]["order"]:
                    planned.append({"arm": arm, "seed": seed, "task": task, "out": str(run_directory(root, arm, seed, task)), "overwrite": overwrite})
            else:
                planned.append({"arm": arm, "seed": seed, "task": None, "out": str(run_directory(root, arm, seed)), "overwrite": overwrite})
    return planned


def execute(payload):
    from ..train import train
    config, job = payload
    tasks = [job["task"]] if job["task"] else None
    train(config, job["arm"], job["seed"], job["out"], tasks, job["overwrite"])
    return job["out"]


def audit(root, config):
    """Reload every checkpoint and re-derive its final row. Artifacts must survive a reread."""
    from ..dqn import load_dqn
    from ..train import streams

    root = Path(root)
    checked, mismatches = 0, []
    for path in sorted(root.rglob("metrics.json")):
        if path.parent == root:
            continue
        metrics = json.loads(path.read_text())
        agent, metadata = load_dqn(path.parent / "model.pt", config, metrics["arm"], metrics["seed"], observation_size(config), action_count(config), streams(metrics["seed"]))
        if metadata["signature"] != metrics["signature"]:
            mismatches.append(f"{path.parent}: checkpoint signature differs from metrics")
        replay = evaluate_all(agent, config, config["eval"]["matrix_episodes"])
        recorded = metrics["matrix"][-1]["evaluations"]
        for task, report in replay.items():
            if abs(report["return_mean"] - recorded[task]["return_mean"]) > 1e-6:
                mismatches.append(f"{path.parent}: {task} re-evaluates to {report['return_mean']} not {recorded[task]['return_mean']}")
        checked += 1
    if mismatches:
        raise RuntimeError("Audit failed:\n" + "\n".join(mismatches))
    report = {"runs": checked, "root": str(root)}
    save_json(root / "audit.json", report)
    return report


def study(config, workers=1, overwrite=False):
    from .compare import compare

    root = Path(config["output"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    # Do not overwrite the study manifest until existing settings are compatible.
    if (root / "config.yaml").exists() and not overwrite:
        from .config import load_config
        if load_config(root / "config.yaml") != config:
            raise ValueError("Study configuration changed; choose another --out or use --overwrite")
    save_config(root / "config.yaml", config)
    (root / "audit.json").unlink(missing_ok=True)

    planned = jobs(config, root, overwrite)
    print(f"[study] {len(planned)} runs, {workers} worker(s)", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done in pool.map(execute, [(config, job) for job in planned]):
                print(f"[study] done {done}", flush=True)
    else:
        for job in planned:
            execute((config, job))

    report = audit(root, config)
    compare(root, config)
    print(f"[study] COMPLETE: {report['runs']} audited runs; report: {root / 'REPORT.md'}", flush=True)


def main():
    args = parser("Run the full continual-learning study").parse_args()
    config = resolve(args)
    study(config, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
