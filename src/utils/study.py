import json
import hashlib
import multiprocessing as mp
import subprocess
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .io import read_run, save_config, save_json, versions
from .config import MAIN_ARMS, parser, resolve
from .envs import task_names
from ..clear.learner import ActorCritic
from .runtime import BudgetExpired, evaluate, run_job

PROFILE_SECONDS = 120
SMOKE_PROFILE_SECONDS = 10


def read_metrics(path):
    return read_run(path)


def run_jobs(jobs, workers):
    results = []
    if not jobs:
        return results
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"out": job["out"], "status": "failed", "error": f"{type(exc).__name__}: {exc}", "completed_blocks": 0}
                Path(job["out"]).mkdir(parents=True, exist_ok=True)
                save_json(Path(job["out"]) / "failure.json", result)
            results.append(result)
            print(f"[study] {job['arm']} {result['status']} blocks={result['completed_blocks']}", flush=True)
    return results


def qualification(suite, runs, random_scores):
    tasks = task_names(suite)
    scratch = {task: runs.get(f"scratch/{task}", {}) for task in tasks}
    joint = runs.get("multitask", {})
    sequential = runs.get("finetune", {})
    required = list(scratch.values()) + [joint, sequential]
    completed = all(run.get("status") == "complete" for run in required)
    scratch_scores = {task: run["matrix"][-1]["scores"][task]["return"] for task, run in scratch.items() if run.get("matrix")}
    joint_scores = {task: joint["matrix"][-1]["scores"][task]["return"] for task in tasks} if joint.get("matrix") else {}
    denominators = {task: 500.0 if suite == "cartpole" else scratch_scores.get(task, 0) - random_scores.get(task, {}).get("return", 0) for task in tasks}
    if suite == "cartpole":
        learnable = all(scratch_scores.get(task, 0) >= 400 and joint_scores.get(task, 0) >= 400 for task in tasks)
    else:
        learnable = all(denominators[task] > max(1, 0.2 * random_scores.get(task, {}).get("return", 0)) and joint_scores.get(task, 0) >= random_scores.get(task, {}).get("return", 0) + 0.8 * denominators[task] for task in tasks)
    drops = retention_drops(sequential.get("matrix", []))
    forgetting_tasks = [task for task in tasks if denominators[task] > 0 and max([row["drop"] for row in drops if row["task"] == task] or [0]) / denominators[task] >= 0.2]
    probes = sequential.get("probes", {})
    deficits, probe_learnable = {}, {}
    for task in task_names(suite, probes=True):
        fresh = probes.get("initial", {}).get(task, {})
        late = probes.get("final", {}).get(task, {})
        baseline = fresh.get("auc")
        current = late.get("auc")
        denominator = 500 if suite == "cartpole" else (fresh.get("curve", [{}])[-1].get("return", 0) - random_scores.get(task, {}).get("return", 0))
        deficits[task] = (baseline - current) / denominator if baseline is not None and current is not None and denominator > 0 else None
        probe_learnable[task] = suite == "cartpole" or denominator > max(1, 0.2 * random_scores.get(task, {}).get("return", 0))
    plasticity_tasks = [task for task, deficit in deficits.items() if deficit is not None and deficit >= 0.1]
    required_probes = 2 if suite == "cartpole" else 1
    plasticity = len([task for task in plasticity_tasks if probe_learnable[task]]) >= required_probes
    return {"qualified": completed and learnable and len(forgetting_tasks) >= 2 and plasticity, "all_pilots_complete": completed, "learnable": learnable, "forgetting_demonstrated": len(forgetting_tasks) >= 2, "plasticity_loss_demonstrated": plasticity, "probe_auc_threshold_met": len(plasticity_tasks) >= required_probes, "probe_reference_learnable": probe_learnable, "scratch_returns": scratch_scores, "joint_returns": joint_scores, "random_scores": random_scores, "forgetting_tasks": forgetting_tasks, "probe_auc_deficits": deficits, "retention_drops": drops, "fresh_reference": "Initial weights equal the seeded scratch network; the initial probes are the paired scratch reference."}


def retention_drops(matrix):
    learned = {}
    results = []
    for row in matrix:
        for task, reference in learned.items():
            if task == row["task"]:
                continue
            score = row["scores"][task]["return"]
            results.append({"block": row["block"], "task": task, "since_block": reference["block"], "drop": reference["return"] - score, "return": score})
        task = row["task"]
        if task in row["scores"]:
            learned[task] = {"block": row["block"], "return": row["scores"][task]["return"]}
    return results


def profile(config, root, deadline=None):
    """Throughput is measured against the clock, so this phase alone carries a time box."""
    budget = SMOKE_PROFILE_SECONDS if config["smoke"] else PROFILE_SECONDS
    deadline = time.time() + budget if deadline is None else deadline
    measurements = []
    modes = [int(config["workers"])] if config["workers"] != "auto" else [1, 2, 4]
    for index, workers in enumerate(modes):
        remaining = deadline - time.time()
        if remaining <= 2:
            break
        phase_end = time.time() + remaining / (len(modes) - index)
        short = deepcopy(config)
        short["train"]["block_steps"] = 2048 if not config["smoke"] else 256
        short["eval"]["episodes"] = 1
        short["eval"]["period"] = 10 ** 9
        arms = ["finetune", "cbp", "clear", "clear_cbp"][:workers]
        jobs = [{"config": short, "suite": "cartpole", "arm": arm, "blocks": ["identity"] * 10000, "out": str(root / "profile" / f"workers_{workers}" / arm), "deadline": phase_end, "quiet": True} for arm in arms]
        began = time.time()
        outputs = run_jobs(jobs, workers)
        elapsed = time.time() - began
        count = sum(read_metrics(result["out"]).get("attempted_env_steps", 0) for result in outputs)
        peaks = [result.get("gpu_peak_bytes", 0) for result in outputs]
        measurements.append({"workers": workers, "seconds": elapsed, "transitions_per_second": count / max(elapsed, 1e-9), "allocated_gpu_peak_bytes": sum(peaks)})
        print(f"[profile] workers={workers} transitions/s={count / max(elapsed, 1e-9):.0f} allocated_gpu_MB={sum(peaks) / 1e6:.1f}", flush=True)
    if not measurements:
        return {"selected_workers": 1, "measurements": []}
    free, _ = torch.cuda.mem_get_info() if config["device"].startswith("cuda") else (2 ** 40, 2 ** 40)
    viable = [row for row in measurements if row["allocated_gpu_peak_bytes"] + row["workers"] * 700_000_000 < 0.8 * free]
    best = max(viable or measurements[:1], key=lambda row: row["transitions_per_second"])
    return {"selected_workers": best["workers"], "measurements": measurements, "memory_headroom_fraction": 0.2}


def pilot(config, suite, root, workers, deadline):
    tasks = task_names(suite)
    pilot_root = root / "pilot" / suite
    settings = deepcopy(config)
    settings["cycles"] = min(config["cycles"], 4 if suite == "cartpole" else 2)
    job_list = []
    for task in tasks:
        job_list.append({"config": settings, "suite": suite, "arm": "scratch", "blocks": [task], "out": str(pilot_root / "scratch" / task), "deadline": deadline})
    job_list.append({"config": settings, "suite": suite, "arm": "multitask", "blocks": ["multitask"], "block_steps": len(tasks) * settings["train"]["block_steps"], "out": str(pilot_root / "multitask"), "deadline": deadline})
    job_list.insert(0, {"config": settings, "suite": suite, "arm": "finetune", "out": str(pilot_root / "finetune"), "deadline": deadline, "probes": True})
    outputs = run_jobs(job_list, workers)
    runs = {str(Path(job["out"]).relative_to(pilot_root)): read_metrics(job["out"]) for job in job_list}
    agent = ActorCritic(config, suite, "finetune")
    try:
        random_scores = evaluate(agent, tasks + task_names(suite, probes=True), config["eval"]["episodes"], deadline, random_policy=True)
    except BudgetExpired:
        random_scores = {}
    report = qualification(suite, runs, random_scores)
    report["jobs"] = outputs
    pilot_root.mkdir(parents=True, exist_ok=True)
    save_json(pilot_root / "qualification.json", report)
    print(f"[qualification {suite}] learnable={report['learnable']} forgetting={report['forgetting_demonstrated']} plasticity={report['plasticity_loss_demonstrated']} qualified={report['qualified']}", flush=True)
    return report, runs


def main_jobs(settings, root, deadline, initial_probes=None):
    suite = settings["suite"]
    tasks = task_names(suite)
    jobs = []
    work = Path(settings.get("workspace", root / "work"))
    for arm in settings.get("arms", MAIN_ARMS):
        if arm == "scratch":
            for task in tasks:
                jobs.append({"config": settings, "suite": suite, "arm": arm, "blocks": [task], "out": str(root / arm / task), "checkpoints": str(work / "checkpoints" / arm / task), "deadline": deadline})
            continue
        blocks = ["multitask"] * settings["cycles"] if arm == "multitask" else tasks * settings["cycles"]
        job = {"config": settings, "suite": suite, "arm": arm, "blocks": blocks, "out": str(root / arm), "checkpoints": str(work / "checkpoints" / arm), "deadline": deadline, "probes": arm in ("finetune", "cbp", "clear", "clear_cbp"), "initial_probes": initial_probes}
        if arm == "multitask":
            job["block_steps"] = len(tasks) * settings["train"]["block_steps"]
        jobs.append(job)
    return jobs + reference_jobs(settings, root, deadline)


def reference_jobs(settings, root, deadline):
    """Scratch and joint baselines at the per-task budget a sequential arm gives each task."""
    if set(settings.get("arms", MAIN_ARMS)) & {"scratch", "multitask"}:
        return []
    suite = settings["suite"]
    tasks = task_names(suite)
    work = Path(settings.get("workspace", root / "work"))
    jobs = [{"config": settings, "suite": suite, "arm": "scratch", "blocks": [task] * settings["cycles"], "out": str(root / "scratch" / task), "checkpoints": str(work / "checkpoints" / "scratch" / task), "deadline": deadline} for task in tasks]
    jobs.append({"config": settings, "suite": suite, "arm": "multitask", "blocks": ["multitask"] * settings["cycles"], "block_steps": len(tasks) * settings["train"]["block_steps"], "out": str(root / "multitask"), "checkpoints": str(work / "checkpoints" / "multitask"), "deadline": deadline})
    return jobs


def study(config, overwrite=False):
    from .report import finalize, finalize_scratch

    root = Path(config["output"])
    work = Path("tmp") / ("stability-plasticity" if root.resolve() == Path("results").resolve() else root.name + "_work")
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise ValueError(f"Output already exists: {root}; select another --out or use --overwrite")
        archive = Path("tmp") / (root.name + f"_previous_{time.time_ns()}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        root.rename(archive)
    if work.exists():
        work.rename(work.with_name(work.name + f"_previous_{time.time_ns()}"))
    work.mkdir(parents=True)
    config = deepcopy(config)
    config["workspace"] = str(work)
    root.mkdir(parents=True, exist_ok=True)
    save_config(root / "config.yaml", config)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable to this process; run with host GPU access")
    started = time.time()
    # No clock bounds the study: cycles and block_steps alone decide how much work it does.
    deadline = float("inf")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    repository = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    with zipfile.ZipFile(work / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        files = list((repository / "src").rglob("*.py")) + list((repository / "config").glob("*.yaml")) + list((repository / "tests").glob("*.py")) + list(repository.glob("requirements*.txt"))
        for file in sorted(files):
            name = str(file.relative_to(repository))
            digest.update(name.encode() + file.read_bytes())
            archive.write(file, name)
    recorded_versions = versions()
    import importlib.metadata

    recorded_versions["minatar"] = importlib.metadata.version("minatar")
    manifest = {"schema": 1, "started_at": started, "seed": 0, "git_revision": revision, "working_tree_dirty": dirty, "source_sha256": digest.hexdigest(), "versions": recorded_versions, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", "status": "running", "workspace": str(work), "qualification_history": {}}
    save_json(work / "manifest.json", manifest)
    profiling = profile(config, work)
    save_json(work / "profile.json", profiling)
    workers = profiling["selected_workers"]
    print(f"[study] selected {workers} concurrent runs", flush=True)
    suite = "cartpole" if config["suite"] == "auto" else config["suite"]
    report, runs = pilot(config, suite, work, workers, deadline)
    manifest["qualification_history"][suite] = report
    if config["suite"] == "auto" and not report["qualified"]:
        suite = "minatar"
        report, runs = pilot(config, suite, work, workers, deadline)
        manifest["qualification_history"][suite] = report
    settings = deepcopy(config)
    settings["suite"] = suite
    save_config(root / "config.yaml", settings)
    initial_probes = runs.get("finetune", {}).get("probes", {}).get("initial")
    jobs = main_jobs(settings, root, deadline, initial_probes)
    manifest["main_jobs"] = run_jobs(jobs, workers)
    manifest.update({"suite": suite, "qualified": report["qualified"], "cycles": settings["cycles"], "workers": workers})
    save_json(work / "manifest.json", manifest)
    finalizer = finalize_scratch if settings["arms"] == ["scratch"] else finalize
    final_report = finalizer(root, settings, report, manifest, deadline)
    manifest["elapsed_seconds"] = time.time() - started
    manifest["status"] = "complete" if final_report["qualified"] and final_report["all_runs_complete"] and final_report["all_checkpoints_verified"] else "inconclusive"
    save_json(work / "manifest.json", manifest)
    final_report["provenance"] = manifest
    final_report["profile"] = profiling
    save_json(root / "summary.json", final_report)
    print(f"[study] {manifest['status'].upper()} elapsed={manifest['elapsed_seconds']:.1f}s report={root / 'REPORT.md'}", flush=True)


def main():
    args = parser().parse_args()
    study(resolve(args), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
