import json
import os
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "tmp" / "matplotlib"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["path.simplify"] = False
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image, ImageDraw

from .io import read_run, save_csv, save_json
from .envs import make_env, seed_for, task_names
from ..clear.learner import ActorCritic
from .runtime import BudgetExpired, check_time, evaluate
from .report_details import diagnostic_notes, findings_notes, probe_result_notes, protocol_notes, tuning_notes

LABELS = {"finetune": "Fine-tuning", "cbp": "CBP", "replay": "Replay without cloning", "clear": "CLEAR", "clear_cbp": "CLEAR + CBP", "scratch": "Scratch", "multitask": "Joint training"}
COLORS = {"finetune": "tab:gray", "cbp": "tab:orange", "replay": "tab:purple", "clear": "tab:blue", "clear_cbp": "tab:green"}
PROBE_SMOOTHING_SIGMA = 200


def experiment_notes(root, runs, summary, qualification, manifest, common):
    elapsed = manifest.get("elapsed_seconds", time.time() - manifest["started_at"])
    steps = [data["compared_env_steps"] for data in summary.values()]
    exposure = f"{steps[0]:,}" if steps and len(set(steps)) == 1 else "See the individual run files for"
    lines = ["", f"The pipeline took **{elapsed / 60:.1f} minutes** on {manifest['gpu']}, with {manifest['workers']} concurrent runs selected by profiling. Each compared arm received {exposure} main-training transitions."]
    return lines


def measurement_notes(runs, summary, tasks, common):
    lines = ["", "Loss since each task's preceding learning block (positive means forgetting; the last trained task necessarily has zero loss at this checkpoint):", "", "| Arm | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for arm, data in summary.items():
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{data['forgetting_since_last_learning'].get(task, 0):.1f}" for task in tasks) + " |")
    lines.extend(["", "Held-out Asterix probes plot every completed training episode against probe environment steps. Plasticity is assessed descriptively, without an AUC score or an automatic pass/fail threshold."])
    for arm in ("cbp", "clear_cbp"):
        if arm in runs and common:
            row = runs[arm]["diagnostics"][common - 1]
            lines.append(f"{LABELS[arm]} replaced {row.get('actor_cbp_replacements', 0)} actor units and {row.get('critic_cbp_replacements', 0)} critic units in total.")
    return lines


def probe_notes(runs):
    lines = ["", "Recorded Asterix training settings, per arm and phase:", "",
             "| Arm | Phase | Steps | Updates | Learning rate | Discount | Entropy weight |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm, run in runs.items():
        for phase, tasks in run.get("probes", {}).items():
            for payload in tasks.values():
                settings = payload.get("train_settings", {})
                values = [payload.get("train_env_steps"), payload.get("updates"),
                          settings.get("learning_rate"), settings.get("gamma"), settings.get("entropy_weight")]
                lines.append(f"| {LABELS[arm]} | {phase} | " + " | ".join("unrecorded" if value is None else str(value) for value in values) + " |")
    return lines if len(lines) > 5 else []


def matched_references(root, run, common, tasks, random_scores):
    exposure = dict.fromkeys(tasks, 0)
    previous = 0
    for row in run["matrix"][1:common + 1]:
        trained = tasks if row["task"] == "multitask" else [row["task"]]
        for task in trained:
            exposure[task] += (row["steps"] - previous) // len(trained)
        previous = row["steps"]
    references = {}
    for task, steps in exposure.items():
        directory = root / "scratch" / task
        scratch = read_run(directory)
        matched = next((row for row in scratch.get("matrix", [])[1:] if row["steps"] == steps), None)
        references[task] = {"path": str(directory), "training_env_steps": steps,
                            "scratch_env_steps": matched["steps"] if matched else None,
                            "scratch_return": matched["scores"][task]["return"] if matched else None,
                            "random_return": random_scores.get(task, {}).get("return")}
    return references


def normalize(value, reference):
    random, scratch = reference["random_return"], reference["scratch_return"]
    if random is None or scratch is None or scratch <= random:
        return None
    return (value - random) / (scratch - random)


def figure_save(fig, root, name):
    fig.tight_layout()
    fig.savefig(root / name, dpi=140)
    plt.close(fig)


def task_segments(run, task, blocks=None):
    """Measured (x, y, style) evaluation runs, starting when the task first trains."""
    trained = {entry["block"] for entry in run.get("block_log", []) if entry["task"] in (task, "multitask")}
    trained |= {row["block"] for row in run.get("evaluations", []) if row["block_task"] in (task, "multitask")}
    if not trained:
        return []
    first = min(trained)
    measured = {}
    for row in run.get("evaluations", []):
        if row["task"] != task or (blocks is not None and row["block"] > blocks):
            continue
        # Before its own first block a task has never been trained, so its curve has not begun.
        if row["block"] < first and not (row["block"] == first - 1 and row["scope"] == "boundary"):
            continue
        # Steps, not episodes: episode length varies by task, so only steps are a shared clock.
        measured.setdefault((row["block"], row["env_steps"]), []).append(row["return"])
    ordered = [(key[1], float(np.mean(values)), key[0]) for key, values in sorted(measured.items())]
    segments, current, style = [], ordered[:1], None
    for previous, point in zip(ordered, ordered[1:]):
        # An interval belongs to the block it ends in: that is the block trained across it.
        mark = point[2] in trained
        if style is not None and mark != style:
            segments.append((current, style))
            current = [previous]
        current.append(point)
        style = mark
    if len(current) > 1:
        segments.append((current, style))
    elif current:
        segments.append((current, current[0][2] in trained))
    return [([x for x, _, _ in points], [y for _, y, _ in points], "-" if mark else "--") for points, mark in segments]


def figures(root, runs, suite, common):
    tasks = task_names(suite)
    cycle = plt.get_cmap("tab10").colors
    resolved_colors = {arm: COLORS.get(arm) or cycle[index % len(cycle)] for index, arm in enumerate(runs)}
    fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 3.2 * len(tasks)), squeeze=False)
    for task, ax in zip(tasks, axes[:, 0]):
        # Solid is this task's own training episodes; dashed is retention between them.
        for arm, run in runs.items():
            color = resolved_colors[arm]
            for xs, ys, style in task_segments(run, task, blocks=common - 1):
                ax.plot(xs, ys, linestyle=style, color=color)
        ax.set(title=task, xlabel="Environment steps", ylabel="Episodic return")
        ax.grid(alpha=0.2)
    handles = [Line2D([0], [0], color=resolved_colors[arm], label=LABELS[arm]) for arm in runs]
    axes[-1, 0].legend(handles=handles, fontsize=7)
    figure_save(fig, root, "performance.png")
    plasticity_figure(root, runs, common)
    probe_figure(root, runs, suite)


def gaussian_probe_returns(values, sigma=PROBE_SMOOTHING_SIGMA):
    """Smooth episode returns with a normalized Gaussian and reflected boundaries."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.copy()
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="reflect"), kernel, mode="valid")


def probe_figure(root, runs, suite):
    probe_names = task_names(suite, probes=True)
    selected = [(arm, run) for arm, run in runs.items() if run.get("probes")]
    if selected:
        fig, axes = plt.subplots(len(selected), len(probe_names), figsize=(10 * len(probe_names), 3.2 * len(selected)), squeeze=False, sharey="col")
        for row, (arm, run) in enumerate(selected):
            for column, task in enumerate(probe_names):
                ax = axes[row, column]
                for phase, style in (("initial", "--"), ("midpoint", ":"), ("final", "-")):
                    probe = run.get("probes", {}).get(phase, {}).get(task, {})
                    points = probe.get("episode_log", [])
                    ax.plot([point["env_steps"] for point in points], gaussian_probe_returns([point["return"] for point in points]), style, label=phase, linewidth=1.2)
                if not any(line.get_xdata().size for line in ax.lines):
                    ax.text(0.5, 0.5, "No completed probe episodes recorded", transform=ax.transAxes, ha="center")
                ax.set(title=f"{LABELS[arm]} / {task} (Gaussian σ={PROBE_SMOOTHING_SIGMA} episodes)", xlabel="Isolated probe env steps", ylabel="Smoothed episodic return")
                ax.grid(alpha=0.2)
                if column == 0:
                    ax.legend(fontsize=7)
        figure_save(fig, root, "probe_curves.png")


def plasticity_figure(root, runs, common):
    keys = (("stable_rank_percent", "Stable rank", "% of final hidden width"),
            ("dormant_percent", "Dormant units", "% of hidden units"),
            ("weight_magnitude_1", "Average weight magnitude", "Mean |weight|, hidden layer 2"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    for row, name in enumerate(("actor", "critic")):
        for ax, (metric, title, ylabel) in zip(axes[row], keys):
            for arm, run in runs.items():
                cutoff = run["matrix"][common]["steps"]
                key = f"{name}_{metric}"
                points = [entry for entry in run.get("plasticity", []) if entry["env_steps"] <= cutoff and entry.get(key) is not None]
                if points:
                    ax.plot([entry["env_steps"] for entry in points], [entry[key] for entry in points], label=LABELS[arm], color=COLORS.get(arm), linewidth=0.8)
            if not ax.lines:
                ax.text(0.5, 0.5, "No interaction diagnostics recorded", transform=ax.transAxes, ha="center", fontsize=9)
            ax.set(title=f"{name.capitalize()}: {title}", xlabel="Environment steps", ylabel=ylabel)
            if metric.endswith("percent"):
                ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=7)
            break
    figure_save(fig, root, "plasticity.png")


def baseline_figure(root, suite):
    tasks = task_names(suite)
    scratch_runs = {task: read_run(root / "scratch" / task) for task in tasks}
    others = {arm: read_run(root / arm) for arm in ("multitask", "finetune")}
    if not any(scratch_runs.values()) and not any(others.values()):
        return False
    colors = dict(zip(tasks, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(3, 1, figsize=(10, 3.2 * 3), squeeze=False)
    for arm, ax in zip(("scratch", "multitask", "finetune"), axes[:, 0]):
        if arm == "scratch":
            for task in tasks:
                run = scratch_runs[task]
                if not run:
                    continue
                for xs, ys, _ in task_segments(run, task):
                    ax.plot(xs, ys, color=colors[task])
            ax.set(xlabel="Environment steps", ylabel="Episodic return")
        else:
            run = others[arm]
            if run.get("evaluations"):
                for task in tasks:
                    for xs, ys, style in task_segments(run, task):
                        ax.plot(xs, ys, linestyle=style if style == "-" else ":", color=colors[task])
            ax.set(xlabel="Environment steps", ylabel="Episodic return")
        ax.set_title(LABELS[arm])
        ax.grid(alpha=0.2)
    handles = [Line2D([0], [0], color=colors[task], label=task) for task in tasks]
    for ax in reversed(axes[:, 0]):
        if ax.lines:
            ax.legend(handles=handles, fontsize=7)
            break
    figure_save(fig, root, "baseline.png")
    return True


def clip(agent, task, path, deadline):
    env = make_env(agent.suite, task, render_mode="rgb_array")
    frames = []
    total, steps = 0.0, 0
    done = False
    try:
        state, _ = env.reset(seed=seed_for(task, 500))
        generator = torch.Generator(device=agent.device).manual_seed(seed_for(task, 77))
        while not done:
            check_time(deadline)
            if steps % agent.config["eval"]["clip_stride"] == 0:
                # MinAtar renders a 10x10 frame, so upscale before the overlay is legible.
                rgb = (np.asarray(env.render()) * 255).astype(np.uint8)
                frame = Image.fromarray(rgb).resize((240, 240), Image.Resampling.NEAREST)
                ImageDraw.Draw(frame).text((5, 5), f"{LABELS[agent.arm]} | {task} | return {total:.0f}", fill="white")
                frames.append(frame)
            action = agent.act([state], rng=generator)[0][0]
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
    finally:
        env.close()
    if frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    return {"task": task, "path": str(path), "return": total, "steps": steps}


def reference_runs(root, tasks):
    """Scratch and joint baselines sit beside the compared arms without being part of the sweep."""
    found = {}
    for task in tasks:
        run = read_run(root / "scratch" / task)
        if run.get("matrix"):
            found[f"scratch/{task}"] = {"arm": "scratch", "directory": root / "scratch" / task, "run": run, "clips": [task]}
    joint = read_run(root / "multitask")
    if joint.get("matrix"):
        found["multitask"] = {"arm": "multitask", "directory": root / "multitask", "run": joint, "clips": list(tasks)}
    return found


def audit_references(references, config, suite, tasks, deadline, audit, clips):
    """Re-evaluate and film each baseline checkpoint, as the compared arms already are."""
    for name, entry in references.items():
        row = entry["run"]["matrix"][-1]
        try:
            check_time(deadline)
            agent = ActorCritic(config, suite, entry["arm"])
            payload = torch.load(entry["directory"] / "model.pt", map_location=agent.device, weights_only=False)
            agent.load_weights(payload["weights"])
            actual = evaluate(agent, tasks, config["eval"]["episodes"], deadline, moment=row["steps"])
            match = all(abs(actual[task]["return"] - row["scores"][task]["return"]) < 1e-6 for task in tasks)
            audit.append({"arm": name, "block": row["block"], "match": match, "scores": actual})
            if config["eval"]["clips"]:
                for task in entry["clips"]:
                    if time.time() >= deadline - 10:
                        break
                    clips.append(clip(agent, task, entry["directory"] / "videos" / f"{task}.gif", deadline - 5))
        except BudgetExpired:
            audit.append({"arm": name, "status": "budget_exhausted"})


def finalize_scratch(root, config, qualification, manifest, deadline, saved_artifacts=None):
    tasks = task_names(config["suite"])
    references = {}
    audit = deepcopy(saved_artifacts["audit"]) if saved_artifacts is not None else []
    clips = deepcopy(saved_artifacts["clips"]) if saved_artifacts is not None else []
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.5), squeeze=False)
    for task, ax in zip(tasks, axes[0]):
        directory = root / "scratch" / task
        run = read_run(directory)
        if not run:
            continue
        references[task] = {"status": run["status"], "completed_blocks": run["completed_blocks"], "env_steps": run["env_steps"], "scores": run["matrix"][-1]["scores"].get(task) if run["matrix"] else None}
        points = run.get("episode_log") or [point for curve in run["curves"] for point in curve["points"]]
        xs, ys = [point["env_steps"] for point in points], [point["return"] for point in points]
        ax.plot(xs, ys)
        ax.set(title=task, xlabel="Environment steps", ylabel="Episodic return")
        if saved_artifacts is not None or not run["matrix"]:
            continue
        try:
            check_time(deadline)
            agent = ActorCritic(config, config["suite"], "scratch")
            payload = torch.load(directory / "model.pt", map_location=agent.device, weights_only=False)
            agent.load_weights(payload["weights"])
            scores = evaluate(agent, [task], config["eval"]["episodes"], deadline, moment=run["matrix"][-1]["steps"])
            match = scores[task] == references[task]["scores"]
            audit.append({"arm": "scratch", "task": task, "match": match, "scores": scores})
            if config["eval"]["clips"]:
                clips.append(clip(agent, task, root / "scratch" / task / "videos" / f"{task}.gif", deadline))
        except BudgetExpired:
            audit.append({"arm": "scratch", "task": task, "status": "budget_exhausted"})
    figure_save(fig, root, "learning_curves.png")
    has_baseline = baseline_figure(root, config["suite"])
    complete = len(references) == len(tasks) and all(row["status"] == "complete" for row in references.values())
    verified = len(audit) == len(tasks) and all(row.get("match") for row in audit)
    report = {"schema": 1, "suite": config["suite"], "seed": 0, "outcome": "scratch references only", "qualified": False, "qualification": qualification, "common_completed_blocks": 0, "all_runs_complete": complete, "all_checkpoints_verified": verified, "references": references, "audit": audit, "clips": clips}
    save_json(root / "summary.json", report)
    save_json(root / "audit.json", audit)
    lines = ["# Independent scratch references", "", "Each task starts from identical seeded weights and a fresh optimizer, and receives one full learning block. These are acquisition references; they do not measure retention or plasticity loss.", "", "| Task | Training transitions | Mean return | Raw episode returns |", "|---|---:|---:|---|"]
    for task, row in references.items():
        score = row["scores"] or {"return": None, "episodes": []}
        lines.append(f"| {task} | {row['env_steps']} | {score['return']} | {score['episodes']} |")
    lines.extend(["", "![Scratch learning](learning_curves.png)", "", f"All runs complete: **{complete}**. All checkpoints verified: **{verified}**.", "", "Timing, source provenance, and pilot qualification are retained in summary.json."])
    if has_baseline:
        lines.extend(["", "![Baseline performance](baseline.png)", "", "One panel per baseline arm (whichever have completed under this root), one line per task."])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def finalize(root, config, qualification, manifest, deadline, saved_artifacts=None):
    from .study import retention_drops

    suite = config["suite"]
    tasks = task_names(suite)
    runs = {}
    for arm in config["arms"]:
        data = read_run(root / arm)
        if data.get("matrix"):
            runs[arm] = data
    common = min([runs.get(arm, {}).get("completed_blocks", 0) for arm in config["arms"]] or [0])
    audit = deepcopy(saved_artifacts["audit"]) if saved_artifacts is not None else []
    clips = deepcopy(saved_artifacts["clips"]) if saved_artifacts is not None else []
    summary = {}
    for arm, run in runs.items():
        row = run["matrix"][common]
        references_for_arm = matched_references(root, run, common, tasks, qualification.get("random_scores", {}))
        normalized = [normalize(row["scores"][task]["return"], references_for_arm[task]) for task in tasks]
        drops = retention_drops(run["matrix"][:common + 1])
        last_learning = {}
        for entry in run["matrix"][1:common + 1]:
            last_learning[entry["task"]] = entry["scores"].get(entry["task"], {}).get("return")
        final_forgetting = {task: last_learning[task] - row["scores"][task]["return"] for task in tasks if last_learning.get(task) is not None}
        summary[arm] = {"status": run["status"], "completed_blocks": run["completed_blocks"], "compared_blocks": common, "compared_env_steps": row["steps"], "final_returns": {task: row["scores"][task]["return"] for task in tasks}, "average_normalized_performance": float(np.mean(normalized)) if all(value is not None for value in normalized) else None, "forgetting_since_last_learning": final_forgetting, "retention_drops": drops, "normalization_references": references_for_arm, "probes": {phase: {task: {key: payload.get(key) for key in ("status", "train_env_steps", "updates", "train_settings")} for task, payload in entries.items()} for phase, entries in run.get("probes", {}).items()}, "probe_checkpoint_blocks": run.get("probe_checkpoint_blocks", {}), "updates": run["diagnostics"][common - 1]["updates"] if common else 0}
        if run["completed_blocks"] != common:
            summary[arm]["probe_comparison_warning"] = "Probe ages differ from the common comparison checkpoint. Read as individual runs only."
        if saved_artifacts is not None:
            continue
        try:
            check_time(deadline)
            agent = ActorCritic(config, suite, arm)
            payload = torch.load((root / arm / "model.pt") if common == run["completed_blocks"] else (Path(config.get("workspace", root / "work")) / "checkpoints" / arm / f"block_{common:04d}.pt"), map_location=agent.device, weights_only=False)
            agent.load_weights(payload["weights"])
            actual = evaluate(agent, tasks, config["eval"]["episodes"], deadline, moment=row["steps"])
            match = all(abs(actual[task]["return"] - row["scores"][task]["return"]) < 1e-6 for task in tasks)
            audit.append({"arm": arm, "block": common, "match": match, "scores": actual})
            if config["eval"]["clips"]:
                for task in tasks:
                    if time.time() >= deadline - 10:
                        break
                    clips.append(clip(agent, task, root / arm / "videos" / f"{task}.gif", deadline - 5))
        except BudgetExpired:
            audit.append({"arm": arm, "status": "budget_exhausted"})
    references = reference_runs(root, tasks)
    if saved_artifacts is None:
        audit_references(references, config, suite, tasks, deadline, audit, clips)
    if runs:
        figures(root, runs, suite, common)
    has_baseline = baseline_figure(root, suite)
    complete = set(runs) == set(config["arms"]) and all(run["status"] == "complete" for run in runs.values())
    complete = complete and all(entry["run"]["status"] == "complete" for entry in references.values())
    verified = len(audit) == len(runs) + len(references) and bool(runs) and all(row.get("match") for row in audit)
    outcome = "completed study" if complete and verified else "incomplete study"
    save_csv(root / "retention.csv", ("arm", "block", "task", "since_block", "drop", "return"),
             [{"arm": arm, **drop} for arm, data in summary.items() for drop in data["retention_drops"]] + [{"arm": "pilot_finetune", **drop} for drop in qualification.get("retention_drops", [])])
    # Tables belong in the CSVs; the JSON keeps the decision and provenance record.
    slim = {arm: {key: value for key, value in data.items() if key != "retention_drops"} for arm, data in summary.items()}
    report = {"schema": 1, "suite": suite, "seed": 0, "outcome": outcome, "qualified": qualification["qualified"], "common_completed_blocks": common, "all_runs_complete": complete, "all_checkpoints_verified": verified, "arms": slim, "qualification": {key: value for key, value in qualification.items() if key != "retention_drops"}, "provenance": manifest, "audit": audit, "clips": clips}
    if (root / "probe_refresh.json").exists():
        report["probe_refresh"] = json.loads((root / "probe_refresh.json").read_text())
    save_json(root / "summary.json", report)
    save_json(root / "audit.json", audit)
    save_csv(root / "summary.csv", ("arm", "task", "final_return", "forgetting_since_last_learning", "average_normalized_performance", "compared_blocks", "compared_env_steps", "updates", "status"),
             [{"arm": arm, "task": task, "final_return": data["final_returns"][task], "forgetting_since_last_learning": data["forgetting_since_last_learning"].get(task),
               "average_normalized_performance": data["average_normalized_performance"], "compared_blocks": data["compared_blocks"],
               "compared_env_steps": data["compared_env_steps"], "updates": data["updates"], "status": data["status"]} for arm, data in summary.items() for task in tasks])
    lines = ["# CLEAR and continual backpropagation", "", f"**{outcome.capitalize()}.** One seed (0), {common} matched completed blocks. Plasticity conclusions are descriptive."]
    lines.extend(experiment_notes(root, runs, summary, qualification, manifest, common))
    lines.extend(findings_notes(runs, summary, common, LABELS))
    lines.extend(protocol_notes(config, runs, common))
    lines.extend(["", "## Pilot checks", "", f"All pilots completed: **{qualification['all_pilots_complete']}**. Scratch/joint learnability: **{qualification['learnable']}**. Forgetting demonstrated: **{qualification['forgetting_demonstrated']}**.", "", "These checks describe the short pilots; they do not determine study completion or provide the final normalization references. Plasticity has no automatic qualification threshold."])
    lines.extend(["", "## Matched final returns", "", "| Arm | " + " | ".join(tasks) + " | Normalized AP |", "|---|" + "---:|" * (len(tasks) + 1)])
    for arm, data in summary.items():
        ap = data["average_normalized_performance"]
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{data['final_returns'][task]:.1f}" for task in tasks) + " | " + (f"{ap:.3f}" if ap is not None else "unavailable") + " |")
    lines.extend(["", "Normalization is (return − random) / (scratch − random), without clipping. Scratch checkpoints come from this output root and match each task's training exposure. Missing matches or nonpositive denominators give unavailable scores; pilot scratch scores are never substituted.", "", "| Arm | Task | Matched scratch steps | Scratch return | Random return |", "|---|---|---:|---:|---:|"])
    for arm, data in summary.items():
        for task, reference in data["normalization_references"].items():
            values = [reference[key] for key in ("scratch_env_steps", "scratch_return", "random_return")]
            lines.append(f"| {LABELS[arm]} | {task} | " + " | ".join("unavailable" if value is None else str(value) for value in values) + " |")
    if has_baseline:
        lines.extend(["", "![Baseline performance](baseline.png)"])
    lines.extend(measurement_notes(runs, summary, tasks, common))
    lines.extend(probe_notes(runs))
    lines.extend(probe_result_notes(runs, LABELS))
    lines.extend(tuning_notes(root))
    if "probe_refresh" in report:
        lines.extend(["", "The Asterix probes were rerun from the saved main-training checkpoints with the settings above. Main-training results and their checkpoint audits are unchanged. Probe rerun provenance is recorded in probe_refresh.json; its additional runtime is separate from the original study runtime."])
    lines.extend(["", "## Retention and fresh-task learning", "", "![Performance](performance.png)", "", "Every task is evaluated at the shared environment-step grid and block boundaries. Each plotted measurement is the mean of the evaluation episodes at that checkpoint. Solid lines indicate training that task; dashed lines indicate training other tasks. Performance, baselines and plasticity diagnostics remain unsmoothed. Only probe curves use Gaussian smoothing; no plot is downsampled.", "", "Initial, midpoint and final Asterix probes use isolated weight copies, fresh optimizers and fresh-only updates with replay and CBP disabled. Probe returns are Gaussian-smoothed across episode order (sigma 200 episodes, kernel truncated at four sigma, reflected boundaries) and plotted at the original environment-step coordinates. Each phase is smoothed separately; raw episode logs are unchanged. The initial probe is the shared scratch reference. No separate probe evaluation or AUC score is used.", "", "![Probe curves](probe_curves.png)", "", "## Plasticity diagnostics", "", "![Plasticity diagnostics](plasticity.png)", "", "Actor and critic occupy separate rows. Each uses the most recent interaction activations, ordered by timestep then environment index across fresh transitions. Evaluation, replay and probes do not enter the window. Windows continue across episode and task boundaries.", "", "Stable rank is the smallest rank containing at least 99% of the uncentered final-hidden-layer singular-value mass, displayed as a percentage of layer width. Dormant units are active on at most 1% of observations, pooled across both hidden layers. Average weight magnitude is mean absolute weight excluding biases; the plot uses the second hidden linear layer, while every layer is logged."])
    for arm, run in runs.items():
        metadata = run.get("plasticity_metrics")
        if metadata:
            lines.append(f"- {LABELS[arm]}: window {metadata['window_steps']:,} transitions; dormant/weights every {metadata['period']:,}; rank every {metadata['rank_period']:,}.")
        else:
            lines.append(f"- {LABELS[arm]}: interaction diagnostics unavailable in these older logs; old metrics are not reused.")
    lines.extend(diagnostic_notes(runs, common, LABELS))
    lines.extend(["", "The rank and dormancy definitions follow the [authors' PPO diagnostics](https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/rl/run_ppo.py), extended here to the critic. The criterion is singular-value mass, not squared-energy stable rank.", "", "## Implementation and verification", "", "All arms retain identical actor/critic initialization, AdamCBP, V-trace, learner batches and main-training budgets. CLEAR adds reservoir replay and replay-only policy/value cloning; CBP replaces low-contribution mature units and resets their optimizer state. This is a single-seed comparison, not a statistical ranking or a full paper reproduction.", "", f"All requested runs completed: **{complete}**. All matched checkpoints verified: **{verified}**.", "", "Per-arm CSV files retain raw episodes and diagnostics. summary.json records matched normalization references and probe completion/budgets. Configurations, checkpoint audits and source provenance are preserved. Pilot runs, profiling and intermediate checkpoints live in tmp/. No wall-clock budget limits the study.", ""])
    if clips:
        lines.extend(["### Representative clips", ""])
        for item in clips:
            path = Path(item["path"])
            lines.append(f"- [{path.relative_to(root).parts[0]}: {item['task']}]({path.relative_to(root)}) — return {item['return']:.0f}.")
    if not any(run.get("probes") for run in runs.values()):
        lines = [line for line in lines if line != "![Probe curves](probe_curves.png)"]
    if all((root / arm / "checkpoints" / f"{phase}.pt").exists() for arm in runs for phase in ("initial", "midpoint", "final")):
        lines.extend(["", "Named phase checkpoints retain both networks and their configuration under each run's checkpoints/ directory. Run isolated probes again with:", "", "```bash", ".venv/bin/python -m src.probes --run results/cbp --out tmp/cbp_probes", "```", "", "Use --phase final to select one phase, or --config to choose probe settings. All three phases run by default; main-task training is not repeated."])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report
