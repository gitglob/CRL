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
from matplotlib.lines import Line2D
import numpy as np
import torch
from PIL import Image, ImageDraw

from .io import read_run, save_csv, save_json
from .envs import make_env, seed_for, task_names
from ..clear.learner import ActorCritic
from .runtime import BudgetExpired, check_time, evaluate

LABELS = {"finetune": "Fine-tuning", "cbp": "CBP", "replay": "Replay without cloning", "clear": "CLEAR", "clear_cbp": "CLEAR + CBP", "scratch": "Scratch", "multitask": "Joint training"}
COLORS = {"finetune": "tab:gray", "cbp": "tab:orange", "replay": "tab:purple", "clear": "tab:blue", "clear_cbp": "tab:green"}


def experiment_notes(root, runs, summary, qualification, manifest, common):
    elapsed = manifest.get("elapsed_seconds", time.time() - manifest["started_at"])
    steps = [data["compared_env_steps"] for data in summary.values()]
    exposure = f"{steps[0]:,}" if steps and len(set(steps)) == 1 else "See the individual run files for"
    lines = ["", f"The pipeline took **{elapsed / 60:.1f} minutes** on {manifest['gpu']}, with {manifest['workers']} concurrent runs selected by profiling. Each compared arm received {exposure} main-training transitions."]
    cart = manifest.get("qualification_history", {}).get("cartpole")
    if manifest["suite"] == "minatar" and cart is not None:
        lines.extend(["", f"CartPole did not qualify (learning **{cart['learnable']}**, forgetting **{cart['forgetting_demonstrated']}**, plasticity loss **{cart['plasticity_loss_demonstrated']}**), so the fallback selected MinAtar."])
    if qualification.get("probe_auc_threshold_met") and not qualification["plasticity_loss_demonstrated"]:
        lines.extend(["", "The raw probe-AUC threshold was crossed, but the fresh reference did not learn enough above random. That threshold alone is not evidence of plasticity loss."])
    return lines


def measurement_notes(runs, summary, tasks, common):
    lines = ["", "Loss since each task's preceding learning block (positive means forgetting; the last trained task necessarily has zero loss at this checkpoint):", "", "| Arm | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for arm, data in summary.items():
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{data['forgetting_since_last_learning'].get(task, 0):.1f}" for task in tasks) + " |")
    lines.extend(["", "Isolated held-out learning AUC (time-averaged raw return over the same probe budget):", "", "| Arm / task | Initial = scratch | Midpoint | Final |", "|---|---:|---:|---:|"])
    for arm, data in summary.items():
        for task in data["probe_auc"].get("initial", {}):
            values = [data["probe_auc"].get(phase, {}).get(task) for phase in ("initial", "midpoint", "final")]
            lines.append("| " + LABELS[arm] + " / " + task + " | " + " | ".join("incomplete" if value is None else f"{value:.3f}" for value in values) + " |")
    lines.extend(["", "Low probe returns and rank changes do not independently establish loss of plasticity. A fresh learner must learn the probe within the allotted budget for an AUC deficit to be persuasive."])
    for arm in ("cbp", "clear_cbp"):
        if arm in runs and common:
            row = runs[arm]["diagnostics"][common - 1]
            lines.append(f"{LABELS[arm]} replaced {row.get('actor_cbp_replacements', 0)} actor units and {row.get('critic_cbp_replacements', 0)} critic units in total.")
    return lines


def normalize(value, task, suite, qualification):
    if suite == "cartpole":
        return value / 500
    random = qualification.get("random_scores", {}).get(task, {}).get("return")
    reference = qualification.get("scratch_returns", {}).get(task)
    if random is None or reference is None or reference <= random:
        return None
    return (value - random) / (reference - random)


def figure_save(fig, root, name):
    fig.tight_layout()
    fig.savefig(root / name, dpi=140)
    plt.close(fig)


def smooth_series(xs, ys, bins=24):
    """Average into equal-width x bins: raw per-episode return is far too noisy to read."""
    if len(xs) <= bins:
        return list(xs), list(ys)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    edges = np.linspace(xs[0], xs[-1], bins + 1)
    index = np.clip(np.digitize(xs, edges[1:-1]), 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    filled = counts > 0
    return (np.bincount(index, weights=xs, minlength=bins)[filled] / counts[filled]).tolist(), (np.bincount(index, weights=ys, minlength=bins)[filled] / counts[filled]).tolist()


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
    return [(*smooth_series([x for x, _, _ in points], [y for _, y, _ in points]), "-" if mark else "--") for points, mark in segments]


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
    keys = (("actor_stable_rank_1", "Actor stable rank"), ("critic_stable_rank_1", "Critic stable rank"), ("actor_dead_fraction_1", "Actor low-activity fraction"), ("critic_max_abs_value", "Value magnitude"))
    fig, axes = plt.subplots(4, 1, figsize=(10, 3.2 * 4))
    for (key, title), ax in zip(keys, axes):
        if key == "critic_max_abs_value":
            key = "max_abs_value"
        for arm, run in runs.items():
            rows = [r for r in run["diagnostics"][:common] if key in r]
            ax.plot([r["steps"] for r in rows], [r[key] for r in rows], label=LABELS[arm], color=COLORS.get(arm))
        ax.set(title=title, xlabel="Training transitions")
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=7)
    figure_save(fig, root, "plasticity.png")
    probe_names = task_names(suite, probes=True)
    selected = [(arm, run) for arm, run in runs.items() if run.get("probes")]
    if selected:
        fig, axes = plt.subplots(len(selected), len(probe_names), figsize=(10 * len(probe_names), 3.2 * len(selected)), squeeze=False, sharey="col")
        for row, (arm, run) in enumerate(selected):
            for column, task in enumerate(probe_names):
                ax = axes[row, column]
                for phase, style in (("initial", "--"), ("midpoint", ":"), ("final", "-")):
                    probe = run.get("probes", {}).get(phase, {}).get(task, {})
                    points = probe.get("episode_log") or probe.get("curve", [])
                    xs, ys = smooth_series([point["env_steps"] for point in points], [point["return"] for point in points])
                    ax.plot(xs, ys, style, label=phase)
                ax.set(title=f"{LABELS[arm]} / {task}", xlabel="Isolated probe env steps", ylabel="Episodic return")
                ax.grid(alpha=0.2)
                if column == 0:
                    ax.legend(fontsize=7)
        figure_save(fig, root, "probe_curves.png")


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
    env = make_env(agent.suite, task, render_mode="rgb_array" if agent.suite == "cartpole" else None)
    frames = []
    total, steps = 0.0, 0
    done = False
    palette = np.asarray(plt.get_cmap("tab10").colors) * 255
    try:
        state, _ = env.reset(seed=seed_for(task, 500))
        generator = torch.Generator(device=agent.device).manual_seed(seed_for(task, 77))
        while not done:
            check_time(deadline)
            if steps % agent.config["eval"]["clip_stride"] == 0:
                if agent.suite == "cartpole":
                    frame = Image.fromarray(env.render()).resize((360, 240))
                else:
                    cells = state.reshape(10, 10, 10)
                    rgb = np.maximum.reduce([cells[:, :, i, None] * palette[i] for i in range(10)])
                    frame = Image.fromarray(rgb.astype(np.uint8)).resize((240, 240), Image.Resampling.NEAREST)
                ImageDraw.Draw(frame).text((5, 5), f"{LABELS[agent.arm]} | {task} | return {total:.0f}", fill="black" if agent.suite == "cartpole" else "white")
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
        xs, ys = smooth_series([point["env_steps"] for point in points], [point["return"] for point in points])
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
        normalized = [normalize(row["scores"][task]["return"], task, suite, qualification) for task in tasks]
        drops = retention_drops(run["matrix"][:common + 1])
        last_learning = {}
        for entry in run["matrix"][1:common + 1]:
            last_learning[entry["task"]] = entry["scores"].get(entry["task"], {}).get("return")
        final_forgetting = {task: last_learning[task] - row["scores"][task]["return"] for task in tasks if last_learning.get(task) is not None}
        summary[arm] = {"status": run["status"], "completed_blocks": run["completed_blocks"], "compared_blocks": common, "compared_env_steps": row["steps"], "final_returns": {task: row["scores"][task]["return"] for task in tasks}, "average_normalized_performance": float(np.mean(normalized)) if all(value is not None for value in normalized) else None, "forgetting_since_last_learning": final_forgetting, "retention_drops": drops, "probe_auc": {phase: {task: payload.get("auc") for task, payload in entries.items()} for phase, entries in run.get("probes", {}).items()}, "probe_checkpoint_blocks": run.get("probe_checkpoint_blocks", {}), "updates": run["diagnostics"][common - 1]["updates"] if common else 0}
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
    verified = len(audit) == len(runs) + len(references) and bool(runs) and all(row.get("match") for row in audit)
    outcome = "qualified demonstration" if qualification["qualified"] and complete and verified else "inconclusive demonstration"
    save_csv(root / "retention.csv", ("arm", "block", "task", "since_block", "drop", "return"),
             [{"arm": arm, **drop} for arm, data in summary.items() for drop in data["retention_drops"]] + [{"arm": "pilot_finetune", **drop} for drop in qualification.get("retention_drops", [])])
    # Tables belong in the CSVs; the JSON keeps the decision and provenance record.
    slim = {arm: {key: value for key, value in data.items() if key != "retention_drops"} for arm, data in summary.items()}
    report = {"schema": 1, "suite": suite, "seed": 0, "outcome": outcome, "qualified": qualification["qualified"], "common_completed_blocks": common, "all_runs_complete": complete, "all_checkpoints_verified": verified, "arms": slim, "qualification": {key: value for key, value in qualification.items() if key != "retention_drops"}, "provenance": manifest, "audit": audit, "clips": clips}
    save_json(root / "summary.json", report)
    save_json(root / "audit.json", audit)
    save_csv(root / "summary.csv", ("arm", "task", "final_return", "forgetting_since_last_learning", "average_normalized_performance", "compared_blocks", "compared_env_steps", "updates", "status"),
             [{"arm": arm, "task": task, "final_return": data["final_returns"][task], "forgetting_since_last_learning": data["forgetting_since_last_learning"].get(task),
               "average_normalized_performance": data["average_normalized_performance"], "compared_blocks": data["compared_blocks"],
               "compared_env_steps": data["compared_env_steps"], "updates": data["updates"], "status": data["status"]} for arm, data in summary.items() for task in tasks])
    lines = ["# CLEAR and continual backpropagation", "", f"**{outcome.capitalize()}.** Suite: **{suite}**. One seed (0), {common} matched completed blocks. No confidence intervals or statistical ranking."]
    lines.extend(experiment_notes(root, runs, summary, qualification, manifest, common))
    lines.extend(["", "## Benchmark qualification", "", f"- All pilots completed: **{qualification['all_pilots_complete']}**.", f"- Scratch and joint learnability gate: **{qualification['learnable']}**.", f"- Forgetting gate: **{qualification['forgetting_demonstrated']}**.", f"- Fresh-task plasticity-loss gate: **{qualification['plasticity_loss_demonstrated']}**.", "", "CartPole uses observable sensor permutations and actuator direction with standard physics. Its gate requires scratch and joint returns of 400 on every task, retention drops of 100 on two tasks, and late isolated-probe AUC deficits of 0.10 on two held-out tasks. A failed CartPole gate triggers MinAtar in auto mode.", "", "MinAtar requires scratch minus random to exceed max(1, 0.2 × random), and joint return to retain at least 80% of that improvement on every recurring game. Forgetting must reach 20% of the scratch improvement on two games. The Asterix probe must show a 10% normalized AUC deficit, with a scratch probe that passes the same above-random learning check. These are operational demonstration thresholds, not significance tests.", "", "| Pilot task | Random | Scratch | Joint |", "|---|---:|---:|---:|"])
    for task in tasks:
        scores = [qualification.get("random_scores", {}).get(task, {}).get("return"), qualification.get("scratch_returns", {}).get(task), qualification.get("joint_returns", {}).get(task)]
        lines.append("| " + task + " | " + " | ".join("unavailable" if value is None else f"{value:.1f}" for value in scores) + " |")
    lines.extend(["", "## Matched final returns", "", "| Arm | " + " | ".join(tasks) + " | Normalized AP |", "|---|" + "---:|" * (len(tasks) + 1)])
    for arm, data in summary.items():
        ap = data["average_normalized_performance"]
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{data['final_returns'][task]:.1f}" for task in tasks) + " | " + (f"{ap:.3f}" if ap is not None else "unavailable") + " |")
    if has_baseline:
        lines.extend(["", "![Baseline performance](baseline.png)", "", "One panel per baseline arm (scratch, joint training, fine-tuning; whichever have completed under this root), one line per task."])
    lines.extend(measurement_notes(runs, summary, tasks, common))
    lines.extend(["", "CartPole normalization is return / 500. MinAtar normalization is (return - random) / (scratch - random), without clipping; it is unavailable when scratch does not outperform random. Raw game returns remain the primary comparison.", "", "## Retention and fresh-task learning", "", "![Performance](performance.png)", "", "One panel per task, plotted against cumulative environment steps. Every task is evaluated on the same fixed step grid (`eval.period`) plus each block boundary, so all tasks and arms share one clock: episode length differs by task, which would otherwise give each task a different axis. Each point is the mean return over `eval.episodes` episodes sampled from the policy, with environment seeds derived from the evaluation point so successive measurements are independent yet reproducible. Solid stretches mark the blocks where that panel's task was being trained and dashed stretches the blocks where others were, but both are measured, not interpolated. A task's line begins at its first training block. A rise during a task's own block is acquisition; the drop across the following dashed span is what the intervening tasks cost it.", "", "The initial probe is the paired scratch reference because every arm starts with the same seeded weights. Midpoint and final probes copy weights into new, isolated learners with fresh optimizers, fresh-only updates, and no replay or CBP. Their scores test the adaptability of the learned parameters; they do not alter the main run or test the accumulated optimizer state.", "", "![Probe curves](probe_curves.png)", "", "![Plasticity diagnostics](plasticity.png)", "", "## Implementation and interpretation", "", "All arms use separate two-layer ReLU actor and critic MLPs, identical initialization, AdamCBP, V-trace targets, and equal learner batch and environment budgets. CLEAR mixes fresh and reservoir unrolls and adds KL(behavior || current) policy cloning and historical-value cloning only on replay. The replay arm disables those cloning losses. CBP resets low-contribution mature units in both networks, including optimizer moments and elementwise counters.", "", "The algorithms can be combined, but a CLEAR+CBP advantage must be observed rather than assumed. Read its retention and probe curves against both single-intervention arms. Results here describe one trajectory; the experiment does not establish a general ranking.", "", "CLEAR loss weights follow the [paper](https://arxiv.org/pdf/1811.11682). CBP uses contribution utility from the [authors' RL implementation](https://github.com/shibhansh/loss-of-plasticity), with miniature-study maturity 1,000 updates, replacement rate 0.0001, decay 0.99, and no weight decay in any arm. This is an algorithm comparison, not a reproduction of either paper's full benchmark.", "", "## Artifacts and verification", "", f"All requested runs completed: **{complete}**. All matched checkpoints re-evaluated correctly: **{verified}**.", "", "Resolved configurations, throughput, code revision, package versions, losses, replay use, replacement counts, checkpoints, and raw curves are saved alongside this report. The comparison uses the shared completed-block prefix if a run did not finish; probe checkpoints with different ages are flagged in summary.json.", "", "The study runs `cycles` passes over the task sequence with no wall-clock budget; only worker profiling is time-boxed. Concurrency and qualification are recorded in summary.json. Pilot runs, profiling, intermediate checkpoints and source snapshots live under tmp/.", ""])
    if clips:
        lines.extend(["### Representative clips", ""])
        for item in clips:
            path = Path(item["path"])
            lines.append(f"- [{path.relative_to(root).parts[0]}: {item['task']}]({path.relative_to(root)}) — return {item['return']:.0f}.")
    if not any(run.get("probes") for run in runs.values()):
        lines = [line for line in lines if line != "![Probe curves](probe_curves.png)"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report
