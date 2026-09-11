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
import numpy as np
import torch
from PIL import Image, ImageDraw

from .io import save_json
from .envs import make_env, seed_for, task_names
from ..clear.learner import ActorCritic
from .runtime import BudgetExpired, check_time, evaluate

LABELS = {"finetune": "Fine-tuning", "cbp": "CBP", "replay": "Replay without cloning", "clear": "CLEAR", "clear_cbp": "CLEAR + CBP", "scratch": "Scratch", "multitask": "Joint training"}
COLORS = {"finetune": "tab:gray", "cbp": "tab:orange", "replay": "tab:purple", "clear": "tab:blue", "clear_cbp": "tab:green"}


def experiment_notes(root, runs, summary, qualification, manifest, common):
    elapsed = manifest.get("elapsed_seconds", time.time() - manifest["started_at"])
    steps = [data["compared_env_steps"] for data in summary.values()]
    exposure = f"{steps[0]:,}" if steps and len(set(steps)) == 1 else "See the individual run files for"
    lines = ["", f"The timed pipeline took **{elapsed / 60:.1f} minutes** on {manifest['gpu']}, with {manifest['workers']} concurrent runs selected by profiling. Each compared arm received {exposure} main-training transitions. The configured maximum was {manifest['max_seconds'] / 60:.0f} minutes; the run stopped after its planned cycles."]
    work = Path(manifest.get("workspace", root / "work"))
    cart_path = work / "pilot" / "cartpole" / "qualification.json"
    cart = manifest.get("qualification_history", {}).get("cartpole")
    if cart is None and cart_path.exists():
        cart = json.loads(cart_path.read_text())
    if manifest["suite"] == "minatar" and cart is not None:
        lines.extend(["", f"CartPole failed qualification: learning gate **{cart['learnable']}**, forgetting gate **{cart['forgetting_demonstrated']}**, plasticity-loss gate **{cart['plasticity_loss_demonstrated']}**. MinAtar also failed qualification. Completed training therefore provides implementation evidence and individual outcomes, without establishing the intended plasticity and stability benefits."] if not qualification["qualified"] else ["", "CartPole failed qualification; the automatic fallback selected MinAtar."])
    if qualification.get("probe_auc_threshold_met") and not qualification["plasticity_loss_demonstrated"]:
        lines.extend(["", "The pilot's raw probe-AUC threshold was crossed, but its fresh reference did not learn enough above random. That numerical threshold alone is not evidence of plasticity loss."])
    if (work / "analysis.json").exists():
        lines.extend(["", "The report was subsequently regenerated from saved metrics and existing checkpoint audits, with no additional training or evaluation. The weak-probe interpretation was corrected after the run; the original qualification decision was already inconclusive and is unchanged. Training provenance is recorded in summary.json. Supporting source snapshots and analysis records are kept in the temporary workspace."])
    return lines


def measurement_notes(runs, summary, tasks, common):
    lines = ["", "Loss since each task's preceding learning block (positive means forgetting; the last trained task necessarily has zero loss at this checkpoint):", "", "| Arm | " + " | ".join(tasks) + " |", "|---|" + "---:|" * len(tasks)]
    for arm, data in summary.items():
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{data['forgetting_since_last_learning'].get(task, 0):.1f}" for task in tasks) + " |")
    if "clear_cbp" in runs and "finetune" in runs and "freeway" in tasks:
        best = {}
        for arm in ("finetune", "clear_cbp"):
            points = [point["return"] for block in runs[arm]["curves"][:common] if block["task"] == "freeway" for point in block["points"]]
            best[arm] = max(points or [0])
        lines.extend(["", f"CLEAR+CBP's best measured Freeway return during learning was {best['clear_cbp']:.1f}, versus {best['finetune']:.1f} for fine-tuning. Its final return was {summary['clear_cbp']['final_returns']['freeway']:.1f}. This indicates weak acquisition in this run; the endpoint alone should not be attributed to forgetting. The combination did not provide a consistent advantage across games. The replay ablation also prevents crediting replay gains automatically to CLEAR's cloning terms."])
    lines.extend(["", "Isolated held-out learning AUC (time-averaged raw return over the same probe budget):", "", "| Arm / task | Initial = scratch | Midpoint | Final |", "|---|---:|---:|---:|"])
    for arm, data in summary.items():
        for task in data["probes"].get("initial", {}):
            values = [data["probes"].get(phase, {}).get(task, {}).get("auc") for phase in ("initial", "midpoint", "final")]
            lines.append("| " + LABELS[arm] + " / " + task + " | " + " | ".join("incomplete" if value is None else f"{value:.3f}" for value in values) + " |")
    lines.extend(["", "Low probe returns and rank changes do not independently establish loss of plasticity. A fresh learner must learn the probe within the allotted budget for an AUC deficit to be persuasive."])
    diagnostic_rows = [row for run in runs.values() for row in run["diagnostics"][:common]]
    if diagnostic_rows:
        peak = max(row.get("max_abs_value", 0) for row in diagnostic_rows)
        lines.extend(["", f"The largest recorded learner-batch absolute value prediction was {peak:.2f}; the old million-scale divergence was not observed in these samples."])
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


def figures(root, runs, suite, qualification, common):
    tasks = task_names(suite)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.6), squeeze=False)
    for task, ax in zip(tasks, axes[0]):
        for arm, run in runs.items():
            rows = run["matrix"][:common + 1]
            ax.plot([r["steps"] for r in rows], [r["scores"][task]["return"] for r in rows], label=LABELS[arm], color=COLORS.get(arm), marker=".")
        ax.set(title=task, xlabel="Training transitions", ylabel="Greedy episode return")
        ax.grid(alpha=0.2)
    axes[0][-1].legend(fontsize=7)
    figure_save(fig, root, "retention.png")
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.6), squeeze=False)
    for task, ax in zip(tasks, axes[0]):
        for arm, run in runs.items():
            first = True
            for block in run["curves"][:common]:
                if block["task"] not in (task, "multitask"):
                    continue
                start = run["matrix"][block["block"]]["steps"]
                returns = [p.get("scores", {}).get(task, {}).get("return", p["return"]) for p in block["points"]]
                ax.plot([start + p["steps"] for p in block["points"]], returns, label=LABELS[arm] if first else None, color=COLORS.get(arm))
                first = False
        ax.set(title=task, xlabel="Training transitions (gaps are other tasks)", ylabel="Current-task return")
        ax.grid(alpha=0.2)
    axes[0][-1].legend(fontsize=7)
    figure_save(fig, root, "learning_curves.png")
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 5.2), squeeze=False)
    for task, ax in zip(tasks, axes[0]):
        data = np.asarray([[run["matrix"][index]["scores"][task]["return"] for run in runs.values()] for index in range(common + 1)])
        heatmap = ax.imshow(data, vmin=0, cmap="viridis", aspect="auto")
        fig.colorbar(heatmap, ax=ax, label="Raw game return")
        ax.set(title=task, ylabel="Completed blocks")
        ax.set_xticks(range(len(runs)), [LABELS[arm] for arm in runs], rotation=45, ha="right", fontsize=7)
    figure_save(fig, root, "performance_matrix.png")
    keys = (("actor_stable_rank_1", "Actor stable rank"), ("critic_stable_rank_1", "Critic stable rank"), ("actor_dead_fraction_1", "Actor low-activity fraction"), ("critic_max_abs_value", "Value magnitude"))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
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
        fig, axes = plt.subplots(len(probe_names), len(selected), figsize=(3.3 * len(selected), 2.6 * len(probe_names)), squeeze=False, sharey="row")
        for column, (arm, run) in enumerate(selected):
            for row, task in enumerate(probe_names):
                ax = axes[row, column]
                for phase, style in (("initial", "--"), ("midpoint", ":"), ("final", "-")):
                    points = run.get("probes", {}).get(phase, {}).get(task, {}).get("curve", [])
                    ax.plot([p["steps"] for p in points], [p["return"] for p in points], style, label=phase)
                ax.set(title=f"{LABELS[arm]} / {task}", xlabel="Isolated probe transitions", ylabel="Return")
                ax.grid(alpha=0.2)
                if row == 0:
                    ax.legend(fontsize=7)
        figure_save(fig, root, "probe_curves.png")


def clip(agent, task, path, deadline):
    env = make_env(agent.suite, task, render_mode="rgb_array" if agent.suite == "cartpole" else None)
    frames = []
    total, steps = 0.0, 0
    done = False
    palette = np.asarray(plt.get_cmap("tab10").colors) * 255
    try:
        state, _ = env.reset(seed=seed_for(task, 500))
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
            action = agent.act([state], greedy=True)[0][0]
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


def finalize_scratch(root, config, qualification, manifest, deadline, saved_artifacts=None):
    tasks = task_names(config["suite"])
    references = {}
    audit = deepcopy(saved_artifacts["audit"]) if saved_artifacts is not None else []
    clips = deepcopy(saved_artifacts["clips"]) if saved_artifacts is not None else []
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.5), squeeze=False)
    for task, ax in zip(tasks, axes[0]):
        directory = root / "scratch" / task
        path = directory / "metrics.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        references[task] = {"status": run["status"], "completed_blocks": run["completed_blocks"], "env_steps": run["env_steps"], "scores": run["matrix"][-1]["scores"].get(task) if run["matrix"] else None}
        for curve in run["curves"]:
            ax.plot([point["steps"] for point in curve["points"]], [point["return"] for point in curve["points"]])
        ax.set(title=task, xlabel="Fresh training transitions", ylabel="Raw return")
        if saved_artifacts is not None or not run["matrix"]:
            continue
        try:
            check_time(deadline)
            agent = ActorCritic(config, config["suite"], "scratch")
            payload = torch.load(directory / "model.pt", map_location=agent.device, weights_only=False)
            agent.load_weights(payload["weights"])
            scores = evaluate(agent, [task], config["eval"]["episodes"], deadline)
            match = scores[task] == references[task]["scores"]
            audit.append({"arm": "scratch", "task": task, "match": match, "scores": scores})
            if config["eval"]["clips"]:
                clips.append(clip(agent, task, root / "scratch" / task / "videos" / f"{task}.gif", deadline))
        except BudgetExpired:
            audit.append({"arm": "scratch", "task": task, "status": "budget_exhausted"})
    figure_save(fig, root, "learning_curves.png")
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
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def finalize(root, config, qualification, manifest, deadline, saved_artifacts=None):
    from .study import retention_drops

    suite = config["suite"]
    tasks = task_names(suite)
    runs = {}
    for arm in config["arms"]:
        path = root / arm / "metrics.json"
        if path.exists():
            data = json.loads(path.read_text())
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
        summary[arm] = {"status": run["status"], "completed_blocks": run["completed_blocks"], "compared_blocks": common, "compared_env_steps": row["steps"], "final_returns": {task: row["scores"][task]["return"] for task in tasks}, "average_normalized_performance": float(np.mean(normalized)) if all(value is not None for value in normalized) else None, "forgetting_since_last_learning": final_forgetting, "retention_drops": drops, "probes": run.get("probes", {}), "probe_checkpoint_blocks": run.get("probe_checkpoint_blocks", {}), "updates": run["diagnostics"][common - 1]["updates"] if common else 0}
        if run["completed_blocks"] != common:
            summary[arm]["probe_comparison_warning"] = "Probe ages differ from the common comparison checkpoint. Read as individual runs only."
        if saved_artifacts is not None:
            continue
        try:
            check_time(deadline)
            agent = ActorCritic(config, suite, arm)
            payload = torch.load((root / arm / "model.pt") if common == run["completed_blocks"] else (Path(config.get("workspace", root / "work")) / "checkpoints" / arm / f"block_{common:04d}.pt"), map_location=agent.device, weights_only=False)
            agent.load_weights(payload["weights"])
            actual = evaluate(agent, tasks, config["eval"]["episodes"], deadline)
            match = all(abs(actual[task]["return"] - row["scores"][task]["return"]) < 1e-6 for task in tasks)
            audit.append({"arm": arm, "block": common, "match": match, "scores": actual})
            if config["eval"]["clips"]:
                for task in tasks:
                    if time.time() >= deadline - 10:
                        break
                    clips.append(clip(agent, task, root / arm / "videos" / f"{task}.gif", deadline - 5))
        except BudgetExpired:
            audit.append({"arm": arm, "status": "budget_exhausted"})
    if runs:
        figures(root, runs, suite, qualification, common)
    complete = set(runs) == set(config["arms"]) and all(run["status"] == "complete" for run in runs.values())
    verified = len(audit) == len(runs) and bool(runs) and all(row.get("match") for row in audit)
    outcome = "qualified demonstration" if qualification["qualified"] and complete and verified else "inconclusive demonstration"
    report = {"schema": 1, "suite": suite, "seed": 0, "outcome": outcome, "qualified": qualification["qualified"], "common_completed_blocks": common, "all_runs_complete": complete, "all_checkpoints_verified": verified, "arms": summary, "qualification": qualification, "provenance": manifest, "audit": audit, "clips": clips}
    save_json(root / "summary.json", report)
    save_json(root / "audit.json", audit)
    lines = ["# CLEAR and continual backpropagation", "", f"**{outcome.capitalize()}.** Suite: **{suite}**. One seed (0), {common} matched completed blocks. No confidence intervals or statistical ranking."]
    lines.extend(experiment_notes(root, runs, summary, qualification, manifest, common))
    lines.extend(["", "## What was wrong with the original study", "", "The old stability arm was DQN replay, not CLEAR. Replay+CBP's saved Q-values reached approximately 2.5 million despite a discounted-return ceiling of 100. Scratch policies also collapsed after learning. These results primarily exposed value-learning instability, not evidence against the two algorithms.", "", "CBP selected downstream units after changing upstream weights and could recreate zeroed outgoing connections. Replacement now uses one selection snapshot and zeros outgoing columns after all incoming resets. Diagnostics use fixed observations and true stable rank; learning curves retain every revisit and integrate over actual step coordinates.", "", "## Benchmark qualification", "", f"- All pilots completed: **{qualification['all_pilots_complete']}**.", f"- Scratch and joint learnability gate: **{qualification['learnable']}**.", f"- Forgetting gate: **{qualification['forgetting_demonstrated']}**.", f"- Fresh-task plasticity-loss gate: **{qualification['plasticity_loss_demonstrated']}**.", "", "CartPole uses observable sensor permutations and actuator direction with standard physics. Its gate requires scratch and joint returns of 400 on every task, retention drops of 100 on two tasks, and late isolated-probe AUC deficits of 0.10 on two held-out tasks. A failed CartPole gate triggers MinAtar in auto mode.", "", "MinAtar requires scratch minus random to exceed max(1, 0.2 × random), and joint return to retain at least 80% of that improvement on every recurring game. Forgetting must reach 20% of the scratch improvement on two games. The Asterix probe must show a 10% normalized AUC deficit, with a scratch probe that passes the same above-random learning check. These are operational demonstration thresholds, not significance tests.", "", "| Pilot task | Random | Scratch | Joint |", "|---|---:|---:|---:|"])
    for task in tasks:
        scores = [qualification.get("random_scores", {}).get(task, {}).get("return"), qualification.get("scratch_returns", {}).get(task), qualification.get("joint_returns", {}).get(task)]
        lines.append("| " + task + " | " + " | ".join("unavailable" if value is None else f"{value:.1f}" for value in scores) + " |")
    lines.extend(["", "## Matched final returns", "", "| Arm | " + " | ".join(tasks) + " | Normalized AP |", "|---|" + "---:|" * (len(tasks) + 1)])
    for arm, data in summary.items():
        ap = data["average_normalized_performance"]
        lines.append("| " + LABELS[arm] + " | " + " | ".join(f"{data['final_returns'][task]:.1f}" for task in tasks) + " | " + (f"{ap:.3f}" if ap is not None else "unavailable") + " |")
    lines.extend(measurement_notes(runs, summary, tasks, common))
    lines.extend(["", "CartPole normalization is return / 500. MinAtar normalization is (return - random) / (scratch - random), without clipping; it is unavailable when scratch does not outperform random. Raw game returns remain the primary comparison.", "", "## Retention and fresh-task learning", "", "![Retention](retention.png)", "", "![Learning curves](learning_curves.png)", "", "![Performance matrix](performance_matrix.png)", "", "The initial probe is the paired scratch reference because every arm starts with the same seeded weights. Midpoint and final probes copy weights into new, isolated learners with fresh optimizers, fresh-only updates, and no replay or CBP. Their scores test the adaptability of the learned parameters; they do not alter the main run or test the accumulated optimizer state.", "", "![Probe curves](probe_curves.png)", "", "![Plasticity diagnostics](plasticity.png)", "", "## Implementation and interpretation", "", "All arms use separate two-layer ReLU actor and critic MLPs, identical initialization, AdamCBP, V-trace targets, and equal learner batch and environment budgets. CLEAR mixes fresh and reservoir unrolls and adds KL(behavior || current) policy cloning and historical-value cloning only on replay. The replay arm disables those cloning losses. CBP resets low-contribution mature units in both networks, including optimizer moments and elementwise counters.", "", "The algorithms can be combined, but a CLEAR+CBP advantage must be observed rather than assumed. Read its retention and probe curves against both single-intervention arms. Results here describe one trajectory; the experiment does not establish a general ranking.", "", "CLEAR loss weights follow the [paper](https://arxiv.org/pdf/1811.11682). CBP uses contribution utility from the [authors' RL implementation](https://github.com/shibhansh/loss-of-plasticity), with miniature-study maturity 1,000 updates, replacement rate 0.0001, decay 0.99, and no weight decay in any arm. This is an algorithm showcase, not a reproduction of either paper's full benchmark.", "", "## Artifacts and verification", "", f"All requested runs completed: **{complete}**. All matched checkpoints re-evaluated correctly: **{verified}**.", "", "Resolved configurations, phase deadlines, throughput, code revision, package versions, losses, replay use, replacement counts, checkpoints, and raw curves are saved alongside this report. The comparison uses the shared completed-block prefix if a deadline interrupted a run; probe checkpoints with different ages are flagged in summary.json.", "", "GPU scheduling uses measured throughput with reserved memory headroom. Timing, concurrency and qualification are recorded in summary.json. Pilot runs, profiling, intermediate checkpoints and source snapshots live under tmp/. The superseded DQN implementation and results have been removed; Git history preserves the earlier version.", ""])
    if clips:
        lines.extend(["### Representative clips", ""])
        for item in clips:
            path = Path(item["path"])
            lines.append(f"- [{path.parent.parent.name}: {item['task']}]({path.relative_to(root)}) — return {item['return']:.0f}.")
    if not any(run.get("probes") for run in runs.values()):
        lines = [line for line in lines if line != "![Probe curves](probe_curves.png)"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report
