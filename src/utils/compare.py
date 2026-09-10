from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np

from .config import load_config, run_directory
from .io import save_json
from .metrics import area_under_curve, average_performance, bootstrap, forgetting, forward_transfer, zero_shot

COLORS = {"finetune": "tab:gray", "replay": "tab:blue", "cbp": "tab:orange", "replay_cbp": "tab:red", "scratch": "tab:green", "multitask": "tab:purple"}
LABELS = {"finetune": "Fine-tuning", "replay": "Persistent replay", "cbp": "Continual backprop", "replay_cbp": "Replay + CBP", "scratch": "From scratch (per task)", "multitask": "Multi-task (joint)"}
STYLES = {"finetune": "-", "replay": "--", "cbp": "-.", "replay_cbp": ":", "scratch": "-", "multitask": "--"}
SEQUENTIAL = ("finetune", "replay", "cbp", "replay_cbp")


def value_text(stat, digits=3):
    if stat["mean"] is None:
        return "—"
    if stat["n"] == 1:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} [{stat['low']:.{digits}f}, {stat['high']:.{digits}f}]"


def load_runs(root, config):
    runs, missing = [], []
    for seed in config["benchmark"]["seeds"]:
        for arm in config["benchmark"]["arms"]:
            targets = [(arm, task) for task in config["benchmark"]["order"]] if arm == "scratch" else [(arm, None)]
            for name, task in targets:
                path = run_directory(root, name, task) / "metrics.json"
                if path.exists():
                    runs.append({"arm": name, "seed": seed, "task": task, **json.loads(path.read_text())})
                else:
                    missing.append(str(path))
    return runs, missing


def scratch_reference(runs, order):
    """One shared vanilla baseline: the seed-mean AUC per task is the FT denominator."""
    reference = {}
    for task in order:
        values = [run["auc"].get(task) for run in runs if run["arm"] == "scratch" and run["task"] == task]
        values = [v for v in values if v is not None]
        if values:
            reference[task] = float(np.mean(values))
    return reference


def summarize(runs, config):
    order = config["benchmark"]["order"]
    reference = scratch_reference(runs, order)
    rows = []
    for run in runs:
        if run["arm"] == "scratch":
            continue
        rows.append({
            "arm": run["arm"], "seed": run["seed"],
            "average_performance": average_performance(run["matrix"], order),
            "forgetting": forgetting(run["matrix"], order),
            "forward_transfer": forward_transfer(run["auc"], reference, order),
            "zero_shot": zero_shot(run["jumpstart"], order),
        })
    grouped = {}
    for arm in config["benchmark"]["arms"]:
        if arm == "scratch":
            continue
        mine = [row for row in rows if row["arm"] == arm]
        if not mine:
            continue
        grouped[arm] = {
            "seeds": len(mine),
            "average_performance": bootstrap([r["average_performance"] for r in mine]),
            "forgetting": bootstrap([r["forgetting"]["mean"] for r in mine]),
            "forgetting_excluding_last": bootstrap([r["forgetting"]["mean_excluding_last"] for r in mine]),
            "forward_transfer": bootstrap([r["forward_transfer"]["mean"] for r in mine]),
            "forward_transfer_excluding_first": bootstrap([r["forward_transfer"]["mean_excluding_first"] for r in mine]),
            "zero_shot": bootstrap([r["zero_shot"]["mean_excluding_first"] for r in mine]),
            "per_task_forgetting": {task: bootstrap([r["forgetting"]["per_task"].get(task) for r in mine]) for task in order},
            "per_task_forward_transfer": {task: bootstrap([r["forward_transfer"]["per_task"].get(task) for r in mine]) for task in order},
            "per_task_delta_auc": {task: bootstrap([r["forward_transfer"]["delta_auc"].get(task) for r in mine]) for task in order},
            "per_task_zero_shot": {task: bootstrap([r["zero_shot"]["per_task"].get(task) for r in mine]) for task in order},
        }
    return {"scratch_auc": reference, "arms": grouped, "per_seed": rows}


def curve_grid(runs, arm, config):
    """Mean normalized performance over seeds on a shared step grid."""
    selected = [run for run in runs if run["arm"] == arm]
    if not selected:
        return None, None, None
    steps = np.array([point["env_steps"] for point in selected[0]["learning_curve"]], dtype=float)
    stacked = []
    for run in selected:
        values = np.array([point["normalized"] for point in run["learning_curve"]], dtype=float)
        if len(values) == len(steps):
            stacked.append(values)
    if not stacked:
        return None, None, None
    stacked = np.vstack(stacked)
    spread = stacked.std(axis=0) / np.sqrt(len(stacked)) if len(stacked) > 1 else np.zeros(stacked.shape[1])
    return steps, stacked.mean(axis=0), spread


def learning_curves(root, runs, config):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    budget = config["train"]["timesteps"]
    for arm in config["benchmark"]["arms"]:
        if arm == "scratch":
            continue
        steps, mean, spread = curve_grid(runs, arm, config)
        if steps is None:
            continue
        ax.plot(steps, mean, STYLES[arm], color=COLORS[arm], label=LABELS[arm], linewidth=1.6)
        ax.fill_between(steps, mean - spread, mean + spread, color=COLORS[arm], alpha=0.15, linewidth=0)
    blocks = [run["blocks"] for run in runs if run["arm"] in SEQUENTIAL]
    if blocks:
        for index, name in enumerate(blocks[0]):
            ax.axvline(index * budget, color="0.75", linewidth=0.8, zorder=0)
            ax.text(index * budget + budget * 0.02, 1.02, name, fontsize=8, color="0.35")
    ax.set(title="Performance on the task being trained, across the whole sequence", xlabel="Cumulative environment steps", ylabel="Normalized return")
    ax.set_ylim(-0.05, 1.12)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=0.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(root) / "learning_curves.png", dpi=150)
    plt.close(fig)


def performance_matrix(root, runs, config):
    order = config["benchmark"]["order"]
    arms = [arm for arm in config["benchmark"]["arms"] if arm != "scratch"]
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(3.1 * len(arms), 3.4), squeeze=False)
    for column, arm in enumerate(arms):
        ax = axes[0][column]
        selected = [run for run in runs if run["arm"] == arm]
        rows = [row for row in selected[0]["matrix"]] if selected else []
        grid = np.full((len(rows), len(order)), np.nan)
        for i, _ in enumerate(rows):
            for j, task in enumerate(order):
                grid[i, j] = float(np.mean([run["matrix"][i]["evaluations"][task]["normalized"] for run in selected]))
        ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows)), ["init"] + [f"after {row['task']}" for row in rows[1:]], fontsize=7)
        ax.set_title(LABELS[arm], fontsize=9)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if grid[i, j] < 0.6 else "black")
    fig.suptitle("Normalized return on every task (columns) after each stage (rows)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(root) / "performance_matrix.png", dpi=150)
    plt.close(fig)


def retention(root, runs, config):
    order = config["benchmark"]["order"]
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 3.2), squeeze=False, sharey=True)
    for column, task in enumerate(order):
        ax = axes[0][column]
        for arm in SEQUENTIAL:
            selected = [run for run in runs if run["arm"] == arm]
            if not selected:
                continue
            stages = range(len(selected[0]["matrix"]))
            values = [float(np.mean([run["matrix"][i]["evaluations"][task]["normalized"] for run in selected])) for i in stages]
            ax.plot(list(stages), values, STYLES[arm], color=COLORS[arm], marker="o", markersize=3, label=LABELS[arm], linewidth=1.4)
        ax.set(title=f"task {task}", xlabel="stage")
        ax.set_xticks(range(len(order) + 1), ["init"] + order, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if column == 0:
            ax.set_ylabel("Normalized return")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle("Retention: how each task degrades while later tasks are learned", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(Path(root) / "retention.png", dpi=150)
    plt.close(fig)


def metric_bars(root, summary, config):
    panels = [("average_performance", "Average performance (higher better)"), ("forward_transfer_excluding_first", "Forward transfer (higher better)"), ("forgetting_excluding_last", "Forgetting (lower better)")]
    arms = [arm for arm in config["benchmark"]["arms"] if arm in summary["arms"] and arm != "multitask"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.6), squeeze=False)
    for column, (key, title) in enumerate(panels):
        ax = axes[0][column]
        for position, arm in enumerate(arms):
            stat = summary["arms"][arm][key]
            if stat["mean"] is None:
                continue
            low = stat["mean"] - stat["low"] if stat["low"] is not None else 0
            high = stat["high"] - stat["mean"] if stat["high"] is not None else 0
            ax.bar(position, stat["mean"], color=COLORS[arm], yerr=[[max(low, 0)], [max(high, 0)]], capsize=3, width=0.62)
        if key == "average_performance" and "multitask" in summary["arms"]:
            ceiling = summary["arms"]["multitask"]["average_performance"]["mean"]
            if ceiling is not None:
                ax.axhline(ceiling, color=COLORS["multitask"], linestyle="--", linewidth=1.2, label="multi-task ceiling")
                ax.legend(fontsize=7)
        ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms], rotation=20, ha="right", fontsize=7.5)
        ax.set(title=title)
        ax.grid(alpha=0.2, axis="y")
        ax.axhline(0, color="0.4", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(Path(root) / "metrics.png", dpi=150)
    plt.close(fig)


def plasticity_figure(root, runs, config):
    keys = [("dead_fraction_0", "Dead units, layer 1"), ("stable_rank_0", "Stable rank, layer 1"), ("weight_norm_0", "Weight norm, layer 1")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.4), squeeze=False)
    for column, (key, title) in enumerate(keys):
        ax = axes[0][column]
        for arm in SEQUENTIAL:
            selected = [run for run in runs if run["arm"] == arm]
            if not selected:
                continue
            # The diagnostic before the first gradient step has none, so drop points not runs.
            series = [{point["env_steps"]: point[key] for point in run["plasticity"] if point.get(key) is not None} for run in selected]
            series = [entry for entry in series if entry]
            if not series:
                continue
            steps = sorted(set.intersection(*[set(entry) for entry in series]))
            if not steps:
                continue
            ax.plot(steps, [float(np.mean([entry[step] for entry in series])) for step in steps], STYLES[arm], color=COLORS[arm], label=LABELS[arm], linewidth=1.4)
        ax.set(title=title, xlabel="Cumulative environment steps")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.2)
    handles, labels = [], []
    for ax in axes[0]:
        found, names = ax.get_legend_handles_labels()
        if len(names) > len(labels):
            handles, labels = found, names
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.suptitle("Plasticity diagnostics: is the network still able to learn?", fontsize=10)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(Path(root) / "plasticity.png", dpi=150)
    plt.close(fig)


def report(root, summary, config, missing):
    order = config["benchmark"]["order"]
    arms = [arm for arm in config["benchmark"]["arms"] if arm in summary["arms"]]
    seeds = len(config["benchmark"]["seeds"])
    lines = ["# Continual RL on CartPole: stability vs plasticity", "",
             f"{seeds} seed(s); {config['train']['timesteps']:,} environment steps per task; order {' -> '.join(order)}.",
             "Cells are mean [95% bootstrap CI] over seeds.", "",
             "## Headline metrics", "",
             "| Agent | AP ↑ | FT ↑ | Forgetting ↓ | Zero-shot ↑ |", "|---|---:|---:|---:|---:|"]
    for arm in arms:
        stat = summary["arms"][arm]
        lines.append(f"| {LABELS[arm]} | {value_text(stat['average_performance'])} | {value_text(stat['forward_transfer_excluding_first'])} | {value_text(stat['forgetting_excluding_last'])} | {value_text(stat['zero_shot'])} |")
    lines += ["",
              "**AP** is the mean normalized return over all tasks once the whole sequence is finished.",
              "**FT** compares the area under each task's learning curve with a fresh agent trained on that task alone,",
              "as `(AUC - AUC_scratch) / (1 - AUC_scratch)`, averaged over tasks 2..N. Task 1 is excluded because no",
              "prior knowledge exists there, so its value is structurally zero.",
              "**Forgetting** is the drop from a task's score right after learning it to its score at the end, averaged",
              "over tasks 1..N-1; the last task is excluded because its two measurements are the same number.",
              "**Zero-shot** is the normalized return on a task the instant before training on it begins.", "",
              "### Both averaging conventions", "",
              "| Agent | F (mean over all N) | F (excluding last) | FT (mean over all N) | FT (excluding first) |", "|---|---:|---:|---:|---:|"]
    for arm in arms:
        stat = summary["arms"][arm]
        lines.append(f"| {LABELS[arm]} | {value_text(stat['forgetting'])} | {value_text(stat['forgetting_excluding_last'])} | {value_text(stat['forward_transfer'])} | {value_text(stat['forward_transfer_excluding_first'])} |")
    lines += ["", "## Per-task detail", ""]
    for key, title, digits in [("per_task_forgetting", "Forgetting by task", 3), ("per_task_forward_transfer", "Forward transfer by task", 3), ("per_task_delta_auc", "Raw AUC difference vs scratch by task", 3), ("per_task_zero_shot", "Zero-shot by task", 3)]:
        lines += [f"### {title}", "", "| Agent | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
        for arm in arms:
            lines.append(f"| {LABELS[arm]} | " + " | ".join(value_text(summary["arms"][arm][key][task], digits) for task in order) + " |")
        lines.append("")
    lines += ["## Scratch baseline", "",
              "| Task | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|",
              "| AUC | " + " | ".join(f"{summary['scratch_auc'].get(task, float('nan')):.3f}" for task in order) + " |",
              "| 1 - AUC (FT denominator) | " + " | ".join(f"{1 - summary['scratch_auc'].get(task, float('nan')):.3f}" for task in order) + " |", "",
              "A small denominator amplifies noise in that task's FT, and it differs per task, so the per-task FT",
              "table above should be read alongside the raw AUC differences.", "",
              "## Figures", "",
              "![learning curves](learning_curves.png)", "", "![task performance matrix](performance_matrix.png)", "",
              "![retention](retention.png)", "", "![metrics](metrics.png)", "", "![plasticity](plasticity.png)", ""]
    if missing:
        lines += [f"**Incomplete study:** {len(missing)} expected metrics files are missing.", ""]
    (Path(root) / "REPORT.md").write_text("\n".join(lines) + "\n")


def compare(root, config):
    root = Path(root)
    runs, missing = load_runs(root, config)
    if not runs:
        raise RuntimeError(f"No metrics found under {root}")
    summary = summarize(runs, config)
    save_json(root / "summary.json", {**summary, "missing": missing})
    learning_curves(root, runs, config)
    performance_matrix(root, runs, config)
    retention(root, runs, config)
    metric_bars(root, summary, config)
    plasticity_figure(root, runs, config)
    report(root, summary, config, missing)
    print(f"[compare] wrote {root / 'REPORT.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Build figures and REPORT.md from a finished study")
    p.add_argument("--out", type=Path, default=Path("results/continual"))
    args = p.parse_args()
    compare(args.out, load_config(args.out / "config.yaml"))


if __name__ == "__main__":
    main()
