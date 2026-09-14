import json


def protocol_notes(config, runs, common):
    train = config["train"]
    clear, cbp = config["clear"], config["cbp"]
    first = next(iter(runs.values()), {})
    phases = first.get("probe_checkpoint_blocks", {})
    lines = ["", "## Experimental protocol", "",
             "The recurring sequence is Breakout → Space Invaders → Freeway. Scratch learns each game alone; joint training interleaves the three games. Asterix is held out from main training.", "",
             "| Setting | Value |", "|---|---|",
             f"| Compared blocks | {common} |",
             f"| Main block budget | {train['block_steps']:,} transitions |",
             f"| Actor and critic hidden widths | {' → '.join(map(str, train['hidden_sizes']))}, ReLU |",
             f"| Main learning rate / discount | {train['learning_rate']} / {train['gamma']} |",
             f"| Environments / unroll length / learner batch | {train['num_envs']} / {train['unroll_length']} / {train['batch_unrolls'] * train['unroll_length']} transitions |",
             f"| Evaluation | {config['eval']['episodes']} episodes every {config['eval']['period']:,} steps and at block boundaries |", "",
             "Both networks take the same 1,000 binary inputs: a 10×10 grid padded to ten object channels and flattened. The policy has six action logits; the critic predicts one state value. No task ID or frame stack is supplied. Episodes use native game rewards, with an outer 2,500-step truncation limit.", "",
             f"Policy, value and entropy loss weights are 1, {train['value_weight']} and {train['entropy_weight']}; gradients are clipped at norm {train['gradient_clip']}. CLEAR mixes fresh interactions with reservoir replay and applies policy/value cloning only to replayed samples. CBP resets selected units and their Adam moments in both networks."]
    lines.extend(["", f"Replay fraction is {clear['replay_ratio']}; the reservoir holds {clear['capacity']:,} transitions. CLEAR's policy/value cloning weights are {clear['policy_cloning_weight']} and {clear['value_cloning_weight']}. CBP uses contribution utility, replacement rate {cbp['replacement_rate']}, maturity {cbp['maturity_threshold']:,} optimizer updates and utility decay {cbp['decay_rate']}."])
    if phases:
        lines.extend(["", "Probe copies come from these completed main-training blocks: " + ", ".join(f"{phase} = {block}" for phase, block in phases.items()) + ". Probe environment steps restart from zero for each isolated copy."])
    return lines


def latest_metric(run, common, key):
    cutoff = run["matrix"][common]["steps"]
    return next((row for row in reversed(run.get("plasticity", []))
                 if row.get("env_steps", float("inf")) <= cutoff and row.get(key) is not None), None)


def findings_notes(runs, summary, common, labels):
    lines = ["", "## Main findings", ""]
    scores = {arm: row["average_normalized_performance"] for arm, row in summary.items()
              if row.get("average_normalized_performance") is not None}
    if scores:
        best = max(scores, key=scores.get)
        lines.append(f"**Final performance:** {labels[best]} has the highest normalized AP in this run ({scores[best]:.3f}). This combines final scores across tasks; it does not measure forgetting or new-task learning by itself.")
    if {"cbp", "finetune"} <= scores.keys():
        lines.extend(["", f"CBP's normalized AP is {scores['cbp']:.3f}, compared with {scores['finetune']:.3f} for fine-tuning. The per-task table below shows where that difference comes from."])
    if {"cbp", "finetune"} <= runs.keys():
        rows = [latest_metric(runs[arm], common, "actor_dormant_percent") for arm in ("cbp", "finetune")]
        if all(rows):
            lines.extend(["", f"**Representation diagnostics:** actor dormancy is {rows[0]['actor_dormant_percent']:.1f}% for CBP and {rows[1]['actor_dormant_percent']:.1f}% for fine-tuning at their latest measurements. Rank, dormancy and weight scale describe the representation; the separate Asterix probes test whether the resulting weights can still learn."])
    lines.extend(["", "**Interpretation:** retention, representation diagnostics and probe learning need not favor the same method. One seed and one held-out game support descriptive comparisons, not a general algorithm ranking. Poorer probe performance can reflect unfavorable transfer as well as reduced plasticity."])
    return lines


def diagnostic_notes(runs, common, labels):
    lines = ["", "### Last recorded diagnostic values", "",
             "| Arm | Network | Rank % | Dormant % | Mean absolute weight | Rank step | Dormancy/weight step |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm, run in runs.items():
        for network in ("actor", "critic"):
            rank = latest_metric(run, common, f"{network}_stable_rank_percent")
            other = latest_metric(run, common, f"{network}_dormant_percent")
            if rank is None or other is None:
                continue
            lines.append(f"| {labels[arm]} | {network} | {rank[network + '_stable_rank_percent']:.2f} | {other[network + '_dormant_percent']:.2f} | {other[network + '_weight_magnitude_1']:.5f} | {rank['env_steps']} | {other['env_steps']} |")
    lines.extend(["", "Rank and the other diagnostics have different recording intervals, so the table includes their actual step coordinates. The weight column uses hidden layer 2; all linear layers are retained in plasticity.csv.", "",
                  "The singular values are summed without squaring or centering the activation matrix. Complete observation windows are required; an all-zero representation has rank zero. A dormant unit fires on at most ten observations in a full 1,000-observation window. Collection uses the weights present during interaction and does not run extra training forwards or feed CBP."])
    return lines


def probe_result_notes(runs, labels):
    lines = ["", "## Asterix adaptation results", "",
             "| Arm | Source phase | Completed episodes | Last episodes used | Mean return | Zero-return % |",
             "|---|---|---:|---:|---:|---:|"]
    for arm, run in runs.items():
        for phase, tasks in run.get("probes", {}).items():
            points = tasks.get("asterix", {}).get("episode_log", [])
            if not points:
                continue
            recent = points[-1000:]
            mean = sum(row["return"] for row in recent) / len(recent)
            zero = 100 * sum(row["return"] == 0 for row in recent) / len(recent)
            lines.append(f"| {labels[arm]} | {phase} | {len(points)} | {len(recent)} | {mean:.3f} | {zero:.1f} |")
    if len(lines) == 5:
        return []
    lines.extend(["", "These are descriptive summaries of up to the last 1,000 completed training episodes. They are not evaluation rollouts, AUCs, gain scores, or thresholds. Episode lengths vary, so these summaries do not cover identical step spans. The table uses raw returns; Gaussian smoothing affects only the plotted probe curves, retaining all original step coordinates.", "",
                  "The shared initial probe is the fresh reference. Midpoint and final probes start from older main-training weights, with fresh optimizers and replay/CBP disabled. This tests the capacity left by earlier training, including CBP's earlier replacements. It does not test CBP running during Asterix learning.", "",
                  "Zero-return Asterix episodes are valid: the agent can hit an enemy before collecting treasure. Actions are sampled, and completed episodes from vector environments are interleaved. Gaussian smoothing makes the probe trends easier to compare; it does not remove zero-return episodes from the logs."])
    if "replay" in runs and not runs["replay"].get("probes"):
        lines.extend(["", "Replay without cloning has saved phase weights but no probe curves in this study."])
    lines.extend(["", "### Why the two learning plots look different", "",
                  "Performance plots average evaluation episodes at each checkpoint and only record those checkpoints periodically. Averaging repeated episodes at the same training age reduces sampling noise without smoothing across training time. Probe plots start from individual training episodes, with three phases overlaid per arm, and now apply a Gaussian filter for readability.", "",
                  "Probe smoothing uses a symmetric Gaussian with sigma 200 episodes, truncated at four sigma, and reflected boundaries. Filtering is separate for each arm and phase. The x coordinates remain the original environment steps; because episode lengths vary, the smoothing span in environment steps also varies. Performance and diagnostic curves remain unsmoothed."])
    return lines


def tuning_notes(root):
    path = root / "probe_refresh.json"
    if not path.exists():
        return []
    refresh = json.loads(path.read_text())
    trials = refresh.get("tuning_trials", {})
    if not trials:
        return []
    lines = ["", "### Reference tuning", "",
             "Settings were selected on the initial reference before comparing aged checkpoints. The following completed trials retained the same seed, initialization, architecture and single-pass learner.", "",
             "| Trial | Transitions | Learning rate | Discount | Entropy weight | First 1,000 mean | Last 1,000 mean | Last zero % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, trial in trials.items():
        cfg, first, last = trial["train_settings"], trial["first1000"], trial["last1000"]
        lines.append(f"| {name} | {trial['transitions']} | {cfg['learning_rate']} | {cfg['gamma']} | {cfg['entropy_weight']} | {first['mean_return']:.3f} | {last['mean_return']:.3f} | {100 * last['zero_fraction']:.1f} |")
    lines.extend(["", f"Selected reference: **{refresh['reference']}**. Comparing budgets separately from optimizer settings shows how much improvement comes from allowing more learning. The trial summaries and source hashes are retained in probe_refresh.json.", "",
                  "The [official MinAtar actor-critic example](https://github.com/kenjyoung/MinAtar/blob/master/examples/AC_lambda.py) uses five million frames. [Published PPO settings](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf) use ten million transitions. Their algorithms differ from this learner; the local trials establish the behavior of the configuration used here."])
    return lines
