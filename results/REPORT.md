# CLEAR and continual backpropagation

**Completed study.** One seed (0), 15 matched completed blocks. Plasticity conclusions are descriptive.

The pipeline took **81.3 minutes** on NVIDIA GeForce RTX 3090, with 4 concurrent runs selected by profiling. Each compared arm received 7,499,520 main-training transitions.

## Main findings

**Final performance:** Replay without cloning has the highest normalized AP in this run (0.956). This combines final scores across tasks; it does not measure forgetting or new-task learning by itself.

CBP's normalized AP is 0.893, compared with 0.763 for fine-tuning. The per-task table below shows where that difference comes from.

**Representation diagnostics:** actor dormancy is 10.7% for CBP and 27.9% for fine-tuning at their latest measurements. Rank, dormancy and weight scale describe the representation; the separate Asterix probes test whether the resulting weights can still learn.

**Interpretation:** retention, representation diagnostics and probe learning need not favor the same method. One seed and one held-out game support descriptive comparisons, not a general algorithm ranking. Poorer probe performance can reflect unfavorable transfer as well as reduced plasticity.

## Experimental protocol

The recurring sequence is Breakout → Space Invaders → Freeway. Scratch learns each game alone; joint training interleaves the three games. Asterix is held out from main training.

| Setting | Value |
|---|---|
| Compared blocks | 15 |
| Main block budget | 499,968 transitions |
| Actor and critic hidden widths | 256 → 256, ReLU |
| Main learning rate / discount | 0.0003 / 0.99 |
| Environments / unroll length / learner batch | 16 / 16 / 256 transitions |
| Evaluation | 10 episodes every 25,000 steps and at block boundaries |

Both networks take the same 1,000 binary inputs: a 10×10 grid padded to ten object channels and flattened. The policy has six action logits; the critic predicts one state value. No task ID or frame stack is supplied. Episodes use native game rewards, with an outer 2,500-step truncation limit.

Policy, value and entropy loss weights are 1, 0.5 and 0.005; gradients are clipped at norm 1.0. CLEAR mixes fresh interactions with reservoir replay and applies policy/value cloning only to replayed samples. CBP resets selected units and their Adam moments in both networks.

Replay fraction is 0.5; the reservoir holds 100,000 transitions. CLEAR's policy/value cloning weights are 0.01 and 0.005. CBP uses contribution utility, replacement rate 0.0001, maturity 1,000 optimizer updates and utility decay 0.99.

Probe copies come from these completed main-training blocks: initial = 0, midpoint = 7, final = 15. Probe environment steps restart from zero for each isolated copy.

## Pilot checks

All pilots completed: **True**. Scratch/joint learnability: **False**. Forgetting demonstrated: **True**.

These checks describe the short pilots; they do not determine study completion or provide the final normalization references. Plasticity has no automatic qualification threshold.

## Matched final returns

| Arm | breakout | space_invaders | freeway | Normalized AP |
|---|---:|---:|---:|---:|
| Fine-tuning | 5.0 | 28.2 | 27.2 | 0.763 |
| CBP | 5.7 | 35.6 | 28.7 | 0.893 |
| Replay without cloning | 5.7 | 47.0 | 23.8 | 0.956 |
| CLEAR | 5.7 | 25.8 | 23.5 | 0.769 |
| CLEAR + CBP | 5.4 | 28.2 | 23.4 | 0.766 |

Normalization is (return − random) / (scratch − random), without clipping. Scratch checkpoints come from this output root and match each task's training exposure. Missing matches or nonpositive denominators give unavailable scores; pilot scratch scores are never substituted.

| Arm | Task | Matched scratch steps | Scratch return | Random return |
|---|---|---:|---:|---:|
| Fine-tuning | breakout | 2499840 | 4.9 | 0.6 |
| Fine-tuning | space_invaders | 2499840 | 40.7 | 2.4 |
| Fine-tuning | freeway | 2499840 | 45.7 | 0.3 |
| CBP | breakout | 2499840 | 4.9 | 0.6 |
| CBP | space_invaders | 2499840 | 40.7 | 2.4 |
| CBP | freeway | 2499840 | 45.7 | 0.3 |
| Replay without cloning | breakout | 2499840 | 4.9 | 0.6 |
| Replay without cloning | space_invaders | 2499840 | 40.7 | 2.4 |
| Replay without cloning | freeway | 2499840 | 45.7 | 0.3 |
| CLEAR | breakout | 2499840 | 4.9 | 0.6 |
| CLEAR | space_invaders | 2499840 | 40.7 | 2.4 |
| CLEAR | freeway | 2499840 | 45.7 | 0.3 |
| CLEAR + CBP | breakout | 2499840 | 4.9 | 0.6 |
| CLEAR + CBP | space_invaders | 2499840 | 40.7 | 2.4 |
| CLEAR + CBP | freeway | 2499840 | 45.7 | 0.3 |

![Baseline performance](baseline.png)

Loss since each task's preceding learning block (positive means forgetting; the last trained task necessarily has zero loss at this checkpoint):

| Arm | breakout | space_invaders | freeway |
|---|---:|---:|---:|
| Fine-tuning | 0.5 | 0.6 | 0.0 |
| CBP | -0.5 | 12.2 | 0.0 |
| Replay without cloning | -0.2 | 28.2 | 0.0 |
| CLEAR | -0.2 | -4.5 | 0.0 |
| CLEAR + CBP | 0.1 | 22.4 | 0.0 |

Held-out Asterix probes plot every completed training episode against probe environment steps. Plasticity is assessed descriptively, without an AUC score or an automatic pass/fail threshold.
CBP replaced 1318 actor units and 1318 critic units in total.
CLEAR + CBP replaced 1318 actor units and 1318 critic units in total.

Recorded Asterix training settings, per arm and phase:

| Arm | Phase | Steps | Updates | Learning rate | Discount | Entropy weight |
|---|---|---:|---:|---:|---:|---:|
| Fine-tuning | initial | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| Fine-tuning | midpoint | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| Fine-tuning | final | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CBP | initial | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CBP | midpoint | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CBP | final | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CLEAR | initial | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CLEAR | midpoint | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CLEAR | final | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CLEAR + CBP | initial | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CLEAR + CBP | midpoint | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |
| CLEAR + CBP | final | 5242880 | 20480 | 0.001 | 0.99 | 0.005 |

## Asterix adaptation results

| Arm | Source phase | Completed episodes | Last episodes used | Mean return | Zero-return % |
|---|---|---:|---:|---:|---:|
| Fine-tuning | initial | 49989 | 1000 | 4.916 | 11.6 |
| Fine-tuning | midpoint | 52028 | 1000 | 3.576 | 16.3 |
| Fine-tuning | final | 53912 | 1000 | 4.059 | 13.2 |
| CBP | initial | 49989 | 1000 | 4.916 | 11.6 |
| CBP | midpoint | 52684 | 1000 | 3.939 | 13.6 |
| CBP | final | 54972 | 1000 | 4.108 | 13.5 |
| CLEAR | initial | 49989 | 1000 | 4.916 | 11.6 |
| CLEAR | midpoint | 52967 | 1000 | 3.775 | 13.5 |
| CLEAR | final | 50588 | 1000 | 3.887 | 15.0 |
| CLEAR + CBP | initial | 49989 | 1000 | 4.916 | 11.6 |
| CLEAR + CBP | midpoint | 53183 | 1000 | 3.453 | 17.1 |
| CLEAR + CBP | final | 47029 | 1000 | 5.027 | 11.2 |

These are descriptive summaries of up to the last 1,000 completed training episodes. They are not evaluation rollouts, AUCs, gain scores, or thresholds. Episode lengths vary, so these summaries do not cover identical step spans. The table uses raw returns; Gaussian smoothing affects only the plotted probe curves, retaining all original step coordinates.

The shared initial probe is the fresh reference. Midpoint and final probes start from older main-training weights, with fresh optimizers and replay/CBP disabled. This tests the capacity left by earlier training, including CBP's earlier replacements. It does not test CBP running during Asterix learning.

Zero-return Asterix episodes are valid: the agent can hit an enemy before collecting treasure. Actions are sampled, and completed episodes from vector environments are interleaved. Gaussian smoothing makes the probe trends easier to compare; it does not remove zero-return episodes from the logs.

Replay without cloning has saved phase weights but no probe curves in this study.

### Why the two learning plots look different

Performance plots average evaluation episodes at each checkpoint and only record those checkpoints periodically. Averaging repeated episodes at the same training age reduces sampling noise without smoothing across training time. Probe plots start from individual training episodes, with three phases overlaid per arm, and now apply a Gaussian filter for readability.

Probe smoothing uses a symmetric Gaussian with sigma 200 episodes, truncated at four sigma, and reflected boundaries. Filtering is separate for each arm and phase. The x coordinates remain the original environment steps; because episode lengths vary, the smoothing span in environment steps also varies. Performance and diagnostic curves remain unsmoothed.

### Reference tuning

Settings were selected on the initial reference before comparing aged checkpoints. The following completed trials retained the same seed, initialization, architecture and single-pass learner.

| Trial | Transitions | Learning rate | Discount | Entropy weight | First 1,000 mean | Last 1,000 mean | Last zero % |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr001 | 2097152 | 0.001 | 0.99 | 0.005 | 0.521 | 1.929 | 27.6 |
| lr001_5m | 5242880 | 0.001 | 0.99 | 0.005 | 0.521 | 4.916 | 11.6 |
| lr001_entropy02 | 2097152 | 0.001 | 0.99 | 0.02 | 0.470 | 2.029 | 28.2 |
| lr001_gamma095_5m | 5242880 | 0.001 | 0.95 | 0.005 | 0.557 | 3.931 | 12.3 |
| lr001_gamma09_5m | 5242880 | 0.001 | 0.9 | 0.005 | 0.513 | 3.071 | 17.1 |
| lr003_entropy01 | 2097152 | 0.003 | 0.99 | 0.01 | 0.509 | 1.414 | 38.1 |
| original_5m | 5242880 | 0.0003 | 0.99 | 0.005 | 0.492 | 4.398 | 12.9 |
| original_long | 2097152 | 0.0003 | 0.99 | 0.005 | 0.492 | 1.880 | 26.2 |

Selected reference: **lr001_5m**. Comparing budgets separately from optimizer settings shows how much improvement comes from allowing more learning. The trial summaries and source hashes are retained in probe_refresh.json.

The [official MinAtar actor-critic example](https://github.com/kenjyoung/MinAtar/blob/master/examples/AC_lambda.py) uses five million frames. [Published PPO settings](https://papers.nips.cc/paper/2024/file/09e1944b7f2372f9f81866470c59b663-Paper-Conference.pdf) use ten million transitions. Their algorithms differ from this learner; the local trials establish the behavior of the configuration used here.

The Asterix probes were rerun from the saved main-training checkpoints with the settings above. Main-training results and their checkpoint audits are unchanged. Probe rerun provenance is recorded in probe_refresh.json; its additional runtime is separate from the original study runtime.

## Retention and fresh-task learning

![Performance](performance.png)

Every task is evaluated at the shared environment-step grid and block boundaries. Each plotted measurement is the mean of the evaluation episodes at that checkpoint. Solid lines indicate training that task; dashed lines indicate training other tasks. Performance, baselines and plasticity diagnostics remain unsmoothed. Only probe curves use Gaussian smoothing; no plot is downsampled.

Initial, midpoint and final Asterix probes use isolated weight copies, fresh optimizers and fresh-only updates with replay and CBP disabled. Probe returns are Gaussian-smoothed across episode order (sigma 200 episodes, kernel truncated at four sigma, reflected boundaries) and plotted at the original environment-step coordinates. Each phase is smoothed separately; raw episode logs are unchanged. The initial probe is the shared scratch reference. No separate probe evaluation or AUC score is used.

![Probe curves](probe_curves.png)

## Plasticity diagnostics

![Plasticity diagnostics](plasticity.png)

Actor and critic occupy separate rows. Each uses the most recent interaction activations, ordered by timestep then environment index across fresh transitions. Evaluation, replay and probes do not enter the window. Windows continue across episode and task boundaries.

Stable rank is the smallest rank containing at least 99% of the uncentered final-hidden-layer singular-value mass, displayed as a percentage of layer width. Dormant units are active on at most 1% of observations, pooled across both hidden layers. Average weight magnitude is mean absolute weight excluding biases; the plot uses the second hidden linear layer, while every layer is logged.
- Fine-tuning: window 1,000 transitions; dormant/weights every 1,000; rank every 10,000.
- CBP: window 1,000 transitions; dormant/weights every 1,000; rank every 10,000.
- Replay without cloning: window 1,000 transitions; dormant/weights every 1,000; rank every 10,000.
- CLEAR: window 1,000 transitions; dormant/weights every 1,000; rank every 10,000.
- CLEAR + CBP: window 1,000 transitions; dormant/weights every 1,000; rank every 10,000.

### Last recorded diagnostic values

| Arm | Network | Rank % | Dormant % | Mean absolute weight | Rank step | Dormancy/weight step |
|---|---|---:|---:|---:|---:|---:|
| Fine-tuning | actor | 37.50 | 27.93 | 0.05405 | 7490000 | 7499000 |
| Fine-tuning | critic | 29.30 | 47.46 | 0.05901 | 7490000 | 7499000 |
| CBP | actor | 57.03 | 10.74 | 0.04196 | 7490000 | 7499000 |
| CBP | critic | 62.50 | 14.45 | 0.04554 | 7490000 | 7499000 |
| Replay without cloning | actor | 46.48 | 22.27 | 0.06672 | 7490000 | 7499000 |
| Replay without cloning | critic | 23.05 | 47.46 | 0.06776 | 7490000 | 7499000 |
| CLEAR | actor | 48.05 | 14.65 | 0.06079 | 7490000 | 7499000 |
| CLEAR | critic | 22.27 | 42.58 | 0.06080 | 7490000 | 7499000 |
| CLEAR + CBP | actor | 69.14 | 13.67 | 0.05401 | 7490000 | 7499000 |
| CLEAR + CBP | critic | 30.47 | 37.70 | 0.05546 | 7490000 | 7499000 |

Rank and the other diagnostics have different recording intervals, so the table includes their actual step coordinates. The weight column uses hidden layer 2; all linear layers are retained in plasticity.csv.

The singular values are summed without squaring or centering the activation matrix. Complete observation windows are required; an all-zero representation has rank zero. A dormant unit fires on at most ten observations in a full 1,000-observation window. Collection uses the weights present during interaction and does not run extra training forwards or feed CBP.

The rank and dormancy definitions follow the [authors' PPO diagnostics](https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/rl/run_ppo.py), extended here to the critic. The criterion is singular-value mass, not squared-energy stable rank.

## Implementation and verification

All arms retain identical actor/critic initialization, AdamCBP, V-trace, learner batches and main-training budgets. CLEAR adds reservoir replay and replay-only policy/value cloning; CBP replaces low-contribution mature units and resets their optimizer state. This is a single-seed comparison, not a statistical ranking or a full paper reproduction.

All requested runs completed: **True**. All matched checkpoints verified: **True**.

Per-arm CSV files retain raw episodes and diagnostics. summary.json records matched normalization references and probe completion/budgets. Configurations, checkpoint audits and source provenance are preserved. Pilot runs, profiling and intermediate checkpoints live in tmp/. No wall-clock budget limits the study.

### Representative clips

- [finetune: breakout](finetune/videos/breakout.gif) — return 5.
- [finetune: space_invaders](finetune/videos/space_invaders.gif) — return 25.
- [finetune: freeway](finetune/videos/freeway.gif) — return 29.
- [cbp: breakout](cbp/videos/breakout.gif) — return 5.
- [cbp: space_invaders](cbp/videos/space_invaders.gif) — return 71.
- [cbp: freeway](cbp/videos/freeway.gif) — return 24.
- [replay: breakout](replay/videos/breakout.gif) — return 5.
- [replay: space_invaders](replay/videos/space_invaders.gif) — return 34.
- [replay: freeway](replay/videos/freeway.gif) — return 9.
- [clear: breakout](clear/videos/breakout.gif) — return 5.
- [clear: space_invaders](clear/videos/space_invaders.gif) — return 38.
- [clear: freeway](clear/videos/freeway.gif) — return 21.
- [clear_cbp: breakout](clear_cbp/videos/breakout.gif) — return 5.
- [clear_cbp: space_invaders](clear_cbp/videos/space_invaders.gif) — return 35.
- [clear_cbp: freeway](clear_cbp/videos/freeway.gif) — return 28.
- [scratch: breakout](scratch/breakout/videos/breakout.gif) — return 5.
- [scratch: space_invaders](scratch/space_invaders/videos/space_invaders.gif) — return 48.
- [scratch: freeway](scratch/freeway/videos/freeway.gif) — return 49.
- [multitask: breakout](multitask/videos/breakout.gif) — return 6.
- [multitask: space_invaders](multitask/videos/space_invaders.gif) — return 137.
- [multitask: freeway](multitask/videos/freeway.gif) — return 29.

Named phase checkpoints retain both networks and their configuration under each run's checkpoints/ directory. Run isolated probes again with:

```bash
.venv/bin/python -m src.probes --run results/cbp --out tmp/cbp_probes
```

Use --phase final to select one phase, or --config to choose probe settings. All three phases run by default; main-task training is not repeated.
