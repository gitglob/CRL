# Continual RL: CLEAR and continual backpropagation

This project compares **fine-tuning**, **CBP**, **CLEAR**, **CLEAR + CBP**, and **replay without
cloning** using one common actor–critic implementation and seed 0.

The current [report](results/REPORT.md) is **inconclusive**: both benchmark suites failed
qualification. All five arms completed 60 MinAtar blocks (1,966,080 transitions each) in a
20.8-minute RTX 3090 experiment, and their checkpoints passed re-evaluation. This validates the
pipeline's execution, but does not establish the intended combined benefit.

## Repository layout

- `src/clear/`: actor–critic learning, V-trace, CLEAR cloning, and reservoir replay.
- `src/cbp/`: continual backpropagation, optimizer resets, and feature diagnostics.
- `src/utils/`: environments, configuration, study execution, evaluation, and reporting.
- `config/study.yaml`: the full experiment, capped at one hour.
- `config/smoke.yaml`: a short pipeline check, writing only under `tmp/`.
- `tests/`: numerical, algorithm, and pipeline regression tests.
- `results/`: the single current study: report, figures, configuration, summary, audit, and
  one directory per arm containing metrics, configuration, final checkpoint, and videos.
- `tmp/`: ignored smoke runs, qualification pilots, throughput profiling, intermediate
  checkpoints, source snapshots, caches, and archived output from `--overwrite`.

There is no separate `continual` or `showcase` result directory. The old DQN implementation,
its configs, tests, documentation, and results have been removed; Git history retains them.
“Preflight” meant an old DQN qualification check. “Smoke” means a tiny execution check, never
an official result. The current study's qualification outcomes are included in `results/summary.json`.

## Install and run

Use Python 3.10+ and the repository virtual environment. All dependencies, including the
CUDA 13.0 PyTorch build used on the RTX 3090, are pinned in one file:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/smoke.yaml
```

Run the full experiment with `.venv/bin/python -m src.study`. Existing results require
`--overwrite`, which archives them under `tmp/` before starting. Exploratory runs belong in `tmp/`:

```bash
.venv/bin/python -m src.study --suite cartpole --out tmp/cartpole
.venv/bin/python -m src.study --suite minatar --max-seconds 1800 --out tmp/minatar
.venv/bin/python -m src.study --arm scratch --out tmp/scratch
.venv/bin/python -m src.study --arm multitask --out tmp/joint
.venv/bin/python -m src.compare                       # regenerate the current report
.venv/bin/python -m src.compare --out tmp/minatar     # regenerate an exploratory report
```

Report regeneration uses saved metrics and audits; it performs no training or evaluation.
`--workers 1|2|4` fixes concurrency; otherwise a short throughput and memory profile selects it.
Learner batch size and updates per transition remain fixed. Real runs use CUDA; numerical tests
also work on CPU. An execution sandbox may hide CUDA even when the host GPU works.

## Algorithms

Actor and critic are separate MLPs with two 256-unit ReLU layers. Every arm receives identical
initial weights, architecture, optimizer settings, environment exposure, and learner update budgets.
Defaults are γ=0.99, learning rate 0.0003, gradient clipping at 1, 16-step unrolls, and 256-transition
learner batches. Adam uses elementwise counters so CBP can reset bias correction for replaced units.

**CLEAR**, from [Experience Replay for Continual Learning](https://arxiv.org/pdf/1811.11682), uses:

- V-trace actor–critic updates on 50% fresh and 50% replayed unrolls.
- A global reservoir capped at 100,000 transitions, with no task labels or boundary callbacks.
- Frozen behavior logits and values, actions, rewards, observations, and episode boundaries.
- `KL(behavior || current)` policy cloning and historical-value cloning on replay only.

Policy-gradient, value, entropy, policy-cloning, and value-cloning weights are respectively
1, 0.5, 0.005, 0.01, and 0.005. Losses average over the complete learner batch, with zero cloning
on fresh samples. The replay ablation disables cloning. Termination disables bootstrap;
truncation bootstraps from the true final observation. Both stop trace propagation across resets.

**CBP**, from [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7),
uses contribution utility from the [authors' RL implementation](https://github.com/shibhansh/loss-of-plasticity).
It selects low-utility mature units in all hidden layers before changing any weights. Replacement
resets incoming weights, zeros outgoing columns last, and clears affected optimizer moments and
counters. Both actor and critic receive CBP. Bias compensation preserves the next preactivation
at the old feature mean, rather than every output for arbitrary inputs.

Miniature-experiment defaults are replacement rate 0.0001, maturity 1,000 optimizer updates, and
decay 0.99. These differ from the paper's full Ant/PPO configuration. No arm uses weight decay.
CLEAR + CBP applies replacement after CLEAR's gradient update; any combined benefit must be measured.

## Tasks and measurements

The experiment starts with standard-physics CartPole. Four recurring tasks cross normal/reversed
actions with identity/swapped observation coordinates. The observation includes the permutation
matrix and actuator direction, making this a contextual synthetic benchmark. Four held-out probes
use two further permutations, each with both action mappings.

If CartPole fails qualification, the automatic fallback cycles MinAtar **Breakout → Space Invaders
→ Freeway**, reserving **Asterix** for probes. All games use six actions and flattened 10×10×10
binary observations with padded channels. An external 2,500-step limit truncates episodes.

Every recurring task is evaluated after every block. Reports retain all revisit curves, actual-step
AUCs, pre-revisit retention, and losses since the task's preceding learning block. At initialization,
midpoint, and completion, isolated learners copy each main arm's weights, reset the optimizer,
and train fresh-only on held-out tasks. Initial probes are the paired scratch reference because
initial weights and random streams are identical. Probes never alter the main learner.

Diagnostics use fixed observations and true stable rank on centered activations:
`sum(singular_values**2) / max(singular_values**2)`. The old formula is named
`singular_value_participation_ratio`.

CartPole qualification requires scratch and joint returns ≥400 on every task, retention drops
≥100 on two tasks, and late-probe normalized AUC deficits ≥0.10 on two held-out tasks. MinAtar
uses measured random/scratch references and the gates detailed in the report. Weak scratch probe
learning cannot establish plasticity loss. These are demonstration thresholds, not significance
tests; there are no confidence intervals or statistical rankings for this single seed.
