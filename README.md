# Continual RL: CLEAR and continual backpropagation

Compares **fine-tuning**, **CBP**, **CLEAR**, **CLEAR + CBP**, and **replay without cloning**
using one common actor–critic implementation and seed 0, with **scratch** and **joint training**
as baselines. Results live in [results/REPORT.md](results/REPORT.md).

## Layout

- `src/clear/`: actor–critic learning, V-trace, CLEAR cloning, and reservoir replay.
- `src/cbp/`: continual backpropagation, optimizer resets, and feature diagnostics.
- `src/utils/`: environments, configuration, study execution, evaluation, and reporting.
- `config/study.yaml`: the full experiment, capped at one hour. `config/smoke.yaml`: a short check.
- `tests/`: numerical, algorithm, and pipeline regression tests.
- `results/`: the current study. `tmp/`: ignored scratch space.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/smoke.yaml   # pipeline check
.venv/bin/python -m src.study                              # full experiment
.venv/bin/python -m src.compare --out results              # rebuild the report, no training
```

Existing output requires `--overwrite`, which archives it under `tmp/` first. `--workers 1|2|4`
fixes concurrency; otherwise a short throughput and memory profile selects it. Real runs use
CUDA; set `device: cpu` in a config to run on CPU.

## Logging

Each run directory holds `config.yaml`, `run.json`, and CSV logs:

| File | Contents |
|---|---|
| `episodes.csv` | every training episode: task, start/end env steps, return |
| `evaluations.csv` | every evaluation episode: env steps, training episodes, task, return |
| `blocks.csv` | one row per training block: task, step range, wall time |
| `diagnostics.csv` | per-block network diagnostics |
| `probes.csv`, `probe_summary.csv` | isolated held-out probe training and its AUC |

Every task is evaluated every `eval.period` environment steps and at each block boundary, so
post-training scripts can plot against env steps, episodes, or task without re-running anything.

## Algorithms

Actor and critic are separate MLPs with two 256-unit ReLU layers. Every arm receives identical
initial weights, architecture, optimizer settings, environment exposure, and learner update
budgets. Defaults are γ=0.99, learning rate 0.0003, gradient clipping at 1, 16-step unrolls, and
256-transition learner batches. Adam uses elementwise counters so CBP can reset bias correction
for replaced units.

**CLEAR**, from [Experience Replay for Continual Learning](https://arxiv.org/pdf/1811.11682):

- V-trace actor–critic updates on 50% fresh and 50% replayed unrolls.
- A global reservoir capped at 100,000 transitions, with no task labels or boundary callbacks.
- `KL(behavior || current)` policy cloning and historical-value cloning on replay only.

Policy-gradient, value, entropy, policy-cloning, and value-cloning weights are 1, 0.5, 0.005,
0.01, and 0.005. The replay arm disables cloning. Termination disables bootstrap; truncation
bootstraps from the true final observation. Both stop trace propagation across resets.

**CBP**, from [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7),
uses contribution utility from the [authors' RL implementation](https://github.com/shibhansh/loss-of-plasticity).
It selects low-utility mature units in all hidden layers before changing any weights. Replacement
resets incoming weights, zeros outgoing columns last, and clears affected optimizer moments and
counters. Both actor and critic receive CBP. Defaults are replacement rate 0.0001, maturity 1,000
optimizer updates, and decay 0.99. No arm uses weight decay.

## Tasks and measurements

Training visits one task at a time for `train.block_steps` environment steps, repeating the
sequence `cycles` times. Joint training covers all tasks at once with a proportionally longer
block, and scratch trains each task alone; both receive the same per-task budget as the sequence.

CartPole crosses normal/reversed actions with identity/swapped observation coordinates, giving
four recurring tasks and four held-out probes; the observation includes the permutation matrix
and actuator direction. If CartPole fails qualification, the fallback cycles MinAtar
**Breakout → Space Invaders → Freeway**, reserving **Asterix** for probes, with six actions and
flattened 10×10×10 binary observations. A 2,500-step limit truncates MinAtar episodes.

At initialization, midpoint, and completion, isolated learners copy each arm's weights, reset the
optimizer, and train fresh-only on held-out tasks; probes never alter the main learner. Diagnostics
use fixed observations and true stable rank, `sum(s**2) / max(s**2)`, on centered activations.

CartPole qualification requires scratch and joint returns ≥400 on every task, retention drops ≥100
on two tasks, and late-probe normalized AUC deficits ≥0.10 on two held-out tasks. MinAtar uses
measured random and scratch references. These are demonstration thresholds, not significance
tests; there are no confidence intervals or statistical rankings for a single seed.
