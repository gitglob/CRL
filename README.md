# CartPole continual RL: stability versus plasticity

One DQN is trained on four CartPole variants one after another, and we measure what it costs.
**Continual RL** means the agent meets its tasks in a sequence and never sees an earlier one again,
so it faces two opposite failure modes. **Catastrophic forgetting** is losing an old skill while
learning a new one; the defence is **stability**. **Loss of plasticity** is the slow decay of the
network's ability to learn anything new at all; the defence is **plasticity**. Methods that buy one
usually cost the other, which is the **stability-plasticity trade-off** this project quantifies.

Six agents are compared over five seeds: fine-tuning, persistent replay, continual backpropagation
(CBP), replay + CBP, a fresh network trained on each task alone, and one network trained on all four
variants jointly.

## Main results

**Not yet run.** This repository is set up but the sweep has not been executed - it needs a machine
with more compute than the one it was written on. Running it writes the table and every figure below.

```bash
.venv/bin/python -m src.study --config config/preflight.yaml   # confirm the suite is measurable
.venv/bin/python -m src.study --config config/base.yaml --workers 8
```

The report at `results/continual/REPORT.md` is generated, not hand-written, and contains:

| Agent | AP ↑ | FT ↑ | Forgetting ↓ | Zero-shot ↑ |
|---|---:|---:|---:|---:|
| Fine-tuning | | | | |
| Persistent replay | | | | |
| Continual backprop | | | | |
| Replay + CBP | | | | |
| Multi-task (joint) | | — | — | — |

along with `learning_curves.png`, `performance_matrix.png`, `retention.png`, `metrics.png` and
`plasticity.png`. A worked example of all of them, at a meaningless 2,000 steps per task, is what
the smoke run produces.

The three questions the study is built to answer:

* Does persistent replay reduce forgetting?
* Does CBP improve forward transfer, or at least preserve the ability to keep learning?
* Does combining them beat either alone on the stability-plasticity trade-off?

## The tasks

Four variants of `CartPole-v1`, differing only in physics. Observations, actions, rewards and the
500-step limit are identical, so one network spans the whole sequence.

| Task | gravity | pole half-length | force | what changes |
|---|---:|---:|---:|---|
| `default` | 9.8 | 0.5 | 10.0 | the reference system |
| `gravity` | 20.0 | 0.5 | 10.0 | the pole falls faster |
| `pole` | 9.8 | 0.1 | 10.0 | a short pole, so much faster dynamics |
| `force` | 9.8 | 0.5 | 4.0 | weak actuation, so less control authority |

Trained in the order `default -> gravity -> pole -> force`.

Two things are easy to get wrong here and both are guarded in code. Gymnasium's `length` is the
**half**-length, and angular acceleration scales as `1/length`, so a *longer* pole is an *easier*
task - the obvious "longer pole" variant cannot create interference, which is why task `pole` is
short instead. And gymnasium `Wrapper` objects do not forward attribute writes, so `env.gravity = 20`
silently sets a field on the `TimeLimit` wrapper and the simulation keeps running at 9.8. Every
environment comes from `make_task()`, which writes to `env.unwrapped`, recomputes the cached
`total_mass` and `polemass_length`, and then asserts the result.

## How it works

* **DQN** - a network maps a state to one value per action; the agent acts greedily on those values and learns them from experience. Here: an MLP of 4-128-128-2 with ReLU.
* **Double DQN** - the online network picks the next action and the target network scores it, which curbs the value overestimation that makes plain DQN collapse late in training.
* **Experience replay** - transitions are stored and re-sampled, so each is learned from many times.
* **Persistent replay (the stability method)** - a bounded 20k memory holding an equal share of every *earlier* task, filled by reservoir sampling throughout each task rather than snapshotted at its end. Half of every batch is drawn from it once there is an earlier task to rehearse.
* **Continual backpropagation (the plasticity method)** - continually reinitialise the least useful mature hidden units, injecting fresh variability so the network does not ossify. Utility is a unit's mean deviation from its own average activation, weighted by its outgoing weights and divided by its incoming ones. Only units older than a maturity threshold are eligible, and a fixed fraction is replaced each step.
* **Task-performance matrix** - after every stage, the agent is evaluated on all four tasks. All three metrics are read off this matrix.

### Details that change the numbers

* **Every agent clears its current-task buffer at each boundary.** Otherwise fine-tuning would carry old data too and "replay" would not be an intervention at all. Weights, optimizer state and the target network do carry over.
* **Replay rehearses only *earlier* tasks.** During task 1 there is nothing to rehearse, so the replay agent is bit-for-bit identical to fine-tuning there - a property the tests assert, and which shows up in the report as a forward transfer of exactly `0.000` on task 1.
* **Truncation is not termination.** Hitting the 500-step limit must still bootstrap; treating it as terminal teaches the agent the horizon is worthless and fights the very ceiling the metrics normalise against.
* **CBP folds a replaced unit's mean contribution into the next layer's bias before zeroing its outgoing weights**, as in the reference implementation. Without it, deleting a unit that is reliably on but low-variance - exactly what a minimum-utility rule selects - shifts every Q-value by a large constant and flips the greedy action.
* **`AdamCBP` keeps an elementwise step counter.** Stock Adam stores one scalar step per tensor, so a reinitialised unit cannot have its bias correction reset and takes up to **6.5x** the intended learning rate for its first ~15 updates, right inside the maturity window. Every agent uses this optimizer, so no arm is advantaged by a different Adam.
* **Evaluation uses common random numbers** - the same fixed episode seeds for every task, agent and seed - which makes cross-agent comparisons paired.
* **L2 regularization defaults to zero** for all agents, so CBP-versus-no-CBP is a single-variable comparison. `config/cbp_l2.yaml` enables the small L2 the CBP paper pairs with the method in its RL experiments.

## Metrics

Let `p[j][i]` be the normalized return (return / 500) on task `i` after finishing stage `j`, for `N` tasks.

```text
AP   = mean_i p[N-1][i]                                    higher is better
F_i  = p[i][i] - p[N-1][i]           F = mean_i F_i        lower  is better
FT_i = (AUC_i - AUC_i_scratch) / (1 - AUC_i_scratch)       higher is better
FT   = mean_i FT_i
```

**Average performance (AP)** is how well the agent does on everything once it is done.
**Forgetting (F)** is how much it lost. **Forward transfer (FT)** compares the area under each task's
learning curve against a fresh agent trained on that task alone, with the same budget and the same
evaluation grid. `AUC_i` averages the normalized greedy evaluations recorded while training on task `i`;
the warmup is included deliberately, because evaluation is greedy and the early points are exactly
where transfer shows itself.

Two structural zeros are worth knowing about, and the report prints both conventions rather than
picking one quietly. `F_{N-1}` is always exactly zero, because the last task's two measurements are
the same number, so averaging over all `N` understates forgetting by a factor of `N/(N-1)`. And `FT_1`
is zero by construction for any agent whose first task is unmodified - which doubles as a correctness
check that the scratch and continual protocols really are identical.

The report also gives **zero-shot** performance: the score on a task the instant before training on it
starts. It needs no baseline and no ratio, so it is the most robust transfer number here. The
**multi-task** agent is not a competitor but a ceiling: four variants share an observation space while
needing different actions, so no single network can be optimal on all of them and some measured
"forgetting" is irreducible. Without that row, an AP of 0.82 has nothing to be compared against.

## Run and regenerate

Needs Python 3.10 or newer.

```bash
uv venv
uv pip install -r requirements.txt      # install these two separately: the torch index
uv pip install -r requirements-nn.txt   # replaces the default one, and would hide gymnasium
                                        # (use requirements-cpu.txt on a machine without CUDA)

.venv/bin/python -m pytest -q                                   # 60 tests, a few seconds

.venv/bin/python -m src.study --config config/smoke.yaml        # ~90s, structure only
.venv/bin/python -m src.study --config config/preflight.yaml    # is the suite measurable?
.venv/bin/python -m src.study --config config/base.yaml --workers 8
.venv/bin/python -m src.compare --out results/continual         # rebuild figures alone
```

**Run the pre-flight before the main sweep.** It trains a fresh agent on each task and evaluates each
one on all the others. It answers two questions that are cheap to ask and expensive to discover late:
if a `default` expert already scores near 500 on the other three, the variants do not interfere and
every agent will finish at AP ≈ 1.0 with nothing to distinguish them; and if task `pole` plateaus far
below 500, it is unsolvable at this budget and should be relaxed from `length: 0.1` to `0.25`.

Individual runs, if you want one on its own:

```bash
.venv/bin/python -m src.train --arm replay_cbp --seed 0
.venv/bin/python -m src.train --arm scratch --task pole --seed 0
```

Runs are independent and the network is small, so `--workers` on CPU beats a single GPU here: at
4-128-128-2 the GPU is kernel-launch bound. Every run writes `config.yaml`, `metrics.json`,
`progress.json` (rewritten at each evaluation, so a live run is inspectable) and `model.pt`. Re-running
reuses finished runs; `--overwrite` replaces them, and a run whose config no longer matches its
artifacts is refused rather than silently mixed. Checkpoints are local artifacts and gitignored.
After every study, `audit()` reloads each checkpoint, re-evaluates it, and raises unless the numbers
reproduce exactly. W&B is opt-in via `--wandb` and can never break a run.

Where the code lives: task variants in `src/utils/envs.py`, the agent and its update in `src/dqn.py`,
the two replay buffers in `src/replay.py`, continual backpropagation and its optimizer in `src/cbp.py`,
the training loop in `src/train.py`, the metric formulas in `src/utils/metrics.py`, and the figures and
report in `src/utils/compare.py`.

## Limitations

* Five seeds against DQN's variance is not much. The report gives bootstrap confidence intervals over seeds rather than a bare mean, and they will overlap more often than not; treat close results as ties.
* One fixed task order. Forgetting depends heavily on order, so the method effect and the order effect are not separated here.
* Four task blocks is few for CBP, which is designed for plasticity loss accumulating over hundreds of task changes. If CBP shows no effect, `plasticity.png` is what distinguishes "CBP did not help" from "there was no plasticity loss to fix". `config/cycles.yaml` runs three passes over the four tasks at the same total budget, where revisits give plasticity loss more room to appear.
* The FT denominator `1 - AUC_scratch` differs per task, so tasks are weighted unequally in the FT mean and a task the baseline nearly solves has an unstable ratio. The report prints the raw AUC difference next to every ratio for this reason.
* Replay draws half of each batch from earlier tasks, so it takes 32 current-task samples per step where fine-tuning takes 64. That mixing ratio is a deliberate hyperparameter, not a controlled variable.
