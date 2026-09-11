# Continual RL: CLEAR, continual backpropagation, and their combination

This mini-project compares **fine-tuning**, **CBP**, **CLEAR**, and **CLEAR + CBP** on the same
actor–critic architecture. **Replay without cloning** separates CLEAR's replay and cloning
components. Scratch and joint-training pilots qualify the task suite.

The experiment uses **one seed (0)** and a **one-hour maximum**, including profiling, pilots,
training, isolated learning probes, evaluation, and reporting. It starts with contextual
CartPole and switches to MinAtar if CartPole does not demonstrate the required failure modes.

The generated [showcase report](results/showcase/REPORT.md) is the source of results. A failed
qualification gate is reported as **inconclusive**, even when all training jobs finish.

The completed seed-0 run took **20.8 minutes** on the RTX 3090. All five arms finished 60 MinAtar
blocks (1,966,080 transitions each), and all checkpoints passed re-evaluation. Both task suites
failed qualification: the fresh Asterix reference barely learned, and CLEAR+CBP failed to acquire
Freeway. The implementation is exercised, but the intended combined benefit is **not established**.

## Run

Use the repository virtual environment and a working CUDA device:

```bash
.venv/bin/python -m pip install -r requirements-showcase.txt
.venv/bin/python -m pip install -r requirements-nn.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/showcase_smoke.yaml
.venv/bin/python -m src.study
```

The default uses [config/showcase.yaml](config/showcase.yaml). Examples:

```bash
.venv/bin/python -m src.study --suite cartpole --out results/cartpole_demo
.venv/bin/python -m src.study --suite minatar --max-seconds 1800 --out results/minatar_demo
.venv/bin/python -m src.study --workers 4 --out results/four_workers
.venv/bin/python -m src.study --arm scratch --out results/scratch_references
.venv/bin/python -m src.study --arm multitask --out results/joint_reference
```

`--workers` fixes concurrency; otherwise a short 1/2/4-worker throughput and memory profile chooses
it. Learner batches and updates per transition remain fixed. CUDA can be hidden by an execution
sandbox even when the host has a GPU; run with host GPU access in that case.

`--arm` restricts the main comparison. Pilots still qualify the suite. `--overwrite` archives an
existing output directory before starting a fresh experiment. No external logging is enabled.
Public showcase runs use CUDA; numerical unit tests also exercise CPU tensors.
Scratch trains a separate fresh learner for one block per task. Joint training balances the tasks
within each rollout and receives the same total transition budget as a sequential run. These two
reference modes run separately from the five-arm comparison.

## Algorithms

Actor and critic are separate MLPs with two 256-unit ReLU hidden layers. Every arm starts with
identical weights and uses the same AdamCBP optimizer, learning rate, rollout length, and learner
batch size. There is no DQN target network in the showcase.

**CLEAR** implements [Experience Replay for Continual Learning](https://arxiv.org/pdf/1811.11682):

- V-trace actor–critic updates on a 50/50 mixture of fresh and replayed unrolls.
- A global, task-agnostic reservoir of at most 100,000 transitions, stored as complete unrolls.
- Stored behavior logits and values, observations, actions, rewards, and episode boundaries.
- Policy cloning using `KL(behavior || current)` and historical-value cloning on replay only.

Loss weights are 1 for policy gradient, 0.5 for value regression, 0.005 for entropy, 0.01 for policy
cloning, and 0.005 for value cloning. Terms are averaged over the complete learner batch, with zero
cloning loss on fresh samples. Truncations bootstrap from true final observations; termination
stops bootstrapping, and both kinds of episode boundary stop trace propagation.

**CBP** follows [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)
and the [authors' implementation](https://github.com/shibhansh/loss-of-plasticity). It tracks
contribution utility in both actor and critic, selects low-utility mature units, reinitializes
incoming weights, and zeros outgoing connections. All layers are selected before any are changed.
Replacement resets Adam moments and elementwise bias-correction counters.

Miniature-study defaults are replacement rate 0.0001, maturity 1,000 **optimizer updates**, and
decay 0.99. No arm uses weight decay. These differ from the paper's Ant/PPO configuration. The
replacement initializer matches the network's initial fan-in bound. Bias compensation preserves
the next preactivation at the old feature mean, not every output for arbitrary inputs.

**CLEAR + CBP** applies CBP after CLEAR's gradient update. They are compatible mechanisms, but
whether their combination helps is an empirical question.

## Tasks and measurements

CartPole keeps standard physics. Four tasks cross normal/reversed actions with identity/swapped
observation coordinates. The observation includes the permutation matrix and actuator direction:
this is an explicitly **contextual synthetic benchmark**, not a hidden-context control problem.
Four held-out tasks use two further permutations and both actuator directions.

MinAtar cycles **Breakout → Space Invaders → Freeway**, with **Asterix** held out for probes.
All games expose six actions and a flattened 10×10×10 binary observation; unused channels are
zero-padded. An external 2,500-step limit truncates episodes; genuine game endings are terminal.

Every recurring task is evaluated after every block. Reports retain all revisit curves and each
performance drop relative to that task's preceding learning block. AUC uses trapezoidal integration
over actual step coordinates.

Plasticity is tested at initialization, midpoint, and completion by copying weights into isolated
fresh-only learners. Probes reset optimizer state and disable CLEAR and CBP. The main learner and
its random streams remain unchanged. Initial probes are the paired scratch reference because all
arms start identically. These probes assess parameter adaptability, not optimizer aging.

Diagnostics use fixed observations. True stable rank is
`sum(singular_values**2) / max(singular_values**2)` on centered activations. The previous formula
is retained under `singular_value_participation_ratio`.

The CartPole gate requires scratch and joint returns ≥400 on every task, retention drops ≥100
on two tasks, and late-probe normalized AUC deficits ≥0.10 on two held-out tasks. MinAtar uses
random/scratch-relative gates documented in the report. These are demonstration thresholds, not
significance tests. No confidence intervals or statistical rankings are reported.
MinAtar probe deficits are distinguished from plasticity-loss evidence when the fresh reference
fails the above-random learning check.

## Artifacts and code

Each study saves resolved configs, a source archive and hash, versions, GPU/profile measurements,
qualification results, per-block metrics, full final checkpoints, and weights at every completed
block. Interrupted studies compare the common completed-block prefix. Checkpoints are re-evaluated.

Reports include learning curves, retention, performance matrices, fixed-input diagnostics, isolated
probe curves, and deterministic gameplay clips. Incomplete phases retain their incomplete status.

Regenerate figures and interpretation from saved data, without training or checkpoint evaluation:

```bash
.venv/bin/python -m src.showcase.reanalyze results/showcase
```

This preserves the original report and qualification, reuses recorded audits, and records analysis
source hashes separately from the timed run's source archive. The report identifies later revisions.

The pipeline is in [src/showcase](src/showcase), shared CBP is in [src/cbp.py](src/cbp.py), and the
new numerical and integration tests are in [tests/test_showcase.py](tests/test_showcase.py).

## Why the original results were misleading

The historical [DQN report](results/continual/REPORT.md) and its artifacts are preserved. Its replay
arm was not CLEAR. Replay+CBP predicted Q-values near 2.5 million, although +1 rewards and γ=0.99
bound true discounted returns at 100. Some scratch policies collapsed after perfect returns.
This establishes instability, not evidence against CLEAR or CBP's ability to maintain plasticity.

CBP had an adjacent-layer replacement bug, the rank diagnostic was mislabeled, and later-visit
AUCs were discarded. The task suite and preflight did not establish plasticity loss. The exact
cause of DQN divergence was not isolated; the new pipeline removes its target-network interaction.

The [original README](docs/legacy_dqn.md) retains historical documentation and its superseded
interpretation. Explicit old configs still run DQN, for example
`python -m src.study --config config/smoke.yaml`; the default now runs the actor–critic showcase.
