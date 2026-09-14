# Continual RL: stability and plasticity

## Description

I compare five reinforcement-learning methods on a repeating sequence of games to study
**stability**—retaining previously learned skills—and **plasticity**—remaining able to learn.
The experiment uses one seed (0), shared network initialization, and matched main-training budgets.

The results below describe this run, not a statistical ranking of the methods.
See the [detailed report](results/REPORT.md) for analysis and the [short practical guide](GUIDE.md)
for installation, commands, repository layout, and logs.

## Tasks

| MinAtar game | Role | Reward |
|---|---|---|
| Breakout | Recurring task 1 | +1 per brick broken |
| Space Invaders | Recurring task 2 | +1 per alien hit |
| Freeway | Recurring task 3 | +1 per completed road crossing |
| Asterix | Unseen task used to probe learning capacity | +1 per treasure collected |

Breakout → Space Invaders → Freeway repeats five times: 15 blocks of 499,968 environment steps,
or **7,499,520 steps per arm**. Scratch baselines learn each task alone; a joint baseline learns
all three together, with the same 2,499,840 steps of exposure per task.

Both networks receive a **1,000-element binary observation**: a 10×10 grid padded to ten object
channels and flattened. Asterix uses player, enemy, trail, and treasure channels; unused channels
are zero. All games expose six actions, with no explicit task ID or frame stack.

## Metrics

| Metric | What it measures |
|---|---|
| Episodic return | Sum of rewards in an episode; main-task evaluation averages ten episodes per checkpoint |
| Normalized average performance (AP) | Average across tasks of `(return − random) / (matched scratch − random)` |
| Forgetting | Return immediately after the task's last learning block minus its current return |
| Representation stable rank | Smallest rank containing 99% of singular-value mass of uncentered final hidden activations, as % of layer width |
| Dormant units | % of units across both hidden layers active on at most 1% of the observation window |
| Average weight magnitude | Mean absolute weight, excluding biases; figures show the second hidden linear layer |

Actor and critic diagnostics use the latest 1,000 fresh interaction observations. Rank is measured
every 10,000 steps; dormancy and weights every 1,000. The rank definition follows the CBP authors'
approximate-rank measure, rather than the conventional squared-norm definition.

Isolated Asterix probes copy the initial, midpoint, or final weights and train for **5,242,880 steps**
with fresh optimizers, replay and CBP disabled. Their raw episode returns describe new-task learning;
no automatic plasticity-loss threshold is applied. **Only probe curves use Gaussian smoothing
(σ = 200 episodes); other plots are unsmoothed.** No measurements are downsampled.

## Algorithms

| Arm | Changes to the common learner |
|---|---|
| Fine-tuning | Learns only from fresh interactions |
| CBP (continual backpropagation) | Replaces mature, low-utility hidden units and resets their optimizer state |
| Replay without cloning | Mixes fresh data with reservoir replay |
| CLEAR | Adds policy and value cloning on replayed data |
| CLEAR + CBP | Combines CLEAR replay/cloning with CBP replacements |

All arms use separate actor and critic MLPs with two 256-unit ReLU layers, V-trace actor–critic
updates, and Adam with per-weight counters for CBP resets. Replay arms use 50% replay from a
100,000-transition reservoir; CBP acts on both networks.

Main training uses learning rate 0.0003, discount 0.99, and 256-transition batches. Asterix uses
learning rate 0.001 with the same discount. The methods are based on
[CLEAR](https://arxiv.org/abs/1811.11682) and [continual backpropagation](https://www.nature.com/articles/s41586-024-07711-7).

## Results

| Arm | Breakout | Space Invaders | Freeway | Normalized AP |
|---|---:|---:|---:|---:|
| Fine-tuning | 5.0 | 28.2 | 27.2 | 0.763 |
| CBP | 5.7 | 35.6 | 28.7 | 0.893 |
| Replay without cloning | 5.7 | 47.0 | 23.8 | 0.956 |
| CLEAR | 5.7 | 25.8 | 23.5 | 0.769 |
| CLEAR + CBP | 5.4 | 28.2 | 23.4 | 0.766 |

These are final evaluation returns after 15 blocks. Replay has the highest normalized AP in this
run; CBP improves all three final returns over fine-tuning. High final performance alone does not
establish that a method forgets less or learns new tasks faster.

![Performance on the three recurring tasks](results/performance.png)

Each point averages ten evaluation episodes at its actual environment-step checkpoint.
Solid segments indicate training that task; dashed segments show performance while training other
tasks. The task-switch fluctuations expose the stability problem.

![Actor and critic plasticity diagnostics](results/plasticity.png)

Rows show actor and critic; columns show rank, dormancy, and weight magnitude. At the last
measurements, CBP has fewer dormant actor units than fine-tuning (10.7% versus 27.9%) and higher
actor rank (57.0% versus 37.5%), evidence of better-preserved representations in this run.

| Arm | Initial Asterix probe | Midpoint probe | Final probe |
|---|---:|---:|---:|
| Fine-tuning | 4.92 | 3.58 | 4.06 |
| CBP | 4.92 | 3.94 | 4.11 |
| CLEAR | 4.92 | 3.78 | 3.89 |
| CLEAR + CBP | 4.92 | 3.45 | 5.03 |

Values summarize the last 1,000 completed training episodes of each probe; the initial reference
is shared. CLEAR + CBP's final probe is close to that reference, while CBP and fine-tuning finish
similarly. These summaries describe adaptation, not a separate plasticity score.

All runs completed and all nine main checkpoint audits passed. The [detailed report](results/REPORT.md)
contains smoothed probe curves, baseline comparisons, forgetting, tuning evidence, and interpretation limits.
