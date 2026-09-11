# CLEAR and continual backpropagation showcase

**Inconclusive demonstration.** Suite: **minatar**. One seed (0), 60 matched completed blocks. No confidence intervals or statistical ranking.

## What was wrong with the original study

The old stability arm was DQN replay, not CLEAR. Replay+CBP's saved Q-values reached approximately 2.5 million despite a discounted-return ceiling of 100. Scratch policies also collapsed after learning. These results primarily exposed value-learning instability, not evidence against the two algorithms.

CBP selected downstream units after changing upstream weights and could recreate zeroed outgoing connections. Replacement now uses one selection snapshot and zeros outgoing columns after all incoming resets. Diagnostics use fixed observations and true stable rank; learning curves retain every revisit and integrate over actual step coordinates.

## Benchmark qualification

- All pilots completed: **True**.
- Scratch and joint learnability gate: **False**.
- Forgetting gate: **False**.
- Fresh-task plasticity-loss gate: **True**.

CartPole uses observable sensor permutations and actuator direction with standard physics. Its gate requires scratch and joint returns of 400 on every task, retention drops of 100 on two tasks, and late isolated-probe AUC deficits of 0.10 on two held-out tasks. A failed CartPole gate triggers MinAtar in auto mode. MinAtar requires measurable learning above random and joint retention of 80% of the scratch improvement, plus normalized forgetting and probe deficits.

Qualification is an operational demonstration check, not a significance test. An unsuccessful pilot is retained; low performance alone is never called plasticity loss.

## Matched final returns

| Arm | breakout | space_invaders | freeway | Normalized AP |
|---|---:|---:|---:|---:|
| Fine-tuning | 4.8 | 43.2 | 11.0 | unavailable |
| CBP | 5.4 | 57.8 | 7.2 | unavailable |
| Replay without cloning | 5.6 | 58.4 | 17.8 | unavailable |
| CLEAR | 5.6 | 34.4 | 13.0 | unavailable |
| CLEAR + CBP | 5.6 | 37.4 | 0.0 | unavailable |

CartPole normalization is return / 500. MinAtar normalization is (return - random) / (scratch - random), without clipping; it is unavailable when scratch does not outperform random. Raw game returns remain the primary comparison.

## Retention and fresh-task learning

![Retention](retention.png)

![Learning curves](learning_curves.png)

![Performance matrix](performance_matrix.png)

The initial probe is the paired scratch reference because every arm starts with the same seeded weights. Midpoint and final probes copy weights into new, isolated learners with fresh optimizers, fresh-only updates, and no replay or CBP. Their scores test the adaptability of the learned parameters; they do not alter the main run or test the accumulated optimizer state.

![Probe curves](probe_curves.png)

![Plasticity diagnostics](plasticity.png)

## Implementation and interpretation

All arms use separate two-layer ReLU actor and critic MLPs, identical initialization, AdamCBP, V-trace targets, and equal learner batch and environment budgets. CLEAR mixes fresh and reservoir unrolls and adds KL(behavior || current) policy cloning and historical-value cloning only on replay. The replay arm disables those cloning losses. CBP resets low-contribution mature units in both networks, including optimizer moments and elementwise counters.

The algorithms can be combined, but a CLEAR+CBP advantage must be observed rather than assumed. Read its retention and probe curves against both single-intervention arms. Results here describe one trajectory; the experiment does not establish a general ranking.

CLEAR loss weights follow the [paper](https://arxiv.org/pdf/1811.11682). CBP uses contribution utility from the [authors' RL implementation](https://github.com/shibhansh/loss-of-plasticity), with miniature-study maturity 1,000 updates, replacement rate 0.0001, decay 0.99, and no weight decay in any arm. This is an algorithm showcase, not a reproduction of either paper's full benchmark.

## Artifacts and verification

All requested runs completed: **True**. All matched checkpoints re-evaluated correctly: **True**.

Resolved configurations, phase deadlines, throughput, code revision, package versions, losses, replay use, replacement counts, checkpoints, and raw curves are saved alongside this report. The comparison uses the shared completed-block prefix if a deadline interrupted a run; probe checkpoints with different ages are flagged in summary.json.

GPU scheduling uses measured throughput with reserved memory headroom. See profile.json and manifest.json for timing and concurrency. Historical DQN results remain in ../continual/.

### Representative clips

- [finetune: breakout](videos/finetune/breakout.gif) — return 5.
- [finetune: space_invaders](videos/finetune/space_invaders.gif) — return 23.
- [finetune: freeway](videos/finetune/freeway.gif) — return 9.
- [cbp: breakout](videos/cbp/breakout.gif) — return 5.
- [cbp: space_invaders](videos/cbp/space_invaders.gif) — return 95.
- [cbp: freeway](videos/cbp/freeway.gif) — return 1.
- [replay: breakout](videos/replay/breakout.gif) — return 6.
- [replay: space_invaders](videos/replay/space_invaders.gif) — return 95.
- [replay: freeway](videos/replay/freeway.gif) — return 27.
- [clear: breakout](videos/clear/breakout.gif) — return 6.
- [clear: space_invaders](videos/clear/space_invaders.gif) — return 69.
- [clear: freeway](videos/clear/freeway.gif) — return 0.
- [clear_cbp: breakout](videos/clear_cbp/breakout.gif) — return 6.
- [clear_cbp: space_invaders](videos/clear_cbp/space_invaders.gif) — return 23.
- [clear_cbp: freeway](videos/clear_cbp/freeway.gif) — return 0.
