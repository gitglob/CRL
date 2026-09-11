# CLAUDE.md

Continual RL mini-project: CLEAR, CBP, CLEAR+CBP, fine-tuning, and replay without cloning.
The default study is an actor-critic showcase with one seed (0) and a one-hour experiment cap.
Start with contextual CartPole; switch to MinAtar when the CartPole qualification gate fails.

## Architecture

- `src/showcase/learner.py` — separate actor/critic MLPs, V-trace, cloning, checkpoint state.
- `src/showcase/replay.py` — global reservoir of complete unrolls, without task labels.
- `src/showcase/envs.py` — contextual CartPole and padded six-action MinAtar adapters.
- `src/showcase/runtime.py` — synchronous collection, evaluation, probes, bounded training.
- `src/showcase/study.py` — profiling, qualification, fallback, matched budgets, phase deadlines.
- `src/showcase/report.py` — checkpoint audit, comparisons, figures, report, clips.
- `src/cbp.py` — CBP transaction, feature probes, and Adam with elementwise counters.
- `config/showcase*.yaml` — full experiment and smoke configurations.
- `src/study.py` — default showcase dispatch; explicit old configs retain the DQN pipeline.
- `results/continual` — preserved historical DQN results; never overwrite for the new study.
- `results/showcase` — generated actor-critic experiment and qualification artifacts.

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/showcase_smoke.yaml
.venv/bin/python -m src.study
```

Install `requirements-showcase.txt` and `requirements-nn.txt` into the repository `.venv`.
The RTX 3090 is accessible outside the execution sandbox; sandbox CUDA failure is not host failure.

## Rules

- This file: at most 80 lines of at most 100 chars. Comments and docstrings: one line, at most
  100 chars. Tests enforce these limits. No type hints; use dicts rather than dataclasses.
- Never stage, commit, or push unless asked. Preserve the old results and source provenance.
- Keep one seed. No confidence intervals or statistical rankings for this mini-project.
- The experiment cap includes profiling, pilots, training, probes, evaluation, and reporting.
- Use GPU access for real training. CPU tensors are supported for numerical unit tests.
- Keep `torch.set_num_threads(1)` to prevent oversubscription across workers.
- Maintain identical initialization, batch size, and update ratio across comparison arms.
- CLEAR replay includes current-task history and uses no task labels or boundary callbacks.
- Store behavior logits, values, and true next observations before any episode reset.
- Termination disables bootstrap. Truncation bootstraps but must stop the V-trace recursion.
- Select CBP units in every layer before changing weights; zero outgoing columns last.
- Probes use isolated weight copies and fresh optimizers, with replay and CBP disabled.
- Use fixed observations for diagnostics. Stable rank uses squared singular values.
- A failed qualification gate is inconclusive. Never relabel poor returns as plasticity loss.
- Run tests and the complete smoke pipeline before launching the timed full experiment.
- Record incomplete work as incomplete; re-evaluate saved checkpoints before claiming validity.

## Historical DQN

`src/dqn.py`, `src/train.py`, and `src/utils/study.py` retain the old workflow. Old first-task
replay equivalences apply only to that workflow. `docs/legacy_dqn.md` preserves its old README.
The shared CBP replacement and rank formula are repaired; archived results remain unchanged.
The old Q-value divergence was observed but its exact causal mechanism was not isolated.
