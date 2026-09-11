# CLAUDE.md

Continual RL mini-project: CLEAR, CBP, CLEAR+CBP, fine-tuning, and replay without cloning.
One actor-critic implementation, seed 0, and a one-hour maximum for the full experiment.
Start with contextual CartPole; fall back to MinAtar if qualification fails.

## Layout

- `src/clear/` — actor/critic MLPs, V-trace, cloning, reservoir replay, checkpoint state.
- `src/cbp/` — replacement algorithm, elementwise Adam, feature probes, rank diagnostics.
- `src/utils/` — environments, configuration, collection, study orchestration, and reporting.
- `src/study.py` and `src/compare.py` — thin CLI entry points.
- `config/study.yaml` and `config/smoke.yaml` — full experiment and pipeline check.
- `results/` — only the current report, figures, config, summary, audit, and per-arm artifacts.
- `tmp/` — ignored pilots, smoke runs, profiling, intermediate checkpoints, caches, and archives.

The DQN implementation, configs, tests, documentation, and results were explicitly removed.
Do not recreate a parallel legacy workflow or a second official results directory.

## Commands

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/smoke.yaml
.venv/bin/python -m src.study
.venv/bin/python -m src.compare
```

Requirements are consolidated in one file, including CUDA PyTorch and MinAtar.
The RTX 3090 is accessible outside the sandbox; sandbox CUDA failure is not host failure.

## Rules

- This file: at most 80 lines of at most 100 chars. Comments and docstrings: one line, at most
  100 chars. Tests enforce these limits. No type hints; use dicts rather than dataclasses.
- Never stage, commit, or push unless asked. Preserve source provenance in Git history.
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
- Run tests and the complete smoke pipeline before launching a timed full experiment.
- Record incomplete work as incomplete; re-evaluate saved checkpoints before claiming validity.
- Keep diagnostics and exploratory runs under `tmp/`; `results/` is the single current study.
