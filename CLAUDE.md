# CLAUDE.md

Continual RL mini-project: CLEAR, CBP, CLEAR+CBP, fine-tuning, and replay without cloning.
One actor-critic implementation, seed 0, and a one-hour maximum for the full experiment.
Start with contextual CartPole; fall back to MinAtar if qualification fails.
The current study fell back to MinAtar: five arms, 60 matched blocks, audited, inconclusive.

## Layout

- `src/clear/learner.py` — actor/critic MLPs, V-trace, cloning, diagnostics, checkpoint state.
- `src/clear/replay.py` — bounded reservoir of whole unrolls and the fresh/replay batch mixer.
- `src/cbp/` — `algorithm.py` replacement, `optimizer.py` elementwise Adam, `diagnostics.py` probes.
- `src/utils/envs.py` — contextual CartPole, padded MinAtar, task names, fixed observations, seeds.
- `src/utils/config.py` — inheritance, validation, CLI parser; `io.py` — atomic JSON/YAML, versions.
- `src/utils/runtime.py` — collection, evaluation, AUC, probe training, and the per-arm `run_job`.
- `src/utils/study.py` — profiling, pilots, qualification gates, job planning, run manifest.
- `src/utils/report.py` — figures, clips, REPORT.md, audits; `compare.py` — report regeneration.
- `src/study.py` and `src/compare.py` — thin CLI entry points.
- `config/study.yaml` and `config/smoke.yaml` — full experiment and pipeline check, by inheritance.
- `tests/` — `test_cbp.py` numerical, `test_study.py` pipeline, `test_style.py` these limits.
- `results/` — report, three figures, config, summary, audit, and one directory per arm holding
  metrics, config, final checkpoint, and videos.
- `tmp/` — ignored pilots, smoke runs, profiling, intermediate checkpoints, caches, and archives.

The DQN implementation, configs, tests, documentation, and results were explicitly removed.
Do not recreate a parallel legacy workflow or a second official results directory.

## Commands

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/smoke.yaml
.venv/bin/python -m src.study
.venv/bin/python -m src.study --suite minatar --arm scratch --out tmp/scratch
.venv/bin/python -m src.compare --out results
```

Requirements are consolidated in one file, including CUDA PyTorch and MinAtar.
The RTX 3090 is accessible outside the sandbox; sandbox CUDA failure is not host failure.
Writing into an existing output requires `--overwrite`, which archives it under `tmp/` first.
`--workers 1|2|4` fixes concurrency; otherwise throughput and memory profiling selects it.

## Rules

- This file: at most 80 lines of at most 100 chars. Comments and docstrings: one line, at most
  100 chars. Tests enforce these limits. No type hints; use dicts rather than dataclasses.
- Never stage, commit, or push unless asked. Preserve source provenance in Git history.
- Keep one seed. No confidence intervals or statistical rankings for this mini-project.
- The experiment cap includes profiling, pilots, training, probes, evaluation, and reporting.
- Every phase receives a deadline; expiry records a status instead of raising past the boundary.
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
- Scratch and joint training are standalone references: select exactly one such arm per run.
- Write metrics, summaries, and checkpoints through a temporary file, then rename.
- `src.compare` rebuilds the report from saved metrics and audits; it never trains or evaluates.
- Run tests and the complete smoke pipeline before launching a timed full experiment.
- Record incomplete work as incomplete; re-evaluate saved checkpoints before claiming validity.
- Keep diagnostics and exploratory runs under `tmp/`; `results/` is the single current study.
