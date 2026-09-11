# CLAUDE.md

Continual RL mini-project: CLEAR, CBP, CLEAR+CBP, fine-tuning, and replay without cloning.
One actor-critic implementation, seed 0, and a study sized by `cycles` and `block_steps`.
MinAtar Breakout, Space Invaders and Freeway recur; Asterix is held out for probes.

## Layout

- `src/clear/learner.py` — actor/critic MLPs, V-trace, cloning, diagnostics, checkpoint state.
- `src/clear/replay.py` — bounded reservoir of whole unrolls and the fresh/replay batch mixer.
- `src/cbp/` — `algorithm.py` replacement, `optimizer.py` elementwise Adam, `diagnostics.py` probes.
- `src/utils/envs.py` — padded MinAtar, task names, fixed observations, seeds.
- `src/utils/config.py` — inheritance, validation, CLI parser; `io.py` — atomic writes, CSV logs.
- `src/utils/runtime.py` — collection, evaluation, probe training, and the per-arm `run_job`.
- `src/utils/study.py` — profiling, pilots, qualification gates, job planning, run manifest.
- `src/utils/report.py` — figures, clips, REPORT.md, audits; `compare.py` — report regeneration.
- `src/study.py` and `src/compare.py` — thin CLI entry points.
- `config/study.yaml` and `config/smoke.yaml` — full experiment and pipeline check, by inheritance.
- `tests/` — `test_cbp.py` numerical, `test_study.py` pipeline, `test_style.py` these limits.
- `results/` — report, figures, summary, audit, and one directory per run.
- `tmp/` — ignored pilots, smoke runs, profiling, intermediate checkpoints, caches, archives.

## Logging

A run directory holds `config.yaml`, `run.json` metadata, and CSV logs: `episodes.csv`,
`evaluations.csv`, `blocks.csv`, `diagnostics.csv`, `probes.csv`, `probe_summary.csv`.
JSON and YAML carry configuration and metadata; CSV carries logged series.
Every task is evaluated every `eval.period` environment steps and at each block boundary,
so post-training scripts can plot against env steps, episodes, or task without re-running.

## Commands

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/smoke.yaml
.venv/bin/python -m src.study
.venv/bin/python -m src.study --arm scratch --out tmp/scratch
.venv/bin/python -m src.compare --out results
```

Writing into an existing output requires `--overwrite`, which archives it under `tmp/` first.
`--workers 1|2|4` fixes concurrency; otherwise throughput and memory profiling selects it.

## Rules

- This file: at most 80 lines of at most 100 chars. Comments and docstrings: one line, at most
  100 chars. Tests enforce these limits. No type hints; use dicts rather than dataclasses.
- Never stage, commit, or push unless asked.
- Keep one seed. No confidence intervals or statistical rankings.
- No wall-clock budget: `cycles` and `block_steps` fix the work. Only profiling is time-boxed.
- A job still honours a passed deadline; expiry records a status instead of raising past it.
- Keep `torch.set_num_threads(1)` to prevent oversubscription across workers.
- Maintain identical initialization, batch size, and update ratio across comparison arms.
- `block_steps` sets how long each task trains before switching; `cycles` repeats the sequence.
- The sweep also runs scratch and joint baselines into the output root at a matched budget.
- CLEAR replay includes current-task history and uses no task labels or boundary callbacks.
- Store behavior logits, values, and true next observations before any episode reset.
- Termination disables bootstrap. Truncation bootstraps but must stop the V-trace recursion.
- Select CBP units in every layer before changing weights; zero outgoing columns last.
- Probes use isolated weight copies and fresh optimizers, with replay and CBP disabled.
- Use fixed observations for diagnostics. Stable rank uses squared singular values.
- A failed qualification gate is inconclusive. Never relabel poor returns as plasticity loss.
- Write metrics, summaries, and checkpoints through a temporary file, then rename.
- `src.compare` rebuilds the report from saved logs; it never trains or evaluates.
- Run tests and the smoke pipeline before launching a timed full experiment.
- Record incomplete work as incomplete; re-evaluate saved checkpoints before claiming validity.
- Keep exploratory runs under `tmp/`; `results/` is the single current study.
- Documentation states the current design only: no past results, no superseded implementations.
