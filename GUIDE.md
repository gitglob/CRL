# Practical guide

## Layout

- `src/clear/`: actor–critic, V-trace and replay; `src/cbp/`: replacement and diagnostics.
- `src/utils/`: configuration, environments, runners, logging and reports.
- `config/study.yaml`: full settings; `config/smoke.yaml`: accelerated pipeline check.
- `tests/`: regression tests; `results/`: current study; `tmp/`: pilots, trials and archives.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m src.study --config config/smoke.yaml
.venv/bin/python -m src.study
.venv/bin/python -m src.compare --out results
.venv/bin/python -m src.probes --run results/cbp --out tmp/cbp_probes
```

The study uses CUDA; set `device: cpu` in a configuration for CPU execution.
Use `--config` to select settings and `--out` to select output. `src.compare` only rebuilds reports.
Study/probe outputs require `--overwrite` when nonempty; previous output is archived under `tmp/`.
`--workers 1|2|4` fixes study concurrency; otherwise profiling selects it.
`src.probes` loads saved weights without main-task training; `--phase final` selects one phase.
All three phases run by default. `probe_train` sets probe optimization; `train.probe_steps` its budget.

## Logging

| File in each run directory | Contents |
|---|---|
| `config.yaml`, `run.json` | Configuration, progress and completion metadata |
| `episodes.csv`, `evaluations.csv` | Raw training episodes and per-episode checkpoint evaluations |
| `blocks.csv`, `diagnostics.csv` | Block timing/statistics, losses, updates and replacement counts |
| `plasticity.csv` | Interaction rank, dormancy and per-layer weight magnitudes |
| `probes.csv`, `probe_summary.csv` | Every probe episode, completion, transitions and updates |
| `probe_settings.json` | Actual optimization settings for every probe phase |
| `model.pt` | Main learner state, including optimizer and replay/CBP state |
| `checkpoints/{initial,midpoint,final}.pt` | Both networks' weights, configuration and training coordinates |

Midpoint is after `max(1, blocks // 2)` blocks; every arm and baseline retains all three phases.
Other per-block weights and source snapshots remain under `tmp/` for audits.
Root `summary.json` and `audit.json` record comparisons and checkpoint checks;
`probe_refresh.json` records the separate probe rerun and its tuning evidence.
Older logs without the interaction-metric schema are marked unavailable, not reinterpreted.
