# CLAUDE.md

Continual RL study: one DQN trained sequentially on four CartPole dynamics variants,
measuring Average Performance, Forward Transfer and Forgetting. Six arms: `scratch`,
`finetune`, `replay`, `cbp`, `replay_cbp`, `multitask`. Structure and style deliberately
mirror the sibling repo `/home/pangr/dev/FourRooms_v0`.

## Architecture

- `config/*.yaml` — nested dicts with `inherits:`. `base` is the real sweep, `smoke` a 2k-step
  structural check, `preflight` the task-suite degeneracy check. Validated eagerly and loudly.
- `src/dqn.py` — `network()`, `DQNAgent`, Double-DQN targets, soft target update.
- `src/replay.py` — `Replay` (current-task FIFO) and `PersistentMemory` (bounded cross-task reservoir).
- `src/cbp.py` — `ContinualBackprop`, `FeatureProbe`, `AdamCBP` (elementwise step counter).
- `src/train.py` — `train()` over a block sequence; covers continual, scratch and multitask shapes.
- `src/utils/` — `envs` (the only env factory), `evaluate`, `metrics` (AP/FT/F formulas),
  `study` (orchestration + audit), `compare` (figures + REPORT.md), `config`, `io`.
- Artifacts: `results/<root>/<arm>[/<task>]/seed<N>/{config.yaml,metrics.json,model.pt}`.

## Commands

```bash
.venv/bin/python -m pytest -q                             # 60 tests, seconds
.venv/bin/python -m src.study --config config/smoke.yaml  # full pipeline, ~40s
.venv/bin/python -m src.study --config config/base.yaml --workers 8
```

## Rules

- Match the existing style: **no type hints**, dicts not dataclasses, long lines are fine,
  rare one-line docstrings that state an invariant, `print(f"[tag] ...", flush=True)` for logging.
- Never set physics on a gymnasium env directly — always go through `make_task()`. Wrappers do
  not forward attribute writes, and `total_mass`/`polemass_length` are cached in `__init__`.
- Bootstrap on `terminated` only. The 500-step limit is truncation, not termination.
- Scratch and continual runs must share an identical eval grid or their AUCs are incomparable.
- Keep `torch.set_num_threads(1)`. Without it parallel workers oversubscribe the cores and a
  study takes an order of magnitude longer.
- Don't launch `config/base.yaml` unprompted: ~10M env steps and hours of compute. Run
  `config/preflight.yaml` first — if the variants don't interfere, the whole study is uninformative.
- Run the tests and the smoke study before claiming a change works. They pin the equalities the
  whole comparison rests on: `replay` is bit-identical to `finetune` on task 1, `cbp` to
  `replay_cbp` on task 1, and `replacement_rate: 0` to plain DQN.
