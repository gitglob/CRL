# CLAUDE.md

Continual RL study: one DQN trained sequentially on four CartPole dynamics variants,
measuring Average Performance, Forward Transfer and Forgetting. Six arms: `scratch`,
`finetune`, `replay`, `cbp`, `replay_cbp`, `multitask`. Structure and style deliberately
mirror the sibling repo `/home/pangr/dev/FourRooms_v0`. Set up and verified, never run.

## Architecture

- `config/*.yaml` — nested dicts with `inherits:`. `base` is the real sweep, `smoke` a
  2k-step structural check, `preflight` the task-suite degeneracy check. Validated
  eagerly and loudly.
- `src/dqn.py` — `network()`, `DQNAgent`, Double-DQN targets, soft target update.
- `src/replay.py` — `Replay` (current-task FIFO) and `PersistentMemory` (bounded
  cross-task reservoir).
- `src/cbp.py` — `ContinualBackprop`, `FeatureProbe`, `AdamCBP` (elementwise step counter).
- `src/train.py` — `train()` over a block sequence; covers continual, scratch and
  multitask shapes.
- `src/utils/` — `envs` (the only env factory), `evaluate`, `metrics` (AP/FT/F formulas),
  `study` (orchestration + audit), `compare` (figures + REPORT.md), `config`, `io`.
- Artifacts: `results/<root>/<arm>[/<task>]/{config.yaml,metrics.json,model.pt}`, plus one
  `videos/<task>.gif` per task from the finished policy. One seed per study, so the seed is not
  in the path and `benchmark.seeds` must hold exactly one entry.

## Commands

```bash
.venv/bin/python -m pytest -q                             # tests, seconds
.venv/bin/python -m src.study --config config/smoke.yaml  # full pipeline, ~40s
.venv/bin/python -m src.study --config config/base.yaml --workers 8
```

## Rules

- This file: at most 80 lines of at most 100 chars. Comments and docstrings: one line, at most
  100 chars. Both enforced by `tests/test_core.py`.
- No type hints, dicts not dataclasses, long code lines fine, `print(f"[tag] ...", flush=True)`.
- Never stage, commit or push unless asked.
- GPU only: `resolve_device` raises without CUDA and `train.device` rejects `cpu`.
- The net is too small for the GPU to notice batch size, so prefer fewer, larger updates.
  Parallelism is `train.num_envs` slots per iteration plus `--workers` across runs.
- `train.num_envs` must divide `train.timesteps` and `eval.every`, or the eval grid drifts and
  scratch and continual AUCs stop being comparable.
- Never set physics on a gymnasium env directly — always `make_task()`. Wrappers do not forward
  attribute writes, and `total_mass`/`polemass_length` are cached in `__init__`.
- Bootstrap on `terminated` only. The 500-step limit is truncation, not termination.
- Keep `torch.set_num_threads(1)`, or parallel workers oversubscribe the cores.
- Run `config/preflight.yaml` before the sweep: if the variants don't interfere, it is
  uninformative.
- Run the tests and the smoke study before claiming a change works. They pin `replay` ==
  `finetune` on task 1, `cbp` == `replay_cbp` on task 1, and `replacement_rate: 0` == plain DQN.

## TODO

The sweep has been run: `results/continual` holds all nine runs, `REPORT.md`, five figures and
24 clips. Headline: both interventions lost to plain fine-tuning (AP 0.749 vs 0.153 replay,
0.019 cbp), which is the opposite of the hypothesis. README `## Main results` has the reading.

- [ ] `config/cbp_slow.yaml` (replacement rate 1e-5) is the first follow-up. At 1e-4 each unit is
      replaced ~20 times per run and stable rank collapses to 1.9 against 6.9 for fine-tuning, so
      the cbp arms plausibly failed from too much plasticity rather than too little.
- [ ] `config/cbp_l2.yaml` is shipped and still unrun.
- [ ] More seeds would matter most. At one seed the single-task collapse in the `scratch` baseline
      is as large as the gaps between arms, so no ranking here is established.
- [ ] `config/cycles.yaml` runs three passes over the four tasks at the same total budget, where
      revisits give plasticity loss more room to appear.
- [ ] Fine-tuning's forgetting is negative: a good `force` policy already solves `default` and
      `gravity`. Only `pole` is genuinely distinct, so the suite may be too similar to separate
      retention from transfer. A more dissimilar fourth task would sharpen the comparison.
