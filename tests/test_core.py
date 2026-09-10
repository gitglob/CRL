from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import numpy as np
import pytest

from src.replay import PersistentMemory, Replay
from src.utils.config import PHYSICS, load_config, run_directory, task_order, validate
from src.utils.envs import make_task, verify_task
from src.utils.metrics import area_under_curve, average_performance, bootstrap, forgetting, forward_transfer, zero_shot


@pytest.fixture
def config():
    return load_config()


def test_every_shipped_config_loads_and_validates():
    for name in ["base", "smoke", "preflight", "cycles", "cbp_slow", "cbp_l2"]:
        validate(load_config(f"config/{name}.yaml"))


def test_validate_rejects_a_task_missing_physics_parameters(config):
    del config["tasks"]["gravity"]["force_mag"]
    with pytest.raises(ValueError, match="missing physics"):
        validate(config)


def test_validate_rejects_several_seeds(config):
    # Run directories no longer carry the seed, so a second seed would overwrite the first.
    config["benchmark"]["seeds"] = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="exactly one seed"):
        validate(config)


def test_validate_accepts_a_single_seed(config):
    config["benchmark"]["seeds"] = [3]
    validate(config)


def test_validate_rejects_the_cpu(config):
    config["train"]["device"] = "cpu"
    with pytest.raises(ValueError, match="CPU is not an option"):
        validate(config)


def test_validate_rejects_a_num_envs_that_breaks_the_eval_grid(config):
    config["train"]["num_envs"] = 3
    with pytest.raises(ValueError, match="must divide"):
        validate(config)


def test_run_directory_omits_the_seed():
    assert run_directory("results/x", "finetune") == Path("results/x/finetune")
    assert run_directory("results/x", "scratch", "pole") == Path("results/x/scratch/pole")


def test_cycles_repeat_the_whole_task_order(config):
    config["benchmark"]["cycles"] = 3
    assert task_order(config) == config["benchmark"]["order"] * 3


def test_task_variants_reach_the_unwrapped_environment(config):
    for name in config["benchmark"]["order"]:
        env = make_task(config, name)
        for key in PHYSICS:
            assert getattr(env.unwrapped, key) == pytest.approx(config["tasks"][name][key])
        env.close()


def test_setting_physics_on_the_wrapper_does_not_reach_the_simulation(config):
    # The bug make_task exists to prevent: gymnasium wrappers do not forward attribute writes.
    env = make_task(config, "default")
    env.gravity = 999.0
    assert env.unwrapped.gravity == pytest.approx(9.8)
    verify_task(env, config, "default")
    env.close()


def test_derived_masses_are_recomputed_for_every_variant(config):
    for name in config["benchmark"]["order"]:
        env = make_task(config, name)
        core = env.unwrapped
        assert core.total_mass == pytest.approx(core.masspole + core.masscart)
        assert core.polemass_length == pytest.approx(core.masspole * core.length)
        env.close()


def test_verify_task_catches_a_stale_derived_mass(config):
    env = make_task(config, "pole")
    env.unwrapped.polemass_length = 0.05
    with pytest.raises(RuntimeError, match="derived masses"):
        verify_task(env, config, "pole")
    env.close()


def test_variants_actually_produce_different_dynamics(config):
    trajectories = {}
    for name in config["benchmark"]["order"]:
        env = make_task(config, name)
        state, _ = env.reset(seed=0)
        for _ in range(8):
            state, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        trajectories[name] = np.asarray(state)
        env.close()
    reference = trajectories[config["benchmark"]["order"][0]]
    for name in config["benchmark"]["order"][1:]:
        assert not np.allclose(reference, trajectories[name]), f"{name} behaves identically to the default task"


def test_replay_ring_buffer_overwrites_oldest_and_reports_its_size():
    buffer = Replay(4, 2)
    for step in range(6):
        buffer.add(np.full(2, step, dtype=np.float32), step % 2, float(step), np.zeros(2, dtype=np.float32), False)
    assert buffer.size == 4 and buffer.total == 6
    assert sorted(buffer.data["reward"].tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_clearing_the_replay_buffer_empties_it_for_the_next_task():
    buffer = Replay(8, 2)
    for step in range(5):
        buffer.add(np.zeros(2, dtype=np.float32), 0, 1.0, np.zeros(2, dtype=np.float32), False)
    buffer.clear()
    assert buffer.size == 0 and buffer.total == 0


def test_persistent_memory_holds_an_equal_share_of_every_task_seen():
    rng = np.random.default_rng(0)
    memory = PersistentMemory(120, 2)
    for index, task in enumerate(["a", "b", "c"]):
        memory.begin(rng, task)
        for step in range(500):
            memory.absorb(rng, task, np.full(2, index, dtype=np.float32), 0, float(step), np.zeros(2, dtype=np.float32), False)
    sizes = {task: entry["size"] for task, entry in memory.tasks.items()}
    assert sizes == {"a": 40, "b": 40, "c": 40}
    assert memory.size <= memory.capacity


def test_persistent_memory_samples_across_the_whole_task_not_only_its_tail():
    # Reservoir sampling, not an end-of-task snapshot: early transitions must survive.
    rng = np.random.default_rng(1)
    memory = PersistentMemory(50, 2)
    memory.begin(rng, "a")
    for step in range(5000):
        memory.absorb(rng, "a", np.zeros(2, dtype=np.float32), 0, float(step), np.zeros(2, dtype=np.float32), False)
    kept = memory.tasks["a"]["data"]["reward"][: memory.tasks["a"]["size"]]
    assert kept.min() < 1000, "memory only retained the end of the task"
    assert kept.max() > 4000


def test_persistent_memory_sampling_covers_every_task():
    rng = np.random.default_rng(2)
    memory = PersistentMemory(60, 2)
    for index, task in enumerate(["a", "b"]):
        memory.begin(rng, task)
        for _ in range(100):
            memory.absorb(rng, task, np.full(2, index, dtype=np.float32), 0, 0.0, np.zeros(2, dtype=np.float32), False)
    batch = memory.sample(rng, 400)
    seen = set(batch["state"][:, 0].tolist())
    assert seen == {0.0, 1.0}


def _matrix(order, retained):
    rows = [{"block": None, "task": None, "evaluations": {t: {"normalized": 0.05} for t in order}}]
    for index, task in enumerate(order):
        scores = {}
        for other in order:
            if other == task:
                scores[other] = 1.0
            elif order.index(other) < index:
                scores[other] = retained
            else:
                scores[other] = 0.1
        rows.append({"block": index, "task": task, "evaluations": {t: {"normalized": v} for t, v in scores.items()}})
    return rows


def test_forgetting_is_zero_when_every_task_is_fully_retained():
    order = ["a", "b", "c", "d"]
    result = forgetting(_matrix(order, retained=1.0), order)
    assert result["mean"] == pytest.approx(0.0)
    assert result["mean_excluding_last"] == pytest.approx(0.0)


def test_forgetting_excludes_the_last_task_whose_value_is_structurally_zero():
    order = ["a", "b", "c", "d"]
    result = forgetting(_matrix(order, retained=0.2), order)
    assert result["per_task"]["d"] == pytest.approx(0.0)
    assert result["mean"] == pytest.approx(0.6)
    assert result["mean_excluding_last"] == pytest.approx(0.8)


def test_average_performance_reads_the_final_row():
    order = ["a", "b", "c", "d"]
    assert average_performance(_matrix(order, retained=0.5), order) == pytest.approx((0.5 * 3 + 1.0) / 4)


def test_forward_transfer_is_zero_when_the_curve_matches_scratch():
    order = ["a", "b"]
    result = forward_transfer({"a": 0.4, "b": 0.7}, {"a": 0.4, "b": 0.7}, order)
    assert result["mean"] == pytest.approx(0.0)
    assert result["delta_auc"] == {"a": pytest.approx(0.0), "b": pytest.approx(0.0)}


def test_forward_transfer_is_one_when_the_curve_is_perfect():
    order = ["a", "b"]
    assert forward_transfer({"a": 1.0, "b": 1.0}, {"a": 0.3, "b": 0.8}, order)["mean"] == pytest.approx(1.0)


def test_forward_transfer_excludes_the_first_task_from_its_mean():
    order = ["a", "b", "c"]
    result = forward_transfer({"a": 0.5, "b": 0.9, "c": 0.9}, {"a": 0.5, "b": 0.5, "c": 0.5}, order)
    assert result["per_task"]["a"] == pytest.approx(0.0)
    assert result["mean_excluding_first"] == pytest.approx(0.8)
    assert result["mean"] == pytest.approx(0.8 * 2 / 3)


def test_area_under_curve_averages_only_its_own_block():
    curve = [{"block": 0, "normalized": 0.0}, {"block": 0, "normalized": 1.0}, {"block": 1, "normalized": 0.5}]
    assert area_under_curve(curve, 0) == pytest.approx(0.5)
    assert area_under_curve(curve, 1) == pytest.approx(0.5)
    assert area_under_curve(curve, 2) is None


def test_zero_shot_keeps_the_first_visit_and_drops_the_first_task():
    order = ["a", "b", "c"]
    records = [{"block": 0, "task": "a", "normalized": 0.02}, {"block": 1, "task": "b", "normalized": 0.6}, {"block": 2, "task": "c", "normalized": 0.4}]
    result = zero_shot(records, order)
    assert result["per_task"]["a"] == pytest.approx(0.02)
    assert result["mean_excluding_first"] == pytest.approx(0.5)


def test_bootstrap_reports_a_degenerate_interval_for_one_seed():
    assert bootstrap([0.5]) == {"mean": 0.5, "low": 0.5, "high": 0.5, "n": 1}


def test_bootstrap_ignores_missing_values():
    assert bootstrap([None, None])["n"] == 0


# --- CLAUDE.md stays short enough to actually be read -----------------


def test_claude_md_respects_its_own_length_and_width_rules():
    # CLAUDE.md pins its own limits so the 80/100 numbers cannot drift between the two files.
    path = Path(__file__).resolve().parents[1] / "CLAUDE.md"
    lines = path.read_text().splitlines()
    assert len(lines) <= 80, f"CLAUDE.md has {len(lines)} lines, limit is 80"
    too_long = [(n, len(line)) for n, line in enumerate(lines, 1) if len(line) > 100]
    assert not too_long, f"CLAUDE.md lines over 100 chars: {too_long}"


def docstrings(source, tree):
    """Yields the line number, raw text and first source line of every docstring in a tree."""
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                yield node.body[0].lineno, text, lines[node.body[0].lineno - 1]


def test_comments_and_docstrings_are_one_short_line():
    # CLAUDE.md pins the rule; walking the tree here keeps the two from drifting apart.
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))):
        source = path.read_text()
        previous = None
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            alone = token.line.lstrip().startswith("#")
            if len(token.line.rstrip()) > 100:
                offenders.append(f"{path.name}:{token.start[0]} comment over 100 chars")
            if alone and previous is not None and token.start[0] == previous + 1:
                offenders.append(f"{path.name}:{token.start[0]} comment block should be one line")
            previous = token.start[0] if alone else None
        for number, text, first in docstrings(source, ast.parse(source)):
            if len(text.strip("\n").split("\n")) > 1:
                offenders.append(f"{path.name}:{number} docstring spans several lines")
            elif len(first.rstrip()) > 100:
                offenders.append(f"{path.name}:{number} docstring over 100 chars")
    assert not offenders, "one line of at most 100 chars, please:\n" + "\n".join(offenders)
