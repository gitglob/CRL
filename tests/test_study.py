from copy import deepcopy
import time

import numpy as np
import pytest
import torch

from src.cbp import AdamCBP, FeatureProbe, attach, plasticity
from src.utils.config import validate
from src.utils.envs import fixed_observations, make_env, task_names
from src.clear.learner import ActorCritic, cloning_losses, load_agent, mlp, vtrace
from src.clear.replay import UnrollReservoir, mix_batch
from src.utils.runtime import BudgetExpired, Collector, auc, evaluate, learn_rollout, probe_tasks, run_job
from src.utils.study import qualification, retention_drops
from src.utils.config import load_config


@pytest.fixture
def config():
    result = load_config("config/smoke.yaml")
    result["device"] = "cpu"
    result["train"]["hidden_sizes"] = [8, 8]
    result["train"]["num_envs"] = 4
    result["train"]["batch_unrolls"] = 4
    result["train"]["unroll_length"] = 4
    result["train"]["block_steps"] = 32
    result["train"]["probe_steps"] = 16
    result["clear"]["capacity"] = 32
    result["eval"]["clips"] = False
    return result


def test_vtrace_matches_two_step_on_policy_return_and_detaches():
    reward = torch.tensor([[1.0], [2.0]], requires_grad=True)
    values = torch.tensor([[1.0], [2.0]])
    next_values = torch.tensor([[2.0], [3.0]])
    done = torch.tensor([[False], [True]])
    targets, advantage = vtrace(reward, values, next_values, done, done, torch.zeros_like(values), 0.9)
    torch.testing.assert_close(targets, torch.tensor([[2.8], [2.0]]))
    torch.testing.assert_close(advantage, torch.tensor([[1.8], [0.0]]))
    assert not targets.requires_grad and not advantage.requires_grad


def test_vtrace_clips_importance_weights_and_corrects_replay():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[1.0], [2.0]])
    next_values = torch.tensor([[2.0], [3.0]])
    done = torch.zeros_like(values, dtype=torch.bool)
    targets, advantages = vtrace(rewards, values, next_values, done, done, torch.full_like(values, np.log(0.5)), 0.9)
    torch.testing.assert_close(targets, torch.tensor([[2.5075], [3.35]]))
    torch.testing.assert_close(advantages, torch.tensor([[1.5075], [1.35]]))
    regular = vtrace(rewards, values, next_values, done, done, torch.zeros_like(values), 0.9)
    clipped = vtrace(rewards, values, next_values, done, done, torch.full_like(values, np.log(9)), 0.9)
    torch.testing.assert_close(regular, clipped)


def test_truncation_bootstraps_final_observation_without_crossing_reset():
    values = torch.zeros(2, 1)
    targets, advantage = vtrace(torch.tensor([[1.0], [100.0]]), values, torch.tensor([[10.0], [9000.0]]), torch.tensor([[False], [True]]), torch.ones(2, 1, dtype=torch.bool), values, 0.9)
    torch.testing.assert_close(targets, torch.tensor([[10.0], [100.0]]))
    torch.testing.assert_close(targets, advantage)


def test_cloning_uses_behavior_to_current_kl_only_on_replay():
    old = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]]).log().requires_grad_()
    current = torch.tensor([[[0.4, 0.6], [0.7, 0.3]]]).log().requires_grad_()
    old_values = torch.tensor([[2.0, 4.0]], requires_grad=True)
    values = torch.tensor([[3.0, 8.0]], requires_grad=True)
    kl, error = cloning_losses(current, values, old, old_values, torch.tensor([True, False]))
    expected = (0.8 * np.log(0.8 / 0.4) + 0.2 * np.log(0.2 / 0.6)) / 2
    assert kl.item() == pytest.approx(expected)
    assert error.item() == pytest.approx(0.5)
    (kl + error).backward()
    assert old.grad is None and old_values.grad is None
    assert current.grad[0, 1].abs().sum() == 0 and values.grad[0, 1] == 0


def test_reservoir_is_bounded_immutable_and_preserves_whole_unrolls():
    reservoir = UnrollReservoir(20, 4, seed=9)
    for step in range(100):
        batch = {"action": np.full((4, 1), step), "behavior_logits": np.full((4, 1, 2), step, dtype=np.float32)}
        reservoir.add(batch)
        batch["action"][:] = -1
    assert reservoir.seen == 100 and len(reservoir.items) == 5
    kept = [item["action"][0] for item in reservoir.items]
    assert min(kept) < 50 and max(kept) > 50
    for item in reservoir.items:
        assert np.all(item["action"] == item["behavior_logits"][:, 0])
        assert len(set(item["action"].tolist())) == 1


def test_zero_replay_preserves_batch_and_sampling_rng():
    rng = np.random.default_rng(2)
    before = deepcopy(rng.bit_generator.state)
    fresh = {"action": np.zeros((4, 4), dtype=np.int64)}
    memory = UnrollReservoir(32, 4)
    memory.add(fresh)
    actual, mask = mix_batch(fresh, memory, 0, rng)
    assert actual is fresh and not mask.any() and rng.bit_generator.state == before


def test_simultaneous_cbp_resets_cannot_resurrect_outgoing_columns():
    net = mlp(4, [3, 3], 2)
    probe = FeatureProbe(net)
    cbp = attach(probe, {"decay_rate": 0.99, "replacement_rate": 0.34, "maturity_threshold": 0, "utility": "contribution"}, torch.Generator().manual_seed(2))
    optimizer = AdamCBP(net.parameters())
    probe.capturing = True
    net(torch.ones(8, 4)).sum().backward()
    probe.capturing = False
    optimizer.step()
    assert cbp.step(optimizer, probe.features) == [1, 1]
    first = torch.where(cbp.state[0]["age"] == 0)[0]
    second = torch.where(cbp.state[1]["age"] == 0)[0]
    assert torch.count_nonzero(net[2].weight[:, first]) == 0
    assert torch.count_nonzero(net[4].weight[:, second]) == 0
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.count_nonzero(optimizer.state[net[2].weight][key][:, first]) == 0
        assert torch.count_nonzero(optimizer.state[net[2].weight][key][second]) == 0


def test_cbp_utilities_are_computed_before_any_replacement():
    net = mlp(4, [3, 3], 2)
    original = deepcopy(net[2].weight.detach())
    probe = FeatureProbe(net)
    cbp = attach(probe, {"decay_rate": 0.99, "replacement_rate": 0.34, "maturity_threshold": 0}, torch.Generator().manual_seed(2))
    observed = []
    select = cbp.select

    def capture(state, utility):
        observed.append(net[2].weight.detach().clone())
        return select(state, utility)

    cbp.select = capture
    cbp.step(None, [torch.ones(8, 3), torch.ones(8, 3)])
    assert len(observed) == 2
    for value in observed:
        torch.testing.assert_close(value, original)


def test_true_stable_rank_differs_from_singular_value_participation_ratio():
    features = torch.tensor([[6.0, 3.0], [0.0, 3.0], [3.0, 4.0], [3.0, 2.0]])
    report = plasticity(mlp(2, [2], 1), [features], 0.025)
    assert report["stable_rank_0"] == pytest.approx(10 / 9)
    assert report["singular_value_participation_ratio_0"] == pytest.approx(1.6)


@pytest.mark.parametrize("left,right,change", [("finetune", "cbp", "replacement"), ("clear", "clear_cbp", "replacement"), ("finetune", "clear", "replay"), ("replay", "clear", "cloning")])
def test_disabled_interventions_are_exact_equivalences(config, left, right, change):
    if change == "replacement":
        config["cbp"]["replacement_rate"] = 0
    elif change == "replay":
        config["clear"]["replay_ratio"] = 0
    else:
        config["clear"]["policy_cloning_weight"] = 0
        config["clear"]["value_cloning_weight"] = 0
    agents = [ActorCritic(config, "minatar", arm) for arm in (left, right)]
    collectors = [Collector(agent, ["breakout"]) for agent in agents]
    try:
        for _ in range(4):
            for agent, collector in zip(agents, collectors):
                learn_rollout(agent, collector.collect(float("inf")))
        for first, second in zip(agents[0].parameters, agents[1].parameters):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        for collector in collectors:
            collector.close()


def test_probe_training_and_diagnostics_leave_parent_unchanged(config):
    agent = ActorCritic(config, "minatar", "clear_cbp")
    collector = Collector(agent, ["breakout"])
    learn_rollout(agent, collector.collect(float("inf")))
    collector.close()
    before = deepcopy(agent.checkpoint())
    fixed = fixed_observations("minatar", 32)
    first = agent.diagnostics(fixed)
    probe_tasks(agent, time.time() + 20, "test")
    second = agent.diagnostics(fixed)
    assert first == second
    for net in ("actor", "critic"):
        for name, parameter in agent.weights()[net].items():
            torch.testing.assert_close(parameter, before["weights"][net][name], rtol=0, atol=0)
    assert agent.memory.seen == before["memory"]["seen"] and agent.updates == before["updates"]
    assert torch.equal(agent.action_rng.get_state(), before["action_rng"])


def test_task_interfaces_and_fixed_diagnostic_inputs():
    suite = "minatar"
    for task in task_names(suite) + task_names(suite, True):
        env = make_env(suite, task)
        state, _ = env.reset(seed=0)
        assert state.shape == (1000,) and env.action_space.n == 6
        # Every game is padded to ten channels, so the tail channels stay empty.
        channels = env.env.observation_space.shape[-1]
        assert not state.reshape(10, 10, 10)[:, :, channels:].any()
        env.step(0)
        env.close()
    np.testing.assert_array_equal(fixed_observations(suite, 32), fixed_observations(suite, 32))


def test_auc_integrates_irregular_coordinates():
    points = [{"steps": 0, "return": 0}, {"steps": 1, "return": 1}, {"steps": 10, "return": 1}]
    assert auc(points) == pytest.approx(0.95)


def test_revisit_forgetting_uses_previous_learning_block():
    rows = []
    for block, (task, a, b) in enumerate((("a", 500, 10), ("b", 100, 500), ("a", 450, 200), ("b", 300, 480))):
        rows.append({"block": block, "task": task, "scores": {"a": {"return": a}, "b": {"return": b}}})
    drops = retention_drops(rows)
    assert [r["drop"] for r in drops if r["task"] == "a"] == [400, 150]
    assert [r["since_block"] for r in drops if r["task"] == "a"] == [0, 2]


def test_incomplete_or_unlearnable_pilot_cannot_qualify():
    report = qualification("minatar", {}, {})
    assert not report["qualified"] and not report["learnable"]


@pytest.mark.parametrize("fresh_auc,late_auc,fresh_return,learned", [(0.5, 0.4, 0.4, False), (5, 4, 4, True)])
def test_probe_deficit_requires_a_learned_fresh_reference(fresh_auc, late_auc, fresh_return, learned):
    probes = {"initial": {"asterix": {"auc": fresh_auc, "curve": [{"return": fresh_return}]}}, "final": {"asterix": {"auc": late_auc}}}
    result = qualification("minatar", {"finetune": {"probes": probes}}, {"asterix": {"return": 0.2}})
    assert result["probe_auc_threshold_met"]
    assert result["probe_reference_learnable"]["asterix"] == learned
    assert result["plasticity_loss_demonstrated"] == learned


def test_checkpoint_restores_behavior_optimizer_memory_and_cbp(config, tmp_path):
    agent = ActorCritic(config, "minatar", "clear_cbp")
    collector = Collector(agent, ["breakout"])
    batch = collector.collect(float("inf"))
    agent.optimize(batch)
    collector.close()
    path = tmp_path / "model.pt"
    torch.save(agent.checkpoint(), path)
    restored, _ = load_agent(path, device="cpu")
    agent.optimize(batch)
    restored.optimize(batch)
    for first, second in zip(agent.parameters, restored.parameters):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    states = fixed_observations("minatar", 4)
    np.testing.assert_array_equal(agent.act(states)[0], restored.act(states)[0])


def test_complete_job_retains_all_revisit_curves_and_can_be_audited(config, tmp_path):
    job = {"config": config, "suite": "minatar", "arm": "clear_cbp", "blocks": ["breakout", "space_invaders", "breakout"], "out": str(tmp_path), "deadline": time.time() + 30, "probes": True}
    result = run_job(job)
    assert result["status"] == "complete"
    from src.utils.io import read_run

    metrics = read_run(tmp_path)
    assert len(metrics["curves"]) == 3 and metrics["completed_blocks"] == 3
    assert set(metrics["probes"]) == {"initial", "midpoint", "final"}
    agent, _ = load_agent(tmp_path / "model.pt", device="cpu")
    # Evaluation replays from the weights and the evaluation point, exactly as the audit does.
    assert evaluate(agent, task_names("minatar"), 1, moment=metrics["matrix"][-1]["steps"]) == metrics["matrix"][-1]["scores"]


def test_expired_job_is_incomplete_and_never_claims_a_checkpoint(config, tmp_path):
    result = run_job({"config": config, "suite": "minatar", "arm": "finetune", "out": str(tmp_path), "deadline": time.time() - 1})
    assert result["status"] == "budget_exhausted" and result["completed_blocks"] == 0
    assert not (tmp_path / "model.pt").exists()


def test_new_configs_validate_and_size_the_study_by_cycles_alone():
    for path in ("config/study.yaml", "config/smoke.yaml"):
        config = validate(load_config(path))
        # No clock: the study is exactly cycles x tasks x block_steps of training.
        assert "max_seconds" not in config and config["cycles"] >= 1
    config = load_config("config/study.yaml")
    config["cycles"] = 0
    with pytest.raises(ValueError, match="Cycles"):
        validate(config)


def test_standalone_reference_jobs_have_explicit_matching_budgets(config, tmp_path):
    from src.utils.study import main_jobs

    config["suite"] = "minatar"
    config["cycles"] = 2
    config["arms"] = ["multitask"]
    joint = main_jobs(config, tmp_path, float("inf"))[0]
    assert len(joint["blocks"]) * joint["block_steps"] == 6 * config["train"]["block_steps"]
    config["arms"] = ["scratch"]
    scratch = main_jobs(config, tmp_path, float("inf"))
    assert len(scratch) == 3
    assert [job["blocks"] for job in scratch] == [[task] for task in task_names("minatar")]
    assert len(set(job["out"] for job in scratch)) == 3


def test_main_run_keeps_intermediate_checkpoints_outside_results(config, tmp_path):
    from pathlib import Path
    from src.utils.study import main_jobs

    root = tmp_path / "results"
    work = tmp_path / "work"
    config.update(suite="minatar", arms=["replay"], cycles=1, workspace=str(work))
    job = main_jobs(config, root, time.time() + 30)[0]
    job["blocks"] = ["breakout"]
    assert Path(job["out"]) == root / "replay"
    assert run_job(job)["status"] == "complete"
    assert (root / "replay" / "model.pt").exists()
    assert (root / "replay" / "config.yaml").exists()
    assert (work / "checkpoints" / "replay" / "block_0001.pt").exists()
    assert not list(root.rglob("checkpoints")) and not (root / "main").exists()


def test_report_reanalysis_reuses_audits_without_instantiating_learners(config, tmp_path, monkeypatch):
    import json
    from src.utils.report import finalize

    config["suite"] = "minatar"
    config["arms"] = ["finetune", "clear"]
    scores = {task: {"return": 10, "episodes": [10]} for task in task_names("minatar")}
    for arm in config["arms"]:
        folder = tmp_path / arm
        folder.mkdir(parents=True)
        data = {"status": "complete", "completed_blocks": 0, "matrix": [{"block": -1, "task": None, "steps": 0, "scores": scores}], "diagnostics": [], "curves": [], "probes": {}}
        (folder / "metrics.json").write_text(json.dumps(data))

    def forbidden(*args, **kwargs):
        raise AssertionError("Reanalysis must not instantiate or evaluate a learner")

    monkeypatch.setattr("src.utils.report.ActorCritic", forbidden)
    monkeypatch.setattr("src.utils.report.figures", lambda *args: None)
    saved = {"audit": [{"arm": arm, "match": True} for arm in config["arms"]], "clips": []}
    manifest = {"suite": "minatar", "gpu": "cpu", "workers": 1, "cycles": 1, "started_at": time.time(), "elapsed_seconds": 10, "source_sha256": "test"}
    report = finalize(tmp_path, config, qualification("minatar", {}, {}), manifest, time.time() - 1, saved_artifacts=saved)
    assert report["all_checkpoints_verified"] and set(report["arms"]) == set(config["arms"])
    from src.utils.compare import refresh
    from src.utils.io import save_config

    config["workspace"] = str(tmp_path / "work")
    save_config(tmp_path / "config.yaml", config)
    regenerated = refresh(tmp_path)
    assert regenerated["all_checkpoints_verified"] and not (tmp_path / "work" / "pilot").exists()


def test_invalid_reference_mixture_and_zero_probe_budget_are_rejected(config):
    config["arms"] = ["scratch", "clear"]
    with pytest.raises(ValueError, match="standalone"):
        validate(config)
    config["arms"] = ["clear"]
    config["train"]["probe_steps"] = 0
    with pytest.raises(ValueError, match="complete rollouts"):
        validate(config)


@pytest.mark.parametrize("arm", ["scratch", "multitask"])
def test_standalone_reference_pipeline_produces_audited_results(config, tmp_path, arm):
    from src.utils.report import finalize, finalize_scratch
    from src.utils.study import main_jobs

    config["suite"] = "minatar"
    config["arms"] = [arm]
    config["cycles"] = 1
    deadline = time.time() + 30
    jobs = main_jobs(config, tmp_path, deadline)
    for job in jobs:
        assert run_job(job)["status"] == "complete"
    finalizer = finalize_scratch if arm == "scratch" else finalize
    manifest = {"suite": "minatar", "gpu": "cpu", "workers": 1, "cycles": 1, "started_at": time.time(), "elapsed_seconds": 1}
    report = finalizer(tmp_path, config, qualification("minatar", {}, {}), manifest, deadline)
    assert report["all_runs_complete"] and report["all_checkpoints_verified"]
    figure = "learning_curves.png" if arm == "scratch" else "performance.png"
    assert (tmp_path / "REPORT.md").exists() and (tmp_path / figure).exists()
    if arm == "scratch":
        assert set(report["references"]) == set(task_names("minatar"))
    else:
        assert report["arms"][arm]["compared_env_steps"] == len(task_names("minatar")) * config["train"]["block_steps"]


def test_collector_stores_truncated_final_observation_before_reset(config, monkeypatch):
    class AlwaysTruncated:
        def reset(self, **kwargs):
            return np.full(1000, -2, dtype=np.float32), {}

        def step(self, action):
            return np.full(1000, 7, dtype=np.float32), 1.0, False, True, {}

        def close(self):
            pass

    monkeypatch.setattr("src.utils.runtime.make_env", lambda *args: AlwaysTruncated())
    collector = Collector(ActorCritic(config, "minatar", "finetune"), ["breakout"])
    batch = collector.collect(float("inf"))
    assert np.all(batch["observation"] == -2) and np.all(batch["next_observation"] == 7)
    assert batch["boundary"].all() and not batch["terminated"].any()
    collector.close()


def test_interrupted_probe_retains_partial_curve(config, monkeypatch):
    agent = ActorCritic(config, "minatar", "finetune")
    calls = 0
    original = evaluate

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BudgetExpired()
        return original(*args, **kwargs)

    monkeypatch.setattr("src.utils.runtime.evaluate", interrupt)
    results = {}
    with pytest.raises(BudgetExpired):
        probe_tasks(agent, time.time() + 10, "interrupted", results)
    partial = results[task_names("minatar", True)[0]]
    assert partial["status"] == "incomplete" and partial["auc"] is None
    assert len(partial["curve"]) == 1 and partial["train_env_steps"] > 0
