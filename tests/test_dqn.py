from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.cbp import AdamCBP, ContinualBackprop, FeatureProbe, attach, plasticity
from src.dqn import DQNAgent, double_targets, epsilon, network
from src.train import streams, train
from src.utils.config import load_config


@pytest.fixture
def config():
    c = load_config()
    c["train"]["hidden_sizes"] = [8, 8]
    c["train"]["batch_size"] = 8
    c["train"]["learning_starts"] = 4
    c["train"]["buffer_capacity"] = 64
    return c


@pytest.fixture
def generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g


def cbp_settings(rate=0.34, maturity=100):
    return {"replacement_rate": rate, "decay_rate": 0.99, "maturity_threshold": maturity, "dead_threshold": 0.025}


def make_pair(width=3, inputs=4, outputs=2, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(inputs, width), nn.Linear(width, outputs)


# --- network and targets -------------------------------------------------


def test_network_maps_observations_to_one_value_per_action():
    net = network(4, [16, 16], 2)
    assert net(torch.zeros(5, 4)).shape == (5, 2)
    assert len([m for m in net if isinstance(m, nn.Linear)]) == 3


def test_epsilon_decays_over_the_configured_fraction_of_each_block():
    settings = {"epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_fraction": 0.1}
    assert epsilon(settings, 0, 1000) == pytest.approx(1.0)
    assert epsilon(settings, 50, 1000) == pytest.approx(0.525)
    assert epsilon(settings, 100, 1000) == pytest.approx(0.05)
    assert epsilon(settings, 900, 1000) == pytest.approx(0.05)


def test_termination_stops_the_bootstrap_but_truncation_does_not():
    reward = torch.tensor([1.0, 1.0])
    terminated = torch.tensor([True, False])
    next_target = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    next_online = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = double_targets(reward, terminated, next_target, next_online, 0.9)
    assert targets.tolist() == pytest.approx([1.0, 5.5])


def test_double_dqn_selects_with_the_online_net_and_scores_with_the_target():
    reward = torch.zeros(1)
    next_target = torch.tensor([[10.0, -10.0]])
    next_online = torch.tensor([[0.0, 1.0]])  # online prefers action 1
    assert double_targets(reward, torch.tensor([False]), next_target, next_online, 1.0).item() == pytest.approx(-10.0)


# --- the elementwise Adam ------------------------------------------------


def test_adamcbp_first_step_equals_the_learning_rate_for_a_constant_gradient():
    p = nn.Parameter(torch.zeros(3))
    optimizer = AdamCBP([p], lr=0.1)
    p.grad = torch.ones(3)
    optimizer.step()
    assert p.detach().abs().tolist() == pytest.approx([0.1, 0.1, 0.1], rel=1e-4)


def test_adamcbp_tracks_stock_adam_when_nothing_is_reset():
    mine = nn.Parameter(torch.zeros(4))
    theirs = nn.Parameter(torch.zeros(4))
    a, b = AdamCBP([mine], lr=0.01), torch.optim.Adam([theirs], lr=0.01)
    for step in range(50):
        gradient = torch.full((4,), 0.1 * (step % 3 + 1))
        mine.grad, theirs.grad = gradient.clone(), gradient.clone()
        a.step()
        b.step()
    torch.testing.assert_close(mine.detach(), theirs.detach(), rtol=1e-5, atol=1e-7)


def step_sizes_after_reset(reset_keys, warmup=5000, steps=25, lr=0.1):
    """Largest per-step move of unit 0 in the steps following a partial state reset."""
    p = nn.Parameter(torch.zeros(2))
    optimizer = AdamCBP([p], lr=lr)
    for _ in range(warmup):
        p.grad = torch.ones(2)
        optimizer.step()
    for key in reset_keys:
        optimizer.state[p][key][0] = 0
    moves = []
    for _ in range(steps):
        before = p.detach().clone()
        p.grad = torch.ones(2)
        optimizer.step()
        moves.append((p[0] - before[0]).abs().item())
    return moves


def test_resetting_a_unit_restores_a_fresh_adam_step_size():
    # A correctly reset unit descends at exactly the learning rate on a constant gradient.
    moves = step_sizes_after_reset(("step", "exp_avg", "exp_avg_sq"))
    assert max(moves) == pytest.approx(0.1, rel=1e-3)
    assert min(moves) == pytest.approx(0.1, rel=1e-3)


def test_zeroing_only_the_moments_inflates_the_step_which_is_why_adamcbp_exists():
    # Stock Adam's scalar step cannot reset per unit, so a replaced unit overshoots: hence AdamCBP.
    moves = step_sizes_after_reset(("exp_avg", "exp_avg_sq"))
    assert max(moves) > 5 * 0.1


def test_reset_unit_clears_the_incoming_row_and_the_outgoing_column():
    incoming, outgoing = make_pair(width=3)
    optimizer = AdamCBP(list(incoming.parameters()) + list(outgoing.parameters()), lr=0.1)
    for parameter in list(incoming.parameters()) + list(outgoing.parameters()):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.reset_unit(incoming, 1, axis=0)
    optimizer.reset_unit(outgoing, 1, axis=1)
    assert optimizer.state[incoming.weight]["exp_avg"][1].abs().sum() == 0
    assert optimizer.state[incoming.weight]["exp_avg"][0].abs().sum() > 0
    assert optimizer.state[outgoing.weight]["exp_avg"][:, 1].abs().sum() == 0
    assert optimizer.state[outgoing.weight]["exp_avg"][:, 0].abs().sum() > 0


# --- continual backpropagation -------------------------------------------


def test_immature_units_are_never_replaced(generator):
    incoming, outgoing = make_pair()
    cbp = ContinualBackprop([(incoming, outgoing)], cbp_settings(rate=1.0), generator)
    state = cbp.state[0]
    state["age"] = torch.full((3,), 10.0)  # below the maturity threshold of 100
    original = outgoing.weight.detach().clone()
    count = cbp.replace(0, incoming, outgoing, state, torch.zeros(3), torch.zeros(3), None)
    assert count == 0
    torch.testing.assert_close(outgoing.weight.detach(), original)


def test_the_lowest_utility_mature_unit_is_the_one_replaced(generator):
    incoming, outgoing = make_pair()
    cbp = ContinualBackprop([(incoming, outgoing)], cbp_settings(rate=0.34), generator)
    state = cbp.state[0]
    state["age"] = torch.full((3,), 200.0)
    count = cbp.replace(0, incoming, outgoing, state, torch.zeros(3), torch.tensor([9.0, 0.1, 9.0]), None)
    assert count == 1
    assert outgoing.weight.detach()[:, 1].abs().sum() == 0
    assert outgoing.weight.detach()[:, 0].abs().sum() > 0
    assert outgoing.weight.detach()[:, 2].abs().sum() > 0
    assert state["age"][1] == 0 and state["age"][0] == 200


def test_replacement_resets_utility_mean_activation_and_age(generator):
    incoming, outgoing = make_pair()
    cbp = ContinualBackprop([(incoming, outgoing)], cbp_settings(rate=0.34), generator)
    state = cbp.state[0]
    state["age"] = torch.full((3,), 200.0)
    state["util"] = torch.tensor([9.0, 0.1, 9.0])
    state["mean_act"] = torch.tensor([1.0, 2.0, 3.0])
    cbp.replace(0, incoming, outgoing, state, torch.zeros(3), state["util"].clone(), None)
    assert state["util"][1] == 0 and state["mean_act"][1] == 0 and state["age"][1] == 0
    assert state["mean_act"][0] == 1.0


def test_replacement_preserves_the_output_at_the_units_mean_activation(generator):
    # Without the bias fold, deleting a strongly-on low-variance unit shifts every Q-value.
    incoming, outgoing = make_pair(width=4)
    cbp = ContinualBackprop([(incoming, outgoing)], cbp_settings(rate=0.26), generator)
    state = cbp.state[0]
    state["age"] = torch.full((4,), 200.0)
    corrected_mean = torch.tensor([0.5, 1.5, 2.5, 3.5])
    with torch.no_grad():
        before = outgoing(corrected_mean).clone()
    count = cbp.replace(0, incoming, outgoing, state, corrected_mean, torch.tensor([9.0, 0.1, 9.0, 9.0]), None)
    with torch.no_grad():
        after = outgoing(corrected_mean)
    assert count == 1
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


def test_dropping_the_bias_fold_would_change_the_output(generator):
    # Guards the test above against silently passing on a zero-mean unit.
    incoming, outgoing = make_pair(width=4)
    cbp = ContinualBackprop([(incoming, outgoing)], cbp_settings(rate=0.26), generator)
    state = cbp.state[0]
    state["age"] = torch.full((4,), 200.0)
    corrected_mean = torch.tensor([0.5, 1.5, 2.5, 3.5])
    with torch.no_grad():
        before = outgoing(corrected_mean).clone()
        naive = before - outgoing.weight[:, 1] * corrected_mean[1]
    cbp.replace(0, incoming, outgoing, state, corrected_mean, torch.tensor([9.0, 0.1, 9.0, 9.0]), None)
    with torch.no_grad():
        after = outgoing(corrected_mean)
    assert not torch.allclose(naive, after, rtol=1e-4, atol=1e-5)


def test_reinitialised_weights_respect_the_layer_specific_fan_in_bound(generator):
    incoming, outgoing = make_pair(width=64, inputs=4)
    cbp = ContinualBackprop([(incoming, outgoing)], cbp_settings(rate=1.0), generator)
    state = cbp.state[0]
    state["age"] = torch.full((64,), 200.0)
    cbp.replace(0, incoming, outgoing, state, torch.zeros(64), torch.arange(64, dtype=torch.float32), None)
    bound = 1.0 / np.sqrt(4)
    assert incoming.weight.detach().abs().max().item() <= bound + 1e-6
    assert incoming.bias.detach().abs().sum().item() == 0


def test_ages_advance_before_the_bias_correction_so_it_is_never_zero(generator):
    net = network(4, [5], 2)
    probe = FeatureProbe(net)
    cbp = attach(probe, cbp_settings(rate=0.0), generator)
    probe.capturing = True
    net(torch.randn(6, 4))
    probe.capturing = False
    cbp.step(None, probe.features)
    assert torch.isfinite(cbp.state[0]["util"]).all()
    assert torch.isfinite(cbp.state[0]["mean_act"]).all()
    assert cbp.state[0]["age"].tolist() == [1.0] * 5


def test_feature_probe_records_only_the_batch_it_is_told_to(generator):
    net = network(4, [5], 2)
    probe = FeatureProbe(net)
    wanted, ignored = torch.ones(1, 4), torch.full((1, 4), -3.0)
    probe.capturing = True
    net(wanted)
    probe.capturing = False
    net(ignored)
    with torch.no_grad():
        expected = torch.relu(net[0](wanted))
    torch.testing.assert_close(probe.features[0], expected)


def test_plasticity_reports_dead_units_and_rank():
    net = network(4, [6], 2)
    features = [torch.cat([torch.zeros(8, 3), torch.rand(8, 3) + 1.0], dim=1)]
    report = plasticity(net, features, 0.025)
    assert report["dead_fraction_0"] == pytest.approx(0.5)
    assert report["stable_rank_0"] > 1.0
    assert report["weight_norm_0"] > 0


# --- the agent -----------------------------------------------------------


def fill(agent, count=32, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        state = rng.normal(size=4).astype(np.float32)
        agent.observe(state, int(rng.integers(0, 2)), 1.0, rng.normal(size=4).astype(np.float32), bool(rng.integers(0, 2)))


def test_a_zero_replacement_rate_reproduces_plain_finetuning_exactly(config):
    config["cbp"]["replacement_rate"] = 0.0
    plain = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    with_cbp = DQNAgent(config, "cbp", 0, 4, 2, streams(0))
    for agent in (plain, with_cbp):
        agent.begin_task("default")
        fill(agent)
    for _ in range(25):
        plain.optimize()
        with_cbp.optimize()
    for left, right in zip(plain.online.parameters(), with_cbp.online.parameters()):
        torch.testing.assert_close(left.detach(), right.detach(), rtol=0, atol=0)


def test_a_positive_replacement_rate_actually_changes_the_network(config):
    config["cbp"]["replacement_rate"] = 0.05
    config["cbp"]["maturity_threshold"] = 2
    plain = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    with_cbp = DQNAgent(config, "cbp", 0, 4, 2, streams(0))
    for agent in (plain, with_cbp):
        agent.begin_task("default")
        fill(agent)
    for _ in range(40):
        plain.optimize()
        with_cbp.optimize()
    assert with_cbp.cbp.replacements > 0
    assert not torch.allclose(next(plain.online.parameters()).detach(), next(with_cbp.online.parameters()).detach())


def test_learning_starts_counts_current_task_transitions_only(config):
    agent = DQNAgent(config, "replay", 0, 4, 2, streams(0))
    agent.begin_task("default")
    fill(agent, count=config["train"]["learning_starts"] + 5)
    assert agent.ready()
    agent.begin_task("gravity")
    # The cross-task memory is full, but the new task has contributed nothing yet.
    assert agent.memory.size > 0
    assert not agent.ready()


def test_every_arm_starts_a_task_with_an_empty_current_buffer(config):
    for arm in ("finetune", "replay", "cbp", "replay_cbp"):
        agent = DQNAgent(config, arm, 0, 4, 2, streams(0))
        agent.begin_task("default")
        fill(agent)
        assert agent.replay.size > 0
        agent.begin_task("gravity")
        assert agent.replay.size == 0


def test_replay_batches_mix_current_and_past_once_the_memory_is_populated(config):
    agent = DQNAgent(config, "replay", 0, 4, 2, streams(0))
    agent.begin_task("default")
    fill(agent, count=40)
    agent.begin_task("gravity")
    fill(agent, count=40, seed=1)
    batch = agent.batch()
    assert batch["state"].shape[0] == config["train"]["batch_size"]
    assert agent.memory.size > 0


def test_finetuning_never_builds_a_cross_task_memory(config):
    agent = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    agent.begin_task("default")
    fill(agent)
    assert agent.memory is None
    assert agent.batch()["state"].shape[0] == config["train"]["batch_size"]


def test_optimize_moves_the_online_net_but_only_nudges_the_target(config):
    agent = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    agent.begin_task("default")
    fill(agent)
    before_online = copy.deepcopy(agent.online.state_dict())
    before_target = copy.deepcopy(agent.target.state_dict())
    agent.optimize()
    key = "0.weight"
    assert not torch.allclose(agent.online.state_dict()[key], before_online[key])
    moved = (agent.target.state_dict()[key] - before_target[key]).abs().max()
    assert 0 < moved < (agent.online.state_dict()[key] - before_online[key]).abs().max()


def test_greedy_action_is_deterministic_and_in_range(config):
    agent = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    state = np.zeros(4, dtype=np.float32)
    assert agent.act(state) == agent.act(state)
    assert agent.act(state) in (0, 1)


# --- end to end ----------------------------------------------------------


def tiny(config, root, arm="finetune"):
    config["train"]["timesteps"] = 300
    config["train"]["learning_starts"] = 50
    config["eval"]["every"] = 150
    config["eval"]["episodes"] = 1
    config["eval"]["matrix_episodes"] = 1
    config["output"]["root"] = str(root)
    config["output"]["videos"] = False
    return train(config, arm, 0, root / arm / "seed0")


def test_a_short_run_produces_a_complete_metrics_file(config, tmp_path):
    metrics = tiny(config, tmp_path)
    order = config["benchmark"]["order"]
    assert len(metrics["matrix"]) == len(order) + 1
    assert metrics["matrix"][0]["block"] is None
    assert set(metrics["auc"]) == set(order)
    assert all(0.0 <= value <= 1.0 for value in metrics["auc"].values())
    assert len(metrics["jumpstart"]) == len(order)
    assert (tmp_path / "finetune" / "seed0" / "model.pt").exists()
    assert metrics["plasticity"] and "dead_fraction_0" in metrics["plasticity"][-1]


def test_truncated_episodes_are_not_recorded_as_terminations(config, tmp_path, monkeypatch):
    config["env"]["max_steps"] = 8
    seen = []
    original = DQNAgent.observe

    def spy(self, state, action, reward, next_state, terminated):
        seen.append(bool(terminated))
        return original(self, state, action, reward, next_state, terminated)

    monkeypatch.setattr(DQNAgent, "observe", spy)
    metrics = tiny(config, tmp_path)
    episodes = len(metrics["train_episodes"])
    assert episodes > 0
    assert sum(seen) < episodes, "every episode end was recorded as a termination"
    assert any(record["return"] == 8.0 for record in metrics["train_episodes"])


def test_a_second_call_reuses_the_run_instead_of_retraining(config, tmp_path):
    first = tiny(config, tmp_path)
    second = tiny(config, tmp_path)
    assert first["signature"] == second["signature"]
    assert first["env_steps"] == second["env_steps"]


def test_multitask_trains_on_every_variant_in_one_block(config, tmp_path):
    metrics = tiny(config, tmp_path, arm="multitask")
    assert metrics["blocks"] == ["multitask"]
    assert len(metrics["matrix"]) == 2
    assert {record["task"] for record in metrics["train_episodes"]} == set(config["benchmark"]["order"])


def test_replay_is_identical_to_finetuning_during_the_first_task(config):
    # Nothing precedes task 1, so any difference here would be a second intervention.
    config["replay"]["capacity"] = 16  # small enough that the reservoir overflows and draws RNG
    plain = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    rehearsing = DQNAgent(config, "replay", 0, 4, 2, streams(0))
    for agent in (plain, rehearsing):
        agent.begin_task("default")
        fill(agent, count=60)
    assert rehearsing.memory.tasks["default"]["seen"] > rehearsing.memory.tasks["default"]["size"]
    for _ in range(25):
        plain.optimize()
        rehearsing.optimize()
    for left, right in zip(plain.online.parameters(), rehearsing.online.parameters()):
        torch.testing.assert_close(left.detach(), right.detach(), rtol=0, atol=0)


def test_replay_rehearses_only_earlier_tasks(config):
    agent = DQNAgent(config, "replay", 0, 4, 2, streams(0))
    agent.begin_task("default")
    fill(agent, count=40)
    assert agent.memory.past_size("default") == 0
    agent.begin_task("gravity")
    assert agent.memory.past_size("gravity") > 0
    batch = agent.memory.sample(agent.sample_rng, 50, exclude="gravity")
    assert batch["state"].shape[0] == 50


def test_torch_is_pinned_to_one_thread():
    # Parallel workers would fight over cores, and threading perturbs float reduction order.
    assert torch.get_num_threads() == 1


def test_two_identical_runs_produce_identical_results(config, tmp_path):
    first = tiny(config, tmp_path / "a")
    second = tiny(config, tmp_path / "b")
    assert first["signature"] == second["signature"]
    assert first["auc"] == second["auc"]
    assert first["matrix"][-1]["evaluations"] == second["matrix"][-1]["evaluations"]
    assert [row["normalized"] for row in first["learning_curve"]] == [row["normalized"] for row in second["learning_curve"]]


def test_different_seeds_produce_different_runs(config, tmp_path):
    config["train"]["timesteps"] = 300
    config["train"]["learning_starts"] = 50
    config["eval"]["every"] = 150
    config["eval"]["episodes"] = 1
    config["eval"]["matrix_episodes"] = 1
    first = train(config, "finetune", 0, tmp_path / "s0")
    second = train(config, "finetune", 1, tmp_path / "s1")
    assert first["signature"] != second["signature"]
    # Not auc: one greedy episode on an 8x8 net is too quantized to separate seeds at this budget.
    assert first["train_episodes"] != second["train_episodes"]
