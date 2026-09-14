import numpy as np
import pytest
import torch
import torch.nn as nn

from src.cbp import AdamCBP, ContinualBackprop, FeatureProbe, attach, plasticity
from src.clear.learner import mlp


@pytest.fixture
def generator():
    g = torch.Generator()
    g.manual_seed(0)
    return g


def cbp_settings(rate=0.34, maturity=100):
    return {"replacement_rate": rate, "decay_rate": 0.99, "maturity_threshold": maturity}


def make_pair(width=3, inputs=4, outputs=2, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(inputs, width), nn.Linear(width, outputs)


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
    # Without the bias fold, deleting a strongly-on low-variance unit shifts every network output.
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
    net = mlp(4, [5], 2)
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
    net = mlp(4, [5], 2)
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
    net = mlp(4, [6], 2)
    features = [torch.cat([torch.zeros(8, 3), torch.rand(8, 3) + 1.0], dim=1)]
    report = plasticity(net, features)
    assert report["dormant_percent"] == pytest.approx(50)
    assert report["stable_rank"] > 1.0
    assert report["weight_magnitude_0"] > 0
