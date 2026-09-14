from copy import deepcopy
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.cbp import InteractionDiagnostics, plasticity
from src.clear.learner import ActorCritic, mlp
from src.utils.config import load_config, validate
from src.utils.io import read_run, save_json
from src.utils.runtime import Collector, evaluate, learn_rollout, probe_tasks, run_job
from src.utils.report import figures, gaussian_probe_returns, matched_references, normalize, plasticity_figure, probe_notes, task_segments


def settings():
    cfg = load_config('config/smoke.yaml')
    cfg['device'] = 'cpu'
    cfg['train'].update(hidden_sizes=[4, 4], num_envs=4, batch_unrolls=4, unroll_length=4)
    cfg['train'].update(block_steps=32, probe_steps=32)
    cfg['eval'].update(clips=False, episodes=1)
    cfg['diagnostics'] = {'window_steps': 20, 'period': 20, 'rank_period': 40}
    return cfg


def test_rank_uses_singular_mass_without_centering_or_squaring():
    net = mlp(2, [2], 1)
    assert plasticity(net, [torch.diag(torch.tensor([99., 1.]))])['stable_rank'] == 1
    assert plasticity(net, [torch.diag(torch.tensor([98., 2.]))])['stable_rank'] == 2
    assert plasticity(net, [torch.ones(1000, 2)])['stable_rank'] == 1
    assert plasticity(net, [torch.zeros(1000, 2)])['stable_rank_percent'] == 0
    assert plasticity(net, [torch.eye(2)])['stable_rank_percent'] == 100


def test_dormancy_pools_units_and_includes_the_one_percent_boundary():
    net = mlp(2, [3, 1], 1)
    first, last = torch.zeros(1000, 3), torch.ones(1000, 1)
    first[:10, 0] = 1
    first[:11, 1] = 1e-8
    report = plasticity(net, [first, last])
    assert report['dormant_percent'] == 50
    with torch.no_grad():
        for layer in net[::2]:
            layer.weight.fill_(-2)
            layer.bias.fill_(1e6)
    assert all(plasticity(net, [first, last])[f'weight_magnitude_{i}'] == 2 for i in range(3))


def test_window_evicts_old_values_and_splits_vector_batches_on_exact_clocks():
    nets = [mlp(2, [2, 2], 1) for _ in range(2)]
    recorder = InteractionDiagnostics(nets)
    for offset in range(0, 10032, 16):
        values = torch.arange(offset, offset + 16).float().unsqueeze(1).repeat(1, 2)
        recorder.observe([[values, values] for _ in nets])
    rows = recorder.drain()
    assert [row['env_steps'] for row in rows] == list(range(1000, 10001, 1000))
    assert [row['env_steps'] for row in rows if 'actor_stable_rank' in row] == [10000]
    assert recorder.steps == 10032
    for buffers in recorder.buffers:
        for buffer in buffers:
            torch.testing.assert_close(buffer[:, 0].sort().values, torch.arange(9032, 10032).float())
    assert recorder.drain() == []


def assert_same_state(first, second):
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same_state(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same_state(a, b)
    else:
        assert first == second


def test_collection_diagnostics_preserve_training_rng_optimizer_and_cbp_across_tasks():
    cfg = settings()
    cfg['cbp'].update(replacement_rate=0.5, maturity_threshold=0)
    agents = [ActorCritic(cfg, 'minatar', 'clear_cbp') for _ in range(2)]
    recorder = InteractionDiagnostics([agents[1].actor, agents[1].critic], cfg['diagnostics'])
    for task in ['breakout', 'space_invaders']:
        collectors = [Collector(agent, [task], recorder=recorder if i else None)
                      for i, agent in enumerate(agents)]
        try:
            for _ in range(3):
                batches = [collector.collect(float('inf')) for collector in collectors]
                assert_same_state(*batches)
                for agent, batch in zip(agents, batches):
                    learn_rollout(agent, batch)
        finally:
            for collector in collectors:
                collector.close()
    assert_same_state(*(agent.checkpoint() for agent in agents))
    assert recorder.steps == 96
    assert [row['env_steps'] for row in recorder.rows] == [20, 40, 60, 80]
    before = deepcopy(recorder.buffers)
    evaluate(agents[1], ['breakout'], 1)
    probe_tasks(agents[1], float('inf'), 'test')
    assert recorder.steps == 96
    assert_same_state(recorder.buffers, before)


def test_probe_logs_each_episode_without_evaluation_or_scores(monkeypatch, tmp_path):
    import src.utils.runtime as runtime

    class ShortEpisode:
        def reset(self, **kwargs):
            return np.zeros(1000, dtype=np.uint8), {}

        def step(self, action):
            return np.zeros(1000, dtype=np.uint8), 1., True, False, {}

        def close(self):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError('Probes must not evaluate')

    monkeypatch.setattr(runtime, 'make_env', lambda *args: ShortEpisode())
    monkeypatch.setattr(runtime, 'evaluate', forbidden)
    agent = ActorCritic(settings(), 'minatar', 'clear_cbp')
    before = agent.checkpoint()
    result = probe_tasks(agent, float('inf'), 'initial')['asterix']
    assert result['status'] == 'complete' and result['train_env_steps'] == 32
    assert len(result['episode_log']) == 32
    assert [row['env_steps'] for row in result['episode_log']] == list(range(1, 33))
    assert 'auc' not in result and 'curve' not in result
    assert_same_state(before, agent.checkpoint())
    runtime.save_probes(tmp_path, 'initial', {'asterix': result})
    save_json(tmp_path / 'run.json', {'schema': 1})
    loaded = read_run(tmp_path)['probes']['initial']['asterix']
    assert len(loaded['episode_log']) == 32 and 'auc' not in loaded
    assert loaded['updates'] == 2
    assert loaded['train_settings'] == result['train_settings']


def test_probe_overrides_preserve_weights_and_main_training_settings():
    cfg = settings()
    cfg['probe_train'] = {'learning_rate': .002, 'entropy_weight': .03, 'gamma': .95}
    validate(cfg)
    source = ActorCritic(cfg, 'minatar', 'clear_cbp')
    before = source.checkpoint()
    probe = source.probe_learner()
    assert_same_state(source.weights(), probe.weights())
    assert_same_state(before, source.checkpoint())
    assert probe.optimizer.param_groups[0]['lr'] == .002
    assert source.optimizer.param_groups[0]['lr'] == cfg['train']['learning_rate']
    assert probe.config['train']['entropy_weight'] == .03
    assert probe.config['train']['gamma'] == .95
    assert not probe.optimizer.state and probe.memory is None and not probe.cbp
    assert probe.config['train']['hidden_sizes'] == cfg['train']['hidden_sizes']


@pytest.mark.parametrize('override', [{'hidden_sizes': [8, 8]}, {'learning_rate': 0},
                                    {'gamma': 1.1}, {'entropy_weight': -1},
                                    {'gradient_clip': 0}, {'value_weight': float('nan')},
                                    {'learning_rate': float('inf')}])
def test_invalid_probe_settings_are_rejected(override):
    cfg = settings()
    cfg['probe_train'] = override
    with pytest.raises(ValueError):
        validate(cfg)


def test_report_uses_recorded_probe_settings_and_marks_legacy_settings_unknown():
    payload = {'status': 'complete', 'train_env_steps': 32, 'updates': 2,
               'train_settings': {'learning_rate': .002, 'gamma': .95, 'entropy_weight': .03}}
    run = {'config': {'train': {'learning_rate': .1}}, 'probes': {'initial': {'asterix': payload}}}
    assert '| Fine-tuning | initial | 32 | 2 | 0.002 | 0.95 | 0.03 |' in probe_notes({'finetune': run})
    del payload['train_settings']
    assert '| Fine-tuning | initial | 32 | 2 | unrecorded | unrecorded | unrecorded |' in probe_notes({'finetune': run})
    assert probe_notes({'finetune': {}}) == []


def test_normalization_selects_the_matching_scratch_checkpoint(tmp_path):
    tasks = ['breakout', 'space_invaders', 'freeway']
    run = {'matrix': [{'steps': 0}] + [{'task': task, 'steps': 100 * (i + 1)}
                                    for i, task in enumerate(tasks * 2)]}
    for task in tasks:
        rows = [{'steps': step, 'block': i - 1, 'task': task, 'scores': {task: {'return': value}}}
                for i, (step, value) in enumerate([(0, 0), (100, 10), (200, 30), (300, 90)])]
        save_json(tmp_path / 'scratch' / task / 'metrics.json', {'matrix': rows})
    refs = matched_references(tmp_path, run, 5, tasks, {t: {'return': 2} for t in tasks})
    assert [refs[t]['scratch_return'] for t in tasks] == [30, 30, 10]
    assert [refs[t]['training_env_steps'] for t in tasks] == [200, 200, 100]
    assert normalize(16, refs['breakout']) == 0.5
    assert normalize(2, {'random_return': 2, 'scratch_return': 2}) is None
    absent = matched_references(tmp_path / 'absent', run, 5, tasks, {})
    assert all(normalize(10, ref) is None for ref in absent.values())
    run['matrix'][1]['steps'] = 99
    assert matched_references(tmp_path, run, 1, tasks, {})['breakout']['scratch_return'] is None


def test_old_diagnostic_columns_are_not_reinterpreted(tmp_path):
    save_json(tmp_path / 'run.json', {'schema': 1})
    (tmp_path / 'diagnostics.csv').write_text('actor_stable_rank_1\n2.1\n')
    (tmp_path / 'plasticity.csv').write_text('env_steps,actor_stable_rank_percent\n1000,25\n')
    assert read_run(tmp_path)['plasticity'] == []


def test_figures_preserve_every_point_and_use_six_correct_panels(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr('src.utils.report.figure_save', lambda fig, root, name: saved.update({name: fig}))
    points = [{'env_steps': 1000 * i, 'actor_stable_rank_percent': 12 + i,
               'critic_stable_rank_percent': 22 + i, 'actor_dormant_percent': 32 + i,
               'critic_dormant_percent': 42 + i, 'actor_weight_magnitude_1': i / 100,
               'critic_weight_magnitude_1': i / 200} for i in range(1, 50)]
    episodes = [{'env_steps': i * 3, 'return': i % 7} for i in range(100)]
    evaluations = [{'block': 0, 'block_task': 'breakout', 'task': 'breakout', 'scope': 'period',
                    'env_steps': i * 7, 'return': i % 9} for i in range(1, 101)]
    run = {'matrix': [{'steps': 0}, {'steps': 50000}], 'plasticity': points,
           'evaluations': evaluations, 'probes': {'initial': {'asterix': {'episode_log': episodes}}}}
    try:
        figures(tmp_path, {'cbp': run}, 'minatar', 1)
        plot = saved['plasticity.png']
        assert len(plot.axes) == 6
        assert ['Actor' in ax.get_title() for ax in plot.axes] == [True] * 3 + [False] * 3
        for ax, key in zip(plot.axes, ['actor_stable_rank_percent', 'actor_dormant_percent',
                                      'actor_weight_magnitude_1', 'critic_stable_rank_percent',
                                      'critic_dormant_percent', 'critic_weight_magnitude_1']):
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), [p[key] for p in points])
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [p['env_steps'] for p in points])
        probe_line = saved['probe_curves.png'].axes[0].lines[0]
        np.testing.assert_array_equal(probe_line.get_xdata(), [p['env_steps'] for p in episodes])
        np.testing.assert_allclose(probe_line.get_ydata(), gaussian_probe_returns([p['return'] for p in episodes]))
        assert not np.array_equal(probe_line.get_ydata(), [p['return'] for p in episodes])
        performance = saved['performance.png'].axes[0].lines[0]
        assert len(performance.get_xdata()) == len(evaluations)
        assert not plt.rcParams['path.simplify']
    finally:
        plt.close('all')


def test_gaussian_probe_smoothing_preserves_constants_and_has_gaussian_shape():
    assert gaussian_probe_returns([]).size == 0
    np.testing.assert_array_equal(gaussian_probe_returns([3]), [3])
    np.testing.assert_allclose(gaussian_probe_returns([3] * 10), 3)
    impulse = np.zeros(101)
    impulse[50] = 1
    result = gaussian_probe_returns(impulse, sigma=2)
    assert result.sum() == pytest.approx(1)
    assert result[52] / result[50] == pytest.approx(np.exp(-0.5))
    np.testing.assert_allclose(result, result[::-1])
    assert impulse[50] == 1 and np.count_nonzero(impulse) == 1


def test_logged_plasticity_windows_continue_across_blocks_and_exclude_probes(tmp_path):
    cfg = settings()
    job = {'config': cfg, 'suite': 'minatar', 'arm': 'cbp', 'blocks': ['breakout', 'space_invaders'],
           'out': str(tmp_path), 'deadline': float('inf'), 'probes': True}
    assert run_job(job)['status'] == 'complete'
    run = read_run(tmp_path)
    assert [row['env_steps'] for row in run['plasticity']] == [20, 40, 60]
    assert [row['block'] for row in run['plasticity']] == [0, 1, 1]
    assert run['plasticity_metrics']['centered'] is False
    assert len(run['diagnostics']) == 2
    assert run['plasticity'][0]['actor_stable_rank'] is None
    assert run['plasticity'][1]['actor_stable_rank'] is not None


def test_default_budget_and_diagnostic_cadence_are_the_requested_values():
    cfg = validate(load_config())
    assert cfg['train']['probe_steps'] == 5242880
    assert cfg['probe_train'] == {'learning_rate': .001, 'gamma': .99, 'entropy_weight': .005}
    assert cfg['train']['learning_rate'] == .0003 and cfg['train']['gamma'] == .99
    assert cfg['diagnostics'] == {'window_steps': 1000, 'period': 1000, 'rank_period': 10000}
