from copy import deepcopy
import hashlib

import pytest
import torch

from src.clear.learner import ActorCritic
from src.utils.config import load_config
from src.utils.io import read_run
from src.utils.probes import run_probes
from src.utils.runtime import save_phase_checkpoint


def test_standalone_probes_load_all_phases_and_leave_sources_unchanged(tmp_path):
    config = load_config("config/smoke.yaml")
    config["device"] = "cpu"
    config["train"].update(hidden_sizes=[4, 4], num_envs=4, batch_unrolls=4,
                           unroll_length=4, block_steps=32, probe_steps=32)
    agent = ActorCritic(config, "minatar", "cbp")
    source, out = tmp_path / "source", tmp_path / "probes"
    for phase, blocks in [("initial", 0), ("midpoint", 7), ("final", 15)]:
        with torch.no_grad():
            next(agent.actor.parameters()).add_(.01)
        save_phase_checkpoint(source, agent, phase, blocks, blocks * 32)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.pt")}
    metadata = run_probes(source, out, config)
    assert metadata["status"] == "complete" and metadata["arm"] == "cbp"
    assert metadata["probe_checkpoint_blocks"] == {"initial": 0, "midpoint": 7, "final": 15}
    logged = read_run(out)
    assert set(logged["probes"]) == {"initial", "midpoint", "final"}
    for tasks in logged["probes"].values():
        assert tasks["asterix"]["train_env_steps"] == 32
        assert tasks["asterix"]["updates"] == 2
        assert tasks["asterix"]["train_settings"]["hidden_sizes"] == [4, 4]
    assert (out / "probe_curves.png").exists()
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    with pytest.raises(ValueError, match="output exists"):
        run_probes(source, out, config)
    with pytest.raises(ValueError, match="separate"):
        run_probes(source, source, config, overwrite=True)
    with pytest.raises(FileNotFoundError):
        run_probes(tmp_path / "missing", tmp_path / "unused", config)
    assert not (tmp_path / "unused").exists()


def test_standalone_probes_use_saved_phase_weights(monkeypatch, tmp_path):
    import src.utils.probes as probes

    config = load_config("config/smoke.yaml")
    config["device"] = "cpu"
    agent = ActorCritic(config, "minatar", "clear_cbp")
    with torch.no_grad():
        for parameter in agent.parameters:
            parameter.fill_(.25)
    expected = deepcopy(agent.weights())
    save_phase_checkpoint(tmp_path / "source", agent, "final", 15, 1234)

    def inspect(source, deadline, phase, results):
        assert phase == "final" and source.memory is None and not source.cbp
        assert not source.optimizer.state
        for network, weights in expected.items():
            for name, value in weights.items():
                torch.testing.assert_close(source.weights()[network][name], value, rtol=0, atol=0)
        results["asterix"] = {"status": "complete", "train_env_steps": 256, "updates": 1,
                              "episode_log": [], "train_settings": source.probe_learner().config["train"]}

    monkeypatch.setattr(probes, "probe_tasks", inspect)
    run_probes(tmp_path / "source", tmp_path / "out", config, phases=["final"])
