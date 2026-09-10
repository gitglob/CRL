from __future__ import annotations

import numpy as np
import pytest

from src.dqn import DQNAgent
from src.train import env_seeds, streams, train
from src.utils.config import load_config, validate
from src.utils.envs import make_task
from src.utils.evaluate import episode_seeds, evaluate


@pytest.fixture
def config():
    return load_config()


class Rightwards:
    """A fixed action makes every episode length a pure function of its seed."""

    def act_batch(self, states, exploration=0.0):
        return [0] * len(states)


def test_slot_seeds_are_distinct_and_start_from_the_serial_seed():
    key = streams(0)["env"]
    serial = int(key.generate_state(1)[0])
    assert env_seeds(key, 1) == [serial]
    seeds = env_seeds(key, 10)
    assert seeds[0] == serial
    assert len(set(seeds)) == 10
    assert env_seeds(key, 10) == seeds


@pytest.mark.parametrize("count", [1, 2, 10])
@pytest.mark.parametrize("exploration", [0.0, 0.3, 1.0])
def test_act_batch_matches_a_sequence_of_scalar_acts(config, count, exploration):
    scalar = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    batched = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    states = np.random.default_rng(3).normal(size=(count, 4)).astype(np.float32)
    assert [scalar.act(states[i], exploration) for i in range(count)] == batched.act_batch(states, exploration)
    # The stream state is the real assertion: equal actions would also pass for reordered draws.
    assert scalar.explore_rng.bit_generator.state == batched.explore_rng.bit_generator.state


def test_greedy_act_batch_draws_no_randomness(config):
    agent = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    before = agent.explore_rng.bit_generator.state
    agent.act_batch(np.zeros((10, 4), dtype=np.float32))
    assert agent.explore_rng.bit_generator.state == before


@pytest.mark.parametrize("width", [1, 2, 4, 6, 20])
def test_vectorized_evaluation_matches_the_serial_loop(config, width):
    expected, env = [], make_task(config, "default")
    for seed in episode_seeds(config, "default", 6):
        state, _ = env.reset(seed=seed)
        total, length = 0.0, 0
        while True:
            state, reward, terminated, truncated, _ = env.step(0)
            total += float(reward)
            length += 1
            if terminated or truncated:
                expected.append({"steps": length, "return": total})
                break
    env.close()
    config["train"]["num_envs"] = width
    report = evaluate(Rightwards(), config, "default", 6)
    assert report["episodes"] == expected
    assert report["eval_env_steps"] == sum(record["steps"] for record in expected)


def test_evaluation_is_repeatable_and_consumes_no_stream(config):
    agent = DQNAgent(config, "finetune", 0, 4, 2, streams(0))
    before = (agent.explore_rng.bit_generator.state, agent.sample_rng.bit_generator.state)
    first = evaluate(agent, config, "default", 5)
    assert first == evaluate(agent, config, "default", 5)
    assert before == (agent.explore_rng.bit_generator.state, agent.sample_rng.bit_generator.state)


@pytest.mark.parametrize("num_envs", [1, 2, 10])
def test_the_eval_grid_and_budget_do_not_depend_on_num_envs(config, tmp_path, num_envs):
    config["train"]["num_envs"] = num_envs
    config["train"]["train_every"] = num_envs
    config["train"]["timesteps"] = 300
    config["train"]["learning_starts"] = 50
    config["eval"]["every"] = 150
    config["eval"]["episodes"] = 1
    config["eval"]["matrix_episodes"] = 1
    config["output"]["root"] = str(tmp_path)
    config["output"]["videos"] = False
    metrics = train(config, "finetune", 0, tmp_path / "finetune")
    blocks = len(metrics["blocks"])
    assert [row["block_steps"] for row in metrics["learning_curve"]] == [0, 150, 300] * blocks
    assert metrics["env_steps"] == 300 * blocks
    assert all(value is not None for value in metrics["auc"].values())
    assert "dead_fraction_0" in metrics["plasticity"][-1]


def test_train_refuses_a_num_envs_that_skews_the_eval_grid(config, tmp_path):
    config["output"]["videos"] = False
    config["train"]["num_envs"] = 4
    config["train"]["timesteps"] = 300
    config["eval"]["every"] = 150
    with pytest.raises(ValueError, match="must divide"):
        train(config, "finetune", 0, tmp_path / "finetune")


def test_every_shipped_config_declares_a_legal_num_envs():
    for name in ["base", "smoke", "preflight", "cycles", "cbp_slow", "cbp_l2"]:
        config = load_config(f"config/{name}.yaml")
        validate(config)
        assert config["train"]["num_envs"] >= 1


@pytest.mark.parametrize("num_envs,train_every,expected", [(1, 1, 250), (10, 1, 250), (10, 2, 125), (10, 5, 50), (10, 10, 25)])
def test_updates_per_env_step_do_not_depend_on_num_envs(config, tmp_path, num_envs, train_every, expected):
    config["train"]["num_envs"] = num_envs
    config["train"]["train_every"] = train_every
    config["train"]["timesteps"] = 300
    config["train"]["learning_starts"] = 50
    config["eval"]["every"] = 150
    config["eval"]["episodes"] = 1
    config["eval"]["matrix_episodes"] = 1
    config["output"]["root"] = str(tmp_path)
    config["output"]["videos"] = False
    metrics = train(config, "scratch", 0, tmp_path / "scratch", tasks=["default"])
    # 300 steps less the learning_starts warmup, divided by train_every, whatever the slot count.
    assert metrics["updates"] == pytest.approx(expected, abs=num_envs)


def test_a_clip_is_written_for_every_task(config, tmp_path):
    config["train"]["timesteps"] = 300
    config["train"]["learning_starts"] = 50
    config["eval"]["every"] = 150
    config["eval"]["episodes"] = 1
    config["eval"]["matrix_episodes"] = 1
    config["output"]["root"] = str(tmp_path)
    config["output"]["videos"] = True
    metrics = train(config, "finetune", 0, tmp_path / "finetune")
    assert [clip["task"] for clip in metrics["clips"]] == config["benchmark"]["order"]
    for task in config["benchmark"]["order"]:
        clip = tmp_path / "finetune" / "videos" / f"{task}.gif"
        assert clip.exists() and clip.stat().st_size > 0


def test_a_scratch_run_only_clips_its_own_task(config, tmp_path):
    config["train"]["timesteps"] = 300
    config["train"]["learning_starts"] = 50
    config["eval"]["every"] = 150
    config["eval"]["episodes"] = 1
    config["eval"]["matrix_episodes"] = 1
    config["output"]["root"] = str(tmp_path)
    config["output"]["videos"] = True
    metrics = train(config, "scratch", 0, tmp_path / "scratch" / "pole", tasks=["pole"])
    assert [clip["task"] for clip in metrics["clips"]] == ["pole"]
    assert sorted(p.name for p in (tmp_path / "scratch" / "pole" / "videos").iterdir()) == ["pole.gif"]
    assert metrics["clips"][0]["episodes"] == 5
