from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbp import AdamCBP, FeatureProbe, attach, plasticity
from .replay import PersistentMemory, Replay

# A 4-128-128-2 net is far too small to benefit from intra-op threading, and a study runs
# dozens of these processes at once: without this they oversubscribe every core.
torch.set_num_threads(1)


def network(inputs, hidden_sizes, outputs):
    layers, previous = [], inputs
    for width in hidden_sizes:
        layers.extend([nn.Linear(previous, width), nn.ReLU()])
        previous = width
    return nn.Sequential(*layers, nn.Linear(previous, outputs))


def epsilon(settings, steps, budget):
    """Restarted at every task boundary, identically for every arm including scratch."""
    span = max(1.0, budget * settings["epsilon_decay_fraction"])
    fraction = min(1.0, steps / span)
    return settings["epsilon_start"] + fraction * (settings["epsilon_end"] - settings["epsilon_start"])


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def double_targets(reward, terminated, next_target, next_online, gamma):
    """Truncation at the step limit is not a terminal state, so only `terminated` stops the bootstrap."""
    actions = next_online.argmax(dim=1)
    continuation = next_target.gather(1, actions[:, None]).squeeze(1)
    return reward + gamma * torch.where(terminated, torch.zeros_like(continuation), continuation)


class DQNAgent:
    def __init__(self, config, arm, seed, obs_size, actions, streams):
        self.settings = config["train"]
        self.config = config
        self.arm = arm
        self.gamma = self.settings["gamma"]
        self.device = resolve_device(self.settings["device"])
        self.uses_replay = arm in ("replay", "replay_cbp")
        self.uses_cbp = arm in ("cbp", "replay_cbp")

        init_seed = int(streams["weight_init"].generate_state(1)[0])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(init_seed)
            self.online = network(obs_size, self.settings["hidden_sizes"], actions).to(self.device)
        self.target = network(obs_size, self.settings["hidden_sizes"], actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)

        self.optimizer = AdamCBP(self.online.parameters(), lr=self.settings["learning_rate"], weight_decay=self.settings["l2_lambda"])
        self.probe = FeatureProbe(self.online)
        self.cbp = None
        if self.uses_cbp:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(streams["cbp_reinit"].generate_state(1)[0]))
            self.cbp = attach(self.probe, config["cbp"], generator, self.device)

        self.replay = Replay(self.settings["buffer_capacity"], obs_size)
        self.memory = PersistentMemory(config["replay"]["capacity"], obs_size) if self.uses_replay else None
        self.sample_rng = np.random.default_rng(streams["replay_sampling"])
        # The reservoir draws on its own stream: sharing one with minibatch sampling would
        # make a replay agent see different batches than fine-tuning on the very first task,
        # where the two are supposed to be identical.
        self.memory_rng = np.random.default_rng(streams["memory"])
        self.explore_rng = np.random.default_rng(streams["exploration"])
        self.actions = actions
        self.updates = 0
        self.task = None
        self.last_loss = None

    def begin_task(self, name):
        """Weights, optimizer and target network carry over; the current-task buffer never does."""
        self.task = name
        self.replay.clear()
        if self.memory is not None:
            self.memory.begin(self.memory_rng, name)

    @torch.no_grad()
    def act(self, state, exploration=0.0):
        if exploration and self.explore_rng.random() < exploration:
            return int(self.explore_rng.integers(0, self.actions))
        values = self.online(torch.as_tensor(state, dtype=torch.float32, device=self.device)[None])
        return int(values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminated):
        self.replay.add(state, action, reward, next_state, terminated)
        if self.memory is not None:
            self.memory.absorb(self.memory_rng, self.task, state, action, reward, next_state, terminated)

    def ready(self):
        # Gated on the current task's own transitions: otherwise a replay agent would open
        # every later task with a thousand gradient steps of pure rehearsal.
        return self.replay.size >= self.settings["learning_starts"]

    def batch(self):
        size = self.settings["batch_size"]
        # Rehearsal only ever draws from *earlier* tasks. During the first task there is
        # nothing to rehearse, so a replay agent must behave exactly like fine-tuning.
        rehearsable = self.memory.past_size(self.task) if self.memory is not None else 0
        past = int(round(size * self.config["replay"]["ratio"])) if rehearsable > 0 else 0
        current = self.replay.sample(self.sample_rng, size - past, self.device)
        if not past:
            return current
        older = self.memory.sample(self.sample_rng, past, self.device, exclude=self.task)
        return {key: torch.cat([current[key], older[key]]) for key in current}

    def optimize(self):
        batch = self.batch()
        # Capture only the s-batch: the s' forward below is off-gradient and its
        # activations describe a distribution the update never touched.
        self.probe.capturing = True
        values = self.online(batch["state"]).gather(1, batch["action"][:, None]).squeeze(1)
        self.probe.capturing = False
        with torch.no_grad():
            next_target = self.target(batch["next_state"])
            next_online = self.online(batch["next_state"])
        targets = double_targets(batch["reward"], batch["terminated"], next_target, next_online, self.gamma)
        loss = F.smooth_l1_loss(values, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.settings["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        if self.cbp is not None:
            self.cbp.step(self.optimizer, self.probe.features)
        self.updates += 1
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.online.parameters()):
                target.lerp_(online, self.settings["target_tau"])
        self.last_loss = float(loss.detach())

    def statistics(self):
        report = {"updates": self.updates, "buffer": self.replay.size, "memory": self.memory.size if self.memory is not None else 0, "loss": self.last_loss}
        report.update(plasticity(self.online, self.probe.features, self.config["cbp"]["dead_threshold"]))
        if self.cbp is not None:
            report.update(self.cbp.statistics())
        return report

    def save(self, path, metadata):
        torch.save({"model": self.online.state_dict(), "metadata": metadata}, path)


def load_dqn(checkpoint, config, arm, seed, obs_size, actions, streams):
    agent = DQNAgent(config, arm, seed, obs_size, actions, streams)
    payload = torch.load(checkpoint, map_location=agent.device, weights_only=False)
    agent.online.load_state_dict(payload["model"])
    agent.target.load_state_dict(payload["model"])
    return agent, payload["metadata"]
