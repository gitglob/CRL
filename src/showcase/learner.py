from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..cbp import AdamCBP, FeatureProbe, attach, plasticity
from .envs import dimensions
from .replay import UnrollReservoir, mix_batch

torch.set_num_threads(1)


@torch.no_grad()
def vtrace(reward, values, next_values, terminated, boundary, log_ratio, gamma):
    """Bootstrap truncations from final observations, but never carry a trace across a reset."""
    ratio = log_ratio.exp()
    rho = ratio.clamp(max=1)
    discount = gamma * (~terminated).float()
    trace_discount = gamma * (~boundary).float()
    delta = rho * (reward + discount * next_values - values)
    correction = torch.zeros_like(values[0])
    targets = torch.empty_like(values)
    for time in range(values.shape[0] - 1, -1, -1):
        correction = delta[time] + trace_discount[time] * rho[time] * correction
        targets[time] = values[time] + correction
    following = torch.cat((targets[1:], next_values[-1:]), dim=0)
    following = torch.where(boundary, next_values, following)
    advantages = rho * (reward + discount * following - values)
    return targets, advantages


def cloning_losses(logits, values, behavior_logits, behavior_values, replay_mask):
    old_log_probs = behavior_logits.detach().log_softmax(dim=-1)
    kl = (old_log_probs.exp() * (old_log_probs - logits.log_softmax(dim=-1))).sum(dim=-1)
    value_error = (values - behavior_values.detach()).square()
    return (kl * replay_mask).mean(), (value_error * replay_mask).mean()


def mlp(inputs, widths, outputs):
    modules = []
    for width in widths:
        modules.extend((nn.Linear(inputs, width), nn.ReLU()))
        inputs = width
    modules.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*modules)


class ActorCritic:
    def __init__(self, config, suite, arm, device=None):
        self.config = deepcopy(config)
        self.suite, self.arm = suite, arm
        self.device = device or config["device"]
        inputs, actions = dimensions(suite)
        self.actions = actions
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            self.actor = mlp(inputs, config["train"]["hidden_sizes"], actions).to(self.device)
            self.critic = mlp(inputs, config["train"]["hidden_sizes"], 1).to(self.device)
        self.parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = AdamCBP(self.parameters, lr=config["train"]["learning_rate"])
        self.probes = [FeatureProbe(self.actor), FeatureProbe(self.critic)]
        self.cbp = []
        if arm in ("cbp", "clear_cbp"):
            for index, probe in enumerate(self.probes):
                generator = torch.Generator(device=self.device).manual_seed(97 + index)
                self.cbp.append(attach(probe, config["cbp"], generator, self.device))
        self.memory = UnrollReservoir(config["clear"]["capacity"], config["train"]["unroll_length"], seed=83) if arm in ("replay", "clear", "clear_cbp") else None
        self.action_rng = torch.Generator(device=self.device).manual_seed(61)
        self.batch_rng = np.random.default_rng(73)
        self.updates = 0
        self.last = {}

    @torch.no_grad()
    def act(self, observations, greedy=False):
        states = torch.as_tensor(np.asarray(observations), device=self.device, dtype=torch.float32)
        logits = self.actor(states)
        actions = logits.argmax(dim=-1) if greedy else torch.multinomial(logits.softmax(dim=-1), 1, generator=self.action_rng).squeeze(-1)
        values = self.critic(states).squeeze(-1)
        return actions.cpu().numpy(), logits.cpu().numpy(), values.cpu().numpy()

    def optimize(self, fresh):
        if self.memory is not None:
            self.memory.add(fresh)
        data, mask = mix_batch(fresh, self.memory, self.config["clear"]["replay_ratio"], self.batch_rng)
        batch = {key: torch.as_tensor(value, device=self.device) for key, value in data.items()}
        states = batch["observation"].float()
        shape = states.shape[:2]
        for probe in self.probes:
            probe.capturing = True
        logits = self.actor(states.flatten(0, 1)).reshape(*shape, self.actions)
        values = self.critic(states.flatten(0, 1)).reshape(shape)
        for probe in self.probes:
            probe.capturing = False
        with torch.no_grad():
            next_values = self.critic(batch["next_observation"].float().flatten(0, 1)).reshape(shape)
        log_policy = logits.log_softmax(dim=-1)
        action_log_prob = log_policy.gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
        behavior_log_prob = batch["behavior_logits"].log_softmax(dim=-1).gather(-1, batch["action"].unsqueeze(-1)).squeeze(-1)
        log_ratio = action_log_prob.detach() - behavior_log_prob
        targets, advantage = vtrace(batch["reward"], values.detach(), next_values, batch["terminated"], batch["boundary"], log_ratio, self.config["train"]["gamma"])
        policy_loss = -(action_log_prob * advantage).mean()
        value_loss = (values - targets).square().mean()
        entropy = -(log_policy.exp() * log_policy).sum(dim=-1).mean()
        policy_clone, value_clone = cloning_losses(logits, values, batch["behavior_logits"], batch["behavior_value"], torch.as_tensor(mask, device=self.device))
        train, clear = self.config["train"], self.config["clear"]
        loss = policy_loss + train["value_weight"] * value_loss - train["entropy_weight"] * entropy
        if self.arm in ("clear", "clear_cbp"):
            loss = loss + clear["policy_cloning_weight"] * policy_clone + clear["value_cloning_weight"] * value_clone
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite actor-critic loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(self.parameters, train["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        for cbp, probe in zip(self.cbp, self.probes):
            cbp.step(self.optimizer, probe.features)
        self.updates += 1
        self.last = {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()), "policy_cloning": float(policy_clone.detach()), "value_cloning": float(value_clone.detach()), "gradient_norm": float(gradient), "max_abs_value": float(values.detach().abs().max()), "mean_rho": float(log_ratio.exp().clamp(max=1).mean()), "replay_fraction": float(mask.mean())}
        return self.last

    @torch.no_grad()
    def diagnostics(self, observations):
        result = {"updates": self.updates, "memory_transitions": len(self.memory.items) * self.memory.unroll_length if self.memory else 0, **self.last}
        states = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        for index, (name, net, probe) in enumerate(zip(("actor", "critic"), (self.actor, self.critic), self.probes)):
            saved = probe.features
            probe.capturing = True
            net(states)
            probe.capturing = False
            values = plasticity(net, probe.features, self.config["cbp"]["dead_threshold"])
            probe.features = saved
            result.update({f"{name}_{key}": value for key, value in values.items()})
            if self.cbp:
                result.update({f"{name}_{key}": value for key, value in self.cbp[index].statistics().items()})
        if str(self.device).startswith("cuda"):
            result["gpu_peak_bytes"] = torch.cuda.max_memory_allocated()
        return result

    def weights(self):
        return {"actor": deepcopy(self.actor.state_dict()), "critic": deepcopy(self.critic.state_dict())}

    def load_weights(self, weights):
        self.actor.load_state_dict(weights["actor"])
        self.critic.load_state_dict(weights["critic"])

    def probe_learner(self):
        agent = ActorCritic(self.config, self.suite, "finetune", device=self.device)
        agent.load_weights(self.weights())
        return agent

    def checkpoint(self):
        return {"schema": 1, "algorithm": "vtrace_actor_critic", "config": self.config, "suite": self.suite, "arm": self.arm, "weights": self.weights(), "optimizer": self.optimizer.state_dict(), "updates": self.updates, "last": self.last, "action_rng": self.action_rng.get_state(), "batch_rng": self.batch_rng.bit_generator.state, "memory": self.memory.state_dict() if self.memory else None, "cbp": [{"state": cbp.state, "replacements": cbp.replacements, "rng": cbp.generator.get_state()} for cbp in self.cbp]}

    def restore(self, payload):
        if payload["schema"] != 1 or payload["suite"] != self.suite or payload["arm"] != self.arm:
            raise ValueError("Incompatible actor-critic checkpoint")
        self.load_weights(payload["weights"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.updates, self.last = payload["updates"], payload["last"]
        self.action_rng.set_state(payload["action_rng"].cpu())
        self.batch_rng.bit_generator.state = payload["batch_rng"]
        if self.memory:
            self.memory.load_state_dict(payload["memory"])
        for cbp, state in zip(self.cbp, payload["cbp"]):
            cbp.state = [{key: value.to(self.device) if torch.is_tensor(value) else value for key, value in entry.items()} for entry in state["state"]]
            cbp.replacements = state["replacements"]
            cbp.generator.set_state(state["rng"].cpu())


def load_agent(path, device=None):
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    agent = ActorCritic(payload["config"], payload["suite"], payload["arm"], device=device)
    agent.restore(payload)
    return agent, payload
