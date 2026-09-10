from __future__ import annotations

import math

import torch
import torch.nn as nn


class AdamCBP(torch.optim.Optimizer):
    """Adam with an elementwise step counter so a replaced unit's bias correction resets too."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                gradient = p.grad if not group["weight_decay"] else p.grad.add(p, alpha=group["weight_decay"])
                state = self.state[p]
                if not state:
                    state["step"] = torch.zeros_like(p)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                first = state["exp_avg"] / (1 - torch.pow(beta1, state["step"]))
                second = state["exp_avg_sq"] / (1 - torch.pow(beta2, state["step"]))
                p.addcdiv_(first, second.sqrt_().add_(group["eps"]), value=-group["lr"])
        return loss

    def reset_unit(self, layer, index, axis):
        """Forget everything Adam learned about one unit's incoming row or outgoing column."""
        for parameter, selector in [(layer.weight, (index, slice(None)) if axis == 0 else (slice(None), index)), (layer.bias, index if axis == 0 else None)]:
            state = self.state.get(parameter)
            if state and selector is not None:
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    state[key][selector] = 0


class ContinualBackprop:
    """Reinitialise the least useful mature hidden units, strictly per layer (Dohare et al.)."""

    def __init__(self, layers, settings, generator, device="cpu"):
        self.pairs = layers
        self.decay = settings["decay_rate"]
        self.rate = settings["replacement_rate"]
        self.maturity = settings["maturity_threshold"]
        self.generator = generator
        self.replacements = 0
        self.state = []
        for incoming, _ in layers:
            width = incoming.out_features
            self.state.append({
                "util": torch.zeros(width, device=device),
                "mean_act": torch.zeros(width, device=device),
                "age": torch.zeros(width, device=device),
                "accumulator": 0.0,
            })

    @torch.no_grad()
    def step(self, optimizer, captured):
        replaced = []
        for index, (incoming, outgoing) in enumerate(self.pairs):
            features = captured[index]
            if features is None:
                continue
            state = self.state[index]
            # Ages advance before the bias correction is formed: 1 - decay**0 is zero.
            state["age"] += 1
            correction = 1 - self.decay ** state["age"]
            state["mean_act"].mul_(self.decay).add_(features.mean(dim=0), alpha=1 - self.decay)
            corrected_mean = state["mean_act"] / correction
            outgoing_magnitude = outgoing.weight.abs().sum(dim=0)
            incoming_magnitude = incoming.weight.abs().sum(dim=1)
            contribution = (features - corrected_mean).abs().mean(dim=0) * outgoing_magnitude
            state["util"].mul_(self.decay).add_(contribution / (incoming_magnitude + 1e-12), alpha=1 - self.decay)
            corrected_util = state["util"] / correction
            replaced.append(self.replace(index, incoming, outgoing, state, corrected_mean, corrected_util, optimizer))
        return replaced

    @torch.no_grad()
    def replace(self, index, incoming, outgoing, state, corrected_mean, corrected_util, optimizer):
        eligible = torch.nonzero(state["age"] > self.maturity, as_tuple=False).flatten()
        state["accumulator"] += self.rate * float(eligible.numel())
        # Clamp to what is mature, and keep the remainder pending rather than dropping it.
        count = min(int(state["accumulator"]), eligible.numel())
        state["accumulator"] -= count
        if count <= 0:
            return 0
        chosen = eligible[torch.argsort(corrected_util[eligible])[:count]]
        bound = 1.0 / math.sqrt(incoming.in_features)
        for unit in chosen.tolist():
            # Fold the unit's mean contribution into the next bias so deleting it shifts no output.
            outgoing.bias.add_(outgoing.weight[:, unit] * corrected_mean[unit])
            incoming.weight[unit, :] = torch.empty(incoming.in_features, device=incoming.weight.device).uniform_(-bound, bound, generator=self.generator)
            incoming.bias[unit] = 0.0
            outgoing.weight[:, unit] = 0.0
            state["util"][unit] = 0.0
            state["mean_act"][unit] = 0.0
            state["age"][unit] = 0.0
            if isinstance(optimizer, AdamCBP):
                optimizer.reset_unit(incoming, unit, axis=0)
                optimizer.reset_unit(outgoing, unit, axis=1)
        self.replacements += count
        return count

    @torch.no_grad()
    def statistics(self):
        return {"cbp_replacements": self.replacements, **{f"mean_utility_{index}": float(state["util"].mean()) for index, state in enumerate(self.state)}}


class FeatureProbe:
    """Stashes hidden activations for the batch the gradient flows through, in every arm."""

    def __init__(self, network):
        self.layers = [module for module in network if isinstance(module, nn.Linear)]
        activations = [module for module in network if isinstance(module, nn.ReLU)]
        if len(activations) != len(self.layers) - 1:
            raise RuntimeError("Expected one ReLU per hidden layer")
        self.features = [None] * len(activations)
        self.capturing = False
        for index, module in enumerate(activations):
            module.register_forward_hook(self.record(index))

    def record(self, index):
        def hook(module, inputs, output):
            if self.capturing:
                self.features[index] = output.detach()
        return hook

    def pairs(self):
        return [(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)]


def attach(probe, settings, generator, device="cpu"):
    return ContinualBackprop(probe.pairs(), settings, generator, device)


@torch.no_grad()
def plasticity(network, features, dead_threshold):
    """Weight and feature-rank diagnostics, so a null CBP result can still be interpreted."""
    report = {}
    for index, module in enumerate(m for m in network if isinstance(m, nn.Linear)):
        report[f"weight_norm_{index}"] = float(module.weight.norm())
        report[f"bias_norm_{index}"] = float(module.bias.norm())
    for index, batch in enumerate(features):
        if batch is None or batch.shape[0] < 2:
            continue
        strength = batch.abs().mean(dim=0)
        report[f"dead_fraction_{index}"] = float((strength <= dead_threshold * strength.mean()).float().mean())
        values = torch.linalg.svdvals(batch - batch.mean(dim=0))
        report[f"stable_rank_{index}"] = float(values.sum() ** 2 / (values.square().sum() + 1e-12))
    return report
