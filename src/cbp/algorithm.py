import math

import torch

from .optimizer import AdamCBP


class ContinualBackprop:
    """Reinitialise the least useful mature hidden units, strictly per layer (Dohare et al.)."""

    def __init__(self, layers, settings, generator, device="cpu"):
        self.pairs = layers
        self.decay = settings["decay_rate"]
        self.rate = settings["replacement_rate"]
        self.maturity = settings["maturity_threshold"]
        self.utility = settings.get("utility", "adaptable_contribution")
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
        selections = []
        for index, (incoming, outgoing) in enumerate(self.pairs):
            features = captured[index]
            if features is None:
                selections.append((torch.empty(0, dtype=torch.long, device=incoming.weight.device), None))
                continue
            state = self.state[index]
            # Ages advance before the bias correction is formed: 1 - decay**0 is zero.
            state["age"] += 1
            correction = 1 - self.decay ** state["age"]
            state["mean_act"].mul_(self.decay).add_(features.mean(dim=0), alpha=1 - self.decay)
            corrected_mean = state["mean_act"] / correction
            outgoing_magnitude = outgoing.weight.abs().sum(dim=0)
            incoming_magnitude = incoming.weight.abs().sum(dim=1)
            if self.utility == "contribution":
                contribution = features.abs().mean(dim=0) * outgoing_magnitude
            else:
                contribution = (features - corrected_mean).abs().mean(dim=0) * outgoing_magnitude / (incoming_magnitude + 1e-12)
            state["util"].mul_(self.decay).add_(contribution, alpha=1 - self.decay)
            corrected_util = state["util"] / correction
            selections.append((self.select(state, corrected_util), corrected_mean.clone()))
        self.apply_replacements(selections, optimizer)
        return [chosen.numel() for chosen, _ in selections]

    def select(self, state, corrected_util):
        eligible = torch.nonzero(state["age"] > self.maturity, as_tuple=False).flatten()
        state["accumulator"] += self.rate * float(eligible.numel())
        count = min(int(state["accumulator"]), eligible.numel())
        state["accumulator"] -= count
        return eligible[torch.argsort(corrected_util[eligible], stable=True)[:count]]

    @torch.no_grad()
    def apply_replacements(self, selections, optimizer):
        # Capture every compensation before any layer is changed.
        corrections = []
        for (_, outgoing), (chosen, mean) in zip(self.pairs, selections):
            corrections.append((outgoing.weight[:, chosen] * mean[chosen]).sum(dim=1) if chosen.numel() else None)
        for (incoming, outgoing), (chosen, _), correction in zip(self.pairs, selections, corrections):
            if not chosen.numel():
                continue
            outgoing.bias.add_(correction)
        for (incoming, _), (chosen, _) in zip(self.pairs, selections):
            if not chosen.numel():
                continue
            bound = 1.0 / math.sqrt(incoming.in_features)
            incoming.weight[chosen] = torch.empty((chosen.numel(), incoming.in_features), device=incoming.weight.device).uniform_(-bound, bound, generator=self.generator)
            incoming.bias[chosen] = 0
        # Zero columns last: a downstream row reset must never resurrect them.
        for index, ((incoming, outgoing), (chosen, _)) in enumerate(zip(self.pairs, selections)):
            if not chosen.numel():
                continue
            outgoing.weight[:, chosen] = 0
            for key in ("util", "mean_act", "age"):
                self.state[index][key][chosen] = 0
            if isinstance(optimizer, AdamCBP):
                optimizer.reset_unit(incoming, chosen, axis=0)
                optimizer.reset_unit(outgoing, chosen, axis=1)
            self.replacements += chosen.numel()

    @torch.no_grad()
    def replace(self, index, incoming, outgoing, state, corrected_mean, corrected_util, optimizer):
        chosen = self.select(state, corrected_util)
        selections = [(torch.empty(0, dtype=torch.long, device=incoming.weight.device), None) for _ in self.pairs]
        selections[index] = (chosen, corrected_mean)
        self.apply_replacements(selections, optimizer)
        return chosen.numel()

    @torch.no_grad()
    def statistics(self):
        return {"cbp_replacements": self.replacements, **{f"mean_utility_{index}": float(state["util"].mean()) for index, state in enumerate(self.state)}}


def attach(probe, settings, generator, device="cpu"):
    return ContinualBackprop(probe.pairs(), settings, generator, device)
