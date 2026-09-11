import torch
import torch.nn as nn


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
        report[f"stable_rank_{index}"] = float(values.square().sum() / (values[0].square() + 1e-12))
        report[f"singular_value_participation_ratio_{index}"] = float(values.sum() ** 2 / (values.square().sum() + 1e-12))
    return report
