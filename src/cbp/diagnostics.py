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
def plasticity(network, features, include_rank=True):
    """Dohare PPO diagnostics on uncentered interaction activations, extended to either network."""
    report = {f"weight_magnitude_{i}": float(layer.weight.abs().mean())
              for i, layer in enumerate(m for m in network if isinstance(m, nn.Linear))}
    dormant = sum(int(((batch > 0).sum(dim=0) <= 0.01 * len(batch)).sum()) for batch in features)
    report["dormant_percent"] = 100 * dormant / sum(batch.shape[1] for batch in features)
    if include_rank:
        singular = torch.linalg.svdvals(features[-1].float())
        total = singular.sum()
        rank = int(torch.searchsorted(singular.cumsum(0), 0.99 * total)) + 1 if total > 0 else 0
        report["stable_rank"] = rank
        report["stable_rank_percent"] = 100 * rank / features[-1].shape[1]
    return report


class InteractionDiagnostics:
    """Bounded activation windows and exact diagnostic clocks across fresh vector transitions."""

    def __init__(self, networks, settings=None):
        settings = settings or {}
        self.window = settings.get("window_steps", 1000)
        self.period = settings.get("period", 1000)
        self.rank_period = settings.get("rank_period", 10000)
        self.networks = networks
        self.buffers = [[torch.empty(self.window, layer.out_features)
                         for layer in list(net)[0:-1:2]] for net in networks]
        self.steps = 0
        self.rows = []

    def metadata(self):
        return {"schema": 1, "window_steps": self.window, "period": self.period,
                "rank_period": self.rank_period, "source": "fresh interaction activations",
                "ordering": "timestep then environment index", "centered": False,
                "stable_rank": "smallest rank containing at least 99% of singular-value mass",
                "dormant": "active on at most 1% of observations, pooled over hidden units",
                "weights": "per-linear-layer mean absolute weight, excluding bias"}

    @torch.no_grad()
    def observe(self, features):
        features = [[batch.detach().cpu() for batch in network] for network in features]
        start, count = 0, len(features[0][0])
        while start < count:
            position = self.steps % self.window
            size = min(count - start, self.window - position, self.period - self.steps % self.period)
            for buffers, batches in zip(self.buffers, features):
                for buffer, batch in zip(buffers, batches):
                    buffer[position:position + size].copy_(batch[start:start + size])
            self.steps += size
            start += size
            if self.steps >= self.window and self.steps % self.period == 0:
                row = {"env_steps": self.steps}
                for name, network, buffers in zip(("actor", "critic"), self.networks, self.buffers):
                    values = plasticity(network, buffers, self.steps % self.rank_period == 0)
                    row.update({f"{name}_{key}": value for key, value in values.items()})
                self.rows.append(row)

    def drain(self):
        rows, self.rows = self.rows, []
        return rows
