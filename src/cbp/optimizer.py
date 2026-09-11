import torch


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
