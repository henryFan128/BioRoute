from __future__ import annotations

import math

import torch


def monitor_gradients(model: torch.nn.Module) -> float:
    total_sq_norm = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().data.norm(2)
        total_sq_norm += float(grad_norm.item()) ** 2
    return math.sqrt(total_sq_norm)
