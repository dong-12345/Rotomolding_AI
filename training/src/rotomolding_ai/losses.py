from __future__ import annotations

import torch
from torch.nn import functional as F


def thickness_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_index: torch.Tensor,
    smooth_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    data_loss = F.smooth_l1_loss(prediction, target)
    source, destination = edge_index
    smooth_loss = (prediction[source] - prediction[destination]).abs().mean()
    total = data_loss + smooth_weight * smooth_loss
    return total, {
        "total": float(total.detach().cpu()),
        "data": float(data_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
    }
