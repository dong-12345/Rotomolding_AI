from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(layers - 1, 1)):
            modules.extend([nn.Linear(current, hidden_dim), nn.SiLU()])
            current = hidden_dim
        modules.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class MeshMessageBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = MLP(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, layers=3)
        self.update = MLP(hidden_dim * 2, hidden_dim, hidden_dim, layers=3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        source, destination = edge_index
        messages = self.message(
            torch.cat([node_state[source], node_state[destination], edge_features], dim=-1)
        )
        aggregate = torch.zeros_like(node_state)
        # index_add_ requires source and destination tensors to share a dtype;
        # AMP may produce FP16 messages while the residual state stays FP32.
        aggregate.index_add_(0, destination, messages.to(dtype=aggregate.dtype))
        updated = self.update(torch.cat([node_state, aggregate], dim=-1))
        return self.norm(node_state + self.dropout(updated))


class RotomoldingMeshGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        process_dim: int,
        hidden_dim: int = 128,
        blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "process_dim": process_dim,
            "hidden_dim": hidden_dim,
            "blocks": blocks,
            "dropout": dropout,
        }
        self.node_encoder = MLP(node_dim, hidden_dim, hidden_dim, layers=3)
        self.process_encoder = MLP(process_dim, hidden_dim, hidden_dim, layers=3)
        self.blocks = nn.ModuleList(
            [MeshMessageBlock(hidden_dim, edge_dim, dropout) for _ in range(blocks)]
        )
        self.output_head = MLP(hidden_dim, hidden_dim, 1, layers=3)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        process: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        node_state = self.node_encoder(node_features)
        process_state = self.process_encoder(process)
        node_state = node_state + process_state[batch]
        for block in self.blocks:
            node_state = block(node_state, edge_index, edge_features)
            node_state = node_state + process_state[batch]
        return F.softplus(self.output_head(node_state).squeeze(-1))
