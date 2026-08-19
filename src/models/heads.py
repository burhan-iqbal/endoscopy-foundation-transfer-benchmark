"""Classification heads for binary CE-NBI prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, num_classes: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_head(kind: str, in_dim: int) -> nn.Module:
    kind = (kind or "linear").lower()
    if kind == "linear":
        return LinearHead(in_dim)
    if kind == "mlp":
        return MLPHead(in_dim)
    raise ValueError(f"Unknown head: {kind}")
