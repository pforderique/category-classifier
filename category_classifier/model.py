"""PyTorch model definitions."""

from __future__ import annotations

import torch


class LinearClassifier(torch.nn.Module):
    """Single linear classification head for fixed embeddings."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear layer."""
        return self.linear(x)
