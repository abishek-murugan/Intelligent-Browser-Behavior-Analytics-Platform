"""PyTorch model used to forecast the next browsing session."""

from __future__ import annotations

import torch
from torch import nn


class NextSessionLSTM(nn.Module):
    """Jointly predict next-session numeric features and category."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_categories: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regression_head = nn.Linear(hidden_size, input_size)
        self.classification_head = nn.Linear(hidden_size, num_categories)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return next-session feature estimates and category logits."""
        output, _ = self.lstm(sequence)
        representation = output[:, -1, :]
        return self.regression_head(representation), self.classification_head(representation)
