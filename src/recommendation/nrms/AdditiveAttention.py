import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # V
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.rand(input.shape)
