import torch
from torch import nn
import math


class ScaledDotProductAttention(nn.Module):
    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ScaledDotProductAttention's forward
        Q -- [batch_size, head_size, seq_len, d_k] | [batch_size, seq_len, d_k]
        K -- [batch_size, head_size, seq_len, d_k] | [batch_size, seq_len, d_k]
        V -- [batch_size, head_size, seq_len, d_k] | [batch_size, seq_len, d_k]
        """
        batch_size, head_size, seq_len, d_k = Q.size()

        attention_mask = (
            torch.zeros(batch_size, seq_len, seq_len).masked_fill_(mask, -float("inf"))
            if mask is not None
            else torch.zeros(batch_size, seq_len, seq_len)
        )

        attention_weight = torch.softmax(
            (torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)).view(head_size, batch_size, seq_len, seq_len)
            + attention_mask,
            dim=-1,
        ).view(batch_size, head_size, seq_len, seq_len)

        return torch.matmul(attention_weight, V)
