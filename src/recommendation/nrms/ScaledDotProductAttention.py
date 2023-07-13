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
        seq_len, d_k = Q.size(-2), Q.size(-1)

        attention_mask = mask if mask is not None else torch.zeros(seq_len, seq_len)  # TODO: attention mask

        attention_weight = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) + attention_mask, dim=-1
        )

        return torch.matmul(attention_weight, V)
