from torch import nn
import torch
from transformers import AutoModel, AutoConfig
from .AdditiveAttention import AdditiveAttention


class PLMBasedNewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "bert-base-uncased",
        last_attn_num_heads: int = 12,
        additive_attn_hidden_dim: int = 200,
        batch_first: bool = True,
    ):
        self.plm = AutoModel.from_pretrained(pretrained)

        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden_size, num_heads=last_attn_num_heads, batch_first=batch_first
        )
        self.additive_attention = AdditiveAttention(plm_hidden_size, additive_attn_hidden_dim)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        return torch.Tensor()
