from torch import nn


class PLMBasedNRMS(nn.Module):
    def __init__(
        self,
        pretrained: str,
        emb_dim: int,
        multihead_attn_num_heads: int,
        additive_attn_hidden_dim: int,
        hist_size: int,
    ) -> None:
        pass

    # def forward(self)
