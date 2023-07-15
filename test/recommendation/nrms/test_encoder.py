from src.recommendation.nrms.PLMBasedNewsEncoder import PLMBasedNewsEncoder
import torch
from transformers import AutoConfig


def test_news_encoder() -> None:
    pretrained: str = "bert-base-uncased"
    last_attn_num_heads: int = 12
    additive_attn_hidden_dim: int = 200
    batch_first: bool = True
    plm_news_encoder = PLMBasedNewsEncoder(pretrained, last_attn_num_heads, additive_attn_hidden_dim, batch_first)

    batch_size, seq_len, emb_dim = 20, 10, AutoConfig.from_pretrained(pretrained).hidden_size

    input_tensor = torch.arange(batch_size * seq_len).view(batch_size, seq_len)

    plm_news_encoder(input_tensor)
