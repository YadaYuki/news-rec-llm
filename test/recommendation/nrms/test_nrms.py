import torch
from transformers import AutoConfig
from src.recommendation.nrms.PLMBasedNRMS import PLMBasedNRMS


def test_nrms() -> None:
    pretrained: str = "bert-base-uncased"
    multihead_attn_num_heads: int = 16
    additive_attn_hidden_dim: int = 200
    batch_size, seq_len, hist_size, emb_dim = 20, 10, 4, AutoConfig.from_pretrained(pretrained).hidden_size

    plm_based_nrms = PLMBasedNRMS(pretrained, emb_dim, multihead_attn_num_heads, additive_attn_hidden_dim, hist_size)

    candidate_news_batch = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
    news_histories_batch = torch.arange(batch_size * hist_size * seq_len).view(batch_size, hist_size, seq_len)

    assert tuple(plm_based_nrms(candidate_news_batch, news_histories_batch)) == (batch_size, 1)
