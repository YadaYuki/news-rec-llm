import torch
from src.recommendation.nrms.AdditiveAttention import AdditiveAttention
from src.recommendation.nrms.ScaledDotProductAttention import ScaledDotProductAttention


def test_additive_attention() -> None:
    batch_size, seq_len, emb_dim, hidden_dim = 20, 10, 30, 5
    attn = AdditiveAttention(emb_dim, hidden_dim)
    input = torch.rand(batch_size, seq_len, emb_dim)
    assert tuple(attn(input).shape) == (batch_size, seq_len, emb_dim)


def test_scaled_dot_product_attention() -> None:
    batch_size, seq_len, d_k, head_size = 20, 10, 40, 8
    attention_mask = None
    attn = ScaledDotProductAttention()
    input = torch.rand(batch_size, head_size, seq_len, d_k)
    assert tuple(attn(input, input, input, attention_mask).shape) == (batch_size, head_size, seq_len, d_k)


def test_scaled_dot_product_attention_with_mask() -> None:
    batch_size, seq_len, d_k, head_size = 20, 10, 40, 8
    attention_mask = torch.zeros(size=(seq_len, seq_len), dtype=torch.bool)
    attention_mask[0, 0] = True
    attn = ScaledDotProductAttention()
    input = torch.rand(batch_size, head_size, seq_len, d_k)
    output = attn(input, input, input, attention_mask)
    assert tuple(output.shape) == (batch_size, head_size, seq_len, d_k)
