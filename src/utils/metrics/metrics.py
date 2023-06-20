import numpy as np


def nDCG_K(
    recommended_top_Kth_items_scores: list[float],
    K: int,
) -> float:
    return DCG_K(recommended_top_Kth_items_scores, K) / DCG_K(np.ones(K).tolist(), K)


def DCG_K(
    recommended_top_Kth_items_scores: list[float],
    K: int,
) -> float:
    assert len(recommended_top_Kth_items_scores) == K
    Kth_items_scores = np.array(recommended_top_Kth_items_scores)
    gains = 2**Kth_items_scores - 1
    discounts = np.log2(np.arange(1, K + 1) + 1)
    return sum(gains / discounts)
