import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import comb


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


def MRR(
    recommended_items_scores: list[float],
) -> float:
    assert sum(recommended_items_scores) > 0
    recommended_items_scores_np = np.array(recommended_items_scores)
    rr = np.sum(recommended_items_scores_np / (np.arange(len(recommended_items_scores_np)) + 1))
    number_of_positive = np.sum(recommended_items_scores_np)
    return rr / number_of_positive


def ILS(
    recommended_top_Kth_items_embeddings: list[list[float]],
    K: int,
) -> float:
    """Paper: https://dl.acm.org/doi/pdf/10.1145/1060745.1060754
    Intra-List Similarity is a metric that intends to capture the diversity of a list.
    Higher scores denote lower diversity.
    """
    assert len(recommended_top_Kth_items_embeddings) == K

    recommended_top_Kth_items_embeddings_np = np.array(recommended_top_Kth_items_embeddings)

    total_cosine_similarity = 0.0
    total_cosine_similarity = (np.sum(cosine_similarity(recommended_top_Kth_items_embeddings_np)) - 1.0 * K) / 2

    pair_count = comb(K, 2)

    return total_cosine_similarity / pair_count
