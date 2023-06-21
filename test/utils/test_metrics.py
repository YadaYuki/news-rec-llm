import pytest
from utils.metrics.metrics import DCG_K, nDCG_K, MRR, ILS


@pytest.mark.parametrize(
    "recommended_item_scores,K,expected",
    [
        ([1.0, 0.0, 1.0], 3, 1.5),
        ([1.0, 1.0, 1.0], 3, 2.1309),
        ([0.0, 0.0, 0.0], 3, 0.0),
    ],
)
def test_DCG_at_K(recommended_item_scores: list[float], K: int, expected: float) -> None:
    assert DCG_K(recommended_item_scores, K) == pytest.approx(expected, 0.001)


@pytest.mark.parametrize(
    "recommended_item_scores,K,expected",
    [
        ([1.0, 0.0, 1.0], 3, 0.7039),
        ([1.0, 1.0, 1.0], 3, 1),
        ([0.0, 0.0, 0.0], 3, 0.0),
    ],
)
def test_nDCG_at_K(
    recommended_item_scores: list[float],
    K: int,
    expected: float,
) -> None:
    assert nDCG_K(recommended_item_scores, K) == pytest.approx(expected, 0.001)


@pytest.mark.parametrize(
    "recommended_item_scores,expected",
    [
        ([1.0, 0.0, 1.0], 0.6667),
        ([1.0, 1.0, 1.0], 0.6111),
    ],
)
def test_MRR(
    recommended_item_scores: list[float],
    expected: float,
) -> None:
    assert MRR(recommended_item_scores) == pytest.approx(expected, 0.001)


@pytest.mark.parametrize(
    "recommended_top_Kth_items_embeddings,K,expected",
    [
        ([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], 3, 1.0),
        ([[1.0, 0.1, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 3, 0.4702),
    ],
)
def test_ILS(
    recommended_top_Kth_items_embeddings: list[list[float]],
    K: int,
    expected: float,
) -> None:
    assert ILS(recommended_top_Kth_items_embeddings, K) == pytest.approx(expected, 0.001)
