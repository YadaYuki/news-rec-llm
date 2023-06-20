import pytest
from utils.metrics.metrics import DCG_K, nDCG_K


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
