import pytest
from utils.metrics.metrics import DCG_K, nDCG_K


@pytest.mark.parametrize(
    "item_ids_to_relevance_scores_map,ordered_recommended_items,K,expected",
    [
        ({"item_a": 1.0, "item_b": 0.0, "item_c": 1.0}, ["item_a", "item_b", "item_c"], 3, 1.5),
        ({"item_a": 1.0, "item_b": 1.0, "item_c": 1.0}, ["item_a", "item_b", "item_c"], 3, 2.1309),
        ({"item_a": 0.0, "item_b": 0.0, "item_c": 0.0}, ["item_a", "item_b", "item_c"], 3, 0.0),
    ],
)
def test_DCG_at_K(
    item_ids_to_relevance_scores_map: dict[str, float], ordered_recommended_items: list[str], K: int, expected: float
) -> None:
    assert DCG_K(item_ids_to_relevance_scores_map, ordered_recommended_items, K) == pytest.approx(expected, 0.001)


@pytest.mark.parametrize(
    "item_ids_to_relevance_scores_map,ordered_recommended_items,K,N,expected",
    [
        ({"item_a": 1.0, "item_b": 0.0, "item_c": 1.0}, ["item_a", "item_b", "item_c"], 3, 5, 0.5087),
        ({"item_a": 1.0, "item_b": 1.0, "item_c": 1.0}, ["item_a", "item_b", "item_c"], 3, 5, 0.7227),
        ({"item_a": 0.0, "item_b": 0.0, "item_c": 0.0}, ["item_a", "item_b", "item_c"], 3, 5, 0.0),
    ],
)
def test_nDCG_at_K(
    item_ids_to_relevance_scores_map: dict[str, float],
    ordered_recommended_items: list[str],
    K: int,
    N: int,
    expected: float,
) -> None:
    assert nDCG_K(item_ids_to_relevance_scores_map, ordered_recommended_items, K, N) == pytest.approx(expected, 0.001)
