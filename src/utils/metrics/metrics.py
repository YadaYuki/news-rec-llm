import math


def nDCG_K(
    item_ids_to_relevance_scores_map: dict[str, float],
    ordered_recommend_items: list[str],
    K: int,
    number_of_positive: int,
) -> float:
    return DCG_K(item_ids_to_relevance_scores_map, ordered_recommend_items, K) / sum(
        [1 / math.log2(1 + (i + 1)) for i in range(number_of_positive)]
    )


def DCG_K(
    item_ids_to_relevance_scores_map: dict[str, float],
    ordered_recommend_items: list[str],
    K: int,
) -> float:
    assert len(item_ids_to_relevance_scores_map) == K
    assert len(ordered_recommend_items) == K

    return sum(
        [
            (2 ** item_ids_to_relevance_scores_map[item] - 1) / math.log2(1 + (i + 1))
            for i, item in enumerate(ordered_recommend_items)
        ]
    )
