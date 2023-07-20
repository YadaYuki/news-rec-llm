import polars as pl
from typing import Tuple
from collections import defaultdict


def create_news_and_user_ids_to_clicked_map(behavior_df: pl.DataFrame) -> dict[Tuple[str, str], int]:
    total_click_df = (
        behavior_df.select(["user_id", "news_id", "clicked"])
        .groupby(["user_id", "news_id"])
        .agg(total_click=pl.col("clicked").sum())
        .filter(pl.col("total_click") > 0)
    )

    user_ids, news_ids, total_click = (
        total_click_df["user_id"].to_list(),
        total_click_df["news_id"].to_list(),
        total_click_df["total_click"].to_list(),
    )

    d: dict[Tuple[str, str], int] = {(user_ids[i], news_ids[i]): total_click[i] for i in range(len(total_click_df))}

    return defaultdict(int, d)
