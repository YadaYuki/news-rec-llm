import numpy as np
import polars as pl
from typing import Tuple
from .NewsRecommenderBase import NewsRecommenderBase
from utils.list import uniq
import random


class RandomNewsRecommender(NewsRecommenderBase):
    def __init__(self) -> None:
        self.unique_news_id: list[str] = []

    def fit(self, behavior_df: pl.DataFrame, news_df: pl.DataFrame) -> None:
        self.unique_news_id = uniq(behavior_df["news_id"].to_list())

    def recommend_top_k_items(
        self, user_ids: np.ndarray, val_behavior_df: pl.DataFrame, val_news_df: pl.DataFrame, K: int = 10
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        user_id_to_recommend_news_ids_map: dict[str, np.ndarray] = {
            user_id: np.array(random.sample(self.unique_news_id, K)) for user_id in user_ids
        }

        return pl.DataFrame(user_id_to_recommend_news_ids_map), pl.DataFrame()

    def predict_score(self, val_behavior_df: pl.DataFrame, val_news_df: pl.DataFrame) -> np.ndarray:
        return np.array([])
