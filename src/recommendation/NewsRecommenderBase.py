import polars as pl
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class NewsRecommenderBase(ABC):
    @abstractmethod
    def fit(self, train_behavior_df: pl.DataFrame, train_news_df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def recommend_top_k_items(
        self, user_ids: np.ndarray, val_behavior_df: pl.DataFrame, val_news_df: pl.DataFrame, K: int
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        pass

    @abstractmethod
    def predict_score(self, val_behavior_df: pl.DataFrame, val_news_df: pl.DataFrame) -> np.ndarray:
        pass
