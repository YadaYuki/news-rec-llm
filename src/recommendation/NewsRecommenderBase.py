import polars as pl
import numpy as np
from abc import ABC, abstractmethod


class RecommenderBase(ABC):
    @abstractmethod
    def fit(self, train: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def recommend_top_k_items(
        self, user_ids: np.ndarray, behavior_df: pl.DataFrame, news_df: pl.DataFrame
    ) -> pl.DataFrame:
        pass

    @abstractmethod
    def predict_score(self, input: pl.DataFrame) -> pl.DataFrame:
        pass
