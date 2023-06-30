from .NewsRecommenderBase import NewsRecommenderBase
import polars as pl
import numpy as np
from implicit.als import AlternatingLeastSquares


class ImplicitMFBasedNewsRecommender(NewsRecommenderBase):
    def __init__(self) -> None:
        self.model = AlternatingLeastSquares()

    def fit(self, train: pl.DataFrame) -> None:
        pass

    def recommend_top_k_items(
        self, user_ids: np.ndarray, behavior_df: pl.DataFrame, news_df: pl.DataFrame
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def predict_score(self, input: pl.DataFrame) -> np.ndarray:
        return np.array([])
