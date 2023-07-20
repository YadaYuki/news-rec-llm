from .NewsRecommenderBase import NewsRecommenderBase
import polars as pl
import numpy as np
from implicit.als import AlternatingLeastSquares
from typing import Tuple
import scipy
from utils.list import uniq
from utils.logger import logging
from pydantic import BaseModel


class MatrixIDMapper(BaseModel):
    user_id_to_idx_map: dict[str, int]
    news_id_to_idx_map: dict[str, int]
    idx_to_user_id: list[str]
    idx_to_news_id: list[str]


class ImplicitMFBasedNewsRecommender(NewsRecommenderBase):
    def __init__(self) -> None:
        #  TODO: Hyperparameter Tuning & Device.
        self.model = AlternatingLeastSquares(factors=64)
        self.feedback_matrix: scipy.sparse.spmatrix = None
        self.id_mapper: MatrixIDMapper | None = None

    def fit(self, behavior_df: pl.DataFrame, news_df: pl.DataFrame) -> None:
        self.feedback_matrix, self.id_mapper = self._create_feedback_matrix_from_behavior_log(behavior_df)
        self.model.fit(self.feedback_matrix)

    def recommend_top_k_items(
        self, user_ids: np.ndarray, val_behavior_df: pl.DataFrame, val_news_df: pl.DataFrame, K: int = 10
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        assert self.feedback_matrix is not None
        assert self.id_mapper is not None

        filtered_user_ids: list[str] = list(
            filter(lambda uid: (uid in self.id_mapper.user_id_to_idx_map), user_ids.tolist())
        )

        user_idxes: np.ndarray = np.array(
            [self.id_mapper.user_id_to_idx_map[user_id] for user_id in filtered_user_ids]
        )  # 未知ユーザに対する推薦は諦める。
        ids, scores = self.model.recommend(
            user_idxes, self.feedback_matrix[user_idxes], N=K, filter_already_liked_items=False
        )

        user_id_to_recommend_news_ids_map: dict[str, list[str]] = {
            self.id_mapper.idx_to_user_id[user_idx]: [self.id_mapper.idx_to_news_id[news_idx] for news_idx in ids[i]]
            for i, user_idx in enumerate(user_idxes)
        }
        user_id_to_recommend_scores_map: dict[int, list[float]] = {
            self.id_mapper.idx_to_user_id[user_idx]: scores[i] for i, user_idx in enumerate(user_idxes)
        }

        return pl.DataFrame(user_id_to_recommend_news_ids_map), pl.DataFrame(user_id_to_recommend_scores_map)

    def predict_score(self, val_behavior_df: pl.DataFrame, val_news_df: pl.DataFrame) -> np.ndarray:
        return np.array([])

    def _create_feedback_matrix_from_behavior_log(
        self,
        behavior_df: pl.DataFrame,
    ) -> Tuple[scipy.sparse.spmatrix, MatrixIDMapper]:
        unique_user_id, unique_news_id = uniq(behavior_df["user_id"].to_list()), uniq(behavior_df["news_id"].to_list())
        user_id_to_idx_map: dict[str, int] = {uid: i for i, uid in enumerate(unique_user_id)}
        news_id_to_idx_map: dict[str, int] = {uid: i for i, uid in enumerate(unique_news_id)}

        clicked_behavior_df = (
            behavior_df.select(["user_id", "news_id", "clicked"])
            .groupby(["user_id", "news_id"])
            .agg(total_click=pl.col("clicked").sum())
            .filter(pl.col("total_click") > 0)
        )
        clicked_users = np.array([user_id_to_idx_map[uid] for uid in clicked_behavior_df["user_id"].to_list()])
        clicked_news = np.array([news_id_to_idx_map[nid] for nid in clicked_behavior_df["news_id"].to_list()])
        total_click = clicked_behavior_df["total_click"].to_numpy()

        matrix_shape = (len(unique_user_id), len(unique_news_id))

        feedback_matrix = scipy.sparse.csr_matrix((total_click, (clicked_users, clicked_news)), shape=matrix_shape)

        logging.info(f"matrix_shape:{matrix_shape}")
        logging.info(f"clicked_behavior_df size:{len(clicked_behavior_df)},clicked_behavior_df:{clicked_behavior_df}")

        return feedback_matrix, MatrixIDMapper(
            **{
                "user_id_to_idx_map": user_id_to_idx_map,
                "news_id_to_idx_map": news_id_to_idx_map,
                "idx_to_user_id": unique_user_id,
                "idx_to_news_id": unique_news_id,
            }
        )

    def _explode_behavior_df(self, behavior_df: pl.DataFrame):
        return (
            behavior_df.explode(pl.col("impressions"))
            .with_columns(
                [
                    pl.col("impressions").struct.field("news_id").alias("news_id"),
                    pl.col("impressions").struct.field("clicked").alias("clicked"),
                ]
            )
            .select(["impression_id", "user_id", "time", "history", "news_id", "clicked"])
        )
