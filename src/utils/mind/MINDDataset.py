from torch.utils.data import Dataset
import torch
from typing import Tuple, Callable
import polars as pl
import numpy as np
import random

EMPTY_NEWS_ID, EMPTY_IMPRESSION_IDX = "EMPTY_NEWS_ID", -1


class MINDTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        batch_transform_texts: Callable[[list[str]], torch.Tensor],
        npratio: int,
        history_size: int,
        random_seed: int,
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.npratio: int = npratio
        self.history_size: int = history_size
        self.random_seed = random_seed

        self.behavior_df = self.behavior_df.with_columns(
            [
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1])
                .alias("clicked_idxes"),
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0])
                .alias("non_clicked_idxes"),
            ]
        )

        self.news_id_to_title_map: dict[str, str] = {EMPTY_NEWS_ID: ""}

    def __getitem__(
        self, behavior_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # TODO: 一行あたりにpositiveが複数存在することも考慮した
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: labels
        """
        behavior_item = self.behavior_df[behavior_idx]
        history: list[str] = behavior_item["history"].item()

        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].item() + [EMPTY_IMPRESSION]
        )  # NOTE: EMPTY_IMPRESSION_IDX = -1なので最後尾に追加する。

        poss_idxes, neg_idxes = behavior_item["clicked_idxes"].item(), behavior_item["non_clicked_idxes"].item()

        # Sampling Positive(clicked) & Negative(non-clicked) Sample
        sample_poss_idxes, sample_neg_idxes = random.sample(poss_idxes, 1), self.__sampling_negative(
            neg_idxes, self.npratio
        )
        sample_impression_idxes = random.shuffle(sample_poss_idxes + sample_neg_idxes)
        sample_impressions = impressions[sample_impression_idxes]

        # Extract candidate_news & history_news based on sample idxes
        candidate_news_ids = [imp_item["news_id"] for imp_item in sample_impressions]
        labels = [imp_item["clicked"] for imp_item in sample_impressions]
        history_news_ids = history[: self.history_size]  # TODO: diverse

        # News ID to News Title
        candidate_news_titles, history_news_titles = [
            self.news_id_to_title_map[news_id] for news_id in candidate_news_ids
        ], [self.news_id_to_title_map[news_id] for news_id in history_news_ids]

        # Conver to Tensor
        candidate_news_tensor, history_news_tensor = self.batch_transform_texts(
            candidate_news_titles
        ), self.batch_transform_texts(history_news_titles)
        labels_tensor = torch.Tensor(labels)

        return history_news_tensor, candidate_news_tensor, labels_tensor

    def __sampling_negative(self, neg_idxes: list[int], npratio: int) -> list[int]:
        if len(neg_idxes) < npratio:
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))

        return random.sample(neg_idxes, self.npratio)


if __name__ == "__main__":
    pass
