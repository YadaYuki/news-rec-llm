from torch.utils.data import Dataset
import torch
from typing import Tuple, Callable
import polars as pl


class MINDTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        transform_text: Callable[[str], torch.Tensor],
        npratio: int,
        history_size: int,
        random_seed: int,
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.transform_text: Callable[[str], torch.Tensor] = transform_text
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

        self.news_id_to_title_map: dict[str, str] = {"EMPTY": ""}

    def __iter__(
        self, behavior_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # TODO: 一行あたりにpositiveが複数存在することも考慮した
        behavior_item = self.behavior_df[behavior_idx]
        history: list[str] = behavior_item["history"].item()
        

        # Sampling Positive(clicked) & Negative(non-clicked) Sample
        


        for history_news_id in history:
            pass

        return torch.Tensor(), torch.Tensor(), torch.Tensor()

    def __sampling_negative(self) -> None:
        return


if __name__ == "__main__":
    pass
