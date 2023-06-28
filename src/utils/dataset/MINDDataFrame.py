import polars as pl
import pandas as pd
from pathlib import Path
from typing import Tuple


class MINDDataFrame:
    @classmethod
    def read_df(cls, path_to_news_tsv: Path, path_to_behavior_tsv: Path) -> Tuple[pl.DataFrame, pl.DataFrame]:
        return cls.read_news_df(path_to_news_tsv), cls.read_behavior_df(path_to_behavior_tsv)

    @classmethod
    def read_news_df(cls, path_to_news_tsv: Path, has_entities: bool = False) -> pl.DataFrame:
        # FIXME:
        # pl.read_csvを直接実行すると、行が欠損するため、pandasでtsvを読み取り、polarsのDataFrameに変換する方式にする。
        news_df = pd.read_csv(path_to_news_tsv, sep="\t", encoding="utf8", header=None)
        news_df.columns = [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]
        news_df = pl.from_dataframe(news_df)

        if has_entities:
            return news_df
        return news_df.drop("title_entities", "abstract_entities")

    @classmethod
    def read_behavior_df(cls, path_to_behavior_tsv: Path) -> pl.DataFrame:
        behavior_df = pl.read_csv(path_to_behavior_tsv, separator="\t", encoding="utf8-lossy", has_header=False)
        behavior_df = behavior_df.rename(
            {
                "column_1": "impression_id",
                "column_2": "user_id",
                "column_3": "time",
                "column_4": "history",
                "column_5": "impressions",
            }
        )
        behavior_df = (
            behavior_df.with_columns((pl.col("impressions").str.split(" ")).alias("impression_news_list"))
            .explode(pl.col("impression_news_list"))
            .with_columns(
                [
                    (pl.col("impression_news_list").str.split("-").list.get(0)).alias("news_id"),
                    (pl.col("impression_news_list").str.split("-").list.get(1).str.parse_int()).alias("clicked"),
                ]
            )
            .select(
                [
                    "impression_id",
                    "user_id",
                    "news_id",
                    "time",
                    "history",
                    "clicked",
                ]
            )
        )

        return behavior_df
