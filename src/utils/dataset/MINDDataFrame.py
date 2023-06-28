import polars as pl
import pandas as pd
from pathlib import Path


class MINDDataFrame:
    def read_news_df(self, path_to_news_tsv: Path, has_entities: bool = False) -> pl.DataFrame:
        # FIXME:
        # pl.read_csvを直接実行すると行が欠損するのでpandasから読み取る形式をとる。
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
