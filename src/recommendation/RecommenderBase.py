import polars as pl


class RecommenderBase:
    def fit(self, train: pl.DataFrame) -> None:
        pass

    def recommend_top_k_items(self, input: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame()

    def predict_score(self, input: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame()
