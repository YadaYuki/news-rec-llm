from src.const.path import DATASET_DIR
from src.utils.dataset.MINDDataFrame import MINDDataFrame


class TestMINDDataFrame:
    def test_read_news_df(self) -> None:
        small_val_news_tsv_path = DATASET_DIR / "mind" / "small" / "val" / "news.tsv"
        news_df = MINDDataFrame().read_news_df(small_val_news_tsv_path)
        assert news_df.columns == ["news_id", "category", "subcategory", "title", "abstract", "url"]
        assert len(news_df) == 42416

    def test_read_news_df_with_entities(self) -> None:
        small_val_news_tsv_path = DATASET_DIR / "mind" / "small" / "val" / "news.tsv"
        news_df = MINDDataFrame().read_news_df(small_val_news_tsv_path, has_entities=True)
        assert news_df.columns == [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ]
        assert len(news_df) == 42416
