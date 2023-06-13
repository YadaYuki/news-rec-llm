from const.path import DATASET_DIR
import polars as pl

df = pl.read_csv(DATASET_DIR / "mind" / "large" / "train" / "behaviors.tsv", separator="\t", encoding="utf8-lossy")
