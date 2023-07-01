from utils.dataset.MINDDataFrame import MINDDataFrame
from utils.dataset.dataset import create_news_and_user_ids_to_clicked_map
from const.path import MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR
from recommendation.RandomNewsRecommender import RandomNewsRecommender
from utils.list import uniq
import numpy as np
from utils.metrics import metrics
from utils.logger import logging


if __name__ == "__main__":
    # Load Train Data.
    logging.info("Loading & Processing Train Data ... ")
    train_news_df, train_behavior_df = MINDDataFrame().read_df(
        MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv", MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv"
    )

    # Train Model
    logging.info("Training Recommender ... ")
    recommender = RandomNewsRecommender()
    recommender.fit(train_behavior_df, train_news_df)

    # Load Validation Data.
    logging.info("Loading & Processing Validation Data ... ")
    val_news_df, val_behavior_df = MINDDataFrame().read_df(
        MIND_SMALL_VAL_DATASET_DIR / "news.tsv", MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv"
    )

    # Recommend
    K = 10
    logging.info(f"Recommend top {K} items.")
    user_ids = np.array(uniq(val_behavior_df["user_id"].to_list()))
    recommended_news_ids, recommended_scores = recommender.recommend_top_k_items(
        user_ids, val_behavior_df, val_news_df, K
    )

    # Evaluate
    logging.info("Evaluation..")
    news_and_user_ids_to_clicked_map = create_news_and_user_ids_to_clicked_map(val_behavior_df)
    ndcg_at_10_list = []
    recommended_news_ids_map = recommended_news_ids.to_dict(as_series=False)  # {[user_id]: [news_id,news_id ... ]}
    for user_id, recommend_news_ids in recommended_news_ids_map.items():
        recommended_top_Kth_items_scores: list[float] = []
        for recommend_news_id in recommend_news_ids:
            recommended_top_Kth_items_scores.append(news_and_user_ids_to_clicked_map[(user_id, recommend_news_id)])
        ndcg_at_10_list.append(metrics.nDCG_K(recommended_top_Kth_items_scores, K))

    logging.info(list(recommended_news_ids_map.items())[:100])
    logging.info(ndcg_at_10_list[:100])
    logging.info(sum(ndcg_at_10_list) / len(ndcg_at_10_list))
    logging.info("Completed")
