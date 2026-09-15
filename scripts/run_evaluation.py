from __future__ import annotations

import argparse
import math

import pandas as pd

from src.data import fixed_time_split, load_data
from src.recommender import LambdaMARTRecommender


def recall_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    if not actual:
        return 0.0
    return sum(item in actual for item in predicted[:k]) / len(actual)


def ndcg_at_k(predicted: list[str], actual: set[str], k: int) -> float:
    if not actual:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(predicted[:k])
        if item in actual
    )
    ideal_hits = min(len(actual), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_recommender(model, test_df: pd.DataFrame, k: int = 10) -> dict[str, float]:
    actual_by_user = test_df.groupby("user_id")["app_id"].apply(set).to_dict()
    recalls, ndcgs = [], []
    empty = 0
    for user_id, actual in actual_by_user.items():
        predicted = model.recommend(str(user_id), k)
        if not predicted:
            empty += 1
        recalls.append(recall_at_k(predicted, actual, k))
        ndcgs.append(ndcg_at_k(predicted, actual, k))
    return {
        "recall_at_10": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "ndcg_at_10": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        "users": len(recalls),
        "empty_predictions": empty,
    }


class PopularityBaseline:
    def fit(self, train_df: pd.DataFrame) -> "PopularityBaseline":
        self.top_apps = train_df["app_id"].value_counts().index.astype(str).tolist()
        return self

    def recommend(self, user_id: str, n: int) -> list[str]:
        return self.top_apps[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cutoff", default="2026-04-23")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    _, _, interactions = load_data(args.data_dir)
    train_df, test_df = fixed_time_split(interactions, args.cutoff)

    print(f"Train interactions: {len(train_df):,}")
    print(f"Test interactions:  {len(test_df):,}")
    print(f"Test users:         {test_df['user_id'].nunique():,}")

    model = LambdaMARTRecommender(data_dir=args.data_dir, precompute_top_n=max(args.k, 50))
    model.fit(train_df)
    ranker_results = evaluate_recommender(model, test_df, args.k)

    popularity = PopularityBaseline().fit(train_df)
    popularity_results = evaluate_recommender(popularity, test_df, args.k)

    comparison = pd.DataFrame(
        [
            {"model": "LambdaMART", **ranker_results},
            {"model": "Popularity", **popularity_results},
        ]
    )
    print("\nFixed-cutoff evaluation")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
