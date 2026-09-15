from __future__ import annotations

import argparse

from scripts.evaluate import evaluate, load_test_split
from src.recommender import LambdaMARTRecommender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    interactions_path = f"{args.data_dir}/interactions.csv"
    train_df, test_df = load_test_split(interactions_path)

    model = LambdaMARTRecommender(data_dir=args.data_dir, precompute_top_n=max(args.k, 50))
    model.fit(train_df)
    evaluate(model, test_df, k=args.k)


if __name__ == "__main__":
    main()
