from __future__ import annotations

from pathlib import Path

import pandas as pd
from lightgbm import LGBMRanker

from src.data import load_data
from src.features import (
    build_app_features,
    build_observed_training_pairs,
    build_user_features,
    normalise_feature_types,
    prepare_model_table,
)


class LambdaMARTRecommender:
    """Feature-based LambdaMART recommender for implicit app engagement.

    Training uses only observed user-app pairs and learns relative engagement strength.
    Unobserved pairs are not labelled as negatives; at inference the learned ranking
    function is extrapolated to the full app catalog.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        precompute_top_n: int = 50,
        random_state: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.precompute_top_n = precompute_top_n
        self.random_state = random_state

        self.model: LGBMRanker | None = None
        self.user_features: pd.DataFrame | None = None
        self.app_features: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.categorical_cols: list[str] = []
        self.category_levels: dict[str, list[str]] = {}
        self.popular_apps: list[str] = []
        self._recommendation_cache: dict[str, list[str]] = {}

    def fit(self, train_df: pd.DataFrame) -> "LambdaMARTRecommender":
        users, apps, _ = load_data(self.data_dir)
        train = train_df.copy()
        train["event_time"] = pd.to_datetime(train["event_time"], errors="coerce")
        train = train.dropna(subset=["event_time"]).copy()
        if train.empty:
            raise ValueError("Training interactions are empty after timestamp validation.")

        reference_time = train["event_time"].max()

        self.user_features = build_user_features(users, train, reference_time)
        self.app_features = build_app_features(apps, train, reference_time)

        observed_pairs = build_observed_training_pairs(train)
        model_table = prepare_model_table(observed_pairs, self.user_features, self.app_features)
        model_table = model_table.sort_values("user_id").reset_index(drop=True)

        excluded = {"user_id", "app_id", "interaction_level", "interaction_rank"}
        model_table, categorical_cols = normalise_feature_types(model_table, exclude=excluded)

        self.feature_cols = [c for c in model_table.columns if c not in excluded]
        self.categorical_cols = [c for c in categorical_cols if c in self.feature_cols]
        self.category_levels = {
            c: model_table[c].cat.categories.astype(str).tolist() for c in self.categorical_cols
        }

        group_sizes = model_table.groupby("user_id", sort=False).size().tolist()

        self.model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            label_gain=[0, 1, 2, 3, 4, 5, 6, 7],
            verbosity=-1,
        )
        self.model.fit(
            model_table[self.feature_cols],
            model_table["interaction_rank"],
            group=group_sizes,
            categorical_feature=self.categorical_cols,
        )

        self.popular_apps = train["app_id"].value_counts().index.astype(str).tolist()
        self._precompute_recommendations()
        return self

    def _align_for_prediction(self, candidates: pd.DataFrame) -> pd.DataFrame:
        out = candidates.copy()
        for col in self.feature_cols:
            if col not in out.columns:
                out[col] = 0

        for col in self.feature_cols:
            if col in self.categorical_cols:
                values = out[col].astype("string").fillna("__MISSING__")
                out[col] = pd.Categorical(values, categories=self.category_levels[col])
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

        return out[self.feature_cols]

    def _candidate_frame_for_users(self, users_chunk: pd.DataFrame) -> pd.DataFrame:
        if self.app_features is None:
            raise RuntimeError("Model is not fitted.")
        return users_chunk.merge(self.app_features, how="cross")

    def _precompute_recommendations(self, chunk_size: int = 200) -> None:
        if self.model is None or self.user_features is None or self.app_features is None:
            raise RuntimeError("Model is not fitted.")

        self._recommendation_cache = {}
        top_n = max(self.precompute_top_n, 10)

        for start in range(0, len(self.user_features), chunk_size):
            users_chunk = self.user_features.iloc[start : start + chunk_size].copy()
            candidates = self._candidate_frame_for_users(users_chunk)
            X = self._align_for_prediction(candidates)
            candidates["score"] = self.model.predict(X)

            top = (
                candidates.sort_values(["user_id", "score"], ascending=[True, False])
                .groupby("user_id", sort=False)
                .head(top_n)
            )
            for user_id, group in top.groupby("user_id", sort=False):
                self._recommendation_cache[str(user_id)] = group["app_id"].astype(str).tolist()

    def recommend(self, user_id: str, n: int) -> list[str]:
        """Return n app_ids ordered by predicted relevance.

        Unknown users fall back to global popularity from the training interactions.
        """
        if self.model is None:
            raise RuntimeError("Call fit(train_df) before recommend().")
        if n <= 0:
            return []

        user_key = str(user_id)
        cached = self._recommendation_cache.get(user_key)
        if cached is not None:
            if len(cached) >= n:
                return cached[:n]

        # Cold/unknown-user fallback.
        return self.popular_apps[:n]
