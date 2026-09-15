from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


EVENT_TYPE_MAPPING = {
    "af_click": "click",
    "af_install": "install",
    "af_complete_registration": "register",
    "af_level_achieved": "level",
    "af_open": "open",
    "af_purchase": "purchase",
    "af_view": "view",
}

EVENT_TYPES = ["click", "install", "register", "level", "open", "purchase", "view"]

# Heuristic engagement weights used in the exploratory notebook.
# They are not probabilities; they only define an ordinal engagement signal.
EVENT_WEIGHTS = {
    "open": 0.10,
    "view": 0.15,
    "click": 0.20,
    "install": 0.20,
    "register": 0.30,
    "level": 0.40,
    "purchase": 0.60,
}

USER_IDENTIFIER_COLUMNS = [
    "customer_user_id",
    "appsflyer_id",
    "advertising_id",
    "idfa",
    "idfv",
    "android_id",
]

APP_DISPLAY_COLUMNS = ["bundle_id", "app_name"]

USER_CATEGORICAL_BASE = [
    "platform",
    "device_model",
    "os_version",
    "country_code",
    "language",
    "carrier",
    "age_bucket",
]

APP_CATEGORICAL_BASE = [
    "platform",
    "status",
    "category",
    "subcategory_1",
    "subcategory_2",
    "subcategory_3",
    "subcategory_4",
    "age_rating",
    "advertiser_id",
]


def add_interaction_type(interactions: pd.DataFrame) -> pd.DataFrame:
    df = interactions.copy()
    df["interaction_type"] = df["event_name"].map(EVENT_TYPE_MAPPING)
    return df


def _event_counts(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    counts = (
        df.dropna(subset=["interaction_type"])
        .groupby([key, "interaction_type"])
        .size()
        .unstack(fill_value=0)
    )
    for event_type in EVENT_TYPES:
        if event_type not in counts.columns:
            counts[event_type] = 0
    counts = counts[EVENT_TYPES]
    counts.columns = [f"{prefix}event_count_{c}" for c in counts.columns]
    return counts.reset_index()


def _duration_features(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    # Keep source values as provided. Negative durations are documented as a data-quality issue
    # rather than silently corrected because their source semantics are not specified.
    if "duration_seconds" not in df.columns:
        return pd.DataFrame({key: df[key].drop_duplicates()})

    duration_df = df.dropna(subset=["duration_seconds"]).copy()
    if duration_df.empty:
        return pd.DataFrame({key: df[key].drop_duplicates()})

    overall = duration_df.groupby(key)["duration_seconds"].agg(["mean", "median"])
    overall.columns = [f"{prefix}overall_duration_avg", f"{prefix}overall_duration_median"]

    avg = (
        duration_df.dropna(subset=["interaction_type"])
        .groupby([key, "interaction_type"])["duration_seconds"]
        .mean()
        .unstack()
    )
    median = (
        duration_df.dropna(subset=["interaction_type"])
        .groupby([key, "interaction_type"])["duration_seconds"]
        .median()
        .unstack()
    )

    for event_type in EVENT_TYPES:
        if event_type not in avg.columns:
            avg[event_type] = np.nan
        if event_type not in median.columns:
            median[event_type] = np.nan

    avg = avg[EVENT_TYPES]
    median = median[EVENT_TYPES]
    avg.columns = [f"{prefix}duration_avg_{c}" for c in avg.columns]
    median.columns = [f"{prefix}duration_median_{c}" for c in median.columns]

    return (
        overall.reset_index()
        .merge(avg.reset_index(), on=key, how="outer")
        .merge(median.reset_index(), on=key, how="outer")
    )


def _impression_counts(df: pd.DataFrame, key: str, name: str) -> pd.DataFrame:
    if "attributed_touch_type" not in df.columns:
        return pd.DataFrame({key: df[key].drop_duplicates(), name: 0})
    out = (
        df.loc[df["attributed_touch_type"].eq("impression")]
        .groupby(key)
        .size()
        .rename(name)
        .reset_index()
    )
    return out


def _simple_event_total(df: pd.DataFrame, key: str, event_type: str, name: str) -> pd.DataFrame:
    out = (
        df.loc[df["interaction_type"].eq(event_type)]
        .groupby(key)
        .size()
        .rename(name)
        .reset_index()
    )
    return out


def build_user_features(
    users: pd.DataFrame,
    train_interactions: pd.DataFrame,
    reference_time: pd.Timestamp,
) -> pd.DataFrame:
    df = users.copy()
    inter = add_interaction_type(train_interactions)

    drop_cols = [c for c in USER_IDENTIFIER_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    if "platform" in df.columns:
        df = df.rename(columns={"platform": "user_platform"})

    if "signup_date" in df.columns:
        signup = pd.to_datetime(df["signup_date"], errors="coerce")
        df["days_since_signup"] = (reference_time.normalize() - signup).dt.days
        df = df.drop(columns=["signup_date"])

    count_features = _event_counts(inter, "user_id", "user_")
    duration_features = _duration_features(inter, "user_id", "user_")
    impressions = _impression_counts(inter, "user_id", "user_impression_count")

    df = df.merge(count_features, on="user_id", how="left")
    df = df.merge(duration_features, on="user_id", how="left")
    df = df.merge(impressions, on="user_id", how="left")

    for event_type, col_name in [
        ("click", "user_total_clicks"),
        ("install", "user_total_installs"),
        ("purchase", "user_total_purchases"),
    ]:
        df = df.merge(
            _simple_event_total(inter, "user_id", event_type, col_name),
            on="user_id",
            how="left",
        )

    return df


def build_app_features(
    apps: pd.DataFrame,
    train_interactions: pd.DataFrame,
    reference_time: pd.Timestamp,
) -> pd.DataFrame:
    df = apps.copy().drop(columns=[c for c in APP_DISPLAY_COLUMNS if c in apps.columns], errors="ignore")
    inter = add_interaction_type(train_interactions)

    if "platform" in df.columns:
        df = df.rename(columns={"platform": "app_platform"})

    if "app_date_inserted" in df.columns:
        inserted = pd.to_datetime(df["app_date_inserted"], errors="coerce")
        df["days_since_app_inserted"] = (reference_time.normalize() - inserted).dt.days
        df = df.drop(columns=["app_date_inserted"])

    count_features = _event_counts(inter, "app_id", "app_")
    duration_features = _duration_features(inter, "app_id", "")
    impressions = _impression_counts(inter, "app_id", "impression_count")

    df = df.merge(count_features, on="app_id", how="left")
    df = df.merge(duration_features, on="app_id", how="left")
    df = df.merge(impressions, on="app_id", how="left")

    for event_type, col_name in [
        ("click", "total_clicks"),
        ("install", "total_installs"),
        ("purchase", "total_purchases"),
    ]:
        df = df.merge(
            _simple_event_total(inter, "app_id", event_type, col_name),
            on="app_id",
            how="left",
        )

    # Explicit base-rate features, computed from train only.
    app_interaction_count = inter.groupby("app_id").size().rename("app_interaction_count").reset_index()
    app_unique_users = inter.groupby("app_id")["user_id"].nunique().rename("app_unique_users").reset_index()
    df = df.merge(app_interaction_count, on="app_id", how="left")
    df = df.merge(app_unique_users, on="app_id", how="left")

    return df


def to_relevance(x: float) -> int:
    x = round(float(x), 2)
    if x <= 0:
        return 0
    if x <= 0.10:
        return 1
    if x <= 0.15:
        return 2
    if x <= 0.20:
        return 3
    if x <= 0.30:
        return 4
    if x <= 0.50:
        return 5
    if x <= 0.80:
        return 6
    return 7


def build_observed_training_pairs(train_interactions: pd.DataFrame) -> pd.DataFrame:
    inter = add_interaction_type(train_interactions)
    counts = (
        inter.dropna(subset=["interaction_type"])
        .groupby(["user_id", "app_id", "interaction_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for event_type in EVENT_TYPES:
        if event_type not in counts.columns:
            counts[event_type] = 0

    counts["interaction_level"] = 0.0
    for event_type, weight in EVENT_WEIGHTS.items():
        counts["interaction_level"] += counts[event_type] * weight

    counts["interaction_rank"] = counts["interaction_level"].map(to_relevance)
    return counts[["user_id", "app_id", "interaction_level", "interaction_rank"]]


def prepare_model_table(
    pairs: pd.DataFrame,
    user_features: pd.DataFrame,
    app_features: pd.DataFrame,
) -> pd.DataFrame:
    return (
        pairs.merge(user_features, on="user_id", how="left")
        .merge(app_features, on="app_id", how="left")
    )


def normalise_feature_types(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Make LightGBM input stable: strings become categorical; numeric missing values become 0."""
    exclude = exclude or set()
    out = df.copy()
    categorical_cols: list[str] = []

    force_categorical = {
        "user_platform",
        "device_model",
        "os_version",
        "country_code",
        "language",
        "carrier",
        "age_bucket",
        "app_platform",
        "status",
        "category",
        "subcategory_1",
        "subcategory_2",
        "subcategory_3",
        "subcategory_4",
        "age_rating",
        "advertiser_id",
    }

    for col in out.columns:
        if col in exclude:
            continue
        if col in force_categorical or out[col].dtype == "object" or str(out[col].dtype).startswith("category"):
            out[col] = out[col].astype("string").fillna("__MISSING__").astype("category")
            categorical_cols.append(col)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out, categorical_cols
