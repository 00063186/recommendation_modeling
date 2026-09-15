from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_USER_COLUMNS = {"user_id"}
REQUIRED_APP_COLUMNS = {"app_id"}
REQUIRED_INTERACTION_COLUMNS = {
    "user_id",
    "app_id",
    "event_time",
    "event_name",
}


def _validate_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and minimally validate the three assignment tables."""
    data_dir = Path(data_dir)

    users = pd.read_csv(data_dir / "users.csv")
    apps = pd.read_csv(data_dir / "apps.csv")
    interactions = pd.read_csv(data_dir / "interactions.csv")

    _validate_columns(users, REQUIRED_USER_COLUMNS, "users.csv")
    _validate_columns(apps, REQUIRED_APP_COLUMNS, "apps.csv")
    _validate_columns(interactions, REQUIRED_INTERACTION_COLUMNS, "interactions.csv")

    interactions = interactions.copy()
    interactions["event_time"] = pd.to_datetime(interactions["event_time"], errors="coerce")
    invalid_times = interactions["event_time"].isna().sum()
    if invalid_times:
        # Invalid timestamps cannot be placed reliably on the timeline.
        interactions = interactions.dropna(subset=["event_time"]).copy()

    if "signup_date" in users.columns:
        users = users.copy()
        users["signup_date"] = pd.to_datetime(users["signup_date"], errors="coerce")

    if "app_date_inserted" in apps.columns:
        apps = apps.copy()
        apps["app_date_inserted"] = pd.to_datetime(apps["app_date_inserted"], errors="coerce")

    return users, apps, interactions


def fixed_time_split(
    interactions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split all users at one global point in time."""
    cutoff = pd.Timestamp(cutoff)
    train = interactions.loc[interactions["event_time"] < cutoff].copy()
    test = interactions.loc[interactions["event_time"] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(
            f"Cutoff {cutoff} produced an empty split: train={len(train)}, test={len(test)}"
        )

    return train, test
