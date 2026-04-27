"""
Merge weekly LDA topic signals into the main weekly measurement matrix.

Reads:
  data/usc_2024_weekly.parquet          -- existing weekly matrix
  data/usc_2024_weekly_topics.parquet   -- LDA topic proportions per week

Writes:
  data/usc_2024_weekly.parquet          -- updated in-place

Run:
    iw/bin/python src/features/merge_topics.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

WEEKLY_PATH = Path("data/usc_2024_weekly.parquet")
TOPICS_PATH = Path("data/usc_2024_weekly_topics.parquet")


def main() -> None:
    if not TOPICS_PATH.exists():
        raise FileNotFoundError(
            f"{TOPICS_PATH} not found. Run src/features/topic_lda.py first."
        )

    df_weekly = pd.read_parquet(WEEKLY_PATH)
    df_topics = pd.read_parquet(TOPICS_PATH)

    topic_cols = [c for c in df_topics.columns if c.startswith("topic_")]
    print(f"Weekly matrix:   {df_weekly.shape}")
    print(f"Topic signals:   {df_topics.shape}  ({len(topic_cols)} topic columns)")

    # Drop any existing topic columns to allow idempotent re-runs
    existing = [c for c in df_weekly.columns if c.startswith("topic_")]
    if existing:
        print(f"  Dropping {len(existing)} existing topic columns for refresh")
        df_weekly = df_weekly.drop(columns=existing)

    merged = df_weekly.merge(df_topics, on="year_week", how="left")

    n_missing = merged[topic_cols[0]].isna().sum() if topic_cols else 0
    if n_missing:
        print(f"  WARNING: {n_missing} weeks have no topic data (will be NaN)")

    merged.to_parquet(WEEKLY_PATH, index=False)
    print(f"\nSaved -> {WEEKLY_PATH}  shape: {merged.shape}")
    print(f"Columns now: {list(merged.columns)}")


if __name__ == "__main__":
    main()
