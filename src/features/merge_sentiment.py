"""
Merge sentiment scores into usc_2024_weekly.parquet.

Run after a sentiment scoring script completes:
    iw/bin/python src/features/merge_sentiment.py --source vader
    iw/bin/python src/features/merge_sentiment.py --source tweetnlp

Re-running is safe — existing columns for the source are replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WEEKLY_PATH = Path("data/usc_2024_weekly.parquet")
SENTIMENT_PATHS = {
    "vader":    Path("data/usc_2024_sentiment_vader.parquet"),
    "tweetnlp": Path("data/usc_2024_sentiment_tweetnlp.parquet"),
}


def merge_sentiment(source: str = "vader") -> pd.DataFrame:
    sent_path = SENTIMENT_PATHS[source]
    if not sent_path.exists():
        raise FileNotFoundError(
            f"Missing sentiment file: {sent_path}\n"
            f"Run sentiment_{source}.py first."
        )
    if not WEEKLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {WEEKLY_PATH}\nRun aggregate_weekly.py first."
        )

    df_weekly = pd.read_parquet(WEEKLY_PATH)
    df_sent   = pd.read_parquet(sent_path)

    # Suffix columns with source name so vader and tweetnlp can coexist
    rename = {
        c: f"{c}_{source}"
        for c in ["avg_sentiment", "pct_positive", "pct_negative", "pct_neutral"]
        if c in df_sent.columns
    }
    df_sent = df_sent.rename(columns=rename).drop(columns=["scored_tweets"], errors="ignore")

    # Drop stale columns for this source before re-merge
    stale = [c for c in df_weekly.columns if c.endswith(f"_{source}")]
    df_weekly = df_weekly.drop(columns=stale)

    df_out = df_weekly.merge(df_sent, on="year_week", how="left")
    df_out.to_parquet(WEEKLY_PATH, index=False)
    print(f"Merged {source} sentiment into {WEEKLY_PATH}")

    sentiment_cols = ["year_week"] + [c for c in df_out.columns if source in c]
    print(df_out[sentiment_cols].to_string(index=False))
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sentiment scores into weekly parquet.")
    parser.add_argument("--source", choices=["vader", "tweetnlp"], default="vader",
                        help="Which sentiment scores to merge in.")
    args = parser.parse_args()
    merge_sentiment(args.source)
