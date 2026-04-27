"""
Merge political leaning scores into usc_2024_weekly.parquet.

Run after a leaning script completes:
    iw/bin/python src/features/merge_leaning.py --source hashtags
    iw/bin/python src/features/merge_leaning.py --source stance

Re-running is safe - existing columns for the source are replaced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

WEEKLY_PATH = Path("data/usc_2024_weekly.parquet")
LEANING_PATHS = {
    "hashtags": Path("data/usc_2024_leaning_hashtags.parquet"),
    "stance":   Path("data/usc_2024_leaning_stance.parquet"),
}

# Columns that are metadata counts, not model output — drop before merging
_DROP_COLS = {"labeled_tweets", "scored_trump", "scored_harris", "scored_biden"}


def merge_leaning(source: str = "hashtags") -> pd.DataFrame:
    lean_path = LEANING_PATHS[source]
    if not lean_path.exists():
        raise FileNotFoundError(
            f"Missing leaning file: {lean_path}\n"
            f"Run political_leaning_{source}.py first."
        )
    if not WEEKLY_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {WEEKLY_PATH}\nRun aggregate_weekly.py first."
        )

    df_weekly = pd.read_parquet(WEEKLY_PATH)
    df_lean   = pd.read_parquet(lean_path)

    # Drop metadata count columns
    df_lean = df_lean.drop(columns=list(_DROP_COLS & set(df_lean.columns)), errors="ignore")

    # Suffix with source so hashtags and stance columns can coexist
    rename = {
        c: f"{c}_{source}"
        for c in df_lean.columns
        if c != "year_week"
    }
    df_lean = df_lean.rename(columns=rename)

    # Remove stale columns for this source
    stale = [c for c in df_weekly.columns if c.endswith(f"_{source}")]
    df_weekly = df_weekly.drop(columns=stale)

    df_out = df_weekly.merge(df_lean, on="year_week", how="left")
    df_out.to_parquet(WEEKLY_PATH, index=False)
    print(f"Merged {source} leaning into {WEEKLY_PATH}")

    show_cols = ["year_week"] + [c for c in df_out.columns if source in c]
    print(df_out[show_cols].to_string(index=False))
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge political leaning into weekly parquet.")
    parser.add_argument("--source", choices=["hashtags", "stance"], default="hashtags",
                        help="Which leaning scores to merge in.")
    args = parser.parse_args()
    merge_leaning(args.source)
