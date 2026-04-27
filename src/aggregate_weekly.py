"""
Aggregate cleaned tweets to weekly counts and engagement metrics.

Reads:   data/usc_2024_clean.parquet
Writes:  data/usc_2024_weekly.parquet  (one row per ISO week: tweet_count + avg engagement)

Run:
    iw/bin/python src/aggregate_weekly.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def aggregate_weekly(
    input_path: str = "data/usc_2024_clean.parquet",
    output_path: str = "data/usc_2024_weekly.parquet",
    batch_size: int = 200_000,
) -> None:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    parquet_file = pq.ParquetFile(str(in_path))

    counts       = defaultdict(int)
    like_sum     = defaultdict(float)
    retweet_sum  = defaultdict(float)
    reply_sum    = defaultdict(float)
    quote_sum    = defaultdict(float)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        if df.empty:
            continue

        grouped = df.groupby("year_week", as_index=False).agg(
            tweet_count=("id", "count"),
            like_sum=("like_count", "sum"),
            retweet_sum=("retweet_count", "sum"),
            reply_sum=("reply_count", "sum"),
            quote_sum=("quote_count", "sum"),
        )

        for row in grouped.itertuples(index=False):
            yw = row.year_week
            counts[yw] += int(row.tweet_count)
            like_sum[yw] += float(row.like_sum)
            retweet_sum[yw] += float(row.retweet_sum)
            reply_sum[yw] += float(row.reply_sum)
            quote_sum[yw] += float(row.quote_sum)

    rows = []
    for yw in sorted(counts.keys()):
        n = counts[yw]
        rows.append(
            {
                "year_week": yw,
                "tweet_count": n,
                "avg_like_count": like_sum[yw] / n if n else 0.0,
                "avg_retweet_count": retweet_sum[yw] / n if n else 0.0,
                "avg_reply_count": reply_sum[yw] / n if n else 0.0,
                "avg_quote_count": quote_sum[yw] / n if n else 0.0,
            }
        )

    out_df = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    print(f"Done. Wrote {len(out_df):,} weekly rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate cleaned USC tweets to weekly metrics.")
    parser.add_argument("--input", default="data/usc_2024_clean.parquet")
    parser.add_argument("--output", default="data/usc_2024_weekly.parquet")
    parser.add_argument("--batch-size", type=int, default=200_000)
    args = parser.parse_args()

    aggregate_weekly(input_path=args.input, output_path=args.output, batch_size=args.batch_size)
