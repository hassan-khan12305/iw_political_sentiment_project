"""
VADER sentiment scoring on usc_2024_clean.parquet.

Reads:   data/usc_2024_clean.parquet   (text_clean, year_week)
Writes:  data/usc_2024_sentiment_vader.parquet  (weekly aggregates)

Run:
    iw/bin/python src/features/sentiment_vader.py
    # or:
    iw/bin/python src/features/sentiment_vader.py --input data/usc_2024_clean.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CLEAN_PATH = Path("data/usc_2024_clean.parquet")
OUT_PATH   = Path("data/usc_2024_sentiment_vader.parquet")
BATCH_SIZE = 500_000


def score_vader(
    clean_path: Path = CLEAN_PATH,
    out_path: Path = OUT_PATH,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}. Run the pipeline first.")

    analyzer = SentimentIntensityAnalyzer()

    # Accumulate per-week stats without storing all scores in memory
    weekly: dict[str, dict] = {}
    total = 0

    pf = pq.ParquetFile(str(clean_path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text_clean", "year_week"]):
        df = batch.to_pandas()
        df = df.dropna(subset=["text_clean", "year_week"])
        df = df[df["text_clean"].str.len() > 0]
        if df.empty:
            continue

        compounds = df["text_clean"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
        df = df.assign(compound=compounds)

        for week, grp in df.groupby("year_week"):
            c = grp["compound"]
            if week not in weekly:
                weekly[week] = {"sum_c": 0.0, "n": 0, "n_pos": 0, "n_neg": 0, "n_neu": 0}
            s = weekly[week]
            s["sum_c"] += float(c.sum())
            s["n"]     += len(c)
            s["n_pos"] += int((c > 0.05).sum())
            s["n_neg"] += int((c < -0.05).sum())
            s["n_neu"] += int(((c >= -0.05) & (c <= 0.05)).sum())

        total += len(df)
        print(f"  Scored {total:,} tweets...", end="\r", flush=True)

    print(f"\nDone. Scored {total:,} tweets total.")

    rows = []
    for week, s in sorted(weekly.items()):
        n = s["n"]
        rows.append({
            "year_week":     week,
            "avg_sentiment": s["sum_c"] / n if n > 0 else 0.0,
            "pct_positive":  s["n_pos"] / n if n > 0 else 0.0,
            "pct_negative":  s["n_neg"] / n if n > 0 else 0.0,
            "pct_neutral":   s["n_neu"] / n if n > 0 else 0.0,
            "scored_tweets": n,
        })

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VADER sentiment scoring for USC 2024 tweets.")
    parser.add_argument("--input",      default=str(CLEAN_PATH))
    parser.add_argument("--output",     default=str(OUT_PATH))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    score_vader(Path(args.input), Path(args.output), args.batch_size)
