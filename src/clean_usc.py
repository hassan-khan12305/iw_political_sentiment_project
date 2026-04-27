"""
Filter, deduplicate, and clean text from the raw USC tweet parquet.

Keeps: English-only, original tweets (no retweets/replies/quotes), W18–W48 only.
Text cleaning: strip RT prefix -> lowercase -> remove URLs -> untag hashtags -> strip emojis.

Reads:   data/usc_2024_full.parquet
Writes:  data/usc_2024_clean.parquet

Run:
    iw/bin/python src/clean_usc.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
RT_RE = re.compile(r"^\s*RT\s+@\w+:\s*", flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    s = str(text).strip()
    s = RT_RE.sub("", s)
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = HASHTAG_RE.sub(r"\1", s)
    s = EMOJI_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def normalize_type(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    t = str(value).strip().lower()
    if t in {"original", "tweet", "tweet-", "tweet_"}:
        return "original"
    if "retweet" in t:
        return "retweet"
    if "reply" in t:
        return "reply"
    if "quote" in t:
        return "quote"
    return t


# START_DATE is set to 2024-04-29 (start of W18)
START_DATE = pd.Timestamp("2024-04-29", tz="UTC")
END_DATE = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")


def clean_usc(input_path: str = "data/usc_2024_full.parquet", output_path: str = "data/usc_2024_clean.parquet", batch_size: int = 100_000) -> None:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(str(in_path))
    writer: pq.ParquetWriter | None = None
    seen_ids: set[str] = set()
    rows_out = 0

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            if "lang" in df.columns:
                df = df[df["lang"].astype("string").str.lower() == "en"]

            if "type" in df.columns:
                df["type_norm"] = df["type"].apply(normalize_type)
                df = df[df["type_norm"] == "original"]

            df = df.dropna(subset=["text"])
            df["id"] = df["id"].astype("string")
            df = df[~df["id"].isna()]

            not_seen = ~df["id"].isin(seen_ids)
            new_ids = df.loc[not_seen, "id"].tolist()
            seen_ids.update(new_ids)
            df = df[not_seen]

            if df.empty:
                continue

            df["text_clean"] = df["text"].apply(clean_text)
            df = df[df["text_clean"].str.len() > 0]
            if df.empty:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df[(df["timestamp"] >= START_DATE) & (df["timestamp"] <= END_DATE)]
            if df.empty:
                continue

            iso = df["timestamp"].dt.isocalendar()
            df["iso_year"] = iso.year.astype(int)
            df["iso_week"] = iso.week.astype(int)
            df["year_week"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out), table.schema)
            writer.write_table(table)
            rows_out += len(df)
    finally:
        if writer is not None:
            writer.close()

    print(f"Done. Wrote {rows_out:,} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and filter USC 2024 tweets.")
    parser.add_argument("--input", default="data/usc_2024_full.parquet")
    parser.add_argument("--output", default="data/usc_2024_clean.parquet")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    clean_usc(input_path=args.input, output_path=args.output, batch_size=args.batch_size)
