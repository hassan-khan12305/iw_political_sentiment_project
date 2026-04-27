"""
Ingest USC 2024 election tweets into a single parquet file.

Reads from (in priority order):
  1. data/interim/2024-x/*.parquet  — pre-converted USC shards (fastest)
  2. data/raw/usc_2024/part_*/**/*.csv.gz  — extracted CSV shards
  3. data/raw/x-24-us-election-usc.zip    — original zip (streaming)

Writes:  data/usc_2024_full.parquet

Run:
    iw/bin/python src/ingest_usc.py
"""
from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path
import zipfile

import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_RE_USER_ID   = re.compile(r"'id':\s*(\d+)")
_RE_USERNAME  = re.compile(r"'username':\s*'([^']*)'")
_RE_RAW_DESC  = re.compile(r"'rawDescription':\s*'((?:[^'\\]|\\.)*)'")

KEEP_COLUMNS = [
    "id",
    "id_str",
    "text",
    "rawContent",
    "epoch",
    "lang",
    "type",
    "user_id",
    "user",
    "replyCount",
    "retweetCount",
    "likeCount",
    "quoteCount",
]


def _user_str(user_val: object) -> str | None:
    """Return the raw user field as a string, or None if missing."""
    if user_val is None or (isinstance(user_val, float) and pd.isna(user_val)):
        return None
    if isinstance(user_val, dict):
        # Already parsed — build a quick repr string so regex still works
        return str(user_val)
    return str(user_val)


def extract_user_id(user_val: object) -> str | None:
    s = _user_str(user_val)
    if s is None:
        return None
    m = _RE_USER_ID.search(s)
    return m.group(1) if m else None


def extract_username(user_val: object) -> str | None:
    s = _user_str(user_val)
    if s is None:
        return None
    m = _RE_USERNAME.search(s)
    return m.group(1) if m else None


def extract_description(user_val: object) -> str | None:
    s = _user_str(user_val)
    if s is None:
        return None
    m = _RE_RAW_DESC.search(s)
    return m.group(1) if m else None


def epoch_to_timestamp(epoch_series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(epoch_series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    unit = "ms" if non_null.median() > 10_000_000_000 else "s"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def _iter_input_sources(raw_root: str, zip_path: str | None, existing_parquet_root: str | None):
    # Fast path: reuse existing old USC parquet shards if they already exist.
    if existing_parquet_root:
        existing = sorted(Path(existing_parquet_root).glob("*.parquet"))
        if existing:
            for fp in existing:
                yield ("existing_parquet", fp)
            return

    files = sorted(Path(raw_root).glob("part_*/**/*.csv.gz"))
    if files:
        for fp in files:
            yield ("file", fp)
        return

    if zip_path and Path(zip_path).exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = sorted([n for n in zf.namelist() if n.endswith(".csv.gz") and "/part_" in n])
        for member in members:
            yield ("zip_member", member)
        return

    raise FileNotFoundError(
        f"No USC files found under {raw_root}/part_*/**/*.csv.gz and zip not found/usable: {zip_path}"
    )


def ingest_usc(
    raw_root: str = "data/raw/usc_2024",
    output_path: str = "data/usc_2024_full.parquet",
    chunk_size: int = 100_000,
    zip_path: str = "data/raw/x-24-us-election-usc.zip",
    existing_parquet_root: str = "data/interim/2024-x",
) -> None:
    input_sources = list(
        _iter_input_sources(
            raw_root=raw_root,
            zip_path=zip_path,
            existing_parquet_root=existing_parquet_root,
        )
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    rows_written = 0

    try:
        zip_handle = zipfile.ZipFile(zip_path, "r") if any(k == "zip_member" for k, _ in input_sources) else None
        for kind, source in input_sources:
            print(f"Processing {source}")
            if kind == "existing_parquet":
                chunk_iter = [pd.read_parquet(source)]
            elif kind == "file":
                chunk_iter = pd.read_csv(source, compression="gzip", chunksize=chunk_size, low_memory=False)
            else:
                raw_bytes = zip_handle.read(source)
                gz_stream = gzip.GzipFile(fileobj=io.BytesIO(raw_bytes))
                chunk_iter = pd.read_csv(gz_stream, chunksize=chunk_size, low_memory=False)

            for chunk in chunk_iter:
                cols = [c for c in KEEP_COLUMNS if c in chunk.columns]
                df = chunk[cols].copy()

                if "id" not in df.columns and "id_str" in df.columns:
                    df["id"] = df["id_str"]

                if "rawContent" in df.columns:
                    df["text"] = df["rawContent"].fillna(df.get("text"))

                if "user" in df.columns:
                    if "user_id" not in df.columns:
                        df["user_id"] = df["user"].apply(extract_user_id)
                    df["username"] = df["user"].apply(extract_username)
                    df["user_description"] = df["user"].apply(extract_description)

                rename_map = {
                    "replyCount": "reply_count",
                    "retweetCount": "retweet_count",
                    "likeCount": "like_count",
                    "quoteCount": "quote_count",
                }
                df = df.rename(columns=rename_map)

                if "epoch" not in df.columns:
                    continue

                df["timestamp"] = epoch_to_timestamp(df["epoch"])
                df = df.dropna(subset=["timestamp", "text"])
                if df.empty:
                    continue

                df["text"] = df["text"].astype("string")
                df = df[df["text"].str.strip().str.len() > 0]
                if df.empty:
                    continue

                for str_col in ["id", "user_id", "username", "user_description", "lang", "type"]:
                    if str_col in df.columns:
                        df[str_col] = df[str_col].astype("string")
                    else:
                        df[str_col] = pd.Series(dtype="string")
                if "epoch" in df.columns:
                    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("float64")
                for c in ["reply_count", "retweet_count", "like_count", "quote_count"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
                    else:
                        df[c] = 0

                keep = [
                    "id",
                    "text",
                    "epoch",
                    "timestamp",
                    "lang",
                    "type",
                    "user_id",
                    "username",
                    "user_description",
                    "reply_count",
                    "retweet_count",
                    "like_count",
                    "quote_count",
                ]
                df = df[[c for c in keep if c in df.columns]]

                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out), table.schema)
                writer.write_table(table)
                rows_written += len(df)
    finally:
        if "zip_handle" in locals() and zip_handle is not None:
            zip_handle.close()
        if writer is not None:
            writer.close()

    print(f"Done. Wrote {rows_written:,} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine USC 2024 chunks into one parquet file.")
    parser.add_argument("--raw-root", default="data/raw/usc_2024")
    parser.add_argument("--zip-path", default="data/raw/x-24-us-election-usc.zip")
    parser.add_argument("--existing-parquet-root", default="data/interim/2024-x")
    parser.add_argument("--output", default="data/usc_2024_full.parquet")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    ingest_usc(
        raw_root=args.raw_root,
        output_path=args.output,
        chunk_size=args.chunk_size,
        zip_path=args.zip_path,
        existing_parquet_root=args.existing_parquet_root,
    )
