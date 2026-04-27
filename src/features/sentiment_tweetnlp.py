"""
CardiffNLP twitter-roberta-base-sentiment-latest scoring on usc_2024_clean.parquet.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  - trained on 124M tweets, 3-class: positive / negative / neutral
  - loaded directly via transformers (no tweetnlp wrapper needed)

Reads:   data/usc_2024_clean.parquet   (text_clean, year_week)
Writes:  data/usc_2024_sentiment_tweetnlp.parquet  (weekly aggregates)

Setup on GPU machine:
    pip install torch transformers pyarrow pandas

Run:
    python src/features/sentiment_tweetnlp.py
    # tune --score-batch for GPU VRAM (256 works on ~16GB, try 512 on A100):
    python src/features/sentiment_tweetnlp.py --score-batch 512
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CLEAN_PATH  = Path("data/usc_2024_clean.parquet")
OUT_PATH    = Path("data/usc_2024_sentiment_tweetnlp.parquet")
MODEL_NAME  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
READ_BATCH  = 500_000   # rows read from parquet per iteration
SCORE_BATCH = 256       # texts per model call (tune for GPU VRAM)


def score_tweetnlp(
    clean_path: Path = CLEAN_PATH,
    out_path: Path = OUT_PATH,
    read_batch: int = READ_BATCH,
    score_batch: int = SCORE_BATCH,
) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}. Run the pipeline first.")

    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} on {'GPU' if device == 0 else 'CPU'} (first run downloads ~500MB)...")
    pipe = hf_pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=3,            # return all 3 class scores
        device=device,
        truncation=True,
        max_length=128,
        batch_size=score_batch,  # tells pipeline to batch internally
    )

    weekly: dict[str, dict] = {}
    total = 0

    pf = pq.ParquetFile(str(clean_path))
    for batch in pf.iter_batches(batch_size=read_batch, columns=["text_clean", "year_week"]):
        df = batch.to_pandas()
        df = df.dropna(subset=["text_clean", "year_week"])
        df = df[df["text_clean"].str.len() > 0]
        if df.empty:
            continue

        texts = df["text_clean"].tolist()
        weeks = df["year_week"].tolist()

        # Score in sub-batches; print progress after each chunk
        scores: list[float] = []
        for i in range(0, len(texts), score_batch):
            chunk = texts[i : i + score_batch]
            results = pipe(chunk)
            for r in results:
                probs = {d["label"]: d["score"] for d in r}
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                scores.append(float(pos - neg))  # net sentiment in [-1, 1]
            print(f"  Scored {total + i + len(chunk):,} tweets...", end="\r", flush=True)

        for week, score in zip(weeks, scores):
            if week not in weekly:
                weekly[week] = {"sum_s": 0.0, "n": 0, "n_pos": 0, "n_neg": 0, "n_neu": 0}
            s = weekly[week]
            s["sum_s"] += score
            s["n"]     += 1
            if score > 0.05:
                s["n_pos"] += 1
            elif score < -0.05:
                s["n_neg"] += 1
            else:
                s["n_neu"] += 1

        total += len(df)
        print(f"  Scored {total:,} tweets...", end="\r", flush=True)

    print(f"\nDone. Scored {total:,} tweets total.")

    rows = []
    for week, s in sorted(weekly.items()):
        n = s["n"]
        rows.append({
            "year_week":     week,
            "avg_sentiment": s["sum_s"] / n if n > 0 else 0.0,
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
    parser = argparse.ArgumentParser(description="TweetNLP sentiment scoring for USC 2024 tweets.")
    parser.add_argument("--input",       default=str(CLEAN_PATH))
    parser.add_argument("--output",      default=str(OUT_PATH))
    parser.add_argument("--read-batch",  type=int, default=READ_BATCH)
    parser.add_argument("--score-batch", type=int, default=SCORE_BATCH)
    args = parser.parse_args()
    score_tweetnlp(Path(args.input), Path(args.output), args.read_batch, args.score_batch)
