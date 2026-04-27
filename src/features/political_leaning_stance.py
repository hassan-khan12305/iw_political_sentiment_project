"""
Zero-shot stance detection toward Trump, Harris, and Biden on usc_2024_clean.parquet.

Covers the full candidate timeline:
  Biden was the Democratic nominee for W01–W29 (dropped out 2024-07-21, W30)
  Harris became the nominee starting W30

Uses a cross-encoder NLI model (zero-shot classification) to score each tweet's
stance toward each candidate without any fine-tuning.

Model default: cross-encoder/nli-deberta-v3-small
  - Smaller and faster than bart-large-mnli, still strong accuracy
  - Use --model facebook/bart-large-mnli for higher quality (needs more VRAM)

Output per week:
  favor_trump, against_trump, neutral_trump  (fractions)
  favor_harris, against_harris, neutral_harris
  favor_biden,  against_biden,  neutral_biden
  trump_net  = favor_trump  - against_trump   (positive -> net pro-Trump)
  harris_net = favor_harris - against_harris
  biden_net  = favor_biden  - against_biden

Reads:   data/usc_2024_clean.parquet
Writes:  data/usc_2024_leaning_stance.parquet

Setup on GPU machine:
    pip install transformers torch

Run:
    python src/features/political_leaning_stance.py
    # Limit tweets per week (recommended: saves time, stats still robust at 10K/week):
    python src/features/political_leaning_stance.py --sample-per-week 10000
    # Tune batch size for your GPU VRAM:
    python src/features/political_leaning_stance.py --score-batch 64
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CLEAN_PATH  = Path("data/usc_2024_clean.parquet")
OUT_PATH    = Path("data/usc_2024_leaning_stance.parquet")
READ_BATCH  = 500_000
SCORE_BATCH = 64      # NLI models are heavier; tune down if OOM
DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"

# Hypothesis templates for zero-shot NLI
# "This tweet is in favor of / against {target}."
#
# Biden dropped out 2024-07-21 (2024-W30). Scoring all three candidates across
# all weeks lets the data reflect the switch naturally: biden_net will be high
# in W01-W29 and drop to near-zero after W30; harris_net rises from W30 onward.
TARGETS = {
    "trump":  "Donald Trump",
    "harris": "Kamala Harris",
    "biden":  "Joe Biden",
}


def _run_zeroshot(pipe, texts: list[str], target_label: str) -> list[str]:
    """Return per-text stance label: 'favor', 'against', or 'neutral'."""
    candidate_labels = [
        f"in favor of {target_label}",
        f"against {target_label}",
        f"neutral toward {target_label}",
    ]
    results = pipe(texts, candidate_labels=candidate_labels, multi_label=False)
    # results is a list when input is a list
    if not isinstance(results, list):
        results = [results]
    labels = []
    for r in results:
        top = r["labels"][0]
        if "favor" in top:
            labels.append("favor")
        elif "against" in top:
            labels.append("against")
        else:
            labels.append("neutral")
    return labels


def score_stance(
    clean_path: Path = CLEAN_PATH,
    out_path: Path = OUT_PATH,
    read_batch: int = READ_BATCH,
    score_batch: int = SCORE_BATCH,
    sample_per_week: int | None = None,
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}. Run the pipeline first.")

    from transformers import pipeline as hf_pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading zero-shot model '{model_name}' on {'GPU' if device == 0 else 'CPU'}...")
    pipe = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    # -- Pass 1: collect all (week, text) pairs, optionally sampling per week --
    print("Pass 1: collecting tweet texts by week...")
    week_texts: dict[str, list[str]] = defaultdict(list)
    pf = pq.ParquetFile(str(clean_path))
    for batch in pf.iter_batches(batch_size=read_batch, columns=["text_clean", "year_week"]):
        df = batch.to_pandas()
        df = df.dropna(subset=["text_clean", "year_week"])
        df = df[df["text_clean"].str.len() > 0]
        for week, text in zip(df["year_week"], df["text_clean"]):
            week_texts[week].append(text)

    if sample_per_week:
        print(f"Sampling up to {sample_per_week:,} tweets per week...")
        for week in week_texts:
            if len(week_texts[week]) > sample_per_week:
                week_texts[week] = random.sample(week_texts[week], sample_per_week)

    total_texts = sum(len(v) for v in week_texts.values())
    print(f"Scoring {total_texts:,} texts across {len(week_texts)} weeks...")

    # -- Pass 2: score each target --
    weekly_stats: dict[str, dict] = {}

    for target_key, target_label in TARGETS.items():
        print(f"\nScoring stance toward '{target_label}'...")
        scored = 0
        for week in sorted(week_texts.keys()):
            texts = week_texts[week]
            if week not in weekly_stats:
                weekly_stats[week] = {}
            s = weekly_stats[week]
            key_prefix = f"{target_key}_"
            s.setdefault(f"{key_prefix}favor",   0)
            s.setdefault(f"{key_prefix}against",  0)
            s.setdefault(f"{key_prefix}neutral",  0)
            s.setdefault(f"{key_prefix}n",        0)

            for i in range(0, len(texts), score_batch):
                chunk = texts[i : i + score_batch]
                stance_labels = _run_zeroshot(pipe, chunk, target_label)
                for lbl in stance_labels:
                    s[f"{key_prefix}{lbl}"] += 1
                s[f"{key_prefix}n"] += len(chunk)

            scored += len(texts)
            print(f"  [{target_label}] {scored:,}/{total_texts:,}", end="\r", flush=True)

    print(f"\nDone scoring.")

    rows = []
    for week in sorted(weekly_stats.keys()):
        s = weekly_stats[week]
        row = {"year_week": week}
        for tk in TARGETS:
            n = s.get(f"{tk}_n", 0)
            fav = s.get(f"{tk}_favor",   0)
            aga = s.get(f"{tk}_against", 0)
            neu = s.get(f"{tk}_neutral", 0)
            row[f"favor_{tk}"]   = fav / n if n > 0 else 0.0
            row[f"against_{tk}"] = aga / n if n > 0 else 0.0
            row[f"neutral_{tk}"] = neu / n if n > 0 else 0.0
            row[f"{tk}_net"]     = (fav - aga) / n if n > 0 else 0.0  # +1 = all favor
            row[f"scored_{tk}"]  = n
        rows.append(row)

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(df_out_path := out_path, index=False)
    print(f"Saved -> {df_out_path}")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot stance detection toward Trump/Harris for USC 2024 tweets."
    )
    parser.add_argument("--input",           default=str(CLEAN_PATH))
    parser.add_argument("--output",          default=str(OUT_PATH))
    parser.add_argument("--read-batch",      type=int,   default=READ_BATCH)
    parser.add_argument("--score-batch",     type=int,   default=SCORE_BATCH,
                        help="Texts per model call. Reduce if GPU OOM.")
    parser.add_argument("--sample-per-week", type=int,   default=None,
                        help="Cap tweets per week (e.g. 10000). Recommended for speed.")
    parser.add_argument("--model",           default=DEFAULT_MODEL,
                        help="HuggingFace zero-shot model name.")
    args = parser.parse_args()
    score_stance(
        clean_path=Path(args.input),
        out_path=Path(args.output),
        read_batch=args.read_batch,
        score_batch=args.score_batch,
        sample_per_week=args.sample_per_week,
        model_name=args.model,
    )
