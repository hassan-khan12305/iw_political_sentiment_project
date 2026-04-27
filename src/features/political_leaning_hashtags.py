"""
Hashtag-based political leaning classification on usc_2024_clean.parquet.

Each tweet is labeled:
  R  — contains R-leaning partisan keywords only
  D  — contains D-leaning partisan keywords only
  M  — contains both (mixed / cross-partisan)
  N  — neither (neutral / unlabeled)

Reads:   data/usc_2024_clean.parquet   (text_clean, year_week)
Writes:  data/usc_2024_leaning_hashtags.parquet  (weekly aggregates)

Run:
    iw/bin/python src/features/political_leaning_hashtags.py

Note: text_clean has # removed but words preserved and lowercased.
So #MAGA -> maga, #VoteBlue -> voteblue. Matching uses word boundaries.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CLEAN_PATH = Path("data/usc_2024_clean.parquet")
OUT_PATH   = Path("data/usc_2024_leaning_hashtags.parquet")
BATCH_SIZE = 500_000

# ---------------------------------------------------------------------------
# Partisan keyword lists  (lowercase, no #)
# Focus on compound hashtag-style tokens that are unambiguous partisan signals.
# These appear in text_clean as plain words after # is stripped.
# ---------------------------------------------------------------------------

R_KEYWORDS = [
    # MAGA / Trump — high-frequency confirmed in dataset (Table 2)
    "maga",                         # #maga:               559,722
    "trump2024",                    # #trump2024:           341,792
    "donaldtrump",                  # #donaldtrump:          40,780
    "trump2024tosaveamerica",       # #trump2024tosaveamerica: 18,455
    "ultramaga",                    # in tracked keywords table
    "trumptrain",                   # in tracked keywords table
    "trump45", "trump47", "trumpvance", "trumpwon",
    "maga2024", "magamovement", "magamoving",
    "magaking", "maganomics", "saveamerica", "americafirst",
    "draintheswamp", "buildthewall", "kag",         # Keep America Great
    # Anti-Biden/Harris coded terms
    "letsgorandon", "letsgobrandon",  # both spellings common
    "fjb", "fjb2024", "nobidenharris", "neverbidenharris",
    "bidenout", "bidenoutofficenow",
    # Republican / conservative general
    "votetrump", "votetrump2024", "votegop", "voterepublican",
    "voterepublicans", "redwave", "redwave2024", "gop2024",
    "republican2024", "wwg1wga",       # QAnon
]

D_KEYWORDS = [
    # Biden (primary nominee W01–W29; dropped out 2024-07-21 / 2024-W30)
    # High-frequency confirmed in dataset (Table 2)
    "joebiden",                     # #joebiden:            31,620
    "biden2024",                    # #biden2024:            39,588
    "voteforBiden", "bidenwon", "joebiden2024",
    # Harris (nominee from W30 onward)
    # High-frequency confirmed in dataset (Table 2)
    "kamalaharris",                 # #kamalaharris:         17,436
    "bidenharris2024",              # #bidenharris2024:     166,669
    "bidenharris", "harris2024", "harris47",
    "kamalaharris2024", "kh47", "voteharris", "voteforharris",
    "harriswalz", "walzharris", "timwalz", "walz2024",
    "voteharriswalz", "voteforharriswalz",
    # Vote blue — confirmed in dataset (Table 2)
    "voteblue",                     # #voteblue:            19,332
    "voteblue2024",                 # #voteblue2024:         17,477
    "bluewave", "bluewave2024",
    "votedem", "votedems", "votedemocrats", "democrats2024",
    "democraticparty2024",
    # Anti-Trump coded terms
    "resisttrump", "nevertrump", "dumptrump", "trumplied",
    "notmypresident", "trumpisacriminal", "convicttrump",
    "lockhimup", "trumpindicted", "trumpguilty",
]

# Pre-compile as single regex patterns for speed
_R_PAT = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in R_KEYWORDS) + r")\b"
)
_D_PAT = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in D_KEYWORDS) + r")\b"
)


def classify_leaning(text: str) -> str:
    has_r = bool(_R_PAT.search(text))
    has_d = bool(_D_PAT.search(text))
    if has_r and has_d:
        return "M"   # mixed
    if has_r:
        return "R"
    if has_d:
        return "D"
    return "N"       # neutral / unclassified


def score_leaning_hashtags(
    clean_path: Path = CLEAN_PATH,
    out_path: Path = OUT_PATH,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing: {clean_path}. Run the pipeline first.")

    weekly: dict[str, dict] = {}
    total = 0

    pf = pq.ParquetFile(str(clean_path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text_clean", "year_week"]):
        df = batch.to_pandas()
        df = df.dropna(subset=["text_clean", "year_week"])
        df = df[df["text_clean"].str.len() > 0]
        if df.empty:
            continue

        df["leaning"] = df["text_clean"].apply(classify_leaning)

        for week, grp in df.groupby("year_week"):
            counts = grp["leaning"].value_counts()
            if week not in weekly:
                weekly[week] = {"n_r": 0, "n_d": 0, "n_m": 0, "n_n": 0}
            s = weekly[week]
            s["n_r"] += int(counts.get("R", 0))
            s["n_d"] += int(counts.get("D", 0))
            s["n_m"] += int(counts.get("M", 0))
            s["n_n"] += int(counts.get("N", 0))

        total += len(df)
        print(f"  Labeled {total:,} tweets...", end="\r", flush=True)

    print(f"\nDone. Labeled {total:,} tweets total.")

    rows = []
    for week, s in sorted(weekly.items()):
        n = s["n_r"] + s["n_d"] + s["n_m"] + s["n_n"]
        n_partisan = s["n_r"] + s["n_d"] + s["n_m"]
        lean = (s["n_r"] - s["n_d"]) / (s["n_r"] + s["n_d"]) if (s["n_r"] + s["n_d"]) > 0 else 0.0
        rows.append({
            "year_week":       week,
            "pct_r":           s["n_r"] / n if n > 0 else 0.0,
            "pct_d":           s["n_d"] / n if n > 0 else 0.0,
            "pct_mixed":       s["n_m"] / n if n > 0 else 0.0,
            "pct_neutral":     s["n_n"] / n if n > 0 else 0.0,
            "pct_partisan":    n_partisan / n if n > 0 else 0.0,
            "lean_score":      lean,    # +1 = all R, -1 = all D (among labeled tweets)
            "labeled_tweets":  n,
        })

    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hashtag-based political leaning classification for USC 2024 tweets."
    )
    parser.add_argument("--input",      default=str(CLEAN_PATH))
    parser.add_argument("--output",     default=str(OUT_PATH))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    score_leaning_hashtags(Path(args.input), Path(args.output), args.batch_size)
