"""
LDA Topic Modeling — 2024 US Election Tweets

Pipeline:
  - Stratified sample ~15K tweets/week (≈465K total) for both fit and inference
  - K sweep over {3,5,7,10,15}, pick best by c_v coherence
  - Preprocessing: NLTK stopwords + expanded political/discourse noise + lemmatization
  - Infer topic distributions on sample, aggregate to weekly means
  - Weekly topic signal is the main deliverable for the latent state model

Preprocessing pipeline (on top of text_clean):
  1. Lowercase, strip URLs/mentions, unhash hashtags
  2. Remove punctuation/numbers
  3. Tokenize, filter < 3 chars
  4. Remove NLTK English stopwords
  5. Remove political noise words (candidate names, generic discourse markers)
  6. Lemmatize with NLTK WordNetLemmatizer

Reads:   data/usc_2024_clean.parquet
Writes:
  models/lda/lda_k{K}.model     — gensim model
  models/lda/dictionary.dict    — gensim dictionary
  results/topic_words.json      — top words per topic for best K
  results/coherence_scores.csv  — K vs c_v score
  data/usc_2024_weekly_topics.parquet  — weekly topic proportions

Run:
    iw/bin/python src/features/topic_lda.py
"""
from __future__ import annotations

import json
import re
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_PATH   = Path("data/usc_2024_clean.parquet")
WEEKLY_PATH  = Path("data/usc_2024_weekly.parquet")
MODEL_DIR    = Path("models/lda")
RESULTS_DIR  = Path("results")

SAMPLE_PER_WEEK = 15_000   # tweets sampled per week for fit + inference
K_VALUES        = [3, 5, 7, 10, 15]
RANDOM_SEED     = 42

# LDA hyperparams
LDA_PASSES      = 10
LDA_ITERATIONS  = 100
LDA_CHUNK       = 2000
NO_BELOW        = 10    # min doc frequency for vocab
NO_ABOVE        = 0.50  # max doc fraction for vocab
KEEP_N          = 50_000


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Political noise words — appear across ALL topics, add no discriminatory signal.
# Two categories:
#   (a) Candidate/party names — ubiquitous, mask topic structure
#   (b) Generic discourse markers — filler words that survived NLTK stopwords
# ---------------------------------------------------------------------------
_POLITICAL_NOISE = {
    # Candidate & party names
    "trump", "donald", "biden", "joe", "harris", "kamala",
    "president", "presidential", "vice",
    "election", "vote", "voting", "voter", "voters", "voted", "elect",
    "democrat", "democratic", "republican", "gop", "maga",
    "white", "house", "government", "america", "american", "country",
    "party", "candidate", "campaign",
    # Generic high-frequency words that survived NLTK stopwords
    "people", "said", "going", "know", "think", "get", "got", "like",
    "just", "would", "could", "also", "one", "way", "make", "time",
    "say", "see", "want", "need", "new", "year", "day", "back", "even",
    "still", "come", "take", "much", "use", "good", "great", "many",
    "never", "always", "every", "really", "yes", "let", "well", "stop",
    "nothing", "anything", "something", "everything", "someone", "everyone",
    "actually", "pretty", "quite", "maybe", "probably", "already",
    "though", "because", "since", "thing", "things", "made", "make",
    "going", "done", "look", "looks", "looking", "trying", "tried",
    "said", "says", "told", "tell", "ask", "asked",
    "two", "three", "four", "five", "first", "second", "next", "last",
    "right", "left",   # overloaded — political AND directional
    # Twitter artifacts
    "amp", "via", "re", "ve", "ll", "https", "http",
    # Year references
    "2024", "2023", "2022", "2020",
}

_STOP:      set[str] | None = None
_LEMMATIZER = None


def _get_stopwords() -> set[str]:
    global _STOP
    if _STOP is not None:
        return _STOP
    try:
        from nltk.corpus import stopwords
        _STOP = set(stopwords.words("english")) | _POLITICAL_NOISE
    except Exception:
        _STOP = _POLITICAL_NOISE
    return _STOP


def _get_lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is not None:
        return _LEMMATIZER
    try:
        from nltk.stem import WordNetLemmatizer
        _LEMMATIZER = WordNetLemmatizer()
    except Exception:
        _LEMMATIZER = None
    return _LEMMATIZER


_RE_URL     = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@\w+")
_RE_HASH    = re.compile(r"#(\w+)")
_RE_NONALPH = re.compile(r"[^a-z\s]")


def tokenize(text: str, stop: set[str]) -> list[str]:
    """
    LDA tokenizer with lemmatization.
    Pipeline: lowercase -> strip URLs/mentions -> unhash -> remove non-alpha
              -> tokenize -> filter stopwords/length -> lemmatize (noun mode)
    """
    text = text.lower()
    text = _RE_URL.sub("", text)
    text = _RE_MENTION.sub("", text)
    text = _RE_HASH.sub(r"\1", text)   # keep hashtag word without #
    text = _RE_NONALPH.sub(" ", text)
    tokens = [t for t in text.split() if len(t) >= 3 and t not in stop]

    lemmatizer = _get_lemmatizer()
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        # Re-filter after lemmatization (lemma might be in stoplist)
        tokens = [t for t in tokens if t not in stop and len(t) >= 3]

    return tokens


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stratified_sample(path: Path, n_per_week: int, seed: int) -> pd.DataFrame:
    """
    Stream through parquet and collect up to n_per_week tweets per year_week.
    Returns a DataFrame with columns [year_week, text_clean].
    """
    print(f"  Sampling up to {n_per_week:,} tweets/week from {path.name}...")
    rng = np.random.default_rng(seed)

    buckets: dict[str, list[str]] = {}
    pf = pq.ParquetFile(str(path))

    for batch in pf.iter_batches(batch_size=500_000, columns=["year_week", "text_clean"]):
        df = batch.to_pandas()
        for week, grp in df.groupby("year_week"):
            texts = grp["text_clean"].dropna().tolist()
            if week not in buckets:
                buckets[week] = []
            buckets[week].extend(texts)

    # Sample from each bucket
    rows = []
    for week, texts in sorted(buckets.items()):
        if len(texts) > n_per_week:
            idx = rng.choice(len(texts), size=n_per_week, replace=False)
            texts = [texts[i] for i in idx]
        for t in texts:
            rows.append({"year_week": week, "text_clean": t})

    df_out = pd.DataFrame(rows)
    weeks = df_out["year_week"].nunique()
    print(f"  Loaded {len(df_out):,} tweets across {weeks} weeks")
    return df_out


# ---------------------------------------------------------------------------
# Build vocab
# ---------------------------------------------------------------------------

def build_corpus(
    texts: list[list[str]],
    no_below: int = NO_BELOW,
    no_above: float = NO_ABOVE,
    keep_n: int = KEEP_N,
):
    from gensim.corpora import Dictionary
    print(f"  Building dictionary (no_below={no_below}, no_above={no_above})...")
    dct = Dictionary(texts)
    dct.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dct.compactify()
    print(f"  Vocab size: {len(dct):,} tokens")
    corpus = [dct.doc2bow(t) for t in texts]
    return dct, corpus


# ---------------------------------------------------------------------------
# Train LDA
# ---------------------------------------------------------------------------

def train_lda(corpus, dct, K: int, seed: int):
    from gensim.models import LdaModel
    print(f"  Training LDA K={K}  (passes={LDA_PASSES}, iter={LDA_ITERATIONS})...")
    model = LdaModel(
        corpus=corpus,
        id2word=dct,
        num_topics=K,
        random_state=seed,
        passes=LDA_PASSES,
        iterations=LDA_ITERATIONS,
        chunksize=LDA_CHUNK,
        alpha="auto",
        eta="auto",
        per_word_topics=False,
    )
    return model


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def coherence_score(model, texts: list[list[str]], dct) -> float:
    from gensim.models import CoherenceModel
    cm = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dct,
        coherence="c_v",
    )
    return cm.get_coherence()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def infer_topic_matrix(model, corpus: list, K: int) -> np.ndarray:
    """Return (N, K) matrix of topic proportions for each document."""
    out = np.zeros((len(corpus), K), dtype=np.float32)
    for i, bow in enumerate(corpus):
        topics = model.get_document_topics(bow, minimum_probability=0.0)
        for tid, prob in topics:
            out[i, tid] = prob
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load stratified sample ──────────────────────────────────────────
    print("\n[1/6] Loading stratified sample...")
    df = load_stratified_sample(CLEAN_PATH, SAMPLE_PER_WEEK, RANDOM_SEED)

    stop = _get_stopwords()
    print("[2/6] Tokenizing...")
    df["tokens"] = df["text_clean"].map(lambda t: tokenize(str(t), stop))
    # Drop very short docs (< 3 tokens after cleaning)
    df = df[df["tokens"].map(len) >= 3].reset_index(drop=True)
    print(f"  {len(df):,} docs after filtering short tweets")
    texts = df["tokens"].tolist()

    # ── 2. Build vocab + corpus ────────────────────────────────────────────
    print("[3/6] Building vocab...")
    dct, corpus = build_corpus(texts)

    # ── 3. K sweep — train + coherence ────────────────────────────────────
    print("[4/6] K sweep: fitting LDA models...")
    scores: dict[int, float] = {}
    models: dict[int, object] = {}
    for K in K_VALUES:
        m = train_lda(corpus, dct, K, RANDOM_SEED)
        cv = coherence_score(m, texts, dct)
        scores[K] = cv
        models[K] = m
        print(f"    K={K:2d}  c_v coherence = {cv:.4f}")
        m.save(str(MODEL_DIR / f"lda_k{K}.model"))

    # Save dictionary
    dct.save(str(MODEL_DIR / "dictionary.dict"))

    # Save coherence scores
    coh_df = pd.DataFrame(
        [{"K": k, "coherence_cv": v} for k, v in scores.items()]
    )
    coh_df.to_csv(RESULTS_DIR / "coherence_scores.csv", index=False)
    print("\n  Coherence scores:")
    print(coh_df.to_string(index=False))

    # ── 4. Pick best K ────────────────────────────────────────────────────
    best_K = max(scores, key=scores.get)
    print(f"\n  Best K by c_v coherence: {best_K}")
    best_model = models[best_K]

    # ── 5. Save top words per topic ───────────────────────────────────────
    print("[5/6] Saving top words per topic...")
    top_words: dict[str, list[str]] = {}
    for tid in range(best_K):
        words = [w for w, _ in best_model.show_topic(tid, topn=20)]
        top_words[f"topic_{tid:02d}"] = words
        print(f"  topic_{tid:02d}: {' | '.join(words[:10])}")

    with open(RESULTS_DIR / "topic_words.json", "w") as fh:
        json.dump({"K": best_K, "topics": top_words}, fh, indent=2)

    # ── 6. Infer topic distributions + aggregate to weekly ────────────────
    print("[6/6] Inferring topic distributions and aggregating to weekly...")
    topic_matrix = infer_topic_matrix(best_model, corpus, best_K)

    topic_cols = [f"topic_{i:02d}_share" for i in range(best_K)]
    df_topics = pd.DataFrame(topic_matrix, columns=topic_cols)
    df_topics["year_week"] = df["year_week"].values

    weekly_topics = (
        df_topics.groupby("year_week")[topic_cols]
        .mean()
        .reset_index()
    )

    # Sanity check — shares should sum to ~1
    share_sum = weekly_topics[topic_cols].sum(axis=1)
    print(f"  Topic share sum per week: mean={share_sum.mean():.4f}  "
          f"min={share_sum.min():.4f}  max={share_sum.max():.4f}")

    # Save
    out_path = Path("data/usc_2024_weekly_topics.parquet")
    weekly_topics.to_parquet(out_path, index=False)
    print(f"\n  Saved weekly topic signals -> {out_path}")
    print(f"  Shape: {weekly_topics.shape}  ({best_K} topics × {len(weekly_topics)} weeks)")

    print("\nDone.")
    return weekly_topics, best_model, dct, top_words, coh_df


if __name__ == "__main__":
    run()
