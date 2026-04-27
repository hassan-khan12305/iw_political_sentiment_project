"""
Observation Matrix Assembly — select, transform, and standardize signals for y_t.

Selects a set of signals from usc_2024_weekly.parquet, 
standardizes them (zero mean, unit variance), checks cross-correlations, and
saves the observation matrix for the latent factor models.

Reads:   data/usc_2024_weekly.parquet
Writes:
  data/observation_matrix.parquet   — standardized y_t  (31 × n_signals)
  results/figures/fig_obs_signals.png
  results/figures/fig_obs_correlation.png

Run:
    iw/bin/python src/models/prepare_observation_matrix.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150,
                     "axes.spines.top": False,
                     "axes.spines.right": False})

WEEKLY_PATH = Path("data/usc_2024_weekly.parquet")
OUT_PATH    = Path("data/observation_matrix.parquet")
FIG_DIR     = Path("results/figures")

# ---------------------------------------------------------------------------
# Signal selection
# Principle: one representative per signal group, low redundancy.
# Prefer net/summary statistics over raw percentages.
# ---------------------------------------------------------------------------

WORDS_PATH = Path("results/topic_words.json")

# Non-topic signals (fixed)
_BASE_SIGNALS: dict[str, str] = {
    "avg_sentiment_vader":    "VADER Sentiment",
    "avg_sentiment_tweetnlp": "RoBERTa Sentiment",
    "lean_score_hashtags":    "Hashtag Lean Score",
    "trump_net_stance":       "Trump Net Stance",
    "harris_net_stance":      "Harris Net Stance",
    "log_tweet_count":        "Log Tweet Volume",
}

def _build_signals() -> dict[str, str]:
    """Return the 6-signal no-topics configuration used in the final model."""
    return dict(_BASE_SIGNALS)

EVENTS = {
    "2024-W30": "Biden exits",
    "2024-W37": "Debate",
    "2024-W45": "Election",
}


def _week_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s + "-1", format="%G-W%V-%u")


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")


def _event_lines(ax, df):
    ymin, ymax = ax.get_ylim()
    for week, label in EVENTS.items():
        row = df[df["year_week"] == week]
        if row.empty:
            continue
        ax.axvline(row["week_start"].iloc[0], color="#9CA3AF", lw=1.2, ls="--")


# ---------------------------------------------------------------------------
# Load + engineer features
# ---------------------------------------------------------------------------

def load_and_engineer(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["week_start"] = _week_to_date(df["year_week"])
    df = df.sort_values("week_start").reset_index(drop=True)

    # Log volume (heavy right skew)
    df["log_tweet_count"] = np.log(df["tweet_count"])

    return df


# ---------------------------------------------------------------------------
# Standardize
# ---------------------------------------------------------------------------

def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[["year_week", "week_start"]].copy()
    for col in cols:
        mean, std = df[col].mean(), df[col].std()
        out[col] = (df[col] - mean) / std
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_signals(df_raw: pd.DataFrame, df_std: pd.DataFrame, labels: dict[str, str]) -> None:
    cols = list(labels.keys())
    n = len(cols)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 2.8),
                             sharex=True, gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    axes = axes.flatten()
    x = df_raw["week_start"]

    colors = plt.cm.tab10(np.linspace(0, 1, n))

    for i, (col, label) in enumerate(labels.items()):
        ax = axes[i]
        raw_col = "tweet_count" if col == "log_tweet_count" else col
        ax.plot(x, df_raw[col], lw=2, color=colors[i], marker="o", markersize=3, zorder=3)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        _event_lines(ax, df_raw)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Observation Matrix Signals — Raw Values (2024 W18–W48)",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig_obs_signals.png")

    # Standardized version — all on one axes for visual comparison
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    for i, (col, label) in enumerate(labels.items()):
        ax2.plot(x, df_std[col], lw=1.8, color=colors[i],
                 marker="o", markersize=3, label=label, zorder=3)
    ax2.axhline(0, color="#9CA3AF", lw=1)
    ax2.set_title("All Signals Standardized (z-scores) — Visual Co-movement Check",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Standard deviations from mean")
    ax2.legend(fontsize=8, frameon=True, ncol=2, loc="lower left")
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.tick_params(axis="x", rotation=20)
    _event_lines(ax2, df_raw)
    fig2.tight_layout()
    _save(fig2, "fig_obs_standardized.png")


def plot_correlation(df_std: pd.DataFrame, labels: dict[str, str]) -> None:
    cols = list(labels.keys())
    rename = {c: labels[c] for c in cols}
    corr = df_std[cols].rename(columns=rename).corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 10},
        linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("Observation Matrix — Signal Correlations\n"
                 "(shared structure -> common latent factor justified)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    _save(fig, "fig_obs_correlation.png")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(df_raw: pd.DataFrame, df_std: pd.DataFrame, labels: dict[str, str]) -> None:
    cols = list(labels.keys())
    print(f"\n{'='*65}")
    print("OBSERVATION MATRIX — SIGNAL SUMMARY")
    print(f"{'='*65}")
    print(f"Signals: {len(cols)}    Weeks: {len(df_raw)}\n")

    # Raw descriptives
    print("Raw signal ranges:")
    for col, label in labels.items():
        s = df_raw[col]
        print(f"  {label:<30}  mean={s.mean():+.3f}  std={s.std():.3f}  "
              f"[{s.min():.3f}, {s.max():.3f}]")

    # Correlation structure
    corr = df_std[cols].corr()
    vals = corr.values[np.tril_indices_from(corr.values, k=-1)]
    print(f"\nPairwise correlations:  mean={vals.mean():.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}")

    # Eigenvalue check — % variance explained by first factor
    eigvals = np.linalg.eigvalsh(corr.values)[::-1]
    pct_var = eigvals / eigvals.sum() * 100
    print(f"\nCorrelation matrix eigenvalues (PCA variance explained):")
    for i, (ev, pv) in enumerate(zip(eigvals, pct_var)):
        print(f"  Factor {i+1}:  λ={ev:.3f}  ({pv:.1f}%)")
        if i >= 4:
            break
    print(f"\n  -> Factor 1 explains {pct_var[0]:.1f}% of shared variance")
    if pct_var[0] > 30:
        print("    Common factor structure is present — 1D latent model is justified.")
    else:
        print("    Weak common factor — consider 2-factor model.")


# ---------------------------------------------------------------------------
# Topic subset experiments
# ---------------------------------------------------------------------------

TOPIC_EXPERIMENTS: dict[str, list[str]] = {
    "no_topics": [
        "avg_sentiment_vader", "avg_sentiment_tweetnlp",
        "lean_score_hashtags", "trump_net_stance", "harris_net_stance",
        "log_tweet_count",
    ],
    "top2_topics": [
        "avg_sentiment_vader", "avg_sentiment_tweetnlp",
        "lean_score_hashtags", "trump_net_stance", "harris_net_stance",
        "topic_03_share", "topic_01_share",
        "log_tweet_count",
    ],
    "all5_topics": [
        "avg_sentiment_vader", "avg_sentiment_tweetnlp",
        "lean_score_hashtags", "trump_net_stance", "harris_net_stance",
        "topic_00_share", "topic_01_share", "topic_02_share",
        "topic_03_share", "topic_04_share",
        "log_tweet_count",
    ],
    "topics_only": [
        "topic_00_share", "topic_01_share", "topic_02_share",
        "topic_03_share", "topic_04_share",
    ],
}


def run_topic_experiments(df: pd.DataFrame) -> None:
    """
    For each topic subset configuration, standardize signals and report
    how much variance the first principal component explains.
    A higher % -> stronger common factor -> 1D latent model more justified.
    """
    print(f"\n{'='*65}")
    print("TOPIC SUBSET EXPERIMENTS — Factor 1 variance explained")
    print(f"{'='*65}")
    print(f"{'Config':<20} {'N signals':>9} {'λ1':>8} {'% var F1':>10} {'% var F1+F2':>12}")
    print("-" * 65)

    results = []
    for name, cols in TOPIC_EXPERIMENTS.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"  {name}: skipping — missing {missing}")
            continue
        mat = df[cols].copy()
        mat = (mat - mat.mean()) / mat.std()
        corr = mat.corr().values
        eigvals = np.linalg.eigvalsh(corr)[::-1]
        pct = eigvals / eigvals.sum() * 100
        results.append({
            "config": name,
            "n_signals": len(cols),
            "lambda1": eigvals[0],
            "pct_var_f1": pct[0],
            "pct_var_f1f2": pct[0] + pct[1],
        })
        print(f"  {name:<18} {len(cols):>9} {eigvals[0]:>8.3f} {pct[0]:>9.1f}% {pct[0]+pct[1]:>11.1f}%")

    best = max(results, key=lambda r: r["pct_var_f1"])
    print(f"\n  Best config by Factor 1 variance: '{best['config']}' ({best['pct_var_f1']:.1f}%)")

    # Save results
    pd.DataFrame(results).to_csv(
        Path("results/topic_experiment_results.csv"), index=False
    )
    print("  Saved -> results/topic_experiment_results.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> pd.DataFrame:
    print("Loading weekly matrix...")
    df = load_and_engineer(WEEKLY_PATH)

    SIGNALS = _build_signals()
    print(f"Signal set ({len(SIGNALS)} signals):")
    for col, label in SIGNALS.items():
        print(f"  {col:<30} -> {label}")

    # Check all selected signals exist
    missing = [c for c in SIGNALS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cols = list(SIGNALS.keys())
    df_std = standardize(df, cols)

    print_stats(df, df_std, SIGNALS)

    print("\nGenerating plots...")
    plot_signals(df, df_std, SIGNALS)
    plot_correlation(df_std, SIGNALS)

    # Topic subset experiments
    # run_topic_experiments(df)

    # Save
    df_std.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved observation matrix -> {OUT_PATH}")
    print(f"Shape: {df_std.shape}  (weeks × signals, standardized)")

    return df_std


if __name__ == "__main__":
    main()
