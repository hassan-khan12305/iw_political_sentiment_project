"""
Static Factor Model — Baseline latent climate estimation.

Implements two estimators under static (time-invariant) loadings:

  1. PCA  — instant, rotation-free baseline.
  2. statsmodels DynamicFactor — ML-estimated state-space model with AR(1)
     latent factor and iid measurement noise.

Both are run in 1-factor and 2-factor configurations.

Signal set: 6-signal no-topics config (50.4% Factor 1 variance).

Reads:   data/usc_2024_weekly.parquet
Writes:
  results/figures/fig_factor_pca.png          — PCA latent series + loadings
  results/figures/fig_factor_2f_pca.png       — 2-factor PCA scores + loadings
  results/figures/fig_factor_ssm_1f.png       — DynamicFactor smoothed 1-factor
  results/figures/fig_factor_ssm_2f.png       — DynamicFactor smoothed 2-factor
  results/figures/fig_factor_loadings.png     — loadings comparison (PCA vs SSM)

Run:
    iw/bin/python src/models/factor_static.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEEKLY_PATH = Path("data/usc_2024_weekly.parquet")
FIG_DIR     = Path("results/figures")
RESULTS_DIR = Path("results")

# Primary signal set — 6 signals, no topics (Factor 1 = 50.4% variance)
SIGNALS_6: dict[str, str] = {
    "avg_sentiment_vader":    "VADER Sentiment",
    "avg_sentiment_tweetnlp": "RoBERTa Sentiment",
    "lean_score_hashtags":    "Hashtag Lean",
    "trump_net_stance":       "Trump Net Stance",
    "harris_net_stance":      "Harris Net Stance",
    "log_tweet_count":        "Log Volume",
}

# Extended 8-signal set (includes 2 topic signals)
SIGNALS_8: dict[str, str] = {
    **SIGNALS_6,
    "topic_01_share":         "Topic 01 Share",
    "topic_03_share":         "Topic 03 Share",
}

# Key political events for annotation
EVENTS = {
    "2024-W30": "Biden exits\nJul 21",
    "2024-W34": "DNC\nAug 19",
    "2024-W37": "Debate\nSep 10",
    "2024-W40": "VP Debate\nOct 1",
    "2024-W45": "Election\nNov 5",
}

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _week_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s + "-1", format="%G-W%V-%u")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")


def _event_lines(ax, df: pd.DataFrame, style: str = "top") -> None:
    """Draw vertical event lines with alternating top/bottom labels."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for i, (week, label) in enumerate(EVENTS.items()):
        row = df[df["year_week"] == week]
        if row.empty:
            continue
        x = row["week_start"].iloc[0]
        ax.axvline(x, color="#9CA3AF", lw=1.1, ls="--", zorder=1)
        pos = "top" if i % 2 == 0 else "bottom"
        y   = ymax - 0.04 * span if pos == "top" else ymin + 0.04 * span
        va  = "top" if pos == "top" else "bottom"
        ax.text(x, y, label, fontsize=7, color="#6B7280",
                ha="center", va=va, linespacing=1.3)


def _month_fmt(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=20, labelsize=8)


def _standardize(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Return (T, n) z-score matrix."""
    mat = df[cols].values.astype(float)
    mat = (mat - mat.mean(axis=0)) / mat.std(axis=0)
    return mat


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(signals: dict[str, str]) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_parquet(WEEKLY_PATH)
    df["week_start"] = _week_to_date(df["year_week"])
    df = df.sort_values("week_start").reset_index(drop=True)
    df["log_tweet_count"] = np.log(df["tweet_count"])

    cols = list(signals.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in weekly parquet: {missing}")

    Y = _standardize(df, cols)
    return df, Y, cols


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_pca(Y: np.ndarray, n_factors: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD.

    Returns:
      scores    (T, n_factors)   — factor scores (latent series)
      loadings  (p, n_factors)   — principal component loadings
      pct_var   (n_factors,)     — % variance explained per factor
    """
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    scores   = U[:, :n_factors] * S[:n_factors]
    loadings = Vt[:n_factors].T           # (p, n_factors)
    total_var = (S ** 2).sum()
    pct_var  = (S[:n_factors] ** 2) / total_var * 100

    # Sign convention: Factor 1 positively loads on sentiment signals
    # (so positive f_t = more positive/favorable political climate)
    ref_cols_pos = ["avg_sentiment_vader", "avg_sentiment_tweetnlp"]
    return scores, loadings, pct_var


def _sign_flip(scores: np.ndarray, loadings: np.ndarray,
               col_names: list[str], ref_positive: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flip factor sign so reference columns load positively."""
    for k in range(scores.shape[1]):
        ref_idx = [i for i, c in enumerate(col_names) if c in ref_positive]
        if ref_idx and loadings[ref_idx[0], k] < 0:
            scores[:, k]   *= -1
            loadings[:, k] *= -1
    return scores, loadings


# ---------------------------------------------------------------------------
# statsmodels DynamicFactor (state-space ML)
# ---------------------------------------------------------------------------

def run_dynamic_factor(Y: np.ndarray, n_factors: int = 1,
                       maxiter: int = 500) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Fit statsmodels DynamicFactor model with static loadings and AR(1) latent factor.

    Returns:
      result    — fitted model result object
      f_smooth  (T, n_factors) — Kalman-smoothed latent factors
      loadings  (p, n_factors) — ML-estimated measurement loadings
    """
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

    print(f"  Fitting DynamicFactor (k_factors={n_factors}, AR(1))...")
    mod = DynamicFactor(Y, k_factors=n_factors, factor_order=1,
                        error_cov_type="diagonal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(maxiter=maxiter, disp=False, method="lbfgs")

    f_smooth = res.factors["smoothed"].T         # statsmodels returns (k, T) -> transpose to (T, k)
    # loadings: shape (p, n_factors) — stored in res.params (array) + mod.param_names
    p = Y.shape[1]
    param_series = pd.Series(res.params, index=mod.param_names)
    loadings = np.zeros((p, n_factors))
    for k in range(n_factors):
        fname = f"f{k+1}"
        for j in range(p):
            key = f"loading.{fname}.{mod.endog_names[j]}"
            if key in param_series.index:
                loadings[j, k] = param_series[key]

    return res, f_smooth, loadings


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pca_1f(scores: np.ndarray, loadings: np.ndarray,
                pct_var: np.ndarray, col_labels: list[str],
                df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 5),
        gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.35}
    )

    # Left — latent factor series
    x = df["week_start"]
    ax1.plot(x, scores[:, 0], lw=2.5, color="#7C3AED", marker="o",
             markersize=4, zorder=3, label="PCA Factor 1")
    ax1.axhline(0, color="#9CA3AF", lw=1)
    ax1.set_title(f"PCA Factor 1  ({pct_var[0]:.1f}% variance explained)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Factor score (a.u.)")
    _month_fmt(ax1)
    _event_lines(ax1, df)

    # Right — loadings bar chart
    y_pos = np.arange(len(col_labels))
    colors = ["#7C3AED" if v >= 0 else "#D7263D" for v in loadings[:, 0]]
    ax2.barh(y_pos, loadings[:, 0], color=colors, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(col_labels, fontsize=9)
    ax2.axvline(0, color="#6B7280", lw=0.8)
    ax2.set_title("Loadings (Λ)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Loading coefficient")

    fig.suptitle("PCA Baseline — 1-Factor Static Model\n"
                 "6-signal no-topics configuration",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.78)
    _save(fig, "fig_factor_pca.png")


def plot_pca_2f(scores: np.ndarray, loadings: np.ndarray,
                pct_var: np.ndarray, col_labels: list[str],
                df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 2, width_ratios=[2.5, 1],
                           hspace=0.42, wspace=0.38)

    x = df["week_start"]
    colors_f = ["#7C3AED", "#059669"]

    for k in range(2):
        ax_ser = fig.add_subplot(gs[k, 0])
        ax_ld  = fig.add_subplot(gs[k, 1])

        ax_ser.plot(x, scores[:, k], lw=2.5, color=colors_f[k],
                    marker="o", markersize=4, zorder=3)
        ax_ser.axhline(0, color="#9CA3AF", lw=1)
        ax_ser.set_title(
            f"Factor {k+1}  ({pct_var[k]:.1f}% variance explained)",
            fontsize=11, fontweight="bold")
        ax_ser.set_ylabel("Factor score (a.u.)")
        _month_fmt(ax_ser)
        _event_lines(ax_ser, df)

        y_pos = np.arange(len(col_labels))
        bar_c = [colors_f[k] if v >= 0 else "#D7263D" for v in loadings[:, k]]
        ax_ld.barh(y_pos, loadings[:, k], color=bar_c, alpha=0.85)
        ax_ld.set_yticks(y_pos)
        ax_ld.set_yticklabels(col_labels, fontsize=8.5)
        ax_ld.axvline(0, color="#6B7280", lw=0.8)
        ax_ld.set_title(f"Loadings F{k+1}", fontsize=10, fontweight="bold")

    fig.suptitle("PCA Baseline — 2-Factor Model\n"
                 "6-signal no-topics configuration",
                 fontsize=13, fontweight="bold")
    _save(fig, "fig_factor_2f_pca.png")


def plot_ssm(f_smooth: np.ndarray, f_pca: np.ndarray,
             pct_var_pca: np.ndarray, df: pd.DataFrame,
             n_factors: int, label: str = "") -> None:
    fig, axes = plt.subplots(n_factors, 1,
                             figsize=(14, 4.5 * n_factors),
                             sharex=True,
                             gridspec_kw={"hspace": 0.38})
    if n_factors == 1:
        axes = [axes]

    x = df["week_start"]
    colors_f = ["#7C3AED", "#059669"]

    for k in range(n_factors):
        ax = axes[k]
        ax.plot(x, f_smooth[:, k], lw=2.5, color=colors_f[k],
                marker="o", markersize=4, zorder=3,
                label=f"SSM smoothed F{k+1}")
        if k == 0 and f_pca is not None:
            # Overlay PCA F1 for comparison (scale-matched)
            f_p = f_pca[:, 0]
            scale = f_smooth[:, 0].std() / (f_p.std() + 1e-9)
            ax.plot(x, f_p * scale, lw=1.5, color="#9CA3AF", ls="--",
                    alpha=0.8, label="PCA F1 (rescaled)")
        ax.axhline(0, color="#9CA3AF", lw=1)
        ax.set_title(f"Smoothed Factor {k+1}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Latent factor (a.u.)")
        ax.legend(fontsize=9, frameon=True)
        _month_fmt(ax)
        _event_lines(ax, df)

    fig.suptitle(f"State-Space Model — {n_factors}-Factor Static Loadings\n"
                 f"{label}",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.82)
    suffix = "1f" if n_factors == 1 else "2f"
    _save(fig, f"fig_factor_ssm_{suffix}.png")


def plot_loadings_comparison(
    loadings_pca: np.ndarray, loadings_ssm: np.ndarray,
    col_labels: list[str], pct_var_pca: np.ndarray
) -> None:
    p = len(col_labels)
    x = np.arange(p)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x - width / 2, loadings_pca[:, 0], height=width,
            color="#7C3AED", alpha=0.85, label=f"PCA F1 ({pct_var_pca[0]:.1f}%)")
    ax.barh(x + width / 2, loadings_ssm[:, 0], height=width,
            color="#059669", alpha=0.85, label="SSM F1 (ML)")
    ax.set_yticks(x)
    ax.set_yticklabels(col_labels, fontsize=10)
    ax.axvline(0, color="#6B7280", lw=0.8)
    ax.set_xlabel("Loading coefficient")
    ax.set_title("Factor 1 Loadings: PCA vs State-Space Model",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_factor_loadings.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STATIC FACTOR MODEL — 6-signal no-topics config")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df, Y, cols = load_data(SIGNALS_6)
    col_labels   = list(SIGNALS_6.values())
    print(f"  Observation matrix: {Y.shape}  (weeks × signals)")

    # ── PCA 1-factor ─────────────────────────────────────────────────────────
    print("\n[2/5] PCA — 1-factor and 2-factor...")
    scores_1f, loadings_1f, pct_1f = run_pca(Y, n_factors=1)
    scores_1f, loadings_1f = _sign_flip(scores_1f, loadings_1f, cols,
                                        ["avg_sentiment_vader",
                                         "avg_sentiment_tweetnlp"])
    scores_2f, loadings_2f, pct_2f = run_pca(Y, n_factors=2)
    scores_2f, loadings_2f = _sign_flip(scores_2f, loadings_2f, cols,
                                        ["avg_sentiment_vader",
                                         "avg_sentiment_tweetnlp"])

    print(f"  PCA Factor 1: {pct_1f[0]:.1f}% variance explained")
    print(f"  PCA Factor 2: {pct_2f[1]:.1f}% variance explained  "
          f"(cumulative {pct_2f[0]+pct_2f[1]:.1f}%)")

    # ── DynamicFactor — 1-factor ──────────────────────────────────────────────
    print("\n[3/5] DynamicFactor SSM — 1-factor with AR(1) latent...")
    res_ssm, f_ssm_1f, loadings_ssm = run_dynamic_factor(Y, n_factors=1)
    # Sign-flip SSM factor to match PCA orientation
    if np.corrcoef(scores_1f[:, 0], f_ssm_1f[:, 0])[0, 1] < 0:
        f_ssm_1f   *= -1
        loadings_ssm *= -1

    # ── DynamicFactor — 2-factor ──────────────────────────────────────────────
    print("\n[4/5] DynamicFactor SSM — 2-factor with AR(1) latent...")
    res_ssm_2f, f_ssm_2f, loadings_ssm_2f = run_dynamic_factor(Y, n_factors=2)
    if np.corrcoef(scores_1f[:, 0], f_ssm_2f[:, 0])[0, 1] < 0:
        f_ssm_2f[:, 0]   *= -1
        loadings_ssm_2f[:, 0] *= -1

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots...")
    plot_pca_1f(scores_1f, loadings_1f, pct_1f, col_labels, df)
    plot_pca_2f(scores_2f, loadings_2f, pct_2f, col_labels, df)
    plot_ssm(f_ssm_1f, scores_1f, pct_1f, df, n_factors=1,
             label="6-signal no-topics")
    plot_ssm(f_ssm_2f, scores_2f, pct_2f, df, n_factors=2,
             label="6-signal no-topics")
    plot_loadings_comparison(loadings_1f, loadings_ssm, col_labels, pct_1f)

    # ── SSM fit stats ─────────────────────────────────────────────────────────
    print("\nSSM 1-factor diagnostics:")
    print(f"  Log-likelihood:  {res_ssm.llf:.4f}")
    print(f"  AIC:             {res_ssm.aic:.4f}")
    print(f"  BIC:             {res_ssm.bic:.4f}")

    print("\nSSM 2-factor diagnostics:")
    print(f"  Log-likelihood:  {res_ssm_2f.llf:.4f}")
    print(f"  AIC:             {res_ssm_2f.aic:.4f}")
    print(f"  BIC:             {res_ssm_2f.bic:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
