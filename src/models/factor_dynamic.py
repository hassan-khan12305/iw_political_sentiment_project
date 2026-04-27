"""
Dynamic Factor Model — Univariate TVP Kalman Filter.

MODEL
-----
  Dynamic (TVP):      poll_t = λ_t · composite_t + ε_t   (time-varying λ)
                      λ_t = λ_{t-1} + ν_t

  poll_t      : Trump net favorability (FiveThirtyEight, standardised to N(0,1))
  composite_t : PLS1 Twitter composite of 6 standardised signals
  λ_t         : time-varying loading — how strongly the composite tracks polls
  σ_ν         : loading drift SD (estimated by MLE — how fast λ can change)
  σ_ε         : measurement noise SD (estimated by MLE — unexplained poll variance)


OUTPUTS
-------
  results/figures/fig_dyn_loading.png  — λ_t trajectory + 95% CI
  results/figures/fig_dyn_fit.png      — static vs dynamic poll tracking
  results/figures/fig_dyn_events.png   — event-window Δλ bar chart
  data/factor_dynamic_output.parquet   — λ_t, CIs, fits per week
  results/event_window_results.csv     — Δλ per event

Run:
    iw/bin/python src/models/factor_dynamic.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIG  = ROOT / "results" / "figures"
RES  = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)

SIGNAL_COLS = [
    "avg_sentiment_vader",
    "avg_sentiment_tweetnlp",
    "lean_score_hashtags",
    "trump_net_stance",
    "harris_net_stance",
    "log_tweet_count",
]

SIGNAL_LABELS = {
    "avg_sentiment_vader":    "VADER Sentiment",
    "avg_sentiment_tweetnlp": "RoBERTa Sentiment",
    "lean_score_hashtags":    "Hashtag Lean",
    "trump_net_stance":       "Trump Net Stance",
    "harris_net_stance":      "Harris Net Stance",
    "log_tweet_count":        "Log Volume",
}

# Key events for event-window analysis and plot annotations
EVENTS = [
    ("2024-W26", "1st Debate\n(Biden-Trump)\nJun 27"),
    ("2024-W29", "Biden exits\n+ RNC\nJul 15–21"),
    ("2024-W30", "Harris enters\n(struct. break)\nJul 22"),
    ("2024-W34", "DNC\nAug 19"),
    ("2024-W37", "2nd Debate\n(Trump-Harris)\nSep 10"),
    ("2024-W40", "VP Debate\nOct 1"),
    ("2024-W45", "Election Day\nNov 5"),
]

EVENT_COLORS = [
    "#2563EB", "#7C3AED", "#059669",
    "#D97706", "#DC2626", "#0891B2", "#DB2777",
]


# ---------------------------------------------------------------------------
# Univariate TVP Kalman Filter
# ---------------------------------------------------------------------------

class TVPKalmanFilter:
    """
    Scalar TVP Kalman filter for:
        poll_t = λ_t · x_t + ε_t,    ε_t ~ N(0, σ_ε²)
        λ_t    = λ_{t-1} + ν_t,       ν_t ~ N(0, σ_ν²)
    """

    def __init__(self, sigma_nu: float, sigma_eps: float) -> None:
        self.q = sigma_nu  ** 2   # state noise variance (loading drift)
        self.r = sigma_eps ** 2   # measurement noise variance

    def filter(self, poll: np.ndarray, x: np.ndarray,
               lam0: float = 0.0, p0: float = 1.0) -> "TVPKalmanFilter":
        """
        Forward Kalman filter pass.

        Parameters
        ----------
        poll  : (T,) observed poll series
        x     : (T,) PC1 composite series
        lam0  : initial loading (default: 0)
        p0    : initial loading variance (default: 1)
        """
        T = len(poll)
        lam_pred = np.zeros(T)
        p_pred   = np.zeros(T)
        lam_filt = np.zeros(T)
        p_filt   = np.zeros(T)
        ll = 0.0

        lam, p = lam0, p0
        for t in range(T):
            # Predict step
            lam_p = lam
            p_p   = p + self.q           # loading uncertainty grows

            # Innovation
            e_t = poll[t] - x[t] * lam_p
            s_t = x[t] ** 2 * p_p + self.r   # innovation variance

            # Log-likelihood contribution (Gaussian)
            s_safe = max(s_t, 1e-12)
            ll += -0.5 * (np.log(2 * np.pi) + np.log(s_safe) + e_t ** 2 / s_safe)

            # Update step (scalar Kalman gain)
            k_t  = p_p * x[t] / s_t
            lam  = lam_p + k_t * e_t
            p    = (1.0 - k_t * x[t]) * p_p   # Joseph-form equivalent (scalar)

            lam_pred[t] = lam_p;  p_pred[t] = p_p
            lam_filt[t] = lam;    p_filt[t] = p

        self.lam_pred = lam_pred
        self.p_pred   = p_pred
        self.lam_filt = lam_filt
        self.p_filt   = p_filt
        self.log_likelihood = ll
        return self

    def smooth(self) -> tuple[np.ndarray, np.ndarray]:
        """
        RTS (Rauch-Tung-Striebel) backward smoother.

        Uses all T observations to produce the best estimate of each λ_t.
        Returns (lam_smooth, p_smooth) — the posterior mean and variance
        of the loading at each week given the full campaign data.
        """
        T = len(self.lam_filt)
        lam_s = self.lam_filt.copy()
        p_s   = self.p_filt.copy()

        for t in range(T - 2, -1, -1):
            g        = self.p_filt[t] / self.p_pred[t + 1]   # smoother gain
            lam_s[t] = self.lam_filt[t] + g * (lam_s[t + 1] - self.lam_pred[t + 1])
            p_s[t]   = self.p_filt[t]   + g ** 2 * (p_s[t + 1] - self.p_pred[t + 1])

        return lam_s, p_s


# ---------------------------------------------------------------------------
# MLE for hyperparameters
# ---------------------------------------------------------------------------

def _neg_loglik(params_log: np.ndarray,
                poll: np.ndarray, x: np.ndarray, lam0: float) -> float:
    sigma_nu, sigma_eps = np.exp(params_log)
    kf = TVPKalmanFilter(sigma_nu, sigma_eps)
    kf.filter(poll, x, lam0=lam0)
    return -kf.log_likelihood


def estimate_params(poll: np.ndarray, x: np.ndarray,
                    lam0: float) -> tuple[float, float, float]:
    """
    MLE for (σ_ν, σ_ε) via L-BFGS-B on the Kalman log-likelihood.

    Parameters optimised in log-space to enforce positivity.
    Lower bound on σ_ε at 0.05: polls are standardised to N(0,1), so
    σ_ε < 0.05 is numerically degenerate (perfect fit with no noise).
    Seven restarts from dispersed starting points to avoid local optima.

    Returns (sigma_nu, sigma_eps, log_likelihood).
    """
    starts = [
        [np.log(0.05), np.log(0.30)],
        [np.log(0.10), np.log(0.50)],
        [np.log(0.20), np.log(0.50)],
        [np.log(0.30), np.log(0.30)],
        [np.log(0.05), np.log(0.80)],
        [np.log(0.15), np.log(0.70)],
        [np.log(0.40), np.log(0.40)],
    ]
    bounds = [(None, None), (np.log(0.05), None)]  # σ_ε ≥ 0.05

    best_ll, best_res = -np.inf, None
    restart_log = []
    for i, x0 in enumerate(starts):
        snu0, sep0 = np.exp(x0)
        try:
            res = minimize(
                _neg_loglik, x0, args=(poll, x, lam0),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-14},
            )
            snu, sep = np.exp(res.x)
            ll = -res.fun
            restart_log.append((i+1, snu0, sep0, snu, sep, ll, res.success))
            if ll > best_ll:
                best_ll, best_res = ll, res
        except Exception:
            restart_log.append((i+1, snu0, sep0, None, None, None, False))
            continue

    print("\n  MLE Restart Log:")
    print(f"  {'#':>2}  {'σ_ν init':>8}  {'σ_ε init':>8}  {'σ_ν opt':>8}  {'σ_ε opt':>8}  {'log-lik':>10}  {'converged':>9}")
    print("  " + "-" * 65)
    for row in restart_log:
        i, snu0, sep0, snu, sep, ll, ok = row
        if snu is not None:
            marker = " ← best" if abs(ll - best_ll) < 1e-6 else ""
            print(f"  {i:>2}  {snu0:>8.4f}  {sep0:>8.4f}  {snu:>8.4f}  {sep:>8.4f}  {ll:>10.4f}  {str(ok):>9}{marker}")
        else:
            print(f"  {i:>2}  {snu0:>8.4f}  {sep0:>8.4f}  {'FAILED':>8}  {'':>8}  {'':>10}  {'False':>9}")

    sigma_nu, sigma_eps = np.exp(best_res.x)
    return float(sigma_nu), float(sigma_eps), float(best_ll)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def event_window_analysis(lam_smooth: np.ndarray,
                           weeks: pd.Series,
                           window: int = 2) -> pd.DataFrame:
    """
    For each key event, compare mean λ_t in the ±2-week window against
    the complement (all other weeks). Δλ = window mean − complement mean.

    Positive Δλ: composite more informative around the event.
    Negative Δλ: composite decoupled from polls around the event.
    """
    week_list = weeks.tolist()
    rows = []
    for week_label, desc in EVENTS:
        if week_label not in week_list:
            continue
        idx     = week_list.index(week_label)
        win_idx = [i for i in range(max(0, idx - window),
                                    min(len(lam_smooth), idx + window + 1))]
        cmp_idx = [i for i in range(len(lam_smooth)) if i not in win_idx]
        rows.append({
            "event":      desc.replace("\n", " "),
            "week":       week_label,
            "lam_window": round(float(lam_smooth[win_idx].mean()), 4),
            "lam_compl":  round(float(lam_smooth[cmp_idx].mean()), 4) if cmp_idx else np.nan,
            "delta_lam":  round(float(lam_smooth[win_idx].mean() -
                                       lam_smooth[cmp_idx].mean()), 4) if cmp_idx else np.nan,
        })
    return pd.DataFrame(rows)


def structural_break(lam_smooth: np.ndarray, weeks: pd.Series) -> dict:
    """
    Compare mean λ before and after W30 (Biden->Harris, 2024-07-22).
    This is the primary structural break in the campaign.
    """
    week_list = weeks.tolist()
    if "2024-W30" not in week_list:
        return {}
    idx  = week_list.index("2024-W30")
    pre  = float(lam_smooth[:idx].mean())
    post = float(lam_smooth[idx:].mean())
    return {"lam_pre": pre, "lam_post": post, "delta_lam": post - pre,
            "break_week": "2024-W30", "n_pre": idx, "n_post": len(lam_smooth) - idx}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _add_event_lines(ax, dates, weeks, ylim_top, ylim_bot=None):
    """Add vertical event lines with staggered labels to a time-series axis."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    y_fracs = [0.97, 0.72, 0.50, 0.28]
    week_list = weeks.tolist()
    visible = [(wl, d, c) for (wl, d), c in zip(EVENTS, EVENT_COLORS)
               if wl in week_list]
    for i, (week_label, desc, color) in enumerate(visible):
        idx = week_list.index(week_label)
        ax.axvline(dates[idx], color=color, linewidth=1.0,
                   linestyle=":", alpha=0.75)
        y_pos = ymax - (1 - y_fracs[i % len(y_fracs)]) * span
        ax.text(dates[idx], y_pos, desc.split("\n")[0],
                rotation=90, ha="right", va="top",
                fontsize=6.5, color=color, alpha=0.85)


def _dual_xaxis_labels(ax, dates, weeks):
    """Week numbers as axis tick labels, month/year as a second row below."""
    from matplotlib.transforms import blended_transform_factory

    # Week tick labels every 2 weeks
    tick_dates, tick_labels = [], []
    for d, w in zip(dates, weeks.tolist()):
        wnum = int(w.split('W')[1])
        if wnum % 2 == 0:
            tick_dates.append(d)
            tick_labels.append(f'W{wnum}')
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(tick_labels, fontsize=7.5, color='#374151')
    ax.tick_params(axis='x', length=4, pad=3)

    # Month/year labels as second row below week numbers
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    month_starts = pd.date_range(dates[0], dates[-1], freq='MS')
    for d in month_starts:
        ax.text(d, -0.13, d.strftime("%b '%y"), transform=trans,
                fontsize=8.5, ha='center', va='top', color='#111827')


def plot_loading(dates, lam_smooth, ci95, beta_ols, weeks):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(dates, lam_smooth - ci95, lam_smooth + ci95,
                    alpha=0.18, color="#2563EB", label="95% CI")
    ax.plot(dates, lam_smooth, color="#2563EB", linewidth=2.5,
            label=r"TVP loading $\lambda_t$ (smoothed)")
    ax.axhline(beta_ols, color="#DC2626", linestyle="--", linewidth=1.6,
               label=f"Static OLS  λ = {beta_ols:.3f}")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.35)

    ylim = ax.get_ylim()
    _add_event_lines(ax, dates, weeks, ylim[1] * 0.97)

    # W30 structural break 
    if "2024-W30" in weeks.tolist():
        idx30 = weeks.tolist().index("2024-W30")
        ax.axvline(dates[idx30], color="black", linewidth=1.8,
                   linestyle="--", alpha=0.6, label="W30: Harris enters")

    ax.set_title(
        r"Time-Varying Loading of Twitter Composite Index on Trump Net Favorability"
        "\n" + r"$\lambda_t$: how strongly the PLS1 composite tracked polls each week",
        fontsize=11, pad=10)
    ax.set_xlabel("Week")
    ax.set_ylabel(r"$\lambda_t$  (loading)")
    ax.legend(loc="lower left", fontsize=8)
    _dual_xaxis_labels(ax, dates, weeks)
    ax.set_xlabel("")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(FIG / "fig_dyn_loading.png", dpi=150)
    plt.close()
    print("  Saved -> results/figures/fig_dyn_loading.png")


def plot_fit(dates, poll, poll_hat_static, poll_hat_dynamic, weeks):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: actual vs fitted
    ax = axes[0]
    ax.plot(dates, poll, "k-", linewidth=1.6, label="Trump polls (standardised)", alpha=0.9)
    ax.plot(dates, poll_hat_static,  "--", color="#DC2626", linewidth=1.8,
            label="Static OLS fit")
    ax.plot(dates, poll_hat_dynamic, "-",  color="#2563EB", linewidth=1.8,
            label="Dynamic TVP fit")
    ylim = ax.get_ylim()
    _add_event_lines(ax, dates, weeks, ylim[1] * 0.97)
    if "2024-W30" in weeks.tolist():
        idx30 = weeks.tolist().index("2024-W30")
        ax.axvline(dates[idx30], color="black", linewidth=1.6,
                   linestyle="--", alpha=0.55, label="W30: Harris enters")
    ax.set_ylabel("Standardised value")
    ax.legend(fontsize=8)
    ax.set_title("Poll Tracking: Static vs Dynamic Twitter Composite Index", fontsize=11)

    # Bottom panel: residuals
    ax2 = axes[1]
    ax2.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax2.plot(dates, poll - poll_hat_static,  "--", color="#DC2626",
             linewidth=1.3, alpha=0.8, label="Static residuals")
    ax2.plot(dates, poll - poll_hat_dynamic, "-",  color="#2563EB",
             linewidth=1.3, alpha=0.8, label="Dynamic residuals")
    ax2.fill_between(dates, poll - poll_hat_dynamic, 0,
                     alpha=0.10, color="#2563EB")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Week")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("")
    _dual_xaxis_labels(ax2, dates, weeks)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(FIG / "fig_dyn_fit.png", dpi=150)
    plt.close()
    print("  Saved -> results/figures/fig_dyn_fit.png")


def plot_events(ew: pd.DataFrame):
    df = ew.dropna(subset=["delta_lam"]).sort_values("delta_lam")
    colors = ["#DC2626" if d < 0 else "#2563EB" for d in df["delta_lam"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df["event"], df["delta_lam"], color=colors, alpha=0.82)
    ax.axvline(0, color="black", linewidth=0.9)

    for bar, val in zip(bars, df["delta_lam"]):
        offset = 0.01 if val >= 0 else -0.01
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=9)

    # Ensure labels outside bars aren't clipped
    vals = df["delta_lam"].values
    ax.set_xlim(vals.min() - 0.12, vals.max() + 0.12)

    ax.set_xlabel(r"$\Delta\lambda$  (loading in ±2-week window − complement mean)",
                  fontsize=10)
    ax.set_title(
        "Event-Window Loading Shifts\n"
        r"$\Delta\lambda > 0$: composite more informative  |  "
        r"$\Delta\lambda < 0$: composite decoupled from polls",
        fontsize=10, pad=10)
    plt.tight_layout()
    fig.savefig(FIG / "fig_dyn_events.png", dpi=150)
    plt.close()
    print("  Saved -> results/figures/fig_dyn_events.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DYNAMIC FACTOR MODEL — Univariate TVP Kalman Filter")
    print("poll_t = λ_t · composite_t + ε_t")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    obs   = pd.read_parquet(DATA / "observation_matrix.parquet")
    polls = pd.read_parquet(DATA / "polls_weekly.parquet")

    df = obs.merge(polls[["year_week", "poll_trump_net"]], on="year_week")
    df = df.sort_values("year_week").reset_index(drop=True)
    weeks = df["year_week"]
    T = len(df)
    print(f"  Weeks: {T}  |  {weeks.iloc[0]} – {weeks.iloc[-1]}")

    # ── Standardise signals ────────────────────────────────────────────────
    print("\n[2/4] Computing PLS1 composite index...")
    X = df[SIGNAL_COLS].values.astype(float)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    # Standardise Trump polls to N(0,1)
    poll_raw = df["poll_trump_net"].values.astype(float)
    poll     = (poll_raw - poll_raw.mean()) / poll_raw.std(ddof=1)

    # ── PLS1 composite ────────────────────────────────────────────────────
    #   direction = X_std.T @ poll  (cross-covariance vector, (6,))
    #   composite = X_std @ (direction / ||direction||)  (T,)
    direction     = X_std.T @ poll                             # (6,)
    direction_norm = direction / np.linalg.norm(direction)     # unit vector
    composite_raw  = X_std @ direction_norm                    # (T,)
    composite      = (composite_raw - composite_raw.mean()) / composite_raw.std(ddof=1)

    # PLS1 weights (signal contributions to the composite)
    pls1_weights = direction_norm

    # PC1 variance explained (for comparison / reporting)
    _, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    var_explained_pc1 = float((S[0] ** 2) / (S ** 2).sum())

    r_composite_poll = float(np.corrcoef(composite, poll)[0, 1])
    print(f"  r(PLS1 composite, Trump polls): {r_composite_poll:.3f}")
    print(f"  PC1 variance explained (ref):   {var_explained_pc1:.1%}")
    print(f"  PLS1 signal weights (composite direction):")
    for col, w in zip(SIGNAL_COLS, pls1_weights):
        print(f"    {SIGNAL_LABELS[col]:<25} {w:+.4f}")

    # ── Static OLS baseline ────────────────────────────────────────────────
    print("\n[3/4] Fitting models...")
    beta_ols        = float(np.dot(composite, poll) / np.dot(composite, composite))
    poll_hat_static = beta_ols * composite
    ss_tot          = float(np.sum((poll - poll.mean()) ** 2))
    ss_res_static   = float(np.sum((poll - poll_hat_static) ** 2))
    r2_static       = 1.0 - ss_res_static / ss_tot
    rmse_static     = float(np.sqrt(ss_res_static / T))
    print(f"  Static OLS:   R² = {r2_static:.4f}  RMSE = {rmse_static:.4f}  β = {beta_ols:.4f}")

    # Static log-likelihood (σ_ε = OLS RMSE, no drift)
    kf0 = TVPKalmanFilter(0.0, rmse_static)
    kf0.filter(poll, composite, lam0=beta_ols)
    ll_static = kf0.log_likelihood

    # ── TVP Kalman (MLE + smoother) ────────────────────────────────────────
    sigma_nu, sigma_eps, ll_dyn = estimate_params(poll, composite, lam0=beta_ols)
    print(f"  TVP MLE:      σ_ν = {sigma_nu:.4f}  σ_ε = {sigma_eps:.4f}  LL = {ll_dyn:.3f}")

    kf = TVPKalmanFilter(sigma_nu, sigma_eps)
    kf.filter(poll, composite, lam0=beta_ols)
    lam_smooth, p_smooth = kf.smooth()
    ci95 = 1.96 * np.sqrt(np.maximum(p_smooth, 0.0))

    poll_hat_dynamic = lam_smooth * composite
    ss_res_dynamic   = float(np.sum((poll - poll_hat_dynamic) ** 2))
    r2_dynamic       = 1.0 - ss_res_dynamic / ss_tot
    rmse_dynamic     = float(np.sqrt(ss_res_dynamic / T))
    lr_stat          = max(0.0, 2.0 * (ll_dyn - ll_static))

    delta_r2 = r2_dynamic - r2_static
    label    = "DYNAMIC BETTER ✓" if delta_r2 > 0 else "STATIC BETTER"
    print(f"  Dynamic TVP:  R² = {r2_dynamic:.4f}  RMSE = {rmse_dynamic:.4f}")
    print(f"  ΔR² = {delta_r2:+.4f}  ({label})")
    print(f"  LR statistic = {lr_stat:.3f}  (boundary χ²(1))")

    # ── Structural break ───────────────────────────────────────────────────
    brk = structural_break(lam_smooth, weeks)
    print(f"\n  Structural break W30 (Biden->Harris):")
    print(f"    Pre-W30  mean λ = {brk['lam_pre']:+.4f}")
    print(f"    Post-W30 mean λ = {brk['lam_post']:+.4f}")
    print(f"    Δλ              = {brk['delta_lam']:+.4f}")

    # ── Event-window analysis ──────────────────────────────────────────────
    ew = event_window_analysis(lam_smooth, weeks)
    print(f"\n  Event-window Δλ (±2 weeks):")
    for _, row in ew.iterrows():
        bar = "█" * max(1, int(abs(row["delta_lam"]) * 15))
        sgn = "+" if row["delta_lam"] > 0 else ""
        print(f"    {row['event'][:38]:<38}  {sgn}{row['delta_lam']:.4f}  {bar}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")
    dates = pd.to_datetime([f"{w}-1" for w in weeks], format="%Y-W%W-%w")
    plot_loading(dates, lam_smooth, ci95, beta_ols, weeks)
    plot_fit(dates, poll, poll_hat_static, poll_hat_dynamic, weeks)
    plot_events(ew)

    # ── Save outputs ───────────────────────────────────────────────────────
    out = pd.DataFrame({
        "week":             weeks.values,
        "date":             dates.values,
        "composite":        composite,
        "poll_trump_std":   poll,
        "lam_smooth":       lam_smooth,
        "lam_ci_lower":     lam_smooth - ci95,
        "lam_ci_upper":     lam_smooth + ci95,
        "poll_hat_static":  poll_hat_static,
        "poll_hat_dynamic": poll_hat_dynamic,
        "residual_static":  poll - poll_hat_static,
        "residual_dynamic": poll - poll_hat_dynamic,
    })
    out.to_parquet(DATA / "factor_dynamic_output.parquet", index=False)
    ew.to_csv(RES / "event_window_results.csv", index=False)
    print("  Saved -> data/factor_dynamic_output.parquet")
    print("  Saved -> results/event_window_results.csv")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print(f"  Static  R² = {r2_static:.4f}  RMSE = {rmse_static:.4f}")
    print(f"  Dynamic R² = {r2_dynamic:.4f}  RMSE = {rmse_dynamic:.4f}")
    print(f"  ΔR²        = {delta_r2:+.4f}")
    print(f"  σ_ν        = {sigma_nu:.4f}  (loading drift per week)")
    print(f"  σ_ε        = {sigma_eps:.4f}  (measurement noise)")
    print(f"  LR stat    = {lr_stat:.3f}")
    print(f"  Structural break Δλ at W30: {brk['delta_lam']:+.4f}")


if __name__ == "__main__":
    main()
