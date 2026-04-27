"""
Polling data aggregation — weekly net favorability series.

Reads raw FiveThirtyEight favorability_polls.csv and aggregates to a clean
weekly series aligned with the W18–W48 2024 tweet window.

Aggregation:
  - National polls only (state == NaN/'')
  - Weighted mean by sample_size (polls with larger N count more)
  - Net favorability = favorable − unfavorable  (negative = underwater)
  - Week assigned by poll end_date (ISO week)

Writes:
  data/polls_weekly.parquet   — 31 rows × columns below
  results/figures/fig_polls.png — validation plot

Columns:
  year_week          — ISO week string e.g. '2024-W30'
  week_start         — Monday of that week
  poll_trump_net     — Trump net favorability
  poll_harris_net    — Harris net favorability
  poll_biden_net     — Biden net favorability
  poll_margin        — Trump net − Harris net
  poll_trump_n       — number of polls aggregated that week
  poll_harris_n      — number of polls aggregated that week
  poll_biden_n       — number of polls aggregated that week
  interpolated       — True if any value was gap-filled by interpolation

Run:
    iw/bin/python src/models/prepare_polls.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLLS_CSV   = Path("data/polls/favorability_polls.csv")
WEEKLY_PATH = Path("data/usc_2024_weekly.parquet")
OUT_PARQUET = Path("data/polls_weekly.parquet")
FIG_DIR     = Path("results/figures")

CANDIDATES = {
    "Donald Trump":  "trump",
    "Kamala Harris": "harris",
    "Joe Biden":     "biden",
}

# ISO weeks we need — must match the tweet window exactly
W18 = "2024-W18"
W48 = "2024-W48"

EVENTS = {
    "2024-W26": "1st Debate (Biden-Trump)",
    "2024-W29": "Biden exits + RNC",
    "2024-W30": "Harris enters",
    "2024-W34": "DNC",
    "2024-W37": "2nd Debate (Trump-Harris)",
    "2024-W40": "VP Debate",
    "2024-W45": "Election Day",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       10,
})


# ---------------------------------------------------------------------------
# Load + aggregate
# ---------------------------------------------------------------------------

def _week_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s + "-1", format="%G-W%V-%u")


def load_and_aggregate() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per ISO week in W18–W48, with
    weighted-mean net favorability for Trump, Harris, Biden.
    """
    raw = pd.read_csv(POLLS_CSV, low_memory=False)
    raw["end_date"] = pd.to_datetime(raw["end_date"])

    # National polls only
    raw = raw[raw["state"].isna() | (raw["state"] == "")].copy()

    # Date window (give a 7-day buffer on each end for edge weeks)
    raw = raw[
        (raw["end_date"] >= "2024-04-22") &
        (raw["end_date"] <= "2024-12-07")
    ].copy()

    raw["net_fav"]   = raw["favorable"] - raw["unfavorable"]
    raw["year_week"] = raw["end_date"].dt.strftime("%G-W%V")

    # Build the target week spine from the tweet parquet
    spine = pd.read_parquet(WEEKLY_PATH, columns=["year_week"])
    spine["week_start"] = _week_to_date(spine["year_week"])
    spine = spine.sort_values("week_start").reset_index(drop=True)

    result = spine.copy()

    for full_name, short in CANDIDATES.items():
        sub = raw[raw["politician"] == full_name].copy()

        def _wagg(g: pd.DataFrame) -> pd.Series:
            w   = g["sample_size"].fillna(1).clip(lower=1)
            net = np.average(g["net_fav"].fillna(g["net_fav"].median()), weights=w)
            return pd.Series({"net": net, "n_polls": len(g)})

        agg = (
            sub.groupby("year_week")
               .apply(_wagg)
               .reset_index()
        )
        agg.columns = ["year_week", f"poll_{short}_net", f"poll_{short}_n"]

        result = result.merge(agg, on="year_week", how="left")

    return result


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear interpolation for missing weeks, flagged in 'interpolated' column.
    Also forward/back fills the very first/last week if needed.
    """
    net_cols = [c for c in df.columns if c.startswith("poll_") and c.endswith("_net")]
    df["interpolated"] = df[net_cols].isna().any(axis=1)

    for col in net_cols:
        df[col] = (
            df[col]
            .interpolate(method="linear", limit_direction="both")
        )

    return df


# ---------------------------------------------------------------------------
# Derived series
# ---------------------------------------------------------------------------

def add_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    poll_margin = Trump net − Harris net.
    Before W30: use Trump − Biden (Harris not yet nominee).
    After  W30: use Trump − Harris.
    This gives a continuous partisan favorability gap series.
    """
    # Unified margin: Trump vs the active Democratic candidate
    break_idx = df[df["year_week"] == "2024-W30"].index[0]  # first full Harris week

    margin = np.where(
        df.index < break_idx,
        df["poll_trump_net"] - df["poll_biden_net"],
        df["poll_trump_net"] - df["poll_harris_net"],
    )
    df["poll_margin"] = margin
    return df


# ---------------------------------------------------------------------------
# Validation plot
# ---------------------------------------------------------------------------

def plot_polls(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = df["week_start"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={"hspace": 0.42})

    # ── Panel 1: individual net favorability ─────────────────────────────
    ax1.plot(x, df["poll_trump_net"],  lw=2, color="#DC2626",
             marker="o", markersize=3.5, label="Trump net fav")
    ax1.plot(x, df["poll_harris_net"], lw=2, color="#2563EB",
             marker="o", markersize=3.5, label="Harris net fav")
    ax1.plot(x, df["poll_biden_net"],  lw=1.5, color="#6B7280", ls="--",
             marker="s", markersize=2.5, label="Biden net fav")

    # Mark interpolated weeks
    interp_weeks = df[df["interpolated"]]["week_start"]
    for iw in interp_weeks:
        ax1.axvspan(iw - pd.Timedelta(days=3),
                    iw + pd.Timedelta(days=3),
                    alpha=0.15, color="#F59E0B", zorder=0)

    ax1.axhline(0, color="#9CA3AF", lw=1)
    ax1.set_ylabel("Net favorability (fav − unfav, pp)", fontsize=10)
    ax1.set_title("Candidate Net Favorability — Weekly Weighted Average\n"
                  "(yellow bands = interpolated weeks)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, frameon=False)

    _add_events(ax1, df)
    _month_fmt(ax1)

    # ── Panel 2: partisan margin ─────────────────────────────────────────
    colors_margin = ["#DC2626" if v > 0 else "#2563EB"
                     for v in df["poll_margin"]]
    ax2.bar(x, df["poll_margin"], color=colors_margin, alpha=0.75, width=5)
    ax2.axhline(0, color="#9CA3AF", lw=1)

    # Biden->Harris transition marker
    brk = df[df["year_week"] == "2024-W30"]["week_start"].iloc[0]
    ax2.axvline(brk, color="#D97706", lw=1.5, ls="--")
    ax2.text(brk, ax2.get_ylim()[1] * 0.9, "Biden exits\n(margin switches\nto Trump−Harris)",
             fontsize=7.5, color="#D97706", ha="center")

    ax2.set_ylabel("Partisan margin (Trump − Dem nominee, pp)", fontsize=10)
    ax2.set_title("Partisan Favorability Margin — Primary Reference Series\n"
                  "Positive = Trump ahead; Negative = Dem nominee ahead",
                  fontsize=11, fontweight="bold")
    _month_fmt(ax2)

    fig.suptitle("Weekly Polling Aggregation — W18–W48 2024",
                 fontsize=13, fontweight="bold")

    p = FIG_DIR / "fig_polls.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")


def _month_fmt(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=20, labelsize=8)


def _add_events(ax, df: pd.DataFrame) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for i, (week, label) in enumerate(EVENTS.items()):
        row = df[df["year_week"] == week]
        if row.empty:
            continue
        xv = row["week_start"].iloc[0]
        ax.axvline(xv, color="#9CA3AF", lw=1, ls="--")
        y  = ymax - 0.05 * span if i % 2 == 0 else ymin + 0.05 * span
        va = "top" if i % 2 == 0 else "bottom"
        ax.text(xv, y, label, fontsize=7, color="#4B5563",
                ha="center", va=va)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("POLLING DATA — Weekly aggregation")
    print("=" * 55)

    print("\n[1/3] Loading and aggregating raw polls...")
    df = load_and_aggregate()

    print("\n[2/3] Filling gaps + computing margin...")
    df = fill_gaps(df)
    df = add_margin(df)

    # Diagnostics
    interp_n = df["interpolated"].sum()
    print(f"  Weeks: {len(df)}  |  Interpolated: {interp_n}")
    print(f"  Trump  net fav: {df['poll_trump_net'].mean():.2f} ± {df['poll_trump_net'].std():.2f}")
    print(f"  Harris net fav: {df['poll_harris_net'].mean():.2f} ± {df['poll_harris_net'].std():.2f}")
    print(f"  Biden  net fav: {df['poll_biden_net'].mean():.2f} ± {df['poll_biden_net'].std():.2f}")
    print(f"  Margin range:   [{df['poll_margin'].min():.2f}, {df['poll_margin'].max():.2f}]")

    print("\n  Weekly margin (Trump − Dem nominee):")
    for _, row in df.iterrows():
        interp_flag = " [interp]" if row["interpolated"] else ""
        print(f"    {row['year_week']}  {row['poll_margin']:+.2f}{interp_flag}")

    print("\n[3/3] Saving output...")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"  Saved -> {OUT_PARQUET}  ({df.shape})")

    plot_polls(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
