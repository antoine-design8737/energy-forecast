"""
explore_data.py — Weekend 1: visualise the cleaned dataset.

Run AFTER fetch_data.py has produced the parquet.

Usage:
    python explore_data.py                          # auto-finds the first parquet
    python explore_data.py --file data/processed/FR_2022_2025_hourly.parquet

Produces 6 plots saved to data/plots/:
    1. raw_timeseries.png       — full load signal over time
    2. daily_cycle.png          — average hourly profile (weekday vs weekend)
    3. weekly_cycle.png         — average load by day of week
    4. seasonal_cycle.png       — monthly averages (summer trough, winter peaks)
    5. temperature_vs_load.png  — the U-shaped demand curve
    6. generation_mix.png       — stacked area of generation sources over a sample month
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PLOT_DIR = SCRIPT_DIR / "data" / "plots"

# --- Styling ----------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
COLORS = {
    "load":    "#2563EB",
    "weekend": "#F59E0B",
    "weekday": "#2563EB",
    "temp":    "#EF4444",
}


# --- Helpers ----------------------------------------------------------------

def find_parquet():
    candidates = sorted((SCRIPT_DIR / "data" / "processed").glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            "No parquet found in data/processed/. Run fetch_data.py first."
        )
    return candidates[0]


def load_data(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)

    # Convenience columns — added locally, not saved back to disk.
    local = df.index.tz_convert(
        "Europe/Paris" if "FR" in str(path) else "Europe/Berlin"
    )
    df["hour"]    = local.hour
    df["dayofweek"] = local.dayofweek   # 0 = Monday
    df["month"]   = local.month
    df["is_weekend"] = df["dayofweek"] >= 5
    return df


def save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / name
    fig.savefig(out, bbox_inches="tight")
    print(f"  saved -> {out}")
    plt.close(fig)


# --- Individual plots -------------------------------------------------------

def plot_raw_timeseries(df):
    """Full load signal — shows seasonality, trends, missing spikes."""
    fig, ax = plt.subplots(figsize=(14, 4))
    # Weekly resample to make the long view readable; daily would be noisy.
    weekly = df["load_mw"].resample("W").mean()
    ax.plot(weekly.index, weekly / 1e3, color=COLORS["load"], linewidth=1.2)
    ax.set(
        title="Electricity demand — weekly average",
        xlabel="Date",
        ylabel="Load (GW)",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    fig.autofmt_xdate()
    save(fig, "1_raw_timeseries.png")


def plot_daily_cycle(df):
    """Average load by hour of day, weekday vs weekend — the most important pattern."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for is_wknd, label, color in [
        (False, "Weekday", COLORS["weekday"]),
        (True,  "Weekend", COLORS["weekend"]),
    ]:
        subset = df.loc[~df["is_weekend"] if not is_wknd else df["is_weekend"]]
        profile = subset.groupby("hour")["load_mw"].mean() / 1e3
        ax.plot(profile.index, profile.values, label=label, color=color, linewidth=2.5)
    ax.set(
        title="Average daily load profile",
        xlabel="Hour of day (local time)",
        ylabel="Average load (GW)",
        xticks=range(0, 24, 2),
    )
    ax.legend()
    save(fig, "2_daily_cycle.png")


def plot_weekly_cycle(df):
    """Average load by day of week — shows the weekend drop."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day = df.groupby("dayofweek")["load_mw"].mean() / 1e3
    colors = [
        COLORS["weekend"] if d >= 5 else COLORS["weekday"]
        for d in by_day.index
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([day_names[d] for d in by_day.index], by_day.values, color=colors)
    ax.set(
        title="Average load by day of week",
        ylabel="Average load (GW)",
    )
    save(fig, "3_weekly_cycle.png")


def plot_seasonal_cycle(df):
    """Monthly average — reveals the winter peak and summer trough."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    by_month = df.groupby("month")["load_mw"].mean() / 1e3
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        [month_names[m - 1] for m in by_month.index],
        by_month.values,
        marker="o", linewidth=2.5, color=COLORS["load"],
    )
    ax.set(
        title="Seasonal demand pattern — monthly average",
        ylabel="Average load (GW)",
    )
    save(fig, "4_seasonal_cycle.png")


def plot_temperature_vs_load(df):
    """
    The classic U-shape: demand rises both in cold weather (heating) and hot
    weather (cooling). This is the most important physical insight for your
    feature engineering — a linear temperature feature won't capture this.
    """
    if "temp_c" not in df.columns:
        print("  skipping temperature plot — temp_c column not found")
        return

    # Bin by temperature and show median load per bin.
    df_copy = df[["temp_c", "load_mw"]].dropna()
    df_copy["temp_bin"] = pd.cut(df_copy["temp_c"], bins=30)
    binned = df_copy.groupby("temp_bin", observed=True)["load_mw"].median() / 1e3

    # Use bin midpoints as x-axis.
    mids = [iv.mid for iv in binned.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(mids, binned.values, color=COLORS["load"], s=40, alpha=0.8)
    ax.set(
        title="Temperature vs electricity demand (the U-shape)",
        xlabel="Temperature (°C)",
        ylabel="Median load (GW)",
    )
    # Annotate the insight.
    ax.annotate(
        "Cold → heating demand ↑",
        xy=(mids[2], binned.iloc[2]),
        xytext=(mids[2] + 4, binned.iloc[2] + 1),
        arrowprops=dict(arrowstyle="->", color="grey"),
        fontsize=10, color="grey",
    )
    ax.annotate(
        "Hot → cooling demand ↑",
        xy=(mids[-3], binned.iloc[-3]),
        xytext=(mids[-3] - 12, binned.iloc[-3] + 1),
        arrowprops=dict(arrowstyle="->", color="grey"),
        fontsize=10, color="grey",
    )
    save(fig, "5_temperature_vs_load.png")


def plot_generation_mix(df, n_weeks=4):
    """
    Stacked area of generation sources over a recent sample window.
    Shows how wind/solar intermittency interacts with dispatchable sources.
    """
    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    if not gen_cols:
        print("  skipping generation mix — no gen_ columns found")
        return

    # Take the last n_weeks of data for a readable window.
    sample = df[gen_cols].iloc[-(n_weeks * 7 * 24):].copy()
    # Drop columns that are mostly zero.
    sample = sample.loc[:, sample.mean() > sample.mean().max() * 0.01]
    # Friendly display names.
    sample.columns = [c.replace("gen_", "").replace("_", " ").title()
                      for c in sample.columns]
    sample = sample.clip(lower=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(
        sample.index,
        [sample[c] / 1e3 for c in sample.columns],
        labels=sample.columns,
        alpha=0.85,
    )
    ax.set(
        title=f"Generation mix — last {n_weeks} weeks",
        xlabel="Date",
        ylabel="Generation (GW)",
    )
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    save(fig, "6_generation_mix.png")


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None,
                        help="Path to the parquet (auto-detected if omitted).")
    args = parser.parse_args()

    path = args.file or find_parquet()
    print(f"Loading {path} ...")
    df = load_data(path)
    print(f"  {len(df):,} rows  |  columns: {list(df.columns)}\n")

    # Quick sanity summary printed to the terminal.
    print("=== Quick summary ===")
    print(f"  Date range : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"  Load range : {df['load_mw'].min():.0f} – {df['load_mw'].max():.0f} MW")
    if "temp_c" in df.columns:
        print(f"  Temperature: {df['temp_c'].min():.1f} – {df['temp_c'].max():.1f} °C")
    remaining_na = df["load_mw"].isna().sum()
    print(f"  Remaining NaN in load: {remaining_na}")
    print()

    print("Generating plots ...")
    plot_raw_timeseries(df)
    plot_daily_cycle(df)
    plot_weekly_cycle(df)
    plot_seasonal_cycle(df)
    plot_temperature_vs_load(df)
    plot_generation_mix(df)
    print(f"\nAll plots saved to {PLOT_DIR.resolve()}")


if __name__ == "__main__":
    main()
