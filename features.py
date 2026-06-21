"""
features.py — Weekend 2: baselines + feature engineering.

Run AFTER fetch_data.py has produced the parquet.

What this script does:
    1. Loads the cleaned hourly dataset.
    2. Defines a leakage-free time-based train/test split.
    3. Computes two honest baselines on the test set.
    4. Engineers the full feature matrix (calendar, lags, rolling stats,
       weather, weather interactions).
    5. Saves the feature matrix + target to data/processed/features.parquet.

Usage:
    python features.py
    python features.py --file data/processed/DE_LU_2022_2025_hourly.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def find_parquet():
    candidates = sorted(PROCESSED_DIR.glob("*.parquet"))
    candidates = [c for c in candidates if "features" not in c.name]
    if not candidates:
        raise FileNotFoundError("No parquet found. Run fetch_data.py first.")
    return candidates[0]


def load(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    print(f"Loaded {len(df):,} rows  ({df.index.min().date()} → {df.index.max().date()})")
    return df


# ---------------------------------------------------------------------------
# 2. Train / test split  ← THE MOST IMPORTANT PART OF THIS WEEKEND
# ---------------------------------------------------------------------------
# We split by time, never by shuffling. Shuffling causes leakage: future data
# bleeds into training, evaluation looks great, model fails in production.
# Rule: last TEST_DAYS days are held out, everything before is training.
# We also define a validation window (the TEST_DAYS block just before the test
# set) for hyperparameter tuning in Weekend 3 without ever touching the test set.

TEST_DAYS = 90      # ~3 months of unseen future
VAL_DAYS  = 90      # validation window for tuning


def make_split(df):
    last  = df.index.max()
    test_start = last - pd.Timedelta(days=TEST_DAYS)
    val_start  = test_start - pd.Timedelta(days=VAL_DAYS)

    train = df[df.index <  val_start]
    val   = df[(df.index >= val_start) & (df.index < test_start)]
    test  = df[df.index >= test_start]

    print(f"\nSplit (no leakage):")
    print(f"  Train : {train.index.min().date()} → {train.index.max().date()}  ({len(train):,} rows)")
    print(f"  Val   : {val.index.min().date()}   → {val.index.max().date()}   ({len(val):,} rows)")
    print(f"  Test  : {test.index.min().date()}  → {test.index.max().date()}  ({len(test):,} rows)")
    return train, val, test


# ---------------------------------------------------------------------------
# 3. Baselines
# ---------------------------------------------------------------------------
# We measure these on the test set BEFORE building any model. If your model
# can't beat them, it isn't useful. Always show baseline numbers alongside
# model numbers — this is the single most common thing missing from student
# portfolios, and interviewers notice.

def mape(actual, predicted):
    mask = actual > 0
    return 100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def compute_baselines(df, test):
    """
    Baseline 1 — Seasonal naive (same hour, one week ago):
        "Tomorrow 9am will look like last Tuesday 9am."
        Strong because it captures both the daily cycle and the weekly cycle.

    Baseline 2 — Yesterday naive (same hour, one day ago):
        "Tomorrow 9am will look like today 9am."
        Good on weekdays, poor over weekends.

    Both are computed by looking backwards in the *full* df (so the week-ago
    value is always available), then evaluating only on test rows.
    """
    actual = test["load_mw"]

    # Seasonal naive: 168 hours = 1 week
    seasonal_pred = df["load_mw"].shift(24 * 7).reindex(test.index)
    # Yesterday naive: 24 hours = 1 day
    yesterday_pred = df["load_mw"].shift(24).reindex(test.index)

    mask = actual.notna() & seasonal_pred.notna() & yesterday_pred.notna()
    a = actual[mask]
    s = seasonal_pred[mask]
    y = yesterday_pred[mask]

    print("\n=== Baselines (test set) ===")
    print(f"  Seasonal naive  (same hour last week)  — MAPE: {mape(a, s):.2f}%  MAE: {mae(a, s)/1e3:.1f} GW  RMSE: {rmse(a, s)/1e3:.1f} GW")
    print(f"  Yesterday naive (same hour yesterday)  — MAPE: {mape(a, y):.2f}%  MAE: {mae(a, y)/1e3:.1f} GW  RMSE: {rmse(a, y)/1e3:.1f} GW")
    print()
    print("  ↑ Your model in Weekend 3 needs to beat these numbers.")
    print("  ↑ Save them — they go in the README next to your model's numbers.")

    return {
        "seasonal_naive_mape": mape(a, s),
        "seasonal_naive_mae":  mae(a, s),
        "yesterday_naive_mape": mape(a, y),
        "yesterday_naive_mae":  mae(a, y),
    }


# ---------------------------------------------------------------------------
# 4. Feature engineering
# ---------------------------------------------------------------------------
# All features are built from information available at forecast time (i.e.
# things you would actually know at 08:00 when forecasting the next 24 hours).
# Anything derived from the target (load_mw) uses a lag >= 24h so it could
# never contain tomorrow's value.

def add_calendar_features(df, tz):
    local = df.index.tz_convert(tz)
    df["hour"]       = local.hour
    df["dayofweek"]  = local.dayofweek          # 0 = Monday
    df["month"]      = local.month
    df["quarter"]    = local.quarter
    df["is_weekend"] = (local.dayofweek >= 5).astype(int)
    # Cyclical encoding: avoids the artificial distance between hour 23 and hour 0
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_holiday_features(df, country_code):
    try:
        import holidays as hol_lib
        # Map our ENTSO-E codes to the holidays library's country codes
        country_map = {"FR": "FR", "DE_LU": "DE"}
        cc = country_map.get(country_code, country_code[:2])
        country_holidays = hol_lib.country_holidays(cc)
        local_dates = df.index.tz_convert("Europe/Paris" if cc == "FR" else "Europe/Berlin").date
        df["is_holiday"] = [int(d in country_holidays) for d in local_dates]
        # Day before and after a holiday behave differently too
        df["is_holiday_eve"] = df["is_holiday"].shift(-24, fill_value=0)
    except ImportError:
        print("  [holidays package not installed — skipping holiday features]")
        print("  Run: pip install holidays")
        df["is_holiday"] = 0
        df["is_holiday_eve"] = 0
    return df


def add_lag_features(df):
    load = df["load_mw"]
    # Lag features: all >= 24h so we never leak the future
    # These are the most predictive features for load forecasting
    for lag_h in [24, 48, 168]:       # 1 day, 2 days, 1 week
        df[f"load_lag_{lag_h}h"] = load.shift(lag_h)
    # Rolling statistics (computed on past data only — shift(24) means the
    # window ends at least 24h before the current row)
    load_shifted = load.shift(24)
    df["load_rolling_mean_24h"]  = load_shifted.rolling(24).mean()
    df["load_rolling_mean_168h"] = load_shifted.rolling(168).mean()
    df["load_rolling_std_24h"]   = load_shifted.rolling(24).std()
    # Same-hour last week: powerful because it captures the weekly + daily
    # cycle simultaneously, but it's already a separate baseline — here it
    # also goes in as a feature so the model can learn to correct it
    df["load_same_hour_last_week"] = load.shift(168)
    return df


def add_weather_features(df):
    if "temp_c" not in df.columns:
        print("  [no temp_c column — skipping weather features]")
        return df

    # Raw weather
    df["temp_c_sq"]    = df["temp_c"] ** 2          # captures the U-shape
    df["wind_ms"]      = df.get("wind_ms", 0)
    df["solar_rad"]    = df.get("solar_rad", 0)

    # Heating / cooling degree concept (thresholds are approximate for comfort)
    df["heating_deg"]  = (18 - df["temp_c"]).clip(lower=0)
    df["cooling_deg"]  = (df["temp_c"] - 22).clip(lower=0)

    # Lagged weather: the forecast model will use tomorrow's weather forecast
    # as an input in production; here we use actual weather lagged by ~0h
    # (we treat it as "known" since weather forecasts 24h ahead are very good)
    # Rolling temperature: persistence effect (today feels cold if yesterday was)
    df["temp_rolling_24h"] = df["temp_c"].rolling(24).mean()

    return df


def build_features(df, country_code):
    tz = "Europe/Paris" if country_code == "FR" else "Europe/Berlin"
    df = df.copy()
    df = add_calendar_features(df, tz)
    df = add_holiday_features(df, country_code)
    df = add_lag_features(df)
    df = add_weather_features(df)
    return df


# ---------------------------------------------------------------------------
# 5. Final clean-up and save
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Calendar
    "hour", "dayofweek", "month", "quarter", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    # Holidays
    "is_holiday", "is_holiday_eve",
    # Load lags
    "load_lag_24h", "load_lag_48h", "load_lag_168h",
    "load_rolling_mean_24h", "load_rolling_mean_168h", "load_rolling_std_24h",
    "load_same_hour_last_week",
    # Weather
    "temp_c", "temp_c_sq", "wind_ms", "solar_rad",
    "heating_deg", "cooling_deg", "temp_rolling_24h",
]

TARGET_COL = "load_mw"


def save_features(df, src_path):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"\n  [Note] Columns not found, skipped: {missing}")

    out = df[available + [TARGET_COL]].copy()

    # Drop rows where any lag feature is NaN (the first 168 hours after start)
    # and rows where the target is NaN.
    before = len(out)
    out = out.dropna(subset=["load_lag_168h", TARGET_COL])
    print(f"\n  Dropped {before - len(out)} warm-up rows (first ~1 week of lags)")
    print(f"  Final feature matrix: {len(out):,} rows × {len(out.columns)} columns")

    out_path = src_path.parent / "features.parquet"
    out.to_parquet(out_path)
    print(f"  Saved -> {out_path}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None)
    args = parser.parse_args()

    src_path = Path(args.file) if args.file else find_parquet()
    df = load(src_path)

    # Infer country code from filename (e.g. "FR_2022_2025_hourly.parquet")
    country_code = src_path.stem.split("_")[0]
    if country_code not in ("FR", "DE_LU"):
        country_code = "FR"
        print(f"  Could not infer country from filename, defaulting to FR")

    # Split — print the ranges so you can verify no leakage visually
    train, val, test = make_split(df)

    # Baselines — measure these before touching the model
    baselines = compute_baselines(df, test)

    # Features — built on the full dataset; the split is applied in Weekend 3
    print("\nEngineering features ...")
    df_feat = build_features(df, country_code)

    # Save
    feat_df = save_features(df_feat, src_path)

    print("\n=== Feature summary ===")
    print(feat_df.describe().T[["mean", "std", "min", "max"]].round(2).to_string())

    print("\nWeekend 2 done. Next: Weekend 3 — train LightGBM, beat these baselines:")
    for k, v in baselines.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
