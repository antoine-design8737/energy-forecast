"""
train.py — Weekend 3: train, evaluate, and save the forecasting model.

Run AFTER features.py has produced data/processed/features.parquet.

What this script does:
    1. Loads the feature matrix produced in Weekend 2.
    2. Applies the same leakage-free split (train / val / test).
    3. Trains LightGBM with early stopping on the validation set.
    4. Evaluates with walk-forward validation on the test set.
    5. Prints metrics vs the baselines from Weekend 2.
    6. Saves four plots to data/plots/ and the trained model to data/model/.

Usage:
    python train.py
    python train.py --file data/processed/features.parquet
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).parent
PLOT_DIR     = SCRIPT_DIR / "data" / "plots"
MODEL_DIR    = SCRIPT_DIR / "data" / "model"
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"

# --- Reproduce the same split constants as features.py --------------------
TEST_DAYS = 90
VAL_DAYS  = 90

# --- Plotting style -------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mape(actual, predicted):
    mask = actual > 0
    return 100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def save_fig(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path}")


# ---------------------------------------------------------------------------
# 1. Load features
# ---------------------------------------------------------------------------

def load_features(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    print(f"Loaded features: {len(df):,} rows × {df.shape[1]} columns")
    return df


def make_split(df):
    last       = df.index.max()
    test_start = last  - pd.Timedelta(days=TEST_DAYS)
    val_start  = test_start - pd.Timedelta(days=VAL_DAYS)

    train = df[df.index <  val_start]
    val   = df[(df.index >= val_start) & (df.index < test_start)]
    test  = df[df.index >= test_start]

    print(f"\nSplit:")
    print(f"  Train : {train.index.min().date()} → {train.index.max().date()}  ({len(train):,} rows)")
    print(f"  Val   : {val.index.min().date()}   → {val.index.max().date()}   ({len(val):,} rows)")
    print(f"  Test  : {test.index.min().date()}  → {test.index.max().date()}  ({len(test):,} rows)")
    return train, val, test


# ---------------------------------------------------------------------------
# 2. Baselines (recomputed directly from the load lag columns)
# ---------------------------------------------------------------------------

def compute_baselines(test):
    actual   = test["load_mw"]
    seasonal = test["load_lag_168h"]   # same hour last week
    yest     = test["load_lag_24h"]    # same hour yesterday

    mask = actual.notna() & seasonal.notna() & yest.notna()
    a, s, y = actual[mask], seasonal[mask], yest[mask]

    results = {
        "seasonal_naive_mape": mape(a, s),
        "seasonal_naive_mae":  mae(a, s),
        "yesterday_naive_mape": mape(a, y),
        "yesterday_naive_mae":  mae(a, y),
    }
    print("\n=== Baselines (test set) ===")
    print(f"  Seasonal naive  — MAPE: {results['seasonal_naive_mape']:.2f}%   MAE: {results['seasonal_naive_mae']/1e3:.2f} GW")
    print(f"  Yesterday naive — MAPE: {results['yesterday_naive_mape']:.2f}%  MAE: {results['yesterday_naive_mae']/1e3:.2f} GW")
    return results


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "hour", "dayofweek", "month", "quarter", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "is_holiday_eve",
    "load_lag_24h", "load_lag_48h", "load_lag_168h",
    "load_rolling_mean_24h", "load_rolling_mean_168h", "load_rolling_std_24h",
    "load_same_hour_last_week",
    "temp_c", "temp_c_sq", "wind_ms", "solar_rad",
    "heating_deg", "cooling_deg", "temp_rolling_24h",
]

TARGET = "load_mw"


def get_xy(df, available_features):
    X = df[available_features]
    y = df[TARGET]
    return X, y


def train_model(train, val, available_features):
    X_tr, y_tr = get_xy(train, available_features)
    X_val, y_val = get_xy(val, available_features)

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print("\nTraining LightGBM (early stopping on val set) ...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return model


# ---------------------------------------------------------------------------
# 4. Walk-forward validation on the test set
# ---------------------------------------------------------------------------
# This is the honest evaluation strategy. Instead of training once and
# predicting all 90 days at once, we roll forward week by week: train on
# everything before the current block, predict it, move forward. This mimics
# real deployment where you'd retrain regularly.

STEP_DAYS = 7   # re-evaluate in weekly blocks


def walk_forward_eval(df, test, available_features, val_start):
    """
    For each weekly block in the test set:
        - Train on all data before the block (no val data in training here).
        - Predict the block.
        - Roll forward.
    Returns a Series of predictions aligned to the test index.
    """
    print("\nWalk-forward validation ...")
    all_preds = []
    block_start = test.index.min()
    block_end   = test.index.max()

    step = pd.Timedelta(days=STEP_DAYS)
    cursor = block_start
    n_blocks = int(np.ceil((block_end - block_start) / step))

    for i in range(n_blocks):
        window_end = min(cursor + step, block_end + pd.Timedelta(hours=1))
        block = test[(test.index >= cursor) & (test.index < window_end)]
        if block.empty:
            break

        # Train on everything strictly before the block
        hist = df[df.index < cursor]
        if len(hist) < 500:
            cursor += step
            continue

        X_hist, y_hist = get_xy(hist, available_features)
        X_block        = block[available_features]

        m = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(X_hist, y_hist)
        preds = m.predict(X_block)
        all_preds.append(pd.Series(preds, index=block.index))

        pct = int(100 * (i + 1) / n_blocks)
        print(f"  block {i+1}/{n_blocks}  {cursor.date()} → {window_end.date()}  [{pct}%]", end="\r")
        cursor += step

    print()
    return pd.concat(all_preds).sort_index()


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def report_metrics(test, predictions, baselines):
    actual = test["load_mw"].reindex(predictions.index)
    mask   = actual.notna() & predictions.notna()
    a, p   = actual[mask], predictions[mask]

    model_mape = mape(a, p)
    model_mae  = mae(a, p)
    model_rmse = rmse(a, p)

    s_mape = baselines["seasonal_naive_mape"]
    s_mae  = baselines["seasonal_naive_mae"]

    improvement_mape = (s_mape - model_mape) / s_mape * 100
    improvement_mae  = (s_mae  - model_mae)  / s_mae  * 100

    print("\n=== Model (walk-forward, test set) ===")
    print(f"  MAPE : {model_mape:.2f}%   (seasonal naive: {s_mape:.2f}%  → improvement: {improvement_mape:+.1f}%)")
    print(f"  MAE  : {model_mae/1e3:.2f} GW  (seasonal naive: {s_mae/1e3:.2f} GW → improvement: {improvement_mae:+.1f}%)")
    print(f"  RMSE : {model_rmse/1e3:.2f} GW")
    print()
    print("  ↑ These numbers go in your README headline and your pitch.")
    print(f'  ↑ "Forecasts next-day demand to within {model_mape:.1f}% MAPE, beating')
    print(f'     the seasonal-naive baseline by {improvement_mape:.1f}%."')

    return {
        "model_mape": model_mape,
        "model_mae":  model_mae,
        "model_rmse": model_rmse,
        "improvement_mape_pct": improvement_mape,
    }


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(test, predictions, n_days=14):
    """Zoom into the last n_days of the test set so individual hours are visible."""
    end   = test.index.max()
    start = end - pd.Timedelta(days=n_days)
    mask  = (test.index >= start)

    actual = test["load_mw"][mask] / 1e3
    pred   = predictions.reindex(test.index)[mask] / 1e3

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual.values, label="Actual",    color="#2563EB", linewidth=1.5)
    ax.plot(pred.index,   pred.values,   label="Forecast",  color="#F59E0B", linewidth=1.5, linestyle="--")
    ax.set(title=f"Forecast vs actual — last {n_days} days of test set",
           ylabel="Load (GW)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate()
    save_fig(fig, "7_forecast_vs_actual.png")


def plot_residuals(test, predictions):
    """Residual distribution — should be centred on zero with no obvious skew."""
    actual = test["load_mw"].reindex(predictions.index)
    resid  = (predictions - actual) / 1e3
    resid  = resid.dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Distribution
    axes[0].hist(resid.values, bins=60, color="#2563EB", alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Residual distribution", xlabel="Error (GW)", ylabel="Count")

    # Residuals over time
    axes[1].plot(resid.index, resid.values, color="#2563EB", linewidth=0.6, alpha=0.7)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Residuals over time", xlabel="Date", ylabel="Error (GW)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    save_fig(fig, "8_residuals.png")


def plot_feature_importance(model, feature_names, top_n=20):
    """Top features by gain — sanity check that physics-driven features dominate."""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38))
    colors = ["#F59E0B" if "load" in n else
              "#EF4444" if any(w in n for w in ["temp","heat","cool","solar","wind"]) else
              "#2563EB"
              for n in imp.index]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set(title=f"Feature importance (top {top_n}, by gain)",
           xlabel="Gain")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#F59E0B", label="Load lags"),
        Patch(color="#EF4444", label="Weather"),
        Patch(color="#2563EB", label="Calendar"),
    ]
    ax.legend(handles=legend, loc="lower right")
    fig.tight_layout()
    save_fig(fig, "9_feature_importance.png")


def plot_error_by_hour(test, predictions):
    """MAPE by hour of day — reveals when the model struggles most."""
    actual = test["load_mw"].reindex(predictions.index)
    df_err = pd.DataFrame({"actual": actual, "pred": predictions})
    df_err = df_err.dropna()
    df_err["hour"] = df_err.index.tz_convert("Europe/Paris").hour
    df_err["pct_err"] = 100 * np.abs(df_err["actual"] - df_err["pred"]) / df_err["actual"]

    by_hour = df_err.groupby("hour")["pct_err"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(by_hour.index, by_hour.values, color="#2563EB", alpha=0.85)
    ax.set(title="MAPE by hour of day (local time)",
           xlabel="Hour", ylabel="MAPE (%)", xticks=range(0, 24, 2))
    save_fig(fig, "10_error_by_hour.png")


# ---------------------------------------------------------------------------
# 7. Save model and metrics
# ---------------------------------------------------------------------------

def save_model(model, metrics, available_features):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path   = MODEL_DIR / "lgbm_model.txt"
    metrics_path = MODEL_DIR / "metrics.json"
    features_path = MODEL_DIR / "features.json"

    model.booster_.save_model(str(model_path))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(features_path, "w") as f:
        json.dump(available_features, f, indent=2)

    print(f"\n  model   -> {model_path}")
    print(f"  metrics -> {metrics_path}")
    print(f"  features-> {features_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None)
    args = parser.parse_args()

    feat_path = Path(args.file) if args.file else PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found at {feat_path}. Run features.py first.")

    df = load_features(feat_path)
    train, val, test = make_split(df)

    # Only keep features that actually exist in the data
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\nUsing {len(available_features)} features:")
    print(" ", available_features)

    # Baselines first — never skip this
    baselines = compute_baselines(test)

    # Train a single model (used for feature importance plot only)
    model = train_model(train, val, available_features)

    # Walk-forward evaluation on the test set (the honest number)
    predictions = walk_forward_eval(df, test, available_features, val.index.min())

    # Metrics vs baselines
    metrics = report_metrics(test, predictions, baselines)

    # Plots
    print("\nGenerating plots ...")
    plot_forecast_vs_actual(test, predictions)
    plot_residuals(test, predictions)
    plot_feature_importance(model, available_features)
    plot_error_by_hour(test, predictions)

    # Save
    save_model(model, {**baselines, **metrics}, available_features)
    print("\nWeekend 3 done. Next: Weekend 4 — build and deploy the Streamlit dashboard.")


if __name__ == "__main__":
    main()
