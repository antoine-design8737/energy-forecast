"""
export_site_data.py — bake the model's outputs into a static JSON for the website.

The dashboard is purely historical, so every number the site needs can be
pre-computed once and served as a flat file. Run this whenever you retrain:

    python export_site_data.py

It writes docs/data.json, which docs/index.html reads at load time. No Python
runs when the site is live — it's just static files.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

SCRIPT_DIR    = Path(__file__).parent
MODEL_DIR     = SCRIPT_DIR / "data" / "model"
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
OUT_PATH      = SCRIPT_DIR / "docs" / "data.json"

WINDOW_DAYS = 365   # how much history the date-picker can scrub through
TEST_DAYS   = 90    # the honest hold-out used for the diagnostics
TOP_FEATS   = 14


def feature_group(name):
    if "load" in name:
        return "load"
    if any(w in name for w in ["temp", "heat", "cool", "solar", "wind"]):
        return "weather"
    return "calendar"


def main():
    model    = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
    metrics  = json.loads((MODEL_DIR / "metrics.json").read_text())
    features = json.loads((MODEL_DIR / "features.json").read_text())

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    avail = [c for c in features if c in df.columns]

    # ---- Headline metrics -------------------------------------------------
    m_mape = metrics["model_mape"]
    s_mape = metrics["seasonal_naive_mape"]
    y_mape = metrics["yesterday_naive_mape"]
    avg_gw = float(df["load_mw"].mean() / 1e3)

    headline = {
        "mape":            round(m_mape, 2),
        "mae_gw":          round(metrics["model_mae"] / 1e3, 2),
        "rmse_gw":         round(metrics["model_rmse"] / 1e3, 2),
        "improvement_pct": round(metrics["improvement_mape_pct"], 1),
        "fold":            round(s_mape / m_mape, 1),
        "seasonal_mape":   round(s_mape, 2),
        "yesterday_mape":  round(y_mape, 2),
        "avg_gw":          round(avg_gw, 1),
    }
    benchmark = [
        {"name": "Seasonal naive\n(last week)", "mape": round(s_mape, 2)},
        {"name": "Yesterday naive",             "mape": round(y_mape, 2)},
        {"name": "LightGBM model",              "mape": round(m_mape, 2)},
    ]

    # ---- Scrubbable hourly window (contiguous, UTC) -----------------------
    window = df.iloc[-WINDOW_DAYS * 24:].copy()
    window["pred_mw"] = model.predict(window[avail])

    t_unix   = (window.index.view("int64") // 10**9).astype("int64")  # seconds
    actual_gw = (window["load_mw"].to_numpy() / 1e3)
    pred_gw   = (window["pred_mw"].to_numpy() / 1e3)

    series = {
        "t":         t_unix.tolist(),
        "actual_gw": [round(float(v), 3) for v in actual_gw],
        "pred_gw":   [round(float(v), 3) for v in pred_gw],
    }

    # ---- Per-day index: one full UTC day = 24 rows ------------------------
    win_dates = pd.Series(window.index.date, index=range(len(window)))
    seasonal_mw = window["load_lag_168h"].to_numpy()  # same hour last week
    days = []
    for d, grp in win_dates.groupby(win_dates):
        idx = grp.index.to_numpy()
        if len(idx) != 24:                 # skip partial first/last day
            continue
        a = window["load_mw"].to_numpy()[idx]
        p = window["pred_mw"].to_numpy()[idx]
        s = seasonal_mw[idx]
        valid = ~np.isnan(a) & ~np.isnan(p)
        if valid.sum() == 0:
            continue
        a_v, p_v = a[valid], p[valid]
        day_mape = float(100 * np.mean(np.abs((a_v - p_v) / np.clip(a_v, 1, None))))
        day_mae  = float(np.mean(np.abs(a_v - p_v)) / 1e3)
        day_rmse = float(np.sqrt(np.mean((a_v - p_v) ** 2)) / 1e3)

        sm = ~np.isnan(s) & valid
        vs_seasonal = None
        if sm.sum() > 0:
            s_mape_day = 100 * np.mean(np.abs((a[sm] - s[sm]) / np.clip(a[sm], 1, None)))
            if s_mape_day > 0:
                vs_seasonal = round(float((s_mape_day - day_mape) / s_mape_day * 100), 0)

        days.append({
            "date":      str(d),
            "start":     int(idx[0]),
            "mape":      round(day_mape, 2),
            "mae_gw":    round(day_mae, 2),
            "rmse_gw":   round(day_rmse, 2),
            "vs_naive":  vs_seasonal,
        })

    # ---- Diagnostics over the honest 90-day hold-out ----------------------
    last = df.index.max()
    test = df[df.index >= last - pd.Timedelta(days=TEST_DAYS)].copy()
    test["pred_mw"] = model.predict(test[avail])
    resid = ((test["pred_mw"] - test["load_mw"]) / 1e3).dropna()

    counts, edges = np.histogram(resid.to_numpy(), bins=45)
    centers = (edges[:-1] + edges[1:]) / 2
    residual_hist = {
        "centers": [round(float(c), 3) for c in centers],
        "counts":  [int(c) for c in counts],
    }

    local_hour = test.index.tz_convert("Europe/Paris").hour
    a = test["load_mw"].to_numpy()
    p = test["pred_mw"].to_numpy()
    pe = 100 * np.abs(a - p) / np.clip(a, 1, None)
    by_hour = pd.Series(pe).groupby(local_hour.values).mean()
    error_by_hour = [{"hour": int(h), "mape": round(float(v), 2)}
                     for h, v in by_hour.items()]

    gains = model.feature_importance(importance_type="gain")
    imp = (pd.Series(gains, index=features).sort_values(ascending=False)
           .head(TOP_FEATS))
    feature_importance = [
        {"name": n, "gain": round(float(g), 1), "group": feature_group(n)}
        for n, g in imp.items()
    ]

    # ---- Assemble + write -------------------------------------------------
    payload = {
        "meta": {
            "rows":       int(len(df)),
            "n_features": len(avail),
            "span_start": str(df.index.min().date()),
            "span_end":   str(df.index.max().date()),
            "avg_gw":     round(avg_gw, 1),
            "generated":  pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
        },
        "headline":  headline,
        "benchmark": benchmark,
        "series":    series,
        "days":      days,
        "diagnostics": {
            "residual_hist":      residual_hist,
            "error_by_hour":      error_by_hour,
            "feature_importance": feature_importance,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH}  ({kb:.0f} KB)")
    print(f"  window: {len(days)} selectable days, {len(series['t'])} hourly points")
    print(f"  headline MAPE {headline['mape']}%  /  -{headline['improvement_pct']}% vs seasonal")


if __name__ == "__main__":
    main()
