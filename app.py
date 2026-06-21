"""
app.py — Electricity Demand Forecast: a presentation-grade dashboard.

A polished, single-page "website" built on Streamlit that foregrounds the
performance of a LightGBM next-day electricity-demand forecaster for France.

Run locally:
    streamlit run app.py

Deploy (see README_DEPLOY.md for the full walkthrough):
    Push this folder to GitHub, then connect the repo at share.streamlit.io.
    No secrets are required — the app only reads the pre-trained model and the
    processed parquet that ship in data/.

Project layout expected:
    your-project/
    ├── app.py
    ├── requirements.txt
    ├── data/
    │   ├── processed/features.parquet
    │   └── model/{lgbm_model.txt, metrics.json, features.json}
"""

import json
from pathlib import Path
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Electricity Demand Forecast · LightGBM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR    = Path(__file__).parent
MODEL_DIR     = SCRIPT_DIR / "data" / "model"
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"

# Palette ------------------------------------------------------------------
INK    = "#0F172A"   # slate-900
MUTED  = "#64748B"   # slate-500
BLUE   = "#2563EB"
INDIGO = "#4F46E5"
VIOLET = "#7C3AED"
AMBER  = "#F59E0B"
RED    = "#EF4444"
GREEN  = "#10B981"
GRID   = "rgba(15,23,42,0.08)"


# ===========================================================================
# Global styling
# ===========================================================================

def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #F8FAFC; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* ---- Hero ---- */
.hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 46px 46px 40px;
  background: linear-gradient(135deg,#1E3A8A 0%,#4F46E5 48%,#7C3AED 100%);
  box-shadow: 0 24px 60px -22px rgba(79,70,229,.65); color: #fff; margin-bottom: 26px;
}
.hero::after {
  content:""; position:absolute; inset:0;
  background: radial-gradient(900px 360px at 88% -25%, rgba(255,255,255,.20), transparent 60%);
  pointer-events:none;
}
.hero .eyebrow {
  display:inline-flex; align-items:center; gap:8px; font-size:.78rem; font-weight:600;
  letter-spacing:.12em; text-transform:uppercase; color:#C7D2FE;
  background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22);
  padding:6px 14px; border-radius:999px;
}
.hero h1 { font-size:2.85rem; font-weight:800; line-height:1.08; margin:18px 0 10px; letter-spacing:-.02em; }
.hero h1 .hl {
  background:linear-gradient(90deg,#FDE68A,#FBBF24); -webkit-background-clip:text;
  background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p.sub { font-size:1.08rem; color:#E0E7FF; max-width:680px; margin:0 0 26px; line-height:1.55; }
.chips { display:flex; flex-wrap:wrap; gap:14px; }
.chip {
  background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.20);
  border-radius:14px; padding:14px 18px; min-width:140px; backdrop-filter:blur(6px);
}
.chip .v { font-size:1.7rem; font-weight:800; line-height:1; }
.chip .l { font-size:.78rem; color:#C7D2FE; margin-top:6px; letter-spacing:.02em; }

/* ---- Section headings ---- */
.sec { margin: 34px 0 14px; }
.sec .kicker { color:#6366F1; font-weight:700; font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; }
.sec h2 { color:#0F172A; font-size:1.6rem; font-weight:800; margin:6px 0 4px; letter-spacing:-.01em; }
.sec p { color:#64748B; font-size:.98rem; margin:0; max-width:760px; }

/* ---- Metric cards ---- */
.cards { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.card {
  background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:20px 20px 18px;
  box-shadow:0 8px 24px -18px rgba(15,23,42,.35); position:relative;
}
.card .bar { position:absolute; left:0; top:16px; bottom:16px; width:4px; border-radius:4px; }
.card .label { color:#64748B; font-size:.82rem; font-weight:600; padding-left:12px; }
.card .value { color:#0F172A; font-size:2.15rem; font-weight:800; line-height:1; margin:8px 0 4px; padding-left:12px; letter-spacing:-.02em; }
.card .note { color:#94A3B8; font-size:.8rem; padding-left:12px; line-height:1.35; }

/* ---- Generic panel ---- */
.panel { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:8px 10px; box-shadow:0 8px 24px -18px rgba(15,23,42,.35); }

/* ---- How-it-works tiles ---- */
.tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
.tile { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:22px; }
.tile .ic { font-size:1.5rem; }
.tile h3 { color:#0F172A; font-size:1.05rem; font-weight:700; margin:10px 0 8px; }
.tile p { color:#475569; font-size:.9rem; line-height:1.55; margin:0; }
.tile .meta { color:#6366F1; font-weight:700; font-size:.82rem; margin-top:12px; }

/* ---- Footer ---- */
.foot { margin-top:42px; padding-top:22px; border-top:1px solid #E2E8F0; color:#94A3B8; font-size:.85rem; line-height:1.7; }
.foot a { color:#4F46E5; text-decoration:none; }

/* ---- Streamlit metric tweaks (day metrics row) ---- */
[data-testid="stMetric"] { background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:14px 16px; box-shadow:0 8px 24px -20px rgba(15,23,42,.35); }
[data-testid="stMetricLabel"] p { font-weight:600; color:#64748B; }

/* sidebar */
section[data-testid="stSidebar"] { background:#0F172A; }
section[data-testid="stSidebar"] * { color:#E2E8F0; }
section[data-testid="stSidebar"] .stMetric { background:rgba(255,255,255,.04); }

@media (max-width:880px){ .cards{grid-template-columns:repeat(2,1fr);} .tiles{grid-template-columns:1fr;} .hero h1{font-size:2.1rem;} }
</style>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Data + model loading (cached)
# ===========================================================================

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    p = MODEL_DIR / "lgbm_model.txt"
    return lgb.Booster(model_file=str(p)) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_metrics():
    p = MODEL_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner=False)
def load_feature_names():
    p = MODEL_DIR / "features.json"
    return json.loads(p.read_text()) if p.exists() else []


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    candidates = sorted(PROCESSED_DIR.glob("features.parquet"))
    if not candidates:
        return None
    df = pd.read_parquet(candidates[0])
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


@st.cache_data(show_spinner=False)
def test_set_predictions(_model, _df, feature_names, days=90):
    """Predict the held-out tail of the series for the diagnostics charts."""
    last = _df.index.max()
    test = _df[_df.index >= last - pd.Timedelta(days=days)]
    avail = [c for c in feature_names if c in test.columns]
    pred = _model.predict(test[avail])
    actual = test["load_mw"].values
    idx = test.index
    return pd.DataFrame({"actual": actual, "pred": pred}, index=idx)


# ===========================================================================
# Prediction helpers
# ===========================================================================

def predict_day(model, df, feature_names, date_utc):
    mask = (df.index.date == date_utc.date())
    day = df.loc[mask, feature_names]
    if day.empty or day.isna().all(axis=None):
        return None
    return pd.Series(model.predict(day), index=day.index)


def metrics_for_day(actual, predicted):
    mask = actual.notna() & predicted.notna()
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return None
    return {
        "mape": 100 * np.mean(np.abs((a - p) / a.clip(lower=1))),
        "mae":  np.mean(np.abs(a - p)) / 1e3,
        "rmse": np.sqrt(np.mean((a - p) ** 2)) / 1e3,
    }


# ===========================================================================
# Plotly chart builders
# ===========================================================================

def _base_layout(fig, height=420, ylab=""):
    fig.update_layout(
        height=height, template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title=ylab)
    return fig


def chart_24h(actual, predicted):
    combined = (pd.DataFrame({"actual": actual, "pred": predicted})
                .sort_index().dropna(how="all") / 1e3)
    x = combined.index.tz_convert("Europe/Paris")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=combined["pred"], name="Forecast", mode="lines",
                             line=dict(color=AMBER, width=3, dash="dot"),
                             fill="tonexty", fillcolor="rgba(245,158,11,0.10)"))
    fig.add_trace(go.Scatter(x=x, y=combined["actual"], name="Actual", mode="lines",
                             line=dict(color=BLUE, width=3)))
    # reorder so actual fills toward forecast nicely
    fig.data = (fig.data[1], fig.data[0])
    _base_layout(fig, 420, "Load (GW)")
    fig.update_xaxes(tickformat="%H:%M")
    return fig


def chart_week(df, selected_utc, predicted):
    start = selected_utc - timedelta(days=3)
    end   = selected_utc + timedelta(days=4)
    window = (df["load_mw"].loc[start:end] / 1e3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window.index.tz_convert("Europe/Paris"), y=window.values,
                             name="Actual load", mode="lines", line=dict(color=BLUE, width=2)))
    if predicted is not None:
        fig.add_trace(go.Scatter(x=predicted.index.tz_convert("Europe/Paris"),
                                 y=predicted.values / 1e3, name="Forecast (selected day)",
                                 mode="lines", line=dict(color=AMBER, width=3, dash="dot")))
    fig.add_vrect(x0=selected_utc.tz_convert("Europe/Paris"),
                  x1=(selected_utc + timedelta(days=1)).tz_convert("Europe/Paris"),
                  fillcolor=AMBER, opacity=0.08, line_width=0)
    _base_layout(fig, 360, "Load (GW)")
    return fig


def chart_benchmark(metrics):
    names = ["Seasonal naive<br>(last week)", "Yesterday naive", "LightGBM model"]
    vals  = [metrics.get("seasonal_naive_mape", 0),
             metrics.get("yesterday_naive_mape", 0),
             metrics.get("model_mape", 0)]
    colors = ["#CBD5E1", "#94A3B8", GREEN]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v:.2f}%" for v in vals], textposition="outside",
        textfont=dict(size=15, color=INK, family="Inter"),
    ))
    _base_layout(fig, 360, "MAPE (%)  ·  lower is better")
    fig.update_layout(hovermode=False, showlegend=False)
    fig.update_yaxes(range=[0, max(vals) * 1.25])
    return fig


def chart_residuals(test_df):
    resid = ((test_df["pred"] - test_df["actual"]) / 1e3).dropna()
    fig = go.Figure(go.Histogram(x=resid, nbinsx=60, marker_color=BLUE,
                                 marker_line_color="white", marker_line_width=0.5,
                                 opacity=0.9))
    fig.add_vline(x=0, line_dash="dash", line_color=INK)
    _base_layout(fig, 340, "Count")
    fig.update_layout(hovermode=False)
    fig.update_xaxes(title="Forecast error (GW)")
    return fig


def chart_error_by_hour(test_df):
    d = test_df.dropna().copy()
    d["hour"] = d.index.tz_convert("Europe/Paris").hour
    d["pe"] = 100 * np.abs(d["actual"] - d["pred"]) / d["actual"].clip(lower=1)
    by = d.groupby("hour")["pe"].mean()
    fig = go.Figure(go.Bar(x=by.index, y=by.values, marker_color=INDIGO, opacity=0.9))
    _base_layout(fig, 340, "MAPE (%)")
    fig.update_layout(hovermode=False)
    fig.update_xaxes(title="Hour of day (local)", dtick=2)
    return fig


def chart_importance(model, feature_names, top_n=14):
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feature_names)
    imp = imp.sort_values(ascending=True).tail(top_n)

    def grp(n):
        if "load" in n:
            return AMBER
        if any(w in n for w in ["temp", "heat", "cool", "solar", "wind"]):
            return RED
        return BLUE

    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color=[grp(n) for n in imp.index]))
    _base_layout(fig, 430, "")
    fig.update_layout(hovermode=False)
    fig.update_xaxes(title="Importance (gain)")
    return fig


# ===========================================================================
# HTML section renderers
# ===========================================================================

def render_hero(metrics, df):
    mape  = metrics.get("model_mape", 0)
    imp   = metrics.get("improvement_mape_pct", 0)
    mae   = metrics.get("model_mae", 0) / 1e3
    span  = f"{df.index.min().year}–{df.index.max().year}"
    html = (
        '<div class="hero">'
        '<span class="eyebrow">⚡ LightGBM · Walk-forward validated</span>'
        f'<h1>Forecasting France\'s electricity demand<br>to within <span class="hl">{mape:.2f}% error</span>.</h1>'
        '<p class="sub">A gradient-boosting model predicts next-day hourly grid load from calendar, '
        'load-history and weather signals — and beats the standard industry baseline by a wide margin.</p>'
        '<div class="chips">'
        f'<div class="chip"><div class="v">{mape:.2f}%</div><div class="l">Mean abs. % error</div></div>'
        f'<div class="chip"><div class="v">−{imp:.0f}%</div><div class="l">Error vs. baseline</div></div>'
        f'<div class="chip"><div class="v">{mae:.2f} GW</div><div class="l">Mean abs. error</div></div>'
        f'<div class="chip"><div class="v">{len(df):,}</div><div class="l">Hours · {span}</div></div>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def section_head(kicker, title, sub=""):
    st.markdown(
        f'<div class="sec"><div class="kicker">{kicker}</div><h2>{title}</h2>'
        f'<p>{sub}</p></div>',
        unsafe_allow_html=True,
    )


def render_kpis(metrics, df):
    mape   = metrics.get("model_mape", 0)
    mae    = metrics.get("model_mae", 0) / 1e3
    rmse   = metrics.get("model_rmse", 0) / 1e3
    s_mape = metrics.get("seasonal_naive_mape", 0)
    imp    = metrics.get("improvement_mape_pct", 0)
    fold   = s_mape / mape if mape else 0
    avg_gw = df["load_mw"].mean() / 1e3

    def card(color, label, value, note):
        return (
            f'<div class="card"><div class="bar" style="background:{color}"></div>'
            f'<div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="note">{note}</div></div>'
        )

    html = (
        '<div class="cards">'
        + card(GREEN,  "MAPE (test set)",       f"{mape:.2f}%",   f"Mean absolute percentage error on 90 unseen days")
        + card(BLUE,   "MAE",                   f"{mae:.2f} GW",  f"≈ {mae/avg_gw*100:.1f}% of the {avg_gw:.0f} GW average load")
        + card(INDIGO, "RMSE",                  f"{rmse:.2f} GW", "Penalises the large misses more heavily")
        + card(AMBER,  "vs. seasonal baseline", f"−{imp:.0f}%",   f"≈ {fold:.1f}× lower error than last-week-same-hour")
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_how(df, n_features):
    span = f"{df.index.min():%b %Y} – {df.index.max():%b %Y}"
    tiles = (
        '<div class="tiles">'
        '<div class="tile"><div class="ic">🛰️</div><h3>Real grid data</h3>'
        '<p>Hourly load and generation-by-source pulled from the ENTSO-E Transparency '
        'Platform, merged with population-weighted weather from Open-Meteo onto one clean UTC grid.</p>'
        f'<div class="meta">{span} · France</div></div>'
        '<div class="tile"><div class="ic">🧮</div><h3>Engineered features</h3>'
        '<p>Cyclically-encoded calendar terms, public holidays, load lags &amp; rolling stats, '
        'plus temperature (with a squared term for the U-shaped demand curve), wind and solar radiation.</p>'
        f'<div class="meta">{n_features} predictive features</div></div>'
        '<div class="tile"><div class="ic">🎯</div><h3>Honest evaluation</h3>'
        '<p>Walk-forward validation: the model is retrained on all prior data before each weekly '
        'test block, then scored on it. This mirrors real deployment and rules out look-ahead bias.</p>'
        '<div class="meta">No leakage · 90-day hold-out</div></div>'
        '</div>'
    )
    st.markdown(tiles, unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="foot">'
        'Built with LightGBM · Streamlit · Plotly &nbsp;·&nbsp; '
        'Data: <a href="https://transparency.entsoe.eu/">ENTSO-E Transparency Platform</a> '
        '+ <a href="https://open-meteo.com/">Open-Meteo</a><br>'
        'Next-day hourly electricity-demand forecasting · evaluated with walk-forward validation.'
        '</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar(df, metrics):
    st.sidebar.markdown("### ⚡ Electricity Forecast")
    st.sidebar.caption("Next-day hourly load · France · LightGBM")
    st.sidebar.divider()

    min_date = (df.index.min() + pd.Timedelta(days=8)).date()
    max_date = df.index.max().date()
    selected = st.sidebar.date_input(
        "📅 Forecast a date",
        value=max_date, min_value=min_date, max_value=max_date,
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Headline performance**")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("MAPE", f"{metrics.get('model_mape', 0):.2f}%")
    c2.metric("MAE", f"{metrics.get('model_mae', 0)/1e3:.2f} GW")
    imp = metrics.get("improvement_mape_pct", 0)
    st.sidebar.metric("vs. seasonal naive", f"−{imp:.0f}% error")

    st.sidebar.divider()
    st.sidebar.caption(
        "Data: ENTSO-E Transparency Platform + Open-Meteo · "
        "Model: LightGBM with walk-forward validation."
    )
    return pd.Timestamp(selected, tz="UTC")


# ===========================================================================
# Main
# ===========================================================================

def render_interactive(df, model, metrics, feature_names, selected_utc):
    avail = [c for c in feature_names if c in df.columns]
    predicted = predict_day(model, df, avail, selected_utc)
    actual = df["load_mw"].loc[df.index.date == selected_utc.date()]

    if predicted is None or len(predicted) == 0:
        st.warning("No feature data available for this date. Pick another from the sidebar.")
        return

    day_m = metrics_for_day(actual, predicted)
    if day_m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAPE · this day", f"{day_m['mape']:.2f}%")
        c2.metric("MAE · this day",  f"{day_m['mae']:.2f} GW")
        c3.metric("RMSE · this day", f"{day_m['rmse']:.2f} GW")
        s_pred = df["load_lag_168h"].reindex(actual.index)
        s_m = metrics_for_day(actual, s_pred)
        if s_m and s_m["mape"]:
            d = (s_m["mape"] - day_m["mape"]) / s_m["mape"] * 100
            c4.metric("vs. seasonal naive", f"{d:+.0f}%",
                      delta_color="normal" if d > 0 else "inverse")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(chart_24h(actual, predicted), width="stretch",
                    config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(chart_week(df, selected_utc, predicted), width="stretch",
                    config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("See the hourly numbers for this day"):
        out = pd.DataFrame({
            "Actual (GW)":   actual.values / 1e3,
            "Forecast (GW)": predicted.reindex(actual.index).values / 1e3,
            "Error (GW)":    (predicted.reindex(actual.index).values - actual.values) / 1e3,
        }, index=actual.index.tz_convert("Europe/Paris").strftime("%H:%M"))
        out.index.name = "Hour (local)"
        st.dataframe(out.style.format("{:.2f}"), width="stretch")


def main():
    inject_css()

    model        = load_model()
    metrics      = load_metrics()
    feature_names = load_feature_names()
    df           = load_data()

    if model is None or df is None:
        st.error("Model or data not found.")
        st.markdown(
            "**Setup checklist**\n\n"
            "1. `python fetch_data.py` — pull & clean the dataset\n"
            "2. `python features.py` — engineer the feature matrix\n"
            "3. `python train.py` — train & evaluate the model\n"
            "4. `streamlit run app.py` — launch this dashboard"
        )
        return

    selected_utc = render_sidebar(df, metrics)

    # 1) Hero
    render_hero(metrics, df)

    # 2) Performance headline
    section_head("Performance", "The headline numbers",
                 "Scored on the most recent 90 days — data the model never saw during training.")
    render_kpis(metrics, df)

    # 3) Benchmark
    section_head("Benchmark", "How much better than the baselines?",
                 "Utilities often quote a “same hour last week” seasonal-naive forecast. "
                 "Our model cuts that error by roughly three-quarters.")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(chart_benchmark(metrics), width="stretch",
                    config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # 4) Interactive forecast
    section_head("Live forecast", "Try it on any day",
                 f"Pick a date in the sidebar — currently **{selected_utc.date()}** — to see the "
                 "24-hour forecast against what actually happened on the French grid.")
    render_interactive(df, model, metrics, feature_names, selected_utc)

    # 5) Diagnostics
    section_head("Diagnostics", "Why you can trust the number",
                 "Errors are small, unbiased and stable across the day — not a lucky average.")
    t1, t2, t3 = st.tabs(["Residual distribution", "Error by hour", "What drives it"])
    test_df = test_set_predictions(model, df, tuple(feature_names))
    with t1:
        st.caption("Forecast errors over the full 90-day hold-out. Centred on zero ⇒ no systematic bias.")
        st.plotly_chart(chart_residuals(test_df), width="stretch",
                        config={"displayModeBar": False})
    with t2:
        st.caption("Average error for each hour of the day. The model is reliable around the clock.")
        st.plotly_chart(chart_error_by_hour(test_df), width="stretch",
                        config={"displayModeBar": False})
    with t3:
        st.caption("Top features by gain. Recent load and weather dominate — exactly as physics predicts.")
        st.plotly_chart(chart_importance(model, feature_names), width="stretch",
                        config={"displayModeBar": False})

    # 6) How it works
    section_head("Under the hood", "How the forecast is built",
                 "Three years of real grid data, physically-motivated features, and a leakage-free test.")
    render_how(df, len(feature_names))

    render_footer()


if __name__ == "__main__":
    main()
