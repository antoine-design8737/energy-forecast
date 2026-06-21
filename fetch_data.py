"""
fetch_data.py — Weekend 1: pull and clean the dataset.

Pulls hourly electricity load + generation-by-source from ENTSO-E and matching
weather (temperature, wind, solar radiation) from Open-Meteo, merges everything
onto a single clean hourly UTC grid, reports gaps, fills them with a stated rule,
and saves a processed parquet ready for feature engineering.

Setup:
    pip install entsoe-py pandas requests pyarrow python-dotenv
    Create a file named ".env" next to this script containing:
        ENTSOE_API_KEY=your-token-here

Run:
    python fetch_data.py                 # defaults: France, last 3 full years
    python fetch_data.py --country DE_LU --start-year 2021 --end-year 2024
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

# --- Config -----------------------------------------------------------------
# Representative cities per bidding zone, with rough metro populations (millions)
# used as weights. Exact values don't matter — they're normalised below — they
# just make the national weather a population-weighted average rather than a
# naive mean, which tracks demand far better.
COUNTRY_CONFIG = {
    "FR": {
        "tz": "Europe/Paris",
        "cities": [
            ("Paris",     48.8566,  2.3522, 11.0),
            ("Lyon",      45.7640,  4.8357,  2.3),
            ("Marseille", 43.2965,  5.3698,  1.9),
            ("Toulouse",  43.6047,  1.4442,  1.4),
            ("Lille",     50.6292,  3.0573,  1.2),
            ("Bordeaux",  44.8378, -0.5792,  1.0),
        ],
    },
    "DE_LU": {
        "tz": "Europe/Berlin",
        "cities": [
            ("Berlin",    52.5200, 13.4050,  4.5),
            ("Hamburg",   53.5511,  9.9937,  2.6),
            ("Munich",    48.1351, 11.5820,  2.9),
            ("Cologne",   50.9375,  6.9603,  2.1),
            ("Frankfurt", 50.1109,  8.6821,  2.3),
            ("Stuttgart", 48.7758,  9.1829,  2.7),
        ],
    },
}

SCRIPT_DIR = Path(__file__).parent
RAW_DIR = SCRIPT_DIR / "data" / "raw"
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"


# --- ENTSO-E ----------------------------------------------------------------
# The raw API caps each request at one year, so we pull year by year and stitch
# the pieces together. We immediately convert to UTC: UTC has no daylight-saving
# jumps, so the duplicated/missing hour you'd otherwise get at each DST switch
# simply disappears. Everything downstream lives in UTC.

def _yearly_windows(start_year, end_year, tz):
    for year in range(start_year, end_year):
        yield (
            pd.Timestamp(f"{year}-01-01", tz=tz),
            pd.Timestamp(f"{year + 1}-01-01", tz=tz),
        )


def fetch_load(client, country, start_year, end_year, tz):
    pieces = []
    for start, end in _yearly_windows(start_year, end_year, tz):
        print(f"  load {start.year} ...")
        df = client.query_load(country, start=start, end=end)
        pieces.append(df)
    raw = pd.concat(pieces)
    # query_load returns a DataFrame; the column is usually "Actual Load".
    col = "Actual Load" if "Actual Load" in raw.columns else raw.columns[0]
    s = raw[col].tz_convert("UTC")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s.resample("h").mean().rename("load_mw")


def fetch_generation(client, country, start_year, end_year, tz):
    pieces = []
    for start, end in _yearly_windows(start_year, end_year, tz):
        print(f"  generation {start.year} ...")
        df = client.query_generation(country, start=start, end=end)
        pieces.append(df)
    raw = pd.concat(pieces)

    # Columns come back as a MultiIndex of (production_type, aggregation), where
    # aggregation is "Actual Aggregated" (generation) or "Actual Consumption"
    # (only for storage that also consumes). We keep generation and flatten.
    if isinstance(raw.columns, pd.MultiIndex):
        agg = "Actual Aggregated"
        keep = [c for c in raw.columns if c[1] == agg]
        gen = raw[keep]
        gen.columns = [c[0] for c in keep]
    else:
        gen = raw

    gen = gen.tz_convert("UTC")
    gen = gen[~gen.index.duplicated(keep="first")].sort_index()
    gen = gen.resample("h").mean()
    # Drop production types this country never reports (all-NaN columns).
    gen = gen.dropna(axis=1, how="all")
    gen.columns = ["gen_" + str(c).lower().replace(" ", "_") for c in gen.columns]
    return gen


# --- Open-Meteo (no token needed) -------------------------------------------
# Free historical archive. We ask for the data already in UTC so it lines up
# with the grid data without further timezone gymnastics.
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]


def _fetch_city_weather(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "UTC",
    }
    for attempt in range(3):
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        if resp.ok:
            h = resp.json()["hourly"]
            idx = pd.to_datetime(h["time"], utc=True)
            return pd.DataFrame({v: h[v] for v in WEATHER_VARS}, index=idx)
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()


def fetch_weather(country, start_date, end_date):
    cities = COUNTRY_CONFIG[country]["cities"]
    total_weight = sum(w for *_, w in cities)
    weighted_sum = None
    for name, lat, lon, weight in cities:
        print(f"  weather {name} ...")
        df = _fetch_city_weather(lat, lon, start_date, end_date)
        contribution = df * (weight / total_weight)
        weighted_sum = contribution if weighted_sum is None else weighted_sum.add(contribution)
        time.sleep(0.5)  # be polite to a free API
    weighted_sum = weighted_sum.rename(
        columns={
            "temperature_2m": "temp_c",
            "wind_speed_10m": "wind_ms",
            "shortwave_radiation": "solar_rad",
        }
    )
    return weighted_sum.resample("h").mean()


# --- Orchestration ----------------------------------------------------------

def build_dataset(country, start_year, end_year, force=False):
    if country not in COUNTRY_CONFIG:
        raise ValueError(f"Unknown country '{country}'. Options: {list(COUNTRY_CONFIG)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{country}_{start_year}_{end_year}_hourly.parquet"
    if out_path.exists() and not force:
        print(f"Already built: {out_path}  (use --force to rebuild)")
        return pd.read_parquet(out_path)

    load_dotenv(SCRIPT_DIR / ".env")
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("ENTSOE_API_KEY not found. Put it in a .env file (see header).")
    client = EntsoePandasClient(api_key=api_key)
    tz = COUNTRY_CONFIG[country]["tz"]

    print("Fetching ENTSO-E load ...")
    load = fetch_load(client, country, start_year, end_year, tz)
    print("Fetching ENTSO-E generation ...")
    generation = fetch_generation(client, country, start_year, end_year, tz)

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year - 1}-12-31"
    print("Fetching Open-Meteo weather ...")
    weather = fetch_weather(country, start_date, end_date)

    # Merge on the common UTC hourly grid, then trim to where demand exists.
    print("Merging ...")
    df = pd.concat([load, generation, weather], axis=1).sort_index()
    df = df.loc[load.first_valid_index():load.last_valid_index()]

    # Missingness report BEFORE filling — this transparency is the point.
    missing = df.isna().sum()
    print("\nMissing values per column (before fill):")
    print(missing[missing > 0] if missing.any() else "  none")

    # Fill rule: linear interpolation over time for short internal gaps (these
    # are smooth physical signals), capped so we never invent long stretches,
    # then fill any remaining edge gaps from the nearest valid value.
    df = df.interpolate(method="time", limit=6, limit_direction="both")
    df = df.ffill().bfill()

    df.index.name = "timestamp_utc"
    df.to_parquet(out_path)
    print(f"\nSaved {len(df):,} hourly rows x {df.shape[1]} columns -> {out_path}")
    print(f"Range: {df.index.min()}  ->  {df.index.max()}")
    return df


def main():
    this_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Pull and clean grid + weather data.")
    parser.add_argument("--country", default="FR", choices=list(COUNTRY_CONFIG))
    parser.add_argument("--start-year", type=int, default=this_year - 3)
    parser.add_argument("--end-year", type=int, default=this_year,
                        help="Exclusive upper bound (data pulled up to end of the year before).")
    parser.add_argument("--force", action="store_true", help="Rebuild even if cached.")
    args = parser.parse_args()
    build_dataset(args.country, args.start_year, args.end_year, force=args.force)


if __name__ == "__main__":
    main()