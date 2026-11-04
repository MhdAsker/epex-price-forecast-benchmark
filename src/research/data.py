"""Chunked, auditable price-history acquisition and dataset manifests."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PRICE_PATH = RAW_DIR / "price_history_2020_2025.parquet"
MANIFEST_PATH = RAW_DIR / "price_history_2020_2025.manifest.json"
EPF_DE_PATH = RAW_DIR / "epftoolbox_epex_de.parquet"
EPF_DE_MANIFEST_PATH = RAW_DIR / "epftoolbox_epex_de.manifest.json"
EPF_DE_URL = "https://zenodo.org/records/4624805/files/DE.csv"
EPF_DATASET_URL = "https://zenodo.org/records/4624805/files/{zone}.csv"


def _normalise_prices(series: pd.Series) -> pd.Series:
    series = series.copy()
    if series.index.tz is None:
        raise ValueError("ENTSO-E price response must have a timezone-aware index")
    series.index = series.index.tz_convert("UTC").tz_localize(None)
    series = series.sort_index().astype(float)
    if series.index.has_duplicates:
        raise ValueError("Duplicate UTC delivery timestamps in price response")
    return series.rename("price_eur_mwh")


def fetch_price_history(
    start_year: int = 2020,
    end_year_exclusive: int = 2026,
    *,
    force: bool = False,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch prices one calendar year at a time and persist a checksum manifest."""
    if PRICE_PATH.exists() and not force:
        return pd.read_parquet(PRICE_PATH)
    load_dotenv(ROOT / ".env")
    token = os.environ.get("ENTSOE_API_KEY")
    if not token:
        raise RuntimeError("ENTSOE_API_KEY is required to fetch price history")
    client = EntsoePandasClient(api_key=token)
    pieces: list[pd.Series] = []
    retrievals: list[dict] = []
    for year in range(start_year, end_year_exclusive):
        start = pd.Timestamp(f"{year}-01-01", tz="Europe/Brussels")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="Europe/Brussels")
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                raw = client.query_day_ahead_prices("DE_LU", start=start, end=end)
                clean = _normalise_prices(raw)
                pieces.append(clean)
                retrievals.append(
                    {"year": year, "rows": len(clean), "attempt": attempt}
                )
                break
            except Exception as exc:  # API/network failures are retried by year
                last_error = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Failed to fetch DE_LU prices for {year}") from last_error
    prices = pd.concat(pieces).sort_index()
    if prices.index.has_duplicates:
        duplicates = prices.index[prices.index.duplicated()].unique()
        raise ValueError(f"Duplicate timestamps across yearly chunks: {duplicates[:3]}")
    expected_start = pd.Timestamp(f"{start_year}-01-01", tz="Europe/Berlin").tz_convert("UTC").tz_localize(None)
    expected_end = pd.Timestamp(f"{end_year_exclusive}-01-01", tz="Europe/Berlin").tz_convert("UTC").tz_localize(None)
    expected = pd.date_range(expected_start, expected_end, freq="h", inclusive="left")
    missing = expected.difference(prices.index)
    if len(missing):
        raise ValueError(f"Price history has {len(missing)} missing UTC hours; first={missing[0]}")
    frame = prices.to_frame()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PRICE_PATH)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    ).hexdigest()
    manifest = {
        "dataset": "ENTSO-E DE_LU day-ahead prices",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_method": "entsoe-py query_day_ahead_prices",
        "start_year": start_year,
        "end_year_exclusive": end_year_exclusive,
        "rows": len(frame),
        "first_utc": frame.index.min().isoformat(),
        "last_utc": frame.index.max().isoformat(),
        "sha256": digest,
        "chunks": retrievals,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def load_price_history() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"{PRICE_PATH} missing; run python -m src.research.data")
    return pd.read_parquet(PRICE_PATH)


def fetch_epftoolbox_de(*, force: bool = False) -> pd.DataFrame:
    """Fetch the six-year open EPEX-Germany benchmark used by Lago et al.

    Only the price column enters the primary information set. The two bundled
    exogenous columns are retained in the raw download checksum but excluded
    because this project has not independently audited their publication
    vintages.
    """
    if EPF_DE_PATH.exists() and not force:
        return pd.read_parquet(EPF_DE_PATH)
    raw = pd.read_csv(EPF_DE_URL, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("EPEX-DE benchmark index must be unique and ordered")
    price_col = "Price" if "Price" in raw.columns else raw.columns[0]
    frame = raw[[price_col]].rename(columns={price_col: "price_eur_mwh"}).astype(float)
    if frame["price_eur_mwh"].isna().any():
        raise ValueError("EPEX-DE benchmark contains missing prices")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(EPF_DE_PATH)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy().tobytes()
    ).hexdigest()
    manifest = {
        "dataset": "Lago et al. epftoolbox EPEX-Germany",
        "source": EPF_DE_URL,
        "zenodo_record": "4624805",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": len(raw),
        "raw_columns": list(raw.columns),
        "used_columns": [price_col],
        "excluded_columns_reason": "exogenous publication vintages not independently audited",
        "first_delivery": frame.index.min().isoformat(),
        "last_delivery": frame.index.max().isoformat(),
        "raw_content_sha256": digest,
    }
    EPF_DE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def fetch_epftoolbox_zone(zone: str, *, force: bool = False) -> pd.DataFrame:
    """Fetch one checksum-pinned open benchmark zone, using prices only."""
    zone = zone.upper()
    if zone not in {"DE", "FR", "BE", "NP"}:
        raise ValueError(f"Unsupported benchmark zone: {zone}")
    if zone == "DE":
        return fetch_epftoolbox_de(force=force)
    path = RAW_DIR / f"epftoolbox_{zone.lower()}.parquet"
    manifest_path = RAW_DIR / f"epftoolbox_{zone.lower()}.manifest.json"
    if path.exists() and not force:
        return pd.read_parquet(path)
    url = EPF_DATASET_URL.format(zone=zone)
    raw = pd.read_csv(url, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError(f"{zone} benchmark index must be unique and ordered")
    price_col = "Price" if "Price" in raw.columns else raw.columns[0]
    frame = raw[[price_col]].rename(columns={price_col: "price_eur_mwh"}).astype(float)
    if frame["price_eur_mwh"].isna().any():
        raise ValueError(f"{zone} benchmark contains missing prices")
    frame.to_parquet(path)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(raw, index=True).to_numpy().tobytes()
    ).hexdigest()
    manifest = {
        "dataset": f"Lago et al. epftoolbox {zone}",
        "source": url,
        "zenodo_record": "4624805",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": len(raw),
        "raw_columns": list(raw.columns),
        "used_columns": [price_col],
        "excluded_columns_reason": "exogenous publication vintages not independently audited",
        "first_delivery": frame.index.min().isoformat(),
        "last_delivery": frame.index.max().isoformat(),
        "raw_content_sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    result = fetch_epftoolbox_de()
    print(f"Saved {len(result):,} hourly prices to {EPF_DE_PATH}")
