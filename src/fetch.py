import os
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class ENTSOFetcher:
    def __init__(self):
        load_dotenv()
        self.client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
        self.zone = "DE_LU"
        logger.info("ENTSO-E client ready")

    def fetch_all(self, start: str, end: str) -> pd.DataFrame:
        """
        Fetch day-ahead prices, load forecast, wind/solar forecast, and generation
        forecast for DE_LU from ENTSO-E. start/end format: "YYYYMMDD".
        Returns a merged hourly DataFrame saved to data/raw/entso_raw.parquet.
        """
        ts_start = pd.Timestamp(start, tz="Europe/Brussels")
        ts_end = pd.Timestamp(end, tz="Europe/Brussels")

        # 1. Day-ahead prices
        prices = self._safe_fetch(
            self.client.query_day_ahead_prices, self.zone, start=ts_start, end=ts_end
        )

        # 2. Load forecast — try forecast first, fall back to actual load
        load = self._safe_fetch(
            self.client.query_load_forecast, self.zone, start=ts_start, end=ts_end
        )
        if isinstance(load, pd.DataFrame) and load.empty:
            logger.warning("query_load_forecast returned nothing — trying query_load as fallback")
            load = self._safe_fetch(
                self.client.query_load, self.zone, start=ts_start, end=ts_end
            )

        # 3. Wind & solar forecast  (DataFrame: Wind Onshore, Wind Offshore, Solar)
        wind_solar = self._safe_fetch(
            self.client.query_wind_and_solar_forecast,
            self.zone,
            start=ts_start,
            end=ts_end,
            psr_type=None,
        )

        # 4. Generation forecast
        gen = self._safe_fetch(
            self.client.query_generation_forecast, self.zone, start=ts_start, end=ts_end
        )

        def to_utc_hourly(obj):
            """Convert tz-aware Series/DataFrame to UTC, strip tz, resample hourly."""
            if obj is None or (isinstance(obj, pd.DataFrame) and obj.empty):
                return obj
            if isinstance(obj, pd.Series):
                obj = obj.tz_convert("UTC").tz_localize(None)
                return obj.resample("h").mean()
            # DataFrame
            obj.index = obj.index.tz_convert("UTC").tz_localize(None)
            return obj.resample("h").mean()

        prices = to_utc_hourly(prices)
        load = to_utc_hourly(load)
        wind_solar = to_utc_hourly(wind_solar)
        gen = to_utc_hourly(gen)

        # Build base DataFrame from prices
        df = pd.DataFrame({"price_eur_mwh": prices})

        # Load forecast
        if isinstance(load, pd.Series) and not load.empty:
            df["load_mw"] = load
        else:
            df["load_mw"] = float("nan")

        # Wind / solar columns
        if isinstance(wind_solar, pd.DataFrame) and not wind_solar.empty:
            cols = wind_solar.columns.tolist()
            onshore_col = next((c for c in cols if "Onshore" in c), None)
            offshore_col = next((c for c in cols if "Offshore" in c), None)
            solar_col = next((c for c in cols if "Solar" in c), None)
            df["wind_onshore_mw"] = wind_solar[onshore_col] if onshore_col else float("nan")
            df["wind_offshore_mw"] = wind_solar[offshore_col] if offshore_col else float("nan")
            df["solar_mw"] = wind_solar[solar_col] if solar_col else float("nan")
        else:
            df["wind_onshore_mw"] = float("nan")
            df["wind_offshore_mw"] = float("nan")
            df["solar_mw"] = float("nan")

        # Generation forecast
        if isinstance(gen, pd.Series) and not gen.empty:
            df["gen_forecast_mw"] = gen
        elif isinstance(gen, pd.DataFrame) and not gen.empty:
            df["gen_forecast_mw"] = gen.iloc[:, 0]
        else:
            df["gen_forecast_mw"] = float("nan")

        df = df.sort_index()
        df = df.ffill(limit=2)

        # Drop rows where target is missing
        df = df.dropna(subset=["price_eur_mwh"])

        out_path = RAW_DIR / "entso_raw.parquet"
        df.to_parquet(out_path)
        metadata = {
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_start": start,
            "requested_end": end,
            "bidding_zone": self.zone,
            "index_semantics": "timezone-naive UTC delivery timestamps",
            "series": {
                "price_eur_mwh": {
                    "source": "ENTSO-E query_day_ahead_prices",
                    "phase1_availability": "primary target/history",
                },
                "load_mw": {
                    "source": "ENTSO-E load forecast, actual-load fallback",
                    "phase1_availability": "oracle/unusable when fallback is actual",
                },
                "wind_onshore_mw": {
                    "source": "ENTSO-E query_wind_and_solar_forecast",
                    "phase1_availability": "post-gate oracle; vintage not preserved",
                },
                "wind_offshore_mw": {
                    "source": "ENTSO-E query_wind_and_solar_forecast",
                    "phase1_availability": "post-gate oracle; vintage not preserved",
                },
                "solar_mw": {
                    "source": "ENTSO-E query_wind_and_solar_forecast",
                    "phase1_availability": "post-gate oracle; vintage not preserved",
                },
                "gen_forecast_mw": {
                    "source": "ENTSO-E query_generation_forecast",
                    "phase1_availability": "post-gate oracle; vintage not preserved",
                },
            },
        }
        (RAW_DIR / "entso_raw.metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        n = len(df)
        logger.info("Fetched %d hourly rows from %s to %s", n, start, end)
        return df

    def _safe_fetch(self, func, *args, **kwargs):
        """Try func(*args, **kwargs); on any exception log a warning and return empty DataFrame."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", func.__name__, exc)
            return pd.DataFrame()


if __name__ == "__main__":
    fetcher = ENTSOFetcher()
    df = fetcher.fetch_all("20220101", "20241201")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
