"""Run multi-zone Phase 3 generalization, then create research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from finalize_phase3 import main as finalize
from src.research.data import fetch_epftoolbox_zone
from src.research.features import build_daily_dataset
from src.research.generalization import run_generalization, split_zone

ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads((ROOT / "config" / "phase3.json").read_text(encoding="utf-8"))
    splits = {}
    for zone, zone_config in config["zones"].items():
        print(f"Preparing {zone}", flush=True)
        data = build_daily_dataset(fetch_epftoolbox_zone(zone))
        splits[zone] = split_zone(data, zone, int(zone_config["test_year"]), int(config["calibration_years"]))
    predictions = run_generalization(splits, int(config["transfer_finetune_days"]))
    output = ROOT / "results" / "predictions" / "phase3_generalization_predictions.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output, index=False)
    print(f"Saved {len(predictions):,} predictions to {output}")
    finalize()


if __name__ == "__main__":
    main()
