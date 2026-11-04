"""Recreate Phase 3 tables, figures, cards, and artifact manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.research.decision import BatterySpec, evaluate_dispatch
from src.research.generalization import generalization_metrics
from src.research.regimes import regime_metrics

ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "results" / "predictions" / "phase3_generalization_predictions.parquet"
METRICS = ROOT / "results" / "metrics"
PLOTS = ROOT / "results" / "plots"
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_reports(general: pd.DataFrame, decision: pd.DataFrame, regimes: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    best = general.loc[general.groupby("zone")["MAE"].idxmin(), ["zone", "model", "MAE", "rMAE_D1"]]
    economic = decision.loc[decision.groupby("zone")["revenue_eur"].idxmax(), ["zone", "model", "revenue_eur", "regret_eur"]]
    report = f"""# Phase 3 research report

## Scope

This release evaluates local, pooled, and transfer 24-product curve models on four European price zones. Each result uses four prior calendar years for calibration and one complete held-out calendar year. Only lagged prices and calendar variables enter models because the bundled exogenous variables do not carry independently audited publication vintages.

## Generalization results

{best.to_markdown(index=False)}

## Decision-value results

{economic.to_markdown(index=False)}

Battery schedules are fixed from the forecast before observing delivery prices. Realized revenue includes 90% round-trip efficiency, EUR 0.50/MWh transaction costs, EUR 2.00/MWh degradation, 1 MW/1 MWh limits, and equal initial/terminal state of charge. The perfect-foresight schedule is used only as a non-rankable regret bound.

## Regime analysis

Regimes are ex-post evaluation labels, never model inputs. This price-only release reports negative prices, top-5% absolute-price spikes, peak hours, weekends, and fixed-date holidays. Energy-crisis, high-renewable, and extreme-weather labels are **not claimed** for these 2016/2017 tests because auditable realized weather and renewable series were not bundled. Adding unverifiable labels would weaken the forecast-time audit.

## Limitations

- Cross-zone experiments freeze the Phase 2 JointCurve specification instead of retuning on test outcomes.
- Pooled and transfer experiments establish geographic robustness, not causal transfer.
- Historical files are immutable checksum-pinned snapshots, but they are retrospective simulations, not live submissions.
- A DOI is not minted automatically: Zenodo account authorization is an external release step.
"""
    (REPORTS / "PHASE3_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    (REPORTS / "DATA_CARD.md").write_text(
        """# Data card

Source: Lago et al. epftoolbox European day-ahead price files, Zenodo record 4624805. Zones: DE, FR, BE, and Nordic system price (NP). Raw source checksums and retrieval metadata are stored beside the ignored raw files. Models use prices only; bundled load and generation columns are excluded because their forecast-time publication vintages were not independently audited. Test years are DE 2017, FR 2016, BE 2016, and NP 2017. NP 2018 is excluded because the source ends on 24 December.
""", encoding="utf-8"
    )
    (REPORTS / "MODEL_CARDS.md").write_text(
        """# Model cards

## Local JointCurve

Multi-task elastic net trained on four years from the target zone. It predicts all 24 auction products jointly from complete D-1, D-2, and D-7 curves plus lag-curve summaries and calendar cycles.

## Pooled JointCurve

The same estimator trained on the union of all four zones with a zone indicator. It tests whether broader geographic data improves a target zone without accessing its test year.

## Transfer JointCurve

The estimator is pretrained on the three source zones and then fine-tuned on the final 365 target-zone calibration days. Warm-starting transfers coefficients; target test outcomes are never used.

All three are intended for benchmark research, not unattended bidding. They do not ingest load, weather, outages, flows, fuel, or carbon until forecast-time vintages are traceable.
""", encoding="utf-8"
    )


def main() -> None:
    if not PREDICTIONS.exists():
        raise FileNotFoundError(f"Missing immutable prediction panel: {PREDICTIONS}")
    METRICS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_parquet(PREDICTIONS)
    general = generalization_metrics(predictions)
    general.to_csv(METRICS / "phase3_generalization_metrics.csv", index=False)
    regimes = regime_metrics(predictions)
    regimes.to_csv(METRICS / "phase3_regime_metrics.csv", index=False)
    daily, decision = evaluate_dispatch(predictions, BatterySpec())
    daily.to_parquet(ROOT / "results" / "predictions" / "phase3_dispatch_daily.parquet", index=False)
    decision.to_csv(METRICS / "phase3_decision_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    general.pivot(index="zone", columns="model", values="MAE").plot.bar(ax=axes[0])
    axes[0].set_ylabel("MAE (EUR/MWh)"); axes[0].set_title("Cross-zone forecast accuracy")
    decision.pivot(index="zone", columns="model", values="revenue_eur").plot.bar(ax=axes[1])
    axes[1].set_ylabel("Annual realized value (EUR/MW)"); axes[1].set_title("Non-anticipative battery value")
    fig.tight_layout(); fig.savefig(PLOTS / "phase3_generalization_and_value.png", dpi=180); plt.close(fig)
    _write_reports(general, decision, regimes)

    artifacts = [
        PREDICTIONS, METRICS / "phase3_generalization_metrics.csv",
        METRICS / "phase3_regime_metrics.csv", METRICS / "phase3_decision_metrics.csv",
        PLOTS / "phase3_generalization_and_value.png", REPORTS / "PHASE3_RESEARCH_REPORT.md",
        REPORTS / "DATA_CARD.md", REPORTS / "MODEL_CARDS.md",
    ]
    manifest = {
        "experiment_version": "3.0.0", "prediction_panel_immutable": True,
        "forecast_time_exogenous_features": [],
        "excluded_unverifiable_regimes": ["energy_crisis", "high_renewable", "extreme_weather"],
        "artifacts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size} for p in artifacts],
    }
    (METRICS / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(general.to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
