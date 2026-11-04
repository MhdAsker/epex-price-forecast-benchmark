"""One-command Phase 3 paper artifact rebuild and reproducibility bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from finalize_phase3 import main as finalize

ROOT = Path(__file__).resolve().parent
VERSION = "0.3.0"


def main() -> None:
    finalize()
    release_dir = ROOT / "dist"
    release_dir.mkdir(exist_ok=True)
    include = [
        "README.md", "LICENSE", "requirements.txt", "CITATION.cff", ".zenodo.json",
        "config/forecast_protocol.json", "config/phase2.json", "config/phase3.json",
        "results/metrics/phase2_point_metrics.csv", "results/metrics/phase2_probabilistic_metrics.csv",
        "results/metrics/phase2_pairwise_tests.csv", "results/metrics/phase2_manifest.json",
        "results/metrics/phase3_generalization_metrics.csv", "results/metrics/phase3_regime_metrics.csv",
        "results/metrics/phase3_decision_metrics.csv", "results/metrics/phase3_manifest.json",
        "results/predictions/phase3_generalization_predictions.parquet",
        "results/predictions/phase3_dispatch_daily.parquet",
        "results/plots/phase2_point_mae_by_year.png", "results/plots/phase2_probabilistic_calibration.png",
        "results/plots/phase2_dm_matrix.png", "results/plots/phase3_generalization_and_value.png",
        "reports/PHASE3_RESEARCH_REPORT.md", "reports/DATA_CARD.md", "reports/MODEL_CARDS.md",
    ]
    archive = release_dir / f"epex-benchmark-{VERSION}.zip"
    checksums = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in include:
            path = ROOT / name
            if not path.exists():
                raise FileNotFoundError(f"Release artifact missing: {name}")
            bundle.write(path, name)
            checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle.writestr("SHA256SUMS.json", json.dumps(checksums, indent=2))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (release_dir / f"epex-benchmark-{VERSION}.zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)\nSHA256 {digest}")


if __name__ == "__main__":
    main()
