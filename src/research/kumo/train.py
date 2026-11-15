"""Verified head-only Kumo training built on NVIDIA's pinned implementation."""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.research.kumo.data import KumoPriceDataset, PriceStandardizer

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class KumoPaths:
    official_checkout: Path
    backbone_snapshot: Path
    checkpoint: Path
    standardizer: Path


@dataclass(frozen=True)
class LoadReport:
    checkpoint_keys: int
    compatible_keys: int
    skipped_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    head_keys: tuple[str, ...]
    compatible_head_keys: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pinned_paths(config: dict) -> KumoPaths:
    dep = ROOT / ".research_deps" / "Kumo-TS"
    forecast = dep / "Kumo-Forecast"
    hf = dep / "hf_cache"
    model_cfg = config["model"]
    backbone = (
        hf / "models--AutonLab--MOMENT-1-large" / "snapshots" / model_cfg["backbone_revision"]
    )
    checkpoint = (
        hf / "models--nvidia--Kumo-Forecast" / "snapshots"
        / model_cfg["checkpoint_revision"] / model_cfg["checkpoint_file"]
    )
    standardizer = checkpoint.parent / model_cfg["standardizer_file"]
    return KumoPaths(forecast, backbone, checkpoint, standardizer)


def load_official_standardizer(config: dict) -> PriceStandardizer:
    import joblib

    path = resolve_pinned_paths(config).standardizer
    if not path.exists() or sha256_file(path) != config["model"]["standardizer_sha256"]:
        raise ValueError("NVIDIA standardizer is missing or checksum-mismatched")
    payload = joblib.load(path)
    return PriceStandardizer(mean=float(payload["mean"][0]), std=float(payload["std"][0]))


def build_verified_model(config: dict, device: str = "cpu") -> tuple[Any, LoadReport]:
    import torch

    paths = resolve_pinned_paths(config)
    for path in (paths.official_checkout, paths.backbone_snapshot, paths.checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Pinned Kumo dependency missing: {path}")
    if sha256_file(paths.checkpoint) != config["model"]["checkpoint_sha256"]:
        raise ValueError("Kumo checkpoint checksum mismatch")
    sys.path.insert(0, str(paths.official_checkout))
    from model import build_model  # type: ignore

    model = build_model(
        model_name=str(paths.backbone_snapshot), forecast_horizon=72, seq_len=512,
        freeze_encoder=True, freeze_embedder=True, freeze_head=False,
        use_cross_channel=False, local_files_only=True, device=device,
    )
    raw = torch.load(paths.checkpoint, map_location=device)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    skipped = tuple(
        key for key, value in state.items()
        if key not in model_state or tuple(value.shape) != tuple(model_state[key].shape)
    )
    loaded = model.load_state_dict(compatible, strict=False)
    head_keys = tuple(key for key in state if key.startswith("head.") or ".head." in key)
    compatible_head = tuple(key for key in head_keys if key in compatible)
    report = LoadReport(
        checkpoint_keys=len(state), compatible_keys=len(compatible), skipped_keys=skipped,
        missing_keys=tuple(loaded.missing_keys), unexpected_keys=tuple(loaded.unexpected_keys),
        head_keys=head_keys, compatible_head_keys=compatible_head,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
    )
    if skipped or report.missing_keys or report.unexpected_keys:
        raise RuntimeError(f"Checkpoint is not fully compatible: {report}")
    if set(head_keys) != set(compatible_head) or not head_keys:
        raise RuntimeError("Pretrained forecasting-head tensors were not loaded")
    non_head_trainable = [name for name, p in model.named_parameters() if p.requires_grad and not name.startswith("head.")]
    if non_head_trainable:
        raise RuntimeError(f"Non-head parameters are trainable: {non_head_trainable}")
    return model, report


def head_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.startswith("head.")}


def head_change(before: dict[str, Any], after: dict[str, Any]) -> dict:
    import torch

    rows = {}
    for name in sorted(before):
        delta = after[name] - before[name]
        rows[name] = {
            "l2_before": float(torch.linalg.vector_norm(before[name])),
            "l2_delta": float(torch.linalg.vector_norm(delta)),
            "max_abs_delta": float(torch.max(torch.abs(delta))),
        }
    return rows


def train_head(
    model: Any,
    train_dataset: KumoPriceDataset,
    validation_dataset: KumoPriceDataset,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    output_dir: Path,
    max_train_windows: int | None = None,
    max_validation_windows: int | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, Subset

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_data = train_dataset
    val_data = validation_dataset
    if max_train_windows is not None:
        train_data = Subset(train_dataset, range(min(len(train_dataset), max_train_windows)))
    if max_validation_windows is not None:
        val_data = Subset(validation_dataset, range(min(len(validation_dataset), max_validation_windows)))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    criterion = torch.nn.MSELoss()
    device = next(model.parameters()).device
    before = head_state(model)
    best = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_enc=x.float(), input_mask=mask).forecast
            loss = criterion(prediction, y.float())
            loss.backward(); optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval(); val_losses = []; val_maes = []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                prediction = model(x_enc=x.float(), input_mask=mask).forecast
                val_losses.append(float(criterion(prediction, y.float())))
                pred_raw = validation_dataset.standardizer.inverse(prediction.detach().cpu().numpy())
                y_raw = validation_dataset.standardizer.inverse(y.detach().cpu().numpy())
                val_maes.append(float(np.mean(np.abs(pred_raw - y_raw))))
        row = {
            "epoch": epoch, "train_mse_standardized": float(np.mean(train_losses)),
            "validation_mse_standardized": float(np.mean(val_losses)),
            "validation_mae_eur_mwh": float(np.mean(val_maes)),
        }
        history.append(row)
        if row["validation_mae_eur_mwh"] < best:
            best = row["validation_mae_eur_mwh"]; best_epoch = epoch
            best_state = head_state(model)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state, strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_head.pt"
    torch.save(best_state, checkpoint_path)
    result = {
        "best_epoch": best_epoch, "best_validation_mae_eur_mwh": best,
        "history": history, "runtime_seconds": time.perf_counter() - started,
        "train_windows": len(train_data), "validation_windows": len(val_data),
        "head_change": head_change(before, best_state),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def save_standardizer(standardizer: PriceStandardizer, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "price_standardizer.json"
    path.write_text(json.dumps(asdict(standardizer), indent=2), encoding="utf-8")
    return path
