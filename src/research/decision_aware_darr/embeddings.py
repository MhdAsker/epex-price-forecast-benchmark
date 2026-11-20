"""Pinned Kumo embedding extraction and resumable batch loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.research.decision_aware_darr.causal import l2_normalize, window_id
from src.research.decision_aware_darr.storage import verify_batch_manifest
from src.research.kumo.data import IssuanceWindow, PriceStandardizer, validate_canonical_hourly


def extract_embeddings(
    model,
    frame: pd.DataFrame,
    origins: Sequence[IssuanceWindow],
    standardizer: PriceStandardizer,
    *,
    batch_size: int = 8,
) -> np.ndarray:
    import torch
    from sdk.forecasting import embed_batch  # type: ignore

    ordered = validate_canonical_hourly(frame)
    values = ordered.price_eur_mwh.to_numpy(dtype=np.float32)
    device = next(model.parameters()).device
    embeddings = []
    model.eval()
    for start in range(0, len(origins), batch_size):
        batch = origins[start:start + batch_size]
        contexts = np.stack([
            standardizer.transform(values[item.context_start:item.context_stop]) for item in batch
        ])[:, None, :]
        mask = torch.ones((len(batch), contexts.shape[-1]), dtype=torch.bool, device=device)
        with torch.no_grad():
            output = embed_batch(model, torch.from_numpy(contexts).float().to(device), mask)
        embeddings.append(output.detach().cpu().numpy().astype(np.float32))
    return l2_normalize(np.concatenate(embeddings))


def load_embedding_store(
    *,
    root: Path,
    manifest_dir: Path,
    expected_ids: Sequence[str],
    config_hash: str,
    source_hash: str,
) -> np.ndarray:
    collected: dict[str, np.ndarray] = {}
    for path in sorted(manifest_dir.glob("embedding_*.json")):
        valid, reason = verify_batch_manifest(path, root, config_hash, source_hash)
        if not valid:
            raise ValueError(f"invalid embedding batch {path.name}: {reason}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        output = root / manifest["outputs"][0]["path"]
        with np.load(output, allow_pickle=False) as payload:
            ids = payload["ids"].astype(str)
            embeddings = payload["embeddings"].astype(np.float32)
        if len(ids) != len(embeddings):
            raise ValueError(f"embedding ID/array mismatch in {output}")
        for identifier, embedding in zip(ids, embeddings):
            if identifier in collected:
                raise ValueError(f"duplicate embedding identifier: {identifier}")
            collected[identifier] = embedding
    missing = [identifier for identifier in expected_ids if identifier not in collected]
    extra = sorted(set(collected) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"embedding store mismatch: missing={len(missing)}, extra={len(extra)}")
    return np.stack([collected[identifier] for identifier in expected_ids]).astype(np.float32)


def embedding_ids(origins: Sequence[IssuanceWindow]) -> list[str]:
    return [window_id(item) for item in origins]
