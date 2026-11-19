"""Diagnostics for the frozen native-DARR experiment.

The archived prediction panel retained the direct and blended forecasts but not
the retrieval-only component. NVIDIA's registered blend is linear, so the
retrieval forecast can be recovered exactly without rerunning or retuning DARR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.kumo.data import KumoPriceDataset, PriceStandardizer, validate_canonical_hourly


def reconstruct_darr_components(
    panel: pd.DataFrame,
    *,
    alpha: float = 0.01,
    direct_model: str = "Kumo fine-tuned",
    blend_model: str = "Kumo fine-tuned + DARR",
) -> pd.DataFrame:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    keys = ["delivery_date", "hour"]
    direct = panel.loc[panel.model == direct_model, keys + ["actual", "prediction"]].rename(
        columns={"prediction": "direct_prediction"}
    )
    blend = panel.loc[panel.model == blend_model, keys + ["actual", "prediction"]].rename(
        columns={"actual": "blend_actual", "prediction": "blend_prediction"}
    )
    merged = direct.merge(blend, on=keys, validate="one_to_one")
    if not np.allclose(merged.actual, merged.blend_actual, rtol=0, atol=1e-8):
        raise ValueError("direct and DARR panels do not contain identical targets")
    merged["retrieval_prediction"] = (
        merged.blend_prediction - alpha * merged.direct_prediction
    ) / (1.0 - alpha)
    merged["test_year"] = pd.to_datetime(merged.delivery_date).dt.year
    return merged.drop(columns="blend_actual")


def daily_darr_diagnostics(components: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for date, day in components.groupby("delivery_date", sort=True):
        day = day.sort_values("hour")
        actual = day.actual.to_numpy(dtype=float)
        row: dict = {
            "delivery_date": date,
            "test_year": int(day.test_year.iloc[0]),
            "actual_spread": float(actual.max() - actual.min()),
            "actual_low_hour": int(day.iloc[np.argmin(actual)].hour),
            "actual_high_hour": int(day.iloc[np.argmax(actual)].hour),
        }
        for label, column in (
            ("direct", "direct_prediction"),
            ("retrieval", "retrieval_prediction"),
            ("blend", "blend_prediction"),
        ):
            pred = day[column].to_numpy(dtype=float)
            row[f"{label}_mae"] = float(np.mean(np.abs(actual - pred)))
            row[f"{label}_bias"] = float(np.mean(pred - actual))
            row[f"{label}_spread"] = float(pred.max() - pred.min())
            row[f"{label}_spread_error"] = float((pred.max() - pred.min()) - row["actual_spread"])
            row[f"{label}_rank_correlation"] = float(
                pd.Series(actual).corr(pd.Series(pred), method="spearman")
            )
            row[f"{label}_low_hour"] = int(day.iloc[np.argmin(pred)].hour)
            row[f"{label}_high_hour"] = int(day.iloc[np.argmax(pred)].hour)
            row[f"{label}_low_correct"] = row[f"{label}_low_hour"] == row["actual_low_hour"]
            row[f"{label}_high_correct"] = row[f"{label}_high_hour"] == row["actual_high_hour"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_darr(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for year, group in [("combined", daily), *[(str(y), g) for y, g in daily.groupby("test_year")]]:
        for component in ("direct", "retrieval", "blend"):
            rows.append({
                "period": year,
                "component": component,
                "MAE": float(group[f"{component}_mae"].mean()),
                "bias": float(group[f"{component}_bias"].mean()),
                "daily_rank_correlation": float(group[f"{component}_rank_correlation"].mean()),
                "mean_predicted_spread": float(group[f"{component}_spread"].mean()),
                "mean_spread_error": float(group[f"{component}_spread_error"].mean()),
                "low_hour_accuracy": float(group[f"{component}_low_correct"].mean()),
                "high_hour_accuracy": float(group[f"{component}_high_correct"].mean()),
                "n_days": int(len(group)),
            })
    return pd.DataFrame(rows)


def revenue_gain_concentration(battery_daily: pd.DataFrame) -> pd.DataFrame:
    pivot = battery_daily.pivot(index="delivery_date", columns="model", values="revenue_eur")
    gain = (pivot["Kumo fine-tuned + DARR"] - pivot["Kumo fine-tuned"]).sort_values(ascending=False)
    positive = gain[gain > 0]
    rows = []
    for period, subset in [("combined", gain), *[(str(y), g) for y, g in gain.groupby(pd.to_datetime(gain.index).year)]]:
        positive_subset = subset[subset > 0].sort_values(ascending=False)
        positive_total = float(positive_subset.sum())
        rows.append({
            "period": period,
            "net_revenue_gain_eur": float(subset.sum()),
            "positive_gain_eur": positive_total,
            "downside_eur": float(subset[subset < 0].sum()),
            "days_improved": int((subset > 0).sum()),
            "days_worsened": int((subset < 0).sum()),
            "top_10_days_share_of_positive_gain": (
                float(positive_subset.head(10).sum() / positive_total) if positive_total else np.nan
            ),
            "top_20_days_share_of_positive_gain": (
                float(positive_subset.head(20).sum() / positive_total) if positive_total else np.nan
            ),
            "largest_daily_gain_eur": float(subset.max()),
            "largest_daily_loss_eur": float(subset.min()),
        })
    return pd.DataFrame(rows)


def neighbor_quality_diagnostics(
    model,
    frame: pd.DataFrame,
    context_origins,
    query_origins,
    standardizer: PriceStandardizer,
    *,
    k: int = 64,
    temperature: float = 0.05,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Recreate frozen DARR similarities and quantify neighbor-target quality."""
    import torch
    from torch.utils.data import DataLoader
    from sdk.forecasting import build_context_memory, embed_batch, l2_normalize  # type: ignore

    ordered = validate_canonical_hourly(frame)
    values = ordered.price_eur_mwh.to_numpy(dtype=np.float32)
    device = next(model.parameters()).device
    loader = DataLoader(
        KumoPriceDataset(frame, context_origins, standardizer),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    db_embeddings, db_targets = build_context_memory(model, loader, device, cosine=True)
    query_embeddings = []
    for start in range(0, len(query_origins), batch_size):
        batch = query_origins[start:start + batch_size]
        x = np.stack([
            standardizer.transform(values[o.context_start:o.context_stop]) for o in batch
        ])[:, None, :]
        mask = torch.ones((len(batch), x.shape[-1]), dtype=torch.bool, device=device)
        with torch.no_grad():
            embedding = embed_batch(model, torch.from_numpy(x).float().to(device), mask)
        query_embeddings.append(embedding.detach().cpu().numpy().astype(np.float32))
    queries = l2_normalize(np.nan_to_num(np.concatenate(query_embeddings), copy=False))
    similarity = queries @ db_embeddings.T
    effective_k = min(k, len(context_origins))
    indices = np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]
    row_indices = np.arange(len(query_origins))[:, None]
    neighbor_sims = similarity[row_indices, indices]
    order = np.argsort(-neighbor_sims, axis=1)
    indices = indices[row_indices, order]
    neighbor_sims = similarity[row_indices, indices]
    logits = neighbor_sims / max(temperature, 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)

    rows = []
    for query_index, origin in enumerate(query_origins):
        top_index = int(indices[query_index, 0])
        top_target = standardizer.inverse(db_targets[top_index, 0, :24])
        actual = values[origin.target_start:origin.target_start + 24]
        top_origin = context_origins[top_index]
        rows.append({
            "delivery_date": origin.delivery_date.date().isoformat(),
            "test_year": int(origin.delivery_date.year),
            "top1_context_delivery_date": top_origin.delivery_date.date().isoformat(),
            "top1_similarity": float(neighbor_sims[query_index, 0]),
            "mean_topk_similarity": float(neighbor_sims[query_index].mean()),
            "min_topk_similarity": float(neighbor_sims[query_index, -1]),
            "effective_neighbor_count": float(1.0 / np.square(weights[query_index]).sum()),
            "top1_weight": float(weights[query_index, 0]),
            "top1_target_mae": float(np.mean(np.abs(top_target - actual))),
            "top1_target_bias": float(np.mean(top_target - actual)),
            "top1_target_spread": float(top_target.max() - top_target.min()),
            "actual_spread": float(actual.max() - actual.min()),
            "top1_same_calendar_month": bool(
                top_origin.delivery_date.month == origin.delivery_date.month
            ),
            "top1_age_days": int((origin.delivery_date - top_origin.delivery_date).days),
        })
    return pd.DataFrame(rows)
