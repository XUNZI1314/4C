from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


SMART_POCKET_COLUMNS = [
    "smart_rank_score",
    "smart_rank_order",
    "smart_rank_label",
    "smart_rank_reason",
    "smart_hotspot_ratio",
    "smart_method_support",
    "smart_confidence_support",
    "smart_residue_support",
    "smart_external_support",
    "smart_external_exact_ratio",
    "smart_external_verified_ratio",
    "smart_external_mapping_quality",
    "smart_evidence_anchor_support",
    "smart_evidence_anchor_risk",
    "smart_conservation_support",
    "smart_burial_support",
    "smart_exposure_penalty",
]


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _normalize_series(values: pd.Series, default: float = 0.5) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(np.full(len(values), default, dtype=float), index=values.index)

    filled = numeric.fillna(float(numeric.dropna().median()))
    minimum = float(filled.min())
    maximum = float(filled.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.full(len(filled), default, dtype=float), index=filled.index)
    return (filled - minimum) / (maximum - minimum)


def _normalize_positive_series(values: pd.Series, default: float = 0.5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any() or not numeric.fillna(0.0).gt(0.0).any():
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return _normalize_series(numeric.fillna(0.0), default=default)


def _center_closeness_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any() or not numeric.fillna(0.0).gt(0.0).any():
        return pd.Series(np.full(len(values), 0.5, dtype=float), index=values.index)
    return 1.0 - _normalize_series(numeric, default=0.5)


def _route_bonus(route_text: object) -> float:
    route = str(route_text or "").strip().lower()
    bonus = 0.0
    if "consensus" in route:
        bonus += 0.24
    if "multiscale" in route:
        bonus += 0.20
    if "p2rank" in route:
        bonus += 0.18
    if "external" in route or "evidence" in route:
        bonus += 0.15
    if "kvfinder" in route:
        bonus += 0.16
    if "ligand" in route:
        bonus += 0.12
    if "geometry" in route:
        bonus += 0.08
    return min(1.0, bonus)


def _rank_label(score: float) -> str:
    if score >= 0.72:
        return "high-priority"
    if score >= 0.54:
        return "promising"
    return "exploratory"


def _rank_reason(row: pd.Series) -> str:
    reasons: list[str] = []

    hotspot_ratio = float(row.get("smart_hotspot_ratio", 0.0) or 0.0)
    method_support = float(row.get("smart_method_support", 0.0) or 0.0)
    confidence_support = float(row.get("smart_confidence_support", 0.0) or 0.0)
    residue_support = float(row.get("smart_residue_support", 0.0) or 0.0)
    external_support = float(row.get("smart_external_support", 0.0) or 0.0)
    external_exact_ratio = float(row.get("smart_external_exact_ratio", 0.0) or 0.0)
    external_verified_ratio = float(row.get("smart_external_verified_ratio", 0.0) or 0.0)
    external_mapping_quality = float(row.get("smart_external_mapping_quality", 0.0) or 0.0)
    evidence_anchor_support = float(row.get("smart_evidence_anchor_support", 0.0) or 0.0)
    evidence_anchor_risk = float(row.get("smart_evidence_anchor_risk", 0.0) or 0.0)
    conservation_support = float(row.get("smart_conservation_support", 0.0) or 0.0)
    burial_support = float(row.get("smart_burial_support", 0.0) or 0.0)
    exposure_penalty = float(row.get("smart_exposure_penalty", 0.0) or 0.0)
    try:
        method_vote_count = int(float(row.get("method_vote_count", 0) or 0))
    except Exception:
        method_vote_count = 0
    detection_route = str(row.get("detection_route") or "").strip()

    if hotspot_ratio >= 0.34:
        reasons.append("热点重叠比例较高")
    elif hotspot_ratio > 0.0:
        reasons.append("存在热点支撑")

    if method_vote_count >= 3:
        reasons.append("多方法共识强")
    elif method_support >= 0.55:
        reasons.append("跨方法共识较稳定")

    if residue_support >= 0.60:
        reasons.append("口袋内部残基得分稳定")

    if confidence_support >= 0.62:
        reasons.append("结构置信度较好")

    if external_exact_ratio >= 0.20:
        reasons.append("命中外部关键位点映射")
    elif external_verified_ratio >= 0.20:
        reasons.append("外部位点与结构残基对齐可靠")
    elif external_support >= 0.48:
        reasons.append("外部功能位点证据支持")
    elif external_mapping_quality >= 0.65:
        reasons.append("外部位点映射质量较好")

    if evidence_anchor_support >= 0.40:
        reasons.append("direct evidence anchor present")
    elif evidence_anchor_risk >= 0.35:
        reasons.append("external evidence mostly neighborhood-expanded")

    if conservation_support >= 0.55:
        reasons.append("保守性信号支持")

    if burial_support >= 0.62:
        reasons.append("buried geometry support")
    elif exposure_penalty >= 0.55:
        reasons.append("shallow exposure penalty")

    if "p2rank" in detection_route.lower():
        reasons.append("P2Rank 预测与本地证据一致")

    if "multiscale" in detection_route.lower():
        reasons.append("多尺度筛选结果稳定")

    if not reasons:
        reasons.append("当前主要由几何与接触证据支持")

    return "；".join(reasons[:3])


def rank_detected_pockets(pocket_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pocket_df is None or getattr(pocket_df, "empty", True):
        return pd.DataFrame(columns=SMART_POCKET_COLUMNS)

    working = pocket_df.copy()
    summary_rows = []
    for pocket_id, group in working.groupby("pocket_id", sort=False):
        residue_count = max(1, int(len(group)))
        hotspot_ratio = float(_numeric_series(group, "is_hotspot", 0.0).astype(bool).mean())
        method_vote_count = int(_numeric_series(group, "method_vote_count", 1.0).max())
        method_support = min(float(method_vote_count) / 3.0, 1.0)
        confidence_support = float(_numeric_series(group, "confidence_score", 0.5).mean())
        seed_support = float(_numeric_series(group, "seed_support", 0.0).mean())
        external_support = float(_numeric_series(group, "external_support", 0.0).mean())
        external_confidence = float(_numeric_series(group, "external_confidence", 0.0).mean())
        external_exact_ratio = float(_numeric_series(group, "external_exact_match", 0.0).astype(bool).mean())
        external_verified_ratio = float(_numeric_series(group, "external_structure_verified", 0.0).astype(bool).mean())
        external_mapping_quality = float(_numeric_series(group, "external_mapping_quality", 0.0).mean())
        external_direct_anchor_ratio = float(_numeric_series(group, "external_direct_anchor", 0.0).astype(bool).mean())
        evidence_route_anchor_ratio = float(_numeric_series(group, "evidence_route_anchor", 0.0).astype(bool).mean())
        evidence_anchor_proximity = float(_numeric_series(group, "evidence_anchor_proximity", 0.0).max())
        conservation_support = float(_numeric_series(group, "conservation_support", 0.0).mean())
        conservation_confidence = float(_numeric_series(group, "conservation_confidence", 0.0).mean())
        contact_density = float(_numeric_series(group, "contact_count", 0.0).mean())
        depth_signal = _numeric_series(group, "depth_avg", np.nan)
        if depth_signal.isna().all():
            depth_signal = _numeric_series(group, "depth_max", 0.0)
        else:
            depth_signal = depth_signal.fillna(0.0)
        depth_support = float(depth_signal.mean())
        center_distance = float(_numeric_series(group, "center_distance", 0.0).mean())
        ligand_contact_ratio = float(_numeric_series(group, "ligand_contact_count", 0.0).gt(0.0).mean())
        external_signal = min(
            1.0,
            (0.34 * external_support)
            + (0.18 * external_confidence)
            + (0.16 * external_exact_ratio)
            + (0.10 * external_verified_ratio)
            + (0.07 * external_mapping_quality)
            + (0.15 * external_direct_anchor_ratio),
        )
        evidence_anchor_signal = min(
            1.0,
            (0.50 * external_direct_anchor_ratio)
            + (0.25 * evidence_route_anchor_ratio)
            + (0.25 * evidence_anchor_proximity),
        )
        evidence_anchor_risk = max(0.0, external_signal - evidence_anchor_signal)
        conservation_signal = min(1.0, (0.65 * conservation_support) + (0.35 * conservation_confidence))

        residue_signal = _numeric_series(group, "consensus_score", np.nan)
        if residue_signal.isna().all():
            residue_signal = _numeric_series(group, "residue_score", np.nan)
        if residue_signal.isna().all():
            residue_signal = _numeric_series(group, "precision_score", 0.0)
        residue_strength = float(residue_signal.fillna(0.0).mean())

        pocket_score = float(_numeric_series(group, "score", 0.0).mean())
        volume = float(_numeric_series(group, "volume", 0.0).iloc[0])
        route_bonus = max(_route_bonus(value) for value in group.get("detection_route", pd.Series(dtype=str)).tolist()) if "detection_route" in group.columns else 0.0

        summary_rows.append(
            {
                "pocket_id": pocket_id,
                "pocket_score_raw": pocket_score,
                "residue_strength_raw": residue_strength,
                "volume_raw": np.log1p(max(volume, 0.0)),
                "smart_hotspot_ratio": round(hotspot_ratio, 3),
                "smart_method_support": round(method_support, 3),
                "smart_confidence_support": round(confidence_support, 3),
                "smart_residue_support_raw": residue_strength,
                "smart_external_support": round(external_signal, 3),
                "smart_external_exact_ratio": round(external_exact_ratio, 3),
                "smart_external_verified_ratio": round(external_verified_ratio, 3),
                "smart_external_mapping_quality": round(external_mapping_quality, 3),
                "smart_evidence_anchor_support": round(evidence_anchor_signal, 3),
                "smart_evidence_anchor_risk": round(evidence_anchor_risk, 3),
                "smart_conservation_support": round(conservation_signal, 3),
                "contact_density_raw": contact_density,
                "depth_support_raw": depth_support,
                "center_distance_raw": center_distance,
                "ligand_contact_ratio": round(ligand_contact_ratio, 3),
                "seed_support_raw": seed_support,
                "route_bonus_raw": route_bonus,
                "method_vote_count": method_vote_count,
                "detection_route": ", ".join(
                    sorted({str(value).strip() for value in group.get("detection_route", pd.Series(dtype=str)).dropna().tolist() if str(value).strip()})
                ) if "detection_route" in group.columns else "",
                "residue_count": residue_count,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return working

    summary_df["pocket_score_norm"] = _normalize_series(summary_df["pocket_score_raw"], default=0.5)
    summary_df["smart_residue_support"] = _normalize_series(summary_df["smart_residue_support_raw"], default=0.5)
    summary_df["volume_norm"] = _normalize_series(summary_df["volume_raw"], default=0.5)
    summary_df["seed_support_norm"] = _normalize_series(summary_df["seed_support_raw"], default=0.4)
    contact_density_norm = _normalize_positive_series(summary_df["contact_density_raw"], default=0.5)
    depth_support_norm = _normalize_positive_series(summary_df["depth_support_raw"], default=0.45)
    center_closeness = _center_closeness_series(summary_df["center_distance_raw"])
    summary_df["smart_burial_support"] = (
        0.45 * contact_density_norm
        + 0.35 * depth_support_norm
        + 0.20 * center_closeness
    ).clip(lower=0.0, upper=1.0)
    rescue_signal = pd.concat(
        [
            summary_df["smart_hotspot_ratio"],
            summary_df["smart_external_support"],
            summary_df["smart_evidence_anchor_support"],
            summary_df["ligand_contact_ratio"],
            0.35 * summary_df["smart_method_support"],
        ],
        axis=1,
    ).max(axis=1)
    summary_df["smart_exposure_penalty"] = (
        (1.0 - summary_df["smart_burial_support"])
        * (1.0 - rescue_signal.clip(lower=0.0, upper=1.0))
    ).clip(lower=0.0, upper=1.0)

    summary_df["smart_rank_score"] = (
        0.20 * summary_df["pocket_score_norm"]
        + 0.19 * summary_df["smart_hotspot_ratio"]
        + 0.16 * summary_df["smart_method_support"]
        + 0.14 * summary_df["smart_residue_support"]
        + 0.09 * summary_df["smart_confidence_support"]
        + 0.08 * summary_df["smart_external_support"]
        + 0.04 * summary_df["smart_evidence_anchor_support"]
        + 0.04 * summary_df["smart_conservation_support"]
        + 0.06 * summary_df["seed_support_norm"]
        + 0.04 * summary_df["smart_burial_support"]
        + 0.04 * summary_df["route_bonus_raw"]
        - 0.05 * summary_df["smart_exposure_penalty"]
        - 0.03 * summary_df["smart_evidence_anchor_risk"]
    ).clip(lower=0.0, upper=1.0)

    summary_df = summary_df.sort_values(
        ["smart_rank_score", "smart_hotspot_ratio", "smart_method_support", "pocket_score_raw", "residue_count"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary_df["smart_rank_order"] = range(1, len(summary_df) + 1)
    summary_df["smart_rank_label"] = summary_df["smart_rank_score"].map(lambda value: _rank_label(float(value)))
    summary_df["smart_rank_reason"] = summary_df.apply(_rank_reason, axis=1)
    summary_df["smart_rank_score"] = summary_df["smart_rank_score"].round(3)
    summary_df["smart_residue_support"] = summary_df["smart_residue_support"].round(3)
    summary_df["smart_burial_support"] = summary_df["smart_burial_support"].round(3)
    summary_df["smart_exposure_penalty"] = summary_df["smart_exposure_penalty"].round(3)
    summary_df["smart_evidence_anchor_support"] = summary_df["smart_evidence_anchor_support"].round(3)
    summary_df["smart_evidence_anchor_risk"] = summary_df["smart_evidence_anchor_risk"].round(3)

    pocket_level_columns = [
        "pocket_id",
        "smart_rank_score",
        "smart_rank_order",
        "smart_rank_label",
        "smart_rank_reason",
        "smart_hotspot_ratio",
        "smart_method_support",
        "smart_confidence_support",
        "smart_residue_support",
        "smart_external_support",
        "smart_external_exact_ratio",
        "smart_external_verified_ratio",
        "smart_external_mapping_quality",
        "smart_evidence_anchor_support",
        "smart_evidence_anchor_risk",
        "smart_conservation_support",
        "smart_burial_support",
        "smart_exposure_penalty",
    ]
    merged = working.merge(summary_df[pocket_level_columns], on="pocket_id", how="left")
    return merged.sort_values(
        ["smart_rank_order", "score", "residue_score", "consensus_score", "pocket_id", "chain", "resid"],
        ascending=[True, False, False, False, True, True, True],
    ).reset_index(drop=True)
