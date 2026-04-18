from typing import Any, Dict, List

import pandas as pd


POCKET_RANKING_COMPARISON_COLUMNS = [
    "pocket_id",
    "status",
    "base_rank",
    "enhanced_rank",
    "rank_delta",
    "base_score",
    "enhanced_score",
    "score_delta",
    "base_label",
    "enhanced_label",
    "base_reason",
    "enhanced_reason",
    "base_evidence_quality_label",
    "enhanced_evidence_quality_label",
    "base_evidence_quality_score",
    "enhanced_evidence_quality_score",
    "evidence_quality_delta",
    "base_evidence_anchor_support",
    "enhanced_evidence_anchor_support",
    "base_evidence_anchor_risk",
    "enhanced_evidence_anchor_risk",
    "base_conservation_support",
    "enhanced_conservation_support",
    "base_conservation_mean",
    "enhanced_conservation_mean",
    "residue_count",
]


def _hotspot_set(df: pd.DataFrame) -> set[tuple[str, int, str]]:
    hotspot_set = set()
    if df is None or getattr(df, "empty", True):
        return hotspot_set

    for row in df.itertuples(index=False):
        try:
            hotspot_set.add((str(row.chain), int(row.resid), str(row.resname)))
        except Exception:
            continue
    return hotspot_set


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return int(numeric)


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _ranking_summary_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or getattr(summary_df, "empty", True) or "pocket_id" not in summary_df.columns:
        return pd.DataFrame(columns=[
            "pocket_id",
            "rank_order",
            "score",
            "label",
            "reason",
            "evidence_quality_label",
            "evidence_quality_score",
            "evidence_anchor_support",
            "evidence_anchor_risk",
            "conservation_support",
            "conservation_mean",
            "residue_count",
        ])

    view = summary_df.copy()
    view = view[view["pocket_id"].notna()].copy()
    if view.empty:
        return pd.DataFrame(columns=[
            "pocket_id",
            "rank_order",
            "score",
            "label",
            "reason",
            "evidence_quality_label",
            "evidence_quality_score",
            "evidence_anchor_support",
            "evidence_anchor_risk",
            "conservation_support",
            "conservation_mean",
            "residue_count",
        ])

    view["pocket_id"] = view["pocket_id"].astype(str)
    row_rank = pd.Series(range(1, len(view) + 1), index=view.index, dtype=float)
    if "smart_rank_order" in view.columns:
        rank_order = pd.to_numeric(view["smart_rank_order"], errors="coerce")
        rank_order = rank_order.where(rank_order.gt(0), row_rank)
        view["rank_order"] = rank_order.fillna(row_rank)
    else:
        view["rank_order"] = row_rank

    if "smart_rank_score" in view.columns:
        view["score"] = pd.to_numeric(view["smart_rank_score"], errors="coerce")
    elif "score" in view.columns:
        view["score"] = pd.to_numeric(view["score"], errors="coerce")
    else:
        view["score"] = 0.0

    text_defaults = {
        "label": "smart_rank_label",
        "reason": "smart_rank_reason",
        "evidence_quality_label": "evidence_quality_label",
    }
    for output_column, source_column in text_defaults.items():
        view[output_column] = view[source_column] if source_column in view.columns else None

    numeric_defaults = {
        "evidence_quality_score": "evidence_quality_score",
        "evidence_anchor_support": "smart_evidence_anchor_support",
        "evidence_anchor_risk": "smart_evidence_anchor_risk",
        "conservation_support": "smart_conservation_support",
        "conservation_mean": "conservation_support_mean",
        "residue_count": "residue_count",
    }
    for output_column, source_column in numeric_defaults.items():
        if source_column in view.columns:
            view[output_column] = pd.to_numeric(view[source_column], errors="coerce")
        else:
            view[output_column] = 0.0

    columns = [
        "pocket_id",
        "rank_order",
        "score",
        "label",
        "reason",
        "evidence_quality_label",
        "evidence_quality_score",
        "evidence_anchor_support",
        "evidence_anchor_risk",
        "conservation_support",
        "conservation_mean",
        "residue_count",
    ]
    return (
        view[columns]
        .sort_values(["rank_order", "score", "pocket_id"], ascending=[True, False, True])
        .drop_duplicates(subset=["pocket_id"], keep="first")
        .reset_index(drop=True)
    )


def compare_pocket_ranking_summaries(base_summary: pd.DataFrame, enhanced_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare two pocket-summary rankings; positive rank_delta means the enhanced run moved up."""
    base_view = _ranking_summary_view(base_summary).set_index("pocket_id", drop=False)
    enhanced_view = _ranking_summary_view(enhanced_summary).set_index("pocket_id", drop=False)
    pocket_ids = sorted(set(base_view.index.tolist()) | set(enhanced_view.index.tolist()))
    if not pocket_ids:
        return pd.DataFrame(columns=POCKET_RANKING_COMPARISON_COLUMNS)

    rows = []
    for pocket_id in pocket_ids:
        base_row = base_view.loc[pocket_id] if pocket_id in base_view.index else None
        enhanced_row = enhanced_view.loc[pocket_id] if pocket_id in enhanced_view.index else None

        base_rank = _safe_int(base_row.get("rank_order")) if base_row is not None else None
        enhanced_rank = _safe_int(enhanced_row.get("rank_order")) if enhanced_row is not None else None
        base_score = _safe_float(base_row.get("score")) if base_row is not None else None
        enhanced_score = _safe_float(enhanced_row.get("score")) if enhanced_row is not None else None
        rank_delta = base_rank - enhanced_rank if base_rank is not None and enhanced_rank is not None else None
        score_delta = enhanced_score - base_score if base_score is not None and enhanced_score is not None else None
        base_evidence_quality_score = (
            _safe_float(base_row.get("evidence_quality_score")) if base_row is not None else None
        )
        enhanced_evidence_quality_score = (
            _safe_float(enhanced_row.get("evidence_quality_score")) if enhanced_row is not None else None
        )
        evidence_quality_delta = (
            enhanced_evidence_quality_score - base_evidence_quality_score
            if base_evidence_quality_score is not None and enhanced_evidence_quality_score is not None
            else None
        )

        if base_rank is None:
            status = "new"
        elif enhanced_rank is None:
            status = "removed"
        elif rank_delta and rank_delta > 0:
            status = "moved_up"
        elif rank_delta and rank_delta < 0:
            status = "moved_down"
        elif score_delta is not None and abs(score_delta) > 1e-9:
            status = "score_changed"
        else:
            status = "unchanged"

        rows.append(
            {
                "pocket_id": pocket_id,
                "status": status,
                "base_rank": base_rank,
                "enhanced_rank": enhanced_rank,
                "rank_delta": rank_delta,
                "base_score": round(base_score, 3) if base_score is not None else None,
                "enhanced_score": round(enhanced_score, 3) if enhanced_score is not None else None,
                "score_delta": round(score_delta, 3) if score_delta is not None else None,
                "base_label": _safe_text(base_row.get("label")) if base_row is not None else None,
                "enhanced_label": _safe_text(enhanced_row.get("label")) if enhanced_row is not None else None,
                "base_reason": _safe_text(base_row.get("reason")) if base_row is not None else None,
                "enhanced_reason": _safe_text(enhanced_row.get("reason")) if enhanced_row is not None else None,
                "base_evidence_quality_label": _safe_text(base_row.get("evidence_quality_label")) if base_row is not None else None,
                "enhanced_evidence_quality_label": _safe_text(enhanced_row.get("evidence_quality_label")) if enhanced_row is not None else None,
                "base_evidence_quality_score": round(base_evidence_quality_score, 3)
                if base_evidence_quality_score is not None
                else None,
                "enhanced_evidence_quality_score": round(enhanced_evidence_quality_score, 3)
                if enhanced_evidence_quality_score is not None
                else None,
                "evidence_quality_delta": round(evidence_quality_delta, 3)
                if evidence_quality_delta is not None
                else None,
                "base_evidence_anchor_support": round(_safe_float(base_row.get("evidence_anchor_support")) or 0.0, 3) if base_row is not None else None,
                "enhanced_evidence_anchor_support": round(_safe_float(enhanced_row.get("evidence_anchor_support")) or 0.0, 3) if enhanced_row is not None else None,
                "base_evidence_anchor_risk": round(_safe_float(base_row.get("evidence_anchor_risk")) or 0.0, 3) if base_row is not None else None,
                "enhanced_evidence_anchor_risk": round(_safe_float(enhanced_row.get("evidence_anchor_risk")) or 0.0, 3) if enhanced_row is not None else None,
                "base_conservation_support": round(_safe_float(base_row.get("conservation_support")) or 0.0, 3) if base_row is not None else None,
                "enhanced_conservation_support": round(_safe_float(enhanced_row.get("conservation_support")) or 0.0, 3) if enhanced_row is not None else None,
                "base_conservation_mean": round(_safe_float(base_row.get("conservation_mean")) or 0.0, 3) if base_row is not None else None,
                "enhanced_conservation_mean": round(_safe_float(enhanced_row.get("conservation_mean")) or 0.0, 3) if enhanced_row is not None else None,
                "residue_count": _safe_int(enhanced_row.get("residue_count")) if enhanced_row is not None else (_safe_int(base_row.get("residue_count")) if base_row is not None else None),
            }
        )

    comparison_df = pd.DataFrame(rows, columns=POCKET_RANKING_COMPARISON_COLUMNS)
    priority = {
        "moved_up": 0,
        "moved_down": 1,
        "new": 2,
        "removed": 3,
        "score_changed": 4,
        "unchanged": 5,
    }
    comparison_df["_priority"] = comparison_df["status"].map(priority).fillna(9)
    comparison_df["_rank_abs_delta"] = pd.to_numeric(comparison_df["rank_delta"], errors="coerce").fillna(0.0).abs()
    comparison_df["_score_abs_delta"] = pd.to_numeric(comparison_df["score_delta"], errors="coerce").fillna(0.0).abs()
    comparison_df["_display_rank"] = pd.to_numeric(comparison_df["enhanced_rank"], errors="coerce").fillna(
        pd.to_numeric(comparison_df["base_rank"], errors="coerce").fillna(1_000_000)
    )
    return (
        comparison_df.sort_values(
            ["_priority", "_rank_abs_delta", "_score_abs_delta", "_display_rank", "pocket_id"],
            ascending=[True, False, False, True, True],
        )
        .drop(columns=["_priority", "_rank_abs_delta", "_score_abs_delta", "_display_rank"])
        .reset_index(drop=True)
    )


def compare_hotspot_sets(hotspot_tables: List[pd.DataFrame]) -> Dict[str, Any]:
    """比较多个热点表，返回共同/并集与每个残基在多少个构象中被标为热点的统计信息。"""
    n = len(hotspot_tables)
    sets = [_hotspot_set(df) for df in hotspot_tables]

    union = set().union(*sets) if sets else set()
    intersection = sets[0].intersection(*sets[1:]) if n > 1 else sets[0] if sets else set()
    consistency_score = float(len(intersection)) / float(len(union)) if len(union) > 0 else 0.0

    counts = {}
    for s in sets:
        for key in s:
            counts[key] = counts.get(key, 0) + 1

    rows = []
    for (chain, resid, resname), cnt in counts.items():
        rows.append(
            {
                "chain": chain,
                "resid": int(resid),
                "resname": resname,
                "count": int(cnt),
                "frequency": round(float(cnt) / float(n), 3) if n > 0 else 0.0,
                "label": f"{resname} {chain}{resid}",
                "is_common": cnt == n,
            }
        )

    per_residue_df = pd.DataFrame(rows)
    if not per_residue_df.empty:
        per_residue_df = per_residue_df.sort_values(["count", "resid", "chain", "resname"], ascending=[False, True, True, True]).reset_index(drop=True)
    else:
        per_residue_df = pd.DataFrame(columns=["chain", "resid", "resname", "count", "frequency", "label", "is_common"])
    common_hotspots = [f"{resname} {chain}{resid}" for chain, resid, resname in sorted(intersection, key=lambda item: (item[0], item[1], item[2]))]

    return {
        "total_conformations": n,
        "consistency_score": consistency_score,
        "union_size": len(union),
        "intersection_size": len(intersection),
        "common_hotspots": common_hotspots,
        "per_residue_df": per_residue_df,
    }


def build_hotspot_stability_tables(per_residue_df: pd.DataFrame, *, stable_threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
    if per_residue_df is None or getattr(per_residue_df, "empty", True):
        empty = pd.DataFrame(columns=["chain", "resid", "resname", "count", "frequency", "label", "is_common"])
        return {
            "stable_hotspots": empty.copy(),
            "variable_hotspots": empty.copy(),
            "common_hotspots": empty.copy(),
        }

    table = per_residue_df.copy()
    if "frequency" not in table.columns:
        table["frequency"] = 0.0

    stable_mask = pd.to_numeric(table["frequency"], errors="coerce").fillna(0.0) >= float(stable_threshold)
    variable_mask = (pd.to_numeric(table["frequency"], errors="coerce").fillna(0.0) > 0.0) & ~stable_mask
    common_mask = table["is_common"] if "is_common" in table.columns else stable_mask & (pd.to_numeric(table["frequency"], errors="coerce").fillna(0.0) >= 1.0)

    stable_hotspots = table[stable_mask].copy()
    variable_hotspots = table[variable_mask].copy()
    common_hotspots = table[common_mask].copy()

    if not stable_hotspots.empty:
        stable_hotspots = stable_hotspots.sort_values(["frequency", "count", "resid", "chain"], ascending=[False, False, True, True]).reset_index(drop=True)
    if not variable_hotspots.empty:
        variable_hotspots = variable_hotspots.sort_values(["frequency", "count", "resid", "chain"], ascending=[True, False, True, True]).reset_index(drop=True)
    if not common_hotspots.empty:
        common_hotspots = common_hotspots.sort_values(["count", "resid", "chain"], ascending=[False, True, True]).reset_index(drop=True)

    return {
        "stable_hotspots": stable_hotspots,
        "variable_hotspots": variable_hotspots,
        "common_hotspots": common_hotspots,
    }


def build_reference_comparison_table(
    per_conformation_rows: List[Dict[str, Any]],
    hotspot_tables: List[pd.DataFrame],
    *,
    reference_index: int = 0,
) -> pd.DataFrame:
    if not per_conformation_rows:
        return pd.DataFrame(
            columns=[
                "conformation",
                "is_reference",
                "residue_count",
                "valid_energy_count",
                "mean_energy",
                "mean_energy_delta_vs_reference",
                "hotspot_count",
                "hotspot_count_delta_vs_reference",
                "reference_overlap_count",
                "reference_overlap_ratio",
                "unique_hotspot_count",
                "energy_coverage",
                "protein_volume",
                "energy_source",
            ]
        )

    if reference_index < 0 or reference_index >= len(per_conformation_rows):
        reference_index = 0

    hotspot_sets = [_hotspot_set(df) for df in hotspot_tables]
    reference_set = hotspot_sets[reference_index] if reference_index < len(hotspot_sets) else set()
    reference_row = per_conformation_rows[reference_index]
    reference_mean = _safe_float(reference_row.get("mean_energy"))
    reference_hotspot_count = int(reference_row.get("hotspot_count", len(reference_set)) or len(reference_set))

    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(per_conformation_rows):
        current_set = hotspot_sets[index] if index < len(hotspot_sets) else set()
        current_mean = _safe_float(row.get("mean_energy"))
        hotspot_count = int(row.get("hotspot_count", len(current_set)) or len(current_set))
        overlap_count = len(current_set & reference_set)
        unique_count = len(current_set - reference_set)

        rows.append(
            {
                "conformation": row.get("conformation", f"构象 {index + 1}"),
                "is_reference": index == reference_index,
                "residue_count": int(row.get("residue_count", 0) or 0),
                "valid_energy_count": int(row.get("valid_energy_count", 0) or 0),
                "mean_energy": current_mean,
                "mean_energy_delta_vs_reference": None if current_mean is None or reference_mean is None else round(current_mean - reference_mean, 3),
                "hotspot_count": hotspot_count,
                "hotspot_count_delta_vs_reference": hotspot_count - reference_hotspot_count,
                "reference_overlap_count": int(overlap_count),
                "reference_overlap_ratio": round(float(overlap_count) / float(len(reference_set)), 3) if reference_set else 0.0,
                "unique_hotspot_count": int(unique_count),
                "energy_coverage": row.get("energy_coverage", 0.0),
                "protein_volume": row.get("protein_volume"),
                "energy_source": row.get("energy_source"),
            }
        )

    comparison_df = pd.DataFrame(rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(["is_reference", "hotspot_count"], ascending=[False, False]).reset_index(drop=True)
    return comparison_df


def build_pairwise_similarity_matrix(hotspot_tables: List[pd.DataFrame]) -> pd.DataFrame:
    sets = []
    labels = []
    for index, df in enumerate(hotspot_tables, start=1):
        s = set()
        if df is not None and not getattr(df, "empty", True):
            for row in df.itertuples(index=False):
                try:
                    s.add((row.chain, int(row.resid), row.resname))
                except Exception:
                    continue
        sets.append(s)
        labels.append(f"构象 {index}")

    rows = []
    for i, left in enumerate(sets):
        row = {"构象": labels[i]}
        for j, right in enumerate(sets):
            union = left.union(right)
            intersection = left.intersection(right)
            score = float(len(intersection)) / float(len(union)) if len(union) > 0 else 0.0
            row[labels[j]] = round(score, 3)
        rows.append(row)

    matrix = pd.DataFrame(rows)
    if not matrix.empty:
        matrix = matrix[["构象", *labels]]
    return matrix


# 兼容旧名称
compare_hotspot_sets.__name__ = "compare_hotspot_sets"
