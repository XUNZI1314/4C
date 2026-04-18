from __future__ import annotations

from typing import Optional

import pandas as pd


JOINT_RECOMMENDATION_COLUMNS = [
    "recommendation_rank",
    "pocket_id",
    "recommendation_score",
    "recommendation_label",
    "recommendation_reason",
    "recommendation_action",
    "evidence_quality_label",
    "evidence_anchor_support",
    "evidence_anchor_risk",
    "smart_rank_score",
    "smart_rank_label",
    "residue_count",
    "hotspot_overlap_count",
    "interface_overlap_count",
    "interface_core_count",
    "triple_overlap_count",
    "external_overlap_count",
    "external_exact_overlap_count",
    "external_weak_overlap_count",
    "hotspot_overlap_ratio",
    "interface_overlap_ratio",
    "triple_overlap_ratio",
    "external_overlap_ratio",
    "external_weighted_overlap_ratio",
    "external_mapping_confidence",
    "external_structure_verified_count",
    "external_structure_verified_ratio",
    "method_vote_count",
    "consensus_methods",
    "pocket_source",
    "interface_region_types",
    "external_evidence_types",
    "residue_labels",
]

POCKET_CONSENSUS_COVERAGE_COLUMNS = [
    "pocket_id",
    "residue_count",
    "consensus_residue_count",
    "consensus_coverage_ratio",
    "rank_safe_anchor_count",
    "validated_anchor_count",
    "supported_anchor_count",
    "ai_supported_anchor_count",
    "conservation_context_count",
    "weak_mapping_count",
    "blocked_ai_count",
    "best_consensus_score",
    "mean_consensus_score",
    "consensus_anchor_residues",
    "consensus_tiers",
    "consensus_sources",
    "pocket_consensus_label",
    "pocket_consensus_action",
]


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _residue_key_frame(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: (str(row["chain"]).strip() or "A", int(row["resid"])), axis=1)


def _empty_pocket_consensus_coverage_df() -> pd.DataFrame:
    return pd.DataFrame(columns=POCKET_CONSENSUS_COVERAGE_COLUMNS)


def _normalize_consensus_table(consensus_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if consensus_df is None or getattr(consensus_df, "empty", True) or "resid" not in consensus_df.columns:
        return pd.DataFrame()

    normalized = consensus_df.copy()
    normalized["resid"] = pd.to_numeric(normalized["resid"], errors="coerce")
    normalized = normalized[normalized["resid"].notna()].copy()
    if normalized.empty:
        return pd.DataFrame()

    normalized["resid"] = normalized["resid"].astype(int)
    if "chain" not in normalized.columns:
        normalized["chain"] = ""
    normalized["chain"] = normalized["chain"].astype(str).str.strip()
    if "residue_anchor" not in normalized.columns:
        normalized["residue_anchor"] = normalized.apply(
            lambda row: f"{row['chain']}:{int(row['resid'])}" if str(row["chain"]).strip() else str(int(row["resid"])),
            axis=1,
        )
    if "consensus_tier" not in normalized.columns:
        normalized["consensus_tier"] = "evidence-context"
    if "consensus_score" not in normalized.columns:
        normalized["consensus_score"] = 0.0
    if "evidence_sources" not in normalized.columns:
        normalized["evidence_sources"] = ""
    normalized["consensus_score"] = pd.to_numeric(normalized["consensus_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    normalized["_exact_key"] = normalized.apply(lambda row: (str(row["chain"]).strip() or "A", int(row["resid"])), axis=1)
    normalized["_resid_only"] = normalized["resid"].astype(int)
    return normalized


def _matched_consensus_rows(consensus_df: pd.DataFrame, pocket_keys: set[tuple[str, int]]) -> pd.DataFrame:
    if consensus_df.empty or not pocket_keys:
        return pd.DataFrame()
    pocket_resids = {resid for _chain, resid in pocket_keys}
    return consensus_df[
        consensus_df.apply(
            lambda row: (
                (str(row.get("chain") or "").strip() and row["_exact_key"] in pocket_keys)
                or (not str(row.get("chain") or "").strip() and int(row["_resid_only"]) in pocket_resids)
            ),
            axis=1,
        )
    ].copy()


def _pocket_consensus_label_and_action(
    *,
    validated_count: int,
    supported_count: int,
    ai_supported_count: int,
    conservation_count: int,
    weak_count: int,
    blocked_count: int,
    rank_safe_count: int,
) -> tuple[str, str]:
    if validated_count > 0:
        return (
            "consensus-validated-pocket",
            "Prioritize this pocket for validation around validated cross-source residue anchors.",
        )
    if supported_count > 0:
        return (
            "consensus-supported-pocket",
            "Shortlist this pocket; verify source details and nearby geometry before validation.",
        )
    if ai_supported_count > 0:
        return (
            "ai-supported-pocket",
            "Keep as AI-supported shortlist; verify cited source before using it as a decisive anchor.",
        )
    if blocked_count > 0 and rank_safe_count == 0:
        return (
            "blocked-ai-evidence-pocket",
            "Do not promote this pocket from AI evidence until blocked residues are fixed or rejected.",
        )
    if weak_count > 0 and rank_safe_count == 0:
        return (
            "mapping-review-pocket",
            "Resolve residue numbering or chain mapping before treating this pocket as evidence-backed.",
        )
    if conservation_count > 0 and rank_safe_count == 0:
        return (
            "conservation-context-pocket",
            "Use as context only; add catalytic, binding, mutagenesis, or curated literature evidence.",
        )
    return (
        "no-consensus-anchor-pocket",
        "No residue-level consensus anchor overlaps this pocket; rely on geometry only or add evidence.",
    )


def build_pocket_consensus_coverage(
    pocket_df: Optional[pd.DataFrame],
    residue_consensus_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if pocket_df is None or getattr(pocket_df, "empty", True) or "pocket_id" not in pocket_df.columns or "resid" not in pocket_df.columns:
        return _empty_pocket_consensus_coverage_df()

    pockets = pocket_df.copy()
    pockets["resid"] = pd.to_numeric(pockets["resid"], errors="coerce")
    pockets = pockets[pockets["resid"].notna()].copy()
    if pockets.empty:
        return _empty_pocket_consensus_coverage_df()
    pockets["resid"] = pockets["resid"].astype(int)
    if "chain" not in pockets.columns:
        pockets["chain"] = "A"
    pockets["chain"] = pockets["chain"].astype(str).str.strip().replace("", "A")

    consensus = _normalize_consensus_table(residue_consensus_df)
    if consensus.empty:
        return _empty_pocket_consensus_coverage_df()

    rows = []
    rank_safe_tiers = {"validated-anchor", "supported-anchor", "ai-supported-anchor"}
    for pocket_id, group in pockets.groupby("pocket_id", sort=False):
        group = group.copy()
        group["_residue_key"] = _residue_key_frame(group)
        pocket_keys = set(group["_residue_key"].tolist())
        residue_count = max(1, len(pocket_keys))
        matched = _matched_consensus_rows(consensus, pocket_keys)

        if matched.empty:
            label, action = _pocket_consensus_label_and_action(
                validated_count=0,
                supported_count=0,
                ai_supported_count=0,
                conservation_count=0,
                weak_count=0,
                blocked_count=0,
                rank_safe_count=0,
            )
            rows.append(
                {
                    "pocket_id": pocket_id,
                    "residue_count": residue_count,
                    "consensus_residue_count": 0,
                    "consensus_coverage_ratio": 0.0,
                    "rank_safe_anchor_count": 0,
                    "validated_anchor_count": 0,
                    "supported_anchor_count": 0,
                    "ai_supported_anchor_count": 0,
                    "conservation_context_count": 0,
                    "weak_mapping_count": 0,
                    "blocked_ai_count": 0,
                    "best_consensus_score": 0.0,
                    "mean_consensus_score": 0.0,
                    "consensus_anchor_residues": "none",
                    "consensus_tiers": "none",
                    "consensus_sources": "none",
                    "pocket_consensus_label": label,
                    "pocket_consensus_action": action,
                }
            )
            continue

        tiers = matched["consensus_tier"].astype(str).str.strip()
        validated_count = int(tiers.eq("validated-anchor").sum())
        supported_count = int(tiers.eq("supported-anchor").sum())
        ai_supported_count = int(tiers.eq("ai-supported-anchor").sum())
        conservation_count = int(tiers.eq("conservation-context").sum())
        weak_count = int(tiers.eq("weak-mapping").sum())
        blocked_count = int(tiers.eq("blocked-ai").sum())
        rank_safe_count = int(tiers.isin(rank_safe_tiers).sum())
        label, action = _pocket_consensus_label_and_action(
            validated_count=validated_count,
            supported_count=supported_count,
            ai_supported_count=ai_supported_count,
            conservation_count=conservation_count,
            weak_count=weak_count,
            blocked_count=blocked_count,
            rank_safe_count=rank_safe_count,
        )
        scores = pd.to_numeric(matched["consensus_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        rows.append(
            {
                "pocket_id": pocket_id,
                "residue_count": residue_count,
                "consensus_residue_count": int(len(matched)),
                "consensus_coverage_ratio": round(float(len(matched)) / float(residue_count), 3),
                "rank_safe_anchor_count": rank_safe_count,
                "validated_anchor_count": validated_count,
                "supported_anchor_count": supported_count,
                "ai_supported_anchor_count": ai_supported_count,
                "conservation_context_count": conservation_count,
                "weak_mapping_count": weak_count,
                "blocked_ai_count": blocked_count,
                "best_consensus_score": round(float(scores.max()), 3),
                "mean_consensus_score": round(float(scores.mean()), 3),
                "consensus_anchor_residues": ", ".join(dict.fromkeys(matched["residue_anchor"].astype(str).tolist())),
                "consensus_tiers": ", ".join(f"{tier}:{count}" for tier, count in tiers.value_counts().to_dict().items()),
                "consensus_sources": ", ".join(
                    sorted(
                        {
                            source.strip()
                            for value in matched["evidence_sources"].astype(str).tolist()
                            for source in value.split(",")
                            if source.strip() and source.strip().lower() != "none"
                        }
                    )
                ) or "none",
                "pocket_consensus_label": label,
                "pocket_consensus_action": action,
            }
        )

    if not rows:
        return _empty_pocket_consensus_coverage_df()
    label_rank = {
        "consensus-validated-pocket": 0,
        "consensus-supported-pocket": 1,
        "ai-supported-pocket": 2,
        "mapping-review-pocket": 3,
        "conservation-context-pocket": 4,
        "blocked-ai-evidence-pocket": 5,
        "no-consensus-anchor-pocket": 6,
    }
    result = pd.DataFrame(rows, columns=POCKET_CONSENSUS_COVERAGE_COLUMNS)
    result["_label_rank"] = result["pocket_consensus_label"].map(label_rank).fillna(99)
    result = result.sort_values(
        ["_label_rank", "best_consensus_score", "rank_safe_anchor_count", "consensus_coverage_ratio", "pocket_id"],
        ascending=[True, False, False, False, True],
    ).drop(columns="_label_rank").reset_index(drop=True)
    return result[POCKET_CONSENSUS_COVERAGE_COLUMNS]


def _recommendation_label(score: float) -> str:
    if score >= 0.70:
        return "优先验证"
    if score >= 0.50:
        return "建议关注"
    return "探索候选"


def _recommendation_reason(
    *,
    smart_rank_score: float,
    evidence_anchor_support: float,
    evidence_anchor_risk: float,
    hotspot_overlap_ratio: float,
    interface_overlap_ratio: float,
    triple_overlap_ratio: float,
    external_exact_ratio: float,
    external_weak_ratio: float,
    external_overlap_ratio: float,
    external_mapping_confidence: float,
    external_structure_verified_ratio: float,
    interface_core_count: int,
    has_interface_signal: bool,
) -> str:
    reasons: list[str] = []
    if smart_rank_score >= 0.70:
        reasons.append("口袋自身排序靠前")
    elif smart_rank_score >= 0.50:
        reasons.append("口袋自身证据中等偏强")

    if evidence_anchor_support >= 0.45:
        reasons.append("direct evidence anchor supports recommendation")
    elif evidence_anchor_risk >= 0.35:
        reasons.append("evidence support may be neighborhood-expanded")

    if triple_overlap_ratio >= 0.20:
        reasons.append("口袋/界面/热点三重交集明显")
    elif hotspot_overlap_ratio >= 0.25:
        reasons.append("热点重叠较集中")

    if external_exact_ratio >= 0.20:
        reasons.append("命中外部位点结构映射证据")
    elif external_exact_ratio > 0.0:
        reasons.append("部分命中外部位点结构映射")
    elif external_overlap_ratio > 0.0:
        reasons.append("部分命中外部位点注释")

    if external_weak_ratio >= 0.20:
        reasons.append("存在残基号弱命中，建议结合映射进一步核验")

    if external_structure_verified_ratio >= 0.20:
        reasons.append("外部位点与结构残基对齐可靠")

    if external_mapping_confidence > 0.0 and external_mapping_confidence < 0.50:
        reasons.append("外部映射置信度偏低")

    if has_interface_signal and interface_overlap_ratio >= 0.30:
        reasons.append("界面残基覆盖较高")
    elif has_interface_signal and interface_core_count > 0:
        reasons.append("命中界面核心区域")
    elif not has_interface_signal:
        reasons.append("当前主要依据口袋与热点证据")

    if not reasons:
        reasons.append("建议结合下游验证继续筛选")
    return "；".join(reasons[:3])


def _joint_evidence_quality_label(
    *,
    evidence_anchor_support: float,
    evidence_anchor_risk: float,
    external_exact_ratio: float,
    external_overlap_ratio: float,
    external_mapping_confidence: float,
    external_structure_verified_ratio: float,
) -> str:
    if evidence_anchor_support >= 0.70 and external_exact_ratio > 0.0 and external_mapping_confidence >= 0.70:
        return "strong-direct-anchor"
    if evidence_anchor_support >= 0.45 or external_exact_ratio > 0.0:
        return "direct-anchor"
    if external_overlap_ratio > 0.0 and external_structure_verified_ratio > 0.0:
        return "structure-verified-external"
    if external_overlap_ratio > 0.0 and evidence_anchor_risk >= 0.35:
        return "neighborhood-expanded"
    if external_overlap_ratio > 0.0:
        return "diffuse-external-support"
    return "no-external-evidence"


def _recommendation_action(
    *,
    recommendation_score: float,
    evidence_quality_label: str,
    evidence_anchor_risk: float,
    triple_overlap_ratio: float,
    interface_overlap_ratio: float,
) -> str:
    if evidence_quality_label == "strong-direct-anchor" and recommendation_score >= 0.25:
        return "validate-prioritize"
    if evidence_quality_label == "direct-anchor" and evidence_anchor_risk < 0.20 and recommendation_score >= 0.25:
        return "validate-prioritize"
    if evidence_anchor_risk >= 0.35 or evidence_quality_label in {"neighborhood-expanded", "diffuse-external-support"}:
        return "review-evidence-mapping"
    if triple_overlap_ratio >= 0.20 or interface_overlap_ratio >= 0.30:
        return "validate-interface-context"
    if recommendation_score >= 0.50:
        return "shortlist-follow-up"
    return "exploratory-only"


def build_joint_candidate_table(
    pocket_df: Optional[pd.DataFrame],
    annotation_df: Optional[pd.DataFrame],
    hotspot_df: Optional[pd.DataFrame],
    external_site_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if pocket_df is None or getattr(pocket_df, "empty", True):
        return pd.DataFrame(columns=JOINT_RECOMMENDATION_COLUMNS)

    hotspot_keys = set()
    if hotspot_df is not None and not getattr(hotspot_df, "empty", True):
        hotspot_keys = {
            (str(row.chain).strip() or "A", int(row.resid))
            for row in hotspot_df.itertuples(index=False)
        }

    normalized_annotations = pd.DataFrame()
    if annotation_df is not None and not getattr(annotation_df, "empty", True):
        normalized_annotations = annotation_df.copy()
        normalized_annotations["_residue_key"] = _residue_key_frame(normalized_annotations)

    external_exact_keys = set()
    external_resid_keys = set()
    normalized_external = pd.DataFrame()
    if external_site_df is not None and not getattr(external_site_df, "empty", True):
        normalized_external = external_site_df.copy()
        if "resid" in normalized_external.columns:
            normalized_external["resid"] = pd.to_numeric(normalized_external["resid"], errors="coerce")
            normalized_external = normalized_external[normalized_external["resid"].notna()].copy()
            normalized_external["resid"] = normalized_external["resid"].astype(int)
        else:
            normalized_external = pd.DataFrame()

        if not normalized_external.empty:
            if "chain" not in normalized_external.columns:
                normalized_external["chain"] = ""
            normalized_external["chain"] = normalized_external["chain"].astype(str).str.strip()
            if "mapping_level" not in normalized_external.columns:
                normalized_external["mapping_level"] = normalized_external["chain"].apply(
                    lambda value: "exact" if str(value).strip() else "weak"
                )
            normalized_external["mapping_level"] = normalized_external["mapping_level"].astype(str).str.strip().str.lower()
            normalized_external.loc[
                ~normalized_external["mapping_level"].isin({"exact", "weak"}),
                "mapping_level",
            ] = "weak"

            if "mapping_confidence" not in normalized_external.columns:
                normalized_external["mapping_confidence"] = normalized_external["mapping_level"].apply(
                    lambda value: 0.90 if value == "exact" else 0.30
                )
            normalized_external["mapping_confidence"] = pd.to_numeric(
                normalized_external["mapping_confidence"],
                errors="coerce",
            ).fillna(0.30)
            normalized_external["mapping_confidence"] = normalized_external["mapping_confidence"].clip(lower=0.0, upper=1.0)
            if "mapping_method" not in normalized_external.columns:
                normalized_external["mapping_method"] = "unknown"
            normalized_external["_structure_verified"] = normalized_external["mapping_method"].astype(str).str.contains(
                "structure|verified",
                case=False,
                regex=True,
                na=False,
            )

            normalized_external["_residue_key"] = normalized_external.apply(
                lambda row: (row["chain"], int(row["resid"])),
                axis=1,
            )
            for row in normalized_external.itertuples(index=False):
                chain_value = str(getattr(row, "chain", "")).strip()
                resid_value = int(getattr(row, "resid", 0))
                level_value = str(getattr(row, "mapping_level", "") or "").strip().lower()
                if level_value == "exact" and chain_value:
                    external_exact_keys.add((chain_value, resid_value))
                else:
                    external_resid_keys.add(resid_value)

    rows = []
    for pocket_id, group in pocket_df.groupby("pocket_id", sort=False):
        working_group = group.copy()
        working_group["_residue_key"] = _residue_key_frame(working_group)
        pocket_keys = set(working_group["_residue_key"].tolist())
        residue_count = max(1, len(pocket_keys))

        hotspot_overlap = pocket_keys & hotspot_keys
        pocket_annotations = (
            normalized_annotations[normalized_annotations["_residue_key"].isin(pocket_keys)].copy()
            if not normalized_annotations.empty
            else pd.DataFrame()
        )
        interface_overlap_keys = set(pocket_annotations["_residue_key"].tolist()) if not pocket_annotations.empty else set()
        triple_overlap_keys = interface_overlap_keys & hotspot_keys
        interface_core_count = 0
        if not pocket_annotations.empty and "region_type" in pocket_annotations.columns:
            interface_core_count = int(
                pocket_annotations["region_type"].astype(str).str.contains("core", case=False, na=False).sum()
            )

        smart_rank_score = float(_numeric_series(working_group, "smart_rank_score", 0.0).max()) if not working_group.empty else 0.0
        evidence_anchor_support = float(_numeric_series(working_group, "smart_evidence_anchor_support", 0.0).max()) if not working_group.empty else 0.0
        evidence_anchor_risk = float(_numeric_series(working_group, "smart_evidence_anchor_risk", 0.0).max()) if not working_group.empty else 0.0
        smart_rank_label = str(working_group.get("smart_rank_label").dropna().astype(str).iloc[0]) if "smart_rank_label" in working_group.columns and working_group["smart_rank_label"].notna().any() else ""
        method_vote_count = int(_numeric_series(working_group, "method_vote_count", 1.0).max()) if not working_group.empty else 1
        consensus_methods = ", ".join(
            sorted({str(value).strip() for value in working_group.get("consensus_methods", pd.Series(dtype=str)).dropna().tolist() if str(value).strip()})
        )
        pocket_source = ", ".join(
            sorted({str(value).strip() for value in working_group.get("pocket_source", pd.Series(dtype=str)).dropna().tolist() if str(value).strip()})
        )

        hotspot_overlap_ratio = float(len(hotspot_overlap)) / float(residue_count)
        interface_overlap_ratio = float(len(interface_overlap_keys)) / float(residue_count)
        triple_overlap_ratio = float(len(triple_overlap_keys)) / float(residue_count)
        core_ratio = float(interface_core_count) / float(max(1, len(interface_overlap_keys)))

        external_exact_overlap_keys = {key for key in pocket_keys if key in external_exact_keys}
        external_weak_overlap_resids = {
            resid
            for chain, resid in pocket_keys
            if resid in external_resid_keys and (chain, resid) not in external_exact_overlap_keys
        }
        external_overlap_count = int(len(external_exact_overlap_keys) + len(external_weak_overlap_resids))
        external_overlap_ratio = float(external_overlap_count) / float(residue_count)
        external_exact_ratio = float(len(external_exact_overlap_keys)) / float(residue_count)
        external_weak_ratio = float(len(external_weak_overlap_resids)) / float(residue_count)
        external_weighted_overlap_ratio = min(1.0, external_exact_ratio + 0.45 * external_weak_ratio)

        external_evidence_types = ""
        external_mapping_confidence = 0.0
        external_structure_verified_count = 0
        if not normalized_external.empty:
            pocket_exact_chain_resid = {(chain, resid) for chain, resid in pocket_keys}
            pocket_resid_only = {resid for _, resid in pocket_keys}
            matched_external = normalized_external[
                normalized_external.apply(
                    lambda row: (
                        (
                            str(row.get("mapping_level", "") or "").strip().lower() == "exact"
                            and (str(row["chain"]).strip(), int(row["resid"])) in pocket_exact_chain_resid
                        )
                        or (
                            str(row.get("mapping_level", "") or "").strip().lower() != "exact"
                            and int(row["resid"]) in pocket_resid_only
                        )
                    ),
                    axis=1,
                )
            ]
            if not matched_external.empty and "evidence_type" in matched_external.columns:
                external_evidence_types = ", ".join(
                    sorted(
                        {
                            str(value).strip()
                            for value in matched_external["evidence_type"].dropna().tolist()
                            if str(value).strip()
                        }
                    )
                )
            if not matched_external.empty and "mapping_confidence" in matched_external.columns:
                external_mapping_confidence = float(
                    pd.to_numeric(matched_external["mapping_confidence"], errors="coerce").fillna(0.0).mean()
                )
            if not matched_external.empty and "_structure_verified" in matched_external.columns:
                external_structure_verified_count = int(
                    matched_external["_structure_verified"].fillna(False).astype(bool).sum()
                )
        external_structure_verified_ratio = float(external_structure_verified_count) / float(residue_count)

        base_recommendation_score = (
            0.42 * smart_rank_score
            + 0.18 * hotspot_overlap_ratio
            + 0.18 * interface_overlap_ratio
            + 0.16 * triple_overlap_ratio
            + 0.06 * core_ratio
        )
        external_bonus = 0.12 * external_weighted_overlap_ratio
        if external_mapping_confidence > 0.0:
            external_bonus += 0.03 * external_mapping_confidence
        if external_structure_verified_ratio > 0.0:
            external_bonus += 0.03 * external_structure_verified_ratio
        evidence_quality_label = _joint_evidence_quality_label(
            evidence_anchor_support=evidence_anchor_support,
            evidence_anchor_risk=evidence_anchor_risk,
            external_exact_ratio=external_exact_ratio,
            external_overlap_ratio=external_overlap_ratio,
            external_mapping_confidence=external_mapping_confidence,
            external_structure_verified_ratio=external_structure_verified_ratio,
        )
        recommendation_score = min(
            1.0,
            max(
                0.0,
                float(
                    base_recommendation_score
                    + external_bonus
                    + 0.05 * evidence_anchor_support
                    - 0.04 * evidence_anchor_risk
                ),
            ),
        )
        recommendation_action = _recommendation_action(
            recommendation_score=float(recommendation_score),
            evidence_quality_label=evidence_quality_label,
            evidence_anchor_risk=evidence_anchor_risk,
            triple_overlap_ratio=triple_overlap_ratio,
            interface_overlap_ratio=interface_overlap_ratio,
        )

        interface_region_types = ""
        if not pocket_annotations.empty and "region_type" in pocket_annotations.columns:
            counts = pocket_annotations["region_type"].astype(str).value_counts()
            interface_region_types = ", ".join(f"{name}:{int(count)}" for name, count in counts.items())

        residue_labels = ", ".join(
            f"{row.resname} {row.chain}{int(row.resid)}"
            for row in working_group.itertuples(index=False)
        )

        rows.append(
            {
                "pocket_id": pocket_id,
                "recommendation_score": round(float(recommendation_score), 3),
                "recommendation_label": _recommendation_label(float(recommendation_score)),
                "recommendation_reason": _recommendation_reason(
                    smart_rank_score=smart_rank_score,
                    evidence_anchor_support=evidence_anchor_support,
                    evidence_anchor_risk=evidence_anchor_risk,
                    hotspot_overlap_ratio=hotspot_overlap_ratio,
                    interface_overlap_ratio=interface_overlap_ratio,
                    triple_overlap_ratio=triple_overlap_ratio,
                    external_exact_ratio=external_exact_ratio,
                    external_weak_ratio=external_weak_ratio,
                    external_overlap_ratio=external_overlap_ratio,
                    external_mapping_confidence=external_mapping_confidence,
                    external_structure_verified_ratio=external_structure_verified_ratio,
                    interface_core_count=interface_core_count,
                    has_interface_signal=not pocket_annotations.empty,
                ),
                "recommendation_action": recommendation_action,
                "evidence_quality_label": evidence_quality_label,
                "evidence_anchor_support": round(evidence_anchor_support, 3),
                "evidence_anchor_risk": round(evidence_anchor_risk, 3),
                "smart_rank_score": round(float(smart_rank_score), 3),
                "smart_rank_label": smart_rank_label or "",
                "residue_count": int(residue_count),
                "hotspot_overlap_count": int(len(hotspot_overlap)),
                "interface_overlap_count": int(len(interface_overlap_keys)),
                "interface_core_count": int(interface_core_count),
                "triple_overlap_count": int(len(triple_overlap_keys)),
                "external_overlap_count": int(external_overlap_count),
                "external_exact_overlap_count": int(len(external_exact_overlap_keys)),
                "external_weak_overlap_count": int(len(external_weak_overlap_resids)),
                "hotspot_overlap_ratio": round(hotspot_overlap_ratio, 3),
                "interface_overlap_ratio": round(interface_overlap_ratio, 3),
                "triple_overlap_ratio": round(triple_overlap_ratio, 3),
                "external_overlap_ratio": round(external_overlap_ratio, 3),
                "external_weighted_overlap_ratio": round(external_weighted_overlap_ratio, 3),
                "external_mapping_confidence": round(external_mapping_confidence, 3),
                "external_structure_verified_count": int(external_structure_verified_count),
                "external_structure_verified_ratio": round(external_structure_verified_ratio, 3),
                "method_vote_count": int(method_vote_count),
                "consensus_methods": consensus_methods,
                "pocket_source": pocket_source,
                "interface_region_types": interface_region_types,
                "external_evidence_types": external_evidence_types,
                "residue_labels": residue_labels,
            }
        )

    if not rows:
        return pd.DataFrame(columns=JOINT_RECOMMENDATION_COLUMNS)

    result = pd.DataFrame(rows).sort_values(
        ["recommendation_score", "triple_overlap_count", "interface_overlap_count", "hotspot_overlap_count", "smart_rank_score"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    result["recommendation_rank"] = range(1, len(result) + 1)
    ordered_columns = JOINT_RECOMMENDATION_COLUMNS + [
        column for column in result.columns if column not in JOINT_RECOMMENDATION_COLUMNS
    ]
    return result[ordered_columns]
