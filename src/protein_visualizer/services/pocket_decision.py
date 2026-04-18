from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO, StringIO
from typing import Any, Optional

import pandas as pd


POCKET_DECISION_COLUMNS = [
    "decision_rank",
    "pocket_id",
    "decision_label",
    "decision_score",
    "functional_confidence",
    "geometry_confidence",
    "recommended_action",
    "audit_status",
    "evidence_quality_label",
    "evidence_quality_score",
    "direct_anchor_count",
    "route_anchor_count",
    "anchor_residues",
    "method_vote_count",
    "smart_rank_label",
    "smart_rank_score",
    "literature_rank_delta",
    "evidence_route_rank_delta",
    "conservation_rank_delta",
    "risk_flags",
    "supporting_evidence",
    "next_step",
    "visual_focus",
]


RESIDUE_LAYER_COLUMNS = [
    "pocket_layer",
    "pocket_layer_score",
    "pocket_layer_reason",
]

POCKET_RELIABILITY_COLUMNS = [
    "pocket_id",
    "check_order",
    "check",
    "status",
    "signal",
    "why_it_matters",
    "next_action",
]

POCKET_TRIAGE_COLUMNS = [
    "pocket_id",
    "decision_rank",
    "precision_tier",
    "triage_priority",
    "triage_action",
    "blocking_checks",
    "review_checks",
    "pass_count",
    "review_count",
    "missing_count",
    "triage_reason",
    "next_data_to_add",
]

CONSENSUS_RERANK_SUGGESTION_COLUMNS = [
    "pocket_id",
    "current_decision_rank",
    "consensus_rank",
    "rank_delta",
    "current_decision_score",
    "pocket_consensus_label",
    "rank_safe_anchor_count",
    "best_consensus_score",
    "blocked_ai_count",
    "weak_mapping_count",
    "suggestion_status",
    "suggestion_reason",
    "recommended_action",
    "consensus_anchor_residues",
]

CONSENSUS_RERANK_PREVIEW_COLUMNS = [
    "pocket_id",
    "current_rank",
    "preview_rank",
    "preview_rank_delta",
    "current_decision_score",
    "consensus_adjustment",
    "preview_score",
    "suggestion_status",
    "pocket_consensus_label",
    "rank_safe_anchor_count",
    "blocked_ai_count",
    "weak_mapping_count",
    "preview_decision",
    "preview_reason",
    "recommended_action",
    "consensus_anchor_residues",
]

CONSENSUS_RERANK_POLICY_GATE_COLUMNS = [
    "preview_rows",
    "changed_rows",
    "promote_rows",
    "demote_rows",
    "blocked_rows",
    "mapping_review_rows",
    "evidence_gap_rows",
    "keep_priority_rows",
    "top_preview_pocket_id",
    "top_preview_decision",
    "top_preview_score",
    "policy_status",
    "blocking_reasons",
    "recommended_action",
]

CONSENSUS_RERANK_ACTION_QUEUE_COLUMNS = [
    "action_priority",
    "pocket_id",
    "issue_type",
    "issue_severity",
    "preview_decision",
    "suggestion_status",
    "current_rank",
    "preview_rank",
    "preview_rank_delta",
    "pocket_consensus_label",
    "rank_safe_anchor_count",
    "blocked_ai_count",
    "weak_mapping_count",
    "required_fix",
    "can_apply_after_fix",
    "policy_status",
    "consensus_anchor_residues",
    "recommended_action",
]

CONSENSUS_RERANK_APPLY_SIMULATION_COLUMNS = [
    "simulated_rank",
    "pocket_id",
    "current_rank",
    "preview_rank",
    "simulated_rank_delta",
    "current_decision_score",
    "preview_score",
    "simulation_score",
    "simulation_score_source",
    "apply_status",
    "apply_decision",
    "policy_status",
    "policy_allows_apply",
    "issue_type",
    "issue_severity",
    "rank_safe_anchor_count",
    "blocked_ai_count",
    "weak_mapping_count",
    "required_before_apply",
    "consensus_anchor_residues",
    "recommended_action",
]

CONSENSUS_RERANK_SIMULATION_DELTA_COLUMNS = [
    "impact_priority",
    "pocket_id",
    "change_type",
    "change_severity",
    "current_rank",
    "simulated_rank",
    "rank_delta",
    "current_decision_score",
    "simulation_score",
    "score_delta",
    "apply_status",
    "apply_decision",
    "issue_type",
    "issue_severity",
    "policy_status",
    "precision_interpretation",
    "explanation",
    "required_before_trust",
    "consensus_anchor_residues",
    "recommended_action",
]

CONSENSUS_RERANK_PRECISION_SCORECARD_COLUMNS = [
    "scorecard_status",
    "precision_improvement_score",
    "simulation_rows",
    "positive_signal_rows",
    "negative_control_rows",
    "open_blocker_rows",
    "rank_up_rows",
    "rank_down_rows",
    "unchanged_ready_rows",
    "score_only_rows",
    "frozen_blocker_rows",
    "mapping_review_rows",
    "evidence_gap_rows",
    "ai_source_review_rows",
    "monitor_rows",
    "policy_status",
    "policy_allows_apply_rows",
    "top_positive_pocket_id",
    "top_blocker_pocket_id",
    "score_reason",
    "recommended_action",
]

CONSENSUS_RERANK_PRECISION_GUARDRAIL_COLUMNS = [
    "guardrail_status",
    "guardrail_decision",
    "apply_mode",
    "can_enable_auto_rerank",
    "can_apply_after_manual_review",
    "manual_review_required",
    "precision_improvement_score",
    "scorecard_status",
    "policy_status",
    "positive_signal_rows",
    "open_blocker_rows",
    "required_clearance_count",
    "first_required_clearance",
    "top_positive_pocket_id",
    "top_blocker_pocket_id",
    "decision_reason",
    "recommended_action",
]

CONSENSUS_RERANK_GUARDRAIL_ARTIFACT_MANIFEST_COLUMNS = [
    "artifact_name",
    "file_name",
    "artifact_type",
    "row_count",
    "byte_size",
    "sha256",
    "status",
    "purpose",
    "recommended_use",
]

CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_COLUMNS = [
    "file_name",
    "expected_byte_size",
    "actual_byte_size",
    "expected_sha256",
    "actual_sha256",
    "verification_status",
    "issue",
    "recommended_action",
]

CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_SUMMARY_COLUMNS = [
    "verification_status",
    "manifest_rows",
    "checked_files",
    "verified_files",
    "failed_files",
    "missing_files",
    "size_mismatch_files",
    "hash_mismatch_files",
    "unlisted_files",
    "invalid_zip_rows",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS = [
    "decision_item_id",
    "decision_scope",
    "pocket_id",
    "decision_item",
    "current_guardrail_status",
    "current_issue_type",
    "current_change_type",
    "recommended_decision",
    "review_decision",
    "reviewer",
    "review_note",
    "verified_anchor_residues",
    "verified_sources",
    "blocker_resolved",
    "manual_approval_allowed",
    "decision_due_to",
    "required_evidence",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_DECISION_VALIDATION_COLUMNS = [
    "row_index",
    "decision_item_id",
    "decision_scope",
    "pocket_id",
    "review_decision",
    "template_match",
    "recommended_decision",
    "manual_approval_allowed",
    "validation_status",
    "issue_flags",
    "can_release",
    "validation_reason",
    "required_fix",
    "reviewer",
    "verified_anchor_residues",
    "verified_sources",
    "blocker_resolved",
]

CONSENSUS_RERANK_RELEASE_DECISION_SUMMARY_COLUMNS = [
    "release_review_status",
    "template_rows",
    "decision_rows",
    "matched_rows",
    "approved_rows",
    "rejected_rows",
    "hold_rows",
    "review_rows",
    "blocked_rows",
    "warning_rows",
    "unmatched_rows",
    "missing_decision_rows",
    "missing_reviewer_rows",
    "missing_evidence_rows",
    "unresolved_blocker_rows",
    "release_allowed",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS = [
    "manual_apply_rank",
    "pocket_id",
    "current_rank",
    "simulated_rank",
    "rank_delta",
    "current_decision_score",
    "simulation_score",
    "apply_status",
    "apply_decision",
    "release_apply_status",
    "release_review_status",
    "release_allowed",
    "approval_reference",
    "required_pre_apply_check",
    "consensus_anchor_residues",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS = [
    "execution_item_id",
    "manual_apply_rank",
    "pocket_id",
    "expected_current_rank",
    "expected_simulated_rank",
    "expected_rank_delta",
    "expected_apply_status",
    "expected_apply_decision",
    "expected_release_apply_status",
    "plan_sha256",
    "execution_decision",
    "applied_rank",
    "operator",
    "executed_at",
    "execution_note",
    "required_pre_apply_check",
    "approval_reference",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_EXECUTION_VALIDATION_COLUMNS = [
    "row_index",
    "execution_item_id",
    "pocket_id",
    "execution_decision",
    "template_match",
    "expected_rank",
    "applied_rank",
    "plan_hash_match",
    "validation_status",
    "issue_flags",
    "execution_accepted",
    "validation_reason",
    "required_fix",
    "operator",
    "executed_at",
    "plan_sha256",
]

CONSENSUS_RERANK_RELEASE_EXECUTION_SUMMARY_COLUMNS = [
    "execution_review_status",
    "template_rows",
    "receipt_rows",
    "matched_rows",
    "applied_rows",
    "skipped_rows",
    "failed_rows",
    "pending_rows",
    "blocked_rows",
    "rank_mismatch_rows",
    "missing_operator_rows",
    "missing_executed_at_rows",
    "plan_hash_mismatch_rows",
    "missing_receipt_rows",
    "execution_complete",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_CLOSURE_LEDGER_COLUMNS = [
    "evidence_item",
    "file_name",
    "artifact_type",
    "row_count",
    "byte_size",
    "sha256",
    "status",
    "required_for_closure",
    "closure_check",
    "issue",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_CLOSURE_SUMMARY_COLUMNS = [
    "closure_readiness_status",
    "ledger_rows",
    "required_rows",
    "ok_rows",
    "blocked_rows",
    "missing_rows",
    "missing_hash_rows",
    "manifest_rows",
    "bundle_verification_status",
    "bundle_failed_files",
    "bundle_verified",
    "release_closed",
    "recommended_action",
]

CONSENSUS_RERANK_RELEASE_CLOSURE_BLOCKER_COLUMNS = [
    "blocker_rank",
    "blocker_source",
    "evidence_item",
    "file_name",
    "blocker_type",
    "severity",
    "current_status",
    "issue",
    "required_fix",
    "recommended_action",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(default)
    if pd.isna(numeric):
        return int(default)
    return int(numeric)


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _safe_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _first_by_pocket(table: Optional[pd.DataFrame], pocket_id: str) -> Optional[pd.Series]:
    if table is None or getattr(table, "empty", True) or "pocket_id" not in table.columns:
        return None
    matched = table[table["pocket_id"].astype(str) == str(pocket_id)]
    if matched.empty:
        return None
    return matched.iloc[0]


def _empty_consensus_rerank_suggestion_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_SUGGESTION_COLUMNS)


def _empty_consensus_rerank_preview_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_PREVIEW_COLUMNS)


def _empty_consensus_rerank_policy_gate_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_POLICY_GATE_COLUMNS)


def _empty_consensus_rerank_action_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_ACTION_QUEUE_COLUMNS)


def _empty_consensus_rerank_apply_simulation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_APPLY_SIMULATION_COLUMNS)


def _empty_consensus_rerank_simulation_delta_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_SIMULATION_DELTA_COLUMNS)


def _empty_consensus_rerank_precision_scorecard_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_PRECISION_SCORECARD_COLUMNS)


def _empty_consensus_rerank_precision_guardrail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_PRECISION_GUARDRAIL_COLUMNS)


def _empty_consensus_rerank_guardrail_artifact_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_GUARDRAIL_ARTIFACT_MANIFEST_COLUMNS)


def _empty_consensus_rerank_guardrail_bundle_verification_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_COLUMNS)


def _empty_consensus_rerank_guardrail_bundle_verification_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_SUMMARY_COLUMNS)


def _empty_consensus_rerank_release_decision_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS)


def _empty_consensus_rerank_release_decision_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_DECISION_VALIDATION_COLUMNS)


def _empty_consensus_rerank_release_decision_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_DECISION_SUMMARY_COLUMNS)


def _empty_consensus_rerank_release_apply_plan_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS)


def _empty_consensus_rerank_release_execution_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS)


def _empty_consensus_rerank_release_execution_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_EXECUTION_VALIDATION_COLUMNS)


def _empty_consensus_rerank_release_execution_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_EXECUTION_SUMMARY_COLUMNS)


def _empty_consensus_rerank_release_closure_ledger_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_CLOSURE_LEDGER_COLUMNS)


def _empty_consensus_rerank_release_closure_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_CLOSURE_SUMMARY_COLUMNS)


def _empty_consensus_rerank_release_closure_blocker_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONSENSUS_RERANK_RELEASE_CLOSURE_BLOCKER_COLUMNS)


def _ab_signal(table: Optional[pd.DataFrame], pocket_id: str) -> dict[str, Any]:
    row = _first_by_pocket(table, pocket_id)
    if row is None:
        return {"status": "", "rank_delta": 0, "score_delta": 0.0, "quality_delta": 0.0}
    return {
        "status": _safe_text(row.get("status")),
        "rank_delta": _safe_int(row.get("rank_delta"), 0),
        "score_delta": _safe_float(row.get("score_delta"), 0.0),
        "quality_delta": _safe_float(row.get("evidence_quality_delta"), 0.0),
    }


def _quality_weight(label: str) -> float:
    normalized = str(label or "").strip().lower()
    weights = {
        "strong-direct-anchor": 1.0,
        "direct-anchor": 0.84,
        "route-anchor": 0.68,
        "structure-verified-external": 0.64,
        "neighborhood-expanded": 0.42,
        "diffuse-external-support": 0.34,
        "no-external-evidence": 0.0,
        "geometry-only": 0.0,
    }
    return weights.get(normalized, 0.20 if normalized else 0.0)


def _functional_confidence(row: pd.Series, joint_row: Optional[pd.Series]) -> float:
    quality_label = _safe_text(row.get("evidence_quality_label"))
    quality_score = _safe_float(row.get("evidence_quality_score"))
    direct_anchor_count = _safe_int(row.get("external_direct_anchor_count"))
    route_anchor_count = _safe_int(row.get("evidence_route_anchor_count"))
    exact_ratio = _safe_float(row.get("external_exact_match_ratio"))
    support_mean = _safe_float(row.get("external_support_mean"))
    mapping_quality = _safe_float(row.get("external_mapping_quality_mean"))
    verified_count = _safe_int(row.get("external_structure_verified_count"))
    anchor_support = _safe_float(row.get("smart_evidence_anchor_support"))
    anchor_risk = _safe_float(row.get("smart_evidence_anchor_risk"))
    if joint_row is not None:
        anchor_support = max(anchor_support, _safe_float(joint_row.get("evidence_anchor_support")))
        anchor_risk = max(anchor_risk, _safe_float(joint_row.get("evidence_anchor_risk")))

    base = max(quality_score, _quality_weight(quality_label), support_mean, anchor_support)
    base += 0.09 if direct_anchor_count > 0 else 0.0
    base += 0.05 if route_anchor_count > 0 else 0.0
    base += 0.07 * _clip(exact_ratio)
    base += 0.06 * _clip(mapping_quality)
    base += 0.04 if verified_count > 0 else 0.0
    base -= 0.14 * _clip(anchor_risk)
    if quality_label in {"neighborhood-expanded", "diffuse-external-support"}:
        base -= 0.08
    if quality_label in {"geometry-only", "no-external-evidence"} and direct_anchor_count <= 0:
        base = min(base, 0.30)
    return round(_clip(base), 3)


def _geometry_confidence(row: pd.Series) -> float:
    smart_score = _safe_float(row.get("smart_rank_score"))
    consensus_score = _safe_float(row.get("consensus_score"))
    method_votes = _safe_int(row.get("method_vote_count"), 1)
    hotspot_count = _safe_int(row.get("hotspot_count"))
    burial_support = _safe_float(row.get("smart_burial_support"))
    exposure_penalty = _safe_float(row.get("smart_exposure_penalty"))

    method_support = _clip(float(method_votes) / 3.0)
    hotspot_support = _clip(float(hotspot_count) / 3.0)
    geometry = (
        0.56 * _clip(smart_score)
        + 0.16 * _clip(consensus_score)
        + 0.12 * method_support
        + 0.10 * hotspot_support
        + 0.10 * _clip(burial_support)
        - 0.12 * _clip(exposure_penalty)
    )
    return round(_clip(geometry), 3)


def _risk_flags(row: pd.Series, functional_confidence: float, literature_ab: dict[str, Any], evidence_route_ab: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    quality_label = _safe_text(row.get("evidence_quality_label"), "unknown")
    warning = _safe_text(row.get("evidence_quality_warning"))
    anchor_risk = _safe_float(row.get("smart_evidence_anchor_risk"))
    mapping_quality = _safe_float(row.get("external_mapping_quality_mean"))
    external_total = _safe_int(row.get("external_evidence_total"))

    if warning:
        flags.append("evidence-warning")
    if quality_label in {"geometry-only", "no-external-evidence"}:
        flags.append("needs-functional-evidence")
    if quality_label in {"neighborhood-expanded", "diffuse-external-support"} or anchor_risk >= 0.35:
        flags.append("neighborhood-expansion-risk")
    if external_total > 0 and mapping_quality < 0.45:
        flags.append("low-mapping-quality")
    if literature_ab.get("rank_delta", 0) < 0:
        flags.append("literature-lowered-rank")
    if evidence_route_ab.get("rank_delta", 0) < 0:
        flags.append("evidence-route-lowered-rank")
    if functional_confidence < 0.30:
        flags.append("geometry-dominated")
    return flags


def _decision_label(action: str, quality_label: str, functional_confidence: float, geometry_confidence: float, flags: list[str]) -> str:
    if action == "validate-prioritize" or (quality_label in {"strong-direct-anchor", "direct-anchor"} and functional_confidence >= 0.70):
        return "Evidence-led active-site candidate"
    if action == "review-evidence-mapping" or "low-mapping-quality" in flags or "neighborhood-expansion-risk" in flags:
        return "Review mapping before validation"
    if action == "validate-interface-context":
        return "Interface-supported candidate"
    if functional_confidence < 0.25 and geometry_confidence >= 0.45:
        return "Geometry-only exploratory pocket"
    if action == "shortlist-follow-up":
        return "Shortlist for follow-up"
    return "Exploratory candidate"


def _audit_status(action: str, flags: list[str]) -> str:
    if action == "validate-prioritize" and not flags:
        return "ready-to-validate"
    if "low-mapping-quality" in flags or "neighborhood-expansion-risk" in flags or "evidence-warning" in flags:
        return "mapping-review-needed"
    if "needs-functional-evidence" in flags or "geometry-dominated" in flags:
        return "needs-functional-evidence"
    if action == "exploratory-only":
        return "exploratory-only"
    return "shortlist"


def _next_step(action: str, flags: list[str]) -> str:
    if action == "validate-prioritize" and not flags:
        return "Prioritize residue-level validation around direct anchors."
    if "low-mapping-quality" in flags:
        return "Check UniProt/PDB residue mapping, chain choice, and numbering before validation."
    if "neighborhood-expansion-risk" in flags:
        return "Inspect direct anchors versus expanded neighborhood residues before trusting the pocket boundary."
    if "needs-functional-evidence" in flags:
        return "Add UniProt/M-CSA/literature or manual key residues before treating this as an active site."
    if action == "validate-interface-context":
        return "Validate with interface and hotspot context; functional residue evidence is still useful."
    return "Keep as a secondary candidate and compare against higher-confidence pockets."


def _visual_focus(row: pd.Series, flags: list[str]) -> str:
    anchors = _safe_text(row.get("evidence_anchor_residues"))
    if anchors:
        return f"Highlight direct anchors first: {anchors}"
    if "neighborhood-expansion-risk" in flags:
        return "Show core/shell split and inspect expanded residues around the nearest anchor."
    if "needs-functional-evidence" in flags:
        return "Show geometry pocket plus hotspot overlap; external key residues are missing."
    return "Show ranked pocket residues with hotspot and interface overlays."


def _supporting_evidence(row: pd.Series) -> str:
    parts = []
    sources = _safe_text(row.get("external_sources"))
    direct_sources = _safe_text(row.get("external_direct_sources"))
    evidence_types = _safe_text(row.get("external_evidence_types"))
    methods = _safe_text(row.get("consensus_methods"))
    if direct_sources:
        parts.append(f"direct: {direct_sources}")
    elif sources:
        parts.append(f"external: {sources}")
    if evidence_types:
        parts.append(f"types: {evidence_types}")
    if methods:
        parts.append(f"methods: {methods}")
    return " | ".join(parts) if parts else "geometry / hotspot context only"


def _risk_flag_set(value: Any) -> set[str]:
    text = _safe_text(value)
    if not text or text.lower() == "none":
        return set()
    return {part.strip() for part in text.split(",") if part.strip() and part.strip().lower() != "none"}


def _check_row(
    pocket_id: str,
    check_order: int,
    check: str,
    status: str,
    signal: str,
    why_it_matters: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "pocket_id": pocket_id,
        "check_order": check_order,
        "check": check,
        "status": status,
        "signal": signal,
        "why_it_matters": why_it_matters,
        "next_action": next_action,
    }


def _check_names_by_status(check_rows: pd.DataFrame, status: str) -> list[str]:
    if check_rows.empty or "status" not in check_rows.columns or "check" not in check_rows.columns:
        return []
    mask = check_rows["status"].astype(str).str.lower() == status
    return [str(value) for value in check_rows.loc[mask, "check"].tolist() if str(value).strip()]


def _triage_next_data(blocking_checks: list[str], review_checks: list[str], risk_flags: set[str]) -> str:
    checks = set(blocking_checks) | set(review_checks)
    suggestions: list[str] = []
    if "Functional anchors" in checks:
        suggestions.append("M-CSA / UniProt active-site annotations / PubMed key residues / manual catalytic residues")
    if "Evidence mapping risk" in checks or {"low-mapping-quality", "neighborhood-expansion-risk", "evidence-warning"} & risk_flags:
        suggestions.append("SIFTS chain mapping, insertion codes, UniProt offsets, and author numbering audit")
    if "Geometry consensus" in checks:
        suggestions.append("P2Rank/fpocket comparison, ligand-neighborhood contacts, or broader geometry detection")
    if "Evidence A/B movement" in checks:
        suggestions.append("literature/evidence-route/conservation A/B comparison")
    if "Actionability" in checks:
        suggestions.append("manual review note that turns the candidate into validate/review/explore")
    return " | ".join(suggestions) if suggestions else "No additional evidence required before validation."


def build_pocket_decision_table(
    pocket_summary: Optional[pd.DataFrame],
    joint_candidate_df: Optional[pd.DataFrame] = None,
    *,
    literature_ab_df: Optional[pd.DataFrame] = None,
    evidence_route_ab_df: Optional[pd.DataFrame] = None,
    conservation_ab_df: Optional[pd.DataFrame] = None,
    max_rows: int = 6,
) -> pd.DataFrame:
    if pocket_summary is None or getattr(pocket_summary, "empty", True) or "pocket_id" not in pocket_summary.columns:
        return pd.DataFrame(columns=POCKET_DECISION_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, summary_row in pocket_summary.iterrows():
        pocket_id = _safe_text(summary_row.get("pocket_id"))
        if not pocket_id:
            continue
        joint_row = _first_by_pocket(joint_candidate_df, pocket_id)
        action = _safe_text(joint_row.get("recommendation_action") if joint_row is not None else "", "shortlist-follow-up")
        quality_label = _safe_text(summary_row.get("evidence_quality_label"), "unknown")
        functional = _functional_confidence(summary_row, joint_row)
        geometry = _geometry_confidence(summary_row)
        literature_ab = _ab_signal(literature_ab_df, pocket_id)
        evidence_route_ab = _ab_signal(evidence_route_ab_df, pocket_id)
        conservation_ab = _ab_signal(conservation_ab_df, pocket_id)
        ab_boost = 0.03 * max(0, literature_ab["rank_delta"]) + 0.04 * max(0, evidence_route_ab["rank_delta"]) + 0.02 * max(0, conservation_ab["rank_delta"])
        flags = _risk_flags(summary_row, functional, literature_ab, evidence_route_ab)
        decision_score = _clip(0.55 * functional + 0.35 * geometry + min(0.10, ab_boost) - 0.035 * len(flags))
        label = _decision_label(action, quality_label, functional, geometry, flags)

        rows.append(
            {
                "decision_rank": 0,
                "pocket_id": pocket_id,
                "decision_label": label,
                "decision_score": round(decision_score, 3),
                "functional_confidence": functional,
                "geometry_confidence": geometry,
                "recommended_action": action,
                "audit_status": _audit_status(action, flags),
                "evidence_quality_label": quality_label,
                "evidence_quality_score": round(_safe_float(summary_row.get("evidence_quality_score")), 3),
                "direct_anchor_count": _safe_int(summary_row.get("external_direct_anchor_count")),
                "route_anchor_count": _safe_int(summary_row.get("evidence_route_anchor_count")),
                "anchor_residues": _safe_text(summary_row.get("evidence_anchor_residues")),
                "method_vote_count": _safe_int(summary_row.get("method_vote_count"), 1),
                "smart_rank_label": _safe_text(summary_row.get("smart_rank_label")),
                "smart_rank_score": round(_safe_float(summary_row.get("smart_rank_score")), 3),
                "literature_rank_delta": literature_ab["rank_delta"],
                "evidence_route_rank_delta": evidence_route_ab["rank_delta"],
                "conservation_rank_delta": conservation_ab["rank_delta"],
                "risk_flags": ", ".join(flags) if flags else "none",
                "supporting_evidence": _supporting_evidence(summary_row),
                "next_step": _next_step(action, flags),
                "visual_focus": _visual_focus(summary_row, flags),
            }
        )

    if not rows:
        return pd.DataFrame(columns=POCKET_DECISION_COLUMNS)

    result = pd.DataFrame(rows, columns=POCKET_DECISION_COLUMNS)
    result = result.sort_values(
        ["decision_score", "functional_confidence", "geometry_confidence", "pocket_id"],
        ascending=[False, False, False, True],
    ).head(max(1, int(max_rows))).reset_index(drop=True)
    result["decision_rank"] = range(1, len(result) + 1)
    return result


def _consensus_suggestion_for_row(
    *,
    current_rank: int,
    consensus_rank: int,
    label: str,
    rank_safe_anchor_count: int,
    best_consensus_score: float,
    blocked_ai_count: int,
    weak_mapping_count: int,
) -> tuple[str, str, str]:
    strong_label = label in {"consensus-validated-pocket", "consensus-supported-pocket"}
    ai_label = label == "ai-supported-pocket"
    has_current_rank = current_rank > 0
    rank_delta = current_rank - consensus_rank if has_current_rank and consensus_rank > 0 else 0

    if blocked_ai_count > 0 and (not has_current_rank or current_rank <= 3) and rank_safe_anchor_count == 0:
        return (
            "demote-or-block",
            "Consensus evidence is dominated by blocked AI residues, so promotion would lower precision.",
            "Keep visible for review, but do not promote until blocked AI evidence is fixed or rejected.",
        )
    if weak_mapping_count > 0 and rank_safe_anchor_count == 0:
        return (
            "review-mapping-before-rerank",
            "Consensus overlap exists but depends on weak residue mapping.",
            "Resolve chain/numbering alignment before using consensus to change ranking.",
        )
    if strong_label and not has_current_rank:
        return (
            "promote-consensus-candidate",
            "A rank-safe consensus pocket is absent from the current decision shortlist.",
            "Inspect this pocket manually and consider adding it to the validation shortlist.",
        )
    if strong_label and rank_delta > 0:
        return (
            "promote-consensus",
            "Consensus evidence ranks this pocket above its current decision rank.",
            "Promote only after confirming the overlapping anchor residues and pocket boundary.",
        )
    if strong_label and current_rank == 1:
        return (
            "keep-prioritized",
            "The current top decision is also supported by rank-safe consensus anchors.",
            "Keep current ranking and use consensus residues as validation anchors.",
        )
    if ai_label and rank_delta > 0:
        return (
            "ai-supported-review",
            "AI-supported consensus suggests a possible promotion, but the evidence still needs source verification.",
            "Review citations/snippets before promoting this pocket.",
        )
    if label in {"conservation-context-pocket", "no-consensus-anchor-pocket"} and has_current_rank and current_rank <= 3:
        return (
            "evidence-gap-review",
            "The current shortlist pocket lacks rank-safe functional consensus anchors.",
            "Collect catalytic, binding, mutagenesis, M-CSA, UniProt, or literature evidence before final validation.",
        )
    if best_consensus_score > 0.0:
        return (
            "monitor-consensus",
            "Consensus evidence is present but does not justify a ranking change yet.",
            "Keep as supporting context and compare against higher-confidence pockets.",
        )
    return (
        "no-consensus-change",
        "No consensus signal is available for this pocket.",
        "Do not change ranking from consensus evidence.",
    )


def build_consensus_rerank_suggestion(
    decision_df: Optional[pd.DataFrame],
    pocket_consensus_coverage_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if (
        (decision_df is None or getattr(decision_df, "empty", True))
        and (pocket_consensus_coverage_df is None or getattr(pocket_consensus_coverage_df, "empty", True))
    ):
        return _empty_consensus_rerank_suggestion_df()

    decision_table = decision_df.copy() if decision_df is not None and not getattr(decision_df, "empty", True) else pd.DataFrame()
    coverage_table = pocket_consensus_coverage_df.copy() if pocket_consensus_coverage_df is not None and not getattr(pocket_consensus_coverage_df, "empty", True) else pd.DataFrame()

    if not decision_table.empty and "pocket_id" not in decision_table.columns:
        decision_table = pd.DataFrame()
    if not coverage_table.empty and "pocket_id" not in coverage_table.columns:
        coverage_table = pd.DataFrame()
    if decision_table.empty and coverage_table.empty:
        return _empty_consensus_rerank_suggestion_df()

    if not decision_table.empty:
        decision_table["pocket_id"] = decision_table["pocket_id"].astype(str)
        if "decision_rank" not in decision_table.columns:
            decision_table["decision_rank"] = range(1, len(decision_table) + 1)
        decision_table["decision_rank"] = pd.to_numeric(decision_table["decision_rank"], errors="coerce")
        if "decision_score" not in decision_table.columns:
            decision_table["decision_score"] = 0.0
        decision_table["decision_score"] = pd.to_numeric(decision_table["decision_score"], errors="coerce").fillna(0.0)
        decision_table = decision_table.sort_values(["decision_rank", "decision_score", "pocket_id"], ascending=[True, False, True])
        decision_table = decision_table.drop_duplicates(subset=["pocket_id"], keep="first").set_index("pocket_id", drop=False)

    if not coverage_table.empty:
        coverage_table["pocket_id"] = coverage_table["pocket_id"].astype(str)
        if "consensus_rank" not in coverage_table.columns:
            coverage_table["consensus_rank"] = range(1, len(coverage_table) + 1)
        coverage_table["consensus_rank"] = pd.to_numeric(coverage_table["consensus_rank"], errors="coerce")
        for column in ("rank_safe_anchor_count", "best_consensus_score", "blocked_ai_count", "weak_mapping_count"):
            if column not in coverage_table.columns:
                coverage_table[column] = 0
        coverage_table = coverage_table.sort_values(["consensus_rank", "best_consensus_score", "pocket_id"], ascending=[True, False, True])
        coverage_table = coverage_table.drop_duplicates(subset=["pocket_id"], keep="first").set_index("pocket_id", drop=False)

    pocket_ids = sorted(set(decision_table.index.tolist() if not decision_table.empty else []) | set(coverage_table.index.tolist() if not coverage_table.empty else []))
    rows: list[dict[str, Any]] = []
    for pocket_id in pocket_ids:
        decision_row = decision_table.loc[pocket_id] if not decision_table.empty and pocket_id in decision_table.index else None
        coverage_row = coverage_table.loc[pocket_id] if not coverage_table.empty and pocket_id in coverage_table.index else None

        current_rank = _safe_int(decision_row.get("decision_rank"), 0) if decision_row is not None else 0
        consensus_rank = _safe_int(coverage_row.get("consensus_rank"), 0) if coverage_row is not None else 0
        decision_score = _safe_float(decision_row.get("decision_score"), 0.0) if decision_row is not None else 0.0
        label = _safe_text(coverage_row.get("pocket_consensus_label"), "no-consensus-anchor-pocket") if coverage_row is not None else "no-consensus-anchor-pocket"
        rank_safe_anchor_count = _safe_int(coverage_row.get("rank_safe_anchor_count"), 0) if coverage_row is not None else 0
        best_consensus_score = _safe_float(coverage_row.get("best_consensus_score"), 0.0) if coverage_row is not None else 0.0
        blocked_ai_count = _safe_int(coverage_row.get("blocked_ai_count"), 0) if coverage_row is not None else 0
        weak_mapping_count = _safe_int(coverage_row.get("weak_mapping_count"), 0) if coverage_row is not None else 0
        rank_delta = current_rank - consensus_rank if current_rank > 0 and consensus_rank > 0 else None
        status, reason, action = _consensus_suggestion_for_row(
            current_rank=current_rank,
            consensus_rank=consensus_rank,
            label=label,
            rank_safe_anchor_count=rank_safe_anchor_count,
            best_consensus_score=best_consensus_score,
            blocked_ai_count=blocked_ai_count,
            weak_mapping_count=weak_mapping_count,
        )
        rows.append(
            {
                "pocket_id": pocket_id,
                "current_decision_rank": current_rank if current_rank > 0 else None,
                "consensus_rank": consensus_rank if consensus_rank > 0 else None,
                "rank_delta": rank_delta,
                "current_decision_score": round(float(decision_score), 3),
                "pocket_consensus_label": label,
                "rank_safe_anchor_count": rank_safe_anchor_count,
                "best_consensus_score": round(float(best_consensus_score), 3),
                "blocked_ai_count": blocked_ai_count,
                "weak_mapping_count": weak_mapping_count,
                "suggestion_status": status,
                "suggestion_reason": reason,
                "recommended_action": action,
                "consensus_anchor_residues": _safe_text(coverage_row.get("consensus_anchor_residues"), "none") if coverage_row is not None else "none",
            }
        )

    if not rows:
        return _empty_consensus_rerank_suggestion_df()
    status_rank = {
        "promote-consensus-candidate": 0,
        "promote-consensus": 1,
        "keep-prioritized": 2,
        "ai-supported-review": 3,
        "review-mapping-before-rerank": 4,
        "demote-or-block": 5,
        "evidence-gap-review": 6,
        "monitor-consensus": 7,
        "no-consensus-change": 8,
    }
    result = pd.DataFrame(rows, columns=CONSENSUS_RERANK_SUGGESTION_COLUMNS)
    result["_status_rank"] = result["suggestion_status"].map(status_rank).fillna(99)
    result["_current_rank_sort"] = pd.to_numeric(result["current_decision_rank"], errors="coerce").fillna(9999)
    result = result.sort_values(
        ["_status_rank", "best_consensus_score", "_current_rank_sort", "pocket_id"],
        ascending=[True, False, True, True],
    ).drop(columns=["_status_rank", "_current_rank_sort"]).reset_index(drop=True)
    return result[CONSENSUS_RERANK_SUGGESTION_COLUMNS]


def _consensus_preview_adjustment(
    *,
    suggestion_status: str,
    best_consensus_score: float,
    rank_safe_anchor_count: int,
    blocked_ai_count: int,
    weak_mapping_count: int,
) -> tuple[float, str]:
    status = _safe_text(suggestion_status).lower()
    best_score = _clip(best_consensus_score)
    safe_anchor_bonus = min(0.03, max(0, rank_safe_anchor_count) * 0.012)
    if status == "promote-consensus-candidate":
        return round(0.12 + 0.04 * best_score + safe_anchor_bonus, 3), "New consensus-backed candidate receives a conservative preview boost."
    if status == "promote-consensus":
        return round(0.09 + 0.035 * best_score + safe_anchor_bonus, 3), "Validated/supported consensus anchors justify a preview promotion."
    if status == "keep-prioritized":
        return round(0.025 + safe_anchor_bonus, 3), "Consensus confirms the current priority without requiring a large score change."
    if status == "ai-supported-review":
        return 0.015, "AI-supported consensus is kept as a small preview boost until source review is complete."
    if status == "review-mapping-before-rerank":
        penalty = 0.04 + min(0.05, max(0, weak_mapping_count) * 0.02)
        return -round(penalty, 3), "Weak residue mapping reduces preview confidence until numbering is resolved."
    if status == "demote-or-block":
        penalty = 0.14 + min(0.10, max(0, blocked_ai_count) * 0.04)
        return -round(penalty, 3), "Blocked AI evidence prevents consensus-based promotion."
    if status == "evidence-gap-review":
        return -0.055, "The pocket lacks rank-safe functional consensus anchors."
    if status == "monitor-consensus":
        return 0.0, "Consensus is present but not strong enough for a preview ranking change."
    return 0.0, "No consensus signal changes the preview score."


def _preview_decision(
    *,
    current_rank: int,
    preview_rank: int,
    suggestion_status: str,
) -> str:
    status = _safe_text(suggestion_status).lower()
    if status in {"demote-or-block", "review-mapping-before-rerank"}:
        return "would-demote-or-block"
    if current_rank <= 0 and preview_rank > 0:
        return "would-enter-preview"
    if current_rank > 0 and preview_rank > 0 and preview_rank < current_rank:
        return "would-move-up"
    if current_rank > 0 and preview_rank > current_rank:
        return "would-move-down"
    if status == "keep-prioritized":
        return "would-keep-priority"
    if status in {"ai-supported-review", "evidence-gap-review", "monitor-consensus"}:
        return "would-review-before-change"
    return "would-keep-current"


def build_consensus_rerank_preview(
    decision_df: Optional[pd.DataFrame],
    consensus_rerank_suggestion_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if consensus_rerank_suggestion_df is None or getattr(consensus_rerank_suggestion_df, "empty", True) or "pocket_id" not in consensus_rerank_suggestion_df.columns:
        return _empty_consensus_rerank_preview_df()

    suggestions = consensus_rerank_suggestion_df.copy()
    suggestions["pocket_id"] = suggestions["pocket_id"].astype(str)
    if "current_decision_rank" not in suggestions.columns:
        suggestions["current_decision_rank"] = 0
    if "current_decision_score" not in suggestions.columns:
        suggestions["current_decision_score"] = 0.0
    if "suggestion_status" not in suggestions.columns:
        suggestions["suggestion_status"] = "no-consensus-change"
    if "pocket_consensus_label" not in suggestions.columns:
        suggestions["pocket_consensus_label"] = "no-consensus-anchor-pocket"
    for column in ("rank_safe_anchor_count", "best_consensus_score", "blocked_ai_count", "weak_mapping_count"):
        if column not in suggestions.columns:
            suggestions[column] = 0
    if "recommended_action" not in suggestions.columns:
        suggestions["recommended_action"] = ""
    if "consensus_anchor_residues" not in suggestions.columns:
        suggestions["consensus_anchor_residues"] = "none"

    decision_table = decision_df.copy() if decision_df is not None and not getattr(decision_df, "empty", True) and "pocket_id" in decision_df.columns else pd.DataFrame()
    if not decision_table.empty:
        decision_table["pocket_id"] = decision_table["pocket_id"].astype(str)
        if "decision_rank" not in decision_table.columns:
            decision_table["decision_rank"] = range(1, len(decision_table) + 1)
        if "decision_score" not in decision_table.columns:
            decision_table["decision_score"] = 0.0
        decision_table["decision_rank"] = pd.to_numeric(decision_table["decision_rank"], errors="coerce")
        decision_table["decision_score"] = pd.to_numeric(decision_table["decision_score"], errors="coerce").fillna(0.0)
        decision_table = decision_table.sort_values(["decision_rank", "decision_score", "pocket_id"], ascending=[True, False, True])
        decision_table = decision_table.drop_duplicates(subset=["pocket_id"], keep="first").set_index("pocket_id", drop=False)

    rows: list[dict[str, Any]] = []
    for _, suggestion in suggestions.drop_duplicates(subset=["pocket_id"], keep="first").iterrows():
        pocket_id = _safe_text(suggestion.get("pocket_id"))
        if not pocket_id:
            continue
        decision_row = decision_table.loc[pocket_id] if not decision_table.empty and pocket_id in decision_table.index else None
        current_rank = _safe_int(
            decision_row.get("decision_rank") if decision_row is not None else suggestion.get("current_decision_rank"),
            0,
        )
        current_score = _safe_float(
            decision_row.get("decision_score") if decision_row is not None else suggestion.get("current_decision_score"),
            0.0,
        )
        best_consensus_score = _safe_float(suggestion.get("best_consensus_score"), 0.0)
        if current_score <= 0.0 and _safe_text(suggestion.get("suggestion_status")) == "promote-consensus-candidate":
            current_score = _clip(0.48 + 0.22 * _clip(best_consensus_score))

        rank_safe_anchor_count = _safe_int(suggestion.get("rank_safe_anchor_count"), 0)
        blocked_ai_count = _safe_int(suggestion.get("blocked_ai_count"), 0)
        weak_mapping_count = _safe_int(suggestion.get("weak_mapping_count"), 0)
        adjustment, reason = _consensus_preview_adjustment(
            suggestion_status=_safe_text(suggestion.get("suggestion_status")),
            best_consensus_score=best_consensus_score,
            rank_safe_anchor_count=rank_safe_anchor_count,
            blocked_ai_count=blocked_ai_count,
            weak_mapping_count=weak_mapping_count,
        )
        preview_score = round(_clip(current_score + adjustment), 3)
        rows.append(
            {
                "pocket_id": pocket_id,
                "current_rank": current_rank if current_rank > 0 else None,
                "preview_rank": 0,
                "preview_rank_delta": None,
                "current_decision_score": round(float(current_score), 3),
                "consensus_adjustment": round(float(adjustment), 3),
                "preview_score": preview_score,
                "suggestion_status": _safe_text(suggestion.get("suggestion_status"), "no-consensus-change"),
                "pocket_consensus_label": _safe_text(suggestion.get("pocket_consensus_label"), "no-consensus-anchor-pocket"),
                "rank_safe_anchor_count": rank_safe_anchor_count,
                "blocked_ai_count": blocked_ai_count,
                "weak_mapping_count": weak_mapping_count,
                "preview_decision": "",
                "preview_reason": reason,
                "recommended_action": _safe_text(suggestion.get("recommended_action"), "Do not change ranking automatically."),
                "consensus_anchor_residues": _safe_text(suggestion.get("consensus_anchor_residues"), "none"),
            }
        )

    if not rows:
        return _empty_consensus_rerank_preview_df()
    result = pd.DataFrame(rows, columns=CONSENSUS_RERANK_PREVIEW_COLUMNS)
    result = result.sort_values(
        ["preview_score", "rank_safe_anchor_count", "current_rank", "pocket_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    result["preview_rank"] = range(1, len(result) + 1)
    preview_rank_by_pocket = result.set_index("pocket_id")["preview_rank"].to_dict()
    preview_decisions = []
    rank_deltas = []
    for _, row in result.iterrows():
        current_rank = _safe_int(row.get("current_rank"), 0)
        preview_rank = int(preview_rank_by_pocket.get(row.get("pocket_id"), 0) or 0)
        rank_deltas.append(current_rank - preview_rank if current_rank > 0 and preview_rank > 0 else None)
        preview_decisions.append(
            _preview_decision(
                current_rank=current_rank,
                preview_rank=preview_rank,
                suggestion_status=_safe_text(row.get("suggestion_status")),
            )
        )
    result["preview_rank_delta"] = rank_deltas
    result["preview_decision"] = preview_decisions
    return result[CONSENSUS_RERANK_PREVIEW_COLUMNS]


def build_consensus_rerank_policy_gate(
    consensus_rerank_preview_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if consensus_rerank_preview_df is None or getattr(consensus_rerank_preview_df, "empty", True) or "pocket_id" not in consensus_rerank_preview_df.columns:
        return _empty_consensus_rerank_policy_gate_df()

    preview = consensus_rerank_preview_df.copy()
    if "preview_decision" not in preview.columns:
        preview["preview_decision"] = "would-keep-current"
    if "suggestion_status" not in preview.columns:
        preview["suggestion_status"] = "no-consensus-change"
    if "preview_rank_delta" not in preview.columns:
        preview["preview_rank_delta"] = 0
    if "preview_score" not in preview.columns:
        preview["preview_score"] = 0.0
    for column in ("blocked_ai_count", "weak_mapping_count", "rank_safe_anchor_count"):
        if column not in preview.columns:
            preview[column] = 0

    preview["preview_rank_delta"] = pd.to_numeric(preview["preview_rank_delta"], errors="coerce").fillna(0).astype(int)
    preview["preview_score"] = pd.to_numeric(preview["preview_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    preview["blocked_ai_count"] = pd.to_numeric(preview["blocked_ai_count"], errors="coerce").fillna(0).astype(int)
    preview["weak_mapping_count"] = pd.to_numeric(preview["weak_mapping_count"], errors="coerce").fillna(0).astype(int)
    decision_series = preview["preview_decision"].astype(str).str.lower()
    status_series = preview["suggestion_status"].astype(str).str.lower()

    changed_mask = preview["preview_rank_delta"].ne(0) | decision_series.isin(
        {"would-move-up", "would-move-down", "would-enter-preview", "would-demote-or-block"}
    )
    promote_mask = decision_series.isin({"would-move-up", "would-enter-preview"}) | status_series.isin(
        {"promote-consensus", "promote-consensus-candidate"}
    )
    demote_mask = decision_series.isin({"would-move-down", "would-demote-or-block"}) | status_series.eq("demote-or-block")
    blocked_mask = status_series.eq("demote-or-block") | (
        preview["blocked_ai_count"].gt(0) & pd.to_numeric(preview.get("rank_safe_anchor_count", 0), errors="coerce").fillna(0).eq(0)
    )
    mapping_mask = status_series.eq("review-mapping-before-rerank") | preview["weak_mapping_count"].gt(0)
    evidence_gap_mask = status_series.eq("evidence-gap-review")
    keep_priority_mask = decision_series.eq("would-keep-priority")

    changed_rows = int(changed_mask.sum())
    promote_rows = int(promote_mask.sum())
    demote_rows = int(demote_mask.sum())
    blocked_rows = int(blocked_mask.sum())
    mapping_review_rows = int(mapping_mask.sum())
    evidence_gap_rows = int(evidence_gap_mask.sum())
    keep_priority_rows = int(keep_priority_mask.sum())

    top_row = preview.sort_values(["preview_score", "pocket_id"], ascending=[False, True]).iloc[0]
    reasons: list[str] = []
    if blocked_rows > 0:
        reasons.append(f"blocked_ai={blocked_rows}")
    if mapping_review_rows > 0:
        reasons.append(f"mapping_review={mapping_review_rows}")
    if evidence_gap_rows > 0:
        reasons.append(f"evidence_gap={evidence_gap_rows}")

    if blocked_rows > 0:
        policy_status = "blocked"
        action = "Do not enable automatic consensus rerank until blocked AI evidence is fixed, rejected, or excluded."
    elif mapping_review_rows > 0:
        policy_status = "mapping-review"
        action = "Resolve weak residue mapping before applying consensus-driven ranking changes."
    elif evidence_gap_rows > 0 and promote_rows == 0:
        policy_status = "needs-evidence"
        action = "Keep preview only; collect functional residue evidence before enabling rerank."
    elif changed_rows == 0 and keep_priority_rows > 0:
        policy_status = "no-change-needed"
        action = "Current ranking already agrees with consensus; use consensus residues as validation anchors."
    elif promote_rows > 0 or demote_rows > 0:
        policy_status = "review-before-apply"
        action = "Review preview rank changes manually before enabling a consensus rerank policy."
    else:
        policy_status = "monitor"
        action = "Keep the preview as a diagnostic layer; no automatic rerank is recommended yet."

    return pd.DataFrame(
        [
            {
                "preview_rows": int(len(preview)),
                "changed_rows": changed_rows,
                "promote_rows": promote_rows,
                "demote_rows": demote_rows,
                "blocked_rows": blocked_rows,
                "mapping_review_rows": mapping_review_rows,
                "evidence_gap_rows": evidence_gap_rows,
                "keep_priority_rows": keep_priority_rows,
                "top_preview_pocket_id": _safe_text(top_row.get("pocket_id"), "-"),
                "top_preview_decision": _safe_text(top_row.get("preview_decision"), "-"),
                "top_preview_score": round(_safe_float(top_row.get("preview_score"), 0.0), 3),
                "policy_status": policy_status,
                "blocking_reasons": ", ".join(reasons) if reasons else "none",
                "recommended_action": action,
            }
        ],
        columns=CONSENSUS_RERANK_POLICY_GATE_COLUMNS,
    )


def _action_queue_row_for_preview(
    row: pd.Series,
    *,
    policy_status: str,
) -> dict[str, Any]:
    suggestion_status = _safe_text(row.get("suggestion_status"), "no-consensus-change")
    preview_decision = _safe_text(row.get("preview_decision"), "would-keep-current")
    blocked_ai_count = _safe_int(row.get("blocked_ai_count"), 0)
    weak_mapping_count = _safe_int(row.get("weak_mapping_count"), 0)
    rank_safe_anchor_count = _safe_int(row.get("rank_safe_anchor_count"), 0)

    if suggestion_status == "demote-or-block" or (blocked_ai_count > 0 and rank_safe_anchor_count == 0):
        priority = 1
        issue_type = "blocked-ai-evidence"
        severity = "blocking"
        required_fix = "Reject or repair blocked AI residue evidence before this pocket can benefit from consensus reranking."
        can_apply = False
    elif suggestion_status == "review-mapping-before-rerank" or weak_mapping_count > 0:
        priority = 2
        issue_type = "weak-residue-mapping"
        severity = "blocking" if rank_safe_anchor_count == 0 else "review"
        required_fix = "Resolve chain, residue numbering, insertion code, UniProt/PDB offset, or SIFTS mapping before applying rerank."
        can_apply = rank_safe_anchor_count > 0
    elif suggestion_status == "evidence-gap-review":
        priority = 3
        issue_type = "functional-evidence-gap"
        severity = "missing-evidence"
        required_fix = "Add catalytic, binding, mutagenesis, M-CSA, UniProt, or literature evidence before treating this pocket as evidence-backed."
        can_apply = False
    elif suggestion_status in {"promote-consensus", "promote-consensus-candidate"} or preview_decision in {"would-move-up", "would-enter-preview"}:
        priority = 4
        issue_type = "promotion-review"
        severity = "review"
        required_fix = "Confirm consensus anchor residues and pocket boundary before accepting the preview promotion."
        can_apply = True
    elif suggestion_status == "ai-supported-review":
        priority = 5
        issue_type = "ai-source-review"
        severity = "review"
        required_fix = "Verify PMID/DOI/title, source snippet, residue identity, and mapping before using AI-supported consensus."
        can_apply = False
    elif suggestion_status == "keep-prioritized" or preview_decision == "would-keep-priority":
        priority = 6
        issue_type = "validation-anchor-ready"
        severity = "pass"
        required_fix = "Use the consensus residues as the validation anchor set; no rerank change is needed."
        can_apply = True
    elif suggestion_status == "monitor-consensus":
        priority = 7
        issue_type = "monitor-consensus"
        severity = "monitor"
        required_fix = "Keep as supporting context; consensus is not strong enough to change ranking."
        can_apply = False
    else:
        priority = 8
        issue_type = "no-consensus-action"
        severity = "none"
        required_fix = "No consensus-driven action is currently recommended."
        can_apply = False

    return {
        "action_priority": priority,
        "pocket_id": _safe_text(row.get("pocket_id")),
        "issue_type": issue_type,
        "issue_severity": severity,
        "preview_decision": preview_decision,
        "suggestion_status": suggestion_status,
        "current_rank": _safe_int(row.get("current_rank"), 0) or None,
        "preview_rank": _safe_int(row.get("preview_rank"), 0) or None,
        "preview_rank_delta": _safe_int(row.get("preview_rank_delta"), 0),
        "pocket_consensus_label": _safe_text(row.get("pocket_consensus_label"), "no-consensus-anchor-pocket"),
        "rank_safe_anchor_count": rank_safe_anchor_count,
        "blocked_ai_count": blocked_ai_count,
        "weak_mapping_count": weak_mapping_count,
        "required_fix": required_fix,
        "can_apply_after_fix": bool(can_apply),
        "policy_status": policy_status or "unknown",
        "consensus_anchor_residues": _safe_text(row.get("consensus_anchor_residues"), "none"),
        "recommended_action": _safe_text(row.get("recommended_action"), required_fix),
    }


def build_consensus_rerank_action_queue(
    consensus_rerank_preview_df: Optional[pd.DataFrame],
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if consensus_rerank_preview_df is None or getattr(consensus_rerank_preview_df, "empty", True) or "pocket_id" not in consensus_rerank_preview_df.columns:
        return _empty_consensus_rerank_action_queue_df()

    preview = consensus_rerank_preview_df.copy()
    for column, default in {
        "preview_decision": "would-keep-current",
        "suggestion_status": "no-consensus-change",
        "pocket_consensus_label": "no-consensus-anchor-pocket",
        "consensus_anchor_residues": "none",
        "recommended_action": "",
    }.items():
        if column not in preview.columns:
            preview[column] = default
    for column in ("current_rank", "preview_rank", "preview_rank_delta", "rank_safe_anchor_count", "blocked_ai_count", "weak_mapping_count"):
        if column not in preview.columns:
            preview[column] = 0

    policy_status = "unknown"
    if consensus_rerank_policy_gate_df is not None and not getattr(consensus_rerank_policy_gate_df, "empty", True):
        policy_status = _safe_text(consensus_rerank_policy_gate_df.iloc[0].get("policy_status"), "unknown")

    rows = [
        _action_queue_row_for_preview(row, policy_status=policy_status)
        for _, row in preview.drop_duplicates(subset=["pocket_id"], keep="first").iterrows()
        if _safe_text(row.get("pocket_id"))
    ]
    if not rows:
        return _empty_consensus_rerank_action_queue_df()
    result = pd.DataFrame(rows, columns=CONSENSUS_RERANK_ACTION_QUEUE_COLUMNS)
    result["_rank_sort"] = pd.to_numeric(result["current_rank"], errors="coerce").fillna(9999)
    result = result.sort_values(
        ["action_priority", "_rank_sort", "preview_rank", "pocket_id"],
        ascending=[True, True, True, True],
    ).drop(columns="_rank_sort").reset_index(drop=True)
    return result[CONSENSUS_RERANK_ACTION_QUEUE_COLUMNS]


def build_consensus_rerank_action_checklist_markdown(
    consensus_rerank_action_queue_df: Optional[pd.DataFrame],
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Consensus rerank action checklist",
) -> str:
    policy_status = "unknown"
    blocking_reasons = "none"
    gate_action = "Review the action queue before applying consensus-driven rank changes."
    if consensus_rerank_policy_gate_df is not None and not getattr(consensus_rerank_policy_gate_df, "empty", True):
        gate_row = consensus_rerank_policy_gate_df.iloc[0]
        policy_status = _safe_text(gate_row.get("policy_status"), "unknown")
        blocking_reasons = _safe_text(gate_row.get("blocking_reasons"), "none")
        gate_action = _safe_text(gate_row.get("recommended_action"), gate_action)

    lines = [
        f"# {title}",
        "",
        "Use this checklist to review consensus-driven rerank changes before they affect the active pocket order.",
        "Only apply a rerank after blocked AI evidence, weak residue mapping, and functional evidence gaps are resolved.",
        "",
        f"- Policy status: `{policy_status}`",
        f"- Blocking reasons: `{blocking_reasons}`",
        f"- Gate recommendation: {gate_action}",
        "",
    ]
    if consensus_rerank_action_queue_df is None or getattr(consensus_rerank_action_queue_df, "empty", True):
        return "\n".join(lines + ["No consensus rerank action items are currently open."])

    queue = consensus_rerank_action_queue_df.copy()
    for column, default in {
        "action_priority": 999,
        "pocket_id": "",
        "issue_type": "review",
        "issue_severity": "review",
        "current_rank": "",
        "preview_rank": "",
        "preview_rank_delta": 0,
        "rank_safe_anchor_count": 0,
        "blocked_ai_count": 0,
        "weak_mapping_count": 0,
        "required_fix": "",
        "recommended_action": "",
        "can_apply_after_fix": False,
        "policy_status": policy_status,
        "consensus_anchor_residues": "none",
    }.items():
        if column not in queue.columns:
            queue[column] = default

    queue["_rank_sort"] = pd.to_numeric(queue["current_rank"], errors="coerce").fillna(9999)
    queue["_preview_rank_sort"] = pd.to_numeric(queue["preview_rank"], errors="coerce").fillna(9999)
    queue = queue.sort_values(
        ["action_priority", "_rank_sort", "_preview_rank_sort", "pocket_id"],
        ascending=[True, True, True, True],
    ).drop(columns=["_rank_sort", "_preview_rank_sort"])

    for index, row in enumerate(queue.itertuples(index=False), start=1):
        current_rank = _safe_text(getattr(row, "current_rank", ""), "-")
        preview_rank = _safe_text(getattr(row, "preview_rank", ""), "-")
        rank_delta = _safe_int(getattr(row, "preview_rank_delta", 0), 0)
        can_apply = _safe_bool(getattr(row, "can_apply_after_fix", False))
        lines.extend(
            [
                f"## {index}. {_safe_text(getattr(row, 'pocket_id', ''), 'Pocket')} - {_safe_text(getattr(row, 'issue_type', ''), 'review')}",
                "",
                f"- [ ] Required fix: {_safe_text(getattr(row, 'required_fix', ''), '-')}",
                f"- [ ] Recommended action: {_safe_text(getattr(row, 'recommended_action', ''), '-')}",
                f"- [ ] Verify consensus anchors: {_safe_text(getattr(row, 'consensus_anchor_residues', ''), 'none')}",
                f"- Priority: `{_safe_text(getattr(row, 'action_priority', ''), '-')}`; severity: `{_safe_text(getattr(row, 'issue_severity', ''), '-')}`",
                f"- Ranks: current `{current_rank}` -> preview `{preview_rank}` (delta {rank_delta:+d})",
                (
                    "- Evidence counters: "
                    f"safe anchors={_safe_int(getattr(row, 'rank_safe_anchor_count', 0), 0)}; "
                    f"blocked AI={_safe_int(getattr(row, 'blocked_ai_count', 0), 0)}; "
                    f"weak mapping={_safe_int(getattr(row, 'weak_mapping_count', 0), 0)}"
                ),
                f"- Policy status: `{_safe_text(getattr(row, 'policy_status', ''), policy_status)}`",
                f"- Can apply after fix: {'yes' if can_apply else 'no'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _default_action_for_preview(row: pd.Series, policy_status: str) -> dict[str, Any]:
    suggestion_status = _safe_text(row.get("suggestion_status"), "no-consensus-change")
    preview_decision = _safe_text(row.get("preview_decision"), "would-keep-current")
    blocked_ai_count = _safe_int(row.get("blocked_ai_count"), 0)
    weak_mapping_count = _safe_int(row.get("weak_mapping_count"), 0)
    rank_safe_anchor_count = _safe_int(row.get("rank_safe_anchor_count"), 0)
    action_row = _action_queue_row_for_preview(row, policy_status=policy_status)
    if suggestion_status in {"promote-consensus", "promote-consensus-candidate"} or preview_decision in {"would-move-up", "would-enter-preview"}:
        action_row["can_apply_after_fix"] = True
    if suggestion_status == "keep-prioritized" and rank_safe_anchor_count > 0:
        action_row["can_apply_after_fix"] = True
    if blocked_ai_count > 0 and rank_safe_anchor_count == 0:
        action_row["can_apply_after_fix"] = False
    if weak_mapping_count > 0 and rank_safe_anchor_count == 0:
        action_row["can_apply_after_fix"] = False
    return action_row


def _simulation_status_and_score(
    *,
    issue_type: str,
    issue_severity: str,
    can_apply_after_fix: bool,
    policy_allows_apply: bool,
    current_score: float,
    preview_score: float,
    preview_decision: str,
) -> tuple[str, float, str]:
    issue = _safe_text(issue_type).lower()
    severity = _safe_text(issue_severity).lower()
    decision = _safe_text(preview_decision).lower()

    if issue == "blocked-ai-evidence":
        return "blocked-currently", min(current_score, preview_score), "conservative-min-score"
    if issue == "weak-residue-mapping" and not can_apply_after_fix:
        return "mapping-review-required", min(current_score, preview_score), "conservative-min-score"
    if issue == "functional-evidence-gap":
        return "evidence-required", min(current_score, preview_score), "conservative-min-score"
    if issue == "ai-source-review":
        return "ai-source-review-required", min(current_score, preview_score), "conservative-min-score"
    if severity == "blocking" and not can_apply_after_fix:
        return "blocked-currently", min(current_score, preview_score), "conservative-min-score"
    if issue == "validation-anchor-ready":
        status = "keep-current-ready" if policy_allows_apply else "keep-current-diagnostic"
        return status, preview_score, "preview-score"
    if can_apply_after_fix and decision in {"would-move-up", "would-move-down", "would-enter-preview", "would-demote-or-block"}:
        status = "apply-ready-after-review" if policy_allows_apply else "candidate-after-fix"
        return status, preview_score, "preview-score"
    if can_apply_after_fix:
        status = "keep-after-review" if policy_allows_apply else "candidate-after-fix"
        return status, preview_score, "preview-score"
    return "monitor-only", current_score if current_score > 0 else preview_score, "current-score"


def build_consensus_rerank_apply_simulation(
    consensus_rerank_preview_df: Optional[pd.DataFrame],
    consensus_rerank_action_queue_df: Optional[pd.DataFrame] = None,
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if consensus_rerank_preview_df is None or getattr(consensus_rerank_preview_df, "empty", True) or "pocket_id" not in consensus_rerank_preview_df.columns:
        return _empty_consensus_rerank_apply_simulation_df()

    preview = consensus_rerank_preview_df.copy()
    for column, default in {
        "current_rank": 0,
        "preview_rank": 0,
        "current_decision_score": 0.0,
        "preview_score": 0.0,
        "preview_decision": "would-keep-current",
        "suggestion_status": "no-consensus-change",
        "pocket_consensus_label": "no-consensus-anchor-pocket",
        "rank_safe_anchor_count": 0,
        "blocked_ai_count": 0,
        "weak_mapping_count": 0,
        "consensus_anchor_residues": "none",
        "recommended_action": "",
    }.items():
        if column not in preview.columns:
            preview[column] = default

    policy_status = "unknown"
    policy_recommended_action = "Review the action queue before applying consensus-driven rank changes."
    if consensus_rerank_policy_gate_df is not None and not getattr(consensus_rerank_policy_gate_df, "empty", True):
        gate_row = consensus_rerank_policy_gate_df.iloc[0]
        policy_status = _safe_text(gate_row.get("policy_status"), "unknown")
        policy_recommended_action = _safe_text(gate_row.get("recommended_action"), policy_recommended_action)
    policy_allows_apply = policy_status in {"no-change-needed", "review-before-apply"}

    queue_by_pocket: dict[str, pd.Series] = {}
    if consensus_rerank_action_queue_df is not None and not getattr(consensus_rerank_action_queue_df, "empty", True) and "pocket_id" in consensus_rerank_action_queue_df.columns:
        queue = consensus_rerank_action_queue_df.copy()
        queue["pocket_id"] = queue["pocket_id"].astype(str)
        queue = queue.drop_duplicates(subset=["pocket_id"], keep="first").set_index("pocket_id", drop=False)
        queue_by_pocket = {str(index): row for index, row in queue.iterrows()}

    rows: list[dict[str, Any]] = []
    for _, row in preview.drop_duplicates(subset=["pocket_id"], keep="first").iterrows():
        pocket_id = _safe_text(row.get("pocket_id"))
        if not pocket_id:
            continue
        action_row = queue_by_pocket.get(pocket_id)
        action = action_row.to_dict() if action_row is not None else _default_action_for_preview(row, policy_status)
        current_score = round(_clip(_safe_float(row.get("current_decision_score"), 0.0)), 3)
        preview_score = round(_clip(_safe_float(row.get("preview_score"), current_score)), 3)
        issue_type = _safe_text(action.get("issue_type"), "no-consensus-action")
        issue_severity = _safe_text(action.get("issue_severity"), "none")
        can_apply_after_fix = _safe_bool(action.get("can_apply_after_fix"))
        apply_status, simulation_score, score_source = _simulation_status_and_score(
            issue_type=issue_type,
            issue_severity=issue_severity,
            can_apply_after_fix=can_apply_after_fix,
            policy_allows_apply=policy_allows_apply,
            current_score=current_score,
            preview_score=preview_score,
            preview_decision=_safe_text(row.get("preview_decision"), "would-keep-current"),
        )
        required_before_apply = _safe_text(action.get("required_fix"), policy_recommended_action)
        if not policy_allows_apply:
            required_before_apply = f"{policy_recommended_action} Required row fix: {required_before_apply}"
        rows.append(
            {
                "simulated_rank": 0,
                "pocket_id": pocket_id,
                "current_rank": _safe_int(row.get("current_rank"), 0) or None,
                "preview_rank": _safe_int(row.get("preview_rank"), 0) or None,
                "simulated_rank_delta": None,
                "current_decision_score": current_score,
                "preview_score": preview_score,
                "simulation_score": round(_clip(simulation_score), 3),
                "simulation_score_source": score_source,
                "apply_status": apply_status,
                "apply_decision": "",
                "policy_status": policy_status,
                "policy_allows_apply": bool(policy_allows_apply),
                "issue_type": issue_type,
                "issue_severity": issue_severity,
                "rank_safe_anchor_count": _safe_int(row.get("rank_safe_anchor_count"), 0),
                "blocked_ai_count": _safe_int(row.get("blocked_ai_count"), 0),
                "weak_mapping_count": _safe_int(row.get("weak_mapping_count"), 0),
                "required_before_apply": required_before_apply,
                "consensus_anchor_residues": _safe_text(row.get("consensus_anchor_residues"), "none"),
                "recommended_action": _safe_text(action.get("recommended_action"), _safe_text(row.get("recommended_action"), required_before_apply)),
            }
        )

    if not rows:
        return _empty_consensus_rerank_apply_simulation_df()

    result = pd.DataFrame(rows, columns=CONSENSUS_RERANK_APPLY_SIMULATION_COLUMNS)
    result["_score_sort"] = pd.to_numeric(result["simulation_score"], errors="coerce").fillna(0.0)
    result["_anchor_sort"] = pd.to_numeric(result["rank_safe_anchor_count"], errors="coerce").fillna(0)
    result["_rank_sort"] = pd.to_numeric(result["current_rank"], errors="coerce").fillna(9999)
    result = result.sort_values(
        ["_score_sort", "_anchor_sort", "_rank_sort", "pocket_id"],
        ascending=[False, False, True, True],
    ).drop(columns=["_score_sort", "_anchor_sort", "_rank_sort"]).reset_index(drop=True)
    result["simulated_rank"] = range(1, len(result) + 1)

    deltas: list[Optional[int]] = []
    decisions: list[str] = []
    for _, row in result.iterrows():
        current_rank = _safe_int(row.get("current_rank"), 0)
        simulated_rank = _safe_int(row.get("simulated_rank"), 0)
        if current_rank <= 0 or simulated_rank <= 0:
            deltas.append(None)
            decisions.append("would-enter-simulated-ranking")
            continue
        delta = current_rank - simulated_rank
        deltas.append(delta)
        if delta > 0:
            decisions.append("would-rank-up")
        elif delta < 0:
            decisions.append("would-rank-down")
        else:
            decisions.append("would-keep-rank")
    result["simulated_rank_delta"] = deltas
    result["apply_decision"] = decisions
    return result[CONSENSUS_RERANK_APPLY_SIMULATION_COLUMNS]


def _simulation_delta_explanation(
    *,
    apply_status: str,
    issue_type: str,
    issue_severity: str,
    policy_status: str,
    rank_delta: int,
    score_delta: float,
    required_before_trust: str,
    consensus_anchor_residues: str,
) -> tuple[int, str, str, str, str]:
    status = _safe_text(apply_status).lower()
    issue = _safe_text(issue_type).lower()
    severity = _safe_text(issue_severity).lower()
    policy = _safe_text(policy_status).lower()
    anchors = _safe_text(consensus_anchor_residues, "none")
    required = _safe_text(required_before_trust, "Review this pocket before trusting a rerank change.")

    if status in {"blocked-currently", "mapping-review-required", "evidence-required", "ai-source-review-required"} or severity == "blocking":
        if issue == "blocked-ai-evidence":
            return (
                1,
                "frozen-blocker",
                "blocking",
                "Precision is protected by preventing unsupported or conflicting AI evidence from improving this pocket.",
                f"Frozen because blocked AI evidence is present. Required before trust: {required}",
            )
        if issue == "weak-residue-mapping":
            return (
                1,
                "frozen-mapping-review",
                "blocking",
                "Precision is limited by residue numbering or chain-mapping uncertainty.",
                f"Frozen until UniProt/PDB residue mapping is resolved. Anchors to inspect: {anchors}. Required before trust: {required}",
            )
        if issue == "functional-evidence-gap":
            return (
                2,
                "evidence-gap-demotion",
                "missing-evidence",
                "The pocket should not outrank consensus-supported pockets until catalytic or binding evidence is added.",
                f"Down-weighted because rank-safe functional anchors are missing. Required before trust: {required}",
            )
        return (
            2,
            "frozen-review-required",
            "review",
            "The simulation keeps this pocket conservative until the review blocker is resolved.",
            f"Frozen by {issue or 'review'} status. Required before trust: {required}",
        )

    if rank_delta > 0:
        change_type = "rank-up-ready" if policy in {"no-change-needed", "review-before-apply"} else "rank-up-candidate"
        return (
            3,
            change_type,
            "review",
            "Potential precision gain: the pocket moves up because consensus anchors support it.",
            f"Moves up {rank_delta} rank position(s) in simulation. Verify anchors before applying: {anchors}.",
        )
    if rank_delta < 0:
        return (
            4,
            "rank-down-conservative",
            "review",
            "Potential false-positive reduction: the pocket loses priority under evidence-aware scoring.",
            f"Moves down {abs(rank_delta)} rank position(s) in simulation because safer consensus-backed pockets outrank it.",
        )
    if score_delta >= 0.025:
        return (
            5,
            "score-up-no-rank-change",
            "monitor",
            "Consensus improves confidence but not enough to change rank order.",
            f"Score increases by {score_delta:+.3f} without changing rank. Keep anchors visible for validation: {anchors}.",
        )
    if score_delta <= -0.025:
        return (
            5,
            "score-down-no-rank-change",
            "monitor",
            "Evidence-aware scoring reduces confidence but not enough to change rank order.",
            f"Score decreases by {score_delta:+.3f} without changing rank. Review evidence before relying on this pocket.",
        )
    if status in {"keep-current-ready", "keep-after-review"}:
        return (
            6,
            "unchanged-ready",
            "pass",
            "The current ranking already agrees with the consensus evidence.",
            f"No material rank change. Use anchors as validation context: {anchors}.",
        )
    return (
        7,
        "unchanged-monitor",
        "monitor",
        "No material precision signal changes the current ranking.",
        "Keep this pocket as diagnostic context unless stronger functional evidence is added.",
    )


def build_consensus_rerank_simulation_delta(
    consensus_rerank_apply_simulation_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if consensus_rerank_apply_simulation_df is None or getattr(consensus_rerank_apply_simulation_df, "empty", True) or "pocket_id" not in consensus_rerank_apply_simulation_df.columns:
        return _empty_consensus_rerank_simulation_delta_df()

    simulation = consensus_rerank_apply_simulation_df.copy()
    for column, default in {
        "current_rank": 0,
        "simulated_rank": 0,
        "simulated_rank_delta": 0,
        "current_decision_score": 0.0,
        "simulation_score": 0.0,
        "apply_status": "monitor-only",
        "apply_decision": "would-keep-rank",
        "issue_type": "no-consensus-action",
        "issue_severity": "none",
        "policy_status": "unknown",
        "required_before_apply": "",
        "consensus_anchor_residues": "none",
        "recommended_action": "",
    }.items():
        if column not in simulation.columns:
            simulation[column] = default

    rows: list[dict[str, Any]] = []
    for _, row in simulation.drop_duplicates(subset=["pocket_id"], keep="first").iterrows():
        pocket_id = _safe_text(row.get("pocket_id"))
        if not pocket_id:
            continue
        current_score = round(_clip(_safe_float(row.get("current_decision_score"), 0.0)), 3)
        simulation_score = round(_clip(_safe_float(row.get("simulation_score"), current_score)), 3)
        score_delta = round(float(simulation_score - current_score), 3)
        rank_delta = _safe_int(row.get("simulated_rank_delta"), 0)
        (
            impact_priority,
            change_type,
            change_severity,
            precision_interpretation,
            explanation,
        ) = _simulation_delta_explanation(
            apply_status=_safe_text(row.get("apply_status"), "monitor-only"),
            issue_type=_safe_text(row.get("issue_type"), "no-consensus-action"),
            issue_severity=_safe_text(row.get("issue_severity"), "none"),
            policy_status=_safe_text(row.get("policy_status"), "unknown"),
            rank_delta=rank_delta,
            score_delta=score_delta,
            required_before_trust=_safe_text(row.get("required_before_apply")),
            consensus_anchor_residues=_safe_text(row.get("consensus_anchor_residues"), "none"),
        )
        rows.append(
            {
                "impact_priority": impact_priority,
                "pocket_id": pocket_id,
                "change_type": change_type,
                "change_severity": change_severity,
                "current_rank": _safe_int(row.get("current_rank"), 0) or None,
                "simulated_rank": _safe_int(row.get("simulated_rank"), 0) or None,
                "rank_delta": rank_delta,
                "current_decision_score": current_score,
                "simulation_score": simulation_score,
                "score_delta": score_delta,
                "apply_status": _safe_text(row.get("apply_status"), "monitor-only"),
                "apply_decision": _safe_text(row.get("apply_decision"), "would-keep-rank"),
                "issue_type": _safe_text(row.get("issue_type"), "no-consensus-action"),
                "issue_severity": _safe_text(row.get("issue_severity"), "none"),
                "policy_status": _safe_text(row.get("policy_status"), "unknown"),
                "precision_interpretation": precision_interpretation,
                "explanation": explanation,
                "required_before_trust": _safe_text(row.get("required_before_apply"), "Review before trusting a rerank change."),
                "consensus_anchor_residues": _safe_text(row.get("consensus_anchor_residues"), "none"),
                "recommended_action": _safe_text(row.get("recommended_action"), "-"),
            }
        )

    if not rows:
        return _empty_consensus_rerank_simulation_delta_df()

    result = pd.DataFrame(rows, columns=CONSENSUS_RERANK_SIMULATION_DELTA_COLUMNS)
    result["_abs_rank_delta"] = pd.to_numeric(result["rank_delta"], errors="coerce").fillna(0).abs()
    result["_abs_score_delta"] = pd.to_numeric(result["score_delta"], errors="coerce").fillna(0.0).abs()
    result["_rank_sort"] = pd.to_numeric(result["current_rank"], errors="coerce").fillna(9999)
    result = result.sort_values(
        ["impact_priority", "_abs_rank_delta", "_abs_score_delta", "_rank_sort", "pocket_id"],
        ascending=[True, False, False, True, True],
    ).drop(columns=["_abs_rank_delta", "_abs_score_delta", "_rank_sort"]).reset_index(drop=True)
    return result[CONSENSUS_RERANK_SIMULATION_DELTA_COLUMNS]


def _top_pocket_from_delta(delta: pd.DataFrame, mask: pd.Series, *, prefer_positive_score: bool) -> str:
    candidates = delta.loc[mask].copy()
    if candidates.empty:
        return ""
    candidates["_abs_rank_delta"] = pd.to_numeric(candidates["rank_delta"], errors="coerce").fillna(0).abs()
    candidates["_score_delta_sort"] = pd.to_numeric(candidates["score_delta"], errors="coerce").fillna(0.0)
    candidates["_impact_sort"] = pd.to_numeric(candidates["impact_priority"], errors="coerce").fillna(9999)
    candidates["_rank_sort"] = pd.to_numeric(candidates["current_rank"], errors="coerce").fillna(9999)
    score_ascending = not bool(prefer_positive_score)
    candidates = candidates.sort_values(
        ["_impact_sort", "_abs_rank_delta", "_score_delta_sort", "_rank_sort", "pocket_id"],
        ascending=[True, False, score_ascending, True, True],
    )
    return _safe_text(candidates.iloc[0].get("pocket_id"))


def build_consensus_rerank_precision_scorecard(
    consensus_rerank_simulation_delta_df: Optional[pd.DataFrame],
    consensus_rerank_apply_simulation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if consensus_rerank_simulation_delta_df is None or getattr(consensus_rerank_simulation_delta_df, "empty", True) or "pocket_id" not in consensus_rerank_simulation_delta_df.columns:
        return _empty_consensus_rerank_precision_scorecard_df()

    delta = consensus_rerank_simulation_delta_df.copy()
    for column, default in {
        "impact_priority": 999,
        "pocket_id": "",
        "change_type": "unchanged-monitor",
        "change_severity": "monitor",
        "current_rank": 0,
        "rank_delta": 0,
        "score_delta": 0.0,
        "apply_status": "monitor-only",
        "issue_type": "no-consensus-action",
        "issue_severity": "none",
        "policy_status": "unknown",
    }.items():
        if column not in delta.columns:
            delta[column] = default

    change = delta["change_type"].astype(str).str.lower()
    issue = delta["issue_type"].astype(str).str.lower()
    severity = delta["issue_severity"].astype(str).str.lower()
    score_delta = pd.to_numeric(delta["score_delta"], errors="coerce").fillna(0.0)

    rank_up_mask = change.isin({"rank-up-ready", "rank-up-candidate"})
    rank_down_mask = change.eq("rank-down-conservative")
    unchanged_ready_mask = change.eq("unchanged-ready")
    score_only_mask = change.isin({"score-up-no-rank-change", "score-down-no-rank-change"})
    frozen_blocker_mask = change.eq("frozen-blocker") | issue.eq("blocked-ai-evidence") | severity.eq("blocking")
    mapping_mask = change.eq("frozen-mapping-review") | issue.eq("weak-residue-mapping")
    evidence_gap_mask = change.eq("evidence-gap-demotion") | issue.eq("functional-evidence-gap")
    ai_source_mask = issue.eq("ai-source-review") | delta["apply_status"].astype(str).str.lower().eq("ai-source-review-required")
    monitor_mask = change.str.contains("monitor", na=False)

    rank_up_rows = int(rank_up_mask.sum())
    rank_down_rows = int(rank_down_mask.sum())
    unchanged_ready_rows = int(unchanged_ready_mask.sum())
    score_only_rows = int(score_only_mask.sum())
    frozen_blocker_rows = int(frozen_blocker_mask.sum())
    mapping_review_rows = int(mapping_mask.sum())
    evidence_gap_rows = int(evidence_gap_mask.sum())
    ai_source_review_rows = int(ai_source_mask.sum())
    monitor_rows = int(monitor_mask.sum())
    positive_signal_rows = int((rank_up_mask | rank_down_mask | unchanged_ready_mask | (change.eq("score-up-no-rank-change") & score_delta.gt(0))).sum())
    negative_control_rows = int((rank_down_mask | frozen_blocker_mask | mapping_mask | evidence_gap_mask | ai_source_mask).sum())
    open_blocker_rows = int((frozen_blocker_mask | mapping_mask | evidence_gap_mask | ai_source_mask).sum())

    policy_status = "unknown"
    if consensus_rerank_policy_gate_df is not None and not getattr(consensus_rerank_policy_gate_df, "empty", True):
        policy_status = _safe_text(consensus_rerank_policy_gate_df.iloc[0].get("policy_status"), "unknown")
    else:
        policy_values = [text for text in delta["policy_status"].astype(str).str.strip().tolist() if text]
        policy_status = policy_values[0] if policy_values else "unknown"

    policy_allows_apply_rows = 0
    if consensus_rerank_apply_simulation_df is not None and not getattr(consensus_rerank_apply_simulation_df, "empty", True) and "policy_allows_apply" in consensus_rerank_apply_simulation_df.columns:
        policy_allows_apply_rows = int(consensus_rerank_apply_simulation_df["policy_allows_apply"].apply(_safe_bool).sum())

    positive_points = (rank_up_rows * 14) + (rank_down_rows * 10) + (unchanged_ready_rows * 6) + (int(change.eq("score-up-no-rank-change").sum()) * 3)
    risk_points = (frozen_blocker_rows * 18) + (mapping_review_rows * 12) + (evidence_gap_rows * 10) + (ai_source_review_rows * 8) + (monitor_rows * 2)
    policy_bonus = 5 if policy_status in {"no-change-needed", "review-before-apply"} and open_blocker_rows == 0 else 0
    policy_penalty = 5 if policy_status in {"blocked", "mapping-review", "needs-evidence"} else 0
    precision_improvement_score = int(round(_clip(50 + min(35, positive_points) - min(45, risk_points) + policy_bonus - policy_penalty, 0, 100)))

    if open_blocker_rows > 0 and positive_signal_rows > 0:
        scorecard_status = "promising-but-blocked"
        recommended_action = "Resolve open evidence or mapping blockers, then rerun the consensus rerank simulation before applying changes."
    elif open_blocker_rows > 0:
        scorecard_status = "blocked-before-precision-gain"
        recommended_action = "Fix blocker rows first; current simulation mainly protects against unsafe precision loss."
    elif positive_signal_rows > 0 and (rank_up_rows > 0 or rank_down_rows > 0):
        scorecard_status = "likely-precision-gain"
        recommended_action = "Review the simulation delta and consider applying the consensus rerank policy after manual approval."
    elif positive_signal_rows > 0:
        scorecard_status = "stable-evidence-aligned"
        recommended_action = "Current ranking is mostly aligned with consensus evidence; use anchors for validation."
    else:
        scorecard_status = "neutral-monitor"
        recommended_action = "Keep the consensus rerank as a diagnostic layer until stronger functional evidence is added."

    top_positive_pocket_id = _top_pocket_from_delta(
        delta,
        rank_up_mask | rank_down_mask | unchanged_ready_mask | change.eq("score-up-no-rank-change"),
        prefer_positive_score=True,
    )
    top_blocker_pocket_id = _top_pocket_from_delta(
        delta,
        frozen_blocker_mask | mapping_mask | evidence_gap_mask | ai_source_mask,
        prefer_positive_score=False,
    )
    score_reason = (
        f"positive={positive_signal_rows}; blockers={open_blocker_rows}; "
        f"rank_up={rank_up_rows}; rank_down={rank_down_rows}; policy={policy_status}"
    )

    return pd.DataFrame(
        [
            {
                "scorecard_status": scorecard_status,
                "precision_improvement_score": precision_improvement_score,
                "simulation_rows": int(len(delta)),
                "positive_signal_rows": positive_signal_rows,
                "negative_control_rows": negative_control_rows,
                "open_blocker_rows": open_blocker_rows,
                "rank_up_rows": rank_up_rows,
                "rank_down_rows": rank_down_rows,
                "unchanged_ready_rows": unchanged_ready_rows,
                "score_only_rows": score_only_rows,
                "frozen_blocker_rows": frozen_blocker_rows,
                "mapping_review_rows": mapping_review_rows,
                "evidence_gap_rows": evidence_gap_rows,
                "ai_source_review_rows": ai_source_review_rows,
                "monitor_rows": monitor_rows,
                "policy_status": policy_status,
                "policy_allows_apply_rows": policy_allows_apply_rows,
                "top_positive_pocket_id": top_positive_pocket_id or "none",
                "top_blocker_pocket_id": top_blocker_pocket_id or "none",
                "score_reason": score_reason,
                "recommended_action": recommended_action,
            }
        ],
        columns=CONSENSUS_RERANK_PRECISION_SCORECARD_COLUMNS,
    )


def _first_guardrail_clearance(consensus_rerank_action_queue_df: Optional[pd.DataFrame]) -> tuple[int, str]:
    if consensus_rerank_action_queue_df is None or getattr(consensus_rerank_action_queue_df, "empty", True):
        return 0, "none"
    queue = consensus_rerank_action_queue_df.copy()
    for column, default in {
        "action_priority": 999,
        "issue_severity": "",
        "can_apply_after_fix": False,
        "required_fix": "",
        "pocket_id": "",
        "issue_type": "",
    }.items():
        if column not in queue.columns:
            queue[column] = default
    priority = pd.to_numeric(queue["action_priority"], errors="coerce").fillna(999).astype(int)
    severity = queue["issue_severity"].astype(str).str.lower()
    can_apply = queue["can_apply_after_fix"].apply(_safe_bool)
    clearance_mask = priority.le(5) & (severity.isin({"blocking", "missing-evidence", "review"}) | ~can_apply)
    clearance = queue.loc[clearance_mask].copy()
    if clearance.empty:
        return 0, "none"
    clearance["_priority_sort"] = pd.to_numeric(clearance["action_priority"], errors="coerce").fillna(999)
    clearance = clearance.sort_values(["_priority_sort", "pocket_id", "issue_type"], ascending=[True, True, True])
    first = clearance.iloc[0]
    pocket = _safe_text(first.get("pocket_id"), "-")
    issue = _safe_text(first.get("issue_type"), "review")
    fix = _safe_text(first.get("required_fix"), "Review this pocket before applying rerank.")
    return int(len(clearance)), f"{pocket} / {issue}: {fix}"


def build_consensus_rerank_precision_guardrail(
    consensus_rerank_precision_scorecard_df: Optional[pd.DataFrame],
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
    consensus_rerank_action_queue_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if consensus_rerank_precision_scorecard_df is None or getattr(consensus_rerank_precision_scorecard_df, "empty", True):
        return _empty_consensus_rerank_precision_guardrail_df()

    scorecard = consensus_rerank_precision_scorecard_df.iloc[0]
    precision_score = _safe_int(scorecard.get("precision_improvement_score"), 0)
    scorecard_status = _safe_text(scorecard.get("scorecard_status"), "neutral-monitor")
    positive_signal_rows = _safe_int(scorecard.get("positive_signal_rows"), 0)
    open_blocker_rows = _safe_int(scorecard.get("open_blocker_rows"), 0)
    top_positive_pocket_id = _safe_text(scorecard.get("top_positive_pocket_id"), "none")
    top_blocker_pocket_id = _safe_text(scorecard.get("top_blocker_pocket_id"), "none")

    policy_status = _safe_text(scorecard.get("policy_status"), "unknown")
    policy_action = ""
    if consensus_rerank_policy_gate_df is not None and not getattr(consensus_rerank_policy_gate_df, "empty", True):
        policy_row = consensus_rerank_policy_gate_df.iloc[0]
        policy_status = _safe_text(policy_row.get("policy_status"), policy_status)
        policy_action = _safe_text(policy_row.get("recommended_action"))

    required_clearance_count, first_required_clearance = _first_guardrail_clearance(consensus_rerank_action_queue_df)
    required_clearance_count = max(required_clearance_count, open_blocker_rows)

    policy_blocks = policy_status in {"blocked", "mapping-review", "needs-evidence"}
    if policy_blocks or open_blocker_rows > 0:
        guardrail_status = "blocked"
        guardrail_decision = "do-not-apply"
        apply_mode = "diagnostic-only"
        can_auto = False
        can_manual = False
        manual_review_required = True
        decision_reason = (
            f"Policy={policy_status}; blockers={open_blocker_rows}; "
            "rerank must remain diagnostic until evidence and mapping blockers are cleared."
        )
        recommended_action = policy_action or _safe_text(scorecard.get("recommended_action"), "Resolve blockers before rerank.")
    elif scorecard_status == "likely-precision-gain" and precision_score >= 70 and positive_signal_rows > 0:
        guardrail_status = "manual-review-ready"
        guardrail_decision = "allow-after-review"
        apply_mode = "manual-consensus-rerank"
        can_auto = False
        can_manual = True
        manual_review_required = True
        decision_reason = (
            f"Score={precision_score}; positive signals={positive_signal_rows}; no open blockers; "
            "manual review is still required before changing active ranking."
        )
        recommended_action = "Review simulation delta and validation anchors, then apply consensus rerank only with manual approval."
    elif scorecard_status == "stable-evidence-aligned":
        guardrail_status = "no-rerank-needed"
        guardrail_decision = "keep-current-ranking"
        apply_mode = "validation-anchors-only"
        can_auto = False
        can_manual = False
        manual_review_required = False
        decision_reason = "Current ranking is already aligned with consensus evidence; use anchors for validation."
        recommended_action = _safe_text(scorecard.get("recommended_action"), "Keep current ranking and validate consensus anchors.")
    elif scorecard_status == "likely-precision-gain" and positive_signal_rows > 0:
        guardrail_status = "review-more-before-apply"
        guardrail_decision = "hold-for-review"
        apply_mode = "review-only"
        can_auto = False
        can_manual = False
        manual_review_required = True
        decision_reason = f"Precision score {precision_score} is below the safe manual-apply threshold despite positive signals."
        recommended_action = "Inspect simulation delta and add stronger functional evidence before applying rerank."
    else:
        guardrail_status = "diagnostic-only"
        guardrail_decision = "do-not-apply"
        apply_mode = "diagnostic-only"
        can_auto = False
        can_manual = False
        manual_review_required = False
        decision_reason = "Consensus rerank does not yet show enough evidence-backed precision gain."
        recommended_action = _safe_text(scorecard.get("recommended_action"), "Keep rerank as diagnostics until stronger evidence is added.")

    return pd.DataFrame(
        [
            {
                "guardrail_status": guardrail_status,
                "guardrail_decision": guardrail_decision,
                "apply_mode": apply_mode,
                "can_enable_auto_rerank": bool(can_auto),
                "can_apply_after_manual_review": bool(can_manual),
                "manual_review_required": bool(manual_review_required),
                "precision_improvement_score": precision_score,
                "scorecard_status": scorecard_status,
                "policy_status": policy_status,
                "positive_signal_rows": positive_signal_rows,
                "open_blocker_rows": open_blocker_rows,
                "required_clearance_count": required_clearance_count,
                "first_required_clearance": first_required_clearance,
                "top_positive_pocket_id": top_positive_pocket_id,
                "top_blocker_pocket_id": top_blocker_pocket_id,
                "decision_reason": decision_reason,
                "recommended_action": recommended_action,
            }
        ],
        columns=CONSENSUS_RERANK_PRECISION_GUARDRAIL_COLUMNS,
    )


def build_consensus_rerank_precision_guardrail_report_markdown(
    consensus_rerank_precision_guardrail_df: Optional[pd.DataFrame],
    consensus_rerank_precision_scorecard_df: Optional[pd.DataFrame] = None,
    consensus_rerank_action_queue_df: Optional[pd.DataFrame] = None,
    consensus_rerank_simulation_delta_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Consensus rerank precision guardrail report",
) -> str:
    lines = [
        f"# {title}",
        "",
        "This report is a non-destructive handoff summary for deciding whether consensus rerank can leave diagnostics.",
        "Automatic rerank should remain disabled unless the guardrail explicitly allows a reviewed release path.",
        "",
    ]
    if consensus_rerank_precision_guardrail_df is None or getattr(consensus_rerank_precision_guardrail_df, "empty", True):
        return "\n".join(lines + ["No precision guardrail decision is currently available."])

    guardrail = consensus_rerank_precision_guardrail_df.iloc[0]
    scorecard = None
    if consensus_rerank_precision_scorecard_df is not None and not getattr(consensus_rerank_precision_scorecard_df, "empty", True):
        scorecard = consensus_rerank_precision_scorecard_df.iloc[0]

    can_auto = _safe_bool(guardrail.get("can_enable_auto_rerank"))
    can_manual = _safe_bool(guardrail.get("can_apply_after_manual_review"))
    manual_required = _safe_bool(guardrail.get("manual_review_required"))
    lines.extend(
        [
            "## Decision summary",
            "",
            f"- Guardrail status: `{_safe_text(guardrail.get('guardrail_status'), '-')}`",
            f"- Decision: `{_safe_text(guardrail.get('guardrail_decision'), '-')}`",
            f"- Apply mode: `{_safe_text(guardrail.get('apply_mode'), '-')}`",
            f"- Can enable automatic rerank: {'yes' if can_auto else 'no'}",
            f"- Can apply after manual review: {'yes' if can_manual else 'no'}",
            f"- Manual review required: {'yes' if manual_required else 'no'}",
            f"- Precision improvement score: `{_safe_int(guardrail.get('precision_improvement_score'), 0)}`",
            f"- Scorecard status: `{_safe_text(guardrail.get('scorecard_status'), '-')}`",
            f"- Policy status: `{_safe_text(guardrail.get('policy_status'), '-')}`",
            f"- Positive signals: `{_safe_int(guardrail.get('positive_signal_rows'), 0)}`",
            f"- Open blockers: `{_safe_int(guardrail.get('open_blocker_rows'), 0)}`",
            f"- Required clearances: `{_safe_int(guardrail.get('required_clearance_count'), 0)}`",
            f"- First required clearance: {_safe_text(guardrail.get('first_required_clearance'), 'none')}",
            f"- Top positive pocket: `{_safe_text(guardrail.get('top_positive_pocket_id'), 'none')}`",
            f"- Top blocker pocket: `{_safe_text(guardrail.get('top_blocker_pocket_id'), 'none')}`",
            "",
            "## Guardrail rationale",
            "",
            _safe_text(guardrail.get("decision_reason"), "-"),
            "",
            "## Recommended action",
            "",
            _safe_text(guardrail.get("recommended_action"), "-"),
            "",
        ]
    )

    if scorecard is not None:
        lines.extend(
            [
                "## Scorecard counters",
                "",
                f"- Rank-up rows: `{_safe_int(scorecard.get('rank_up_rows'), 0)}`",
                f"- Rank-down rows: `{_safe_int(scorecard.get('rank_down_rows'), 0)}`",
                f"- Negative controls: `{_safe_int(scorecard.get('negative_control_rows'), 0)}`",
                f"- Frozen blockers: `{_safe_int(scorecard.get('frozen_blocker_rows'), 0)}`",
                f"- Mapping review rows: `{_safe_int(scorecard.get('mapping_review_rows'), 0)}`",
                f"- Evidence-gap rows: `{_safe_int(scorecard.get('evidence_gap_rows'), 0)}`",
                f"- Score reason: {_safe_text(scorecard.get('score_reason'), '-')}",
                "",
            ]
        )

    if consensus_rerank_action_queue_df is not None and not getattr(consensus_rerank_action_queue_df, "empty", True):
        queue = consensus_rerank_action_queue_df.copy()
        for column, default in {
            "action_priority": 999,
            "pocket_id": "",
            "issue_type": "review",
            "issue_severity": "review",
            "required_fix": "",
            "can_apply_after_fix": False,
            "consensus_anchor_residues": "none",
        }.items():
            if column not in queue.columns:
                queue[column] = default
        queue["_priority_sort"] = pd.to_numeric(queue["action_priority"], errors="coerce").fillna(999)
        queue = queue.sort_values(["_priority_sort", "pocket_id", "issue_type"], ascending=[True, True, True]).head(5)
        lines.extend(["## Required clearances and review items", ""])
        for index, row in enumerate(queue.itertuples(index=False), start=1):
            lines.extend(
                [
                    f"### {index}. {_safe_text(getattr(row, 'pocket_id', ''), 'Pocket')} - {_safe_text(getattr(row, 'issue_type', ''), 'review')}",
                    "",
                    f"- [ ] Required fix: {_safe_text(getattr(row, 'required_fix', ''), '-')}",
                    f"- Severity: `{_safe_text(getattr(row, 'issue_severity', ''), '-')}`",
                    f"- Can apply after fix: {'yes' if _safe_bool(getattr(row, 'can_apply_after_fix', False)) else 'no'}",
                    f"- Consensus anchors: {_safe_text(getattr(row, 'consensus_anchor_residues', ''), 'none')}",
                    "",
                ]
            )

    if consensus_rerank_simulation_delta_df is not None and not getattr(consensus_rerank_simulation_delta_df, "empty", True):
        delta = consensus_rerank_simulation_delta_df.copy()
        if "impact_priority" not in delta.columns:
            delta["impact_priority"] = 999
        if "pocket_id" not in delta.columns:
            delta["pocket_id"] = ""
        delta["_impact_sort"] = pd.to_numeric(delta["impact_priority"], errors="coerce").fillna(999)
        delta = delta.sort_values(["_impact_sort", "pocket_id"], ascending=[True, True]).head(5)
        lines.extend(["## Key simulation deltas", ""])
        for row in delta.itertuples(index=False):
            lines.append(
                "- "
                f"`{_safe_text(getattr(row, 'pocket_id', ''), 'Pocket')}`: "
                f"{_safe_text(getattr(row, 'change_type', ''), 'change')} / "
                f"rank delta {_safe_int(getattr(row, 'rank_delta', 0), 0):+d}. "
                f"{_safe_text(getattr(row, 'precision_interpretation', ''), '-')}"
            )
        lines.append("")

    lines.extend(
        [
            "## Release checklist",
            "",
            "- [ ] Confirm the guardrail decision with a human reviewer.",
            "- [ ] Verify every consensus anchor residue against chain, numbering, insertion code, and source evidence.",
            "- [ ] Resolve required clearances before changing active ranking.",
            "- [ ] Re-run the policy gate, scorecard, and guardrail after any evidence or mapping edit.",
            "- [ ] Archive this report with the exported CSV tables used to generate it.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _rerank_table_len(table: Optional[pd.DataFrame]) -> int:
    return 0 if table is None or getattr(table, "empty", True) else int(len(table))


def _rerank_csv_artifact_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode("utf-8")


def _rerank_artifact_integrity(data: bytes) -> tuple[int, str]:
    return len(data), hashlib.sha256(data).hexdigest()


def build_consensus_rerank_guardrail_artifact_manifest(
    *,
    consensus_rerank_suggestion_df: Optional[pd.DataFrame] = None,
    consensus_rerank_preview_df: Optional[pd.DataFrame] = None,
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
    consensus_rerank_action_queue_df: Optional[pd.DataFrame] = None,
    consensus_rerank_action_checklist_markdown: str = "",
    consensus_rerank_apply_simulation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_simulation_delta_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_scorecard_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_guardrail_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_guardrail_report_markdown: str = "",
    consensus_rerank_release_decision_template_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_validation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_apply_report_markdown: str = "",
    consensus_rerank_release_execution_template_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_receipt_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_validation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_report_markdown: str = "",
    consensus_rerank_release_closure_certificate_markdown: str = "",
    consensus_rerank_release_closure_ledger_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_artifact(
        artifact_name: str,
        file_name: str,
        artifact_type: str,
        row_count: int,
        data: bytes,
        status: str,
        purpose: str,
        recommended_use: str,
    ) -> None:
        if row_count <= 0 or not data:
            return
        byte_size, digest = _rerank_artifact_integrity(data)
        rows.append(
            {
                "artifact_name": artifact_name,
                "file_name": file_name,
                "artifact_type": artifact_type,
                "row_count": int(row_count),
                "byte_size": int(byte_size),
                "sha256": digest,
                "status": status,
                "purpose": purpose,
                "recommended_use": recommended_use,
            }
        )

    def add_csv(
        artifact_name: str,
        file_name: str,
        table: Optional[pd.DataFrame],
        status: str,
        purpose: str,
        recommended_use: str,
    ) -> None:
        if table is None or getattr(table, "empty", True):
            return
        add_artifact(
            artifact_name,
            file_name,
            "csv",
            _rerank_table_len(table),
            _rerank_csv_artifact_bytes(table),
            status,
            purpose,
            recommended_use,
        )

    guardrail_status = (
        _safe_text(consensus_rerank_precision_guardrail_df.iloc[0].get("guardrail_status"), "not-generated")
        if consensus_rerank_precision_guardrail_df is not None and not getattr(consensus_rerank_precision_guardrail_df, "empty", True)
        else "not-generated"
    )
    scorecard_status = (
        _safe_text(consensus_rerank_precision_scorecard_df.iloc[0].get("scorecard_status"), "not-generated")
        if consensus_rerank_precision_scorecard_df is not None and not getattr(consensus_rerank_precision_scorecard_df, "empty", True)
        else "not-generated"
    )
    policy_status = (
        _safe_text(consensus_rerank_policy_gate_df.iloc[0].get("policy_status"), "not-generated")
        if consensus_rerank_policy_gate_df is not None and not getattr(consensus_rerank_policy_gate_df, "empty", True)
        else "not-generated"
    )
    release_review_status = (
        _safe_text(consensus_rerank_release_decision_summary_df.iloc[0].get("release_review_status"), "decision-uploaded")
        if consensus_rerank_release_decision_summary_df is not None and not getattr(consensus_rerank_release_decision_summary_df, "empty", True)
        else "not-uploaded"
    )
    release_validation_status = "not-uploaded"
    if consensus_rerank_release_decision_validation_df is not None and not getattr(consensus_rerank_release_decision_validation_df, "empty", True):
        validation_statuses = consensus_rerank_release_decision_validation_df["validation_status"].astype(str).str.lower() if "validation_status" in consensus_rerank_release_decision_validation_df.columns else pd.Series(dtype=str)
        release_validation_status = "blocked" if (validation_statuses == "blocked").any() else "validated"

    report_text = "" if consensus_rerank_precision_guardrail_report_markdown is None else str(consensus_rerank_precision_guardrail_report_markdown)
    report_lines = len([line for line in report_text.splitlines() if line.strip()])
    add_artifact(
        "Consensus rerank precision guardrail report",
        "consensus_rerank_precision_guardrail_report.md",
        "markdown",
        report_lines,
        report_text.encode("utf-8") if _safe_text(report_text) else b"",
        guardrail_status,
        "Human-readable release-review handoff for the consensus rerank guardrail.",
        "Open this first to understand the go/no-go decision, required clearances, and release checklist.",
    )

    add_csv(
        "Consensus rerank precision guardrail",
        "consensus_rerank_precision_guardrail.csv",
        consensus_rerank_precision_guardrail_df,
        guardrail_status,
        "One-row go/no-go decision for applying consensus rerank.",
        "Use this as the primary safety decision before any ranking change.",
    )
    add_csv(
        "Consensus rerank release decision template",
        "consensus_rerank_release_decision_template.csv",
        consensus_rerank_release_decision_template_df,
        "ready-for-review",
        "Editable reviewer sign-off template for guardrail release decisions.",
        "Fill reviewer, review_decision, notes, verified anchors, and sources before any manual rerank release.",
    )
    add_csv(
        "Consensus rerank release decisions normalized",
        "consensus_rerank_release_decisions_normalized.csv",
        consensus_rerank_release_decision_df,
        release_review_status,
        "Normalized reviewer-uploaded release decision rows.",
        "Archive this with the template to prove which reviewer decisions were evaluated.",
    )
    add_csv(
        "Consensus rerank release decision validation",
        "consensus_rerank_release_decision_validation.csv",
        consensus_rerank_release_decision_validation_df,
        release_validation_status,
        "Per-row validation of reviewer, source, anchor, blocker, template, and guardrail requirements.",
        "Use this to identify why an uploaded release decision can or cannot authorize rerank.",
    )
    add_csv(
        "Consensus rerank release decision summary",
        "consensus_rerank_release_decision_summary.csv",
        consensus_rerank_release_decision_summary_df,
        release_review_status,
        "One-row final review status for manual rerank release.",
        "Use this as the reviewer sign-off outcome; release is allowed only when release_allowed is true.",
    )
    add_csv(
        "Consensus rerank release apply plan",
        "consensus_rerank_release_apply_plan.csv",
        consensus_rerank_release_apply_plan_df,
        "ready-for-manual-apply",
        "Approved manual rank order derived from the reviewed release decision and clean apply simulation.",
        "Use this as the final manual application worksheet after archiving the certificate and ZIP hash.",
    )
    apply_report_text = "" if consensus_rerank_release_apply_report_markdown is None else str(consensus_rerank_release_apply_report_markdown)
    apply_report_lines = len([line for line in apply_report_text.splitlines() if line.strip()])
    add_artifact(
        "Consensus rerank release apply report",
        "consensus_rerank_release_apply_report.md",
        "markdown",
        apply_report_lines,
        apply_report_text.encode("utf-8") if _safe_text(apply_report_text) else b"",
        "ready-for-manual-apply",
        "Human-readable execution worksheet for an approved manual rerank release.",
        "Open this with the apply plan CSV before manually changing ranking.",
    )
    add_csv(
        "Consensus rerank release execution template",
        "consensus_rerank_release_execution_template.csv",
        consensus_rerank_release_execution_template_df,
        "execution-pending",
        "Operator-facing execution receipt template for recording applied manual ranks.",
        "Fill operator, executed_at, execution_decision, applied_rank, and notes after applying the approved plan.",
    )
    execution_review_status = (
        _safe_text(consensus_rerank_release_execution_summary_df.iloc[0].get("execution_review_status"), "execution-uploaded")
        if consensus_rerank_release_execution_summary_df is not None and not getattr(consensus_rerank_release_execution_summary_df, "empty", True)
        else "not-uploaded"
    )
    execution_validation_status = "not-uploaded"
    if consensus_rerank_release_execution_validation_df is not None and not getattr(consensus_rerank_release_execution_validation_df, "empty", True):
        execution_statuses = consensus_rerank_release_execution_validation_df["validation_status"].astype(str).str.lower() if "validation_status" in consensus_rerank_release_execution_validation_df.columns else pd.Series(dtype=str)
        execution_validation_status = "blocked" if (execution_statuses == "blocked").any() else "validated"
    add_csv(
        "Consensus rerank release execution receipt",
        "consensus_rerank_release_execution_receipt_normalized.csv",
        consensus_rerank_release_execution_receipt_df,
        execution_review_status,
        "Normalized operator-uploaded execution receipt rows.",
        "Archive this with the execution template to prove which applied ranks were reported.",
    )
    add_csv(
        "Consensus rerank release execution validation",
        "consensus_rerank_release_execution_validation.csv",
        consensus_rerank_release_execution_validation_df,
        execution_validation_status,
        "Per-row validation of operator, timestamp, applied rank, template matching, and apply-plan hash.",
        "Use this to identify whether the execution receipt matches the approved manual apply plan.",
    )
    add_csv(
        "Consensus rerank release execution summary",
        "consensus_rerank_release_execution_summary.csv",
        consensus_rerank_release_execution_summary_df,
        execution_review_status,
        "One-row final execution status for the approved manual rerank.",
        "Use this as the operational receipt outcome; execution is complete only when execution_complete is true.",
    )
    execution_report_text = "" if consensus_rerank_release_execution_report_markdown is None else str(consensus_rerank_release_execution_report_markdown)
    execution_report_lines = len([line for line in execution_report_text.splitlines() if line.strip()])
    add_artifact(
        "Consensus rerank release execution report",
        "consensus_rerank_release_execution_report.md",
        "markdown",
        execution_report_lines,
        execution_report_text.encode("utf-8") if _safe_text(execution_report_text) else b"",
        execution_review_status,
        "Human-readable operational receipt report for the manual rerank execution.",
        "Archive this after uploading the execution receipt to prove whether the approved rerank was actually applied.",
    )
    closure_certificate_text = "" if consensus_rerank_release_closure_certificate_markdown is None else str(consensus_rerank_release_closure_certificate_markdown)
    closure_certificate_lines = len([line for line in closure_certificate_text.splitlines() if line.strip()])
    add_artifact(
        "Consensus rerank release closure certificate",
        "consensus_rerank_release_closure_certificate.md",
        "markdown",
        closure_certificate_lines,
        closure_certificate_text.encode("utf-8") if _safe_text(closure_certificate_text) else b"",
        execution_review_status,
        "Detached final closure certificate tying the approved apply plan, release review, execution receipt, and execution report together.",
        "Archive this as the final human-readable proof of whether the approved consensus rerank release is closed.",
    )
    closure_ledger_status = "not-generated"
    if consensus_rerank_release_closure_ledger_df is not None and not getattr(consensus_rerank_release_closure_ledger_df, "empty", True):
        if "closure_check" in consensus_rerank_release_closure_ledger_df.columns:
            checks = consensus_rerank_release_closure_ledger_df["closure_check"].astype(str).str.lower()
            closure_ledger_status = "closed-executed" if (checks == "ok").all() else "not-closed"
        else:
            closure_ledger_status = "generated"
    add_csv(
        "Consensus rerank release closure ledger",
        "consensus_rerank_release_closure_ledger.csv",
        consensus_rerank_release_closure_ledger_df,
        closure_ledger_status,
        "Machine-readable closure evidence ledger for apply plan, release review, receipt, validation, report, and certificate artifacts.",
        "Use this table for automated audit checks before treating a consensus rerank release as closed.",
    )
    add_csv(
        "Consensus rerank precision scorecard",
        "consensus_rerank_precision_scorecard.csv",
        consensus_rerank_precision_scorecard_df,
        scorecard_status,
        "One-row precision improvement score and blocker summary.",
        "Use it to understand whether the simulation likely improves precision.",
    )
    add_csv(
        "Consensus rerank simulation delta",
        "consensus_rerank_simulation_delta.csv",
        consensus_rerank_simulation_delta_df,
        "explanation-layer",
        "Pocket-level explanation of simulated rank and score changes.",
        "Use it to trace scorecard and guardrail decisions back to individual pockets.",
    )
    add_csv(
        "Consensus rerank apply simulation",
        "consensus_rerank_apply_simulation.csv",
        consensus_rerank_apply_simulation_df,
        "simulation",
        "Non-destructive simulated rank order after conservative consensus rerank rules.",
        "Use it to inspect how ranking would change without modifying active ranking.",
    )
    add_csv(
        "Consensus rerank action queue",
        "consensus_rerank_action_queue.csv",
        consensus_rerank_action_queue_df,
        "clearance-worklist",
        "Prioritized blocker and review queue for making rerank safer.",
        "Use it to resolve evidence, mapping, and AI-source issues before rerank.",
    )
    checklist_text = "" if consensus_rerank_action_checklist_markdown is None else str(consensus_rerank_action_checklist_markdown)
    checklist_lines = len([line for line in checklist_text.splitlines() if line.strip()])
    add_artifact(
        "Consensus rerank action checklist",
        "consensus_rerank_action_checklist.md",
        "markdown",
        checklist_lines,
        checklist_text.encode("utf-8") if _safe_text(checklist_text) else b"",
        "clearance-worklist",
        "Markdown checklist for resolving action queue items.",
        "Use it as the manual review worksheet before rerunning the guardrail.",
    )
    add_csv(
        "Consensus rerank policy gate",
        "consensus_rerank_policy_gate.csv",
        consensus_rerank_policy_gate_df,
        policy_status,
        "Global safety gate for whether the preview can progress toward application.",
        "Use it to confirm whether blockers are global or only pocket-level.",
    )
    add_csv(
        "Consensus rerank preview",
        "consensus_rerank_preview.csv",
        consensus_rerank_preview_df,
        "preview",
        "Conservative score-adjustment preview before apply simulation.",
        "Use it to inspect the raw evidence-aware rerank preview.",
    )
    add_csv(
        "Consensus rerank suggestions",
        "consensus_rerank_suggestions.csv",
        consensus_rerank_suggestion_df,
        "suggestions",
        "Evidence-consensus suggestions before preview scoring.",
        "Use it to trace preview inputs back to consensus coverage.",
    )

    if not rows:
        return _empty_consensus_rerank_guardrail_artifact_manifest_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_GUARDRAIL_ARTIFACT_MANIFEST_COLUMNS).reset_index(drop=True)


def build_consensus_rerank_guardrail_handoff_zip(
    *,
    consensus_rerank_suggestion_df: Optional[pd.DataFrame] = None,
    consensus_rerank_preview_df: Optional[pd.DataFrame] = None,
    consensus_rerank_policy_gate_df: Optional[pd.DataFrame] = None,
    consensus_rerank_action_queue_df: Optional[pd.DataFrame] = None,
    consensus_rerank_action_checklist_markdown: str = "",
    consensus_rerank_apply_simulation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_simulation_delta_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_scorecard_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_guardrail_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_guardrail_report_markdown: str = "",
    consensus_rerank_release_decision_template_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_validation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_apply_report_markdown: str = "",
    consensus_rerank_release_execution_template_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_receipt_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_validation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_report_markdown: str = "",
    consensus_rerank_release_closure_certificate_markdown: str = "",
    consensus_rerank_release_closure_ledger_df: Optional[pd.DataFrame] = None,
    artifact_manifest_df: Optional[pd.DataFrame] = None,
) -> bytes:
    artifacts: list[tuple[str, bytes]] = []

    def add_csv(file_name: str, table: Optional[pd.DataFrame]) -> None:
        if table is not None and not getattr(table, "empty", True):
            artifacts.append((file_name, _rerank_csv_artifact_bytes(table)))

    if _safe_text(consensus_rerank_precision_guardrail_report_markdown):
        artifacts.append(
            (
                "consensus_rerank_precision_guardrail_report.md",
                str(consensus_rerank_precision_guardrail_report_markdown).encode("utf-8"),
            )
        )
    add_csv("consensus_rerank_precision_guardrail.csv", consensus_rerank_precision_guardrail_df)
    add_csv("consensus_rerank_release_decision_template.csv", consensus_rerank_release_decision_template_df)
    add_csv("consensus_rerank_release_decisions_normalized.csv", consensus_rerank_release_decision_df)
    add_csv("consensus_rerank_release_decision_validation.csv", consensus_rerank_release_decision_validation_df)
    add_csv("consensus_rerank_release_decision_summary.csv", consensus_rerank_release_decision_summary_df)
    add_csv("consensus_rerank_release_apply_plan.csv", consensus_rerank_release_apply_plan_df)
    if _safe_text(consensus_rerank_release_apply_report_markdown):
        artifacts.append(
            (
                "consensus_rerank_release_apply_report.md",
                str(consensus_rerank_release_apply_report_markdown).encode("utf-8"),
            )
        )
    add_csv("consensus_rerank_release_execution_template.csv", consensus_rerank_release_execution_template_df)
    add_csv("consensus_rerank_release_execution_receipt_normalized.csv", consensus_rerank_release_execution_receipt_df)
    add_csv("consensus_rerank_release_execution_validation.csv", consensus_rerank_release_execution_validation_df)
    add_csv("consensus_rerank_release_execution_summary.csv", consensus_rerank_release_execution_summary_df)
    if _safe_text(consensus_rerank_release_execution_report_markdown):
        artifacts.append(
            (
                "consensus_rerank_release_execution_report.md",
                str(consensus_rerank_release_execution_report_markdown).encode("utf-8"),
            )
        )
    if _safe_text(consensus_rerank_release_closure_certificate_markdown):
        artifacts.append(
            (
                "consensus_rerank_release_closure_certificate.md",
                str(consensus_rerank_release_closure_certificate_markdown).encode("utf-8"),
            )
        )
    add_csv("consensus_rerank_release_closure_ledger.csv", consensus_rerank_release_closure_ledger_df)
    add_csv("consensus_rerank_precision_scorecard.csv", consensus_rerank_precision_scorecard_df)
    add_csv("consensus_rerank_simulation_delta.csv", consensus_rerank_simulation_delta_df)
    add_csv("consensus_rerank_apply_simulation.csv", consensus_rerank_apply_simulation_df)
    add_csv("consensus_rerank_action_queue.csv", consensus_rerank_action_queue_df)
    if _safe_text(consensus_rerank_action_checklist_markdown):
        artifacts.append(("consensus_rerank_action_checklist.md", str(consensus_rerank_action_checklist_markdown).encode("utf-8")))
    add_csv("consensus_rerank_policy_gate.csv", consensus_rerank_policy_gate_df)
    add_csv("consensus_rerank_preview.csv", consensus_rerank_preview_df)
    add_csv("consensus_rerank_suggestions.csv", consensus_rerank_suggestion_df)
    add_csv("consensus_rerank_guardrail_artifact_manifest.csv", artifact_manifest_df)

    if not artifacts:
        return b""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name, data in artifacts:
            archive.writestr(file_name, data)
    return buffer.getvalue()


def verify_consensus_rerank_guardrail_handoff_zip(
    handoff_zip: bytes,
    artifact_manifest_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if not handoff_zip or artifact_manifest_df is None or getattr(artifact_manifest_df, "empty", True):
        return _empty_consensus_rerank_guardrail_bundle_verification_df()
    required_columns = {"file_name", "byte_size", "sha256"}
    if not required_columns.issubset(set(artifact_manifest_df.columns)):
        return _empty_consensus_rerank_guardrail_bundle_verification_df()

    rows: list[dict[str, Any]] = []
    try:
        archive = zipfile.ZipFile(BytesIO(handoff_zip), mode="r")
    except zipfile.BadZipFile:
        return pd.DataFrame(
            [
                {
                    "file_name": "consensus_rerank_guardrail_handoff.zip",
                    "expected_byte_size": 0,
                    "actual_byte_size": len(handoff_zip),
                    "expected_sha256": "",
                    "actual_sha256": hashlib.sha256(handoff_zip).hexdigest() if handoff_zip else "",
                    "verification_status": "invalid-zip",
                    "issue": "The handoff bundle is not a readable ZIP archive.",
                    "recommended_action": "Regenerate the rerank guardrail handoff ZIP before archival or review.",
                }
            ],
            columns=CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_COLUMNS,
        )

    with archive:
        zip_names = set(archive.namelist())
        expected_names = set()
        for _, manifest_row in artifact_manifest_df.iterrows():
            file_name = _safe_text(manifest_row.get("file_name"))
            if not file_name:
                continue
            expected_names.add(file_name)
            expected_size = _safe_int(manifest_row.get("byte_size"), 0)
            expected_hash = _safe_text(manifest_row.get("sha256"))
            if file_name not in zip_names:
                rows.append(
                    {
                        "file_name": file_name,
                        "expected_byte_size": expected_size,
                        "actual_byte_size": 0,
                        "expected_sha256": expected_hash,
                        "actual_sha256": "",
                        "verification_status": "missing",
                        "issue": "Artifact listed in manifest is missing from the handoff ZIP.",
                        "recommended_action": "Regenerate the handoff ZIP from the same manifest and source tables.",
                    }
                )
                continue

            data = archive.read(file_name)
            actual_size, actual_hash = _rerank_artifact_integrity(data)
            if actual_size != expected_size:
                status = "size-mismatch"
                issue = "Artifact byte size differs from the manifest."
                action = "Do not archive this bundle; regenerate it and verify again."
            elif actual_hash != expected_hash:
                status = "hash-mismatch"
                issue = "Artifact SHA-256 differs from the manifest."
                action = "Treat the bundle as modified or corrupted; regenerate it before review."
            else:
                status = "verified"
                issue = "none"
                action = "No action required."
            rows.append(
                {
                    "file_name": file_name,
                    "expected_byte_size": expected_size,
                    "actual_byte_size": actual_size,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "verification_status": status,
                    "issue": issue,
                    "recommended_action": action,
                }
            )

        ignored_unlisted = {"consensus_rerank_guardrail_artifact_manifest.csv"}
        for file_name in sorted(zip_names - expected_names - ignored_unlisted):
            data = archive.read(file_name)
            actual_size, actual_hash = _rerank_artifact_integrity(data)
            rows.append(
                {
                    "file_name": file_name,
                    "expected_byte_size": 0,
                    "actual_byte_size": actual_size,
                    "expected_sha256": "",
                    "actual_sha256": actual_hash,
                    "verification_status": "unlisted-file",
                    "issue": "Artifact exists in the ZIP but is not listed in the manifest.",
                    "recommended_action": "Regenerate the manifest and ZIP together before handoff.",
                }
            )

    if not rows:
        return _empty_consensus_rerank_guardrail_bundle_verification_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_COLUMNS).reset_index(drop=True)


def build_consensus_rerank_guardrail_bundle_verification_summary(
    verification_df: Optional[pd.DataFrame],
    artifact_manifest_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if verification_df is None or getattr(verification_df, "empty", True):
        return _empty_consensus_rerank_guardrail_bundle_verification_summary_df()

    statuses = verification_df["verification_status"].astype(str).str.lower() if "verification_status" in verification_df.columns else pd.Series(dtype=str)
    checked_files = int(len(verification_df))
    verified_files = int((statuses == "verified").sum())
    missing_files = int((statuses == "missing").sum())
    size_mismatch_files = int((statuses == "size-mismatch").sum())
    hash_mismatch_files = int((statuses == "hash-mismatch").sum())
    unlisted_files = int((statuses == "unlisted-file").sum())
    invalid_zip_rows = int((statuses == "invalid-zip").sum())
    failed_files = int(checked_files - verified_files)
    manifest_rows = 0 if artifact_manifest_df is None or getattr(artifact_manifest_df, "empty", True) else int(len(artifact_manifest_df))

    if invalid_zip_rows > 0:
        verification_status = "invalid-zip"
        recommended_action = "Regenerate the handoff ZIP; the current file cannot be read."
    elif failed_files > 0:
        verification_status = "failed"
        recommended_action = "Do not archive or hand off this rerank package until missing, mismatched, or unlisted files are resolved."
    elif manifest_rows and verified_files >= manifest_rows:
        verification_status = "verified"
        recommended_action = "The handoff ZIP matches its manifest; archive it with the guardrail report."
    else:
        verification_status = "incomplete"
        recommended_action = "Regenerate the manifest and ZIP together, then rerun verification."

    return pd.DataFrame(
        [
            {
                "verification_status": verification_status,
                "manifest_rows": manifest_rows,
                "checked_files": checked_files,
                "verified_files": verified_files,
                "failed_files": failed_files,
                "missing_files": missing_files,
                "size_mismatch_files": size_mismatch_files,
                "hash_mismatch_files": hash_mismatch_files,
                "unlisted_files": unlisted_files,
                "invalid_zip_rows": invalid_zip_rows,
                "recommended_action": recommended_action,
            }
        ],
        columns=CONSENSUS_RERANK_GUARDRAIL_BUNDLE_VERIFICATION_SUMMARY_COLUMNS,
    )


def build_consensus_rerank_guardrail_handoff_certificate_markdown(
    handoff_zip: bytes,
    verification_summary_df: Optional[pd.DataFrame],
    artifact_manifest_df: Optional[pd.DataFrame],
    consensus_rerank_precision_guardrail_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Consensus rerank guardrail handoff certificate",
) -> str:
    if not handoff_zip:
        return ""

    bundle_size, bundle_hash = _rerank_artifact_integrity(handoff_zip)
    manifest_rows = 0 if artifact_manifest_df is None or getattr(artifact_manifest_df, "empty", True) else int(len(artifact_manifest_df))
    summary_row = (
        verification_summary_df.iloc[0]
        if verification_summary_df is not None and not getattr(verification_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    guardrail_row = (
        consensus_rerank_precision_guardrail_df.iloc[0]
        if consensus_rerank_precision_guardrail_df is not None and not getattr(consensus_rerank_precision_guardrail_df, "empty", True)
        else pd.Series(dtype=object)
    )
    release_row = (
        consensus_rerank_release_decision_summary_df.iloc[0]
        if consensus_rerank_release_decision_summary_df is not None and not getattr(consensus_rerank_release_decision_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )

    verification_status = _safe_text(summary_row.get("verification_status"), "not-verified")
    checked_files = _safe_text(summary_row.get("checked_files"), "0")
    verified_files = _safe_text(summary_row.get("verified_files"), "0")
    failed_files = _safe_text(summary_row.get("failed_files"), "0")
    recommended_action = _safe_text(summary_row.get("recommended_action"), "Run bundle verification before handoff.")
    guardrail_status = _safe_text(guardrail_row.get("guardrail_status"), "unknown")
    guardrail_decision = _safe_text(guardrail_row.get("guardrail_decision"), "unknown")
    apply_mode = _safe_text(guardrail_row.get("apply_mode"), "unknown")
    can_manual = _safe_bool(guardrail_row.get("can_apply_after_manual_review"))
    can_auto = _safe_bool(guardrail_row.get("can_enable_auto_rerank"))
    release_review_status = _safe_text(release_row.get("release_review_status"))
    release_allowed = _safe_bool(release_row.get("release_allowed"))
    release_decision_rows = _safe_text(release_row.get("decision_rows"), "0")
    release_blocked_rows = _safe_text(release_row.get("blocked_rows"), "0")
    release_recommended_action = _safe_text(release_row.get("recommended_action"))

    lines = [
        f"# {title}",
        "",
        "This certificate records the rerank guardrail ZIP identity, integrity result, and release decision.",
        "",
        "## Bundle identity",
        "",
        "- File: `consensus_rerank_guardrail_handoff.zip`",
        f"- Byte size: {bundle_size}",
        f"- SHA-256: `{bundle_hash}`",
        f"- Manifest rows: {manifest_rows}",
        "",
        "## Verification summary",
        "",
        f"- Status: `{verification_status}`",
        f"- Checked files: {checked_files}",
        f"- Verified files: {verified_files}",
        f"- Failed files: {failed_files}",
        f"- Recommended action: {recommended_action}",
        "",
        "## Guardrail decision",
        "",
        f"- Guardrail status: `{guardrail_status}`",
        f"- Decision: `{guardrail_decision}`",
        f"- Apply mode: `{apply_mode}`",
        f"- Can enable automatic rerank: {'yes' if can_auto else 'no'}",
        f"- Can apply after manual review: {'yes' if can_manual else 'no'}",
    ]
    if release_review_status:
        lines.extend(
            [
                "",
                "## Release review",
                "",
                f"- Status: `{release_review_status}`",
                f"- Release allowed: {'yes' if release_allowed else 'no'}",
                f"- Decision rows: {release_decision_rows}",
                f"- Blocked rows: {release_blocked_rows}",
                f"- Recommended action: {release_recommended_action or 'Review the release decision summary before applying rerank.'}",
            ]
        )
    lines.extend(
        [
            "",
            "## How to use",
            "",
            "- Keep this certificate next to the handoff ZIP.",
            "- Recompute the ZIP SHA-256 before handoff or archival and compare it with this certificate.",
            "- Open `consensus_rerank_precision_guardrail_report.md` inside the ZIP before changing active ranking.",
            "- If verification is not `verified`, regenerate the ZIP and manifest before review.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _release_template_decision_for_guardrail(guardrail_decision: str, guardrail_status: str) -> str:
    decision = _safe_text(guardrail_decision).lower()
    status = _safe_text(guardrail_status).lower()
    if decision == "allow-after-review":
        return "approve-after-manual-review"
    if decision == "keep-current-ranking":
        return "keep-current-ranking"
    if decision == "hold-for-review":
        return "hold-for-more-evidence"
    if status == "blocked" or decision == "do-not-apply":
        return "reject-or-hold"
    return "review-required"


def build_consensus_rerank_release_decision_template(
    consensus_rerank_precision_guardrail_df: Optional[pd.DataFrame],
    consensus_rerank_action_queue_df: Optional[pd.DataFrame] = None,
    consensus_rerank_simulation_delta_df: Optional[pd.DataFrame] = None,
    *,
    max_review_items: int = 8,
) -> pd.DataFrame:
    if consensus_rerank_precision_guardrail_df is None or getattr(consensus_rerank_precision_guardrail_df, "empty", True):
        return _empty_consensus_rerank_release_decision_template_df()

    guardrail = consensus_rerank_precision_guardrail_df.iloc[0]
    guardrail_status = _safe_text(guardrail.get("guardrail_status"), "unknown")
    guardrail_decision = _safe_text(guardrail.get("guardrail_decision"), "unknown")
    can_manual = _safe_bool(guardrail.get("can_apply_after_manual_review"))
    release_pocket = _safe_text(guardrail.get("top_positive_pocket_id"), "none")
    if release_pocket == "none":
        release_pocket = _safe_text(guardrail.get("top_blocker_pocket_id"), "none")
    rows: list[dict[str, Any]] = [
        {
            "decision_item_id": "release-guardrail",
            "decision_scope": "release",
            "pocket_id": release_pocket,
            "decision_item": "Overall consensus rerank release decision",
            "current_guardrail_status": guardrail_status,
            "current_issue_type": "-",
            "current_change_type": "-",
            "recommended_decision": _release_template_decision_for_guardrail(guardrail_decision, guardrail_status),
            "review_decision": "review",
            "reviewer": "",
            "review_note": "",
            "verified_anchor_residues": "",
            "verified_sources": "",
            "blocker_resolved": "no" if _safe_int(guardrail.get("open_blocker_rows"), 0) > 0 else "not-applicable",
            "manual_approval_allowed": bool(can_manual),
            "decision_due_to": _safe_text(guardrail.get("decision_reason"), "-"),
            "required_evidence": _safe_text(guardrail.get("first_required_clearance"), "Confirm all anchors and guardrail evidence."),
            "recommended_action": _safe_text(guardrail.get("recommended_action"), "Review the guardrail report before applying rerank."),
        }
    ]

    max_items = max(1, int(max_review_items))
    item_count = 0
    if consensus_rerank_action_queue_df is not None and not getattr(consensus_rerank_action_queue_df, "empty", True):
        queue = consensus_rerank_action_queue_df.copy()
        for column, default in {
            "action_priority": 999,
            "pocket_id": "",
            "issue_type": "review",
            "issue_severity": "review",
            "required_fix": "",
            "can_apply_after_fix": False,
            "consensus_anchor_residues": "",
            "recommended_action": "",
        }.items():
            if column not in queue.columns:
                queue[column] = default
        queue["_priority_sort"] = pd.to_numeric(queue["action_priority"], errors="coerce").fillna(999)
        queue = queue.sort_values(["_priority_sort", "pocket_id", "issue_type"], ascending=[True, True, True])
        for _, row in queue.head(max_items).iterrows():
            item_count += 1
            issue_type = _safe_text(row.get("issue_type"), "review")
            rows.append(
                {
                    "decision_item_id": f"clearance-{item_count}",
                    "decision_scope": "clearance",
                    "pocket_id": _safe_text(row.get("pocket_id"), "none"),
                    "decision_item": f"Resolve {issue_type}",
                    "current_guardrail_status": guardrail_status,
                    "current_issue_type": issue_type,
                    "current_change_type": "-",
                    "recommended_decision": "resolve-before-approval" if _safe_text(row.get("issue_severity")).lower() in {"blocking", "missing-evidence", "review"} else "acknowledge",
                    "review_decision": "review",
                    "reviewer": "",
                    "review_note": "",
                    "verified_anchor_residues": _safe_text(row.get("consensus_anchor_residues"), ""),
                    "verified_sources": "",
                    "blocker_resolved": "no",
                    "manual_approval_allowed": bool(_safe_bool(row.get("can_apply_after_fix")) and can_manual),
                    "decision_due_to": _safe_text(row.get("issue_severity"), "review"),
                    "required_evidence": _safe_text(row.get("required_fix"), "Resolve this review item before release."),
                    "recommended_action": _safe_text(row.get("recommended_action"), _safe_text(row.get("required_fix"), "-")),
                }
            )

    if consensus_rerank_simulation_delta_df is not None and not getattr(consensus_rerank_simulation_delta_df, "empty", True):
        delta = consensus_rerank_simulation_delta_df.copy()
        for column, default in {
            "impact_priority": 999,
            "pocket_id": "",
            "change_type": "unchanged-monitor",
            "issue_type": "",
            "rank_delta": 0,
            "precision_interpretation": "",
            "required_before_trust": "",
            "consensus_anchor_residues": "",
            "recommended_action": "",
        }.items():
            if column not in delta.columns:
                delta[column] = default
        delta["_impact_sort"] = pd.to_numeric(delta["impact_priority"], errors="coerce").fillna(999)
        delta["_abs_rank_delta"] = pd.to_numeric(delta["rank_delta"], errors="coerce").fillna(0).abs()
        delta = delta.sort_values(["_impact_sort", "_abs_rank_delta", "pocket_id"], ascending=[True, False, True])
        existing_keys = {(row["decision_scope"], row["pocket_id"], row["current_change_type"]) for row in rows}
        for _, row in delta.iterrows():
            if item_count >= max_items:
                break
            change_type = _safe_text(row.get("change_type"), "unchanged-monitor")
            if change_type in {"unchanged-monitor", "score-down-no-rank-change"}:
                continue
            key = ("rank-change", _safe_text(row.get("pocket_id"), "none"), change_type)
            if key in existing_keys:
                continue
            item_count += 1
            existing_keys.add(key)
            rank_delta = _safe_int(row.get("rank_delta"), 0)
            if change_type.startswith("rank-up"):
                recommended_decision = "approve-rank-up-after-anchor-review"
            elif change_type.startswith("rank-down"):
                recommended_decision = "approve-conservative-rank-down"
            elif change_type.startswith("frozen"):
                recommended_decision = "acknowledge-freeze"
            else:
                recommended_decision = "review-change"
            rows.append(
                {
                    "decision_item_id": f"rank-change-{item_count}",
                    "decision_scope": "rank-change",
                    "pocket_id": _safe_text(row.get("pocket_id"), "none"),
                    "decision_item": f"{change_type} (rank delta {rank_delta:+d})",
                    "current_guardrail_status": guardrail_status,
                    "current_issue_type": _safe_text(row.get("issue_type"), "-"),
                    "current_change_type": change_type,
                    "recommended_decision": recommended_decision,
                    "review_decision": "review",
                    "reviewer": "",
                    "review_note": "",
                    "verified_anchor_residues": _safe_text(row.get("consensus_anchor_residues"), ""),
                    "verified_sources": "",
                    "blocker_resolved": "not-applicable",
                    "manual_approval_allowed": bool(can_manual and not change_type.startswith("frozen")),
                    "decision_due_to": _safe_text(row.get("precision_interpretation"), "-"),
                    "required_evidence": _safe_text(row.get("required_before_trust"), "Verify anchors and rerank evidence before approval."),
                    "recommended_action": _safe_text(row.get("recommended_action"), "Review this simulated rank change."),
                }
            )

    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS).reset_index(drop=True)


def _normalize_release_column_key(value: Any) -> str:
    text = _safe_text(value).lower()
    chars: list[str] = []
    last_was_separator = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_was_separator = False
        elif not last_was_separator:
            chars.append("_")
            last_was_separator = True
    return "".join(chars).strip("_")


def _pick_release_column(columns: list[str], aliases: set[str]) -> str:
    normalized = {_normalize_release_column_key(column): column for column in columns}
    for alias in aliases:
        column = normalized.get(alias)
        if column:
            return column
    return ""


def _normalize_release_review_decision(value: Any) -> str:
    text = _safe_text(value).lower()
    if not text:
        return "review"
    if text in {"1", "true", "yes", "y", "approve", "approved", "accept", "accepted", "release", "signoff", "signed_off"}:
        return "approve"
    if text in {"0", "false", "no", "n", "reject", "rejected", "deny", "denied", "do_not_apply", "do-not-apply"}:
        return "reject"
    if text in {"hold", "held", "defer", "deferred", "blocked", "wait"}:
        return "hold"
    if text in {"review", "pending", "manual_review", "manual-review", "needs_review", "needs-review"}:
        return "review"
    return text


def _release_blocker_resolved(value: Any, *, strict: bool = False) -> bool:
    text = _safe_text(value).lower()
    resolved_values = {"1", "true", "yes", "y", "resolved", "done", "clear", "cleared"}
    if strict:
        return text in resolved_values
    return text in {*resolved_values, "not_applicable", "not-applicable", "n_a", "n/a", "na", "none", "-"}


def parse_consensus_rerank_release_decision_table(decision_text: str | bytes | None) -> tuple[pd.DataFrame, dict[str, str]]:
    if isinstance(decision_text, bytes):
        text = decision_text.decode("utf-8", errors="ignore")
    else:
        text = _safe_text(decision_text)
    if not text:
        return _empty_consensus_rerank_release_decision_template_df(), {
            "status": "empty",
            "input_rows": "0",
            "decision_rows": "0",
            "skipped_rows": "0",
        }

    try:
        raw = pd.read_csv(StringIO(text), sep=None, engine="python")
    except Exception as exc:
        return _empty_consensus_rerank_release_decision_template_df(), {
            "status": "parse-error",
            "input_rows": "0",
            "decision_rows": "0",
            "skipped_rows": "0",
            "message": str(exc),
        }
    if raw.empty:
        return _empty_consensus_rerank_release_decision_template_df(), {
            "status": "empty",
            "input_rows": "0",
            "decision_rows": "0",
            "skipped_rows": "0",
        }

    columns = [str(column) for column in raw.columns]
    aliases = {
        "decision_item_id": {"decision_item_id", "item_id", "release_item_id", "id"},
        "decision_scope": {"decision_scope", "scope", "review_scope"},
        "pocket_id": {"pocket_id", "pocket", "candidate_pocket"},
        "decision_item": {"decision_item", "item", "review_item"},
        "current_guardrail_status": {"current_guardrail_status", "guardrail_status"},
        "current_issue_type": {"current_issue_type", "issue_type"},
        "current_change_type": {"current_change_type", "change_type"},
        "recommended_decision": {"recommended_decision", "recommendation"},
        "review_decision": {"review_decision", "decision", "review_status", "approval"},
        "reviewer": {"reviewer", "curator", "approver", "user"},
        "review_note": {"review_note", "note", "notes", "comment", "comments"},
        "verified_anchor_residues": {"verified_anchor_residues", "verified_anchors", "anchor_residues", "anchors", "residues"},
        "verified_sources": {"verified_sources", "verified_source", "source", "sources", "citation", "reference", "references"},
        "blocker_resolved": {"blocker_resolved", "resolved", "clearance_resolved"},
        "manual_approval_allowed": {"manual_approval_allowed", "approval_allowed", "can_approve"},
        "decision_due_to": {"decision_due_to", "reason", "rationale"},
        "required_evidence": {"required_evidence", "evidence_required", "required_fix"},
        "recommended_action": {"recommended_action", "next_action", "action"},
    }
    selected = {column: _pick_release_column(columns, column_aliases) for column, column_aliases in aliases.items()}
    if not selected["decision_item_id"] or not selected["review_decision"]:
        return _empty_consensus_rerank_release_decision_template_df(), {
            "status": "missing-required-columns",
            "input_rows": str(len(raw)),
            "decision_rows": "0",
            "skipped_rows": str(len(raw)),
            "message": "decision_item_id and review_decision columns are required.",
        }

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in raw.iterrows():
        decision_item_id = _safe_text(row.get(selected["decision_item_id"]))
        if not decision_item_id:
            skipped += 1
            continue
        normalized: dict[str, Any] = {}
        for column in CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS:
            source_column = selected.get(column, "")
            if column == "decision_item_id":
                normalized[column] = decision_item_id
            elif column == "review_decision":
                normalized[column] = _normalize_release_review_decision(row.get(source_column)) if source_column else "review"
            elif column == "manual_approval_allowed":
                normalized[column] = _safe_bool(row.get(source_column)) if source_column else ""
            else:
                normalized[column] = _safe_text(row.get(source_column)) if source_column else ""
        rows.append(normalized)

    if not rows:
        return _empty_consensus_rerank_release_decision_template_df(), {
            "status": "empty-after-normalization",
            "input_rows": str(len(raw)),
            "decision_rows": "0",
            "skipped_rows": str(skipped),
        }

    decisions = pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS).reset_index(drop=True)
    review_decisions = decisions["review_decision"].astype(str).str.lower()
    return decisions, {
        "status": "ok",
        "input_rows": str(len(raw)),
        "decision_rows": str(len(decisions)),
        "skipped_rows": str(skipped),
        "approve_rows": str(int((review_decisions == "approve").sum())),
        "reject_rows": str(int((review_decisions == "reject").sum())),
        "hold_rows": str(int((review_decisions == "hold").sum())),
        "review_rows": str(int((review_decisions == "review").sum())),
    }


def validate_consensus_rerank_release_decisions(
    decision_df: Optional[pd.DataFrame],
    release_decision_template_df: Optional[pd.DataFrame] = None,
    consensus_rerank_precision_guardrail_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if decision_df is None or getattr(decision_df, "empty", True):
        return _empty_consensus_rerank_release_decision_validation_df()

    working = decision_df.copy()
    for column in CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS:
        if column not in working.columns:
            working[column] = ""

    template_map: dict[str, pd.Series] = {}
    if release_decision_template_df is not None and not getattr(release_decision_template_df, "empty", True):
        template = release_decision_template_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_DECISION_TEMPLATE_COLUMNS:
            if column not in template.columns:
                template[column] = ""
        for _, template_row in template.iterrows():
            item_id = _safe_text(template_row.get("decision_item_id"))
            if item_id and item_id not in template_map:
                template_map[item_id] = template_row

    guardrail = (
        consensus_rerank_precision_guardrail_df.iloc[0]
        if consensus_rerank_precision_guardrail_df is not None and not getattr(consensus_rerank_precision_guardrail_df, "empty", True)
        else pd.Series(dtype=object)
    )

    duplicate_counts: dict[str, int] = {}
    duplicate_decisions: dict[str, set[str]] = {}
    for _, row in working.iterrows():
        item_id = _safe_text(row.get("decision_item_id"))
        if not item_id:
            continue
        duplicate_counts[item_id] = duplicate_counts.get(item_id, 0) + 1
        duplicate_decisions.setdefault(item_id, set()).add(_normalize_release_review_decision(row.get("review_decision")))

    rows: list[dict[str, Any]] = []
    for row_index, (_, decision) in enumerate(working.iterrows(), start=1):
        item_id = _safe_text(decision.get("decision_item_id"))
        template_row = template_map.get(item_id)
        template_match = bool(template_row is not None) if template_map else True

        def merged_text(column: str, default: str = "") -> str:
            uploaded = _safe_text(decision.get(column))
            if uploaded:
                return uploaded
            if template_row is not None:
                return _safe_text(template_row.get(column), default)
            return default

        scope = merged_text("decision_scope", "release" if item_id == "release-guardrail" else "review")
        pocket_id = merged_text("pocket_id", "none")
        recommended_decision = merged_text("recommended_decision", "review-required")
        review_decision = _normalize_release_review_decision(decision.get("review_decision"))
        reviewer = _safe_text(decision.get("reviewer"))
        review_note = _safe_text(decision.get("review_note"))
        verified_anchors = _safe_text(decision.get("verified_anchor_residues"))
        verified_sources = _safe_text(decision.get("verified_sources"))
        blocker_resolved = _safe_text(decision.get("blocker_resolved"))
        if not blocker_resolved and template_row is not None:
            blocker_resolved = _safe_text(template_row.get("blocker_resolved"))
        manual_uploaded = _safe_text(decision.get("manual_approval_allowed"))
        manual_allowed = _safe_bool(decision.get("manual_approval_allowed")) if manual_uploaded else (
            _safe_bool(template_row.get("manual_approval_allowed")) if template_row is not None else False
        )
        required_evidence = merged_text("required_evidence", "")

        flags: list[str] = []
        reasons: list[str] = []
        fixes: list[str] = []
        can_release = False

        if not item_id:
            flags.append("missing-decision-item-id")
            reasons.append("Decision row has no decision_item_id.")
            fixes.append("Use an unmodified decision_item_id from the release decision template.")
        if template_map and not template_match:
            flags.append("unmatched-template-item")
            reasons.append("Decision item does not exist in the current release decision template.")
            fixes.append("Download the latest template and copy decisions into matching decision_item_id rows.")

        duplicate_count = duplicate_counts.get(item_id, 0)
        if duplicate_count > 1:
            if len(duplicate_decisions.get(item_id, set())) > 1:
                flags.append("conflicting-duplicate")
                reasons.append("Multiple uploaded rows for the same decision item disagree.")
                fixes.append("Keep one non-conflicting row per decision_item_id.")
            else:
                flags.append("duplicate")
                reasons.append("Repeated identical decision item found.")
                fixes.append("Remove duplicate rows to keep the approval trail clean.")

        if review_decision not in {"approve", "reject", "hold", "review"}:
            flags.append("invalid-decision")
            reasons.append("review_decision must be approve, reject, hold, or review.")
            fixes.append("Normalize review_decision before upload.")

        if review_decision == "approve":
            if item_id == "release-guardrail" and not _safe_bool(guardrail.get("can_apply_after_manual_review")):
                flags.append("guardrail-blocks-manual-approval")
                reasons.append("The current guardrail does not allow manual rerank application.")
                fixes.append("Resolve guardrail blockers and regenerate the release decision template.")
            if not manual_allowed:
                flags.append("manual-approval-not-allowed")
                reasons.append("This decision item is not eligible for manual approval in the current guardrail state.")
                fixes.append("Resolve blockers or keep this item on hold.")
            if not reviewer:
                flags.append("missing-reviewer")
                reasons.append("Approval lacks a reviewer.")
                fixes.append("Fill the reviewer column.")
            if not verified_sources:
                flags.append("missing-verified-sources")
                reasons.append("Approval lacks verified_sources.")
                fixes.append("Add PMID, DOI, database entry, report path, or another citable source.")
            needs_anchors = (
                scope in {"release", "rank-change"}
                or bool(_safe_text(template_row.get("verified_anchor_residues")) if template_row is not None else "")
                or "anchor" in required_evidence.lower()
                or "residue" in required_evidence.lower()
            )
            if needs_anchors and not verified_anchors:
                flags.append("missing-verified-anchors")
                reasons.append("Approval lacks verified_anchor_residues.")
                fixes.append("List the reviewed chain:residue anchors before approval.")
            if scope == "clearance":
                if not _release_blocker_resolved(blocker_resolved, strict=True):
                    flags.append("unresolved-blocker")
                    reasons.append("Clearance approval requires blocker_resolved=yes.")
                    fixes.append("Resolve the blocker and mark blocker_resolved=yes.")
            elif not _release_blocker_resolved(blocker_resolved, strict=False):
                flags.append("unresolved-blocker")
                reasons.append("Approval still has an unresolved blocker flag.")
                fixes.append("Resolve the blocker or set blocker_resolved=not-applicable when justified.")

        blocking_flags = {
            "missing-decision-item-id",
            "unmatched-template-item",
            "conflicting-duplicate",
            "invalid-decision",
            "guardrail-blocks-manual-approval",
            "manual-approval-not-allowed",
            "missing-reviewer",
            "missing-verified-sources",
            "missing-verified-anchors",
            "unresolved-blocker",
        }
        if any(flag in blocking_flags for flag in flags):
            validation_status = "blocked"
        elif review_decision == "approve":
            validation_status = "warning" if flags else "approved"
            can_release = True
        elif review_decision == "reject":
            validation_status = "rejected"
        elif review_decision == "hold":
            validation_status = "held"
        else:
            validation_status = "pending-review"

        rows.append(
            {
                "row_index": row_index,
                "decision_item_id": item_id,
                "decision_scope": scope,
                "pocket_id": pocket_id,
                "review_decision": review_decision,
                "template_match": bool(template_match),
                "recommended_decision": recommended_decision,
                "manual_approval_allowed": bool(manual_allowed),
                "validation_status": validation_status,
                "issue_flags": ", ".join(dict.fromkeys(flags)) if flags else "none",
                "can_release": bool(can_release),
                "validation_reason": " ".join(reasons) if reasons else (
                    "Decision item is approved for manual release." if can_release else "Decision item is not approved for release."
                ),
                "required_fix": " ".join(dict.fromkeys(fixes)) if fixes else "none",
                "reviewer": reviewer,
                "verified_anchor_residues": verified_anchors,
                "verified_sources": verified_sources,
                "blocker_resolved": blocker_resolved,
            }
        )

    if not rows:
        return _empty_consensus_rerank_release_decision_validation_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_DECISION_VALIDATION_COLUMNS).reset_index(drop=True)


def build_consensus_rerank_release_decision_summary(
    validation_df: Optional[pd.DataFrame],
    decision_df: Optional[pd.DataFrame] = None,
    release_decision_template_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if validation_df is None or getattr(validation_df, "empty", True):
        return _empty_consensus_rerank_release_decision_summary_df()

    statuses = validation_df["validation_status"].astype(str).str.lower() if "validation_status" in validation_df.columns else pd.Series(dtype=str)
    issue_flags = validation_df["issue_flags"].astype(str).str.lower() if "issue_flags" in validation_df.columns else pd.Series(dtype=str)
    template_rows = 0 if release_decision_template_df is None or getattr(release_decision_template_df, "empty", True) else int(len(release_decision_template_df))
    decision_rows = 0 if decision_df is None or getattr(decision_df, "empty", True) else int(len(decision_df))
    matched_rows = int(validation_df["template_match"].astype(bool).sum()) if "template_match" in validation_df.columns else int(len(validation_df))
    approved_rows = int(statuses.isin(["approved", "warning"]).sum())
    rejected_rows = int((statuses == "rejected").sum())
    hold_rows = int((statuses == "held").sum())
    review_rows = int((statuses == "pending-review").sum())
    blocked_rows = int((statuses == "blocked").sum())
    warning_rows = int((statuses == "warning").sum())
    unmatched_rows = int(issue_flags.str.contains("unmatched-template-item", regex=False).sum()) if not issue_flags.empty else 0
    missing_reviewer_rows = int(issue_flags.str.contains("missing-reviewer", regex=False).sum()) if not issue_flags.empty else 0
    missing_evidence_rows = int(
        (
            issue_flags.str.contains("missing-verified-sources", regex=False)
            | issue_flags.str.contains("missing-verified-anchors", regex=False)
        ).sum()
    ) if not issue_flags.empty else 0
    unresolved_blocker_rows = int(issue_flags.str.contains("unresolved-blocker", regex=False).sum()) if not issue_flags.empty else 0

    missing_decision_rows = 0
    if template_rows > 0 and release_decision_template_df is not None and "decision_item_id" in release_decision_template_df.columns:
        expected_ids = {_safe_text(value) for value in release_decision_template_df["decision_item_id"].tolist() if _safe_text(value)}
        decided_ids = set()
        if "decision_item_id" in validation_df.columns:
            decided_ids = {_safe_text(value) for value in validation_df["decision_item_id"].tolist() if _safe_text(value)}
        missing_decision_rows = int(len(expected_ids - decided_ids))

    release_rows = validation_df[validation_df["decision_item_id"].astype(str) == "release-guardrail"] if "decision_item_id" in validation_df.columns else pd.DataFrame()
    release_row_approved = bool(
        not release_rows.empty
        and str(release_rows.iloc[0].get("validation_status")).lower() in {"approved", "warning"}
        and _safe_bool(release_rows.iloc[0].get("can_release"))
    )
    release_allowed = bool(
        release_row_approved
        and blocked_rows == 0
        and warning_rows == 0
        and rejected_rows == 0
        and hold_rows == 0
        and review_rows == 0
        and unmatched_rows == 0
        and missing_decision_rows == 0
    )

    if blocked_rows > 0 or unmatched_rows > 0 or missing_decision_rows > 0:
        release_review_status = "blocked"
        recommended_action = "Fix blocked, unmatched, or missing release decision rows before applying rerank."
    elif rejected_rows > 0:
        release_review_status = "rejected"
        recommended_action = "Do not apply rerank; reviewer rejected at least one release decision item."
    elif hold_rows > 0:
        release_review_status = "hold"
        recommended_action = "Keep rerank diagnostic until held items are resolved."
    elif review_rows > 0:
        release_review_status = "pending-review"
        recommended_action = "Complete review_decision values for every release template row."
    elif warning_rows > 0:
        release_review_status = "needs-cleanup"
        recommended_action = "Clean duplicate or warning rows, then re-upload the decision CSV."
    elif release_allowed:
        release_review_status = "approved-for-manual-release"
        recommended_action = "Manual rerank release is approved; archive the decision CSV with the guardrail bundle."
    else:
        release_review_status = "pending-review"
        recommended_action = "Upload a completed release decision CSV before applying rerank."

    return pd.DataFrame(
        [
            {
                "release_review_status": release_review_status,
                "template_rows": template_rows,
                "decision_rows": decision_rows,
                "matched_rows": matched_rows,
                "approved_rows": approved_rows,
                "rejected_rows": rejected_rows,
                "hold_rows": hold_rows,
                "review_rows": review_rows,
                "blocked_rows": blocked_rows,
                "warning_rows": warning_rows,
                "unmatched_rows": unmatched_rows,
                "missing_decision_rows": missing_decision_rows,
                "missing_reviewer_rows": missing_reviewer_rows,
                "missing_evidence_rows": missing_evidence_rows,
                "unresolved_blocker_rows": unresolved_blocker_rows,
                "release_allowed": release_allowed,
                "recommended_action": recommended_action,
            }
        ],
        columns=CONSENSUS_RERANK_RELEASE_DECISION_SUMMARY_COLUMNS,
    )


def build_consensus_rerank_release_apply_plan(
    consensus_rerank_apply_simulation_df: Optional[pd.DataFrame],
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame],
    consensus_rerank_release_decision_validation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if (
        consensus_rerank_apply_simulation_df is None
        or getattr(consensus_rerank_apply_simulation_df, "empty", True)
        or consensus_rerank_release_decision_summary_df is None
        or getattr(consensus_rerank_release_decision_summary_df, "empty", True)
    ):
        return _empty_consensus_rerank_release_apply_plan_df()

    summary = consensus_rerank_release_decision_summary_df.iloc[0]
    release_allowed = _safe_bool(summary.get("release_allowed"))
    release_review_status = _safe_text(summary.get("release_review_status"), "unknown")
    if not release_allowed or release_review_status != "approved-for-manual-release":
        return _empty_consensus_rerank_release_apply_plan_df()

    simulation = consensus_rerank_apply_simulation_df.copy()
    for column, default in {
        "simulated_rank": 0,
        "pocket_id": "",
        "current_rank": 0,
        "simulated_rank_delta": 0,
        "current_decision_score": 0.0,
        "simulation_score": 0.0,
        "apply_status": "",
        "apply_decision": "",
        "policy_allows_apply": False,
        "required_before_apply": "",
        "consensus_anchor_residues": "none",
        "recommended_action": "",
    }.items():
        if column not in simulation.columns:
            simulation[column] = default

    statuses = simulation["apply_status"].astype(str).str.lower()
    allowed_statuses = {"keep-current-ready", "apply-ready-after-review", "keep-after-review", "monitor-only"}
    has_unready_status = bool((~statuses.isin(allowed_statuses)).any())
    policy_ready = bool(simulation["policy_allows_apply"].apply(_safe_bool).all()) if "policy_allows_apply" in simulation.columns else False
    if has_unready_status or not policy_ready:
        return _empty_consensus_rerank_release_apply_plan_df()

    validation_reference = "release decision summary"
    if consensus_rerank_release_decision_validation_df is not None and not getattr(consensus_rerank_release_decision_validation_df, "empty", True):
        validation = consensus_rerank_release_decision_validation_df.copy()
        release_validation = validation[validation["decision_item_id"].astype(str) == "release-guardrail"] if "decision_item_id" in validation.columns else pd.DataFrame()
        if not release_validation.empty:
            row = release_validation.iloc[0]
            reviewer = _safe_text(row.get("reviewer"), "reviewer")
            sources = _safe_text(row.get("verified_sources"), "verified sources")
            validation_reference = f"{reviewer}; {sources}"

    decision_rows = _safe_int(summary.get("decision_rows"), 0)
    approved_rows = _safe_int(summary.get("approved_rows"), 0)
    approval_reference = (
        f"{release_review_status}; decisions={decision_rows}; approved={approved_rows}; "
        f"blocked={_safe_int(summary.get('blocked_rows'), 0)}; {validation_reference}"
    )

    simulation["_manual_apply_rank_sort"] = pd.to_numeric(simulation["simulated_rank"], errors="coerce").fillna(9999)
    simulation = simulation.sort_values(["_manual_apply_rank_sort", "pocket_id"], ascending=[True, True]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for manual_rank, (_, row) in enumerate(simulation.iterrows(), start=1):
        current_rank = _safe_int(row.get("current_rank"), 0)
        simulated_rank = _safe_int(row.get("simulated_rank"), manual_rank)
        rank_delta = _safe_int(row.get("simulated_rank_delta"), 0)
        required_check = _safe_text(
            row.get("required_before_apply"),
            "Archive the handoff certificate and compare ZIP SHA-256 before applying this rank order.",
        )
        rows.append(
            {
                "manual_apply_rank": manual_rank,
                "pocket_id": _safe_text(row.get("pocket_id"), "Pocket"),
                "current_rank": current_rank,
                "simulated_rank": simulated_rank,
                "rank_delta": rank_delta,
                "current_decision_score": round(_clip(_safe_float(row.get("current_decision_score"), 0.0)), 3),
                "simulation_score": round(_clip(_safe_float(row.get("simulation_score"), 0.0)), 3),
                "apply_status": _safe_text(row.get("apply_status"), "-"),
                "apply_decision": _safe_text(row.get("apply_decision"), "-"),
                "release_apply_status": "ready-for-manual-apply",
                "release_review_status": release_review_status,
                "release_allowed": True,
                "approval_reference": approval_reference,
                "required_pre_apply_check": (
                    f"{required_check} Archive the release decision summary and handoff certificate before changing active ranking."
                ),
                "consensus_anchor_residues": _safe_text(row.get("consensus_anchor_residues"), "none"),
                "recommended_action": _safe_text(
                    row.get("recommended_action"),
                    "Apply this approved manual rank order only after recording the handoff ZIP hash.",
                ),
            }
        )

    if not rows:
        return _empty_consensus_rerank_release_apply_plan_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS).reset_index(drop=True)


def build_consensus_rerank_release_apply_report_markdown(
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame],
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Consensus rerank release apply report",
) -> str:
    if consensus_rerank_release_apply_plan_df is None or getattr(consensus_rerank_release_apply_plan_df, "empty", True):
        return ""

    plan = consensus_rerank_release_apply_plan_df.copy()
    for column, default in {
        "manual_apply_rank": 999,
        "pocket_id": "",
        "current_rank": 0,
        "simulated_rank": 0,
        "rank_delta": 0,
        "current_decision_score": 0.0,
        "simulation_score": 0.0,
        "apply_status": "",
        "apply_decision": "",
        "release_apply_status": "ready-for-manual-apply",
        "release_review_status": "",
        "release_allowed": False,
        "approval_reference": "",
        "required_pre_apply_check": "",
        "consensus_anchor_residues": "none",
        "recommended_action": "",
    }.items():
        if column not in plan.columns:
            plan[column] = default
    plan["_rank_sort"] = pd.to_numeric(plan["manual_apply_rank"], errors="coerce").fillna(999)
    plan = plan.sort_values(["_rank_sort", "pocket_id"], ascending=[True, True]).drop(columns=["_rank_sort"]).reset_index(drop=True)

    summary = (
        consensus_rerank_release_decision_summary_df.iloc[0]
        if consensus_rerank_release_decision_summary_df is not None and not getattr(consensus_rerank_release_decision_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    top = plan.iloc[0]
    rank_delta_series = pd.to_numeric(plan["rank_delta"], errors="coerce").fillna(0)
    rank_up_rows = int((rank_delta_series > 0).sum())
    rank_down_rows = int((rank_delta_series < 0).sum())
    unchanged_rows = int((rank_delta_series == 0).sum())
    plan_size, plan_hash = _rerank_artifact_integrity(_rerank_csv_artifact_bytes(plan[CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS]))

    release_review_status = _safe_text(
        summary.get("release_review_status"),
        _safe_text(top.get("release_review_status"), "approved-for-manual-release"),
    )
    release_allowed = _safe_bool(summary.get("release_allowed")) or _safe_bool(top.get("release_allowed"))
    approval_reference = _safe_text(top.get("approval_reference"), "release decision summary")

    lines = [
        f"# {title}",
        "",
        "This report is the human-readable execution worksheet for a reviewer-approved consensus rerank.",
        "It is generated only when the release decision summary allows manual release and the current apply simulation is clean.",
        "",
        "## Release gate",
        "",
        f"- Release review status: `{release_review_status}`",
        f"- Release allowed: {'yes' if release_allowed else 'no'}",
        f"- Apply plan rows: `{len(plan)}`",
        f"- Top manual rank: `{_safe_text(top.get('pocket_id'), '-')}`",
        f"- Rank-up rows: `{rank_up_rows}`",
        f"- Rank-down rows: `{rank_down_rows}`",
        f"- Unchanged rows: `{unchanged_rows}`",
        f"- Decision rows: `{_safe_int(summary.get('decision_rows'), 0)}`",
        f"- Blocked decision rows: `{_safe_int(summary.get('blocked_rows'), 0)}`",
        f"- Approval reference: {approval_reference}",
        f"- Apply plan CSV byte size: `{plan_size}`",
        f"- Apply plan CSV SHA-256: `{plan_hash}`",
        "",
        "## Manual apply order",
        "",
    ]

    for _, row in plan.head(12).iterrows():
        rank_delta = _safe_int(row.get("rank_delta"), 0)
        lines.append(
            "- "
            f"Rank `{_safe_int(row.get('manual_apply_rank'), 0)}`: "
            f"`{_safe_text(row.get('pocket_id'), '-')}` "
            f"(current `{_safe_int(row.get('current_rank'), 0)}`, simulated `{_safe_int(row.get('simulated_rank'), 0)}`, "
            f"delta {rank_delta:+d}, score `{_safe_float(row.get('simulation_score'), 0.0):.3f}`) - "
            f"{_safe_text(row.get('apply_status'), '-')}"
        )
    if len(plan) > 12:
        lines.append(f"- ... {len(plan) - 12} additional rows omitted from this report; see the CSV for the full order.")
    lines.extend(
        [
            "",
            "## Required pre-apply checks",
            "",
            "- [ ] Confirm `release_allowed` is `yes` in this report and the release decision summary.",
            "- [ ] Compare the `consensus_rerank_release_apply_plan.csv` SHA-256 above against the exported file before applying.",
            "- [ ] Archive the release decision summary, validation table, handoff certificate, manifest, and ZIP hash together.",
            "- [ ] Apply this as a manual rank order only; do not enable automatic rerank from this report.",
            "- [ ] Re-run the full guardrail workflow if any evidence, residue mapping, or reviewer decision changes.",
            "",
            "## First-row required check",
            "",
            _safe_text(top.get("required_pre_apply_check"), "Archive the reviewed handoff package before changing active ranking."),
            "",
            "## Recommended action",
            "",
            _safe_text(top.get("recommended_action"), "Apply the approved manual rank order only after archival checks are complete."),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _release_apply_plan_sha256(consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame]) -> str:
    if consensus_rerank_release_apply_plan_df is None or getattr(consensus_rerank_release_apply_plan_df, "empty", True):
        return ""
    plan = consensus_rerank_release_apply_plan_df.copy()
    for column in CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS:
        if column not in plan.columns:
            plan[column] = ""
    if "manual_apply_rank" in plan.columns:
        plan["_rank_sort"] = pd.to_numeric(plan["manual_apply_rank"], errors="coerce").fillna(999)
        plan = plan.sort_values(["_rank_sort", "pocket_id"], ascending=[True, True]).drop(columns=["_rank_sort"]).reset_index(drop=True)
    _, plan_hash = _rerank_artifact_integrity(_rerank_csv_artifact_bytes(plan[CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS]))
    return plan_hash


def build_consensus_rerank_release_execution_template(
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if consensus_rerank_release_apply_plan_df is None or getattr(consensus_rerank_release_apply_plan_df, "empty", True):
        return _empty_consensus_rerank_release_execution_template_df()

    plan = consensus_rerank_release_apply_plan_df.copy()
    for column, default in {
        "manual_apply_rank": 999,
        "pocket_id": "",
        "current_rank": 0,
        "simulated_rank": 0,
        "rank_delta": 0,
        "apply_status": "",
        "apply_decision": "",
        "release_apply_status": "",
        "required_pre_apply_check": "",
        "approval_reference": "",
        "recommended_action": "",
    }.items():
        if column not in plan.columns:
            plan[column] = default

    plan["_rank_sort"] = pd.to_numeric(plan["manual_apply_rank"], errors="coerce").fillna(999)
    plan = plan.sort_values(["_rank_sort", "pocket_id"], ascending=[True, True]).drop(columns=["_rank_sort"]).reset_index(drop=True)
    plan_hash = _release_apply_plan_sha256(plan)

    rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        manual_rank = _safe_int(row.get("manual_apply_rank"), 0)
        pocket_id = _safe_text(row.get("pocket_id"), "Pocket")
        rows.append(
            {
                "execution_item_id": f"apply-rank-{manual_rank}",
                "manual_apply_rank": manual_rank,
                "pocket_id": pocket_id,
                "expected_current_rank": _safe_int(row.get("current_rank"), 0),
                "expected_simulated_rank": _safe_int(row.get("simulated_rank"), manual_rank),
                "expected_rank_delta": _safe_int(row.get("rank_delta"), 0),
                "expected_apply_status": _safe_text(row.get("apply_status"), "-"),
                "expected_apply_decision": _safe_text(row.get("apply_decision"), "-"),
                "expected_release_apply_status": _safe_text(row.get("release_apply_status"), "ready-for-manual-apply"),
                "plan_sha256": plan_hash,
                "execution_decision": "pending",
                "applied_rank": "",
                "operator": "",
                "executed_at": "",
                "execution_note": "",
                "required_pre_apply_check": _safe_text(
                    row.get("required_pre_apply_check"),
                    "Compare plan SHA-256, archive the handoff package, then record the applied rank.",
                ),
                "approval_reference": _safe_text(row.get("approval_reference"), "release decision summary"),
                "recommended_action": _safe_text(
                    row.get("recommended_action"),
                    "Record whether this row was applied exactly as approved.",
                ),
            }
        )

    if not rows:
        return _empty_consensus_rerank_release_execution_template_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS).reset_index(drop=True)


def _normalize_release_execution_decision(value: Any) -> str:
    text = _safe_text(value).lower()
    if not text:
        return "pending"
    if text in {"1", "true", "yes", "y", "apply", "applied", "execute", "executed", "done", "completed"}:
        return "applied"
    if text in {"0", "false", "no", "n", "skip", "skipped", "not_applied", "not-applied", "not applied"}:
        return "skipped"
    if text in {"fail", "failed", "error", "blocked"}:
        return "failed"
    if text in {"pending", "review", "hold", "todo", "to-do"}:
        return "pending"
    return text


def parse_consensus_rerank_release_execution_table(execution_text: str | bytes | None) -> tuple[pd.DataFrame, dict[str, str]]:
    if isinstance(execution_text, bytes):
        text = execution_text.decode("utf-8", errors="ignore")
    else:
        text = _safe_text(execution_text)
    if not text:
        return _empty_consensus_rerank_release_execution_template_df(), {
            "status": "empty",
            "input_rows": "0",
            "receipt_rows": "0",
            "skipped_rows": "0",
        }

    try:
        raw = pd.read_csv(StringIO(text), sep=None, engine="python")
    except Exception as exc:
        return _empty_consensus_rerank_release_execution_template_df(), {
            "status": "parse-error",
            "input_rows": "0",
            "receipt_rows": "0",
            "skipped_rows": "0",
            "message": str(exc),
        }
    if raw.empty:
        return _empty_consensus_rerank_release_execution_template_df(), {
            "status": "empty",
            "input_rows": "0",
            "receipt_rows": "0",
            "skipped_rows": "0",
        }

    columns = [str(column) for column in raw.columns]
    aliases = {
        "execution_item_id": {"execution_item_id", "execution_id", "item_id", "id"},
        "manual_apply_rank": {"manual_apply_rank", "manual_rank", "expected_rank", "rank"},
        "pocket_id": {"pocket_id", "pocket", "candidate_pocket"},
        "expected_current_rank": {"expected_current_rank", "current_rank"},
        "expected_simulated_rank": {"expected_simulated_rank", "simulated_rank", "expected_apply_rank"},
        "expected_rank_delta": {"expected_rank_delta", "rank_delta"},
        "expected_apply_status": {"expected_apply_status", "apply_status"},
        "expected_apply_decision": {"expected_apply_decision", "apply_decision"},
        "expected_release_apply_status": {"expected_release_apply_status", "release_apply_status"},
        "plan_sha256": {"plan_sha256", "apply_plan_sha256", "sha256", "plan_hash"},
        "execution_decision": {"execution_decision", "decision", "status", "execution_status"},
        "applied_rank": {"applied_rank", "actual_rank", "final_rank"},
        "operator": {"operator", "user", "executor", "applied_by"},
        "executed_at": {"executed_at", "execution_time", "applied_at", "timestamp"},
        "execution_note": {"execution_note", "note", "notes", "comment", "comments"},
        "required_pre_apply_check": {"required_pre_apply_check", "pre_apply_check", "check"},
        "approval_reference": {"approval_reference", "approval", "reference"},
        "recommended_action": {"recommended_action", "action", "next_action"},
    }
    selected = {column: _pick_release_column(columns, column_aliases) for column, column_aliases in aliases.items()}
    if not selected["execution_item_id"]:
        return _empty_consensus_rerank_release_execution_template_df(), {
            "status": "missing-required-columns",
            "input_rows": str(len(raw)),
            "receipt_rows": "0",
            "skipped_rows": str(len(raw)),
            "message": "execution_item_id column is required.",
        }

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in raw.iterrows():
        execution_item_id = _safe_text(row.get(selected["execution_item_id"]))
        if not execution_item_id:
            skipped += 1
            continue
        normalized: dict[str, Any] = {}
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS:
            source_column = selected.get(column, "")
            if column == "execution_item_id":
                normalized[column] = execution_item_id
            elif column == "execution_decision":
                normalized[column] = _normalize_release_execution_decision(row.get(source_column)) if source_column else "pending"
            else:
                normalized[column] = _safe_text(row.get(source_column)) if source_column else ""
        rows.append(normalized)

    if not rows:
        return _empty_consensus_rerank_release_execution_template_df(), {
            "status": "empty-after-normalization",
            "input_rows": str(len(raw)),
            "receipt_rows": "0",
            "skipped_rows": str(skipped),
        }

    receipts = pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS).reset_index(drop=True)
    decisions = receipts["execution_decision"].astype(str).str.lower()
    return receipts, {
        "status": "ok",
        "input_rows": str(len(raw)),
        "receipt_rows": str(len(receipts)),
        "skipped_rows": str(skipped),
        "applied_rows": str(int((decisions == "applied").sum())),
        "skipped_execution_rows": str(int((decisions == "skipped").sum())),
        "failed_rows": str(int((decisions == "failed").sum())),
        "pending_rows": str(int((decisions == "pending").sum())),
    }


def validate_consensus_rerank_release_execution_receipt(
    execution_receipt_df: Optional[pd.DataFrame],
    execution_template_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if execution_receipt_df is None or getattr(execution_receipt_df, "empty", True):
        return _empty_consensus_rerank_release_execution_validation_df()

    working = execution_receipt_df.copy()
    for column in CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS:
        if column not in working.columns:
            working[column] = ""

    template_map: dict[str, pd.Series] = {}
    if execution_template_df is not None and not getattr(execution_template_df, "empty", True):
        template = execution_template_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS:
            if column not in template.columns:
                template[column] = ""
        for _, template_row in template.iterrows():
            item_id = _safe_text(template_row.get("execution_item_id"))
            if item_id and item_id not in template_map:
                template_map[item_id] = template_row

    current_plan_hash = _release_apply_plan_sha256(consensus_rerank_release_apply_plan_df)
    rows: list[dict[str, Any]] = []
    for row_index, (_, receipt) in enumerate(working.iterrows(), start=1):
        item_id = _safe_text(receipt.get("execution_item_id"))
        template_row = template_map.get(item_id)
        template_match = bool(template_row is not None) if template_map else True

        def merged_text(column: str, default: str = "") -> str:
            uploaded = _safe_text(receipt.get(column))
            if uploaded:
                return uploaded
            if template_row is not None:
                return _safe_text(template_row.get(column), default)
            return default

        execution_decision = _normalize_release_execution_decision(receipt.get("execution_decision"))
        pocket_id = merged_text("pocket_id", "Pocket")
        expected_rank = _safe_int(merged_text("manual_apply_rank"), 0)
        applied_rank_text = _safe_text(receipt.get("applied_rank"))
        applied_rank = _safe_int(applied_rank_text, 0) if applied_rank_text else 0
        operator = _safe_text(receipt.get("operator"))
        executed_at = _safe_text(receipt.get("executed_at"))
        plan_hash = _safe_text(receipt.get("plan_sha256"), merged_text("plan_sha256"))
        expected_plan_hash = current_plan_hash or merged_text("plan_sha256")
        plan_hash_match = bool(plan_hash and expected_plan_hash and plan_hash == expected_plan_hash)

        flags: list[str] = []
        reasons: list[str] = []
        fixes: list[str] = []
        execution_accepted = False

        if not item_id:
            flags.append("missing-execution-item-id")
            reasons.append("Execution row has no execution_item_id.")
            fixes.append("Use an unmodified execution_item_id from the execution template.")
        if template_map and not template_match:
            flags.append("unmatched-template-item")
            reasons.append("Execution item does not exist in the current execution template.")
            fixes.append("Download the latest execution template and keep matching execution_item_id values.")
        if execution_decision not in {"applied", "skipped", "failed", "pending"}:
            flags.append("invalid-execution-decision")
            reasons.append("execution_decision must be applied, skipped, failed, or pending.")
            fixes.append("Normalize execution_decision before upload.")
        if expected_plan_hash and not plan_hash_match:
            flags.append("plan-hash-mismatch")
            reasons.append("Uploaded plan_sha256 does not match the current approved apply plan.")
            fixes.append("Confirm the operator executed the same apply plan CSV that was exported from this run.")

        if execution_decision == "applied":
            if applied_rank <= 0:
                flags.append("missing-applied-rank")
                reasons.append("Applied execution row lacks applied_rank.")
                fixes.append("Record the actual applied rank.")
            elif expected_rank > 0 and applied_rank != expected_rank:
                flags.append("rank-mismatch")
                reasons.append("Applied rank differs from the approved manual_apply_rank.")
                fixes.append("Correct the applied rank or mark the row skipped/failed with a note.")
            if not operator:
                flags.append("missing-operator")
                reasons.append("Applied execution row lacks operator.")
                fixes.append("Fill the operator column.")
            if not executed_at:
                flags.append("missing-executed-at")
                reasons.append("Applied execution row lacks executed_at.")
                fixes.append("Fill the executed_at timestamp.")
        elif execution_decision == "skipped":
            if not operator:
                flags.append("missing-operator")
                reasons.append("Skipped execution row lacks operator.")
                fixes.append("Fill the operator column.")
        elif execution_decision == "failed":
            if not operator:
                flags.append("missing-operator")
                reasons.append("Failed execution row lacks operator.")
                fixes.append("Fill the operator column.")

        blocking_flags = {
            "missing-execution-item-id",
            "unmatched-template-item",
            "invalid-execution-decision",
            "plan-hash-mismatch",
            "missing-applied-rank",
            "rank-mismatch",
            "missing-operator",
            "missing-executed-at",
        }
        if any(flag in blocking_flags for flag in flags):
            validation_status = "blocked"
        elif execution_decision == "applied":
            validation_status = "applied"
            execution_accepted = True
        elif execution_decision == "skipped":
            validation_status = "skipped"
        elif execution_decision == "failed":
            validation_status = "failed"
        else:
            validation_status = "pending"

        rows.append(
            {
                "row_index": row_index,
                "execution_item_id": item_id,
                "pocket_id": pocket_id,
                "execution_decision": execution_decision,
                "template_match": bool(template_match),
                "expected_rank": expected_rank,
                "applied_rank": applied_rank if applied_rank > 0 else "",
                "plan_hash_match": bool(plan_hash_match),
                "validation_status": validation_status,
                "issue_flags": ", ".join(dict.fromkeys(flags)) if flags else "none",
                "execution_accepted": bool(execution_accepted),
                "validation_reason": " ".join(reasons) if reasons else (
                    "Execution row was applied exactly as approved." if execution_accepted else "Execution row is not complete."
                ),
                "required_fix": " ".join(dict.fromkeys(fixes)) if fixes else "none",
                "operator": operator,
                "executed_at": executed_at,
                "plan_sha256": plan_hash,
            }
        )

    if not rows:
        return _empty_consensus_rerank_release_execution_validation_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_EXECUTION_VALIDATION_COLUMNS).reset_index(drop=True)


def build_consensus_rerank_release_execution_summary(
    execution_validation_df: Optional[pd.DataFrame],
    execution_receipt_df: Optional[pd.DataFrame] = None,
    execution_template_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if execution_validation_df is None or getattr(execution_validation_df, "empty", True):
        return _empty_consensus_rerank_release_execution_summary_df()

    statuses = execution_validation_df["validation_status"].astype(str).str.lower() if "validation_status" in execution_validation_df.columns else pd.Series(dtype=str)
    issue_flags = execution_validation_df["issue_flags"].astype(str).str.lower() if "issue_flags" in execution_validation_df.columns else pd.Series(dtype=str)
    template_rows = 0 if execution_template_df is None or getattr(execution_template_df, "empty", True) else int(len(execution_template_df))
    receipt_rows = 0 if execution_receipt_df is None or getattr(execution_receipt_df, "empty", True) else int(len(execution_receipt_df))
    matched_rows = int(execution_validation_df["template_match"].astype(bool).sum()) if "template_match" in execution_validation_df.columns else int(len(execution_validation_df))
    applied_rows = int((statuses == "applied").sum())
    skipped_rows = int((statuses == "skipped").sum())
    failed_rows = int((statuses == "failed").sum())
    pending_rows = int((statuses == "pending").sum())
    blocked_rows = int((statuses == "blocked").sum())
    rank_mismatch_rows = int(issue_flags.str.contains("rank-mismatch", regex=False).sum()) if not issue_flags.empty else 0
    missing_operator_rows = int(issue_flags.str.contains("missing-operator", regex=False).sum()) if not issue_flags.empty else 0
    missing_executed_at_rows = int(issue_flags.str.contains("missing-executed-at", regex=False).sum()) if not issue_flags.empty else 0
    plan_hash_mismatch_rows = int(issue_flags.str.contains("plan-hash-mismatch", regex=False).sum()) if not issue_flags.empty else 0

    missing_receipt_rows = 0
    if template_rows > 0 and execution_template_df is not None and "execution_item_id" in execution_template_df.columns:
        expected_ids = {_safe_text(value) for value in execution_template_df["execution_item_id"].tolist() if _safe_text(value)}
        received_ids = set()
        if "execution_item_id" in execution_validation_df.columns:
            received_ids = {_safe_text(value) for value in execution_validation_df["execution_item_id"].tolist() if _safe_text(value)}
        missing_receipt_rows = int(len(expected_ids - received_ids))

    execution_complete = bool(
        template_rows > 0
        and applied_rows == template_rows
        and blocked_rows == 0
        and failed_rows == 0
        and skipped_rows == 0
        and pending_rows == 0
        and missing_receipt_rows == 0
    )

    if blocked_rows > 0 or missing_receipt_rows > 0:
        execution_review_status = "blocked"
        recommended_action = "Fix blocked, mismatched, or missing execution receipt rows before treating the manual rerank as executed."
    elif failed_rows > 0:
        execution_review_status = "failed"
        recommended_action = "Do not mark rerank as complete; at least one execution row failed."
    elif skipped_rows > 0:
        execution_review_status = "partial"
        recommended_action = "Review skipped rows before claiming the full approved plan was applied."
    elif pending_rows > 0:
        execution_review_status = "pending"
        recommended_action = "Complete all pending execution receipt rows."
    elif execution_complete:
        execution_review_status = "executed"
        recommended_action = "Manual rerank execution is complete; archive the receipt with the approved apply plan and handoff package."
    else:
        execution_review_status = "pending"
        recommended_action = "Upload a completed execution receipt after applying the approved manual rank order."

    return pd.DataFrame(
        [
            {
                "execution_review_status": execution_review_status,
                "template_rows": template_rows,
                "receipt_rows": receipt_rows,
                "matched_rows": matched_rows,
                "applied_rows": applied_rows,
                "skipped_rows": skipped_rows,
                "failed_rows": failed_rows,
                "pending_rows": pending_rows,
                "blocked_rows": blocked_rows,
                "rank_mismatch_rows": rank_mismatch_rows,
                "missing_operator_rows": missing_operator_rows,
                "missing_executed_at_rows": missing_executed_at_rows,
                "plan_hash_mismatch_rows": plan_hash_mismatch_rows,
                "missing_receipt_rows": missing_receipt_rows,
                "execution_complete": execution_complete,
                "recommended_action": recommended_action,
            }
        ],
        columns=CONSENSUS_RERANK_RELEASE_EXECUTION_SUMMARY_COLUMNS,
    )


def build_consensus_rerank_release_execution_report_markdown(
    execution_summary_df: Optional[pd.DataFrame],
    execution_validation_df: Optional[pd.DataFrame] = None,
    execution_receipt_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Consensus rerank release execution report",
) -> str:
    if execution_summary_df is None or getattr(execution_summary_df, "empty", True):
        return ""

    summary = execution_summary_df.iloc[0]
    execution_status = _safe_text(summary.get("execution_review_status"), "unknown")
    execution_complete = _safe_bool(summary.get("execution_complete"))
    receipt_rows = _safe_int(summary.get("receipt_rows"), 0)
    applied_rows = _safe_int(summary.get("applied_rows"), 0)
    blocked_rows = _safe_int(summary.get("blocked_rows"), 0)
    failed_rows = _safe_int(summary.get("failed_rows"), 0)
    skipped_rows = _safe_int(summary.get("skipped_rows"), 0)
    pending_rows = _safe_int(summary.get("pending_rows"), 0)
    recommended_action = _safe_text(summary.get("recommended_action"), "Review execution receipt before treating rerank as complete.")

    receipt_hash = ""
    receipt_size = 0
    if execution_receipt_df is not None and not getattr(execution_receipt_df, "empty", True):
        receipt = execution_receipt_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS:
            if column not in receipt.columns:
                receipt[column] = ""
        receipt_size, receipt_hash = _rerank_artifact_integrity(_rerank_csv_artifact_bytes(receipt[CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS]))

    operator_text = "none"
    if execution_validation_df is not None and not getattr(execution_validation_df, "empty", True) and "operator" in execution_validation_df.columns:
        operators = sorted({_safe_text(value) for value in execution_validation_df["operator"].tolist() if _safe_text(value)})
        if operators:
            operator_text = ", ".join(operators[:8])
            if len(operators) > 8:
                operator_text += f", +{len(operators) - 8} more"

    lines = [
        f"# {title}",
        "",
        "This report records whether the approved manual consensus rerank was actually executed.",
        "It is generated from an uploaded execution receipt and should be archived with the apply plan, receipt CSV, and handoff ZIP.",
        "",
        "## Execution summary",
        "",
        f"- Execution status: `{execution_status}`",
        f"- Execution complete: {'yes' if execution_complete else 'no'}",
        f"- Template rows: `{_safe_int(summary.get('template_rows'), 0)}`",
        f"- Receipt rows: `{receipt_rows}`",
        f"- Applied rows: `{applied_rows}`",
        f"- Skipped rows: `{skipped_rows}`",
        f"- Failed rows: `{failed_rows}`",
        f"- Pending rows: `{pending_rows}`",
        f"- Blocked rows: `{blocked_rows}`",
        f"- Missing receipt rows: `{_safe_int(summary.get('missing_receipt_rows'), 0)}`",
        f"- Rank mismatch rows: `{_safe_int(summary.get('rank_mismatch_rows'), 0)}`",
        f"- Plan hash mismatch rows: `{_safe_int(summary.get('plan_hash_mismatch_rows'), 0)}`",
        f"- Missing operator rows: `{_safe_int(summary.get('missing_operator_rows'), 0)}`",
        f"- Missing executed_at rows: `{_safe_int(summary.get('missing_executed_at_rows'), 0)}`",
        f"- Operators: {operator_text}",
    ]
    if receipt_hash:
        lines.extend(
            [
                f"- Execution receipt CSV byte size: `{receipt_size}`",
                f"- Execution receipt CSV SHA-256: `{receipt_hash}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommended action",
            "",
            recommended_action,
            "",
        ]
    )

    if execution_validation_df is not None and not getattr(execution_validation_df, "empty", True):
        validation = execution_validation_df.copy()
        for column, default in {
            "row_index": 0,
            "execution_item_id": "",
            "pocket_id": "",
            "execution_decision": "",
            "validation_status": "",
            "issue_flags": "none",
            "operator": "",
            "executed_at": "",
            "applied_rank": "",
            "expected_rank": "",
        }.items():
            if column not in validation.columns:
                validation[column] = default
        validation["_status_sort"] = validation["validation_status"].astype(str).map(
            {"blocked": 0, "failed": 1, "pending": 2, "skipped": 3, "applied": 4}
        ).fillna(5)
        validation["_row_sort"] = pd.to_numeric(validation["row_index"], errors="coerce").fillna(9999)
        validation = validation.sort_values(["_status_sort", "_row_sort"], ascending=[True, True]).head(12)
        lines.extend(["## Execution rows", ""])
        for _, row in validation.iterrows():
            lines.append(
                "- "
                f"`{_safe_text(row.get('execution_item_id'), '-')}` / `{_safe_text(row.get('pocket_id'), '-')}`: "
                f"{_safe_text(row.get('validation_status'), '-')} "
                f"(decision `{_safe_text(row.get('execution_decision'), '-')}`, "
                f"expected `{_safe_text(row.get('expected_rank'), '-')}`, applied `{_safe_text(row.get('applied_rank'), '-')}`, "
                f"operator `{_safe_text(row.get('operator'), '-')}`) - "
                f"{_safe_text(row.get('issue_flags'), 'none')}"
            )
        if len(execution_validation_df) > 12:
            lines.append(f"- ... {len(execution_validation_df) - 12} additional rows omitted; see validation CSV for all rows.")
        lines.append("")

    lines.extend(
        [
            "## Archival checklist",
            "",
            "- [ ] Archive this report with `consensus_rerank_release_execution_receipt_normalized.csv`.",
            "- [ ] Compare the receipt SHA-256 in this report against the exported receipt CSV.",
            "- [ ] Keep the approved apply plan, execution summary, handoff certificate, manifest, and ZIP together.",
            "- [ ] If execution status is not `executed`, do not claim the manual rerank was fully applied.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_consensus_rerank_release_closure_certificate_markdown(
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame],
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_receipt_df: Optional[pd.DataFrame] = None,
    execution_report_markdown: str = "",
    *,
    title: str = "Consensus rerank release closure certificate",
) -> str:
    if consensus_rerank_release_execution_summary_df is None or getattr(consensus_rerank_release_execution_summary_df, "empty", True):
        return ""

    execution_summary = consensus_rerank_release_execution_summary_df.iloc[0]
    release_summary = (
        consensus_rerank_release_decision_summary_df.iloc[0]
        if consensus_rerank_release_decision_summary_df is not None and not getattr(consensus_rerank_release_decision_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )

    plan_rows = 0
    plan_size = 0
    plan_hash = ""
    if consensus_rerank_release_apply_plan_df is not None and not getattr(consensus_rerank_release_apply_plan_df, "empty", True):
        plan = consensus_rerank_release_apply_plan_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS:
            if column not in plan.columns:
                plan[column] = ""
        if "manual_apply_rank" in plan.columns:
            plan["_rank_sort"] = pd.to_numeric(plan["manual_apply_rank"], errors="coerce").fillna(999)
            plan = plan.sort_values(["_rank_sort", "pocket_id"], ascending=[True, True]).drop(columns=["_rank_sort"]).reset_index(drop=True)
        plan_rows = int(len(plan))
        plan_size, plan_hash = _rerank_artifact_integrity(_rerank_csv_artifact_bytes(plan[CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS]))

    release_review_status = _safe_text(release_summary.get("release_review_status"), "not-recorded")
    release_allowed = _safe_bool(release_summary.get("release_allowed"))
    if not release_allowed and consensus_rerank_release_apply_plan_df is not None and not getattr(consensus_rerank_release_apply_plan_df, "empty", True):
        if "release_allowed" in consensus_rerank_release_apply_plan_df.columns:
            release_allowed = any(_safe_bool(value) for value in consensus_rerank_release_apply_plan_df["release_allowed"].tolist())

    execution_review_status = _safe_text(execution_summary.get("execution_review_status"), "unknown")
    execution_complete = _safe_bool(execution_summary.get("execution_complete"))
    recommended_action = _safe_text(
        execution_summary.get("recommended_action"),
        "Review execution receipt before treating the consensus rerank release as closed.",
    )

    receipt_rows = _safe_int(execution_summary.get("receipt_rows"), 0)
    receipt_size = 0
    receipt_hash = ""
    operators = "none"
    if consensus_rerank_release_execution_receipt_df is not None and not getattr(consensus_rerank_release_execution_receipt_df, "empty", True):
        receipt = consensus_rerank_release_execution_receipt_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS:
            if column not in receipt.columns:
                receipt[column] = ""
        receipt_size, receipt_hash = _rerank_artifact_integrity(
            _rerank_csv_artifact_bytes(receipt[CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS])
        )
        if "operator" in receipt.columns:
            operator_values = sorted({_safe_text(value) for value in receipt["operator"].tolist() if _safe_text(value)})
            if operator_values:
                operators = ", ".join(operator_values[:8])
                if len(operator_values) > 8:
                    operators += f", +{len(operator_values) - 8} more"

    report_text = "" if execution_report_markdown is None else str(execution_report_markdown)
    report_size = 0
    report_hash = ""
    if _safe_text(report_text):
        report_size, report_hash = _rerank_artifact_integrity(report_text.encode("utf-8"))

    if release_allowed and execution_complete:
        closure_status = "closed-executed"
        closure_action = "Archive this closure certificate with the manifest, ZIP, approved apply plan, execution receipt, and execution report."
    else:
        closure_status = "not-closed"
        if not release_allowed:
            closure_action = "Do not close release: reviewer approval is not recorded as allowed."
        else:
            closure_action = recommended_action

    lines = [
        f"# {title}",
        "",
        "This certificate records final closure state for a reviewer-approved manual consensus rerank release.",
        "It ties the approved apply plan, release review, uploaded execution receipt, and execution report into one auditable page.",
        "",
        "## Closure decision",
        "",
        f"- Closure status: `{closure_status}`",
        f"- Release review status: `{release_review_status}`",
        f"- Release allowed: {'yes' if release_allowed else 'no'}",
        f"- Execution status: `{execution_review_status}`",
        f"- Execution complete: {'yes' if execution_complete else 'no'}",
        f"- Recommended action: {closure_action}",
        "",
        "## Approved apply plan identity",
        "",
        f"- Apply plan rows: `{plan_rows}`",
    ]
    if plan_hash:
        lines.extend(
            [
                f"- Apply plan CSV byte size: `{plan_size}`",
                f"- Apply plan SHA-256: `{plan_hash}`",
            ]
        )
    else:
        lines.append("- Apply plan SHA-256: `not-available`")

    lines.extend(
        [
            "",
            "## Release review evidence",
            "",
            f"- Decision rows: `{_safe_int(release_summary.get('decision_rows'), 0)}`",
            f"- Blocked decision rows: `{_safe_int(release_summary.get('blocked_rows'), 0)}`",
            f"- Missing reviewer rows: `{_safe_int(release_summary.get('missing_reviewer_rows'), 0)}`",
            f"- Missing evidence rows: `{_safe_int(release_summary.get('missing_evidence_rows'), 0)}`",
            f"- Unresolved blocker rows: `{_safe_int(release_summary.get('unresolved_blocker_rows'), 0)}`",
            "",
            "## Execution evidence",
            "",
            f"- Template rows: `{_safe_int(execution_summary.get('template_rows'), 0)}`",
            f"- Receipt rows: `{receipt_rows}`",
            f"- Applied rows: `{_safe_int(execution_summary.get('applied_rows'), 0)}`",
            f"- Skipped rows: `{_safe_int(execution_summary.get('skipped_rows'), 0)}`",
            f"- Failed rows: `{_safe_int(execution_summary.get('failed_rows'), 0)}`",
            f"- Pending rows: `{_safe_int(execution_summary.get('pending_rows'), 0)}`",
            f"- Blocked rows: `{_safe_int(execution_summary.get('blocked_rows'), 0)}`",
            f"- Missing receipt rows: `{_safe_int(execution_summary.get('missing_receipt_rows'), 0)}`",
            f"- Rank mismatch rows: `{_safe_int(execution_summary.get('rank_mismatch_rows'), 0)}`",
            f"- Plan hash mismatch rows: `{_safe_int(execution_summary.get('plan_hash_mismatch_rows'), 0)}`",
            f"- Operators: {operators}",
        ]
    )
    if receipt_hash:
        lines.extend(
            [
                f"- Execution receipt CSV byte size: `{receipt_size}`",
                f"- Execution receipt CSV SHA-256: `{receipt_hash}`",
            ]
        )
    else:
        lines.append("- Execution receipt CSV SHA-256: `not-available`")
    if report_hash:
        lines.extend(
            [
                f"- Execution report byte size: `{report_size}`",
                f"- Execution report SHA-256: `{report_hash}`",
            ]
        )
    else:
        lines.append("- Execution report SHA-256: `not-available`")

    lines.extend(
        [
            "",
            "## Closure checklist",
            "",
            "- [ ] Compare the apply plan SHA-256 above with `consensus_rerank_release_apply_plan.csv`.",
            "- [ ] Compare the execution receipt SHA-256 above with `consensus_rerank_release_execution_receipt_normalized.csv`.",
            "- [ ] Compare the execution report SHA-256 above with `consensus_rerank_release_execution_report.md`.",
            "- [ ] Keep this certificate with the verified handoff ZIP and artifact manifest.",
            "- [ ] Treat the release as closed only when closure status is `closed-executed`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_consensus_rerank_release_closure_ledger(
    consensus_rerank_release_apply_plan_df: Optional[pd.DataFrame],
    consensus_rerank_release_decision_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_receipt_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_validation_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_execution_report_markdown: str = "",
    consensus_rerank_release_closure_certificate_markdown: str = "",
) -> pd.DataFrame:
    if (
        (consensus_rerank_release_execution_summary_df is None or getattr(consensus_rerank_release_execution_summary_df, "empty", True))
        and not _safe_text(consensus_rerank_release_closure_certificate_markdown)
    ):
        return _empty_consensus_rerank_release_closure_ledger_df()

    release_summary = (
        consensus_rerank_release_decision_summary_df.iloc[0]
        if consensus_rerank_release_decision_summary_df is not None and not getattr(consensus_rerank_release_decision_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    execution_summary = (
        consensus_rerank_release_execution_summary_df.iloc[0]
        if consensus_rerank_release_execution_summary_df is not None and not getattr(consensus_rerank_release_execution_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    release_allowed = _safe_bool(release_summary.get("release_allowed"))
    execution_complete = _safe_bool(execution_summary.get("execution_complete"))
    closure_status = "closed-executed" if release_allowed and execution_complete else "not-closed"
    execution_review_status = _safe_text(execution_summary.get("execution_review_status"), "not-uploaded")
    release_review_status = _safe_text(release_summary.get("release_review_status"), "not-recorded")
    rows: list[dict[str, Any]] = []

    def add_row(
        evidence_item: str,
        file_name: str,
        artifact_type: str,
        row_count: int,
        data: bytes,
        status: str,
        closure_check: str,
        issue: str,
        recommended_action: str,
        *,
        required_for_closure: bool = True,
    ) -> None:
        byte_size = 0
        digest = ""
        if data:
            byte_size, digest = _rerank_artifact_integrity(data)
        rows.append(
            {
                "evidence_item": evidence_item,
                "file_name": file_name,
                "artifact_type": artifact_type,
                "row_count": int(row_count),
                "byte_size": int(byte_size),
                "sha256": digest,
                "status": status,
                "required_for_closure": bool(required_for_closure),
                "closure_check": closure_check,
                "issue": issue,
                "recommended_action": recommended_action,
            }
        )

    plan_data = b""
    plan_rows = 0
    if consensus_rerank_release_apply_plan_df is not None and not getattr(consensus_rerank_release_apply_plan_df, "empty", True):
        plan = consensus_rerank_release_apply_plan_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS:
            if column not in plan.columns:
                plan[column] = ""
        if "manual_apply_rank" in plan.columns:
            plan["_rank_sort"] = pd.to_numeric(plan["manual_apply_rank"], errors="coerce").fillna(999)
            plan = plan.sort_values(["_rank_sort", "pocket_id"], ascending=[True, True]).drop(columns=["_rank_sort"]).reset_index(drop=True)
        plan_rows = int(len(plan))
        plan_data = _rerank_csv_artifact_bytes(plan[CONSENSUS_RERANK_RELEASE_APPLY_PLAN_COLUMNS])
    add_row(
        "approved apply plan",
        "consensus_rerank_release_apply_plan.csv",
        "csv",
        plan_rows,
        plan_data,
        "ready-for-manual-apply" if plan_rows > 0 else "missing",
        "ok" if plan_rows > 0 and plan_data else "missing",
        "none" if plan_rows > 0 and plan_data else "Approved apply plan is missing.",
        "Regenerate the approved apply plan from a clean simulation and approved release decision.",
    )

    release_summary_data = b""
    release_summary_rows = 0
    if consensus_rerank_release_decision_summary_df is not None and not getattr(consensus_rerank_release_decision_summary_df, "empty", True):
        table = consensus_rerank_release_decision_summary_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_DECISION_SUMMARY_COLUMNS:
            if column not in table.columns:
                table[column] = ""
        release_summary_rows = int(len(table))
        release_summary_data = _rerank_csv_artifact_bytes(table[CONSENSUS_RERANK_RELEASE_DECISION_SUMMARY_COLUMNS])
    add_row(
        "release decision summary",
        "consensus_rerank_release_decision_summary.csv",
        "csv",
        release_summary_rows,
        release_summary_data,
        release_review_status,
        "ok" if release_allowed else "blocked",
        "none" if release_allowed else "Release decision summary does not allow manual release.",
        "Upload or fix the reviewer decision file until release_allowed is true.",
    )

    receipt_data = b""
    receipt_rows = 0
    if consensus_rerank_release_execution_receipt_df is not None and not getattr(consensus_rerank_release_execution_receipt_df, "empty", True):
        receipt = consensus_rerank_release_execution_receipt_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS:
            if column not in receipt.columns:
                receipt[column] = ""
        receipt_rows = int(len(receipt))
        receipt_data = _rerank_csv_artifact_bytes(receipt[CONSENSUS_RERANK_RELEASE_EXECUTION_TEMPLATE_COLUMNS])
    add_row(
        "execution receipt",
        "consensus_rerank_release_execution_receipt_normalized.csv",
        "csv",
        receipt_rows,
        receipt_data,
        execution_review_status,
        "ok" if receipt_rows > 0 and receipt_data else "missing",
        "none" if receipt_rows > 0 and receipt_data else "Execution receipt is missing.",
        "Upload a completed execution receipt after applying the approved manual rank order.",
    )

    validation_data = b""
    validation_rows = 0
    blocked_rows = _safe_int(execution_summary.get("blocked_rows"), 0)
    if consensus_rerank_release_execution_validation_df is not None and not getattr(consensus_rerank_release_execution_validation_df, "empty", True):
        validation = consensus_rerank_release_execution_validation_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_VALIDATION_COLUMNS:
            if column not in validation.columns:
                validation[column] = ""
        validation_rows = int(len(validation))
        validation_data = _rerank_csv_artifact_bytes(validation[CONSENSUS_RERANK_RELEASE_EXECUTION_VALIDATION_COLUMNS])
    add_row(
        "execution validation",
        "consensus_rerank_release_execution_validation.csv",
        "csv",
        validation_rows,
        validation_data,
        "validated" if validation_rows > 0 and blocked_rows == 0 else "blocked" if blocked_rows > 0 else "missing",
        "ok" if validation_rows > 0 and blocked_rows == 0 else "blocked" if blocked_rows > 0 else "missing",
        "none" if validation_rows > 0 and blocked_rows == 0 else "Execution validation has blocked rows or is missing.",
        "Fix blocked execution receipt rows before closure.",
    )

    execution_summary_data = b""
    execution_summary_rows = 0
    if consensus_rerank_release_execution_summary_df is not None and not getattr(consensus_rerank_release_execution_summary_df, "empty", True):
        summary_table = consensus_rerank_release_execution_summary_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_EXECUTION_SUMMARY_COLUMNS:
            if column not in summary_table.columns:
                summary_table[column] = ""
        execution_summary_rows = int(len(summary_table))
        execution_summary_data = _rerank_csv_artifact_bytes(summary_table[CONSENSUS_RERANK_RELEASE_EXECUTION_SUMMARY_COLUMNS])
    add_row(
        "execution summary",
        "consensus_rerank_release_execution_summary.csv",
        "csv",
        execution_summary_rows,
        execution_summary_data,
        execution_review_status,
        "ok" if execution_complete else "blocked",
        "none" if execution_complete else "Execution summary is not complete.",
        "Complete all execution rows exactly as approved before closure.",
    )

    report_text = "" if consensus_rerank_release_execution_report_markdown is None else str(consensus_rerank_release_execution_report_markdown)
    report_data = report_text.encode("utf-8") if _safe_text(report_text) else b""
    add_row(
        "execution report",
        "consensus_rerank_release_execution_report.md",
        "markdown",
        len([line for line in report_text.splitlines() if line.strip()]),
        report_data,
        execution_review_status if report_data else "missing",
        "ok" if report_data else "missing",
        "none" if report_data else "Execution report is missing.",
        "Regenerate the execution report from the uploaded receipt and validation table.",
    )

    certificate_text = "" if consensus_rerank_release_closure_certificate_markdown is None else str(consensus_rerank_release_closure_certificate_markdown)
    certificate_data = certificate_text.encode("utf-8") if _safe_text(certificate_text) else b""
    add_row(
        "closure certificate",
        "consensus_rerank_release_closure_certificate.md",
        "markdown",
        len([line for line in certificate_text.splitlines() if line.strip()]),
        certificate_data,
        closure_status if certificate_data else "missing",
        "ok" if closure_status == "closed-executed" and certificate_data else "blocked" if certificate_data else "missing",
        "none" if closure_status == "closed-executed" and certificate_data else "Closure certificate is present but release is not closed." if certificate_data else "Closure certificate is missing.",
        "Archive the closure certificate only after release approval and execution completion are both true.",
    )

    if not rows:
        return _empty_consensus_rerank_release_closure_ledger_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_CLOSURE_LEDGER_COLUMNS).reset_index(drop=True)


def build_consensus_rerank_release_closure_summary(
    consensus_rerank_release_closure_ledger_df: Optional[pd.DataFrame],
    consensus_rerank_guardrail_bundle_verification_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if consensus_rerank_release_closure_ledger_df is None or getattr(consensus_rerank_release_closure_ledger_df, "empty", True):
        return _empty_consensus_rerank_release_closure_summary_df()

    ledger = consensus_rerank_release_closure_ledger_df.copy()
    for column, default in {
        "evidence_item": "",
        "file_name": "",
        "sha256": "",
        "required_for_closure": True,
        "closure_check": "",
    }.items():
        if column not in ledger.columns:
            ledger[column] = default

    required_mask = ledger["required_for_closure"].map(_safe_bool)
    required = ledger[required_mask].copy()
    checks = required["closure_check"].astype(str).str.lower()
    hashes = required["sha256"].astype(str).str.strip()
    ledger_rows = int(len(ledger))
    required_rows = int(len(required))
    ok_rows = int((checks == "ok").sum())
    blocked_rows = int((checks == "blocked").sum())
    missing_rows = int((checks == "missing").sum())
    missing_hash_rows = int((hashes.str.len() != 64).sum())

    verification_summary = (
        consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0]
        if consensus_rerank_guardrail_bundle_verification_summary_df is not None
        and not getattr(consensus_rerank_guardrail_bundle_verification_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    manifest_rows = _safe_int(verification_summary.get("manifest_rows"), 0)
    bundle_status = _safe_text(verification_summary.get("verification_status"), "not-verified")
    bundle_failed_files = _safe_int(verification_summary.get("failed_files"), 0)
    bundle_verified = bool(bundle_status == "verified" and bundle_failed_files == 0 and manifest_rows > 0)

    ledger_ready = bool(required_rows > 0 and ok_rows == required_rows and blocked_rows == 0 and missing_rows == 0 and missing_hash_rows == 0)
    release_closed = bool(ledger_ready and bundle_verified)

    if release_closed:
        closure_readiness_status = "closed-and-verified"
        recommended_action = "Release closure evidence is complete and the handoff ZIP matches its manifest; archive the detached summary with the certificate."
    elif not ledger_ready:
        closure_readiness_status = "ledger-blocked"
        recommended_action = "Fix missing, blocked, or unhashed closure ledger evidence before treating the release as closed."
    elif not bundle_verified:
        closure_readiness_status = "package-verification-blocked"
        recommended_action = "Regenerate or verify the handoff ZIP and manifest before treating the release as closed."
    else:
        closure_readiness_status = "not-closed"
        recommended_action = "Review closure ledger and bundle verification before closing the release."

    return pd.DataFrame(
        [
            {
                "closure_readiness_status": closure_readiness_status,
                "ledger_rows": ledger_rows,
                "required_rows": required_rows,
                "ok_rows": ok_rows,
                "blocked_rows": blocked_rows,
                "missing_rows": missing_rows,
                "missing_hash_rows": missing_hash_rows,
                "manifest_rows": manifest_rows,
                "bundle_verification_status": bundle_status,
                "bundle_failed_files": bundle_failed_files,
                "bundle_verified": bundle_verified,
                "release_closed": release_closed,
                "recommended_action": recommended_action,
            }
        ],
        columns=CONSENSUS_RERANK_RELEASE_CLOSURE_SUMMARY_COLUMNS,
    )


def build_consensus_rerank_release_closure_blocker_queue(
    consensus_rerank_release_closure_summary_df: Optional[pd.DataFrame],
    consensus_rerank_release_closure_ledger_df: Optional[pd.DataFrame] = None,
    consensus_rerank_guardrail_bundle_verification_df: Optional[pd.DataFrame] = None,
    consensus_rerank_guardrail_bundle_verification_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if consensus_rerank_release_closure_summary_df is None or getattr(consensus_rerank_release_closure_summary_df, "empty", True):
        return _empty_consensus_rerank_release_closure_blocker_df()

    summary = consensus_rerank_release_closure_summary_df.iloc[0]
    if _safe_bool(summary.get("release_closed")):
        return _empty_consensus_rerank_release_closure_blocker_df()

    rows: list[dict[str, Any]] = []

    def add_blocker(
        blocker_source: str,
        evidence_item: str,
        file_name: str,
        blocker_type: str,
        severity: str,
        current_status: str,
        issue: str,
        required_fix: str,
        recommended_action: str,
    ) -> None:
        rows.append(
            {
                "blocker_rank": 0,
                "blocker_source": blocker_source,
                "evidence_item": evidence_item,
                "file_name": file_name,
                "blocker_type": blocker_type,
                "severity": severity,
                "current_status": current_status,
                "issue": issue,
                "required_fix": required_fix,
                "recommended_action": recommended_action,
            }
        )

    if consensus_rerank_release_closure_ledger_df is not None and not getattr(consensus_rerank_release_closure_ledger_df, "empty", True):
        ledger = consensus_rerank_release_closure_ledger_df.copy()
        for column, default in {
            "evidence_item": "",
            "file_name": "",
            "sha256": "",
            "status": "",
            "required_for_closure": True,
            "closure_check": "",
            "issue": "",
            "recommended_action": "",
        }.items():
            if column not in ledger.columns:
                ledger[column] = default
        required = ledger[ledger["required_for_closure"].map(_safe_bool)].copy()
        for _, row in required.iterrows():
            closure_check = _safe_text(row.get("closure_check"), "missing").lower()
            sha256 = _safe_text(row.get("sha256"))
            if closure_check == "ok" and len(sha256) == 64:
                continue
            if closure_check == "missing":
                blocker_type = "missing-evidence"
                severity = "critical"
                required_fix = "Regenerate or upload the missing closure artifact, then rebuild the closure ledger."
            elif closure_check == "blocked":
                blocker_type = "blocked-evidence"
                severity = "critical"
                required_fix = "Resolve the blocked release or execution evidence before rerunning closure readiness."
            elif len(sha256) != 64:
                blocker_type = "missing-artifact-hash"
                severity = "high"
                required_fix = "Regenerate the artifact from the current table or Markdown so the ledger records a 64-character SHA-256."
            else:
                blocker_type = "ledger-not-ok"
                severity = "high"
                required_fix = "Review and fix this ledger row before treating the release as closed."
            add_blocker(
                "closure-ledger",
                _safe_text(row.get("evidence_item"), "-"),
                _safe_text(row.get("file_name"), "-"),
                blocker_type,
                severity,
                _safe_text(row.get("status"), closure_check),
                _safe_text(row.get("issue"), "Closure ledger row is not ok."),
                required_fix,
                _safe_text(row.get("recommended_action"), "Fix this closure ledger row."),
            )

    verification_summary = (
        consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0]
        if consensus_rerank_guardrail_bundle_verification_summary_df is not None
        and not getattr(consensus_rerank_guardrail_bundle_verification_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    bundle_status = _safe_text(
        verification_summary.get("verification_status"),
        _safe_text(summary.get("bundle_verification_status"), "not-verified"),
    )
    bundle_failed_files = _safe_int(verification_summary.get("failed_files"), _safe_int(summary.get("bundle_failed_files"), 0))
    bundle_verified = bool(bundle_status == "verified" and bundle_failed_files == 0)
    if not bundle_verified:
        if consensus_rerank_guardrail_bundle_verification_df is not None and not getattr(consensus_rerank_guardrail_bundle_verification_df, "empty", True):
            verification = consensus_rerank_guardrail_bundle_verification_df.copy()
            for column, default in {
                "file_name": "",
                "verification_status": "",
                "issue": "",
                "recommended_action": "",
            }.items():
                if column not in verification.columns:
                    verification[column] = default
            failed = verification[verification["verification_status"].astype(str).str.lower() != "verified"].copy()
            if failed.empty:
                failed = verification.head(1).copy()
            for _, row in failed.iterrows():
                status = _safe_text(row.get("verification_status"), bundle_status)
                add_blocker(
                    "handoff-zip-verification",
                    "handoff ZIP artifact",
                    _safe_text(row.get("file_name"), "consensus_rerank_guardrail_handoff.zip"),
                    "package-verification-failed",
                    "critical",
                    status,
                    _safe_text(row.get("issue"), "Handoff ZIP verification is not clean."),
                    "Regenerate the manifest and handoff ZIP together, then rerun verification.",
                    _safe_text(row.get("recommended_action"), "Regenerate and verify the handoff package before closure."),
                )
        else:
            add_blocker(
                "handoff-zip-verification",
                "handoff ZIP artifact",
                "consensus_rerank_guardrail_handoff.zip",
                "package-verification-missing",
                "critical",
                bundle_status,
                "Handoff ZIP verification summary is missing or not verified.",
                "Run handoff ZIP verification and require zero failed files before closure.",
                "Verify the handoff ZIP against the current artifact manifest.",
            )

    if not rows:
        return _empty_consensus_rerank_release_closure_blocker_df()

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    table = pd.DataFrame(rows, columns=CONSENSUS_RERANK_RELEASE_CLOSURE_BLOCKER_COLUMNS)
    table["_severity_sort"] = table["severity"].astype(str).map(severity_order).fillna(9)
    table = table.sort_values(["_severity_sort", "blocker_source", "file_name"], ascending=[True, True, True]).drop(columns=["_severity_sort"]).reset_index(drop=True)
    table["blocker_rank"] = range(1, len(table) + 1)
    return table[CONSENSUS_RERANK_RELEASE_CLOSURE_BLOCKER_COLUMNS].reset_index(drop=True)


def build_consensus_rerank_release_closure_remediation_checklist_markdown(
    consensus_rerank_release_closure_blocker_df: Optional[pd.DataFrame],
    consensus_rerank_release_closure_summary_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Consensus rerank release closure remediation checklist",
) -> str:
    if consensus_rerank_release_closure_blocker_df is None or getattr(consensus_rerank_release_closure_blocker_df, "empty", True):
        return ""

    blockers = consensus_rerank_release_closure_blocker_df.copy()
    for column, default in {
        "blocker_rank": 999,
        "blocker_source": "",
        "evidence_item": "",
        "file_name": "",
        "blocker_type": "",
        "severity": "",
        "current_status": "",
        "issue": "",
        "required_fix": "",
        "recommended_action": "",
    }.items():
        if column not in blockers.columns:
            blockers[column] = default
    blockers["_rank_sort"] = pd.to_numeric(blockers["blocker_rank"], errors="coerce").fillna(999)
    blockers = blockers.sort_values(["_rank_sort", "severity", "file_name"], ascending=[True, True, True]).drop(columns=["_rank_sort"]).reset_index(drop=True)

    summary = (
        consensus_rerank_release_closure_summary_df.iloc[0]
        if consensus_rerank_release_closure_summary_df is not None and not getattr(consensus_rerank_release_closure_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    severity = blockers["severity"].astype(str).str.lower()
    critical_rows = int((severity == "critical").sum())
    high_rows = int((severity == "high").sum())
    readiness_status = _safe_text(summary.get("closure_readiness_status"), "not-closed")
    release_closed = _safe_bool(summary.get("release_closed"))
    recommended_action = _safe_text(
        summary.get("recommended_action"),
        "Resolve closure blockers, regenerate affected artifacts, and rerun handoff ZIP verification.",
    )
    top = blockers.iloc[0]

    lines = [
        f"# {title}",
        "",
        "This checklist turns the detached closure blocker queue into a human repair worksheet.",
        "It is generated only when the release is not closed and should be archived after all blockers are resolved.",
        "",
        "## Closure state",
        "",
        f"- Readiness status: `{readiness_status}`",
        f"- Release closed: {'yes' if release_closed else 'no'}",
        f"- Blocker rows: `{len(blockers)}`",
        f"- Critical blockers: `{critical_rows}`",
        f"- High blockers: `{high_rows}`",
        f"- Top blocker: `{_safe_text(top.get('blocker_type'), '-')}` / `{_safe_text(top.get('file_name'), '-')}`",
        f"- Summary action: {recommended_action}",
        "",
        "## Remediation checklist",
        "",
    ]

    for _, row in blockers.head(20).iterrows():
        lines.append(
            "- [ ] "
            f"Rank `{_safe_int(row.get('blocker_rank'), 0)}` / `{_safe_text(row.get('severity'), '-')}` / "
            f"`{_safe_text(row.get('blocker_type'), '-')}`: "
            f"`{_safe_text(row.get('file_name'), '-')}` from `{_safe_text(row.get('blocker_source'), '-')}`. "
            f"Issue: {_safe_text(row.get('issue'), 'not recorded')} "
            f"Required fix: {_safe_text(row.get('required_fix'), 'fix this blocker before closure')} "
            f"Recommended action: {_safe_text(row.get('recommended_action'), 'rerun closure readiness after the fix')}"
        )
    if len(blockers) > 20:
        lines.append(f"- [ ] Review {len(blockers) - 20} additional blocker rows in the CSV queue.")

    lines.extend(
        [
            "",
            "## Re-run sequence",
            "",
            "- [ ] Apply the required fixes above to source evidence, execution receipt, generated artifacts, or ZIP package.",
            "- [ ] Regenerate closure ledger and closure certificate from the corrected artifacts.",
            "- [ ] Regenerate handoff ZIP and manifest together when any packaged artifact changes.",
            "- [ ] Rerun ZIP verification and closure readiness summary.",
            "- [ ] Treat release as closed only when readiness status is `closed-and-verified` and this checklist no longer generates.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_consensus_rerank_release_closure_detached_manifest(
    consensus_rerank_release_closure_summary_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_closure_blocker_df: Optional[pd.DataFrame] = None,
    consensus_rerank_release_closure_remediation_checklist_markdown: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_artifact(
        artifact_name: str,
        file_name: str,
        artifact_type: str,
        row_count: int,
        data: bytes,
        status: str,
        purpose: str,
        recommended_use: str,
    ) -> None:
        if row_count <= 0 or not data:
            return
        byte_size, digest = _rerank_artifact_integrity(data)
        rows.append(
            {
                "artifact_name": artifact_name,
                "file_name": file_name,
                "artifact_type": artifact_type,
                "row_count": int(row_count),
                "byte_size": int(byte_size),
                "sha256": digest,
                "status": status,
                "purpose": purpose,
                "recommended_use": recommended_use,
            }
        )

    closure_status = "not-generated"
    if consensus_rerank_release_closure_summary_df is not None and not getattr(consensus_rerank_release_closure_summary_df, "empty", True):
        summary = consensus_rerank_release_closure_summary_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_CLOSURE_SUMMARY_COLUMNS:
            if column not in summary.columns:
                summary[column] = ""
        closure_status = _safe_text(summary.iloc[0].get("closure_readiness_status"), "generated")
        add_artifact(
            "Consensus rerank release closure readiness summary",
            "consensus_rerank_release_closure_summary.csv",
            "csv",
            int(len(summary)),
            _rerank_csv_artifact_bytes(summary[CONSENSUS_RERANK_RELEASE_CLOSURE_SUMMARY_COLUMNS]),
            closure_status,
            "Detached one-row final closure gate derived after handoff ZIP verification.",
            "Archive this with the handoff ZIP, closure certificate, and detached manifest.",
        )

    if consensus_rerank_release_closure_blocker_df is not None and not getattr(consensus_rerank_release_closure_blocker_df, "empty", True):
        blockers = consensus_rerank_release_closure_blocker_df.copy()
        for column in CONSENSUS_RERANK_RELEASE_CLOSURE_BLOCKER_COLUMNS:
            if column not in blockers.columns:
                blockers[column] = ""
        add_artifact(
            "Consensus rerank release closure blocker queue",
            "consensus_rerank_release_closure_blocker_queue.csv",
            "csv",
            int(len(blockers)),
            _rerank_csv_artifact_bytes(blockers[CONSENSUS_RERANK_RELEASE_CLOSURE_BLOCKER_COLUMNS]),
            "remediation-required",
            "Detached action queue for closure ledger and handoff ZIP verification blockers.",
            "Use this to drive repair work before rerunning closure readiness.",
        )

    checklist_text = "" if consensus_rerank_release_closure_remediation_checklist_markdown is None else str(consensus_rerank_release_closure_remediation_checklist_markdown)
    checklist_lines = len([line for line in checklist_text.splitlines() if line.strip()])
    add_artifact(
        "Consensus rerank release closure remediation checklist",
        "consensus_rerank_release_closure_remediation_checklist.md",
        "markdown",
        checklist_lines,
        checklist_text.encode("utf-8") if _safe_text(checklist_text) else b"",
        "remediation-required",
        "Detached human repair checklist generated from closure blockers.",
        "Use this as the manual remediation worksheet, then rerun closure readiness.",
    )

    if not rows:
        return _empty_consensus_rerank_guardrail_artifact_manifest_df()
    return pd.DataFrame(rows, columns=CONSENSUS_RERANK_GUARDRAIL_ARTIFACT_MANIFEST_COLUMNS).reset_index(drop=True)


def build_pocket_reliability_checklist(decision_df: Optional[pd.DataFrame], *, max_pockets: int = 3) -> pd.DataFrame:
    if decision_df is None or getattr(decision_df, "empty", True) or "pocket_id" not in decision_df.columns:
        return pd.DataFrame(columns=POCKET_RELIABILITY_COLUMNS)

    table = decision_df.copy()
    if "decision_rank" in table.columns:
        table = table.sort_values(["decision_rank", "decision_score"], ascending=[True, False])
    elif "decision_score" in table.columns:
        table = table.sort_values("decision_score", ascending=False)
    table = table.head(max(1, int(max_pockets)))

    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        pocket_id = _safe_text(row.get("pocket_id"))
        if not pocket_id:
            continue

        direct_anchor_count = _safe_int(row.get("direct_anchor_count"))
        route_anchor_count = _safe_int(row.get("route_anchor_count"))
        quality_label = _safe_text(row.get("evidence_quality_label"), "unknown")
        functional = _safe_float(row.get("functional_confidence"))
        geometry = _safe_float(row.get("geometry_confidence"))
        method_votes = _safe_int(row.get("method_vote_count"))
        risk_flags = _risk_flag_set(row.get("risk_flags"))
        audit_status = _safe_text(row.get("audit_status"))
        recommended_action = _safe_text(row.get("recommended_action"))
        lit_delta = _safe_int(row.get("literature_rank_delta"))
        route_delta = _safe_int(row.get("evidence_route_rank_delta"))
        cons_delta = _safe_int(row.get("conservation_rank_delta"))

        if direct_anchor_count > 0:
            anchor_status = "pass"
            anchor_action = "Use direct anchors as the pocket core before expanding the boundary."
        elif route_anchor_count > 0 or quality_label in {"route-anchor", "structure-verified-external"} or functional >= 0.45:
            anchor_status = "review"
            anchor_action = "Review route-derived anchors against residue numbering before treating them as catalytic points."
        else:
            anchor_status = "missing"
            anchor_action = "Add UniProt, M-CSA, literature, or manual key residues to avoid geometry-only ranking."
        rows.append(
            _check_row(
                pocket_id,
                1,
                "Functional anchors",
                anchor_status,
                f"direct={direct_anchor_count}; route={route_anchor_count}; quality={quality_label}; functional={functional:.3f}",
                "Enzyme active sites should be tied to catalytic/binding residues, not only to surface cavities.",
                anchor_action,
            )
        )

        mapping_risk_flags = {"low-mapping-quality", "neighborhood-expansion-risk", "evidence-warning"}
        if risk_flags & mapping_risk_flags:
            mapping_status = "review"
            mapping_action = "Inspect chain, insertion codes, UniProt/PDB offsets, and expanded-neighborhood residues."
        elif quality_label in {"geometry-only", "no-external-evidence"}:
            mapping_status = "missing"
            mapping_action = "Fetch or upload external residue evidence before relying on this candidate."
        else:
            mapping_status = "pass"
            mapping_action = "Keep the mapped evidence visible as the validation anchor layer."
        rows.append(
            _check_row(
                pocket_id,
                2,
                "Evidence mapping risk",
                mapping_status,
                ", ".join(sorted(risk_flags)) if risk_flags else "none",
                "A correct catalytic residue is not useful if it was mapped onto the wrong chain or numbering system.",
                mapping_action,
            )
        )

        if geometry >= 0.55 and method_votes >= 2:
            geometry_status = "pass"
            geometry_action = "Use geometry support to define the shell around evidence anchors."
        elif geometry >= 0.35 or method_votes >= 2:
            geometry_status = "review"
            geometry_action = "Compare pocket boundary against neighboring cavities and hotspot overlap."
        else:
            geometry_status = "missing"
            geometry_action = "Treat this as weak geometry; add ligand/contact context or rerun detection with broader settings."
        rows.append(
            _check_row(
                pocket_id,
                3,
                "Geometry consensus",
                geometry_status,
                f"geometry={geometry:.3f}; method_votes={method_votes}",
                "Reliable pockets need both functional anchors and a physically plausible cavity boundary.",
                geometry_action,
            )
        )

        ab_deltas = [lit_delta, route_delta, cons_delta]
        if any(delta > 0 for delta in ab_deltas):
            ab_status = "pass"
            ab_action = "Keep the evidence route enabled; it improves this candidate's ranking."
        elif any(delta < 0 for delta in ab_deltas):
            ab_status = "review"
            ab_action = "Compare before/after rankings to understand why evidence lowered this pocket."
        elif direct_anchor_count > 0 or route_anchor_count > 0:
            ab_status = "review"
            ab_action = "Evidence exists but did not move the rank; inspect whether weights are too conservative."
        else:
            ab_status = "missing"
            ab_action = "Run literature/evidence/conservation comparison after adding functional evidence."
        rows.append(
            _check_row(
                pocket_id,
                4,
                "Evidence A/B movement",
                ab_status,
                f"literature={lit_delta:+d}; route={route_delta:+d}; conservation={cons_delta:+d}",
                "A/B movement shows whether external evidence is actually changing the product recommendation.",
                ab_action,
            )
        )

        if audit_status == "ready-to-validate" or recommended_action == "validate-prioritize":
            action_status = "pass"
            action_step = _safe_text(row.get("next_step"), "Prioritize validation around the top evidence anchors.")
        elif audit_status in {"mapping-review-needed", "shortlist"}:
            action_status = "review"
            action_step = _safe_text(row.get("next_step"), "Review the listed risks before wet-lab or docking follow-up.")
        else:
            action_status = "missing"
            action_step = _safe_text(row.get("next_step"), "Do not treat this as a final active-site call yet.")
        rows.append(
            _check_row(
                pocket_id,
                5,
                "Actionability",
                action_status,
                f"audit={audit_status or '-'}; action={recommended_action or '-'}",
                "The UI should end with an explicit next step instead of a raw score that users must interpret.",
                action_step,
            )
        )

    if not rows:
        return pd.DataFrame(columns=POCKET_RELIABILITY_COLUMNS)
    return pd.DataFrame(rows, columns=POCKET_RELIABILITY_COLUMNS)


def build_pocket_precision_triage(
    decision_df: Optional[pd.DataFrame],
    checklist_df: Optional[pd.DataFrame] = None,
    *,
    max_pockets: int = 3,
) -> pd.DataFrame:
    if decision_df is None or getattr(decision_df, "empty", True) or "pocket_id" not in decision_df.columns:
        return pd.DataFrame(columns=POCKET_TRIAGE_COLUMNS)

    decision_table = decision_df.copy()
    if "decision_rank" in decision_table.columns:
        decision_table = decision_table.sort_values(["decision_rank", "decision_score"], ascending=[True, False])
    elif "decision_score" in decision_table.columns:
        decision_table = decision_table.sort_values("decision_score", ascending=False)
    decision_table = decision_table.head(max(1, int(max_pockets)))

    if checklist_df is None or getattr(checklist_df, "empty", True):
        checklist_table = build_pocket_reliability_checklist(decision_table, max_pockets=max_pockets)
    else:
        checklist_table = checklist_df.copy()

    rows: list[dict[str, Any]] = []
    for _, decision_row in decision_table.iterrows():
        pocket_id = _safe_text(decision_row.get("pocket_id"))
        if not pocket_id:
            continue
        checks = (
            checklist_table[checklist_table["pocket_id"].astype(str) == pocket_id].copy()
            if not checklist_table.empty and "pocket_id" in checklist_table.columns
            else pd.DataFrame(columns=POCKET_RELIABILITY_COLUMNS)
        )
        pass_checks = _check_names_by_status(checks, "pass")
        review_checks = _check_names_by_status(checks, "review")
        missing_checks = _check_names_by_status(checks, "missing")
        pass_count = len(pass_checks)
        review_count = len(review_checks)
        missing_count = len(missing_checks)
        functional = _safe_float(decision_row.get("functional_confidence"))
        geometry = _safe_float(decision_row.get("geometry_confidence"))
        decision_score = _safe_float(decision_row.get("decision_score"))
        audit_status = _safe_text(decision_row.get("audit_status"))
        risk_flags = _risk_flag_set(decision_row.get("risk_flags"))

        if missing_count == 0 and review_count == 0 and (audit_status == "ready-to-validate" or decision_score >= 0.68):
            tier = "validation-ready"
            priority = 1
            action = "Proceed to validation around core evidence anchors."
            reason = "All reliability gates pass and the decision audit is ready."
        elif "Functional anchors" in missing_checks or "Evidence mapping risk" in missing_checks:
            tier = "evidence-gap"
            priority = 2
            action = "Do not finalize; add or verify functional residue evidence first."
            reason = "The candidate lacks the residue-level enzyme evidence needed for a high-precision active-site call."
        elif "Evidence mapping risk" in review_checks or {"low-mapping-quality", "neighborhood-expansion-risk", "evidence-warning"} & risk_flags:
            tier = "mapping-review"
            priority = 2
            action = "Review chain/numbering/mapping before validation."
            reason = "Functional evidence exists, but residue mapping or neighborhood expansion can shift the pocket core."
        elif "Geometry consensus" in missing_checks or "Geometry consensus" in review_checks or geometry < 0.45:
            tier = "geometry-review"
            priority = 3
            action = "Check cavity boundary against alternate geometry and ligand/contact context."
            reason = "Evidence may be useful, but the physical pocket boundary is not yet stable."
        elif missing_count > 0 or review_count > 0:
            tier = "evidence-review"
            priority = 3
            action = "Keep shortlisted and resolve the remaining review/missing gates."
            reason = "The pocket is plausible but still has unresolved evidence or actionability gaps."
        elif functional < 0.35:
            tier = "exploratory"
            priority = 4
            action = "Use only as exploratory geometry until functional evidence is added."
            reason = "The candidate is dominated by geometry rather than enzyme-specific evidence."
        else:
            tier = "shortlist"
            priority = 3
            action = "Keep as a secondary candidate and compare with stronger evidence-led pockets."
            reason = "No hard blocker is visible, but support is not strong enough for a primary validation call."

        rows.append(
            {
                "pocket_id": pocket_id,
                "decision_rank": _safe_int(decision_row.get("decision_rank")),
                "precision_tier": tier,
                "triage_priority": priority,
                "triage_action": action,
                "blocking_checks": ", ".join(missing_checks) if missing_checks else "none",
                "review_checks": ", ".join(review_checks) if review_checks else "none",
                "pass_count": pass_count,
                "review_count": review_count,
                "missing_count": missing_count,
                "triage_reason": reason,
                "next_data_to_add": _triage_next_data(missing_checks, review_checks, risk_flags),
            }
        )

    if not rows:
        return pd.DataFrame(columns=POCKET_TRIAGE_COLUMNS)
    result = pd.DataFrame(rows, columns=POCKET_TRIAGE_COLUMNS)
    return result.sort_values(["triage_priority", "decision_rank", "pocket_id"], ascending=[True, True, True]).reset_index(drop=True)


def add_pocket_residue_layers(pocket_rows: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pocket_rows is None or getattr(pocket_rows, "empty", True):
        columns = list(getattr(pocket_rows, "columns", [])) + [column for column in RESIDUE_LAYER_COLUMNS if column not in getattr(pocket_rows, "columns", [])]
        return pd.DataFrame(columns=columns)

    table = pocket_rows.copy()
    layer_values = []
    scores = []
    reasons = []
    for _, row in table.iterrows():
        direct_anchor = _safe_bool(row.get("external_direct_anchor", False))
        route_anchor = _safe_bool(row.get("evidence_route_anchor", False))
        hotspot = _safe_bool(row.get("is_hotspot", False))
        external_support = _safe_float(row.get("external_support"))
        anchor_proximity = _safe_float(row.get("evidence_anchor_proximity"))
        residue_score = max(_safe_float(row.get("residue_score")), _safe_float(row.get("precision_score")))
        ligand_contacts = _safe_int(row.get("ligand_contact_count"))
        contact_count = _safe_int(row.get("contact_count"))

        layer_score = (
            0.34 * float(direct_anchor)
            + 0.18 * float(route_anchor)
            + 0.16 * float(hotspot)
            + 0.14 * _clip(external_support)
            + 0.08 * _clip(anchor_proximity)
            + 0.06 * _clip(residue_score)
            + 0.04 * _clip(float(ligand_contacts) / 2.0)
        )
        reason_parts = []
        if direct_anchor:
            reason_parts.append("direct evidence anchor")
        if route_anchor:
            reason_parts.append("evidence-route anchor")
        if hotspot:
            reason_parts.append("hotspot")
        if ligand_contacts > 0:
            reason_parts.append("ligand contact")
        if anchor_proximity >= 0.40 and not direct_anchor:
            reason_parts.append("near evidence anchor")
        if contact_count >= 4:
            reason_parts.append("dense local contacts")

        if direct_anchor or (route_anchor and external_support >= 0.45) or (hotspot and external_support >= 0.30):
            layer = "core"
        elif anchor_proximity >= 0.40 or external_support >= 0.28 or residue_score >= 0.55 or ligand_contacts > 0 or contact_count >= 4:
            layer = "shell"
        else:
            layer = "rim"

        layer_values.append(layer)
        scores.append(round(_clip(layer_score), 3))
        reasons.append(", ".join(reason_parts) if reason_parts else "boundary residue")

    table["pocket_layer"] = layer_values
    table["pocket_layer_score"] = scores
    table["pocket_layer_reason"] = reasons
    return table
