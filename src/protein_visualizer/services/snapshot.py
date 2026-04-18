from __future__ import annotations

import html
import json
import textwrap
from datetime import datetime
from typing import Any, Optional, Sequence

import pandas as pd

from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return value.item()  # numpy scalar
    except Exception:
        return str(value)


def _format_volume_text(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.1f} A³"
    except (TypeError, ValueError):
        return "-"


def _frame_preview(table: Optional[pd.DataFrame], columns: Optional[Sequence[str]] = None, max_rows: int = 8) -> list[dict[str, Any]]:
    if table is None or getattr(table, "empty", True):
        return []

    preview = table.copy()
    if columns:
        available_columns = [column for column in columns if column in preview.columns]
        preview = preview[available_columns]
    preview = preview.head(max_rows)

    records: list[dict[str, Any]] = []
    for row in preview.to_dict(orient="records"):
        records.append({key: _safe_value(value) for key, value in row.items()})
    return records


def build_analysis_snapshot(
    energy_table: Optional[pd.DataFrame],
    *,
    title: str = "ProteinInsight 分析快照",
    annotation_table: Optional[pd.DataFrame] = None,
    hotspot_df: Optional[pd.DataFrame] = None,
    pocket_summary: Optional[pd.DataFrame] = None,
    joint_candidate_df: Optional[pd.DataFrame] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    protein_volume: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    summary = build_analysis_summary(energy_table)
    if protein_volume is not None:
        summary["protein_volume"] = float(protein_volume)

    snapshot = {
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {key: _safe_value(value) for key, value in summary.items()},
        "protein_volume": _safe_value(protein_volume),
        "hotspot_count": int(len(hotspot_df)) if hotspot_df is not None else 0,
        "annotation_rows": int(len(annotation_table)) if annotation_table is not None else 0,
        "pocket_rows": int(len(pocket_summary)) if pocket_summary is not None else 0,
        "joint_candidate_rows": int(len(joint_candidate_df)) if joint_candidate_df is not None else 0,
        "comparison_rows": int(len(comparison_df)) if comparison_df is not None else 0,
        "top_hotspots": _frame_preview(hotspot_df, ["label", "chain", "resid", "resname", "delta_total", "hotspot_rank", "energy"]),
        "annotation_preview": _frame_preview(
            annotation_table,
            ["residue_label", "classification_label", "energy", "hotspot_rank", "is_hotspot", "is_pocket"],
            max_rows=12,
        ),
        "pocket_summary": _frame_preview(
            pocket_summary,
            [
                "pocket_id",
                "smart_rank_label",
                "smart_rank_score",
                "smart_rank_reason",
                "evidence_quality_label",
                "evidence_quality_score",
                "evidence_quality_warning",
                "detection_route",
                "consensus_methods",
                "method_vote_count",
                "consensus_score",
                "volume",
                "score",
                "residue_count",
                "hotspot_count",
                "residue_labels",
            ],
            max_rows=10,
        ),
        "joint_candidate_preview": _frame_preview(
            joint_candidate_df,
            [
                "recommendation_rank",
                "pocket_id",
                "recommendation_score",
                "recommendation_label",
                "recommendation_action",
                "evidence_quality_label",
                "recommendation_reason",
                "smart_rank_label",
                "hotspot_overlap_count",
                "interface_overlap_count",
                "triple_overlap_count",
            ],
            max_rows=10,
        ),
        "comparison_preview": _frame_preview(
            comparison_df,
            ["chain", "resid", "resname", "count", "label", "is_common"],
            max_rows=20,
        ),
        "extra": {key: _safe_value(value) for key, value in (extra or {}).items()},
    }
    return snapshot


def snapshot_to_json_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _reliability_status_counts(extra: dict[str, Any]) -> dict[str, int]:
    records = extra.get("pocket_reliability") or []
    counts = {"pass": 0, "review": 0, "missing": 0}
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            status = str(record.get("status") or "").strip().lower()
            if status in counts:
                counts[status] += 1
    for status in counts:
        key = f"pocket_reliability_{status}_count"
        if key in extra:
            try:
                counts[status] = int(extra.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return counts


def snapshot_to_summary_lines(snapshot: dict[str, Any]) -> list[str]:
    summary = snapshot.get("summary") or {}
    extra = snapshot.get("extra") or {}
    lines = [
        f"生成时间: {snapshot.get('generated_at', '-')}",
        f"标题: {snapshot.get('title', '-')}",
        f"残基总数: {summary.get('residue_count', '-')}",
        f"有效能量数: {summary.get('valid_energy_count', '-')}/{summary.get('residue_count', '-')}",
        f"能量来源: {summary.get('energy_source') or '未标注'}",
        f"平均能量: {format_energy_value(summary.get('mean_energy'))}",
        f"最低能量: {format_energy_value(summary.get('min_energy'))}",
        f"最高能量: {format_energy_value(summary.get('max_energy'))}",
    ]

    protein_volume = summary.get("protein_volume")
    if protein_volume is None:
        protein_volume = snapshot.get("protein_volume")
    volume_text = _format_volume_text(protein_volume)
    if volume_text != "-":
        lines.append(f"蛋白质体积: {volume_text}")

    if snapshot.get("hotspot_count") is not None:
        lines.append(f"热点残基数: {snapshot.get('hotspot_count')}")
    if snapshot.get("pocket_rows") is not None:
        lines.append(f"口袋条目数: {snapshot.get('pocket_rows')}")
    if snapshot.get("joint_candidate_rows") is not None:
        lines.append(f"鑱斿悎鎺ㄨ崘鏉＄洰鏁? {snapshot.get('joint_candidate_rows')}")
    methods_used = str(extra.get("auto_detection_methods_used") or "").strip()
    if methods_used:
        lines.append(f"鑷姩鍙ｈ鏂规硶: {methods_used}")
    status_summary = str(extra.get("auto_detection_status_summary") or "").strip()
    if status_summary:
        lines.append(f"妫€娴嬬姸鎬? {status_summary}")
    p2rank_status = str(extra.get("auto_detection_p2rank_status") or "").strip()
    if p2rank_status:
        lines.append(
            f"P2Rank: {p2rank_status} / pred {int(extra.get('auto_detection_p2rank_prediction_rows') or 0)} / "
            f"res {int(extra.get('auto_detection_p2rank_residue_rows') or 0)}"
        )
    if bool(extra.get("p2rank_ab_enabled")):
        comparison_rows = len(extra.get("p2rank_ab_comparison") or [])
        lines.append(f"P2Rank A/B: enabled / rows {comparison_rows}")
    try:
        external_rows = int(extra.get("auto_detection_external_rows") or 0)
    except (TypeError, ValueError):
        external_rows = 0
    if external_rows > 0:
        source_text = str(extra.get("auto_detection_external_sources") or "external").strip()
        lines.append(f"澶栭儴浣嶇偣璇佹嵁: {external_rows} ({source_text})")
    ai_rows = int(extra.get("ai_evidence_rows") or 0)
    ai_status = str(extra.get("ai_evidence_status") or "").strip()
    if ai_rows > 0 or ai_status:
        lines.append(f"AI evidence: {ai_rows} rows / status {ai_status or '-'}")
    ai_ranked = int(extra.get("ai_evidence_ranked_rows") or 0)
    if ai_rows > 0 or ai_ranked > 0:
        lines.append(f"AI evidence used for ranking: {ai_ranked} rows")
    ai_review_decisions = int(extra.get("ai_review_decision_rows") or 0)
    if ai_review_decisions > 0:
        ai_review_status = str(extra.get("ai_review_decision_status") or "-").strip()
        ai_review_applied = int(extra.get("ai_review_decision_applied_rows") or ai_review_decisions)
        lines.append(f"AI review decisions: {ai_review_decisions} rows / applied {ai_review_applied} / status {ai_review_status or '-'}")
    ai_review_validation_rows = int(extra.get("ai_review_decision_validation_rows") or 0)
    if ai_review_validation_rows > 0:
        blocked_rows = int(extra.get("ai_review_decision_validation_blocked_rows") or 0)
        lines.append(f"AI review decision validation: {ai_review_validation_rows} rows / blocked {blocked_rows}")
    ai_review_round_status = str(extra.get("ai_review_round_status") or "").strip()
    if ai_review_round_status:
        rankable_rows = int(extra.get("ai_review_round_rankable_rows") or 0)
        lines.append(f"AI review round: {ai_review_round_status} / rankable {rankable_rows}")
    ai_review_effect = str(extra.get("ai_review_ranking_effect_status") or "").strip()
    if ai_review_effect:
        promoted_rows = int(extra.get("ai_review_ranking_promoted_rows") or 0)
        removed_rows = int(extra.get("ai_review_ranking_removed_rows") or 0)
        lines.append(f"AI review ranking delta: {ai_review_effect} / promoted {promoted_rows}, removed {removed_rows}")
    ai_review_manifest_rows = int(extra.get("ai_review_artifact_manifest_rows") or 0)
    if ai_review_manifest_rows > 0:
        lines.append(f"AI review artifact manifest: {ai_review_manifest_rows} files")
    if bool(extra.get("ai_review_bundle_readme_available")):
        lines.append("AI review bundle README: available")
    if bool(extra.get("ai_review_artifact_bundle_available")):
        lines.append("AI review artifact bundle: available")
    ai_review_bundle_verification_rows = int(extra.get("ai_review_bundle_verification_rows") or 0)
    if ai_review_bundle_verification_rows > 0:
        failed_rows = int(extra.get("ai_review_bundle_verification_failed_rows") or 0)
        lines.append(f"AI review bundle verification: {ai_review_bundle_verification_rows} files / failed {failed_rows}")
        verification_status = str(extra.get("ai_review_bundle_verification_status") or "").strip()
        if verification_status:
            lines.append(f"AI review bundle verification summary: {verification_status}")
    if bool(extra.get("ai_review_bundle_certificate_available")):
        lines.append("AI review bundle certificate: available")
    ai_review_outcomes = int(extra.get("ai_review_decision_outcome_rows") or 0)
    if ai_review_outcomes > 0:
        lines.append(f"AI review decision outcomes: {ai_review_outcomes} rows")
    ai_review_template_rows = int(extra.get("ai_review_decision_template_rows") or 0)
    if ai_review_template_rows > 0:
        lines.append(f"AI review decision template rows: {ai_review_template_rows}")
    ai_influence = str(extra.get("ai_influence_level") or "").strip()
    if ai_influence:
        top_ai_residues = str(extra.get("top_pocket_ai_residues") or "none").strip()
        lines.append(f"AI ranking influence: {ai_influence} / Top pocket AI residues {top_ai_residues or 'none'}")
    ai_supported = int(extra.get("ai_evidence_audit_supported_count") or 0)
    ai_review = int(extra.get("ai_evidence_audit_review_count") or 0)
    if ai_supported > 0 or ai_review > 0:
        lines.append(f"AI evidence audit: supported {ai_supported}, review {ai_review}")
    ai_review_queue_rows = int(extra.get("ai_evidence_review_queue_rows") or 0)
    if ai_review_queue_rows > 0:
        top_fix = str(extra.get("top_ai_review_fix_type") or "-").strip()
        lines.append(f"AI evidence review queue: {ai_review_queue_rows} rows / top fix {top_fix or '-'}")
    ai_followup_rows = int(extra.get("ai_followup_plan_rows") or 0)
    if ai_followup_rows > 0:
        lines.append(f"AI follow-up plan rows: {ai_followup_rows}")
        top_query = str(extra.get("top_ai_followup_query") or "").strip()
        if top_query:
            lines.append(f"Top AI follow-up query: {top_query}")
    residue_consensus_rows = int(extra.get("residue_evidence_consensus_rows") or 0)
    if residue_consensus_rows > 0:
        top_anchor = str(extra.get("top_residue_consensus_anchor") or "-").strip()
        top_tier = str(extra.get("top_residue_consensus_tier") or "-").strip()
        top_score = format_energy_value(extra.get("top_residue_consensus_score"))
        lines.append(f"Residue evidence consensus: {residue_consensus_rows} rows / top {top_anchor or '-'} ({top_tier or '-'}, score {top_score})")
    pocket_consensus_rows = int(extra.get("pocket_consensus_coverage_rows") or 0)
    if pocket_consensus_rows > 0:
        top_pocket = str(extra.get("top_pocket_consensus_coverage_id") or "-").strip()
        top_label = str(extra.get("top_pocket_consensus_label") or "-").strip()
        anchor_count = int(extra.get("top_pocket_consensus_anchor_count") or 0)
        best_score = format_energy_value(extra.get("top_pocket_consensus_best_score"))
        lines.append(
            f"Pocket consensus coverage: {pocket_consensus_rows} rows / top {top_pocket or '-'} ({top_label or '-'}, anchors {anchor_count}, score {best_score})"
        )
    benchmark_reference_rows = int(extra.get("pocket_benchmark_reference_rows") or 0)
    if benchmark_reference_rows > 0:
        top1_coverage = format_energy_value(extra.get("pocket_benchmark_top1_coverage"))
        top3_coverage = format_energy_value(extra.get("pocket_benchmark_top3_coverage"))
        top1_status = str(extra.get("pocket_benchmark_top1_status") or "-").strip()
        top3_status = str(extra.get("pocket_benchmark_top3_status") or "-").strip()
        lines.append(
            f"Catalytic pocket benchmark: references {benchmark_reference_rows} / Top-1 {top1_coverage} ({top1_status or '-'}) / Top-3 {top3_coverage} ({top3_status or '-'})"
        )
    benchmark_case_summary = extra.get("pocket_benchmark_case_summary") or []
    benchmark_dataset_rows = int(extra.get("pocket_benchmark_dataset_summary_rows") or 0)
    if benchmark_dataset_rows > 0 or benchmark_case_summary:
        case_ids = {
            str(row.get("benchmark_id") or "").strip()
            for row in benchmark_case_summary
            if isinstance(row, dict) and str(row.get("benchmark_id") or "").strip()
        }
        case_count = len(case_ids) if case_ids else int(extra.get("pocket_benchmark_case_summary_rows") or 0)
        lines.append(f"Catalytic benchmark dataset: cases {case_count} / summary rows {benchmark_dataset_rows}")
    benchmark_variant_rows = int(extra.get("pocket_benchmark_variant_comparison_rows") or 0)
    if benchmark_variant_rows > 0:
        lines.append(f"Catalytic benchmark variants: {benchmark_variant_rows} rows")
    benchmark_variant_case_rows = int(extra.get("pocket_benchmark_variant_case_comparison_rows") or 0)
    benchmark_variant_dataset_rows = int(extra.get("pocket_benchmark_variant_dataset_comparison_rows") or 0)
    if benchmark_variant_case_rows > 0 or benchmark_variant_dataset_rows > 0:
        lines.append(
            f"Catalytic benchmark variant cases: {benchmark_variant_case_rows} rows / dataset rows {benchmark_variant_dataset_rows}"
        )
    benchmark_variant_detail_rows = int(extra.get("pocket_benchmark_variant_detail_comparison_rows") or 0)
    if benchmark_variant_detail_rows > 0:
        lines.append(f"Catalytic benchmark variant residues: {benchmark_variant_detail_rows} rows")
    consensus_rerank_rows = int(extra.get("consensus_rerank_suggestion_rows") or 0)
    if consensus_rerank_rows > 0:
        rerank_pocket = str(extra.get("top_consensus_rerank_pocket_id") or "-").strip()
        rerank_status = str(extra.get("top_consensus_rerank_status") or "-").strip()
        rerank_delta = int(extra.get("top_consensus_rerank_rank_delta") or 0)
        lines.append(
            f"Consensus rerank suggestions: {consensus_rerank_rows} rows / top {rerank_pocket or '-'} ({rerank_status or '-'}, rank delta {rerank_delta:+d})"
        )
    consensus_preview_rows = int(extra.get("consensus_rerank_preview_rows") or 0)
    if consensus_preview_rows > 0:
        preview_pocket = str(extra.get("top_consensus_preview_pocket_id") or "-").strip()
        preview_decision = str(extra.get("top_consensus_preview_decision") or "-").strip()
        preview_delta = int(extra.get("top_consensus_preview_rank_delta") or 0)
        preview_score = format_energy_value(extra.get("top_consensus_preview_score"))
        lines.append(
            f"Consensus rerank preview: {consensus_preview_rows} rows / top {preview_pocket or '-'} ({preview_decision or '-'}, rank delta {preview_delta:+d}, score {preview_score})"
        )
    rerank_policy = str(extra.get("consensus_rerank_policy_status") or "").strip()
    if rerank_policy:
        changed_rows = int(extra.get("consensus_rerank_policy_changed_rows") or 0)
        blocked_rows = int(extra.get("consensus_rerank_policy_blocked_rows") or 0)
        lines.append(f"Consensus rerank policy gate: {rerank_policy} / changed {changed_rows}, blocked {blocked_rows}")
    rerank_action_rows = int(extra.get("consensus_rerank_action_queue_rows") or 0)
    if rerank_action_rows > 0:
        action_pocket = str(extra.get("top_consensus_rerank_action_pocket_id") or "-").strip()
        issue_type = str(extra.get("top_consensus_rerank_issue_type") or "-").strip()
        issue_severity = str(extra.get("top_consensus_rerank_issue_severity") or "-").strip()
        lines.append(
            f"Consensus rerank action queue: {rerank_action_rows} rows / top {action_pocket or '-'} ({issue_type or '-'}, {issue_severity or '-'})"
        )
    if bool(extra.get("consensus_rerank_action_checklist_available")):
        lines.append("Consensus rerank action checklist: available")
    rerank_apply_rows = int(extra.get("consensus_rerank_apply_simulation_rows") or 0)
    if rerank_apply_rows > 0:
        apply_pocket = str(extra.get("top_consensus_rerank_apply_pocket_id") or "-").strip()
        apply_status = str(extra.get("top_consensus_rerank_apply_status") or "-").strip()
        apply_delta = int(extra.get("top_consensus_rerank_apply_rank_delta") or 0)
        lines.append(
            f"Consensus rerank apply simulation: {rerank_apply_rows} rows / top {apply_pocket or '-'} ({apply_status or '-'}, rank delta {apply_delta:+d})"
        )
    rerank_delta_rows = int(extra.get("consensus_rerank_simulation_delta_rows") or 0)
    if rerank_delta_rows > 0:
        delta_pocket = str(extra.get("top_consensus_rerank_delta_pocket_id") or "-").strip()
        delta_type = str(extra.get("top_consensus_rerank_delta_change_type") or "-").strip()
        delta_rank = int(extra.get("top_consensus_rerank_delta_rank_delta") or 0)
        lines.append(
            f"Consensus rerank simulation delta: {rerank_delta_rows} rows / top {delta_pocket or '-'} ({delta_type or '-'}, rank delta {delta_rank:+d})"
        )
    scorecard_rows = int(extra.get("consensus_rerank_precision_scorecard_rows") or 0)
    if scorecard_rows > 0:
        precision_status = str(extra.get("consensus_rerank_precision_status") or "-").strip()
        precision_score = int(extra.get("consensus_rerank_precision_score") or 0)
        positive_rows = int(extra.get("consensus_rerank_positive_signal_rows") or 0)
        blocker_rows = int(extra.get("consensus_rerank_open_blocker_rows") or 0)
        lines.append(
            f"Consensus rerank precision scorecard: {precision_status or '-'} / score {precision_score} / positive {positive_rows}, blockers {blocker_rows}"
        )
    guardrail_rows = int(extra.get("consensus_rerank_precision_guardrail_rows") or 0)
    if guardrail_rows > 0:
        guardrail_status = str(extra.get("consensus_rerank_guardrail_status") or "-").strip()
        guardrail_decision = str(extra.get("consensus_rerank_guardrail_decision") or "-").strip()
        apply_mode = str(extra.get("consensus_rerank_guardrail_apply_mode") or "-").strip()
        lines.append(
            f"Consensus rerank precision guardrail: {guardrail_status or '-'} / decision {guardrail_decision or '-'}, mode {apply_mode or '-'}"
        )
    if bool(extra.get("consensus_rerank_guardrail_report_available")):
        lines.append("Consensus rerank precision guardrail report: available")
    manifest_rows = int(extra.get("consensus_rerank_guardrail_artifact_manifest_rows") or 0)
    if manifest_rows > 0 or bool(extra.get("consensus_rerank_guardrail_handoff_zip_available")):
        bundle_status = "available" if bool(extra.get("consensus_rerank_guardrail_handoff_zip_available")) else "not available"
        lines.append(f"Consensus rerank guardrail handoff bundle: {bundle_status} / manifest {manifest_rows} files")
    verification_status = str(extra.get("consensus_rerank_guardrail_bundle_verification_status") or "").strip()
    if verification_status:
        verification_rows = int(extra.get("consensus_rerank_guardrail_bundle_verification_rows") or 0)
        failed_rows = int(extra.get("consensus_rerank_guardrail_bundle_verification_failed_rows") or 0)
        lines.append(
            f"Consensus rerank guardrail bundle verification: {verification_status} / files {verification_rows}, failed {failed_rows}"
        )
    if bool(extra.get("consensus_rerank_guardrail_handoff_certificate_available")):
        lines.append("Consensus rerank guardrail handoff certificate: available")
    release_template_rows = int(extra.get("consensus_rerank_release_decision_template_rows") or 0)
    if release_template_rows > 0:
        lines.append(f"Consensus rerank release decision template: {release_template_rows} rows")
    release_decision_rows = int(extra.get("consensus_rerank_release_decision_rows") or 0)
    release_decision_status = str(extra.get("consensus_rerank_release_decision_status") or "").strip()
    if release_decision_rows > 0 or (release_decision_status and release_decision_status != "not-uploaded"):
        lines.append(f"Consensus rerank release decisions: {release_decision_rows} rows / status {release_decision_status or '-'}")
    release_validation_rows = int(extra.get("consensus_rerank_release_decision_validation_rows") or 0)
    if release_validation_rows > 0:
        blocked_rows = int(extra.get("consensus_rerank_release_decision_blocked_rows") or 0)
        lines.append(f"Consensus rerank release decision validation: {release_validation_rows} rows / blocked {blocked_rows}")
    release_review_status = str(extra.get("consensus_rerank_release_review_status") or "").strip()
    if release_review_status:
        release_allowed = "yes" if bool(extra.get("consensus_rerank_release_allowed")) else "no"
        lines.append(f"Consensus rerank release review: {release_review_status} / allowed {release_allowed}")
    release_apply_plan_rows = int(extra.get("consensus_rerank_release_apply_plan_rows") or 0)
    if release_apply_plan_rows > 0:
        top_apply_pocket = str(extra.get("top_consensus_rerank_release_apply_pocket_id") or "-").strip()
        apply_status = str(extra.get("top_consensus_rerank_release_apply_status") or "-").strip()
        lines.append(
            f"Consensus rerank release apply plan: {release_apply_plan_rows} rows / top {top_apply_pocket or '-'} ({apply_status or '-'})"
        )
    if bool(extra.get("consensus_rerank_release_apply_report_available")):
        lines.append("Consensus rerank release apply report: available")
    execution_template_rows = int(extra.get("consensus_rerank_release_execution_template_rows") or 0)
    if execution_template_rows > 0:
        lines.append(f"Consensus rerank release execution template: {execution_template_rows} rows")
    execution_receipt_rows = int(extra.get("consensus_rerank_release_execution_receipt_rows") or 0)
    execution_receipt_status = str(extra.get("consensus_rerank_release_execution_receipt_status") or "").strip()
    if execution_receipt_rows > 0 or (execution_receipt_status and execution_receipt_status != "not-uploaded"):
        lines.append(f"Consensus rerank release execution receipt: {execution_receipt_rows} rows / status {execution_receipt_status or '-'}")
    execution_validation_rows = int(extra.get("consensus_rerank_release_execution_validation_rows") or 0)
    if execution_validation_rows > 0:
        blocked_rows = int(extra.get("consensus_rerank_release_execution_blocked_rows") or 0)
        lines.append(f"Consensus rerank release execution validation: {execution_validation_rows} rows / blocked {blocked_rows}")
    execution_review_status = str(extra.get("consensus_rerank_release_execution_review_status") or "").strip()
    if execution_review_status:
        execution_complete = "yes" if bool(extra.get("consensus_rerank_release_execution_complete")) else "no"
        lines.append(f"Consensus rerank release execution: {execution_review_status} / complete {execution_complete}")
    if bool(extra.get("consensus_rerank_release_execution_report_available")):
        lines.append("Consensus rerank release execution report: available")
    if bool(extra.get("consensus_rerank_release_closure_certificate_available")):
        lines.append("Consensus rerank release closure certificate: available")
    closure_ledger_rows = int(extra.get("consensus_rerank_release_closure_ledger_rows") or 0)
    if closure_ledger_rows > 0:
        blocked_rows = int(extra.get("consensus_rerank_release_closure_ledger_blocked_rows") or 0)
        lines.append(f"Consensus rerank release closure ledger: {closure_ledger_rows} rows / blocked {blocked_rows}")
    closure_readiness_status = str(extra.get("consensus_rerank_release_closure_readiness_status") or "").strip()
    if closure_readiness_status:
        release_closed = "yes" if bool(extra.get("consensus_rerank_release_closed")) else "no"
        lines.append(f"Consensus rerank release closure readiness: {closure_readiness_status} / closed {release_closed}")
    closure_blocker_rows = int(extra.get("consensus_rerank_release_closure_blocker_rows") or 0)
    if closure_blocker_rows > 0:
        blocker_type = str(extra.get("top_consensus_rerank_release_closure_blocker_type") or "-").strip()
        lines.append(f"Consensus rerank release closure blockers: {closure_blocker_rows} rows / top {blocker_type or '-'}")
    if bool(extra.get("consensus_rerank_release_closure_remediation_checklist_available")):
        lines.append("Consensus rerank release closure remediation checklist: available")
    detached_manifest_rows = int(extra.get("consensus_rerank_release_closure_detached_manifest_rows") or 0)
    if detached_manifest_rows > 0:
        lines.append(f"Consensus rerank release closure detached manifest: {detached_manifest_rows} files")
    pocket_preview = snapshot.get("pocket_summary") or []
    if pocket_preview:
        top_pocket = pocket_preview[0]
        pocket_id = top_pocket.get("pocket_id") or "-"
        rank_label = top_pocket.get("smart_rank_label") or "-"
        lines.append(f"Top 鍙ｈ: {pocket_id} ({rank_label})")
        evidence_quality = top_pocket.get("evidence_quality_label") or ""
        evidence_score = top_pocket.get("evidence_quality_score")
        if evidence_quality:
            lines.append(f"Top pocket evidence quality: {evidence_quality} ({format_energy_value(evidence_score)})")
    joint_preview = snapshot.get("joint_candidate_preview") or []
    if joint_preview:
        top_joint = joint_preview[0]
        joint_id = top_joint.get("pocket_id") or "-"
        joint_label = top_joint.get("recommendation_label") or "-"
        joint_action = top_joint.get("recommendation_action") or "-"
        if joint_action != "-":
            lines.append(f"Top joint action: {joint_action}")
        lines.append(f"Top 鑱斿悎鎺ㄨ崘: {joint_id} ({joint_label})")
    decision_label = str(extra.get("top_pocket_decision_label") or "").strip()
    decision_score = extra.get("top_pocket_decision_score")
    audit_status = str(extra.get("top_pocket_audit_status") or "").strip()
    if decision_label or audit_status:
        lines.append(
            f"Top active-site decision: {decision_label or '-'} / score {format_energy_value(decision_score)} / audit {audit_status or '-'}"
        )
    precision_tier = str(extra.get("top_pocket_precision_tier") or "").strip()
    triage_action = str(extra.get("top_pocket_triage_action") or "").strip()
    if precision_tier or triage_action:
        lines.append(f"Top pocket precision tier: {precision_tier or '-'} / action {triage_action or '-'}")
    reliability_counts = _reliability_status_counts(extra)
    if any(reliability_counts.values()):
        lines.append(
            "Pocket reliability checks: "
            f"pass {reliability_counts['pass']}, review {reliability_counts['review']}, missing {reliability_counts['missing']}"
        )
    reliability_gaps = str(extra.get("top_pocket_reliability_gaps") or "").strip()
    if reliability_gaps:
        lines.append(f"Top pocket reliability gaps: {reliability_gaps}")
    return lines


def _svg_escape(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=False)


def _svg_multiline_text(x: int, y: int, text: str, *, font_size: int = 16, fill: str = "#0f1724", max_chars: int = 38) -> str:
    lines = textwrap.wrap(text, width=max_chars) or [""]
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else font_size + 4
        tspans.append(f'<tspan x="{x}" dy="{dy}">{_svg_escape(line)}</tspan>')
    return f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{font_size}" fill="{fill}">' + "".join(tspans) + "</text>"


def build_snapshot_svg(snapshot: dict[str, Any], *, width: int = 1280, height: int = 920) -> bytes:
    summary = snapshot.get("summary") or {}
    protein_volume = summary.get("protein_volume")
    if protein_volume is None:
        protein_volume = snapshot.get("protein_volume")
    cards = [
        ("残基总数", summary.get("residue_count", "-")),
        ("平均能量", format_energy_value(summary.get("mean_energy"))),
        ("蛋白质体积", _format_volume_text(protein_volume)),
        ("热点残基", snapshot.get("hotspot_count", 0)),
        ("口袋条目", snapshot.get("pocket_rows", 0)),
        ("能量来源", summary.get("energy_source") or "未标注"),
    ]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f'<rect width="100%" height="100%" fill="#f6f9fc"/>')
    parts.append(f'<defs><linearGradient id="headerGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#2b6cb0"/><stop offset="100%" stop-color="#805ad5"/></linearGradient></defs>')
    parts.append(f'<rect x="32" y="28" width="{width - 64}" height="130" rx="20" fill="url(#headerGrad)"/>')
    parts.append(f'<text x="58" y="74" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="28" fill="#ffffff" font-weight="700">{_svg_escape(snapshot.get("title", "ProteinInsight 分析快照"))}</text>')
    parts.append(f'<text x="58" y="104" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14" fill="#eff6ff">{_svg_escape(snapshot.get("generated_at", ""))}</text>')
    parts.append(_svg_multiline_text(58, 132, f"能量来源：{summary.get('energy_source') or '未标注'}；平均能量：{format_energy_value(summary.get('mean_energy'))}", font_size=15, fill="#eff6ff", max_chars=80))

    card_width = (width - 84) / 3
    card_height = 84
    card_y = 186
    for index, (label, value) in enumerate(cards):
        row = index // 3
        col = index % 3
        x = 32 + col * (card_width + 10)
        y = card_y + row * (card_height + 12)
        parts.append(f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="16" fill="#ffffff" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 28}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13" fill="#64748b">{_svg_escape(label)}</text>')
        parts.append(f'<text x="{x + 16}" y="{y + 58}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="22" fill="#0f1724" font-weight="700">{_svg_escape(value)}</text>')

    sections_y = 332
    section_width = (width - 84) / 2
    section_height = 260
    sections = [
        ("自动分析摘要", snapshot_to_summary_lines(snapshot)[:10]),
        ("热点与口袋预览", [
            *[f"热点: {item.get('label', '-')}; ΔG={item.get('delta_total', item.get('energy', '-'))}" for item in snapshot.get("top_hotspots", [])[:4]],
            *[
                "口袋: {pocket_id}{route_text}{method_text} | votes={vote_count}, volume={volume}, score={score}, hotspot={hotspot}".format(
                    pocket_id=item.get("pocket_id", "-"),
                    route_text=f" | route={item.get('detection_route')}" if item.get("detection_route") else "",
                    method_text=f" | methods={item.get('consensus_methods')}" if item.get("consensus_methods") else "",
                    vote_count=item.get("method_vote_count") if item.get("method_vote_count") not in (None, "") else "-",
                    volume=item.get("volume", "-"),
                    score=item.get("score", "-"),
                    hotspot=item.get("hotspot_count", "-"),
                )
                for item in snapshot.get("pocket_summary", [])[:4]
            ],
        ]),
    ]

    for index, (title, lines) in enumerate(sections):
        x = 32 + index * (section_width + 10)
        y = sections_y
        parts.append(f'<rect x="{x}" y="{y}" width="{section_width}" height="{section_height}" rx="18" fill="#ffffff" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 30}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="16" fill="#0f1724" font-weight="700">{_svg_escape(title)}</text>')
        body_y = y + 58
        if not lines:
            lines = ["无可用数据"]
        for line_index, line in enumerate(lines[:10]):
            line_y = body_y + line_index * 22
            parts.append(_svg_multiline_text(x + 16, line_y, str(line), font_size=13, fill="#334155", max_chars=48))

    bottom_y = 614
    parts.append(f'<rect x="32" y="{bottom_y}" width="{width - 64}" height="250" rx="18" fill="#ffffff" stroke="#e2e8f0"/>')
    parts.append(f'<text x="48" y="{bottom_y + 28}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="16" fill="#0f1724" font-weight="700">结果快照预览</text>')

    preview_lines = snapshot_to_summary_lines(snapshot)
    preview_lines.extend([
        f"注释行数: {snapshot.get('annotation_rows', 0)}",
        f"比较条目: {snapshot.get('comparison_rows', 0)}",
    ])
    for index, line in enumerate(preview_lines[:8]):
        parts.append(_svg_multiline_text(48, bottom_y + 58 + index * 22, line, font_size=13, fill="#334155", max_chars=90))

    parts.append(f'<text x="32" y="{height - 20}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#94a3b8">ProteinInsight snapshot export</text>')
    parts.append("</svg>")
    return "".join(parts).encode("utf-8")
