import pandas as pd

from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms
from protein_visualizer.services.pdf_export import build_simple_pdf
from protein_visualizer.services.snapshot import (
    build_analysis_snapshot,
    build_snapshot_svg,
    snapshot_to_json_bytes,
    snapshot_to_summary_lines,
)
from protein_visualizer.services.structure_energy import estimate_protein_volume


def test_snapshot_exports_generate_json_svg_and_pdf():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    hotspot_df = identify_hotspots(energy_table)
    protein_volume = estimate_protein_volume(PDB_TEXT)

    snapshot = build_analysis_snapshot(
        energy_table,
        hotspot_df=hotspot_df,
        protein_volume=protein_volume,
        title="Snapshot Test",
    )

    json_bytes = snapshot_to_json_bytes(snapshot)
    svg_bytes = build_snapshot_svg(snapshot)
    pdf_bytes = build_simple_pdf("测试报告", snapshot=snapshot)

    assert b'"title": "Snapshot Test"' in json_bytes
    assert b"<svg" in svg_bytes
    assert "A³".encode("utf-8") in svg_bytes
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_snapshot_summary_lines_do_not_expose_mojibake_labels():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    pocket_summary = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "smart_rank_label": "优先验证",
                "evidence_quality_label": "direct-anchor",
                "evidence_quality_score": 0.91,
            }
        ]
    )
    joint_candidate_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "recommendation_label": "建议验证",
                "recommendation_action": "validate",
            }
        ]
    )
    snapshot = build_analysis_snapshot(
        energy_table,
        pocket_summary=pocket_summary,
        joint_candidate_df=joint_candidate_df,
        extra={
            "auto_detection_methods_used": "geometry-cluster",
            "auto_detection_status_summary": "geometry-cluster:used",
            "auto_detection_external_rows": 2,
            "auto_detection_external_sources": "M-CSA,UniProt",
        },
    )

    summary_lines = snapshot_to_summary_lines(snapshot)
    summary_text = "\n".join(summary_lines)

    for snippet in ["鑱", "鍙", "妫", "澶", "鎺", "鐑", "鏅"]:
        assert snippet not in summary_text
    assert "联合推荐条目数: 1" in summary_text
    assert "自动口袋方法: geometry-cluster" in summary_text
    assert "检测状态: geometry-cluster:used" in summary_text
    assert "外部位点证据: 2 (M-CSA,UniProt)" in summary_text
    assert "Top 口袋: Pocket-1 (优先验证)" in summary_text
    assert "Top 口袋证据质量: direct-anchor (0.910)" in summary_text
    assert "Top 联合动作: validate" in summary_text
    assert "Top 联合推荐: Pocket-1 (建议验证)" in summary_text


def test_snapshot_extra_preserves_nested_detection_payloads():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)

    snapshot = build_analysis_snapshot(
        energy_table,
        title="Detection Payload Snapshot",
        extra={
            "auto_detection_methods_used": "geometry-cluster",
            "auto_detection_status_summary": "geometry-cluster:used; consensus:single-method",
            "auto_detection_p2rank_status": "ok",
            "auto_detection_p2rank_prediction_rows": 2,
            "auto_detection_p2rank_residue_rows": 7,
            "p2rank_ab_enabled": True,
            "p2rank_ab_comparison": [{"pocket_id": "Pocket-1", "rank_delta": 1}],
            "auto_detection_external_rows": 2,
            "auto_detection_external_sources": "M-CSA,UniProt",
            "pocket_benchmark_reference_candidate_rows": 2,
            "pocket_benchmark_reference_import_summary_rows": 1,
            "pocket_benchmark_reference_import_status": "review-needed",
            "pocket_benchmark_reference_candidate_review_rows": 4,
            "pocket_benchmark_reference_candidate_review_p1_rows": 2,
            "pocket_benchmark_reference_candidate_review_p2_rows": 2,
            "pocket_benchmark_reference_candidate_review_checklist_available": True,
            "pocket_benchmark_reference_candidate_review_decision_rows": 4,
            "pocket_benchmark_reference_candidate_review_decision_validation_blocked_rows": 1,
            "pocket_benchmark_reference_candidate_review_outcome_accepted_rows": 3,
            "pocket_benchmark_reference_candidate_accepted_rows": 1,
            "pocket_benchmark_reference_is_provisional": True,
            "pocket_benchmark_reference_is_reviewed_candidate": False,
            "pocket_benchmark_reference_source_mode": "provisional-external-evidence",
            "pocket_benchmark_reference_source_audit_rows": 3,
            "pocket_benchmark_reference_source_audit_summary_rows": 1,
            "pocket_benchmark_reference_source_audit_summary_status": "blocked-provisional",
            "pocket_benchmark_reference_source_audit_summary_independent_claim_status": "no",
            "pocket_benchmark_reference_source_audit_action_queue_rows": 3,
            "pocket_benchmark_reference_source_audit_action_queue_blocker_rows": 3,
            "pocket_benchmark_reference_source_audit_action_queue_review_rows": 0,
            "pocket_benchmark_reference_source_audit_case_summary_rows": 2,
            "pocket_benchmark_reference_source_audit_case_summary_blocked_cases": 1,
            "pocket_benchmark_reference_source_audit_case_summary_review_cases": 0,
            "pocket_benchmark_reference_source_audit_case_checklist_available": True,
            "pocket_benchmark_reference_source_audit_case_decision_template_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_validation_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_validation_blocked_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_rows": 3,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_blocked_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_pending_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_cleared_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_status": "blocked",
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_open_cases": 2,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_blocker_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_review_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_top_status": "blocked",
            "pocket_benchmark_reference_source_audit_case_decision_closure_checklist_available": True,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_rows": 3,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_cleared_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_open_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_status": "blocked",
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases": 2,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_net_blocker_delta": -1,
            "pocket_benchmark_reference_source_audit_checklist_available": True,
            "pocket_benchmark_reference_source_claim_status": "blocked-provisional",
            "pocket_benchmark_reference_source_independent_claim_status": "no",
            "pocket_benchmark_reference_source_provisional_rows": 3,
            "pocket_benchmark_reference_source_reviewed_candidate_rows": 0,
            "pocket_benchmark_reference_rows": 3,
            "pocket_benchmark_reference_template_rows": 3,
            "pocket_benchmark_reference_template_notes_available": True,
            "pocket_benchmark_reference_quality_issue_rows": 2,
            "pocket_benchmark_reference_quality_summary_rows": 1,
            "pocket_benchmark_reference_quality_checklist_available": True,
            "pocket_benchmark_reference_structure_validation_issue_rows": 3,
            "pocket_benchmark_reference_structure_validation_summary_rows": 2,
            "pocket_benchmark_reference_structure_validation_checklist_available": True,
            "pocket_benchmark_reference_readiness_queue_rows": 8,
            "pocket_benchmark_reference_readiness_summary_rows": 1,
            "pocket_benchmark_reference_readiness_case_summary_rows": 4,
            "pocket_benchmark_reference_readiness_status": "blocked",
            "pocket_benchmark_reference_readiness_blocker_rows": 6,
            "pocket_benchmark_reference_readiness_review_rows": 2,
            "pocket_benchmark_reference_readiness_source_audit_issue_rows": 3,
            "pocket_benchmark_reference_readiness_blocked_cases": 1,
            "pocket_benchmark_reference_readiness_review_cases": 2,
            "pocket_benchmark_reference_readiness_checklist_available": True,
            "pocket_benchmark_interpretation_rows": 3,
            "pocket_benchmark_top1_claim_status": "blocked",
            "pocket_benchmark_top3_claim_status": "blocked",
            "pocket_benchmark_case_interpretation_rows": 6,
            "pocket_benchmark_case_interpretation_blocked_rows": 2,
            "pocket_benchmark_case_interpretation_review_rows": 1,
            "pocket_benchmark_case_interpretation_matrix_rows": 2,
            "pocket_benchmark_case_interpretation_matrix_blocked_rows": 1,
            "pocket_benchmark_case_interpretation_matrix_review_rows": 1,
            "pocket_benchmark_case_interpretation_matrix_summary_rows": 1,
            "pocket_benchmark_case_interpretation_matrix_summary_status": "blocked",
            "pocket_benchmark_case_interpretation_matrix_summary_usable_cases": 1,
            "pocket_benchmark_case_interpretation_matrix_queue_rows": 2,
            "pocket_benchmark_case_interpretation_matrix_queue_blocker_rows": 1,
            "pocket_benchmark_case_interpretation_matrix_queue_review_rows": 1,
            "pocket_benchmark_dataset_interpretation_rows": 3,
            "pocket_benchmark_dataset_interpretation_blocked_rows": 1,
            "pocket_benchmark_dataset_interpretation_review_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_rows": 3,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_blocker_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_review_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_mismatch_rows": 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_rows": 6,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_blocker_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_review_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_mismatch_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_rows": 3,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_blocker_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_review_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_mismatch_rows": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_rows": 3,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_action_count": 4,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_p0_rows": 2,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_mismatch_count": 1,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_priority": "P0",
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_source_impact": "source-gate-mismatch",
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_available": True,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_available": True,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_rows": 6,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_bytes": 12345,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_hash_rows": 6,
            "pocket_benchmark_dataset_interpretation_queue_rows": 2,
            "pocket_benchmark_dataset_interpretation_queue_blocker_rows": 1,
            "pocket_benchmark_dataset_interpretation_queue_review_rows": 1,
            "pocket_benchmark_dataset_interpretation_checklist_available": True,
            "pocket_benchmark_dataset_interpretation_report_available": True,
            "pocket_benchmark_top1_coverage": 0.667,
            "pocket_benchmark_top1_status": "top1-partial-hit",
            "pocket_benchmark_top3_coverage": 1.0,
            "pocket_benchmark_top3_status": "topn-complete-hit",
            "pocket_benchmark_case_summary_rows": 2,
            "pocket_benchmark_case_summary": [
                {"benchmark_id": "enzyme-a", "top_n": 1},
                {"benchmark_id": "enzyme-b", "top_n": 1},
            ],
            "pocket_benchmark_dataset_summary_rows": 2,
            "pocket_benchmark_variant_comparison_rows": 4,
            "pocket_benchmark_variant_case_comparison_rows": 4,
            "pocket_benchmark_variant_dataset_comparison_rows": 2,
            "pocket_benchmark_variant_detail_comparison_rows": 6,
            "pocket_benchmark_variant_remediation_rows": 2,
            "pocket_benchmark_variant_remediation_summary_rows": 1,
            "pocket_benchmark_variant_remediation_checklist_available": True,
            "auto_detection_metadata": {
                "methods_used": "geometry-cluster",
                "diagnostics": [{"method": "geometry-cluster", "status": "used"}],
            },
        },
    )

    summary_lines = snapshot_to_summary_lines(snapshot)
    json_bytes = snapshot_to_json_bytes(snapshot)

    assert snapshot["extra"]["auto_detection_metadata"]["diagnostics"][0]["method"] == "geometry-cluster"
    assert any("geometry-cluster" in line for line in summary_lines)
    assert any("P2Rank: ok" in line for line in summary_lines)
    assert any("P2Rank A/B: enabled / rows 1" in line for line in summary_lines)
    assert any("Benchmark reference candidate: 2 rows / import review-needed / provisional used yes" in line for line in summary_lines)
    assert any("Benchmark reference source: provisional-external-evidence / provisional yes / reviewed candidate no" in line for line in summary_lines)
    assert any("Benchmark reference source audit: 3 rows / claim status blocked-provisional / independent claim no" in line for line in summary_lines)
    assert any("Benchmark reference source audit summary: 1 rows / top status blocked-provisional / independent claim no" in line for line in summary_lines)
    assert any("Benchmark reference source audit action queue: 3 rows / blockers 3 / review 0" in line for line in summary_lines)
    assert any("Benchmark reference source audit cases: 2 rows / blocked 1 / review 0" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision template: 2 rows" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decisions: 2 rows / validation blocked 1" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision outcome summary: 1 rows / status blocked / open 2" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision closure queue: 2 rows / blockers 1 / review 1 / top blocked" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision readiness impact: 3 rows / cleared 1 / open 2" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision readiness impact summary: 1 rows / status blocked / open 2 / net blocker delta -1" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision closure checklist: available" in line for line in summary_lines)
    assert any("Benchmark reference source audit case decision outcomes: 3 rows / blocked 1 / pending 1 / cleared 1" in line for line in summary_lines)
    assert any("Benchmark reference source audit case checklist: available" in line for line in summary_lines)
    assert any("Benchmark reference source audit checklist: available" in line for line in summary_lines)
    assert any("Benchmark reference candidate review: 4 rows / P1 2 / P2 2 / checklist available" in line for line in summary_lines)
    assert any("Benchmark reference candidate review decisions: 4 rows / validation blocked 1 / accepted actions 3 / accepted references 1" in line for line in summary_lines)
    assert any("Catalytic pocket benchmark: references 3 / Top-1 0.667" in line for line in summary_lines)
    assert any("Benchmark reference template: 3 rows / notes available" in line for line in summary_lines)
    assert any("Benchmark reference curation quality: 2 issues / summary 1 rows / checklist available" in line for line in summary_lines)
    assert any("Benchmark reference structure validation: 3 issues / summary 2 rows / checklist available" in line for line in summary_lines)
    assert any("Benchmark reference readiness: blocked / blockers 6 / review 2 / source audit 3 / queue 8 / checklist available" in line for line in summary_lines)
    assert any("Benchmark reference readiness cases: 4 rows / blocked 1 / review 2" in line for line in summary_lines)
    assert any("Benchmark interpretation: 3 rows / Top-1 claim blocked / Top-3 claim blocked" in line for line in summary_lines)
    assert any("Benchmark case interpretation: 6 rows / blocked 2 / review 1" in line for line in summary_lines)
    assert any("Benchmark case interpretation matrix: 2 rows / blocked 1 / review 1" in line for line in summary_lines)
    assert any("Benchmark case interpretation matrix summary: blocked / usable 1" in line for line in summary_lines)
    assert any("Benchmark case interpretation matrix queue: 2 rows / blockers 1 / review 1" in line for line in summary_lines)
    assert any("Benchmark dataset interpretation: 3 rows / blocked 1 / review 1" in line for line in summary_lines)
    assert any("Benchmark source-audit decision dataset impact: 3 rows / blockers 1 / review 1 / mismatch 0" in line for line in summary_lines)
    assert any("Benchmark source-audit decision dataset impact cases: 6 rows / blockers 2 / review 1 / mismatch 1 / checklist available / report available" in line for line in summary_lines)
    assert any("Benchmark source-audit decision dataset impact action queue: 3 rows / blockers 2 / review 1 / mismatch 1" in line for line in summary_lines)
    assert any("Benchmark source-audit decision dataset impact action summary: 3 rows / actions 4 / P0 groups 2 / mismatches 1 / top P0 source-gate-mismatch" in line for line in summary_lines)
    assert any("Benchmark source-audit decision dataset impact artifacts: 6 files / bytes 12345 / hashes 6" in line for line in summary_lines)
    assert any("Benchmark dataset interpretation queue: 2 rows / blockers 1 / review 1 / checklist available / report available" in line for line in summary_lines)
    assert any("Catalytic benchmark dataset: cases 2 / summary rows 2" in line for line in summary_lines)
    assert any("Catalytic benchmark variants: 4 rows" in line for line in summary_lines)
    assert any("Catalytic benchmark variant cases: 4 rows / dataset rows 2" in line for line in summary_lines)
    assert any("Catalytic benchmark variant residues: 6 rows" in line for line in summary_lines)
    assert any("Catalytic benchmark remediation queue: 2 rows / summary 1 rows / checklist available" in line for line in summary_lines)
    assert b'"auto_detection_metadata": {' in json_bytes
    assert b'"diagnostics": [' in json_bytes


def test_snapshot_summary_lines_include_release_closure_blockers():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)

    snapshot = build_analysis_snapshot(
        energy_table,
        title="Closure Blocker Snapshot",
        extra={
            "consensus_rerank_release_closure_readiness_status": "ledger-blocked",
            "consensus_rerank_release_closed": False,
            "consensus_rerank_release_closure_blocker_rows": 2,
            "top_consensus_rerank_release_closure_blocker_type": "missing-evidence",
            "consensus_rerank_release_closure_remediation_checklist_available": True,
            "consensus_rerank_release_closure_detached_manifest_rows": 3,
        },
    )

    summary_lines = snapshot_to_summary_lines(snapshot)

    assert any("共识重排发布关闭就绪: 台账阻断 / 关闭 否" in line for line in summary_lines)
    assert any("共识重排发布关闭阻断项: 2 行 / Top 缺少证据" in line for line in summary_lines)
    assert any("共识重排发布关闭修复清单: 可用" in line for line in summary_lines)
    assert any("共识重排发布关闭外置清单: 3 个文件" in line for line in summary_lines)


def test_snapshot_summary_lines_include_pocket_reliability_audit():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)

    snapshot = build_analysis_snapshot(
        energy_table,
        title="Reliability Snapshot",
        extra={
            "top_pocket_decision_label": "Review mapping before validation",
            "top_pocket_decision_score": 0.512,
            "top_pocket_audit_status": "mapping-review-needed",
            "top_pocket_precision_tier": "mapping-review",
            "top_pocket_triage_action": "Review chain/numbering/mapping before validation.",
            "top_pocket_reliability_gaps": "Evidence mapping risk: review; Actionability: review",
            "ai_evidence_rows": 2,
            "ai_evidence_status": "ok",
            "ai_evidence_ranked_rows": 1,
            "ai_review_decision_rows": 1,
            "ai_review_decision_applied_rows": 1,
            "ai_review_decision_status": "ok",
            "ai_review_decision_validation_rows": 2,
            "ai_review_decision_validation_blocked_rows": 1,
            "ai_review_round_status": "blocked",
            "ai_review_round_rankable_rows": 1,
            "ai_review_ranking_effect_status": "promoted",
            "ai_review_ranking_promoted_rows": 1,
            "ai_review_ranking_removed_rows": 0,
            "ai_review_artifact_manifest_rows": 8,
            "ai_review_bundle_readme_available": True,
            "ai_review_artifact_bundle_available": True,
            "ai_review_bundle_verification_rows": 2,
            "ai_review_bundle_verification_failed_rows": 0,
            "ai_review_bundle_verification_status": "verified",
            "ai_review_bundle_certificate_available": True,
            "ai_review_decision_outcome_rows": 1,
            "ai_review_decision_template_rows": 2,
            "ai_influence_level": "top-pocket-supported",
            "top_pocket_ai_residues": "A:10",
            "ai_evidence_audit_supported_count": 1,
            "ai_evidence_audit_review_count": 1,
            "ai_evidence_review_queue_rows": 1,
            "top_ai_review_fix_type": "missing-citation-or-snippet",
            "ai_followup_plan_rows": 2,
            "top_ai_followup_query": "Example enzyme active site catalytic residue",
            "residue_evidence_consensus_rows": 3,
            "top_residue_consensus_anchor": "A:10",
            "top_residue_consensus_tier": "validated-anchor",
            "top_residue_consensus_score": 0.93,
            "top_residue_consensus_sources": "M-CSA, AI-Literature",
            "pocket_consensus_coverage_rows": 2,
            "top_pocket_consensus_coverage_id": "Pocket-1",
            "top_pocket_consensus_label": "consensus-validated-pocket",
            "top_pocket_consensus_anchor_count": 1,
            "top_pocket_consensus_best_score": 0.93,
            "consensus_rerank_suggestion_rows": 2,
            "top_consensus_rerank_pocket_id": "Pocket-1",
            "top_consensus_rerank_status": "keep-prioritized",
            "top_consensus_rerank_rank_delta": 0,
            "consensus_rerank_preview_rows": 2,
            "top_consensus_preview_pocket_id": "Pocket-1",
            "top_consensus_preview_decision": "would-keep-priority",
            "top_consensus_preview_rank_delta": 0,
            "top_consensus_preview_score": 0.72,
            "consensus_rerank_policy_status": "no-change-needed",
            "consensus_rerank_policy_changed_rows": 0,
            "consensus_rerank_policy_blocked_rows": 0,
            "consensus_rerank_action_queue_rows": 2,
            "top_consensus_rerank_action_pocket_id": "Pocket-1",
            "top_consensus_rerank_issue_type": "validation-anchor-ready",
            "top_consensus_rerank_issue_severity": "pass",
            "consensus_rerank_action_checklist_available": True,
            "consensus_rerank_apply_simulation_rows": 2,
            "top_consensus_rerank_apply_pocket_id": "Pocket-1",
            "top_consensus_rerank_apply_status": "keep-current-ready",
            "top_consensus_rerank_apply_rank_delta": 0,
            "consensus_rerank_simulation_delta_rows": 2,
            "top_consensus_rerank_delta_pocket_id": "Pocket-2",
            "top_consensus_rerank_delta_change_type": "frozen-blocker",
            "top_consensus_rerank_delta_rank_delta": -1,
            "consensus_rerank_precision_scorecard_rows": 1,
            "consensus_rerank_precision_score": 72,
            "consensus_rerank_precision_status": "likely-precision-gain",
            "consensus_rerank_positive_signal_rows": 2,
            "consensus_rerank_open_blocker_rows": 0,
            "consensus_rerank_precision_guardrail_rows": 1,
            "consensus_rerank_guardrail_status": "manual-review-ready",
            "consensus_rerank_guardrail_decision": "allow-after-review",
            "consensus_rerank_guardrail_apply_mode": "manual-consensus-rerank",
            "consensus_rerank_guardrail_report_available": True,
            "consensus_rerank_guardrail_artifact_manifest_rows": 13,
            "consensus_rerank_guardrail_handoff_zip_available": True,
            "consensus_rerank_guardrail_bundle_verification_rows": 13,
            "consensus_rerank_guardrail_bundle_verification_status": "verified",
            "consensus_rerank_guardrail_bundle_verification_failed_rows": 0,
            "consensus_rerank_guardrail_handoff_certificate_available": True,
            "consensus_rerank_release_decision_template_rows": 3,
            "consensus_rerank_release_decision_rows": 3,
            "consensus_rerank_release_decision_status": "ok",
            "consensus_rerank_release_decision_validation_rows": 3,
            "consensus_rerank_release_decision_blocked_rows": 0,
            "consensus_rerank_release_review_status": "approved-for-manual-release",
            "consensus_rerank_release_allowed": True,
            "consensus_rerank_release_apply_plan_rows": 2,
            "top_consensus_rerank_release_apply_pocket_id": "Pocket-1",
            "top_consensus_rerank_release_apply_status": "ready-for-manual-apply",
            "consensus_rerank_release_apply_report_available": True,
            "consensus_rerank_release_execution_template_rows": 2,
            "consensus_rerank_release_execution_receipt_rows": 2,
            "consensus_rerank_release_execution_receipt_status": "ok",
            "consensus_rerank_release_execution_validation_rows": 2,
            "consensus_rerank_release_execution_blocked_rows": 0,
            "consensus_rerank_release_execution_review_status": "executed",
            "consensus_rerank_release_execution_complete": True,
            "consensus_rerank_release_execution_report_available": True,
            "consensus_rerank_release_closure_certificate_available": True,
            "consensus_rerank_release_closure_ledger_rows": 7,
            "consensus_rerank_release_closure_ledger_blocked_rows": 0,
            "consensus_rerank_release_closure_summary_rows": 1,
            "consensus_rerank_release_closure_readiness_status": "closed-and-verified",
            "consensus_rerank_release_closed": True,
            "pocket_reliability": [
                {"pocket_id": "Pocket-1", "check": "Functional anchors", "status": "pass"},
                {"pocket_id": "Pocket-1", "check": "Evidence mapping risk", "status": "review"},
                {"pocket_id": "Pocket-1", "check": "Geometry consensus", "status": "pass"},
                {"pocket_id": "Pocket-1", "check": "Evidence A/B movement", "status": "missing"},
            ],
        },
    )

    summary_lines = snapshot_to_summary_lines(snapshot)

    assert any("Top 活性位点决策: 验证前复核映射 / 评分 0.512 / 审计 映射需复核" in line for line in summary_lines)
    assert any("Top 口袋精度分层: 映射需复核 / 动作 验证前复核链、编号和映射。" in line for line in summary_lines)
    assert any("AI 证据: 2 行 / 状态 正常" in line for line in summary_lines)
    assert any("AI 排名可用证据: 1 行" in line for line in summary_lines)
    assert any("AI 复核决策: 1 行 / 已应用 1 / 状态 正常" in line for line in summary_lines)
    assert any("AI 复核决策校验: 2 行 / 阻断 1" in line for line in summary_lines)
    assert any("AI 复核轮次: 阻断 / 可排名 1" in line for line in summary_lines)
    assert any("AI 复核排名变化: 提升 / 提升 1, 移除 0" in line for line in summary_lines)
    assert any("AI 复核产物清单: 8 个文件" in line for line in summary_lines)
    assert any("AI 复核包 README: 可用" in line for line in summary_lines)
    assert any("AI 复核产物包: 可用" in line for line in summary_lines)
    assert any("AI 复核包校验: 2 个文件 / 失败 0" in line for line in summary_lines)
    assert any("AI 复核包校验汇总: 已校验" in line for line in summary_lines)
    assert any("AI 复核包证书: 可用" in line for line in summary_lines)
    assert any("AI 复核决策结果: 1 行" in line for line in summary_lines)
    assert any("AI 复核决策模板行数: 2" in line for line in summary_lines)
    assert any("AI 排名影响: Top 口袋受支持 / Top 口袋 AI 残基 A:10" in line for line in summary_lines)
    assert any("AI 证据审计: 已支持 1, 复核 1" in line for line in summary_lines)
    assert any("AI 证据复核队列: 1 行 / Top 修复项 缺少引用或证据片段" in line for line in summary_lines)
    assert any("AI 后续取证计划: 2 行" in line for line in summary_lines)
    assert any("Example enzyme active site catalytic residue" in line for line in summary_lines)
    assert any("残基证据共识: 3 行 / Top A:10 (已验证锚点" in line for line in summary_lines)
    assert any("口袋共识覆盖: 2 行 / Top Pocket-1 (共识已验证口袋" in line for line in summary_lines)
    assert any("共识重排建议: 2 行 / Top Pocket-1 (保持优先" in line for line in summary_lines)
    assert any("共识重排预览: 2 行 / Top Pocket-1 (将保持优先" in line for line in summary_lines)
    assert any("共识重排策略门控: 无需变更 / 变化 0, 阻断 0" in line for line in summary_lines)
    assert any("共识重排行动队列: 2 行 / Top Pocket-1 (验证锚点就绪, 通过)" in line for line in summary_lines)
    assert any("共识重排行动清单: 可用" in line for line in summary_lines)
    assert any("共识重排应用模拟: 2 行 / Top Pocket-1 (保留当前且就绪, 排名变化 +0)" in line for line in summary_lines)
    assert any("共识重排模拟变化: 2 行 / Top Pocket-2 (冻结阻断, 排名变化 -1)" in line for line in summary_lines)
    assert any("共识重排精度评分卡: 可能提升精度 / 评分 72 / 正向 2, 阻断项 0" in line for line in summary_lines)
    assert any("共识重排精度护栏: 人工复核就绪 / 决策 复核后允许, 模式 人工共识重排" in line for line in summary_lines)
    assert any("共识重排精度护栏报告: 可用" in line for line in summary_lines)
    assert any("共识重排护栏交接包: 可用 / 清单 13 个文件" in line for line in summary_lines)
    assert any("共识重排护栏包校验: 已校验 / 文件数 13, 失败 0" in line for line in summary_lines)
    assert any("共识重排护栏交接证书: 可用" in line for line in summary_lines)
    assert any("共识重排发布决策模板: 3 行" in line for line in summary_lines)
    assert any("共识重排发布决策: 3 行 / 状态 正常" in line for line in summary_lines)
    assert any("共识重排发布决策校验: 3 行 / 阻断 0" in line for line in summary_lines)
    assert any("共识重排发布复核: 批准人工发布 / 允许 是" in line for line in summary_lines)
    assert any("共识重排发布应用计划: 2 行 / Top Pocket-1 (可人工应用)" in line for line in summary_lines)
    assert any("共识重排发布应用报告: 可用" in line for line in summary_lines)
    assert any("共识重排发布执行模板: 2 行" in line for line in summary_lines)
    assert any("共识重排发布执行回执: 2 行 / 状态 正常" in line for line in summary_lines)
    assert any("共识重排发布执行校验: 2 行 / 阻断 0" in line for line in summary_lines)
    assert any("共识重排发布执行: 已执行 / 完成 是" in line for line in summary_lines)
    assert any("共识重排发布执行报告: 可用" in line for line in summary_lines)
    assert any("共识重排发布关闭证书: 可用" in line for line in summary_lines)
    assert any("共识重排发布关闭台账: 7 行 / 阻断 0" in line for line in summary_lines)
    assert any("共识重排发布关闭就绪: 已关闭并校验 / 关闭 是" in line for line in summary_lines)
    assert any("口袋可靠性检查: 通过 2, 复核 1, 缺失 1" in line for line in summary_lines)
    assert any("证据映射风险: 复核" in line for line in summary_lines)
    joined_summary = "\n".join(summary_lines)
    for forbidden_label in [
        "Top active-site decision:",
        "Top pocket precision tier:",
        "Pocket reliability checks:",
        "Top pocket reliability gaps:",
        "Residue evidence consensus:",
        "Pocket consensus coverage:",
        "Consensus rerank",
    ]:
        assert forbidden_label not in joined_summary
