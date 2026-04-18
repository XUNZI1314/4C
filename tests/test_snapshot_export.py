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

    assert any("Consensus rerank release closure readiness: ledger-blocked / closed no" in line for line in summary_lines)
    assert any("Consensus rerank release closure blockers: 2 rows / top missing-evidence" in line for line in summary_lines)
    assert any("Consensus rerank release closure remediation checklist: available" in line for line in summary_lines)
    assert any("Consensus rerank release closure detached manifest: 3 files" in line for line in summary_lines)


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

    assert any("Top active-site decision: Review mapping before validation" in line for line in summary_lines)
    assert any("Top pocket precision tier: mapping-review" in line for line in summary_lines)
    assert any("AI evidence: 2 rows / status ok" in line for line in summary_lines)
    assert any("AI evidence used for ranking: 1 rows" in line for line in summary_lines)
    assert any("AI review decisions: 1 rows / applied 1 / status ok" in line for line in summary_lines)
    assert any("AI review decision validation: 2 rows / blocked 1" in line for line in summary_lines)
    assert any("AI review round: blocked / rankable 1" in line for line in summary_lines)
    assert any("AI review ranking delta: promoted / promoted 1, removed 0" in line for line in summary_lines)
    assert any("AI review artifact manifest: 8 files" in line for line in summary_lines)
    assert any("AI review bundle README: available" in line for line in summary_lines)
    assert any("AI review artifact bundle: available" in line for line in summary_lines)
    assert any("AI review bundle verification: 2 files / failed 0" in line for line in summary_lines)
    assert any("AI review bundle verification summary: verified" in line for line in summary_lines)
    assert any("AI review bundle certificate: available" in line for line in summary_lines)
    assert any("AI review decision outcomes: 1 rows" in line for line in summary_lines)
    assert any("AI review decision template rows: 2" in line for line in summary_lines)
    assert any("AI ranking influence: top-pocket-supported / Top pocket AI residues A:10" in line for line in summary_lines)
    assert any("AI evidence audit: supported 1, review 1" in line for line in summary_lines)
    assert any("AI evidence review queue: 1 rows / top fix missing-citation-or-snippet" in line for line in summary_lines)
    assert any("AI follow-up plan rows: 2" in line for line in summary_lines)
    assert any("Example enzyme active site catalytic residue" in line for line in summary_lines)
    assert any("Residue evidence consensus: 3 rows / top A:10 (validated-anchor" in line for line in summary_lines)
    assert any("Pocket consensus coverage: 2 rows / top Pocket-1 (consensus-validated-pocket" in line for line in summary_lines)
    assert any("Consensus rerank suggestions: 2 rows / top Pocket-1 (keep-prioritized" in line for line in summary_lines)
    assert any("Consensus rerank preview: 2 rows / top Pocket-1 (would-keep-priority" in line for line in summary_lines)
    assert any("Consensus rerank policy gate: no-change-needed / changed 0, blocked 0" in line for line in summary_lines)
    assert any("Consensus rerank action queue: 2 rows / top Pocket-1 (validation-anchor-ready, pass)" in line for line in summary_lines)
    assert any("Consensus rerank action checklist: available" in line for line in summary_lines)
    assert any("Consensus rerank apply simulation: 2 rows / top Pocket-1 (keep-current-ready, rank delta +0)" in line for line in summary_lines)
    assert any("Consensus rerank simulation delta: 2 rows / top Pocket-2 (frozen-blocker, rank delta -1)" in line for line in summary_lines)
    assert any("Consensus rerank precision scorecard: likely-precision-gain / score 72 / positive 2, blockers 0" in line for line in summary_lines)
    assert any("Consensus rerank precision guardrail: manual-review-ready / decision allow-after-review, mode manual-consensus-rerank" in line for line in summary_lines)
    assert any("Consensus rerank precision guardrail report: available" in line for line in summary_lines)
    assert any("Consensus rerank guardrail handoff bundle: available / manifest 13 files" in line for line in summary_lines)
    assert any("Consensus rerank guardrail bundle verification: verified / files 13, failed 0" in line for line in summary_lines)
    assert any("Consensus rerank guardrail handoff certificate: available" in line for line in summary_lines)
    assert any("Consensus rerank release decision template: 3 rows" in line for line in summary_lines)
    assert any("Consensus rerank release decisions: 3 rows / status ok" in line for line in summary_lines)
    assert any("Consensus rerank release decision validation: 3 rows / blocked 0" in line for line in summary_lines)
    assert any("Consensus rerank release review: approved-for-manual-release / allowed yes" in line for line in summary_lines)
    assert any("Consensus rerank release apply plan: 2 rows / top Pocket-1 (ready-for-manual-apply)" in line for line in summary_lines)
    assert any("Consensus rerank release apply report: available" in line for line in summary_lines)
    assert any("Consensus rerank release execution template: 2 rows" in line for line in summary_lines)
    assert any("Consensus rerank release execution receipt: 2 rows / status ok" in line for line in summary_lines)
    assert any("Consensus rerank release execution validation: 2 rows / blocked 0" in line for line in summary_lines)
    assert any("Consensus rerank release execution: executed / complete yes" in line for line in summary_lines)
    assert any("Consensus rerank release execution report: available" in line for line in summary_lines)
    assert any("Consensus rerank release closure certificate: available" in line for line in summary_lines)
    assert any("Consensus rerank release closure ledger: 7 rows / blocked 0" in line for line in summary_lines)
    assert any("Consensus rerank release closure readiness: closed-and-verified / closed yes" in line for line in summary_lines)
    assert any("Pocket reliability checks: pass 2, review 1, missing 1" in line for line in summary_lines)
    assert any("Evidence mapping risk: review" in line for line in summary_lines)
