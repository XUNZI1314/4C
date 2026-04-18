import hashlib
import zipfile
from io import BytesIO

import pandas as pd

from protein_visualizer.services.pocket_decision import (
    add_pocket_residue_layers,
    build_consensus_rerank_guardrail_artifact_manifest,
    build_consensus_rerank_guardrail_handoff_zip,
    build_consensus_rerank_guardrail_handoff_certificate_markdown,
    build_consensus_rerank_guardrail_bundle_verification_summary,
    build_consensus_rerank_action_checklist_markdown,
    build_consensus_rerank_action_queue,
    build_consensus_rerank_apply_simulation,
    build_consensus_rerank_policy_gate,
    build_consensus_rerank_preview,
    build_consensus_rerank_precision_guardrail,
    build_consensus_rerank_precision_guardrail_report_markdown,
    build_consensus_rerank_precision_scorecard,
    build_consensus_rerank_release_apply_plan,
    build_consensus_rerank_release_apply_report_markdown,
    build_consensus_rerank_release_closure_blocker_queue,
    build_consensus_rerank_release_closure_certificate_markdown,
    build_consensus_rerank_release_closure_detached_manifest,
    build_consensus_rerank_release_closure_ledger,
    build_consensus_rerank_release_closure_remediation_checklist_markdown,
    build_consensus_rerank_release_closure_summary,
    build_consensus_rerank_release_decision_summary,
    build_consensus_rerank_release_decision_template,
    build_consensus_rerank_release_execution_report_markdown,
    build_consensus_rerank_release_execution_summary,
    build_consensus_rerank_release_execution_template,
    build_consensus_rerank_simulation_delta,
    build_consensus_rerank_suggestion,
    build_pocket_decision_table,
    build_pocket_precision_triage,
    build_pocket_reliability_checklist,
    parse_consensus_rerank_release_decision_table,
    parse_consensus_rerank_release_execution_table,
    validate_consensus_rerank_release_execution_receipt,
    validate_consensus_rerank_release_decisions,
    verify_consensus_rerank_guardrail_handoff_zip,
)


def test_build_pocket_decision_table_prioritizes_direct_anchor_candidate():
    pocket_summary = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Direct",
                "smart_rank_score": 0.66,
                "smart_rank_label": "promising",
                "consensus_score": 0.62,
                "method_vote_count": 2,
                "hotspot_count": 1,
                "evidence_quality_label": "direct-anchor",
                "evidence_quality_score": 0.72,
                "external_direct_anchor_count": 1,
                "evidence_route_anchor_count": 1,
                "evidence_anchor_residues": "A:10",
                "external_exact_match_ratio": 0.25,
                "external_support_mean": 0.48,
                "external_mapping_quality_mean": 0.82,
                "external_structure_verified_count": 1,
                "smart_evidence_anchor_support": 0.78,
                "smart_evidence_anchor_risk": 0.04,
                "external_direct_sources": "M-CSA",
                "external_evidence_types": "Catalytic residue",
                "consensus_methods": "external-evidence, geometry-cluster",
            },
            {
                "pocket_id": "Pocket-Geometry",
                "smart_rank_score": 0.71,
                "smart_rank_label": "promising",
                "consensus_score": 0.70,
                "method_vote_count": 2,
                "hotspot_count": 0,
                "evidence_quality_label": "geometry-only",
                "evidence_quality_score": 0.0,
                "external_direct_anchor_count": 0,
                "evidence_route_anchor_count": 0,
                "smart_evidence_anchor_support": 0.0,
                "smart_evidence_anchor_risk": 0.0,
                "consensus_methods": "geometry-cluster",
            },
        ]
    )
    joint_candidate_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Direct",
                "recommendation_action": "validate-prioritize",
                "evidence_anchor_support": 0.80,
                "evidence_anchor_risk": 0.02,
            }
        ]
    )
    evidence_route_ab_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Direct",
                "status": "moved_up",
                "rank_delta": 1,
                "score_delta": 0.08,
                "evidence_quality_delta": 0.30,
            }
        ]
    )

    result = build_pocket_decision_table(
        pocket_summary,
        joint_candidate_df,
        evidence_route_ab_df=evidence_route_ab_df,
    )

    assert result.iloc[0]["pocket_id"] == "Pocket-Direct"
    assert result.iloc[0]["decision_label"] == "Evidence-led active-site candidate"
    assert result.iloc[0]["audit_status"] == "ready-to-validate"
    assert result.iloc[0]["evidence_route_rank_delta"] == 1
    assert "M-CSA" in result.iloc[0]["supporting_evidence"]
    assert result[result["pocket_id"] == "Pocket-Geometry"].iloc[0]["risk_flags"] == "needs-functional-evidence, geometry-dominated"


def test_build_pocket_reliability_checklist_marks_strong_candidate_passes():
    decision_df = pd.DataFrame(
        [
            {
                "decision_rank": 1,
                "pocket_id": "Pocket-Direct",
                "decision_score": 0.76,
                "functional_confidence": 0.82,
                "geometry_confidence": 0.62,
                "recommended_action": "validate-prioritize",
                "audit_status": "ready-to-validate",
                "evidence_quality_label": "direct-anchor",
                "direct_anchor_count": 2,
                "route_anchor_count": 1,
                "method_vote_count": 2,
                "literature_rank_delta": 1,
                "evidence_route_rank_delta": 0,
                "conservation_rank_delta": 0,
                "risk_flags": "none",
                "next_step": "Prioritize residue-level validation around direct anchors.",
            }
        ]
    )

    checklist = build_pocket_reliability_checklist(decision_df, max_pockets=1)

    assert checklist["check"].tolist() == [
        "Functional anchors",
        "Evidence mapping risk",
        "Geometry consensus",
        "Evidence A/B movement",
        "Actionability",
    ]
    assert set(checklist["status"]) == {"pass"}
    assert "direct=2" in checklist.loc[checklist["check"] == "Functional anchors", "signal"].iloc[0]


def test_build_pocket_precision_triage_marks_validation_ready_candidate():
    decision_df = pd.DataFrame(
        [
            {
                "decision_rank": 1,
                "pocket_id": "Pocket-Direct",
                "decision_score": 0.76,
                "functional_confidence": 0.82,
                "geometry_confidence": 0.62,
                "recommended_action": "validate-prioritize",
                "audit_status": "ready-to-validate",
                "evidence_quality_label": "direct-anchor",
                "direct_anchor_count": 2,
                "route_anchor_count": 1,
                "method_vote_count": 2,
                "literature_rank_delta": 1,
                "evidence_route_rank_delta": 0,
                "conservation_rank_delta": 0,
                "risk_flags": "none",
            }
        ]
    )
    checklist = build_pocket_reliability_checklist(decision_df, max_pockets=1)

    triage = build_pocket_precision_triage(decision_df, checklist, max_pockets=1)

    assert triage.iloc[0]["precision_tier"] == "validation-ready"
    assert triage.iloc[0]["triage_priority"] == 1
    assert triage.iloc[0]["blocking_checks"] == "none"
    assert triage.iloc[0]["next_data_to_add"] == "No additional evidence required before validation."


def test_build_pocket_decision_table_flags_neighborhood_expansion_risk():
    pocket_summary = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Expanded",
                "smart_rank_score": 0.62,
                "consensus_score": 0.58,
                "method_vote_count": 1,
                "evidence_quality_label": "neighborhood-expanded",
                "evidence_quality_score": 0.38,
                "external_support_mean": 0.42,
                "external_mapping_quality_mean": 0.60,
                "smart_evidence_anchor_support": 0.20,
                "smart_evidence_anchor_risk": 0.56,
                "external_evidence_total": 2,
            }
        ]
    )

    result = build_pocket_decision_table(pocket_summary)

    assert result.iloc[0]["decision_label"] == "Review mapping before validation"
    assert result.iloc[0]["audit_status"] == "mapping-review-needed"
    assert "neighborhood-expansion-risk" in result.iloc[0]["risk_flags"]
    assert "Inspect direct anchors" in result.iloc[0]["next_step"]


def test_build_pocket_reliability_checklist_exposes_geometry_only_gaps():
    decision_df = pd.DataFrame(
        [
            {
                "decision_rank": 1,
                "pocket_id": "Pocket-Geometry",
                "decision_score": 0.31,
                "functional_confidence": 0.18,
                "geometry_confidence": 0.58,
                "recommended_action": "shortlist-follow-up",
                "audit_status": "needs-functional-evidence",
                "evidence_quality_label": "geometry-only",
                "direct_anchor_count": 0,
                "route_anchor_count": 0,
                "method_vote_count": 2,
                "literature_rank_delta": 0,
                "evidence_route_rank_delta": 0,
                "conservation_rank_delta": 0,
                "risk_flags": "needs-functional-evidence, geometry-dominated",
                "next_step": "Add UniProt/M-CSA/literature or manual key residues.",
            }
        ]
    )

    checklist = build_pocket_reliability_checklist(decision_df, max_pockets=1)
    statuses = dict(zip(checklist["check"], checklist["status"]))

    assert statuses["Functional anchors"] == "missing"
    assert statuses["Evidence mapping risk"] == "missing"
    assert statuses["Geometry consensus"] == "pass"
    assert statuses["Evidence A/B movement"] == "missing"
    assert statuses["Actionability"] == "missing"


def test_build_pocket_precision_triage_blocks_geometry_only_candidate_on_evidence_gap():
    decision_df = pd.DataFrame(
        [
            {
                "decision_rank": 1,
                "pocket_id": "Pocket-Geometry",
                "decision_score": 0.31,
                "functional_confidence": 0.18,
                "geometry_confidence": 0.58,
                "recommended_action": "shortlist-follow-up",
                "audit_status": "needs-functional-evidence",
                "evidence_quality_label": "geometry-only",
                "direct_anchor_count": 0,
                "route_anchor_count": 0,
                "method_vote_count": 2,
                "literature_rank_delta": 0,
                "evidence_route_rank_delta": 0,
                "conservation_rank_delta": 0,
                "risk_flags": "needs-functional-evidence, geometry-dominated",
            }
        ]
    )
    checklist = build_pocket_reliability_checklist(decision_df, max_pockets=1)

    triage = build_pocket_precision_triage(decision_df, checklist, max_pockets=1)

    assert triage.iloc[0]["precision_tier"] == "evidence-gap"
    assert "Functional anchors" in triage.iloc[0]["blocking_checks"]
    assert "Evidence mapping risk" in triage.iloc[0]["blocking_checks"]
    assert "M-CSA" in triage.iloc[0]["next_data_to_add"]


def test_build_consensus_rerank_suggestion_promotes_validated_pocket_and_blocks_ai():
    decision_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-Geometry", "decision_rank": 1, "decision_score": 0.68},
            {"pocket_id": "Pocket-Validated", "decision_rank": 3, "decision_score": 0.52},
            {"pocket_id": "Pocket-Blocked", "decision_rank": 2, "decision_score": 0.60},
        ]
    )
    coverage_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Validated",
                "pocket_consensus_label": "consensus-validated-pocket",
                "rank_safe_anchor_count": 1,
                "best_consensus_score": 0.93,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "consensus_anchor_residues": "A:10",
            },
            {
                "pocket_id": "Pocket-Blocked",
                "pocket_consensus_label": "blocked-ai-evidence-pocket",
                "rank_safe_anchor_count": 0,
                "best_consensus_score": 0.42,
                "blocked_ai_count": 1,
                "weak_mapping_count": 0,
                "consensus_anchor_residues": "A:30",
            },
            {
                "pocket_id": "Pocket-Geometry",
                "pocket_consensus_label": "no-consensus-anchor-pocket",
                "rank_safe_anchor_count": 0,
                "best_consensus_score": 0.0,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "consensus_anchor_residues": "none",
            },
        ]
    )

    suggestions = build_consensus_rerank_suggestion(decision_df, coverage_df)

    statuses = dict(zip(suggestions["pocket_id"], suggestions["suggestion_status"]))
    assert statuses["Pocket-Validated"] == "promote-consensus"
    assert statuses["Pocket-Blocked"] == "demote-or-block"
    assert statuses["Pocket-Geometry"] == "evidence-gap-review"
    assert int(suggestions[suggestions["pocket_id"] == "Pocket-Validated"].iloc[0]["rank_delta"]) == 2


def test_build_consensus_rerank_preview_simulates_promote_and_demote():
    decision_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-Geometry", "decision_rank": 1, "decision_score": 0.68},
            {"pocket_id": "Pocket-Validated", "decision_rank": 3, "decision_score": 0.52},
            {"pocket_id": "Pocket-Blocked", "decision_rank": 2, "decision_score": 0.60},
        ]
    )
    suggestions = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Validated",
                "current_decision_rank": 3,
                "rank_delta": 2,
                "current_decision_score": 0.52,
                "pocket_consensus_label": "consensus-validated-pocket",
                "rank_safe_anchor_count": 1,
                "best_consensus_score": 0.93,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "suggestion_status": "promote-consensus",
                "recommended_action": "Promote after boundary review.",
                "consensus_anchor_residues": "A:10",
            },
            {
                "pocket_id": "Pocket-Blocked",
                "current_decision_rank": 2,
                "rank_delta": 0,
                "current_decision_score": 0.60,
                "pocket_consensus_label": "blocked-ai-evidence-pocket",
                "rank_safe_anchor_count": 0,
                "best_consensus_score": 0.42,
                "blocked_ai_count": 1,
                "weak_mapping_count": 0,
                "suggestion_status": "demote-or-block",
                "recommended_action": "Do not promote.",
                "consensus_anchor_residues": "A:30",
            },
            {
                "pocket_id": "Pocket-Geometry",
                "current_decision_rank": 1,
                "rank_delta": 0,
                "current_decision_score": 0.68,
                "pocket_consensus_label": "no-consensus-anchor-pocket",
                "rank_safe_anchor_count": 0,
                "best_consensus_score": 0.0,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "suggestion_status": "evidence-gap-review",
                "recommended_action": "Collect evidence.",
                "consensus_anchor_residues": "none",
            },
        ]
    )

    preview = build_consensus_rerank_preview(decision_df, suggestions)

    by_pocket = preview.set_index("pocket_id")
    assert by_pocket.loc["Pocket-Validated", "preview_rank"] < by_pocket.loc["Pocket-Validated", "current_rank"]
    assert by_pocket.loc["Pocket-Validated", "preview_decision"] == "would-move-up"
    assert by_pocket.loc["Pocket-Blocked", "preview_decision"] == "would-demote-or-block"
    assert float(by_pocket.loc["Pocket-Validated", "consensus_adjustment"]) > 0.0
    assert float(by_pocket.loc["Pocket-Blocked", "consensus_adjustment"]) < 0.0


def test_build_consensus_rerank_policy_gate_blocks_risky_preview():
    preview = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Validated",
                "preview_rank_delta": 2,
                "preview_score": 0.72,
                "suggestion_status": "promote-consensus",
                "preview_decision": "would-move-up",
                "rank_safe_anchor_count": 1,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
            },
            {
                "pocket_id": "Pocket-Blocked",
                "preview_rank_delta": -1,
                "preview_score": 0.40,
                "suggestion_status": "demote-or-block",
                "preview_decision": "would-demote-or-block",
                "rank_safe_anchor_count": 0,
                "blocked_ai_count": 1,
                "weak_mapping_count": 0,
            },
        ]
    )

    gate = build_consensus_rerank_policy_gate(preview)
    row = gate.iloc[0]

    assert row["policy_status"] == "blocked"
    assert int(row["changed_rows"]) == 2
    assert int(row["blocked_rows"]) == 1
    assert "blocked_ai=1" in row["blocking_reasons"]


def test_build_consensus_rerank_action_queue_lists_blockers_first():
    preview = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Validated",
                "current_rank": 3,
                "preview_rank": 1,
                "preview_rank_delta": 2,
                "preview_score": 0.72,
                "suggestion_status": "promote-consensus",
                "preview_decision": "would-move-up",
                "pocket_consensus_label": "consensus-validated-pocket",
                "rank_safe_anchor_count": 1,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "recommended_action": "Promote after review.",
                "consensus_anchor_residues": "A:10",
            },
            {
                "pocket_id": "Pocket-Blocked",
                "current_rank": 2,
                "preview_rank": 3,
                "preview_rank_delta": -1,
                "preview_score": 0.40,
                "suggestion_status": "demote-or-block",
                "preview_decision": "would-demote-or-block",
                "pocket_consensus_label": "blocked-ai-evidence-pocket",
                "rank_safe_anchor_count": 0,
                "blocked_ai_count": 1,
                "weak_mapping_count": 0,
                "recommended_action": "Do not promote.",
                "consensus_anchor_residues": "A:30",
            },
        ]
    )
    gate = build_consensus_rerank_policy_gate(preview)

    queue = build_consensus_rerank_action_queue(preview, gate)

    assert queue.iloc[0]["pocket_id"] == "Pocket-Blocked"
    assert queue.iloc[0]["issue_type"] == "blocked-ai-evidence"
    assert queue.iloc[0]["policy_status"] == "blocked"
    assert bool(queue.iloc[0]["can_apply_after_fix"]) is False
    assert queue[queue["pocket_id"] == "Pocket-Validated"].iloc[0]["issue_type"] == "promotion-review"


def test_build_consensus_rerank_action_checklist_markdown_exports_queue_context():
    queue = pd.DataFrame(
        [
            {
                "action_priority": 1,
                "pocket_id": "Pocket-Blocked",
                "issue_type": "blocked-ai-evidence",
                "issue_severity": "blocking",
                "current_rank": 2,
                "preview_rank": 3,
                "preview_rank_delta": -1,
                "rank_safe_anchor_count": 0,
                "blocked_ai_count": 1,
                "weak_mapping_count": 0,
                "required_fix": "Reject or repair blocked AI residue evidence.",
                "can_apply_after_fix": False,
                "policy_status": "blocked",
                "consensus_anchor_residues": "A:30",
                "recommended_action": "Do not apply rerank yet.",
            }
        ]
    )
    gate = pd.DataFrame(
        [
            {
                "policy_status": "blocked",
                "blocking_reasons": "blocked_ai=1",
                "recommended_action": "Do not enable automatic consensus rerank yet.",
            }
        ]
    )

    markdown = build_consensus_rerank_action_checklist_markdown(queue, gate)

    assert markdown.startswith("# Consensus rerank action checklist")
    assert "- Policy status: `blocked`" in markdown
    assert "## 1. Pocket-Blocked - blocked-ai-evidence" in markdown
    assert "- [ ] Required fix: Reject or repair blocked AI residue evidence." in markdown
    assert "- [ ] Verify consensus anchors: A:30" in markdown
    assert "Ranks: current `2` -> preview `3` (delta -1)" in markdown
    assert "Can apply after fix: no" in markdown


def test_build_consensus_rerank_apply_simulation_keeps_blockers_conservative():
    preview = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Validated",
                "current_rank": 3,
                "preview_rank": 1,
                "preview_rank_delta": 2,
                "current_decision_score": 0.52,
                "preview_score": 0.72,
                "suggestion_status": "promote-consensus",
                "preview_decision": "would-move-up",
                "pocket_consensus_label": "consensus-validated-pocket",
                "rank_safe_anchor_count": 1,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "recommended_action": "Promote after review.",
                "consensus_anchor_residues": "A:10",
            },
            {
                "pocket_id": "Pocket-Blocked",
                "current_rank": 2,
                "preview_rank": 3,
                "preview_rank_delta": -1,
                "current_decision_score": 0.60,
                "preview_score": 0.40,
                "suggestion_status": "demote-or-block",
                "preview_decision": "would-demote-or-block",
                "pocket_consensus_label": "blocked-ai-evidence-pocket",
                "rank_safe_anchor_count": 0,
                "blocked_ai_count": 1,
                "weak_mapping_count": 0,
                "recommended_action": "Do not promote.",
                "consensus_anchor_residues": "A:30",
            },
            {
                "pocket_id": "Pocket-Geometry",
                "current_rank": 1,
                "preview_rank": 2,
                "preview_rank_delta": -1,
                "current_decision_score": 0.68,
                "preview_score": 0.625,
                "suggestion_status": "evidence-gap-review",
                "preview_decision": "would-move-down",
                "pocket_consensus_label": "no-consensus-anchor-pocket",
                "rank_safe_anchor_count": 0,
                "blocked_ai_count": 0,
                "weak_mapping_count": 0,
                "recommended_action": "Collect evidence.",
                "consensus_anchor_residues": "none",
            },
        ]
    )
    gate = build_consensus_rerank_policy_gate(preview)
    queue = build_consensus_rerank_action_queue(preview, gate)

    simulation = build_consensus_rerank_apply_simulation(preview, queue, gate)

    assert simulation.iloc[0]["pocket_id"] == "Pocket-Validated"
    assert simulation.iloc[0]["apply_status"] == "candidate-after-fix"
    assert bool(simulation.iloc[0]["policy_allows_apply"]) is False
    assert int(simulation.iloc[0]["simulated_rank_delta"]) == 2
    blocked = simulation[simulation["pocket_id"] == "Pocket-Blocked"].iloc[0]
    assert blocked["apply_status"] == "blocked-currently"
    assert blocked["simulation_score_source"] == "conservative-min-score"
    assert float(blocked["simulation_score"]) == 0.40


def test_build_consensus_rerank_simulation_delta_explains_rank_changes_and_freezes():
    simulation = pd.DataFrame(
        [
            {
                "simulated_rank": 1,
                "pocket_id": "Pocket-Validated",
                "current_rank": 3,
                "simulated_rank_delta": 2,
                "current_decision_score": 0.52,
                "simulation_score": 0.72,
                "apply_status": "candidate-after-fix",
                "apply_decision": "would-rank-up",
                "policy_status": "blocked",
                "issue_type": "promotion-review",
                "issue_severity": "review",
                "required_before_apply": "Review preview rank changes manually.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after review.",
            },
            {
                "simulated_rank": 3,
                "pocket_id": "Pocket-Blocked",
                "current_rank": 2,
                "simulated_rank_delta": -1,
                "current_decision_score": 0.60,
                "simulation_score": 0.40,
                "apply_status": "blocked-currently",
                "apply_decision": "would-rank-down",
                "policy_status": "blocked",
                "issue_type": "blocked-ai-evidence",
                "issue_severity": "blocking",
                "required_before_apply": "Reject or repair blocked AI residue evidence.",
                "consensus_anchor_residues": "A:30",
                "recommended_action": "Do not promote.",
            },
        ]
    )

    delta = build_consensus_rerank_simulation_delta(simulation)

    assert delta.iloc[0]["pocket_id"] == "Pocket-Blocked"
    assert delta.iloc[0]["change_type"] == "frozen-blocker"
    assert delta.iloc[0]["change_severity"] == "blocking"
    assert "Precision is protected" in delta.iloc[0]["precision_interpretation"]
    promoted = delta[delta["pocket_id"] == "Pocket-Validated"].iloc[0]
    assert promoted["change_type"] == "rank-up-candidate"
    assert promoted["rank_delta"] == 2
    assert promoted["score_delta"] == 0.20
    assert "A:10" in promoted["explanation"]


def test_build_consensus_rerank_precision_scorecard_summarizes_gain_and_blockers():
    delta = pd.DataFrame(
        [
            {
                "impact_priority": 3,
                "pocket_id": "Pocket-Validated",
                "change_type": "rank-up-candidate",
                "change_severity": "review",
                "current_rank": 3,
                "rank_delta": 2,
                "score_delta": 0.20,
                "apply_status": "candidate-after-fix",
                "issue_type": "promotion-review",
                "issue_severity": "review",
                "policy_status": "blocked",
            },
            {
                "impact_priority": 1,
                "pocket_id": "Pocket-Blocked",
                "change_type": "frozen-blocker",
                "change_severity": "blocking",
                "current_rank": 2,
                "rank_delta": -1,
                "score_delta": -0.20,
                "apply_status": "blocked-currently",
                "issue_type": "blocked-ai-evidence",
                "issue_severity": "blocking",
                "policy_status": "blocked",
            },
            {
                "impact_priority": 4,
                "pocket_id": "Pocket-Geometry",
                "change_type": "rank-down-conservative",
                "change_severity": "review",
                "current_rank": 1,
                "rank_delta": -1,
                "score_delta": -0.05,
                "apply_status": "evidence-required",
                "issue_type": "functional-evidence-gap",
                "issue_severity": "missing-evidence",
                "policy_status": "blocked",
            },
        ]
    )
    simulation = pd.DataFrame(
        [
            {"pocket_id": "Pocket-Validated", "policy_allows_apply": False},
            {"pocket_id": "Pocket-Blocked", "policy_allows_apply": False},
            {"pocket_id": "Pocket-Geometry", "policy_allows_apply": False},
        ]
    )

    scorecard = build_consensus_rerank_precision_scorecard(delta, simulation)
    row = scorecard.iloc[0]

    assert row["scorecard_status"] == "promising-but-blocked"
    assert int(row["positive_signal_rows"]) == 2
    assert int(row["open_blocker_rows"]) == 2
    assert int(row["rank_up_rows"]) == 1
    assert int(row["rank_down_rows"]) == 1
    assert row["top_positive_pocket_id"] == "Pocket-Validated"
    assert row["top_blocker_pocket_id"] == "Pocket-Blocked"
    assert "positive=2" in row["score_reason"]


def test_build_consensus_rerank_precision_guardrail_blocks_when_clearance_required():
    scorecard = pd.DataFrame(
        [
            {
                "scorecard_status": "promising-but-blocked",
                "precision_improvement_score": 44,
                "positive_signal_rows": 2,
                "open_blocker_rows": 2,
                "policy_status": "blocked",
                "top_positive_pocket_id": "Pocket-Validated",
                "top_blocker_pocket_id": "Pocket-Blocked",
                "recommended_action": "Resolve blockers.",
            }
        ]
    )
    gate = pd.DataFrame(
        [
            {
                "policy_status": "blocked",
                "recommended_action": "Do not enable automatic consensus rerank.",
            }
        ]
    )
    queue = pd.DataFrame(
        [
            {
                "action_priority": 1,
                "pocket_id": "Pocket-Blocked",
                "issue_type": "blocked-ai-evidence",
                "issue_severity": "blocking",
                "required_fix": "Reject or repair blocked AI residue evidence.",
                "can_apply_after_fix": False,
            }
        ]
    )

    guardrail = build_consensus_rerank_precision_guardrail(scorecard, gate, queue)
    row = guardrail.iloc[0]

    assert row["guardrail_status"] == "blocked"
    assert row["guardrail_decision"] == "do-not-apply"
    assert row["apply_mode"] == "diagnostic-only"
    assert bool(row["can_enable_auto_rerank"]) is False
    assert bool(row["can_apply_after_manual_review"]) is False
    assert int(row["required_clearance_count"]) == 2
    assert "Pocket-Blocked / blocked-ai-evidence" in row["first_required_clearance"]


def test_build_consensus_rerank_precision_guardrail_allows_manual_review_when_clean():
    scorecard = pd.DataFrame(
        [
            {
                "scorecard_status": "likely-precision-gain",
                "precision_improvement_score": 78,
                "positive_signal_rows": 2,
                "open_blocker_rows": 0,
                "policy_status": "review-before-apply",
                "top_positive_pocket_id": "Pocket-Validated",
                "top_blocker_pocket_id": "none",
                "recommended_action": "Review delta before applying.",
            }
        ]
    )
    gate = pd.DataFrame(
        [
            {
                "policy_status": "review-before-apply",
                "recommended_action": "Review preview rank changes manually.",
            }
        ]
    )

    guardrail = build_consensus_rerank_precision_guardrail(scorecard, gate)
    row = guardrail.iloc[0]

    assert row["guardrail_status"] == "manual-review-ready"
    assert row["guardrail_decision"] == "allow-after-review"
    assert row["apply_mode"] == "manual-consensus-rerank"
    assert bool(row["can_enable_auto_rerank"]) is False
    assert bool(row["can_apply_after_manual_review"]) is True
    assert bool(row["manual_review_required"]) is True
    assert int(row["required_clearance_count"]) == 0


def test_build_consensus_rerank_precision_guardrail_report_markdown_exports_handoff():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "blocked",
                "guardrail_decision": "do-not-apply",
                "apply_mode": "diagnostic-only",
                "can_enable_auto_rerank": False,
                "can_apply_after_manual_review": False,
                "manual_review_required": True,
                "precision_improvement_score": 44,
                "scorecard_status": "promising-but-blocked",
                "policy_status": "blocked",
                "positive_signal_rows": 2,
                "open_blocker_rows": 1,
                "required_clearance_count": 1,
                "first_required_clearance": "Pocket-Blocked / blocked-ai-evidence: Reject or repair blocked AI residue evidence.",
                "top_positive_pocket_id": "Pocket-Validated",
                "top_blocker_pocket_id": "Pocket-Blocked",
                "decision_reason": "Policy=blocked; blockers=1.",
                "recommended_action": "Resolve blockers before rerank.",
            }
        ]
    )
    scorecard = pd.DataFrame(
        [
            {
                "rank_up_rows": 1,
                "rank_down_rows": 1,
                "negative_control_rows": 2,
                "frozen_blocker_rows": 1,
                "mapping_review_rows": 0,
                "evidence_gap_rows": 1,
                "score_reason": "positive=2; blockers=1; policy=blocked",
            }
        ]
    )
    queue = pd.DataFrame(
        [
            {
                "action_priority": 1,
                "pocket_id": "Pocket-Blocked",
                "issue_type": "blocked-ai-evidence",
                "issue_severity": "blocking",
                "required_fix": "Reject or repair blocked AI residue evidence.",
                "can_apply_after_fix": False,
                "consensus_anchor_residues": "A:30",
            }
        ]
    )
    delta = pd.DataFrame(
        [
            {
                "impact_priority": 1,
                "pocket_id": "Pocket-Blocked",
                "change_type": "frozen-blocker",
                "rank_delta": -1,
                "precision_interpretation": "Precision is protected by blocking unsafe promotion.",
            }
        ]
    )

    markdown = build_consensus_rerank_precision_guardrail_report_markdown(guardrail, scorecard, queue, delta)

    assert markdown.startswith("# Consensus rerank precision guardrail report")
    assert "- Guardrail status: `blocked`" in markdown
    assert "- Can enable automatic rerank: no" in markdown
    assert "Pocket-Blocked / blocked-ai-evidence" in markdown
    assert "- [ ] Required fix: Reject or repair blocked AI residue evidence." in markdown
    assert "## Release checklist" in markdown


def test_build_consensus_rerank_guardrail_artifact_manifest_lists_handoff_files():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "blocked",
                "guardrail_decision": "do-not-apply",
            }
        ]
    )
    scorecard = pd.DataFrame(
        [
            {
                "scorecard_status": "promising-but-blocked",
                "precision_improvement_score": 44,
            }
        ]
    )
    report = "# Report\n\nBody"

    manifest = build_consensus_rerank_guardrail_artifact_manifest(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_scorecard_df=scorecard,
        consensus_rerank_precision_guardrail_report_markdown=report,
    )

    file_names = set(manifest["file_name"])
    assert "consensus_rerank_precision_guardrail_report.md" in file_names
    assert "consensus_rerank_precision_guardrail.csv" in file_names
    assert "consensus_rerank_precision_scorecard.csv" in file_names
    report_row = manifest[manifest["file_name"] == "consensus_rerank_precision_guardrail_report.md"].iloc[0]
    assert int(report_row["byte_size"]) == len(report.encode("utf-8"))
    assert report_row["sha256"] == hashlib.sha256(report.encode("utf-8")).hexdigest()
    assert report_row["status"] == "blocked"


def test_build_consensus_rerank_guardrail_handoff_zip_contains_manifest_and_report():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
            }
        ]
    )
    scorecard = pd.DataFrame(
        [
            {
                "scorecard_status": "likely-precision-gain",
                "precision_improvement_score": 78,
            }
        ]
    )
    release_template = pd.DataFrame(
        [
            {
                "decision_item_id": "release-guardrail",
                "decision_scope": "release",
                "review_decision": "review",
            }
        ]
    )
    report = "# Guardrail Report\n\nReady."
    manifest = build_consensus_rerank_guardrail_artifact_manifest(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_scorecard_df=scorecard,
        consensus_rerank_precision_guardrail_report_markdown=report,
        consensus_rerank_release_decision_template_df=release_template,
    )

    bundle = build_consensus_rerank_guardrail_handoff_zip(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_scorecard_df=scorecard,
        consensus_rerank_precision_guardrail_report_markdown=report,
        consensus_rerank_release_decision_template_df=release_template,
        artifact_manifest_df=manifest,
    )

    assert bundle
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        assert "consensus_rerank_precision_guardrail_report.md" in names
        assert "consensus_rerank_precision_guardrail.csv" in names
        assert "consensus_rerank_release_decision_template.csv" in names
        assert "consensus_rerank_guardrail_artifact_manifest.csv" in names
        assert archive.read("consensus_rerank_precision_guardrail_report.md") == report.encode("utf-8")


def test_consensus_rerank_guardrail_handoff_zip_includes_release_decision_results():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
            }
        ]
    )
    release_decisions = pd.DataFrame(
        [
            {
                "decision_item_id": "release-guardrail",
                "decision_scope": "release",
                "review_decision": "approve",
                "reviewer": "Alice",
                "verified_anchor_residues": "A:10",
                "verified_sources": "PMID:1",
            }
        ]
    )
    release_validation = pd.DataFrame(
        [
            {
                "decision_item_id": "release-guardrail",
                "validation_status": "approved",
                "can_release": True,
            }
        ]
    )
    release_summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "decision_rows": 1,
                "blocked_rows": 0,
                "release_allowed": True,
            }
        ]
    )
    release_apply_plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-Validated",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "current_decision_score": 0.62,
                "simulation_score": 0.81,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
                "release_review_status": "approved-for-manual-release",
                "release_allowed": True,
                "approval_reference": "Alice; PMID:1",
                "required_pre_apply_check": "Confirm A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Apply after archival.",
            }
        ]
    )
    apply_report = build_consensus_rerank_release_apply_report_markdown(release_apply_plan, release_summary)
    execution_template = build_consensus_rerank_release_execution_template(release_apply_plan)
    execution_receipt = execution_template.copy()
    execution_receipt.loc[0, "execution_decision"] = "applied"
    execution_receipt.loc[0, "applied_rank"] = "1"
    execution_receipt.loc[0, "operator"] = "Ops"
    execution_receipt.loc[0, "executed_at"] = "2026-04-18T10:00:00+08:00"
    execution_validation = validate_consensus_rerank_release_execution_receipt(
        execution_receipt,
        execution_template,
        release_apply_plan,
    )
    execution_summary = build_consensus_rerank_release_execution_summary(
        execution_validation,
        execution_receipt,
        execution_template,
    )
    execution_report = build_consensus_rerank_release_execution_report_markdown(
        execution_summary,
        execution_validation,
        execution_receipt,
    )
    closure_certificate = build_consensus_rerank_release_closure_certificate_markdown(
        release_apply_plan,
        release_summary,
        execution_summary,
        execution_receipt,
        execution_report,
    )
    closure_ledger = build_consensus_rerank_release_closure_ledger(
        release_apply_plan,
        release_summary,
        execution_receipt,
        execution_validation,
        execution_summary,
        execution_report,
        closure_certificate,
    )

    manifest = build_consensus_rerank_guardrail_artifact_manifest(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_release_decision_df=release_decisions,
        consensus_rerank_release_decision_validation_df=release_validation,
        consensus_rerank_release_decision_summary_df=release_summary,
        consensus_rerank_release_apply_plan_df=release_apply_plan,
        consensus_rerank_release_apply_report_markdown=apply_report,
        consensus_rerank_release_execution_template_df=execution_template,
        consensus_rerank_release_execution_receipt_df=execution_receipt,
        consensus_rerank_release_execution_validation_df=execution_validation,
        consensus_rerank_release_execution_summary_df=execution_summary,
        consensus_rerank_release_execution_report_markdown=execution_report,
        consensus_rerank_release_closure_certificate_markdown=closure_certificate,
        consensus_rerank_release_closure_ledger_df=closure_ledger,
    )
    bundle = build_consensus_rerank_guardrail_handoff_zip(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_release_decision_df=release_decisions,
        consensus_rerank_release_decision_validation_df=release_validation,
        consensus_rerank_release_decision_summary_df=release_summary,
        consensus_rerank_release_apply_plan_df=release_apply_plan,
        consensus_rerank_release_apply_report_markdown=apply_report,
        consensus_rerank_release_execution_template_df=execution_template,
        consensus_rerank_release_execution_receipt_df=execution_receipt,
        consensus_rerank_release_execution_validation_df=execution_validation,
        consensus_rerank_release_execution_summary_df=execution_summary,
        consensus_rerank_release_execution_report_markdown=execution_report,
        consensus_rerank_release_closure_certificate_markdown=closure_certificate,
        consensus_rerank_release_closure_ledger_df=closure_ledger,
        artifact_manifest_df=manifest,
    )

    file_names = set(manifest["file_name"])
    assert "consensus_rerank_release_decisions_normalized.csv" in file_names
    assert "consensus_rerank_release_decision_validation.csv" in file_names
    assert "consensus_rerank_release_decision_summary.csv" in file_names
    assert "consensus_rerank_release_apply_plan.csv" in file_names
    assert "consensus_rerank_release_apply_report.md" in file_names
    assert "consensus_rerank_release_execution_template.csv" in file_names
    assert "consensus_rerank_release_execution_receipt_normalized.csv" in file_names
    assert "consensus_rerank_release_execution_validation.csv" in file_names
    assert "consensus_rerank_release_execution_summary.csv" in file_names
    assert "consensus_rerank_release_execution_report.md" in file_names
    assert "consensus_rerank_release_closure_certificate.md" in file_names
    assert "consensus_rerank_release_closure_ledger.csv" in file_names
    summary_row = manifest[manifest["file_name"] == "consensus_rerank_release_decision_summary.csv"].iloc[0]
    assert summary_row["status"] == "approved-for-manual-release"

    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        assert "consensus_rerank_release_decisions_normalized.csv" in names
        assert "consensus_rerank_release_decision_validation.csv" in names
        assert "consensus_rerank_release_decision_summary.csv" in names
        assert "consensus_rerank_release_apply_plan.csv" in names
        assert "consensus_rerank_release_apply_report.md" in names
        assert "consensus_rerank_release_execution_template.csv" in names
        assert "consensus_rerank_release_execution_receipt_normalized.csv" in names
        assert "consensus_rerank_release_execution_validation.csv" in names
        assert "consensus_rerank_release_execution_summary.csv" in names
        assert "consensus_rerank_release_execution_report.md" in names
        assert "consensus_rerank_release_closure_certificate.md" in names
        assert "consensus_rerank_release_closure_ledger.csv" in names
        assert archive.read("consensus_rerank_release_apply_report.md") == apply_report.encode("utf-8")
        assert archive.read("consensus_rerank_release_execution_report.md") == execution_report.encode("utf-8")
        assert archive.read("consensus_rerank_release_closure_certificate.md") == closure_certificate.encode("utf-8")
        assert archive.read("consensus_rerank_release_closure_ledger.csv") == closure_ledger.to_csv(index=False).encode("utf-8")


def test_verify_consensus_rerank_guardrail_handoff_zip_checks_manifest_hashes():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
            }
        ]
    )
    report = "# Guardrail Report\n\nReady."
    manifest = build_consensus_rerank_guardrail_artifact_manifest(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_guardrail_report_markdown=report,
    )
    bundle = build_consensus_rerank_guardrail_handoff_zip(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_guardrail_report_markdown=report,
        artifact_manifest_df=manifest,
    )

    verification = verify_consensus_rerank_guardrail_handoff_zip(bundle, manifest)
    summary = build_consensus_rerank_guardrail_bundle_verification_summary(verification, manifest)

    assert set(verification["verification_status"]) == {"verified"}
    assert summary.iloc[0]["verification_status"] == "verified"
    assert int(summary.iloc[0]["verified_files"]) == len(manifest)

    tampered_manifest = manifest.copy()
    tampered_manifest.loc[
        tampered_manifest["file_name"] == "consensus_rerank_precision_guardrail_report.md",
        "sha256",
    ] = "0" * 64
    tampered = verify_consensus_rerank_guardrail_handoff_zip(bundle, tampered_manifest)
    tampered_summary = build_consensus_rerank_guardrail_bundle_verification_summary(tampered, tampered_manifest)

    assert "hash-mismatch" in set(tampered["verification_status"])
    assert tampered_summary.iloc[0]["verification_status"] == "failed"
    assert int(tampered_summary.iloc[0]["hash_mismatch_files"]) == 1


def test_build_consensus_rerank_guardrail_handoff_certificate_markdown_records_zip_identity():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
                "apply_mode": "manual-consensus-rerank",
                "can_enable_auto_rerank": False,
                "can_apply_after_manual_review": True,
            }
        ]
    )
    report = "# Guardrail Report\n\nReady."
    manifest = build_consensus_rerank_guardrail_artifact_manifest(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_guardrail_report_markdown=report,
    )
    bundle = build_consensus_rerank_guardrail_handoff_zip(
        consensus_rerank_precision_guardrail_df=guardrail,
        consensus_rerank_precision_guardrail_report_markdown=report,
        artifact_manifest_df=manifest,
    )
    verification = verify_consensus_rerank_guardrail_handoff_zip(bundle, manifest)
    summary = build_consensus_rerank_guardrail_bundle_verification_summary(verification, manifest)
    release_summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "decision_rows": 1,
                "blocked_rows": 0,
                "release_allowed": True,
                "recommended_action": "Manual rerank release is approved.",
            }
        ]
    )

    certificate = build_consensus_rerank_guardrail_handoff_certificate_markdown(
        bundle,
        summary,
        manifest,
        guardrail,
        release_summary,
    )

    assert certificate.startswith("# Consensus rerank guardrail handoff certificate")
    assert "consensus_rerank_guardrail_handoff.zip" in certificate
    assert hashlib.sha256(bundle).hexdigest() in certificate
    assert "- Status: `verified`" in certificate
    assert "- Guardrail status: `manual-review-ready`" in certificate
    assert "- Can apply after manual review: yes" in certificate
    assert "- Status: `approved-for-manual-release`" in certificate
    assert "- Release allowed: yes" in certificate


def test_build_consensus_rerank_release_decision_template_lists_release_and_review_items():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
                "can_apply_after_manual_review": True,
                "open_blocker_rows": 0,
                "top_positive_pocket_id": "Pocket-Validated",
                "top_blocker_pocket_id": "none",
                "decision_reason": "Score=78; no open blockers.",
                "first_required_clearance": "none",
                "recommended_action": "Review simulation delta before applying.",
            }
        ]
    )
    queue = pd.DataFrame(
        [
            {
                "action_priority": 4,
                "pocket_id": "Pocket-Validated",
                "issue_type": "promotion-review",
                "issue_severity": "review",
                "required_fix": "Confirm consensus anchor residues.",
                "can_apply_after_fix": True,
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after review.",
            }
        ]
    )
    delta = pd.DataFrame(
        [
            {
                "impact_priority": 3,
                "pocket_id": "Pocket-Validated",
                "change_type": "rank-up-candidate",
                "issue_type": "promotion-review",
                "rank_delta": 2,
                "precision_interpretation": "Potential precision gain.",
                "required_before_trust": "Verify A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Approve after anchor review.",
            }
        ]
    )

    template = build_consensus_rerank_release_decision_template(guardrail, queue, delta)

    assert template.iloc[0]["decision_item_id"] == "release-guardrail"
    assert template.iloc[0]["recommended_decision"] == "approve-after-manual-review"
    assert template.iloc[0]["review_decision"] == "review"
    assert template[template["decision_scope"] == "clearance"].iloc[0]["pocket_id"] == "Pocket-Validated"
    rank_item = template[template["decision_scope"] == "rank-change"].iloc[0]
    assert rank_item["recommended_decision"] == "approve-rank-up-after-anchor-review"
    assert rank_item["verified_anchor_residues"] == "A:10"


def test_parse_consensus_rerank_release_decision_table_normalizes_approval():
    text = "decision_item_id,review_decision,reviewer,verified_sources\nrelease-guardrail,approved,Alice,PMID:1\n"

    decisions, metadata = parse_consensus_rerank_release_decision_table(text)

    assert metadata["status"] == "ok"
    assert metadata["approve_rows"] == "1"
    assert decisions.iloc[0]["decision_item_id"] == "release-guardrail"
    assert decisions.iloc[0]["review_decision"] == "approve"
    assert decisions.iloc[0]["reviewer"] == "Alice"


def test_validate_consensus_rerank_release_decisions_blocks_under_sourced_approval():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
                "can_apply_after_manual_review": True,
                "open_blocker_rows": 0,
                "top_positive_pocket_id": "Pocket-Validated",
                "decision_reason": "Score=78; no open blockers.",
                "first_required_clearance": "Confirm A:10.",
                "recommended_action": "Review simulation delta before applying.",
            }
        ]
    )
    template = build_consensus_rerank_release_decision_template(guardrail)
    decisions = template.copy()
    decisions.loc[0, "review_decision"] = "approve"

    validation = validate_consensus_rerank_release_decisions(decisions, template, guardrail)
    row = validation.iloc[0]

    assert row["validation_status"] == "blocked"
    assert bool(row["can_release"]) is False
    assert "missing-reviewer" in row["issue_flags"]
    assert "missing-verified-sources" in row["issue_flags"]


def test_build_consensus_rerank_release_decision_summary_allows_complete_manual_release():
    guardrail = pd.DataFrame(
        [
            {
                "guardrail_status": "manual-review-ready",
                "guardrail_decision": "allow-after-review",
                "can_apply_after_manual_review": True,
                "open_blocker_rows": 0,
                "top_positive_pocket_id": "Pocket-Validated",
                "decision_reason": "Score=78; no open blockers.",
                "first_required_clearance": "Confirm A:10.",
                "recommended_action": "Review simulation delta before applying.",
            }
        ]
    )
    template = build_consensus_rerank_release_decision_template(guardrail)
    decisions = template.copy()
    decisions.loc[0, "review_decision"] = "approve"
    decisions.loc[0, "reviewer"] = "Alice"
    decisions.loc[0, "verified_anchor_residues"] = "A:10"
    decisions.loc[0, "verified_sources"] = "PMID:1"
    decisions.loc[0, "blocker_resolved"] = "not-applicable"

    validation = validate_consensus_rerank_release_decisions(decisions, template, guardrail)
    summary = build_consensus_rerank_release_decision_summary(validation, decisions, template)

    assert validation.iloc[0]["validation_status"] == "approved"
    assert summary.iloc[0]["release_review_status"] == "approved-for-manual-release"
    assert bool(summary.iloc[0]["release_allowed"]) is True


def test_build_consensus_rerank_release_apply_plan_requires_approved_release():
    simulation = pd.DataFrame(
        [
            {
                "simulated_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 1,
                "simulated_rank_delta": 0,
                "current_decision_score": 0.70,
                "simulation_score": 0.72,
                "apply_status": "keep-current-ready",
                "apply_decision": "would-keep-rank",
                "policy_allows_apply": True,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "release_review_status": "pending-review",
                "release_allowed": False,
            }
        ]
    )

    plan = build_consensus_rerank_release_apply_plan(simulation, summary)

    assert plan.empty


def test_build_consensus_rerank_release_apply_plan_exports_clean_manual_order():
    simulation = pd.DataFrame(
        [
            {
                "simulated_rank": 2,
                "pocket_id": "Pocket-2",
                "current_rank": 1,
                "simulated_rank_delta": -1,
                "current_decision_score": 0.70,
                "simulation_score": 0.64,
                "apply_status": "keep-after-review",
                "apply_decision": "would-rank-down",
                "policy_allows_apply": True,
                "required_before_apply": "Confirm A:20.",
                "consensus_anchor_residues": "A:20",
                "recommended_action": "Keep after review.",
            },
            {
                "simulated_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank_delta": 1,
                "current_decision_score": 0.62,
                "simulation_score": 0.81,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "policy_allows_apply": True,
                "required_before_apply": "Confirm A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after review.",
            },
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "decision_rows": 2,
                "approved_rows": 2,
                "blocked_rows": 0,
                "release_allowed": True,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "decision_item_id": "release-guardrail",
                "reviewer": "Alice",
                "verified_sources": "PMID:1",
            }
        ]
    )

    plan = build_consensus_rerank_release_apply_plan(simulation, summary, validation)

    assert plan["pocket_id"].tolist() == ["Pocket-1", "Pocket-2"]
    assert plan.iloc[0]["manual_apply_rank"] == 1
    assert plan.iloc[0]["release_apply_status"] == "ready-for-manual-apply"
    assert "Alice" in plan.iloc[0]["approval_reference"]
    assert "Archive the release decision summary" in plan.iloc[0]["required_pre_apply_check"]


def test_build_consensus_rerank_release_apply_plan_blocks_stale_simulation():
    simulation = pd.DataFrame(
        [
            {
                "simulated_rank": 1,
                "pocket_id": "Pocket-Blocked",
                "current_rank": 1,
                "simulated_rank_delta": 0,
                "apply_status": "blocked-currently",
                "policy_allows_apply": True,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "release_allowed": True,
            }
        ]
    )

    plan = build_consensus_rerank_release_apply_plan(simulation, summary)

    assert plan.empty


def test_build_consensus_rerank_release_apply_report_markdown_records_order_and_hash():
    plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "current_decision_score": 0.62,
                "simulation_score": 0.81,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
                "release_review_status": "approved-for-manual-release",
                "release_allowed": True,
                "approval_reference": "Alice; PMID:1",
                "required_pre_apply_check": "Confirm A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after archival.",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "decision_rows": 1,
                "blocked_rows": 0,
                "release_allowed": True,
            }
        ]
    )

    report = build_consensus_rerank_release_apply_report_markdown(plan, summary)

    assert report.startswith("# Consensus rerank release apply report")
    assert "- Release allowed: yes" in report
    assert "- Top manual rank: `Pocket-1`" in report
    assert "Rank `1`: `Pocket-1`" in report
    assert "Apply plan CSV SHA-256" in report
    assert "Compare the `consensus_rerank_release_apply_plan.csv` SHA-256" in report


def test_build_consensus_rerank_release_execution_template_records_plan_hash_and_pending_fields():
    plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
                "approval_reference": "Alice; PMID:1",
                "required_pre_apply_check": "Confirm A:10.",
                "recommended_action": "Promote after archival.",
            }
        ]
    )

    execution_template = build_consensus_rerank_release_execution_template(plan)

    assert execution_template.iloc[0]["execution_item_id"] == "apply-rank-1"
    assert execution_template.iloc[0]["execution_decision"] == "pending"
    assert execution_template.iloc[0]["applied_rank"] == ""
    assert execution_template.iloc[0]["operator"] == ""
    assert len(execution_template.iloc[0]["plan_sha256"]) == 64
    assert execution_template.iloc[0]["approval_reference"] == "Alice; PMID:1"


def test_parse_consensus_rerank_release_execution_table_normalizes_applied_rows():
    text = "execution_item_id,execution_decision,applied_rank,operator,executed_at,plan_sha256\napply-rank-1,executed,1,Ops,2026-04-18T10:00:00+08:00,abc\n"

    receipt, metadata = parse_consensus_rerank_release_execution_table(text)

    assert metadata["status"] == "ok"
    assert metadata["applied_rows"] == "1"
    assert receipt.iloc[0]["execution_decision"] == "applied"
    assert receipt.iloc[0]["operator"] == "Ops"


def test_validate_consensus_rerank_release_execution_receipt_accepts_exact_application():
    plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "current_decision_score": 0.62,
                "simulation_score": 0.81,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
                "release_review_status": "approved-for-manual-release",
                "release_allowed": True,
                "approval_reference": "Alice; PMID:1",
                "required_pre_apply_check": "Confirm A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after archival.",
            }
        ]
    )
    template = build_consensus_rerank_release_execution_template(plan)
    receipt = template.copy()
    receipt.loc[0, "execution_decision"] = "applied"
    receipt.loc[0, "applied_rank"] = "1"
    receipt.loc[0, "operator"] = "Ops"
    receipt.loc[0, "executed_at"] = "2026-04-18T10:00:00+08:00"

    validation = validate_consensus_rerank_release_execution_receipt(receipt, template, plan)
    summary = build_consensus_rerank_release_execution_summary(validation, receipt, template)

    assert validation.iloc[0]["validation_status"] == "applied"
    assert bool(validation.iloc[0]["execution_accepted"]) is True
    assert summary.iloc[0]["execution_review_status"] == "executed"
    assert bool(summary.iloc[0]["execution_complete"]) is True


def test_validate_consensus_rerank_release_execution_receipt_blocks_rank_and_hash_mismatch():
    plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
            }
        ]
    )
    template = build_consensus_rerank_release_execution_template(plan)
    receipt = template.copy()
    receipt.loc[0, "execution_decision"] = "applied"
    receipt.loc[0, "applied_rank"] = "2"
    receipt.loc[0, "operator"] = "Ops"
    receipt.loc[0, "executed_at"] = "2026-04-18T10:00:00+08:00"
    receipt.loc[0, "plan_sha256"] = "0" * 64

    validation = validate_consensus_rerank_release_execution_receipt(receipt, template, plan)
    summary = build_consensus_rerank_release_execution_summary(validation, receipt, template)

    assert validation.iloc[0]["validation_status"] == "blocked"
    assert "rank-mismatch" in validation.iloc[0]["issue_flags"]
    assert "plan-hash-mismatch" in validation.iloc[0]["issue_flags"]
    assert summary.iloc[0]["execution_review_status"] == "blocked"
    assert int(summary.iloc[0]["rank_mismatch_rows"]) == 1
    assert int(summary.iloc[0]["plan_hash_mismatch_rows"]) == 1


def test_build_consensus_rerank_release_execution_report_markdown_records_receipt_status_and_hash():
    receipt = pd.DataFrame(
        [
            {
                "execution_item_id": "apply-rank-1",
                "manual_apply_rank": "1",
                "pocket_id": "Pocket-1",
                "plan_sha256": "a" * 64,
                "execution_decision": "applied",
                "applied_rank": "1",
                "operator": "Ops",
                "executed_at": "2026-04-18T10:00:00+08:00",
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "row_index": 1,
                "execution_item_id": "apply-rank-1",
                "pocket_id": "Pocket-1",
                "execution_decision": "applied",
                "template_match": True,
                "expected_rank": 1,
                "applied_rank": 1,
                "plan_hash_match": True,
                "validation_status": "applied",
                "issue_flags": "none",
                "execution_accepted": True,
                "operator": "Ops",
                "executed_at": "2026-04-18T10:00:00+08:00",
                "plan_sha256": "a" * 64,
            }
        ]
    )
    summary = build_consensus_rerank_release_execution_summary(validation, receipt, receipt)

    report = build_consensus_rerank_release_execution_report_markdown(summary, validation, receipt)

    assert report.startswith("# Consensus rerank release execution report")
    assert "- Execution status: `executed`" in report
    assert "- Execution complete: yes" in report
    assert "- Operators: Ops" in report
    assert "Execution receipt CSV SHA-256" in report
    assert "`apply-rank-1` / `Pocket-1`: applied" in report


def test_build_consensus_rerank_release_closure_certificate_markdown_records_final_status():
    plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "current_decision_score": 0.62,
                "simulation_score": 0.81,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
                "release_review_status": "approved-for-manual-release",
                "release_allowed": True,
                "approval_reference": "Alice; PMID:1",
                "required_pre_apply_check": "Confirm A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after archival.",
            }
        ]
    )
    release_summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "decision_rows": 1,
                "blocked_rows": 0,
                "release_allowed": True,
            }
        ]
    )
    execution_template = build_consensus_rerank_release_execution_template(plan)
    receipt = execution_template.copy()
    receipt.loc[0, "execution_decision"] = "applied"
    receipt.loc[0, "applied_rank"] = "1"
    receipt.loc[0, "operator"] = "Ops"
    receipt.loc[0, "executed_at"] = "2026-04-18T10:00:00+08:00"
    validation = validate_consensus_rerank_release_execution_receipt(receipt, execution_template, plan)
    execution_summary = build_consensus_rerank_release_execution_summary(validation, receipt, execution_template)
    report = build_consensus_rerank_release_execution_report_markdown(execution_summary, validation, receipt)

    certificate = build_consensus_rerank_release_closure_certificate_markdown(
        plan,
        release_summary,
        execution_summary,
        receipt,
        report,
    )

    assert certificate.startswith("# Consensus rerank release closure certificate")
    assert "closed-executed" in certificate
    assert "Execution complete: yes" in certificate
    assert "Apply plan SHA-256" in certificate
    assert "Execution receipt CSV SHA-256" in certificate
    assert "Execution report SHA-256" in certificate


def test_build_consensus_rerank_release_closure_ledger_tracks_required_artifacts():
    plan = pd.DataFrame(
        [
            {
                "manual_apply_rank": 1,
                "pocket_id": "Pocket-1",
                "current_rank": 2,
                "simulated_rank": 1,
                "rank_delta": 1,
                "current_decision_score": 0.62,
                "simulation_score": 0.81,
                "apply_status": "apply-ready-after-review",
                "apply_decision": "would-rank-up",
                "release_apply_status": "ready-for-manual-apply",
                "release_review_status": "approved-for-manual-release",
                "release_allowed": True,
                "approval_reference": "Alice; PMID:1",
                "required_pre_apply_check": "Confirm A:10.",
                "consensus_anchor_residues": "A:10",
                "recommended_action": "Promote after archival.",
            }
        ]
    )
    release_summary = pd.DataFrame(
        [
            {
                "release_review_status": "approved-for-manual-release",
                "decision_rows": 1,
                "blocked_rows": 0,
                "release_allowed": True,
            }
        ]
    )
    template = build_consensus_rerank_release_execution_template(plan)
    receipt = template.copy()
    receipt.loc[0, "execution_decision"] = "applied"
    receipt.loc[0, "applied_rank"] = "1"
    receipt.loc[0, "operator"] = "Ops"
    receipt.loc[0, "executed_at"] = "2026-04-18T10:00:00+08:00"
    validation = validate_consensus_rerank_release_execution_receipt(receipt, template, plan)
    execution_summary = build_consensus_rerank_release_execution_summary(validation, receipt, template)
    report = build_consensus_rerank_release_execution_report_markdown(execution_summary, validation, receipt)
    certificate = build_consensus_rerank_release_closure_certificate_markdown(
        plan,
        release_summary,
        execution_summary,
        receipt,
        report,
    )

    ledger = build_consensus_rerank_release_closure_ledger(
        plan,
        release_summary,
        receipt,
        validation,
        execution_summary,
        report,
        certificate,
    )

    assert list(ledger["evidence_item"]) == [
        "approved apply plan",
        "release decision summary",
        "execution receipt",
        "execution validation",
        "execution summary",
        "execution report",
        "closure certificate",
    ]
    assert set(ledger["closure_check"]) == {"ok"}
    assert ledger.loc[ledger["file_name"] == "consensus_rerank_release_closure_certificate.md", "status"].iloc[0] == "closed-executed"
    assert ledger["sha256"].astype(str).str.len().min() == 64


def test_build_consensus_rerank_release_closure_summary_requires_verified_bundle():
    ledger = pd.DataFrame(
        [
            {
                "evidence_item": f"item-{idx}",
                "file_name": f"artifact-{idx}.csv",
                "artifact_type": "csv",
                "row_count": 1,
                "byte_size": 12,
                "sha256": "a" * 64,
                "status": "ok",
                "required_for_closure": True,
                "closure_check": "ok",
                "issue": "none",
                "recommended_action": "No action required.",
            }
            for idx in range(7)
        ]
    )
    verified_bundle = pd.DataFrame(
        [
            {
                "verification_status": "verified",
                "manifest_rows": 13,
                "failed_files": 0,
            }
        ]
    )

    summary = build_consensus_rerank_release_closure_summary(ledger, verified_bundle)

    assert summary.iloc[0]["closure_readiness_status"] == "closed-and-verified"
    assert bool(summary.iloc[0]["bundle_verified"]) is True
    assert bool(summary.iloc[0]["release_closed"]) is True

    failed_bundle = verified_bundle.copy()
    failed_bundle.loc[0, "verification_status"] = "failed"
    failed_bundle.loc[0, "failed_files"] = 1
    blocked_summary = build_consensus_rerank_release_closure_summary(ledger, failed_bundle)

    assert blocked_summary.iloc[0]["closure_readiness_status"] == "package-verification-blocked"
    assert bool(blocked_summary.iloc[0]["release_closed"]) is False


def test_build_consensus_rerank_release_closure_blocker_queue_lists_ledger_and_package_fixes():
    ledger = pd.DataFrame(
        [
            {
                "evidence_item": "execution receipt",
                "file_name": "consensus_rerank_release_execution_receipt_normalized.csv",
                "artifact_type": "csv",
                "row_count": 0,
                "byte_size": 0,
                "sha256": "",
                "status": "missing",
                "required_for_closure": True,
                "closure_check": "missing",
                "issue": "Execution receipt is missing.",
                "recommended_action": "Upload a completed execution receipt.",
            },
            {
                "evidence_item": "closure certificate",
                "file_name": "consensus_rerank_release_closure_certificate.md",
                "artifact_type": "markdown",
                "row_count": 12,
                "byte_size": 256,
                "sha256": "a" * 64,
                "status": "closed-executed",
                "required_for_closure": True,
                "closure_check": "ok",
                "issue": "none",
                "recommended_action": "No action required.",
            },
        ]
    )
    bundle_summary = pd.DataFrame(
        [
            {
                "verification_status": "failed",
                "manifest_rows": 13,
                "failed_files": 1,
            }
        ]
    )
    bundle_verification = pd.DataFrame(
        [
            {
                "file_name": "consensus_rerank_release_execution_receipt_normalized.csv",
                "verification_status": "missing",
                "issue": "Artifact listed in manifest is missing from the handoff ZIP.",
                "recommended_action": "Regenerate the handoff ZIP.",
            }
        ]
    )
    summary = build_consensus_rerank_release_closure_summary(ledger, bundle_summary)

    blockers = build_consensus_rerank_release_closure_blocker_queue(
        summary,
        ledger,
        bundle_verification,
        bundle_summary,
    )

    assert list(blockers["blocker_rank"]) == [1, 2]
    assert set(blockers["blocker_type"]) == {"missing-evidence", "package-verification-failed"}
    assert blockers.iloc[0]["severity"] == "critical"
    assert "execution receipt" in set(blockers["evidence_item"])


def test_build_consensus_rerank_release_closure_remediation_checklist_markdown_lists_actions():
    blockers = pd.DataFrame(
        [
            {
                "blocker_rank": 1,
                "blocker_source": "closure-ledger",
                "evidence_item": "execution receipt",
                "file_name": "consensus_rerank_release_execution_receipt_normalized.csv",
                "blocker_type": "missing-evidence",
                "severity": "critical",
                "current_status": "missing",
                "issue": "Execution receipt is missing.",
                "required_fix": "Upload a completed execution receipt.",
                "recommended_action": "Regenerate closure ledger.",
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "closure_readiness_status": "ledger-blocked",
                "release_closed": False,
                "recommended_action": "Fix missing closure evidence.",
            }
        ]
    )

    checklist = build_consensus_rerank_release_closure_remediation_checklist_markdown(blockers, summary)

    assert checklist.startswith("# Consensus rerank release closure remediation checklist")
    assert "Readiness status: `ledger-blocked`" in checklist
    assert "Release closed: no" in checklist
    assert "- [ ] Rank `1` / `critical` / `missing-evidence`" in checklist
    assert "Upload a completed execution receipt." in checklist
    assert "Treat release as closed only when readiness status is `closed-and-verified`" in checklist


def test_build_consensus_rerank_release_closure_detached_manifest_hashes_external_artifacts():
    summary = pd.DataFrame(
        [
            {
                "closure_readiness_status": "ledger-blocked",
                "ledger_rows": 7,
                "required_rows": 7,
                "ok_rows": 5,
                "blocked_rows": 1,
                "missing_rows": 1,
                "missing_hash_rows": 1,
                "manifest_rows": 13,
                "bundle_verification_status": "failed",
                "bundle_failed_files": 1,
                "bundle_verified": False,
                "release_closed": False,
                "recommended_action": "Fix closure blockers.",
            }
        ]
    )
    blockers = pd.DataFrame(
        [
            {
                "blocker_rank": 1,
                "blocker_source": "closure-ledger",
                "evidence_item": "execution receipt",
                "file_name": "consensus_rerank_release_execution_receipt_normalized.csv",
                "blocker_type": "missing-evidence",
                "severity": "critical",
                "current_status": "missing",
                "issue": "Execution receipt is missing.",
                "required_fix": "Upload a completed execution receipt.",
                "recommended_action": "Regenerate closure ledger.",
            }
        ]
    )
    checklist = build_consensus_rerank_release_closure_remediation_checklist_markdown(blockers, summary)

    manifest = build_consensus_rerank_release_closure_detached_manifest(summary, blockers, checklist)

    file_names = set(manifest["file_name"])
    assert file_names == {
        "consensus_rerank_release_closure_summary.csv",
        "consensus_rerank_release_closure_blocker_queue.csv",
        "consensus_rerank_release_closure_remediation_checklist.md",
    }
    assert set(manifest["artifact_type"]) == {"csv", "markdown"}
    assert manifest["sha256"].astype(str).str.len().min() == 64
    summary_row = manifest[manifest["file_name"] == "consensus_rerank_release_closure_summary.csv"].iloc[0]
    assert summary_row["status"] == "ledger-blocked"


def test_add_pocket_residue_layers_marks_core_shell_and_rim():
    pocket_rows = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "external_direct_anchor": True,
                "evidence_route_anchor": True,
                "external_support": 0.82,
                "is_hotspot": False,
                "residue_score": 0.70,
            },
            {
                "chain": "A",
                "resid": 11,
                "external_direct_anchor": False,
                "evidence_route_anchor": False,
                "external_support": 0.20,
                "evidence_anchor_proximity": 0.55,
                "contact_count": 5,
                "residue_score": 0.48,
            },
            {
                "chain": "A",
                "resid": 18,
                "external_direct_anchor": False,
                "evidence_route_anchor": False,
                "external_support": 0.0,
                "evidence_anchor_proximity": 0.05,
                "contact_count": 1,
                "residue_score": 0.12,
            },
        ]
    )

    layered = add_pocket_residue_layers(pocket_rows)

    assert layered.loc[layered["resid"] == 10, "pocket_layer"].iloc[0] == "core"
    assert layered.loc[layered["resid"] == 11, "pocket_layer"].iloc[0] == "shell"
    assert layered.loc[layered["resid"] == 18, "pocket_layer"].iloc[0] == "rim"
    assert "direct evidence anchor" in layered.loc[layered["resid"] == 10, "pocket_layer_reason"].iloc[0]
