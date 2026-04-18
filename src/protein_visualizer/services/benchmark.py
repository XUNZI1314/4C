from __future__ import annotations

import io
import re
from typing import Optional, Sequence

import pandas as pd


BENCHMARK_REFERENCE_COLUMNS = [
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "reference_type",
    "reference_source",
    "reference_note",
    "expected_pocket_id",
]

BENCHMARK_REFERENCE_TEMPLATE_COLUMNS = [
    *BENCHMARK_REFERENCE_COLUMNS,
]

BENCHMARK_REFERENCE_IMPORT_SUMMARY_COLUMNS = [
    "source",
    "import_status",
    "evidence_rows",
    "reference_rows",
    "case_count",
    "chain_specific_rows",
    "wildcard_chain_rows",
    "exact_mapping_rows",
    "weak_mapping_rows",
    "structure_verified_rows",
    "manual_review_rows",
    "missing_resname_rows",
    "duplicate_rows",
    "skipped_rows",
    "recommended_action",
    "import_warning",
]

BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "issue_type",
    "reference_type",
    "reference_source",
    "mapping_level",
    "mapping_confidence",
    "mapping_method",
    "suggested_action",
    "review_warning",
]

BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS = [
    *BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS,
    "review_decision",
    "reviewer",
    "verified_source",
    "verified_mapping",
    "review_note",
]

BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_VALIDATION_COLUMNS = [
    "row_index",
    "action_id",
    "review_decision",
    "validation_status",
    "issue_flags",
    "required_fix",
]

BENCHMARK_REFERENCE_CANDIDATE_REVIEW_OUTCOME_COLUMNS = [
    "action_id",
    "priority",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "issue_type",
    "review_decision",
    "applied_status",
    "outcome_reason",
    "next_action",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS = [
    "audit_id",
    "source_mode",
    "source_claim_status",
    "can_support_independent_claim",
    "is_provisional",
    "is_reviewed_candidate",
    *BENCHMARK_REFERENCE_COLUMNS,
    "review_status",
    "provenance_warning",
    "recommended_action",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_SUMMARY_COLUMNS = [
    "summary_id",
    "source_mode",
    "source_claim_status",
    "can_support_independent_claim",
    "reference_rows",
    "case_count",
    "provisional_rows",
    "reviewed_candidate_rows",
    "recommended_action",
    "summary_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_ACTION_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "issue_type",
    "source_mode",
    "source_claim_status",
    "can_support_independent_claim",
    *BENCHMARK_REFERENCE_COLUMNS,
    "residue_label",
    "suggested_action",
    "action_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS = [
    "benchmark_id",
    "reference_rows",
    "action_rows",
    "blocker_rows",
    "review_rows",
    "top_priority",
    "top_issue_type",
    "source_claim_statuses",
    "source_modes",
    "recommended_action",
    "case_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS = [
    *BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS,
    "source_decision",
    "reviewer",
    "decision_date",
    "verified_source_mode",
    "verified_independence",
    "replacement_reference_source",
    "decision_note",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_VALIDATION_COLUMNS = [
    "row_index",
    "benchmark_id",
    "source_decision",
    "validation_status",
    "issue_flags",
    "required_fix",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS = [
    "benchmark_id",
    "source_decision",
    "validation_status",
    "applied_status",
    "reference_rows",
    "action_rows",
    "blocker_rows",
    "review_rows",
    "top_priority",
    "top_issue_type",
    "source_claim_statuses",
    "source_modes",
    "reviewer",
    "verified_source_mode",
    "verified_independence",
    "replacement_reference_source",
    "outcome_reason",
    "next_action",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_SUMMARY_COLUMNS = [
    "summary_id",
    "closure_status",
    "case_count",
    "blocked_cases",
    "pending_cases",
    "held_cases",
    "replaced_cases",
    "cleared_cases",
    "source_ready_cases",
    "actionable_case_count",
    "closed_actionable_case_count",
    "open_actionable_case_count",
    "recommended_action",
    "summary_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_CLOSURE_QUEUE_COLUMNS = [
    "closure_action_id",
    "benchmark_id",
    "applied_status",
    "closure_priority",
    "closure_action_status",
    "closure_issue_type",
    "source_decision",
    "validation_status",
    "reference_rows",
    "action_rows",
    "blocker_rows",
    "review_rows",
    "top_priority",
    "top_issue_type",
    "source_claim_statuses",
    "source_modes",
    "reviewer",
    "verified_source_mode",
    "verified_independence",
    "replacement_reference_source",
    "outcome_reason",
    "required_action",
    "closure_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_COLUMNS = [
    "benchmark_id",
    "applied_status",
    "source_decision",
    "validation_status",
    "original_source_priority",
    "original_source_issue_type",
    "adjusted_source_priority",
    "adjusted_source_issue_type",
    "readiness_impact",
    "reference_rows",
    "source_claim_statuses",
    "source_modes",
    "outcome_reason",
    "readiness_note",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_SUMMARY_COLUMNS = [
    "summary_id",
    "readiness_impact_status",
    "case_count",
    "cleared_by_decision_cases",
    "decision_adjusted_open_cases",
    "decision_open_cases",
    "unchanged_open_cases",
    "source_ready_cases",
    "open_after_decision_cases",
    "original_p0_cases",
    "original_p1_cases",
    "original_p2_cases",
    "adjusted_p0_cases",
    "adjusted_p1_cases",
    "adjusted_p2_cases",
    "net_blocker_delta",
    "recommended_action",
    "summary_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_COLUMNS = [
    "top_n",
    "case_count",
    "claim_ready_case_count",
    "source_tracked_case_count",
    "source_open_case_count",
    "source_blocker_case_count",
    "source_review_case_count",
    "source_cleared_case_count",
    "source_ready_case_count",
    "dataset_blocked_by_source_cases",
    "dataset_review_by_source_cases",
    "source_gate_mismatch_case_count",
    "mean_source_open_coverage",
    "mean_source_cleared_coverage",
    "dataset_source_impact_status",
    "recommended_action",
    "impact_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS = [
    "impact_case_id",
    "top_n",
    "benchmark_id",
    "claim_status",
    "claim_ready",
    "coverage_ratio",
    "best_rank",
    "best_pocket_id",
    "benchmark_status",
    "readiness_status",
    "source_tracked",
    "applied_status",
    "source_decision",
    "validation_status",
    "original_source_priority",
    "original_source_issue_type",
    "adjusted_source_priority",
    "adjusted_source_issue_type",
    "readiness_impact",
    "source_impact_status",
    "source_action_status",
    "source_gate_mismatch",
    "outcome_reason",
    "readiness_note",
    "recommended_action",
    "impact_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "impact_case_id",
    "top_n",
    "benchmark_id",
    "source_impact_status",
    "source_gate_mismatch",
    "claim_status",
    "coverage_ratio",
    "best_rank",
    "best_pocket_id",
    "readiness_status",
    "applied_status",
    "source_decision",
    "validation_status",
    "adjusted_source_priority",
    "adjusted_source_issue_type",
    "readiness_impact",
    "recommended_action",
    "impact_warning",
]

BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_SUMMARY_COLUMNS = [
    "summary_id",
    "priority",
    "action_status",
    "source_impact_status",
    "action_count",
    "affected_case_count",
    "top_n_values",
    "mismatch_count",
    "mean_coverage_ratio",
    "top_action_id",
    "top_benchmark_id",
    "recommended_action",
    "summary_warning",
]

BENCHMARK_DETAIL_COLUMNS = [
    *BENCHMARK_REFERENCE_COLUMNS,
    "residue_label",
    "matched",
    "matched_pocket_id",
    "matched_rank",
    "matched_pocket_ids",
    "matched_top1",
    "matched_top3",
    "matched_top5",
    "expected_pocket_matched",
    "benchmark_warning",
]

BENCHMARK_SUMMARY_COLUMNS = [
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "any_hit",
    "all_hit",
    "best_rank",
    "best_pocket_id",
    "top_pocket_id",
    "top_pocket_hit",
    "matched_residues",
    "missed_residues",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_CASE_SUMMARY_COLUMNS = [
    "benchmark_id",
    *BENCHMARK_SUMMARY_COLUMNS,
]

BENCHMARK_DATASET_SUMMARY_COLUMNS = [
    "top_n",
    "case_count",
    "reference_residue_count",
    "matched_reference_count",
    "mean_coverage_ratio",
    "median_coverage_ratio",
    "min_coverage_ratio",
    "max_coverage_ratio",
    "any_hit_case_count",
    "all_hit_case_count",
    "miss_case_count",
    "any_hit_rate",
    "all_hit_rate",
    "mean_best_rank",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "reference_coverage_ratio",
    "coverage_delta_vs_reference",
    "coverage_loss_vs_reference",
    "best_rank",
    "reference_best_rank",
    "best_rank_delta_vs_reference",
    "best_pocket_id",
    "top_pocket_id",
    "top_pocket_hit",
    "matched_residues",
    "missed_residues",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_CASE_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "benchmark_id",
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "reference_coverage_ratio",
    "coverage_delta_vs_reference",
    "coverage_loss_vs_reference",
    "best_rank",
    "reference_best_rank",
    "best_rank_delta_vs_reference",
    "best_pocket_id",
    "top_pocket_id",
    "top_pocket_hit",
    "matched_residues",
    "missed_residues",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_DATASET_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "top_n",
    "case_count",
    "mean_coverage_ratio",
    "reference_mean_coverage_ratio",
    "mean_coverage_delta_vs_reference",
    "mean_coverage_loss_vs_reference",
    "any_hit_rate",
    "reference_any_hit_rate",
    "any_hit_rate_delta_vs_reference",
    "all_hit_rate",
    "reference_all_hit_rate",
    "all_hit_rate_delta_vs_reference",
    "mean_best_rank",
    "reference_mean_best_rank",
    "mean_best_rank_delta_vs_reference",
    "case_loss_count",
    "case_gain_count",
    "case_unchanged_count",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "reference_type",
    "reference_source",
    "expected_pocket_id",
    "variant_matched",
    "reference_matched",
    "match_delta",
    "variant_matched_rank",
    "reference_matched_rank",
    "rank_delta_vs_reference",
    "variant_matched_pocket_id",
    "reference_matched_pocket_id",
    "variant_matched_pocket_ids",
    "reference_matched_pocket_ids",
    "variant_expected_pocket_matched",
    "reference_expected_pocket_matched",
    "benchmark_warning",
]

BENCHMARK_VARIANT_REMEDIATION_COLUMNS = [
    "action_id",
    "priority",
    "issue_type",
    "variant_label",
    "reference_variant_label",
    "benchmark_id",
    "residue_label",
    "chain",
    "resid",
    "resname",
    "match_delta",
    "reference_matched_pocket_id",
    "variant_matched_pocket_id",
    "reference_matched_rank",
    "variant_matched_rank",
    "expected_pocket_id",
    "suggested_action",
    "benchmark_warning",
]

BENCHMARK_VARIANT_REMEDIATION_SUMMARY_COLUMNS = [
    "priority",
    "issue_type",
    "variant_label",
    "action_count",
    "affected_case_count",
    "affected_residue_count",
    "top_residues",
    "suggested_action",
    "summary_status",
    "summary_warning",
]

BENCHMARK_REFERENCE_QUALITY_COLUMNS = [
    "issue_id",
    "severity",
    "issue_type",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "reference_type",
    "reference_source",
    "reference_note",
    "suggested_action",
    "quality_warning",
]

BENCHMARK_REFERENCE_QUALITY_SUMMARY_COLUMNS = [
    "severity",
    "issue_type",
    "issue_count",
    "affected_case_count",
    "affected_residue_count",
    "suggested_action",
    "summary_status",
    "summary_warning",
]

BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS = [
    "issue_id",
    "severity",
    "issue_type",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "structure_chains",
    "structure_resnames",
    "matched_chain",
    "matched_resname",
    "reference_source",
    "reference_note",
    "suggested_action",
    "validation_warning",
]

BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_SUMMARY_COLUMNS = [
    "severity",
    "issue_type",
    "issue_count",
    "affected_case_count",
    "affected_residue_count",
    "suggested_action",
    "summary_status",
    "summary_warning",
]

BENCHMARK_REFERENCE_READINESS_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "issue_source",
    "issue_type",
    "benchmark_id",
    "residue_label",
    "chain",
    "resid",
    "resname",
    "suggested_action",
    "readiness_warning",
]

BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS = [
    "readiness_status",
    "reference_residue_count",
    "curation_issue_count",
    "structure_validation_issue_count",
    "source_audit_issue_count",
    "p0_p1_issue_count",
    "p2_issue_count",
    "blocking_issue_types",
    "review_issue_types",
    "recommended_action",
    "readiness_warning",
]

BENCHMARK_REFERENCE_READINESS_CASE_SUMMARY_COLUMNS = [
    "benchmark_id",
    *BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS,
]

BENCHMARK_INTERPRETATION_COLUMNS = [
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "benchmark_status",
    "readiness_status",
    "claim_status",
    "claim_ready",
    "best_rank",
    "best_pocket_id",
    "interpretation_label",
    "recommended_action",
    "interpretation_warning",
]

BENCHMARK_CASE_INTERPRETATION_COLUMNS = [
    "benchmark_id",
    *BENCHMARK_INTERPRETATION_COLUMNS,
]

BENCHMARK_CASE_INTERPRETATION_MATRIX_BASE_COLUMNS = [
    "benchmark_id",
    "top_n_count",
    "best_claim_ready_top_n",
    "best_claim_ready_coverage",
    "best_claim_ready_rank",
    "any_blocked",
    "any_review_needed",
    "any_readiness_unknown",
    "case_interpretation_status",
    "recommended_action",
]

BENCHMARK_CASE_INTERPRETATION_MATRIX_SUMMARY_COLUMNS = [
    "case_count",
    "usable_claim_ready_case_count",
    "blocked_case_count",
    "review_case_count",
    "readiness_unknown_case_count",
    "no_claim_ready_case_count",
    "earliest_top1_claim_ready_case_count",
    "earliest_top3_claim_ready_case_count",
    "earliest_top5_claim_ready_case_count",
    "mean_usable_claim_ready_coverage",
    "mean_usable_claim_ready_rank",
    "summary_status",
    "recommended_action",
    "summary_warning",
]

BENCHMARK_CASE_INTERPRETATION_MATRIX_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "benchmark_id",
    "case_interpretation_status",
    "best_claim_ready_top_n",
    "best_claim_ready_coverage",
    "best_claim_ready_rank",
    "top1_claim_status",
    "top3_claim_status",
    "top5_claim_status",
    "issue_type",
    "suggested_action",
    "case_warning",
]

BENCHMARK_DATASET_INTERPRETATION_COLUMNS = [
    "top_n",
    "case_count",
    "claim_ready_case_count",
    "blocked_case_count",
    "review_case_count",
    "unknown_case_count",
    "mean_claim_ready_coverage",
    "mean_all_case_coverage",
    "claim_ready_rate",
    "blocked_case_rate",
    "review_case_rate",
    "dataset_claim_status",
    "interpretation_label",
    "recommended_action",
    "interpretation_warning",
]

BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "top_n",
    "benchmark_id",
    "claim_status",
    "coverage_ratio",
    "best_rank",
    "best_pocket_id",
    "benchmark_status",
    "readiness_status",
    "issue_type",
    "suggested_action",
    "interpretation_warning",
]

CHAIN_ALIASES = {"chain", "chainid", "chain_id", "authasymid", "auth_asym_id", "asymid", "asym_id"}
RESID_ALIASES = {
    "resid",
    "residue",
    "residuenumber",
    "residue_number",
    "residueid",
    "residue_id",
    "position",
    "seqnum",
    "seq_num",
    "seqnumber",
    "sequenceposition",
    "pdbresiduenumber",
    "uniprotresid",
    "uniprot_resid",
}
RESNAME_ALIASES = {"resname", "residue_name", "aa", "aminoacid", "amino_acid"}
RESIDUE_TOKEN_ALIASES = {"residuelabel", "residue_label", "site", "positiontext", "position_text"}
TYPE_ALIASES = {"reference_type", "type", "evidence_type", "role", "annotation", "function"}
SOURCE_ALIASES = {"reference_source", "evidence_source", "source", "dataset", "citation", "reference", "pmid", "doi"}
NOTE_ALIASES = {"reference_note", "note", "notes", "comment", "description", "evidence_note"}
EXPECTED_POCKET_ALIASES = {"expected_pocket_id", "pocket_id", "active_site_pocket", "validated_pocket_id"}
BENCHMARK_ID_ALIASES = {"benchmark_id", "case_id", "dataset_id", "enzyme_id", "pdb_id", "entry_id", "mcsa_id"}
PDB_ID_ALIASES = {"pdb_id", "pdb", "pdbcode", "pdb_code", "structure_id", "structure"}
GENERIC_REFERENCE_SOURCE_LABELS = {
    "",
    "source",
    "reference",
    "curatedbenchmark",
    "curatedcatalyticbenchmark",
    "curatedliterature",
    "mcsapmiddoi",
}
SUPPORTED_REFERENCE_TYPE_TOKENS = (
    "active",
    "binding",
    "catalytic",
    "cofactor",
    "ligand",
    "metal",
    "mutation",
    "mutagenesis",
    "substrate",
)
MAPPING_ASSUMPTION_TOKENS = ("uniprot", "mature", "isoform", "offset", "precursor")


def _empty_reference_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_COLUMNS)


def _empty_reference_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_TEMPLATE_COLUMNS)


def _empty_reference_import_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_IMPORT_SUMMARY_COLUMNS)


def _empty_reference_candidate_review_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS)


def _empty_reference_candidate_review_decision_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS)


def _empty_reference_candidate_review_decision_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_VALIDATION_COLUMNS)


def _empty_reference_candidate_review_outcome_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_CANDIDATE_REVIEW_OUTCOME_COLUMNS)


def _empty_reference_source_audit_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS)


def _empty_reference_source_audit_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_SUMMARY_COLUMNS)


def _empty_reference_source_audit_action_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_ACTION_QUEUE_COLUMNS)


def _empty_reference_source_audit_case_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS)


def _empty_reference_source_audit_case_decision_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS)


def _empty_reference_source_audit_case_decision_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_VALIDATION_COLUMNS)


def _empty_reference_source_audit_case_decision_outcome_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS)


def _empty_reference_source_audit_case_decision_outcome_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_SUMMARY_COLUMNS)


def _empty_reference_source_audit_case_decision_closure_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_CLOSURE_QUEUE_COLUMNS)


def _empty_reference_source_audit_case_decision_readiness_impact_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_COLUMNS)


def _empty_reference_source_audit_case_decision_readiness_impact_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_SUMMARY_COLUMNS)


def _empty_reference_source_audit_case_decision_dataset_impact_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_COLUMNS)


def _empty_reference_source_audit_case_decision_dataset_impact_case_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS)


def _empty_reference_source_audit_case_decision_dataset_impact_action_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_COLUMNS)


def _empty_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_SUMMARY_COLUMNS
    )


def _empty_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DETAIL_COLUMNS)


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_SUMMARY_COLUMNS)


def _empty_case_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_CASE_SUMMARY_COLUMNS)


def _empty_dataset_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DATASET_SUMMARY_COLUMNS)


def _empty_variant_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_COMPARISON_COLUMNS)


def _empty_variant_case_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_CASE_COMPARISON_COLUMNS)


def _empty_variant_dataset_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_DATASET_COMPARISON_COLUMNS)


def _empty_variant_detail_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS)


def _empty_variant_remediation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_REMEDIATION_COLUMNS)


def _empty_variant_remediation_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_REMEDIATION_SUMMARY_COLUMNS)


def _empty_reference_quality_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_QUALITY_COLUMNS)


def _empty_reference_quality_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_QUALITY_SUMMARY_COLUMNS)


def _empty_reference_structure_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS)


def _empty_reference_structure_validation_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_SUMMARY_COLUMNS)


def _empty_reference_readiness_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_READINESS_QUEUE_COLUMNS)


def _empty_reference_readiness_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS)


def _empty_reference_readiness_case_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_READINESS_CASE_SUMMARY_COLUMNS)


def _empty_interpretation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_INTERPRETATION_COLUMNS)


def _empty_case_interpretation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_CASE_INTERPRETATION_COLUMNS)


def _case_interpretation_matrix_columns(top_ns: Sequence[int] = (1, 3, 5)) -> list[str]:
    columns = list(BENCHMARK_CASE_INTERPRETATION_MATRIX_BASE_COLUMNS)
    for top_n in top_ns:
        prefix = f"top{int(top_n)}"
        columns.extend(
            [
                f"{prefix}_claim_status",
                f"{prefix}_claim_ready",
                f"{prefix}_coverage_ratio",
                f"{prefix}_best_rank",
                f"{prefix}_best_pocket_id",
                f"{prefix}_benchmark_status",
            ]
        )
    return columns


def _empty_case_interpretation_matrix_df(top_ns: Sequence[int] = (1, 3, 5)) -> pd.DataFrame:
    return pd.DataFrame(columns=_case_interpretation_matrix_columns(top_ns))


def _empty_case_interpretation_matrix_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_CASE_INTERPRETATION_MATRIX_SUMMARY_COLUMNS)


def _empty_case_interpretation_matrix_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_CASE_INTERPRETATION_MATRIX_QUEUE_COLUMNS)


def _empty_dataset_interpretation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DATASET_INTERPRETATION_COLUMNS)


def _empty_dataset_interpretation_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS)


def _simplify_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _find_column(frame: pd.DataFrame, aliases: set[str]) -> Optional[str]:
    simplified = {_simplify_column_name(column): column for column in frame.columns}
    for alias in aliases:
        normalized_alias = _simplify_column_name(alias)
        if normalized_alias in simplified:
            return simplified[normalized_alias]
    return None


def _find_column_in_order(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    simplified = {_simplify_column_name(column): column for column in frame.columns}
    for alias in aliases:
        normalized_alias = _simplify_column_name(alias)
        if normalized_alias in simplified:
            return simplified[normalized_alias]
    return None


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_int(value: object) -> Optional[int]:
    try:
        text = _safe_text(value)
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _read_delimited_table(text: str) -> pd.DataFrame:
    payload = str(text or "").strip()
    if not payload:
        return pd.DataFrame()

    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ","},
        {"sep": r"\s+", "engine": "python"},
    )
    for kwargs in attempts:
        try:
            frame = pd.read_csv(io.StringIO(payload), comment="#", **kwargs)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        if len(frame.columns) == 1 and kwargs.get("sep") is None:
            continue
        return frame
    return pd.DataFrame()


def build_pocket_benchmark_reference_template(*, include_examples: bool = True) -> pd.DataFrame:
    """Build an editable curated catalytic residue benchmark template."""

    if not include_examples:
        return _empty_reference_template_df()
    rows = [
        {
            "benchmark_id": "case-001",
            "chain": "A",
            "resid": 195,
            "resname": "SER",
            "reference_type": "Catalytic residue",
            "reference_source": "M-CSA / PMID / DOI",
            "reference_note": "Example nucleophile; replace with curated residue evidence.",
            "expected_pocket_id": "",
        },
        {
            "benchmark_id": "case-001",
            "chain": "A",
            "resid": 57,
            "resname": "HIS",
            "reference_type": "Catalytic residue",
            "reference_source": "M-CSA / PMID / DOI",
            "reference_note": "Example catalytic base; keep benchmark_id identical for the same enzyme/structure.",
            "expected_pocket_id": "",
        },
        {
            "benchmark_id": "case-002",
            "chain": "",
            "resid": 123,
            "resname": "",
            "reference_type": "Binding residue",
            "reference_source": "curated literature",
            "reference_note": "Blank chain acts as wildcard when source numbering is chain-agnostic.",
            "expected_pocket_id": "",
        },
    ]
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_TEMPLATE_COLUMNS)


def build_pocket_benchmark_reference_template_markdown() -> str:
    """Explain the curated benchmark reference template fields."""

    lines = [
        "# Pocket benchmark reference template",
        "",
        "Use this template to collect curated catalytic or binding residues for enzyme pocket accuracy checks.",
        "",
        "## Columns",
        "",
        "- `benchmark_id`: Case identifier. Use one ID per enzyme/structure; examples include a PDB ID, M-CSA entry, or local dataset case ID.",
        "- `chain`: PDB chain. Leave blank only when the curated source is chain-agnostic; blank chain is treated as wildcard.",
        "- `resid`: Required residue number in the numbering system you want to benchmark.",
        "- `resname`: Optional three-letter residue name such as `SER`, `HIS`, or `ASP`.",
        "- `reference_type`: Evidence role, for example `Catalytic residue`, `Binding residue`, `Metal binding`, or `Mutagenesis`.",
        "- `reference_source`: Stable source label such as `M-CSA`, `PMID:...`, `DOI:...`, or a curated dataset name.",
        "- `reference_note`: Free-text note for EC number, UniProt accession, mature-chain offset, paper quote, or curation caveat.",
        "- `expected_pocket_id`: Optional pocket ID if a validated pocket label is already known.",
        "",
        "## Rules",
        "",
        "- Keep all catalytic residues from the same enzyme/structure under the same `benchmark_id`.",
        "- Prefer structure author residue numbering when benchmarking uploaded PDB coordinates.",
        "- If using UniProt or mature-chain numbering, record the mapping assumption in `reference_note`.",
        "- Do not mix different structures under the same `benchmark_id` unless residue numbering is intentionally shared.",
    ]
    return "\n".join(lines).strip() + "\n"


def _extract_residue_token(value: object) -> tuple[str, str, Optional[int]]:
    text = _safe_text(value)
    if not text:
        return "", "", None

    aa3_match = re.search(
        r"\b(?P<resname>Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\.?\s*[- ]?\s*(?P<resid>-?\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if aa3_match:
        return "", aa3_match.group("resname").upper()[:3], _safe_int(aa3_match.group("resid"))

    chain_first = re.search(r"\b(?P<chain>[A-Za-z0-9])\s*[:/_-]\s*(?P<resid>-?\d+)\b", text)
    if chain_first:
        return chain_first.group("chain").strip(), "", _safe_int(chain_first.group("resid"))

    resid_first = re.search(r"\b(?P<resid>-?\d+)\s*[:/_-]\s*(?P<chain>[A-Za-z0-9])\b", text)
    if resid_first:
        return resid_first.group("chain").strip(), "", _safe_int(resid_first.group("resid"))

    one_letter_mutation = re.search(r"\b[A-Z](?P<resid>-?\d+)[A-Z]\b", text)
    if one_letter_mutation:
        return "", "", _safe_int(one_letter_mutation.group("resid"))

    return "", "", _safe_int(text)


def _residue_label(chain: str, resid: int, resname: str = "") -> str:
    resname_text = _safe_text(resname).upper()
    prefix = f"{resname_text} " if resname_text else ""
    chain_text = _safe_text(chain)
    return f"{prefix}{chain_text}{int(resid)}".strip()


def parse_benchmark_reference_table(text: str, *, source_hint: str = "Curated benchmark") -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse curated catalytic/reference residues for pocket accuracy checks.

    Accepted tables are CSV/TSV/whitespace-delimited. Required information is
    only a residue number; chain is optional and acts as a wildcard when blank.
    """

    frame = _read_delimited_table(text)
    if frame.empty:
        return _empty_reference_df(), {
            "status": "empty",
            "reference_rows": "0",
            "reason": "No benchmark rows could be parsed.",
        }

    resid_column = _find_column(frame, RESID_ALIASES)
    token_column = _find_column(frame, RESIDUE_TOKEN_ALIASES)
    chain_column = _find_column(frame, CHAIN_ALIASES)
    resname_column = _find_column(frame, RESNAME_ALIASES)
    type_column = _find_column(frame, TYPE_ALIASES)
    source_column = _find_column_in_order(
        frame,
        (
            "reference_source",
            "evidence_source",
            "source",
            "dataset",
            "citation",
            "reference",
            "pmid",
            "doi",
        ),
    )
    note_column = _find_column(frame, NOTE_ALIASES)
    expected_pocket_column = _find_column(frame, EXPECTED_POCKET_ALIASES)
    benchmark_id_column = _find_column_in_order(
        frame,
        ("benchmark_id", "case_id", "dataset_id", "enzyme_id", "mcsa_id", "entry_id", "pdb_id"),
    )

    rows: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        row_dict = row.to_dict()
        token_chain = ""
        token_resname = ""
        token_resid = None
        if token_column:
            token_chain, token_resname, token_resid = _extract_residue_token(row_dict.get(token_column))

        resid = _safe_int(row_dict.get(resid_column)) if resid_column else None
        if resid is None:
            resid = token_resid
        if resid is None:
            continue

        chain = _safe_text(row_dict.get(chain_column, "") if chain_column else "") or token_chain
        resname = _safe_text(row_dict.get(resname_column, "") if resname_column else "").upper() or token_resname
        rows.append(
            {
                "benchmark_id": _safe_text(row_dict.get(benchmark_id_column, "") if benchmark_id_column else ""),
                "chain": chain,
                "resid": int(resid),
                "resname": resname,
                "reference_type": _safe_text(row_dict.get(type_column, "") if type_column else "Catalytic residue") or "Catalytic residue",
                "reference_source": _safe_text(row_dict.get(source_column, "") if source_column else source_hint) or source_hint,
                "reference_note": _safe_text(row_dict.get(note_column, "") if note_column else ""),
                "expected_pocket_id": _safe_text(row_dict.get(expected_pocket_column, "") if expected_pocket_column else ""),
            }
        )

    if not rows:
        return _empty_reference_df(), {
            "status": "empty",
            "reference_rows": "0",
            "reason": "No valid residue numbers were found.",
        }

    reference_df = pd.DataFrame(rows)
    reference_df = reference_df.drop_duplicates(subset=["benchmark_id", "chain", "resid", "reference_type"]).sort_values(
        ["benchmark_id", "chain", "resid", "reference_type"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return reference_df[BENCHMARK_REFERENCE_COLUMNS], {
        "status": "ok",
        "reference_rows": str(len(reference_df)),
        "source": source_hint,
        "chain_specific_rows": str(int(reference_df["chain"].astype(str).str.strip().ne("").sum())),
        "wildcard_chain_rows": str(int(reference_df["chain"].astype(str).str.strip().eq("").sum())),
    }


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "review", "needs-review", "manual-review"}


def _external_reference_source(row: dict[str, object], source_hint: str) -> str:
    values = [_safe_text(row.get("evidence_source")) or source_hint]
    for column, prefix in (("pmid", "PMID"), ("pmcid", "PMCID"), ("doi", "DOI")):
        value = _safe_text(row.get(column))
        if value:
            values.append(f"{prefix}:{value}")
    return "; ".join(dict.fromkeys(value for value in values if value))


def _external_reference_note(row: dict[str, object]) -> str:
    parts: list[str] = []
    for column in ("evidence_note", "article_title", "evidence_snippet"):
        value = _safe_text(row.get(column))
        if value:
            parts.append(value[:240])

    mapping_parts: list[str] = []
    mapping_level = _safe_text(row.get("mapping_level"))
    mapping_method = _safe_text(row.get("mapping_method"))
    mapping_confidence = _safe_text(row.get("mapping_confidence"))
    uniprot_resid = _safe_text(row.get("uniprot_resid"))
    if mapping_level:
        mapping_parts.append(f"mapping_level={mapping_level}")
    if mapping_confidence:
        mapping_parts.append(f"mapping_confidence={mapping_confidence}")
    if mapping_method:
        mapping_parts.append(f"mapping_method={mapping_method}")
    if uniprot_resid and uniprot_resid not in {"0", "0.0"}:
        mapping_parts.append(f"uniprot_resid={uniprot_resid}")
    if _safe_bool(row.get("requires_manual_review")):
        mapping_parts.append("requires_manual_review=true")
    if mapping_parts:
        parts.append("; ".join(mapping_parts))
    return " | ".join(dict.fromkeys(part for part in parts if part))


def build_pocket_benchmark_reference_from_external_evidence(
    evidence_df: Optional[pd.DataFrame],
    *,
    default_benchmark_id: str = "current-structure",
    source_hint: str = "External residue evidence",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert loaded residue evidence into a benchmark reference candidate table.

    The returned table is intentionally a candidate: if the same evidence was
    used for pocket detection, it still needs independent curation before it is
    used as an accuracy claim.
    """

    if evidence_df is None or getattr(evidence_df, "empty", True):
        return _empty_reference_df(), {
            "status": "empty",
            "source": source_hint,
            "evidence_rows": "0",
            "reference_rows": "0",
            "reason": "No external evidence rows were available.",
        }

    frame = evidence_df.copy()
    resid_column = _find_column(frame, RESID_ALIASES)
    token_column = _find_column(frame, RESIDUE_TOKEN_ALIASES)
    chain_column = _find_column(frame, CHAIN_ALIASES)
    resname_column = _find_column(frame, RESNAME_ALIASES)
    type_column = _find_column(frame, TYPE_ALIASES)
    source_column = _find_column_in_order(
        frame,
        (
            "reference_source",
            "evidence_source",
            "source",
            "dataset",
            "citation",
            "reference",
            "pmid",
            "doi",
        ),
    )
    note_column = _find_column(frame, NOTE_ALIASES)
    expected_pocket_column = _find_column(frame, EXPECTED_POCKET_ALIASES)
    benchmark_id_column = _find_column_in_order(
        frame,
        ("benchmark_id", "case_id", "dataset_id", "enzyme_id", "mcsa_id", "entry_id", "pdb_id"),
    )
    pdb_id_column = _find_column(frame, PDB_ID_ALIASES)

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        token_chain = ""
        token_resname = ""
        token_resid = None
        if token_column:
            token_chain, token_resname, token_resid = _extract_residue_token(row_dict.get(token_column))

        resid = _safe_int(row_dict.get(resid_column)) if resid_column else None
        if resid is None:
            resid = token_resid
        if resid is None:
            continue

        benchmark_id = _safe_text(row_dict.get(benchmark_id_column, "") if benchmark_id_column else "")
        if not benchmark_id and pdb_id_column:
            benchmark_id = _safe_text(row_dict.get(pdb_id_column))
        if not benchmark_id:
            benchmark_id = _safe_text(default_benchmark_id) or "current-structure"

        chain = _safe_text(row_dict.get(chain_column, "") if chain_column else "") or token_chain
        resname = _safe_text(row_dict.get(resname_column, "") if resname_column else "").upper() or token_resname
        evidence_type = _safe_text(row_dict.get(type_column, "") if type_column else "") or "Functional site"
        source_value = _safe_text(row_dict.get(source_column, "") if source_column else "")
        note_value = _safe_text(row_dict.get(note_column, "") if note_column else "")
        enriched_row = {
            **row_dict,
            "evidence_source": source_value or row_dict.get("evidence_source") or source_hint,
            "evidence_note": note_value or row_dict.get("evidence_note") or "",
        }
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "chain": chain,
                "resid": int(resid),
                "resname": resname,
                "reference_type": evidence_type,
                "reference_source": _external_reference_source(enriched_row, source_hint),
                "reference_note": _external_reference_note(enriched_row),
                "expected_pocket_id": _safe_text(row_dict.get(expected_pocket_column, "") if expected_pocket_column else ""),
            }
        )

    evidence_rows = int(len(frame))
    if not rows:
        return _empty_reference_df(), {
            "status": "empty",
            "source": source_hint,
            "evidence_rows": str(evidence_rows),
            "reference_rows": "0",
            "skipped_rows": str(evidence_rows),
            "reason": "No valid residue numbers were found in external evidence.",
        }

    raw_df = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_COLUMNS)
    grouped_rows: list[dict[str, object]] = []
    group_columns = ["benchmark_id", "chain", "resid", "reference_type", "expected_pocket_id"]
    for _, group in raw_df.groupby(group_columns, dropna=False, sort=True):
        sources = "; ".join(
            dict.fromkeys(group["reference_source"].map(_safe_text).replace("", pd.NA).dropna().tolist())
        )
        notes = " | ".join(
            dict.fromkeys(group["reference_note"].map(_safe_text).replace("", pd.NA).dropna().tolist())
        )
        resnames = group["resname"].map(_safe_text).replace("", pd.NA).dropna().tolist()
        first = group.iloc[0]
        grouped_rows.append(
            {
                "benchmark_id": _safe_text(first.get("benchmark_id")),
                "chain": _safe_text(first.get("chain")),
                "resid": int(first.get("resid")),
                "resname": _safe_text(resnames[0]).upper() if resnames else "",
                "reference_type": _safe_text(first.get("reference_type")),
                "reference_source": sources or source_hint,
                "reference_note": notes,
                "expected_pocket_id": _safe_text(first.get("expected_pocket_id")),
            }
        )

    reference_df = pd.DataFrame(grouped_rows, columns=BENCHMARK_REFERENCE_COLUMNS).sort_values(
        ["benchmark_id", "chain", "resid", "reference_type"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    mapping_levels = frame["mapping_level"].astype(str).str.lower() if "mapping_level" in frame.columns else pd.Series(dtype=str)
    manual_review_rows = (
        int(frame["requires_manual_review"].map(_safe_bool).sum())
        if "requires_manual_review" in frame.columns
        else 0
    )
    structure_verified_rows = 0
    if "structure_verified" in frame.columns:
        structure_verified_rows = int(frame["structure_verified"].map(_safe_bool).sum())
    elif not mapping_levels.empty:
        structure_verified_rows = int(mapping_levels.eq("exact").sum())

    return reference_df[BENCHMARK_REFERENCE_COLUMNS], {
        "status": "ok",
        "source": source_hint,
        "evidence_rows": str(evidence_rows),
        "reference_rows": str(len(reference_df)),
        "case_count": str(int(reference_df["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().nunique())),
        "chain_specific_rows": str(int(reference_df["chain"].map(_safe_text).ne("").sum())),
        "wildcard_chain_rows": str(int(reference_df["chain"].map(_safe_text).eq("").sum())),
        "exact_mapping_rows": str(int(mapping_levels.eq("exact").sum()) if not mapping_levels.empty else 0),
        "weak_mapping_rows": str(int(mapping_levels.eq("weak").sum()) if not mapping_levels.empty else 0),
        "structure_verified_rows": str(structure_verified_rows),
        "manual_review_rows": str(manual_review_rows),
        "missing_resname_rows": str(int(reference_df["resname"].map(_safe_text).eq("").sum())),
        "duplicate_rows": str(max(0, len(raw_df) - len(reference_df))),
        "skipped_rows": str(max(0, evidence_rows - len(raw_df))),
    }


def build_pocket_benchmark_reference_import_summary(
    reference_df: Optional[pd.DataFrame],
    metadata: Optional[dict[str, object]] = None,
) -> pd.DataFrame:
    """Summarize an external-evidence-to-benchmark-reference import."""

    meta = dict(metadata or {})
    references = _reference_rows(reference_df)
    if references.empty:
        evidence_rows = int(meta.get("evidence_rows") or 0)
        skipped_rows = int(meta.get("skipped_rows") or evidence_rows or 0)
        return pd.DataFrame(
            [
                {
                    "source": _safe_text(meta.get("source")) or "External residue evidence",
                    "import_status": "empty",
                    "evidence_rows": evidence_rows,
                    "reference_rows": 0,
                    "case_count": 0,
                    "chain_specific_rows": 0,
                    "wildcard_chain_rows": 0,
                    "exact_mapping_rows": int(meta.get("exact_mapping_rows") or 0),
                    "weak_mapping_rows": int(meta.get("weak_mapping_rows") or 0),
                    "structure_verified_rows": int(meta.get("structure_verified_rows") or 0),
                    "manual_review_rows": int(meta.get("manual_review_rows") or 0),
                    "missing_resname_rows": 0,
                    "duplicate_rows": int(meta.get("duplicate_rows") or 0),
                    "skipped_rows": skipped_rows,
                    "recommended_action": "Fetch UniProt/M-CSA/literature evidence or upload a curated benchmark reference CSV.",
                    "import_warning": _safe_text(meta.get("reason")) or "No benchmark reference candidate rows were produced.",
                }
            ],
            columns=BENCHMARK_REFERENCE_IMPORT_SUMMARY_COLUMNS,
        )

    wildcard_chain_rows = int(meta.get("wildcard_chain_rows") or references["chain"].map(_safe_text).eq("").sum())
    weak_mapping_rows = int(meta.get("weak_mapping_rows") or 0)
    manual_review_rows = int(meta.get("manual_review_rows") or 0)
    missing_resname_rows = int(meta.get("missing_resname_rows") or references["resname"].map(_safe_text).eq("").sum())
    review_needed = any(value > 0 for value in (wildcard_chain_rows, weak_mapping_rows, manual_review_rows, missing_resname_rows))
    import_status = "review-needed" if review_needed else "candidate-ready"
    recommended_action = (
        "Review weak mappings, wildcard chains, manual-review rows and missing residue names before using this as a benchmark."
        if review_needed
        else "Export this candidate, then lock it as an independently curated benchmark before claiming accuracy."
    )
    warning = (
        "External evidence candidates can overlap with detection inputs; treat them as curation input, not independent accuracy proof."
    )

    return pd.DataFrame(
        [
            {
                "source": _safe_text(meta.get("source")) or "External residue evidence",
                "import_status": import_status,
                "evidence_rows": int(meta.get("evidence_rows") or len(references)),
                "reference_rows": int(meta.get("reference_rows") or len(references)),
                "case_count": int(meta.get("case_count") or references["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().nunique()),
                "chain_specific_rows": int(meta.get("chain_specific_rows") or references["chain"].map(_safe_text).ne("").sum()),
                "wildcard_chain_rows": wildcard_chain_rows,
                "exact_mapping_rows": int(meta.get("exact_mapping_rows") or 0),
                "weak_mapping_rows": weak_mapping_rows,
                "structure_verified_rows": int(meta.get("structure_verified_rows") or 0),
                "manual_review_rows": manual_review_rows,
                "missing_resname_rows": missing_resname_rows,
                "duplicate_rows": int(meta.get("duplicate_rows") or 0),
                "skipped_rows": int(meta.get("skipped_rows") or 0),
                "recommended_action": recommended_action,
                "import_warning": warning,
            }
        ],
        columns=BENCHMARK_REFERENCE_IMPORT_SUMMARY_COLUMNS,
    )


def _reference_note_key_value(note_text: object, key: str) -> str:
    pattern = rf"(?:^|[;|]\s*){re.escape(key)}\s*=\s*([^;|]+)"
    match = re.search(pattern, _safe_text(note_text), flags=re.IGNORECASE)
    return _safe_text(match.group(1)) if match else ""


def _candidate_review_issue(
    issue_type: str,
) -> tuple[str, str, str, str]:
    if issue_type == "manual-review-required":
        return (
            "P1",
            "review",
            "Open the source article or AI evidence audit and accept/reject this residue before benchmark use.",
            "Manual-review evidence is not safe as a benchmark reference until a reviewer confirms it.",
        )
    if issue_type == "weak-mapping":
        return (
            "P1",
            "review",
            "Resolve the residue through SIFTS/structure numbering or keep it out of accuracy claims.",
            "Weak mapping can shift catalytic residues onto the wrong PDB position.",
        )
    if issue_type == "generic-source":
        return (
            "P1",
            "review",
            "Replace the source with M-CSA ID, UniProt accession, PMID, DOI, or a named curated dataset.",
            "Generic sources make external-evidence candidate references hard to reproduce.",
        )
    if issue_type == "wildcard-chain":
        return (
            "P2",
            "review",
            "Assign the PDB chain or document why chain-agnostic matching is intended.",
            "Wildcard chain rows can overestimate benchmark coverage in multi-chain structures.",
        )
    if issue_type == "missing-resname":
        return (
            "P2",
            "review",
            "Fill the expected three-letter residue name and verify it against the uploaded structure.",
            "Missing residue identity weakens residue-number validation.",
        )
    if issue_type == "unknown-mapping":
        return (
            "P2",
            "review",
            "Record mapping_level, mapping_method and mapping_confidence before benchmark use.",
            "Unknown mapping provenance makes it unclear whether the candidate is structure-verified.",
        )
    return "P3", "review", "Document the curation decision.", "Candidate reference row needs reviewer documentation."


def build_pocket_benchmark_reference_candidate_review_queue(reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Build a row-level review queue for benchmark reference candidates."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_candidate_review_queue_df()

    rows: list[dict[str, object]] = []
    for _, reference in references.iterrows():
        chain = _safe_text(reference.get("chain"))
        resid = int(reference.get("resid"))
        resname = _safe_text(reference.get("resname")).upper()
        reference_source = _safe_text(reference.get("reference_source"))
        reference_note = _safe_text(reference.get("reference_note"))
        mapping_level = _reference_note_key_value(reference_note, "mapping_level").lower()
        mapping_confidence = _reference_note_key_value(reference_note, "mapping_confidence")
        mapping_method = _reference_note_key_value(reference_note, "mapping_method")
        issue_types: list[str] = []

        if "requires_manual_review=true" in reference_note.lower():
            issue_types.append("manual-review-required")
        if mapping_level == "weak":
            issue_types.append("weak-mapping")
        elif not mapping_level:
            issue_types.append("unknown-mapping")
        if not chain:
            issue_types.append("wildcard-chain")
        if not resname:
            issue_types.append("missing-resname")
        if _source_is_generic(reference_source):
            issue_types.append("generic-source")

        for issue_type in dict.fromkeys(issue_types):
            priority, action_status, suggested_action, warning = _candidate_review_issue(issue_type)
            rows.append(
                {
                    "priority": priority,
                    "action_status": action_status,
                    "benchmark_id": _safe_text(reference.get("benchmark_id")),
                    "chain": chain,
                    "resid": resid,
                    "resname": resname,
                    "residue_label": _residue_label(chain, resid, resname),
                    "issue_type": issue_type,
                    "reference_type": _safe_text(reference.get("reference_type")),
                    "reference_source": reference_source,
                    "mapping_level": mapping_level,
                    "mapping_confidence": mapping_confidence,
                    "mapping_method": mapping_method,
                    "suggested_action": suggested_action,
                    "review_warning": warning,
                }
            )

    if not rows:
        return _empty_reference_candidate_review_queue_df()

    frame = pd.DataFrame(rows)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(
        ["_priority_rank", "benchmark_id", "resid", "issue_type"],
        ascending=[True, True, True, True],
    ).drop(columns=["_priority_rank"]).reset_index(drop=True)
    frame["action_id"] = [f"REFC-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS]


def build_pocket_benchmark_reference_candidate_review_checklist_markdown(
    review_queue_df: Optional[pd.DataFrame],
) -> str:
    """Render reference-candidate review actions as a Markdown checklist."""

    if review_queue_df is None or getattr(review_queue_df, "empty", True):
        return ""
    queue = review_queue_df.copy()
    for column in BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""

    lines = [
        "# Benchmark reference candidate review checklist",
        "",
        "Resolve these items before promoting external-evidence candidates to an independent benchmark reference table.",
        "",
        "## Summary",
        "",
    ]
    for (priority, issue_type), group in queue.groupby(["priority", "issue_type"], dropna=False):
        affected_residues = int(group[["benchmark_id", "chain", "resid"]].astype(str).drop_duplicates().shape[0])
        lines.append(
            f"- {priority} `{issue_type}`: {len(group)} actions / {affected_residues} residues."
        )
    lines.extend(["", "## Actions", ""])
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    queue["_priority_rank"] = queue["priority"].map(priority_rank).fillna(99)
    queue = queue.sort_values(["_priority_rank", "benchmark_id", "resid", "issue_type"]).drop(columns=["_priority_rank"])
    for row in queue.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        lines.append(
            f"- [ ] {row.priority} `{row.issue_type}` for {case_text}, residue `{row.residue_label}`: {row.suggested_action}"
        )
    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_reference_candidate_review_decision_template(
    review_queue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build an editable decision template for candidate reference review actions."""

    if review_queue_df is None or getattr(review_queue_df, "empty", True):
        return _empty_reference_candidate_review_decision_template_df()
    queue = review_queue_df.copy()
    for column in BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    template = queue[BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS].copy()
    template["review_decision"] = "review"
    template["reviewer"] = ""
    template["verified_source"] = ""
    template["verified_mapping"] = ""
    template["review_note"] = ""
    return template[BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS]


def _normalize_candidate_review_decision(value: object) -> str:
    text = _safe_text(value).lower()
    if text in {"accept", "accepted", "approve", "approved", "yes", "pass"}:
        return "accept"
    if text in {"reject", "rejected", "deny", "denied", "no", "fail"}:
        return "reject"
    if text in {"hold", "defer", "deferred", "blocked", "block"}:
        return "hold"
    if text in {"", "review", "needs-review", "pending"}:
        return "review"
    return "unknown"


def parse_pocket_benchmark_reference_candidate_review_decision_table(
    decision_text: str | bytes | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse reviewer decisions for benchmark reference candidate review actions."""

    if decision_text is None:
        return _empty_reference_candidate_review_decision_template_df(), {
            "status": "empty",
            "decision_rows": "0",
            "reason": "No decision table was provided.",
        }
    if isinstance(decision_text, bytes):
        decision_text = decision_text.decode("utf-8", errors="ignore")
    frame = _read_delimited_table(str(decision_text or ""))
    if frame.empty:
        return _empty_reference_candidate_review_decision_template_df(), {
            "status": "empty",
            "decision_rows": "0",
            "reason": "No decision rows could be parsed.",
        }

    column_aliases: dict[str, Sequence[str]] = {
        "action_id": ("action_id", "actionid", "id", "review_action_id"),
        "review_decision": ("review_decision", "decision", "review", "status", "approval"),
        "reviewer": ("reviewer", "reviewer_name", "curator", "user"),
        "verified_source": ("verified_source", "source", "verified_sources", "evidence_source"),
        "verified_mapping": ("verified_mapping", "mapping", "verified_mapping_note", "mapping_note"),
        "review_note": ("review_note", "note", "notes", "comment", "comments"),
    }
    selected = {column: _find_column_in_order(frame, aliases) for column, aliases in column_aliases.items()}
    if not selected["action_id"] or not selected["review_decision"]:
        return _empty_reference_candidate_review_decision_template_df(), {
            "status": "invalid",
            "decision_rows": "0",
            "reason": "action_id and review_decision columns are required.",
        }

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        action_id = _safe_text(row_dict.get(selected["action_id"]))
        if not action_id:
            continue
        rows.append(
            {
                "action_id": action_id,
                "review_decision": _normalize_candidate_review_decision(row_dict.get(selected["review_decision"])),
                "reviewer": _safe_text(row_dict.get(selected["reviewer"])) if selected["reviewer"] else "",
                "verified_source": _safe_text(row_dict.get(selected["verified_source"])) if selected["verified_source"] else "",
                "verified_mapping": _safe_text(row_dict.get(selected["verified_mapping"])) if selected["verified_mapping"] else "",
                "review_note": _safe_text(row_dict.get(selected["review_note"])) if selected["review_note"] else "",
            }
        )

    if not rows:
        return _empty_reference_candidate_review_decision_template_df(), {
            "status": "empty",
            "decision_rows": "0",
            "reason": "No valid action_id rows were found.",
        }

    decisions = pd.DataFrame(rows)
    for column in BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    decision_counts = decisions["review_decision"].astype(str).value_counts().to_dict()
    return decisions[BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS], {
        "status": "ok",
        "decision_rows": str(len(decisions)),
        "accept_rows": str(int(decision_counts.get("accept", 0))),
        "reject_rows": str(int(decision_counts.get("reject", 0))),
        "hold_rows": str(int(decision_counts.get("hold", 0))),
        "review_rows": str(int(decision_counts.get("review", 0))),
        "unknown_rows": str(int(decision_counts.get("unknown", 0))),
    }


def build_pocket_benchmark_reference_candidate_review_decision_validation(
    decision_df: Optional[pd.DataFrame],
    review_queue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Validate uploaded candidate-reference review decisions before applying them."""

    if decision_df is None or getattr(decision_df, "empty", True):
        return _empty_reference_candidate_review_decision_validation_df()
    decisions = decision_df.copy()
    for column in BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    queue = review_queue_df.copy() if review_queue_df is not None and not getattr(review_queue_df, "empty", True) else _empty_reference_candidate_review_queue_df()
    valid_action_ids = set(queue["action_id"].map(_safe_text).tolist()) if "action_id" in queue.columns else set()
    duplicate_decisions: dict[str, set[str]] = {}
    for _, row in decisions.iterrows():
        action_id = _safe_text(row.get("action_id"))
        if action_id:
            duplicate_decisions.setdefault(action_id, set()).add(_safe_text(row.get("review_decision")))

    rows: list[dict[str, object]] = []
    for index, decision in decisions.reset_index(drop=True).iterrows():
        action_id = _safe_text(decision.get("action_id"))
        review_decision = _normalize_candidate_review_decision(decision.get("review_decision"))
        reviewer = _safe_text(decision.get("reviewer"))
        verified_source = _safe_text(decision.get("verified_source"))
        verified_mapping = _safe_text(decision.get("verified_mapping"))
        review_note = _safe_text(decision.get("review_note"))
        issues: list[str] = []
        fixes: list[str] = []

        if not action_id:
            issues.append("missing-action-id")
            fixes.append("Fill action_id from the exported review queue.")
        elif action_id not in valid_action_ids:
            issues.append("unknown-action-id")
            fixes.append("Use action_id values from the current review queue.")
        if len(duplicate_decisions.get(action_id, set())) > 1:
            issues.append("conflicting-duplicate")
            fixes.append("Keep only one decision per action_id.")
        if review_decision == "unknown":
            issues.append("unknown-decision")
            fixes.append("Use accept, reject, hold, or review.")
        if review_decision in {"accept", "reject", "hold"} and not reviewer:
            issues.append("missing-reviewer")
            fixes.append("Fill reviewer for every non-review decision.")
        if review_decision == "accept" and not (verified_source or verified_mapping or review_note):
            issues.append("missing-acceptance-evidence")
            fixes.append("Fill verified_source, verified_mapping, or review_note before accepting a candidate action.")

        validation_status = "blocked" if issues else ("review" if review_decision == "review" else "ok")
        rows.append(
            {
                "row_index": int(index + 1),
                "action_id": action_id,
                "review_decision": review_decision,
                "validation_status": validation_status,
                "issue_flags": ";".join(dict.fromkeys(issues)),
                "required_fix": " ".join(dict.fromkeys(fixes)),
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_VALIDATION_COLUMNS)


def build_pocket_benchmark_reference_candidate_review_outcomes(
    review_queue_df: Optional[pd.DataFrame],
    decision_df: Optional[pd.DataFrame],
    validation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply validated reviewer decisions to each candidate-reference review action."""

    if review_queue_df is None or getattr(review_queue_df, "empty", True):
        return _empty_reference_candidate_review_outcome_df()
    queue = review_queue_df.copy()
    for column in BENCHMARK_REFERENCE_CANDIDATE_REVIEW_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    decisions = decision_df.copy() if decision_df is not None and not getattr(decision_df, "empty", True) else pd.DataFrame()
    for column in BENCHMARK_REFERENCE_CANDIDATE_REVIEW_DECISION_TEMPLATE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    validation = validation_df.copy() if validation_df is not None and not getattr(validation_df, "empty", True) else pd.DataFrame()
    if validation.empty and not decisions.empty:
        validation = build_pocket_benchmark_reference_candidate_review_decision_validation(decisions, queue)

    decision_by_action = {
        _safe_text(row.get("action_id")): row
        for _, row in decisions.iterrows()
        if _safe_text(row.get("action_id"))
    }
    validation_by_action = {
        _safe_text(row.get("action_id")): row
        for _, row in validation.iterrows()
        if _safe_text(row.get("action_id"))
    }

    rows: list[dict[str, object]] = []
    for _, action in queue.iterrows():
        action_id = _safe_text(action.get("action_id"))
        decision = decision_by_action.get(action_id)
        validation_row = validation_by_action.get(action_id)
        review_decision = _normalize_candidate_review_decision(decision.get("review_decision")) if decision is not None else "review"
        validation_status = _safe_text(validation_row.get("validation_status")) if validation_row is not None else ""
        if decision is None:
            applied_status = "pending"
            reason = "No reviewer decision uploaded for this action."
            next_action = "Fill the candidate review decision template."
        elif validation_status == "blocked":
            applied_status = "blocked"
            reason = _safe_text(validation_row.get("issue_flags"))
            next_action = _safe_text(validation_row.get("required_fix")) or "Fix validation issues and re-upload decisions."
        elif review_decision == "accept":
            applied_status = "accepted"
            reason = "Reviewer accepted this candidate-review action."
            next_action = "No action needed for this issue."
        elif review_decision == "reject":
            applied_status = "rejected"
            reason = "Reviewer rejected this candidate-reference issue or residue."
            next_action = "Do not promote this residue candidate until evidence is replaced."
        elif review_decision == "hold":
            applied_status = "held"
            reason = "Reviewer placed this candidate action on hold."
            next_action = "Resolve blocker or add evidence before promotion."
        else:
            applied_status = "pending"
            reason = "Reviewer left this action in review."
            next_action = "Complete accept/reject/hold decision."

        rows.append(
            {
                "action_id": action_id,
                "priority": _safe_text(action.get("priority")),
                "benchmark_id": _safe_text(action.get("benchmark_id")),
                "chain": _safe_text(action.get("chain")),
                "resid": int(action.get("resid") or 0),
                "resname": _safe_text(action.get("resname")).upper(),
                "residue_label": _safe_text(action.get("residue_label")),
                "issue_type": _safe_text(action.get("issue_type")),
                "review_decision": review_decision,
                "applied_status": applied_status,
                "outcome_reason": reason,
                "next_action": next_action,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_CANDIDATE_REVIEW_OUTCOME_COLUMNS)


def build_pocket_benchmark_reference_candidate_accepted_reference(
    reference_df: Optional[pd.DataFrame],
    review_queue_df: Optional[pd.DataFrame],
    review_outcome_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Promote only clean or fully accepted candidate reference residues."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_df()
    queue = review_queue_df.copy() if review_queue_df is not None and not getattr(review_queue_df, "empty", True) else _empty_reference_candidate_review_queue_df()
    outcomes = review_outcome_df.copy() if review_outcome_df is not None and not getattr(review_outcome_df, "empty", True) else _empty_reference_candidate_review_outcome_df()

    def key_from_row(row: object) -> tuple[str, str, int]:
        return (_safe_text(row.get("benchmark_id")), _safe_text(row.get("chain")), int(row.get("resid") or 0))

    risk_actions_by_key: dict[tuple[str, str, int], set[str]] = {}
    for _, action in queue.iterrows():
        risk_actions_by_key.setdefault(key_from_row(action), set()).add(_safe_text(action.get("action_id")))

    accepted_actions = {
        _safe_text(row.get("action_id"))
        for _, row in outcomes.iterrows()
        if _safe_text(row.get("applied_status")) == "accepted"
    }

    accepted_rows: list[dict[str, object]] = []
    for _, reference in references.iterrows():
        key = key_from_row(reference)
        risk_action_ids = risk_actions_by_key.get(key, set())
        if risk_action_ids and not risk_action_ids.issubset(accepted_actions):
            continue
        accepted_rows.append(reference.to_dict())

    if not accepted_rows:
        return _empty_reference_df()
    return pd.DataFrame(accepted_rows, columns=BENCHMARK_REFERENCE_COLUMNS).reset_index(drop=True)


def select_pocket_benchmark_reference_source(
    curated_reference_df: Optional[pd.DataFrame],
    curated_reference_meta: Optional[dict[str, object]] = None,
    *,
    curated_reference_uploaded: bool = False,
    external_candidate_df: Optional[pd.DataFrame] = None,
    external_candidate_meta: Optional[dict[str, object]] = None,
    accepted_candidate_df: Optional[pd.DataFrame] = None,
    prefer_reviewed_candidate: bool = True,
    allow_provisional_candidate: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Select the benchmark reference source with curated/reviewed/provisional priority."""

    if curated_reference_uploaded:
        return (
            _reference_rows(curated_reference_df),
            dict(curated_reference_meta or {}),
            {
                "loaded": True,
                "source_mode": "uploaded-curated",
                "is_provisional": False,
                "is_reviewed_candidate": False,
                "message": "",
            },
        )

    candidate_meta = dict(external_candidate_meta or {})
    accepted_references = _reference_rows(accepted_candidate_df)
    if prefer_reviewed_candidate and not accepted_references.empty:
        return (
            accepted_references,
            {
                **candidate_meta,
                "source": "Accepted reviewed external-evidence benchmark reference",
            },
            {
                "loaded": True,
                "source_mode": "accepted-reviewed-candidate",
                "is_provisional": False,
                "is_reviewed_candidate": True,
                "message": "Benchmark reference: using accepted reviewed candidate references.",
            },
        )

    candidate_references = _reference_rows(external_candidate_df)
    if allow_provisional_candidate:
        if candidate_references.empty:
            return (
                _empty_reference_df(),
                {},
                {
                    "loaded": False,
                    "source_mode": "",
                    "is_provisional": False,
                    "is_reviewed_candidate": False,
                    "message": "Benchmark reference: external evidence candidate is empty; upload a curated reference file instead.",
                },
            )
        return (
            candidate_references,
            {
                **candidate_meta,
                "source": "Provisional external evidence benchmark reference",
            },
            {
                "loaded": True,
                "source_mode": "provisional-external-evidence",
                "is_provisional": True,
                "is_reviewed_candidate": False,
                "message": "Benchmark reference: using provisional external-evidence candidate.",
            },
        )

    message = ""
    if prefer_reviewed_candidate and not candidate_references.empty:
        message = (
            "Benchmark reference: no accepted reviewed candidate rows yet; "
            "upload accepted review decisions or a curated reference file."
        )
    return (
        _empty_reference_df(),
        {},
        {
            "loaded": False,
            "source_mode": "",
            "is_provisional": False,
            "is_reviewed_candidate": False,
            "message": message,
        },
    )


def build_pocket_benchmark_reference_source_audit(
    reference_df: Optional[pd.DataFrame],
    *,
    source_mode: str = "",
    is_provisional: bool = False,
    is_reviewed_candidate: bool = False,
) -> pd.DataFrame:
    """Build a row-level provenance audit for the benchmark references actually used."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_source_audit_df()

    normalized_source_mode = _safe_text(source_mode) or "unknown"
    if normalized_source_mode == "uploaded-curated":
        source_claim_status = "source-ready"
        can_support_independent_claim = "yes"
        review_status = "curated-upload"
        provenance_warning = ""
        recommended_action = "Use readiness and structure-validation gates before reporting benchmark coverage."
    elif normalized_source_mode == "accepted-reviewed-candidate":
        source_claim_status = "review-qualified"
        can_support_independent_claim = "review-required"
        review_status = "reviewer-accepted"
        provenance_warning = (
            "Reference was promoted from evidence that may also have influenced candidate detection or reranking."
        )
        recommended_action = (
            "Keep the source audit with benchmark exports and verify independence before making accuracy claims."
        )
    elif bool(is_provisional) or normalized_source_mode == "provisional-external-evidence":
        source_claim_status = "blocked-provisional"
        can_support_independent_claim = "no"
        review_status = "unreviewed-provisional"
        provenance_warning = (
            "Provisional external evidence can overlap with detection inputs and is not an independent benchmark."
        )
        recommended_action = "Curate or accept review decisions before using this reference for precision claims."
    else:
        source_claim_status = "source-unknown"
        can_support_independent_claim = "review-required"
        review_status = "unknown"
        provenance_warning = "Benchmark reference source mode is not recorded."
        recommended_action = "Record whether this reference came from curated upload, reviewed candidate, or provisional evidence."

    rows: list[dict[str, object]] = []
    for index, reference in references.reset_index(drop=True).iterrows():
        rows.append(
            {
                "audit_id": f"BRS-{index + 1:03d}",
                "source_mode": normalized_source_mode,
                "source_claim_status": source_claim_status,
                "can_support_independent_claim": can_support_independent_claim,
                "is_provisional": bool(is_provisional),
                "is_reviewed_candidate": bool(is_reviewed_candidate),
                **{column: reference.get(column, "") for column in BENCHMARK_REFERENCE_COLUMNS},
                "review_status": review_status,
                "provenance_warning": provenance_warning,
                "recommended_action": recommended_action,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS)


def _source_audit_summary_action(source_claim_status: str, independent_claim_status: str) -> tuple[str, str]:
    if source_claim_status == "blocked-provisional" or independent_claim_status == "no":
        return (
            "Replace provisional rows with curated references or accepted review decisions before reporting precision.",
            "These rows cannot support independent benchmark claims.",
        )
    if source_claim_status == "source-unknown":
        return (
            "Record source mode and provenance before using these rows in benchmark claims.",
            "Unknown source provenance blocks reliable benchmark interpretation.",
        )
    if source_claim_status == "review-qualified" or independent_claim_status == "review-required":
        return (
            "Confirm reviewed candidate references are independent from detection/reranking evidence.",
            "These rows are useful for triage, but final accuracy claims need independence review.",
        )
    return (
        "Keep readiness and structure-validation outputs attached to benchmark reports.",
        "Source provenance is ready; remaining blockers, if any, come from other readiness gates.",
    )


def build_pocket_benchmark_reference_source_audit_summary(
    source_audit_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize benchmark reference source provenance by claim-safety status."""

    if source_audit_df is None or getattr(source_audit_df, "empty", True):
        return _empty_reference_source_audit_summary_df()
    working = source_audit_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    for column in ("source_mode", "source_claim_status", "can_support_independent_claim", "benchmark_id"):
        working[column] = working[column].map(_safe_text)
    working["is_provisional"] = working["is_provisional"].astype(bool)
    working["is_reviewed_candidate"] = working["is_reviewed_candidate"].astype(bool)

    rows: list[dict[str, object]] = []
    grouped = working.groupby(["source_mode", "source_claim_status", "can_support_independent_claim"], dropna=False)
    for (source_mode, source_claim_status, independent_claim_status), group in grouped:
        action, warning = _source_audit_summary_action(
            _safe_text(source_claim_status),
            _safe_text(independent_claim_status),
        )
        case_count = int(group["benchmark_id"].replace("", pd.NA).dropna().nunique())
        rows.append(
            {
                "summary_id": "",
                "source_mode": _safe_text(source_mode) or "unknown",
                "source_claim_status": _safe_text(source_claim_status) or "source-unknown",
                "can_support_independent_claim": _safe_text(independent_claim_status) or "review-required",
                "reference_rows": int(len(group)),
                "case_count": case_count,
                "provisional_rows": int(group["is_provisional"].sum()),
                "reviewed_candidate_rows": int(group["is_reviewed_candidate"].sum()),
                "recommended_action": action,
                "summary_warning": warning,
            }
        )

    if not rows:
        return _empty_reference_source_audit_summary_df()
    rank = {
        "blocked-provisional": 0,
        "source-unknown": 1,
        "review-qualified": 2,
        "source-ready": 3,
    }
    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_SUMMARY_COLUMNS)
    frame["_rank"] = frame["source_claim_status"].map(rank).fillna(9)
    frame = frame.sort_values(["_rank", "source_mode", "can_support_independent_claim"]).drop(columns=["_rank"]).reset_index(drop=True)
    frame["summary_id"] = [f"BRSS-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_SUMMARY_COLUMNS]


def build_pocket_benchmark_reference_source_audit_checklist_markdown(
    source_audit_summary_df: Optional[pd.DataFrame],
    source_audit_df: Optional[pd.DataFrame],
) -> str:
    """Render source-audit risks as a manual review checklist."""

    summary = (
        source_audit_summary_df.copy()
        if source_audit_summary_df is not None and not getattr(source_audit_summary_df, "empty", True)
        else _empty_reference_source_audit_summary_df()
    )
    audit = (
        source_audit_df.copy()
        if source_audit_df is not None and not getattr(source_audit_df, "empty", True)
        else _empty_reference_source_audit_df()
    )
    if summary.empty and audit.empty:
        return ""
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = ""
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""

    lines = [
        "# Benchmark reference source audit checklist",
        "",
        "Use this checklist before treating catalytic pocket benchmark coverage as an independent precision claim.",
        "",
    ]
    if not summary.empty:
        lines.extend(["## Source Status Summary", ""])
        for row in summary.itertuples(index=False):
            lines.append(
                f"- [ ] {row.summary_id} `{row.source_claim_status}` / `{row.source_mode}`: "
                f"{int(row.reference_rows)} references, {int(row.case_count)} cases. {row.recommended_action}"
            )
        lines.append("")

    risk_rows = audit[audit["source_claim_status"].map(_safe_text).ne("source-ready")].copy()
    if not risk_rows.empty:
        lines.extend(["## Reference Row Actions", ""])
        for row in risk_rows.itertuples(index=False):
            case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
            residue_text = _residue_label(_safe_text(row.chain), int(row.resid), _safe_text(row.resname)) if _safe_text(row.resid) else "-"
            lines.append(
                f"- [ ] {row.audit_id} `{row.source_claim_status}` for {case_text}, residue `{residue_text}`: {row.recommended_action}"
            )
        lines.append("")
    elif not audit.empty:
        lines.extend(
            [
                "## Reference Row Actions",
                "",
                "- [ ] All reference rows are source-ready; keep source audit, readiness, and structure-validation exports with the report.",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_reference_source_audit_action_queue(
    source_audit_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a source-only remediation queue from benchmark reference source audit rows."""

    if source_audit_df is None or getattr(source_audit_df, "empty", True):
        return _empty_reference_source_audit_action_queue_df()
    audit = source_audit_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""

    rows: list[dict[str, object]] = []
    for _, audit_row in audit.iterrows():
        issue = _source_audit_readiness_issue(audit_row)
        if issue is None:
            continue
        priority, issue_type, fallback_action, fallback_warning = issue
        chain = _safe_text(audit_row.get("chain"))
        resid = _safe_int(audit_row.get("resid")) or 0
        resname = _safe_text(audit_row.get("resname")).upper()
        rows.append(
            {
                "action_id": "",
                "priority": priority,
                "action_status": "blocker" if priority in {"P0", "P1"} else "review",
                "issue_type": issue_type,
                "source_mode": _safe_text(audit_row.get("source_mode")),
                "source_claim_status": _safe_text(audit_row.get("source_claim_status")),
                "can_support_independent_claim": _safe_text(audit_row.get("can_support_independent_claim")),
                **{column: audit_row.get(column, "") for column in BENCHMARK_REFERENCE_COLUMNS},
                "residue_label": _residue_label(chain, resid, resname) if resid else "",
                "suggested_action": _safe_text(audit_row.get("recommended_action")) or fallback_action,
                "action_warning": _safe_text(audit_row.get("provenance_warning")) or fallback_warning,
            }
        )

    if not rows:
        return _empty_reference_source_audit_action_queue_df()
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_ACTION_QUEUE_COLUMNS)
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(9)
    frame = frame.sort_values(
        ["_priority_rank", "source_claim_status", "benchmark_id", "chain", "resid"],
        ascending=[True, True, True, True, True],
    ).drop(columns=["_priority_rank"]).reset_index(drop=True)
    frame["action_id"] = [f"BRSA-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_ACTION_QUEUE_COLUMNS]


def _ordered_source_values(values: Sequence[object], rank: dict[str, int] | None = None) -> str:
    unique_values = list(dict.fromkeys(_safe_text(value) for value in values if _safe_text(value)))
    if rank:
        unique_values = sorted(unique_values, key=lambda value: (rank.get(value, 99), value))
    return "; ".join(unique_values)


def build_pocket_benchmark_reference_source_audit_case_summary(
    source_audit_df: Optional[pd.DataFrame],
    action_queue_df: Optional[pd.DataFrame] = None,
    *,
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Summarize source-audit remediation status by benchmark case."""

    if source_audit_df is None or getattr(source_audit_df, "empty", True):
        return _empty_reference_source_audit_case_summary_df()

    fallback_id = _safe_text(default_benchmark_id) or "current"
    audit = source_audit_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""
    audit["benchmark_id"] = audit["benchmark_id"].map(lambda value: _safe_text(value) or fallback_id)

    if action_queue_df is None or getattr(action_queue_df, "empty", True):
        queue = build_pocket_benchmark_reference_source_audit_action_queue(audit)
    else:
        queue = action_queue_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_ACTION_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    if not queue.empty:
        queue["benchmark_id"] = queue["benchmark_id"].map(lambda value: _safe_text(value) or fallback_id)

    case_ids = sorted(
        set(audit["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
        | set(queue["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
    )
    if not case_ids:
        return _empty_reference_source_audit_case_summary_df()

    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    claim_rank = {
        "blocked-provisional": 0,
        "source-unknown": 1,
        "review-qualified": 2,
        "source-ready": 3,
    }
    rows: list[dict[str, object]] = []
    for benchmark_id in case_ids:
        case_audit = audit[audit["benchmark_id"].astype(str).eq(benchmark_id)]
        case_queue = queue[queue["benchmark_id"].astype(str).eq(benchmark_id)] if not queue.empty else queue
        action_rows = int(len(case_queue))
        blocker_rows = int(case_queue["action_status"].map(_safe_text).eq("blocker").sum()) if action_rows else 0
        review_rows = int(case_queue["action_status"].map(_safe_text).eq("review").sum()) if action_rows else 0
        top_priority = ""
        top_issue_type = ""
        if action_rows:
            ranked_queue = case_queue.copy()
            ranked_queue["_priority_rank"] = ranked_queue["priority"].map(_safe_text).map(priority_rank).fillna(99)
            ranked_queue = ranked_queue.sort_values(["_priority_rank", "issue_type", "chain", "resid"]).reset_index(drop=True)
            top_priority = _safe_text(ranked_queue.iloc[0].get("priority"))
            top_issue_type = _safe_text(ranked_queue.iloc[0].get("issue_type"))

        if blocker_rows:
            recommended_action = "Resolve source-audit blockers for this case before using its benchmark coverage in precision claims."
            case_warning = "This case has benchmark references that cannot support independent precision claims yet."
        elif review_rows:
            recommended_action = "Confirm source independence for this case before treating benchmark coverage as publication-ready."
            case_warning = "This case is useful for triage, but final accuracy claims still need source-independence review."
        else:
            recommended_action = "Keep source audit with this case's benchmark report."
            case_warning = "Source provenance is ready; remaining risks, if any, are outside the source audit."

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "reference_rows": int(len(case_audit)),
                "action_rows": action_rows,
                "blocker_rows": blocker_rows,
                "review_rows": review_rows,
                "top_priority": top_priority,
                "top_issue_type": top_issue_type,
                "source_claim_statuses": _ordered_source_values(case_audit["source_claim_status"].tolist(), claim_rank),
                "source_modes": _ordered_source_values(case_audit["source_mode"].tolist()),
                "recommended_action": recommended_action,
                "case_warning": case_warning,
            }
        )

    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS)
    frame["_risk_rank"] = frame.apply(
        lambda row: 0 if int(row["blocker_rows"]) > 0 else 1 if int(row["review_rows"]) > 0 else 2,
        axis=1,
    )
    frame["_priority_rank"] = frame["top_priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["_risk_rank", "_priority_rank", "benchmark_id"]).drop(
        columns=["_risk_rank", "_priority_rank"]
    ).reset_index(drop=True)
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_checklist_markdown(
    case_summary_df: Optional[pd.DataFrame],
    action_queue_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render source-audit case risks as a case-first manual checklist."""

    case_summary = (
        case_summary_df.copy()
        if case_summary_df is not None and not getattr(case_summary_df, "empty", True)
        else _empty_reference_source_audit_case_summary_df()
    )
    action_queue = (
        action_queue_df.copy()
        if action_queue_df is not None and not getattr(action_queue_df, "empty", True)
        else _empty_reference_source_audit_action_queue_df()
    )
    if case_summary.empty and action_queue.empty:
        return ""
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS:
        if column not in case_summary.columns:
            case_summary[column] = ""
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_ACTION_QUEUE_COLUMNS:
        if column not in action_queue.columns:
            action_queue[column] = ""

    lines = [
        "# Benchmark reference source audit case checklist",
        "",
        "Use this case-first checklist to resolve source provenance before interpreting catalytic pocket benchmark coverage as precision.",
        "",
    ]
    if not case_summary.empty:
        lines.extend(["## Case Actions", ""])
        for row in case_summary.itertuples(index=False):
            priority = _safe_text(row.top_priority) or "ready"
            issue_type = _safe_text(row.top_issue_type) or "source-ready"
            lines.append(
                f"- [ ] case `{row.benchmark_id}` `{priority}` `{issue_type}`: "
                f"{int(row.reference_rows)} references, {int(row.action_rows)} source actions "
                f"(blockers {int(row.blocker_rows)}, review {int(row.review_rows)}). {row.recommended_action}"
            )
        lines.append("")

    if not action_queue.empty:
        lines.extend(["## Source Row Evidence", ""])
        for row in action_queue.itertuples(index=False):
            case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
            residue_text = _safe_text(row.residue_label) or "-"
            lines.append(
                f"- [ ] {row.action_id} `{row.priority}` `{row.issue_type}` for {case_text}, "
                f"residue `{residue_text}`: {row.suggested_action}"
            )
        lines.append("")
    elif not case_summary.empty:
        lines.extend(
            [
                "## Source Row Evidence",
                "",
                "- [ ] No source-only blockers or review rows remain; keep the case summary with the benchmark export.",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_reference_source_audit_case_decision_template(
    case_summary_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build an editable case-level decision template for source-audit remediation."""

    if case_summary_df is None or getattr(case_summary_df, "empty", True):
        return _empty_reference_source_audit_case_decision_template_df()
    summary = case_summary_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = ""
    for column in ("reference_rows", "action_rows", "blocker_rows", "review_rows"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)

    actionable = summary[
        summary["action_rows"].gt(0)
        | summary["blocker_rows"].gt(0)
        | summary["review_rows"].gt(0)
        | summary["top_priority"].map(_safe_text).ne("")
    ].copy()
    if actionable.empty:
        return _empty_reference_source_audit_case_decision_template_df()

    template = actionable[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS].copy()
    template["source_decision"] = "review"
    template["reviewer"] = ""
    template["decision_date"] = ""
    template["verified_source_mode"] = ""
    template["verified_independence"] = ""
    template["replacement_reference_source"] = ""
    template["decision_note"] = ""
    return template[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS].reset_index(drop=True)


def _normalize_source_audit_case_decision(value: object) -> str:
    text = _safe_text(value).lower()
    if text in {"accept", "accepted", "approve", "approved", "resolved", "source-ready", "ready", "yes", "pass"}:
        return "accept"
    if text in {"replace", "replaced", "curate", "curated", "source-replaced", "replacement"}:
        return "replace"
    if text in {"hold", "held", "defer", "deferred", "blocked", "block", "reject", "rejected", "no", "fail"}:
        return "hold"
    if text in {"", "review", "needs-review", "pending"}:
        return "review"
    return "unknown"


def parse_pocket_benchmark_reference_source_audit_case_decision_table(
    decision_text: str | bytes | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse reviewer decisions for source-audit case remediation."""

    if decision_text is None:
        return _empty_reference_source_audit_case_decision_template_df(), {
            "status": "empty",
            "decision_rows": "0",
            "reason": "No source-audit case decision table was provided.",
        }
    if isinstance(decision_text, bytes):
        decision_text = decision_text.decode("utf-8", errors="ignore")
    frame = _read_delimited_table(str(decision_text or ""))
    if frame.empty:
        return _empty_reference_source_audit_case_decision_template_df(), {
            "status": "empty",
            "decision_rows": "0",
            "reason": "No source-audit case decision rows could be parsed.",
        }

    column_aliases: dict[str, Sequence[str]] = {
        "benchmark_id": ("benchmark_id", "case_id", "enzyme_id", "pdb_id", "case"),
        "source_decision": ("source_decision", "decision", "review_decision", "status", "approval"),
        "reviewer": ("reviewer", "reviewer_name", "curator", "user"),
        "decision_date": ("decision_date", "review_date", "date", "reviewed_at"),
        "verified_source_mode": ("verified_source_mode", "source_mode", "verified_mode", "source"),
        "verified_independence": ("verified_independence", "independence", "independent", "independence_status"),
        "replacement_reference_source": (
            "replacement_reference_source",
            "replacement_source",
            "curated_source",
            "verified_reference_source",
        ),
        "decision_note": ("decision_note", "note", "notes", "comment", "comments", "review_note"),
    }
    selected = {column: _find_column_in_order(frame, aliases) for column, aliases in column_aliases.items()}
    if not selected["benchmark_id"] or not selected["source_decision"]:
        return _empty_reference_source_audit_case_decision_template_df(), {
            "status": "invalid",
            "decision_rows": "0",
            "reason": "benchmark_id and source_decision columns are required.",
        }

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        benchmark_id = _safe_text(row_dict.get(selected["benchmark_id"]))
        if not benchmark_id:
            continue
        parsed_row: dict[str, object] = {
            "benchmark_id": benchmark_id,
            "source_decision": _normalize_source_audit_case_decision(row_dict.get(selected["source_decision"])),
            "reviewer": _safe_text(row_dict.get(selected["reviewer"])) if selected["reviewer"] else "",
            "decision_date": _safe_text(row_dict.get(selected["decision_date"])) if selected["decision_date"] else "",
            "verified_source_mode": _safe_text(row_dict.get(selected["verified_source_mode"])) if selected["verified_source_mode"] else "",
            "verified_independence": _safe_text(row_dict.get(selected["verified_independence"])) if selected["verified_independence"] else "",
            "replacement_reference_source": _safe_text(row_dict.get(selected["replacement_reference_source"]))
            if selected["replacement_reference_source"]
            else "",
            "decision_note": _safe_text(row_dict.get(selected["decision_note"])) if selected["decision_note"] else "",
        }
        for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS:
            if column not in parsed_row:
                parsed_row[column] = _safe_text(row_dict.get(_find_column_in_order(frame, (column,)))) if column in frame.columns else ""
        rows.append(parsed_row)

    if not rows:
        return _empty_reference_source_audit_case_decision_template_df(), {
            "status": "empty",
            "decision_rows": "0",
            "reason": "No valid benchmark_id rows were found.",
        }

    decisions = pd.DataFrame(rows)
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    decision_counts = decisions["source_decision"].astype(str).value_counts().to_dict()
    return decisions[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS], {
        "status": "ok",
        "decision_rows": str(len(decisions)),
        "accept_rows": str(int(decision_counts.get("accept", 0))),
        "replace_rows": str(int(decision_counts.get("replace", 0))),
        "hold_rows": str(int(decision_counts.get("hold", 0))),
        "review_rows": str(int(decision_counts.get("review", 0))),
        "unknown_rows": str(int(decision_counts.get("unknown", 0))),
    }


def build_pocket_benchmark_reference_source_audit_case_decision_validation(
    decision_df: Optional[pd.DataFrame],
    case_summary_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Validate source-audit case decisions before they are used to clear benchmark source risks."""

    if decision_df is None or getattr(decision_df, "empty", True):
        return _empty_reference_source_audit_case_decision_validation_df()
    decisions = decision_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    summary = (
        case_summary_df.copy()
        if case_summary_df is not None and not getattr(case_summary_df, "empty", True)
        else _empty_reference_source_audit_case_summary_df()
    )
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = ""
    valid_case_ids = set(summary["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
    duplicate_decisions: dict[str, set[str]] = {}
    for _, row in decisions.iterrows():
        benchmark_id = _safe_text(row.get("benchmark_id"))
        if benchmark_id:
            duplicate_decisions.setdefault(benchmark_id, set()).add(_safe_text(row.get("source_decision")))

    rows: list[dict[str, object]] = []
    for index, decision in decisions.reset_index(drop=True).iterrows():
        benchmark_id = _safe_text(decision.get("benchmark_id"))
        source_decision = _normalize_source_audit_case_decision(decision.get("source_decision"))
        reviewer = _safe_text(decision.get("reviewer"))
        verified_independence = _safe_text(decision.get("verified_independence")).lower()
        verified_source_mode = _safe_text(decision.get("verified_source_mode"))
        replacement_source = _safe_text(decision.get("replacement_reference_source"))
        decision_note = _safe_text(decision.get("decision_note"))
        issues: list[str] = []
        fixes: list[str] = []

        if not benchmark_id:
            issues.append("missing-benchmark-id")
            fixes.append("Fill benchmark_id from the exported source-audit case decision template.")
        elif valid_case_ids and benchmark_id not in valid_case_ids:
            issues.append("unknown-benchmark-id")
            fixes.append("Use benchmark_id values from the current source-audit case summary.")
        if len(duplicate_decisions.get(benchmark_id, set())) > 1:
            issues.append("conflicting-duplicate")
            fixes.append("Keep only one source decision per benchmark_id.")
        if source_decision == "unknown":
            issues.append("unknown-decision")
            fixes.append("Use accept, replace, hold, or review.")
        if source_decision in {"accept", "replace", "hold"} and not reviewer:
            issues.append("missing-reviewer")
            fixes.append("Fill reviewer for every non-review source decision.")
        if source_decision == "accept" and verified_independence not in {"yes", "y", "true", "independent", "verified"}:
            issues.append("missing-independence-evidence")
            fixes.append("Set verified_independence to yes/independent after confirming the benchmark source is independent.")
        if source_decision == "replace" and not (replacement_source or verified_source_mode):
            issues.append("missing-replacement-source")
            fixes.append("Fill replacement_reference_source or verified_source_mode for replacement decisions.")
        if source_decision == "hold" and not decision_note:
            issues.append("missing-hold-note")
            fixes.append("Explain why this case remains blocked or deferred in decision_note.")

        validation_status = "blocked" if issues else ("review" if source_decision == "review" else "ok")
        rows.append(
            {
                "row_index": int(index + 1),
                "benchmark_id": benchmark_id,
                "source_decision": source_decision,
                "validation_status": validation_status,
                "issue_flags": ";".join(dict.fromkeys(issues)),
                "required_fix": " ".join(dict.fromkeys(fixes)),
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_VALIDATION_COLUMNS)


def build_pocket_benchmark_reference_source_audit_case_decision_outcomes(
    case_summary_df: Optional[pd.DataFrame],
    decision_df: Optional[pd.DataFrame] = None,
    validation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply validated source-audit case decisions to case-level source risks."""

    if case_summary_df is None or getattr(case_summary_df, "empty", True):
        return _empty_reference_source_audit_case_decision_outcome_df()
    summary = case_summary_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = ""
    for column in ("reference_rows", "action_rows", "blocker_rows", "review_rows"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)

    decisions = (
        decision_df.copy()
        if decision_df is not None and not getattr(decision_df, "empty", True)
        else _empty_reference_source_audit_case_decision_template_df()
    )
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_TEMPLATE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""

    validation = (
        validation_df.copy()
        if validation_df is not None and not getattr(validation_df, "empty", True)
        else pd.DataFrame()
    )
    if validation.empty and not decisions.empty:
        validation = build_pocket_benchmark_reference_source_audit_case_decision_validation(decisions, summary)
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_VALIDATION_COLUMNS:
        if column not in validation.columns:
            validation[column] = ""

    decision_by_case = {
        _safe_text(row.get("benchmark_id")): row
        for _, row in decisions.iterrows()
        if _safe_text(row.get("benchmark_id"))
    }
    validation_by_case = {
        _safe_text(row.get("benchmark_id")): row
        for _, row in validation.iterrows()
        if _safe_text(row.get("benchmark_id"))
    }

    rows: list[dict[str, object]] = []
    for _, case in summary.iterrows():
        benchmark_id = _safe_text(case.get("benchmark_id"))
        reference_rows = int(case.get("reference_rows") or 0)
        action_rows = int(case.get("action_rows") or 0)
        blocker_rows = int(case.get("blocker_rows") or 0)
        review_rows = int(case.get("review_rows") or 0)
        has_source_risk = bool(action_rows or blocker_rows or review_rows or _safe_text(case.get("top_priority")))

        decision = decision_by_case.get(benchmark_id)
        validation_row = validation_by_case.get(benchmark_id)
        source_decision = _normalize_source_audit_case_decision(decision.get("source_decision")) if decision is not None else ""
        validation_status = _safe_text(validation_row.get("validation_status")) if validation_row is not None else ""
        reviewer = _safe_text(decision.get("reviewer")) if decision is not None else ""
        verified_source_mode = _safe_text(decision.get("verified_source_mode")) if decision is not None else ""
        verified_independence = _safe_text(decision.get("verified_independence")) if decision is not None else ""
        replacement_source = _safe_text(decision.get("replacement_reference_source")) if decision is not None else ""
        issue_flags = _safe_text(validation_row.get("issue_flags")) if validation_row is not None else ""
        required_fix = _safe_text(validation_row.get("required_fix")) if validation_row is not None else ""

        if not has_source_risk:
            applied_status = "source-ready"
            outcome_reason = "No source-audit blocker or review action is recorded for this case."
            next_action = "Keep source audit with the benchmark report."
        elif decision is None:
            source_decision = "review"
            validation_status = "review"
            applied_status = "pending"
            outcome_reason = "No source-audit case decision has been uploaded for this case."
            next_action = "Fill and upload the source-audit case decision template."
        elif validation_status == "blocked":
            applied_status = "blocked"
            outcome_reason = issue_flags or "Uploaded source-audit case decision failed validation."
            next_action = required_fix or "Fix validation issues and re-upload source-audit case decisions."
        elif source_decision == "accept":
            applied_status = "cleared"
            outcome_reason = "Reviewer accepted the source decision and recorded independence evidence."
            next_action = "Keep the validated case decision with benchmark exports before making precision claims."
        elif source_decision == "replace":
            applied_status = "replaced"
            outcome_reason = "Reviewer replaced or curated the benchmark reference source for this case."
            next_action = "Rebuild benchmark references from the replacement source before reporting final coverage."
        elif source_decision == "hold":
            applied_status = "held"
            outcome_reason = _safe_text(decision.get("decision_note")) or "Reviewer held this case for later source remediation."
            next_action = "Do not use this case for independent precision claims until the hold is resolved."
        else:
            applied_status = "pending"
            outcome_reason = "Source decision is still pending review."
            next_action = "Complete source audit case review before clearing this case."

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "source_decision": source_decision,
                "validation_status": validation_status,
                "applied_status": applied_status,
                "reference_rows": reference_rows,
                "action_rows": action_rows,
                "blocker_rows": blocker_rows,
                "review_rows": review_rows,
                "top_priority": _safe_text(case.get("top_priority")),
                "top_issue_type": _safe_text(case.get("top_issue_type")),
                "source_claim_statuses": _safe_text(case.get("source_claim_statuses")),
                "source_modes": _safe_text(case.get("source_modes")),
                "reviewer": reviewer,
                "verified_source_mode": verified_source_mode,
                "verified_independence": verified_independence,
                "replacement_reference_source": replacement_source,
                "outcome_reason": outcome_reason,
                "next_action": next_action,
            }
        )

    if not rows:
        return _empty_reference_source_audit_case_decision_outcome_df()
    status_rank = {"blocked": 0, "pending": 1, "held": 2, "replaced": 3, "cleared": 4, "source-ready": 5}
    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS)
    frame["_status_rank"] = frame["applied_status"].map(status_rank).fillna(99)
    frame = frame.sort_values(["_status_rank", "benchmark_id"]).drop(columns=["_status_rank"]).reset_index(drop=True)
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_decision_outcome_summary(
    outcome_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize whether source-audit case decision outcomes close source risk."""

    if outcome_df is None or getattr(outcome_df, "empty", True):
        return _empty_reference_source_audit_case_decision_outcome_summary_df()

    outcomes = outcome_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = ""
    status = outcomes["applied_status"].astype(str).str.strip().str.lower()

    blocked_cases = int(status.eq("blocked").sum())
    pending_cases = int(status.eq("pending").sum())
    held_cases = int(status.eq("held").sum())
    replaced_cases = int(status.eq("replaced").sum())
    cleared_cases = int(status.eq("cleared").sum())
    source_ready_cases = int(status.eq("source-ready").sum())
    recognized_statuses = {"blocked", "pending", "held", "replaced", "cleared", "source-ready"}
    unknown_status_cases = int((~status.isin(recognized_statuses)).sum())
    case_count = int(len(outcomes))
    actionable_case_count = int(max(case_count - source_ready_cases, 0))
    closed_actionable_case_count = int(cleared_cases + replaced_cases)
    open_actionable_case_count = int(blocked_cases + pending_cases + held_cases + unknown_status_cases)

    if blocked_cases:
        closure_status = "blocked"
        recommended_action = "Fix blocked source-audit case decisions before using benchmark coverage in precision claims."
        summary_warning = "At least one source-audit case decision is invalid or blocked."
    elif pending_cases:
        closure_status = "pending"
        recommended_action = "Upload or complete source-audit case decisions for pending cases."
        summary_warning = "Some source-risk cases still have no clearing decision."
    elif held_cases:
        closure_status = "held"
        recommended_action = "Resolve held source-risk cases or exclude them from independent benchmark claims."
        summary_warning = "Held cases remain outside claim-ready source provenance."
    elif actionable_case_count > 0 and open_actionable_case_count == 0:
        closure_status = "closed"
        recommended_action = "Keep validated source-audit decisions with benchmark exports before reporting precision."
        summary_warning = "All actionable source-risk cases have been cleared or replaced."
    elif actionable_case_count == 0:
        closure_status = "source-ready"
        recommended_action = "Keep source audit with benchmark exports."
        summary_warning = "No actionable source-risk cases were present."
    else:
        closure_status = "review"
        recommended_action = "Review source-audit case decision outcomes before reporting benchmark precision."
        summary_warning = "Source-audit case closure status could not be determined from applied statuses."

    return pd.DataFrame(
        [
            {
                "summary_id": "BRSDOS-001",
                "closure_status": closure_status,
                "case_count": case_count,
                "blocked_cases": blocked_cases,
                "pending_cases": pending_cases,
                "held_cases": held_cases,
                "replaced_cases": replaced_cases,
                "cleared_cases": cleared_cases,
                "source_ready_cases": source_ready_cases,
                "actionable_case_count": actionable_case_count,
                "closed_actionable_case_count": closed_actionable_case_count,
                "open_actionable_case_count": open_actionable_case_count,
                "recommended_action": recommended_action,
                "summary_warning": summary_warning,
            }
        ],
        columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_SUMMARY_COLUMNS,
    )


def build_pocket_benchmark_reference_source_audit_case_decision_closure_queue(
    outcome_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a machine-readable queue for open source-audit decision outcomes."""

    if outcome_df is None or getattr(outcome_df, "empty", True):
        return _empty_reference_source_audit_case_decision_closure_queue_df()

    outcomes = outcome_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = ""

    closed_statuses = {"cleared", "replaced", "source-ready"}
    status = outcomes["applied_status"].astype(str).str.strip().str.lower()
    open_rows = outcomes[~status.isin(closed_statuses)].copy()
    if open_rows.empty:
        return _empty_reference_source_audit_case_decision_closure_queue_df()

    rows: list[dict[str, object]] = []
    for _, outcome in open_rows.iterrows():
        applied_status = _safe_text(outcome.get("applied_status")).lower() or "unknown"
        if applied_status == "blocked":
            closure_priority = "P0"
            closure_action_status = "blocker"
            closure_issue_type = "invalid_source_audit_case_decision"
            fallback_action = "Fix validation issues and re-upload source-audit case decisions."
        elif applied_status == "pending":
            closure_priority = "P1"
            closure_action_status = "blocker"
            closure_issue_type = "missing_source_audit_case_decision"
            fallback_action = "Fill and upload the source-audit case decision template."
        elif applied_status == "held":
            closure_priority = "P2"
            closure_action_status = "review"
            closure_issue_type = "held_source_audit_case"
            fallback_action = "Resolve the hold decision or exclude this case from independent precision claims."
        else:
            closure_priority = "P1"
            closure_action_status = "blocker"
            closure_issue_type = "unknown_source_audit_case_status"
            fallback_action = "Review this source-audit case decision outcome before reporting precision."

        rows.append(
            {
                "closure_action_id": "",
                "benchmark_id": _safe_text(outcome.get("benchmark_id")),
                "applied_status": applied_status,
                "closure_priority": closure_priority,
                "closure_action_status": closure_action_status,
                "closure_issue_type": closure_issue_type,
                "source_decision": _safe_text(outcome.get("source_decision")),
                "validation_status": _safe_text(outcome.get("validation_status")),
                "reference_rows": _safe_int(outcome.get("reference_rows")) or 0,
                "action_rows": _safe_int(outcome.get("action_rows")) or 0,
                "blocker_rows": _safe_int(outcome.get("blocker_rows")) or 0,
                "review_rows": _safe_int(outcome.get("review_rows")) or 0,
                "top_priority": _safe_text(outcome.get("top_priority")),
                "top_issue_type": _safe_text(outcome.get("top_issue_type")),
                "source_claim_statuses": _safe_text(outcome.get("source_claim_statuses")),
                "source_modes": _safe_text(outcome.get("source_modes")),
                "reviewer": _safe_text(outcome.get("reviewer")),
                "verified_source_mode": _safe_text(outcome.get("verified_source_mode")),
                "verified_independence": _safe_text(outcome.get("verified_independence")),
                "replacement_reference_source": _safe_text(outcome.get("replacement_reference_source")),
                "outcome_reason": _safe_text(outcome.get("outcome_reason")),
                "required_action": _safe_text(outcome.get("next_action")) or fallback_action,
                "closure_warning": _safe_text(outcome.get("outcome_reason")) or "Source-audit case is not closed yet.",
            }
        )

    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    status_rank = {"blocked": 0, "pending": 1, "held": 2}
    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_CLOSURE_QUEUE_COLUMNS)
    frame["_priority_rank"] = frame["closure_priority"].map(priority_rank).fillna(9)
    frame["_status_rank"] = frame["applied_status"].map(status_rank).fillna(9)
    frame = frame.sort_values(["_priority_rank", "_status_rank", "benchmark_id"]).drop(
        columns=["_priority_rank", "_status_rank"]
    ).reset_index(drop=True)
    frame["closure_action_id"] = [f"BRSDQ-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_CLOSURE_QUEUE_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact(
    source_audit_df: Optional[pd.DataFrame],
    decision_outcome_df: Optional[pd.DataFrame] = None,
    *,
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Explain how source-audit case decisions adjust readiness source issues."""

    if (
        (source_audit_df is None or getattr(source_audit_df, "empty", True))
        and (decision_outcome_df is None or getattr(decision_outcome_df, "empty", True))
    ):
        return _empty_reference_source_audit_case_decision_readiness_impact_df()

    fallback_id = _safe_text(default_benchmark_id) or "current"
    audit = (
        source_audit_df.copy()
        if source_audit_df is not None and not getattr(source_audit_df, "empty", True)
        else _empty_reference_source_audit_df()
    )
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = ""
    if "benchmark_id" not in audit.columns:
        audit["benchmark_id"] = ""
    audit["benchmark_id"] = audit["benchmark_id"].map(lambda value: _safe_text(value) or fallback_id)

    outcomes = (
        decision_outcome_df.copy()
        if decision_outcome_df is not None and not getattr(decision_outcome_df, "empty", True)
        else _empty_reference_source_audit_case_decision_outcome_df()
    )
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = ""
    outcomes["benchmark_id"] = outcomes["benchmark_id"].map(lambda value: _safe_text(value) or fallback_id)
    decision_by_case = {
        _safe_text(row.get("benchmark_id")): row
        for _, row in outcomes.iterrows()
        if _safe_text(row.get("benchmark_id"))
    }

    case_ids = sorted(
        set(audit["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
        | set(outcomes["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
    )
    if not case_ids:
        return _empty_reference_source_audit_case_decision_readiness_impact_df()

    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    claim_rank = {
        "blocked-provisional": 0,
        "source-unknown": 1,
        "review-qualified": 2,
        "source-ready": 3,
    }
    rows: list[dict[str, object]] = []
    for benchmark_id in case_ids:
        case_audit = audit[audit["benchmark_id"].astype(str).eq(benchmark_id)]
        original_issues = [
            issue
            for _, audit_row in case_audit.iterrows()
            if (issue := _source_audit_readiness_issue(audit_row)) is not None
        ]
        original_issue = (
            sorted(original_issues, key=lambda item: (priority_rank.get(item[0], 99), item[1]))[0]
            if original_issues
            else None
        )
        outcome = decision_by_case.get(benchmark_id)
        adjusted_issue = _source_audit_decision_readiness_issue(outcome) if outcome is not None else original_issue

        original_priority = original_issue[0] if original_issue else ""
        original_issue_type = original_issue[1] if original_issue else ""
        adjusted_priority = adjusted_issue[0] if adjusted_issue else ""
        adjusted_issue_type = adjusted_issue[1] if adjusted_issue else ""
        applied_status = _safe_text(outcome.get("applied_status")).lower() if outcome is not None else ""

        if original_issue is None and adjusted_issue is None:
            readiness_impact = "source-ready"
            readiness_note = "No source-audit readiness issue for this case."
        elif outcome is not None and original_issue is not None and adjusted_issue is None:
            readiness_impact = "cleared-by-decision"
            readiness_note = "Source-audit case decision closed this case for readiness; keep decision exports with the benchmark report."
        elif outcome is not None and adjusted_issue is not None and (
            adjusted_priority != original_priority or adjusted_issue_type != original_issue_type
        ):
            readiness_impact = "decision-adjusted-open"
            readiness_note = "Source-audit case decision changed the readiness issue; resolve the adjusted issue before claims."
        elif outcome is not None and adjusted_issue is not None:
            readiness_impact = "decision-open"
            readiness_note = "Source-audit case decision outcome remains open for readiness."
        else:
            readiness_impact = "unchanged-open"
            readiness_note = "No source-audit case decision has closed this original readiness issue."

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "applied_status": applied_status,
                "source_decision": _safe_text(outcome.get("source_decision")) if outcome is not None else "",
                "validation_status": _safe_text(outcome.get("validation_status")) if outcome is not None else "",
                "original_source_priority": original_priority,
                "original_source_issue_type": original_issue_type,
                "adjusted_source_priority": adjusted_priority,
                "adjusted_source_issue_type": adjusted_issue_type,
                "readiness_impact": readiness_impact,
                "reference_rows": int(len(case_audit)) if not case_audit.empty else (_safe_int(outcome.get("reference_rows")) if outcome is not None else 0) or 0,
                "source_claim_statuses": _ordered_source_values(case_audit["source_claim_status"].tolist(), claim_rank) if "source_claim_status" in case_audit.columns else "",
                "source_modes": _ordered_source_values(case_audit["source_mode"].tolist()) if "source_mode" in case_audit.columns else "",
                "outcome_reason": _safe_text(outcome.get("outcome_reason")) if outcome is not None else "",
                "readiness_note": readiness_note,
            }
        )

    impact_rank = {
        "decision-adjusted-open": 0,
        "decision-open": 1,
        "unchanged-open": 2,
        "cleared-by-decision": 3,
        "source-ready": 4,
    }
    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_COLUMNS)
    frame["_impact_rank"] = frame["readiness_impact"].map(impact_rank).fillna(9)
    frame["_priority_rank"] = frame["adjusted_source_priority"].map(priority_rank).fillna(9)
    frame = frame.sort_values(["_impact_rank", "_priority_rank", "benchmark_id"]).drop(
        columns=["_impact_rank", "_priority_rank"]
    ).reset_index(drop=True)
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary(
    readiness_impact_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize source-audit decision readiness impact across cases."""

    if readiness_impact_df is None or getattr(readiness_impact_df, "empty", True):
        return _empty_reference_source_audit_case_decision_readiness_impact_summary_df()

    impact = readiness_impact_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_COLUMNS:
        if column not in impact.columns:
            impact[column] = ""

    status = impact["readiness_impact"].astype(str).str.strip().str.lower()
    original_priority = impact["original_source_priority"].astype(str).str.strip().str.upper()
    adjusted_priority = impact["adjusted_source_priority"].astype(str).str.strip().str.upper()

    case_count = int(len(impact))
    cleared_cases = int(status.eq("cleared-by-decision").sum())
    adjusted_open_cases = int(status.eq("decision-adjusted-open").sum())
    decision_open_cases = int(status.eq("decision-open").sum())
    unchanged_open_cases = int(status.eq("unchanged-open").sum())
    source_ready_cases = int(status.eq("source-ready").sum())
    open_after_decision_cases = int((~status.isin({"cleared-by-decision", "source-ready"})).sum())
    original_p0_cases = int(original_priority.eq("P0").sum())
    original_p1_cases = int(original_priority.eq("P1").sum())
    original_p2_cases = int(original_priority.eq("P2").sum())
    adjusted_p0_cases = int(adjusted_priority.eq("P0").sum())
    adjusted_p1_cases = int(adjusted_priority.eq("P1").sum())
    adjusted_p2_cases = int(adjusted_priority.eq("P2").sum())
    original_blocker_cases = original_p0_cases + original_p1_cases
    adjusted_blocker_cases = adjusted_p0_cases + adjusted_p1_cases
    net_blocker_delta = int(adjusted_blocker_cases - original_blocker_cases)

    if adjusted_blocker_cases > 0:
        readiness_impact_status = "blocked"
        recommended_action = "Resolve adjusted P0/P1 source-readiness issues before using benchmark coverage in precision claims."
        summary_warning = "Some source-audit decision outcomes still block independent benchmark claims."
    elif adjusted_p2_cases > 0:
        readiness_impact_status = "review-needed"
        recommended_action = "Resolve or explicitly sign off adjusted P2 source-readiness review items before publication-ready claims."
        summary_warning = "Source-audit decision outcomes still need reviewer sign-off."
    elif cleared_cases > 0:
        readiness_impact_status = "cleared-by-decision"
        recommended_action = "Keep decision outcomes and readiness impact exports with the benchmark report."
        summary_warning = "Source-audit readiness issues represented here were cleared or are source-ready."
    else:
        readiness_impact_status = "source-ready"
        recommended_action = "Keep source audit and readiness impact exports with the benchmark report."
        summary_warning = "No source-audit readiness issues are open in the impact table."

    return pd.DataFrame(
        [
            {
                "summary_id": "BRSDIS-001",
                "readiness_impact_status": readiness_impact_status,
                "case_count": case_count,
                "cleared_by_decision_cases": cleared_cases,
                "decision_adjusted_open_cases": adjusted_open_cases,
                "decision_open_cases": decision_open_cases,
                "unchanged_open_cases": unchanged_open_cases,
                "source_ready_cases": source_ready_cases,
                "open_after_decision_cases": open_after_decision_cases,
                "original_p0_cases": original_p0_cases,
                "original_p1_cases": original_p1_cases,
                "original_p2_cases": original_p2_cases,
                "adjusted_p0_cases": adjusted_p0_cases,
                "adjusted_p1_cases": adjusted_p1_cases,
                "adjusted_p2_cases": adjusted_p2_cases,
                "net_blocker_delta": net_blocker_delta,
                "recommended_action": recommended_action,
                "summary_warning": summary_warning,
            }
        ],
        columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_SUMMARY_COLUMNS,
    )


def _source_dataset_impact_case_action(
    *,
    readiness_impact: str,
    adjusted_priority: str,
    claim_status: str,
    source_tracked: bool,
) -> tuple[str, str, bool, str, str]:
    source_open = readiness_impact in {"decision-adjusted-open", "decision-open", "unchanged-open"}
    source_blocker = adjusted_priority in {"P0", "P1"}
    source_review = adjusted_priority == "P2"
    source_cleared = readiness_impact == "cleared-by-decision"
    source_ready = readiness_impact == "source-ready"
    claim_ready = claim_status == "claim-ready"
    source_gate_mismatch = bool(
        source_tracked
        and (
            (source_blocker and claim_status != "blocked")
            or (source_review and claim_ready)
            or (source_open and claim_ready)
        )
    )

    if source_gate_mismatch:
        return (
            "source-gate-mismatch",
            "blocker",
            True,
            "Regenerate readiness interpretation for this case because source decision impact and claim status diverge.",
            "Source-audit decision impact is not reflected in the case claim status.",
        )
    if source_blocker:
        return (
            "source-blocked",
            "blocker",
            False,
            "Resolve adjusted P0/P1 source-readiness issues before using this case in dataset-level precision claims.",
            "Source-audit decision outcomes still block this benchmark case.",
        )
    if source_review:
        return (
            "source-review-needed",
            "review",
            False,
            "Complete source independence review for adjusted P2 cases before publication-ready dataset claims.",
            "Source-audit decision outcomes still require reviewer sign-off for this case.",
        )
    if source_open:
        return (
            "source-open",
            "review",
            False,
            "Review open source-audit decision cases before interpreting this case in dataset-level coverage.",
            "Source-audit decision outcome remains open for this case.",
        )
    if source_cleared:
        return (
            "source-cleared",
            "closed",
            False,
            "Keep the source-audit case decision and readiness impact exports with the dataset benchmark report.",
            "Source-audit issue was cleared by decision for this case.",
        )
    if source_ready:
        return (
            "source-ready",
            "ready",
            False,
            "Keep source-audit readiness impact exports with the dataset benchmark report.",
            "Tracked source-audit case is source-ready.",
        )
    return (
        "no-source-impact",
        "none",
        False,
        "No source-audit decision impact is tracked for this case and Top-N level.",
        "No source-audit decision impact is tracked for this case.",
    )


def build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases(
    case_interpretation_df: Optional[pd.DataFrame],
    readiness_impact_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build per-case Top-N source-audit decision impact details."""

    if (
        case_interpretation_df is None
        or getattr(case_interpretation_df, "empty", True)
        or "top_n" not in case_interpretation_df.columns
    ):
        return _empty_reference_source_audit_case_decision_dataset_impact_case_df()

    cases = case_interpretation_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_COLUMNS:
        if column not in cases.columns:
            cases[column] = ""
    cases["top_n"] = pd.to_numeric(cases["top_n"], errors="coerce")
    cases = cases[cases["top_n"].notna()].copy()
    if cases.empty:
        return _empty_reference_source_audit_case_decision_dataset_impact_case_df()

    cases["benchmark_id"] = [
        _safe_text(value) or f"case-{index}"
        for index, value in zip(cases.index, cases["benchmark_id"])
    ]
    cases["claim_status"] = cases["claim_status"].map(_normalized_claim_status)
    cases.loc[cases["claim_status"].eq(""), "claim_status"] = "readiness-unknown"
    cases["claim_ready"] = cases["claim_ready"].map(_claim_ready_bool)
    cases.loc[cases["claim_ready"] & cases["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"
    cases["coverage_ratio"] = pd.to_numeric(cases["coverage_ratio"], errors="coerce").fillna(0.0)
    cases["best_rank"] = pd.to_numeric(cases["best_rank"], errors="coerce").fillna(0).astype(int)
    cases = cases.drop_duplicates(subset=["top_n", "benchmark_id"], keep="first").reset_index(drop=True)

    impact = (
        readiness_impact_df.copy()
        if readiness_impact_df is not None and not getattr(readiness_impact_df, "empty", True)
        else _empty_reference_source_audit_case_decision_readiness_impact_df()
    )
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_READINESS_IMPACT_COLUMNS:
        if column not in impact.columns:
            impact[column] = ""
    impact["benchmark_id"] = impact["benchmark_id"].map(_safe_text)
    impact = impact[impact["benchmark_id"].ne("")].drop_duplicates(subset=["benchmark_id"], keep="first")
    impact["source_tracked"] = True

    joined = cases.merge(impact, on="benchmark_id", how="left")
    joined["source_tracked"] = joined["source_tracked"].fillna(False).map(bool)
    for column in (
        "applied_status",
        "source_decision",
        "validation_status",
        "original_source_priority",
        "original_source_issue_type",
        "adjusted_source_priority",
        "adjusted_source_issue_type",
        "readiness_impact",
        "outcome_reason",
        "readiness_note",
    ):
        if column not in joined.columns:
            joined[column] = ""
        joined[column] = joined[column].fillna("").astype(str).str.strip()
    joined["readiness_impact"] = joined["readiness_impact"].str.lower()
    joined["adjusted_source_priority"] = joined["adjusted_source_priority"].str.upper()

    rows: list[dict[str, object]] = []
    for _, row in joined.iterrows():
        source_impact_status, action_status, mismatch, action, warning = _source_dataset_impact_case_action(
            readiness_impact=_safe_text(row.get("readiness_impact")),
            adjusted_priority=_safe_text(row.get("adjusted_source_priority")),
            claim_status=_safe_text(row.get("claim_status")),
            source_tracked=bool(row.get("source_tracked")),
        )
        rows.append(
            {
                "impact_case_id": "",
                "top_n": int(row.get("top_n") or 0),
                "benchmark_id": _safe_text(row.get("benchmark_id")) or "current",
                "claim_status": _safe_text(row.get("claim_status")) or "readiness-unknown",
                "claim_ready": bool(row.get("claim_ready")),
                "coverage_ratio": round(float(row.get("coverage_ratio") or 0.0), 3),
                "best_rank": int(row.get("best_rank") or 0),
                "best_pocket_id": _safe_text(row.get("best_pocket_id")),
                "benchmark_status": _safe_text(row.get("benchmark_status")),
                "readiness_status": _safe_text(row.get("readiness_status")),
                "source_tracked": bool(row.get("source_tracked")),
                "applied_status": _safe_text(row.get("applied_status")),
                "source_decision": _safe_text(row.get("source_decision")),
                "validation_status": _safe_text(row.get("validation_status")),
                "original_source_priority": _safe_text(row.get("original_source_priority")),
                "original_source_issue_type": _safe_text(row.get("original_source_issue_type")),
                "adjusted_source_priority": _safe_text(row.get("adjusted_source_priority")),
                "adjusted_source_issue_type": _safe_text(row.get("adjusted_source_issue_type")),
                "readiness_impact": _safe_text(row.get("readiness_impact")),
                "source_impact_status": source_impact_status,
                "source_action_status": action_status,
                "source_gate_mismatch": bool(mismatch),
                "outcome_reason": _safe_text(row.get("outcome_reason")),
                "readiness_note": _safe_text(row.get("readiness_note")),
                "recommended_action": action,
                "impact_warning": warning,
            }
        )

    if not rows:
        return _empty_reference_source_audit_case_decision_dataset_impact_case_df()

    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS)
    status_rank = {
        "source-gate-mismatch": 0,
        "source-blocked": 1,
        "source-review-needed": 2,
        "source-open": 3,
        "source-cleared": 4,
        "source-ready": 5,
        "no-source-impact": 6,
    }
    frame["_status_rank"] = frame["source_impact_status"].map(status_rank).fillna(9)
    frame = frame.sort_values(["top_n", "_status_rank", "benchmark_id"]).drop(columns=["_status_rank"]).reset_index(drop=True)
    frame["impact_case_id"] = [f"BRSDIC-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact(
    case_interpretation_df: Optional[pd.DataFrame],
    readiness_impact_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize source-audit decision readiness impact at dataset Top-N levels."""

    impact_cases = build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases(
        case_interpretation_df,
        readiness_impact_df,
    )
    if impact_cases.empty:
        return _empty_reference_source_audit_case_decision_dataset_impact_df()

    rows: list[dict[str, object]] = []
    for top_n, joined in impact_cases.groupby("top_n", sort=True, dropna=False):
        top_n_int = int(top_n)
        source_open_mask = joined["readiness_impact"].isin(
            {"decision-adjusted-open", "decision-open", "unchanged-open"}
        )
        source_blocker_mask = joined["adjusted_source_priority"].isin({"P0", "P1"})
        source_review_mask = joined["adjusted_source_priority"].eq("P2")
        source_cleared_mask = joined["readiness_impact"].eq("cleared-by-decision")
        source_ready_mask = joined["readiness_impact"].eq("source-ready")
        claim_ready_mask = joined["claim_status"].eq("claim-ready")
        dataset_blocked_by_source_mask = joined["claim_status"].eq("blocked") & source_blocker_mask
        dataset_review_by_source_mask = joined["claim_status"].eq("review-needed") & (
            source_review_mask | source_open_mask
        )

        source_gate_mismatch_case_count = int(joined["source_gate_mismatch"].map(bool).sum())
        source_blocker_case_count = int(source_blocker_mask.sum())
        source_review_case_count = int(source_review_mask.sum())
        source_open_case_count = int(source_open_mask.sum())
        source_cleared_case_count = int(source_cleared_mask.sum())
        source_ready_case_count = int(source_ready_mask.sum())
        dataset_blocked_by_source_cases = int(dataset_blocked_by_source_mask.sum())
        dataset_review_by_source_cases = int(dataset_review_by_source_mask.sum())
        open_coverage = joined.loc[source_open_mask, "coverage_ratio"]
        cleared_coverage = joined.loc[source_cleared_mask, "coverage_ratio"]

        if source_gate_mismatch_case_count > 0:
            impact_status = "source-gate-mismatch"
            recommended_action = "Regenerate readiness interpretation because source-open cases are not reflected in case claim status."
            warning = "Source-audit decision impact and case interpretation are inconsistent."
        elif dataset_blocked_by_source_cases > 0 or source_blocker_case_count > 0:
            impact_status = "source-blocked"
            recommended_action = "Resolve adjusted P0/P1 source decisions before using this Top-N dataset coverage as a precision claim."
            warning = "Source-audit decision outcomes still block dataset-level interpretation."
        elif dataset_review_by_source_cases > 0 or source_review_case_count > 0:
            impact_status = "source-review-needed"
            recommended_action = "Complete independence/source review for adjusted P2 cases before publication-ready dataset claims."
            warning = "Source-audit decision outcomes still require reviewer sign-off."
        elif source_open_case_count > 0:
            impact_status = "source-open"
            recommended_action = "Review open source-audit decision cases before interpreting dataset-level coverage."
            warning = "Open source-audit cases remain, but they are not classified as blocker/review by adjusted priority."
        elif source_cleared_case_count > 0:
            impact_status = "source-cleared"
            recommended_action = "Keep source-audit decision and readiness impact exports with the dataset benchmark report."
            warning = "Source-audit issues for this Top-N level were cleared or are source-ready."
        elif source_ready_case_count > 0:
            impact_status = "source-ready"
            recommended_action = "Keep source-audit readiness impact exports with the dataset benchmark report."
            warning = "Tracked source-audit cases are source-ready for this Top-N level."
        else:
            impact_status = "no-source-impact"
            recommended_action = "No source-audit decision impact is tracked for this Top-N dataset interpretation."
            warning = "Dataset source-impact table has no tracked source-audit cases for this Top-N level."

        rows.append(
            {
                "top_n": top_n_int,
                "case_count": int(len(joined)),
                "claim_ready_case_count": int(claim_ready_mask.sum()),
                "source_tracked_case_count": int(joined["source_tracked"].map(bool).sum()),
                "source_open_case_count": source_open_case_count,
                "source_blocker_case_count": source_blocker_case_count,
                "source_review_case_count": source_review_case_count,
                "source_cleared_case_count": source_cleared_case_count,
                "source_ready_case_count": source_ready_case_count,
                "dataset_blocked_by_source_cases": dataset_blocked_by_source_cases,
                "dataset_review_by_source_cases": dataset_review_by_source_cases,
                "source_gate_mismatch_case_count": source_gate_mismatch_case_count,
                "mean_source_open_coverage": round(float(open_coverage.mean()), 3) if not open_coverage.empty else 0.0,
                "mean_source_cleared_coverage": round(float(cleared_coverage.mean()), 3) if not cleared_coverage.empty else 0.0,
                "dataset_source_impact_status": impact_status,
                "recommended_action": recommended_action,
                "impact_warning": warning,
            }
        )

    if not rows:
        return _empty_reference_source_audit_case_decision_dataset_impact_df()
    return pd.DataFrame(
        rows,
        columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_COLUMNS,
    )


def build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown(
    impact_case_df: Optional[pd.DataFrame],
    *,
    title: str = "Benchmark source-audit decision dataset impact case checklist",
    max_actions: int = 80,
) -> str:
    """Render source-audit dataset-impact case actions as a reviewer checklist."""

    if impact_case_df is None or getattr(impact_case_df, "empty", True):
        return ""

    working = impact_case_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce").fillna(0).astype(int)
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)
    working["source_gate_mismatch"] = working["source_gate_mismatch"].map(_claim_ready_bool)
    actionable = working[working["source_action_status"].astype(str).isin(["blocker", "review"])].copy()
    if actionable.empty:
        return ""

    action_rank = {"blocker": 0, "review": 1, "closed": 2, "ready": 3, "none": 4}
    actionable["_action_rank"] = actionable["source_action_status"].map(action_rank).fillna(9)
    actionable["_mismatch_rank"] = actionable["source_gate_mismatch"].map(lambda value: 0 if bool(value) else 1)
    actionable = actionable.sort_values(
        ["top_n", "_action_rank", "_mismatch_rank", "benchmark_id", "source_impact_status"]
    ).drop(columns=["_action_rank", "_mismatch_rank"])

    summary = (
        actionable.groupby(["source_action_status", "source_impact_status"], sort=False, dropna=False)
        .agg(
            action_rows=("impact_case_id", "count"),
            affected_cases=("benchmark_id", "nunique"),
            mismatch_rows=("source_gate_mismatch", "sum"),
        )
        .reset_index()
    )
    summary["_action_rank"] = summary["source_action_status"].map(action_rank).fillna(9)
    summary = summary.sort_values(["_action_rank", "source_impact_status"]).drop(columns=["_action_rank"])

    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases.csv`.",
        "",
        "Use this checklist to close source-decision blockers, reviewer items and gate mismatches before treating dataset-level coverage as a precision claim.",
        "",
        "## Summary",
        "",
        "| Action | Source impact | Rows | Cases | Mismatches |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {action} | {impact} | {rows} | {cases} | {mismatches} |".format(
                action=_safe_text(row.get("source_action_status")) or "-",
                impact=_safe_text(row.get("source_impact_status")) or "-",
                rows=int(row.get("action_rows") or 0),
                cases=int(row.get("affected_cases") or 0),
                mismatches=int(row.get("mismatch_rows") or 0),
            )
        )

    lines.extend(["", "## Actions", ""])
    for _, row in actionable.head(max_actions).iterrows():
        mismatch = " mismatch" if bool(row.get("source_gate_mismatch")) else ""
        adjusted = _safe_text(row.get("adjusted_source_priority")) or "-"
        issue = _safe_text(row.get("adjusted_source_issue_type")) or "-"
        lines.append(
            "- [ ] `{action}` Top-{top_n} case `{case}` `{impact}`{mismatch} coverage `{coverage:.3f}` "
            "claim `{claim}` adjusted `{adjusted}` `{issue}`: {recommended}".format(
                action=_safe_text(row.get("source_action_status")) or "-",
                top_n=int(row.get("top_n") or 0),
                case=_safe_text(row.get("benchmark_id")) or "current",
                impact=_safe_text(row.get("source_impact_status")) or "-",
                mismatch=mismatch,
                coverage=float(row.get("coverage_ratio") or 0.0),
                claim=_safe_text(row.get("claim_status")) or "-",
                adjusted=adjusted,
                issue=issue,
                recommended=_safe_text(row.get("recommended_action"))
                or "Review this source-audit dataset impact case before reporting benchmark claims.",
            )
        )

    remaining = int(len(actionable) - max_actions)
    if remaining > 0:
        lines.append(f"- [ ] Review {remaining} additional source-impact case actions in the CSV export.")
    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue(
    impact_case_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a machine-readable action queue from source-audit dataset impact cases."""

    if impact_case_df is None or getattr(impact_case_df, "empty", True):
        return _empty_reference_source_audit_case_decision_dataset_impact_action_queue_df()

    working = impact_case_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce").fillna(0).astype(int)
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)
    working["source_gate_mismatch"] = working["source_gate_mismatch"].map(_claim_ready_bool)
    working["source_action_status"] = working["source_action_status"].astype(str).str.strip().str.lower()
    actionable = working[
        working["source_action_status"].isin(["blocker", "review"]) | working["source_gate_mismatch"].map(bool)
    ].copy()
    if actionable.empty:
        return _empty_reference_source_audit_case_decision_dataset_impact_action_queue_df()

    rows: list[dict[str, object]] = []
    for _, row in actionable.iterrows():
        action_status = _safe_text(row.get("source_action_status")) or "review"
        source_impact_status = _safe_text(row.get("source_impact_status"))
        if bool(row.get("source_gate_mismatch")) or action_status == "blocker" or source_impact_status == "source-blocked":
            priority = "P0"
        elif action_status == "review" or source_impact_status in {"source-review-needed", "source-open"}:
            priority = "P2"
        else:
            priority = "P3"
        rows.append(
            {
                "action_id": "",
                "priority": priority,
                "action_status": action_status,
                "impact_case_id": _safe_text(row.get("impact_case_id")),
                "top_n": int(row.get("top_n") or 0),
                "benchmark_id": _safe_text(row.get("benchmark_id")) or "current",
                "source_impact_status": source_impact_status,
                "source_gate_mismatch": bool(row.get("source_gate_mismatch")),
                "claim_status": _safe_text(row.get("claim_status")) or "readiness-unknown",
                "coverage_ratio": round(float(row.get("coverage_ratio") or 0.0), 3),
                "best_rank": int(row.get("best_rank") or 0),
                "best_pocket_id": _safe_text(row.get("best_pocket_id")),
                "readiness_status": _safe_text(row.get("readiness_status")),
                "applied_status": _safe_text(row.get("applied_status")),
                "source_decision": _safe_text(row.get("source_decision")),
                "validation_status": _safe_text(row.get("validation_status")),
                "adjusted_source_priority": _safe_text(row.get("adjusted_source_priority")),
                "adjusted_source_issue_type": _safe_text(row.get("adjusted_source_issue_type")),
                "readiness_impact": _safe_text(row.get("readiness_impact")),
                "recommended_action": _safe_text(row.get("recommended_action"))
                or "Review this source-audit dataset impact case before reporting benchmark claims.",
                "impact_warning": _safe_text(row.get("impact_warning")),
            }
        )

    frame = pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_COLUMNS)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    status_rank = {"blocker": 0, "review": 1}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(9)
    frame["_status_rank"] = frame["action_status"].map(status_rank).fillna(9)
    frame = frame.sort_values(["_priority_rank", "_status_rank", "top_n", "benchmark_id", "source_impact_status"]).drop(
        columns=["_priority_rank", "_status_rank"]
    ).reset_index(drop=True)
    frame["action_id"] = [f"BRSDIA-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary(
    action_queue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize source-audit dataset impact actions by priority and impact status."""

    if action_queue_df is None or getattr(action_queue_df, "empty", True):
        return _empty_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df()

    working = action_queue_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    for column in ("priority", "action_status", "source_impact_status", "benchmark_id", "action_id"):
        working[column] = working[column].map(_safe_text)
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce").fillna(0).astype(int)
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["source_gate_mismatch"] = working["source_gate_mismatch"].map(_claim_ready_bool)
    working = working[working["priority"].ne("") | working["action_status"].ne("") | working["source_impact_status"].ne("")]
    if working.empty:
        return _empty_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df()

    rows: list[dict[str, object]] = []
    for (priority, action_status, source_impact_status), group in working.groupby(
        ["priority", "action_status", "source_impact_status"],
        sort=False,
        dropna=False,
    ):
        sorted_group = group.sort_values(["top_n", "benchmark_id", "action_id"]).reset_index(drop=True)
        top_row = sorted_group.iloc[0]
        mismatch_count = int(sorted_group["source_gate_mismatch"].map(bool).sum())
        top_n_values = sorted({int(value) for value in sorted_group["top_n"].tolist() if int(value) > 0})
        if mismatch_count > 0:
            recommended_action = (
                "Regenerate readiness interpretation for source-gate mismatches before dataset-level claims."
            )
            warning = "Source gate mismatch remains in this action group."
        elif str(priority).upper() in {"P0", "P1"} or action_status == "blocker":
            recommended_action = "Resolve source blockers before using affected Top-N coverage as a precision claim."
            warning = "Source blockers remain in this action group."
        elif str(priority).upper() == "P2" or action_status == "review":
            recommended_action = "Complete source review before treating affected cases as publication-ready."
            warning = "Source review remains open in this action group."
        else:
            recommended_action = "Review source-impact actions before reporting benchmark claims."
            warning = "Source-impact action group needs reviewer confirmation."
        rows.append(
            {
                "summary_id": "",
                "priority": str(priority).upper(),
                "action_status": action_status,
                "source_impact_status": source_impact_status,
                "action_count": int(len(sorted_group)),
                "affected_case_count": int(sorted_group["benchmark_id"].replace("", pd.NA).dropna().nunique()),
                "top_n_values": ",".join(str(value) for value in top_n_values),
                "mismatch_count": mismatch_count,
                "mean_coverage_ratio": round(float(sorted_group["coverage_ratio"].mean()), 3),
                "top_action_id": _safe_text(top_row.get("action_id")),
                "top_benchmark_id": _safe_text(top_row.get("benchmark_id")),
                "recommended_action": recommended_action,
                "summary_warning": warning,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_SUMMARY_COLUMNS,
    )
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    status_rank = {"blocker": 0, "review": 1}
    impact_rank = {"source-gate-mismatch": 0, "source-blocked": 1, "source-review-needed": 2, "source-open": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(9)
    frame["_status_rank"] = frame["action_status"].map(status_rank).fillna(9)
    frame["_impact_rank"] = frame["source_impact_status"].map(impact_rank).fillna(9)
    frame = frame.sort_values(
        ["_priority_rank", "_status_rank", "_impact_rank", "source_impact_status"]
    ).drop(columns=["_priority_rank", "_status_rank", "_impact_rank"]).reset_index(drop=True)
    frame["summary_id"] = [f"BRSDIAS-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_SUMMARY_COLUMNS]


def build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown(
    dataset_impact_df: Optional[pd.DataFrame],
    impact_case_df: Optional[pd.DataFrame],
    *,
    action_summary_df: Optional[pd.DataFrame] = None,
    checklist_available: bool = False,
    title: str = "Benchmark source-audit decision dataset impact report",
    max_case_actions: int = 30,
) -> str:
    """Render a compact report for source-audit dataset impact."""

    has_dataset = dataset_impact_df is not None and not getattr(dataset_impact_df, "empty", True)
    has_cases = impact_case_df is not None and not getattr(impact_case_df, "empty", True)
    has_action_summary = action_summary_df is not None and not getattr(action_summary_df, "empty", True)
    if not has_dataset and not has_cases and not has_action_summary:
        return ""

    dataset = dataset_impact_df.copy() if has_dataset else _empty_reference_source_audit_case_decision_dataset_impact_df()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = ""
    if not dataset.empty:
        dataset["top_n"] = pd.to_numeric(dataset["top_n"], errors="coerce").fillna(0).astype(int)
        for column in (
            "case_count",
            "source_open_case_count",
            "source_blocker_case_count",
            "source_review_case_count",
            "source_gate_mismatch_case_count",
        ):
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce").fillna(0).astype(int)
        for column in ("mean_source_open_coverage", "mean_source_cleared_coverage"):
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce").fillna(0.0)
        dataset = dataset.sort_values("top_n")

    cases = impact_case_df.copy() if has_cases else _empty_reference_source_audit_case_decision_dataset_impact_case_df()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_CASE_COLUMNS:
        if column not in cases.columns:
            cases[column] = ""
    if not cases.empty:
        cases["top_n"] = pd.to_numeric(cases["top_n"], errors="coerce").fillna(0).astype(int)
        cases["coverage_ratio"] = pd.to_numeric(cases["coverage_ratio"], errors="coerce").fillna(0.0)
        cases["source_gate_mismatch"] = cases["source_gate_mismatch"].map(_claim_ready_bool)
        action_rank = {"blocker": 0, "review": 1, "closed": 2, "ready": 3, "none": 4}
        cases["_action_rank"] = cases["source_action_status"].map(action_rank).fillna(9)
        cases["_mismatch_rank"] = cases["source_gate_mismatch"].map(lambda value: 0 if bool(value) else 1)
        cases = cases.sort_values(
            ["top_n", "_action_rank", "_mismatch_rank", "benchmark_id", "source_impact_status"]
        ).drop(columns=["_action_rank", "_mismatch_rank"])

    action_summary = (
        action_summary_df.copy()
        if has_action_summary
        else _empty_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df()
    )
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_DATASET_IMPACT_ACTION_QUEUE_SUMMARY_COLUMNS:
        if column not in action_summary.columns:
            action_summary[column] = ""
    if not action_summary.empty:
        for column in ("action_count", "affected_case_count", "mismatch_count"):
            action_summary[column] = pd.to_numeric(action_summary[column], errors="coerce").fillna(0).astype(int)
        action_summary["mean_coverage_ratio"] = pd.to_numeric(
            action_summary["mean_coverage_ratio"], errors="coerce"
        ).fillna(0.0)
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        status_rank = {"blocker": 0, "review": 1}
        impact_rank = {"source-gate-mismatch": 0, "source-blocked": 1, "source-review-needed": 2, "source-open": 3}
        action_summary["_priority_rank"] = action_summary["priority"].map(priority_rank).fillna(9)
        action_summary["_status_rank"] = action_summary["action_status"].map(status_rank).fillna(9)
        action_summary["_impact_rank"] = action_summary["source_impact_status"].map(impact_rank).fillna(9)
        action_summary = action_summary.sort_values(
            ["_priority_rank", "_status_rank", "_impact_rank", "source_impact_status"]
        ).drop(columns=["_priority_rank", "_status_rank", "_impact_rank"])

    statuses = set(dataset["dataset_source_impact_status"].astype(str).str.strip().str.lower()) if not dataset.empty else set()
    if "source-gate-mismatch" in statuses:
        report_status = "source-gate-mismatch"
        next_action = "Regenerate readiness interpretation and resolve source gate mismatches before reporting dataset claims."
    elif "source-blocked" in statuses:
        report_status = "source-blocked"
        next_action = "Resolve source blocker cases before using benchmark coverage as a precision claim."
    elif "source-review-needed" in statuses:
        report_status = "source-review-needed"
        next_action = "Complete source independence review before publication-ready dataset claims."
    elif "source-open" in statuses:
        report_status = "source-open"
        next_action = "Review open source-audit decision cases before interpreting dataset-level coverage."
    elif {"source-cleared", "source-ready"} & statuses:
        report_status = "source-cleared"
        next_action = "Keep source-audit decision exports with the dataset benchmark report."
    else:
        report_status = "no-source-impact"
        next_action = "No source-audit dataset impact actions are currently tracked."

    actionable = cases[cases["source_action_status"].astype(str).isin(["blocker", "review"])].copy()
    blocker_rows = int(cases["source_action_status"].astype(str).eq("blocker").sum()) if not cases.empty else 0
    review_rows = int(cases["source_action_status"].astype(str).eq("review").sum()) if not cases.empty else 0
    mismatch_rows = int(cases["source_gate_mismatch"].map(bool).sum()) if not cases.empty else 0

    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_reference_source_audit_case_decision_dataset_impact.csv` and `pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases.csv`.",
        "",
        "## Gate",
        "",
        f"- Dataset source-impact status: `{report_status}`.",
        f"- Dataset impact rows: {int(len(dataset))}.",
        f"- Case impact rows: {int(len(cases))}.",
        f"- Action summary rows: {int(len(action_summary))}.",
        f"- Blocker case actions: {blocker_rows}.",
        f"- Review case actions: {review_rows}.",
        f"- Source gate mismatches: {mismatch_rows}.",
        f"- Checklist: {'available' if checklist_available else 'not available'}.",
        f"- Recommended action: {next_action}",
        "",
        "## Top-N Source Impact",
        "",
    ]
    if dataset.empty:
        lines.append("No dataset source-impact rows are available.")
    else:
        lines.append("| Top-N | Status | Cases | Open | Blockers | Review | Mismatch | Mean open coverage | Mean cleared coverage |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in dataset.iterrows():
            lines.append(
                "| {top_n} | {status} | {cases} | {open_cases} | {blockers} | {review} | {mismatch} | {open_cov:.3f} | {cleared_cov:.3f} |".format(
                    top_n=int(row.get("top_n") or 0),
                    status=_safe_text(row.get("dataset_source_impact_status")) or "-",
                    cases=int(row.get("case_count") or 0),
                    open_cases=int(row.get("source_open_case_count") or 0),
                    blockers=int(row.get("source_blocker_case_count") or 0),
                    review=int(row.get("source_review_case_count") or 0),
                    mismatch=int(row.get("source_gate_mismatch_case_count") or 0),
                    open_cov=float(row.get("mean_source_open_coverage") or 0.0),
                    cleared_cov=float(row.get("mean_source_cleared_coverage") or 0.0),
                )
            )

    lines.extend(["", "## Action Summary", ""])
    if action_summary.empty:
        lines.append("No source-impact action summary rows are available.")
    else:
        lines.append("| Priority | Action | Source impact | Actions | Cases | Top-N | Mismatch | Mean coverage | Top action | Recommended action |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |")
        for _, row in action_summary.iterrows():
            lines.append(
                "| {priority} | {action} | {impact} | {actions} | {cases} | {top_n} | {mismatch} | {coverage:.3f} | {top_action} | {recommended} |".format(
                    priority=_safe_text(row.get("priority")) or "-",
                    action=_safe_text(row.get("action_status")) or "-",
                    impact=_safe_text(row.get("source_impact_status")) or "-",
                    actions=int(row.get("action_count") or 0),
                    cases=int(row.get("affected_case_count") or 0),
                    top_n=_safe_text(row.get("top_n_values")) or "-",
                    mismatch=int(row.get("mismatch_count") or 0),
                    coverage=float(row.get("mean_coverage_ratio") or 0.0),
                    top_action=_safe_text(row.get("top_action_id")) or "-",
                    recommended=_safe_text(row.get("recommended_action")) or "Review source-impact actions.",
                )
            )

    lines.extend(["", "## Case Actions", ""])
    if actionable.empty:
        lines.append("No blocker or review source-impact case actions are queued.")
    else:
        lines.append("| Action | Top-N | Case | Source impact | Claim | Coverage | Adjusted priority | Recommended action |")
        lines.append("| --- | ---: | --- | --- | --- | ---: | --- | --- |")
        for _, row in actionable.head(max_case_actions).iterrows():
            impact = _safe_text(row.get("source_impact_status")) or "-"
            if bool(row.get("source_gate_mismatch")):
                impact = f"{impact} mismatch"
            lines.append(
                "| {action} | {top_n} | {case} | {impact} | {claim} | {coverage:.3f} | {priority} | {recommended} |".format(
                    action=_safe_text(row.get("source_action_status")) or "-",
                    top_n=int(row.get("top_n") or 0),
                    case=_safe_text(row.get("benchmark_id")) or "current",
                    impact=impact,
                    claim=_safe_text(row.get("claim_status")) or "-",
                    coverage=float(row.get("coverage_ratio") or 0.0),
                    priority=_safe_text(row.get("adjusted_source_priority")) or "-",
                    recommended=_safe_text(row.get("recommended_action")) or "Review this source-impact case.",
                )
            )
        remaining = int(len(actionable) - max_case_actions)
        if remaining > 0:
            lines.append(f"\n{remaining} additional source-impact case actions are available in the CSV export.")

    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_reference_source_audit_case_decision_closure_checklist_markdown(
    outcome_summary_df: Optional[pd.DataFrame],
    outcome_df: Optional[pd.DataFrame],
) -> str:
    """Render source-audit decision outcomes as a closure checklist."""

    summary = (
        outcome_summary_df.copy()
        if outcome_summary_df is not None and not getattr(outcome_summary_df, "empty", True)
        else _empty_reference_source_audit_case_decision_outcome_summary_df()
    )
    outcomes = (
        outcome_df.copy()
        if outcome_df is not None and not getattr(outcome_df, "empty", True)
        else _empty_reference_source_audit_case_decision_outcome_df()
    )
    if summary.empty and outcomes.empty:
        return ""
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = ""
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = ""

    lines = [
        "# Benchmark reference source audit decision closure checklist",
        "",
        "Use this checklist after uploading source-audit case decisions and before treating benchmark coverage as a precision claim.",
        "",
    ]
    if not summary.empty:
        row = summary.iloc[0]
        lines.extend(
            [
                "## Closure Gate",
                "",
                f"- [ ] Status `{_safe_text(row.get('closure_status')) or '-'}` with {int(row.get('open_actionable_case_count') or 0)} open actionable cases.",
                f"- [ ] Closed/actionable cases: {int(row.get('closed_actionable_case_count') or 0)} / {int(row.get('actionable_case_count') or 0)}.",
                f"- [ ] Recommended action: {_safe_text(row.get('recommended_action')) or '-'}",
                f"- [ ] Warning: {_safe_text(row.get('summary_warning')) or '-'}",
                "",
            ]
        )

    closed_statuses = {"cleared", "replaced", "source-ready"}
    status = outcomes["applied_status"].astype(str).str.strip().str.lower()
    open_rows = outcomes[~status.isin(closed_statuses)].copy()
    if not open_rows.empty:
        status_rank = {"blocked": 0, "pending": 1, "held": 2}
        open_rows["_status_rank"] = open_rows["applied_status"].map(_safe_text).map(status_rank).fillna(9)
        open_rows = open_rows.sort_values(["_status_rank", "benchmark_id"]).drop(columns=["_status_rank"])
        lines.extend(["## Open Case Actions", ""])
        for index, row in enumerate(open_rows.itertuples(index=False), start=1):
            case_id = _safe_text(row.benchmark_id) or "unnamed"
            applied_status = _safe_text(row.applied_status) or "unknown"
            decision = _safe_text(row.source_decision) or "review"
            validation = _safe_text(row.validation_status) or "review"
            next_action = _safe_text(row.next_action) or "Complete source-audit decision review."
            reason = _safe_text(row.outcome_reason) or "Source-audit case is not closed yet."
            lines.append(
                f"- [ ] BRSDC-{index:03d} case `{case_id}` `{applied_status}` "
                f"(decision `{decision}`, validation `{validation}`): {next_action} Reason: {reason}"
            )
        lines.append("")
    elif not outcomes.empty:
        lines.extend(
            [
                "## Open Case Actions",
                "",
                "- [ ] No open source-audit decision cases remain; keep decisions, validation, outcomes, and outcome summary exports with the benchmark report.",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _normalize_pocket_rows(pocket_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pocket_df is None or getattr(pocket_df, "empty", True) or "pocket_id" not in pocket_df.columns or "resid" not in pocket_df.columns:
        return pd.DataFrame(columns=["benchmark_id", "pocket_id", "chain", "resid", "resname"])
    working = pocket_df.copy()
    benchmark_id_column = _find_column(working, BENCHMARK_ID_ALIASES)
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["benchmark_id", "pocket_id", "chain", "resid", "resname"])
    working["benchmark_id"] = working[benchmark_id_column].astype(str).str.strip() if benchmark_id_column else ""
    working["pocket_id"] = working["pocket_id"].astype(str).str.strip()
    working["chain"] = working["chain"].astype(str).str.strip() if "chain" in working.columns else ""
    working["resname"] = working["resname"].astype(str).str.strip().str.upper() if "resname" in working.columns else ""
    working["resid"] = working["resid"].astype(int)
    return working[["benchmark_id", "pocket_id", "chain", "resid", "resname"]].drop_duplicates().reset_index(drop=True)


def _filter_rows_for_benchmark_id(frame: Optional[pd.DataFrame], benchmark_id: str) -> Optional[pd.DataFrame]:
    if frame is None or getattr(frame, "empty", True):
        return frame
    benchmark_id_column = _find_column(frame, BENCHMARK_ID_ALIASES)
    if not benchmark_id_column:
        return frame
    working = frame.copy()
    expected = str(benchmark_id or "").strip()
    values = working[benchmark_id_column].astype(str).str.strip()
    return working[values.eq(expected) | values.eq("")].reset_index(drop=True)


def _ranked_pocket_ids(pocket_df: pd.DataFrame, pocket_summary_df: Optional[pd.DataFrame]) -> list[str]:
    if pocket_df.empty:
        return []

    if pocket_summary_df is not None and not getattr(pocket_summary_df, "empty", True) and "pocket_id" in pocket_summary_df.columns:
        summary = pocket_summary_df.copy()
        summary["pocket_id"] = summary["pocket_id"].astype(str).str.strip()
        if "smart_rank_order" in summary.columns:
            summary["_rank_value"] = pd.to_numeric(summary["smart_rank_order"], errors="coerce")
        elif "rank" in summary.columns:
            summary["_rank_value"] = pd.to_numeric(summary["rank"], errors="coerce")
        else:
            summary["_rank_value"] = range(1, len(summary) + 1)
        fallback_rank = pd.Series(range(1, len(summary) + 1), index=summary.index)
        summary["_rank_value"] = pd.to_numeric(summary["_rank_value"], errors="coerce").fillna(fallback_rank)
        if "smart_rank_score" in summary.columns:
            summary["_score_value"] = pd.to_numeric(summary["smart_rank_score"], errors="coerce").fillna(0.0)
        elif "score" in summary.columns:
            summary["_score_value"] = pd.to_numeric(summary["score"], errors="coerce").fillna(0.0)
        else:
            summary["_score_value"] = 0.0
        ranked = summary.sort_values(["_rank_value", "_score_value", "pocket_id"], ascending=[True, False, True])["pocket_id"].tolist()
    else:
        ranked = pocket_df["pocket_id"].dropna().astype(str).str.strip().drop_duplicates().tolist()

    pocket_ids = set(pocket_df["pocket_id"].astype(str).str.strip().tolist())
    ordered = [pocket_id for pocket_id in ranked if pocket_id in pocket_ids]
    for pocket_id in pocket_df["pocket_id"].astype(str).str.strip().drop_duplicates().tolist():
        if pocket_id not in ordered:
            ordered.append(pocket_id)
    return ordered


def _reference_rows(reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if reference_df is None or getattr(reference_df, "empty", True):
        return _empty_reference_df()
    working = reference_df.copy()
    if "resid" not in working.columns:
        return _empty_reference_df()
    for column in BENCHMARK_REFERENCE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return _empty_reference_df()
    working["resid"] = working["resid"].astype(int)
    for column in ("benchmark_id", "chain", "resname", "reference_type", "reference_source", "reference_note", "expected_pocket_id"):
        working[column] = working[column].map(_safe_text)
    working["resname"] = working["resname"].str.upper()
    return working[BENCHMARK_REFERENCE_COLUMNS].drop_duplicates(subset=["benchmark_id", "chain", "resid", "reference_type"]).reset_index(drop=True)


def _reference_quality_issue(
    issue_number: int,
    reference_row: pd.Series,
    *,
    severity: str,
    issue_type: str,
    suggested_action: str,
    quality_warning: str,
) -> dict[str, object]:
    chain = _safe_text(reference_row.get("chain"))
    resid = int(reference_row.get("resid"))
    resname = _safe_text(reference_row.get("resname")).upper()
    return {
        "issue_id": f"REFQ-{issue_number:03d}",
        "severity": severity,
        "issue_type": issue_type,
        "benchmark_id": _safe_text(reference_row.get("benchmark_id")),
        "chain": chain,
        "resid": resid,
        "resname": resname,
        "residue_label": _residue_label(chain, resid, resname),
        "reference_type": _safe_text(reference_row.get("reference_type")),
        "reference_source": _safe_text(reference_row.get("reference_source")),
        "reference_note": _safe_text(reference_row.get("reference_note")),
        "suggested_action": suggested_action,
        "quality_warning": quality_warning,
    }


def _source_is_generic(value: object) -> bool:
    return _simplify_column_name(value) in GENERIC_REFERENCE_SOURCE_LABELS


def build_pocket_benchmark_reference_quality_issues(reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Flag curation risks in uploaded benchmark reference residues."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_quality_df()

    issues: list[dict[str, object]] = []

    def add_issue(reference_row: pd.Series, severity: str, issue_type: str, suggested_action: str, quality_warning: str) -> None:
        issues.append(
            _reference_quality_issue(
                len(issues) + 1,
                reference_row,
                severity=severity,
                issue_type=issue_type,
                suggested_action=suggested_action,
                quality_warning=quality_warning,
            )
        )

    for _, reference in references.iterrows():
        benchmark_id = _safe_text(reference.get("benchmark_id"))
        chain = _safe_text(reference.get("chain"))
        resname = _safe_text(reference.get("resname"))
        reference_type = _safe_text(reference.get("reference_type"))
        reference_source = _safe_text(reference.get("reference_source"))
        note_text = f"{reference_source} {_safe_text(reference.get('reference_note'))}".lower()

        if not benchmark_id:
            add_issue(
                reference,
                "P1",
                "missing_benchmark_id",
                "Assign a stable benchmark_id/case_id before using this row in batch benchmark aggregation.",
                "Rows without benchmark_id are grouped as an unnamed case and can hide case-level errors.",
            )
        if _source_is_generic(reference_source):
            add_issue(
                reference,
                "P1",
                "generic_reference_source",
                "Replace the generic source with M-CSA ID, PDB ID, PMID, DOI, or another stable curated dataset label.",
                "Generic sources make benchmark residues difficult to audit or reproduce.",
            )
        if not chain:
            add_issue(
                reference,
                "P2",
                "wildcard_chain",
                "Map the residue to a PDB chain, or document why chain-agnostic matching is intended.",
                "Blank chain matches any pocket chain and can overestimate catalytic coverage.",
            )
        if not resname:
            add_issue(
                reference,
                "P2",
                "missing_resname",
                "Fill the expected three-letter residue name and verify it against the structure.",
                "Missing residue identity weakens numbering and mapping validation.",
            )
        if reference_type and not any(token in reference_type.lower() for token in SUPPORTED_REFERENCE_TYPE_TOKENS):
            add_issue(
                reference,
                "P2",
                "unsupported_reference_type",
                "Use a functional role such as Catalytic residue, Binding residue, Metal binding, Mutagenesis, Substrate, or Cofactor.",
                "Unrecognized reference_type values can make benchmark interpretation ambiguous.",
            )
        if any(token in note_text for token in MAPPING_ASSUMPTION_TOKENS):
            add_issue(
                reference,
                "P2",
                "mapping_assumption_note",
                "Record the exact UniProt/PDB numbering conversion and verify residue identity in the uploaded structure.",
                "UniProt, mature-chain, isoform, precursor, or offset assumptions can shift catalytic residue numbers.",
            )

    for _, group in references.groupby(["benchmark_id", "chain", "resid"], dropna=False):
        if len(group) <= 1:
            continue
        reference_types = group["reference_type"].map(_safe_text).str.lower().drop_duplicates()
        sources = group["reference_source"].map(_safe_text).str.lower().drop_duplicates()
        if len(reference_types) > 1 or len(sources) > 1:
            add_issue(
                group.iloc[0],
                "P3",
                "multi_role_or_source_residue",
                "Keep the duplicated evidence if intentional, but consolidate notes so reviewers understand why the same residue has multiple roles or sources.",
                "Multiple roles or sources for one residue are valid evidence, but should be explicit in curated benchmark records.",
            )

    if not issues:
        return _empty_reference_quality_df()
    return pd.DataFrame(issues, columns=BENCHMARK_REFERENCE_QUALITY_COLUMNS)


def build_pocket_benchmark_reference_quality_summary(quality_issue_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize benchmark reference curation issues by severity and type."""

    if quality_issue_df is None or getattr(quality_issue_df, "empty", True):
        return _empty_reference_quality_summary_df()
    working = quality_issue_df.copy()
    for column in BENCHMARK_REFERENCE_QUALITY_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    rows: list[dict[str, object]] = []
    severity_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    for (severity, issue_type), group in working.groupby(["severity", "issue_type"], dropna=False):
        severity_text = _safe_text(severity) or "P3"
        case_count = int(group["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().nunique())
        residue_count = int(
            group[["benchmark_id", "chain", "resid"]]
            .astype(str)
            .drop_duplicates()
            .shape[0]
        )
        if severity_text in {"P0", "P1"}:
            status = "needs-curation"
            warning = "Fix before using this reference set as a batch accuracy benchmark."
        elif severity_text == "P2":
            status = "review"
            warning = "Review before interpreting precision metrics."
        else:
            status = "informational"
            warning = "Document the curation decision for traceability."
        rows.append(
            {
                "severity": severity_text,
                "issue_type": _safe_text(issue_type),
                "issue_count": int(len(group)),
                "affected_case_count": case_count,
                "affected_residue_count": residue_count,
                "suggested_action": _safe_text(group.iloc[0].get("suggested_action")),
                "summary_status": status,
                "summary_warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_QUALITY_SUMMARY_COLUMNS).sort_values(
        ["severity", "issue_count", "issue_type"],
        key=lambda series: series.map(severity_rank).fillna(series) if series.name == "severity" else series,
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_pocket_benchmark_reference_quality_checklist_markdown(
    quality_issue_df: Optional[pd.DataFrame],
    quality_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render curation issues as a reviewer checklist."""

    if quality_issue_df is None or getattr(quality_issue_df, "empty", True):
        return ""
    issues = quality_issue_df.copy()
    summary = build_pocket_benchmark_reference_quality_summary(issues) if quality_summary_df is None else quality_summary_df
    lines = [
        "# Benchmark reference curation checklist",
        "",
        "Resolve or explicitly accept these issues before using the reference table as a precision benchmark.",
        "",
    ]
    if summary is not None and not getattr(summary, "empty", True):
        lines.extend(["## Summary", ""])
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.severity} `{row.issue_type}`: {int(row.issue_count)} issues / {int(row.affected_residue_count)} residues / {row.summary_status}."
            )
        lines.append("")
    lines.extend(["## Actions", ""])
    severity_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    issues["_severity_order"] = issues["severity"].map(severity_order).fillna(9)
    issues = issues.sort_values(["_severity_order", "issue_type", "benchmark_id", "chain", "resid"]).drop(columns=["_severity_order"])
    for row in issues.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        lines.append(f"- [ ] {row.severity} `{row.issue_type}` for {case_text}, residue `{row.residue_label}`: {row.suggested_action}")
    return "\n".join(lines).strip() + "\n"


def _structure_residue_rows(atom_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if atom_df is None or getattr(atom_df, "empty", True):
        return pd.DataFrame(columns=["chain", "resid", "resname"])
    working = atom_df.copy()
    if "resid" not in working.columns:
        return pd.DataFrame(columns=["chain", "resid", "resname"])
    for column in ("chain", "resname"):
        if column not in working.columns:
            working[column] = ""
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["chain", "resid", "resname"])
    working["resid"] = working["resid"].astype(int)
    working["chain"] = working["chain"].map(_safe_text)
    working["resname"] = working["resname"].map(_safe_text).str.upper()
    if "record_type" in working.columns:
        atom_rows = working[working["record_type"].astype(str).str.upper().eq("ATOM")].copy()
        if not atom_rows.empty:
            working = atom_rows
    return working[["chain", "resid", "resname"]].drop_duplicates().reset_index(drop=True)


def _structure_validation_issue(
    issue_number: int,
    reference_row: pd.Series,
    *,
    severity: str,
    issue_type: str,
    structure_chains: str,
    structure_resnames: str,
    matched_chain: str,
    matched_resname: str,
    suggested_action: str,
    validation_warning: str,
) -> dict[str, object]:
    chain = _safe_text(reference_row.get("chain"))
    resid = int(reference_row.get("resid"))
    resname = _safe_text(reference_row.get("resname")).upper()
    return {
        "issue_id": f"REFS-{issue_number:03d}",
        "severity": severity,
        "issue_type": issue_type,
        "benchmark_id": _safe_text(reference_row.get("benchmark_id")),
        "chain": chain,
        "resid": resid,
        "resname": resname,
        "residue_label": _residue_label(chain, resid, resname),
        "structure_chains": structure_chains,
        "structure_resnames": structure_resnames,
        "matched_chain": matched_chain,
        "matched_resname": matched_resname,
        "reference_source": _safe_text(reference_row.get("reference_source")),
        "reference_note": _safe_text(reference_row.get("reference_note")),
        "suggested_action": suggested_action,
        "validation_warning": validation_warning,
    }


def build_pocket_benchmark_reference_structure_validation(
    reference_df: Optional[pd.DataFrame],
    atom_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Validate benchmark reference residues against residues present in the uploaded structure."""

    references = _reference_rows(reference_df)
    structure_residues = _structure_residue_rows(atom_df)
    if references.empty or structure_residues.empty:
        return _empty_reference_structure_validation_df()

    issues: list[dict[str, object]] = []

    def add_issue(
        reference_row: pd.Series,
        *,
        severity: str,
        issue_type: str,
        structure_chains: str,
        structure_resnames: str,
        matched_chain: str,
        matched_resname: str,
        suggested_action: str,
        validation_warning: str,
    ) -> None:
        issues.append(
            _structure_validation_issue(
                len(issues) + 1,
                reference_row,
                severity=severity,
                issue_type=issue_type,
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain=matched_chain,
                matched_resname=matched_resname,
                suggested_action=suggested_action,
                validation_warning=validation_warning,
            )
        )

    for _, reference in references.iterrows():
        chain = _safe_text(reference.get("chain"))
        resid = int(reference.get("resid"))
        resname = _safe_text(reference.get("resname")).upper()
        same_resid = structure_residues[structure_residues["resid"].astype(int).eq(resid)]
        if chain:
            same_resid = same_resid[same_resid["chain"].map(_safe_text).eq(chain)]

        if same_resid.empty:
            all_same_number = structure_residues[structure_residues["resid"].astype(int).eq(resid)]
            structure_chains = ";".join(sorted(all_same_number["chain"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist()))
            structure_resnames = ";".join(sorted(all_same_number["resname"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist()))
            add_issue(
                reference,
                severity="P1",
                issue_type="reference_residue_absent",
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain="",
                matched_resname="",
                suggested_action="Check whether the reference uses UniProt, mature-chain, isoform, or another PDB author's numbering before interpreting coverage.",
                validation_warning="The curated reference residue was not found in the uploaded structure.",
            )
            continue

        chains = sorted(same_resid["chain"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist())
        resnames = sorted(same_resid["resname"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist())
        structure_chains = ";".join(chains)
        structure_resnames = ";".join(resnames)

        if not chain and len(chains) > 1:
            add_issue(
                reference,
                severity="P2",
                issue_type="wildcard_chain_ambiguous_in_structure",
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain="",
                matched_resname=structure_resnames,
                suggested_action="Choose the intended PDB chain for this benchmark residue or split the reference into chain-specific rows.",
                validation_warning="Blank chain matches multiple chains in the uploaded structure and can inflate benchmark coverage.",
            )

        if resname and resname not in resnames:
            add_issue(
                reference,
                severity="P1",
                issue_type="reference_resname_mismatch",
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain=structure_chains,
                matched_resname=structure_resnames,
                suggested_action="Verify residue numbering and expected residue identity against the uploaded PDB before using this row for accuracy claims.",
                validation_warning="The reference residue name does not match the residue identity in the uploaded structure.",
            )

    if not issues:
        return _empty_reference_structure_validation_df()
    return pd.DataFrame(issues, columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS)


def build_pocket_benchmark_reference_structure_validation_summary(
    validation_issue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize structure validation issues for benchmark reference residues."""

    if validation_issue_df is None or getattr(validation_issue_df, "empty", True):
        return _empty_reference_structure_validation_summary_df()
    working = validation_issue_df.copy()
    for column in BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    rows: list[dict[str, object]] = []
    severity_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    for (severity, issue_type), group in working.groupby(["severity", "issue_type"], dropna=False):
        severity_text = _safe_text(severity) or "P3"
        if severity_text == "P1":
            status = "mapping-blocked"
            warning = "Fix before treating misses as pocket-detection errors."
        elif severity_text == "P2":
            status = "mapping-review"
            warning = "Review before trusting wildcard-chain coverage."
        else:
            status = "informational"
            warning = "Document the structure validation decision."
        rows.append(
            {
                "severity": severity_text,
                "issue_type": _safe_text(issue_type),
                "issue_count": int(len(group)),
                "affected_case_count": int(group["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().nunique()),
                "affected_residue_count": int(group[["benchmark_id", "chain", "resid"]].astype(str).drop_duplicates().shape[0]),
                "suggested_action": _safe_text(group.iloc[0].get("suggested_action")),
                "summary_status": status,
                "summary_warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_SUMMARY_COLUMNS).sort_values(
        ["severity", "issue_count", "issue_type"],
        key=lambda series: series.map(severity_rank).fillna(series) if series.name == "severity" else series,
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_pocket_benchmark_reference_structure_validation_checklist_markdown(
    validation_issue_df: Optional[pd.DataFrame],
    validation_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render structure validation issues as a reviewer checklist."""

    if validation_issue_df is None or getattr(validation_issue_df, "empty", True):
        return ""
    issues = validation_issue_df.copy()
    summary = (
        build_pocket_benchmark_reference_structure_validation_summary(issues)
        if validation_summary_df is None
        else validation_summary_df
    )
    lines = [
        "# Benchmark reference structure validation checklist",
        "",
        "Resolve these structure-mapping issues before interpreting catalytic pocket benchmark misses as detection failures.",
        "",
    ]
    if summary is not None and not getattr(summary, "empty", True):
        lines.extend(["## Summary", ""])
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.severity} `{row.issue_type}`: {int(row.issue_count)} issues / {int(row.affected_residue_count)} residues / {row.summary_status}."
            )
        lines.append("")
    lines.extend(["## Actions", ""])
    severity_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    issues["_severity_order"] = issues["severity"].map(severity_order).fillna(9)
    issues = issues.sort_values(["_severity_order", "issue_type", "benchmark_id", "chain", "resid"]).drop(columns=["_severity_order"])
    for row in issues.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        structure_text = f"structure chains `{row.structure_chains or '-'}`, resnames `{row.structure_resnames or '-'}`"
        lines.append(
            f"- [ ] {row.severity} `{row.issue_type}` for {case_text}, residue `{row.residue_label}` ({structure_text}): {row.suggested_action}"
        )
    return "\n".join(lines).strip() + "\n"


def _reference_issue_queue_rows(issue_df: Optional[pd.DataFrame], *, issue_source: str, warning_column: str) -> list[dict[str, object]]:
    if issue_df is None or getattr(issue_df, "empty", True):
        return []
    working = issue_df.copy()
    for column in ("severity", "issue_type", "benchmark_id", "residue_label", "chain", "resid", "resname", "suggested_action", warning_column):
        if column not in working.columns:
            working[column] = ""
    rows: list[dict[str, object]] = []
    for _, issue in working.iterrows():
        priority = _safe_text(issue.get("severity")) or "P3"
        if priority not in {"P0", "P1", "P2"}:
            continue
        rows.append(
            {
                "priority": priority,
                "action_status": "blocker" if priority in {"P0", "P1"} else "review",
                "issue_source": issue_source,
                "issue_type": _safe_text(issue.get("issue_type")),
                "benchmark_id": _safe_text(issue.get("benchmark_id")),
                "residue_label": _safe_text(issue.get("residue_label")),
                "chain": _safe_text(issue.get("chain")),
                "resid": _safe_text(issue.get("resid")),
                "resname": _safe_text(issue.get("resname")).upper(),
                "suggested_action": _safe_text(issue.get("suggested_action")),
                "readiness_warning": _safe_text(issue.get(warning_column)),
            }
        )
    return rows


def _source_audit_readiness_issue(audit_row: pd.Series) -> tuple[str, str, str, str] | None:
    source_claim_status = _safe_text(audit_row.get("source_claim_status"))
    independent_claim_status = _safe_text(audit_row.get("can_support_independent_claim"))
    if source_claim_status == "blocked-provisional" or independent_claim_status == "no":
        return (
            "P0",
            "provisional_reference_source",
            "Replace the provisional reference with a curated upload or accepted review decision before using coverage as a precision claim.",
            "Provisional external evidence can overlap with detection inputs and cannot support independent benchmark claims.",
        )
    if source_claim_status == "source-unknown":
        return (
            "P1",
            "unknown_reference_source",
            "Record whether the reference came from curated upload, reviewed candidate, or provisional evidence before using benchmark coverage.",
            "Unknown reference provenance can invalidate benchmark interpretation.",
        )
    if source_claim_status == "review-qualified" or independent_claim_status == "review-required":
        return (
            "P2",
            "review_qualified_reference_source",
            "Confirm the accepted candidate reference is independent from detection/reranking evidence before reporting final accuracy.",
            "Reviewer-accepted candidate references are useful for triage but still need independence confirmation for precision claims.",
        )
    return None


def _source_audit_decision_outcome_by_case(decision_outcome_df: Optional[pd.DataFrame]) -> dict[str, pd.Series]:
    if decision_outcome_df is None or getattr(decision_outcome_df, "empty", True):
        return {}
    outcomes = decision_outcome_df.copy()
    for column in BENCHMARK_REFERENCE_SOURCE_AUDIT_CASE_DECISION_OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = ""
    return {
        _safe_text(row.get("benchmark_id")): row
        for _, row in outcomes.iterrows()
        if _safe_text(row.get("benchmark_id"))
    }


def _source_audit_decision_readiness_issue(outcome_row: pd.Series) -> tuple[str, str, str, str] | None:
    applied_status = _safe_text(outcome_row.get("applied_status")).lower()
    next_action = _safe_text(outcome_row.get("next_action"))
    outcome_reason = _safe_text(outcome_row.get("outcome_reason"))
    if applied_status in {"cleared", "replaced", "source-ready"}:
        return None
    if applied_status == "blocked":
        return (
            "P0",
            "source_audit_case_decision_blocked",
            next_action or "Fix blocked source-audit case decisions before using benchmark coverage in precision claims.",
            outcome_reason or "At least one source-audit case decision is invalid or blocked.",
        )
    if applied_status == "pending":
        return (
            "P1",
            "source_audit_case_decision_pending",
            next_action or "Upload or complete source-audit case decisions for pending cases.",
            outcome_reason or "Some source-risk cases still have no clearing decision.",
        )
    if applied_status == "held":
        return (
            "P2",
            "source_audit_case_decision_held",
            next_action or "Resolve held source-risk cases or exclude them from independent benchmark claims.",
            outcome_reason or "Held source-risk cases remain outside claim-ready source provenance.",
        )
    return (
        "P1",
        "source_audit_case_decision_unknown",
        next_action or "Review source-audit case decision outcomes before reporting benchmark precision.",
        outcome_reason or "Source-audit decision outcome status is unknown.",
    )


def _source_audit_queue_rows(
    source_audit_df: Optional[pd.DataFrame],
    source_audit_decision_outcome_df: Optional[pd.DataFrame] = None,
) -> list[dict[str, object]]:
    if source_audit_df is None or getattr(source_audit_df, "empty", True):
        return []
    working = source_audit_df.copy()
    for column in (
        "benchmark_id",
        "chain",
        "resid",
        "resname",
        "source_claim_status",
        "can_support_independent_claim",
        "recommended_action",
        "provenance_warning",
    ):
        if column not in working.columns:
            working[column] = ""

    decision_by_case = _source_audit_decision_outcome_by_case(source_audit_decision_outcome_df)
    rows: list[dict[str, object]] = []
    for _, audit in working.iterrows():
        benchmark_id = _safe_text(audit.get("benchmark_id"))
        decision_outcome = decision_by_case.get(benchmark_id)
        issue = (
            _source_audit_decision_readiness_issue(decision_outcome)
            if decision_outcome is not None
            else _source_audit_readiness_issue(audit)
        )
        if issue is None:
            continue
        priority, issue_type, suggested_action, warning = issue
        resid_value = _safe_int(audit.get("resid")) or 0
        chain = _safe_text(audit.get("chain"))
        resname = _safe_text(audit.get("resname")).upper()
        rows.append(
            {
                "priority": priority,
                "action_status": "blocker" if priority in {"P0", "P1"} else "review",
                "issue_source": "source_audit",
                "issue_type": issue_type,
                "benchmark_id": benchmark_id,
                "residue_label": _residue_label(chain, resid_value, resname) if resid_value else "",
                "chain": chain,
                "resid": _safe_text(audit.get("resid")),
                "resname": resname,
                "suggested_action": _safe_text(audit.get("recommended_action")) or suggested_action,
                "readiness_warning": _safe_text(audit.get("provenance_warning")) or warning,
            }
        )
    return rows


def build_pocket_benchmark_reference_readiness_queue(
    quality_issue_df: Optional[pd.DataFrame],
    structure_validation_issue_df: Optional[pd.DataFrame],
    source_audit_df: Optional[pd.DataFrame] = None,
    source_audit_decision_outcome_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine reference curation and structure validation issues into one readiness queue."""

    rows = [
        *_reference_issue_queue_rows(quality_issue_df, issue_source="curation_quality", warning_column="quality_warning"),
        *_reference_issue_queue_rows(
            structure_validation_issue_df,
            issue_source="structure_validation",
            warning_column="validation_warning",
        ),
        *_source_audit_queue_rows(source_audit_df, source_audit_decision_outcome_df),
    ]
    if not rows:
        return _empty_reference_readiness_queue_df()

    severity_order = {"P0": 0, "P1": 1, "P2": 2}
    queue = pd.DataFrame(rows)
    queue["_priority_order"] = queue["priority"].map(severity_order).fillna(9)
    queue = queue.sort_values(
        ["_priority_order", "issue_source", "issue_type", "benchmark_id", "chain", "resid"],
        ascending=[True, True, True, True, True, True],
    ).drop(columns=["_priority_order"]).reset_index(drop=True)
    queue["action_id"] = [f"REFR-{index + 1:03d}" for index in range(len(queue))]
    return queue[BENCHMARK_REFERENCE_READINESS_QUEUE_COLUMNS]


def build_pocket_benchmark_reference_readiness_summary(
    reference_df: Optional[pd.DataFrame],
    quality_issue_df: Optional[pd.DataFrame],
    structure_validation_issue_df: Optional[pd.DataFrame],
    source_audit_df: Optional[pd.DataFrame] = None,
    source_audit_decision_outcome_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a one-row gate for whether benchmark reference rows are ready for accuracy claims."""

    references = _reference_rows(reference_df)
    queue = build_pocket_benchmark_reference_readiness_queue(
        quality_issue_df,
        structure_validation_issue_df,
        source_audit_df,
        source_audit_decision_outcome_df,
    )
    reference_count = int(len(references))
    curation_count = 0 if quality_issue_df is None or getattr(quality_issue_df, "empty", True) else int(len(quality_issue_df))
    structure_count = (
        0
        if structure_validation_issue_df is None or getattr(structure_validation_issue_df, "empty", True)
        else int(len(structure_validation_issue_df))
    )
    source_audit_count = int(queue["issue_source"].astype(str).eq("source_audit").sum()) if not queue.empty and "issue_source" in queue.columns else 0

    if reference_count <= 0:
        row = {
            "readiness_status": "no-reference",
            "reference_residue_count": 0,
            "curation_issue_count": curation_count,
            "structure_validation_issue_count": structure_count,
            "source_audit_issue_count": source_audit_count,
            "p0_p1_issue_count": 0,
            "p2_issue_count": 0,
            "blocking_issue_types": "",
            "review_issue_types": "",
            "recommended_action": "Upload curated catalytic residues before running benchmark readiness checks.",
            "readiness_warning": "Benchmark accuracy cannot be interpreted without reference residues.",
        }
        return pd.DataFrame([row], columns=BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS)

    p0_p1_count = (
        int(queue["priority"].astype(str).isin(["P0", "P1"]).sum())
        if not queue.empty and "priority" in queue.columns
        else 0
    )
    p2_count = int(queue["priority"].astype(str).eq("P2").sum()) if not queue.empty and "priority" in queue.columns else 0
    blocking_types = ""
    review_types = ""
    if not queue.empty:
        blocking_types = "; ".join(
            sorted(queue.loc[queue["priority"].astype(str).isin(["P0", "P1"]), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )
        review_types = "; ".join(
            sorted(queue.loc[queue["priority"].astype(str).eq("P2"), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )

    if p0_p1_count > 0:
        readiness_status = "blocked"
        if "provisional_reference_source" in blocking_types or "unknown_reference_source" in blocking_types:
            recommended_action = "Resolve P0/P1 reference provenance blockers before using coverage as a precision claim."
            warning = "Current benchmark coverage may be based on non-independent or unknown reference evidence."
        else:
            recommended_action = "Resolve P0/P1 reference curation or structure-mapping blockers before using coverage as a precision claim."
            warning = "Current benchmark misses may reflect reference numbering or curation errors, not pocket-detection errors."
    elif p2_count > 0:
        readiness_status = "review-needed"
        if "review_qualified_reference_source" in review_types:
            recommended_action = "Confirm source independence for reviewed candidate references before publishing benchmark coverage."
            warning = "Benchmark coverage can be inspected, but reviewed candidate references should be labeled as independence-pending."
        else:
            recommended_action = "Review P2 issues, especially wildcard chain and residue identity assumptions, before publishing benchmark coverage."
            warning = "Benchmark coverage can be inspected, but should be labeled as reviewer-pending."
    else:
        readiness_status = "ready"
        recommended_action = "Reference residues are ready for catalytic pocket coverage interpretation."
        warning = "No benchmark reference readiness blockers detected."

    row = {
        "readiness_status": readiness_status,
        "reference_residue_count": reference_count,
        "curation_issue_count": curation_count,
        "structure_validation_issue_count": structure_count,
        "source_audit_issue_count": source_audit_count,
        "p0_p1_issue_count": p0_p1_count,
        "p2_issue_count": p2_count,
        "blocking_issue_types": blocking_types,
        "review_issue_types": review_types,
        "recommended_action": recommended_action,
        "readiness_warning": warning,
    }
    return pd.DataFrame([row], columns=BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS)


def _normalized_case_id(value: object, default_benchmark_id: str) -> str:
    return _safe_text(value) or _safe_text(default_benchmark_id) or "current"


def _readiness_issue_frame(issue_df: Optional[pd.DataFrame], *, issue_source: str, default_benchmark_id: str) -> pd.DataFrame:
    if issue_df is None or getattr(issue_df, "empty", True):
        return pd.DataFrame(columns=["benchmark_id", "severity", "issue_type", "issue_source"])
    working = issue_df.copy()
    for column in ("benchmark_id", "severity", "issue_type"):
        if column not in working.columns:
            working[column] = ""
    working["benchmark_id"] = working["benchmark_id"].map(lambda value: _normalized_case_id(value, default_benchmark_id))
    working["severity"] = working["severity"].map(_safe_text)
    working["issue_type"] = working["issue_type"].map(_safe_text)
    working["issue_source"] = issue_source
    return working[["benchmark_id", "severity", "issue_type", "issue_source"]].reset_index(drop=True)


def _source_audit_readiness_issue_frame(
    source_audit_df: Optional[pd.DataFrame],
    source_audit_decision_outcome_df: Optional[pd.DataFrame] = None,
    *,
    default_benchmark_id: str,
) -> pd.DataFrame:
    rows = _source_audit_queue_rows(source_audit_df, source_audit_decision_outcome_df)
    if not rows:
        return pd.DataFrame(columns=["benchmark_id", "severity", "issue_type", "issue_source"])
    frame = pd.DataFrame(rows)
    frame["benchmark_id"] = frame["benchmark_id"].map(lambda value: _normalized_case_id(value, default_benchmark_id))
    frame["severity"] = frame["priority"].map(_safe_text)
    frame["issue_source"] = "source_audit"
    return frame[["benchmark_id", "severity", "issue_type", "issue_source"]].reset_index(drop=True)


def build_pocket_benchmark_reference_readiness_case_summary(
    reference_df: Optional[pd.DataFrame],
    quality_issue_df: Optional[pd.DataFrame],
    structure_validation_issue_df: Optional[pd.DataFrame],
    source_audit_df: Optional[pd.DataFrame] = None,
    source_audit_decision_outcome_df: Optional[pd.DataFrame] = None,
    *,
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Build per-case benchmark reference readiness gates."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_readiness_case_summary_df()

    fallback_id = _safe_text(default_benchmark_id) or "current"
    references = references.copy()
    references["benchmark_id"] = references["benchmark_id"].map(lambda value: _normalized_case_id(value, fallback_id))
    quality_issues = _readiness_issue_frame(quality_issue_df, issue_source="curation_quality", default_benchmark_id=fallback_id)
    structure_issues = _readiness_issue_frame(
        structure_validation_issue_df,
        issue_source="structure_validation",
        default_benchmark_id=fallback_id,
    )
    source_issues = _source_audit_readiness_issue_frame(
        source_audit_df,
        source_audit_decision_outcome_df,
        default_benchmark_id=fallback_id,
    )
    issue_frame = pd.concat([quality_issues, structure_issues, source_issues], ignore_index=True)

    case_ids = sorted(
        set(references["benchmark_id"].map(_safe_text).tolist())
        | set(issue_frame["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
    )
    rows: list[dict[str, object]] = []
    for benchmark_id in case_ids:
        case_references = references[references["benchmark_id"].astype(str).eq(benchmark_id)]
        case_issues = issue_frame[issue_frame["benchmark_id"].astype(str).eq(benchmark_id)]
        curation_count = int(case_issues["issue_source"].astype(str).eq("curation_quality").sum())
        structure_count = int(case_issues["issue_source"].astype(str).eq("structure_validation").sum())
        source_audit_count = int(case_issues["issue_source"].astype(str).eq("source_audit").sum())
        p0_p1_count = int(case_issues["severity"].astype(str).isin(["P0", "P1"]).sum())
        p2_count = int(case_issues["severity"].astype(str).eq("P2").sum())
        blocking_types = "; ".join(
            sorted(case_issues.loc[case_issues["severity"].astype(str).isin(["P0", "P1"]), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )
        review_types = "; ".join(
            sorted(case_issues.loc[case_issues["severity"].astype(str).eq("P2"), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )

        if p0_p1_count > 0:
            readiness_status = "blocked"
            if "provisional_reference_source" in blocking_types or "unknown_reference_source" in blocking_types:
                recommended_action = "Resolve this case's reference provenance blockers before using its coverage in dataset-level claims."
                warning = "This case can distort dataset coverage until provisional or unknown reference sources are replaced."
            else:
                recommended_action = "Resolve this case's P0/P1 reference curation or structure-mapping blockers before using its coverage in dataset-level claims."
                warning = "This case can distort dataset coverage until reference numbering and curation blockers are fixed."
        elif p2_count > 0:
            readiness_status = "review-needed"
            if "review_qualified_reference_source" in review_types:
                recommended_action = "Confirm this case's reviewed candidate references are independent before treating coverage as publication-ready."
                warning = "This case can be inspected, but should be labeled source-independence-pending."
            else:
                recommended_action = "Review this case's P2 assumptions before treating its coverage as publication-ready."
                warning = "This case can be inspected, but should be labeled reviewer-pending."
        else:
            readiness_status = "ready"
            recommended_action = "This case is ready for catalytic pocket coverage interpretation."
            warning = "No case-level readiness blockers detected."

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "readiness_status": readiness_status,
                "reference_residue_count": int(len(case_references)),
                "curation_issue_count": curation_count,
                "structure_validation_issue_count": structure_count,
                "source_audit_issue_count": source_audit_count,
                "p0_p1_issue_count": p0_p1_count,
                "p2_issue_count": p2_count,
                "blocking_issue_types": blocking_types,
                "review_issue_types": review_types,
                "recommended_action": recommended_action,
                "readiness_warning": warning,
            }
        )

    if not rows:
        return _empty_reference_readiness_case_summary_df()
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_READINESS_CASE_SUMMARY_COLUMNS)


def build_pocket_benchmark_reference_readiness_checklist_markdown(
    readiness_queue_df: Optional[pd.DataFrame],
    readiness_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render the combined benchmark reference readiness gate as a checklist."""

    if readiness_queue_df is None or getattr(readiness_queue_df, "empty", True):
        return ""
    queue = readiness_queue_df.copy()
    summary = _empty_reference_readiness_summary_df() if readiness_summary_df is None else readiness_summary_df
    lines = [
        "# Benchmark reference readiness checklist",
        "",
        "Use this checklist before interpreting catalytic pocket benchmark coverage as an accuracy claim.",
        "",
    ]
    if summary is not None and not getattr(summary, "empty", True):
        row = summary.iloc[0]
        lines.extend(
            [
                "## Gate",
                "",
                f"- Status: `{_safe_text(row.get('readiness_status')) or '-'}`.",
                f"- P0/P1 blockers: {int(row.get('p0_p1_issue_count') or 0)}.",
                f"- P2 review items: {int(row.get('p2_issue_count') or 0)}.",
                f"- Recommended action: {_safe_text(row.get('recommended_action')) or '-'}",
                "",
            ]
        )
    lines.extend(["## Actions", ""])
    for row in queue.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        lines.append(
            f"- [ ] {row.priority} `{row.issue_source}/{row.issue_type}` for {case_text}, residue `{row.residue_label or '-'}`: {row.suggested_action}"
        )
    return "\n".join(lines).strip() + "\n"


def _matches_reference(pocket_row: pd.Series, reference_row: pd.Series) -> bool:
    ref_benchmark_id = str(reference_row.get("benchmark_id") or "").strip()
    pocket_benchmark_id = str(pocket_row.get("benchmark_id") or "").strip()
    if ref_benchmark_id and pocket_benchmark_id and ref_benchmark_id != pocket_benchmark_id:
        return False
    if int(pocket_row.get("resid")) != int(reference_row.get("resid")):
        return False
    ref_chain = str(reference_row.get("chain") or "").strip()
    pocket_chain = str(pocket_row.get("chain") or "").strip()
    return not ref_chain or not pocket_chain or ref_chain == pocket_chain


def build_pocket_benchmark_details(
    reference_df: Optional[pd.DataFrame],
    pocket_df: Optional[pd.DataFrame],
    pocket_summary_df: Optional[pd.DataFrame] = None,
    *,
    top_thresholds: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    references = _reference_rows(reference_df)
    pockets = _normalize_pocket_rows(pocket_df)
    if references.empty:
        return _empty_detail_df()

    ranked_ids = _ranked_pocket_ids(pockets, pocket_summary_df)
    rank_map = {pocket_id: rank + 1 for rank, pocket_id in enumerate(ranked_ids)}
    rows: list[dict[str, object]] = []
    for reference in references.itertuples(index=False):
        ref_series = pd.Series(reference._asdict())
        matching_rows = []
        for pocket_row in pockets.itertuples(index=False):
            pocket_series = pd.Series(pocket_row._asdict())
            if _matches_reference(pocket_series, ref_series):
                matching_rows.append(pocket_series)
        matching_rows = sorted(matching_rows, key=lambda row: (rank_map.get(str(row.get("pocket_id")), 999999), str(row.get("pocket_id"))))
        matched_pocket_ids = [str(row.get("pocket_id")) for row in matching_rows if str(row.get("pocket_id"))]
        matched_rank = min([rank_map.get(pocket_id, 999999) for pocket_id in matched_pocket_ids], default=0)
        matched_pocket_id = matched_pocket_ids[0] if matched_pocket_ids else ""
        expected_pocket_id = str(ref_series.get("expected_pocket_id") or "").strip()
        rows.append(
            {
                **{column: ref_series.get(column, "") for column in BENCHMARK_REFERENCE_COLUMNS},
                "residue_label": _residue_label(str(ref_series.get("chain") or ""), int(ref_series.get("resid")), str(ref_series.get("resname") or "")),
                "matched": bool(matched_pocket_ids),
                "matched_pocket_id": matched_pocket_id,
                "matched_rank": int(matched_rank) if matched_rank and matched_rank < 999999 else 0,
                "matched_pocket_ids": ", ".join(dict.fromkeys(matched_pocket_ids)),
                "matched_top1": bool(matched_rank and matched_rank <= 1),
                "matched_top3": bool(matched_rank and matched_rank <= 3),
                "matched_top5": bool(matched_rank and matched_rank <= 5),
                "expected_pocket_matched": bool(expected_pocket_id and expected_pocket_id in matched_pocket_ids),
                "benchmark_warning": "" if matched_pocket_ids else "reference-residue-not-covered-by-any-pocket",
            }
        )

    detail_df = pd.DataFrame(rows)
    for threshold in top_thresholds:
        column = f"matched_top{int(threshold)}"
        if column not in detail_df.columns:
            detail_df[column] = pd.to_numeric(detail_df["matched_rank"], errors="coerce").fillna(0).between(1, int(threshold))
    return detail_df[BENCHMARK_DETAIL_COLUMNS]


def _summary_status(*, top_n: int, coverage_ratio: float, any_hit: bool, all_hit: bool, reference_count: int, pocket_count: int) -> tuple[str, str]:
    if reference_count <= 0:
        return "no-reference", "Upload curated catalytic residues to compute benchmark coverage."
    if pocket_count <= 0:
        return "no-pockets", "No pocket rows are available for benchmark comparison."
    if all_hit:
        return "topn-complete-hit", f"All reference residues are covered within Top-{top_n} pockets."
    if any_hit and top_n == 1:
        return "top1-partial-hit", "Top-1 covers at least one reference residue but misses part of the catalytic set."
    if any_hit:
        return "topn-partial-hit", f"Top-{top_n} covers some reference residues; inspect missed residues before claiming active-site precision."
    if coverage_ratio <= 0.0 and top_n == 1:
        return "top1-miss", "Top-1 does not cover the curated catalytic residues."
    return "topn-miss", f"Top-{top_n} does not cover the curated catalytic residues."


def build_pocket_benchmark_summary(
    reference_df: Optional[pd.DataFrame],
    pocket_df: Optional[pd.DataFrame],
    pocket_summary_df: Optional[pd.DataFrame] = None,
    *,
    top_ns: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    details = build_pocket_benchmark_details(reference_df, pocket_df, pocket_summary_df, top_thresholds=top_ns)
    references = _reference_rows(reference_df)
    pockets = _normalize_pocket_rows(pocket_df)
    ranked_ids = _ranked_pocket_ids(pockets, pocket_summary_df)
    reference_count = int(len(references))
    pocket_count = int(len(ranked_ids))
    if reference_count <= 0:
        return _empty_summary_df()

    rows: list[dict[str, object]] = []
    matched_rank_series = pd.to_numeric(details.get("matched_rank", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    best_rank_values = matched_rank_series[matched_rank_series > 0]
    best_rank = int(best_rank_values.min()) if not best_rank_values.empty else 0
    best_pocket_id = ""
    if best_rank:
        best_rows = details[matched_rank_series == best_rank]
        if not best_rows.empty:
            best_pocket_id = str(best_rows.iloc[0].get("matched_pocket_id") or "")

    for top_n in top_ns:
        top_n = max(1, int(top_n))
        matched_mask = matched_rank_series.between(1, top_n)
        matched_count = int(matched_mask.sum())
        coverage_ratio = round(float(matched_count) / float(reference_count), 3) if reference_count else 0.0
        any_hit = matched_count > 0
        all_hit = matched_count == reference_count and reference_count > 0
        matched_residues = ", ".join(details.loc[matched_mask, "residue_label"].astype(str).tolist()) if matched_count else "none"
        missed_residues = ", ".join(details.loc[~matched_mask, "residue_label"].astype(str).tolist()) if matched_count < reference_count else "none"
        top_pocket_id = ranked_ids[0] if ranked_ids else ""
        top_pocket_hit = bool((matched_rank_series == 1).any())
        status, warning = _summary_status(
            top_n=top_n,
            coverage_ratio=coverage_ratio,
            any_hit=any_hit,
            all_hit=all_hit,
            reference_count=reference_count,
            pocket_count=pocket_count,
        )
        rows.append(
            {
                "top_n": top_n,
                "reference_residue_count": reference_count,
                "matched_reference_count": matched_count,
                "coverage_ratio": coverage_ratio,
                "any_hit": any_hit,
                "all_hit": all_hit,
                "best_rank": best_rank,
                "best_pocket_id": best_pocket_id,
                "top_pocket_id": top_pocket_id,
                "top_pocket_hit": top_pocket_hit,
                "matched_residues": matched_residues,
                "missed_residues": missed_residues,
                "benchmark_status": status,
                "benchmark_warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_SUMMARY_COLUMNS)


def build_pocket_benchmark_interpretation_summary(
    benchmark_summary_df: Optional[pd.DataFrame],
    readiness_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach reference readiness to each Top-N benchmark coverage row."""

    if benchmark_summary_df is None or getattr(benchmark_summary_df, "empty", True):
        return _empty_interpretation_df()

    benchmark = benchmark_summary_df.copy()
    for column in BENCHMARK_SUMMARY_COLUMNS:
        if column not in benchmark.columns:
            benchmark[column] = 0 if column in {"top_n", "reference_residue_count", "matched_reference_count", "best_rank"} else ""

    readiness_status = "unknown"
    readiness_action = "Run benchmark reference readiness checks before using coverage as a precision claim."
    readiness_warning = "Benchmark reference readiness was not available."
    blocker_count = 0
    review_count = 0
    if readiness_summary_df is not None and not getattr(readiness_summary_df, "empty", True):
        row = readiness_summary_df.iloc[0]
        readiness_status = _safe_text(row.get("readiness_status")) or "unknown"
        readiness_action = _safe_text(row.get("recommended_action")) or readiness_action
        readiness_warning = _safe_text(row.get("readiness_warning")) or readiness_warning
        blocker_count = int(row.get("p0_p1_issue_count") or 0)
        review_count = int(row.get("p2_issue_count") or 0)

    rows: list[dict[str, object]] = []
    for _, benchmark_row in benchmark.iterrows():
        top_n = int(pd.to_numeric(pd.Series([benchmark_row.get("top_n")]), errors="coerce").fillna(0).iloc[0] or 0)
        coverage_ratio = round(float(pd.to_numeric(pd.Series([benchmark_row.get("coverage_ratio")]), errors="coerce").fillna(0.0).iloc[0]), 3)
        matched_count = int(pd.to_numeric(pd.Series([benchmark_row.get("matched_reference_count")]), errors="coerce").fillna(0).iloc[0])
        reference_count = int(pd.to_numeric(pd.Series([benchmark_row.get("reference_residue_count")]), errors="coerce").fillna(0).iloc[0])
        best_rank = int(pd.to_numeric(pd.Series([benchmark_row.get("best_rank")]), errors="coerce").fillna(0).iloc[0])
        benchmark_status = _safe_text(benchmark_row.get("benchmark_status"))

        if readiness_status in {"blocked", "no-reference"} or blocker_count > 0:
            claim_status = "blocked"
            claim_ready = False
            interpretation_label = f"Top-{top_n} coverage is not claimable until benchmark reference blockers are fixed."
            recommended_action = readiness_action
            warning = readiness_warning
        elif readiness_status == "review-needed" or review_count > 0:
            claim_status = "review-needed"
            claim_ready = False
            interpretation_label = f"Top-{top_n} coverage is reviewer-pending because benchmark reference assumptions need sign-off."
            recommended_action = readiness_action
            warning = readiness_warning
        elif readiness_status == "ready":
            claim_status = "claim-ready"
            claim_ready = True
            if reference_count > 0 and matched_count == reference_count:
                interpretation_label = f"Top-{top_n} has complete curated residue coverage."
                recommended_action = "Use this coverage row as supported active-site pocket evidence, while still reporting the benchmark dataset size."
                warning = "Reference readiness passed; coverage can be interpreted directly."
            elif matched_count > 0:
                interpretation_label = f"Top-{top_n} has partial curated residue coverage."
                recommended_action = "Report partial coverage and inspect missed residues before claiming complete active-site localization."
                warning = "Reference readiness passed; incomplete coverage likely reflects detection/ranking limits or pocket boundary choices."
            else:
                interpretation_label = f"Top-{top_n} misses the curated residues."
                recommended_action = "Treat this as a pocket detection or ranking miss and inspect candidate generation, evidence routes, and thresholds."
                warning = "Reference readiness passed; this miss is more likely to reflect model behavior than reference curation."
        else:
            claim_status = "readiness-unknown"
            claim_ready = False
            interpretation_label = f"Top-{top_n} coverage cannot be interpreted as a precision claim until readiness is available."
            recommended_action = readiness_action
            warning = readiness_warning

        rows.append(
            {
                "top_n": top_n,
                "reference_residue_count": reference_count,
                "matched_reference_count": matched_count,
                "coverage_ratio": coverage_ratio,
                "benchmark_status": benchmark_status,
                "readiness_status": readiness_status,
                "claim_status": claim_status,
                "claim_ready": bool(claim_ready),
                "best_rank": best_rank,
                "best_pocket_id": _safe_text(benchmark_row.get("best_pocket_id")),
                "interpretation_label": interpretation_label,
                "recommended_action": recommended_action,
                "interpretation_warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_INTERPRETATION_COLUMNS)


def build_pocket_benchmark_case_interpretation_summary(
    case_summary_df: Optional[pd.DataFrame],
    readiness_case_summary_df: Optional[pd.DataFrame] = None,
    *,
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Attach case-level readiness to each case-level Top-N benchmark coverage row."""

    if case_summary_df is None or getattr(case_summary_df, "empty", True) or "benchmark_id" not in case_summary_df.columns:
        return _empty_case_interpretation_df()

    case_summary = case_summary_df.copy()
    fallback_id = _safe_text(default_benchmark_id) or "current"
    case_summary["benchmark_id"] = case_summary["benchmark_id"].map(lambda value: _normalized_case_id(value, fallback_id))

    readiness_by_case: dict[str, pd.DataFrame] = {}
    if readiness_case_summary_df is not None and not getattr(readiness_case_summary_df, "empty", True) and "benchmark_id" in readiness_case_summary_df.columns:
        readiness = readiness_case_summary_df.copy()
        readiness["benchmark_id"] = readiness["benchmark_id"].map(lambda value: _normalized_case_id(value, fallback_id))
        for benchmark_id, group in readiness.groupby("benchmark_id", sort=False, dropna=False):
            readiness_by_case[_safe_text(benchmark_id)] = group.drop(columns=["benchmark_id"], errors="ignore").head(1).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for benchmark_id, group in case_summary.groupby("benchmark_id", sort=True, dropna=False):
        benchmark_id_text = _safe_text(benchmark_id) or fallback_id
        interpretation = build_pocket_benchmark_interpretation_summary(
            group.drop(columns=["benchmark_id"], errors="ignore"),
            readiness_by_case.get(benchmark_id_text),
        )
        for row in interpretation.to_dict(orient="records"):
            rows.append({"benchmark_id": benchmark_id_text, **row})

    if not rows:
        return _empty_case_interpretation_df()
    return pd.DataFrame(rows, columns=BENCHMARK_CASE_INTERPRETATION_COLUMNS)


def _claim_ready_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_text(value).strip().lower()
    return text in {"1", "true", "yes", "y", "ready", "claim-ready"}


def _normalized_claim_status(value: object) -> str:
    return re.sub(r"[\s_]+", "-", _safe_text(value).strip().lower())


def _case_matrix_status(
    has_claim_ready: bool,
    any_blocked: bool,
    any_review_needed: bool,
    any_readiness_unknown: bool,
) -> tuple[str, str]:
    if any_blocked:
        return "blocked", "Fix blocked Top-N case interpretation rows before using this case in dataset-level claims."
    if any_review_needed or any_readiness_unknown:
        return "review-needed", "Resolve reviewer-pending or unknown Top-N rows before using this case in dataset-level claims."
    if has_claim_ready:
        return "claim-ready", "Use the earliest claim-ready Top-N row as case-level benchmark support."
    return "no-claim-ready", "Inspect coverage misses, ranking thresholds or reference readiness before claiming this case."


def build_pocket_benchmark_case_interpretation_matrix(
    case_interpretation_df: Optional[pd.DataFrame],
    *,
    top_ns: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Pivot case-level Top-N interpretation rows into one row per benchmark case."""

    normalized_top_values: list[int] = []
    for value in top_ns or ():
        try:
            top_n_value = int(value)
        except (TypeError, ValueError):
            continue
        if top_n_value > 0:
            normalized_top_values.append(top_n_value)
    normalized_top_ns = tuple(sorted(set(normalized_top_values)))
    if (
        case_interpretation_df is None
        or getattr(case_interpretation_df, "empty", True)
        or "benchmark_id" not in case_interpretation_df.columns
        or "top_n" not in case_interpretation_df.columns
    ):
        return _empty_case_interpretation_matrix_df(normalized_top_ns or (1, 3, 5))

    working = case_interpretation_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_case_interpretation_matrix_df(normalized_top_ns or (1, 3, 5))

    working["top_n"] = working["top_n"].astype(int)
    if not normalized_top_ns:
        normalized_top_ns = tuple(sorted({int(value) for value in working["top_n"].dropna().tolist() if int(value) > 0}))
    if not normalized_top_ns:
        normalized_top_ns = (1, 3, 5)

    working = working[working["top_n"].isin(normalized_top_ns)].copy()
    if working.empty:
        return _empty_case_interpretation_matrix_df(normalized_top_ns)

    working["benchmark_id"] = working["benchmark_id"].map(lambda value: _safe_text(value) or "current")
    working["claim_status"] = working["claim_status"].map(_normalized_claim_status)
    working.loc[working["claim_status"].eq(""), "claim_status"] = "readiness-unknown"
    working["claim_ready"] = working["claim_ready"].map(_claim_ready_bool)
    working.loc[working["claim_ready"] & working["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    for benchmark_id, group in working.groupby("benchmark_id", sort=True, dropna=False):
        group = group.sort_values("top_n")
        claim_ready_rows = group[group["claim_status"].eq("claim-ready")]
        best_claim = claim_ready_rows.iloc[0] if not claim_ready_rows.empty else None
        any_blocked = bool(group["claim_status"].eq("blocked").any())
        any_review_needed = bool(group["claim_status"].eq("review-needed").any())
        any_readiness_unknown = bool(group["claim_status"].eq("readiness-unknown").any())
        case_status, action = _case_matrix_status(
            best_claim is not None,
            any_blocked,
            any_review_needed,
            any_readiness_unknown,
        )
        row: dict[str, object] = {
            "benchmark_id": _safe_text(benchmark_id) or "current",
            "top_n_count": int(group["top_n"].nunique()),
            "best_claim_ready_top_n": int(best_claim.get("top_n") or 0) if best_claim is not None else 0,
            "best_claim_ready_coverage": round(float(best_claim.get("coverage_ratio") or 0.0), 3) if best_claim is not None else 0.0,
            "best_claim_ready_rank": int(best_claim.get("best_rank") or 0) if best_claim is not None else 0,
            "any_blocked": any_blocked,
            "any_review_needed": any_review_needed,
            "any_readiness_unknown": any_readiness_unknown,
            "case_interpretation_status": case_status,
            "recommended_action": action,
        }
        for top_n in normalized_top_ns:
            prefix = f"top{int(top_n)}"
            top_group = group[group["top_n"].eq(int(top_n))]
            top_row = top_group.iloc[0] if not top_group.empty else None
            row[f"{prefix}_claim_status"] = _safe_text(top_row.get("claim_status")) if top_row is not None else ""
            row[f"{prefix}_claim_ready"] = bool(top_row.get("claim_ready")) if top_row is not None else False
            row[f"{prefix}_coverage_ratio"] = round(float(top_row.get("coverage_ratio") or 0.0), 3) if top_row is not None else 0.0
            row[f"{prefix}_best_rank"] = int(top_row.get("best_rank") or 0) if top_row is not None else 0
            row[f"{prefix}_best_pocket_id"] = _safe_text(top_row.get("best_pocket_id")) if top_row is not None else ""
            row[f"{prefix}_benchmark_status"] = _safe_text(top_row.get("benchmark_status")) if top_row is not None else ""
        rows.append(row)

    if not rows:
        return _empty_case_interpretation_matrix_df(normalized_top_ns)
    return pd.DataFrame(rows, columns=_case_interpretation_matrix_columns(normalized_top_ns))


def _case_matrix_summary_status(
    case_count: int,
    blocked_count: int,
    review_count: int,
    unknown_count: int,
    no_claim_ready_count: int,
    usable_claim_ready_count: int,
) -> tuple[str, str, str]:
    if case_count <= 0:
        return (
            "no-cases",
            "Generate case interpretation rows before summarizing benchmark case readiness.",
            "No benchmark cases are available in the interpretation matrix.",
        )
    if blocked_count > 0:
        return (
            "blocked",
            "Fix blocked case interpretation rows before reporting dataset-level benchmark claims.",
            "One or more benchmark cases are blocked by readiness or reference issues.",
        )
    if review_count > 0 or unknown_count > 0:
        return (
            "review-needed",
            "Resolve reviewer-pending or unknown cases before publishing dataset-level benchmark claims.",
            "One or more benchmark cases still require review.",
        )
    if no_claim_ready_count > 0:
        return (
            "coverage-review",
            "Inspect no-claim-ready cases for missed catalytic residues, ranking thresholds or reference issues.",
            "Some benchmark cases have no claim-ready Top-N interpretation row.",
        )
    if usable_claim_ready_count == case_count:
        return (
            "claim-ready",
            "Report usable claim-ready case counts and earliest Top-N distribution.",
            "All benchmark cases are usable for readiness-aware benchmark claims.",
        )
    return (
        "review-needed",
        "Review unresolved case interpretation states before reporting benchmark claims.",
        "Case interpretation matrix summary could not resolve all cases as claim-ready.",
    )


def build_pocket_benchmark_case_interpretation_matrix_summary(matrix_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize the case interpretation matrix into dataset-level case readiness counts."""

    if matrix_df is None or getattr(matrix_df, "empty", True):
        return _empty_case_interpretation_matrix_summary_df()

    working = matrix_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_MATRIX_BASE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["case_interpretation_status"] = working["case_interpretation_status"].map(_normalized_claim_status)
    working.loc[working["case_interpretation_status"].eq(""), "case_interpretation_status"] = "readiness-unknown"
    for column in ("best_claim_ready_top_n", "best_claim_ready_rank"):
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int)
    working["best_claim_ready_coverage"] = pd.to_numeric(working["best_claim_ready_coverage"], errors="coerce").fillna(0.0)
    if "any_readiness_unknown" not in working.columns:
        working["any_readiness_unknown"] = False
    working["any_readiness_unknown"] = working["any_readiness_unknown"].map(_claim_ready_bool)

    case_count = int(len(working))
    usable_claim_ready = working[working["case_interpretation_status"].eq("claim-ready")]
    usable_claim_ready_count = int(len(usable_claim_ready))
    blocked_count = int(working["case_interpretation_status"].eq("blocked").sum())
    review_count = int(working["case_interpretation_status"].eq("review-needed").sum())
    unknown_count = int(working["any_readiness_unknown"].sum())
    no_claim_ready_count = int(working["case_interpretation_status"].eq("no-claim-ready").sum())
    summary_status, action, warning = _case_matrix_summary_status(
        case_count,
        blocked_count,
        review_count,
        unknown_count,
        no_claim_ready_count,
        usable_claim_ready_count,
    )
    usable_ranks = usable_claim_ready.loc[usable_claim_ready["best_claim_ready_rank"] > 0, "best_claim_ready_rank"]

    row = {
        "case_count": case_count,
        "usable_claim_ready_case_count": usable_claim_ready_count,
        "blocked_case_count": blocked_count,
        "review_case_count": review_count,
        "readiness_unknown_case_count": unknown_count,
        "no_claim_ready_case_count": no_claim_ready_count,
        "earliest_top1_claim_ready_case_count": int(usable_claim_ready["best_claim_ready_top_n"].eq(1).sum()) if not usable_claim_ready.empty else 0,
        "earliest_top3_claim_ready_case_count": int(usable_claim_ready["best_claim_ready_top_n"].eq(3).sum()) if not usable_claim_ready.empty else 0,
        "earliest_top5_claim_ready_case_count": int(usable_claim_ready["best_claim_ready_top_n"].eq(5).sum()) if not usable_claim_ready.empty else 0,
        "mean_usable_claim_ready_coverage": round(float(usable_claim_ready["best_claim_ready_coverage"].mean()), 3) if not usable_claim_ready.empty else 0.0,
        "mean_usable_claim_ready_rank": round(float(usable_ranks.mean()), 3) if not usable_ranks.empty else 0.0,
        "summary_status": summary_status,
        "recommended_action": action,
        "summary_warning": warning,
    }
    return pd.DataFrame([row], columns=BENCHMARK_CASE_INTERPRETATION_MATRIX_SUMMARY_COLUMNS)


def _case_matrix_queue_issue(case_status: str) -> tuple[str, str, str, str]:
    if case_status == "blocked":
        return (
            "P0",
            "blocker",
            "blocked-case",
            "Fix readiness or reference blockers for this benchmark case before using it in dataset-level claims.",
        )
    if case_status == "no-claim-ready":
        return (
            "P1",
            "review",
            "no-claim-ready-case",
            "Inspect missed catalytic residues, ranking thresholds and pocket boundary choices for this benchmark case.",
        )
    if case_status == "review-needed":
        return (
            "P2",
            "review",
            "review-needed-case",
            "Complete reviewer sign-off for this benchmark case before publishing dataset-level claims.",
        )
    if case_status == "readiness-unknown":
        return (
            "P2",
            "review",
            "readiness-unknown-case",
            "Generate or repair readiness evidence for this benchmark case before reporting benchmark accuracy.",
        )
    return "", "", "", ""


def build_pocket_benchmark_case_interpretation_matrix_queue(matrix_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Turn case matrix statuses into a one-row-per-case triage queue."""

    if matrix_df is None or getattr(matrix_df, "empty", True):
        return _empty_case_interpretation_matrix_queue_df()

    working = matrix_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_MATRIX_BASE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    for column in ("top1_claim_status", "top3_claim_status", "top5_claim_status"):
        if column not in working.columns:
            working[column] = ""
    working["case_interpretation_status"] = working["case_interpretation_status"].map(_normalized_claim_status)
    working.loc[working["case_interpretation_status"].eq(""), "case_interpretation_status"] = "readiness-unknown"
    for column in ("best_claim_ready_top_n", "best_claim_ready_rank"):
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int)
    working["best_claim_ready_coverage"] = pd.to_numeric(working["best_claim_ready_coverage"], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        case_status = _safe_text(row.get("case_interpretation_status"))
        priority, action_status, issue_type, fallback_action = _case_matrix_queue_issue(case_status)
        if not priority:
            continue
        rows.append(
            {
                "priority": priority,
                "action_status": action_status,
                "benchmark_id": _safe_text(row.get("benchmark_id")) or "current",
                "case_interpretation_status": case_status,
                "best_claim_ready_top_n": int(row.get("best_claim_ready_top_n") or 0),
                "best_claim_ready_coverage": round(float(row.get("best_claim_ready_coverage") or 0.0), 3),
                "best_claim_ready_rank": int(row.get("best_claim_ready_rank") or 0),
                "top1_claim_status": _safe_text(row.get("top1_claim_status")),
                "top3_claim_status": _safe_text(row.get("top3_claim_status")),
                "top5_claim_status": _safe_text(row.get("top5_claim_status")),
                "issue_type": issue_type,
                "suggested_action": _safe_text(row.get("recommended_action")) or fallback_action,
                "case_warning": fallback_action,
            }
        )

    if not rows:
        return _empty_case_interpretation_matrix_queue_df()

    frame = pd.DataFrame(rows)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"]).reset_index(drop=True)
    frame["action_id"] = [f"BCMQ-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_CASE_INTERPRETATION_MATRIX_QUEUE_COLUMNS]


def _dataset_interpretation_status(
    case_count: int,
    claim_ready_count: int,
    blocked_count: int,
    review_count: int,
    unknown_count: int,
    top_n: int,
) -> tuple[str, str, str, str]:
    if case_count <= 0:
        return (
            "no-cases",
            f"Top-{top_n} dataset interpretation has no benchmark cases.",
            "Add benchmark cases before reporting dataset-level catalytic pocket coverage.",
            "No benchmark cases are available for dataset-level interpretation.",
        )
    if blocked_count > 0:
        return (
            "blocked",
            f"Top-{top_n} dataset interpretation is blocked because one or more benchmark cases are not claim-ready.",
            "Fix blocked cases before using dataset-level coverage as a precision claim.",
            "At least one benchmark case is blocked; dataset-level coverage is not claimable.",
        )
    if review_count > 0 or unknown_count > 0:
        return (
            "review-needed",
            f"Top-{top_n} dataset interpretation needs review because some cases are reviewer-pending or unknown.",
            "Review non-claim-ready cases before publishing dataset-level coverage.",
            "Some benchmark cases are not claim-ready; dataset-level coverage needs reviewer interpretation.",
        )
    if claim_ready_count == case_count:
        return (
            "claim-ready",
            f"Top-{top_n} dataset interpretation is claim-ready across all benchmark cases.",
            "Report dataset coverage with case count and mean claim-ready coverage.",
            "All benchmark cases are claim-ready for this Top-N dataset interpretation.",
        )
    return (
        "review-needed",
        f"Top-{top_n} dataset interpretation needs review because not all cases are claim-ready.",
        "Review non-claim-ready cases before publishing dataset-level coverage.",
        "Dataset-level readiness could not be fully resolved from case-level interpretation.",
    )


def build_pocket_benchmark_dataset_interpretation(case_interpretation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate case-level benchmark interpretations into dataset-level claim readiness."""

    if case_interpretation_df is None or getattr(case_interpretation_df, "empty", True) or "top_n" not in case_interpretation_df.columns:
        return _empty_dataset_interpretation_df()

    working = case_interpretation_df.copy()
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_dataset_interpretation_df()

    if "coverage_ratio" not in working.columns:
        working["coverage_ratio"] = 0.0
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)

    if "claim_ready" not in working.columns:
        working["claim_ready"] = False
    working["claim_ready"] = working["claim_ready"].map(_claim_ready_bool)

    if "claim_status" not in working.columns:
        working["claim_status"] = ""
    working["claim_status"] = working.apply(
        lambda row: _normalized_claim_status(row.get("claim_status"))
        or ("claim-ready" if bool(row.get("claim_ready")) else "readiness-unknown"),
        axis=1,
    )
    known_statuses = {"claim-ready", "blocked", "review-needed", "readiness-unknown"}
    working.loc[~working["claim_status"].isin(known_statuses), "claim_status"] = "readiness-unknown"
    working.loc[working["claim_ready"] & working["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"

    if "benchmark_id" not in working.columns:
        working["benchmark_id"] = [f"case-{index}" for index in working.index]
    else:
        working["benchmark_id"] = [
            _safe_text(value).strip() or f"case-{index}"
            for index, value in zip(working.index, working["benchmark_id"])
        ]

    rows: list[dict[str, object]] = []
    for top_n, group in working.groupby("top_n", sort=True, dropna=False):
        top_n_int = int(top_n)
        case_group = group.drop_duplicates(subset=["benchmark_id"], keep="first").copy()
        case_count = int(len(case_group))
        claim_ready_count = int(case_group["claim_status"].eq("claim-ready").sum())
        blocked_count = int(case_group["claim_status"].eq("blocked").sum())
        review_count = int(case_group["claim_status"].eq("review-needed").sum())
        unknown_count = int(case_group["claim_status"].eq("readiness-unknown").sum())
        claim_ready_rows = case_group[case_group["claim_status"].eq("claim-ready")]
        dataset_status, label, action, warning = _dataset_interpretation_status(
            case_count,
            claim_ready_count,
            blocked_count,
            review_count,
            unknown_count,
            top_n_int,
        )

        rows.append(
            {
                "top_n": top_n_int,
                "case_count": case_count,
                "claim_ready_case_count": claim_ready_count,
                "blocked_case_count": blocked_count,
                "review_case_count": review_count,
                "unknown_case_count": unknown_count,
                "mean_claim_ready_coverage": round(float(claim_ready_rows["coverage_ratio"].mean()), 3) if not claim_ready_rows.empty else 0.0,
                "mean_all_case_coverage": round(float(case_group["coverage_ratio"].mean()), 3) if case_count else 0.0,
                "claim_ready_rate": round(float(claim_ready_count) / float(case_count), 3) if case_count else 0.0,
                "blocked_case_rate": round(float(blocked_count) / float(case_count), 3) if case_count else 0.0,
                "review_case_rate": round(float(review_count) / float(case_count), 3) if case_count else 0.0,
                "dataset_claim_status": dataset_status,
                "interpretation_label": label,
                "recommended_action": action,
                "interpretation_warning": warning,
            }
        )

    if not rows:
        return _empty_dataset_interpretation_df()
    return pd.DataFrame(rows, columns=BENCHMARK_DATASET_INTERPRETATION_COLUMNS)


def _dataset_interpretation_queue_issue(claim_status: str) -> tuple[str, str, str]:
    if claim_status == "blocked":
        return "P0", "blocker", "blocked-case"
    if claim_status == "review-needed":
        return "P2", "review", "review-needed-case"
    if claim_status == "readiness-unknown":
        return "P2", "review", "readiness-unknown-case"
    return "", "", ""


def build_pocket_benchmark_dataset_interpretation_queue(case_interpretation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """List non-claim-ready benchmark cases that block or weaken dataset-level interpretation."""

    if case_interpretation_df is None or getattr(case_interpretation_df, "empty", True) or "top_n" not in case_interpretation_df.columns:
        return _empty_dataset_interpretation_queue_df()

    working = case_interpretation_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_dataset_interpretation_queue_df()

    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)
    working["claim_status"] = working["claim_status"].map(_normalized_claim_status)
    working.loc[working["claim_status"].eq(""), "claim_status"] = "readiness-unknown"
    if "claim_ready" in working.columns:
        working["claim_ready"] = working["claim_ready"].map(_claim_ready_bool)
        working.loc[working["claim_ready"] & working["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        claim_status = _safe_text(row.get("claim_status"))
        priority, action_status, issue_type = _dataset_interpretation_queue_issue(claim_status)
        if not priority:
            continue
        rows.append(
            {
                "priority": priority,
                "action_status": action_status,
                "top_n": int(row.get("top_n") or 0),
                "benchmark_id": _safe_text(row.get("benchmark_id")) or "current",
                "claim_status": claim_status,
                "coverage_ratio": round(float(row.get("coverage_ratio") or 0.0), 3),
                "best_rank": int(row.get("best_rank") or 0),
                "best_pocket_id": _safe_text(row.get("best_pocket_id")),
                "benchmark_status": _safe_text(row.get("benchmark_status")),
                "readiness_status": _safe_text(row.get("readiness_status")),
                "issue_type": issue_type,
                "suggested_action": _safe_text(row.get("recommended_action")),
                "interpretation_warning": _safe_text(row.get("interpretation_warning")),
            }
        )

    if not rows:
        return _empty_dataset_interpretation_queue_df()

    frame = pd.DataFrame(rows)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["top_n", "_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"]).reset_index(drop=True)
    frame["action_id"] = [f"BDSI-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS]


def build_pocket_benchmark_dataset_interpretation_checklist_markdown(
    dataset_interpretation_queue_df: Optional[pd.DataFrame],
    *,
    title: str = "Benchmark dataset interpretation checklist",
    max_actions: int = 80,
) -> str:
    """Render dataset-level benchmark interpretation blockers as a reviewer checklist."""

    if dataset_interpretation_queue_df is None or getattr(dataset_interpretation_queue_df, "empty", True):
        return ""

    working = dataset_interpretation_queue_df.copy()
    for column in BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce").fillna(0).astype(int)
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    working["_priority_rank"] = working["priority"].map(priority_rank).fillna(99)
    working = working.sort_values(["top_n", "_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"])

    summary = (
        working.groupby(["priority", "action_status", "issue_type"], sort=False, dropna=False)
        .agg(action_count=("action_id", "count"), affected_case_count=("benchmark_id", "nunique"))
        .reset_index()
    )
    summary["_priority_rank"] = summary["priority"].map(priority_rank).fillna(99)
    summary = summary.sort_values(["_priority_rank", "issue_type"]).drop(columns=["_priority_rank"])

    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_dataset_interpretation_queue.csv`.",
        "",
        "Use this checklist before reporting dataset-level benchmark coverage as a precision claim.",
        "",
        "## Summary",
        "",
        "| Priority | Status | Issue | Actions | Cases |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {priority} | {status} | {issue} | {actions} | {cases} |".format(
                priority=_safe_text(row.get("priority")) or "-",
                status=_safe_text(row.get("action_status")) or "-",
                issue=_safe_text(row.get("issue_type")) or "-",
                actions=int(row.get("action_count") or 0),
                cases=int(row.get("affected_case_count") or 0),
            )
        )

    lines.extend(["", "## Actions", ""])
    for _, row in working.head(max_actions).iterrows():
        lines.append(
            "- [ ] `{priority}` Top-{top_n} `{issue}` case `{case}` coverage `{coverage}` best rank `{best_rank}` pocket `{pocket}`: {action}".format(
                priority=_safe_text(row.get("priority")) or "-",
                top_n=int(row.get("top_n") or 0),
                issue=_safe_text(row.get("issue_type")) or "-",
                case=_safe_text(row.get("benchmark_id")) or "current",
                coverage=round(float(row.get("coverage_ratio") or 0.0), 3),
                best_rank=int(row.get("best_rank") or 0),
                pocket=_safe_text(row.get("best_pocket_id")) or "-",
                action=_safe_text(row.get("suggested_action")) or "Review this benchmark case before reporting dataset-level coverage.",
            )
        )

    remaining = int(len(working) - max_actions)
    if remaining > 0:
        lines.append(f"- [ ] Review {remaining} additional queued actions in the CSV export.")
    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_dataset_interpretation_report_markdown(
    dataset_interpretation_df: Optional[pd.DataFrame],
    dataset_interpretation_queue_df: Optional[pd.DataFrame] = None,
    *,
    checklist_available: bool = False,
    title: str = "Benchmark dataset interpretation report",
    max_queue_actions: int = 20,
) -> str:
    """Render a compact report for dataset-level benchmark claim readiness."""

    has_interpretation = dataset_interpretation_df is not None and not getattr(dataset_interpretation_df, "empty", True)
    has_queue = dataset_interpretation_queue_df is not None and not getattr(dataset_interpretation_queue_df, "empty", True)
    if not has_interpretation and not has_queue:
        return ""

    interpretation = dataset_interpretation_df.copy() if has_interpretation else _empty_dataset_interpretation_df()
    for column in BENCHMARK_DATASET_INTERPRETATION_COLUMNS:
        if column not in interpretation.columns:
            interpretation[column] = ""
    if not interpretation.empty:
        interpretation["top_n"] = pd.to_numeric(interpretation["top_n"], errors="coerce").fillna(0).astype(int)
        for column in (
            "case_count",
            "claim_ready_case_count",
            "blocked_case_count",
            "review_case_count",
            "unknown_case_count",
        ):
            interpretation[column] = pd.to_numeric(interpretation[column], errors="coerce").fillna(0).astype(int)
        for column in ("mean_claim_ready_coverage", "mean_all_case_coverage", "claim_ready_rate"):
            interpretation[column] = pd.to_numeric(interpretation[column], errors="coerce").fillna(0.0)
        interpretation = interpretation.sort_values("top_n")

    queue = dataset_interpretation_queue_df.copy() if has_queue else _empty_dataset_interpretation_queue_df()
    for column in BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    if not queue.empty:
        queue["top_n"] = pd.to_numeric(queue["top_n"], errors="coerce").fillna(0).astype(int)
        queue["coverage_ratio"] = pd.to_numeric(queue["coverage_ratio"], errors="coerce").fillna(0.0)
        queue["best_rank"] = pd.to_numeric(queue["best_rank"], errors="coerce").fillna(0).astype(int)
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        queue["_priority_rank"] = queue["priority"].map(priority_rank).fillna(99)
        queue = queue.sort_values(["top_n", "_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"])

    blocked_rows = int(interpretation["dataset_claim_status"].astype(str).eq("blocked").sum()) if not interpretation.empty else 0
    review_rows = int(interpretation["dataset_claim_status"].astype(str).eq("review-needed").sum()) if not interpretation.empty else 0
    claim_ready_rows = int(interpretation["dataset_claim_status"].astype(str).eq("claim-ready").sum()) if not interpretation.empty else 0
    report_status = "blocked" if blocked_rows else "review-needed" if review_rows else "claim-ready" if claim_ready_rows else "no-dataset-interpretation"
    next_action = (
        "Fix P0 blocker cases before reporting dataset-level coverage."
        if blocked_rows
        else "Complete reviewer sign-off before publishing dataset-level coverage."
        if review_rows
        else "Report dataset coverage with case counts and Top-N status."
        if claim_ready_rows
        else "Generate dataset interpretation before reporting benchmark accuracy."
    )

    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_dataset_interpretation.csv` and `pocket_benchmark_dataset_interpretation_queue.csv`.",
        "",
        "## Gate",
        "",
        f"- Dataset claim status: `{report_status}`.",
        f"- Dataset interpretation rows: {int(len(interpretation))}.",
        f"- Queued case actions: {int(len(queue))}.",
        f"- Checklist: {'available' if checklist_available else 'not available'}.",
        f"- Recommended action: {next_action}",
        "",
        "## Top-N Interpretation",
        "",
    ]
    if interpretation.empty:
        lines.append("No dataset interpretation rows are available.")
    else:
        lines.append("| Top-N | Status | Cases | Claim-ready | Blocked | Review | Unknown | Mean claim-ready coverage | Mean all-case coverage |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in interpretation.iterrows():
            lines.append(
                "| {top_n} | {status} | {cases} | {ready} | {blocked} | {review} | {unknown} | {ready_cov:.3f} | {all_cov:.3f} |".format(
                    top_n=int(row.get("top_n") or 0),
                    status=_safe_text(row.get("dataset_claim_status")) or "-",
                    cases=int(row.get("case_count") or 0),
                    ready=int(row.get("claim_ready_case_count") or 0),
                    blocked=int(row.get("blocked_case_count") or 0),
                    review=int(row.get("review_case_count") or 0),
                    unknown=int(row.get("unknown_case_count") or 0),
                    ready_cov=float(row.get("mean_claim_ready_coverage") or 0.0),
                    all_cov=float(row.get("mean_all_case_coverage") or 0.0),
                )
            )

    lines.extend(["", "## Queued Actions", ""])
    if queue.empty:
        lines.append("No non-claim-ready benchmark cases are queued.")
    else:
        lines.append("| Priority | Top-N | Case | Issue | Coverage | Best rank | Action |")
        lines.append("| --- | ---: | --- | --- | ---: | ---: | --- |")
        for _, row in queue.head(max_queue_actions).iterrows():
            lines.append(
                "| {priority} | {top_n} | {case} | {issue} | {coverage:.3f} | {best_rank} | {action} |".format(
                    priority=_safe_text(row.get("priority")) or "-",
                    top_n=int(row.get("top_n") or 0),
                    case=_safe_text(row.get("benchmark_id")) or "current",
                    issue=_safe_text(row.get("issue_type")) or "-",
                    coverage=float(row.get("coverage_ratio") or 0.0),
                    best_rank=int(row.get("best_rank") or 0),
                    action=_safe_text(row.get("suggested_action")) or "Review this benchmark case.",
                )
            )
        remaining = int(len(queue) - max_queue_actions)
        if remaining > 0:
            lines.append(f"\n{remaining} additional queued actions are available in the CSV export.")

    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_case_summary(
    reference_df: Optional[pd.DataFrame],
    pocket_df: Optional[pd.DataFrame],
    pocket_summary_df: Optional[pd.DataFrame] = None,
    *,
    top_ns: Sequence[int] = (1, 3),
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Build Top-N coverage summaries per benchmark case.

    A curated dataset can contain multiple `benchmark_id`/`case_id` values.
    Keeping case rows separate prevents enzymes with many curated residues from
    dominating the reported accuracy.
    """

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_case_summary_df()

    working = references.copy()
    fallback_id = str(default_benchmark_id or "current").strip() or "current"
    working["benchmark_id"] = working["benchmark_id"].astype(str).str.strip().replace("", fallback_id)

    rows: list[dict[str, object]] = []
    for benchmark_id, case_reference_df in working.groupby("benchmark_id", sort=True, dropna=False):
        case_pocket_df = _filter_rows_for_benchmark_id(pocket_df, str(benchmark_id or fallback_id))
        case_pocket_summary_df = _filter_rows_for_benchmark_id(pocket_summary_df, str(benchmark_id or fallback_id))
        summary = build_pocket_benchmark_summary(
            case_reference_df,
            case_pocket_df,
            case_pocket_summary_df,
            top_ns=top_ns,
        )
        for row in summary.to_dict(orient="records"):
            rows.append({"benchmark_id": str(benchmark_id or fallback_id), **row})

    if not rows:
        return _empty_case_summary_df()
    return pd.DataFrame(rows, columns=BENCHMARK_CASE_SUMMARY_COLUMNS)


def _dataset_summary_status(case_count: int, any_hit_count: int, all_hit_count: int) -> tuple[str, str]:
    if case_count <= 0:
        return "no-cases", "No benchmark cases are available."
    if all_hit_count == case_count:
        return "all-cases-complete-hit", "All benchmark cases have complete catalytic residue coverage."
    if any_hit_count == case_count:
        return "all-cases-any-hit", "Every benchmark case has at least one catalytic residue covered."
    if any_hit_count > 0:
        return "mixed-hit", "Some benchmark cases are hit while others miss; inspect case_summary before claiming dataset accuracy."
    return "all-cases-miss", "No benchmark case has catalytic residue coverage."


def build_pocket_benchmark_dataset_summary(case_summary_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-case benchmark coverage into dataset-level metrics."""

    if case_summary_df is None or getattr(case_summary_df, "empty", True) or "top_n" not in case_summary_df.columns:
        return _empty_dataset_summary_df()

    working = case_summary_df.copy()
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_dataset_summary_df()

    for column in ["reference_residue_count", "matched_reference_count", "best_rank"]:
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    if "coverage_ratio" not in working.columns:
        working["coverage_ratio"] = 0.0
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    if "any_hit" not in working.columns:
        working["any_hit"] = False
    if "all_hit" not in working.columns:
        working["all_hit"] = False
    working["any_hit"] = working["any_hit"].astype(bool)
    working["all_hit"] = working["all_hit"].astype(bool)

    rows: list[dict[str, object]] = []
    for top_n, group in working.groupby("top_n", sort=True, dropna=False):
        case_count = int(len(group))
        any_hit_count = int(group["any_hit"].sum())
        all_hit_count = int(group["all_hit"].sum())
        best_ranks = group.loc[group["best_rank"] > 0, "best_rank"]
        status, warning = _dataset_summary_status(case_count, any_hit_count, all_hit_count)
        rows.append(
            {
                "top_n": int(top_n),
                "case_count": case_count,
                "reference_residue_count": int(group["reference_residue_count"].sum()),
                "matched_reference_count": int(group["matched_reference_count"].sum()),
                "mean_coverage_ratio": round(float(group["coverage_ratio"].mean()), 3) if case_count else 0.0,
                "median_coverage_ratio": round(float(group["coverage_ratio"].median()), 3) if case_count else 0.0,
                "min_coverage_ratio": round(float(group["coverage_ratio"].min()), 3) if case_count else 0.0,
                "max_coverage_ratio": round(float(group["coverage_ratio"].max()), 3) if case_count else 0.0,
                "any_hit_case_count": any_hit_count,
                "all_hit_case_count": all_hit_count,
                "miss_case_count": int(case_count - any_hit_count),
                "any_hit_rate": round(float(any_hit_count) / float(case_count), 3) if case_count else 0.0,
                "all_hit_rate": round(float(all_hit_count) / float(case_count), 3) if case_count else 0.0,
                "mean_best_rank": round(float(best_ranks.mean()), 3) if not best_ranks.empty else 0.0,
                "benchmark_status": status,
                "benchmark_warning": warning,
            }
        )

    if not rows:
        return _empty_dataset_summary_df()
    return pd.DataFrame(rows, columns=BENCHMARK_DATASET_SUMMARY_COLUMNS)


def build_pocket_benchmark_variant_comparison(
    reference_df: Optional[pd.DataFrame],
    variants: Sequence[tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    *,
    reference_variant_label: str = "current",
    top_ns: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    """Compare catalytic coverage across pocket-ranking variants.

    `variants` contains `(label, pocket_df, pocket_summary_df)` tuples. The
    reference variant is usually the active/current run; ablated variants then
    show coverage loss when literature, evidence-route or conservation support
    is removed.
    """

    references = _reference_rows(reference_df)
    if references.empty or not variants:
        return _empty_variant_comparison_df()

    summaries: dict[str, pd.DataFrame] = {}
    labels: list[str] = []
    for label, pocket_df, pocket_summary_df in variants:
        cleaned_label = str(label or "").strip()
        if not cleaned_label or cleaned_label in summaries:
            continue
        summary = build_pocket_benchmark_summary(
            references,
            pocket_df,
            pocket_summary_df,
            top_ns=top_ns,
        )
        if summary.empty:
            continue
        summaries[cleaned_label] = summary
        labels.append(cleaned_label)

    if not summaries:
        return _empty_variant_comparison_df()

    reference_label = str(reference_variant_label or "").strip()
    if reference_label not in summaries:
        reference_label = labels[0]
    reference_summary = summaries[reference_label].copy()
    reference_by_top_n = {
        int(row.top_n): row
        for row in reference_summary.itertuples(index=False)
        if pd.notna(getattr(row, "top_n", None))
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        summary = summaries[label]
        for row in summary.itertuples(index=False):
            top_n = int(getattr(row, "top_n"))
            reference_row = reference_by_top_n.get(top_n)
            reference_coverage = float(getattr(reference_row, "coverage_ratio", 0.0)) if reference_row is not None else 0.0
            coverage = float(getattr(row, "coverage_ratio", 0.0))
            reference_best_rank = int(getattr(reference_row, "best_rank", 0) or 0) if reference_row is not None else 0
            best_rank = int(getattr(row, "best_rank", 0) or 0)
            if best_rank <= 0 or reference_best_rank <= 0:
                rank_delta = 0
            else:
                rank_delta = int(best_rank - reference_best_rank)
            rows.append(
                {
                    "variant_label": label,
                    "reference_variant_label": reference_label,
                    "top_n": top_n,
                    "reference_residue_count": int(getattr(row, "reference_residue_count", 0) or 0),
                    "matched_reference_count": int(getattr(row, "matched_reference_count", 0) or 0),
                    "coverage_ratio": coverage,
                    "reference_coverage_ratio": reference_coverage,
                    "coverage_delta_vs_reference": round(float(coverage - reference_coverage), 3),
                    "coverage_loss_vs_reference": round(float(reference_coverage - coverage), 3),
                    "best_rank": best_rank,
                    "reference_best_rank": reference_best_rank,
                    "best_rank_delta_vs_reference": rank_delta,
                    "best_pocket_id": str(getattr(row, "best_pocket_id", "") or ""),
                    "top_pocket_id": str(getattr(row, "top_pocket_id", "") or ""),
                    "top_pocket_hit": bool(getattr(row, "top_pocket_hit", False)),
                    "matched_residues": str(getattr(row, "matched_residues", "") or ""),
                    "missed_residues": str(getattr(row, "missed_residues", "") or ""),
                    "benchmark_status": str(getattr(row, "benchmark_status", "") or ""),
                    "benchmark_warning": str(getattr(row, "benchmark_warning", "") or ""),
                }
            )

    if not rows:
        return _empty_variant_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_COMPARISON_COLUMNS)


def build_pocket_benchmark_variant_case_comparison(
    reference_df: Optional[pd.DataFrame],
    variants: Sequence[tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    *,
    reference_variant_label: str = "current",
    top_ns: Sequence[int] = (1, 3),
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Compare variant coverage per benchmark case."""

    references = _reference_rows(reference_df)
    if references.empty or not variants:
        return _empty_variant_case_comparison_df()

    case_summaries: dict[str, pd.DataFrame] = {}
    labels: list[str] = []
    for label, pocket_df, pocket_summary_df in variants:
        cleaned_label = str(label or "").strip()
        if not cleaned_label or cleaned_label in case_summaries:
            continue
        summary = build_pocket_benchmark_case_summary(
            references,
            pocket_df,
            pocket_summary_df,
            top_ns=top_ns,
            default_benchmark_id=default_benchmark_id,
        )
        if summary.empty:
            continue
        case_summaries[cleaned_label] = summary
        labels.append(cleaned_label)

    if not case_summaries:
        return _empty_variant_case_comparison_df()

    reference_label = str(reference_variant_label or "").strip()
    if reference_label not in case_summaries:
        reference_label = labels[0]
    reference_summary = case_summaries[reference_label].copy()
    reference_by_case_top_n = {
        (str(row.benchmark_id), int(row.top_n)): row
        for row in reference_summary.itertuples(index=False)
        if pd.notna(getattr(row, "top_n", None))
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        summary = case_summaries[label]
        for row in summary.itertuples(index=False):
            benchmark_id = str(getattr(row, "benchmark_id", "") or "")
            top_n = int(getattr(row, "top_n"))
            reference_row = reference_by_case_top_n.get((benchmark_id, top_n))
            reference_coverage = float(getattr(reference_row, "coverage_ratio", 0.0)) if reference_row is not None else 0.0
            coverage = float(getattr(row, "coverage_ratio", 0.0))
            reference_best_rank = int(getattr(reference_row, "best_rank", 0) or 0) if reference_row is not None else 0
            best_rank = int(getattr(row, "best_rank", 0) or 0)
            rank_delta = int(best_rank - reference_best_rank) if best_rank > 0 and reference_best_rank > 0 else 0
            rows.append(
                {
                    "variant_label": label,
                    "reference_variant_label": reference_label,
                    "benchmark_id": benchmark_id,
                    "top_n": top_n,
                    "reference_residue_count": int(getattr(row, "reference_residue_count", 0) or 0),
                    "matched_reference_count": int(getattr(row, "matched_reference_count", 0) or 0),
                    "coverage_ratio": coverage,
                    "reference_coverage_ratio": reference_coverage,
                    "coverage_delta_vs_reference": round(float(coverage - reference_coverage), 3),
                    "coverage_loss_vs_reference": round(float(reference_coverage - coverage), 3),
                    "best_rank": best_rank,
                    "reference_best_rank": reference_best_rank,
                    "best_rank_delta_vs_reference": rank_delta,
                    "best_pocket_id": str(getattr(row, "best_pocket_id", "") or ""),
                    "top_pocket_id": str(getattr(row, "top_pocket_id", "") or ""),
                    "top_pocket_hit": bool(getattr(row, "top_pocket_hit", False)),
                    "matched_residues": str(getattr(row, "matched_residues", "") or ""),
                    "missed_residues": str(getattr(row, "missed_residues", "") or ""),
                    "benchmark_status": str(getattr(row, "benchmark_status", "") or ""),
                    "benchmark_warning": str(getattr(row, "benchmark_warning", "") or ""),
                }
            )

    if not rows:
        return _empty_variant_case_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_CASE_COMPARISON_COLUMNS)


def build_pocket_benchmark_variant_dataset_comparison(variant_case_comparison_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate case-level variant comparison into dataset-level deltas."""

    if (
        variant_case_comparison_df is None
        or getattr(variant_case_comparison_df, "empty", True)
        or "variant_label" not in variant_case_comparison_df.columns
        or "top_n" not in variant_case_comparison_df.columns
    ):
        return _empty_variant_dataset_comparison_df()

    working = variant_case_comparison_df.copy()
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_variant_dataset_comparison_df()

    for column in [
        "coverage_ratio",
        "reference_coverage_ratio",
        "coverage_delta_vs_reference",
        "coverage_loss_vs_reference",
        "best_rank",
        "reference_best_rank",
    ]:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    group_columns = ["variant_label", "reference_variant_label", "top_n"]
    for (variant_label, reference_label, top_n), group in working.groupby(group_columns, sort=True, dropna=False):
        case_count = int(len(group))
        any_hit_rate = round(float((group["coverage_ratio"] > 0.0).sum()) / float(case_count), 3) if case_count else 0.0
        reference_any_hit_rate = round(float((group["reference_coverage_ratio"] > 0.0).sum()) / float(case_count), 3) if case_count else 0.0
        all_hit_rate = round(float((group["coverage_ratio"] >= 1.0).sum()) / float(case_count), 3) if case_count else 0.0
        reference_all_hit_rate = round(float((group["reference_coverage_ratio"] >= 1.0).sum()) / float(case_count), 3) if case_count else 0.0
        best_ranks = group.loc[group["best_rank"] > 0, "best_rank"]
        reference_best_ranks = group.loc[group["reference_best_rank"] > 0, "reference_best_rank"]
        case_loss_count = int((group["coverage_loss_vs_reference"] > 0.0).sum())
        case_gain_count = int((group["coverage_delta_vs_reference"] > 0.0).sum())
        case_unchanged_count = int((group["coverage_delta_vs_reference"].abs() <= 0.0005).sum())
        mean_coverage = round(float(group["coverage_ratio"].mean()), 3) if case_count else 0.0
        reference_mean_coverage = round(float(group["reference_coverage_ratio"].mean()), 3) if case_count else 0.0
        mean_best_rank = round(float(best_ranks.mean()), 3) if not best_ranks.empty else 0.0
        reference_mean_best_rank = round(float(reference_best_ranks.mean()), 3) if not reference_best_ranks.empty else 0.0
        status, warning = _dataset_summary_status(
            case_count,
            int((group["coverage_ratio"] > 0.0).sum()),
            int((group["coverage_ratio"] >= 1.0).sum()),
        )
        if case_loss_count > 0:
            warning = f"{case_loss_count} benchmark cases lost catalytic coverage versus {reference_label}."
        rows.append(
            {
                "variant_label": str(variant_label or ""),
                "reference_variant_label": str(reference_label or ""),
                "top_n": int(top_n),
                "case_count": case_count,
                "mean_coverage_ratio": mean_coverage,
                "reference_mean_coverage_ratio": reference_mean_coverage,
                "mean_coverage_delta_vs_reference": round(float(mean_coverage - reference_mean_coverage), 3),
                "mean_coverage_loss_vs_reference": round(float(reference_mean_coverage - mean_coverage), 3),
                "any_hit_rate": any_hit_rate,
                "reference_any_hit_rate": reference_any_hit_rate,
                "any_hit_rate_delta_vs_reference": round(float(any_hit_rate - reference_any_hit_rate), 3),
                "all_hit_rate": all_hit_rate,
                "reference_all_hit_rate": reference_all_hit_rate,
                "all_hit_rate_delta_vs_reference": round(float(all_hit_rate - reference_all_hit_rate), 3),
                "mean_best_rank": mean_best_rank,
                "reference_mean_best_rank": reference_mean_best_rank,
                "mean_best_rank_delta_vs_reference": round(float(mean_best_rank - reference_mean_best_rank), 3)
                if mean_best_rank > 0 and reference_mean_best_rank > 0
                else 0.0,
                "case_loss_count": case_loss_count,
                "case_gain_count": case_gain_count,
                "case_unchanged_count": case_unchanged_count,
                "benchmark_status": status,
                "benchmark_warning": warning,
            }
        )

    if not rows:
        return _empty_variant_dataset_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_DATASET_COMPARISON_COLUMNS)


def _detail_key(row: pd.Series) -> tuple[str, str, int, str]:
    return (
        str(row.get("benchmark_id") or "").strip(),
        str(row.get("chain") or "").strip(),
        int(row.get("resid") or 0),
        str(row.get("reference_type") or "").strip(),
    )


def _match_delta(variant_matched: bool, reference_matched: bool) -> str:
    if variant_matched and reference_matched:
        return "unchanged-hit"
    if not variant_matched and not reference_matched:
        return "unchanged-miss"
    if reference_matched and not variant_matched:
        return "lost"
    return "gained"


def build_pocket_benchmark_variant_detail_comparison(
    reference_df: Optional[pd.DataFrame],
    variants: Sequence[tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    *,
    reference_variant_label: str = "current",
    top_thresholds: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Compare exact catalytic residue matches across ranking variants."""

    references = _reference_rows(reference_df)
    if references.empty or not variants:
        return _empty_variant_detail_comparison_df()

    details_by_label: dict[str, pd.DataFrame] = {}
    labels: list[str] = []
    for label, pocket_df, pocket_summary_df in variants:
        cleaned_label = str(label or "").strip()
        if not cleaned_label or cleaned_label in details_by_label:
            continue
        details = build_pocket_benchmark_details(
            references,
            pocket_df,
            pocket_summary_df,
            top_thresholds=top_thresholds,
        )
        if details.empty:
            continue
        details_by_label[cleaned_label] = details
        labels.append(cleaned_label)

    if not details_by_label:
        return _empty_variant_detail_comparison_df()

    reference_label = str(reference_variant_label or "").strip()
    if reference_label not in details_by_label:
        reference_label = labels[0]
    reference_details = details_by_label[reference_label]
    reference_by_key = {
        _detail_key(row): row
        for _, row in reference_details.iterrows()
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        details = details_by_label[label]
        for _, row in details.iterrows():
            key = _detail_key(row)
            reference_row = reference_by_key.get(key)
            variant_matched = bool(row.get("matched"))
            reference_matched = bool(reference_row.get("matched")) if reference_row is not None else False
            variant_rank = int(row.get("matched_rank") or 0)
            reference_rank = int(reference_row.get("matched_rank") or 0) if reference_row is not None else 0
            rank_delta = int(variant_rank - reference_rank) if variant_rank > 0 and reference_rank > 0 else 0
            delta = _match_delta(variant_matched, reference_matched)
            warning = str(row.get("benchmark_warning") or "")
            if delta == "lost":
                warning = "reference-residue-lost-vs-current"
            elif delta == "gained":
                warning = "reference-residue-gained-vs-current"
            rows.append(
                {
                    "variant_label": label,
                    "reference_variant_label": reference_label,
                    "benchmark_id": str(row.get("benchmark_id") or ""),
                    "chain": str(row.get("chain") or ""),
                    "resid": int(row.get("resid") or 0),
                    "resname": str(row.get("resname") or ""),
                    "residue_label": str(row.get("residue_label") or ""),
                    "reference_type": str(row.get("reference_type") or ""),
                    "reference_source": str(row.get("reference_source") or ""),
                    "expected_pocket_id": str(row.get("expected_pocket_id") or ""),
                    "variant_matched": variant_matched,
                    "reference_matched": reference_matched,
                    "match_delta": delta,
                    "variant_matched_rank": variant_rank,
                    "reference_matched_rank": reference_rank,
                    "rank_delta_vs_reference": rank_delta,
                    "variant_matched_pocket_id": str(row.get("matched_pocket_id") or ""),
                    "reference_matched_pocket_id": str(reference_row.get("matched_pocket_id") or "") if reference_row is not None else "",
                    "variant_matched_pocket_ids": str(row.get("matched_pocket_ids") or ""),
                    "reference_matched_pocket_ids": str(reference_row.get("matched_pocket_ids") or "") if reference_row is not None else "",
                    "variant_expected_pocket_matched": bool(row.get("expected_pocket_matched")),
                    "reference_expected_pocket_matched": bool(reference_row.get("expected_pocket_matched")) if reference_row is not None else False,
                    "benchmark_warning": warning,
                }
            )

    if not rows:
        return _empty_variant_detail_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS)


def _variant_issue_action(variant_label: str, issue_type: str) -> str:
    variant = str(variant_label or "").strip().lower()
    if issue_type == "current-missed-residue":
        return "Check curated residue numbering, chain mapping and pocket detection thresholds; current ranking does not cover this catalytic residue."
    if "no-literature" in variant:
        return "Literature evidence appears to support this residue; verify citation/snippet quality, numbering assumptions and manual-review flags before weakening the literature route."
    if "no-evidence-route" in variant:
        return "External-evidence route appears to preserve this residue; review route thresholds, mapping quality and neighborhood radius before disabling the route."
    if "no-p2rank" in variant:
        return "P2Rank appears to preserve this residue; verify the local P2Rank install/profile and keep a fallback when P2Rank is unavailable."
    if "no-conservation" in variant:
        return "Conservation rerank appears to preserve this residue; review conservation score direction and keep it rerank-only unless benchmark loss persists."
    return "Inspect this ablation before accepting the rank change; removing the evidence path changed catalytic residue coverage."


def build_pocket_benchmark_variant_remediation_queue(variant_detail_comparison_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Turn residue-level variant errors into a review/remediation queue."""

    if (
        variant_detail_comparison_df is None
        or getattr(variant_detail_comparison_df, "empty", True)
        or "match_delta" not in variant_detail_comparison_df.columns
    ):
        return _empty_variant_remediation_df()

    working = variant_detail_comparison_df.copy()
    for column in BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["match_delta"] = working["match_delta"].astype(str).str.strip()
    working["variant_label"] = working["variant_label"].astype(str).str.strip()
    working["reference_variant_label"] = working["reference_variant_label"].astype(str).str.strip()
    reference_label = str(working["reference_variant_label"].replace("", pd.NA).dropna().iloc[0]) if working["reference_variant_label"].replace("", pd.NA).dropna().shape[0] else "current"

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        variant_label = str(row.get("variant_label") or "").strip()
        delta = str(row.get("match_delta") or "").strip()
        issue_type = ""
        priority = ""
        if delta == "lost" and variant_label != reference_label:
            issue_type = "ablation-lost-residue"
            priority = "P0"
        elif delta == "unchanged-miss" and variant_label == reference_label:
            issue_type = "current-missed-residue"
            priority = "P1"
        else:
            continue

        benchmark_id = str(row.get("benchmark_id") or "").strip() or "current"
        residue_label = str(row.get("residue_label") or "").strip() or f"{row.get('chain', '')}{row.get('resid', '')}"
        action_id = f"{issue_type}:{variant_label}:{benchmark_id}:{residue_label}".replace(" ", "-")
        rows.append(
            {
                "action_id": action_id,
                "priority": priority,
                "issue_type": issue_type,
                "variant_label": variant_label,
                "reference_variant_label": reference_label,
                "benchmark_id": benchmark_id,
                "residue_label": residue_label,
                "chain": str(row.get("chain") or ""),
                "resid": int(row.get("resid") or 0),
                "resname": str(row.get("resname") or ""),
                "match_delta": delta,
                "reference_matched_pocket_id": str(row.get("reference_matched_pocket_id") or ""),
                "variant_matched_pocket_id": str(row.get("variant_matched_pocket_id") or ""),
                "reference_matched_rank": int(row.get("reference_matched_rank") or 0),
                "variant_matched_rank": int(row.get("variant_matched_rank") or 0),
                "expected_pocket_id": str(row.get("expected_pocket_id") or ""),
                "suggested_action": _variant_issue_action(variant_label, issue_type),
                "benchmark_warning": str(row.get("benchmark_warning") or ""),
            }
        )

    if not rows:
        return _empty_variant_remediation_df()
    frame = pd.DataFrame(rows, columns=BENCHMARK_VARIANT_REMEDIATION_COLUMNS)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["_priority_rank", "variant_label", "benchmark_id", "resid", "residue_label"]).drop(columns=["_priority_rank"])
    return frame.reset_index(drop=True)


def build_pocket_benchmark_variant_remediation_summary(remediation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize remediation actions by priority, issue type and variant."""

    if remediation_df is None or getattr(remediation_df, "empty", True):
        return _empty_variant_remediation_summary_df()

    working = remediation_df.copy()
    for column in BENCHMARK_VARIANT_REMEDIATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["priority"] = working["priority"].astype(str).str.strip()
    working["issue_type"] = working["issue_type"].astype(str).str.strip()
    working["variant_label"] = working["variant_label"].astype(str).str.strip()
    working["benchmark_id"] = working["benchmark_id"].astype(str).str.strip()
    working["residue_label"] = working["residue_label"].astype(str).str.strip()
    working["suggested_action"] = working["suggested_action"].astype(str).str.strip()
    if working.empty:
        return _empty_variant_remediation_summary_df()

    rows: list[dict[str, object]] = []
    group_columns = ["priority", "issue_type", "variant_label"]
    for (priority, issue_type, variant_label), group in working.groupby(group_columns, sort=True, dropna=False):
        residue_keys = (
            group["benchmark_id"].astype(str).str.strip()
            + "|"
            + group["chain"].astype(str).str.strip()
            + "|"
            + group["resid"].astype(str).str.strip()
        )
        top_residues = ", ".join(group["residue_label"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().drop_duplicates().head(8).tolist())
        action_count = int(len(group))
        status = "review-required" if action_count else "no-actions"
        warning = ""
        if str(priority).upper() == "P0":
            warning = "Ablation removes residues that current covers; do not weaken this evidence path without review."
        elif str(issue_type) == "current-missed-residue":
            warning = "Current run misses curated residues; validate numbering, mapping and detection thresholds."
        rows.append(
            {
                "priority": str(priority or ""),
                "issue_type": str(issue_type or ""),
                "variant_label": str(variant_label or ""),
                "action_count": action_count,
                "affected_case_count": int(group["benchmark_id"].replace("", pd.NA).dropna().nunique()),
                "affected_residue_count": int(residue_keys.nunique()),
                "top_residues": top_residues or "none",
                "suggested_action": str(group["suggested_action"].dropna().astype(str).head(1).iloc[0]) if not group.empty else "",
                "summary_status": status,
                "summary_warning": warning,
            }
        )

    if not rows:
        return _empty_variant_remediation_summary_df()
    frame = pd.DataFrame(rows, columns=BENCHMARK_VARIANT_REMEDIATION_SUMMARY_COLUMNS)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["_priority_rank", "variant_label", "issue_type"]).drop(columns=["_priority_rank"])
    return frame.reset_index(drop=True)


def build_pocket_benchmark_variant_remediation_checklist_markdown(
    remediation_df: Optional[pd.DataFrame],
    summary_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Pocket benchmark remediation checklist",
    max_actions: int = 80,
) -> str:
    """Render a manual review checklist for benchmark remediation actions."""

    if remediation_df is None or getattr(remediation_df, "empty", True):
        return ""

    working = remediation_df.copy()
    for column in BENCHMARK_VARIANT_REMEDIATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    working["_priority_rank"] = working["priority"].map(priority_rank).fillna(99)
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce").fillna(0).astype(int)
    working = working.sort_values(["_priority_rank", "variant_label", "benchmark_id", "resid", "residue_label"]).drop(columns=["_priority_rank"])

    summary = summary_df.copy() if summary_df is not None and not getattr(summary_df, "empty", True) else build_pocket_benchmark_variant_remediation_summary(working)
    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_variant_remediation_queue.csv`.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No remediation actions are currently required.")
    else:
        lines.append("| Priority | Issue | Variant | Actions | Cases | Residues |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for _, row in summary.iterrows():
            lines.append(
                "| {priority} | {issue} | {variant} | {actions} | {cases} | {residues} |".format(
                    priority=str(row.get("priority") or "-"),
                    issue=str(row.get("issue_type") or "-"),
                    variant=str(row.get("variant_label") or "-"),
                    actions=int(row.get("action_count") or 0),
                    cases=int(row.get("affected_case_count") or 0),
                    residues=int(row.get("affected_residue_count") or 0),
                )
            )

    lines.extend(["", "## Actions", ""])
    for index, (_, row) in enumerate(working.head(max_actions).iterrows(), start=1):
        lines.append(
            "- [ ] `{priority}` `{issue}` `{variant}` {case} {residue}: {action} Reference pocket `{ref_pocket}` rank `{ref_rank}`, variant pocket `{variant_pocket}` rank `{variant_rank}`.".format(
                priority=str(row.get("priority") or "-"),
                issue=str(row.get("issue_type") or "-"),
                variant=str(row.get("variant_label") or "-"),
                case=str(row.get("benchmark_id") or "current"),
                residue=str(row.get("residue_label") or f"{row.get('chain', '')}{row.get('resid', '')}"),
                action=str(row.get("suggested_action") or "Review this benchmark action."),
                ref_pocket=str(row.get("reference_matched_pocket_id") or "-"),
                ref_rank=int(row.get("reference_matched_rank") or 0),
                variant_pocket=str(row.get("variant_matched_pocket_id") or "-"),
                variant_rank=int(row.get("variant_matched_rank") or 0),
            )
        )
    if len(working) > max_actions:
        lines.append("")
        lines.append(f"Additional actions omitted: {len(working) - max_actions}. Export the CSV for the full queue.")
    return "\n".join(lines).strip() + "\n"
