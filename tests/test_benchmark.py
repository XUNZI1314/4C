import pandas as pd

from protein_visualizer.services.benchmark import (
    build_pocket_benchmark_case_interpretation_summary,
    build_pocket_benchmark_case_interpretation_matrix,
    build_pocket_benchmark_case_interpretation_matrix_queue,
    build_pocket_benchmark_case_interpretation_matrix_summary,
    build_pocket_benchmark_dataset_interpretation_checklist_markdown,
    build_pocket_benchmark_dataset_interpretation_report_markdown,
    build_pocket_benchmark_dataset_interpretation,
    build_pocket_benchmark_dataset_interpretation_queue,
    build_pocket_benchmark_interpretation_summary,
    build_pocket_benchmark_case_summary,
    build_pocket_benchmark_dataset_summary,
    build_pocket_benchmark_details,
    build_pocket_benchmark_reference_quality_checklist_markdown,
    build_pocket_benchmark_reference_quality_issues,
    build_pocket_benchmark_reference_quality_summary,
    build_pocket_benchmark_reference_from_external_evidence,
    build_pocket_benchmark_reference_import_summary,
    build_pocket_benchmark_reference_readiness_case_summary,
    build_pocket_benchmark_reference_readiness_checklist_markdown,
    build_pocket_benchmark_reference_readiness_queue,
    build_pocket_benchmark_reference_readiness_summary,
    build_pocket_benchmark_reference_structure_validation,
    build_pocket_benchmark_reference_structure_validation_checklist_markdown,
    build_pocket_benchmark_reference_structure_validation_summary,
    build_pocket_benchmark_reference_template,
    build_pocket_benchmark_reference_template_markdown,
    build_pocket_benchmark_summary,
    build_pocket_benchmark_variant_comparison,
    build_pocket_benchmark_variant_case_comparison,
    build_pocket_benchmark_variant_dataset_comparison,
    build_pocket_benchmark_variant_detail_comparison,
    build_pocket_benchmark_variant_remediation_checklist_markdown,
    build_pocket_benchmark_variant_remediation_queue,
    build_pocket_benchmark_variant_remediation_summary,
    parse_benchmark_reference_table,
)


def test_build_pocket_benchmark_reference_template_is_parseable():
    template = build_pocket_benchmark_reference_template()
    markdown = build_pocket_benchmark_reference_template_markdown()
    reference_df, metadata = parse_benchmark_reference_table(template.to_csv(index=False))

    assert list(template.columns) == [
        "benchmark_id",
        "chain",
        "resid",
        "resname",
        "reference_type",
        "reference_source",
        "reference_note",
        "expected_pocket_id",
    ]
    assert metadata["status"] == "ok"
    assert int(metadata["reference_rows"]) == len(template)
    assert "blank chain is treated as wildcard" in markdown.lower()


def test_build_pocket_benchmark_reference_quality_flags_curation_risks():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname,type,source,note
, ,195,,Catalytic residue,,UniProt mature-chain numbering needs offset check
enzyme-a,A,57,HIS,Catalytic residue,M-CSA,
enzyme-a,A,57,HIS,Mutagenesis,PMID:12345,
"""
    )

    issues = build_pocket_benchmark_reference_quality_issues(reference_df)
    summary = build_pocket_benchmark_reference_quality_summary(issues)
    checklist = build_pocket_benchmark_reference_quality_checklist_markdown(issues, summary)

    issue_types = set(issues["issue_type"].astype(str).tolist())
    assert {
        "missing_benchmark_id",
        "generic_reference_source",
        "wildcard_chain",
        "missing_resname",
        "mapping_assumption_note",
        "multi_role_or_source_residue",
    }.issubset(issue_types)
    assert summary["issue_count"].astype(int).sum() == len(issues)
    assert "Benchmark reference curation checklist" in checklist
    assert '"nan"' not in reference_df.to_csv(index=False).lower()


def test_build_pocket_benchmark_reference_structure_validation_flags_mapping_risks():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname,type,source
enzyme-a,A,57,HIS,Catalytic residue,M-CSA
enzyme-a,A,102,ASP,Catalytic residue,M-CSA
enzyme-a,,195,SER,Catalytic residue,M-CSA
enzyme-a,B,999,GLY,Catalytic residue,M-CSA
"""
    )
    atom_df = pd.DataFrame(
        [
            {"record_type": "ATOM", "chain": "A", "resid": 57, "resname": "HIS"},
            {"record_type": "ATOM", "chain": "A", "resid": 102, "resname": "ASN"},
            {"record_type": "ATOM", "chain": "A", "resid": 195, "resname": "SER"},
            {"record_type": "ATOM", "chain": "B", "resid": 195, "resname": "SER"},
        ]
    )

    issues = build_pocket_benchmark_reference_structure_validation(reference_df, atom_df)
    summary = build_pocket_benchmark_reference_structure_validation_summary(issues)
    checklist = build_pocket_benchmark_reference_structure_validation_checklist_markdown(issues, summary)

    issue_types = set(issues["issue_type"].astype(str).tolist())
    assert issue_types == {
        "reference_residue_absent",
        "reference_resname_mismatch",
        "wildcard_chain_ambiguous_in_structure",
    }
    assert summary["issue_count"].astype(int).sum() == len(issues)
    assert "Benchmark reference structure validation checklist" in checklist


def test_build_pocket_benchmark_reference_readiness_gate_blocks_bad_references():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname,type,source,note
, ,195,,Catalytic residue,,UniProt mature-chain numbering needs offset check
enzyme-a,A,102,ASP,Catalytic residue,M-CSA,
"""
    )
    atom_df = pd.DataFrame(
        [
            {"record_type": "ATOM", "chain": "A", "resid": 102, "resname": "ASN"},
            {"record_type": "ATOM", "chain": "A", "resid": 195, "resname": "SER"},
            {"record_type": "ATOM", "chain": "B", "resid": 195, "resname": "SER"},
        ]
    )
    quality_issues = build_pocket_benchmark_reference_quality_issues(reference_df)
    structure_issues = build_pocket_benchmark_reference_structure_validation(reference_df, atom_df)

    queue = build_pocket_benchmark_reference_readiness_queue(quality_issues, structure_issues)
    summary = build_pocket_benchmark_reference_readiness_summary(reference_df, quality_issues, structure_issues)
    checklist = build_pocket_benchmark_reference_readiness_checklist_markdown(queue, summary)

    gate = summary.iloc[0]
    assert str(gate["readiness_status"]) == "blocked"
    assert int(gate["reference_residue_count"]) == 2
    assert int(gate["p0_p1_issue_count"]) >= 1
    assert set(queue["action_status"].astype(str)).issuperset({"blocker", "review"})
    assert "Benchmark reference readiness checklist" in checklist


def test_build_pocket_benchmark_reference_readiness_case_summary_splits_cases():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname,type,source
enzyme-a,,195,SER,Catalytic residue,
enzyme-b,A,57,HIS,Catalytic residue,M-CSA
enzyme-c,A,102,ASP,Catalytic residue,M-CSA
"""
    )
    atom_df = pd.DataFrame(
        [
            {"record_type": "ATOM", "chain": "A", "resid": 57, "resname": "HIS"},
            {"record_type": "ATOM", "chain": "A", "resid": 102, "resname": "ASP"},
            {"record_type": "ATOM", "chain": "B", "resid": 102, "resname": "ASP"},
        ]
    )
    quality_issues = build_pocket_benchmark_reference_quality_issues(reference_df)
    structure_issues = build_pocket_benchmark_reference_structure_validation(reference_df, atom_df)

    case_summary = build_pocket_benchmark_reference_readiness_case_summary(
        reference_df,
        quality_issues,
        structure_issues,
    )

    statuses = dict(zip(case_summary["benchmark_id"], case_summary["readiness_status"]))
    assert statuses["enzyme-a"] == "blocked"
    assert statuses["enzyme-b"] == "ready"
    assert statuses["enzyme-c"] == "ready"
    enzyme_a = case_summary[case_summary["benchmark_id"] == "enzyme-a"].iloc[0]
    assert int(enzyme_a["p0_p1_issue_count"]) >= 1


def test_build_pocket_benchmark_interpretation_respects_readiness_gate():
    benchmark_summary = pd.DataFrame(
        [
            {
                "top_n": 1,
                "reference_residue_count": 2,
                "matched_reference_count": 1,
                "coverage_ratio": 0.5,
                "benchmark_status": "top1-partial-hit",
                "best_rank": 1,
                "best_pocket_id": "Pocket-1",
            }
        ]
    )
    blocked_readiness = pd.DataFrame(
        [
            {
                "readiness_status": "blocked",
                "p0_p1_issue_count": 2,
                "p2_issue_count": 1,
                "recommended_action": "Fix mapping blockers.",
                "readiness_warning": "Reference numbering is not trustworthy.",
            }
        ]
    )
    ready_readiness = pd.DataFrame(
        [
            {
                "readiness_status": "ready",
                "p0_p1_issue_count": 0,
                "p2_issue_count": 0,
                "recommended_action": "Ready.",
                "readiness_warning": "No blockers.",
            }
        ]
    )

    blocked = build_pocket_benchmark_interpretation_summary(benchmark_summary, blocked_readiness)
    ready = build_pocket_benchmark_interpretation_summary(benchmark_summary, ready_readiness)

    assert str(blocked.iloc[0]["claim_status"]) == "blocked"
    assert not bool(blocked.iloc[0]["claim_ready"])
    assert "not claimable" in str(blocked.iloc[0]["interpretation_label"])
    assert str(ready.iloc[0]["claim_status"]) == "claim-ready"
    assert bool(ready.iloc[0]["claim_ready"])
    assert "partial curated residue coverage" in str(ready.iloc[0]["interpretation_label"])


def test_build_pocket_benchmark_case_interpretation_uses_case_readiness():
    case_summary = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "top_n": 1,
                "reference_residue_count": 1,
                "matched_reference_count": 1,
                "coverage_ratio": 1.0,
                "benchmark_status": "topn-complete-hit",
                "best_rank": 1,
                "best_pocket_id": "Pocket-1",
            },
            {
                "benchmark_id": "enzyme-b",
                "top_n": 1,
                "reference_residue_count": 1,
                "matched_reference_count": 1,
                "coverage_ratio": 1.0,
                "benchmark_status": "topn-complete-hit",
                "best_rank": 1,
                "best_pocket_id": "Pocket-1",
            },
        ]
    )
    readiness_cases = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "readiness_status": "blocked",
                "p0_p1_issue_count": 1,
                "p2_issue_count": 0,
                "recommended_action": "Fix enzyme-a mapping.",
                "readiness_warning": "Mapping blocked.",
            },
            {
                "benchmark_id": "enzyme-b",
                "readiness_status": "ready",
                "p0_p1_issue_count": 0,
                "p2_issue_count": 0,
                "recommended_action": "Ready.",
                "readiness_warning": "No blockers.",
            },
        ]
    )

    interpretation = build_pocket_benchmark_case_interpretation_summary(case_summary, readiness_cases)

    statuses = dict(zip(interpretation["benchmark_id"], interpretation["claim_status"]))
    assert statuses["enzyme-a"] == "blocked"
    assert statuses["enzyme-b"] == "claim-ready"


def test_build_pocket_benchmark_case_interpretation_matrix_pivots_topn_rows():
    case_interpretation = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "top_n": 1,
                "coverage_ratio": 0.0,
                "claim_status": "review-needed",
                "claim_ready": False,
                "best_rank": 0,
                "best_pocket_id": "",
                "benchmark_status": "top1-miss",
            },
            {
                "benchmark_id": "enzyme-a",
                "top_n": 3,
                "coverage_ratio": 0.8,
                "claim_status": "claim-ready",
                "claim_ready": True,
                "best_rank": 2,
                "best_pocket_id": "Pocket-2",
                "benchmark_status": "topn-partial-hit",
            },
            {
                "benchmark_id": "enzyme-b",
                "top_n": 1,
                "coverage_ratio": 0.5,
                "claim_status": "blocked",
                "claim_ready": False,
                "best_rank": 1,
                "best_pocket_id": "Pocket-1",
                "benchmark_status": "top1-partial-hit",
            },
        ]
    )

    matrix = build_pocket_benchmark_case_interpretation_matrix(case_interpretation, top_ns=(1, 3))

    enzyme_a = matrix[matrix["benchmark_id"] == "enzyme-a"].iloc[0]
    enzyme_b = matrix[matrix["benchmark_id"] == "enzyme-b"].iloc[0]
    assert int(enzyme_a["best_claim_ready_top_n"]) == 3
    assert float(enzyme_a["best_claim_ready_coverage"]) == 0.8
    assert str(enzyme_a["case_interpretation_status"]) == "review-needed"
    assert str(enzyme_a["top1_claim_status"]) == "review-needed"
    assert str(enzyme_a["top3_claim_status"]) == "claim-ready"
    assert str(enzyme_a["top3_best_pocket_id"]) == "Pocket-2"
    assert bool(enzyme_b["any_blocked"])
    assert str(enzyme_b["case_interpretation_status"]) == "blocked"
    assert int(enzyme_b["top3_best_rank"]) == 0


def test_build_pocket_benchmark_case_interpretation_matrix_summary_counts_case_states():
    matrix = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "best_claim_ready_top_n": 1,
                "best_claim_ready_coverage": 1.0,
                "best_claim_ready_rank": 1,
                "any_readiness_unknown": False,
                "case_interpretation_status": "claim-ready",
            },
            {
                "benchmark_id": "enzyme-b",
                "best_claim_ready_top_n": 0,
                "best_claim_ready_coverage": 0.0,
                "best_claim_ready_rank": 0,
                "any_readiness_unknown": False,
                "case_interpretation_status": "blocked",
            },
            {
                "benchmark_id": "enzyme-c",
                "best_claim_ready_top_n": 0,
                "best_claim_ready_coverage": 0.0,
                "best_claim_ready_rank": 0,
                "any_readiness_unknown": True,
                "case_interpretation_status": "review-needed",
            },
            {
                "benchmark_id": "enzyme-d",
                "best_claim_ready_top_n": 0,
                "best_claim_ready_coverage": 0.0,
                "best_claim_ready_rank": 0,
                "any_readiness_unknown": False,
                "case_interpretation_status": "no-claim-ready",
            },
        ]
    )

    summary = build_pocket_benchmark_case_interpretation_matrix_summary(matrix)

    row = summary.iloc[0]
    assert int(row["case_count"]) == 4
    assert int(row["usable_claim_ready_case_count"]) == 1
    assert int(row["blocked_case_count"]) == 1
    assert int(row["review_case_count"]) == 1
    assert int(row["readiness_unknown_case_count"]) == 1
    assert int(row["no_claim_ready_case_count"]) == 1
    assert int(row["earliest_top1_claim_ready_case_count"]) == 1
    assert float(row["mean_usable_claim_ready_coverage"]) == 1.0
    assert str(row["summary_status"]) == "blocked"


def test_build_pocket_benchmark_case_interpretation_matrix_queue_prioritizes_cases():
    matrix = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "case_interpretation_status": "claim-ready",
                "best_claim_ready_top_n": 1,
                "best_claim_ready_coverage": 1.0,
                "best_claim_ready_rank": 1,
                "top1_claim_status": "claim-ready",
                "recommended_action": "Ready.",
            },
            {
                "benchmark_id": "enzyme-b",
                "case_interpretation_status": "blocked",
                "best_claim_ready_top_n": 0,
                "best_claim_ready_coverage": 0.0,
                "best_claim_ready_rank": 0,
                "top1_claim_status": "blocked",
                "top3_claim_status": "blocked",
                "recommended_action": "Fix numbering.",
            },
            {
                "benchmark_id": "enzyme-c",
                "case_interpretation_status": "no-claim-ready",
                "best_claim_ready_top_n": 0,
                "best_claim_ready_coverage": 0.0,
                "best_claim_ready_rank": 0,
                "top1_claim_status": "readiness-unknown",
                "top3_claim_status": "review-needed",
            },
            {
                "benchmark_id": "enzyme-d",
                "case_interpretation_status": "review-needed",
                "best_claim_ready_top_n": 3,
                "best_claim_ready_coverage": 0.7,
                "best_claim_ready_rank": 3,
                "top1_claim_status": "review-needed",
                "top3_claim_status": "claim-ready",
            },
        ]
    )

    queue = build_pocket_benchmark_case_interpretation_matrix_queue(matrix)

    assert queue["action_id"].tolist() == ["BCMQ-001", "BCMQ-002", "BCMQ-003"]
    assert queue["benchmark_id"].tolist() == ["enzyme-b", "enzyme-c", "enzyme-d"]
    assert queue["priority"].tolist() == ["P0", "P1", "P2"]
    assert queue["issue_type"].tolist() == ["blocked-case", "no-claim-ready-case", "review-needed-case"]
    assert str(queue.iloc[0]["suggested_action"]) == "Fix numbering."


def test_build_pocket_benchmark_dataset_interpretation_aggregates_case_claims():
    case_interpretation = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "top_n": 1,
                "coverage_ratio": 1.0,
                "claim_status": "claim-ready",
                "claim_ready": True,
            },
            {
                "benchmark_id": "enzyme-b",
                "top_n": 1,
                "coverage_ratio": 0.5,
                "claim_status": "blocked",
                "claim_ready": False,
            },
            {
                "benchmark_id": "enzyme-c",
                "top_n": 1,
                "coverage_ratio": 0.0,
                "claim_status": "review-needed",
                "claim_ready": False,
            },
            {
                "benchmark_id": "enzyme-a",
                "top_n": 3,
                "coverage_ratio": 1.0,
                "claim_status": "claim-ready",
                "claim_ready": True,
            },
            {
                "benchmark_id": "enzyme-b",
                "top_n": 3,
                "coverage_ratio": 0.8,
                "claim_status": "claim-ready",
                "claim_ready": True,
            },
        ]
    )

    dataset_interpretation = build_pocket_benchmark_dataset_interpretation(case_interpretation)

    top1 = dataset_interpretation[dataset_interpretation["top_n"] == 1].iloc[0]
    assert int(top1["case_count"]) == 3
    assert int(top1["claim_ready_case_count"]) == 1
    assert int(top1["blocked_case_count"]) == 1
    assert int(top1["review_case_count"]) == 1
    assert float(top1["mean_claim_ready_coverage"]) == 1.0
    assert float(top1["mean_all_case_coverage"]) == 0.5
    assert str(top1["dataset_claim_status"]) == "blocked"

    top3 = dataset_interpretation[dataset_interpretation["top_n"] == 3].iloc[0]
    assert int(top3["case_count"]) == 2
    assert float(top3["mean_claim_ready_coverage"]) == 0.9
    assert str(top3["dataset_claim_status"]) == "claim-ready"


def test_build_pocket_benchmark_dataset_interpretation_queue_lists_blocking_cases():
    case_interpretation = pd.DataFrame(
        [
            {
                "benchmark_id": "enzyme-a",
                "top_n": 1,
                "coverage_ratio": 1.0,
                "claim_status": "claim-ready",
                "claim_ready": True,
                "recommended_action": "Ready.",
            },
            {
                "benchmark_id": "enzyme-b",
                "top_n": 1,
                "coverage_ratio": 0.5,
                "claim_status": "blocked",
                "claim_ready": False,
                "best_rank": 2,
                "best_pocket_id": "Pocket-2",
                "benchmark_status": "top1-partial-hit",
                "readiness_status": "blocked",
                "recommended_action": "Fix numbering.",
                "interpretation_warning": "Reference blocked.",
            },
            {
                "benchmark_id": "enzyme-c",
                "top_n": 1,
                "coverage_ratio": 0.0,
                "claim_status": "review needed",
                "claim_ready": False,
                "recommended_action": "Reviewer sign-off required.",
            },
            {
                "benchmark_id": "enzyme-b",
                "top_n": 3,
                "coverage_ratio": 0.8,
                "claim_status": "claim-ready",
                "claim_ready": True,
            },
        ]
    )

    queue = build_pocket_benchmark_dataset_interpretation_queue(case_interpretation)

    assert queue["action_id"].tolist() == ["BDSI-001", "BDSI-002"]
    assert queue["benchmark_id"].tolist() == ["enzyme-b", "enzyme-c"]
    assert queue["priority"].tolist() == ["P0", "P2"]
    assert queue["issue_type"].tolist() == ["blocked-case", "review-needed-case"]
    assert str(queue.iloc[0]["suggested_action"]) == "Fix numbering."


def test_build_pocket_benchmark_dataset_interpretation_checklist_markdown():
    queue = pd.DataFrame(
        [
            {
                "action_id": "BDSI-001",
                "priority": "P0",
                "action_status": "blocker",
                "top_n": 1,
                "benchmark_id": "enzyme-b",
                "claim_status": "blocked",
                "coverage_ratio": 0.5,
                "best_rank": 2,
                "best_pocket_id": "Pocket-2",
                "benchmark_status": "top1-partial-hit",
                "readiness_status": "blocked",
                "issue_type": "blocked-case",
                "suggested_action": "Fix numbering.",
                "interpretation_warning": "Reference blocked.",
            },
            {
                "action_id": "BDSI-002",
                "priority": "P2",
                "action_status": "review",
                "top_n": 1,
                "benchmark_id": "enzyme-c",
                "claim_status": "review-needed",
                "coverage_ratio": 0.0,
                "best_rank": 0,
                "best_pocket_id": "",
                "benchmark_status": "top1-miss",
                "readiness_status": "review-needed",
                "issue_type": "review-needed-case",
                "suggested_action": "Reviewer sign-off required.",
                "interpretation_warning": "Reference needs review.",
            },
        ]
    )

    checklist = build_pocket_benchmark_dataset_interpretation_checklist_markdown(queue)

    assert checklist.startswith("# Benchmark dataset interpretation checklist")
    assert "| P0 | blocker | blocked-case | 1 | 1 |" in checklist
    assert "| P2 | review | review-needed-case | 1 | 1 |" in checklist
    assert "`P0` Top-1 `blocked-case` case `enzyme-b` coverage `0.5` best rank `2` pocket `Pocket-2`: Fix numbering." in checklist


def test_build_pocket_benchmark_dataset_interpretation_report_markdown():
    interpretation = pd.DataFrame(
        [
            {
                "top_n": 1,
                "case_count": 2,
                "claim_ready_case_count": 1,
                "blocked_case_count": 1,
                "review_case_count": 0,
                "unknown_case_count": 0,
                "mean_claim_ready_coverage": 1.0,
                "mean_all_case_coverage": 0.75,
                "claim_ready_rate": 0.5,
                "dataset_claim_status": "blocked",
                "recommended_action": "Fix blocked cases.",
            },
            {
                "top_n": 3,
                "case_count": 2,
                "claim_ready_case_count": 2,
                "blocked_case_count": 0,
                "review_case_count": 0,
                "unknown_case_count": 0,
                "mean_claim_ready_coverage": 0.9,
                "mean_all_case_coverage": 0.9,
                "claim_ready_rate": 1.0,
                "dataset_claim_status": "claim-ready",
                "recommended_action": "Report coverage.",
            },
        ]
    )
    queue = pd.DataFrame(
        [
            {
                "action_id": "BDSI-001",
                "priority": "P0",
                "action_status": "blocker",
                "top_n": 1,
                "benchmark_id": "enzyme-b",
                "claim_status": "blocked",
                "coverage_ratio": 0.5,
                "best_rank": 2,
                "best_pocket_id": "Pocket-2",
                "issue_type": "blocked-case",
                "suggested_action": "Fix numbering.",
            }
        ]
    )

    report = build_pocket_benchmark_dataset_interpretation_report_markdown(
        interpretation,
        queue,
        checklist_available=True,
    )

    assert report.startswith("# Benchmark dataset interpretation report")
    assert "- Dataset claim status: `blocked`." in report
    assert "| 1 | blocked | 2 | 1 | 1 | 0 | 0 | 1.000 | 0.750 |" in report
    assert "| P0 | 1 | enzyme-b | blocked-case | 0.500 | 2 | Fix numbering. |" in report
    assert "- Checklist: available." in report


def test_parse_benchmark_reference_table_accepts_catalytic_residue_aliases():
    text = """case_id,chain,residue_label,type,source,note,validated_pocket_id
trypsin,A,Ser195,Catalytic residue,M-CSA,nucleophile,Pocket-2
trypsin,A,His57,Catalytic residue,M-CSA,base,Pocket-2
trypsin,A,Asp102,Catalytic residue,M-CSA,acid,Pocket-3
"""

    reference_df, metadata = parse_benchmark_reference_table(text, source_hint="M-CSA benchmark")

    assert metadata["status"] == "ok"
    assert metadata["reference_rows"] == "3"
    assert set(reference_df["resid"].astype(int).tolist()) == {57, 102, 195}
    assert set(reference_df["resname"].astype(str).tolist()) == {"SER", "HIS", "ASP"}
    assert reference_df["benchmark_id"].eq("trypsin").all()
    assert reference_df["expected_pocket_id"].isin({"Pocket-2", "Pocket-3"}).all()


def test_build_pocket_benchmark_reference_from_external_evidence_creates_candidate_summary():
    evidence_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 195,
                "resname": "SER",
                "evidence_source": "M-CSA",
                "evidence_type": "Catalytic residue",
                "evidence_note": "nucleophile",
                "mapping_level": "exact",
                "mapping_confidence": 0.98,
                "mapping_method": "sifts",
                "pmid": "12345",
            },
            {
                "chain": "A",
                "resid": 195,
                "resname": "SER",
                "evidence_source": "UniProt",
                "evidence_type": "Catalytic residue",
                "evidence_note": "active site",
                "mapping_level": "exact",
                "mapping_confidence": 0.96,
                "mapping_method": "sifts",
            },
            {
                "chain": "",
                "resid": 57,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_note": "general acid/base",
                "mapping_level": "weak",
                "mapping_confidence": 0.4,
                "mapping_method": "assumed-structure-numbering",
                "requires_manual_review": True,
            },
        ]
    )

    reference_df, metadata = build_pocket_benchmark_reference_from_external_evidence(
        evidence_df,
        default_benchmark_id="1ABC",
        source_hint="Loaded external evidence",
    )
    summary = build_pocket_benchmark_reference_import_summary(reference_df, metadata)

    assert metadata["status"] == "ok"
    assert metadata["evidence_rows"] == "3"
    assert metadata["reference_rows"] == "2"
    assert metadata["duplicate_rows"] == "1"
    assert reference_df["benchmark_id"].eq("1ABC").all()
    catalytic_195 = reference_df[reference_df["resid"].astype(int) == 195].iloc[0]
    assert str(catalytic_195["reference_source"]) == "M-CSA; PMID:12345; UniProt"
    assert "mapping_level=exact" in str(catalytic_195["reference_note"])
    summary_row = summary.iloc[0]
    assert str(summary_row["import_status"]) == "review-needed"
    assert int(summary_row["weak_mapping_rows"]) == 1
    assert int(summary_row["manual_review_rows"]) == 1
    assert int(summary_row["wildcard_chain_rows"]) == 1
    assert int(summary_row["missing_resname_rows"]) == 1


def test_build_pocket_benchmark_summary_reports_top1_and_top3_coverage():
    reference_df, _ = parse_benchmark_reference_table(
        """chain,resid,resname,reference_type
A,195,SER,Catalytic residue
A,57,HIS,Catalytic residue
A,102,ASP,Catalytic residue
"""
    )
    pocket_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-1", "chain": "A", "resid": 30, "resname": "GLY"},
            {"pocket_id": "Pocket-2", "chain": "A", "resid": 195, "resname": "SER"},
            {"pocket_id": "Pocket-2", "chain": "A", "resid": 57, "resname": "HIS"},
            {"pocket_id": "Pocket-3", "chain": "A", "resid": 102, "resname": "ASP"},
        ]
    )
    pocket_summary = pd.DataFrame(
        [
            {"pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.91},
            {"pocket_id": "Pocket-2", "smart_rank_order": 2, "smart_rank_score": 0.88},
            {"pocket_id": "Pocket-3", "smart_rank_order": 3, "smart_rank_score": 0.72},
        ]
    )

    summary = build_pocket_benchmark_summary(reference_df, pocket_df, pocket_summary)

    top1 = summary[summary["top_n"] == 1].iloc[0]
    top3 = summary[summary["top_n"] == 3].iloc[0]
    assert int(top1["matched_reference_count"]) == 0
    assert float(top1["coverage_ratio"]) == 0.0
    assert str(top1["benchmark_status"]) == "top1-miss"
    assert int(top3["matched_reference_count"]) == 3
    assert float(top3["coverage_ratio"]) == 1.0
    assert bool(top3["all_hit"])
    assert int(top3["best_rank"]) == 2
    assert str(top3["best_pocket_id"]) == "Pocket-2"


def test_build_pocket_benchmark_details_allows_wildcard_reference_chain():
    reference_df, _ = parse_benchmark_reference_table(
        """residue_label,reference_type
Asp102,Catalytic residue
"""
    )
    pocket_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-B", "chain": "B", "resid": 102, "resname": "ASP"},
        ]
    )

    details = build_pocket_benchmark_details(reference_df, pocket_df)

    assert len(details) == 1
    assert bool(details.iloc[0]["matched"])
    assert str(details.iloc[0]["matched_pocket_id"]) == "Pocket-B"
    assert bool(details.iloc[0]["matched_top1"])


def test_build_pocket_benchmark_summary_handles_missing_reference():
    summary = build_pocket_benchmark_summary(pd.DataFrame(), pd.DataFrame())

    assert summary.empty


def test_build_pocket_benchmark_case_and_dataset_summary_separate_cases():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname
enzyme-a,A,10,SER
enzyme-a,A,20,HIS
enzyme-b,B,5,ASP
"""
    )
    pocket_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-1", "chain": "A", "resid": 10, "resname": "SER"},
            {"pocket_id": "Pocket-1", "chain": "A", "resid": 20, "resname": "HIS"},
            {"pocket_id": "Pocket-2", "chain": "B", "resid": 99, "resname": "GLY"},
        ]
    )
    pocket_summary = pd.DataFrame(
        [
            {"pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90},
            {"pocket_id": "Pocket-2", "smart_rank_order": 2, "smart_rank_score": 0.70},
        ]
    )

    case_summary = build_pocket_benchmark_case_summary(
        reference_df,
        pocket_df,
        pocket_summary,
        top_ns=(1,),
    )
    dataset_summary = build_pocket_benchmark_dataset_summary(case_summary)

    assert set(case_summary["benchmark_id"].astype(str).tolist()) == {"enzyme-a", "enzyme-b"}
    enzyme_a = case_summary[case_summary["benchmark_id"] == "enzyme-a"].iloc[0]
    enzyme_b = case_summary[case_summary["benchmark_id"] == "enzyme-b"].iloc[0]
    assert float(enzyme_a["coverage_ratio"]) == 1.0
    assert bool(enzyme_a["all_hit"])
    assert float(enzyme_b["coverage_ratio"]) == 0.0
    assert str(enzyme_b["benchmark_status"]) == "top1-miss"

    top1_dataset = dataset_summary[dataset_summary["top_n"] == 1].iloc[0]
    assert int(top1_dataset["case_count"]) == 2
    assert int(top1_dataset["reference_residue_count"]) == 3
    assert int(top1_dataset["matched_reference_count"]) == 2
    assert float(top1_dataset["mean_coverage_ratio"]) == 0.5
    assert float(top1_dataset["any_hit_rate"]) == 0.5
    assert str(top1_dataset["benchmark_status"]) == "mixed-hit"


def test_benchmark_case_summary_filters_batch_pockets_by_benchmark_id():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname
enzyme-a,A,10,SER
enzyme-b,A,10,SER
"""
    )
    pocket_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "chain": "A", "resid": 10, "resname": "SER"},
            {"benchmark_id": "enzyme-b", "pocket_id": "Pocket-1", "chain": "A", "resid": 99, "resname": "GLY"},
        ]
    )
    pocket_summary = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90},
            {"benchmark_id": "enzyme-b", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90},
        ]
    )

    case_summary = build_pocket_benchmark_case_summary(
        reference_df,
        pocket_df,
        pocket_summary,
        top_ns=(1,),
    )

    enzyme_a = case_summary[case_summary["benchmark_id"] == "enzyme-a"].iloc[0]
    enzyme_b = case_summary[case_summary["benchmark_id"] == "enzyme-b"].iloc[0]
    assert float(enzyme_a["coverage_ratio"]) == 1.0
    assert float(enzyme_b["coverage_ratio"]) == 0.0


def test_build_pocket_benchmark_variant_comparison_reports_ablation_loss():
    reference_df, _ = parse_benchmark_reference_table(
        """chain,resid,resname
A,195,SER
A,57,HIS
"""
    )
    current_pocket_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-1", "chain": "A", "resid": 195, "resname": "SER"},
            {"pocket_id": "Pocket-1", "chain": "A", "resid": 57, "resname": "HIS"},
        ]
    )
    current_summary_df = pd.DataFrame([{"pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90}])
    ablated_pocket_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-X", "chain": "A", "resid": 30, "resname": "GLY"},
            {"pocket_id": "Pocket-2", "chain": "A", "resid": 195, "resname": "SER"},
            {"pocket_id": "Pocket-3", "chain": "A", "resid": 57, "resname": "HIS"},
        ]
    )
    ablated_summary_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-X", "smart_rank_order": 1, "smart_rank_score": 0.95},
            {"pocket_id": "Pocket-2", "smart_rank_order": 2, "smart_rank_score": 0.70},
            {"pocket_id": "Pocket-3", "smart_rank_order": 3, "smart_rank_score": 0.65},
        ]
    )

    comparison = build_pocket_benchmark_variant_comparison(
        reference_df,
        [
            ("current", current_pocket_df, current_summary_df),
            ("no-literature", ablated_pocket_df, ablated_summary_df),
        ],
        reference_variant_label="current",
        top_ns=(1, 3),
    )

    current_top1 = comparison[(comparison["variant_label"] == "current") & (comparison["top_n"] == 1)].iloc[0]
    ablated_top1 = comparison[(comparison["variant_label"] == "no-literature") & (comparison["top_n"] == 1)].iloc[0]
    ablated_top3 = comparison[(comparison["variant_label"] == "no-literature") & (comparison["top_n"] == 3)].iloc[0]
    assert float(current_top1["coverage_ratio"]) == 1.0
    assert float(ablated_top1["coverage_ratio"]) == 0.0
    assert float(ablated_top1["coverage_loss_vs_reference"]) == 1.0
    assert float(ablated_top3["coverage_ratio"]) == 1.0


def test_variant_case_and_dataset_comparison_report_case_level_loss():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname
enzyme-a,A,10,SER
enzyme-b,B,5,ASP
"""
    )
    current_pocket_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "chain": "A", "resid": 10, "resname": "SER"},
            {"benchmark_id": "enzyme-b", "pocket_id": "Pocket-1", "chain": "B", "resid": 5, "resname": "ASP"},
        ]
    )
    current_summary_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90},
            {"benchmark_id": "enzyme-b", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90},
        ]
    )
    ablated_pocket_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-X", "chain": "A", "resid": 99, "resname": "GLY"},
            {"benchmark_id": "enzyme-b", "pocket_id": "Pocket-1", "chain": "B", "resid": 5, "resname": "ASP"},
        ]
    )
    ablated_summary_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-X", "smart_rank_order": 1, "smart_rank_score": 0.70},
            {"benchmark_id": "enzyme-b", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90},
        ]
    )

    case_comparison = build_pocket_benchmark_variant_case_comparison(
        reference_df,
        [
            ("current", current_pocket_df, current_summary_df),
            ("no-literature", ablated_pocket_df, ablated_summary_df),
        ],
        reference_variant_label="current",
        top_ns=(1,),
    )
    dataset_comparison = build_pocket_benchmark_variant_dataset_comparison(case_comparison)

    enzyme_a_loss = case_comparison[
        (case_comparison["variant_label"] == "no-literature")
        & (case_comparison["benchmark_id"] == "enzyme-a")
        & (case_comparison["top_n"] == 1)
    ].iloc[0]
    no_lit_dataset = dataset_comparison[
        (dataset_comparison["variant_label"] == "no-literature")
        & (dataset_comparison["top_n"] == 1)
    ].iloc[0]
    assert float(enzyme_a_loss["coverage_loss_vs_reference"]) == 1.0
    assert int(no_lit_dataset["case_count"]) == 2
    assert float(no_lit_dataset["mean_coverage_ratio"]) == 0.5
    assert float(no_lit_dataset["mean_coverage_loss_vs_reference"]) == 0.5
    assert int(no_lit_dataset["case_loss_count"]) == 1


def test_variant_detail_comparison_reports_lost_residue():
    reference_df, _ = parse_benchmark_reference_table(
        """case_id,chain,resid,resname
enzyme-a,A,10,SER
enzyme-a,A,20,HIS
"""
    )
    current_pocket_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "chain": "A", "resid": 10, "resname": "SER"},
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "chain": "A", "resid": 20, "resname": "HIS"},
        ]
    )
    current_summary_df = pd.DataFrame(
        [{"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.90}]
    )
    ablated_pocket_df = pd.DataFrame(
        [
            {"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "chain": "A", "resid": 10, "resname": "SER"},
        ]
    )
    ablated_summary_df = pd.DataFrame(
        [{"benchmark_id": "enzyme-a", "pocket_id": "Pocket-1", "smart_rank_order": 1, "smart_rank_score": 0.80}]
    )

    detail_comparison = build_pocket_benchmark_variant_detail_comparison(
        reference_df,
        [
            ("current", current_pocket_df, current_summary_df),
            ("no-literature", ablated_pocket_df, ablated_summary_df),
        ],
        reference_variant_label="current",
    )

    lost_his = detail_comparison[
        (detail_comparison["variant_label"] == "no-literature")
        & (detail_comparison["resid"] == 20)
    ].iloc[0]
    kept_ser = detail_comparison[
        (detail_comparison["variant_label"] == "no-literature")
        & (detail_comparison["resid"] == 10)
    ].iloc[0]
    assert str(lost_his["match_delta"]) == "lost"
    assert bool(lost_his["reference_matched"])
    assert not bool(lost_his["variant_matched"])
    assert str(lost_his["benchmark_warning"]) == "reference-residue-lost-vs-current"
    assert str(kept_ser["match_delta"]) == "unchanged-hit"


def test_variant_remediation_queue_prioritizes_lost_and_current_missed_residues():
    detail_comparison = pd.DataFrame(
        [
            {
                "variant_label": "current",
                "reference_variant_label": "current",
                "benchmark_id": "enzyme-a",
                "chain": "A",
                "resid": 10,
                "resname": "SER",
                "residue_label": "SER A10",
                "match_delta": "unchanged-miss",
                "variant_matched": False,
                "reference_matched": False,
                "variant_matched_rank": 0,
                "reference_matched_rank": 0,
                "variant_matched_pocket_id": "",
                "reference_matched_pocket_id": "",
                "expected_pocket_id": "",
                "benchmark_warning": "reference-residue-not-covered-by-any-pocket",
            },
            {
                "variant_label": "no-literature",
                "reference_variant_label": "current",
                "benchmark_id": "enzyme-a",
                "chain": "A",
                "resid": 20,
                "resname": "HIS",
                "residue_label": "HIS A20",
                "match_delta": "lost",
                "variant_matched": False,
                "reference_matched": True,
                "variant_matched_rank": 0,
                "reference_matched_rank": 1,
                "variant_matched_pocket_id": "",
                "reference_matched_pocket_id": "Pocket-1",
                "expected_pocket_id": "Pocket-1",
                "benchmark_warning": "reference-residue-lost-vs-current",
            },
            {
                "variant_label": "no-literature",
                "reference_variant_label": "current",
                "benchmark_id": "enzyme-a",
                "chain": "A",
                "resid": 30,
                "resname": "ASP",
                "residue_label": "ASP A30",
                "match_delta": "unchanged-hit",
                "variant_matched": True,
                "reference_matched": True,
            },
        ]
    )

    queue = build_pocket_benchmark_variant_remediation_queue(detail_comparison)

    assert queue["issue_type"].tolist() == ["ablation-lost-residue", "current-missed-residue"]
    lost_row = queue.iloc[0]
    current_miss = queue.iloc[1]
    assert str(lost_row["priority"]) == "P0"
    assert str(lost_row["variant_label"]) == "no-literature"
    assert "Literature evidence" in str(lost_row["suggested_action"])
    assert str(current_miss["priority"]) == "P1"
    assert str(current_miss["issue_type"]) == "current-missed-residue"


def test_variant_remediation_summary_and_checklist_markdown():
    queue = pd.DataFrame(
        [
            {
                "action_id": "a1",
                "priority": "P0",
                "issue_type": "ablation-lost-residue",
                "variant_label": "no-evidence-route",
                "reference_variant_label": "current",
                "benchmark_id": "enzyme-a",
                "residue_label": "SER A10",
                "chain": "A",
                "resid": 10,
                "resname": "SER",
                "match_delta": "lost",
                "reference_matched_pocket_id": "Pocket-1",
                "variant_matched_pocket_id": "",
                "reference_matched_rank": 1,
                "variant_matched_rank": 0,
                "expected_pocket_id": "Pocket-1",
                "suggested_action": "Review route threshold.",
                "benchmark_warning": "reference-residue-lost-vs-current",
            },
            {
                "action_id": "a2",
                "priority": "P0",
                "issue_type": "ablation-lost-residue",
                "variant_label": "no-evidence-route",
                "reference_variant_label": "current",
                "benchmark_id": "enzyme-b",
                "residue_label": "HIS B20",
                "chain": "B",
                "resid": 20,
                "resname": "HIS",
                "match_delta": "lost",
                "reference_matched_pocket_id": "Pocket-2",
                "variant_matched_pocket_id": "",
                "reference_matched_rank": 2,
                "variant_matched_rank": 0,
                "expected_pocket_id": "",
                "suggested_action": "Review route threshold.",
                "benchmark_warning": "reference-residue-lost-vs-current",
            },
        ]
    )

    summary = build_pocket_benchmark_variant_remediation_summary(queue)
    checklist = build_pocket_benchmark_variant_remediation_checklist_markdown(queue, summary)

    assert len(summary) == 1
    assert int(summary.iloc[0]["action_count"]) == 2
    assert int(summary.iloc[0]["affected_case_count"]) == 2
    assert int(summary.iloc[0]["affected_residue_count"]) == 2
    assert "Ablation removes residues" in str(summary.iloc[0]["summary_warning"])
    assert checklist.startswith("# Pocket benchmark remediation checklist")
    assert "| P0 | ablation-lost-residue | no-evidence-route | 2 | 2 | 2 |" in checklist
    assert "`P0` `ablation-lost-residue` `no-evidence-route` enzyme-a SER A10" in checklist
