import pandas as pd

from protein_visualizer.services.benchmark import (
    build_pocket_benchmark_case_summary,
    build_pocket_benchmark_dataset_summary,
    build_pocket_benchmark_details,
    build_pocket_benchmark_summary,
    build_pocket_benchmark_variant_comparison,
    parse_benchmark_reference_table,
)


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
