import pandas as pd

from protein_visualizer.sample_data import ANNOTATION_TEXT, POCKET_TEXT, PDB_TEXT, MMPBSA_TEXT
from protein_visualizer.services.comparison import (
    build_hotspot_stability_tables,
    build_pairwise_similarity_matrix,
    build_reference_comparison_table,
    compare_hotspot_sets,
    compare_pocket_ranking_summaries,
)
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.interface import (
    build_inferred_interface_annotations,
    build_interface_overlap_summary,
    build_interface_summary,
    enrich_interface_annotations,
    merge_interface_annotation_tables,
    parse_interface_annotation_table,
)
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms
from protein_visualizer.services.pocket import parse_pocket_table
from protein_visualizer.services.reporting import build_analysis_summary
from protein_visualizer.services.structure_energy import resolve_energy_table


def test_pairwise_similarity_matrix_handles_empty_and_non_empty_tables():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    hotspot_df = identify_hotspots(energy_table)

    matrix = build_pairwise_similarity_matrix([hotspot_df, hotspot_df.iloc[:2].copy(), pd.DataFrame()])

    assert list(matrix.columns)[0] == "构象"
    assert matrix.shape == (3, 4)
    assert matrix.iloc[0]["构象 1"] == 1.0


def test_compare_hotspot_sets_returns_empty_frame_safely():
    comparison = compare_hotspot_sets([pd.DataFrame(), pd.DataFrame()])

    assert comparison["union_size"] == 0
    assert comparison["intersection_size"] == 0
    assert comparison["per_residue_df"].empty


def test_compare_hotspot_sets_adds_frequency_column():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    hotspot_df = identify_hotspots(energy_table)

    comparison = compare_hotspot_sets([hotspot_df, hotspot_df.copy()])

    assert "frequency" in comparison["per_residue_df"].columns
    assert comparison["per_residue_df"]["frequency"].max() <= 1.0


def test_build_hotspot_stability_tables_splits_stable_and_variable_hotspots():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    hotspot_df = identify_hotspots(energy_table)

    comparison = compare_hotspot_sets([hotspot_df, hotspot_df.copy(), hotspot_df.iloc[:3].copy()])
    stability = build_hotspot_stability_tables(comparison["per_residue_df"], stable_threshold=0.67)

    assert set(stability.keys()) == {"stable_hotspots", "variable_hotspots", "common_hotspots"}
    assert not stability["stable_hotspots"].empty
    assert stability["stable_hotspots"]["frequency"].min() >= 0.67
    if not stability["variable_hotspots"].empty:
        assert stability["variable_hotspots"]["frequency"].max() < 0.67


def test_build_reference_comparison_table_marks_reference_and_deltas():
    atom_df_1 = parse_pdb_atoms(PDB_TEXT)
    energy_df_1 = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    table_1 = prepare_energy_table(atom_df_1, energy_df_1)
    hotspot_df_1 = identify_hotspots(table_1)
    summary_1 = build_analysis_summary(table_1)

    atom_df_2 = parse_pdb_atoms(PDB_TEXT)
    energy_df_2 = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    table_2 = prepare_energy_table(atom_df_2, energy_df_2)
    hotspot_df_2 = hotspot_df_1.iloc[:3].copy()
    summary_2 = build_analysis_summary(table_2)

    reference_df = build_reference_comparison_table(
        [
            {
                "conformation": "构象 1",
                "residue_count": summary_1["residue_count"],
                "valid_energy_count": summary_1["valid_energy_count"],
                "mean_energy": summary_1["mean_energy"],
                "hotspot_count": len(hotspot_df_1),
                "energy_coverage": summary_1["energy_coverage"],
                "protein_volume": 100.0,
                "energy_source": summary_1["energy_source"],
            },
            {
                "conformation": "构象 2",
                "residue_count": summary_2["residue_count"],
                "valid_energy_count": summary_2["valid_energy_count"],
                "mean_energy": summary_2["mean_energy"],
                "hotspot_count": len(hotspot_df_2),
                "energy_coverage": summary_2["energy_coverage"],
                "protein_volume": 100.0,
                "energy_source": summary_2["energy_source"],
            },
        ],
        [hotspot_df_1, hotspot_df_2],
        reference_index=0,
    )

    assert not reference_df.empty
    assert "mean_energy_delta_vs_reference" in reference_df.columns
    assert bool(reference_df.iloc[0]["is_reference"])
    assert reference_df.iloc[0]["mean_energy_delta_vs_reference"] in (0, 0.0)


def test_compare_pocket_ranking_summaries_reports_rank_and_score_delta():
    base_summary = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-A",
                "smart_rank_order": 1,
                "smart_rank_score": 0.70,
                "smart_rank_label": "promising",
                "smart_rank_reason": "baseline",
                "evidence_quality_label": "no-external-evidence",
                "evidence_quality_score": 0.0,
                "smart_evidence_anchor_support": 0.0,
                "smart_evidence_anchor_risk": 0.0,
                "smart_conservation_support": 0.0,
                "conservation_support_mean": 0.0,
                "residue_count": 5,
            },
            {
                "pocket_id": "Pocket-B",
                "smart_rank_order": 2,
                "smart_rank_score": 0.68,
                "smart_rank_label": "promising",
                "smart_rank_reason": "baseline",
                "evidence_quality_label": "no-external-evidence",
                "evidence_quality_score": 0.0,
                "smart_evidence_anchor_support": 0.0,
                "smart_evidence_anchor_risk": 0.0,
                "smart_conservation_support": 0.0,
                "conservation_support_mean": 0.0,
                "residue_count": 4,
            },
        ]
    )
    enhanced_summary = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-B",
                "smart_rank_order": 1,
                "smart_rank_score": 0.73,
                "smart_rank_label": "high-priority",
                "smart_rank_reason": "conservation support",
                "evidence_quality_label": "direct-anchor",
                "evidence_quality_score": 0.62,
                "smart_evidence_anchor_support": 0.75,
                "smart_evidence_anchor_risk": 0.05,
                "smart_conservation_support": 0.72,
                "conservation_support_mean": 0.68,
                "residue_count": 4,
            },
            {
                "pocket_id": "Pocket-A",
                "smart_rank_order": 2,
                "smart_rank_score": 0.70,
                "smart_rank_label": "promising",
                "smart_rank_reason": "baseline",
                "evidence_quality_label": "no-external-evidence",
                "evidence_quality_score": 0.0,
                "smart_evidence_anchor_support": 0.0,
                "smart_evidence_anchor_risk": 0.0,
                "smart_conservation_support": 0.0,
                "conservation_support_mean": 0.0,
                "residue_count": 5,
            },
        ]
    )

    comparison = compare_pocket_ranking_summaries(base_summary, enhanced_summary)
    pocket_b = comparison[comparison["pocket_id"] == "Pocket-B"].iloc[0]

    assert pocket_b["status"] == "moved_up"
    assert pocket_b["rank_delta"] == 1
    assert pocket_b["score_delta"] == 0.05
    assert pocket_b["base_evidence_quality_label"] == "no-external-evidence"
    assert pocket_b["enhanced_evidence_quality_label"] == "direct-anchor"
    assert pocket_b["evidence_quality_delta"] == 0.62
    assert pocket_b["enhanced_evidence_anchor_support"] == 0.75
    assert pocket_b["enhanced_evidence_anchor_risk"] == 0.05
    assert pocket_b["enhanced_conservation_support"] == 0.72


def test_interface_summary_marks_pocket_and_hotspot_overlap():
    annotation_df = parse_interface_annotation_table(ANNOTATION_TEXT)
    pocket_df = parse_pocket_table(POCKET_TEXT)

    enriched = enrich_interface_annotations(
        annotation_df,
        pocket_residues=[(row.chain, int(row.resid)) for row in pocket_df.itertuples(index=False)],
        hotspot_residues=[("A", 2), ("A", 4)],
    )
    summary = build_interface_summary(enriched)

    assert not enriched.empty
    assert "is_overlap" in enriched.columns
    assert enriched["is_pocket"].any()
    assert enriched["is_hotspot"].any()
    assert not summary.empty
    assert summary["region_type"].isin({"binding-rim", "hotspot", "pocket-edge", "interface", "flexible"}).all()


def test_build_inferred_interface_annotations_uses_structure_contacts():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df, _ = resolve_energy_table(PDB_TEXT, energy_mode="estimate")
    energy_table = prepare_energy_table(atom_df, energy_df)

    inferred = build_inferred_interface_annotations(energy_table)

    assert not inferred.empty
    assert {"chain", "resid", "resname", "annotation", "region_type", "annotation_source", "inference_basis"}.issubset(inferred.columns)
    assert inferred["annotation_source"].eq("structure-inference").all()
    assert inferred["inference_basis"].isin({"inter-chain-contact", "surface-contact"}).all()


def test_build_inferred_interface_annotations_handles_missing_density_columns():
    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "resname": "ALA", "contact_count": 4, "contact_score": 2.2, "surface_proxy": 0.80},
            {"chain": "A", "resid": 2, "resname": "TYR", "contact_count": 5, "contact_score": 2.8, "surface_proxy": 0.72},
            {"chain": "A", "resid": 3, "resname": "GLU", "contact_count": 1, "contact_score": 0.4, "surface_proxy": 0.20},
        ]
    )

    inferred = build_inferred_interface_annotations(energy_table, top_n=2)

    assert not inferred.empty
    assert inferred["annotation_source"].eq("structure-inference").all()
    assert inferred["inference_basis"].eq("surface-contact").all()


def test_merge_interface_annotation_tables_prefers_primary_rows():
    primary = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "resname": "ALA", "annotation": "manual", "region_type": "binding-rim"},
            {"chain": "A", "resid": 2, "resname": "TYR", "annotation": "manual-2", "region_type": "interface"},
        ]
    )
    secondary = pd.DataFrame(
        [
            {"chain": "A", "resid": 2, "resname": "TYR", "annotation": "inferred-2", "region_type": "interface-core"},
            {"chain": "A", "resid": 3, "resname": "GLU", "annotation": "inferred-3", "region_type": "interface-rim"},
        ]
    )

    merged = merge_interface_annotation_tables(primary, secondary)

    assert len(merged) == 3
    assert merged.loc[(merged["chain"] == "A") & (merged["resid"] == 2), "annotation"].iloc[0] == "manual-2"
    assert merged["resid"].tolist() == [1, 2, 3]


def test_build_interface_overlap_summary_counts_intersections():
    annotation_df = parse_interface_annotation_table(ANNOTATION_TEXT)
    enriched = enrich_interface_annotations(
        annotation_df,
        pocket_residues=[("A", 1), ("A", 2), ("A", 3)],
        hotspot_residues=[("A", 2), ("A", 4)],
    )

    overlap_summary = build_interface_overlap_summary(
        enriched,
        pocket_residues=[("A", 1), ("A", 2), ("A", 3)],
        hotspot_residues=[("A", 2), ("A", 4)],
    )

    assert not overlap_summary.empty
    counts = dict(zip(overlap_summary["category"], overlap_summary["count"]))
    assert counts["interface_residues"] == len(enriched)
    assert counts["pocket_and_hotspot"] == 1
    assert counts["triple_overlap"] == 1
