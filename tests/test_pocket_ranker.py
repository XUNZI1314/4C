import pandas as pd

from protein_visualizer.sample_data import PDB_TEXT
from protein_visualizer.services.pocket import build_pocket_summary, build_pocket_summary_without_conservation_signal, detect_auto_pocket_table
from protein_visualizer.services.pocket_ranker import rank_detected_pockets


def test_rank_detected_pockets_adds_smart_columns():
    pocket_df = detect_auto_pocket_table(PDB_TEXT, prefer_kvfinder=False)

    assert not pocket_df.empty
    assert {
        "smart_rank_score",
        "smart_rank_order",
        "smart_rank_label",
        "smart_rank_reason",
        "smart_evidence_anchor_support",
        "smart_evidence_anchor_risk",
    }.issubset(pocket_df.columns)
    assert pocket_df["smart_rank_score"].astype(float).between(0.0, 1.0).all()
    assert pocket_df["smart_rank_order"].astype(int).ge(1).all()


def test_rank_detected_pockets_prefers_hotspot_supported_consensus_pocket():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-A",
                "chain": "A",
                "resid": 1,
                "resname": "ALA",
                "volume": 120.0,
                "score": 0.92,
                "residue_score": 0.88,
                "consensus_score": 0.90,
                "method_vote_count": 3,
                "consensus_methods": "geometry+kvfinder+ligand",
                "confidence_score": 0.82,
                "seed_support": 0.76,
                "is_hotspot": True,
                "detection_route": "precision-consensus-multiscale",
            },
            {
                "pocket_id": "Pocket-A",
                "chain": "A",
                "resid": 2,
                "resname": "TYR",
                "volume": 120.0,
                "score": 0.92,
                "residue_score": 0.84,
                "consensus_score": 0.86,
                "method_vote_count": 3,
                "consensus_methods": "geometry+kvfinder+ligand",
                "confidence_score": 0.79,
                "seed_support": 0.72,
                "is_hotspot": False,
                "detection_route": "precision-consensus-multiscale",
            },
            {
                "pocket_id": "Pocket-B",
                "chain": "A",
                "resid": 6,
                "resname": "GLU",
                "volume": 118.0,
                "score": 0.88,
                "residue_score": 0.73,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.55,
                "seed_support": 0.20,
                "is_hotspot": False,
                "detection_route": "precision-geometry",
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    summary = build_pocket_summary(ranked, pd.DataFrame())

    assert summary.iloc[0]["pocket_id"] == "Pocket-A"
    assert summary.iloc[0]["smart_rank_order"] == 1
    assert isinstance(summary.iloc[0]["smart_rank_reason"], str)


def test_ranked_summary_exposes_external_evidence_features():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-A",
                "chain": "A",
                "resid": 1,
                "resname": "SER",
                "volume": 100.0,
                "score": 0.78,
                "residue_score": 0.74,
                "consensus_score": 0.76,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.40,
                "is_hotspot": False,
                "external_support": 0.85,
                "external_confidence": 0.92,
                "external_evidence_count": 2,
                "external_exact_match": True,
                "detection_route": "precision-p2rank",
            },
            {
                "pocket_id": "Pocket-B",
                "chain": "A",
                "resid": 5,
                "resname": "GLY",
                "volume": 101.0,
                "score": 0.78,
                "residue_score": 0.74,
                "consensus_score": 0.76,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.40,
                "is_hotspot": False,
                "external_support": 0.0,
                "external_confidence": 0.0,
                "external_evidence_count": 0,
                "external_exact_match": False,
                "detection_route": "precision-geometry",
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    summary = build_pocket_summary(ranked, pd.DataFrame())

    assert summary.iloc[0]["pocket_id"] == "Pocket-A"
    assert summary.iloc[0]["external_exact_match_count"] == 1
    assert summary.iloc[0]["external_evidence_total"] == 2
    assert summary.iloc[0]["external_support_mean"] > 0.5
    assert summary.iloc[0]["smart_external_support"] > 0.5
    assert summary.iloc[0]["evidence_quality_label"] in {"direct-anchor", "strong-direct-anchor"}
    assert float(summary.iloc[0]["evidence_quality_score"]) > 0.0
    assert "外部" in str(summary.iloc[0]["smart_rank_reason"]) or "P2Rank" in str(summary.iloc[0]["smart_rank_reason"])


def test_ranked_summary_prefers_structure_verified_external_alignment():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Verified",
                "chain": "A",
                "resid": 10,
                "resname": "SER",
                "volume": 100.0,
                "score": 0.76,
                "residue_score": 0.72,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.38,
                "is_hotspot": False,
                "external_support": 0.72,
                "external_confidence": 0.88,
                "external_evidence_count": 1,
                "external_exact_match": True,
                "external_structure_verified": True,
                "external_mapping_quality": 0.98,
                "detection_route": "precision-geometry",
            },
            {
                "pocket_id": "Pocket-Fallback",
                "chain": "A",
                "resid": 25,
                "resname": "GLY",
                "volume": 100.0,
                "score": 0.76,
                "residue_score": 0.72,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.38,
                "is_hotspot": False,
                "external_support": 0.72,
                "external_confidence": 0.88,
                "external_evidence_count": 1,
                "external_exact_match": True,
                "external_structure_verified": False,
                "external_mapping_quality": 0.34,
                "detection_route": "precision-geometry",
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    summary = build_pocket_summary(ranked, pd.DataFrame())

    assert summary.iloc[0]["pocket_id"] == "Pocket-Verified"
    assert int(summary.iloc[0]["external_structure_verified_count"]) == 1
    assert float(summary.iloc[0]["external_mapping_quality_mean"]) > float(summary.iloc[1]["external_mapping_quality_mean"])
    assert float(summary.iloc[0]["smart_external_verified_ratio"]) > float(summary.iloc[1]["smart_external_verified_ratio"])


def test_ranked_summary_prefers_direct_evidence_anchor_over_neighborhood_support():
    common = {
        "volume": 100.0,
        "score": 0.76,
        "residue_score": 0.72,
        "consensus_score": 0.74,
        "method_vote_count": 1,
        "consensus_methods": "external-evidence",
        "confidence_score": 0.66,
        "seed_support": 0.70,
        "is_hotspot": False,
        "external_support": 0.80,
        "external_confidence": 0.90,
        "external_evidence_count": 1,
        "external_exact_match": False,
        "external_structure_verified": False,
        "external_mapping_quality": 0.70,
        "detection_route": "precision-external-evidence",
    }
    pocket_df = pd.DataFrame(
        [
            {
                **common,
                "pocket_id": "Pocket-Direct",
                "chain": "A",
                "resid": 10,
                "resname": "SER",
                "external_direct_anchor": True,
                "evidence_route_anchor": True,
                "evidence_anchor_proximity": 1.0,
            },
            {
                **common,
                "pocket_id": "Pocket-Neighborhood",
                "chain": "A",
                "resid": 20,
                "resname": "GLY",
                "external_direct_anchor": False,
                "evidence_route_anchor": False,
                "evidence_anchor_proximity": 0.25,
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    summary = build_pocket_summary(ranked, pd.DataFrame())

    assert summary.iloc[0]["pocket_id"] == "Pocket-Direct"
    assert float(summary.iloc[0]["smart_evidence_anchor_support"]) > float(summary.iloc[1]["smart_evidence_anchor_support"])
    assert float(summary.iloc[1]["smart_evidence_anchor_risk"]) > float(summary.iloc[0]["smart_evidence_anchor_risk"])
    assert "direct evidence anchor" in str(summary.iloc[0]["smart_rank_reason"])
    assert summary.iloc[0]["evidence_quality_label"] == "direct-anchor"
    assert summary.iloc[1]["evidence_quality_label"] == "neighborhood-expanded"
    assert float(summary.iloc[0]["evidence_quality_score"]) > float(summary.iloc[1]["evidence_quality_score"])


def test_ranked_summary_penalizes_shallow_exposed_pocket_without_rescue_signal():
    base_row = {
        "volume": 100.0,
        "score": 0.76,
        "residue_score": 0.72,
        "consensus_score": 0.74,
        "method_vote_count": 1,
        "consensus_methods": "geometry",
        "confidence_score": 0.66,
        "seed_support": 0.30,
        "is_hotspot": False,
        "external_support": 0.0,
        "external_confidence": 0.0,
        "external_evidence_count": 0,
        "external_exact_match": False,
        "ligand_contact_count": 0,
        "detection_route": "precision-geometry",
    }
    pocket_df = pd.DataFrame(
        [
            {
                **base_row,
                "pocket_id": "Pocket-Buried",
                "chain": "A",
                "resid": 10,
                "resname": "ASP",
                "contact_count": 8,
                "center_distance": 2.0,
                "depth_avg": 2.4,
            },
            {
                **base_row,
                "pocket_id": "Pocket-Buried",
                "chain": "A",
                "resid": 11,
                "resname": "HIS",
                "contact_count": 7,
                "center_distance": 2.3,
                "depth_avg": 2.1,
            },
            {
                **base_row,
                "pocket_id": "Pocket-Shallow",
                "chain": "A",
                "resid": 40,
                "resname": "LYS",
                "contact_count": 1,
                "center_distance": 16.0,
                "depth_avg": 0.1,
            },
            {
                **base_row,
                "pocket_id": "Pocket-Shallow",
                "chain": "A",
                "resid": 41,
                "resname": "SER",
                "contact_count": 1,
                "center_distance": 17.0,
                "depth_avg": 0.0,
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    summary = build_pocket_summary(ranked, pd.DataFrame())
    buried = summary[summary["pocket_id"] == "Pocket-Buried"].iloc[0]
    shallow = summary[summary["pocket_id"] == "Pocket-Shallow"].iloc[0]

    assert summary.iloc[0]["pocket_id"] == "Pocket-Buried"
    assert float(buried["smart_burial_support"]) > float(shallow["smart_burial_support"])
    assert float(shallow["smart_exposure_penalty"]) > float(buried["smart_exposure_penalty"])


def test_ranked_summary_uses_conservation_as_small_independent_signal():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Conservation",
                "chain": "A",
                "resid": 8,
                "resname": "ASP",
                "volume": 100.0,
                "score": 0.76,
                "residue_score": 0.72,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.38,
                "is_hotspot": False,
                "conservation_support": 0.72,
                "conservation_confidence": 0.86,
                "conservation_evidence_count": 2,
                "detection_route": "precision-geometry",
            },
            {
                "pocket_id": "Pocket-Plain",
                "chain": "A",
                "resid": 18,
                "resname": "GLY",
                "volume": 100.0,
                "score": 0.76,
                "residue_score": 0.72,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.38,
                "is_hotspot": False,
                "conservation_support": 0.0,
                "conservation_confidence": 0.0,
                "conservation_evidence_count": 0,
                "detection_route": "precision-geometry",
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    summary = build_pocket_summary(ranked, pd.DataFrame())

    assert summary.iloc[0]["pocket_id"] == "Pocket-Conservation"
    assert float(summary.iloc[0]["conservation_support_mean"]) > 0.5
    assert float(summary.iloc[0]["smart_conservation_support"]) > 0.5


def test_summary_without_conservation_signal_zeros_only_conservation_rerank_features():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Conservation",
                "chain": "A",
                "resid": 8,
                "resname": "ASP",
                "volume": 100.0,
                "score": 0.76,
                "residue_score": 0.72,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.38,
                "is_hotspot": False,
                "conservation_support": 0.72,
                "conservation_confidence": 0.86,
                "conservation_evidence_count": 2,
                "detection_route": "precision-geometry",
            },
            {
                "pocket_id": "Pocket-Plain",
                "chain": "A",
                "resid": 18,
                "resname": "GLY",
                "volume": 100.0,
                "score": 0.76,
                "residue_score": 0.72,
                "consensus_score": 0.74,
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "confidence_score": 0.66,
                "seed_support": 0.38,
                "is_hotspot": False,
                "conservation_support": 0.0,
                "conservation_confidence": 0.0,
                "conservation_evidence_count": 0,
                "detection_route": "precision-geometry",
            },
        ]
    )

    ranked = rank_detected_pockets(pocket_df)
    enhanced_summary = build_pocket_summary(ranked, pd.DataFrame())
    base_summary = build_pocket_summary_without_conservation_signal(ranked, pd.DataFrame())

    enhanced_row = enhanced_summary[enhanced_summary["pocket_id"] == "Pocket-Conservation"].iloc[0]
    base_row = base_summary[base_summary["pocket_id"] == "Pocket-Conservation"].iloc[0]

    assert float(enhanced_row["smart_conservation_support"]) > 0.5
    assert float(base_row["smart_conservation_support"]) == 0.0
    assert float(base_row["conservation_support_mean"]) == 0.0
    assert float(enhanced_row["smart_rank_score"]) > float(base_row["smart_rank_score"])
