import pandas as pd

from protein_visualizer.services.candidate_fusion import (
    build_joint_candidate_table,
    build_pocket_consensus_coverage,
)


def test_build_joint_candidate_table_prioritizes_triple_overlap():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "chain": "A",
                "resid": 1,
                "resname": "ALA",
                "smart_rank_score": 0.84,
                "smart_rank_label": "high-priority",
                "method_vote_count": 3,
                "consensus_methods": "geometry+kvfinder+ligand",
                "pocket_source": "auto",
            },
            {
                "pocket_id": "Pocket-1",
                "chain": "A",
                "resid": 2,
                "resname": "TYR",
                "smart_rank_score": 0.84,
                "smart_rank_label": "high-priority",
                "method_vote_count": 3,
                "consensus_methods": "geometry+kvfinder+ligand",
                "pocket_source": "auto",
            },
            {
                "pocket_id": "Pocket-2",
                "chain": "A",
                "resid": 5,
                "resname": "GLU",
                "smart_rank_score": 0.61,
                "smart_rank_label": "promising",
                "method_vote_count": 1,
                "consensus_methods": "geometry",
                "pocket_source": "auto",
            },
        ]
    )
    annotation_df = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "resname": "ALA", "region_type": "interface-core"},
            {"chain": "A", "resid": 2, "resname": "TYR", "region_type": "interface-rim"},
        ]
    )
    hotspot_df = pd.DataFrame(
        [
            {"chain": "A", "resid": 1},
            {"chain": "A", "resid": 2},
        ]
    )

    result = build_joint_candidate_table(pocket_df, annotation_df, hotspot_df)

    assert not result.empty
    assert result.iloc[0]["pocket_id"] == "Pocket-1"
    assert result.iloc[0]["triple_overlap_count"] == 2
    assert result.iloc[0]["recommendation_label"] in {"优先验证", "建议关注"}


def test_build_joint_candidate_table_uses_external_site_evidence_boost():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "chain": "A",
                "resid": 10,
                "resname": "SER",
                "smart_rank_score": 0.62,
                "smart_rank_label": "promising",
                "method_vote_count": 2,
                "consensus_methods": "geometry+kvfinder",
                "pocket_source": "auto",
            },
            {
                "pocket_id": "Pocket-2",
                "chain": "A",
                "resid": 40,
                "resname": "GLU",
                "smart_rank_score": 0.62,
                "smart_rank_label": "promising",
                "method_vote_count": 2,
                "consensus_methods": "geometry+kvfinder",
                "pocket_source": "auto",
            },
        ]
    )
    annotation_df = pd.DataFrame()
    hotspot_df = pd.DataFrame()
    external_site_df = pd.DataFrame(
        [
            {"resid": 10, "evidence_type": "Active site"},
        ]
    )

    result = build_joint_candidate_table(
        pocket_df,
        annotation_df,
        hotspot_df,
        external_site_df=external_site_df,
    )

    assert not result.empty
    assert result.iloc[0]["pocket_id"] == "Pocket-1"
    assert int(result.iloc[0]["external_overlap_count"]) == 1
    assert float(result.iloc[0]["external_overlap_ratio"]) > 0.0
    assert "外部" in str(result.iloc[0]["recommendation_reason"])


def test_build_joint_candidate_table_uses_evidence_anchor_quality_in_recommendation():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Direct",
                "chain": "A",
                "resid": 10,
                "resname": "SER",
                "smart_rank_score": 0.60,
                "smart_rank_label": "promising",
                "smart_evidence_anchor_support": 0.85,
                "smart_evidence_anchor_risk": 0.02,
                "method_vote_count": 1,
                "consensus_methods": "external-evidence",
                "pocket_source": "auto",
            },
            {
                "pocket_id": "Pocket-Expanded",
                "chain": "A",
                "resid": 30,
                "resname": "GLY",
                "smart_rank_score": 0.60,
                "smart_rank_label": "promising",
                "smart_evidence_anchor_support": 0.15,
                "smart_evidence_anchor_risk": 0.55,
                "method_vote_count": 1,
                "consensus_methods": "external-evidence",
                "pocket_source": "auto",
            },
        ]
    )

    result = build_joint_candidate_table(pocket_df, pd.DataFrame(), pd.DataFrame())

    assert result.iloc[0]["pocket_id"] == "Pocket-Direct"
    assert result.iloc[0]["evidence_quality_label"] == "direct-anchor"
    assert result.iloc[0]["recommendation_action"] == "validate-prioritize"
    assert float(result.iloc[0]["evidence_anchor_support"]) > float(result.iloc[1]["evidence_anchor_support"])
    assert float(result.iloc[1]["evidence_anchor_risk"]) > float(result.iloc[0]["evidence_anchor_risk"])
    assert result.iloc[1]["recommendation_action"] == "review-evidence-mapping"
    assert "direct evidence anchor" in str(result.iloc[0]["recommendation_reason"])


def test_build_joint_candidate_table_prefers_exact_external_mapping_over_weak_signal():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Exact",
                "chain": "A",
                "resid": 15,
                "resname": "SER",
                "smart_rank_score": 0.58,
                "smart_rank_label": "promising",
                "method_vote_count": 2,
                "consensus_methods": "geometry+kvfinder",
                "pocket_source": "auto",
            },
            {
                "pocket_id": "Pocket-Weak",
                "chain": "B",
                "resid": 15,
                "resname": "SER",
                "smart_rank_score": 0.58,
                "smart_rank_label": "promising",
                "method_vote_count": 2,
                "consensus_methods": "geometry+kvfinder",
                "pocket_source": "auto",
            },
        ]
    )
    annotation_df = pd.DataFrame()
    hotspot_df = pd.DataFrame()
    external_site_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 15,
                "mapping_level": "exact",
                "mapping_confidence": 0.94,
                "evidence_type": "Active site",
            },
            {
                "chain": "",
                "resid": 15,
                "mapping_level": "weak",
                "mapping_confidence": 0.28,
                "evidence_type": "Binding site",
            },
        ]
    )

    result = build_joint_candidate_table(
        pocket_df,
        annotation_df,
        hotspot_df,
        external_site_df=external_site_df,
    )

    assert not result.empty
    assert result.iloc[0]["pocket_id"] == "Pocket-Exact"
    assert int(result.iloc[0]["external_exact_overlap_count"]) == 1
    assert float(result.iloc[0]["external_mapping_confidence"]) > 0.5


def test_build_joint_candidate_table_surfaces_structure_verified_external_alignment():
    pocket_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Verified",
                "chain": "A",
                "resid": 15,
                "resname": "SER",
                "smart_rank_score": 0.58,
                "smart_rank_label": "promising",
                "method_vote_count": 2,
                "consensus_methods": "geometry+kvfinder",
                "pocket_source": "auto",
            },
            {
                "pocket_id": "Pocket-Plain",
                "chain": "A",
                "resid": 25,
                "resname": "GLY",
                "smart_rank_score": 0.58,
                "smart_rank_label": "promising",
                "method_vote_count": 2,
                "consensus_methods": "geometry+kvfinder",
                "pocket_source": "auto",
            },
        ]
    )
    annotation_df = pd.DataFrame()
    hotspot_df = pd.DataFrame()
    external_site_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 15,
                "mapping_level": "exact",
                "mapping_confidence": 0.94,
                "mapping_method": "sifts-structure-order",
                "evidence_type": "Active site",
            },
            {
                "chain": "A",
                "resid": 25,
                "mapping_level": "exact",
                "mapping_confidence": 0.94,
                "mapping_method": "sifts-linear-interpolated",
                "evidence_type": "Active site",
            },
        ]
    )

    result = build_joint_candidate_table(
        pocket_df,
        annotation_df,
        hotspot_df,
        external_site_df=external_site_df,
    )

    assert not result.empty
    assert result.iloc[0]["pocket_id"] == "Pocket-Verified"
    assert int(result.iloc[0]["external_structure_verified_count"]) == 1
    assert float(result.iloc[0]["external_structure_verified_ratio"]) > 0.0


def test_build_pocket_consensus_coverage_prioritizes_validated_anchor_overlap():
    pocket_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-Validated", "chain": "A", "resid": 10, "resname": "ASP"},
            {"pocket_id": "Pocket-Validated", "chain": "A", "resid": 11, "resname": "GLY"},
            {"pocket_id": "Pocket-Blocked", "chain": "A", "resid": 30, "resname": "HIS"},
        ]
    )
    consensus_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "residue_anchor": "A:10",
                "consensus_score": 0.93,
                "consensus_tier": "validated-anchor",
                "evidence_sources": "M-CSA, AI-Literature",
            },
            {
                "chain": "A",
                "resid": 30,
                "residue_anchor": "A:30",
                "consensus_score": 0.42,
                "consensus_tier": "blocked-ai",
                "evidence_sources": "AI-Literature",
            },
        ]
    )

    coverage = build_pocket_consensus_coverage(pocket_df, consensus_df)

    assert coverage.iloc[0]["pocket_id"] == "Pocket-Validated"
    assert coverage.iloc[0]["pocket_consensus_label"] == "consensus-validated-pocket"
    assert int(coverage.iloc[0]["validated_anchor_count"]) == 1
    assert float(coverage.iloc[0]["consensus_coverage_ratio"]) == 0.5
    blocked = coverage[coverage["pocket_id"] == "Pocket-Blocked"].iloc[0]
    assert blocked["pocket_consensus_label"] == "blocked-ai-evidence-pocket"
    assert int(blocked["blocked_ai_count"]) == 1
