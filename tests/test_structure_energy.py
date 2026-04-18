import math

from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
from protein_visualizer.services.reporting import build_analysis_summary
from protein_visualizer.services.structure_energy import estimate_protein_volume, estimate_structure_energy_table, resolve_energy_table


def test_estimate_structure_energy_table_produces_energy_values():
    table = estimate_structure_energy_table(PDB_TEXT)

    assert not table.empty
    assert {
        "chain",
        "resid",
        "resname",
        "delta_total",
        "energy",
        "energy_source",
        "estimate_method",
        "contact_score",
        "contact_density",
        "interface_contact_density",
        "pairwise_energy",
    }.issubset(table.columns)
    assert table["energy_source"].eq("结构估算").all()
    assert table["estimate_method"].eq("structure_contact_proxy_v2").all()
    assert table["contact_score"].ge(0).all()
    assert table["delta_total"].dtype.kind in "fi"
    assert table["energy"].equals(table["delta_total"])

    summary = build_analysis_summary(table)
    assert summary["energy_source"] == "结构估算"
    assert summary["valid_energy_count"] == len(table)
    assert summary["mean_energy"] is not None
    assert summary["min_energy"] < summary["max_energy"]


def test_resolve_energy_table_prefers_mmpbsa_in_auto_mode():
    table, source = resolve_energy_table(PDB_TEXT, energy_mode="auto", mmpbsa_text=MMPBSA_TEXT)

    assert source == "MMPBSA数据"
    assert table is not None
    assert table["energy_source"].eq("MMPBSA数据").all()
    assert table["energy"].equals(table["delta_total"])


def test_resolve_energy_table_can_force_structure_estimation():
    table, source = resolve_energy_table(PDB_TEXT, energy_mode="estimate", mmpbsa_text=MMPBSA_TEXT)

    assert source == "结构估算"
    assert table is not None
    assert table["energy_source"].eq("结构估算").all()
    assert {"contact_density", "interface_contact_density"}.issubset(table.columns)


def test_resolve_energy_table_requires_uploaded_data_in_mmpbsa_mode():
    table, source = resolve_energy_table(PDB_TEXT, energy_mode="mmpbsa")

    assert table is None
    assert source == "无可用能量数据"


def test_estimate_protein_volume_returns_positive_value():
    volume = estimate_protein_volume(PDB_TEXT)

    assert math.isfinite(volume)
    assert volume > 0
