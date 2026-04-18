from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.reporting import build_analysis_summary, build_text_report, format_energy_value


def test_build_analysis_summary_ignores_imputed_zero_values():
    atom_df = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "resname": "ALA"},
            {"chain": "A", "resid": 2, "resname": "GLY"},
            {"chain": "A", "resid": 3, "resname": "SER"},
        ]
    )
    energy_df = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "resname": "ALA", "delta_total": -2.0},
            {"chain": "A", "resid": 3, "resname": "SER", "delta_total": -4.0},
        ]
    )

    table = prepare_energy_table(atom_df, energy_df)
    summary = build_analysis_summary(table)

    assert "delta_total_raw" in table.columns
    assert table.loc[table["resid"] == 2, "delta_total"].iloc[0] == 0.0
    assert pd.isna(table.loc[table["resid"] == 2, "delta_total_raw"]).iloc[0]
    assert summary["residue_count"] == 3
    assert summary["valid_energy_count"] == 2
    assert summary["energy_coverage"] == 2 / 3
    assert summary["mean_energy"] == -3.0
    assert summary["lowest_residue"] == "SER A3"
    assert summary["highest_residue"] == "ALA A1"


def test_build_analysis_summary_supports_energy_column():
    table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "residue_label": "ALA A1", "energy": -1.0},
            {"chain": "A", "resid": 2, "residue_label": "GLY A2", "energy": -3.0},
        ]
    )

    summary = build_analysis_summary(table)

    assert summary["valid_energy_count"] == 2
    assert summary["energy_coverage"] == 1.0
    assert summary["mean_energy"] == -2.0
    assert summary["lowest_residue"] == "GLY A2"


def test_build_analysis_summary_handles_empty_table():
    summary = build_analysis_summary(pd.DataFrame())

    assert summary["residue_count"] == 0
    assert summary["valid_energy_count"] == 0
    assert summary["mean_energy"] is None
    assert summary["lowest_residue"] == "-"
    assert format_energy_value(summary["mean_energy"]) == "-"
    report = build_text_report(pd.DataFrame())
    assert "无可用残基能量明细" in report