from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.services.reporting import build_text_report
from protein_visualizer.services.snapshot import build_analysis_snapshot, snapshot_to_summary_lines


def test_build_text_report_includes_top_pocket_summary():
    table = pd.DataFrame(
        [
            {"label": "ALA A1", "heat_color": "#2563eb", "delta_total": -1.0},
            {"label": "GLY A2", "heat_color": "#ef4444", "delta_total": -2.0},
        ]
    )
    pocket_summary = pd.DataFrame(
        [
            {
                "pocket_id": "AutoPocket-1",
                "smart_rank_label": "高优先级",
                "smart_rank_score": 1.418,
                "hotspot_count": 2,
                "evidence_quality_label": "direct-anchor",
                "evidence_quality_score": 0.735,
                "evidence_quality_warning": "Direct external residue anchors are present; inspect mapping/source confidence.",
                "smart_rank_reason": "多方法共识且覆盖热点",
            }
        ]
    )

    report = build_text_report(table, pocket_summary=pocket_summary)

    assert "AutoPocket-1" in report
    assert "高优先级" in report
    assert "1.418" in report
    assert "direct-anchor" in report
    assert "0.735" in report
    assert "Direct external residue anchors" in report
    assert "多方法共识且覆盖热点" in report


def test_snapshot_summary_lines_include_top_pocket_label():
    table = pd.DataFrame(
        [
            {"label": "ALA A1", "heat_color": "#2563eb", "delta_total": -1.0},
            {"label": "GLY A2", "heat_color": "#ef4444", "delta_total": -2.0},
        ]
    )
    pocket_summary = pd.DataFrame(
        [
            {
                "pocket_id": "AutoPocket-2",
                "smart_rank_label": "候选优先级",
                "smart_rank_score": 1.102,
                "hotspot_count": 1,
                "residue_count": 3,
                "evidence_quality_label": "direct-anchor",
                "evidence_quality_score": 0.735,
            }
        ]
    )

    snapshot = build_analysis_snapshot(table, pocket_summary=pocket_summary)
    lines = snapshot_to_summary_lines(snapshot)

    assert any("AutoPocket-2" in line for line in lines)
    assert any("direct-anchor" in line for line in lines)
    assert snapshot["pocket_summary"][0]["evidence_quality_label"] == "direct-anchor"
    assert any("候选优先级" in line for line in lines)
