from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.services.reporting import build_text_report
from protein_visualizer.services.snapshot import build_analysis_snapshot, snapshot_to_summary_lines


def test_build_text_report_includes_top_joint_recommendation():
    energy_table = pd.DataFrame(
        [
            {"label": "ALA A1", "heat_color": "#2563eb", "delta_total": -1.4},
            {"label": "GLY A2", "heat_color": "#ef4444", "delta_total": -2.8},
        ]
    )
    joint_candidate_table = pd.DataFrame(
        [
            {
                "pocket_id": "AutoPocket-2",
                "recommendation_label": "优先验证",
                "recommendation_score": 0.812,
                "recommendation_action": "validate-prioritize",
                "evidence_quality_label": "direct-anchor",
                "recommendation_reason": "口袋/界面/热点三重交集明显",
            }
        ]
    )

    report = build_text_report(energy_table, joint_candidate_table=joint_candidate_table)

    assert "AutoPocket-2" in report
    assert "优先验证" in report
    assert "0.812" in report
    assert "validate-prioritize" in report
    assert "direct-anchor" in report
    assert "三重交集明显" in report
    assert "推荐动作" in report
    assert "证据质量" in report


def test_build_text_report_localizes_smart_pocket_summary():
    energy_table = pd.DataFrame(
        [
            {"label": "ALA A1", "heat_color": "#2563eb", "delta_total": -1.4},
            {"label": "GLY A2", "heat_color": "#ef4444", "delta_total": -2.8},
        ]
    )
    pocket_summary = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "smart_rank_label": "优先验证",
                "smart_rank_score": 0.912,
                "smart_rank_reason": "文献关键残基支持",
                "hotspot_count": 3,
                "evidence_quality_label": "direct-anchor",
                "evidence_quality_score": 0.9,
                "evidence_quality_warning": "需核对编号",
            }
        ]
    )

    report = build_text_report(energy_table, pocket_summary=pocket_summary)

    for snippet in ["鏅", "鍙", "鎺", "鐑"]:
        assert snippet not in report
    assert "智能口袋摘要:" in report
    assert "- Top1 口袋: Pocket-1" in report
    assert "- 排序等级: 优先验证" in report
    assert "- 热点覆盖数: 3" in report
    assert "- 证据质量: direct-anchor (0.900)" in report
    assert "- 证据提醒: 需核对编号" in report


def test_snapshot_summary_lines_include_top_joint_recommendation():
    energy_table = pd.DataFrame(
        [
            {"label": "ALA A1", "heat_color": "#2563eb", "delta_total": -1.4},
            {"label": "GLY A2", "heat_color": "#ef4444", "delta_total": -2.8},
        ]
    )
    joint_candidate_table = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "pocket_id": "AutoPocket-3",
                "recommendation_label": "建议关注",
                "recommendation_score": 0.655,
                "recommendation_action": "validate-interface-context",
                "recommendation_reason": "界面覆盖较高",
            }
        ]
    )

    snapshot = build_analysis_snapshot(energy_table, joint_candidate_df=joint_candidate_table)
    summary_lines = snapshot_to_summary_lines(snapshot)

    assert any("AutoPocket-3" in line for line in summary_lines)
    assert any("validate-interface-context" in line for line in summary_lines)
    assert any("建议关注" in line for line in summary_lines)
