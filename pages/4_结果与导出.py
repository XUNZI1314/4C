from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms
from protein_visualizer.services.pdf_export import PDF_EXPORT_AVAILABLE, build_simple_pdf
from protein_visualizer.services.reporting import build_analysis_summary, build_text_report, format_energy_value
from protein_visualizer.services.session_state import (
    get_current_annotation_table,
    get_current_energy_table,
    get_current_joint_candidate_table,
    get_current_pocket_summary,
    get_current_pocket_table,
    get_current_pdb_text,
    initialize_state,
)
from protein_visualizer.services.snapshot import build_analysis_snapshot, build_snapshot_svg, snapshot_to_json_bytes
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.structure_energy import estimate_protein_volume

st.set_page_config(page_title="结果与导出", layout="wide")
st.title("结果与导出")
st.markdown(
    """
    <style>
    .summary-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_state()
energy_table = get_current_energy_table()
annotation_table = get_current_annotation_table()
joint_candidate_table = get_current_joint_candidate_table()
pocket_table = get_current_pocket_table()
pocket_summary = get_current_pocket_summary()

if energy_table is None:
    st.warning("当前未检测到共享分析结果。请先在“结构可视化”页完成一次分析。")
    use_examples = st.checkbox("使用示例结果预览", value=False)
    if not use_examples:
        st.stop()
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    pocket_table = pd.DataFrame()
    pocket_summary = pd.DataFrame()
    st.info("当前正在展示示例结果，仅用于预览导出格式。")
else:
    st.success("当前显示的是结构可视化页面最近一次分析得到的共享结果。")

if pocket_table is None:
    pocket_table = pd.DataFrame()
if pocket_summary is None:
    pocket_summary = pd.DataFrame()
if joint_candidate_table is None:
    joint_candidate_table = pd.DataFrame()

if annotation_table is not None and not annotation_table.empty:
    display_table = annotation_table
else:
    display_table = energy_table
    if annotation_table is not None:
        st.caption("当前注释表为空，已回退展示原始能量表。")
display_columns = [
    "residue_label",
    "classification_label",
    "classification_color",
    "classification_description",
    "energy",
    "delta_total",
    "delta_total_raw",
    "hotspot_rank",
    "is_hotspot",
    "is_pocket",
]
display_columns = [column for column in display_columns if display_table is not None and column in display_table.columns]
st.dataframe(display_table[display_columns] if display_columns else display_table, use_container_width=True)

if annotation_table is not None and not annotation_table.empty:
    classification_summary = (
        annotation_table.groupby("classification_label", dropna=False)
        .agg(
            count=("resid", "count"),
            mean_energy=("delta_total", "mean"),
            hotspots=("is_hotspot", "sum"),
            pocket=("is_pocket", "sum"),
        )
        .reset_index()
        .sort_values(by="count", ascending=False)
    )

    st.subheader("分类统计")
    st.dataframe(classification_summary, use_container_width=True)

    annotated_csv = annotation_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="导出残基注释 CSV",
        data=annotated_csv,
        file_name="protein_residue_annotations.csv",
        mime="text/csv",
    )

if not pocket_summary.empty:
    st.subheader("鏅鸿兘鍙ｈ鎽樿")
    top_pocket = pocket_summary.iloc[0]
    top_pocket_id = str(top_pocket.get("pocket_id") or "-")
    top_rank_label = str(top_pocket.get("smart_rank_label") or "-")
    top_reason = str(top_pocket.get("smart_rank_reason") or top_pocket.get("detection_route") or "-")
    st.caption(f"Top1 鍙ｈ锛?{top_pocket_id} / {top_rank_label} / {top_reason}")

    pocket_columns = [
        "pocket_id",
        "smart_rank_label",
        "smart_rank_score",
        "hotspot_count",
        "residue_count",
        "detection_route",
        "consensus_methods",
        "method_vote_count",
        "consensus_score",
        "volume",
        "score",
        "smart_rank_reason",
    ]
    pocket_columns = [column for column in pocket_columns if column in pocket_summary.columns]
    st.dataframe(pocket_summary[pocket_columns] if pocket_columns else pocket_summary, use_container_width=True)

    pocket_summary_csv = pocket_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="瀵煎嚭鏅鸿兘鍙ｈ鎽樿 CSV",
        data=pocket_summary_csv,
        file_name="smart_pocket_summary.csv",
        mime="text/csv",
    )

    if not pocket_table.empty:
        pocket_detail_csv = pocket_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="瀵煎嚭鍙ｈ鏄庣粏 CSV",
            data=pocket_detail_csv,
            file_name="smart_pocket_detail.csv",
            mime="text/csv",
        )

if not joint_candidate_table.empty:
    st.subheader("联合推荐摘要")
    top_joint = joint_candidate_table.iloc[0]
    top_joint_id = str(top_joint.get("pocket_id") or "-")
    top_joint_label = str(top_joint.get("recommendation_label") or "-")
    top_joint_action = str(top_joint.get("recommendation_action") or "-")
    top_joint_reason = str(top_joint.get("recommendation_reason") or "-")
    st.caption(f"Top1 联合推荐：{top_joint_id} / {top_joint_label} / {top_joint_reason}")

    joint_columns = [
        "recommendation_rank",
        "pocket_id",
        "recommendation_label",
        "recommendation_score",
        "recommendation_action",
        "evidence_quality_label",
        "evidence_anchor_support",
        "evidence_anchor_risk",
        "recommendation_reason",
        "smart_rank_label",
        "smart_rank_score",
        "hotspot_overlap_count",
        "interface_overlap_count",
        "triple_overlap_count",
        "method_vote_count",
        "consensus_methods",
    ]
    joint_columns = [column for column in joint_columns if column in joint_candidate_table.columns]
    st.dataframe(
        joint_candidate_table[joint_columns] if joint_columns else joint_candidate_table,
        use_container_width=True,
    )

    joint_candidate_csv = joint_candidate_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="导出联合推荐 CSV",
        data=joint_candidate_csv,
        file_name="joint_candidate_recommendations.csv",
        mime="text/csv",
    )

summary = build_analysis_summary(energy_table)
try:
    protein_volume = estimate_protein_volume(get_current_pdb_text())
except Exception:
    protein_volume = None
hotspot_df = identify_hotspots(energy_table) if energy_table is not None and not energy_table.empty else None
snapshot = build_analysis_snapshot(
    energy_table,
    title="ProteinInsight 结果快照",
    annotation_table=annotation_table,
    hotspot_df=hotspot_df,
    pocket_summary=pocket_summary,
    joint_candidate_df=joint_candidate_table,
    protein_volume=protein_volume,
)

st.subheader("分析摘要")
col1, col2, col3, col4 = st.columns(4)
col1.metric("残基总数", summary["residue_count"])
col2.metric("最低能量", format_energy_value(summary["min_energy"]))
col3.metric("最高能量", format_energy_value(summary["max_energy"]))
energy_metric_label = "平均能量（估算）" if summary.get("energy_source") == "结构估算" else "平均能量"
col4.metric(energy_metric_label, format_energy_value(summary["mean_energy"]))

protein_volume_text = f"{protein_volume:,.1f} A³" if protein_volume is not None else "-"
st.metric("蛋白质体积（估算）", protein_volume_text)

st.caption(f"平均能量基于 {summary['valid_energy_count']}/{summary['residue_count']} 个有效能量值计算")
if summary.get("energy_source"):
    source_note = "（不是标准 MMPBSA）" if summary.get("energy_source") == "结构估算" else ""
    st.caption(f"能量来源：{summary.get('energy_source')}{source_note}")
st.markdown(
    f"<div class='summary-card'><b>最低能量残基</b><br>{summary['lowest_residue']}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='summary-card'><b>最高能量残基</b><br>{summary['highest_residue']}</div>",
    unsafe_allow_html=True,
)

csv_data = energy_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="导出 CSV 结果",
    data=csv_data,
    file_name="protein_energy_results.csv",
    mime="text/csv",
)

report_text = build_text_report(
    energy_table,
    pocket_summary=pocket_summary,
    joint_candidate_table=joint_candidate_table,
)
st.download_button(
    label="导出文本分析报告",
    data=report_text.encode("utf-8"),
    file_name="protein_analysis_report.txt",
    mime="text/plain",
)

snapshot_json = snapshot_to_json_bytes(snapshot)
st.download_button(
    label="导出结果快照 JSON",
    data=snapshot_json,
    file_name="protein_analysis_snapshot.json",
    mime="application/json",
)

snapshot_svg = build_snapshot_svg(snapshot)
st.download_button(
    label="导出结果快照 SVG",
    data=snapshot_svg,
    file_name="protein_analysis_snapshot.svg",
    mime="image/svg+xml",
)

if PDF_EXPORT_AVAILABLE:
    pdf_bytes = build_simple_pdf(report_text, snapshot=snapshot)
    st.download_button(
        label="导出 PDF 报告（新版模板）",
        data=pdf_bytes,
        file_name="protein_analysis_report.pdf",
        mime="application/pdf",
    )
else:
    st.info("当前环境未安装 reportlab，PDF 报告暂不可用。请安装依赖后重试。")

st.info("导出内容已支持跨页面共享。若重新上传并分析数据，请先在“结构可视化”页完成操作。")
