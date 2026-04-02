from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms
from protein_visualizer.services.pdf_export import build_simple_pdf
from protein_visualizer.services.reporting import build_analysis_summary, build_text_report
from protein_visualizer.services.session_state import get_current_energy_table, initialize_state

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

if energy_table is None:
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    st.warning("当前未检测到共享分析结果，已回退到内置示例数据。请先在“结构可视化”页完成一次分析。")
else:
    st.success("当前显示的是结构可视化页面最近一次分析得到的共享结果。")

st.subheader("残基能量结果表")
st.dataframe(energy_table, use_container_width=True)

summary = build_analysis_summary(energy_table)

st.subheader("分析摘要")
col1, col2, col3 = st.columns(3)
col1.metric("残基总数", summary["residue_count"])
col2.metric("最低能量", f"{summary['min_energy']:.3f}")
col3.metric("最高能量", f"{summary['max_energy']:.3f}")

st.markdown(f"**平均能量**: {summary['mean_energy']:.3f}")
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

report_text = build_text_report(energy_table)
st.download_button(
    label="导出文本分析报告",
    data=report_text.encode("utf-8"),
    file_name="protein_analysis_report.txt",
    mime="text/plain",
)

pdf_bytes = build_simple_pdf(report_text)
st.download_button(
    label="导出 PDF 报告",
    data=pdf_bytes,
    file_name="protein_analysis_report.pdf",
    mime="application/pdf",
)

st.info("导出内容已支持跨页面共享。若重新上传并分析数据，请先在“结构可视化”页完成操作。")
