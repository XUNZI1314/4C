from pathlib import Path
import sys

import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.sample_data import PDB_TEXT, PDB_TEXT_ALT, MMPBSA_TEXT, MMPBSA_TEXT_ALT
from protein_visualizer.services.parsers import parse_pdb_atoms, parse_mmpbsa_delta_total
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.comparison import compare_hotspot_sets
from protein_visualizer.services.explainer import explain_comparison

st.set_page_config(page_title="多构象比较", layout="wide")
st.title("多构象比较")
st.markdown("在此页面可上传多个构象及其对应的 MMPBSA 文件，系统会自动计算共同热点与一致性评分。")

uploaded_pdbs = st.file_uploader("上传 PDB 文件（可多选）", type=["pdb"], accept_multiple_files=True)
uploaded_mmpbs = st.file_uploader("上传 MMPBSA 文件（可多选，按构象顺序）", type=["txt", "dat", "out", "csv"], accept_multiple_files=True)

if uploaded_pdbs:
    pdb_texts = [f.getvalue().decode("utf-8", errors="ignore") for f in uploaded_pdbs]
else:
    pdb_texts = [PDB_TEXT, PDB_TEXT_ALT]

if uploaded_mmpbs:
    mmpbsa_texts = [f.getvalue().decode("utf-8", errors="ignore") for f in uploaded_mmpbs]
else:
    mmpbsa_texts = [MMPBSA_TEXT, MMPBSA_TEXT_ALT]

# pad mmpbsa_texts
if len(mmpbsa_texts) < len(pdb_texts):
    last = mmpbsa_texts[-1] if mmpbsa_texts else MMPBSA_TEXT
    mmpbsa_texts = mmpbsa_texts + [last] * (len(pdb_texts) - len(mmpbsa_texts))

energy_tables = []
for i, pdb_text in enumerate(pdb_texts):
    try:
        atom_df = parse_pdb_atoms(pdb_text)
        energy_df = parse_mmpbsa_delta_total(mmpbsa_texts[i])
        energy_table = prepare_energy_table(atom_df, energy_df)
    except Exception as exc:
        st.error(f"解析第 {i+1} 个构象失败：{exc}")
        energy_table = pd.DataFrame()
    energy_tables.append(energy_table)

if all(t.empty for t in energy_tables):
    st.warning("未解析到有效构象数据；请上传或使用示例数据。")
else:
    st.success(f"加载 {len(energy_tables)} 个构象（含示例或上传数据）。")

hotspot_lists = [identify_hotspots(t) if not t.empty else pd.DataFrame() for t in energy_tables]
comparison = compare_hotspot_sets(hotspot_lists)

st.subheader("一致性得分与总体信息")
st.metric("一致性得分", f"{comparison['consistency_score']:.2f}")
st.markdown(explain_comparison(comparison))

st.subheader("共同/差异热点统计（每个残基在多少个构象中被标为热点）")
st.dataframe(comparison["per_residue_df"].head(200), use_container_width=True)

# 导出共同热点
if comparison.get("per_residue_df") is not None and not comparison["per_residue_df"].empty:
    csv_bytes = comparison["per_residue_df"].to_csv(index=False).encode("utf-8")
    st.download_button("导出热点统计 CSV", data=csv_bytes, file_name="hotspot_comparison.csv", mime="text/csv")

st.info("提示：一致性得分为共同热点数 / 热点并集数，数值越高说明不同构象中热点越稳定。")
