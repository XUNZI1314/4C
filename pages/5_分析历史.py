from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.services.session_state import get_history_records, initialize_state

st.set_page_config(page_title="分析历史", layout="wide")
st.title("分析历史")
st.caption("查看当前会话最近完成的蛋白质结构分析记录。")

initialize_state()
history = get_history_records()

if not history:
    st.warning("当前还没有分析历史。请先前往“结构可视化”页完成至少一次分析。")
else:
    st.success(f"当前共保存最近 {len(history)} 条分析记录。")
    latest = history[0]
    st.subheader("最近一次分析摘要")
    col1, col2, col3 = st.columns(3)
    col1.metric("最近分析时间", latest.get("generated_at", "-"))
    col2.metric("最近残基数", latest.get("residue_count", "-"))
    mean_energy = latest.get("mean_energy")
    col3.metric("最近平均能量", f"{mean_energy:.3f}" if isinstance(mean_energy, (int, float)) else "-")

    st.markdown(f"- PDB 来源：`{latest.get('source_name', '-')}`")
    st.markdown(f"- MMPBSA 来源：`{latest.get('energy_source_name', '-')}`")
    st.markdown(f"- 最低能量残基：`{latest.get('lowest_residue', '-')}`")
    st.markdown(f"- 最高能量残基：`{latest.get('highest_residue', '-')}`")

    st.subheader("历史记录表")
    history_df = pd.DataFrame(history)
    preferred_columns = [
        "generated_at",
        "source_name",
        "energy_source_name",
        "residue_count",
        "min_energy",
        "max_energy",
        "mean_energy",
        "lowest_residue",
        "highest_residue",
    ]
    available_columns = [column for column in preferred_columns if column in history_df.columns]
    st.dataframe(history_df[available_columns], use_container_width=True)

    st.info("当前历史记录保存在会话中，刷新页面或重启服务后会清空。后续可继续扩展为本地持久化。")
