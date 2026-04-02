import numpy as np
import pandas as pd
import streamlit as st

from protein_visualizer.config.settings import SETTINGS
from protein_visualizer.sample_data import (
    MMPBSA_TEXT,
    PDB_TEXT,
    PDB_TEXT_ALT,
    MMPBSA_TEXT_ALT,
    POCKET_TEXT,
)
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.logging_utils import get_logger
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms
from protein_visualizer.services.reporting import build_analysis_summary
from protein_visualizer.services.session_state import (
    append_history_record,
    initialize_state,
    set_analysis_state,
)
from protein_visualizer.services.viewer import build_view
from protein_visualizer.services.hotspot import identify_hotspots, summarize_hotspot_clusters
from protein_visualizer.services.pocket import parse_pocket_table, build_pocket_summary
from protein_visualizer.services.comparison import compare_hotspot_sets
from protein_visualizer.services.explainer import explain_analysis, explain_comparison


LOGGER = get_logger(__name__)


def render_app() -> None:
    st.set_page_config(page_title=SETTINGS.page_title, layout=SETTINGS.layout)
    initialize_state()

    st.title(SETTINGS.page_title)
    st.caption("支持单/多构象 PDB 上传、MMPBSA 能量映射、口袋高亮与热点比较")

    with st.sidebar:
        st.header("数据输入")
        uploaded_pdbs = st.file_uploader("上传 PDB 文件（可多选）", type=["pdb"], accept_multiple_files=True)
        uploaded_mmpbs = st.file_uploader(
            "上传 MMPBSA 文件（可多选，按构象顺序）", type=["txt", "dat", "out", "csv"], accept_multiple_files=True
        )
        uploaded_pocket = st.file_uploader("上传 Pocket 文件（CSV）", type=["csv", "txt"], accept_multiple_files=False)

        # 准备构象与 MMPBSA 文本列表
        pdb_texts = []
        mmpbsa_texts = []
        if uploaded_pdbs:
            for f in uploaded_pdbs:
                try:
                    pdb_texts.append(f.getvalue().decode("utf-8", errors="ignore"))
                except Exception:
                    pdb_texts.append("")
        else:
            # 默认提供两个示例构象用于比较演示
            pdb_texts = [PDB_TEXT, PDB_TEXT_ALT]

        if uploaded_mmpbs:
            for f in uploaded_mmpbs:
                try:
                    mmpbsa_texts.append(f.getvalue().decode("utf-8", errors="ignore"))
                except Exception:
                    mmpbsa_texts.append("")
        else:
            mmpbsa_texts = [MMPBSA_TEXT, MMPBSA_TEXT_ALT]

        # 将 mmpbsa_texts 填充到与 pdb_texts 相同长度
        if len(mmpbsa_texts) < len(pdb_texts):
            last = mmpbsa_texts[-1] if mmpbsa_texts else MMPBSA_TEXT
            mmpbsa_texts = mmpbsa_texts + [last] * (len(pdb_texts) - len(mmpbsa_texts))

    # 解析所有构象数据
    energy_tables = []
    atom_dfs = []
    energy_dfs = []
    for i, pdb_text in enumerate(pdb_texts):
        try:
            atom_df = parse_pdb_atoms(pdb_text)
            energy_df = parse_mmpbsa_delta_total(mmpbsa_texts[i])
            energy_table = prepare_energy_table(atom_df, energy_df)
        except Exception as exc:
            LOGGER.exception("解析构象失败")
            atom_df = pd.DataFrame()
            energy_df = pd.DataFrame()
            energy_table = pd.DataFrame()
        atom_dfs.append(atom_df)
        energy_dfs.append(energy_df)
        energy_tables.append(energy_table)

    # 侧边栏显示控制（基于第一构象的能量范围作为参考）
    sample_table = next((t for t in energy_tables if not t.empty), None)
    energy_limit = float(max(0.1, abs(sample_table["delta_total"].min()) if sample_table is not None else 0.1, abs(sample_table["delta_total"].max()) if sample_table is not None else 0.1))

    with st.sidebar:
        st.header("显示控制")
        threshold = st.slider("MMPBSA |阈值| (绝对值)", 0.0, energy_limit, 0.0, 0.1)
        display_mode = st.radio(
            "显示模式",
            ["ball_stick", "surface", "transparent"],
            format_func=lambda x: {"ball_stick": "球棒", "surface": "表面", "transparent": "透明"}[x],
        )
        opacity = st.slider("透明度", 0.0, 1.0, SETTINGS.default_opacity, 0.05)
        show_backbone = st.checkbox("显示主链", value=True)
        show_sidechain = st.checkbox("显示侧链", value=True)
        show_surface = st.checkbox("显示表面", value=True)

        st.header("构象与比较")
        options = list(range(len(pdb_texts)))
        selected_conf = st.selectbox("选择构象 (显示)", options, format_func=lambda i: f"构象 {i+1}")
        compare_mode = st.checkbox("对比多个构象 (计算共同/差异热点)", value=False) if len(pdb_texts) > 1 else False

    # 取当前构象并渲染
    current_table = energy_tables[selected_conf] if energy_tables else pd.DataFrame()
    if current_table.empty:
        st.warning("当前构象未能解析出有效数据。请检查输入或使用示例数据。")
        return

    # 识别热点
    hotspot_df = identify_hotspots(current_table, energy_threshold=-abs(threshold) if threshold > 0 else -1.0)
    hotspot_clusters = summarize_hotspot_clusters(hotspot_df)

    # Pocket 解析（可选）
    pocket_df = None
    pocket_summary = pd.DataFrame()
    try:
        if uploaded_pocket:
            pocket_text = uploaded_pocket.getvalue().decode("utf-8", errors="ignore")
            pocket_df = parse_pocket_table(pocket_text)
        else:
            pocket_df = parse_pocket_table(POCKET_TEXT)
        pocket_summary = build_pocket_summary(pocket_df, hotspot_df)
    except Exception:
        pocket_df = None

    # 将当前分析写入会话状态（仅保存当前构象的结果）
    try:
        set_analysis_state(pdb_texts[selected_conf], mmpbsa_texts[selected_conf], atom_dfs[selected_conf], energy_dfs[selected_conf], current_table)
        summary = build_analysis_summary(current_table)
        append_history_record(
            {
                "generated_at": summary["generated_at"],
                "source_name": f"构象 {selected_conf+1}",
                "energy_source_name": f"构象 {selected_conf+1} MMPBSA",
                "residue_count": summary["residue_count"],
                "min_energy": summary["min_energy"],
                "max_energy": summary["max_energy"],
                "mean_energy": summary["mean_energy"],
                "lowest_residue": summary["lowest_residue"],
                "highest_residue": summary["highest_residue"],
            }
        )
    except Exception:
        LOGGER.exception("写入会话状态失败")

    col1, col2 = st.columns([2.2, 1.0])

    with col1:
        viewer = build_view(
            pdb_text=pdb_texts[selected_conf],
            energy_table=current_table,
            threshold=threshold,
            display_mode=display_mode,
            show_backbone=show_backbone,
            show_sidechain=show_sidechain,
            show_surface=show_surface,
            opacity=opacity,
            selected_chain=hotspot_df.iloc[0]["chain"] if not hotspot_df.empty else current_table.iloc[0]["chain"],
            selected_resid=int(hotspot_df.iloc[0]["resid"]) if not hotspot_df.empty else int(current_table.iloc[0]["resid"]),
        )
        st.components.v1.html(viewer._make_html(), height=SETTINGS.viewer_height + 20, scrolling=False)

    with col2:
        st.subheader("当前状态")
        highlighted_count = int((current_table["delta_total"].abs() >= threshold).sum())
        st.metric("高亮残基数", f"{highlighted_count}/{len(current_table)}")
        st.metric("当前模式", {"ball_stick": "球棒", "surface": "表面", "transparent": "透明"}[display_mode])
        st.metric("平均能量", f"{summary['mean_energy']:.3f}")

        st.subheader("自动分析摘要")
        st.write(explain_analysis(current_table, hotspot_df, pocket_summary))

        st.subheader("热点残基（示例）")
        if hotspot_df.empty:
            st.info("未检测到满足阈值的热点残基，已展示能量最低的前几位。")
        st.dataframe(hotspot_df[["label", "delta_total", "hotspot_rank"]].head(10), use_container_width=True)

    st.subheader("口袋/区域摘要")
    if not pocket_summary.empty:
        st.dataframe(pocket_summary, use_container_width=True)
    else:
        st.info("未加载口袋信息或口袋文件解析失败。")

    # 多构象比较
    if compare_mode and len(energy_tables) > 1:
        st.markdown("---")
        st.subheader("多构象比较：共同/差异热点")
        hotspot_lists = [identify_hotspots(t, energy_threshold=-abs(threshold) if threshold > 0 else -1.0) for t in energy_tables]
        comparison = compare_hotspot_sets(hotspot_lists)
        st.metric("共同热点一致性得分", f"{comparison['consistency_score']:.2f}")
        st.markdown(explain_comparison(comparison))
        st.dataframe(comparison["per_residue_df"].head(40), use_container_width=True)

    st.info("可上传多个 PDB/MMPBSA 文件进行并列比较；默认示例包含两个构象用于演示。")
