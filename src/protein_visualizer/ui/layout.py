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
from protein_visualizer.services.parsers import parse_pdb_atoms
from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value
from protein_visualizer.services.structure_energy import estimate_protein_volume, resolve_energy_table
from protein_visualizer.services.session_state import (
    append_history_record,
    get_uploaded_inputs_cache,
    initialize_state,
    set_uploaded_inputs_cache,
    set_analysis_state,
)
from protein_visualizer.services.coloring import (
    CLASSIFICATION_THEME_OPTIONS,
    build_legacy_annotation_table,
    build_legacy_legend,
)
from protein_visualizer.services.viewer import build_view
from protein_visualizer.services.hotspot import identify_hotspots, summarize_hotspot_clusters
from protein_visualizer.services.pocket import (
    build_pocket_summary,
    detect_auto_pocket_table,
    get_pocket_detection_metadata,
    parse_pocket_table,
    summarize_pocket_detection_metadata,
)
from protein_visualizer.services.comparison import compare_hotspot_sets
from protein_visualizer.services.explainer import explain_analysis, explain_comparison


LOGGER = get_logger(__name__)


DISPLAY_COLUMN_LABELS = {
    "rank": "序号",
    "chain": "链",
    "resid": "残基编号",
    "resname": "残基名",
    "label": "标签",
    "residue_label": "残基",
    "delta_total": "总能量变化",
    "delta_total_raw": "原始总能量变化",
    "energy": "能量",
    "energy_source": "能量来源",
    "classification_label": "分类",
    "classification_color": "分类颜色",
    "classification_description": "分类说明",
    "display_color": "显示颜色",
    "hotspot_score": "热点得分",
    "hotspot_rank": "热点排名",
    "neighborhood_count": "邻近残基数",
    "cluster_id": "聚类 ID",
    "is_hotspot": "是否热点",
    "is_pocket": "是否口袋",
    "pocket_id": "口袋 ID",
    "volume": "体积",
    "score": "得分",
    "residue_count": "残基数",
    "hotspot_count": "热点数",
    "detection_route": "识别路径",
    "consensus_methods": "共识方法",
    "method_vote_count": "方法投票数",
    "consensus_score": "共识得分",
    "consensus_overlap_ratio": "共识重叠率",
    "smart_rank_score": "智能排名得分",
    "smart_rank_order": "智能排名顺序",
    "smart_rank_label": "智能排名标签",
    "smart_rank_reason": "智能排名理由",
    "evidence_quality_label": "证据质量",
    "evidence_quality_score": "证据质量得分",
    "evidence_quality_warning": "证据质量提醒",
    "smart_external_support": "智能外部证据支持",
    "smart_external_exact_ratio": "智能外部证据精确匹配率",
    "smart_external_verified_ratio": "智能外部结构验证率",
    "smart_external_mapping_quality": "智能外部证据映射质量",
    "smart_evidence_anchor_support": "智能证据锚点支持",
    "smart_evidence_anchor_risk": "智能证据锚点风险",
    "smart_conservation_support": "智能保守性支持",
    "smart_burial_support": "智能埋藏支持",
    "smart_exposure_penalty": "智能暴露惩罚",
    "external_supported_residue_count": "外部证据支持残基数",
    "external_evidence_total": "外部证据总数",
    "external_exact_match_count": "外部精确匹配数",
    "external_exact_match_ratio": "外部精确匹配率",
    "external_structure_verified_count": "外部结构验证数",
    "external_support_mean": "外部支持均值",
    "external_confidence_mean": "外部置信度均值",
    "external_mapping_quality_mean": "外部映射质量均值",
    "external_direct_anchor_count": "外部直接锚点数",
    "evidence_route_anchor_count": "证据路径锚点数",
    "evidence_anchor_min_distance": "证据锚点最小距离",
    "evidence_anchor_max_proximity": "证据锚点最大接近度",
    "evidence_anchor_residues": "证据锚点残基",
    "external_direct_sources": "外部直接来源",
    "external_evidence_types": "外部证据类型",
    "external_evidence_notes": "外部证据备注",
    "conservation_supported_residue_count": "保守性支持残基数",
    "conservation_evidence_total": "保守性证据总数",
    "conservation_support_mean": "保守性支持均值",
    "conservation_confidence_mean": "保守性置信度均值",
    "conservation_sources": "保守性来源",
    "external_sources": "外部来源",
    "residue_labels": "残基列表",
    "count": "出现次数",
    "frequency": "出现频率",
    "is_common": "是否共同热点",
}


DISPLAY_COLUMN_TOKEN_LABELS = {
    "ai": "AI",
    "anchor": "锚点",
    "burial": "埋藏",
    "chain": "链",
    "classification": "分类",
    "cluster": "聚类",
    "color": "颜色",
    "confidence": "置信度",
    "conservation": "保守性",
    "consensus": "共识",
    "count": "数量",
    "delta": "变化",
    "description": "说明",
    "detection": "识别",
    "display": "显示",
    "distance": "距离",
    "energy": "能量",
    "evidence": "证据",
    "exact": "精确",
    "external": "外部",
    "frequency": "频率",
    "hotspot": "热点",
    "id": "ID",
    "is": "是否",
    "label": "标签",
    "mapping": "映射",
    "max": "最大",
    "mean": "均值",
    "method": "方法",
    "methods": "方法",
    "min": "最小",
    "neighborhood": "邻近",
    "notes": "备注",
    "order": "顺序",
    "overlap": "重叠",
    "pocket": "口袋",
    "proximity": "接近度",
    "quality": "质量",
    "rank": "排名",
    "reason": "理由",
    "residue": "残基",
    "resid": "残基编号",
    "resname": "残基名",
    "route": "路径",
    "score": "得分",
    "source": "来源",
    "sources": "来源",
    "support": "支持",
    "supported": "支持",
    "total": "总计",
    "types": "类型",
    "verified": "验证",
    "volume": "体积",
    "vote": "投票",
    "warning": "提醒",
}


DISPLAY_VALUE_REPLACEMENTS = {
    "precision-consensus": "精度共识",
    "precision-external-evidence": "精度外部证据",
    "precision-fpocket": "精度 fpocket",
    "precision-kvfinder": "精度 KVFinder",
    "precision-kvfinder-multiscale": "精度 KVFinder 多尺度",
    "precision-p2rank": "精度 P2Rank",
    "precision-geometry": "精度几何",
    "fallback-": "回退-",
    "consensus": "共识",
    "external-evidence": "外部证据",
    "high": "高",
    "medium": "中",
    "low": "低",
    "excellent": "优秀",
    "good": "良好",
    "review": "需复核",
    "weak": "弱",
}


def localize_column_name(column: object) -> object:
    if not isinstance(column, str):
        return column
    if column in DISPLAY_COLUMN_LABELS:
        return DISPLAY_COLUMN_LABELS[column]
    if "_" not in column:
        return DISPLAY_COLUMN_TOKEN_LABELS.get(column.lower(), column)
    parts = column.split("_")
    localized_parts = [DISPLAY_COLUMN_TOKEN_LABELS.get(part.lower(), part) for part in parts]
    if localized_parts != parts:
        return " ".join(localized_parts)
    return column


def localize_display_value(value: object) -> object:
    if value is None:
        return value
    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text in DISPLAY_VALUE_REPLACEMENTS:
        return DISPLAY_VALUE_REPLACEMENTS[text]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) > 1:
        localized_parts = [DISPLAY_VALUE_REPLACEMENTS.get(part, part) for part in parts]
        if localized_parts != parts:
            return "，".join(localized_parts)
    for prefix, target_prefix in {
        "precision-": "精度-",
        "fallback-": "回退-",
    }.items():
        if text.startswith(prefix):
            return text.replace(prefix, target_prefix, 1)
    return text


def localize_display_table(table: pd.DataFrame) -> pd.DataFrame:
    if table is None:
        return table
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_bool_dtype(display[column]):
            display[column] = display[column].map(lambda value: "是" if bool(value) else "否")
        elif pd.api.types.is_object_dtype(display[column]) or pd.api.types.is_string_dtype(display[column]):
            display[column] = display[column].map(localize_display_value)
    return display.rename(columns={column: localize_column_name(column) for column in display.columns})


def _decode_uploaded_entries(uploaded_files, default_name: str) -> list[dict]:
    entries = []
    if not uploaded_files:
        return entries
    for f in uploaded_files:
        try:
            text = f.getvalue().decode("utf-8", errors="ignore")
            if not text:
                continue
            entries.append({"name": getattr(f, "name", default_name), "text": text})
        except Exception:
            continue
    return entries


def _uploader_prev_names_key(name: str) -> str:
    return f"protein_visualizer_uploader_prev_names_{name}"


def render_app() -> None:
    st.set_page_config(page_title=SETTINGS.page_title, layout=SETTINGS.layout)
    initialize_state()
    resolved_pocket_text = None

    st.title(SETTINGS.page_title)
    st.caption("支持单/多构象 PDB 上传、MMPBSA 能量映射、口袋高亮与热点比较")

    with st.sidebar:
        st.header("数据输入")
        uploaded_pdbs = st.file_uploader(
            "上传 PDB 文件（可多选）",
            type=["pdb"],
            accept_multiple_files=True,
            key="protein_visualizer_uploader_pdb",
        )
        uploaded_mmpbs = st.file_uploader(
            "上传 MMPBSA 文件（可多选，按构象顺序）",
            type=["txt", "dat", "out", "csv"],
            accept_multiple_files=True,
            key="protein_visualizer_uploader_mmpbsa",
        )
        uploaded_pocket = st.file_uploader(
            "上传 Pocket 文件（CSV）",
            type=["csv", "txt"],
            accept_multiple_files=False,
            key="protein_visualizer_uploader_pocket",
        )
        use_examples = st.checkbox("使用示例数据", value=False)

        cached_inputs = get_uploaded_inputs_cache()
        pdb_cached_entries = list(cached_inputs.get("pdb_files", []))
        mmpbsa_cached_entries = list(cached_inputs.get("mmpbsa_files", []))
        pocket_raw = cached_inputs.get("pocket_file")
        pocket_cached_entry = pocket_raw if isinstance(pocket_raw, dict) else None

        pdb_prev_names = st.session_state.get(_uploader_prev_names_key("pdb"))
        mmpbsa_prev_names = st.session_state.get(_uploader_prev_names_key("mmpbsa"))
        pocket_prev_names = st.session_state.get(_uploader_prev_names_key("pocket"))

        uploaded_pdb_entries = _decode_uploaded_entries(uploaded_pdbs, "uploaded.pdb")
        uploaded_mmpbsa_entries = _decode_uploaded_entries(uploaded_mmpbs, "uploaded.csv")
        uploaded_pocket_entries = _decode_uploaded_entries([uploaded_pocket] if uploaded_pocket else [], "uploaded_pocket.csv")
        uploaded_pocket_entry = uploaded_pocket_entries[0] if uploaded_pocket_entries else None

        if not use_examples:
            cache_changed = False
            if uploaded_pdb_entries:
                pdb_cached_entries = uploaded_pdb_entries
                cache_changed = True
            elif isinstance(pdb_prev_names, list) and len(pdb_prev_names) > 0:
                pdb_cached_entries = []
                cache_changed = True

            if uploaded_mmpbsa_entries:
                mmpbsa_cached_entries = uploaded_mmpbsa_entries
                cache_changed = True
            elif isinstance(mmpbsa_prev_names, list) and len(mmpbsa_prev_names) > 0:
                mmpbsa_cached_entries = []
                cache_changed = True

            if uploaded_pocket_entry is not None:
                pocket_cached_entry = uploaded_pocket_entry
                cache_changed = True
            elif isinstance(pocket_prev_names, list) and len(pocket_prev_names) > 0:
                pocket_cached_entry = None
                cache_changed = True

            if cache_changed:
                set_uploaded_inputs_cache(
                    pdb_files=pdb_cached_entries,
                    mmpbsa_files=mmpbsa_cached_entries,
                    pocket_file=pocket_cached_entry,
                )

        st.session_state[_uploader_prev_names_key("pdb")] = [
            str(item.get("name") or "") for item in uploaded_pdb_entries
        ]
        st.session_state[_uploader_prev_names_key("mmpbsa")] = [
            str(item.get("name") or "") for item in uploaded_mmpbsa_entries
        ]
        st.session_state[_uploader_prev_names_key("pocket")] = [
            str(uploaded_pocket_entry.get("name") or "")
        ] if uploaded_pocket_entry is not None else []

        energy_mode = st.selectbox(
            "能量来源模式",
            ["auto", "mmpbsa", "estimate"],
            index=0,
            format_func=lambda x: {"auto": "自动", "mmpbsa": "上传 MMPBSA", "estimate": "结构估算"}[x],
        )

        pdb_texts = []
        mmpbsa_texts = []
        pdb_input_source = "未提供"
        mmpbsa_input_source = "未提供"
        pocket_input_source = "未提供"

        if use_examples:
            pdb_texts = [PDB_TEXT, PDB_TEXT_ALT]
            mmpbsa_texts = [MMPBSA_TEXT, MMPBSA_TEXT_ALT]
            pdb_input_source = "示例数据"
            mmpbsa_input_source = "示例数据"
            pocket_input_source = "示例数据"
        else:
            for item in pdb_cached_entries:
                text = str(item.get("text") or "")
                if text:
                    pdb_texts.append(text)
            for item in mmpbsa_cached_entries:
                text = str(item.get("text") or "")
                if text:
                    mmpbsa_texts.append(text)

            if uploaded_pocket_entry is not None:
                resolved_pocket_text = str(uploaded_pocket_entry.get("text") or "")
            elif pocket_cached_entry and pocket_cached_entry.get("text"):
                resolved_pocket_text = str(pocket_cached_entry.get("text") or "")

            if uploaded_pdb_entries:
                pdb_input_source = "本次上传"
            elif pdb_texts:
                pdb_input_source = "缓存恢复"

            if uploaded_mmpbsa_entries:
                mmpbsa_input_source = "本次上传"
            elif mmpbsa_texts:
                mmpbsa_input_source = "缓存恢复"

            if uploaded_pocket_entry is not None:
                pocket_input_source = "本次上传"
            elif resolved_pocket_text:
                pocket_input_source = "缓存恢复"

        if not mmpbsa_texts:
            mmpbsa_texts = [None for _ in pdb_texts]
        elif len(mmpbsa_texts) < len(pdb_texts):
            last = mmpbsa_texts[-1]
            mmpbsa_texts = mmpbsa_texts + [last] * (len(pdb_texts) - len(mmpbsa_texts))

    if not pdb_texts:
        st.warning("请上传至少一个 PDB 文件，或在侧边栏勾选“使用示例数据”。")
        st.stop()

    # 解析所有构象数据
    energy_tables = []
    atom_dfs = []
    energy_dfs = []
    energy_sources = []
    for i, pdb_text in enumerate(pdb_texts):
        energy_source = "无可用能量数据"
        try:
            atom_df = parse_pdb_atoms(pdb_text)
            mmpbsa_text = mmpbsa_texts[i] if i < len(mmpbsa_texts) else None
            energy_df, energy_source = resolve_energy_table(
                pdb_text,
                energy_mode=energy_mode,
                mmpbsa_text=mmpbsa_text,
            )
            if energy_df is not None and not energy_df.empty:
                energy_table = prepare_energy_table(atom_df, energy_df)
                energy_table["energy_source"] = energy_source
            else:
                energy_table = pd.DataFrame()
        except Exception as exc:
            LOGGER.exception("解析构象失败")
            atom_df = pd.DataFrame()
            energy_df = pd.DataFrame()
            energy_table = pd.DataFrame()
        atom_dfs.append(atom_df)
        energy_dfs.append(energy_df)
        energy_tables.append(energy_table)
        energy_sources.append(energy_source)

    # 侧边栏显示控制（基于第一构象的能量范围作为参考）
    sample_table = next((t for t in energy_tables if not t.empty), None)
    energy_limit = float(max(0.1, abs(sample_table["delta_total"].min()) if sample_table is not None else 0.1, abs(sample_table["delta_total"].max()) if sample_table is not None else 0.1))

    with st.sidebar:
        st.header("显示控制")
        threshold = st.slider("MMPBSA |阈值| (绝对值)", 0.0, energy_limit, 0.0, 0.1)
        display_mode = st.radio(
            "显示模式",
            ["cartoon", "sticks", "surface"],
            format_func=lambda x: {"cartoon": "卡通", "sticks": "球棍", "surface": "表面"}[x],
        )
        color_mode = st.selectbox(
            "颜色分类方式",
            [
                "按DELTA TOTAL 热度",
                "按氨基酸理化性质",
                "按电荷状态",
                "按侧链极性",
                "按MMPBSA等级",
                "按热点等级",
                "按口袋识别",
                "按链",
                "单色",
            ],
            index=0,
        )
        classification_theme = CLASSIFICATION_THEME_OPTIONS[0] if CLASSIFICATION_THEME_OPTIONS else None
        st.caption("分类颜色为固定映射：同一分类在不同结构中保持同色。")
        chain_palette = st.selectbox(
            "链颜色方案",
            ["经典", "柔和", "高对比"],
            index=0,
            disabled=color_mode != "按链",
        )
        mono_color = st.selectbox(
            "单色颜色",
            ["#d1d5db", "#ef4444", "#2563eb", "#10b981", "#f97316"],
            index=0,
            disabled=color_mode != "单色",
            format_func=lambda x: {
                "#d1d5db": "灰色",
                "#ef4444": "红色",
                "#2563eb": "蓝色",
                "#10b981": "绿色",
                "#f97316": "橙色",
            }[x],
        )
        opacity = st.slider("表面透明度", 0.0, 1.0, SETTINGS.default_opacity, 0.05, disabled=display_mode != "surface")
        show_backbone = st.checkbox("显示主链", value=True, disabled=display_mode != "cartoon")
        surface_single_color = color_mode == "单色"
        surface_uniform_color = mono_color if surface_single_color else SETTINGS.neutral_color
        if display_mode == "surface":
            if surface_single_color:
                st.caption("当前为单色模式：表面使用统一颜色。")
            else:
                st.caption("当前为精准分类着色：表面直接使用当前分类颜色。")

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
    pocket_detection_summary: dict[str, object] = {}
    try:
        if resolved_pocket_text:
            pocket_df = parse_pocket_table(resolved_pocket_text)
        elif use_examples:
            pocket_df = parse_pocket_table(POCKET_TEXT)
        else:
            pocket_df = pd.DataFrame()
        if not pocket_df.empty:
            pocket_summary = build_pocket_summary(pocket_df, hotspot_df)
        elif not use_examples:
            try:
                auto_pocket_df = detect_auto_pocket_table(
                    pdb_texts[selected_conf],
                    hotspot_residues=[(row.chain, int(row.resid)) for row in hotspot_df.itertuples(index=False)],
                )
                pocket_detection_summary = summarize_pocket_detection_metadata(
                    get_pocket_detection_metadata(auto_pocket_df)
                )
            except Exception:
                auto_pocket_df = pd.DataFrame()
                pocket_detection_summary = {}
            if not auto_pocket_df.empty:
                pocket_df = auto_pocket_df
                pocket_summary = build_pocket_summary(pocket_df, hotspot_df)
                resolved_pocket_text = "__AUTO_CONSENSUS__"
    except Exception:
        pocket_df = None

    pocket_residues = []
    if pocket_df is not None and not pocket_df.empty:
        pocket_residues = [(row.chain, int(row.resid)) for row in pocket_df.itertuples(index=False)]

    if energy_mode == "mmpbsa" and all(t.empty for t in energy_tables):
        st.warning("当前选择为上传 MMPBSA 模式，但未提供可用的 MMPBSA 数据。")

    hotspot_rank_map = {
        (row.chain, int(row.resid)): int(row.hotspot_rank)
        for row in hotspot_df.itertuples(index=False)
        if getattr(row, "hotspot_rank", None) is not None
    }

    annotation_table = build_legacy_annotation_table(
        current_table,
        color_mode,
        palette_name=chain_palette,
        mono_color=mono_color,
        hotspot_df=hotspot_df,
        pocket_residues=pocket_residues,
        theme_name=classification_theme,
    )
    classification_summary = "-"
    if not annotation_table.empty and "classification_label" in annotation_table.columns:
        classification_counts = (
            annotation_table.groupby("classification_label", dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        summary_parts = [f"{label}: {int(count)}" for label, count in classification_counts.head(3).items()]
        classification_summary = "；".join(summary_parts) if summary_parts else "-"

    # 将当前分析写入会话状态（仅保存当前构象的结果）
    try:
        summary = build_analysis_summary(current_table)
        try:
            protein_volume = estimate_protein_volume(pdb_texts[selected_conf])
        except Exception:
            protein_volume = None
        energy_source_label = summary.get("energy_source") or (energy_sources[selected_conf] if energy_sources else None) or "未知"
        stored_mmpbsa_text = mmpbsa_texts[selected_conf] if (selected_conf < len(mmpbsa_texts) and mmpbsa_texts[selected_conf]) else "结构估算（未上传 MMPBSA 文件）"
        stored_pocket_table = pocket_df if pocket_df is not None else pd.DataFrame()
        top_pocket = pocket_summary.iloc[0] if not pocket_summary.empty else None
        set_analysis_state(
            pdb_texts[selected_conf],
            stored_mmpbsa_text,
            atom_dfs[selected_conf],
            energy_dfs[selected_conf],
            current_table,
            annotation_table=annotation_table,
            pocket_table=stored_pocket_table,
            pocket_summary=pocket_summary,
            color_mode=color_mode,
        )
        append_history_record(
            {
                "generated_at": summary["generated_at"],
                "source_name": f"构象 {selected_conf+1}",
                "energy_source_name": f"构象 {selected_conf+1} {energy_source_label}",
                "residue_count": summary["residue_count"],
                "min_energy": summary["min_energy"],
                "max_energy": summary["max_energy"],
                "mean_energy": summary["mean_energy"],
                "lowest_residue": summary["lowest_residue"],
                "highest_residue": summary["highest_residue"],
                "valid_energy_count": summary["valid_energy_count"],
                "energy_coverage": summary["energy_coverage"],
                "protein_volume": protein_volume,
                "display_mode": display_mode,
                "color_mode": color_mode,
                "hotspot_count": int(len(hotspot_df)),
                "pocket_count": int(len(pocket_residues)),
                "annotation_rows": int(len(annotation_table)),
                **pocket_detection_summary,
                "top_pocket_id": str(top_pocket.get("pocket_id")) if top_pocket is not None and pd.notna(top_pocket.get("pocket_id")) else None,
                "top_pocket_smart_rank_label": str(top_pocket.get("smart_rank_label")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_label")) else None,
                "top_pocket_smart_rank_score": float(top_pocket.get("smart_rank_score")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_score")) else None,
                "top_pocket_hotspot_count": int(top_pocket.get("hotspot_count")) if top_pocket is not None and pd.notna(top_pocket.get("hotspot_count")) else None,
                "top_pocket_detection_route": str(top_pocket.get("detection_route")) if top_pocket is not None and pd.notna(top_pocket.get("detection_route")) else None,
                "top_pocket_reason": str(top_pocket.get("smart_rank_reason")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_reason")) else None,
                "classification_summary": classification_summary,
            }
        )
    except Exception:
        LOGGER.exception("写入会话状态失败")

    col1, col2 = st.columns([2.4, 1.0])

    with col1:
        viewer = build_view(
            pdb_text=pdb_texts[selected_conf],
            energy_table=annotation_table,
            threshold=threshold,
            display_mode=display_mode,
            show_backbone=show_backbone,
            opacity=opacity,
            selected_chain=None,
            selected_resid=None,
            color_mode=color_mode,
            surface_single_color=surface_single_color,
            surface_uniform_color=surface_uniform_color,
        )
        st.components.v1.html(viewer._make_html(), height=SETTINGS.viewer_height + 20, scrolling=False)

    with col2:
        st.subheader("当前状态")
        highlighted_count = int((current_table["delta_total"].abs() >= threshold).sum())
        st.metric("高亮残基数", f"{highlighted_count}/{len(current_table)}")
        st.metric("当前模式", {"cartoon": "卡通", "sticks": "球棍", "surface": "表面"}[display_mode])
        st.metric("颜色模式", color_mode)
        energy_metric_label = "平均能量（估算）" if summary.get("energy_source") == "结构估算" else "平均能量"
        st.metric(energy_metric_label, format_energy_value(summary["mean_energy"]))
        protein_volume_text = f"{protein_volume:,.1f} A³" if protein_volume is not None else "-"
        st.metric("蛋白质体积（估算）", protein_volume_text)
        if summary.get("energy_source"):
            source_note = "（不是标准 MMPBSA）" if summary.get("energy_source") == "结构估算" else ""
            st.caption(f"能量来源：{summary.get('energy_source')}{source_note}")
        st.caption(f"基于 {summary['valid_energy_count']}/{summary['residue_count']} 个有效能量值计算")

        legend_items, legend_note = build_legacy_legend(
            color_mode,
            annotation_table,
            palette_name=chain_palette,
            hotspot_rank_map=hotspot_rank_map,
            theme_name=classification_theme,
        )
        if legend_items:
            st.caption("图例")
            for item in legend_items:
                color_hex = str(item.get("color") or "#9ca3af")
                label_text = str(item.get("label") or "")
                count_text = item.get("count")
                if count_text is None:
                    count_suffix = ""
                else:
                    count_suffix = f" ({int(count_text)})"
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
                    f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:{color_hex};'></span>"
                    f"<span style='font-size:12px;'>{label_text}{count_suffix}</span></div>",
                    unsafe_allow_html=True,
                )
        if legend_note:
            st.caption(legend_note)

    st.markdown("---")
    st.subheader("残基能量明细")
    display_cols = ["chain", "resid", "resname", "delta_total", "classification_label", "display_color"]
    available_cols = [c for c in display_cols if c in annotation_table.columns]
    df_display = annotation_table[available_cols].copy() if available_cols else annotation_table.copy()
    df_display = df_display.sort_values(["chain", "resid"]).reset_index(drop=True)
    df_display.insert(0, "rank", np.arange(1, len(df_display) + 1))
    st.dataframe(localize_display_table(df_display), use_container_width=True, height=300)

    st.subheader("热点摘要")
    if hotspot_df.empty:
        st.info("在当前阈值下未识别到热点残基。")
    else:
        show_cols = [
            c
            for c in [
                "chain",
                "resid",
                "resname",
                "delta_total",
                "hotspot_score",
                "hotspot_rank",
                "neighborhood_count",
                "cluster_id",
            ]
            if c in hotspot_df.columns
        ]
        hotspot_display = hotspot_df[show_cols].sort_values(["hotspot_rank", "chain", "resid"])
        st.dataframe(localize_display_table(hotspot_display), use_container_width=True, height=220)

    if isinstance(hotspot_clusters, dict) and hotspot_clusters:
        st.caption("热点聚类提示")
        cluster_count = int(hotspot_clusters.get("count", 0) or 0)
        lowest_hotspot = str(hotspot_clusters.get("lowest_hotspot") or "-")
        cluster_hint = str(hotspot_clusters.get("cluster_hint") or "")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("热点数量", str(cluster_count))
        metric_col2.metric("最低能量热点", lowest_hotspot)
        if cluster_hint:
            st.caption(cluster_hint)

    if pocket_df is not None and not pocket_df.empty:
        st.subheader("口袋摘要")
        if resolved_pocket_text == "__AUTO_CONSENSUS__":
            st.caption("未上传 Pocket 文件，已使用自动共识口袋检测。")
        st.dataframe(localize_display_table(pocket_summary), use_container_width=True, height=180)

    if compare_mode and len(energy_tables) > 1:
        st.markdown("---")
        st.subheader("多构象热点比较")
        hotspot_tables = []
        for idx, table in enumerate(energy_tables):
            if table.empty:
                hotspot_tables.append(pd.DataFrame())
                continue
            hs = identify_hotspots(table, energy_threshold=-abs(threshold) if threshold > 0 else -1.0)
            hotspot_tables.append(hs)
        comparison = compare_hotspot_sets(hotspot_tables)
        comparison_table = comparison.get("per_residue_df", pd.DataFrame())
        st.dataframe(localize_display_table(comparison_table), use_container_width=True, height=180)
        st.markdown(explain_comparison(comparison))

    st.markdown("---")
    st.subheader("分析解释")
    st.markdown(explain_analysis(current_table, hotspot_df, pocket_summary))
    return
