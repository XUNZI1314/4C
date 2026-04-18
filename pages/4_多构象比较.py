from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.sample_data import MMPBSA_TEXT, MMPBSA_TEXT_ALT, PDB_TEXT, PDB_TEXT_ALT
from protein_visualizer.services.comparison import build_hotspot_stability_tables, compare_hotspot_sets, build_pairwise_similarity_matrix, build_reference_comparison_table
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.parsers import parse_pdb_atoms
from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value
from protein_visualizer.services.snapshot import build_analysis_snapshot, build_snapshot_svg, snapshot_to_json_bytes
from protein_visualizer.services.structure_energy import estimate_protein_volume, resolve_energy_table


DISPLAY_COLUMN_LABELS = {
    "conformation": "构象",
    "energy_source": "能量来源",
    "residue_count": "残基数",
    "valid_energy_count": "有效能量数",
    "mean_energy": "平均能量",
    "min_energy": "最低能量",
    "max_energy": "最高能量",
    "protein_volume": "蛋白体积",
    "energy_coverage": "能量覆盖率",
    "hotspot_count": "热点数",
    "is_reference": "是否参考",
    "mean_energy_delta_vs_reference": "相对参考平均能量差",
    "hotspot_count_delta_vs_reference": "相对参考热点数差",
    "reference_overlap_count": "参考重叠数",
    "reference_overlap_ratio": "参考重叠率",
    "unique_hotspot_count": "新增热点数",
    "chain": "链",
    "resid": "残基编号",
    "resname": "残基名",
    "count": "出现次数",
    "frequency": "出现频率",
    "label": "残基",
    "is_common": "是否共同热点",
}


DISPLAY_COLUMN_TOKEN_LABELS = {
    "conformation": "构象",
    "energy": "能量",
    "source": "来源",
    "residue": "残基",
    "count": "数量",
    "valid": "有效",
    "mean": "平均",
    "min": "最低",
    "max": "最高",
    "protein": "蛋白",
    "volume": "体积",
    "coverage": "覆盖率",
    "hotspot": "热点",
    "is": "是否",
    "reference": "参考",
    "delta": "差值",
    "vs": "相对",
    "overlap": "重叠",
    "ratio": "比例",
    "unique": "新增",
    "chain": "链",
    "resid": "残基编号",
    "resname": "残基名",
    "frequency": "频率",
    "label": "标签",
    "common": "共同",
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


def localize_display_table(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_bool_dtype(display[column]):
            display[column] = display[column].map(lambda value: "是" if bool(value) else "否")
    return display.rename(columns={column: localize_column_name(column) for column in display.columns})


st.set_page_config(page_title="多构象比较", layout="wide")
st.title("多构象比较")
st.caption("用于比较多个构象的热点稳定性、能量来源和结构尺度，并支持自动 / MMPBSA / 结构估算三种能量来源模式。")

with st.sidebar:
    st.header("输入与模式")
    uploaded_pdbs = st.file_uploader("上传 PDB 文件（可多选）", type=["pdb"], accept_multiple_files=True)
    uploaded_mmpbs = st.file_uploader(
        "上传 MMPBSA 文件（可多选，按构象顺序）",
        type=["txt", "dat", "out", "csv"],
        accept_multiple_files=True,
    )
    use_examples = st.checkbox("使用示例数据", value=False)
    energy_mode = st.selectbox(
        "能量来源模式",
        ["auto", "mmpbsa", "estimate"],
        index=0,
        format_func=lambda x: {"auto": "自动", "mmpbsa": "上传 MMPBSA", "estimate": "结构估算"}[x],
    )

if uploaded_pdbs:
    pdb_texts = [file.getvalue().decode("utf-8", errors="ignore") for file in uploaded_pdbs]
elif use_examples:
    pdb_texts = [PDB_TEXT, PDB_TEXT_ALT]
else:
    pdb_texts = []

if uploaded_mmpbs:
    mmpbsa_texts = [file.getvalue().decode("utf-8", errors="ignore") for file in uploaded_mmpbs]
elif use_examples:
    mmpbsa_texts = [MMPBSA_TEXT, MMPBSA_TEXT_ALT]
else:
    mmpbsa_texts = []

if len(mmpbsa_texts) < len(pdb_texts):
    last = mmpbsa_texts[-1] if mmpbsa_texts else None
    mmpbsa_texts = mmpbsa_texts + [last] * (len(pdb_texts) - len(mmpbsa_texts))

energy_tables = []
per_conformation_rows = []

for index, pdb_text in enumerate(pdb_texts):
    mmpbsa_text = mmpbsa_texts[index] if index < len(mmpbsa_texts) else None
    try:
        atom_df = parse_pdb_atoms(pdb_text)
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
            energy_source = "无可用能量数据"
    except Exception as exc:
        st.error(f"解析第 {index + 1} 个构象失败：{exc}")
        atom_df = pd.DataFrame()
        energy_table = pd.DataFrame()
        energy_source = "解析失败"

    try:
        protein_volume = estimate_protein_volume(pdb_text)
    except Exception:
        protein_volume = None

    if not energy_table.empty:
        summary = build_analysis_summary(energy_table)
    else:
        summary = {
            "residue_count": 0,
            "mean_energy": None,
            "valid_energy_count": 0,
            "energy_source": energy_source,
            "energy_coverage": 0.0,
            "min_energy": None,
            "max_energy": None,
        }

    energy_tables.append(energy_table)
    per_conformation_rows.append(
        {
            "conformation": f"构象 {index + 1}",
            "energy_source": summary.get("energy_source") or energy_source,
            "residue_count": summary.get("residue_count", 0),
            "valid_energy_count": summary.get("valid_energy_count", 0),
            "mean_energy": summary.get("mean_energy"),
            "min_energy": summary.get("min_energy"),
            "max_energy": summary.get("max_energy"),
            "protein_volume": protein_volume,
            "energy_coverage": summary.get("energy_coverage", 0.0),
        }
    )

if not energy_tables:
    st.warning("没有可比较的构象。请上传 PDB 或勾选使用示例数据。")
    st.stop()

energy_limits = []
for table in energy_tables:
    if table.empty or "delta_total" not in table.columns:
        continue
    series = pd.to_numeric(table["delta_total"], errors="coerce").dropna()
    if not series.empty:
        energy_limits.append(float(max(abs(series.min()), abs(series.max()))))

energy_limit = max([0.1, *energy_limits]) if energy_limits else 0.1

with st.sidebar:
    st.header("热点筛选")
    hotspot_threshold = st.slider(
        "MMPBSA |阈值| (绝对值)",
        0.0,
        energy_limit,
        min(1.0, energy_limit),
        0.1,
    )
    hotspot_top_n = st.slider("热点保底数量", 1, 20, 5, 1)
    reference_index = st.selectbox("参考构象", list(range(len(pdb_texts))), index=0, format_func=lambda i: f"构象 {i + 1}")

hotspot_cutoff = -abs(hotspot_threshold) if hotspot_threshold > 0 else -1.0
hotspot_tables = [
    identify_hotspots(table, energy_threshold=hotspot_cutoff, top_n=hotspot_top_n) if not table.empty else pd.DataFrame()
    for table in energy_tables
]
for row, hotspot_df in zip(per_conformation_rows, hotspot_tables):
    row["hotspot_count"] = int(len(hotspot_df))

comparison = compare_hotspot_sets(hotspot_tables)
reference_comparison_df = build_reference_comparison_table(per_conformation_rows, hotspot_tables, reference_index=reference_index)
stability_views = build_hotspot_stability_tables(comparison["per_residue_df"], stable_threshold=0.67)
stable_hotspots_df = stability_views["stable_hotspots"]
variable_hotspots_df = stability_views["variable_hotspots"]
common_hotspots_df = stability_views["common_hotspots"]
pairwise_matrix = build_pairwise_similarity_matrix(hotspot_tables)
per_conformation_df = pd.DataFrame(per_conformation_rows)
reference_label = per_conformation_rows[reference_index].get("conformation", f"构象 {reference_index + 1}") if per_conformation_rows else "构象 1"


def _render_combo_chart(
    frame: pd.DataFrame,
    *,
    x_col: str,
    bar_col: str,
    line_col: str,
    title: str,
    bar_name: str,
    line_name: str,
    bar_color: str,
    line_color: str,
) -> None:
    if frame.empty:
        st.info("没有足够的数据生成趋势图。")
        return

    chart_frame = frame[[x_col, bar_col, line_col]].copy()
    if go is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=chart_frame[x_col],
                y=chart_frame[bar_col],
                name=bar_name,
                marker_color=bar_color,
                opacity=0.78,
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_frame[x_col],
                y=chart_frame[line_col],
                name=line_name,
                mode="lines+markers",
                line=dict(color=line_color, width=3),
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=title,
            margin=dict(l=30, r=30, t=48, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(title="构象", tickangle=-15),
            yaxis=dict(title=bar_name),
            yaxis2=dict(title=line_name, overlaying="y", side="right"),
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fallback_chart = chart_frame.rename(columns={x_col: "构象", bar_col: bar_name, line_col: line_name})
        st.bar_chart(fallback_chart.set_index("构象")[[bar_name]])
        st.line_chart(fallback_chart.set_index("构象")[[line_name]])

st.success(f"已加载 {len(energy_tables)} 个构象。")

metric_cols = st.columns(4)
metric_cols[0].metric("构象数量", len(energy_tables))
metric_cols[1].metric("热点并集", comparison["union_size"])
metric_cols[2].metric("共同热点", comparison["intersection_size"])
metric_cols[3].metric("一致性得分", f"{comparison['consistency_score']:.2f}")

snapshot_index = next((index for index, table in enumerate(energy_tables) if not table.empty), 0)
snapshot_energy_table = energy_tables[snapshot_index]
snapshot_hotspot_df = hotspot_tables[snapshot_index] if hotspot_tables else None

comparison_snapshot = build_analysis_snapshot(
    snapshot_energy_table,
    title="ProteinInsight 多构象比较快照",
    hotspot_df=snapshot_hotspot_df,
    comparison_df=comparison["per_residue_df"],
    protein_volume=per_conformation_rows[snapshot_index].get("protein_volume") if per_conformation_rows else None,
    extra={
        "energy_mode": energy_mode,
        "hotspot_threshold": hotspot_threshold,
        "hotspot_top_n": hotspot_top_n,
        "reference_index": reference_index,
        "reference_label": reference_label,
        "conformation_count": len(energy_tables),
        "union_size": comparison["union_size"],
        "intersection_size": comparison["intersection_size"],
        "consistency_score": comparison["consistency_score"],
        "pairwise_matrix": pairwise_matrix.to_dict(orient="records"),
        "reference_comparison": reference_comparison_df.to_dict(orient="records"),
        "stable_hotspots": stable_hotspots_df.to_dict(orient="records"),
        "variable_hotspots": variable_hotspots_df.to_dict(orient="records"),
        "common_hotspots_table": common_hotspots_df.to_dict(orient="records"),
        "per_conformation": per_conformation_rows,
    },
)

tab_overview, tab_trend, tab_reference, tab_matrix, tab_hotspots, tab_export = st.tabs(["总体", "趋势洞察", "参考对比", "成对相似度", "热点明细", "导出"])

with tab_overview:
    st.subheader("构象摘要")
    st.dataframe(localize_display_table(per_conformation_df), use_container_width=True, hide_index=True)
    st.markdown(f"**比较说明**：{comparison['consistency_score']:.2f} 的一致性得分反映了热点在不同构象中的稳定程度。")
    if comparison.get("common_hotspots"):
        st.write("共同热点示例：", ", ".join(comparison["common_hotspots"][:8]))
    st.subheader("自动解释")
    st.info(
        f"在 {comparison['total_conformations']} 个构象中检测到 {comparison['union_size']} 个候选热点，其中 {comparison['intersection_size']} 个在所有构象中一致出现。"
    )
    st.caption(f"当前参考构象：{reference_label}")

with tab_trend:
    st.subheader("趋势洞察")
    st.caption(f"以 {reference_label} 作为参考，观察其它构象在能量、热点和重叠上的变化。")

    trend_df = reference_comparison_df.copy()
    if trend_df.empty:
        st.info("没有足够的数据生成趋势洞察。")
    else:
        for column in [
            "mean_energy",
            "mean_energy_delta_vs_reference",
            "hotspot_count",
            "hotspot_count_delta_vs_reference",
            "reference_overlap_ratio",
            "reference_overlap_count",
            "unique_hotspot_count",
            "energy_coverage",
            "protein_volume",
        ]:
            if column in trend_df.columns:
                trend_df[column] = pd.to_numeric(trend_df[column], errors="coerce")

        non_reference_df = trend_df[~trend_df["is_reference"]].copy() if "is_reference" in trend_df.columns else trend_df.copy()
        if non_reference_df.empty:
            non_reference_df = trend_df.copy()

        overlap_series = pd.to_numeric(non_reference_df.get("reference_overlap_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
        closest_row = non_reference_df.loc[overlap_series.idxmax()] if not overlap_series.empty else non_reference_df.iloc[0]

        energy_series = pd.to_numeric(trend_df.get("mean_energy", pd.Series(dtype=float)), errors="coerce").dropna()
        lowest_energy_row = trend_df.loc[energy_series.idxmin()] if not energy_series.empty else trend_df.iloc[0]

        unique_series = pd.to_numeric(non_reference_df.get("unique_hotspot_count", pd.Series(dtype=float)), errors="coerce").dropna()
        least_unique_row = non_reference_df.loc[unique_series.idxmin()] if not unique_series.empty else non_reference_df.iloc[0]

        trend_metrics = st.columns(4)
        trend_metrics[0].metric(
            "最接近参考",
            closest_row.get("conformation", "-"),
            f"{float(closest_row.get('reference_overlap_ratio', 0.0)):.2f} 重叠",
        )
        trend_metrics[1].metric(
            "最低平均能量",
            lowest_energy_row.get("conformation", "-"),
            format_energy_value(lowest_energy_row.get("mean_energy")),
        )
        trend_metrics[2].metric(
            "最少新热点",
            least_unique_row.get("conformation", "-"),
            f"{int(least_unique_row.get('unique_hotspot_count', 0))}",
        )
        trend_metrics[3].metric(
            "平均热点覆盖",
            f"{float(trend_df['reference_overlap_ratio'].fillna(0.0).mean()):.2f}" if "reference_overlap_ratio" in trend_df.columns else "-",
            f"参考构象：{reference_label}",
        )

        chart_left, chart_right = st.columns(2)
        with chart_left:
            _render_combo_chart(
                trend_df,
                x_col="conformation",
                bar_col="hotspot_count",
                line_col="mean_energy",
                title="热点数与平均能量趋势",
                bar_name="热点数",
                line_name="平均能量",
                bar_color="#2563eb",
                line_color="#f97316",
            )
        with chart_right:
            _render_combo_chart(
                trend_df,
                x_col="conformation",
                bar_col="reference_overlap_ratio",
                line_col="unique_hotspot_count",
                title="参考重叠率与新增热点趋势",
                bar_name="重叠率",
                line_name="新增热点数",
                bar_color="#10b981",
                line_color="#7c3aed",
            )

        summary_cols = st.columns(2)
        with summary_cols[0]:
            st.subheader("稳定热点（频率 >= 0.67）")
            if stable_hotspots_df.empty:
                st.info("没有稳定热点。")
            else:
                st.dataframe(localize_display_table(stable_hotspots_df.head(10)), use_container_width=True, hide_index=True)
                st.download_button(
                    "导出稳定热点 CSV",
                    data=stable_hotspots_df.to_csv(index=False).encode("utf-8"),
                    file_name="stable_hotspots.csv",
                    mime="text/csv",
                )
        with summary_cols[1]:
            st.subheader("可变热点（频率 < 0.67）")
            if variable_hotspots_df.empty:
                st.info("没有可变热点。")
            else:
                st.dataframe(localize_display_table(variable_hotspots_df.head(10)), use_container_width=True, hide_index=True)
                st.download_button(
                    "导出可变热点 CSV",
                    data=variable_hotspots_df.to_csv(index=False).encode("utf-8"),
                    file_name="variable_hotspots.csv",
                    mime="text/csv",
                )

with tab_reference:
    st.subheader("相对于参考构象的变化")
    st.caption(f"以 {reference_label} 作为基准，查看其它构象在能量和热点上的变化。")
    reference_metrics = reference_comparison_df[reference_comparison_df["is_reference"]] if not reference_comparison_df.empty else pd.DataFrame()
    if not reference_metrics.empty:
        ref_row = reference_metrics.iloc[0]
        metric_cols = st.columns(4)
        metric_cols[0].metric("参考热点数", int(ref_row.get("hotspot_count", 0)))
        metric_cols[1].metric("参考能量", format_energy_value(ref_row.get("mean_energy")))
        metric_cols[2].metric("热点覆盖率", f"{float(ref_row.get('reference_overlap_ratio', 0.0)):.2f}")
        metric_cols[3].metric("唯一热点数", int(ref_row.get("unique_hotspot_count", 0)))

    if reference_comparison_df.empty:
        st.info("没有足够的数据生成参考对比表。")
    else:
        st.dataframe(localize_display_table(reference_comparison_df), use_container_width=True, hide_index=True)
        st.caption("正值的能量差表示相对参考构象更高，热点数差表示相对参考构象增减的热点数量。")

with tab_matrix:
    st.subheader("构象两两热点相似度矩阵")
    if pairwise_matrix.empty:
        st.info("没有足够的数据生成相似度矩阵。")
    else:
        st.dataframe(localize_display_table(pairwise_matrix), use_container_width=True, hide_index=True)

with tab_hotspots:
    st.subheader("热点明细")
    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.caption("稳定热点：在多数构象中反复出现的残基。")
        if stable_hotspots_df.empty:
            st.info("没有稳定热点。")
        else:
            st.dataframe(localize_display_table(stable_hotspots_df), use_container_width=True, hide_index=True)
    with detail_right:
        st.caption("可变热点：仅在部分构象中出现的残基。")
        if variable_hotspots_df.empty:
            st.info("没有可变热点。")
        else:
            st.dataframe(localize_display_table(variable_hotspots_df), use_container_width=True, hide_index=True)

    st.subheader("共同热点")
    if common_hotspots_df.empty:
        st.info("没有共同热点统计可展示。")
    else:
        st.dataframe(localize_display_table(common_hotspots_df), use_container_width=True, hide_index=True)

with tab_export:
    st.subheader("导出比较结果")
    st.download_button(
        "导出构象摘要 CSV",
        data=per_conformation_df.to_csv(index=False).encode("utf-8"),
        file_name="conformation_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出相似度矩阵 CSV",
        data=pairwise_matrix.to_csv(index=False).encode("utf-8"),
        file_name="pairwise_similarity_matrix.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出热点统计 CSV",
        data=comparison["per_residue_df"].to_csv(index=False).encode("utf-8"),
        file_name="hotspot_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出稳定热点 CSV",
        data=stable_hotspots_df.to_csv(index=False).encode("utf-8"),
        file_name="stable_hotspots.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出可变热点 CSV",
        data=variable_hotspots_df.to_csv(index=False).encode("utf-8"),
        file_name="variable_hotspots.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出参考对比 CSV",
        data=reference_comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="reference_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        "导出比较快照 JSON",
        data=snapshot_to_json_bytes(comparison_snapshot),
        file_name="multi_conformation_comparison_snapshot.json",
        mime="application/json",
    )
    st.download_button(
        "导出比较快照 SVG",
        data=build_snapshot_svg(comparison_snapshot),
        file_name="multi_conformation_comparison_snapshot.svg",
        mime="image/svg+xml",
    )

st.info("提示：自动模式会优先使用上传的 MMPBSA 数据，没有上传时会回退到结构估算；示例数据需要手动勾选。")
