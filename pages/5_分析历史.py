from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.services.session_state import get_history_records, initialize_state
from protein_visualizer.services.reporting import format_energy_value


DISPLAY_COLUMN_LABELS = {
    "generated_at": "生成时间",
    "source_name": "PDB 来源",
    "energy_source_name": "能量来源",
    "display_mode": "显示模式",
    "color_mode": "颜色模式",
    "residue_count": "残基数",
    "valid_energy_count": "有效能量数",
    "energy_coverage": "能量覆盖率",
    "protein_volume": "蛋白体积",
    "hotspot_count": "热点数",
    "pocket_count": "口袋残基数",
    "annotation_rows": "注释行数",
    "auto_detection_methods_used": "自动口袋方法",
    "auto_detection_result_pocket_count": "自动识别口袋数",
    "auto_detection_result_residue_rows": "自动识别残基行数",
    "auto_detection_p2rank_status": "P2Rank 状态",
    "auto_detection_p2rank_prediction_rows": "P2Rank 预测行数",
    "auto_detection_p2rank_residue_rows": "P2Rank 残基行数",
    "auto_detection_external_rows": "外部证据行数",
    "auto_detection_external_sources": "外部证据来源",
    "auto_detection_status_summary": "自动检测状态汇总",
    "top_pocket_id": "Top1 口袋",
    "top_pocket_smart_rank_label": "Top1 口袋标签",
    "top_pocket_smart_rank_score": "Top1 口袋得分",
    "top_pocket_hotspot_count": "Top1 口袋热点数",
    "top_pocket_detection_route": "Top1 口袋识别路径",
    "top_pocket_reason": "Top1 口袋排序理由",
    "top_pocket_evidence_quality_label": "Top1 证据质量",
    "top_pocket_evidence_quality_score": "Top1 证据质量得分",
    "top_pocket_evidence_quality_warning": "Top1 证据质量提醒",
    "top_joint_pocket_id": "Top1 联合推荐口袋",
    "top_joint_recommendation_label": "Top1 联合推荐等级",
    "top_joint_recommendation_score": "Top1 联合推荐得分",
    "top_joint_reason": "Top1 联合推荐理由",
    "min_energy": "最低能量",
    "max_energy": "最高能量",
    "mean_energy": "平均能量",
    "lowest_residue": "最低能量残基",
    "highest_residue": "最高能量残基",
    "classification_summary": "分类摘要",
}


def localize_history_table(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    display = table.copy()
    return display.rename(columns={column: DISPLAY_COLUMN_LABELS.get(column, column) for column in display.columns})


st.set_page_config(page_title="分析历史", layout="wide")
st.title("分析历史")
st.caption("查看最近已保存的蛋白质结构分析记录。")

initialize_state()
history = get_history_records()

if not history:
    st.warning("当前还没有分析历史。请先前往“结构可视化”页完成至少一次分析。")
else:
    st.success(f"当前共保存最近 {len(history)} 条分析记录。")
    latest = history[0]
    st.subheader("最近一次分析摘要")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("最近分析时间", latest.get("generated_at", "-"))
    col2.metric("最近残基数", latest.get("residue_count", "-"))
    mean_energy = latest.get("mean_energy")
    top_pocket_id = latest.get("top_pocket_id", "-")
    mean_label = "最近平均能量（估算）" if latest.get("energy_source_name", "").find("结构估算") != -1 else "最近平均能量"
    col3.metric(mean_label, format_energy_value(mean_energy))
    col4.metric("最近颜色模式", latest.get("color_mode", "-"))

    col5.metric("Top1 口袋", top_pocket_id)

    col6.metric("Top1 联合推荐", latest.get("top_joint_pocket_id", "-"))

    volume_value = latest.get("protein_volume")
    st.metric("最近蛋白体积（估算）", f"{volume_value:,.1f} A³" if volume_value is not None else "-")

    st.markdown(f"- PDB 来源：`{latest.get('source_name', '-')}`")
    st.markdown(f"- MMPBSA 来源：`{latest.get('energy_source_name', '-')}`")
    st.markdown(f"- 显示模式：`{latest.get('display_mode', '-')}`")
    if latest.get("valid_energy_count") is not None:
        st.markdown(f"- 平均能量基于：`{latest.get('valid_energy_count', '-')}/{latest.get('residue_count', '-')}` 个有效能量值")
    if latest.get("energy_source_name", "").find("结构估算") != -1:
        st.markdown("- 说明：当前记录使用结构估算能量，不是标准 MMPBSA 文件。")
    st.markdown(f"- 热点数：`{latest.get('hotspot_count', '-')}` 口袋残基数：`{latest.get('pocket_count', '-')}` 注释行数：`{latest.get('annotation_rows', '-')}`")
    if latest.get("auto_detection_methods_used"):
        st.markdown(f"- 自动口袋方法：`{latest.get('auto_detection_methods_used')}`")
    if latest.get("auto_detection_status_summary"):
        st.markdown(f"- 自动检测状态：`{latest.get('auto_detection_status_summary')}`")
    if latest.get("auto_detection_p2rank_status"):
        st.markdown(
            f"- P2Rank：`{latest.get('auto_detection_p2rank_status')}` / "
            f"预测行 `{latest.get('auto_detection_p2rank_prediction_rows', 0)}` / "
            f"残基行 `{latest.get('auto_detection_p2rank_residue_rows', 0)}`"
        )
    if latest.get("auto_detection_external_rows") is not None:
        st.markdown(
            f"- 外部证据：`{latest.get('auto_detection_external_rows', 0)}` 条 / "
            f"精确 `{latest.get('auto_detection_external_exact_rows', 0)}` / "
            f"弱匹配 `{latest.get('auto_detection_external_weak_rows', 0)}`"
        )
    st.markdown(f"- 最低能量残基：`{latest.get('lowest_residue', '-')}`")
    st.markdown(f"- 最高能量残基：`{latest.get('highest_residue', '-')}`")
    if latest.get("top_pocket_id"):
        st.markdown(
            f"- 智能口袋 Top1：`{latest.get('top_pocket_id', '-')}` / "
            f"`{latest.get('top_pocket_smart_rank_label', '-')}` / "
            f"`{format_energy_value(latest.get('top_pocket_smart_rank_score'))}`"
        )
    if latest.get("top_pocket_reason"):
        st.markdown(f"- 口袋排序理由：`{latest.get('top_pocket_reason')}`")
    if latest.get("top_pocket_evidence_quality_label"):
        st.markdown(
            f"- Top1 证据质量：`{latest.get('top_pocket_evidence_quality_label')}` / "
            f"`{format_energy_value(latest.get('top_pocket_evidence_quality_score'))}`"
        )
    if latest.get("top_pocket_evidence_quality_warning"):
        st.markdown(f"- Top1 证据提醒：`{latest.get('top_pocket_evidence_quality_warning')}`")
    if latest.get("classification_summary"):
        st.markdown(f"- 分类摘要：`{latest.get('classification_summary')}`")

    st.subheader("历史记录表")
    if latest.get("top_joint_pocket_id"):
        st.markdown(
            f"- 联合推荐 Top1：`{latest.get('top_joint_pocket_id', '-')}` / "
            f"`{latest.get('top_joint_recommendation_label', '-')}` / "
            f"`{format_energy_value(latest.get('top_joint_recommendation_score'))}`"
        )
    if latest.get("top_joint_reason"):
        st.markdown(f"- 联合推荐理由：`{latest.get('top_joint_reason')}`")

    history_df = pd.DataFrame(history)
    preferred_columns = [
        "generated_at",
        "source_name",
        "energy_source_name",
        "display_mode",
        "color_mode",
        "residue_count",
        "valid_energy_count",
        "energy_coverage",
        "protein_volume",
        "hotspot_count",
        "pocket_count",
        "annotation_rows",
        "auto_detection_methods_used",
        "auto_detection_result_pocket_count",
        "auto_detection_result_residue_rows",
        "auto_detection_p2rank_status",
        "auto_detection_p2rank_prediction_rows",
        "auto_detection_p2rank_residue_rows",
        "auto_detection_external_rows",
        "auto_detection_external_sources",
        "auto_detection_status_summary",
        "top_pocket_id",
        "top_pocket_smart_rank_label",
        "top_pocket_smart_rank_score",
        "top_pocket_hotspot_count",
        "top_pocket_detection_route",
        "top_pocket_reason",
        "top_pocket_evidence_quality_label",
        "top_pocket_evidence_quality_score",
        "top_pocket_evidence_quality_warning",
        "top_joint_pocket_id",
        "top_joint_recommendation_label",
        "top_joint_recommendation_score",
        "top_joint_reason",
        "min_energy",
        "max_energy",
        "mean_energy",
        "lowest_residue",
        "highest_residue",
        "classification_summary",
    ]
    available_columns = [column for column in preferred_columns if column in history_df.columns]
    st.dataframe(localize_history_table(history_df[available_columns]), use_container_width=True)

    st.download_button(
        "导出历史记录 CSV",
        data=history_df[available_columns].to_csv(index=False).encode("utf-8"),
        file_name="analysis_history.csv",
        mime="text/csv",
    )

    st.info("历史记录会写入本地数据文件，刷新页面后仍可保留；如果删除数据文件或切换运行目录，历史会重新生成。")
