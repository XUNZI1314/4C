from __future__ import annotations

from datetime import datetime
import html
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.config.settings import SETTINGS
from protein_visualizer.sample_data import ANNOTATION_TEXT, MMPBSA_TEXT, PDB_TEXT, POCKET_TEXT
from protein_visualizer.services.ai_evidence import (
    apply_ai_review_decisions_to_audit,
    build_ai_evidence_audit_table,
    build_ai_followup_evidence_plan,
    build_ai_followup_prompt_bundle,
    build_ai_evidence_review_queue,
    build_ai_review_artifact_manifest,
    build_ai_review_artifact_bundle_zip,
    build_ai_review_checklist_markdown,
    build_ai_review_bundle_certificate_markdown,
    build_ai_review_bundle_readme_markdown,
    build_ai_review_bundle_verification_summary,
    build_ai_review_decision_template,
    build_ai_review_decision_outcome_table,
    build_ai_review_decision_validation_table,
    build_ai_review_ranking_delta,
    build_ai_review_round_report_markdown,
    build_ai_review_round_summary,
    build_ai_ranking_impact_summary,
    build_residue_evidence_consensus,
    fetch_ai_residue_evidence,
    filter_ai_evidence_for_ranking,
    parse_ai_review_decision_table,
    parse_ai_residue_evidence_payload,
    verify_ai_review_artifact_bundle_zip,
)
from protein_visualizer.services.benchmark import (
    build_pocket_benchmark_case_interpretation_summary,
    build_pocket_benchmark_case_interpretation_matrix,
    build_pocket_benchmark_case_interpretation_matrix_queue,
    build_pocket_benchmark_case_interpretation_matrix_summary,
    build_pocket_benchmark_dataset_interpretation_checklist_markdown,
    build_pocket_benchmark_dataset_interpretation_report_markdown,
    build_pocket_benchmark_dataset_interpretation,
    build_pocket_benchmark_dataset_interpretation_queue,
    build_pocket_benchmark_interpretation_summary,
    build_pocket_benchmark_case_summary,
    build_pocket_benchmark_dataset_summary,
    build_pocket_benchmark_details,
    build_pocket_benchmark_reference_quality_checklist_markdown,
    build_pocket_benchmark_reference_quality_issues,
    build_pocket_benchmark_reference_quality_summary,
    build_pocket_benchmark_reference_candidate_accepted_reference,
    build_pocket_benchmark_reference_candidate_review_checklist_markdown,
    build_pocket_benchmark_reference_candidate_review_decision_template,
    build_pocket_benchmark_reference_candidate_review_decision_validation,
    build_pocket_benchmark_reference_candidate_review_outcomes,
    build_pocket_benchmark_reference_candidate_review_queue,
    build_pocket_benchmark_reference_from_external_evidence,
    build_pocket_benchmark_reference_import_summary,
    build_pocket_benchmark_reference_source_audit,
    build_pocket_benchmark_reference_source_audit_action_queue,
    build_pocket_benchmark_reference_source_audit_case_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_closure_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_closure_queue,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_outcomes,
    build_pocket_benchmark_reference_source_audit_case_decision_outcome_summary,
    build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact,
    build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary,
    build_pocket_benchmark_reference_source_audit_case_decision_template,
    build_pocket_benchmark_reference_source_audit_case_decision_validation,
    build_pocket_benchmark_reference_source_audit_case_summary,
    build_pocket_benchmark_reference_source_audit_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_summary,
    build_pocket_benchmark_reference_readiness_case_summary,
    build_pocket_benchmark_reference_readiness_checklist_markdown,
    build_pocket_benchmark_reference_readiness_queue,
    build_pocket_benchmark_reference_readiness_summary,
    build_pocket_benchmark_reference_structure_validation,
    build_pocket_benchmark_reference_structure_validation_checklist_markdown,
    build_pocket_benchmark_reference_structure_validation_summary,
    build_pocket_benchmark_reference_template,
    build_pocket_benchmark_reference_template_markdown,
    build_pocket_benchmark_summary,
    build_pocket_benchmark_variant_comparison,
    build_pocket_benchmark_variant_case_comparison,
    build_pocket_benchmark_variant_dataset_comparison,
    build_pocket_benchmark_variant_detail_comparison,
    build_pocket_benchmark_variant_remediation_checklist_markdown,
    build_pocket_benchmark_variant_remediation_queue,
    build_pocket_benchmark_variant_remediation_summary,
    parse_benchmark_reference_table,
    parse_pocket_benchmark_reference_candidate_review_decision_table,
    parse_pocket_benchmark_reference_source_audit_case_decision_table,
    select_pocket_benchmark_reference_source,
)
from protein_visualizer.services.candidate_fusion import build_joint_candidate_table, build_pocket_consensus_coverage
from protein_visualizer.services.comparison import compare_pocket_ranking_summaries
from protein_visualizer.services.conservation import parse_conservation_evidence_table
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.external_sites import (
    build_manual_key_residue_template,
    extract_pdb_id_from_text,
    fetch_combined_functional_sites_for_structure,
    merge_external_evidence_tables,
    parse_manual_key_residue_table,
)
from protein_visualizer.services.explainer import explain_analysis
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.interface import (
    build_inferred_interface_annotations,
    build_interface_overlap_summary,
    build_interface_summary,
    enrich_interface_annotations,
    merge_interface_annotation_tables,
    parse_interface_annotation_table,
)
from protein_visualizer.services.literature_sites import fetch_literature_residue_evidence_for_structure, remove_literature_evidence
from protein_visualizer.services.parsers import parse_pdb_atoms
from protein_visualizer.services.pdf_export import PDF_EXPORT_AVAILABLE, build_simple_pdf
from protein_visualizer.services.pocket_decision import (
    add_pocket_residue_layers,
    build_consensus_rerank_action_checklist_markdown,
    build_consensus_rerank_action_queue,
    build_consensus_rerank_apply_simulation,
    build_consensus_rerank_guardrail_artifact_manifest,
    build_consensus_rerank_guardrail_bundle_verification_summary,
    build_consensus_rerank_guardrail_handoff_certificate_markdown,
    build_consensus_rerank_guardrail_handoff_zip,
    build_consensus_rerank_policy_gate,
    build_consensus_rerank_preview,
    build_consensus_rerank_precision_guardrail,
    build_consensus_rerank_precision_guardrail_report_markdown,
    build_consensus_rerank_precision_scorecard,
    build_consensus_rerank_release_apply_plan,
    build_consensus_rerank_release_apply_report_markdown,
    build_consensus_rerank_release_closure_blocker_queue,
    build_consensus_rerank_release_closure_certificate_markdown,
    build_consensus_rerank_release_closure_detached_manifest,
    build_consensus_rerank_release_closure_ledger,
    build_consensus_rerank_release_closure_remediation_checklist_markdown,
    build_consensus_rerank_release_closure_summary,
    build_consensus_rerank_release_decision_summary,
    build_consensus_rerank_release_decision_template,
    build_consensus_rerank_release_execution_report_markdown,
    build_consensus_rerank_release_execution_template,
    build_consensus_rerank_release_execution_summary,
    build_consensus_rerank_simulation_delta,
    build_consensus_rerank_suggestion,
    build_pocket_decision_table,
    build_pocket_precision_triage,
    build_pocket_reliability_checklist,
    parse_consensus_rerank_release_decision_table,
    parse_consensus_rerank_release_execution_table,
    validate_consensus_rerank_release_execution_receipt,
    validate_consensus_rerank_release_decisions,
    verify_consensus_rerank_guardrail_handoff_zip,
)
from protein_visualizer.services.pocket import (
    PYKVFINDER_AVAILABLE,
    build_auto_pocket_display_table,
    build_pocket_detection_diagnostics_table,
    build_pocket_summary,
    build_pocket_summary_without_conservation_signal,
    detect_auto_pocket_table,
    get_pocket_detection_metadata,
    parse_pocket_table,
    summarize_pocket_detection_metadata,
)
from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value
from protein_visualizer.services.session_state import (
    append_history_record,
    get_current_energy_table,
    get_current_mmpbsa_text,
    get_current_pdb_text,
    get_uploaded_inputs_cache,
    initialize_state,
    set_analysis_state,
    set_uploaded_inputs_cache,
)
from protein_visualizer.services.snapshot import build_analysis_snapshot, build_snapshot_svg, snapshot_to_json_bytes
from protein_visualizer.services.structure_energy import estimate_protein_volume, resolve_energy_table
from protein_visualizer.services.viewer import build_view


st.set_page_config(page_title="口袋与界面", layout="wide")
st.title("口袋 / 界面专页")
st.caption("聚焦口袋、界面注释和热点交集，可作为结构可视化页之外的独立补充分析页。")
initialize_state()
st.markdown(
    """
    <style>
    .decision-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 10px 0 16px 0; }
    .decision-card { border-radius: 16px; padding: 15px; border: 1px solid #dbe4f0; background: linear-gradient(145deg, #ffffff, #f8fbff); box-shadow: 0 8px 24px rgba(15,23,42,0.06); min-height: 220px; }
    .decision-card.ready { border-color: #34d399; background: linear-gradient(145deg, #ecfdf5, #ffffff); }
    .decision-card.review { border-color: #f59e0b; background: linear-gradient(145deg, #fffbeb, #ffffff); }
    .decision-card.explore { border-color: #93c5fd; background: linear-gradient(145deg, #eff6ff, #ffffff); }
    .decision-kicker { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; font-weight: 800; }
    .decision-title { font-size: 18px; font-weight: 800; color: #0f172a; margin: 6px 0; }
    .decision-score { font-size: 28px; font-weight: 900; color: #1d4ed8; margin: 4px 0; }
    .decision-meta { font-size: 12px; color: #334155; line-height: 1.55; margin-top: 8px; }
    .decision-pill { display: inline-block; border-radius: 999px; padding: 3px 8px; background: rgba(15,23,42,0.06); color: #0f172a; font-size: 11px; margin: 3px 4px 0 0; }
    @media (max-width: 980px) { .decision-grid { grid-template-columns: 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入数据")
    uploaded_pdb = st.file_uploader("上传 PDB 文件（可选）", type=["pdb"], accept_multiple_files=False)
    uploaded_mmpbsa = st.file_uploader("上传 MMPBSA 文件（可选）", type=["txt", "dat", "out", "csv"], accept_multiple_files=False)
    uploaded_pocket = st.file_uploader("上传 Pocket CSV（可选）", type=["csv", "txt"], accept_multiple_files=False)
    uploaded_annotation = st.file_uploader("上传界面注释 CSV（可选）", type=["csv", "txt"], accept_multiple_files=False)
    use_examples = st.checkbox("使用示例数据", value=False)
    energy_mode = st.selectbox(
        "能量来源模式",
        ["auto", "mmpbsa", "estimate"],
        index=0,
        format_func=lambda x: {"auto": "自动", "mmpbsa": "上传 MMPBSA", "estimate": "结构估算"}[x],
    )

def _read_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def _uploaded_file_entry(uploaded_file) -> dict | None:
    if uploaded_file is None:
        return None
    text = _read_uploaded_text(uploaded_file)
    if not text:
        return None
    return {
        "name": str(getattr(uploaded_file, "name", "uploaded_file")),
        "text": text,
    }


def _to_csv_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode("utf-8")


DOWNLOAD_LABEL_REPLACEMENTS = [
    ("normalized ", "标准化"),
    ("consensus rerank", "共识重排"),
    ("benchmark reference", "基准参考"),
    ("pocket benchmark", "口袋基准"),
    ("AI review", "AI 复核"),
    ("AI evidence", "AI 证据"),
    ("AI ranking", "AI 排名"),
    ("ranking-gated", "排名门控"),
    ("source audit", "来源审计"),
    ("candidate review", "候选复核"),
    ("candidate", "候选"),
    ("accepted", "已接受"),
    ("release decision", "发布决策"),
    ("release execution", "发布执行"),
    ("release closure", "发布关闭"),
    ("decision template", "决策模板"),
    ("decision validation", "决策校验"),
    ("decision outcomes", "决策结果"),
    ("decisions", "决策"),
    ("decision", "决策"),
    ("review queue", "复核队列"),
    ("review checklist", "复核清单"),
    ("round summary", "轮次汇总"),
    ("round report", "轮次报告"),
    ("ranking delta", "排名变化"),
    ("artifact manifest", "产物清单"),
    ("artifact bundle", "产物包"),
    ("bundle verification summary", "包校验汇总"),
    ("bundle verification", "包校验"),
    ("bundle handoff certificate", "包交接证书"),
    ("bundle README", "包 README"),
    ("handoff certificate", "交接证书"),
    ("follow-up evidence plan", "后续取证计划"),
    ("follow-up prompt bundle", "后续提示词包"),
    ("release apply plan", "发布应用计划"),
    ("release apply report", "发布应用报告"),
    ("release execution template", "发布执行模板"),
    ("release execution receipt", "发布执行回执"),
    ("release execution validation", "发布执行校验"),
    ("release execution summary", "发布执行汇总"),
    ("release execution report", "发布执行报告"),
    ("release closure certificate", "发布关闭证书"),
    ("release closure ledger", "发布关闭台账"),
    ("release closure readiness summary", "发布关闭就绪汇总"),
    ("release closure blocker queue", "发布关闭阻断队列"),
    ("release closure remediation checklist", "发布关闭修复清单"),
    ("release closure detached manifest", "发布关闭外置清单"),
    ("suggestions", "建议"),
    ("preview", "预览"),
    ("policy gate", "策略门控"),
    ("action queue summary", "行动队列汇总"),
    ("action queue", "行动队列"),
    ("action checklist", "行动检查清单"),
    ("apply simulation", "应用模拟"),
    ("simulation delta", "模拟变化"),
    ("precision scorecard", "精度评分卡"),
    ("precision guardrail report", "精度护栏报告"),
    ("precision guardrail", "精度护栏"),
    ("guardrail artifact manifest", "护栏产物清单"),
    ("guardrail handoff ZIP", "护栏交接 ZIP"),
    ("guardrail bundle verification summary", "护栏包校验汇总"),
    ("guardrail bundle verification", "护栏包校验"),
    ("guardrail handoff certificate", "护栏交接证书"),
    ("import summary", "导入汇总"),
    ("quality issues", "质量问题"),
    ("quality summary", "质量汇总"),
    ("curation checklist", "整理清单"),
    ("structure validation summary", "结构校验汇总"),
    ("structure validation checklist", "结构校验清单"),
    ("structure validation", "结构校验"),
    ("readiness summary", "就绪汇总"),
    ("readiness case summary", "就绪 case 汇总"),
    ("readiness queue", "就绪队列"),
    ("readiness checklist", "就绪清单"),
    ("readiness impact summary", "就绪影响汇总"),
    ("readiness impact", "就绪影响"),
    ("case decision closure checklist", "case 决策关闭清单"),
    ("case decision outcome summary", "case 决策结果汇总"),
    ("case decision closure queue", "case 决策关闭队列"),
    ("case decision validation", "case 决策校验"),
    ("case decision template", "case 决策模板"),
    ("case decision dataset impact action queue summary", "case 决策数据集影响行动队列汇总"),
    ("case decision dataset impact action queue", "case 决策数据集影响行动队列"),
    ("case decision dataset impact case checklist", "case 决策数据集影响 case 清单"),
    ("case decision dataset impact artifact manifest", "case 决策数据集影响产物清单"),
    ("case decision dataset impact cases", "case 决策数据集影响 case"),
    ("case decision dataset impact report", "case 决策数据集影响报告"),
    ("case decision dataset impact", "case 决策数据集影响"),
    ("case summary", "case 汇总"),
    ("case checklist", "case 清单"),
    ("case", "案例"),
    ("template notes", "模板说明"),
    ("template", "模板"),
    ("reference", "参考"),
    ("interpretation matrix summary", "解释矩阵汇总"),
    ("interpretation matrix queue", "解释矩阵队列"),
    ("interpretation matrix", "解释矩阵"),
    ("dataset interpretation queue", "数据集解释队列"),
    ("dataset interpretation checklist", "数据集解释清单"),
    ("dataset interpretation report", "数据集解释报告"),
    ("dataset interpretation", "数据集解释"),
    ("dataset summary", "数据集汇总"),
    ("case interpretation", "case 解释"),
    ("case summary", "case 汇总"),
    ("variant dataset comparison", "变体数据集对比"),
    ("variant case comparison", "变体 case 对比"),
    ("variant residue comparison", "变体残基对比"),
    ("variant comparison", "变体对比"),
    ("remediation queue", "修复队列"),
    ("remediation summary", "修复汇总"),
    ("remediation checklist", "修复清单"),
    ("details", "明细"),
    ("summary", "汇总"),
    ("audit", "审计"),
    ("coverage", "覆盖"),
    ("evidence", "证据"),
    ("residue", "残基"),
    ("pocket", "口袋"),
    ("CSV", "CSV"),
    ("MD", "MD"),
    ("ZIP", "ZIP"),
]


def _localize_download_label(label: object) -> object:
    if not isinstance(label, str):
        return label
    if not label.startswith(("Export ", "Download ")):
        return label
    action = "导出" if label.startswith("Export ") else "下载"
    text = label.split(" ", 1)[1]
    for source, target in DOWNLOAD_LABEL_REPLACEMENTS:
        text = text.replace(source, target)
    text = " ".join(text.split())
    separator = " " if text and text[0].isascii() else ""
    return f"{action}{separator}{text}"


if not hasattr(st, "_protein_visualizer_original_download_button"):
    st._protein_visualizer_original_download_button = st.download_button


def _localized_download_button(*args, **kwargs):
    if args:
        args = (_localize_download_label(args[0]), *args[1:])
    elif "label" in kwargs:
        kwargs["label"] = _localize_download_label(kwargs["label"])
    return st._protein_visualizer_original_download_button(*args, **kwargs)


st.download_button = _localized_download_button


REPORT_LINE_REPLACEMENTS = [
    ("Benchmark reference source audit case decision readiness impact summary", "基准参考来源审计 case 决策就绪影响汇总"),
    ("Benchmark reference source audit case decision readiness impact", "基准参考来源审计 case 决策就绪影响"),
    ("Benchmark reference source audit case decision closure checklist", "基准参考来源审计 case 决策关闭清单"),
    ("Benchmark reference source audit case decision closure queue", "基准参考来源审计 case 决策关闭队列"),
    ("Benchmark reference source audit case decision outcome summary", "基准参考来源审计 case 决策结果汇总"),
    ("Benchmark reference source audit case decision outcomes", "基准参考来源审计 case 决策结果"),
    ("Benchmark reference source audit case decision template", "基准参考来源审计 case 决策模板"),
    ("Benchmark reference source audit case decisions", "基准参考来源审计 case 决策"),
    ("Benchmark source-audit decision dataset impact action summary", "基准来源审计决策数据集影响行动汇总"),
    ("Benchmark source-audit decision dataset impact action queue", "基准来源审计决策数据集影响行动队列"),
    ("Benchmark source-audit decision dataset impact artifacts", "基准来源审计决策数据集影响产物"),
    ("Benchmark source-audit decision dataset impact cases", "基准来源审计决策数据集影响 case"),
    ("Benchmark source-audit decision dataset impact", "基准来源审计决策数据集影响"),
    ("Benchmark reference source audit action queue", "基准参考来源审计行动队列"),
    ("Benchmark reference source audit case checklist", "基准参考来源审计 case 清单"),
    ("Benchmark reference source audit checklist", "基准参考来源审计清单"),
    ("Benchmark reference source audit summary", "基准参考来源审计汇总"),
    ("Benchmark reference source audit cases", "基准参考来源审计 case"),
    ("Benchmark reference source audit", "基准参考来源审计"),
    ("Benchmark reference candidate review decisions", "基准参考候选复核决策"),
    ("Benchmark reference candidate review", "基准参考候选复核"),
    ("Benchmark reference candidate", "基准参考候选"),
    ("Benchmark reference curation quality", "基准参考整理质量"),
    ("Benchmark reference structure validation", "基准参考结构校验"),
    ("Benchmark reference readiness cases", "基准参考就绪 case"),
    ("Benchmark reference readiness", "基准参考就绪"),
    ("Benchmark reference template", "基准参考模板"),
    ("Benchmark reference source", "基准参考来源"),
    ("Catalytic pocket benchmark", "催化口袋基准"),
    ("Catalytic benchmark remediation queue", "催化基准修复队列"),
    ("Catalytic benchmark variant residues", "催化基准变体残基"),
    ("Catalytic benchmark variant cases", "催化基准变体 case"),
    ("Catalytic benchmark variants", "催化基准变体"),
    ("Catalytic benchmark dataset", "催化基准数据集"),
    ("Benchmark case interpretation matrix summary", "基准 case 解释矩阵汇总"),
    ("Benchmark case interpretation matrix queue", "基准 case 解释矩阵队列"),
    ("Benchmark case interpretation matrix", "基准 case 解释矩阵"),
    ("Benchmark case interpretation", "基准 case 解释"),
    ("case", "案例"),
    ("Benchmark dataset interpretation queue", "基准数据集解释队列"),
    ("Benchmark dataset interpretation", "基准数据集解释"),
    ("Benchmark interpretation", "基准解释"),
    ("Consensus rerank release closure detached manifest", "共识重排发布关闭外置清单"),
    ("Consensus rerank release closure remediation checklist", "共识重排发布关闭修复清单"),
    ("Consensus rerank release closure blockers", "共识重排发布关闭阻断项"),
    ("Consensus rerank release closure readiness", "共识重排发布关闭就绪"),
    ("Consensus rerank release closure ledger", "共识重排发布关闭台账"),
    ("Consensus rerank release closure certificate", "共识重排发布关闭证书"),
    ("Consensus rerank release execution validation", "共识重排发布执行校验"),
    ("Consensus rerank release execution template", "共识重排发布执行模板"),
    ("Consensus rerank release execution receipt", "共识重排发布执行回执"),
    ("Consensus rerank release execution report", "共识重排发布执行报告"),
    ("Consensus rerank release execution", "共识重排发布执行"),
    ("Consensus rerank release apply report", "共识重排发布应用报告"),
    ("Consensus rerank release apply plan", "共识重排发布应用计划"),
    ("Consensus rerank release decision validation", "共识重排发布决策校验"),
    ("Consensus rerank release decision template", "共识重排发布决策模板"),
    ("Consensus rerank release decisions", "共识重排发布决策"),
    ("Consensus rerank release review", "共识重排发布复核"),
    ("Consensus rerank guardrail bundle verification", "共识重排护栏包校验"),
    ("Consensus rerank guardrail handoff certificate", "共识重排护栏交接证书"),
    ("Consensus rerank guardrail handoff bundle", "共识重排护栏交接包"),
    ("Consensus rerank precision guardrail report", "共识重排精度护栏报告"),
    ("Consensus rerank precision guardrail", "共识重排精度护栏"),
    ("Consensus rerank precision scorecard", "共识重排精度评分卡"),
    ("Consensus rerank simulation delta", "共识重排模拟变化"),
    ("Consensus rerank apply simulation", "共识重排应用模拟"),
    ("Consensus rerank action checklist", "共识重排行动清单"),
    ("Consensus rerank action queue", "共识重排行动队列"),
    ("Consensus rerank policy gate", "共识重排策略门控"),
    ("Consensus rerank suggestions", "共识重排建议"),
    ("Consensus rerank preview", "共识重排预览"),
    ("AI evidence used for ranking", "AI 排名可用证据"),
    ("AI evidence audit", "AI 证据审计"),
    ("AI evidence", "AI 证据"),
    ("AI review bundle verification summary", "AI 复核包校验汇总"),
    ("AI review bundle verification", "AI 复核包校验"),
    ("AI review bundle certificate", "AI 复核包证书"),
    ("AI review artifact manifest", "AI 复核产物清单"),
    ("AI review artifact bundle", "AI 复核产物包"),
    ("AI review bundle README", "AI 复核包 README"),
    ("AI review decision validation", "AI 复核决策校验"),
    ("AI review decision outcomes", "AI 复核决策结果"),
    ("AI review decision template", "AI 复核决策模板"),
    ("AI review decisions", "AI 复核决策"),
    ("AI review ranking delta", "AI 复核排名变化"),
    ("AI review round", "AI 复核轮次"),
    ("AI review queue", "AI 复核队列"),
    ("AI follow-up plan", "AI 后续取证计划"),
    ("AI influence", "AI 影响"),
    ("Residue evidence consensus", "残基证据共识"),
    ("Pocket consensus coverage", "口袋共识覆盖"),
    ("Top active-site decision", "Top 活性位点决策"),
    ("Top decision score", "Top 决策评分"),
    ("Precision tier", "精度分层"),
    ("Triage action", "分诊动作"),
    ("Reliability checks", "可靠性检查"),
    ("Reliability gaps", "可靠性缺口"),
    ("Next step", "下一步"),
    ("P2Rank A/B", "P2Rank A/B"),
    ("Top pocket AI residues", "Top 口袋 AI 残基"),
    ("Top-1 claim", "Top-1 结论"),
    ("Top-3 claim", "Top-3 结论"),
    ("Top-1", "Top-1"),
    ("Top-3", "Top-3"),
    ("current vs ablations", "当前与消融对比"),
    ("best rank", "最佳排名"),
    ("dataset rows", "数据集行"),
    ("accepted actions", "接受动作"),
    ("accepted references", "接受参考"),
    ("validation blocked", "校验阻断"),
    ("independent claim", "独立结论"),
    ("claim status", "结论状态"),
    ("top status", "Top 状态"),
    ("top fix", "Top 修复项"),
    ("provisional used", "使用临时参考"),
    ("reviewed candidate", "已复核候选"),
    ("provisional", "临时参考"),
    ("rankable", "可排名"),
    ("manifest", "清单"),
    ("checklist", "清单"),
    ("references", "条参考"),
    ("blockers", "阻断项"),
    ("blocked", "阻断"),
    ("mismatches", "不匹配"),
    ("mismatch", "不匹配"),
    ("cleared", "已清除"),
    ("changed", "已变化"),
    ("failed", "失败"),
    ("usable", "可用"),
    ("complete", "完成"),
    ("closed", "关闭"),
    ("allowed", "允许"),
    ("actions", "动作"),
    ("hashes", "哈希"),
    ("bytes", "字节"),
    ("score", "评分"),
    ("mode", "模式"),
    ("issue", "问题"),
    ("files", "个文件"),
    ("issues", "个问题"),
    ("cases", "个案例"),
    ("rows", "行"),
    ("status", "状态"),
    ("summary", "汇总"),
    ("decision", "决策"),
    ("audit", "审计"),
    ("source", "来源"),
    ("import", "导入"),
    ("notes", "说明"),
    ("tier", "分层"),
    ("label", "标签"),
    ("top", "Top"),
    ("source-review-needed", "来源需复核"),
    ("source-blocked", "来源阻断"),
    ("source-gate-mismatch", "来源门控不匹配"),
    ("review-needed", "需复核"),
    ("cleared-by-decision", "决策已清除"),
    ("decision-adjusted-open", "决策调整后未关闭"),
    ("decision-open", "决策未关闭"),
    ("unchanged-open", "未改变且未关闭"),
    ("verified", "已校验"),
    ("accepted", "已接受"),
    ("review", "复核"),
    ("open", "未关闭"),
    ("pending", "待处理"),
    ("applied", "已应用"),
    ("not available", "不可用"),
    ("not enabled", "未启用"),
    ("available", "可用"),
    ("enabled", "已启用"),
    ("none", "无"),
    ("yes", "是"),
    ("no", "否"),
    ("pass", "通过"),
    ("missing", "缺失"),
]


REPORT_STATUS_REPLACEMENTS = [
    ("source-review-needed", "来源需复核"),
    ("source-blocked", "来源阻断"),
    ("source-gate-mismatch", "来源门控不匹配"),
    ("review-needed", "需复核"),
    ("cleared-by-decision", "决策已清除"),
    ("decision-adjusted-open", "决策调整后未关闭"),
    ("decision-open", "决策未关闭"),
    ("unchanged-open", "未改变且未关闭"),
]


REPORT_VALUE_REPLACEMENTS = [
    ("interface_and_pocket", "界面与口袋"),
    ("interface_and_hotspot", "界面与热点"),
    ("pocket_and_hotspot", "口袋与热点"),
    ("interface_residues", "界面残基"),
    ("pocket_residues", "口袋残基"),
    ("hotspot_residues", "热点残基"),
    ("triple_overlap", "三重交集"),
    ("empty-input", "输入为空"),
    ("not-uploaded", "未上传"),
    ("source-ready", "来源就绪"),
    ("top-pocket-supported", "Top 口袋受支持"),
    ("missing-citation-or-snippet", "缺少引用或证据片段"),
    ("Review mapping before validation", "验证前需要复核映射"),
    ("Review chain/numbering/mapping before validation.", "验证前检查链、编号和映射。"),
    ("mapping-review-needed", "映射需要复核"),
    ("mapping-review", "映射复核"),
    ("Evidence mapping risk", "证据映射风险"),
    ("Actionability", "可行动性"),
    ("needs-review", "需复核"),
    ("unsupported", "无支持"),
    ("conflicting", "冲突"),
    ("supported", "已支持"),
    ("unchanged", "未变化"),
    ("changed", "已变化"),
    ("cleared", "已清除"),
    ("replaced", "已替换"),
    ("rejected", "已拒绝"),
    ("promoted", "提升"),
    ("removed", "移除"),
    ("complete", "完成"),
    ("failed", "失败"),
    ("external", "外部"),
    ("literature", "文献"),
    ("manual", "人工"),
    ("uploaded", "上传"),
    ("combined", "合并"),
    ("inferred", "推断"),
    ("core", "核心"),
    ("rim", "边缘"),
    ("surface", "表面"),
    ("contact", "接触"),
    ("empty", "空"),
    ("ok", "正常"),
    ("not available", "不可用"),
    ("not enabled", "未启用"),
    ("available", "可用"),
    ("enabled", "已启用"),
    ("verified", "已校验"),
    ("accepted", "已接受"),
    ("blocked", "阻断"),
    ("pending", "待处理"),
    ("review", "复核"),
    ("pass", "通过"),
    ("missing", "缺失"),
    ("none", "无"),
    ("yes", "是"),
    ("no", "否"),
]


STATUS_TEXT_LABELS = {
    source: target
    for source, target in [
        *REPORT_STATUS_REPLACEMENTS,
        *REPORT_VALUE_REPLACEMENTS,
    ]
}
STATUS_TEXT_LABELS.update({source.lower(): target for source, target in STATUS_TEXT_LABELS.items()})


def _localize_report_line(line: str) -> str:
    text = str(line)
    for source, target in REPORT_STATUS_REPLACEMENTS:
        text = text.replace(source, target)
    for source, target in REPORT_LINE_REPLACEMENTS:
        text = text.replace(source, target)
    for source, target in REPORT_VALUE_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def _localize_status_text(value: object, default: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    if text in STATUS_TEXT_LABELS:
        return STATUS_TEXT_LABELS[text]
    if text.lower() in STATUS_TEXT_LABELS:
        return STATUS_TEXT_LABELS[text.lower()]
    if "," in text:
        return ", ".join(
            _localize_status_text(part.strip(), default=part.strip())
            for part in text.split(",")
            if part.strip()
        )
    return text


DATAFRAME_COLUMN_LABELS = {
    "pocket_id": "口袋 ID",
    "rank": "排名",
    "old_rank": "原排名",
    "new_rank": "新排名",
    "best_rank": "最佳排名",
    "rank_delta": "排名变化",
    "score": "得分",
    "original_score": "原始得分",
    "adjusted_score": "调整后得分",
    "consensus_score": "共识得分",
    "confidence": "置信度",
    "confidence_score": "置信度",
    "coverage_ratio": "覆盖率",
    "coverage": "覆盖",
    "status": "状态",
    "audit_status": "审计状态",
    "validation_status": "校验状态",
    "applied_status": "应用状态",
    "action_status": "行动状态",
    "readiness_status": "就绪状态",
    "benchmark_status": "基准状态",
    "claim_status": "结论状态",
    "source_claim_status": "来源结论状态",
    "decision_label": "决策标签",
    "decision_score": "决策得分",
    "decision_reason": "决策理由",
    "next_step": "下一步",
    "precision_tier": "精度分层",
    "triage_action": "分诊动作",
    "issue_type": "问题类型",
    "issue_flags": "问题标记",
    "priority": "优先级",
    "reason": "原因",
    "recommendation_label": "推荐等级",
    "recommendation_reason": "推荐理由",
    "residue_label": "残基",
    "residue_anchor": "残基锚点",
    "residue_name": "残基名称",
    "residue_number": "残基编号",
    "residue_index": "残基序号",
    "chain_id": "链 ID",
    "chain": "链",
    "insertion_code": "插入码",
    "uniprot_position": "UniProt 位点",
    "sequence_position": "序列位点",
    "structure_residue_number": "结构残基编号",
    "source": "来源",
    "sources": "来源",
    "source_mode": "来源模式",
    "source_id": "来源 ID",
    "source_url": "来源链接",
    "evidence_source": "证据来源",
    "evidence_type": "证据类型",
    "mapping_level": "映射等级",
    "matching_chain": "匹配链",
    "method": "方法",
    "detection_method": "识别方法",
    "region_type": "区域类型",
    "annotation": "注释",
    "annotation_source": "注释来源",
    "inference_basis": "推断依据",
    "is_overlap": "是否交集",
    "is_pocket": "是否口袋",
    "is_hotspot": "是否热点",
    "category": "类别",
    "count": "数量",
    "residue_count": "残基数",
    "pocket_count": "口袋命中数",
    "hotspot_count": "热点命中数",
    "overlap_count": "交集数",
    "residue_labels": "残基列表",
    "delta_total": "总能量变化",
    "hotspot_rank": "热点排名",
    "top_n": "Top-N",
    "top_pocket_id": "Top 口袋 ID",
    "best_pocket_id": "最佳口袋 ID",
    "reference_rows": "参考残基数",
    "import_status": "导入状态",
    "manual_review_rows": "人工复核行数",
    "rankable_after_review_rows": "复核后可排名行数",
    "promoted_rows": "提升行数",
    "removed_rows": "移除行数",
    "failed_files": "失败文件数",
    "byte_size": "字节数",
    "sha256": "SHA256",
}


DATAFRAME_COLUMN_TOKEN_LABELS = {
    "auto": "自动",
    "detection": "识别",
    "min": "最小",
    "max": "最大",
    "ai": "AI",
    "ab": "A/B",
    "p2rank": "P2Rank",
    "pocket": "口袋",
    "benchmark": "基准",
    "reference": "参考",
    "candidate": "候选",
    "review": "复核",
    "decision": "决策",
    "validation": "校验",
    "outcome": "结果",
    "source": "来源",
    "audit": "审计",
    "case": "案例",
    "dataset": "数据集",
    "impact": "影响",
    "action": "行动",
    "queue": "队列",
    "summary": "汇总",
    "status": "状态",
    "reason": "原因",
    "issue": "问题",
    "issues": "问题",
    "priority": "优先级",
    "rank": "排名",
    "ranking": "排名",
    "delta": "变化",
    "score": "得分",
    "coverage": "覆盖",
    "ratio": "比例",
    "residue": "残基",
    "residues": "残基",
    "chain": "链",
    "position": "位点",
    "number": "编号",
    "name": "名称",
    "label": "标签",
    "type": "类型",
    "mode": "模式",
    "method": "方法",
    "evidence": "证据",
    "consensus": "共识",
    "confidence": "置信度",
    "mapping": "映射",
    "level": "等级",
    "support": "支持",
    "supported": "支持",
    "manual": "人工",
    "external": "外部",
    "route": "路径",
    "readiness": "就绪",
    "closure": "关闭",
    "closed": "关闭",
    "blocker": "阻断项",
    "blocked": "阻断",
    "missing": "缺失",
    "pass": "通过",
    "failed": "失败",
    "files": "文件",
    "file": "文件",
    "bytes": "字节",
    "size": "大小",
    "hash": "哈希",
    "rows": "行数",
    "row": "行",
    "count": "数量",
    "top": "Top",
    "recommendation": "推荐",
    "best": "最佳",
    "old": "原",
    "new": "新",
    "current": "当前",
    "baseline": "基线",
    "variant": "变体",
    "comparison": "对比",
    "interpretation": "解释",
    "matrix": "矩阵",
    "artifact": "产物",
    "manifest": "清单",
    "bundle": "包",
    "verification": "校验",
    "certificate": "证书",
    "readme": "README",
    "template": "模板",
    "notes": "说明",
    "quality": "质量",
    "structure": "结构",
    "curation": "整理",
    "remediation": "修复",
    "guardrail": "护栏",
    "release": "发布",
    "execution": "执行",
    "receipt": "回执",
    "apply": "应用",
    "plan": "计划",
    "effect": "效果",
    "influence": "影响",
    "fix": "修复",
    "query": "检索词",
    "title": "标题",
    "snippet": "片段",
    "abstract": "摘要",
    "url": "链接",
    "id": "ID",
    "ec": "EC",
    "uniprot": "UniProt",
    "pdb": "PDB",
}


DATAFRAME_VALUE_COLUMN_HINTS = (
    "status",
    "decision",
    "reason",
    "action",
    "issue",
    "tier",
    "mode",
    "check",
    "outcome",
    "effect",
    "support",
    "route",
    "category",
    "basis",
    "readiness",
    "validation",
    "closure",
    "claim",
)


def _localize_dataframe_column(column: object) -> object:
    if not isinstance(column, str):
        return column
    if column in DATAFRAME_COLUMN_LABELS:
        return DATAFRAME_COLUMN_LABELS[column]
    if "_" not in column:
        return DATAFRAME_COLUMN_TOKEN_LABELS.get(column.lower(), column)
    parts = column.split("_")
    localized_parts = [DATAFRAME_COLUMN_TOKEN_LABELS.get(part.lower(), part) for part in parts]
    if localized_parts != parts:
        return " ".join(localized_parts)
    return column


def _should_localize_dataframe_values(column: object) -> bool:
    name = str(column).lower()
    return any(hint in name for hint in DATAFRAME_VALUE_COLUMN_HINTS)


def _localize_dataframe_value(value: object) -> object:
    if isinstance(value, bool):
        return "是" if value else "否"
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return _localize_status_text(value, default="")
    return value


def _localize_dataframe_for_display(table: pd.DataFrame) -> pd.DataFrame:
    display = table.copy()
    for column in list(display.columns):
        if pd.api.types.is_bool_dtype(display[column]):
            display[column] = display[column].map(lambda value: "是" if bool(value) else "否")
        elif _should_localize_dataframe_values(column) and (
            pd.api.types.is_object_dtype(display[column]) or pd.api.types.is_string_dtype(display[column])
        ):
            display[column] = display[column].map(_localize_dataframe_value)
    return display.rename(columns={column: _localize_dataframe_column(column) for column in display.columns})


if not hasattr(st, "_protein_visualizer_original_dataframe"):
    st._protein_visualizer_original_dataframe = st.dataframe


def _localized_dataframe(*args, **kwargs):
    if args and isinstance(args[0], pd.DataFrame):
        args = (_localize_dataframe_for_display(args[0]), *args[1:])
    elif isinstance(kwargs.get("data"), pd.DataFrame):
        kwargs["data"] = _localize_dataframe_for_display(kwargs["data"])
    return st._protein_visualizer_original_dataframe(*args, **kwargs)


st.dataframe = _localized_dataframe


def _localize_json_for_display(value):
    if isinstance(value, dict):
        return {_localize_dataframe_column(key): _localize_json_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_localize_json_for_display(item) for item in value]
    if isinstance(value, tuple):
        return [_localize_json_for_display(item) for item in value]
    if isinstance(value, str):
        return _localize_status_text(value, default=value)
    return value


benchmark_reference_template_df = build_pocket_benchmark_reference_template()
benchmark_reference_template_markdown = build_pocket_benchmark_reference_template_markdown()
manual_key_residue_template_df = build_manual_key_residue_template()


def _external_evidence_counts(table: pd.DataFrame) -> dict[str, int]:
    if table is None or getattr(table, "empty", True):
        return {"rows": 0, "exact": 0, "weak": 0}
    exact_rows = 0
    weak_rows = 0
    if "mapping_level" in table.columns:
        level_series = table["mapping_level"].astype(str).str.lower()
        exact_rows = int((level_series == "exact").sum())
        weak_rows = int((level_series == "weak").sum())
    return {"rows": int(len(table)), "exact": exact_rows, "weak": weak_rows}


def _source_audit_case_summary_counts(table: pd.DataFrame) -> tuple[int, int]:
    if table is None or getattr(table, "empty", True) or not {"blocker_rows", "review_rows"}.issubset(table.columns):
        return 0, 0
    blocker_rows = pd.to_numeric(table["blocker_rows"], errors="coerce").fillna(0)
    review_rows = pd.to_numeric(table["review_rows"], errors="coerce").fillna(0)
    blocked_cases = int(blocker_rows.gt(0).sum())
    review_cases = int((blocker_rows.eq(0) & review_rows.gt(0)).sum())
    return blocked_cases, review_cases


@st.cache_data(show_spinner=False, ttl=3600)
def _load_external_evidence(
    accession: str,
    chain_hint: str,
    pdb_id: str,
    pdb_text: str,
    ec_number: str,
    enable_uniprot: bool,
    enable_mcsa: bool,
) -> tuple[pd.DataFrame, dict]:
    return fetch_combined_functional_sites_for_structure(
        accession,
        ec_number=ec_number,
        chain_hint=chain_hint,
        pdb_id=pdb_id,
        pdb_text=pdb_text,
        enable_uniprot=enable_uniprot,
        enable_mcsa=enable_mcsa,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _load_literature_evidence(
    query: str,
    manual_text: str,
    accession: str,
    ec_number: str,
    pdb_id: str,
    protein_name: str,
    chain_hint: str,
    pdb_text: str,
    max_articles: int,
    enable_pubmed: bool,
    enable_europepmc: bool,
    include_europepmc_fulltext: bool,
    max_fulltext_articles: int,
    assume_structure_numbering: bool,
) -> tuple[pd.DataFrame, dict]:
    return fetch_literature_residue_evidence_for_structure(
        query=query,
        manual_text=manual_text,
        accession=accession,
        ec_number=ec_number,
        pdb_id=pdb_id,
        protein_name=protein_name,
        chain_hint=chain_hint,
        pdb_text=pdb_text,
        max_articles=max_articles,
        enable_pubmed=enable_pubmed,
        enable_europepmc=enable_europepmc,
        include_europepmc_fulltext=include_europepmc_fulltext,
        max_fulltext_articles=max_fulltext_articles,
        assume_structure_numbering=assume_structure_numbering,
    )


def _normalize_pocket_table(table: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if table is None or getattr(table, "empty", True):
        return pd.DataFrame()

    normalized = table.copy()
    normalized["pocket_source"] = source_label
    if "residue_label" not in normalized.columns:
        normalized["residue_label"] = normalized.apply(
            lambda row: f"{row.resname} {row.chain}{int(row.resid)}",
            axis=1,
        )
    return normalized


def _merge_pocket_tables(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    valid_tables = [table for table in [primary_df, secondary_df] if table is not None and not getattr(table, "empty", True)]
    if not valid_tables:
        return pd.DataFrame()

    combined = pd.concat(valid_tables, ignore_index=True)
    for column in ("score", "residue_score", "consensus_score"):
        if column not in combined.columns:
            combined[column] = 0.0

    combined = combined.sort_values(
        ["score", "residue_score", "consensus_score", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    )
    combined = combined.drop_duplicates(subset=["pocket_id", "chain", "resid", "resname"], keep="first")
    return combined.reset_index(drop=True)


def _resolve_pocket_source(manual_df: pd.DataFrame, auto_df: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    if source_mode == "uploaded":
        return manual_df.copy() if manual_df is not None else pd.DataFrame()
    if source_mode == "auto":
        return auto_df.copy() if auto_df is not None else pd.DataFrame()
    return _merge_pocket_tables(manual_df, auto_df)


def _resolve_annotation_source(uploaded_df: pd.DataFrame, inferred_df: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    if source_mode == "uploaded":
        return uploaded_df.copy() if uploaded_df is not None else pd.DataFrame()
    if source_mode == "inferred":
        return inferred_df.copy() if inferred_df is not None else pd.DataFrame()
    return merge_interface_annotation_tables(uploaded_df, inferred_df)


def _residue_pairs(table: pd.DataFrame) -> list[tuple[str, int]]:
    if table is None or getattr(table, "empty", True):
        return []
    return [(str(row.chain).strip() or "A", int(row.resid)) for row in table.itertuples(index=False)]


POCKET_DECISION_VALUE_LABELS = {
    "Evidence-led active-site candidate": "证据主导的活性位点候选",
    "Review mapping before validation": "验证前需要复核映射",
    "Interface-supported candidate": "界面证据支持的候选口袋",
    "Geometry-only exploratory pocket": "仅几何支持的探索口袋",
    "Shortlist for follow-up": "列入后续跟进候选",
    "Exploratory candidate": "探索性候选",
    "ready-to-validate": "可进入验证",
    "mapping-review-needed": "映射需要复核",
    "needs-functional-evidence": "缺少功能证据",
    "exploratory-only": "仅探索使用",
    "shortlist": "候选保留",
    "validate-prioritize": "优先验证",
    "review-evidence-mapping": "复核证据映射",
    "validate-interface-context": "结合界面与热点验证",
    "shortlist-follow-up": "列入后续跟进",
    "strong-direct-anchor": "强直接残基锚点",
    "direct-anchor": "直接残基锚点",
    "route-anchor": "证据路径锚点",
    "structure-verified-external": "结构已验证的外部证据",
    "neighborhood-expanded": "邻域扩展证据",
    "diffuse-external-support": "分散外部支持",
    "geometry-only": "仅几何支持",
    "no-external-evidence": "无外部证据",
    "unknown": "未知",
    "evidence-warning": "证据警告",
    "neighborhood-expansion-risk": "邻域扩展风险",
    "low-mapping-quality": "映射质量低",
    "literature-lowered-rank": "文献证据降低排名",
    "evidence-route-lowered-rank": "证据路径降低排名",
    "geometry-dominated": "几何信号主导",
    "none": "无",
    "pass": "通过",
    "review": "需复核",
    "missing": "缺失",
    "Functional anchors": "功能残基锚点",
    "Evidence mapping risk": "证据映射风险",
    "Geometry consensus": "几何一致性",
    "Evidence A/B movement": "证据 A/B 变化",
    "Actionability": "可行动性",
    "validation-ready": "可进入验证",
    "evidence-gap": "证据缺口",
    "mapping-review": "映射复核",
    "geometry-review": "几何复核",
    "evidence-review": "证据复核",
    "exploratory": "探索性",
}

POCKET_DECISION_TEXT_LABELS = {
    "Prioritize residue-level validation around direct anchors.": "优先围绕直接证据锚点做残基层验证。",
    "Check UniProt/PDB residue mapping, chain choice, and numbering before validation.": "验证前检查 UniProt/PDB 残基映射、链选择和编号。",
    "Inspect direct anchors versus expanded neighborhood residues before trusting the pocket boundary.": "信任口袋边界前，先检查直接锚点与邻域扩展残基是否一致。",
    "Add UniProt/M-CSA/literature or manual key residues before treating this as an active site.": "先补充 UniProt、M-CSA、文献或人工关键残基，再把它作为活性位点。",
    "Validate with interface and hotspot context; functional residue evidence is still useful.": "结合界面和热点上下文验证，同时继续补充功能残基证据。",
    "Keep as a secondary candidate and compare against higher-confidence pockets.": "作为次级候选保留，并与更高置信度口袋比较。",
    "No decision panel is available yet. Run auto-pocket detection or add pocket evidence first.": "暂时没有可用的决策面板。请先运行自动口袋识别，或补充口袋证据。",
    "Use direct anchors as the pocket core before expanding the boundary.": "先把直接证据锚点作为口袋核心，再扩展边界。",
    "Review route-derived anchors against residue numbering before treating them as catalytic points.": "把证据路径推导出的锚点视为催化位点前，先核对残基编号。",
    "Add UniProt, M-CSA, literature, or manual key residues to avoid geometry-only ranking.": "补充 UniProt、M-CSA、文献或人工关键残基，避免只靠几何排名。",
    "Enzyme active sites should be tied to catalytic/binding residues, not only to surface cavities.": "酶活性位点应绑定到催化/结合残基，而不能只依赖表面凹腔。",
    "Inspect chain, insertion codes, UniProt/PDB offsets, and expanded-neighborhood residues.": "检查链、插入码、UniProt/PDB 编号偏移，以及邻域扩展残基。",
    "Fetch or upload external residue evidence before relying on this candidate.": "依赖该候选前，先获取或上传外部残基证据。",
    "Keep the mapped evidence visible as the validation anchor layer.": "保留映射后的证据层，作为验证时的锚点层。",
    "A correct catalytic residue is not useful if it was mapped onto the wrong chain or numbering system.": "即使催化残基本身正确，如果映射到错误链或编号系统，也不能作为可靠证据。",
    "Use geometry support to define the shell around evidence anchors.": "用几何支持定义证据锚点周围的口袋外壳。",
    "Compare pocket boundary against neighboring cavities and hotspot overlap.": "对比邻近腔体和热点重叠，复核口袋边界。",
    "Treat this as weak geometry; add ligand/contact context or rerun detection with broader settings.": "当前几何支持较弱；建议补充配体/接触上下文，或放宽参数重新识别。",
    "Reliable pockets need both functional anchors and a physically plausible cavity boundary.": "可靠口袋需要同时具备功能锚点和物理上合理的腔体边界。",
    "Keep the evidence route enabled; it improves this candidate's ranking.": "保留证据路径，它提升了该候选的排名。",
    "Compare before/after rankings to understand why evidence lowered this pocket.": "对比证据加入前后的排名，确认为什么该口袋被下调。",
    "Evidence exists but did not move the rank; inspect whether weights are too conservative.": "已有证据但排名未变化，需要检查权重是否过于保守。",
    "Run literature/evidence/conservation comparison after adding functional evidence.": "补充功能证据后，再运行文献/证据路径/保守性对比。",
    "A/B movement shows whether external evidence is actually changing the product recommendation.": "A/B 变化用于判断外部证据是否真正改变产品推荐结果。",
    "Prioritize validation around the top evidence anchors.": "优先围绕最高证据锚点开展验证。",
    "Review the listed risks before wet-lab or docking follow-up.": "开展湿实验或 docking 前，先复核列出的风险。",
    "Do not treat this as a final active-site call yet.": "暂时不要把它当作最终活性位点结论。",
    "The UI should end with an explicit next step instead of a raw score that users must interpret.": "界面应给出明确下一步，而不是只给用户一个需要自行解释的原始分数。",
    "Proceed to validation around core evidence anchors.": "围绕核心证据锚点进入验证。",
    "All reliability gates pass and the decision audit is ready.": "所有可靠性门槛均通过，决策审计已准备好。",
    "Do not finalize; add or verify functional residue evidence first.": "不要定稿；先补充或验证功能残基证据。",
    "The candidate lacks the residue-level enzyme evidence needed for a high-precision active-site call.": "该候选缺少高精度活性位点判断所需的残基层酶学证据。",
    "Review chain/numbering/mapping before validation.": "验证前复核链、编号和映射关系。",
    "Functional evidence exists, but residue mapping or neighborhood expansion can shift the pocket core.": "已有功能证据，但残基映射或邻域扩展可能改变口袋核心。",
    "Check cavity boundary against alternate geometry and ligand/contact context.": "结合替代几何结果和配体/接触上下文检查腔体边界。",
    "Evidence may be useful, but the physical pocket boundary is not yet stable.": "证据可能有用，但物理口袋边界尚不稳定。",
    "Keep shortlisted and resolve the remaining review/missing gates.": "保留为候选，并解决剩余需复核/缺失项。",
    "The pocket is plausible but still has unresolved evidence or actionability gaps.": "该口袋有一定合理性，但仍存在未解决的证据或可行动性缺口。",
    "Use only as exploratory geometry until functional evidence is added.": "在补充功能证据前，仅作为探索性几何结果使用。",
    "The candidate is dominated by geometry rather than enzyme-specific evidence.": "该候选主要由几何信号驱动，而非酶特异性证据。",
    "Keep as a secondary candidate and compare with stronger evidence-led pockets.": "作为次级候选保留，并与证据更强的口袋比较。",
    "No hard blocker is visible, but support is not strong enough for a primary validation call.": "目前没有硬性阻断项，但支持强度不足以作为首要验证结论。",
    "No additional evidence required before validation.": "验证前无需额外证据。",
}

POCKET_DECISION_TEXT_REPLACEMENTS = [
    ("M-CSA / UniProt active-site annotations / PubMed key residues / manual catalytic residues", "M-CSA / UniProt 活性位点注释 / PubMed 关键残基 / 人工催化残基"),
    ("SIFTS chain mapping, insertion codes, UniProt offsets, and author numbering audit", "SIFTS 链映射、插入码、UniProt 偏移和作者编号审计"),
    ("P2Rank/fpocket comparison, ligand-neighborhood contacts, or broader geometry detection", "P2Rank/fpocket 对比、配体邻域接触或更宽松的几何检测"),
    ("literature/evidence-route/conservation A/B comparison", "文献/证据路径/保守性 A/B 对比"),
    ("manual review note that turns the candidate into validate/review/explore", "人工复核说明，用于把候选转为验证/复核/探索状态"),
    ("direct=", "直接锚点="),
    ("route=", "路径锚点="),
    ("quality=", "证据质量="),
    ("functional=", "功能分="),
    ("geometry=", "几何分="),
    ("method_votes=", "方法票数="),
    ("literature=", "文献="),
    ("conservation=", "保守性="),
    ("audit=", "审计="),
    ("action=", "动作="),
]

POCKET_DECISION_COLUMN_LABELS = {
    "decision_rank": "决策排名",
    "pocket_id": "口袋 ID",
    "decision_label": "判断结果",
    "decision_score": "决策分数",
    "functional_confidence": "功能证据分",
    "geometry_confidence": "几何支持分",
    "recommended_action": "建议动作",
    "audit_status": "审计状态",
    "evidence_quality_label": "证据质量",
    "evidence_quality_score": "证据质量分",
    "direct_anchor_count": "直接锚点数",
    "route_anchor_count": "路径锚点数",
    "anchor_residues": "锚点残基",
    "method_vote_count": "方法票数",
    "smart_rank_label": "智能排名标签",
    "smart_rank_score": "智能排名分",
    "literature_rank_delta": "文献排名变化",
    "evidence_route_rank_delta": "证据路径排名变化",
    "conservation_rank_delta": "保守性排名变化",
    "risk_flags": "风险标签",
    "supporting_evidence": "支持证据",
    "next_step": "下一步",
    "visual_focus": "可视化重点",
}

POCKET_RELIABILITY_COLUMN_LABELS = {
    "pocket_id": "口袋 ID",
    "check_order": "检查顺序",
    "check": "检查项",
    "status": "状态",
    "signal": "信号",
    "why_it_matters": "为什么重要",
    "next_action": "下一步",
}

POCKET_TRIAGE_COLUMN_LABELS = {
    "pocket_id": "口袋 ID",
    "decision_rank": "决策排名",
    "precision_tier": "精度分层",
    "triage_priority": "处理优先级",
    "triage_action": "处理动作",
    "blocking_checks": "阻断项",
    "review_checks": "需复核项",
    "pass_count": "通过数",
    "review_count": "复核数",
    "missing_count": "缺失数",
    "triage_reason": "处理原因",
    "next_data_to_add": "建议补充数据",
}

MANUAL_KEY_RESIDUE_FOLLOWUP_COLUMNS = [
    "pocket_id",
    "decision_rank",
    "triage_priority",
    "precision_tier",
    "evidence_quality_label",
    "risk_flags",
    "blocking_checks",
    "review_checks",
    "manual_evidence_status",
    "manual_evidence_count",
    "manual_evidence_residues",
    "suggested_sources",
    "manual_template_columns",
    "closure_action",
    "recommended_action",
    "next_step",
]

MANUAL_KEY_RESIDUE_FOLLOWUP_COLUMN_LABELS = {
    "pocket_id": "口袋 ID",
    "decision_rank": "决策排名",
    "triage_priority": "补证优先级",
    "precision_tier": "精度分层",
    "evidence_quality_label": "证据质量",
    "risk_flags": "风险标签",
    "blocking_checks": "阻断项",
    "review_checks": "需复核项",
    "manual_evidence_status": "人工证据状态",
    "manual_evidence_count": "人工证据命中数",
    "manual_evidence_residues": "人工证据残基",
    "suggested_sources": "建议补证来源",
    "manual_template_columns": "人工证据模板列",
    "closure_action": "闭环动作",
    "recommended_action": "建议动作",
    "next_step": "下一步",
}

MANUAL_KEY_RESIDUE_FOLLOWUP_SUMMARY_COLUMNS = [
    "total_tasks",
    "closed_tasks",
    "mapping_review_tasks",
    "open_tasks",
    "manual_evidence_hits",
    "decision_gap_tasks",
    "release_gate_status",
    "closure_status",
    "recommended_next_step",
]

MANUAL_KEY_RESIDUE_FOLLOWUP_SUMMARY_COLUMN_LABELS = {
    "total_tasks": "补证任务数",
    "closed_tasks": "已补证任务",
    "mapping_review_tasks": "链/编号待确认",
    "open_tasks": "仍需补证",
    "manual_evidence_hits": "人工证据命中数",
    "decision_gap_tasks": "决策缺口任务数",
    "release_gate_status": "发布门控",
    "closure_status": "闭环状态",
    "recommended_next_step": "建议下一步",
}


def _localize_pocket_decision_text(value: object) -> object:
    if value is None:
        return value
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return text
    localized = POCKET_DECISION_VALUE_LABELS.get(text) or POCKET_DECISION_TEXT_LABELS.get(text)
    if localized:
        return localized
    for source, target in POCKET_DECISION_VALUE_LABELS.items():
        text = text.replace(source, target)
    for source, target in POCKET_DECISION_TEXT_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def _localize_pocket_decision_list(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "none":
        return "无"
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return "，".join(str(_localize_pocket_decision_text(part)) for part in parts) if parts else "无"


def _localize_pocket_decision_df(table: pd.DataFrame, column_labels: dict[str, str]) -> pd.DataFrame:
    if table is None or getattr(table, "empty", True):
        return table
    display = table.copy()
    for column in display.columns:
        if pd.api.types.is_object_dtype(display[column]) or pd.api.types.is_string_dtype(display[column]):
            display[column] = display[column].map(_localize_pocket_decision_text)
    return display.rename(columns=column_labels)


MANUAL_KEY_RESIDUE_GAP_MARKERS = [
    "evidence-gap",
    "functional-evidence-gap",
    "needs-functional-evidence",
    "geometry-only",
    "no-external-evidence",
    "manual key residues",
    "manual-key-residue",
    "add uniprot",
    "fetch or upload external residue evidence",
]


def _has_manual_key_residue_gap_text(text_parts: list[str]) -> bool:
    combined = " ".join(str(part or "") for part in text_parts).lower()
    return any(marker in combined for marker in MANUAL_KEY_RESIDUE_GAP_MARKERS)


def _pocket_row_text(row: pd.Series | None, column: str, default: str = "-") -> str:
    if row is None or column not in row.index:
        return default
    value = row.get(column)
    if value is None:
        return default
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _residue_key_from_row(row: pd.Series) -> tuple[str, int] | None:
    if "resid" not in row.index:
        return None
    try:
        resid = int(float(row.get("resid")))
    except (TypeError, ValueError):
        return None
    chain = str(row.get("chain") or "").strip()
    return chain, resid


def _format_residue_key(chain: str, resid: int) -> str:
    return f"{chain}:{int(resid)}" if str(chain or "").strip() else str(int(resid))


def _manual_evidence_overlap_for_pocket(
    pocket_id: str,
    pocket_residue_df: pd.DataFrame | None,
    manual_evidence_df: pd.DataFrame | None,
) -> tuple[str, int, str]:
    if (
        pocket_residue_df is None
        or getattr(pocket_residue_df, "empty", True)
        or manual_evidence_df is None
        or getattr(manual_evidence_df, "empty", True)
        or "pocket_id" not in pocket_residue_df.columns
    ):
        return "仍需补证", 0, "none"

    pocket_rows = pocket_residue_df[pocket_residue_df["pocket_id"].astype(str) == str(pocket_id)]
    if pocket_rows.empty:
        return "仍需补证", 0, "none"

    pocket_keys = {
        key
        for _, row in pocket_rows.iterrows()
        for key in [_residue_key_from_row(row)]
        if key is not None
    }
    manual_keys = {
        key
        for _, row in manual_evidence_df.iterrows()
        for key in [_residue_key_from_row(row)]
        if key is not None
    }
    if not pocket_keys or not manual_keys:
        return "仍需补证", 0, "none"

    exact_matches = pocket_keys & manual_keys
    if exact_matches:
        residues = ", ".join(
            _format_residue_key(chain, resid)
            for chain, resid in sorted(exact_matches, key=lambda item: (item[0], item[1]))
        )
        return "已补人工证据", len(exact_matches), residues

    pocket_resids = {resid for _chain, resid in pocket_keys}
    residue_only_matches = {key for key in manual_keys if key[1] in pocket_resids}
    if residue_only_matches:
        residues = ", ".join(
            _format_residue_key(chain, resid)
            for chain, resid in sorted(residue_only_matches, key=lambda item: (item[0], item[1]))
        )
        return "已补人工证据（需确认链/编号）", len(residue_only_matches), residues

    return "仍需补证", 0, "none"


def _needs_manual_key_residue_evidence(decision_df: pd.DataFrame | None, triage_df: pd.DataFrame | None) -> bool:
    text_parts: list[str] = []
    for table, columns in [
        (decision_df, ["decision_label", "evidence_quality_label", "recommended_action", "next_step", "risk_flags"]),
        (triage_df, ["precision_tier", "triage_action", "triage_reason", "blocking_checks", "review_checks"]),
    ]:
        if table is None or getattr(table, "empty", True):
            continue
        for column in columns:
            if column in table.columns:
                text_parts.extend(table[column].dropna().astype(str).tolist())
    return _has_manual_key_residue_gap_text(text_parts)


def _build_manual_key_residue_followup_df(
    decision_df: pd.DataFrame | None,
    triage_df: pd.DataFrame | None,
    pocket_residue_df: pd.DataFrame | None = None,
    manual_evidence_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    decision_rows: dict[str, pd.Series] = {}
    triage_rows: dict[str, pd.Series] = {}
    pocket_order: list[str] = []

    for table, target in [(decision_df, decision_rows), (triage_df, triage_rows)]:
        if table is None or getattr(table, "empty", True) or "pocket_id" not in table.columns:
            continue
        for _, row in table.iterrows():
            pocket_id = _pocket_row_text(row, "pocket_id", "")
            if not pocket_id:
                continue
            target[pocket_id] = row
            if pocket_id not in pocket_order:
                pocket_order.append(pocket_id)

    records = []
    for pocket_id in pocket_order:
        decision_row = decision_rows.get(pocket_id)
        triage_row = triage_rows.get(pocket_id)
        gap_text = [
            _pocket_row_text(decision_row, "decision_label", ""),
            _pocket_row_text(decision_row, "evidence_quality_label", ""),
            _pocket_row_text(decision_row, "recommended_action", ""),
            _pocket_row_text(decision_row, "next_step", ""),
            _pocket_row_text(decision_row, "risk_flags", ""),
            _pocket_row_text(triage_row, "precision_tier", ""),
            _pocket_row_text(triage_row, "triage_action", ""),
            _pocket_row_text(triage_row, "triage_reason", ""),
            _pocket_row_text(triage_row, "blocking_checks", ""),
            _pocket_row_text(triage_row, "review_checks", ""),
            _pocket_row_text(triage_row, "next_data_to_add", ""),
        ]
        if not _has_manual_key_residue_gap_text(gap_text):
            continue
        manual_status, manual_count, manual_residues = _manual_evidence_overlap_for_pocket(
            pocket_id,
            pocket_residue_df,
            manual_evidence_df,
        )
        closure_action = (
            "重新运行自动口袋识别，并检查该口袋是否转为功能证据锚定。"
            if manual_status == "已补人工证据"
            else (
                "确认链和编号映射后重新运行自动口袋识别。"
                if manual_status.startswith("已补人工证据")
                else "补充 UniProt、M-CSA、文献或人工关键残基后重新运行自动口袋识别。"
            )
        )
        records.append(
            {
                "pocket_id": pocket_id,
                "decision_rank": _pocket_row_text(decision_row, "decision_rank", _pocket_row_text(triage_row, "decision_rank")),
                "triage_priority": _pocket_row_text(triage_row, "triage_priority"),
                "precision_tier": _pocket_row_text(triage_row, "precision_tier"),
                "evidence_quality_label": _pocket_row_text(decision_row, "evidence_quality_label"),
                "risk_flags": _pocket_row_text(decision_row, "risk_flags"),
                "blocking_checks": _pocket_row_text(triage_row, "blocking_checks"),
                "review_checks": _pocket_row_text(triage_row, "review_checks"),
                "manual_evidence_status": manual_status,
                "manual_evidence_count": manual_count,
                "manual_evidence_residues": manual_residues,
                "suggested_sources": "UniProt 活性/结合位点；M-CSA 催化残基；PMID/DOI 文献；人工关键残基 CSV",
                "manual_template_columns": "chain,resid,resname,evidence_type,evidence_source,evidence_note,pmid,doi,evidence_snippet",
                "closure_action": closure_action,
                "recommended_action": _pocket_row_text(decision_row, "recommended_action"),
                "next_step": _pocket_row_text(decision_row, "next_step", _pocket_row_text(triage_row, "next_data_to_add")),
            }
        )

    return pd.DataFrame(records, columns=MANUAL_KEY_RESIDUE_FOLLOWUP_COLUMNS)


def _summarize_manual_key_residue_followup_df(followup_df: pd.DataFrame | None) -> pd.DataFrame:
    if followup_df is None or getattr(followup_df, "empty", True):
        return pd.DataFrame(columns=MANUAL_KEY_RESIDUE_FOLLOWUP_SUMMARY_COLUMNS)

    table = followup_df.copy()
    status_series = table.get("manual_evidence_status", pd.Series(["仍需补证"] * len(table), index=table.index))
    status_text = status_series.fillna("").astype(str)
    closed_mask = status_text.eq("已补人工证据")
    mapping_review_mask = status_text.str.contains("需确认链/编号", na=False)
    open_mask = ~(closed_mask | mapping_review_mask)
    evidence_hits = (
        pd.to_numeric(table.get("manual_evidence_count", pd.Series([0] * len(table), index=table.index)), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    total_tasks = int(len(table))
    closed_tasks = int(closed_mask.sum())
    mapping_review_tasks = int(mapping_review_mask.sum())
    open_tasks = int(open_mask.sum())
    decision_gap_tasks = total_tasks

    if total_tasks == 0:
        closure_status = "无补证任务"
        release_gate_status = "可进入验证"
        recommended_next_step = "暂无需要补证的候选口袋。"
    elif open_tasks == 0 and mapping_review_tasks == 0:
        closure_status = "补证已上传，等待决策缺口复核"
        release_gate_status = "不可直接作为活性位点"
        recommended_next_step = "重新运行自动口袋识别；只有精度分层不再是证据缺口，才可进入活性位点验证。"
    elif open_tasks == 0:
        closure_status = "补证已覆盖但需复核映射"
        release_gate_status = "不可直接作为活性位点"
        recommended_next_step = "先确认链和编号映射，再重新运行自动口袋识别。"
    elif closed_tasks or mapping_review_tasks:
        closure_status = "部分补证"
        release_gate_status = "不可直接作为活性位点"
        recommended_next_step = "继续补齐未命中的口袋，并复核链/编号待确认的人工证据。"
    else:
        closure_status = "仍需补证"
        release_gate_status = "不可直接作为活性位点"
        recommended_next_step = "优先补充 UniProt、M-CSA、文献或人工关键残基。"

    return pd.DataFrame(
        [
            {
                "total_tasks": total_tasks,
                "closed_tasks": closed_tasks,
                "mapping_review_tasks": mapping_review_tasks,
                "open_tasks": open_tasks,
                "manual_evidence_hits": int(evidence_hits.sum()),
                "decision_gap_tasks": decision_gap_tasks,
                "release_gate_status": release_gate_status,
                "closure_status": closure_status,
                "recommended_next_step": recommended_next_step,
            }
        ],
        columns=MANUAL_KEY_RESIDUE_FOLLOWUP_SUMMARY_COLUMNS,
    )


def _build_manual_key_residue_followup_checklist_markdown(
    followup_df: pd.DataFrame | None,
    summary_df: pd.DataFrame | None = None,
) -> str:
    if followup_df is None or getattr(followup_df, "empty", True):
        return ""
    if summary_df is None or getattr(summary_df, "empty", True):
        summary_df = _summarize_manual_key_residue_followup_df(followup_df)
    summary = summary_df.iloc[0] if summary_df is not None and not getattr(summary_df, "empty", True) else {}

    release_gate = str(summary.get("release_gate_status") or "-")
    closure_status = str(summary.get("closure_status") or "-")
    recommended_next_step = str(summary.get("recommended_next_step") or "-")
    lines = [
        "# 人工关键残基补证复跑检查清单",
        "",
        "## 当前门控",
        "",
        f"- 发布门控: {release_gate}",
        f"- 补证闭环状态: {closure_status}",
        f"- 补证任务数: {int(summary.get('total_tasks') or 0)}",
        f"- 已补证任务: {int(summary.get('closed_tasks') or 0)}",
        f"- 链/编号待确认: {int(summary.get('mapping_review_tasks') or 0)}",
        f"- 仍需补证: {int(summary.get('open_tasks') or 0)}",
        f"- 决策缺口任务数: {int(summary.get('decision_gap_tasks') or 0)}",
        "",
        "## 必做检查",
        "",
        "- [ ] 已补齐仍需补证的口袋，至少包含 UniProt、M-CSA、PMID/DOI 文献或人工关键残基来源。",
        "- [ ] 每条人工关键残基都确认 chain、resid、resname 和 PDB 编号体系一致。",
        "- [ ] 对链/编号待确认的记录完成映射复核，再重新运行自动口袋识别。",
        "- [ ] 重新运行后检查精度分层不再是 evidence-gap、geometry-only 或 no-external-evidence。",
        "- [ ] 重新运行后检查直接锚点数、锚点残基、可靠性检查表和发布门控。",
        "- [ ] 导出活性位点决策、可靠性检查、精度处理建议和补证任务，作为本轮审计记录。",
        "",
        "## 建议下一步",
        "",
        recommended_next_step,
        "",
        "## 口袋级闭环动作",
        "",
    ]

    for row in followup_df.head(12).itertuples(index=False):
        pocket_id = str(getattr(row, "pocket_id", "-") or "-")
        status = str(getattr(row, "manual_evidence_status", "-") or "-")
        residues = str(getattr(row, "manual_evidence_residues", "-") or "-")
        action = str(getattr(row, "closure_action", "-") or "-")
        lines.append(f"- {pocket_id}: {status}; 人工证据残基: {residues}; 闭环动作: {action}")

    if len(followup_df) > 12:
        lines.append(f"- 其余 {len(followup_df) - 12} 条任务请查看 CSV 明细。")

    return "\n".join(lines).rstrip() + "\n"


def _render_pocket_decision_panel(
    decision_df: pd.DataFrame,
    checklist_df: pd.DataFrame | None = None,
    triage_df: pd.DataFrame | None = None,
    manual_template_df: pd.DataFrame | None = None,
    manual_followup_df: pd.DataFrame | None = None,
    manual_followup_summary_df: pd.DataFrame | None = None,
    manual_followup_checklist_markdown: str = "",
) -> None:
    st.subheader("活性位点决策面板")
    st.caption(
        "把口袋排名转换成可审计的产品视图：同时展示功能证据、几何支持、A/B 变化、风险标签和下一步动作。"
    )
    if decision_df is None or getattr(decision_df, "empty", True):
        st.info("暂时没有可用的决策面板。请先运行自动口袋识别，或补充口袋证据。")
        return

    cards = []
    for _, row in decision_df.head(3).iterrows():
        audit_status = str(row.get("audit_status") or "")
        card_class = "ready" if audit_status == "ready-to-validate" else ("review" if "review" in audit_status or "needed" in audit_status else "explore")
        risk_flags = str(row.get("risk_flags") or "none")
        pills = "".join(
            f'<span class="decision-pill">{html.escape(_localize_pocket_decision_list(flag.strip()))}</span>'
            for flag in risk_flags.split(",")
            if flag.strip()
        )
        cards.append(
            """
            <div class="decision-card {card_class}">
              <div class="decision-kicker">排名 #{rank} | {audit_status}</div>
              <div class="decision-title">{pocket_id}</div>
              <div>{decision_label}</div>
              <div class="decision-score">{decision_score:.3f}</div>
              <div class="decision-meta">
                功能证据 {functional:.3f} / 几何支持 {geometry:.3f}<br/>
                证据质量：{quality}<br/>
                建议动作：{action}<br/>
                A/B：文献 {lit_delta:+d}，证据路径 {route_delta:+d}，保守性 {cons_delta:+d}
              </div>
              <div style="margin-top:8px;">{pills}</div>
              <div class="decision-meta"><strong>下一步：</strong>{next_step}</div>
            </div>
            """.format(
                card_class=card_class,
                rank=int(row.get("decision_rank") or 0),
                audit_status=html.escape(str(_localize_pocket_decision_text(audit_status)) or "-"),
                pocket_id=html.escape(str(row.get("pocket_id") or "-")),
                decision_label=html.escape(str(_localize_pocket_decision_text(row.get("decision_label") or "-"))),
                decision_score=float(row.get("decision_score") or 0.0),
                functional=float(row.get("functional_confidence") or 0.0),
                geometry=float(row.get("geometry_confidence") or 0.0),
                quality=html.escape(str(_localize_pocket_decision_text(row.get("evidence_quality_label") or "-"))),
                action=html.escape(str(_localize_pocket_decision_text(row.get("recommended_action") or "-"))),
                lit_delta=int(row.get("literature_rank_delta") or 0),
                route_delta=int(row.get("evidence_route_rank_delta") or 0),
                cons_delta=int(row.get("conservation_rank_delta") or 0),
                pills=pills or '<span class="decision-pill">无风险标签</span>',
                next_step=html.escape(str(_localize_pocket_decision_text(row.get("next_step") or "-"))),
            )
        )
    st.markdown(f'<div class="decision-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    if checklist_df is not None and not getattr(checklist_df, "empty", True):
        st.markdown("##### 可靠性检查表")
        st.caption(
            "通过 = 可用信号；需复核 = 有价值但需要人工检查；缺失 = 在认定为活性位点前仍有精度缺口。"
        )
        st.dataframe(
            _localize_pocket_decision_df(checklist_df, POCKET_RELIABILITY_COLUMN_LABELS),
            use_container_width=True,
            hide_index=True,
        )

    if triage_df is not None and not getattr(triage_df, "empty", True):
        st.markdown("##### 精度处理建议")
        st.caption(
            "把检查结果压缩成产品级动作：进入验证、复核映射、补证据、复核几何边界，或仅保留为探索结果。"
        )
        st.dataframe(
            _localize_pocket_decision_df(triage_df, POCKET_TRIAGE_COLUMN_LABELS),
            use_container_width=True,
            hide_index=True,
        )

    needs_manual_key_residue_evidence = _needs_manual_key_residue_evidence(decision_df, triage_df)
    if manual_followup_df is None:
        manual_followup_df = _build_manual_key_residue_followup_df(decision_df, triage_df)
    if needs_manual_key_residue_evidence:
        st.warning(
            "当前候选口袋仍缺少可审计功能残基证据。请先补充 UniProt、M-CSA、文献或人工关键残基，"
            "再把它作为活性位点。"
        )
        st.caption(
            "推荐列：chain、resid、resname、evidence_type、evidence_source、evidence_note、pmid、doi、"
            "evidence_snippet。上传后会并入外部位点证据，参与自动口袋识别、证据路径和最终重排。"
        )
        if manual_template_df is not None and not getattr(manual_template_df, "empty", True):
            st.download_button(
                "下载人工关键残基补证模板 CSV",
                data=_to_csv_bytes(manual_template_df),
                file_name="manual_key_residue_evidence_template.csv",
                mime="text/csv",
                key="decision_manual_key_residue_template",
            )

    if not manual_followup_df.empty:
        st.markdown("##### 人工关键残基补证任务")
        st.caption(
            "只列出缺少功能残基证据或仍主要依赖几何排名的候选口袋，并自动标记上传的人工残基是否已经命中该口袋。"
        )
        if manual_followup_summary_df is None:
            manual_followup_summary_df = _summarize_manual_key_residue_followup_df(manual_followup_df)
        if not manual_followup_checklist_markdown:
            manual_followup_checklist_markdown = _build_manual_key_residue_followup_checklist_markdown(
                manual_followup_df,
                manual_followup_summary_df,
            )
        if not manual_followup_summary_df.empty:
            summary_row = manual_followup_summary_df.iloc[0]
            summary_cols = st.columns(4)
            summary_cols[0].metric("补证任务数", int(summary_row.get("total_tasks") or 0))
            summary_cols[1].metric("已补证任务", int(summary_row.get("closed_tasks") or 0))
            summary_cols[2].metric("链/编号待确认", int(summary_row.get("mapping_review_tasks") or 0))
            summary_cols[3].metric("仍需补证", int(summary_row.get("open_tasks") or 0))
            release_gate_status = str(summary_row.get("release_gate_status") or "-")
            if release_gate_status != "可进入验证":
                st.error(f"发布门控：{release_gate_status}。任务表仍存在决策缺口时，不要把候选口袋直接作为活性位点。")
            st.caption(
                f"补证闭环状态：{summary_row.get('closure_status') or '-'}；"
                f"决策缺口任务数：{summary_row.get('decision_gap_tasks') or 0}；"
                f"建议下一步：{summary_row.get('recommended_next_step') or '-'}"
            )
            st.download_button(
                "导出人工关键残基补证闭环总览 CSV",
                data=_to_csv_bytes(manual_followup_summary_df),
                file_name="manual_key_residue_followup_summary.csv",
                mime="text/csv",
                key="download_manual_key_residue_followup_summary",
            )
        if manual_followup_checklist_markdown:
            with st.expander("补证复跑检查清单", expanded=False):
                st.markdown(manual_followup_checklist_markdown)
                st.download_button(
                    "导出人工关键残基补证复跑检查清单 MD",
                    data=manual_followup_checklist_markdown.encode("utf-8"),
                    file_name="manual_key_residue_followup_checklist.md",
                    mime="text/markdown",
                    key="download_manual_key_residue_followup_checklist",
                )
        st.dataframe(
            _localize_pocket_decision_df(manual_followup_df, MANUAL_KEY_RESIDUE_FOLLOWUP_COLUMN_LABELS),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "导出人工关键残基补证任务 CSV",
            data=_to_csv_bytes(manual_followup_df),
            file_name="manual_key_residue_followup_tasks.csv",
            mime="text/csv",
            key="download_manual_key_residue_followup_tasks",
        )

    with st.expander("决策审计明细", expanded=False):
        st.dataframe(
            _localize_pocket_decision_df(decision_df, POCKET_DECISION_COLUMN_LABELS),
            use_container_width=True,
            hide_index=True,
        )


POCKET_SOURCE_LABELS = {
    "uploaded": "上传 Pocket CSV",
    "auto": "自动识别口袋",
    "combined": "上传 + 自动合并",
}

ANNOTATION_SOURCE_LABELS = {
    "uploaded": "上传界面注释",
    "inferred": "结构推断界面",
    "combined": "上传 + 推断合并",
}

INFERENCE_BASIS_LABELS = {
    "inter-chain-contact": "跨链接触",
    "surface-contact": "表面接触",
}


cached_inputs = get_uploaded_inputs_cache()
cached_pdb_files = list(cached_inputs.get("pdb_files", []))
cached_mmpbsa_files = list(cached_inputs.get("mmpbsa_files", []))
cached_pocket_entry = cached_inputs.get("pocket_file")
shared_energy_table = get_current_energy_table()
shared_pdb_text = str(get_current_pdb_text() or "")
shared_mmpbsa_text = str(get_current_mmpbsa_text() or "")

if uploaded_pdb is not None:
    pdb_text = _read_uploaded_text(uploaded_pdb)
    pdb_input_note = "当前使用本页上传的 PDB。"
elif use_examples:
    pdb_text = PDB_TEXT
    pdb_input_note = "当前使用示例 PDB。"
elif shared_energy_table is not None and shared_pdb_text.strip():
    pdb_text = shared_pdb_text
    pdb_input_note = "当前复用结构可视化页最近一次分析的 PDB。"
elif cached_pdb_files:
    pdb_text = str(cached_pdb_files[0].get("text") or "")
    pdb_input_note = f"当前复用已缓存的 PDB：{cached_pdb_files[0].get('name', 'uploaded.pdb')}。"
else:
    pdb_text = ""
    pdb_input_note = ""

if uploaded_mmpbsa is not None:
    mmpbsa_text = _read_uploaded_text(uploaded_mmpbsa)
    mmpbsa_input_note = "当前使用本页上传的 MMPBSA。"
elif use_examples:
    mmpbsa_text = MMPBSA_TEXT
    mmpbsa_input_note = "当前使用示例 MMPBSA。"
elif shared_mmpbsa_text.strip() and "结构估算" not in shared_mmpbsa_text:
    mmpbsa_text = shared_mmpbsa_text
    mmpbsa_input_note = "当前复用结构可视化页最近一次分析的 MMPBSA。"
elif cached_mmpbsa_files:
    mmpbsa_text = str(cached_mmpbsa_files[0].get("text") or "")
    mmpbsa_input_note = f"当前复用已缓存的 MMPBSA：{cached_mmpbsa_files[0].get('name', 'uploaded_mmpbsa')}。"
else:
    mmpbsa_text = ""
    mmpbsa_input_note = ""

if uploaded_pocket is not None:
    pocket_text = _read_uploaded_text(uploaded_pocket)
    pocket_input_note = "当前使用本页上传的 Pocket。"
elif use_examples:
    pocket_text = POCKET_TEXT
    pocket_input_note = "当前使用示例 Pocket。"
elif cached_pocket_entry and cached_pocket_entry.get("text"):
    pocket_text = str(cached_pocket_entry.get("text") or "")
    pocket_input_note = f"当前复用已缓存的 Pocket：{cached_pocket_entry.get('name', 'uploaded_pocket.csv')}。"
else:
    pocket_text = ""
    pocket_input_note = ""

if uploaded_annotation is not None:
    annotation_text = _read_uploaded_text(uploaded_annotation)
elif use_examples:
    annotation_text = ANNOTATION_TEXT
else:
    annotation_text = ""

if not use_examples:
    new_pdb_entry = _uploaded_file_entry(uploaded_pdb)
    new_mmpbsa_entry = _uploaded_file_entry(uploaded_mmpbsa)
    new_pocket_entry = _uploaded_file_entry(uploaded_pocket)
    if new_pdb_entry is not None or new_mmpbsa_entry is not None or new_pocket_entry is not None:
        set_uploaded_inputs_cache(
            pdb_files=[new_pdb_entry] if new_pdb_entry is not None else cached_pdb_files,
            mmpbsa_files=[new_mmpbsa_entry] if new_mmpbsa_entry is not None else cached_mmpbsa_files,
            pocket_file=new_pocket_entry if new_pocket_entry is not None else cached_pocket_entry,
        )

if not pdb_text:
    st.warning("请上传 PDB 文件或勾选使用示例数据。")
    st.stop()

for input_note in [pdb_input_note, mmpbsa_input_note, pocket_input_note]:
    if input_note:
        st.caption(input_note)

try:
    atom_df = parse_pdb_atoms(pdb_text)
except Exception as exc:
    st.error(f"PDB 解析失败：{exc}")
    st.stop()

try:
    energy_df, energy_source = resolve_energy_table(
        pdb_text,
        energy_mode=energy_mode,
        mmpbsa_text=mmpbsa_text or None,
    )
    if energy_df is not None and not energy_df.empty:
        energy_table = prepare_energy_table(atom_df, energy_df)
        energy_table["energy_source"] = energy_source
    else:
        energy_table = pd.DataFrame()
except Exception:
    energy_table = pd.DataFrame()

try:
    structure_energy_df, _ = resolve_energy_table(
        pdb_text,
        energy_mode="estimate",
        mmpbsa_text=None,
    )
    if structure_energy_df is not None and not structure_energy_df.empty:
        structure_energy_table = prepare_energy_table(atom_df, structure_energy_df)
        structure_energy_table["energy_source"] = "结构估算"
    else:
        structure_energy_table = pd.DataFrame()
except Exception:
    structure_energy_table = pd.DataFrame()

if not energy_table.empty and "delta_total" in energy_table.columns:
    energy_series = pd.to_numeric(energy_table["delta_total"], errors="coerce").dropna()
    energy_limit = float(max(abs(energy_series.min()), abs(energy_series.max()))) if not energy_series.empty else 0.1
else:
    energy_limit = 0.1

hotspot_slider_max = max(energy_limit, 0.1)
hotspot_slider_value = 0.0 if energy_limit <= 0 else min(1.0, hotspot_slider_max)

with st.sidebar:
    st.header("热点筛选")
    hotspot_threshold = st.slider(
        "MMPBSA |阈值| (绝对值)",
        0.0,
        hotspot_slider_max,
        hotspot_slider_value,
        0.1,
        disabled=energy_table.empty,
    )
    hotspot_top_n = st.slider("热点保底数量", 1, 20, 5, 1, disabled=energy_table.empty)

    with st.expander("自动口袋识别参数", expanded=False):
        auto_adaptive_profile = st.checkbox("启用结构自适应参数", value=True)
        auto_use_kvfinder = st.checkbox(
            "优先使用 pyKVFinder 口袋引擎（推荐）",
            value=True,
            disabled=not PYKVFINDER_AVAILABLE,
        )
        auto_use_p2rank = st.checkbox("本地已安装时启用 P2Rank 增强", value=False)
        p2rank_profile = st.selectbox("P2Rank 配置", ["default", "alphafold"], index=0)
        p2rank_executable = st.text_input(
            "P2Rank 可执行文件路径（可选）",
            value="",
            placeholder="例如: C:\\tools\\p2rank\\prank.bat",
        )
        enable_p2rank_ab = st.checkbox(
            "显示 P2Rank A/B 口袋对比",
            value=False,
            disabled=not auto_use_p2rank,
        )
        if PYKVFINDER_AVAILABLE:
            st.caption("当前环境已检测到 pyKVFinder，将优先使用库算法进行口袋识别。")
        else:
            st.caption("当前环境未检测到 pyKVFinder，将使用几何启发式检测。")

        auto_detection_mode = st.selectbox(
            "识别策略",
            ["auto", "geometry"],
            index=0,
            format_func=lambda x: {"auto": "配体优先", "geometry": "纯几何"}[x],
        )
        auto_ligand_radius = st.slider("配体邻域半径", 3.0, 8.0, 5.0, 0.5)
        auto_contact_cutoff = st.slider("残基接触半径", 6.0, 12.0, 8.0, 0.5)
        auto_cluster_cutoff = st.slider("聚类距离阈值", 4.0, 14.0, 8.5, 0.5)
        auto_candidate_fraction = st.slider("候选残基比例", 0.15, 0.60, 0.35, 0.05)
        auto_max_candidates = st.slider("最大候选残基数", 3, 24, 18, 1)
        auto_max_pockets = st.slider("最多口袋数", 1, 8, 6, 1)

    with st.expander("外部关键位点证据（可选）", expanded=False):
        enable_uniprot_evidence = st.checkbox("启用 UniProt 功能位点增强", value=False)
        enable_mcsa_evidence = st.checkbox("启用 M-CSA 催化位点增强", value=False)
        uniprot_accession = st.text_input("UniProt 编号", value="", placeholder="例如: P00533")
        enzyme_ec_number = st.text_input("EC 编号（可选）", value="", placeholder="例如: 3.2.1.4")
        uniprot_chain_hint = st.text_input("链提示（可选）", value="", placeholder="例如: A")
        enable_literature_evidence = st.checkbox("启用文献残基挖掘", value=False)
        literature_query = st.text_input("文献检索词覆盖（可选）", value="", placeholder="例如: enzyme name catalytic residue")
        literature_protein_name = st.text_input("用于文献检索的蛋白名称（可选）", value="")
        enable_europepmc_evidence = st.checkbox(
            "启用 Europe PMC 开放文本挖掘",
            value=False,
            disabled=not enable_literature_evidence,
        )
        include_europepmc_fulltext = st.checkbox(
            "包含 Europe PMC 开放全文",
            value=True,
            disabled=not enable_literature_evidence or not enable_europepmc_evidence,
        )
        literature_max_articles = st.slider(
            "扫描文献数量",
            1,
            12,
            6,
            1,
            disabled=not enable_literature_evidence,
        )
        literature_max_fulltext = st.slider(
            "扫描 Europe PMC 全文数量",
            0,
            4,
            2,
            1,
            disabled=not enable_literature_evidence or not enable_europepmc_evidence or not include_europepmc_fulltext,
        )
        uploaded_literature = st.file_uploader(
            "上传文献正文 / 摘要（可选）",
            type=["txt", "md", "xml"],
            accept_multiple_files=False,
        )
        uploaded_manual_key_residues = st.file_uploader(
            "上传人工关键残基 CSV/TSV（可选）",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption(
            "人工关键残基列建议：chain,resid,resname,evidence_type,evidence_source,evidence_note,pmid,doi,evidence_snippet。"
            "这些行会并入外部位点证据，参与自动口袋识别、证据路径和最终重排。"
        )
        st.download_button(
            "下载人工关键残基模板 CSV",
            data=_to_csv_bytes(manual_key_residue_template_df),
            file_name="manual_key_residue_evidence_template.csv",
            mime="text/csv",
        )
        literature_assume_structure_numbering = st.checkbox(
            "假设文献残基编号与上传 PDB 链编号一致",
            value=False,
            disabled=not bool(str(uniprot_chain_hint or "").strip()),
        )
        enable_ai_evidence = st.checkbox("启用 AI 残基证据助手", value=False)
        ai_context_text = st.text_area(
            "AI 来源文本 / 备注",
            value="",
            height=120,
            disabled=not enable_ai_evidence,
            placeholder="粘贴摘要、论文片段或审核备注。AI 只应提取有来源支持的酶残基。",
        )
        ai_payload_text = st.text_area(
            "粘贴 AI 残基 JSON（可选）",
            value="",
            height=110,
            disabled=not enable_ai_evidence,
            placeholder='{"residues":[{"resname":"SER","position_text":"Ser195","confidence":0.86,"evidence_snippet":"..."}]}',
        )
        ai_api_url = st.text_input(
            "AI API 地址（兼容 OpenAI，可选）",
            value=os.getenv("AI_EVIDENCE_API_URL", ""),
            disabled=not enable_ai_evidence or bool(str(ai_payload_text or "").strip()),
            placeholder="https://.../v1/chat/completions",
        )
        ai_model = st.text_input(
            "AI 模型",
            value=os.getenv("AI_EVIDENCE_MODEL", ""),
            disabled=not enable_ai_evidence or bool(str(ai_payload_text or "").strip()),
            placeholder="模型名称",
        )
        ai_api_key = st.text_input(
            "AI API Key（可选；也支持环境变量 AI_EVIDENCE_API_KEY）",
            value="",
            type="password",
            disabled=not enable_ai_evidence or bool(str(ai_payload_text or "").strip()),
        )
        ai_min_confidence = st.slider(
            "AI 残基最低置信度",
            0.20,
            0.90,
            0.35,
            0.05,
            disabled=not enable_ai_evidence,
        )
        ai_assume_structure_numbering = st.checkbox(
            "假设 AI 残基编号与上传 PDB 链编号一致",
            value=False,
            disabled=not enable_ai_evidence or not bool(str(uniprot_chain_hint or "").strip()),
        )
        ai_allow_review_ranking = st.checkbox(
            "允许复核级 AI 证据影响排名",
            value=False,
            disabled=not enable_ai_evidence,
        )
        uploaded_ai_review_decisions = st.file_uploader(
            "上传 AI 复核决策 CSV/TSV（可选）",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
            disabled=not enable_ai_evidence,
        )
        st.caption("决策列：chain,resid,evidence_type,review_decision,verified_source,verified_snippet,review_note。")
        uploaded_consensus_rerank_release_decisions = st.file_uploader(
            "上传共识重排发布决策 CSV（可选）",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption("发布决策列：decision_item_id,review_decision,reviewer,verified_anchor_residues,verified_sources,blocker_resolved。")
        uploaded_consensus_rerank_release_execution_receipt = st.file_uploader(
            "上传共识重排发布执行回执 CSV（可选）",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption("执行回执列：execution_item_id,execution_decision,applied_rank,operator,executed_at,plan_sha256。")
        st.caption("AI 证据受审计门控保护：冲突或无来源支持的 AI 残基会保留用于复核/导出，但不会影响排名。")
        enable_literature_ab = st.checkbox(
            "显示文献 A/B 口袋对比",
            value=False,
            disabled=not enable_literature_evidence and uploaded_literature is None,
        )
        auto_external_evidence_route = st.checkbox("启用外部证据引导的口袋路径", value=True)
        external_route_min_support = st.slider(
            "证据路径最低支持度",
            0.30,
            1.00,
            0.58,
            0.02,
            disabled=not auto_external_evidence_route,
        )
        external_route_min_confidence = st.slider(
            "证据路径最低映射置信度",
            0.30,
            1.00,
            0.55,
            0.02,
            disabled=not auto_external_evidence_route,
        )
        external_route_min_quality = st.slider(
            "证据路径最低映射质量",
            0.50,
            1.00,
            0.82,
            0.02,
            disabled=not auto_external_evidence_route,
        )
        external_route_radius_mode = st.selectbox(
            "证据路径邻域半径",
            ["auto", "manual"],
            index=0,
            disabled=not auto_external_evidence_route,
        )
        external_route_radius = (
            st.slider(
                "手动证据路径半径",
                3.5,
                12.0,
                6.0,
                0.5,
                disabled=not auto_external_evidence_route or external_route_radius_mode != "manual",
            )
            if external_route_radius_mode == "manual"
            else None
        )
        enable_evidence_route_ab = st.checkbox(
            "显示证据路径 A/B 口袋对比",
            value=False,
            disabled=not auto_external_evidence_route,
        )
        st.caption("UniProt、M-CSA 和高置信文献残基证据会用于自动口袋识别和最终重排。")

    with st.expander("保守性证据（可选）", expanded=False):
        uploaded_conservation = st.file_uploader(
            "上传 ConSurf / 保守性表格",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        conservation_source_name = st.text_input("保守性来源标签", value="ConSurf")
        enable_conservation_ab = st.checkbox(
            "显示保守性 A/B 排名对比",
            value=False,
            disabled=uploaded_conservation is None,
        )
        st.caption("导入的保守性分数只作为独立重排信号，不参与候选口袋生成。")

    with st.expander("基准参考残基（可选）", expanded=False):
        uploaded_benchmark_reference = st.file_uploader(
            "上传人工整理的催化残基 CSV/TSV",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        benchmark_source_name = st.text_input("基准来源标签", value="人工整理催化口袋基准")
        use_external_evidence_as_benchmark_reference = st.checkbox(
            "未上传人工基准时，使用已加载外部证据作为临时基准参考",
            value=False,
        )
        use_reviewed_candidate_as_benchmark_reference = st.checkbox(
            "未上传人工基准时，使用已接受的复核候选作为基准参考",
            value=True,
        )
        uploaded_benchmark_reference_candidate_review_decisions = st.file_uploader(
            "上传基准参考候选复核决策 CSV/TSV",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        uploaded_benchmark_reference_source_audit_case_decisions = st.file_uploader(
            "上传基准来源审计案例决策 CSV/TSV",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption(
            "可包含列：chain,resid,resname,reference_type,reference_source,reference_note,expected_pocket_id。"
            "空 chain 会按通配处理。临时外部证据参考可用于分诊，但单独整理前不应作为独立准确率证明。"
        )
        st.caption("候选复核决策需要：action_id,review_decision,reviewer,verified_source,verified_mapping,review_note。")
        st.caption("来源审计案例决策需要：benchmark_id,source_decision,reviewer,verified_independence,decision_note。")
        st.download_button(
            "下载基准参考模板 CSV",
            data=_to_csv_bytes(benchmark_reference_template_df),
            file_name="pocket_benchmark_reference_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "下载基准参考模板说明",
            data=benchmark_reference_template_markdown.encode("utf-8"),
            file_name="pocket_benchmark_reference_template.md",
            mime="text/markdown",
        )

benchmark_reference_text = _read_uploaded_text(uploaded_benchmark_reference) if uploaded_benchmark_reference is not None else ""

hotspot_df = (
    identify_hotspots(
        energy_table,
        energy_threshold=-abs(hotspot_threshold) if hotspot_threshold > 0 else -1.0,
        top_n=hotspot_top_n,
    )
    if not energy_table.empty
    else pd.DataFrame()
)

try:
    uploaded_pocket_df = parse_pocket_table(pocket_text) if pocket_text else pd.DataFrame()
except Exception as exc:
    st.warning(f"口袋文件解析失败：{exc}")
    uploaded_pocket_df = pd.DataFrame()

try:
    uploaded_annotation_df = parse_interface_annotation_table(annotation_text) if annotation_text else pd.DataFrame()
except Exception as exc:
    st.warning(f"界面注释文件解析失败：{exc}")
    uploaded_annotation_df = pd.DataFrame()

hotspot_residues = _residue_pairs(hotspot_df)

external_site_df = pd.DataFrame()
external_site_meta: dict = {}
manual_key_residue_df = pd.DataFrame()
manual_key_residue_meta: dict = {}
literature_site_df = pd.DataFrame()
literature_site_meta: dict = {}
ai_evidence_df = pd.DataFrame()
ai_evidence_meta: dict = {}
ai_evidence_audit_df = pd.DataFrame()
ai_evidence_review_queue_df = pd.DataFrame()
ai_review_decision_template_df = pd.DataFrame()
ai_review_decision_df = pd.DataFrame()
ai_review_decision_validation_df = pd.DataFrame()
ai_review_decision_outcome_df = pd.DataFrame()
ai_review_round_summary_df = pd.DataFrame()
ai_review_ranking_delta_df = pd.DataFrame()
ai_review_round_report_markdown = ""
ai_review_artifact_manifest_df = pd.DataFrame()
ai_review_bundle_readme_markdown = ""
ai_review_artifact_bundle_zip = b""
ai_review_bundle_verification_df = pd.DataFrame()
ai_review_bundle_verification_summary_df = pd.DataFrame()
ai_review_bundle_certificate_markdown = ""
ai_review_decision_meta: dict = {}
rankable_ai_evidence_df = pd.DataFrame()
rankable_ai_evidence_before_review_df = pd.DataFrame()
rankable_ai_evidence_meta: dict = {}
conservation_site_df = pd.DataFrame()
conservation_site_meta: dict = {}
residue_evidence_consensus_df = pd.DataFrame()
pocket_consensus_coverage_df = pd.DataFrame()
benchmark_reference_df = pd.DataFrame()
benchmark_reference_meta: dict = {}
benchmark_reference_candidate_df = pd.DataFrame()
benchmark_reference_candidate_meta: dict = {}
benchmark_reference_import_summary_df = pd.DataFrame()
benchmark_reference_candidate_review_queue_df = pd.DataFrame()
benchmark_reference_candidate_review_checklist_markdown = ""
benchmark_reference_candidate_review_decision_template_df = pd.DataFrame()
benchmark_reference_candidate_review_decision_df = pd.DataFrame()
benchmark_reference_candidate_review_decision_meta: dict = {}
benchmark_reference_candidate_review_decision_validation_df = pd.DataFrame()
benchmark_reference_candidate_review_outcome_df = pd.DataFrame()
benchmark_reference_candidate_accepted_df = pd.DataFrame()
benchmark_reference_is_provisional = False
benchmark_reference_is_reviewed_candidate = False
benchmark_reference_source_mode = ""
benchmark_reference_source_audit_df = pd.DataFrame()
benchmark_reference_source_audit_summary_df = pd.DataFrame()
benchmark_reference_source_audit_action_queue_df = pd.DataFrame()
benchmark_reference_source_audit_case_summary_df = pd.DataFrame()
benchmark_reference_source_audit_case_summary_blocked_cases = 0
benchmark_reference_source_audit_case_summary_review_cases = 0
benchmark_reference_source_audit_case_checklist_markdown = ""
benchmark_reference_source_audit_case_decision_template_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_meta: dict = {}
benchmark_reference_source_audit_case_decision_validation_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_outcome_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_outcome_summary_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_outcome_summary_status = ""
benchmark_reference_source_audit_case_decision_outcome_summary_open_cases = 0
benchmark_reference_source_audit_case_decision_closure_queue_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_closure_checklist_markdown = ""
benchmark_reference_source_audit_case_decision_readiness_impact_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_readiness_impact_summary_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_readiness_impact_summary_status = ""
benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases = 0
benchmark_reference_source_audit_checklist_markdown = ""
pocket_benchmark_reference_quality_issue_df = pd.DataFrame()
pocket_benchmark_reference_quality_summary_df = pd.DataFrame()
pocket_benchmark_reference_quality_checklist_markdown = ""
pocket_benchmark_reference_structure_validation_df = pd.DataFrame()
pocket_benchmark_reference_structure_validation_summary_df = pd.DataFrame()
pocket_benchmark_reference_structure_validation_checklist_markdown = ""
pocket_benchmark_reference_readiness_queue_df = pd.DataFrame()
pocket_benchmark_reference_readiness_summary_df = pd.DataFrame()
pocket_benchmark_reference_readiness_case_summary_df = pd.DataFrame()
pocket_benchmark_reference_readiness_checklist_markdown = ""
pocket_benchmark_interpretation_df = pd.DataFrame()
pocket_benchmark_case_interpretation_df = pd.DataFrame()
pocket_benchmark_case_interpretation_matrix_df = pd.DataFrame()
pocket_benchmark_case_interpretation_matrix_summary_df = pd.DataFrame()
pocket_benchmark_case_interpretation_matrix_queue_df = pd.DataFrame()
pocket_benchmark_dataset_interpretation_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown = ""
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown = ""
pocket_benchmark_dataset_interpretation_queue_df = pd.DataFrame()
pocket_benchmark_dataset_interpretation_checklist_markdown = ""
pocket_benchmark_dataset_interpretation_report_markdown = ""
pocket_benchmark_summary_df = pd.DataFrame()
pocket_benchmark_details_df = pd.DataFrame()
pocket_benchmark_case_summary_df = pd.DataFrame()
pocket_benchmark_dataset_summary_df = pd.DataFrame()
pocket_benchmark_variant_comparison_df = pd.DataFrame()
pocket_benchmark_variant_case_comparison_df = pd.DataFrame()
pocket_benchmark_variant_dataset_comparison_df = pd.DataFrame()
pocket_benchmark_variant_detail_comparison_df = pd.DataFrame()
pocket_benchmark_variant_remediation_df = pd.DataFrame()
pocket_benchmark_variant_remediation_summary_df = pd.DataFrame()
pocket_benchmark_variant_remediation_checklist_markdown = ""
consensus_rerank_suggestion_df = pd.DataFrame()
consensus_rerank_preview_df = pd.DataFrame()
consensus_rerank_policy_gate_df = pd.DataFrame()
consensus_rerank_action_queue_df = pd.DataFrame()
consensus_rerank_action_checklist_markdown = ""
consensus_rerank_apply_simulation_df = pd.DataFrame()
consensus_rerank_simulation_delta_df = pd.DataFrame()
consensus_rerank_precision_scorecard_df = pd.DataFrame()
consensus_rerank_precision_guardrail_df = pd.DataFrame()
consensus_rerank_precision_guardrail_report_markdown = ""
consensus_rerank_guardrail_artifact_manifest_df = pd.DataFrame()
consensus_rerank_guardrail_handoff_zip = b""
consensus_rerank_guardrail_bundle_verification_df = pd.DataFrame()
consensus_rerank_guardrail_bundle_verification_summary_df = pd.DataFrame()
consensus_rerank_guardrail_handoff_certificate_markdown = ""
consensus_rerank_release_decision_template_df = pd.DataFrame()
consensus_rerank_release_apply_plan_df = pd.DataFrame()
consensus_rerank_release_apply_report_markdown = ""
consensus_rerank_release_execution_template_df = pd.DataFrame()
consensus_rerank_release_execution_receipt_df = pd.DataFrame()
consensus_rerank_release_execution_receipt_meta: dict = {}
consensus_rerank_release_execution_validation_df = pd.DataFrame()
consensus_rerank_release_execution_summary_df = pd.DataFrame()
consensus_rerank_release_execution_report_markdown = ""
consensus_rerank_release_closure_certificate_markdown = ""
consensus_rerank_release_closure_ledger_df = pd.DataFrame()
consensus_rerank_release_closure_summary_df = pd.DataFrame()
consensus_rerank_release_closure_blocker_df = pd.DataFrame()
consensus_rerank_release_closure_remediation_checklist_markdown = ""
consensus_rerank_release_closure_detached_manifest_df = pd.DataFrame()
structure_pdb_id = str(extract_pdb_id_from_text(pdb_text) or "").strip().upper()
if (enable_uniprot_evidence and str(uniprot_accession or "").strip()) or (
    enable_mcsa_evidence and (str(uniprot_accession or "").strip() or str(enzyme_ec_number or "").strip())
):
    with st.spinner("正在加载外部功能位点证据..."):
        external_site_df, external_site_meta = _load_external_evidence(
            str(uniprot_accession or "").strip(),
            str(uniprot_chain_hint or "").strip(),
            structure_pdb_id,
            pdb_text,
            str(enzyme_ec_number or "").strip(),
            bool(enable_uniprot_evidence),
            bool(enable_mcsa_evidence),
        )
manual_key_residue_text = _read_uploaded_text(uploaded_manual_key_residues) if uploaded_manual_key_residues is not None else ""
if manual_key_residue_text.strip():
    try:
        manual_key_residue_df, manual_key_residue_meta = parse_manual_key_residue_table(
            manual_key_residue_text,
            source_hint="manual",
        )
    except Exception as exc:
        st.warning(f"人工关键残基解析失败：{exc}")
        manual_key_residue_df = pd.DataFrame()
        manual_key_residue_meta = {"status": "parse-error", "error": str(exc)}
    if not manual_key_residue_df.empty:
        external_site_df = merge_external_evidence_tables(external_site_df, manual_key_residue_df)
        counts = _external_evidence_counts(external_site_df)
        source_values = []
        if str(external_site_meta.get("sources") or "").strip():
            source_values.extend(str(external_site_meta.get("sources")).split(","))
        source_values.extend(str(manual_key_residue_meta.get("sources") or "manual").split(","))
        external_site_meta = {
            **external_site_meta,
            "status": "ok",
            "sources": ",".join(dict.fromkeys(source.strip() for source in source_values if source.strip())),
            "evidence_rows": str(counts["rows"]),
            "exact_rows": str(counts["exact"]),
            "weak_rows": str(counts["weak"]),
            "manual_key_residue_rows": str(len(manual_key_residue_df)),
            "manual_key_residue": manual_key_residue_meta,
        }
        st.success(f"人工关键残基：{len(manual_key_residue_df)} 条已并入外部位点证据。")
    elif str(manual_key_residue_meta.get("status") or "") != "empty":
        st.warning("人工关键残基表未产生可用残基行，请至少提供 resid 或 residue_number 列。")
literature_manual_text = _read_uploaded_text(uploaded_literature) if uploaded_literature is not None else ""
if bool(enable_literature_evidence) or literature_manual_text.strip():
    with st.spinner("正在加载文献残基证据..."):
        literature_site_df, literature_site_meta = _load_literature_evidence(
            str(literature_query or "").strip(),
            literature_manual_text,
            str(uniprot_accession or "").strip(),
            str(enzyme_ec_number or "").strip(),
            structure_pdb_id,
            str(literature_protein_name or "").strip(),
            str(uniprot_chain_hint or "").strip(),
            pdb_text,
            int(literature_max_articles),
            bool(enable_literature_evidence),
            bool(enable_europepmc_evidence),
            bool(include_europepmc_fulltext),
            int(literature_max_fulltext),
            bool(literature_assume_structure_numbering),
        )
    if not literature_site_df.empty:
        external_site_df = merge_external_evidence_tables(external_site_df, literature_site_df)
        counts = _external_evidence_counts(external_site_df)
        source_values = []
        if str(external_site_meta.get("sources") or "").strip():
            source_values.extend(str(external_site_meta.get("sources")).split(","))
        source_values.append("literature")
        external_site_meta = {
            **external_site_meta,
            "status": "ok",
            "sources": ",".join(dict.fromkeys(source.strip() for source in source_values if source.strip())),
            "evidence_rows": str(counts["rows"]),
            "exact_rows": str(counts["exact"]),
            "weak_rows": str(counts["weak"]),
            "literature": literature_site_meta,
        }
if bool(enable_literature_evidence) or literature_manual_text.strip():
    if literature_site_df.empty:
        st.sidebar.caption("文献残基挖掘未产生可用的高置信残基。")
    else:
        st.sidebar.caption(
            f"文献证据：{len(literature_site_df)} 行 / "
            f"状态 {_localize_status_text(literature_site_meta.get('status'))} / "
            f"检索词 {literature_site_meta.get('query') or '-'}"
        )
if bool(enable_ai_evidence):
    ai_source_text = str(ai_context_text or literature_manual_text or "").strip()
    reference_evidence_before_ai_df = external_site_df.copy()
    with st.spinner("正在加载 AI 残基证据..."):
        if str(ai_payload_text or "").strip():
            ai_evidence_df, ai_evidence_meta = parse_ai_residue_evidence_payload(
                str(ai_payload_text or ""),
                chain_hint=str(uniprot_chain_hint or "").strip(),
                min_confidence=float(ai_min_confidence),
                assume_structure_numbering=bool(ai_assume_structure_numbering),
                pdb_text=pdb_text,
            )
        elif ai_source_text.strip():
            ai_evidence_df, ai_evidence_meta = fetch_ai_residue_evidence(
                ai_source_text,
                api_url=str(ai_api_url or "").strip(),
                api_key=str(ai_api_key or "").strip(),
                model=str(ai_model or "").strip(),
                chain_hint=str(uniprot_chain_hint or "").strip(),
                protein_name=str(literature_protein_name or "").strip(),
                accession=str(uniprot_accession or "").strip(),
                pdb_id=structure_pdb_id,
                ec_number=str(enzyme_ec_number or "").strip(),
                triage_context="",
                min_confidence=float(ai_min_confidence),
                assume_structure_numbering=bool(ai_assume_structure_numbering),
                pdb_text=pdb_text,
            )
        else:
            ai_evidence_df, ai_evidence_meta = pd.DataFrame(), {
                "status": "empty-input",
                "evidence_rows": "0",
                "message": "Paste source text, upload literature text, paste AI JSON, or configure an AI API call.",
            }
    ai_evidence_audit_df = build_ai_evidence_audit_table(ai_evidence_df, reference_evidence_before_ai_df)
    rankable_ai_evidence_before_review_df, _rankable_ai_evidence_before_review_meta = filter_ai_evidence_for_ranking(
        ai_evidence_df,
        ai_evidence_audit_df,
        allow_review=bool(ai_allow_review_ranking),
    )
    if uploaded_ai_review_decisions is not None:
        ai_review_decision_text = _read_uploaded_text(uploaded_ai_review_decisions)
        if ai_review_decision_text.strip():
            ai_review_decision_df, ai_review_decision_meta = parse_ai_review_decision_table(ai_review_decision_text)
            ai_review_decision_validation_df = build_ai_review_decision_validation_table(
                ai_review_decision_df,
                ai_evidence_audit_df,
            )
            decisions_for_apply_df = ai_review_decision_df
            if not ai_review_decision_validation_df.empty and "issue_flags" in ai_review_decision_validation_df.columns:
                blocked_duplicate_rows = ai_review_decision_validation_df[
                    ai_review_decision_validation_df["issue_flags"].astype(str).str.contains("conflicting-duplicate", case=False, na=False)
                ]
                if not blocked_duplicate_rows.empty and "row_index" in blocked_duplicate_rows.columns:
                    blocked_row_numbers = set(pd.to_numeric(blocked_duplicate_rows["row_index"], errors="coerce").dropna().astype(int).tolist())
                    decisions_for_apply_df = ai_review_decision_df.reset_index(drop=True).iloc[
                        [index for index in range(len(ai_review_decision_df)) if index + 1 not in blocked_row_numbers]
                    ].copy()
            ai_evidence_audit_df, ai_review_decision_apply_meta = apply_ai_review_decisions_to_audit(
                ai_evidence_audit_df,
                decisions_for_apply_df,
            )
            ai_review_decision_outcome_df = build_ai_review_decision_outcome_table(
                ai_review_decision_df,
                ai_evidence_audit_df,
            )
            ai_review_decision_meta = {
                **ai_review_decision_meta,
                "apply_status": str(ai_review_decision_apply_meta.get("status") or ""),
                "applied_rows": str(ai_review_decision_apply_meta.get("applied_rows") or "0"),
                "accepted_rows": str(ai_review_decision_apply_meta.get("accepted_rows") or "0"),
                "rejected_rows": str(ai_review_decision_apply_meta.get("rejected_rows") or "0"),
                "review_rows_after_apply": str(ai_review_decision_apply_meta.get("review_rows") or "0"),
                "conflict_blocked_rows": str(ai_review_decision_apply_meta.get("conflict_blocked_rows") or "0"),
                "validation_rows": str(len(ai_review_decision_validation_df)),
                "validation_blocked_rows": str(
                    int((ai_review_decision_validation_df["validation_status"].astype(str) == "blocked").sum())
                    if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns
                    else 0
                ),
                "outcome_rows": str(len(ai_review_decision_outcome_df)),
            }
    ai_evidence_review_queue_df = build_ai_evidence_review_queue(ai_evidence_audit_df)
    ai_review_decision_template_df = build_ai_review_decision_template(ai_evidence_review_queue_df)
    rankable_ai_evidence_df, rankable_ai_evidence_meta = filter_ai_evidence_for_ranking(
        ai_evidence_df,
        ai_evidence_audit_df,
        allow_review=bool(ai_allow_review_ranking),
    )
    ai_review_round_summary_df = build_ai_review_round_summary(
        ai_review_decision_df,
        ai_review_decision_validation_df,
        ai_review_decision_outcome_df,
        rankable_ai_evidence_df,
    )
    if not ai_review_decision_df.empty:
        ai_review_ranking_delta_df = build_ai_review_ranking_delta(
            rankable_ai_evidence_before_review_df,
            rankable_ai_evidence_df,
        )
        ai_review_round_report_markdown = build_ai_review_round_report_markdown(
            ai_review_round_summary_df,
            ai_review_decision_validation_df,
            ai_review_decision_outcome_df,
            ai_review_ranking_delta_df,
        )
    ai_review_artifact_manifest_df = build_ai_review_artifact_manifest(
        review_queue_df=ai_evidence_review_queue_df,
        decision_template_df=ai_review_decision_template_df,
        normalized_decision_df=ai_review_decision_df,
        validation_df=ai_review_decision_validation_df,
        round_summary_df=ai_review_round_summary_df,
        ranking_delta_df=ai_review_ranking_delta_df,
        outcome_df=ai_review_decision_outcome_df,
        round_report_markdown=ai_review_round_report_markdown,
    )
    if not ai_review_artifact_manifest_df.empty:
        ai_review_bundle_readme_markdown = build_ai_review_bundle_readme_markdown(ai_review_artifact_manifest_df)
        ai_review_artifact_manifest_df = build_ai_review_artifact_manifest(
            review_queue_df=ai_evidence_review_queue_df,
            decision_template_df=ai_review_decision_template_df,
            normalized_decision_df=ai_review_decision_df,
            validation_df=ai_review_decision_validation_df,
            round_summary_df=ai_review_round_summary_df,
            ranking_delta_df=ai_review_ranking_delta_df,
            outcome_df=ai_review_decision_outcome_df,
            round_report_markdown=ai_review_round_report_markdown,
            bundle_readme_markdown=ai_review_bundle_readme_markdown,
        )
    ai_review_artifact_bundle_zip = build_ai_review_artifact_bundle_zip(
        review_queue_df=ai_evidence_review_queue_df,
        decision_template_df=ai_review_decision_template_df,
        normalized_decision_df=ai_review_decision_df,
        validation_df=ai_review_decision_validation_df,
        round_summary_df=ai_review_round_summary_df,
        ranking_delta_df=ai_review_ranking_delta_df,
        outcome_df=ai_review_decision_outcome_df,
        artifact_manifest_df=ai_review_artifact_manifest_df,
        round_report_markdown=ai_review_round_report_markdown,
        bundle_readme_markdown=ai_review_bundle_readme_markdown,
    )
    ai_review_bundle_verification_df = verify_ai_review_artifact_bundle_zip(
        ai_review_artifact_bundle_zip,
        ai_review_artifact_manifest_df,
    )
    ai_review_bundle_verification_summary_df = build_ai_review_bundle_verification_summary(
        ai_review_bundle_verification_df
    )
    ai_review_bundle_certificate_markdown = build_ai_review_bundle_certificate_markdown(
        ai_review_artifact_bundle_zip,
        ai_review_bundle_verification_summary_df,
        ai_review_artifact_manifest_df,
    )
    if not rankable_ai_evidence_df.empty:
        external_site_df = merge_external_evidence_tables(external_site_df, rankable_ai_evidence_df)
        counts = _external_evidence_counts(external_site_df)
        source_values = []
        if str(external_site_meta.get("sources") or "").strip():
            source_values.extend(str(external_site_meta.get("sources")).split(","))
        source_values.append("AI")
        external_site_meta = {
            **external_site_meta,
            "status": "ok",
            "sources": ",".join(dict.fromkeys(source.strip() for source in source_values if source.strip())),
            "evidence_rows": str(counts["rows"]),
            "exact_rows": str(counts["exact"]),
            "weak_rows": str(counts["weak"]),
            "ai": ai_evidence_meta,
            "ai_ranking": rankable_ai_evidence_meta,
        }
    if ai_evidence_df.empty:
        st.sidebar.caption(f"AI 证据助手：{_localize_status_text(ai_evidence_meta.get('status') or 'empty')}")
    else:
        audit_status_text = (
            ", ".join(f"{_localize_status_text(status)}:{count}" for status, count in ai_evidence_audit_df["audit_status"].astype(str).value_counts().to_dict().items())
            if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns
            else "无"
        )
        st.sidebar.caption(
            f"AI 证据：{len(ai_evidence_df)} 行 / 状态 {_localize_status_text(ai_evidence_meta.get('status'))} / "
            f"进入排名 {len(rankable_ai_evidence_df)} / 需人工复核 {ai_evidence_meta.get('manual_review_rows') or '0'} / 审计 {audit_status_text}"
        )
    if ai_review_decision_meta:
        st.sidebar.caption(
            f"AI 复核决策：{len(ai_review_decision_df)} 行 / "
            f"状态 {_localize_status_text(ai_review_decision_meta.get('status'))} / "
            f"已应用 {ai_review_decision_meta.get('applied_rows') or '0'} / "
            f"接受 {ai_review_decision_meta.get('accepted_rows') or '0'} / "
            f"拒绝 {ai_review_decision_meta.get('rejected_rows') or '0'}"
        )
        if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns:
            validation_text = ", ".join(
                f"{_localize_status_text(status)}:{count}"
                for status, count in ai_review_decision_validation_df["validation_status"].astype(str).value_counts().to_dict().items()
            )
            st.sidebar.caption(f"AI 复核校验：{validation_text}")
        if not ai_review_decision_outcome_df.empty and "applied_status" in ai_review_decision_outcome_df.columns:
            outcome_text = ", ".join(
                f"{_localize_status_text(status)}:{count}"
                for status, count in ai_review_decision_outcome_df["applied_status"].astype(str).value_counts().to_dict().items()
            )
            st.sidebar.caption(f"AI 复核结果：{outcome_text}")
        if not ai_review_round_summary_df.empty:
            summary_row = ai_review_round_summary_df.iloc[0]
            st.sidebar.caption(
                f"AI 复核轮次：{_localize_status_text(summary_row.get('review_round_status'))} / "
                f"可进入排名 {summary_row.get('rankable_after_review_rows') or 0}"
            )
        if not ai_review_ranking_delta_df.empty:
            delta_row = ai_review_ranking_delta_df.iloc[0]
            st.sidebar.caption(
                f"AI 复核排名变化：{_localize_status_text(delta_row.get('review_effect_status'))} / "
                f"+{delta_row.get('promoted_rows') or 0} / -{delta_row.get('removed_rows') or 0}"
            )
        if not ai_review_artifact_manifest_df.empty:
            st.sidebar.caption(f"AI 复核产物清单：{len(ai_review_artifact_manifest_df)} 个文件")
        if ai_review_artifact_bundle_zip:
            st.sidebar.caption("AI 复核产物包：已就绪")
        if not ai_review_bundle_verification_df.empty and "verification_status" in ai_review_bundle_verification_df.columns:
            failed_verification = int((ai_review_bundle_verification_df["verification_status"].astype(str) != "verified").sum())
            st.sidebar.caption(f"AI 复核包校验：{len(ai_review_bundle_verification_df)} 个文件 / 失败 {failed_verification}")
        if not ai_review_bundle_verification_summary_df.empty:
            verify_row = ai_review_bundle_verification_summary_df.iloc[0]
            st.sidebar.caption(
                f"AI 复核包校验汇总：{_localize_status_text(verify_row.get('verification_status'))} / "
                f"失败 {verify_row.get('failed_files') or 0}"
            )
        if ai_review_bundle_certificate_markdown:
            st.sidebar.caption("AI 复核包交接证书：已就绪")
if not external_site_df.empty:
    benchmark_reference_candidate_df, benchmark_reference_candidate_meta = build_pocket_benchmark_reference_from_external_evidence(
        external_site_df,
        default_benchmark_id=str(structure_pdb_id or uniprot_accession or enzyme_ec_number or "current-structure").strip(),
        source_hint=str(external_site_meta.get("sources") or "Loaded external evidence").strip() or "Loaded external evidence",
    )
    benchmark_reference_import_summary_df = build_pocket_benchmark_reference_import_summary(
        benchmark_reference_candidate_df,
        benchmark_reference_candidate_meta,
    )
    benchmark_reference_candidate_review_queue_df = build_pocket_benchmark_reference_candidate_review_queue(
        benchmark_reference_candidate_df
    )
    benchmark_reference_candidate_review_checklist_markdown = (
        build_pocket_benchmark_reference_candidate_review_checklist_markdown(
            benchmark_reference_candidate_review_queue_df
        )
    )
    benchmark_reference_candidate_review_decision_template_df = (
        build_pocket_benchmark_reference_candidate_review_decision_template(
            benchmark_reference_candidate_review_queue_df
        )
    )
    review_decision_text = (
        _read_uploaded_text(uploaded_benchmark_reference_candidate_review_decisions)
        if uploaded_benchmark_reference_candidate_review_decisions is not None
        else ""
    )
    if review_decision_text.strip():
        (
            benchmark_reference_candidate_review_decision_df,
            benchmark_reference_candidate_review_decision_meta,
        ) = parse_pocket_benchmark_reference_candidate_review_decision_table(review_decision_text)
        benchmark_reference_candidate_review_decision_validation_df = (
            build_pocket_benchmark_reference_candidate_review_decision_validation(
                benchmark_reference_candidate_review_decision_df,
                benchmark_reference_candidate_review_queue_df,
            )
        )
        benchmark_reference_candidate_review_outcome_df = (
            build_pocket_benchmark_reference_candidate_review_outcomes(
                benchmark_reference_candidate_review_queue_df,
                benchmark_reference_candidate_review_decision_df,
                benchmark_reference_candidate_review_decision_validation_df,
            )
        )
    else:
        benchmark_reference_candidate_review_outcome_df = (
            build_pocket_benchmark_reference_candidate_review_outcomes(
                benchmark_reference_candidate_review_queue_df,
                pd.DataFrame(),
            )
        )
    benchmark_reference_candidate_accepted_df = (
        build_pocket_benchmark_reference_candidate_accepted_reference(
            benchmark_reference_candidate_df,
            benchmark_reference_candidate_review_queue_df,
            benchmark_reference_candidate_review_outcome_df,
        )
    )
    if not benchmark_reference_import_summary_df.empty:
        import_row = benchmark_reference_import_summary_df.iloc[0]
        st.sidebar.caption(
            f"基准参考候选：{import_row.get('reference_rows') or 0} 个残基 / "
            f"{_localize_status_text(import_row.get('import_status'))}。"
        )
    if not benchmark_reference_candidate_review_queue_df.empty:
        st.sidebar.caption(
            f"基准参考候选复核：{len(benchmark_reference_candidate_review_queue_df)} 个动作。"
        )
    if benchmark_reference_candidate_review_decision_meta:
        st.sidebar.caption(
            f"基准参考候选决策：{benchmark_reference_candidate_review_decision_meta.get('decision_rows') or 0} 行 / "
            f"状态 {_localize_status_text(benchmark_reference_candidate_review_decision_meta.get('status'))}。"
        )
if uploaded_conservation is not None:
    conservation_text = _read_uploaded_text(uploaded_conservation)
    if conservation_text.strip():
        with st.spinner("正在加载保守性证据..."):
            conservation_site_df, conservation_site_meta = parse_conservation_evidence_table(
                conservation_text,
                chain_hint=str(uniprot_chain_hint or "").strip(),
                source_hint=str(conservation_source_name or "").strip() or "ConSurf",
            )
benchmark_reference_loaded = False
benchmark_reference_uploaded = bool(benchmark_reference_text.strip())
if benchmark_reference_uploaded:
    benchmark_reference_df, benchmark_reference_meta = parse_benchmark_reference_table(
        benchmark_reference_text,
        source_hint=str(benchmark_source_name or "").strip() or "Curated catalytic benchmark",
    )
benchmark_reference_df, benchmark_reference_meta, benchmark_reference_selection = select_pocket_benchmark_reference_source(
    benchmark_reference_df,
    benchmark_reference_meta,
    curated_reference_uploaded=benchmark_reference_uploaded,
    external_candidate_df=benchmark_reference_candidate_df,
    external_candidate_meta=benchmark_reference_candidate_meta,
    accepted_candidate_df=benchmark_reference_candidate_accepted_df,
    prefer_reviewed_candidate=bool(use_reviewed_candidate_as_benchmark_reference),
    allow_provisional_candidate=bool(use_external_evidence_as_benchmark_reference),
)
benchmark_reference_loaded = bool(benchmark_reference_selection.get("loaded"))
benchmark_reference_is_provisional = bool(benchmark_reference_selection.get("is_provisional"))
benchmark_reference_is_reviewed_candidate = bool(benchmark_reference_selection.get("is_reviewed_candidate"))
benchmark_reference_source_mode = str(benchmark_reference_selection.get("source_mode") or "")
if benchmark_reference_selection.get("message"):
    st.sidebar.caption(_localize_report_line(str(benchmark_reference_selection.get("message"))))
if benchmark_reference_loaded:
    if benchmark_reference_df.empty:
        st.sidebar.caption(
            f"基准参考：无可用行（{_localize_status_text(benchmark_reference_meta.get('reason') or benchmark_reference_meta.get('status'))}）。"
        )
    else:
        benchmark_reference_source_audit_df = build_pocket_benchmark_reference_source_audit(
            benchmark_reference_df,
            source_mode=benchmark_reference_source_mode,
            is_provisional=benchmark_reference_is_provisional,
            is_reviewed_candidate=benchmark_reference_is_reviewed_candidate,
        )
        benchmark_reference_source_audit_summary_df = build_pocket_benchmark_reference_source_audit_summary(
            benchmark_reference_source_audit_df
        )
        benchmark_reference_source_audit_action_queue_df = build_pocket_benchmark_reference_source_audit_action_queue(
            benchmark_reference_source_audit_df
        )
        benchmark_reference_source_audit_case_summary_df = build_pocket_benchmark_reference_source_audit_case_summary(
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_action_queue_df,
        )
        (
            benchmark_reference_source_audit_case_summary_blocked_cases,
            benchmark_reference_source_audit_case_summary_review_cases,
        ) = _source_audit_case_summary_counts(benchmark_reference_source_audit_case_summary_df)
        benchmark_reference_source_audit_case_checklist_markdown = (
            build_pocket_benchmark_reference_source_audit_case_checklist_markdown(
                benchmark_reference_source_audit_case_summary_df,
                benchmark_reference_source_audit_action_queue_df,
            )
        )
        benchmark_reference_source_audit_case_decision_template_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_template(
                benchmark_reference_source_audit_case_summary_df
            )
        )
        source_audit_case_decision_text = (
            _read_uploaded_text(uploaded_benchmark_reference_source_audit_case_decisions)
            if uploaded_benchmark_reference_source_audit_case_decisions is not None
            else ""
        )
        if source_audit_case_decision_text.strip():
            (
                benchmark_reference_source_audit_case_decision_df,
                benchmark_reference_source_audit_case_decision_meta,
            ) = parse_pocket_benchmark_reference_source_audit_case_decision_table(
                source_audit_case_decision_text
            )
            benchmark_reference_source_audit_case_decision_validation_df = (
                build_pocket_benchmark_reference_source_audit_case_decision_validation(
                    benchmark_reference_source_audit_case_decision_df,
                    benchmark_reference_source_audit_case_summary_df,
                )
            )
        benchmark_reference_source_audit_case_decision_outcome_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_outcomes(
                benchmark_reference_source_audit_case_summary_df,
                benchmark_reference_source_audit_case_decision_df,
                benchmark_reference_source_audit_case_decision_validation_df,
            )
        )
        benchmark_reference_source_audit_case_decision_outcome_summary_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_outcome_summary(
                benchmark_reference_source_audit_case_decision_outcome_df
            )
        )
        if not benchmark_reference_source_audit_case_decision_outcome_summary_df.empty:
            _source_audit_outcome_summary_row = benchmark_reference_source_audit_case_decision_outcome_summary_df.iloc[0]
            benchmark_reference_source_audit_case_decision_outcome_summary_status = str(
                _source_audit_outcome_summary_row.get("closure_status") or ""
            )
            benchmark_reference_source_audit_case_decision_outcome_summary_open_cases = int(
                _source_audit_outcome_summary_row.get("open_actionable_case_count") or 0
            )
        benchmark_reference_source_audit_case_decision_closure_queue_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_closure_queue(
                benchmark_reference_source_audit_case_decision_outcome_df
            )
        )
        benchmark_reference_source_audit_case_decision_closure_checklist_markdown = (
            build_pocket_benchmark_reference_source_audit_case_decision_closure_checklist_markdown(
                benchmark_reference_source_audit_case_decision_outcome_summary_df,
                benchmark_reference_source_audit_case_decision_outcome_df,
            )
        )
        benchmark_reference_source_audit_case_decision_readiness_impact_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact(
                benchmark_reference_source_audit_df,
                benchmark_reference_source_audit_case_decision_outcome_df,
            )
        )
        benchmark_reference_source_audit_case_decision_readiness_impact_summary_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary(
                benchmark_reference_source_audit_case_decision_readiness_impact_df
            )
        )
        if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty:
            _source_audit_readiness_impact_summary_row = benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.iloc[0]
            benchmark_reference_source_audit_case_decision_readiness_impact_summary_status = str(
                _source_audit_readiness_impact_summary_row.get("readiness_impact_status") or ""
            )
            benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases = int(
                _source_audit_readiness_impact_summary_row.get("open_after_decision_cases") or 0
            )
        benchmark_reference_source_audit_checklist_markdown = build_pocket_benchmark_reference_source_audit_checklist_markdown(
            benchmark_reference_source_audit_summary_df,
            benchmark_reference_source_audit_df,
        )
        if benchmark_reference_is_provisional:
            st.sidebar.caption(
                "基准参考提醒：当前候选可能与识别输入重叠，做精度声明前需要单独整理和复核。"
            )
        pocket_benchmark_reference_quality_issue_df = build_pocket_benchmark_reference_quality_issues(benchmark_reference_df)
        pocket_benchmark_reference_quality_summary_df = build_pocket_benchmark_reference_quality_summary(
            pocket_benchmark_reference_quality_issue_df
        )
        pocket_benchmark_reference_quality_checklist_markdown = build_pocket_benchmark_reference_quality_checklist_markdown(
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_quality_summary_df,
        )
        pocket_benchmark_reference_structure_validation_df = build_pocket_benchmark_reference_structure_validation(
            benchmark_reference_df,
            atom_df,
        )
        pocket_benchmark_reference_structure_validation_summary_df = build_pocket_benchmark_reference_structure_validation_summary(
            pocket_benchmark_reference_structure_validation_df
        )
        pocket_benchmark_reference_structure_validation_checklist_markdown = (
            build_pocket_benchmark_reference_structure_validation_checklist_markdown(
                pocket_benchmark_reference_structure_validation_df,
                pocket_benchmark_reference_structure_validation_summary_df,
            )
        )
        pocket_benchmark_reference_readiness_queue_df = build_pocket_benchmark_reference_readiness_queue(
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_structure_validation_df,
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_case_decision_outcome_df,
        )
        pocket_benchmark_reference_readiness_summary_df = build_pocket_benchmark_reference_readiness_summary(
            benchmark_reference_df,
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_structure_validation_df,
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_case_decision_outcome_df,
        )
        pocket_benchmark_reference_readiness_case_summary_df = build_pocket_benchmark_reference_readiness_case_summary(
            benchmark_reference_df,
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_structure_validation_df,
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_case_decision_outcome_df,
        )
        pocket_benchmark_reference_readiness_checklist_markdown = build_pocket_benchmark_reference_readiness_checklist_markdown(
            pocket_benchmark_reference_readiness_queue_df,
            pocket_benchmark_reference_readiness_summary_df,
        )
        p1_quality_issues = (
            int(pocket_benchmark_reference_quality_issue_df["severity"].astype(str).isin(["P0", "P1"]).sum())
            if not pocket_benchmark_reference_quality_issue_df.empty and "severity" in pocket_benchmark_reference_quality_issue_df.columns
            else 0
        )
        p1_structure_issues = (
            int(pocket_benchmark_reference_structure_validation_df["severity"].astype(str).isin(["P0", "P1"]).sum())
            if not pocket_benchmark_reference_structure_validation_df.empty and "severity" in pocket_benchmark_reference_structure_validation_df.columns
            else 0
        )
        st.sidebar.caption(
            f"基准参考：{len(benchmark_reference_df)} 个残基 / "
            f"指定链 {benchmark_reference_meta.get('chain_specific_rows') or 0} / "
            f"通配链 {benchmark_reference_meta.get('wildcard_chain_rows') or 0}。"
        )
        st.sidebar.caption(
            f"基准参考整理：{len(pocket_benchmark_reference_quality_issue_df)} 个问题 / P0-P1 {p1_quality_issues}。"
        )
        st.sidebar.caption(
            f"基准参考结构校验：{len(pocket_benchmark_reference_structure_validation_df)} 个问题 / P0-P1 {p1_structure_issues}。"
        )
        if not pocket_benchmark_reference_readiness_summary_df.empty:
            readiness_row = pocket_benchmark_reference_readiness_summary_df.iloc[0]
            st.sidebar.caption(
                f"基准就绪：{_localize_status_text(readiness_row.get('readiness_status'))} / "
                f"阻断 {readiness_row.get('p0_p1_issue_count') or 0} / 复核 {readiness_row.get('p2_issue_count') or 0}。"
            )

residue_evidence_consensus_df = build_residue_evidence_consensus(
    external_site_df,
    ai_evidence_df=ai_evidence_df,
    ai_audit_df=ai_evidence_audit_df,
    rankable_ai_evidence_df=rankable_ai_evidence_df,
    conservation_df=conservation_site_df,
)

auto_pocket_df = detect_auto_pocket_table(
    pdb_text,
    hotspot_residues=hotspot_residues,
    external_site_df=external_site_df,
    conservation_site_df=conservation_site_df,
    adaptive_profile=auto_adaptive_profile,
    prefer_kvfinder=auto_use_kvfinder,
    prefer_p2rank=auto_use_p2rank,
    prefer_ligand=auto_detection_mode == "auto",
    enable_external_evidence_route=auto_external_evidence_route,
    external_evidence_min_support=external_route_min_support,
    external_evidence_min_confidence=external_route_min_confidence,
    external_evidence_min_mapping_quality=external_route_min_quality,
    external_evidence_radius=external_route_radius,
    contact_cutoff=auto_contact_cutoff,
    cluster_cutoff=auto_cluster_cutoff,
    ligand_radius=auto_ligand_radius,
    top_fraction=auto_candidate_fraction,
    min_candidates=3,
    max_candidates=auto_max_candidates,
    max_pockets=auto_max_pockets,
    p2rank_profile=p2rank_profile,
    p2rank_executable=str(p2rank_executable or "").strip() or None,
)
auto_detection_meta = get_pocket_detection_metadata(auto_pocket_df)
auto_detection_diag_df = build_pocket_detection_diagnostics_table(auto_pocket_df)
auto_detection_summary = summarize_pocket_detection_metadata(auto_detection_meta)
uploaded_pocket_df = _normalize_pocket_table(uploaded_pocket_df, "uploaded")
auto_pocket_df = _normalize_pocket_table(auto_pocket_df, "auto")
auto_pocket_summary = build_pocket_summary(auto_pocket_df, hotspot_df) if not auto_pocket_df.empty else pd.DataFrame()
uploaded_pocket_summary = build_pocket_summary(uploaded_pocket_df, hotspot_df) if not uploaded_pocket_df.empty else pd.DataFrame()
p2rank_ab_enabled = bool(enable_p2rank_ab and auto_use_p2rank and not auto_pocket_summary.empty)
auto_pocket_df_without_p2rank = pd.DataFrame()
auto_pocket_summary_without_p2rank = pd.DataFrame()
p2rank_ab_df = pd.DataFrame()
if p2rank_ab_enabled:
    auto_pocket_df_without_p2rank = detect_auto_pocket_table(
        pdb_text,
        hotspot_residues=hotspot_residues,
        external_site_df=external_site_df,
        conservation_site_df=conservation_site_df,
        adaptive_profile=auto_adaptive_profile,
        prefer_kvfinder=auto_use_kvfinder,
        prefer_p2rank=False,
        prefer_ligand=auto_detection_mode == "auto",
        enable_external_evidence_route=auto_external_evidence_route,
        external_evidence_min_support=external_route_min_support,
        external_evidence_min_confidence=external_route_min_confidence,
        external_evidence_min_mapping_quality=external_route_min_quality,
        external_evidence_radius=external_route_radius,
        contact_cutoff=auto_contact_cutoff,
        cluster_cutoff=auto_cluster_cutoff,
        ligand_radius=auto_ligand_radius,
        top_fraction=auto_candidate_fraction,
        min_candidates=3,
        max_candidates=auto_max_candidates,
        max_pockets=auto_max_pockets,
        p2rank_profile=p2rank_profile,
        p2rank_executable=str(p2rank_executable or "").strip() or None,
    )
    auto_pocket_df_without_p2rank = _normalize_pocket_table(auto_pocket_df_without_p2rank, "auto-no-p2rank")
    auto_pocket_summary_without_p2rank = (
        build_pocket_summary(auto_pocket_df_without_p2rank, hotspot_df)
        if not auto_pocket_df_without_p2rank.empty
        else pd.DataFrame()
    )
    p2rank_ab_df = compare_pocket_ranking_summaries(auto_pocket_summary_without_p2rank, auto_pocket_summary)
literature_ab_enabled = bool(enable_literature_ab and not literature_site_df.empty and not auto_pocket_summary.empty)
auto_pocket_df_without_literature = pd.DataFrame()
auto_pocket_summary_without_literature = pd.DataFrame()
literature_ab_df = pd.DataFrame()
if literature_ab_enabled:
    baseline_external_site_df = remove_literature_evidence(external_site_df)
    auto_pocket_df_without_literature = detect_auto_pocket_table(
        pdb_text,
        hotspot_residues=hotspot_residues,
        external_site_df=baseline_external_site_df,
        conservation_site_df=conservation_site_df,
        adaptive_profile=auto_adaptive_profile,
        prefer_kvfinder=auto_use_kvfinder,
        prefer_p2rank=auto_use_p2rank,
        prefer_ligand=auto_detection_mode == "auto",
        enable_external_evidence_route=auto_external_evidence_route,
        external_evidence_min_support=external_route_min_support,
        external_evidence_min_confidence=external_route_min_confidence,
        external_evidence_min_mapping_quality=external_route_min_quality,
        external_evidence_radius=external_route_radius,
        contact_cutoff=auto_contact_cutoff,
        cluster_cutoff=auto_cluster_cutoff,
        ligand_radius=auto_ligand_radius,
        top_fraction=auto_candidate_fraction,
        min_candidates=3,
        max_candidates=auto_max_candidates,
        max_pockets=auto_max_pockets,
        p2rank_profile=p2rank_profile,
        p2rank_executable=str(p2rank_executable or "").strip() or None,
    )
    auto_pocket_df_without_literature = _normalize_pocket_table(auto_pocket_df_without_literature, "auto-no-literature")
    auto_pocket_summary_without_literature = (
        build_pocket_summary(auto_pocket_df_without_literature, hotspot_df)
        if not auto_pocket_df_without_literature.empty
        else pd.DataFrame()
    )
    literature_ab_df = compare_pocket_ranking_summaries(auto_pocket_summary_without_literature, auto_pocket_summary)
evidence_route_ab_enabled = bool(
    enable_evidence_route_ab
    and auto_external_evidence_route
    and not external_site_df.empty
    and not auto_pocket_summary.empty
)
auto_pocket_df_without_evidence_route = pd.DataFrame()
auto_pocket_summary_without_evidence_route = pd.DataFrame()
evidence_route_ab_df = pd.DataFrame()
if evidence_route_ab_enabled:
    auto_pocket_df_without_evidence_route = detect_auto_pocket_table(
        pdb_text,
        hotspot_residues=hotspot_residues,
        external_site_df=external_site_df,
        conservation_site_df=conservation_site_df,
        adaptive_profile=auto_adaptive_profile,
        prefer_kvfinder=auto_use_kvfinder,
        prefer_p2rank=auto_use_p2rank,
        prefer_ligand=auto_detection_mode == "auto",
        enable_external_evidence_route=False,
        external_evidence_min_support=external_route_min_support,
        external_evidence_min_confidence=external_route_min_confidence,
        external_evidence_min_mapping_quality=external_route_min_quality,
        external_evidence_radius=external_route_radius,
        contact_cutoff=auto_contact_cutoff,
        cluster_cutoff=auto_cluster_cutoff,
        ligand_radius=auto_ligand_radius,
        top_fraction=auto_candidate_fraction,
        min_candidates=3,
        max_candidates=auto_max_candidates,
        max_pockets=auto_max_pockets,
        p2rank_profile=p2rank_profile,
        p2rank_executable=str(p2rank_executable or "").strip() or None,
    )
    auto_pocket_df_without_evidence_route = _normalize_pocket_table(auto_pocket_df_without_evidence_route, "auto-no-evidence-route")
    auto_pocket_summary_without_evidence_route = (
        build_pocket_summary(auto_pocket_df_without_evidence_route, hotspot_df)
        if not auto_pocket_df_without_evidence_route.empty
        else pd.DataFrame()
    )
    evidence_route_ab_df = compare_pocket_ranking_summaries(
        auto_pocket_summary_without_evidence_route,
        auto_pocket_summary,
    )
conservation_ab_enabled = bool(enable_conservation_ab and not conservation_site_df.empty and not auto_pocket_df.empty)
auto_pocket_summary_without_conservation = (
    build_pocket_summary_without_conservation_signal(auto_pocket_df, hotspot_df)
    if conservation_ab_enabled
    else pd.DataFrame()
)
conservation_ab_df = (
    compare_pocket_ranking_summaries(auto_pocket_summary_without_conservation, auto_pocket_summary)
    if conservation_ab_enabled and not auto_pocket_summary.empty
    else pd.DataFrame()
)

if not uploaded_annotation_df.empty:
    uploaded_annotation_df = uploaded_annotation_df.copy()
    uploaded_annotation_df["annotation_source"] = "uploaded"

inferred_annotation_df = build_inferred_interface_annotations(
    structure_energy_table if not structure_energy_table.empty else energy_table,
)

if (
    (enable_uniprot_evidence and str(uniprot_accession or "").strip())
    or enable_mcsa_evidence
    or enable_literature_evidence
    or not literature_site_df.empty
):
    if external_site_df.empty:
        st.sidebar.caption("未获取到可映射的外部功能位点证据，可能是编号不一致或网络不可用。")
    else:
        exact_rows = 0
        weak_rows = 0
        if "mapping_level" in external_site_df.columns:
            level_series = external_site_df["mapping_level"].astype(str).str.lower()
            exact_rows = int((level_series == "exact").sum())
            weak_rows = int((level_series == "weak").sum())
        mapping_status = _localize_status_text(external_site_meta.get("status") or external_site_meta.get("mapping_status"))
        mapping_pdb = str(external_site_meta.get("pdb_id") or structure_pdb_id or "-")
        source_text = _localize_status_text(external_site_meta.get("sources") or "external")
        st.sidebar.caption(
            f"已加载 {source_text} 位点证据 {len(external_site_df)} 条（精确 {exact_rows} / 弱命中 {weak_rows}，PDB {mapping_pdb}，{mapping_status}）。"
        )
if uploaded_conservation is not None:
    if conservation_site_df.empty:
        st.sidebar.caption("保守性证据文件未产生可用残基行。")
    else:
        st.sidebar.caption(
            f"保守性导入：{len(conservation_site_df)} 行 / 来源 {conservation_site_meta.get('source') or conservation_source_name} / "
            f"平均分 {conservation_site_meta.get('score_mean') or '-'}"
        )
if not residue_evidence_consensus_df.empty:
    top_consensus = residue_evidence_consensus_df.iloc[0]
    st.sidebar.caption(
        f"残基证据共识：{len(residue_evidence_consensus_df)} 个残基 / "
        f"最高 {top_consensus.get('residue_anchor') or '-'} / {top_consensus.get('consensus_tier') or '-'}"
    )

pocket_source_options: list[str] = []
if not uploaded_pocket_df.empty:
    pocket_source_options.append("uploaded")
if not auto_pocket_df.empty:
    pocket_source_options.append("auto")
if not uploaded_pocket_df.empty and not auto_pocket_df.empty:
    pocket_source_options.append("combined")

annotation_source_options: list[str] = []
if not uploaded_annotation_df.empty:
    annotation_source_options.append("uploaded")
if not inferred_annotation_df.empty:
    annotation_source_options.append("inferred")
if not uploaded_annotation_df.empty and not inferred_annotation_df.empty:
    annotation_source_options.append("combined")

with st.sidebar:
    st.header("分析来源")
    if pocket_source_options:
        default_pocket_mode = "combined" if "combined" in pocket_source_options else pocket_source_options[0]
        effective_pocket_mode = st.radio(
            "界面分析使用的口袋来源",
            pocket_source_options,
            index=pocket_source_options.index(default_pocket_mode),
            format_func=lambda value: POCKET_SOURCE_LABELS.get(value, value),
        )
    else:
        effective_pocket_mode = "auto"
        st.caption("当前没有可用 Pocket 数据源。")

    if annotation_source_options:
        default_annotation_mode = "combined" if "combined" in annotation_source_options else annotation_source_options[0]
        effective_annotation_mode = st.radio(
            "界面分析使用的注释来源",
            annotation_source_options,
            index=annotation_source_options.index(default_annotation_mode),
            format_func=lambda value: ANNOTATION_SOURCE_LABELS.get(value, value),
        )
    else:
        effective_annotation_mode = "inferred"
        st.caption("当前没有可用界面注释来源。")

effective_pocket_df = _resolve_pocket_source(uploaded_pocket_df, auto_pocket_df, effective_pocket_mode)
effective_pocket_summary = build_pocket_summary(effective_pocket_df, hotspot_df) if not effective_pocket_df.empty else pd.DataFrame()
effective_pocket_residues = _residue_pairs(effective_pocket_df)
pocket_consensus_coverage_df = build_pocket_consensus_coverage(
    effective_pocket_df,
    residue_evidence_consensus_df,
)

effective_annotation_base_df = _resolve_annotation_source(
    uploaded_annotation_df,
    inferred_annotation_df,
    effective_annotation_mode,
)
enriched_annotations = enrich_interface_annotations(
    effective_annotation_base_df,
    pocket_residues=effective_pocket_residues,
    hotspot_residues=hotspot_residues,
)
interface_summary = build_interface_summary(enriched_annotations)
overlap_summary = build_interface_overlap_summary(
    enriched_annotations,
    pocket_residues=effective_pocket_residues,
    hotspot_residues=hotspot_residues,
)
joint_candidate_df = build_joint_candidate_table(
    effective_pocket_df,
    enriched_annotations,
    hotspot_df,
    external_site_df=external_site_df,
)
top_joint_candidate = joint_candidate_df.iloc[0] if not joint_candidate_df.empty else None
top_pocket_consensus_coverage = pocket_consensus_coverage_df.iloc[0] if not pocket_consensus_coverage_df.empty else None
pocket_benchmark_summary_df = build_pocket_benchmark_summary(
    benchmark_reference_df,
    effective_pocket_df,
    effective_pocket_summary,
    top_ns=(1, 3, 5),
) if not benchmark_reference_df.empty else pd.DataFrame()
pocket_benchmark_details_df = build_pocket_benchmark_details(
    benchmark_reference_df,
    effective_pocket_df,
    effective_pocket_summary,
    top_thresholds=(1, 3, 5),
) if not benchmark_reference_df.empty else pd.DataFrame()
pocket_benchmark_case_summary_df = build_pocket_benchmark_case_summary(
    benchmark_reference_df,
    effective_pocket_df,
    effective_pocket_summary,
    top_ns=(1, 3, 5),
) if not benchmark_reference_df.empty else pd.DataFrame()
pocket_benchmark_dataset_summary_df = (
    build_pocket_benchmark_dataset_summary(pocket_benchmark_case_summary_df)
    if not pocket_benchmark_case_summary_df.empty
    else pd.DataFrame()
)
top1_benchmark = (
    pocket_benchmark_summary_df[pocket_benchmark_summary_df["top_n"].astype(int) == 1].iloc[0]
    if not pocket_benchmark_summary_df.empty and "top_n" in pocket_benchmark_summary_df.columns and (pocket_benchmark_summary_df["top_n"].astype(int) == 1).any()
    else None
)
top3_benchmark = (
    pocket_benchmark_summary_df[pocket_benchmark_summary_df["top_n"].astype(int) == 3].iloc[0]
    if not pocket_benchmark_summary_df.empty and "top_n" in pocket_benchmark_summary_df.columns and (pocket_benchmark_summary_df["top_n"].astype(int) == 3).any()
    else None
)
pocket_benchmark_interpretation_df = build_pocket_benchmark_interpretation_summary(
    pocket_benchmark_summary_df,
    pocket_benchmark_reference_readiness_summary_df,
)
pocket_benchmark_case_interpretation_df = build_pocket_benchmark_case_interpretation_summary(
    pocket_benchmark_case_summary_df,
    pocket_benchmark_reference_readiness_case_summary_df,
)
pocket_benchmark_case_interpretation_matrix_df = build_pocket_benchmark_case_interpretation_matrix(
    pocket_benchmark_case_interpretation_df,
    top_ns=(1, 3, 5),
)
pocket_benchmark_case_interpretation_matrix_summary_df = (
    build_pocket_benchmark_case_interpretation_matrix_summary(
        pocket_benchmark_case_interpretation_matrix_df
    )
)
pocket_benchmark_case_interpretation_matrix_queue_df = (
    build_pocket_benchmark_case_interpretation_matrix_queue(
        pocket_benchmark_case_interpretation_matrix_df
    )
)
pocket_benchmark_dataset_interpretation_df = build_pocket_benchmark_dataset_interpretation(
    pocket_benchmark_case_interpretation_df
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact(
        pocket_benchmark_case_interpretation_df,
        benchmark_reference_source_audit_case_decision_readiness_impact_df,
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases(
        pocket_benchmark_case_interpretation_df,
        benchmark_reference_source_audit_case_decision_readiness_impact_df,
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue(
        pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary(
        pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown(
        pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown(
        pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df,
        pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df,
        action_summary_df=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df,
        checklist_available=bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown),
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest(
        dataset_impact_df=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df,
        impact_case_df=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df,
        action_queue_df=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df,
        action_summary_df=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df,
        case_checklist_markdown=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown,
        report_markdown=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown,
    )
)
pocket_benchmark_dataset_interpretation_queue_df = build_pocket_benchmark_dataset_interpretation_queue(
    pocket_benchmark_case_interpretation_df
)
pocket_benchmark_dataset_interpretation_checklist_markdown = (
    build_pocket_benchmark_dataset_interpretation_checklist_markdown(
        pocket_benchmark_dataset_interpretation_queue_df
    )
)
pocket_benchmark_dataset_interpretation_report_markdown = (
    build_pocket_benchmark_dataset_interpretation_report_markdown(
        pocket_benchmark_dataset_interpretation_df,
        pocket_benchmark_dataset_interpretation_queue_df,
        checklist_available=bool(pocket_benchmark_dataset_interpretation_checklist_markdown),
    )
)
benchmark_variants: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
if not benchmark_reference_df.empty:
    benchmark_variants.append(("current", effective_pocket_df, effective_pocket_summary))
    if p2rank_ab_enabled and not auto_pocket_df_without_p2rank.empty:
        benchmark_variants.append(("no-p2rank", auto_pocket_df_without_p2rank, auto_pocket_summary_without_p2rank))
    if literature_ab_enabled and not auto_pocket_df_without_literature.empty:
        benchmark_variants.append(("no-literature", auto_pocket_df_without_literature, auto_pocket_summary_without_literature))
    if evidence_route_ab_enabled and not auto_pocket_df_without_evidence_route.empty:
        benchmark_variants.append(("no-evidence-route", auto_pocket_df_without_evidence_route, auto_pocket_summary_without_evidence_route))
    if conservation_ab_enabled and not auto_pocket_summary_without_conservation.empty:
        benchmark_variants.append(("no-conservation-rerank", auto_pocket_df, auto_pocket_summary_without_conservation))
pocket_benchmark_variant_comparison_df = (
    build_pocket_benchmark_variant_comparison(
        benchmark_reference_df,
        benchmark_variants,
        reference_variant_label="current",
        top_ns=(1, 3, 5),
    )
    if len(benchmark_variants) > 1
    else pd.DataFrame()
)
pocket_benchmark_variant_case_comparison_df = (
    build_pocket_benchmark_variant_case_comparison(
        benchmark_reference_df,
        benchmark_variants,
        reference_variant_label="current",
        top_ns=(1, 3, 5),
    )
    if len(benchmark_variants) > 1
    else pd.DataFrame()
)
pocket_benchmark_variant_dataset_comparison_df = (
    build_pocket_benchmark_variant_dataset_comparison(pocket_benchmark_variant_case_comparison_df)
    if not pocket_benchmark_variant_case_comparison_df.empty
    else pd.DataFrame()
)
pocket_benchmark_variant_detail_comparison_df = (
    build_pocket_benchmark_variant_detail_comparison(
        benchmark_reference_df,
        benchmark_variants,
        reference_variant_label="current",
        top_thresholds=(1, 3, 5),
    )
    if len(benchmark_variants) > 1
    else pd.DataFrame()
)
pocket_benchmark_variant_remediation_df = build_pocket_benchmark_variant_remediation_queue(
    pocket_benchmark_variant_detail_comparison_df
)
pocket_benchmark_variant_remediation_summary_df = build_pocket_benchmark_variant_remediation_summary(
    pocket_benchmark_variant_remediation_df
)
pocket_benchmark_variant_remediation_checklist_markdown = build_pocket_benchmark_variant_remediation_checklist_markdown(
    pocket_benchmark_variant_remediation_df,
    pocket_benchmark_variant_remediation_summary_df,
)
pocket_decision_df = build_pocket_decision_table(
    effective_pocket_summary,
    joint_candidate_df,
    literature_ab_df=literature_ab_df,
    evidence_route_ab_df=evidence_route_ab_df,
    conservation_ab_df=conservation_ab_df,
)
consensus_rerank_suggestion_df = build_consensus_rerank_suggestion(
    pocket_decision_df,
    pocket_consensus_coverage_df,
)
consensus_rerank_preview_df = build_consensus_rerank_preview(
    pocket_decision_df,
    consensus_rerank_suggestion_df,
)
consensus_rerank_policy_gate_df = build_consensus_rerank_policy_gate(consensus_rerank_preview_df)
consensus_rerank_action_queue_df = build_consensus_rerank_action_queue(
    consensus_rerank_preview_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_action_checklist_markdown = build_consensus_rerank_action_checklist_markdown(
    consensus_rerank_action_queue_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_apply_simulation_df = build_consensus_rerank_apply_simulation(
    consensus_rerank_preview_df,
    consensus_rerank_action_queue_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_simulation_delta_df = build_consensus_rerank_simulation_delta(
    consensus_rerank_apply_simulation_df,
)
consensus_rerank_precision_scorecard_df = build_consensus_rerank_precision_scorecard(
    consensus_rerank_simulation_delta_df,
    consensus_rerank_apply_simulation_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_precision_guardrail_df = build_consensus_rerank_precision_guardrail(
    consensus_rerank_precision_scorecard_df,
    consensus_rerank_policy_gate_df,
    consensus_rerank_action_queue_df,
)
consensus_rerank_precision_guardrail_report_markdown = build_consensus_rerank_precision_guardrail_report_markdown(
    consensus_rerank_precision_guardrail_df,
    consensus_rerank_precision_scorecard_df,
    consensus_rerank_action_queue_df,
    consensus_rerank_simulation_delta_df,
)
consensus_rerank_release_decision_template_df = build_consensus_rerank_release_decision_template(
    consensus_rerank_precision_guardrail_df,
    consensus_rerank_action_queue_df,
    consensus_rerank_simulation_delta_df,
)
if uploaded_consensus_rerank_release_decisions is not None:
    consensus_rerank_release_decision_df, consensus_rerank_release_decision_meta = parse_consensus_rerank_release_decision_table(
        _read_uploaded_text(uploaded_consensus_rerank_release_decisions)
    )
else:
    consensus_rerank_release_decision_df, consensus_rerank_release_decision_meta = pd.DataFrame(), {
        "status": "not-uploaded",
        "input_rows": "0",
        "decision_rows": "0",
        "skipped_rows": "0",
    }
consensus_rerank_release_decision_validation_df = validate_consensus_rerank_release_decisions(
    consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_template_df,
    consensus_rerank_precision_guardrail_df,
)
consensus_rerank_release_decision_summary_df = build_consensus_rerank_release_decision_summary(
    consensus_rerank_release_decision_validation_df,
    consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_template_df,
)
consensus_rerank_release_apply_plan_df = build_consensus_rerank_release_apply_plan(
    consensus_rerank_apply_simulation_df,
    consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_decision_validation_df,
)
consensus_rerank_release_apply_report_markdown = build_consensus_rerank_release_apply_report_markdown(
    consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_decision_summary_df,
)
consensus_rerank_release_execution_template_df = build_consensus_rerank_release_execution_template(
    consensus_rerank_release_apply_plan_df
)
if uploaded_consensus_rerank_release_execution_receipt is not None:
    consensus_rerank_release_execution_receipt_df, consensus_rerank_release_execution_receipt_meta = parse_consensus_rerank_release_execution_table(
        _read_uploaded_text(uploaded_consensus_rerank_release_execution_receipt)
    )
else:
    consensus_rerank_release_execution_receipt_df, consensus_rerank_release_execution_receipt_meta = pd.DataFrame(), {
        "status": "not-uploaded",
        "input_rows": "0",
        "receipt_rows": "0",
        "skipped_rows": "0",
    }
consensus_rerank_release_execution_validation_df = validate_consensus_rerank_release_execution_receipt(
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_template_df,
    consensus_rerank_release_apply_plan_df,
)
consensus_rerank_release_execution_summary_df = build_consensus_rerank_release_execution_summary(
    consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_template_df,
)
consensus_rerank_release_execution_report_markdown = build_consensus_rerank_release_execution_report_markdown(
    consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_receipt_df,
)
consensus_rerank_release_closure_certificate_markdown = build_consensus_rerank_release_closure_certificate_markdown(
    consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_report_markdown,
)
consensus_rerank_release_closure_ledger_df = build_consensus_rerank_release_closure_ledger(
    consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_report_markdown,
    consensus_rerank_release_closure_certificate_markdown,
)
consensus_rerank_guardrail_artifact_manifest_df = build_consensus_rerank_guardrail_artifact_manifest(
    consensus_rerank_suggestion_df=consensus_rerank_suggestion_df,
    consensus_rerank_preview_df=consensus_rerank_preview_df,
    consensus_rerank_policy_gate_df=consensus_rerank_policy_gate_df,
    consensus_rerank_action_queue_df=consensus_rerank_action_queue_df,
    consensus_rerank_action_checklist_markdown=consensus_rerank_action_checklist_markdown,
    consensus_rerank_apply_simulation_df=consensus_rerank_apply_simulation_df,
    consensus_rerank_simulation_delta_df=consensus_rerank_simulation_delta_df,
    consensus_rerank_precision_scorecard_df=consensus_rerank_precision_scorecard_df,
    consensus_rerank_precision_guardrail_df=consensus_rerank_precision_guardrail_df,
    consensus_rerank_precision_guardrail_report_markdown=consensus_rerank_precision_guardrail_report_markdown,
    consensus_rerank_release_decision_template_df=consensus_rerank_release_decision_template_df,
    consensus_rerank_release_decision_df=consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_validation_df=consensus_rerank_release_decision_validation_df,
    consensus_rerank_release_decision_summary_df=consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_apply_plan_df=consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_apply_report_markdown=consensus_rerank_release_apply_report_markdown,
    consensus_rerank_release_execution_template_df=consensus_rerank_release_execution_template_df,
    consensus_rerank_release_execution_receipt_df=consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_validation_df=consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_summary_df=consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_report_markdown=consensus_rerank_release_execution_report_markdown,
    consensus_rerank_release_closure_certificate_markdown=consensus_rerank_release_closure_certificate_markdown,
    consensus_rerank_release_closure_ledger_df=consensus_rerank_release_closure_ledger_df,
)
consensus_rerank_guardrail_handoff_zip = build_consensus_rerank_guardrail_handoff_zip(
    consensus_rerank_suggestion_df=consensus_rerank_suggestion_df,
    consensus_rerank_preview_df=consensus_rerank_preview_df,
    consensus_rerank_policy_gate_df=consensus_rerank_policy_gate_df,
    consensus_rerank_action_queue_df=consensus_rerank_action_queue_df,
    consensus_rerank_action_checklist_markdown=consensus_rerank_action_checklist_markdown,
    consensus_rerank_apply_simulation_df=consensus_rerank_apply_simulation_df,
    consensus_rerank_simulation_delta_df=consensus_rerank_simulation_delta_df,
    consensus_rerank_precision_scorecard_df=consensus_rerank_precision_scorecard_df,
    consensus_rerank_precision_guardrail_df=consensus_rerank_precision_guardrail_df,
    consensus_rerank_precision_guardrail_report_markdown=consensus_rerank_precision_guardrail_report_markdown,
    consensus_rerank_release_decision_template_df=consensus_rerank_release_decision_template_df,
    consensus_rerank_release_decision_df=consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_validation_df=consensus_rerank_release_decision_validation_df,
    consensus_rerank_release_decision_summary_df=consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_apply_plan_df=consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_apply_report_markdown=consensus_rerank_release_apply_report_markdown,
    consensus_rerank_release_execution_template_df=consensus_rerank_release_execution_template_df,
    consensus_rerank_release_execution_receipt_df=consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_validation_df=consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_summary_df=consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_report_markdown=consensus_rerank_release_execution_report_markdown,
    consensus_rerank_release_closure_certificate_markdown=consensus_rerank_release_closure_certificate_markdown,
    consensus_rerank_release_closure_ledger_df=consensus_rerank_release_closure_ledger_df,
    artifact_manifest_df=consensus_rerank_guardrail_artifact_manifest_df,
)
consensus_rerank_guardrail_bundle_verification_df = verify_consensus_rerank_guardrail_handoff_zip(
    consensus_rerank_guardrail_handoff_zip,
    consensus_rerank_guardrail_artifact_manifest_df,
)
consensus_rerank_guardrail_bundle_verification_summary_df = build_consensus_rerank_guardrail_bundle_verification_summary(
    consensus_rerank_guardrail_bundle_verification_df,
    consensus_rerank_guardrail_artifact_manifest_df,
)
consensus_rerank_release_closure_summary_df = build_consensus_rerank_release_closure_summary(
    consensus_rerank_release_closure_ledger_df,
    consensus_rerank_guardrail_bundle_verification_summary_df,
)
consensus_rerank_release_closure_blocker_df = build_consensus_rerank_release_closure_blocker_queue(
    consensus_rerank_release_closure_summary_df,
    consensus_rerank_release_closure_ledger_df,
    consensus_rerank_guardrail_bundle_verification_df,
    consensus_rerank_guardrail_bundle_verification_summary_df,
)
consensus_rerank_release_closure_remediation_checklist_markdown = build_consensus_rerank_release_closure_remediation_checklist_markdown(
    consensus_rerank_release_closure_blocker_df,
    consensus_rerank_release_closure_summary_df,
)
consensus_rerank_release_closure_detached_manifest_df = build_consensus_rerank_release_closure_detached_manifest(
    consensus_rerank_release_closure_summary_df,
    consensus_rerank_release_closure_blocker_df,
    consensus_rerank_release_closure_remediation_checklist_markdown,
)
consensus_rerank_guardrail_handoff_certificate_markdown = build_consensus_rerank_guardrail_handoff_certificate_markdown(
    consensus_rerank_guardrail_handoff_zip,
    consensus_rerank_guardrail_bundle_verification_summary_df,
    consensus_rerank_guardrail_artifact_manifest_df,
    consensus_rerank_precision_guardrail_df,
    consensus_rerank_release_decision_summary_df,
)
top_pocket_decision = pocket_decision_df.iloc[0] if not pocket_decision_df.empty else None
top_consensus_rerank_suggestion = consensus_rerank_suggestion_df.iloc[0] if not consensus_rerank_suggestion_df.empty else None
top_consensus_rerank_preview = consensus_rerank_preview_df.iloc[0] if not consensus_rerank_preview_df.empty else None
top_consensus_rerank_policy_gate = consensus_rerank_policy_gate_df.iloc[0] if not consensus_rerank_policy_gate_df.empty else None
top_consensus_rerank_action = consensus_rerank_action_queue_df.iloc[0] if not consensus_rerank_action_queue_df.empty else None
top_consensus_rerank_apply = consensus_rerank_apply_simulation_df.iloc[0] if not consensus_rerank_apply_simulation_df.empty else None
top_consensus_rerank_delta = consensus_rerank_simulation_delta_df.iloc[0] if not consensus_rerank_simulation_delta_df.empty else None
top_consensus_rerank_scorecard = consensus_rerank_precision_scorecard_df.iloc[0] if not consensus_rerank_precision_scorecard_df.empty else None
top_consensus_rerank_guardrail = consensus_rerank_precision_guardrail_df.iloc[0] if not consensus_rerank_precision_guardrail_df.empty else None
top_consensus_rerank_release_decision_summary = (
    consensus_rerank_release_decision_summary_df.iloc[0]
    if not consensus_rerank_release_decision_summary_df.empty
    else None
)
top_consensus_rerank_release_apply_plan = (
    consensus_rerank_release_apply_plan_df.iloc[0]
    if not consensus_rerank_release_apply_plan_df.empty
    else None
)
top_consensus_rerank_release_execution_summary = (
    consensus_rerank_release_execution_summary_df.iloc[0]
    if not consensus_rerank_release_execution_summary_df.empty
    else None
)
top_consensus_rerank_release_closure_summary = (
    consensus_rerank_release_closure_summary_df.iloc[0]
    if not consensus_rerank_release_closure_summary_df.empty
    else None
)
pocket_reliability_df = build_pocket_reliability_checklist(pocket_decision_df, max_pockets=3)
pocket_triage_df = build_pocket_precision_triage(pocket_decision_df, pocket_reliability_df, max_pockets=3)
manual_key_residue_followup_df = _build_manual_key_residue_followup_df(
    pocket_decision_df,
    pocket_triage_df,
    effective_pocket_df,
    manual_key_residue_df,
)
manual_key_residue_followup_summary_df = _summarize_manual_key_residue_followup_df(manual_key_residue_followup_df)
manual_key_residue_followup_checklist_markdown = _build_manual_key_residue_followup_checklist_markdown(
    manual_key_residue_followup_df,
    manual_key_residue_followup_summary_df,
)
ai_ranking_impact_df = build_ai_ranking_impact_summary(
    ai_evidence_df,
    rankable_ai_evidence_df,
    ai_evidence_audit_df,
    pocket_decision_df,
    pocket_triage_df,
)
ai_followup_plan_df = build_ai_followup_evidence_plan(
    pocket_decision_df,
    pocket_reliability_df,
    pocket_triage_df,
    protein_name=str(literature_protein_name or "").strip(),
    accession=str(uniprot_accession or "").strip(),
    pdb_id=structure_pdb_id,
    ec_number=str(enzyme_ec_number or "").strip(),
    max_pockets=3,
)
ai_followup_prompt_bundle = build_ai_followup_prompt_bundle(ai_followup_plan_df)
ai_review_checklist_markdown = build_ai_review_checklist_markdown(ai_evidence_review_queue_df)
top_precision_triage = pocket_triage_df.iloc[0] if not pocket_triage_df.empty else None
top_residue_consensus = residue_evidence_consensus_df.iloc[0] if not residue_evidence_consensus_df.empty else None
top_pocket_triage = None
if top_pocket_decision is not None and not pocket_reliability_df.empty:
    top_pocket_id = str(top_pocket_decision.get("pocket_id") or "")
    top_reliability_rows = pocket_reliability_df[pocket_reliability_df["pocket_id"].astype(str) == top_pocket_id]
    top_reliability_gaps = "; ".join(
        f"{row.check}: {row.status}"
        for row in top_reliability_rows.itertuples(index=False)
        if str(row.status) != "pass"
    )
    if not pocket_triage_df.empty:
        top_triage_rows = pocket_triage_df[pocket_triage_df["pocket_id"].astype(str) == top_pocket_id]
        if not top_triage_rows.empty:
            top_pocket_triage = top_triage_rows.iloc[0]
else:
    top_reliability_gaps = ""

try:
    protein_volume = estimate_protein_volume(pdb_text)
except Exception:
    protein_volume = None

summary = build_analysis_summary(energy_table)
analysis_text = explain_analysis(energy_table, hotspot_df, effective_pocket_summary)
try:
    stored_mmpbsa_text = mmpbsa_text or "结构估算（未上传 MMPBSA 文件）"
    top_pocket = effective_pocket_summary.iloc[0] if not effective_pocket_summary.empty else None
    set_analysis_state(
        pdb_text,
        stored_mmpbsa_text,
        atom_df,
        energy_df,
        energy_table,
        annotation_table=enriched_annotations,
        pocket_table=effective_pocket_df,
        pocket_summary=effective_pocket_summary,
        joint_candidate_table=joint_candidate_df,
    )
    append_history_record(
        {
            "generated_at": summary["generated_at"],
            "source_name": "口袋/界面专页",
            "energy_source_name": summary.get("energy_source") or energy_source or "未知",
            "residue_count": summary["residue_count"],
            "min_energy": summary["min_energy"],
            "max_energy": summary["max_energy"],
            "mean_energy": summary["mean_energy"],
            "lowest_residue": summary["lowest_residue"],
            "highest_residue": summary["highest_residue"],
            "valid_energy_count": summary["valid_energy_count"],
            "energy_coverage": summary["energy_coverage"],
            "protein_volume": protein_volume,
            "display_mode": "pocket-interface",
            "color_mode": "联合注释",
            "hotspot_count": int(len(hotspot_df)),
            "pocket_count": int(len(effective_pocket_df)),
            "annotation_rows": int(len(enriched_annotations)),
            **auto_detection_summary,
            "top_pocket_id": str(top_pocket.get("pocket_id")) if top_pocket is not None and pd.notna(top_pocket.get("pocket_id")) else None,
            "top_pocket_smart_rank_label": str(top_pocket.get("smart_rank_label")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_label")) else None,
            "top_pocket_smart_rank_score": float(top_pocket.get("smart_rank_score")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_score")) else None,
            "top_pocket_hotspot_count": int(top_pocket.get("hotspot_count")) if top_pocket is not None and pd.notna(top_pocket.get("hotspot_count")) else None,
            "top_pocket_detection_route": str(top_pocket.get("detection_route")) if top_pocket is not None and pd.notna(top_pocket.get("detection_route")) else None,
            "top_pocket_reason": str(top_pocket.get("smart_rank_reason")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_reason")) else None,
            "top_pocket_evidence_quality_label": str(top_pocket.get("evidence_quality_label")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_label")) else None,
            "top_pocket_evidence_quality_score": float(top_pocket.get("evidence_quality_score")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_score")) else None,
            "top_pocket_evidence_quality_warning": str(top_pocket.get("evidence_quality_warning")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_warning")) else None,
            "top_pocket_decision_label": str(top_pocket_decision.get("decision_label")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_label")) else None,
            "top_pocket_decision_score": float(top_pocket_decision.get("decision_score")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_score")) else None,
            "top_pocket_audit_status": str(top_pocket_decision.get("audit_status")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("audit_status")) else None,
            "top_pocket_next_step": str(top_pocket_decision.get("next_step")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("next_step")) else None,
            "top_pocket_reliability_gaps": top_reliability_gaps or None,
            "pocket_reliability_pass_count": int((pocket_reliability_df["status"].astype(str) == "pass").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0,
            "pocket_reliability_missing_count": int((pocket_reliability_df["status"].astype(str) == "missing").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0,
            "top_pocket_precision_tier": str(top_pocket_triage.get("precision_tier")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("precision_tier")) else None,
            "top_pocket_triage_action": str(top_pocket_triage.get("triage_action")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("triage_action")) else None,
            "top_precision_triage_pocket_id": str(top_precision_triage.get("pocket_id")) if top_precision_triage is not None and pd.notna(top_precision_triage.get("pocket_id")) else None,
            "top_joint_pocket_id": str(top_joint_candidate.get("pocket_id")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("pocket_id")) else None,
            "top_joint_recommendation_label": str(top_joint_candidate.get("recommendation_label")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("recommendation_label")) else None,
            "top_joint_recommendation_score": float(top_joint_candidate.get("recommendation_score")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("recommendation_score")) else None,
            "top_joint_reason": str(top_joint_candidate.get("recommendation_reason")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("recommendation_reason")) else None,
            "manual_key_residue_rows": int(len(manual_key_residue_df)),
            "manual_key_residue_status": str(manual_key_residue_meta.get("status") or ""),
            "literature_site_rows": int(len(literature_site_df)),
            "literature_status": str(literature_site_meta.get("status") or ""),
            "literature_query": str(literature_site_meta.get("query") or ""),
            "ai_evidence_enabled": bool(enable_ai_evidence),
            "ai_evidence_rows": int(len(ai_evidence_df)),
            "ai_evidence_status": str(ai_evidence_meta.get("status") or ""),
            "ai_evidence_manual_review_rows": int(ai_evidence_meta.get("manual_review_rows") or 0)
            if str(ai_evidence_meta.get("manual_review_rows") or "").strip().isdigit()
            else 0,
            "ai_evidence_ranked_rows": int(len(rankable_ai_evidence_df)),
            "ai_evidence_ranking_status": str(rankable_ai_evidence_meta.get("status") or ""),
            "ai_review_decision_rows": int(len(ai_review_decision_df)),
            "ai_review_decision_status": str(ai_review_decision_meta.get("status") or ""),
            "ai_review_decision_applied_rows": int(ai_review_decision_meta.get("applied_rows") or 0)
            if str(ai_review_decision_meta.get("applied_rows") or "").strip().isdigit()
            else 0,
            "ai_review_decision_validation_rows": int(len(ai_review_decision_validation_df)),
            "ai_review_decision_validation_blocked_rows": int(
                (ai_review_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
            ) if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns else 0,
            "ai_review_round_status": str(ai_review_round_summary_df.iloc[0].get("review_round_status")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_status")) else "",
            "ai_review_round_reason": str(ai_review_round_summary_df.iloc[0].get("review_round_reason")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_reason")) else "",
            "ai_review_round_rankable_rows": int(ai_review_round_summary_df.iloc[0].get("rankable_after_review_rows") or 0) if not ai_review_round_summary_df.empty else 0,
            "ai_review_ranking_effect_status": str(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) if not ai_review_ranking_delta_df.empty and pd.notna(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) else "",
            "ai_review_ranking_promoted_rows": int(ai_review_ranking_delta_df.iloc[0].get("promoted_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
            "ai_review_ranking_removed_rows": int(ai_review_ranking_delta_df.iloc[0].get("removed_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
            "ai_review_artifact_manifest_rows": int(len(ai_review_artifact_manifest_df)),
            "ai_review_bundle_readme_available": bool(ai_review_bundle_readme_markdown),
            "ai_review_artifact_bundle_available": bool(ai_review_artifact_bundle_zip),
            "ai_review_bundle_verification_rows": int(len(ai_review_bundle_verification_df)),
            "ai_review_bundle_verification_failed_rows": int(
                (ai_review_bundle_verification_df["verification_status"].astype(str) != "verified").sum()
            ) if not ai_review_bundle_verification_df.empty and "verification_status" in ai_review_bundle_verification_df.columns else 0,
            "ai_review_bundle_verification_status": str(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) if not ai_review_bundle_verification_summary_df.empty and pd.notna(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) else "",
            "ai_review_bundle_certificate_available": bool(ai_review_bundle_certificate_markdown),
            "ai_review_decision_outcome_rows": int(len(ai_review_decision_outcome_df)),
            "ai_review_decision_template_rows": int(len(ai_review_decision_template_df)),
            "ai_evidence_supported_count": int((ai_evidence_audit_df["audit_status"].astype(str) == "supported").sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
            "ai_evidence_review_count": int((ai_evidence_audit_df["audit_status"].astype(str).isin(["needs-review", "unsupported", "conflicting"])).sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
            "ai_evidence_review_queue_rows": int(len(ai_evidence_review_queue_df)),
            "top_ai_review_fix_type": str(ai_evidence_review_queue_df.iloc[0].get("fix_type")) if not ai_evidence_review_queue_df.empty and pd.notna(ai_evidence_review_queue_df.iloc[0].get("fix_type")) else None,
            "ai_influence_level": str(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) else None,
            "top_pocket_has_ai_support": bool(ai_ranking_impact_df.iloc[0].get("top_pocket_has_ai_support")) if not ai_ranking_impact_df.empty else False,
            "top_pocket_ai_residues": str(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) else None,
            "ai_followup_plan_rows": int(len(ai_followup_plan_df)),
            "top_ai_followup_query": str(ai_followup_plan_df.iloc[0].get("search_query")) if not ai_followup_plan_df.empty and pd.notna(ai_followup_plan_df.iloc[0].get("search_query")) else None,
            "residue_evidence_consensus_rows": int(len(residue_evidence_consensus_df)),
            "top_residue_consensus_anchor": str(top_residue_consensus.get("residue_anchor")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("residue_anchor")) else None,
            "top_residue_consensus_tier": str(top_residue_consensus.get("consensus_tier")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_tier")) else None,
            "top_residue_consensus_score": float(top_residue_consensus.get("consensus_score")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_score")) else None,
            "top_residue_consensus_sources": str(top_residue_consensus.get("evidence_sources")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("evidence_sources")) else None,
            "pocket_consensus_coverage_rows": int(len(pocket_consensus_coverage_df)),
            "top_pocket_consensus_coverage_id": str(top_pocket_consensus_coverage.get("pocket_id")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_id")) else None,
            "top_pocket_consensus_label": str(top_pocket_consensus_coverage.get("pocket_consensus_label")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_consensus_label")) else None,
            "top_pocket_consensus_anchor_count": int(top_pocket_consensus_coverage.get("rank_safe_anchor_count") or 0) if top_pocket_consensus_coverage is not None else 0,
            "top_pocket_consensus_best_score": float(top_pocket_consensus_coverage.get("best_consensus_score")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("best_consensus_score")) else None,
            "pocket_benchmark_reference_candidate_rows": int(len(benchmark_reference_candidate_df)),
            "pocket_benchmark_reference_import_summary_rows": int(len(benchmark_reference_import_summary_df)),
            "pocket_benchmark_reference_import_status": str(benchmark_reference_import_summary_df.iloc[0].get("import_status") or "") if not benchmark_reference_import_summary_df.empty else "",
            "pocket_benchmark_reference_candidate_review_rows": int(len(benchmark_reference_candidate_review_queue_df)),
            "pocket_benchmark_reference_candidate_review_p1_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P1").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
            "pocket_benchmark_reference_candidate_review_p2_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P2").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
            "pocket_benchmark_reference_candidate_review_checklist_available": bool(benchmark_reference_candidate_review_checklist_markdown),
            "pocket_benchmark_reference_candidate_review_decision_template_rows": int(len(benchmark_reference_candidate_review_decision_template_df)),
            "pocket_benchmark_reference_candidate_review_decision_rows": int(len(benchmark_reference_candidate_review_decision_df)),
            "pocket_benchmark_reference_candidate_review_decision_status": str(benchmark_reference_candidate_review_decision_meta.get("status") or ""),
            "pocket_benchmark_reference_candidate_review_decision_validation_rows": int(len(benchmark_reference_candidate_review_decision_validation_df)),
            "pocket_benchmark_reference_candidate_review_decision_validation_blocked_rows": int(benchmark_reference_candidate_review_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_candidate_review_decision_validation_df.empty and "validation_status" in benchmark_reference_candidate_review_decision_validation_df.columns else 0,
            "pocket_benchmark_reference_candidate_review_outcome_rows": int(len(benchmark_reference_candidate_review_outcome_df)),
            "pocket_benchmark_reference_candidate_review_outcome_accepted_rows": int(benchmark_reference_candidate_review_outcome_df["applied_status"].astype(str).eq("accepted").sum()) if not benchmark_reference_candidate_review_outcome_df.empty and "applied_status" in benchmark_reference_candidate_review_outcome_df.columns else 0,
            "pocket_benchmark_reference_candidate_accepted_rows": int(len(benchmark_reference_candidate_accepted_df)),
            "pocket_benchmark_reference_is_provisional": bool(benchmark_reference_is_provisional),
            "pocket_benchmark_reference_is_reviewed_candidate": bool(benchmark_reference_is_reviewed_candidate),
            "pocket_benchmark_reference_source_mode": str(benchmark_reference_source_mode or ""),
            "pocket_benchmark_reference_source_audit_rows": int(len(benchmark_reference_source_audit_df)),
            "pocket_benchmark_reference_source_audit_summary_rows": int(len(benchmark_reference_source_audit_summary_df)),
            "pocket_benchmark_reference_source_audit_summary_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
            "pocket_benchmark_reference_source_audit_summary_independent_claim_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
            "pocket_benchmark_reference_source_audit_action_queue_rows": int(len(benchmark_reference_source_audit_action_queue_df)),
            "pocket_benchmark_reference_source_audit_action_queue_blocker_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_action_queue_review_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_summary_rows": int(len(benchmark_reference_source_audit_case_summary_df)),
            "pocket_benchmark_reference_source_audit_case_summary_blocked_cases": int(benchmark_reference_source_audit_case_summary_blocked_cases),
            "pocket_benchmark_reference_source_audit_case_summary_review_cases": int(benchmark_reference_source_audit_case_summary_review_cases),
            "pocket_benchmark_reference_source_audit_case_checklist_available": bool(benchmark_reference_source_audit_case_checklist_markdown),
            "pocket_benchmark_reference_source_audit_case_decision_template_rows": int(len(benchmark_reference_source_audit_case_decision_template_df)),
            "pocket_benchmark_reference_source_audit_case_decision_rows": int(len(benchmark_reference_source_audit_case_decision_df)),
            "pocket_benchmark_reference_source_audit_case_decision_status": str(benchmark_reference_source_audit_case_decision_meta.get("status") or ""),
            "pocket_benchmark_reference_source_audit_case_decision_validation_rows": int(len(benchmark_reference_source_audit_case_decision_validation_df)),
            "pocket_benchmark_reference_source_audit_case_decision_validation_blocked_rows": int(benchmark_reference_source_audit_case_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_validation_df.empty and "validation_status" in benchmark_reference_source_audit_case_decision_validation_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_df)),
            "pocket_benchmark_reference_source_audit_case_decision_outcome_blocked_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_pending_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("pending").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_cleared_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).isin(["cleared", "replaced", "source-ready"]).sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_summary_df)),
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_status": benchmark_reference_source_audit_case_decision_outcome_summary_status,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_open_cases": int(benchmark_reference_source_audit_case_decision_outcome_summary_open_cases),
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_rows": int(len(benchmark_reference_source_audit_case_decision_closure_queue_df)),
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_blocker_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_review_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_top_status": str(benchmark_reference_source_audit_case_decision_closure_queue_df.iloc[0].get("applied_status") or "") if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty else "",
            "pocket_benchmark_reference_source_audit_case_decision_closure_checklist_available": bool(benchmark_reference_source_audit_case_decision_closure_checklist_markdown),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_df)),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_cleared_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).eq("cleared-by-decision").sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_open_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).isin(["decision-adjusted-open", "decision-open", "unchanged-open"]).sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df)),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_status": benchmark_reference_source_audit_case_decision_readiness_impact_summary_status,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_net_blocker_delta": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.iloc[0].get("net_blocker_delta") or 0) if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty else 0,
            "pocket_benchmark_reference_source_audit_checklist_available": bool(benchmark_reference_source_audit_checklist_markdown),
            "pocket_benchmark_reference_source_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_df.empty else "",
            "pocket_benchmark_reference_source_independent_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_df.empty else "",
            "pocket_benchmark_reference_source_provisional_rows": int(benchmark_reference_source_audit_df["is_provisional"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_provisional" in benchmark_reference_source_audit_df.columns else 0,
            "pocket_benchmark_reference_source_reviewed_candidate_rows": int(benchmark_reference_source_audit_df["is_reviewed_candidate"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_reviewed_candidate" in benchmark_reference_source_audit_df.columns else 0,
            "pocket_benchmark_reference_rows": int(len(benchmark_reference_df)),
            "pocket_benchmark_reference_template_rows": int(len(benchmark_reference_template_df)),
            "pocket_benchmark_reference_template_notes_available": bool(benchmark_reference_template_markdown),
            "pocket_benchmark_reference_quality_issue_rows": int(len(pocket_benchmark_reference_quality_issue_df)),
            "pocket_benchmark_reference_quality_summary_rows": int(len(pocket_benchmark_reference_quality_summary_df)),
            "pocket_benchmark_reference_quality_checklist_available": bool(pocket_benchmark_reference_quality_checklist_markdown),
            "pocket_benchmark_reference_structure_validation_issue_rows": int(len(pocket_benchmark_reference_structure_validation_df)),
            "pocket_benchmark_reference_structure_validation_summary_rows": int(len(pocket_benchmark_reference_structure_validation_summary_df)),
            "pocket_benchmark_reference_structure_validation_checklist_available": bool(pocket_benchmark_reference_structure_validation_checklist_markdown),
            "pocket_benchmark_reference_readiness_queue_rows": int(len(pocket_benchmark_reference_readiness_queue_df)),
            "pocket_benchmark_reference_readiness_summary_rows": int(len(pocket_benchmark_reference_readiness_summary_df)),
            "pocket_benchmark_reference_readiness_case_summary_rows": int(len(pocket_benchmark_reference_readiness_case_summary_df)),
            "pocket_benchmark_reference_readiness_status": str(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("readiness_status") or "") if not pocket_benchmark_reference_readiness_summary_df.empty else "",
            "pocket_benchmark_reference_readiness_blocker_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p0_p1_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
            "pocket_benchmark_reference_readiness_review_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p2_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
            "pocket_benchmark_reference_readiness_source_audit_issue_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("source_audit_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
            "pocket_benchmark_reference_readiness_blocked_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
            "pocket_benchmark_reference_readiness_review_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
            "pocket_benchmark_reference_readiness_checklist_available": bool(pocket_benchmark_reference_readiness_checklist_markdown),
            "pocket_benchmark_interpretation_rows": int(len(pocket_benchmark_interpretation_df)),
            "pocket_benchmark_top1_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 1].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 1).any() else "",
            "pocket_benchmark_top3_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 3].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 3).any() else "",
            "pocket_benchmark_case_interpretation_rows": int(len(pocket_benchmark_case_interpretation_df)),
            "pocket_benchmark_case_interpretation_blocked_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
            "pocket_benchmark_case_interpretation_review_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_rows": int(len(pocket_benchmark_case_interpretation_matrix_df)),
            "pocket_benchmark_case_interpretation_matrix_blocked_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_review_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_summary_rows": int(len(pocket_benchmark_case_interpretation_matrix_summary_df)),
            "pocket_benchmark_case_interpretation_matrix_summary_status": str(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("summary_status") or "") if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else "",
            "pocket_benchmark_case_interpretation_matrix_summary_usable_cases": int(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("usable_claim_ready_case_count") or 0) if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else 0,
            "pocket_benchmark_case_interpretation_matrix_queue_rows": int(len(pocket_benchmark_case_interpretation_matrix_queue_df)),
            "pocket_benchmark_case_interpretation_matrix_queue_blocker_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_queue_review_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_rows": int(len(pocket_benchmark_dataset_interpretation_df)),
            "pocket_benchmark_dataset_interpretation_blocked_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_review_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-blocked").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-review-needed").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-gate-mismatch").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_gate_mismatch"].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_gate_mismatch" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and "action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and "action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df["source_gate_mismatch"].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and "source_gate_mismatch" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_action_count": int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df["action_count"], errors="coerce").fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and "action_count" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_p0_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df["priority"].astype(str).eq("P0").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and "priority" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_mismatch_count": int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df["mismatch_count"], errors="coerce").fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and "mismatch_count" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_priority": str(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.iloc[0].get("priority") or "") if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty else "",
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_source_impact": str(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.iloc[0].get("source_impact_status") or "") if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty else "",
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_available": bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_available": bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_bytes": int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df["byte_size"], errors="coerce").fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty and "byte_size" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_hash_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty and "sha256" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_queue_rows": int(len(pocket_benchmark_dataset_interpretation_queue_df)),
            "pocket_benchmark_dataset_interpretation_queue_blocker_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_queue_review_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_checklist_available": bool(pocket_benchmark_dataset_interpretation_checklist_markdown),
            "pocket_benchmark_dataset_interpretation_report_available": bool(pocket_benchmark_dataset_interpretation_report_markdown),
            "pocket_benchmark_top1_coverage": float(top1_benchmark.get("coverage_ratio") or 0.0) if top1_benchmark is not None else None,
            "pocket_benchmark_top1_status": str(top1_benchmark.get("benchmark_status") or "") if top1_benchmark is not None else None,
            "pocket_benchmark_top3_coverage": float(top3_benchmark.get("coverage_ratio") or 0.0) if top3_benchmark is not None else None,
            "pocket_benchmark_top3_status": str(top3_benchmark.get("benchmark_status") or "") if top3_benchmark is not None else None,
            "pocket_benchmark_best_rank": int(top3_benchmark.get("best_rank") or 0) if top3_benchmark is not None else 0,
            "pocket_benchmark_best_pocket_id": str(top3_benchmark.get("best_pocket_id") or "") if top3_benchmark is not None else None,
            "pocket_benchmark_case_summary_rows": int(len(pocket_benchmark_case_summary_df)),
            "pocket_benchmark_dataset_summary_rows": int(len(pocket_benchmark_dataset_summary_df)),
            "pocket_benchmark_variant_comparison_rows": int(len(pocket_benchmark_variant_comparison_df)),
            "pocket_benchmark_variant_case_comparison_rows": int(len(pocket_benchmark_variant_case_comparison_df)),
            "pocket_benchmark_variant_dataset_comparison_rows": int(len(pocket_benchmark_variant_dataset_comparison_df)),
            "pocket_benchmark_variant_detail_comparison_rows": int(len(pocket_benchmark_variant_detail_comparison_df)),
            "pocket_benchmark_variant_remediation_rows": int(len(pocket_benchmark_variant_remediation_df)),
            "pocket_benchmark_variant_remediation_summary_rows": int(len(pocket_benchmark_variant_remediation_summary_df)),
            "pocket_benchmark_variant_remediation_checklist_available": bool(pocket_benchmark_variant_remediation_checklist_markdown),
            "p2rank_ab_enabled": bool(p2rank_ab_enabled),
            "p2rank_ab_changed_count": int((p2rank_ab_df["status"].astype(str) != "unchanged").sum())
            if not p2rank_ab_df.empty and "status" in p2rank_ab_df.columns
            else 0,
            "consensus_rerank_suggestion_rows": int(len(consensus_rerank_suggestion_df)),
            "top_consensus_rerank_pocket_id": str(top_consensus_rerank_suggestion.get("pocket_id")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("pocket_id")) else None,
            "top_consensus_rerank_status": str(top_consensus_rerank_suggestion.get("suggestion_status")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("suggestion_status")) else None,
            "top_consensus_rerank_rank_delta": int(top_consensus_rerank_suggestion.get("rank_delta") or 0) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("rank_delta")) else 0,
            "consensus_rerank_preview_rows": int(len(consensus_rerank_preview_df)),
            "top_consensus_preview_pocket_id": str(top_consensus_rerank_preview.get("pocket_id")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("pocket_id")) else None,
            "top_consensus_preview_decision": str(top_consensus_rerank_preview.get("preview_decision")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_decision")) else None,
            "top_consensus_preview_rank_delta": int(top_consensus_rerank_preview.get("preview_rank_delta") or 0) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_rank_delta")) else 0,
            "top_consensus_preview_score": float(top_consensus_rerank_preview.get("preview_score")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_score")) else None,
            "consensus_rerank_policy_status": str(top_consensus_rerank_policy_gate.get("policy_status")) if top_consensus_rerank_policy_gate is not None and pd.notna(top_consensus_rerank_policy_gate.get("policy_status")) else "",
            "consensus_rerank_policy_changed_rows": int(top_consensus_rerank_policy_gate.get("changed_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
            "consensus_rerank_policy_blocked_rows": int(top_consensus_rerank_policy_gate.get("blocked_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
            "consensus_rerank_action_queue_rows": int(len(consensus_rerank_action_queue_df)),
            "top_consensus_rerank_action_pocket_id": str(top_consensus_rerank_action.get("pocket_id")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("pocket_id")) else None,
            "top_consensus_rerank_issue_type": str(top_consensus_rerank_action.get("issue_type")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_type")) else None,
            "top_consensus_rerank_issue_severity": str(top_consensus_rerank_action.get("issue_severity")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_severity")) else None,
            "consensus_rerank_action_checklist_available": bool(consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty),
            "consensus_rerank_apply_simulation_rows": int(len(consensus_rerank_apply_simulation_df)),
            "top_consensus_rerank_apply_pocket_id": str(top_consensus_rerank_apply.get("pocket_id")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("pocket_id")) else None,
            "top_consensus_rerank_apply_status": str(top_consensus_rerank_apply.get("apply_status")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("apply_status")) else None,
            "top_consensus_rerank_apply_rank_delta": int(top_consensus_rerank_apply.get("simulated_rank_delta") or 0) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("simulated_rank_delta")) else 0,
            "consensus_rerank_simulation_delta_rows": int(len(consensus_rerank_simulation_delta_df)),
            "top_consensus_rerank_delta_pocket_id": str(top_consensus_rerank_delta.get("pocket_id")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("pocket_id")) else None,
            "top_consensus_rerank_delta_change_type": str(top_consensus_rerank_delta.get("change_type")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("change_type")) else None,
            "top_consensus_rerank_delta_rank_delta": int(top_consensus_rerank_delta.get("rank_delta") or 0) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("rank_delta")) else 0,
            "consensus_rerank_precision_scorecard_rows": int(len(consensus_rerank_precision_scorecard_df)),
            "consensus_rerank_precision_score": int(top_consensus_rerank_scorecard.get("precision_improvement_score") or 0) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("precision_improvement_score")) else 0,
            "consensus_rerank_precision_status": str(top_consensus_rerank_scorecard.get("scorecard_status")) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("scorecard_status")) else None,
            "consensus_rerank_positive_signal_rows": int(top_consensus_rerank_scorecard.get("positive_signal_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
            "consensus_rerank_open_blocker_rows": int(top_consensus_rerank_scorecard.get("open_blocker_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
            "consensus_rerank_precision_guardrail_rows": int(len(consensus_rerank_precision_guardrail_df)),
            "consensus_rerank_guardrail_status": str(top_consensus_rerank_guardrail.get("guardrail_status")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_status")) else None,
            "consensus_rerank_guardrail_decision": str(top_consensus_rerank_guardrail.get("guardrail_decision")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_decision")) else None,
            "consensus_rerank_guardrail_apply_mode": str(top_consensus_rerank_guardrail.get("apply_mode")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("apply_mode")) else None,
            "consensus_rerank_guardrail_can_apply_after_review": bool(top_consensus_rerank_guardrail.get("can_apply_after_manual_review")) if top_consensus_rerank_guardrail is not None else False,
            "consensus_rerank_guardrail_report_available": bool(consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty),
            "consensus_rerank_guardrail_artifact_manifest_rows": int(len(consensus_rerank_guardrail_artifact_manifest_df)),
            "consensus_rerank_guardrail_handoff_zip_available": bool(consensus_rerank_guardrail_handoff_zip),
            "consensus_rerank_guardrail_bundle_verification_rows": int(len(consensus_rerank_guardrail_bundle_verification_df)),
            "consensus_rerank_guardrail_bundle_verification_status": str(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty and pd.notna(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) else None,
            "consensus_rerank_guardrail_bundle_verification_failed_rows": int(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("failed_files") or 0) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else 0,
            "consensus_rerank_guardrail_handoff_certificate_available": bool(consensus_rerank_guardrail_handoff_certificate_markdown),
            "consensus_rerank_release_decision_template_rows": int(len(consensus_rerank_release_decision_template_df)),
            "consensus_rerank_release_decision_rows": int(len(consensus_rerank_release_decision_df)),
            "consensus_rerank_release_decision_status": str(consensus_rerank_release_decision_meta.get("status") or ""),
            "consensus_rerank_release_decision_validation_rows": int(len(consensus_rerank_release_decision_validation_df)),
            "consensus_rerank_release_decision_blocked_rows": int(
                (consensus_rerank_release_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
            ) if not consensus_rerank_release_decision_validation_df.empty and "validation_status" in consensus_rerank_release_decision_validation_df.columns else 0,
            "consensus_rerank_release_review_status": str(top_consensus_rerank_release_decision_summary.get("release_review_status")) if top_consensus_rerank_release_decision_summary is not None and pd.notna(top_consensus_rerank_release_decision_summary.get("release_review_status")) else "",
            "consensus_rerank_release_allowed": bool(top_consensus_rerank_release_decision_summary.get("release_allowed")) if top_consensus_rerank_release_decision_summary is not None else False,
            "consensus_rerank_release_apply_plan_rows": int(len(consensus_rerank_release_apply_plan_df)),
            "top_consensus_rerank_release_apply_pocket_id": str(top_consensus_rerank_release_apply_plan.get("pocket_id")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("pocket_id")) else None,
            "top_consensus_rerank_release_apply_status": str(top_consensus_rerank_release_apply_plan.get("release_apply_status")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("release_apply_status")) else None,
            "consensus_rerank_release_apply_report_available": bool(consensus_rerank_release_apply_report_markdown),
            "consensus_rerank_release_execution_template_rows": int(len(consensus_rerank_release_execution_template_df)),
            "consensus_rerank_release_execution_receipt_rows": int(len(consensus_rerank_release_execution_receipt_df)),
            "consensus_rerank_release_execution_receipt_status": str(consensus_rerank_release_execution_receipt_meta.get("status") or ""),
            "consensus_rerank_release_execution_validation_rows": int(len(consensus_rerank_release_execution_validation_df)),
            "consensus_rerank_release_execution_blocked_rows": int(
                (consensus_rerank_release_execution_validation_df["validation_status"].astype(str) == "blocked").sum()
            ) if not consensus_rerank_release_execution_validation_df.empty and "validation_status" in consensus_rerank_release_execution_validation_df.columns else 0,
            "consensus_rerank_release_execution_review_status": str(top_consensus_rerank_release_execution_summary.get("execution_review_status")) if top_consensus_rerank_release_execution_summary is not None and pd.notna(top_consensus_rerank_release_execution_summary.get("execution_review_status")) else "",
            "consensus_rerank_release_execution_complete": bool(top_consensus_rerank_release_execution_summary.get("execution_complete")) if top_consensus_rerank_release_execution_summary is not None else False,
            "consensus_rerank_release_execution_report_available": bool(consensus_rerank_release_execution_report_markdown),
            "consensus_rerank_release_closure_certificate_available": bool(consensus_rerank_release_closure_certificate_markdown),
            "consensus_rerank_release_closure_ledger_rows": int(len(consensus_rerank_release_closure_ledger_df)),
            "consensus_rerank_release_closure_ledger_blocked_rows": int(
                consensus_rerank_release_closure_ledger_df["closure_check"].astype(str).str.lower().isin({"blocked", "missing"}).sum()
            ) if not consensus_rerank_release_closure_ledger_df.empty and "closure_check" in consensus_rerank_release_closure_ledger_df.columns else 0,
            "consensus_rerank_release_closure_summary_rows": int(len(consensus_rerank_release_closure_summary_df)),
            "consensus_rerank_release_closure_readiness_status": str(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) if top_consensus_rerank_release_closure_summary is not None and pd.notna(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) else "",
            "consensus_rerank_release_closed": bool(top_consensus_rerank_release_closure_summary.get("release_closed")) if top_consensus_rerank_release_closure_summary is not None else False,
            "consensus_rerank_release_closure_blocker_rows": int(len(consensus_rerank_release_closure_blocker_df)),
            "top_consensus_rerank_release_closure_blocker_type": str(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) if not consensus_rerank_release_closure_blocker_df.empty and pd.notna(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) else "",
            "consensus_rerank_release_closure_remediation_checklist_available": bool(consensus_rerank_release_closure_remediation_checklist_markdown),
            "consensus_rerank_release_closure_detached_manifest_rows": int(len(consensus_rerank_release_closure_detached_manifest_df)),
            "literature_ab_enabled": bool(literature_ab_enabled),
            "literature_ab_changed_count": int((literature_ab_df["status"].astype(str) != "unchanged").sum())
            if not literature_ab_df.empty and "status" in literature_ab_df.columns
            else 0,
            "evidence_route_enabled": bool(auto_external_evidence_route),
            "evidence_route_status": str(auto_detection_summary.get("auto_detection_external_route_status") or ""),
            "evidence_route_ab_enabled": bool(evidence_route_ab_enabled),
            "evidence_route_ab_changed_count": int((evidence_route_ab_df["status"].astype(str) != "unchanged").sum())
            if not evidence_route_ab_df.empty and "status" in evidence_route_ab_df.columns
            else 0,
            "conservation_site_rows": int(len(conservation_site_df)),
            "conservation_source_name": str(conservation_site_meta.get("source") or ""),
            "conservation_score_mean": float(conservation_site_meta.get("score_mean")) if conservation_site_meta.get("score_mean") not in {None, ""} else None,
            "conservation_ab_enabled": bool(conservation_ab_enabled),
            "conservation_ab_changed_count": int((conservation_ab_df["status"].astype(str) != "unchanged").sum())
            if not conservation_ab_df.empty and "status" in conservation_ab_df.columns
            else 0,
        }
    )
except Exception:
    pass

snapshot = build_analysis_snapshot(
    energy_table,
    title="ProteinInsight 口袋 / 界面快照",
    annotation_table=enriched_annotations,
    hotspot_df=hotspot_df,
    pocket_summary=effective_pocket_summary,
    joint_candidate_df=joint_candidate_df,
    protein_volume=protein_volume,
    extra={
        **auto_detection_summary,
        "auto_detection_metadata": auto_detection_meta,
        "effective_pocket_source": effective_pocket_mode,
        "effective_annotation_source": effective_annotation_mode,
        "external_site_rows": int(len(external_site_df)),
        "manual_key_residue_rows": int(len(manual_key_residue_df)),
        "manual_key_residue_metadata": manual_key_residue_meta,
        "external_site_accession": str(external_site_meta.get("accession") or ""),
        "external_site_pdb_id": str(external_site_meta.get("pdb_id") or structure_pdb_id),
        "external_mapping_status": str(external_site_meta.get("mapping_status") or ""),
        "literature_site_rows": int(len(literature_site_df)),
        "literature_status": str(literature_site_meta.get("status") or ""),
        "literature_query": str(literature_site_meta.get("query") or ""),
        "literature_metadata": literature_site_meta,
        "ai_evidence_enabled": bool(enable_ai_evidence),
        "ai_evidence_rows": int(len(ai_evidence_df)),
        "ai_evidence_status": str(ai_evidence_meta.get("status") or ""),
        "ai_evidence_metadata": ai_evidence_meta,
        "ai_evidence": ai_evidence_df.to_dict(orient="records"),
        "ai_evidence_ranked_rows": int(len(rankable_ai_evidence_df)),
        "ai_evidence_ranking_metadata": rankable_ai_evidence_meta,
        "ai_evidence_ranked": rankable_ai_evidence_df.to_dict(orient="records"),
        "ai_review_decision_rows": int(len(ai_review_decision_df)),
        "ai_review_decision_status": str(ai_review_decision_meta.get("status") or ""),
        "ai_review_decision_applied_rows": int(ai_review_decision_meta.get("applied_rows") or 0)
        if str(ai_review_decision_meta.get("applied_rows") or "").strip().isdigit()
        else 0,
        "ai_review_decision_validation_rows": int(len(ai_review_decision_validation_df)),
        "ai_review_decision_validation_blocked_rows": int(
            (ai_review_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
        ) if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns else 0,
        "ai_review_decision_validation": ai_review_decision_validation_df.to_dict(orient="records"),
        "ai_review_round_status": str(ai_review_round_summary_df.iloc[0].get("review_round_status")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_status")) else "",
        "ai_review_round_reason": str(ai_review_round_summary_df.iloc[0].get("review_round_reason")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_reason")) else "",
        "ai_review_round_rankable_rows": int(ai_review_round_summary_df.iloc[0].get("rankable_after_review_rows") or 0) if not ai_review_round_summary_df.empty else 0,
        "ai_review_round_summary": ai_review_round_summary_df.to_dict(orient="records"),
        "ai_review_ranking_effect_status": str(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) if not ai_review_ranking_delta_df.empty and pd.notna(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) else "",
        "ai_review_ranking_promoted_rows": int(ai_review_ranking_delta_df.iloc[0].get("promoted_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
        "ai_review_ranking_removed_rows": int(ai_review_ranking_delta_df.iloc[0].get("removed_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
        "ai_review_ranking_delta": ai_review_ranking_delta_df.to_dict(orient="records"),
        "ai_review_round_report_available": bool(ai_review_round_report_markdown),
        "ai_review_artifact_manifest_rows": int(len(ai_review_artifact_manifest_df)),
        "ai_review_artifact_manifest": ai_review_artifact_manifest_df.to_dict(orient="records"),
        "ai_review_bundle_readme_available": bool(ai_review_bundle_readme_markdown),
        "ai_review_artifact_bundle_available": bool(ai_review_artifact_bundle_zip),
        "ai_review_bundle_verification_rows": int(len(ai_review_bundle_verification_df)),
        "ai_review_bundle_verification_failed_rows": int(
            (ai_review_bundle_verification_df["verification_status"].astype(str) != "verified").sum()
        ) if not ai_review_bundle_verification_df.empty and "verification_status" in ai_review_bundle_verification_df.columns else 0,
        "ai_review_bundle_verification": ai_review_bundle_verification_df.to_dict(orient="records"),
        "ai_review_bundle_verification_status": str(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) if not ai_review_bundle_verification_summary_df.empty and pd.notna(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) else "",
        "ai_review_bundle_verification_summary": ai_review_bundle_verification_summary_df.to_dict(orient="records"),
        "ai_review_bundle_certificate_available": bool(ai_review_bundle_certificate_markdown),
        "ai_review_decision_outcome_rows": int(len(ai_review_decision_outcome_df)),
        "ai_review_decision_outcomes": ai_review_decision_outcome_df.to_dict(orient="records"),
        "ai_review_decision_template_rows": int(len(ai_review_decision_template_df)),
        "ai_review_decision_template": ai_review_decision_template_df.to_dict(orient="records"),
        "ai_review_decision_metadata": ai_review_decision_meta,
        "ai_review_decisions": ai_review_decision_df.to_dict(orient="records"),
        "ai_evidence_audit_supported_count": int((ai_evidence_audit_df["audit_status"].astype(str) == "supported").sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
        "ai_evidence_audit_review_count": int((ai_evidence_audit_df["audit_status"].astype(str).isin(["needs-review", "unsupported", "conflicting"])).sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
        "ai_evidence_audit": ai_evidence_audit_df.to_dict(orient="records"),
        "ai_evidence_review_queue_rows": int(len(ai_evidence_review_queue_df)),
        "top_ai_review_fix_type": str(ai_evidence_review_queue_df.iloc[0].get("fix_type")) if not ai_evidence_review_queue_df.empty and pd.notna(ai_evidence_review_queue_df.iloc[0].get("fix_type")) else None,
        "ai_evidence_review_queue": ai_evidence_review_queue_df.to_dict(orient="records"),
        "ai_ranking_impact": ai_ranking_impact_df.to_dict(orient="records"),
        "ai_influence_level": str(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) else None,
        "top_pocket_has_ai_support": bool(ai_ranking_impact_df.iloc[0].get("top_pocket_has_ai_support")) if not ai_ranking_impact_df.empty else False,
        "top_pocket_ai_residues": str(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) else None,
        "ai_followup_plan_rows": int(len(ai_followup_plan_df)),
        "top_ai_followup_query": str(ai_followup_plan_df.iloc[0].get("search_query")) if not ai_followup_plan_df.empty and pd.notna(ai_followup_plan_df.iloc[0].get("search_query")) else None,
        "ai_followup_plan": ai_followup_plan_df.to_dict(orient="records"),
        "residue_evidence_consensus_rows": int(len(residue_evidence_consensus_df)),
        "residue_evidence_consensus": residue_evidence_consensus_df.to_dict(orient="records"),
        "top_residue_consensus_anchor": str(top_residue_consensus.get("residue_anchor")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("residue_anchor")) else None,
        "top_residue_consensus_tier": str(top_residue_consensus.get("consensus_tier")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_tier")) else None,
        "top_residue_consensus_score": float(top_residue_consensus.get("consensus_score")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_score")) else None,
        "top_residue_consensus_sources": str(top_residue_consensus.get("evidence_sources")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("evidence_sources")) else None,
        "pocket_consensus_coverage_rows": int(len(pocket_consensus_coverage_df)),
        "pocket_consensus_coverage": pocket_consensus_coverage_df.to_dict(orient="records"),
        "top_pocket_consensus_coverage_id": str(top_pocket_consensus_coverage.get("pocket_id")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_id")) else None,
        "top_pocket_consensus_label": str(top_pocket_consensus_coverage.get("pocket_consensus_label")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_consensus_label")) else None,
        "top_pocket_consensus_anchor_count": int(top_pocket_consensus_coverage.get("rank_safe_anchor_count") or 0) if top_pocket_consensus_coverage is not None else 0,
        "top_pocket_consensus_best_score": float(top_pocket_consensus_coverage.get("best_consensus_score")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("best_consensus_score")) else None,
        "pocket_benchmark_reference_candidate_rows": int(len(benchmark_reference_candidate_df)),
        "pocket_benchmark_reference_candidate": benchmark_reference_candidate_df.to_dict(orient="records"),
        "pocket_benchmark_reference_import_summary_rows": int(len(benchmark_reference_import_summary_df)),
        "pocket_benchmark_reference_import_summary": benchmark_reference_import_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_import_status": str(benchmark_reference_import_summary_df.iloc[0].get("import_status") or "") if not benchmark_reference_import_summary_df.empty else "",
        "pocket_benchmark_reference_candidate_review_rows": int(len(benchmark_reference_candidate_review_queue_df)),
        "pocket_benchmark_reference_candidate_review_queue": benchmark_reference_candidate_review_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_p1_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P1").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_p2_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P2").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_checklist_available": bool(benchmark_reference_candidate_review_checklist_markdown),
        "pocket_benchmark_reference_candidate_review_checklist": benchmark_reference_candidate_review_checklist_markdown,
        "pocket_benchmark_reference_candidate_review_decision_template_rows": int(len(benchmark_reference_candidate_review_decision_template_df)),
        "pocket_benchmark_reference_candidate_review_decision_template": benchmark_reference_candidate_review_decision_template_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_decision_rows": int(len(benchmark_reference_candidate_review_decision_df)),
        "pocket_benchmark_reference_candidate_review_decisions": benchmark_reference_candidate_review_decision_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_decision_status": str(benchmark_reference_candidate_review_decision_meta.get("status") or ""),
        "pocket_benchmark_reference_candidate_review_decision_metadata": benchmark_reference_candidate_review_decision_meta,
        "pocket_benchmark_reference_candidate_review_decision_validation_rows": int(len(benchmark_reference_candidate_review_decision_validation_df)),
        "pocket_benchmark_reference_candidate_review_decision_validation_blocked_rows": int(benchmark_reference_candidate_review_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_candidate_review_decision_validation_df.empty and "validation_status" in benchmark_reference_candidate_review_decision_validation_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_decision_validation": benchmark_reference_candidate_review_decision_validation_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_outcome_rows": int(len(benchmark_reference_candidate_review_outcome_df)),
        "pocket_benchmark_reference_candidate_review_outcome_accepted_rows": int(benchmark_reference_candidate_review_outcome_df["applied_status"].astype(str).eq("accepted").sum()) if not benchmark_reference_candidate_review_outcome_df.empty and "applied_status" in benchmark_reference_candidate_review_outcome_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_outcomes": benchmark_reference_candidate_review_outcome_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_accepted_rows": int(len(benchmark_reference_candidate_accepted_df)),
        "pocket_benchmark_reference_candidate_accepted": benchmark_reference_candidate_accepted_df.to_dict(orient="records"),
        "pocket_benchmark_reference_is_provisional": bool(benchmark_reference_is_provisional),
        "pocket_benchmark_reference_is_reviewed_candidate": bool(benchmark_reference_is_reviewed_candidate),
        "pocket_benchmark_reference_source_mode": str(benchmark_reference_source_mode or ""),
        "pocket_benchmark_reference_source_audit_rows": int(len(benchmark_reference_source_audit_df)),
        "pocket_benchmark_reference_source_audit": benchmark_reference_source_audit_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_summary_rows": int(len(benchmark_reference_source_audit_summary_df)),
        "pocket_benchmark_reference_source_audit_summary": benchmark_reference_source_audit_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_summary_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
        "pocket_benchmark_reference_source_audit_summary_independent_claim_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
        "pocket_benchmark_reference_source_audit_action_queue_rows": int(len(benchmark_reference_source_audit_action_queue_df)),
        "pocket_benchmark_reference_source_audit_action_queue": benchmark_reference_source_audit_action_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_action_queue_blocker_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_action_queue_review_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_summary_rows": int(len(benchmark_reference_source_audit_case_summary_df)),
        "pocket_benchmark_reference_source_audit_case_summary": benchmark_reference_source_audit_case_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_summary_blocked_cases": int(benchmark_reference_source_audit_case_summary_blocked_cases),
        "pocket_benchmark_reference_source_audit_case_summary_review_cases": int(benchmark_reference_source_audit_case_summary_review_cases),
        "pocket_benchmark_reference_source_audit_case_checklist_available": bool(benchmark_reference_source_audit_case_checklist_markdown),
        "pocket_benchmark_reference_source_audit_case_checklist": benchmark_reference_source_audit_case_checklist_markdown,
        "pocket_benchmark_reference_source_audit_case_decision_template_rows": int(len(benchmark_reference_source_audit_case_decision_template_df)),
        "pocket_benchmark_reference_source_audit_case_decision_template": benchmark_reference_source_audit_case_decision_template_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_rows": int(len(benchmark_reference_source_audit_case_decision_df)),
        "pocket_benchmark_reference_source_audit_case_decision_status": str(benchmark_reference_source_audit_case_decision_meta.get("status") or ""),
        "pocket_benchmark_reference_source_audit_case_decisions": benchmark_reference_source_audit_case_decision_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_validation_rows": int(len(benchmark_reference_source_audit_case_decision_validation_df)),
        "pocket_benchmark_reference_source_audit_case_decision_validation_blocked_rows": int(benchmark_reference_source_audit_case_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_validation_df.empty and "validation_status" in benchmark_reference_source_audit_case_decision_validation_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_validation": benchmark_reference_source_audit_case_decision_validation_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_df)),
        "pocket_benchmark_reference_source_audit_case_decision_outcomes": benchmark_reference_source_audit_case_decision_outcome_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_blocked_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_pending_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("pending").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_cleared_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).isin(["cleared", "replaced", "source-ready"]).sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_summary_df)),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_status": benchmark_reference_source_audit_case_decision_outcome_summary_status,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_open_cases": int(benchmark_reference_source_audit_case_decision_outcome_summary_open_cases),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary": benchmark_reference_source_audit_case_decision_outcome_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_rows": int(len(benchmark_reference_source_audit_case_decision_closure_queue_df)),
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_blocker_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_review_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_top_status": str(benchmark_reference_source_audit_case_decision_closure_queue_df.iloc[0].get("applied_status") or "") if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty else "",
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue": benchmark_reference_source_audit_case_decision_closure_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_closure_checklist_available": bool(benchmark_reference_source_audit_case_decision_closure_checklist_markdown),
        "pocket_benchmark_reference_source_audit_case_decision_closure_checklist": benchmark_reference_source_audit_case_decision_closure_checklist_markdown,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_df)),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_cleared_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).eq("cleared-by-decision").sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_open_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).isin(["decision-adjusted-open", "decision-open", "unchanged-open"]).sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact": benchmark_reference_source_audit_case_decision_readiness_impact_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df)),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_status": benchmark_reference_source_audit_case_decision_readiness_impact_summary_status,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_net_blocker_delta": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.iloc[0].get("net_blocker_delta") or 0) if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty else 0,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary": benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_checklist_available": bool(benchmark_reference_source_audit_checklist_markdown),
        "pocket_benchmark_reference_source_audit_checklist": benchmark_reference_source_audit_checklist_markdown,
        "pocket_benchmark_reference_source_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_df.empty else "",
        "pocket_benchmark_reference_source_independent_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_df.empty else "",
        "pocket_benchmark_reference_source_provisional_rows": int(benchmark_reference_source_audit_df["is_provisional"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_provisional" in benchmark_reference_source_audit_df.columns else 0,
        "pocket_benchmark_reference_source_reviewed_candidate_rows": int(benchmark_reference_source_audit_df["is_reviewed_candidate"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_reviewed_candidate" in benchmark_reference_source_audit_df.columns else 0,
        "pocket_benchmark_reference_rows": int(len(benchmark_reference_df)),
        "pocket_benchmark_reference": benchmark_reference_df.to_dict(orient="records"),
        "pocket_benchmark_reference_template_rows": int(len(benchmark_reference_template_df)),
        "pocket_benchmark_reference_template": benchmark_reference_template_df.to_dict(orient="records"),
        "pocket_benchmark_reference_template_notes_available": bool(benchmark_reference_template_markdown),
        "pocket_benchmark_reference_quality_issue_rows": int(len(pocket_benchmark_reference_quality_issue_df)),
        "pocket_benchmark_reference_quality_issues": pocket_benchmark_reference_quality_issue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_quality_summary_rows": int(len(pocket_benchmark_reference_quality_summary_df)),
        "pocket_benchmark_reference_quality_summary": pocket_benchmark_reference_quality_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_quality_checklist_available": bool(pocket_benchmark_reference_quality_checklist_markdown),
        "pocket_benchmark_reference_quality_checklist": pocket_benchmark_reference_quality_checklist_markdown,
        "pocket_benchmark_reference_structure_validation_issue_rows": int(len(pocket_benchmark_reference_structure_validation_df)),
        "pocket_benchmark_reference_structure_validation_issues": pocket_benchmark_reference_structure_validation_df.to_dict(orient="records"),
        "pocket_benchmark_reference_structure_validation_summary_rows": int(len(pocket_benchmark_reference_structure_validation_summary_df)),
        "pocket_benchmark_reference_structure_validation_summary": pocket_benchmark_reference_structure_validation_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_structure_validation_checklist_available": bool(pocket_benchmark_reference_structure_validation_checklist_markdown),
        "pocket_benchmark_reference_structure_validation_checklist": pocket_benchmark_reference_structure_validation_checklist_markdown,
        "pocket_benchmark_reference_readiness_queue_rows": int(len(pocket_benchmark_reference_readiness_queue_df)),
        "pocket_benchmark_reference_readiness_queue": pocket_benchmark_reference_readiness_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_readiness_summary_rows": int(len(pocket_benchmark_reference_readiness_summary_df)),
        "pocket_benchmark_reference_readiness_summary": pocket_benchmark_reference_readiness_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_readiness_case_summary_rows": int(len(pocket_benchmark_reference_readiness_case_summary_df)),
        "pocket_benchmark_reference_readiness_case_summary": pocket_benchmark_reference_readiness_case_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_readiness_status": str(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("readiness_status") or "") if not pocket_benchmark_reference_readiness_summary_df.empty else "",
        "pocket_benchmark_reference_readiness_blocker_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p0_p1_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
        "pocket_benchmark_reference_readiness_review_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p2_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
        "pocket_benchmark_reference_readiness_source_audit_issue_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("source_audit_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
        "pocket_benchmark_reference_readiness_blocked_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
        "pocket_benchmark_reference_readiness_review_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
        "pocket_benchmark_reference_readiness_checklist_available": bool(pocket_benchmark_reference_readiness_checklist_markdown),
        "pocket_benchmark_reference_readiness_checklist": pocket_benchmark_reference_readiness_checklist_markdown,
        "pocket_benchmark_interpretation_rows": int(len(pocket_benchmark_interpretation_df)),
        "pocket_benchmark_interpretation": pocket_benchmark_interpretation_df.to_dict(orient="records"),
        "pocket_benchmark_top1_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 1].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 1).any() else "",
        "pocket_benchmark_top3_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 3].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 3).any() else "",
        "pocket_benchmark_case_interpretation_rows": int(len(pocket_benchmark_case_interpretation_df)),
        "pocket_benchmark_case_interpretation": pocket_benchmark_case_interpretation_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_blocked_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
        "pocket_benchmark_case_interpretation_review_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_rows": int(len(pocket_benchmark_case_interpretation_matrix_df)),
        "pocket_benchmark_case_interpretation_matrix": pocket_benchmark_case_interpretation_matrix_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_matrix_blocked_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_review_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_summary_rows": int(len(pocket_benchmark_case_interpretation_matrix_summary_df)),
        "pocket_benchmark_case_interpretation_matrix_summary": pocket_benchmark_case_interpretation_matrix_summary_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_matrix_summary_status": str(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("summary_status") or "") if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else "",
        "pocket_benchmark_case_interpretation_matrix_summary_usable_cases": int(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("usable_claim_ready_case_count") or 0) if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else 0,
        "pocket_benchmark_case_interpretation_matrix_queue_rows": int(len(pocket_benchmark_case_interpretation_matrix_queue_df)),
        "pocket_benchmark_case_interpretation_matrix_queue": pocket_benchmark_case_interpretation_matrix_queue_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_matrix_queue_blocker_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_queue_review_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_rows": int(len(pocket_benchmark_dataset_interpretation_df)),
        "pocket_benchmark_dataset_interpretation": pocket_benchmark_dataset_interpretation_df.to_dict(orient="records"),
        "pocket_benchmark_dataset_interpretation_blocked_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_review_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-blocked").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-review-needed").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-gate-mismatch").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_gate_mismatch"].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_gate_mismatch" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and "action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and "action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df["source_gate_mismatch"].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and "source_gate_mismatch" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_action_count": int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df["action_count"], errors="coerce").fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and "action_count" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_p0_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df["priority"].astype(str).eq("P0").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and "priority" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_mismatch_count": int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df["mismatch_count"], errors="coerce").fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and "mismatch_count" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_priority": str(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.iloc[0].get("priority") or "") if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty else "",
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_source_impact": str(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.iloc[0].get("source_impact_status") or "") if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty else "",
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_available": bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_available": bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_bytes": int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df["byte_size"], errors="coerce").fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty and "byte_size" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_hash_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty and "sha256" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_queue_rows": int(len(pocket_benchmark_dataset_interpretation_queue_df)),
        "pocket_benchmark_dataset_interpretation_queue": pocket_benchmark_dataset_interpretation_queue_df.to_dict(orient="records"),
        "pocket_benchmark_dataset_interpretation_queue_blocker_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_queue_review_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_checklist_available": bool(pocket_benchmark_dataset_interpretation_checklist_markdown),
        "pocket_benchmark_dataset_interpretation_checklist": pocket_benchmark_dataset_interpretation_checklist_markdown,
        "pocket_benchmark_dataset_interpretation_report_available": bool(pocket_benchmark_dataset_interpretation_report_markdown),
        "pocket_benchmark_dataset_interpretation_report": pocket_benchmark_dataset_interpretation_report_markdown,
        "pocket_benchmark_summary_rows": int(len(pocket_benchmark_summary_df)),
        "pocket_benchmark_summary": pocket_benchmark_summary_df.to_dict(orient="records"),
        "pocket_benchmark_details_rows": int(len(pocket_benchmark_details_df)),
        "pocket_benchmark_details": pocket_benchmark_details_df.to_dict(orient="records"),
        "pocket_benchmark_case_summary_rows": int(len(pocket_benchmark_case_summary_df)),
        "pocket_benchmark_case_summary": pocket_benchmark_case_summary_df.to_dict(orient="records"),
        "pocket_benchmark_dataset_summary_rows": int(len(pocket_benchmark_dataset_summary_df)),
        "pocket_benchmark_dataset_summary": pocket_benchmark_dataset_summary_df.to_dict(orient="records"),
        "pocket_benchmark_variant_comparison_rows": int(len(pocket_benchmark_variant_comparison_df)),
        "pocket_benchmark_variant_comparison": pocket_benchmark_variant_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_case_comparison_rows": int(len(pocket_benchmark_variant_case_comparison_df)),
        "pocket_benchmark_variant_case_comparison": pocket_benchmark_variant_case_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_dataset_comparison_rows": int(len(pocket_benchmark_variant_dataset_comparison_df)),
        "pocket_benchmark_variant_dataset_comparison": pocket_benchmark_variant_dataset_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_detail_comparison_rows": int(len(pocket_benchmark_variant_detail_comparison_df)),
        "pocket_benchmark_variant_detail_comparison": pocket_benchmark_variant_detail_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_remediation_rows": int(len(pocket_benchmark_variant_remediation_df)),
        "pocket_benchmark_variant_remediation": pocket_benchmark_variant_remediation_df.to_dict(orient="records"),
        "pocket_benchmark_variant_remediation_summary_rows": int(len(pocket_benchmark_variant_remediation_summary_df)),
        "pocket_benchmark_variant_remediation_summary": pocket_benchmark_variant_remediation_summary_df.to_dict(orient="records"),
        "pocket_benchmark_variant_remediation_checklist_available": bool(pocket_benchmark_variant_remediation_checklist_markdown),
        "pocket_benchmark_variant_remediation_checklist": pocket_benchmark_variant_remediation_checklist_markdown,
        "pocket_benchmark_top1_coverage": float(top1_benchmark.get("coverage_ratio") or 0.0) if top1_benchmark is not None else None,
        "pocket_benchmark_top1_status": str(top1_benchmark.get("benchmark_status") or "") if top1_benchmark is not None else None,
        "pocket_benchmark_top3_coverage": float(top3_benchmark.get("coverage_ratio") or 0.0) if top3_benchmark is not None else None,
        "pocket_benchmark_top3_status": str(top3_benchmark.get("benchmark_status") or "") if top3_benchmark is not None else None,
        "pocket_benchmark_best_rank": int(top3_benchmark.get("best_rank") or 0) if top3_benchmark is not None else 0,
        "pocket_benchmark_best_pocket_id": str(top3_benchmark.get("best_pocket_id") or "") if top3_benchmark is not None else None,
        "consensus_rerank_suggestion_rows": int(len(consensus_rerank_suggestion_df)),
        "consensus_rerank_suggestions": consensus_rerank_suggestion_df.to_dict(orient="records"),
        "top_consensus_rerank_pocket_id": str(top_consensus_rerank_suggestion.get("pocket_id")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("pocket_id")) else None,
        "top_consensus_rerank_status": str(top_consensus_rerank_suggestion.get("suggestion_status")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("suggestion_status")) else None,
        "top_consensus_rerank_rank_delta": int(top_consensus_rerank_suggestion.get("rank_delta") or 0) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("rank_delta")) else 0,
        "consensus_rerank_preview_rows": int(len(consensus_rerank_preview_df)),
        "consensus_rerank_preview": consensus_rerank_preview_df.to_dict(orient="records"),
        "top_consensus_preview_pocket_id": str(top_consensus_rerank_preview.get("pocket_id")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("pocket_id")) else None,
        "top_consensus_preview_decision": str(top_consensus_rerank_preview.get("preview_decision")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_decision")) else None,
        "top_consensus_preview_rank_delta": int(top_consensus_rerank_preview.get("preview_rank_delta") or 0) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_rank_delta")) else 0,
        "top_consensus_preview_score": float(top_consensus_rerank_preview.get("preview_score")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_score")) else None,
        "consensus_rerank_policy_gate": consensus_rerank_policy_gate_df.to_dict(orient="records"),
        "consensus_rerank_policy_status": str(top_consensus_rerank_policy_gate.get("policy_status")) if top_consensus_rerank_policy_gate is not None and pd.notna(top_consensus_rerank_policy_gate.get("policy_status")) else "",
        "consensus_rerank_policy_changed_rows": int(top_consensus_rerank_policy_gate.get("changed_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
        "consensus_rerank_policy_blocked_rows": int(top_consensus_rerank_policy_gate.get("blocked_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
        "consensus_rerank_action_queue_rows": int(len(consensus_rerank_action_queue_df)),
        "consensus_rerank_action_queue": consensus_rerank_action_queue_df.to_dict(orient="records"),
        "top_consensus_rerank_action_pocket_id": str(top_consensus_rerank_action.get("pocket_id")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("pocket_id")) else None,
        "top_consensus_rerank_issue_type": str(top_consensus_rerank_action.get("issue_type")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_type")) else None,
        "top_consensus_rerank_issue_severity": str(top_consensus_rerank_action.get("issue_severity")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_severity")) else None,
        "consensus_rerank_action_checklist_available": bool(consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty),
        "consensus_rerank_apply_simulation_rows": int(len(consensus_rerank_apply_simulation_df)),
        "consensus_rerank_apply_simulation": consensus_rerank_apply_simulation_df.to_dict(orient="records"),
        "top_consensus_rerank_apply_pocket_id": str(top_consensus_rerank_apply.get("pocket_id")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("pocket_id")) else None,
        "top_consensus_rerank_apply_status": str(top_consensus_rerank_apply.get("apply_status")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("apply_status")) else None,
        "top_consensus_rerank_apply_rank_delta": int(top_consensus_rerank_apply.get("simulated_rank_delta") or 0) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("simulated_rank_delta")) else 0,
        "consensus_rerank_simulation_delta_rows": int(len(consensus_rerank_simulation_delta_df)),
        "consensus_rerank_simulation_delta": consensus_rerank_simulation_delta_df.to_dict(orient="records"),
        "top_consensus_rerank_delta_pocket_id": str(top_consensus_rerank_delta.get("pocket_id")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("pocket_id")) else None,
        "top_consensus_rerank_delta_change_type": str(top_consensus_rerank_delta.get("change_type")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("change_type")) else None,
        "top_consensus_rerank_delta_rank_delta": int(top_consensus_rerank_delta.get("rank_delta") or 0) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("rank_delta")) else 0,
        "consensus_rerank_precision_scorecard_rows": int(len(consensus_rerank_precision_scorecard_df)),
        "consensus_rerank_precision_scorecard": consensus_rerank_precision_scorecard_df.to_dict(orient="records"),
        "consensus_rerank_precision_score": int(top_consensus_rerank_scorecard.get("precision_improvement_score") or 0) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("precision_improvement_score")) else 0,
        "consensus_rerank_precision_status": str(top_consensus_rerank_scorecard.get("scorecard_status")) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("scorecard_status")) else None,
        "consensus_rerank_positive_signal_rows": int(top_consensus_rerank_scorecard.get("positive_signal_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
        "consensus_rerank_open_blocker_rows": int(top_consensus_rerank_scorecard.get("open_blocker_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
        "consensus_rerank_precision_guardrail_rows": int(len(consensus_rerank_precision_guardrail_df)),
        "consensus_rerank_precision_guardrail": consensus_rerank_precision_guardrail_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_status": str(top_consensus_rerank_guardrail.get("guardrail_status")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_status")) else None,
        "consensus_rerank_guardrail_decision": str(top_consensus_rerank_guardrail.get("guardrail_decision")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_decision")) else None,
        "consensus_rerank_guardrail_apply_mode": str(top_consensus_rerank_guardrail.get("apply_mode")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("apply_mode")) else None,
        "consensus_rerank_guardrail_can_apply_after_review": bool(top_consensus_rerank_guardrail.get("can_apply_after_manual_review")) if top_consensus_rerank_guardrail is not None else False,
        "consensus_rerank_guardrail_report_available": bool(consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty),
        "consensus_rerank_guardrail_artifact_manifest_rows": int(len(consensus_rerank_guardrail_artifact_manifest_df)),
        "consensus_rerank_guardrail_artifact_manifest": consensus_rerank_guardrail_artifact_manifest_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_handoff_zip_available": bool(consensus_rerank_guardrail_handoff_zip),
        "consensus_rerank_guardrail_bundle_verification_rows": int(len(consensus_rerank_guardrail_bundle_verification_df)),
        "consensus_rerank_guardrail_bundle_verification": consensus_rerank_guardrail_bundle_verification_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_bundle_verification_summary": consensus_rerank_guardrail_bundle_verification_summary_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_bundle_verification_status": str(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty and pd.notna(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) else None,
        "consensus_rerank_guardrail_bundle_verification_failed_rows": int(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("failed_files") or 0) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else 0,
        "consensus_rerank_guardrail_handoff_certificate_available": bool(consensus_rerank_guardrail_handoff_certificate_markdown),
        "consensus_rerank_release_decision_template_rows": int(len(consensus_rerank_release_decision_template_df)),
        "consensus_rerank_release_decision_template": consensus_rerank_release_decision_template_df.to_dict(orient="records"),
        "consensus_rerank_release_decision_rows": int(len(consensus_rerank_release_decision_df)),
        "consensus_rerank_release_decision_status": str(consensus_rerank_release_decision_meta.get("status") or ""),
        "consensus_rerank_release_decision_metadata": consensus_rerank_release_decision_meta,
        "consensus_rerank_release_decisions": consensus_rerank_release_decision_df.to_dict(orient="records"),
        "consensus_rerank_release_decision_validation_rows": int(len(consensus_rerank_release_decision_validation_df)),
        "consensus_rerank_release_decision_blocked_rows": int(
            (consensus_rerank_release_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
        ) if not consensus_rerank_release_decision_validation_df.empty and "validation_status" in consensus_rerank_release_decision_validation_df.columns else 0,
        "consensus_rerank_release_decision_validation": consensus_rerank_release_decision_validation_df.to_dict(orient="records"),
        "consensus_rerank_release_decision_summary": consensus_rerank_release_decision_summary_df.to_dict(orient="records"),
        "consensus_rerank_release_review_status": str(top_consensus_rerank_release_decision_summary.get("release_review_status")) if top_consensus_rerank_release_decision_summary is not None and pd.notna(top_consensus_rerank_release_decision_summary.get("release_review_status")) else "",
        "consensus_rerank_release_allowed": bool(top_consensus_rerank_release_decision_summary.get("release_allowed")) if top_consensus_rerank_release_decision_summary is not None else False,
        "consensus_rerank_release_apply_plan_rows": int(len(consensus_rerank_release_apply_plan_df)),
        "consensus_rerank_release_apply_plan": consensus_rerank_release_apply_plan_df.to_dict(orient="records"),
        "top_consensus_rerank_release_apply_pocket_id": str(top_consensus_rerank_release_apply_plan.get("pocket_id")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("pocket_id")) else None,
        "top_consensus_rerank_release_apply_status": str(top_consensus_rerank_release_apply_plan.get("release_apply_status")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("release_apply_status")) else None,
        "consensus_rerank_release_apply_report_available": bool(consensus_rerank_release_apply_report_markdown),
        "consensus_rerank_release_execution_template_rows": int(len(consensus_rerank_release_execution_template_df)),
        "consensus_rerank_release_execution_template": consensus_rerank_release_execution_template_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_receipt_rows": int(len(consensus_rerank_release_execution_receipt_df)),
        "consensus_rerank_release_execution_receipt_status": str(consensus_rerank_release_execution_receipt_meta.get("status") or ""),
        "consensus_rerank_release_execution_receipt_metadata": consensus_rerank_release_execution_receipt_meta,
        "consensus_rerank_release_execution_receipt": consensus_rerank_release_execution_receipt_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_validation_rows": int(len(consensus_rerank_release_execution_validation_df)),
        "consensus_rerank_release_execution_blocked_rows": int(
            (consensus_rerank_release_execution_validation_df["validation_status"].astype(str) == "blocked").sum()
        ) if not consensus_rerank_release_execution_validation_df.empty and "validation_status" in consensus_rerank_release_execution_validation_df.columns else 0,
        "consensus_rerank_release_execution_validation": consensus_rerank_release_execution_validation_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_summary": consensus_rerank_release_execution_summary_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_review_status": str(top_consensus_rerank_release_execution_summary.get("execution_review_status")) if top_consensus_rerank_release_execution_summary is not None and pd.notna(top_consensus_rerank_release_execution_summary.get("execution_review_status")) else "",
        "consensus_rerank_release_execution_complete": bool(top_consensus_rerank_release_execution_summary.get("execution_complete")) if top_consensus_rerank_release_execution_summary is not None else False,
        "consensus_rerank_release_execution_report_available": bool(consensus_rerank_release_execution_report_markdown),
        "consensus_rerank_release_closure_certificate_available": bool(consensus_rerank_release_closure_certificate_markdown),
        "consensus_rerank_release_closure_ledger_rows": int(len(consensus_rerank_release_closure_ledger_df)),
        "consensus_rerank_release_closure_ledger_blocked_rows": int(
            consensus_rerank_release_closure_ledger_df["closure_check"].astype(str).str.lower().isin({"blocked", "missing"}).sum()
        ) if not consensus_rerank_release_closure_ledger_df.empty and "closure_check" in consensus_rerank_release_closure_ledger_df.columns else 0,
        "consensus_rerank_release_closure_ledger": consensus_rerank_release_closure_ledger_df.to_dict(orient="records"),
        "consensus_rerank_release_closure_summary_rows": int(len(consensus_rerank_release_closure_summary_df)),
        "consensus_rerank_release_closure_summary": consensus_rerank_release_closure_summary_df.to_dict(orient="records"),
        "consensus_rerank_release_closure_readiness_status": str(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) if top_consensus_rerank_release_closure_summary is not None and pd.notna(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) else "",
        "consensus_rerank_release_closed": bool(top_consensus_rerank_release_closure_summary.get("release_closed")) if top_consensus_rerank_release_closure_summary is not None else False,
        "consensus_rerank_release_closure_blocker_rows": int(len(consensus_rerank_release_closure_blocker_df)),
        "top_consensus_rerank_release_closure_blocker_type": str(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) if not consensus_rerank_release_closure_blocker_df.empty and pd.notna(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) else "",
        "consensus_rerank_release_closure_blockers": consensus_rerank_release_closure_blocker_df.to_dict(orient="records"),
        "consensus_rerank_release_closure_remediation_checklist_available": bool(consensus_rerank_release_closure_remediation_checklist_markdown),
        "consensus_rerank_release_closure_detached_manifest_rows": int(len(consensus_rerank_release_closure_detached_manifest_df)),
        "consensus_rerank_release_closure_detached_manifest": consensus_rerank_release_closure_detached_manifest_df.to_dict(orient="records"),
        "p2rank_ab_enabled": bool(p2rank_ab_enabled),
        "p2rank_ab_comparison": p2rank_ab_df.to_dict(orient="records"),
        "literature_ab_enabled": bool(literature_ab_enabled),
        "literature_ab_comparison": literature_ab_df.to_dict(orient="records"),
        "evidence_route_enabled": bool(auto_external_evidence_route),
        "evidence_route_min_support": float(external_route_min_support),
        "evidence_route_min_confidence": float(external_route_min_confidence),
        "evidence_route_min_mapping_quality": float(external_route_min_quality),
        "evidence_route_radius": float(external_route_radius) if external_route_radius is not None else None,
        "evidence_route_ab_enabled": bool(evidence_route_ab_enabled),
        "evidence_route_ab_comparison": evidence_route_ab_df.to_dict(orient="records"),
        "conservation_site_rows": int(len(conservation_site_df)),
        "conservation_source_name": str(conservation_site_meta.get("source") or ""),
        "conservation_score_mean": str(conservation_site_meta.get("score_mean") or ""),
        "conservation_ab_enabled": bool(conservation_ab_enabled),
        "conservation_ab_comparison": conservation_ab_df.to_dict(orient="records"),
        "external_exact_rows": int(
            (external_site_df["mapping_level"].astype(str).str.lower() == "exact").sum()
        ) if not external_site_df.empty and "mapping_level" in external_site_df.columns else 0,
        "external_weak_rows": int(
            (external_site_df["mapping_level"].astype(str).str.lower() == "weak").sum()
        ) if not external_site_df.empty and "mapping_level" in external_site_df.columns else 0,
        "interface_rows": int(len(enriched_annotations)),
        "interface_summary": interface_summary.to_dict(orient="records"),
        "overlap_summary": overlap_summary.to_dict(orient="records"),
        "auto_detection_mode": auto_detection_mode,
        "uploaded_pocket_rows": int(len(uploaded_pocket_df)),
        "auto_pocket_rows": int(len(auto_pocket_df)),
        "inferred_annotation_rows": int(len(inferred_annotation_df)),
        "top_pocket_evidence_quality_label": str(top_pocket.get("evidence_quality_label")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_label")) else None,
        "top_pocket_evidence_quality_score": float(top_pocket.get("evidence_quality_score")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_score")) else None,
        "top_pocket_evidence_quality_warning": str(top_pocket.get("evidence_quality_warning")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_warning")) else None,
        "top_pocket_decision_label": str(top_pocket_decision.get("decision_label")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_label")) else None,
        "top_pocket_decision_score": float(top_pocket_decision.get("decision_score")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_score")) else None,
        "top_pocket_audit_status": str(top_pocket_decision.get("audit_status")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("audit_status")) else None,
        "top_pocket_reliability_gaps": top_reliability_gaps or None,
        "top_pocket_precision_tier": str(top_pocket_triage.get("precision_tier")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("precision_tier")) else None,
        "top_pocket_triage_action": str(top_pocket_triage.get("triage_action")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("triage_action")) else None,
        "top_precision_triage_pocket_id": str(top_precision_triage.get("pocket_id")) if top_precision_triage is not None and pd.notna(top_precision_triage.get("pocket_id")) else None,
        "pocket_decision": pocket_decision_df.to_dict(orient="records"),
        "pocket_reliability": pocket_reliability_df.to_dict(orient="records"),
        "pocket_precision_triage": pocket_triage_df.to_dict(orient="records"),
        "auto_pocket_summary": auto_pocket_summary.to_dict(orient="records"),
        "joint_candidates": joint_candidate_df.to_dict(orient="records"),
    },
)

uploaded_annotation_summary = build_interface_summary(
    enrich_interface_annotations(
        uploaded_annotation_df,
        pocket_residues=effective_pocket_residues,
        hotspot_residues=hotspot_residues,
    )
) if not uploaded_annotation_df.empty else pd.DataFrame()
inferred_annotation_summary = build_interface_summary(
    enrich_interface_annotations(
        inferred_annotation_df,
        pocket_residues=effective_pocket_residues,
        hotspot_residues=hotspot_residues,
    )
) if not inferred_annotation_df.empty else pd.DataFrame()

if not inferred_annotation_df.empty and "inference_basis" in inferred_annotation_df.columns:
    inferred_basis_counts = inferred_annotation_df["inference_basis"].astype(str).value_counts().to_dict()
    inferred_basis_text = "；".join(
        f"{INFERENCE_BASIS_LABELS.get(basis, basis)}: {count}"
        for basis, count in inferred_basis_counts.items()
    )
else:
    inferred_basis_text = ""

pocket_hotspot_df = (
    effective_pocket_df[effective_pocket_df["is_hotspot"].fillna(False).astype(bool)].copy()
    if not effective_pocket_df.empty and "is_hotspot" in effective_pocket_df.columns
    else pd.DataFrame()
)
interface_hotspot_df = (
    enriched_annotations[enriched_annotations["is_hotspot"]].copy()
    if not enriched_annotations.empty and "is_hotspot" in enriched_annotations.columns
    else pd.DataFrame()
)
interface_pocket_df = (
    enriched_annotations[enriched_annotations["is_pocket"]].copy()
    if not enriched_annotations.empty and "is_pocket" in enriched_annotations.columns
    else pd.DataFrame()
)
triple_overlap_df = (
    enriched_annotations[enriched_annotations["is_overlap"]].copy()
    if not enriched_annotations.empty and "is_overlap" in enriched_annotations.columns
    else pd.DataFrame()
)

metric_cols = st.columns(6)
metric_cols[0].metric("有效口袋数", len(effective_pocket_summary) if not effective_pocket_summary.empty else 0)
metric_cols[1].metric("自动口袋数", len(auto_pocket_summary) if not auto_pocket_summary.empty else 0)
metric_cols[2].metric("界面注释数", len(enriched_annotations))
metric_cols[3].metric("三重交集", len(triple_overlap_df))
metric_cols[4].metric("热点数", len(hotspot_df))
metric_cols[5].metric("蛋白体积（估算）", f"{protein_volume:,.1f} A³" if protein_volume is not None else "-")

st.caption(
    f"当前主分析使用口袋来源：{POCKET_SOURCE_LABELS.get(effective_pocket_mode, effective_pocket_mode)}；"
    f"界面来源：{ANNOTATION_SOURCE_LABELS.get(effective_annotation_mode, effective_annotation_mode)}；"
    f"热点判定：ΔG ≤ -{hotspot_threshold:.1f}，至少保留 {hotspot_top_n} 个残基。"
)
if inferred_basis_text:
    st.caption(f"结构推断界面依据：{inferred_basis_text}")
if top_joint_candidate is not None:
    st.caption(
        f"联合推荐 Top1：{top_joint_candidate['pocket_id']} / {top_joint_candidate['recommendation_label']} / {top_joint_candidate['recommendation_reason']}"
    )
st.markdown(analysis_text)

def _render_evidence_context_panels() -> None:
    has_content = (
        (not ai_evidence_audit_df.empty)
        or (not residue_evidence_consensus_df.empty)
        or (not pocket_consensus_coverage_df.empty)
    )
    if not has_content:
        return
    st.subheader("证据复核与残基共识")
    st.caption("这些表格用于解释外部证据、AI 证据、保守性证据如何落到残基层面；默认不直接覆盖当前口袋排名。")
    if not ai_evidence_audit_df.empty:
        with st.expander("AI 证据审计", expanded=False):
            st.caption("AI 提取的残基会与非 AI 证据、来源片段、映射置信度和结构身份比对后，才允许进入排名信号。")
            st.dataframe(ai_evidence_audit_df, use_container_width=True, hide_index=True)
    if not residue_evidence_consensus_df.empty:
        with st.expander("残基证据共识", expanded=False):
            st.caption("把外部数据库、文献、AI 和保守性证据聚合到残基锚点层，用于精度复核。")
            st.dataframe(residue_evidence_consensus_df, use_container_width=True, hide_index=True)
    if not pocket_consensus_coverage_df.empty:
        with st.expander("口袋共识覆盖", expanded=False):
            st.caption("把残基层共识锚点映射回每个口袋，用于解释候选口袋是否覆盖关键残基；不改变原始排名。")
            st.dataframe(pocket_consensus_coverage_df, use_container_width=True, hide_index=True)


def _render_benchmark_review_panels() -> None:
    has_content = (
        (not benchmark_reference_candidate_df.empty)
        or (not benchmark_reference_source_audit_df.empty)
        or (not pocket_benchmark_summary_df.empty)
    )
    if not has_content:
        return
    st.subheader("基准参考与精度评估")
    st.caption("这些面板用于评估当前口袋是否覆盖人工或外部证据整理出的催化残基；属于评测层，不会自动改变排名。")

    if not benchmark_reference_candidate_df.empty:
        with st.expander("从外部证据生成的基准参考候选", expanded=False):
            st.caption("由当前加载的 UniProt、M-CSA、文献或 AI 残基证据生成。作为独立基准使用前，需要先人工整理。")
            if not benchmark_reference_import_summary_df.empty:
                st.dataframe(benchmark_reference_import_summary_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_candidate_review_queue_df.empty:
                st.caption("候选复核队列：这些项目需要修复后，才能提升为可信基准参考。")
                st.dataframe(benchmark_reference_candidate_review_queue_df, use_container_width=True, hide_index=True)
                if benchmark_reference_candidate_review_checklist_markdown:
                    with st.expander("基准参考候选复核清单", expanded=False):
                        st.markdown(benchmark_reference_candidate_review_checklist_markdown)
            if not benchmark_reference_candidate_review_decision_template_df.empty:
                st.caption("决策模板：填写 review_decision、审核人和验证证据后，再上传回系统。")
                st.dataframe(benchmark_reference_candidate_review_decision_template_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_candidate_review_decision_validation_df.empty:
                st.caption("决策校验：存在阻断行时，不能把候选提升为正式参考。")
                st.dataframe(benchmark_reference_candidate_review_decision_validation_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_candidate_review_outcome_df.empty:
                st.caption("决策结果：只有该残基的所有风险动作都被接受时，候选残基才可提升。")
                st.dataframe(benchmark_reference_candidate_review_outcome_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_candidate_accepted_df.empty:
                st.caption("已接受参考候选：包含干净行，以及复核动作全部通过的候选残基。")
                st.dataframe(benchmark_reference_candidate_accepted_df, use_container_width=True, hide_index=True)
            st.dataframe(benchmark_reference_candidate_df, use_container_width=True, hide_index=True)

    if not benchmark_reference_source_audit_df.empty:
        with st.expander("基准参考来源审计", expanded=False):
            st.caption("审计最终用于基准评分的参考行，包括来源模式，以及能否支撑独立精度声明。")
            if not benchmark_reference_source_audit_summary_df.empty:
                st.dataframe(benchmark_reference_source_audit_summary_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_summary_df.empty:
                st.caption("来源审计案例汇总：按 benchmark_id 汇总仅来源导致的阻断或复核需求。")
                st.dataframe(benchmark_reference_source_audit_case_summary_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_template_df.empty:
                st.caption("来源审计案例决策模板：在把覆盖率当作精度前，需要为每个受影响 benchmark_id 填写一条决策。")
                st.dataframe(benchmark_reference_source_audit_case_decision_template_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_df.empty:
                st.caption("来源审计案例决策：上传的来源风险关闭决策标准化结果。")
                st.dataframe(benchmark_reference_source_audit_case_decision_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_validation_df.empty:
                st.caption("来源审计案例决策校验：存在阻断行时，不能清除来源风险案例。")
                st.dataframe(benchmark_reference_source_audit_case_decision_validation_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_outcome_summary_df.empty:
                st.caption("来源审计案例决策结果汇总：展示关闭状态、开放案例和下一步动作。")
                st.dataframe(benchmark_reference_source_audit_case_decision_outcome_summary_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty:
                st.caption("来源审计案例决策关闭队列：决策结果后的机器可读开放案例动作。")
                st.dataframe(benchmark_reference_source_audit_case_decision_closure_queue_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty:
                st.caption("来源审计案例决策就绪影响汇总：决策后被清除或仍开放的就绪影响。")
                st.dataframe(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df, use_container_width=True, hide_index=True)
            if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty:
                st.caption("来源审计案例决策就绪影响：逐案例对比原始来源问题和决策调整后的就绪问题。")
                st.dataframe(benchmark_reference_source_audit_case_decision_readiness_impact_df, use_container_width=True, hide_index=True)
            if benchmark_reference_source_audit_case_decision_closure_checklist_markdown:
                with st.expander("基准参考来源审计决策关闭清单", expanded=False):
                    st.markdown(benchmark_reference_source_audit_case_decision_closure_checklist_markdown)
            if not benchmark_reference_source_audit_case_decision_outcome_df.empty:
                st.caption("来源审计案例决策结果：每个来源风险案例在校验后的应用状态。")
                st.dataframe(benchmark_reference_source_audit_case_decision_outcome_df, use_container_width=True, hide_index=True)
            if benchmark_reference_source_audit_case_checklist_markdown:
                with st.expander("基准参考来源审计案例清单", expanded=False):
                    st.markdown(benchmark_reference_source_audit_case_checklist_markdown)
            if not benchmark_reference_source_audit_action_queue_df.empty:
                st.caption("来源审计行动队列：从非就绪基准参考来源中提取的来源修复动作。")
                st.dataframe(benchmark_reference_source_audit_action_queue_df, use_container_width=True, hide_index=True)
            if benchmark_reference_source_audit_checklist_markdown:
                with st.expander("基准参考来源审计清单", expanded=False):
                    st.markdown(benchmark_reference_source_audit_checklist_markdown)
            st.dataframe(benchmark_reference_source_audit_df, use_container_width=True, hide_index=True)

    if not pocket_benchmark_summary_df.empty:
        with st.expander("催化口袋 Benchmark", expanded=False):
            st.caption("把当前口袋排名与上传或整理出的催化残基参考集比较。这里只做评估，不改变排名。")
            benchmark_metric_cols = st.columns(3)
            benchmark_metric_cols[0].metric(
                "Top-1 覆盖率",
                f"{float(top1_benchmark.get('coverage_ratio') or 0.0):.2f}" if top1_benchmark is not None else "-",
                str(top1_benchmark.get("benchmark_status") or "") if top1_benchmark is not None else "",
            )
            benchmark_metric_cols[1].metric(
                "Top-3 覆盖率",
                f"{float(top3_benchmark.get('coverage_ratio') or 0.0):.2f}" if top3_benchmark is not None else "-",
                str(top3_benchmark.get("benchmark_status") or "") if top3_benchmark is not None else "",
            )
            benchmark_metric_cols[2].metric(
                "最佳命中排名",
                str(top3_benchmark.get("best_rank") or "-") if top3_benchmark is not None else "-",
                str(top3_benchmark.get("best_pocket_id") or "") if top3_benchmark is not None else "",
            )
            if not pocket_benchmark_reference_quality_issue_df.empty:
                st.caption("基准参考整理质量：信任覆盖率前，先复核 P1/P2 问题行。")
                if not pocket_benchmark_reference_quality_summary_df.empty:
                    st.dataframe(pocket_benchmark_reference_quality_summary_df, use_container_width=True, hide_index=True)
                st.dataframe(pocket_benchmark_reference_quality_issue_df, use_container_width=True, hide_index=True)
                if pocket_benchmark_reference_quality_checklist_markdown:
                    with st.expander("基准参考整理清单", expanded=False):
                        st.markdown(pocket_benchmark_reference_quality_checklist_markdown)
            if not pocket_benchmark_reference_structure_validation_df.empty:
                st.caption("基准参考结构校验：把漏检视为检测失败前，先确认参考残基确实存在于上传 PDB 中。")
                if not pocket_benchmark_reference_structure_validation_summary_df.empty:
                    st.dataframe(pocket_benchmark_reference_structure_validation_summary_df, use_container_width=True, hide_index=True)
                st.dataframe(pocket_benchmark_reference_structure_validation_df, use_container_width=True, hide_index=True)
                if pocket_benchmark_reference_structure_validation_checklist_markdown:
                    with st.expander("基准参考结构校验清单", expanded=False):
                        st.markdown(pocket_benchmark_reference_structure_validation_checklist_markdown)
            if not pocket_benchmark_reference_readiness_summary_df.empty:
                st.caption("基准参考就绪门控：在把覆盖率作为准确率声明前，先给出综合决策。")
                st.dataframe(pocket_benchmark_reference_readiness_summary_df, use_container_width=True, hide_index=True)
                if not pocket_benchmark_reference_readiness_case_summary_df.empty:
                    st.dataframe(pocket_benchmark_reference_readiness_case_summary_df, use_container_width=True, hide_index=True)
                if not pocket_benchmark_reference_readiness_queue_df.empty:
                    st.dataframe(pocket_benchmark_reference_readiness_queue_df, use_container_width=True, hide_index=True)
                if pocket_benchmark_reference_readiness_checklist_markdown:
                    with st.expander("基准参考就绪清单", expanded=False):
                        st.markdown(pocket_benchmark_reference_readiness_checklist_markdown)
            if not pocket_benchmark_interpretation_df.empty:
                st.caption("基准解释：结合 Top-N 覆盖率与参考就绪状态，再决定是否可做准确率声明。")
                st.dataframe(pocket_benchmark_interpretation_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_case_interpretation_df.empty:
                st.caption("基准案例解释：按案例结合 Top-N 覆盖率和案例级就绪状态。")
                st.dataframe(pocket_benchmark_case_interpretation_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_case_interpretation_matrix_df.empty:
                st.caption("基准案例解释矩阵：每个 benchmark_id 一行，展示 Top-1/Top-3/Top-5 声明状态和覆盖率。")
                if not pocket_benchmark_case_interpretation_matrix_summary_df.empty:
                    st.dataframe(pocket_benchmark_case_interpretation_matrix_summary_df, use_container_width=True, hide_index=True)
                if not pocket_benchmark_case_interpretation_matrix_queue_df.empty:
                    st.caption("基准案例解释矩阵队列：每个未达到可声明状态的案例对应一条动作。")
                    st.dataframe(pocket_benchmark_case_interpretation_matrix_queue_df, use_container_width=True, hide_index=True)
                st.dataframe(pocket_benchmark_case_interpretation_matrix_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_dataset_interpretation_df.empty:
                st.caption("基准数据集解释：按 Top-N 汇总可声明、阻断和需复核 case。")
                st.dataframe(pocket_benchmark_dataset_interpretation_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty:
                st.caption("基准来源审计决策数据集影响：Top-N 来源决策对数据集声明就绪性的影响。")
                st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty:
                st.caption("基准来源审计决策数据集影响案例：逐案例展示 Top-N 来源决策影响和门控不一致。")
                st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty:
                st.caption("基准来源审计决策数据集影响行动队列：机器可读的阻断/复核/不一致案例动作。")
                if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty:
                    st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df, use_container_width=True, hide_index=True)
                st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df, use_container_width=True, hide_index=True)
                if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown:
                    with st.expander("基准来源审计决策数据集影响案例清单", expanded=False):
                        st.markdown(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown)
            if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown:
                with st.expander("基准来源审计决策数据集影响报告", expanded=False):
                    st.markdown(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown)
            if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty:
                st.caption("基准来源审计决策数据集影响产物清单：数据集影响复核导出的完整性索引。")
                st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_dataset_interpretation_queue_df.empty:
                st.caption("基准数据集解释队列：阻断或削弱数据集级精度声明的非就绪 case。")
                st.dataframe(pocket_benchmark_dataset_interpretation_queue_df, use_container_width=True, hide_index=True)
                if pocket_benchmark_dataset_interpretation_checklist_markdown:
                    with st.expander("基准数据集解释清单", expanded=False):
                        st.markdown(pocket_benchmark_dataset_interpretation_checklist_markdown)
            if pocket_benchmark_dataset_interpretation_report_markdown:
                with st.expander("基准数据集解释报告", expanded=False):
                    st.markdown(pocket_benchmark_dataset_interpretation_report_markdown)
            st.dataframe(pocket_benchmark_summary_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_dataset_summary_df.empty:
                st.caption("基准数据集汇总：按案例聚合，避免大型催化残基集合主导准确率。")
                st.dataframe(pocket_benchmark_dataset_summary_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_case_summary_df.empty:
                st.caption("基准案例汇总：提供 benchmark_id/case_id 时，Top-N 覆盖率按案例拆分。")
                st.dataframe(pocket_benchmark_case_summary_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_variant_comparison_df.empty:
                st.caption("基准变体对比：coverage_delta 为负或 coverage_loss 为正，说明移除该证据路径会损伤催化残基覆盖。")
                st.dataframe(pocket_benchmark_variant_comparison_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_variant_dataset_comparison_df.empty:
                st.caption("基准变体数据集对比：跨基准案例汇总覆盖损失。")
                st.dataframe(pocket_benchmark_variant_dataset_comparison_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_variant_case_comparison_df.empty:
                st.caption("基准变体案例对比：检查移除某条证据路径时，哪些案例失去覆盖。")
                st.dataframe(pocket_benchmark_variant_case_comparison_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_variant_detail_comparison_df.empty:
                st.caption("基准变体残基对比：逐一列出每次消融中丢失、获得或保持不变的催化残基。")
                st.dataframe(pocket_benchmark_variant_detail_comparison_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_variant_remediation_df.empty:
                st.caption("基准修复队列：把丢失或当前漏检的催化残基转换成复核动作。")
                if not pocket_benchmark_variant_remediation_summary_df.empty:
                    st.dataframe(pocket_benchmark_variant_remediation_summary_df, use_container_width=True, hide_index=True)
                st.dataframe(pocket_benchmark_variant_remediation_df, use_container_width=True, hide_index=True)
                if pocket_benchmark_variant_remediation_checklist_markdown:
                    with st.expander("基准修复清单", expanded=False):
                        st.markdown(pocket_benchmark_variant_remediation_checklist_markdown)
            if not pocket_benchmark_details_df.empty:
                st.dataframe(pocket_benchmark_details_df, use_container_width=True, hide_index=True)
def _render_consensus_rerank_review_panels() -> None:
    def show_df(title: str, caption: str, table: pd.DataFrame) -> None:
        if table is None or getattr(table, "empty", True):
            return
        with st.expander(title, expanded=False):
            if caption:
                st.caption(caption)
            st.dataframe(table, use_container_width=True, hide_index=True)

    def show_markdown(title: str, caption: str, markdown_text: str) -> None:
        if not markdown_text:
            return
        with st.expander(title, expanded=False):
            if caption:
                st.caption(caption)
            st.markdown(markdown_text)

    consensus_tables = [
        consensus_rerank_suggestion_df,
        consensus_rerank_preview_df,
        consensus_rerank_policy_gate_df,
        consensus_rerank_action_queue_df,
        consensus_rerank_apply_simulation_df,
        consensus_rerank_simulation_delta_df,
        consensus_rerank_precision_scorecard_df,
        consensus_rerank_precision_guardrail_df,
        consensus_rerank_guardrail_artifact_manifest_df,
        consensus_rerank_guardrail_bundle_verification_summary_df,
        consensus_rerank_guardrail_bundle_verification_df,
        consensus_rerank_release_decision_template_df,
        consensus_rerank_release_decision_summary_df,
        consensus_rerank_release_apply_plan_df,
        consensus_rerank_release_execution_template_df,
        consensus_rerank_release_execution_summary_df,
        consensus_rerank_release_closure_summary_df,
        consensus_rerank_release_closure_blocker_df,
        consensus_rerank_release_closure_detached_manifest_df,
        consensus_rerank_release_closure_ledger_df,
        consensus_rerank_release_execution_validation_df,
        consensus_rerank_release_execution_receipt_df,
        consensus_rerank_release_decision_validation_df,
        consensus_rerank_release_decision_df,
    ]
    consensus_markdown = [
        consensus_rerank_precision_guardrail_report_markdown,
        consensus_rerank_guardrail_handoff_certificate_markdown,
        consensus_rerank_release_apply_report_markdown,
        consensus_rerank_release_execution_report_markdown,
        consensus_rerank_release_closure_certificate_markdown,
        consensus_rerank_release_closure_remediation_checklist_markdown,
        consensus_rerank_action_checklist_markdown if not consensus_rerank_action_queue_df.empty else "",
    ]
    has_failed_upload = (
        uploaded_consensus_rerank_release_decisions is not None
        and consensus_rerank_release_decision_df.empty
    ) or (
        uploaded_consensus_rerank_release_execution_receipt is not None
        and consensus_rerank_release_execution_receipt_df.empty
    )
    if not (
        any(table is not None and not getattr(table, "empty", True) for table in consensus_tables)
        or any(bool(markdown) for markdown in consensus_markdown)
        or has_failed_upload
    ):
        return

    st.subheader("共识重排审计")
    st.caption(
        "这些面板用于审计证据共识是否应该改变口袋排序，默认不会自动覆盖当前排名；适合放在主识别结果和活性位点判断之后复核。"
    )

    show_df(
        "共识重排建议",
        "保守判断共识证据是否应提升、保留、下调或复核某个口袋；不会自动修改排名。",
        consensus_rerank_suggestion_df,
    )
    show_df(
        "共识重排预览",
        "模拟共识证据带来的保守分数调整；只是预览，不替代当前有效排名。",
        consensus_rerank_preview_df,
    )
    show_df(
        "共识重排策略门控",
        "单行安全门控：判断重排预览是否可应用、需要复核，或应保持阻断。",
        consensus_rerank_policy_gate_df,
    )
    show_df(
        "共识重排行动队列",
        "在共识重排可信或启用前必须处理的可执行修复项。",
        consensus_rerank_action_queue_df,
    )
    show_df(
        "共识重排应用模拟",
        "非破坏性模拟保守重排规则后的排序；被阻断的证据会继续作为诊断信息，直到修复。",
        consensus_rerank_apply_simulation_df,
    )
    show_df(
        "共识重排变化明细",
        "解释每个口袋在非破坏性应用模拟中为何上升、保持不变或被冻结。",
        consensus_rerank_simulation_delta_df,
    )
    show_df(
        "共识重排精度评分卡",
        "单行汇总：模拟重排是否可能提升精度、仍被阻断，或只能保持诊断状态。",
        consensus_rerank_precision_scorecard_df,
    )
    show_df(
        "共识重排精度护栏",
        "Go/No-Go 护栏：判断共识重排可否应用、是否需要人工复核，或必须继续仅作诊断。",
        consensus_rerank_precision_guardrail_df,
    )
    if not consensus_rerank_precision_guardrail_df.empty:
        show_markdown(
            "共识重排精度护栏报告",
            "用于交接的 Markdown 报告，包含护栏决策、评分卡计数、放行情况和发布检查清单。",
            consensus_rerank_precision_guardrail_report_markdown,
        )
    show_df(
        "共识重排护栏产物清单",
        "重排护栏交接 ZIP 的完整性清单，记录每个产物的字节数和 SHA-256。",
        consensus_rerank_guardrail_artifact_manifest_df,
    )
    show_df(
        "共识重排护栏包校验汇总",
        "针对交接 ZIP 与清单的一行完整性校验结果。",
        consensus_rerank_guardrail_bundle_verification_summary_df,
    )
    show_df(
        "共识重排护栏包逐文件校验",
        "按清单中的字节数和 SHA-256 对 ZIP 内每个文件进行校验。",
        consensus_rerank_guardrail_bundle_verification_df,
    )
    show_markdown(
        "共识重排护栏交接证书",
        "独立交接证书，记录 ZIP 身份、校验结果和护栏发布决策。",
        consensus_rerank_guardrail_handoff_certificate_markdown,
    )
    show_df(
        "共识重排发布决策模板",
        "审核人签核模板，用于记录发布决策、阻断项清理和关键模拟排名变化。",
        consensus_rerank_release_decision_template_df,
    )
    if uploaded_consensus_rerank_release_decisions is not None and consensus_rerank_release_decision_df.empty:
        st.warning(
            f"共识重排发布决策上传不可用：{consensus_rerank_release_decision_meta.get('status') or '未知'}。"
        )
    show_df(
        "共识重排发布决策汇总",
        "上传发布决策 CSV 后的一行审核结果；只有该汇总允许时，重排才可发布。",
        consensus_rerank_release_decision_summary_df,
    )
    show_df(
        "共识重排发布应用计划",
        "获批的人工排序顺序；仅在发布审核通过且当前应用模拟干净时生成。",
        consensus_rerank_release_apply_plan_df,
    )
    show_markdown(
        "共识重排发布应用报告",
        "获批人工排序的执行工作表，包含门控状态、哈希、排序和应用前检查。",
        consensus_rerank_release_apply_report_markdown,
    )
    show_df(
        "共识重排发布执行模板",
        "操作员回执模板，用于记录每个获批人工排名是否实际应用。",
        consensus_rerank_release_execution_template_df,
    )
    if uploaded_consensus_rerank_release_execution_receipt is not None and consensus_rerank_release_execution_receipt_df.empty:
        st.warning(
            f"共识重排发布执行回执上传不可用：{consensus_rerank_release_execution_receipt_meta.get('status') or '未知'}。"
        )
    show_df(
        "共识重排发布执行汇总",
        "一行操作回执结果；只有每一行都按批准计划精确应用时，执行才算完成。",
        consensus_rerank_release_execution_summary_df,
    )
    show_markdown(
        "共识重排发布执行报告",
        "操作回执报告，包含执行状态、回执哈希、操作员、逐行结果和归档检查。",
        consensus_rerank_release_execution_report_markdown,
    )
    show_markdown(
        "共识重排发布关闭证书",
        "独立关闭证书，关联应用计划、发布审核、执行回执、执行报告和最终关闭状态。",
        consensus_rerank_release_closure_certificate_markdown,
    )
    show_df(
        "共识重排发布关闭就绪汇总",
        "一行发布关闭门控，综合闭环台账完整性和交接 ZIP 校验。",
        consensus_rerank_release_closure_summary_df,
    )
    show_df(
        "共识重排发布关闭阻断队列",
        "从闭环台账和交接 ZIP 校验中提取的可执行阻断项；发布关闭前需要解决。",
        consensus_rerank_release_closure_blocker_df,
    )
    show_markdown(
        "共识重排发布关闭修复清单",
        "由关闭阻断项生成的人工修复清单；每次修复后需要重新运行关闭就绪检查。",
        consensus_rerank_release_closure_remediation_checklist_markdown,
    )
    show_df(
        "共识重排发布关闭外置清单",
        "交接 ZIP 校验后生成的 ZIP 外部关闭产物清单。",
        consensus_rerank_release_closure_detached_manifest_df,
    )
    show_df(
        "共识重排发布关闭台账",
        "机器可读的关闭证据台账，包含状态、行数、字节数、SHA-256 和关闭检查结果。",
        consensus_rerank_release_closure_ledger_df,
    )
    show_df(
        "共识重排发布执行校验",
        "逐行校验执行回执中的排名匹配、操作员、时间戳、模板项和获批应用计划哈希。",
        consensus_rerank_release_execution_validation_df,
    )
    show_df(
        "共识重排发布执行回执",
        "从 CSV/TSV 解析出的操作员上传执行回执标准化行。",
        consensus_rerank_release_execution_receipt_df,
    )
    show_df(
        "共识重排发布决策校验",
        "逐行校验审核人、来源证据、锚点残基、阻断项清理、模板匹配和护栏权限。",
        consensus_rerank_release_decision_validation_df,
    )
    show_df(
        "共识重排发布决策上传结果",
        "从 CSV/TSV 解析出的审核人上传发布决策标准化行。",
        consensus_rerank_release_decision_df,
    )
    if not consensus_rerank_action_queue_df.empty:
        show_markdown(
            "共识重排行动检查清单",
            "用于验证重排阻断项、锚点残基和应用就绪性的 Markdown 交接清单。",
            consensus_rerank_action_checklist_markdown,
        )


def _render_ai_review_panels() -> None:
    has_content = (
        (not ai_review_decision_df.empty)
        or (not ai_review_round_summary_df.empty)
        or (not ai_review_ranking_delta_df.empty)
        or bool(ai_review_round_report_markdown)
        or (not ai_review_artifact_manifest_df.empty)
        or bool(ai_review_bundle_readme_markdown)
        or bool(ai_review_artifact_bundle_zip)
        or (not ai_review_bundle_verification_df.empty)
        or (not ai_review_bundle_verification_summary_df.empty)
        or bool(ai_review_bundle_certificate_markdown)
        or (not ai_review_decision_validation_df.empty)
        or (not ai_review_decision_outcome_df.empty)
        or (not ai_evidence_review_queue_df.empty)
        or (not ai_ranking_impact_df.empty)
        or (not ai_followup_plan_df.empty)
    )
    if not has_content:
        return

    st.subheader("AI 证据复核")
    st.caption("这些面板用于人工复核 AI 提取的关键残基证据。未通过来源、片段和结构校验的 AI 证据不会直接影响排名。")
    if not ai_review_decision_df.empty:
        with st.expander("AI 复核决策应用结果", expanded=False):
            st.caption("人工决策会被保守应用：接受项必须有已验证来源和片段，结构冲突仍保持阻断。")
            st.dataframe(ai_review_decision_df, use_container_width=True, hide_index=True)
    if not ai_review_round_summary_df.empty:
        with st.expander("AI 复核轮次汇总", expanded=False):
            st.caption("本次人工复核上传的一行汇总：是否被阻断、是否仍需复核，或是否已安全应用。")
            st.dataframe(ai_review_round_summary_df, use_container_width=True, hide_index=True)
    if not ai_review_ranking_delta_df.empty:
        with st.expander("AI 复核排名变化", expanded=False):
            st.caption("比较人工复核上传前后，允许进入排名的 AI 残基发生了哪些变化。")
            st.dataframe(ai_review_ranking_delta_df, use_container_width=True, hide_index=True)
    if ai_review_round_report_markdown:
        with st.expander("AI 复核轮次报告", expanded=False):
            st.markdown(ai_review_round_report_markdown)
    if not ai_review_artifact_manifest_df.empty:
        with st.expander("AI 复核产物清单", expanded=False):
            st.caption("生成的 AI 复核产物索引，包含文件名、行数、状态、用途和推荐使用方式。")
            st.dataframe(ai_review_artifact_manifest_df, use_container_width=True, hide_index=True)
    if ai_review_bundle_readme_markdown:
        with st.expander("AI 复核包 README", expanded=False):
            st.markdown(ai_review_bundle_readme_markdown)
    if ai_review_artifact_bundle_zip:
        st.caption("AI 复核产物包已生成，可在“导出”页签中一键导出。")
    if not ai_review_bundle_verification_df.empty:
        with st.expander("AI 复核包逐文件校验", expanded=False):
            st.caption("根据清单中的字节数和 SHA-256 对 ZIP 进行自动自检。")
            st.dataframe(ai_review_bundle_verification_df, use_container_width=True, hide_index=True)
    if not ai_review_bundle_verification_summary_df.empty:
        with st.expander("AI 复核包校验汇总", expanded=False):
            st.caption("由 ZIP 校验结果生成的一行完整性状态。")
            st.dataframe(ai_review_bundle_verification_summary_df, use_container_width=True, hide_index=True)
    if ai_review_bundle_certificate_markdown:
        with st.expander("AI 复核包交接证书", expanded=False):
            st.markdown(ai_review_bundle_certificate_markdown)
    if not ai_review_decision_validation_df.empty:
        with st.expander("AI 复核决策校验", expanded=False):
            st.caption("上传决策应用前检查重复、冲突、未匹配或来源不足的人工决策；冲突重复行不会应用。")
            st.dataframe(ai_review_decision_validation_df, use_container_width=True, hide_index=True)
    if not ai_review_decision_outcome_df.empty:
        with st.expander("AI 复核决策结果", expanded=False):
            st.caption("逐行反馈上传决策的应用结果，包括接受、拒绝、阻断、缺少来源和未匹配。")
            st.dataframe(ai_review_decision_outcome_df, use_container_width=True, hide_index=True)
    if not ai_evidence_review_queue_df.empty:
        with st.expander("AI 证据复核队列", expanded=False):
            st.caption("这些 AI 残基暂时不能安全提升排名置信度，需要先处理队列中的修复项。")
            st.dataframe(ai_evidence_review_queue_df, use_container_width=True, hide_index=True)
    if not ai_ranking_impact_df.empty:
        with st.expander("AI 排名影响汇总", expanded=False):
            st.caption("区分 AI 证据收集和真实排名影响，避免被排除的 AI 行悄悄影响 Top 口袋解释。")
            st.dataframe(ai_ranking_impact_df, use_container_width=True, hide_index=True)
    if not ai_followup_plan_df.empty:
        with st.expander("AI 后续取证计划", expanded=False):
            st.caption("用于收集下一轮文献或数据库证据；生成的提示词预期输入检索到的来源文本，而不是无来源模型记忆。")
            st.dataframe(ai_followup_plan_df, use_container_width=True, hide_index=True)

tab_auto, tab_overview, tab_annotations, tab_overlap, tab_export = st.tabs(["自动识别", "总览", "界面注释", "交集分析", "导出"])

with tab_auto:
    st.subheader("自动口袋识别")
    st.caption("无需上传 Pocket CSV，系统会基于 PDB 坐标自动筛入口袋候选；若结构中含有 HETATM，则会优先尝试配体邻域识别。")
    st.caption(f"当前自动口袋结果{'已' if effective_pocket_mode in {'auto', 'combined'} else '未'}接入界面主分析。")
    auto_metric_cols = st.columns(4)
    auto_metric_cols[0].metric("自动口袋数", len(auto_pocket_summary) if not auto_pocket_summary.empty else 0)
    auto_metric_cols[1].metric("候选残基数", len(auto_pocket_df) if not auto_pocket_df.empty else 0)
    auto_metric_cols[2].metric("热点重叠", int(auto_pocket_df["is_hotspot"].sum()) if not auto_pocket_df.empty and "is_hotspot" in auto_pocket_df.columns else 0)
    auto_metric_cols[3].metric("识别策略", "配体优先" if auto_detection_mode == "auto" else "纯几何")

    auto_overlay_hotspots = st.checkbox("叠加热点残基（仅当前口袋）", value=True, key="pocket_interface_auto_overlay_hotspots")
    auto_view_mode = st.radio(
        "自动口袋展示模式",
        ["surface", "cartoon", "sticks"],
        index=0,
        format_func=lambda x: {"surface": "表面", "cartoon": "卡通", "sticks": "球棍"}[x],
        key="pocket_interface_auto_view_mode",
    )
    auto_show_backbone = st.checkbox("显示主链", value=True, disabled=auto_view_mode != "cartoon", key="pocket_interface_auto_show_backbone")
    auto_surface_opacity = st.slider(
        "表面透明度",
        0.0,
        1.0,
        SETTINGS.default_opacity,
        0.05,
        disabled=auto_view_mode != "surface",
        key="pocket_interface_auto_surface_opacity",
    )

    selected_auto_pocket_id = None
    selected_auto_rows = pd.DataFrame()
    if not auto_pocket_summary.empty:
        auto_pocket_ids = auto_pocket_summary["pocket_id"].astype(str).tolist()
        selected_auto_pocket_id = st.selectbox("查看自动口袋", auto_pocket_ids, index=0, key="pocket_interface_selected_auto_pocket")
        if not auto_pocket_df.empty:
            selected_auto_rows = auto_pocket_df[auto_pocket_df["pocket_id"].astype(str) == selected_auto_pocket_id].copy()
            if selected_auto_rows.empty and auto_pocket_ids:
                selected_auto_pocket_id = auto_pocket_ids[0]
                selected_auto_rows = auto_pocket_df[auto_pocket_df["pocket_id"].astype(str) == selected_auto_pocket_id].copy()
        if not selected_auto_rows.empty:
            selected_auto_rows = add_pocket_residue_layers(selected_auto_rows)

    view_col, detail_col = st.columns([2.3, 1.0])

    with view_col:
        selected_pocket_residues = []
        selected_focus_row = None
        if selected_auto_pocket_id and not selected_auto_rows.empty:
            selected_pocket_residues = [(row.chain, int(row.resid)) for row in selected_auto_rows.itertuples(index=False)]
            selected_focus_row = selected_auto_rows.iloc[0]

        render_atom_df = atom_df
        if not render_atom_df.empty and "record_type" in render_atom_df.columns:
            protein_render_atom_df = render_atom_df[render_atom_df["record_type"].astype(str).str.upper() == "ATOM"].copy()
            if not protein_render_atom_df.empty:
                render_atom_df = protein_render_atom_df

        hotspot_residues_for_view = hotspot_residues if (auto_overlay_hotspots and selected_pocket_residues) else []
        auto_view_table = build_auto_pocket_display_table(
            render_atom_df,
            selected_pocket_residues,
            hotspot_residues=hotspot_residues_for_view,
            pocket_id=selected_auto_pocket_id,
            limit_hotspots_to_pocket=True,
            pocket_residue_layers=selected_auto_rows,
        )
        if auto_view_table.empty:
            st.info("当前没有可渲染的自动口袋结构。")
        else:
            focus_chain = None if auto_view_mode == "surface" else (getattr(selected_focus_row, "chain", None) if selected_focus_row is not None else None)
            focus_resid = None if auto_view_mode == "surface" else (int(getattr(selected_focus_row, "resid", 0)) if selected_focus_row is not None else None)
            viewer = build_view(
                pdb_text=pdb_text,
                energy_table=auto_view_table,
                threshold=0.0,
                display_mode=auto_view_mode,
                show_backbone=auto_show_backbone,
                opacity=auto_surface_opacity if auto_view_mode == "surface" else 0.85,
                selected_chain=focus_chain,
                selected_resid=focus_resid,
                color_mode="按口袋识别",
                surface_single_color=False,
                surface_uniform_color=SETTINGS.neutral_color,
                viewer_width=max(680, SETTINGS.viewer_width - 120),
                viewer_height=max(520, SETTINGS.viewer_height - 120),
            )
            st.components.v1.html(viewer._make_html(), height=max(520, SETTINGS.viewer_height - 120) + 20, scrolling=False)
            st.caption("仅当前选中口袋显色：蓝色为当前口袋，红色为当前口袋内热点，灰色为背景。")

    with detail_col:
        if auto_pocket_summary.empty:
            st.info("当前没有识别出自动口袋。你可以试着增大聚类距离、提高候选残基比例，或者上传含配体的结构。")
        else:
            selected_summary = auto_pocket_summary[auto_pocket_summary["pocket_id"].astype(str) == selected_auto_pocket_id].copy()
            st.subheader("当前口袋摘要")
            st.dataframe(selected_summary, use_container_width=True, hide_index=True)

            if not selected_auto_rows.empty:
                st.subheader("当前口袋残基")
                residue_columns = [
                    column
                    for column in [
                        "residue_label",
                        "pocket_layer",
                        "pocket_layer_score",
                        "pocket_layer_reason",
                        "residue_score",
                        "score",
                        "detection_route",
                        "consensus_methods",
                        "method_vote_count",
                        "smart_rank_score",
                        "smart_rank_label",
                        "smart_evidence_anchor_support",
                        "smart_evidence_anchor_risk",
                        "external_direct_anchor",
                        "evidence_route_anchor",
                        "evidence_anchor_distance",
                        "evidence_anchor_proximity",
                        "evidence_anchor_residue",
                        "external_structure_verified",
                        "external_mapping_quality",
                        "external_direct_sources",
                        "external_evidence_types",
                        "external_evidence_notes",
                        "conservation_support",
                        "conservation_confidence",
                        "contact_count",
                        "center_distance",
                        "ligand_contact_count",
                        "is_hotspot",
                    ]
                    if column in selected_auto_rows.columns
                ]
                if residue_columns:
                    st.dataframe(selected_auto_rows[residue_columns], use_container_width=True, hide_index=True)

                st.download_button(
                    "导出当前口袋残基 CSV",
                    data=_to_csv_bytes(selected_auto_rows),
                    file_name=f"{selected_auto_pocket_id}_residues.csv",
                    mime="text/csv",
                )

            st.download_button(
                "导出自动口袋汇总 CSV",
                data=_to_csv_bytes(auto_pocket_summary),
                file_name="auto_pocket_summary.csv",
                mime="text/csv",
            )
            st.download_button(
                "导出自动口袋明细 CSV",
                data=_to_csv_bytes(auto_pocket_df),
                file_name="auto_pocket_candidates.csv",
                mime="text/csv",
            )

        with st.expander("自动检测诊断", expanded=False):
            methods_used_text = str(auto_detection_summary.get("auto_detection_methods_used") or "-")
            status_text = str(auto_detection_summary.get("auto_detection_status_summary") or "-")
            st.caption(f"识别方法：{methods_used_text}")
            st.caption(f"运行状态：{status_text}")
            p2rank_status_text = str(auto_detection_summary.get("auto_detection_p2rank_status") or "").strip()
            if p2rank_status_text:
                st.caption(
                    f"P2Rank：{p2rank_status_text} / 预测口袋 {int(auto_detection_summary.get('auto_detection_p2rank_prediction_rows', 0) or 0)} / "
                    f"残基 {int(auto_detection_summary.get('auto_detection_p2rank_residue_rows', 0) or 0)}"
                )
            if int(auto_detection_summary.get("auto_detection_external_rows", 0) or 0) > 0:
                external_text = str(auto_detection_summary.get("auto_detection_external_sources") or "external")
                st.caption(
                    f"外部证据：{int(auto_detection_summary.get('auto_detection_external_rows', 0) or 0)} 行 / "
                    f"精确映射 {int(auto_detection_summary.get('auto_detection_external_exact_rows', 0) or 0)} / "
                    f"弱映射 {int(auto_detection_summary.get('auto_detection_external_weak_rows', 0) or 0)} "
                    f"({external_text})"
                )
            if not external_site_df.empty:
                source_detail_columns = [
                    column
                    for column in [
                        "chain",
                        "resid",
                        "evidence_source",
                        "evidence_type",
                        "evidence_score",
                        "mapping_level",
                        "mapping_confidence",
                        "article_title",
                        "pmid",
                        "pmcid",
                        "doi",
                        "evidence_snippet",
                        "sentence_index",
                        "extraction_pattern",
                        "requires_manual_review",
                    ]
                    if column in external_site_df.columns
                ]
                if source_detail_columns:
                    st.caption("外部证据来源明细：展示口袋重排使用的结构化引用、片段和人工复核标记。")
                    st.dataframe(
                        external_site_df[source_detail_columns].head(80),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.download_button(
                    "导出外部证据明细 CSV",
                    data=_to_csv_bytes(external_site_df),
                    file_name="external_residue_evidence_details.csv",
                    mime="text/csv",
                )
            if p2rank_ab_enabled:
                st.caption("P2Rank A/B：基线为关闭 P2Rank 后重新自动识别；排名变化值为正表示 P2Rank 提升了该口袋排名。")
                if p2rank_ab_df.empty:
                    st.caption("P2Rank A/B：没有可比较的口袋排名行。")
                else:
                    st.dataframe(p2rank_ab_df, use_container_width=True, hide_index=True)
            route_status_text = _localize_status_text(auto_detection_summary.get("auto_detection_external_route_status"), default="")
            if route_status_text:
                st.caption(
                    f"证据路径：{route_status_text} / "
                    f"支持度>={float(auto_detection_summary.get('auto_detection_external_route_min_support') or 0.0):.2f} / "
                    f"置信度>={float(auto_detection_summary.get('auto_detection_external_route_min_confidence') or 0.0):.2f} / "
                    f"映射质量>={float(auto_detection_summary.get('auto_detection_external_route_min_mapping_quality') or 0.0):.2f}"
                )
            if literature_ab_enabled:
                st.caption("文献 A/B：基线为移除文献证据后重新自动识别；排名变化值为正表示文献证据提升了该口袋排名。")
                if literature_ab_df.empty:
                    st.caption("文献 A/B：没有可比较的口袋排名行。")
                else:
                    st.dataframe(literature_ab_df, use_container_width=True, hide_index=True)
            if evidence_route_ab_enabled:
                st.caption("证据路径 A/B：基线为保留相同外部证据但关闭证据路径；排名变化值为正表示证据路径提升了该口袋排名。")
                if evidence_route_ab_df.empty:
                    st.caption("证据路径 A/B：没有可比较的口袋排名行。")
                else:
                    st.dataframe(evidence_route_ab_df, use_container_width=True, hide_index=True)
            if int(auto_detection_summary.get("auto_detection_conservation_rows", 0) or 0) > 0:
                conservation_text = str(auto_detection_summary.get("auto_detection_conservation_sources") or "conservation")
                st.caption(
                    f"保守性：{int(auto_detection_summary.get('auto_detection_conservation_rows', 0) or 0)} 行 "
                    f"({conservation_text}，仅用于重排)"
                )
            if conservation_ab_enabled:
                st.caption("保守性 A/B：基线为保守性列置零；排名变化值为正表示该口袋排名上升。")
                if conservation_ab_df.empty:
                    st.caption("保守性 A/B：没有可比较的口袋排名行。")
                else:
                    st.dataframe(conservation_ab_df, use_container_width=True, hide_index=True)
            if not auto_detection_diag_df.empty:
                st.dataframe(auto_detection_diag_df, use_container_width=True, hide_index=True)
            elif auto_detection_meta:
                st.json(_localize_json_for_display(auto_detection_meta))

_render_pocket_decision_panel(
    pocket_decision_df,
    pocket_reliability_df,
    pocket_triage_df,
    manual_key_residue_template_df,
    manual_key_residue_followup_df,
    manual_key_residue_followup_summary_df,
    manual_key_residue_followup_checklist_markdown,
)
_render_evidence_context_panels()
_render_consensus_rerank_review_panels()
_render_ai_review_panels()
_render_benchmark_review_panels()

with tab_overview:
    st.subheader("主分析总览")
    overview_cols = st.columns(2)
    with overview_cols[0]:
        st.markdown(f"**当前口袋来源**：{POCKET_SOURCE_LABELS.get(effective_pocket_mode, effective_pocket_mode)}")
        if effective_pocket_summary.empty:
            st.info("当前口袋来源下没有可用口袋。")
        else:
            st.dataframe(effective_pocket_summary, use_container_width=True, hide_index=True)
    with overview_cols[1]:
        st.markdown(f"**当前界面来源**：{ANNOTATION_SOURCE_LABELS.get(effective_annotation_mode, effective_annotation_mode)}")
        if interface_summary.empty:
            st.info("当前界面来源下没有可用注释。")
        else:
            st.dataframe(interface_summary, use_container_width=True, hide_index=True)

    if not interface_summary.empty:
        chart_df = interface_summary.set_index("region_type")[
            [column for column in ["residue_count", "pocket_count", "hotspot_count", "overlap_count"] if column in interface_summary.columns]
        ]
        chart_df = chart_df.rename(
            index=lambda value: _localize_status_text(value),
            columns=lambda column: _localize_dataframe_column(column),
        )
        st.subheader("区域类型计数对比")
        st.bar_chart(chart_df)

    st.subheader("联合推荐")
    if joint_candidate_df.empty:
        st.info("当前还没有可用的联合推荐结果。至少需要有效口袋，界面和热点证据会在存在时自动叠加。")
    else:
        recommendation_columns = [
            "recommendation_rank",
            "pocket_id",
            "recommendation_label",
            "recommendation_score",
            "recommendation_action",
            "evidence_quality_label",
            "evidence_anchor_support",
            "evidence_anchor_risk",
            "smart_rank_label",
            "smart_rank_score",
            "hotspot_overlap_count",
            "interface_overlap_count",
            "triple_overlap_count",
            "external_overlap_count",
            "external_exact_overlap_count",
            "external_weak_overlap_count",
            "external_overlap_ratio",
            "external_weighted_overlap_ratio",
            "external_mapping_confidence",
            "external_structure_verified_count",
            "external_structure_verified_ratio",
            "external_evidence_types",
            "method_vote_count",
            "recommendation_reason",
        ]
        recommendation_columns = [column for column in recommendation_columns if column in joint_candidate_df.columns]
        st.dataframe(joint_candidate_df[recommendation_columns], use_container_width=True, hide_index=True)

    with st.expander("数据源详情", expanded=False):
        source_cols = st.columns(2)
        with source_cols[0]:
            st.markdown("**上传 Pocket 汇总**")
            if uploaded_pocket_summary.empty:
                st.caption("无上传 Pocket 数据。")
            else:
                st.dataframe(uploaded_pocket_summary, use_container_width=True, hide_index=True)
            st.markdown("**自动 Pocket 汇总**")
            if auto_pocket_summary.empty:
                st.caption("无自动口袋结果。")
            else:
                st.dataframe(auto_pocket_summary, use_container_width=True, hide_index=True)
        with source_cols[1]:
            st.markdown("**上传界面汇总**")
            if uploaded_annotation_summary.empty:
                st.caption("无上传界面注释。")
            else:
                st.dataframe(uploaded_annotation_summary, use_container_width=True, hide_index=True)
            st.markdown("**结构推断界面汇总**")
            if inferred_annotation_summary.empty:
                st.caption("当前未生成结构推断界面。")
            else:
                st.dataframe(inferred_annotation_summary, use_container_width=True, hide_index=True)

with tab_annotations:
    st.subheader("界面注释明细")
    if enriched_annotations.empty:
        st.info("当前没有可显示的界面注释。你可以上传界面 CSV，或使用结构估算生成推断界面。")
    else:
        filter_cols = st.columns([1.1, 1.1, 1.1, 1.2, 1.5])
        only_overlap = filter_cols[0].checkbox("仅三重交集", value=False)
        only_pocket = filter_cols[1].checkbox("仅口袋命中", value=False)
        only_hotspot = filter_cols[2].checkbox("仅热点命中", value=False)
        only_interface_core = filter_cols[3].checkbox("仅核心界面", value=False)
        keyword = filter_cols[4].text_input("按残基/注释搜索", value="")

        region_options = sorted(
            value for value in enriched_annotations.get("region_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if value
        )
        selected_regions = st.multiselect("区域类型过滤", region_options, default=region_options)

        view = enriched_annotations.copy()
        if selected_regions:
            view = view[view["region_type"].astype(str).isin(selected_regions)]
        if only_overlap and "is_overlap" in view.columns:
            view = view[view["is_overlap"]]
        if only_pocket and "is_pocket" in view.columns:
            view = view[view["is_pocket"]]
        if only_hotspot and "is_hotspot" in view.columns:
            view = view[view["is_hotspot"]]
        if only_interface_core and "region_type" in view.columns:
            view = view[view["region_type"].astype(str).str.contains("core", case=False, na=False)]
        if keyword.strip():
            pattern = keyword.strip().lower()
            mask = pd.Series(False, index=view.index)
            for column in [col for col in ["residue_label", "annotation", "region_type"] if col in view.columns]:
                mask = mask | view[column].astype(str).str.lower().str.contains(pattern, na=False)
            view = view[mask]

        display_view = view.copy()
        if "inference_basis" in display_view.columns:
            display_view["inference_basis"] = (
                display_view["inference_basis"]
                .astype(str)
                .map(INFERENCE_BASIS_LABELS)
                .fillna(display_view["inference_basis"].astype(str))
            )

        display_columns = [
            "residue_label",
            "region_type",
            "annotation",
            "annotation_source",
            "inference_basis",
            "is_pocket",
            "is_hotspot",
            "is_overlap",
        ]
        display_columns = [column for column in display_columns if column in view.columns]
        st.dataframe(display_view[display_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "导出当前筛选结果 CSV",
            data=_to_csv_bytes(view),
            file_name="filtered_interface_annotations.csv",
            mime="text/csv",
        )

with tab_overlap:
    st.subheader("交集分析")
    overlap_metric_cols = st.columns(4)
    overlap_metric_cols[0].metric("口袋 ∩ 热点", len(pocket_hotspot_df))
    overlap_metric_cols[1].metric("界面 ∩ 热点", len(interface_hotspot_df))
    overlap_metric_cols[2].metric("界面 ∩ 口袋", len(interface_pocket_df))
    overlap_metric_cols[3].metric("界面 ∩ 口袋 ∩ 热点", len(triple_overlap_df))

    if not overlap_summary.empty:
        overlap_chart_df = overlap_summary.copy()
        overlap_chart_df["category"] = overlap_chart_df["category"].map(_localize_status_text)
        overlap_chart_df = overlap_chart_df.rename(columns={"category": "类别", "count": "数量"})
        st.bar_chart(overlap_chart_df.set_index("类别"))

    overlap_cols = st.columns(2)
    with overlap_cols[0]:
        st.markdown("**口袋与热点重叠残基**")
        if pocket_hotspot_df.empty:
            st.info("当前没有发现口袋与热点的重叠残基。")
        else:
            st.dataframe(
                pocket_hotspot_df[
                    [col for col in ["pocket_id", "residue_label", "score", "consensus_methods", "detection_route"] if col in pocket_hotspot_df.columns]
                ],
                use_container_width=True,
                hide_index=True,
            )
    with overlap_cols[1]:
        st.markdown("**界面、口袋、热点三重交集**")
        if triple_overlap_df.empty:
            st.info("当前没有发现三重交集残基。")
        else:
            st.dataframe(
                triple_overlap_df[[col for col in ["residue_label", "region_type", "annotation", "annotation_source"] if col in triple_overlap_df.columns]],
                use_container_width=True,
                hide_index=True,
            )

    if not hotspot_df.empty:
        st.subheader("热点列表")
        st.dataframe(hotspot_df[[col for col in ["label", "delta_total", "hotspot_rank"] if col in hotspot_df.columns]], use_container_width=True, hide_index=True)

with tab_export:
    st.subheader("导出")
    export_cols = st.columns(2)
    with export_cols[0]:
        if not effective_pocket_summary.empty:
            st.download_button(
                "导出当前口袋汇总 CSV",
                data=_to_csv_bytes(effective_pocket_summary),
                file_name="effective_pocket_summary.csv",
                mime="text/csv",
            )
        if not effective_pocket_df.empty:
            st.download_button(
                "导出当前口袋明细 CSV",
                data=_to_csv_bytes(effective_pocket_df),
                file_name="effective_pocket_candidates.csv",
                mime="text/csv",
            )
        if not enriched_annotations.empty:
            st.download_button(
                "导出当前界面注释 CSV",
                data=_to_csv_bytes(enriched_annotations),
                file_name="effective_interface_annotations.csv",
                mime="text/csv",
            )
        if not interface_summary.empty:
            st.download_button(
                "导出当前界面汇总 CSV",
                data=_to_csv_bytes(interface_summary),
                file_name="effective_interface_summary.csv",
                mime="text/csv",
            )
        if not joint_candidate_df.empty:
            st.download_button(
                "导出联合推荐 CSV",
                data=_to_csv_bytes(joint_candidate_df),
                file_name="joint_candidate_recommendations.csv",
                mime="text/csv",
            )
        if not pocket_decision_df.empty:
            st.download_button(
                "导出活性位点决策 CSV",
                data=_to_csv_bytes(pocket_decision_df),
                file_name="active_site_decision.csv",
                mime="text/csv",
            )
        if not pocket_reliability_df.empty:
            st.download_button(
                "导出可靠性检查表 CSV",
                data=_to_csv_bytes(pocket_reliability_df),
                file_name="pocket_reliability_checklist.csv",
                mime="text/csv",
            )
        if not pocket_triage_df.empty:
            st.download_button(
                "导出精度处理建议 CSV",
                data=_to_csv_bytes(pocket_triage_df),
                file_name="pocket_precision_triage.csv",
                mime="text/csv",
            )
        if not manual_key_residue_followup_df.empty:
            if not manual_key_residue_followup_summary_df.empty:
                st.download_button(
                    "导出人工关键残基补证闭环总览 CSV",
                    data=_to_csv_bytes(manual_key_residue_followup_summary_df),
                    file_name="manual_key_residue_followup_summary.csv",
                    mime="text/csv",
                )
            if manual_key_residue_followup_checklist_markdown:
                st.download_button(
                    "导出人工关键残基补证复跑检查清单 MD",
                    data=manual_key_residue_followup_checklist_markdown.encode("utf-8"),
                    file_name="manual_key_residue_followup_checklist.md",
                    mime="text/markdown",
                    key="export_manual_key_residue_followup_checklist",
                )
            st.download_button(
                "导出人工关键残基补证任务 CSV",
                data=_to_csv_bytes(manual_key_residue_followup_df),
                file_name="manual_key_residue_followup_tasks.csv",
                mime="text/csv",
            )
        if not manual_key_residue_df.empty:
            st.download_button(
                "导出人工关键残基证据 CSV",
                data=_to_csv_bytes(manual_key_residue_df),
                file_name="manual_key_residue_evidence.csv",
                mime="text/csv",
            )
        if not ai_evidence_df.empty:
            st.download_button(
                "导出 AI 残基证据 CSV",
                data=_to_csv_bytes(ai_evidence_df),
                file_name="ai_residue_evidence.csv",
                mime="text/csv",
            )
        if not ai_evidence_audit_df.empty:
            st.download_button(
                "导出 AI 证据审计 CSV",
                data=_to_csv_bytes(ai_evidence_audit_df),
                file_name="ai_residue_evidence_audit.csv",
                mime="text/csv",
            )
        if not residue_evidence_consensus_df.empty:
            st.download_button(
                "导出残基证据共识 CSV",
                data=_to_csv_bytes(residue_evidence_consensus_df),
                file_name="residue_evidence_consensus.csv",
                mime="text/csv",
            )
        if not pocket_consensus_coverage_df.empty:
            st.download_button(
                "导出口袋共识覆盖 CSV",
                data=_to_csv_bytes(pocket_consensus_coverage_df),
                file_name="pocket_consensus_coverage.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_df.empty:
            st.download_button(
                "Export benchmark reference candidate CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_df),
                file_name="pocket_benchmark_reference_candidate.csv",
                mime="text/csv",
            )
        if not benchmark_reference_import_summary_df.empty:
            st.download_button(
                "Export benchmark reference import summary CSV",
                data=_to_csv_bytes(benchmark_reference_import_summary_df),
                file_name="pocket_benchmark_reference_import_summary.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_queue_df.empty:
            st.download_button(
                "Export benchmark reference candidate review queue CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_queue_df),
                file_name="pocket_benchmark_reference_candidate_review_queue.csv",
                mime="text/csv",
            )
        if benchmark_reference_candidate_review_checklist_markdown:
            st.download_button(
                "Export benchmark reference candidate review checklist MD",
                data=benchmark_reference_candidate_review_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_candidate_review_checklist.md",
                mime="text/markdown",
            )
        if not benchmark_reference_candidate_review_decision_template_df.empty:
            st.download_button(
                "Export benchmark reference candidate review decision template CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_decision_template_df),
                file_name="pocket_benchmark_reference_candidate_review_decision_template.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_decision_df.empty:
            st.download_button(
                "Export benchmark reference candidate review decisions normalized CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_decision_df),
                file_name="pocket_benchmark_reference_candidate_review_decisions_normalized.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_decision_validation_df.empty:
            st.download_button(
                "Export benchmark reference candidate review decision validation CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_decision_validation_df),
                file_name="pocket_benchmark_reference_candidate_review_decision_validation.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_outcome_df.empty:
            st.download_button(
                "Export benchmark reference candidate review outcomes CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_outcome_df),
                file_name="pocket_benchmark_reference_candidate_review_outcomes.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_accepted_df.empty:
            st.download_button(
                "Export accepted benchmark reference candidates CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_accepted_df),
                file_name="pocket_benchmark_reference_candidate_accepted.csv",
                mime="text/csv",
            )
        if not benchmark_reference_df.empty:
            st.download_button(
                "Export benchmark reference CSV",
                data=_to_csv_bytes(benchmark_reference_df),
                file_name="pocket_benchmark_reference.csv",
                mime="text/csv",
            )
        if not benchmark_reference_source_audit_df.empty:
            if not benchmark_reference_source_audit_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_case_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_template_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision template CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_template_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_template.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decisions normalized CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decisions_normalized.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_validation_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision validation CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_validation_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_validation.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_outcome_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision outcome summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_outcome_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_outcome_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision closure queue CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_closure_queue_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_closure_queue.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision readiness impact summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision readiness impact CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_readiness_impact_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_readiness_impact.csv",
                    mime="text/csv",
                )
            if benchmark_reference_source_audit_case_decision_closure_checklist_markdown:
                st.download_button(
                    "Export benchmark reference source audit case decision closure checklist MD",
                    data=benchmark_reference_source_audit_case_decision_closure_checklist_markdown.encode("utf-8"),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_closure_checklist.md",
                    mime="text/markdown",
                )
            if not benchmark_reference_source_audit_case_decision_outcome_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision outcomes CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_outcome_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_outcomes.csv",
                    mime="text/csv",
                )
            if benchmark_reference_source_audit_case_checklist_markdown:
                st.download_button(
                    "Export benchmark reference source audit case checklist MD",
                    data=benchmark_reference_source_audit_case_checklist_markdown.encode("utf-8"),
                    file_name="pocket_benchmark_reference_source_audit_case_checklist.md",
                    mime="text/markdown",
                )
            if not benchmark_reference_source_audit_action_queue_df.empty:
                st.download_button(
                    "Export benchmark reference source audit action queue CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_action_queue_df),
                    file_name="pocket_benchmark_reference_source_audit_action_queue.csv",
                    mime="text/csv",
                )
            if benchmark_reference_source_audit_checklist_markdown:
                st.download_button(
                    "Export benchmark reference source audit checklist MD",
                    data=benchmark_reference_source_audit_checklist_markdown.encode("utf-8"),
                    file_name="pocket_benchmark_reference_source_audit_checklist.md",
                    mime="text/markdown",
                )
            st.download_button(
                "Export benchmark reference source audit CSV",
                data=_to_csv_bytes(benchmark_reference_source_audit_df),
                file_name="pocket_benchmark_reference_source_audit.csv",
                mime="text/csv",
            )
        st.download_button(
            "Export benchmark reference template CSV",
            data=_to_csv_bytes(benchmark_reference_template_df),
            file_name="pocket_benchmark_reference_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Export benchmark reference template notes",
            data=benchmark_reference_template_markdown.encode("utf-8"),
            file_name="pocket_benchmark_reference_template.md",
            mime="text/markdown",
        )
        if not pocket_benchmark_reference_quality_issue_df.empty:
            st.download_button(
                "Export benchmark reference quality issues CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_quality_issue_df),
                file_name="pocket_benchmark_reference_quality_issues.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_quality_summary_df.empty:
            st.download_button(
                "Export benchmark reference quality summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_quality_summary_df),
                file_name="pocket_benchmark_reference_quality_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_quality_checklist_markdown:
            st.download_button(
                "Export benchmark reference curation checklist",
                data=pocket_benchmark_reference_quality_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_quality_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_reference_structure_validation_df.empty:
            st.download_button(
                "Export benchmark reference structure validation CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_structure_validation_df),
                file_name="pocket_benchmark_reference_structure_validation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_structure_validation_summary_df.empty:
            st.download_button(
                "Export benchmark reference structure validation summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_structure_validation_summary_df),
                file_name="pocket_benchmark_reference_structure_validation_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_structure_validation_checklist_markdown:
            st.download_button(
                "Export benchmark reference structure validation checklist",
                data=pocket_benchmark_reference_structure_validation_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_structure_validation_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_reference_readiness_summary_df.empty:
            st.download_button(
                "Export benchmark reference readiness summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_readiness_summary_df),
                file_name="pocket_benchmark_reference_readiness_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_readiness_case_summary_df.empty:
            st.download_button(
                "Export benchmark reference readiness case summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_readiness_case_summary_df),
                file_name="pocket_benchmark_reference_readiness_case_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_readiness_queue_df.empty:
            st.download_button(
                "Export benchmark reference readiness queue CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_readiness_queue_df),
                file_name="pocket_benchmark_reference_readiness_queue.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_readiness_checklist_markdown:
            st.download_button(
                "Export benchmark reference readiness checklist",
                data=pocket_benchmark_reference_readiness_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_readiness_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_interpretation_df.empty:
            st.download_button(
                "Export pocket benchmark interpretation CSV",
                data=_to_csv_bytes(pocket_benchmark_interpretation_df),
                file_name="pocket_benchmark_interpretation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_df),
                file_name="pocket_benchmark_case_interpretation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_matrix_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation matrix CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_matrix_df),
                file_name="pocket_benchmark_case_interpretation_matrix.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_matrix_summary_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation matrix summary CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_matrix_summary_df),
                file_name="pocket_benchmark_case_interpretation_matrix_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_matrix_queue_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation matrix queue CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_matrix_queue_df),
                file_name="pocket_benchmark_case_interpretation_matrix_queue.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_dataset_interpretation_df.empty:
            st.download_button(
                "Export pocket benchmark dataset interpretation CSV",
                data=_to_csv_bytes(pocket_benchmark_dataset_interpretation_df),
                file_name="pocket_benchmark_dataset_interpretation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact cases CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact action queue CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact action queue summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact case checklist MD",
                data=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist.md",
                mime="text/markdown",
            )
        if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact report MD",
                data=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact artifact manifest CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_dataset_interpretation_queue_df.empty:
            st.download_button(
                "Export pocket benchmark dataset interpretation queue CSV",
                data=_to_csv_bytes(pocket_benchmark_dataset_interpretation_queue_df),
                file_name="pocket_benchmark_dataset_interpretation_queue.csv",
                mime="text/csv",
            )
        if pocket_benchmark_dataset_interpretation_checklist_markdown:
            st.download_button(
                "Export pocket benchmark dataset interpretation checklist MD",
                data=pocket_benchmark_dataset_interpretation_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_dataset_interpretation_checklist.md",
                mime="text/markdown",
            )
        if pocket_benchmark_dataset_interpretation_report_markdown:
            st.download_button(
                "Export pocket benchmark dataset interpretation report MD",
                data=pocket_benchmark_dataset_interpretation_report_markdown.encode("utf-8"),
                file_name="pocket_benchmark_dataset_interpretation_report.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_summary_df.empty:
            st.download_button(
                "Export pocket benchmark summary CSV",
                data=_to_csv_bytes(pocket_benchmark_summary_df),
                file_name="pocket_benchmark_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_dataset_summary_df.empty:
            st.download_button(
                "Export pocket benchmark dataset summary CSV",
                data=_to_csv_bytes(pocket_benchmark_dataset_summary_df),
                file_name="pocket_benchmark_dataset_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_summary_df.empty:
            st.download_button(
                "Export pocket benchmark case summary CSV",
                data=_to_csv_bytes(pocket_benchmark_case_summary_df),
                file_name="pocket_benchmark_case_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_comparison_df),
                file_name="pocket_benchmark_variant_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_dataset_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant dataset comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_dataset_comparison_df),
                file_name="pocket_benchmark_variant_dataset_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_case_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant case comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_case_comparison_df),
                file_name="pocket_benchmark_variant_case_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_detail_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant residue comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_detail_comparison_df),
                file_name="pocket_benchmark_variant_residue_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_remediation_df.empty:
            st.download_button(
                "Export pocket benchmark remediation queue CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_remediation_df),
                file_name="pocket_benchmark_variant_remediation_queue.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_remediation_summary_df.empty:
            st.download_button(
                "Export pocket benchmark remediation summary CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_remediation_summary_df),
                file_name="pocket_benchmark_variant_remediation_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_variant_remediation_checklist_markdown:
            st.download_button(
                "Export pocket benchmark remediation checklist",
                data=pocket_benchmark_variant_remediation_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_variant_remediation_checklist.md",
                mime="text/markdown",
            )
        if not p2rank_ab_df.empty:
            st.download_button(
                "Export P2Rank A/B comparison CSV",
                data=_to_csv_bytes(p2rank_ab_df),
                file_name="p2rank_ab_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_details_df.empty:
            st.download_button(
                "Export pocket benchmark details CSV",
                data=_to_csv_bytes(pocket_benchmark_details_df),
                file_name="pocket_benchmark_details.csv",
                mime="text/csv",
            )
        if not consensus_rerank_suggestion_df.empty:
            st.download_button(
                "Export consensus rerank suggestions CSV",
                data=_to_csv_bytes(consensus_rerank_suggestion_df),
                file_name="consensus_rerank_suggestions.csv",
                mime="text/csv",
            )
        if not consensus_rerank_preview_df.empty:
            st.download_button(
                "Export consensus rerank preview CSV",
                data=_to_csv_bytes(consensus_rerank_preview_df),
                file_name="consensus_rerank_preview.csv",
                mime="text/csv",
            )
        if not consensus_rerank_policy_gate_df.empty:
            st.download_button(
                "Export consensus rerank policy gate CSV",
                data=_to_csv_bytes(consensus_rerank_policy_gate_df),
                file_name="consensus_rerank_policy_gate.csv",
                mime="text/csv",
            )
        if not consensus_rerank_action_queue_df.empty:
            st.download_button(
                "Export consensus rerank action queue CSV",
                data=_to_csv_bytes(consensus_rerank_action_queue_df),
                file_name="consensus_rerank_action_queue.csv",
                mime="text/csv",
            )
        if not consensus_rerank_apply_simulation_df.empty:
            st.download_button(
                "Export consensus rerank apply simulation CSV",
                data=_to_csv_bytes(consensus_rerank_apply_simulation_df),
                file_name="consensus_rerank_apply_simulation.csv",
                mime="text/csv",
            )
        if not consensus_rerank_simulation_delta_df.empty:
            st.download_button(
                "Export consensus rerank simulation delta CSV",
                data=_to_csv_bytes(consensus_rerank_simulation_delta_df),
                file_name="consensus_rerank_simulation_delta.csv",
                mime="text/csv",
            )
        if not consensus_rerank_precision_scorecard_df.empty:
            st.download_button(
                "Export consensus rerank precision scorecard CSV",
                data=_to_csv_bytes(consensus_rerank_precision_scorecard_df),
                file_name="consensus_rerank_precision_scorecard.csv",
                mime="text/csv",
            )
        if not consensus_rerank_precision_guardrail_df.empty:
            st.download_button(
                "Export consensus rerank precision guardrail CSV",
                data=_to_csv_bytes(consensus_rerank_precision_guardrail_df),
                file_name="consensus_rerank_precision_guardrail.csv",
                mime="text/csv",
            )
        if consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty:
            st.download_button(
                "Export consensus rerank precision guardrail report",
                data=consensus_rerank_precision_guardrail_report_markdown.encode("utf-8"),
                file_name="consensus_rerank_precision_guardrail_report.md",
                mime="text/markdown",
            )
        if not consensus_rerank_guardrail_artifact_manifest_df.empty:
            st.download_button(
                "Export consensus rerank guardrail artifact manifest CSV",
                data=_to_csv_bytes(consensus_rerank_guardrail_artifact_manifest_df),
                file_name="consensus_rerank_guardrail_artifact_manifest.csv",
                mime="text/csv",
            )
        if consensus_rerank_guardrail_handoff_zip:
            st.download_button(
                "Export consensus rerank guardrail handoff ZIP",
                data=consensus_rerank_guardrail_handoff_zip,
                file_name="consensus_rerank_guardrail_handoff.zip",
                mime="application/zip",
            )
        if not consensus_rerank_guardrail_bundle_verification_summary_df.empty:
            st.download_button(
                "Export consensus rerank guardrail bundle verification summary CSV",
                data=_to_csv_bytes(consensus_rerank_guardrail_bundle_verification_summary_df),
                file_name="consensus_rerank_guardrail_bundle_verification_summary.csv",
                mime="text/csv",
            )
        if not consensus_rerank_guardrail_bundle_verification_df.empty:
            st.download_button(
                "Export consensus rerank guardrail bundle verification CSV",
                data=_to_csv_bytes(consensus_rerank_guardrail_bundle_verification_df),
                file_name="consensus_rerank_guardrail_bundle_verification.csv",
                mime="text/csv",
            )
        if consensus_rerank_guardrail_handoff_certificate_markdown:
            st.download_button(
                "Export consensus rerank guardrail handoff certificate",
                data=consensus_rerank_guardrail_handoff_certificate_markdown.encode("utf-8"),
                file_name="consensus_rerank_guardrail_handoff_certificate.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_decision_template_df.empty:
            st.download_button(
                "Export consensus rerank release decision template CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_template_df),
                file_name="consensus_rerank_release_decision_template.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_decision_df.empty:
            st.download_button(
                "Export normalized consensus rerank release decisions CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_df),
                file_name="consensus_rerank_release_decisions_normalized.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_decision_validation_df.empty:
            st.download_button(
                "Export consensus rerank release decision validation CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_validation_df),
                file_name="consensus_rerank_release_decision_validation.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_decision_summary_df.empty:
            st.download_button(
                "Export consensus rerank release decision summary CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_summary_df),
                file_name="consensus_rerank_release_decision_summary.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_apply_plan_df.empty:
            st.download_button(
                "Export consensus rerank release apply plan CSV",
                data=_to_csv_bytes(consensus_rerank_release_apply_plan_df),
                file_name="consensus_rerank_release_apply_plan.csv",
                mime="text/csv",
            )
        if consensus_rerank_release_apply_report_markdown:
            st.download_button(
                "Export consensus rerank release apply report",
                data=consensus_rerank_release_apply_report_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_apply_report.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_execution_template_df.empty:
            st.download_button(
                "Export consensus rerank release execution template CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_template_df),
                file_name="consensus_rerank_release_execution_template.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_execution_receipt_df.empty:
            st.download_button(
                "Export normalized consensus rerank release execution receipt CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_receipt_df),
                file_name="consensus_rerank_release_execution_receipt_normalized.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_execution_validation_df.empty:
            st.download_button(
                "Export consensus rerank release execution validation CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_validation_df),
                file_name="consensus_rerank_release_execution_validation.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_execution_summary_df.empty:
            st.download_button(
                "Export consensus rerank release execution summary CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_summary_df),
                file_name="consensus_rerank_release_execution_summary.csv",
                mime="text/csv",
            )
        if consensus_rerank_release_execution_report_markdown:
            st.download_button(
                "Export consensus rerank release execution report",
                data=consensus_rerank_release_execution_report_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_execution_report.md",
                mime="text/markdown",
            )
        if consensus_rerank_release_closure_certificate_markdown:
            st.download_button(
                "Export consensus rerank release closure certificate",
                data=consensus_rerank_release_closure_certificate_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_closure_certificate.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_closure_ledger_df.empty:
            st.download_button(
                "Export consensus rerank release closure ledger CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_ledger_df),
                file_name="consensus_rerank_release_closure_ledger.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_closure_summary_df.empty:
            st.download_button(
                "Export consensus rerank release closure readiness summary CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_summary_df),
                file_name="consensus_rerank_release_closure_summary.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_closure_blocker_df.empty:
            st.download_button(
                "Export consensus rerank release closure blocker queue CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_blocker_df),
                file_name="consensus_rerank_release_closure_blocker_queue.csv",
                mime="text/csv",
            )
        if consensus_rerank_release_closure_remediation_checklist_markdown:
            st.download_button(
                "Export consensus rerank release closure remediation checklist",
                data=consensus_rerank_release_closure_remediation_checklist_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_closure_remediation_checklist.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_closure_detached_manifest_df.empty:
            st.download_button(
                "Export consensus rerank release closure detached manifest CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_detached_manifest_df),
                file_name="consensus_rerank_release_closure_detached_manifest.csv",
                mime="text/csv",
            )
        if consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty:
            st.download_button(
                "Export consensus rerank action checklist",
                data=consensus_rerank_action_checklist_markdown.encode("utf-8"),
                file_name="consensus_rerank_action_checklist.md",
                mime="text/markdown",
            )
        if not ai_review_decision_df.empty:
            st.download_button(
                "导出规范化 AI 复核决策 CSV",
                data=_to_csv_bytes(ai_review_decision_df),
                file_name="ai_review_decisions_normalized.csv",
                mime="text/csv",
            )
        if not ai_review_decision_validation_df.empty:
            st.download_button(
                "导出 AI 复核决策校验 CSV",
                data=_to_csv_bytes(ai_review_decision_validation_df),
                file_name="ai_review_decision_validation.csv",
                mime="text/csv",
            )
        if not ai_review_round_summary_df.empty:
            st.download_button(
                "导出 AI 复核轮次汇总 CSV",
                data=_to_csv_bytes(ai_review_round_summary_df),
                file_name="ai_review_round_summary.csv",
                mime="text/csv",
            )
        if not ai_review_ranking_delta_df.empty:
            st.download_button(
                "导出 AI 复核排名变化 CSV",
                data=_to_csv_bytes(ai_review_ranking_delta_df),
                file_name="ai_review_ranking_delta.csv",
                mime="text/csv",
            )
        if ai_review_round_report_markdown:
            st.download_button(
                "导出 AI 复核轮次报告",
                data=ai_review_round_report_markdown.encode("utf-8"),
                file_name="ai_review_round_report.md",
                mime="text/markdown",
            )
        if not ai_review_artifact_manifest_df.empty:
            st.download_button(
                "导出 AI 复核产物清单 CSV",
                data=_to_csv_bytes(ai_review_artifact_manifest_df),
                file_name="ai_review_artifact_manifest.csv",
                mime="text/csv",
            )
        if ai_review_bundle_readme_markdown:
            st.download_button(
                "导出 AI 复核包 README",
                data=ai_review_bundle_readme_markdown.encode("utf-8"),
                file_name="ai_review_bundle_README.md",
                mime="text/markdown",
            )
        if ai_review_artifact_bundle_zip:
            st.download_button(
                "导出 AI 复核产物包 ZIP",
                data=ai_review_artifact_bundle_zip,
                file_name="ai_review_artifacts.zip",
                mime="application/zip",
            )
        if not ai_review_bundle_verification_df.empty:
            st.download_button(
                "导出 AI 复核包校验 CSV",
                data=_to_csv_bytes(ai_review_bundle_verification_df),
                file_name="ai_review_bundle_verification.csv",
                mime="text/csv",
            )
        if not ai_review_bundle_verification_summary_df.empty:
            st.download_button(
                "导出 AI 复核包校验汇总 CSV",
                data=_to_csv_bytes(ai_review_bundle_verification_summary_df),
                file_name="ai_review_bundle_verification_summary.csv",
                mime="text/csv",
            )
        if ai_review_bundle_certificate_markdown:
            st.download_button(
                "导出 AI 复核交接证书",
                data=ai_review_bundle_certificate_markdown.encode("utf-8"),
                file_name="ai_review_bundle_certificate.md",
                mime="text/markdown",
            )
        if not ai_review_decision_outcome_df.empty:
            st.download_button(
                "导出 AI 复核决策结果 CSV",
                data=_to_csv_bytes(ai_review_decision_outcome_df),
                file_name="ai_review_decision_outcomes.csv",
                mime="text/csv",
            )
        if not ai_evidence_review_queue_df.empty:
            st.download_button(
                "导出 AI 证据复核队列 CSV",
                data=_to_csv_bytes(ai_evidence_review_queue_df),
                file_name="ai_evidence_review_queue.csv",
                mime="text/csv",
            )
            if not ai_review_decision_template_df.empty:
                st.download_button(
                    "导出 AI 复核决策模板 CSV",
                    data=_to_csv_bytes(ai_review_decision_template_df),
                    file_name="ai_review_decision_template.csv",
                    mime="text/csv",
                )
            st.download_button(
                "导出 AI 证据复核清单",
                data=ai_review_checklist_markdown.encode("utf-8"),
                file_name="ai_evidence_review_checklist.md",
                mime="text/markdown",
            )
        if not rankable_ai_evidence_df.empty:
            st.download_button(
                "导出通过排名门控的 AI 证据 CSV",
                data=_to_csv_bytes(rankable_ai_evidence_df),
                file_name="ai_residue_evidence_ranked.csv",
                mime="text/csv",
            )
        if not ai_ranking_impact_df.empty:
            st.download_button(
                "导出 AI 排名影响 CSV",
                data=_to_csv_bytes(ai_ranking_impact_df),
                file_name="ai_ranking_impact_summary.csv",
                mime="text/csv",
            )
        if not ai_followup_plan_df.empty:
            st.download_button(
                "导出 AI 后续取证计划 CSV",
                data=_to_csv_bytes(ai_followup_plan_df),
                file_name="ai_followup_evidence_plan.csv",
                mime="text/csv",
            )
            st.download_button(
                "导出 AI 后续提示词包",
                data=ai_followup_prompt_bundle.encode("utf-8"),
                file_name="ai_followup_prompt_bundle.md",
                mime="text/markdown",
            )
    with export_cols[1]:
        if not overlap_summary.empty:
            st.download_button(
                "导出交集摘要 CSV",
                data=_to_csv_bytes(overlap_summary),
                file_name="overlap_summary.csv",
                mime="text/csv",
            )
        if not triple_overlap_df.empty:
            st.download_button(
                "导出三重交集 CSV",
                data=_to_csv_bytes(triple_overlap_df),
                file_name="triple_overlap.csv",
                mime="text/csv",
            )
        st.download_button(
            "导出快照 JSON",
            data=snapshot_to_json_bytes(snapshot),
            file_name="pocket_interface_snapshot.json",
            mime="application/json",
        )
        st.download_button(
            "导出快照 SVG",
            data=build_snapshot_svg(snapshot),
            file_name="pocket_interface_snapshot.svg",
            mime="image/svg+xml",
        )

    report_lines = [
        "ProteinInsight 口袋 / 界面分析报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"能量来源: {summary.get('energy_source') or '未标注'}",
        f"平均能量: {format_energy_value(summary.get('mean_energy'))}",
        f"蛋白体积: {f'{protein_volume:,.1f} A³' if protein_volume is not None else '-'}",
        f"当前口袋来源: {POCKET_SOURCE_LABELS.get(effective_pocket_mode, effective_pocket_mode)}",
        f"当前界面来源: {ANNOTATION_SOURCE_LABELS.get(effective_annotation_mode, effective_annotation_mode)}",
        f"结构推断依据: {inferred_basis_text or '-'}",
        f"口袋数量: {len(effective_pocket_summary) if not effective_pocket_summary.empty else 0}",
        f"自动口袋数: {len(auto_pocket_summary) if not auto_pocket_summary.empty else 0}",
        f"界面注释数: {len(enriched_annotations)}",
        f"三重交集: {len(triple_overlap_df)}",
        f"AI evidence: {len(ai_evidence_df)} rows / status {ai_evidence_meta.get('status') or '-'}",
        f"AI evidence used for ranking: {len(rankable_ai_evidence_df)} rows / status {rankable_ai_evidence_meta.get('status') or '-'}",
        f"Residue evidence consensus: {len(residue_evidence_consensus_df)} rows / top {top_residue_consensus.get('residue_anchor') if top_residue_consensus is not None else '-'} / tier {top_residue_consensus.get('consensus_tier') if top_residue_consensus is not None else '-'}",
        f"Pocket consensus coverage: {len(pocket_consensus_coverage_df)} rows / top {top_pocket_consensus_coverage.get('pocket_id') if top_pocket_consensus_coverage is not None else '-'} / label {top_pocket_consensus_coverage.get('pocket_consensus_label') if top_pocket_consensus_coverage is not None else '-'}",
        f"Benchmark reference candidate: {len(benchmark_reference_candidate_df)} rows / import {benchmark_reference_import_summary_df.iloc[0].get('import_status') if not benchmark_reference_import_summary_df.empty else '-'} / provisional used {'yes' if benchmark_reference_is_provisional else 'no'}",
        f"Benchmark reference source: {benchmark_reference_source_mode or '-'} / provisional {'yes' if benchmark_reference_is_provisional else 'no'} / reviewed candidate {'yes' if benchmark_reference_is_reviewed_candidate else 'no'}",
        f"Benchmark reference source audit: {len(benchmark_reference_source_audit_df)} rows / claim status {benchmark_reference_source_audit_df.iloc[0].get('source_claim_status') if not benchmark_reference_source_audit_df.empty else '-'} / independent claim {benchmark_reference_source_audit_df.iloc[0].get('can_support_independent_claim') if not benchmark_reference_source_audit_df.empty else '-'}",
        f"Benchmark reference source audit summary: {len(benchmark_reference_source_audit_summary_df)} rows / top status {benchmark_reference_source_audit_summary_df.iloc[0].get('source_claim_status') if not benchmark_reference_source_audit_summary_df.empty else '-'} / independent claim {benchmark_reference_source_audit_summary_df.iloc[0].get('can_support_independent_claim') if not benchmark_reference_source_audit_summary_df.empty else '-'}",
        f"Benchmark reference source audit action queue: {len(benchmark_reference_source_audit_action_queue_df)} rows / blockers {int(benchmark_reference_source_audit_action_queue_df['action_status'].astype(str).eq('blocker').sum()) if not benchmark_reference_source_audit_action_queue_df.empty and 'action_status' in benchmark_reference_source_audit_action_queue_df.columns else 0} / review {int(benchmark_reference_source_audit_action_queue_df['action_status'].astype(str).eq('review').sum()) if not benchmark_reference_source_audit_action_queue_df.empty and 'action_status' in benchmark_reference_source_audit_action_queue_df.columns else 0}",
        f"Benchmark reference source audit cases: {len(benchmark_reference_source_audit_case_summary_df)} rows / blocked {benchmark_reference_source_audit_case_summary_blocked_cases} / review {benchmark_reference_source_audit_case_summary_review_cases}",
        f"Benchmark reference source audit case decision template: {len(benchmark_reference_source_audit_case_decision_template_df)} rows",
        f"Benchmark reference source audit case decisions: {len(benchmark_reference_source_audit_case_decision_df)} rows / validation blocked {int(benchmark_reference_source_audit_case_decision_validation_df['validation_status'].astype(str).eq('blocked').sum()) if not benchmark_reference_source_audit_case_decision_validation_df.empty and 'validation_status' in benchmark_reference_source_audit_case_decision_validation_df.columns else 0}",
        f"Benchmark reference source audit case decision outcome summary: {len(benchmark_reference_source_audit_case_decision_outcome_summary_df)} rows / status {benchmark_reference_source_audit_case_decision_outcome_summary_status or '-'} / open {benchmark_reference_source_audit_case_decision_outcome_summary_open_cases}",
        f"Benchmark reference source audit case decision closure queue: {len(benchmark_reference_source_audit_case_decision_closure_queue_df)} rows / blockers {int(benchmark_reference_source_audit_case_decision_closure_queue_df['closure_action_status'].astype(str).eq('blocker').sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and 'closure_action_status' in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0} / review {int(benchmark_reference_source_audit_case_decision_closure_queue_df['closure_action_status'].astype(str).eq('review').sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and 'closure_action_status' in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0}",
        f"Benchmark reference source audit case decision readiness impact summary: {len(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df)} rows / status {benchmark_reference_source_audit_case_decision_readiness_impact_summary_status or '-'} / open {benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases}",
        f"Benchmark reference source audit case decision readiness impact: {len(benchmark_reference_source_audit_case_decision_readiness_impact_df)} rows / cleared {int(benchmark_reference_source_audit_case_decision_readiness_impact_df['readiness_impact'].astype(str).eq('cleared-by-decision').sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and 'readiness_impact' in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0} / open {int(benchmark_reference_source_audit_case_decision_readiness_impact_df['readiness_impact'].astype(str).isin(['decision-adjusted-open', 'decision-open', 'unchanged-open']).sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and 'readiness_impact' in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0}",
        f"Benchmark reference source audit case decision closure checklist: {'available' if benchmark_reference_source_audit_case_decision_closure_checklist_markdown else 'not available'}",
        f"Benchmark reference source audit case decision outcomes: {len(benchmark_reference_source_audit_case_decision_outcome_df)} rows / blocked {int(benchmark_reference_source_audit_case_decision_outcome_df['applied_status'].astype(str).eq('blocked').sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and 'applied_status' in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0} / pending {int(benchmark_reference_source_audit_case_decision_outcome_df['applied_status'].astype(str).eq('pending').sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and 'applied_status' in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0}",
        f"Benchmark reference source audit case checklist: {'available' if benchmark_reference_source_audit_case_checklist_markdown else 'not available'}",
        f"Benchmark reference source audit checklist: {'available' if benchmark_reference_source_audit_checklist_markdown else 'not available'}",
        f"Benchmark reference candidate review: {len(benchmark_reference_candidate_review_queue_df)} rows / P1 {int(benchmark_reference_candidate_review_queue_df['priority'].astype(str).eq('P1').sum()) if not benchmark_reference_candidate_review_queue_df.empty and 'priority' in benchmark_reference_candidate_review_queue_df.columns else 0} / P2 {int(benchmark_reference_candidate_review_queue_df['priority'].astype(str).eq('P2').sum()) if not benchmark_reference_candidate_review_queue_df.empty and 'priority' in benchmark_reference_candidate_review_queue_df.columns else 0} / checklist {'available' if benchmark_reference_candidate_review_checklist_markdown else 'not available'}",
        f"Benchmark reference candidate review decisions: {len(benchmark_reference_candidate_review_decision_df)} rows / validation blocked {int(benchmark_reference_candidate_review_decision_validation_df['validation_status'].astype(str).eq('blocked').sum()) if not benchmark_reference_candidate_review_decision_validation_df.empty and 'validation_status' in benchmark_reference_candidate_review_decision_validation_df.columns else 0} / accepted actions {int(benchmark_reference_candidate_review_outcome_df['applied_status'].astype(str).eq('accepted').sum()) if not benchmark_reference_candidate_review_outcome_df.empty and 'applied_status' in benchmark_reference_candidate_review_outcome_df.columns else 0} / accepted references {len(benchmark_reference_candidate_accepted_df)}",
        f"Catalytic pocket benchmark: references {len(benchmark_reference_df)} / Top-1 {top1_benchmark.get('coverage_ratio') if top1_benchmark is not None else '-'} / Top-3 {top3_benchmark.get('coverage_ratio') if top3_benchmark is not None else '-'} / best rank {top3_benchmark.get('best_rank') if top3_benchmark is not None else '-'}",
        f"Benchmark reference template: {len(benchmark_reference_template_df)} rows / notes {'available' if benchmark_reference_template_markdown else 'not available'}",
        f"Benchmark reference curation quality: {len(pocket_benchmark_reference_quality_issue_df)} issues / summary {len(pocket_benchmark_reference_quality_summary_df)} rows / checklist {'available' if pocket_benchmark_reference_quality_checklist_markdown else 'not available'}",
        f"Benchmark reference structure validation: {len(pocket_benchmark_reference_structure_validation_df)} issues / summary {len(pocket_benchmark_reference_structure_validation_summary_df)} rows / checklist {'available' if pocket_benchmark_reference_structure_validation_checklist_markdown else 'not available'}",
        f"Benchmark reference readiness: {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('readiness_status') if not pocket_benchmark_reference_readiness_summary_df.empty else '-'} / blockers {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('p0_p1_issue_count') if not pocket_benchmark_reference_readiness_summary_df.empty else 0} / review {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('p2_issue_count') if not pocket_benchmark_reference_readiness_summary_df.empty else 0} / source audit {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('source_audit_issue_count') if not pocket_benchmark_reference_readiness_summary_df.empty else 0}",
        f"Benchmark reference readiness cases: {len(pocket_benchmark_reference_readiness_case_summary_df)} rows / blocked {int(pocket_benchmark_reference_readiness_case_summary_df['readiness_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and 'readiness_status' in pocket_benchmark_reference_readiness_case_summary_df.columns else 0} / review {int(pocket_benchmark_reference_readiness_case_summary_df['readiness_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and 'readiness_status' in pocket_benchmark_reference_readiness_case_summary_df.columns else 0}",
        f"Benchmark interpretation: {len(pocket_benchmark_interpretation_df)} rows / Top-1 claim {pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df['top_n'].astype(int) == 1].iloc[0].get('claim_status') if not pocket_benchmark_interpretation_df.empty and 'top_n' in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df['top_n'].astype(int) == 1).any() else '-'} / Top-3 claim {pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df['top_n'].astype(int) == 3].iloc[0].get('claim_status') if not pocket_benchmark_interpretation_df.empty and 'top_n' in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df['top_n'].astype(int) == 3).any() else '-'}",
        f"Benchmark case interpretation: {len(pocket_benchmark_case_interpretation_df)} rows / blocked {int(pocket_benchmark_case_interpretation_df['claim_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_case_interpretation_df.empty and 'claim_status' in pocket_benchmark_case_interpretation_df.columns else 0} / review {int(pocket_benchmark_case_interpretation_df['claim_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_case_interpretation_df.empty and 'claim_status' in pocket_benchmark_case_interpretation_df.columns else 0}",
        f"Benchmark case interpretation matrix: {len(pocket_benchmark_case_interpretation_matrix_df)} rows / blocked {int(pocket_benchmark_case_interpretation_matrix_df['case_interpretation_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and 'case_interpretation_status' in pocket_benchmark_case_interpretation_matrix_df.columns else 0} / review {int(pocket_benchmark_case_interpretation_matrix_df['case_interpretation_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and 'case_interpretation_status' in pocket_benchmark_case_interpretation_matrix_df.columns else 0}",
        f"Benchmark case interpretation matrix summary: {pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get('summary_status') if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else '-'} / usable {pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get('usable_claim_ready_case_count') if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else 0}",
        f"Benchmark case interpretation matrix queue: {len(pocket_benchmark_case_interpretation_matrix_queue_df)} rows / blockers {int(pocket_benchmark_case_interpretation_matrix_queue_df['action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and 'action_status' in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0} / review {int(pocket_benchmark_case_interpretation_matrix_queue_df['action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and 'action_status' in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0}",
        f"Benchmark dataset interpretation: {len(pocket_benchmark_dataset_interpretation_df)} rows / blocked {int(pocket_benchmark_dataset_interpretation_df['dataset_claim_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_dataset_interpretation_df.empty and 'dataset_claim_status' in pocket_benchmark_dataset_interpretation_df.columns else 0} / review {int(pocket_benchmark_dataset_interpretation_df['dataset_claim_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_dataset_interpretation_df.empty and 'dataset_claim_status' in pocket_benchmark_dataset_interpretation_df.columns else 0}",
        f"Benchmark source-audit decision dataset impact: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df)} rows / blockers {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df['dataset_source_impact_status'].astype(str).eq('source-blocked').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and 'dataset_source_impact_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0} / review {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df['dataset_source_impact_status'].astype(str).eq('source-review-needed').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and 'dataset_source_impact_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0} / mismatch {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df['dataset_source_impact_status'].astype(str).eq('source-gate-mismatch').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and 'dataset_source_impact_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0}",
        f"Benchmark source-audit decision dataset impact cases: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df)} rows / blockers {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df['source_action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and 'source_action_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0} / review {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df['source_action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and 'source_action_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0} / mismatch {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df['source_gate_mismatch'].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and 'source_gate_mismatch' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0} / checklist {'available' if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown else 'not available'} / report {'available' if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_markdown else 'not available'}",
        f"Benchmark source-audit decision dataset impact action queue: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df)} rows / blockers {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df['action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and 'action_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0} / review {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df['action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and 'action_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0} / mismatch {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df['source_gate_mismatch'].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.empty and 'source_gate_mismatch' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_df.columns else 0}",
        f"Benchmark source-audit decision dataset impact action summary: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df)} rows / actions {int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df['action_count'], errors='coerce').fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and 'action_count' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0} / P0 groups {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df['priority'].astype(str).eq('P0').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and 'priority' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0} / mismatches {int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df['mismatch_count'], errors='coerce').fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.empty and 'mismatch_count' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_df.columns else 0}",
        f"Benchmark source-audit decision dataset impact artifacts: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df)} files / bytes {int(pd.to_numeric(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df['byte_size'], errors='coerce').fillna(0).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty and 'byte_size' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.columns else 0} / hashes {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df['sha256'].astype(str).str.fullmatch(r'[0-9a-f]{64}').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.empty and 'sha256' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_df.columns else 0}",
        f"Benchmark dataset interpretation queue: {len(pocket_benchmark_dataset_interpretation_queue_df)} rows / blockers {int(pocket_benchmark_dataset_interpretation_queue_df['action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and 'action_status' in pocket_benchmark_dataset_interpretation_queue_df.columns else 0} / review {int(pocket_benchmark_dataset_interpretation_queue_df['action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and 'action_status' in pocket_benchmark_dataset_interpretation_queue_df.columns else 0} / checklist {'available' if pocket_benchmark_dataset_interpretation_checklist_markdown else 'not available'} / report {'available' if pocket_benchmark_dataset_interpretation_report_markdown else 'not available'}",
        f"Catalytic benchmark dataset: cases {int(pocket_benchmark_case_summary_df['benchmark_id'].nunique()) if not pocket_benchmark_case_summary_df.empty and 'benchmark_id' in pocket_benchmark_case_summary_df.columns else 0} / dataset rows {len(pocket_benchmark_dataset_summary_df)}",
        f"Catalytic benchmark variants: {len(pocket_benchmark_variant_comparison_df)} rows / current vs ablations {'available' if not pocket_benchmark_variant_comparison_df.empty else 'not available'}",
        f"Catalytic benchmark variant cases: {len(pocket_benchmark_variant_case_comparison_df)} rows / variant dataset rows {len(pocket_benchmark_variant_dataset_comparison_df)}",
        f"Catalytic benchmark variant residues: {len(pocket_benchmark_variant_detail_comparison_df)} rows",
        f"Catalytic benchmark remediation queue: {len(pocket_benchmark_variant_remediation_df)} rows / summary {len(pocket_benchmark_variant_remediation_summary_df)} rows / checklist {'available' if pocket_benchmark_variant_remediation_checklist_markdown else 'not available'}",
        f"P2Rank A/B: {len(p2rank_ab_df)} rows / {'enabled' if p2rank_ab_enabled else 'not enabled'}",
        f"Consensus rerank suggestions: {len(consensus_rerank_suggestion_df)} rows / top {top_consensus_rerank_suggestion.get('pocket_id') if top_consensus_rerank_suggestion is not None else '-'} / status {top_consensus_rerank_suggestion.get('suggestion_status') if top_consensus_rerank_suggestion is not None else '-'}",
        f"Consensus rerank preview: {len(consensus_rerank_preview_df)} rows / top {top_consensus_rerank_preview.get('pocket_id') if top_consensus_rerank_preview is not None else '-'} / decision {top_consensus_rerank_preview.get('preview_decision') if top_consensus_rerank_preview is not None else '-'}",
        f"Consensus rerank policy gate: {top_consensus_rerank_policy_gate.get('policy_status') if top_consensus_rerank_policy_gate is not None else '-'} / changed {top_consensus_rerank_policy_gate.get('changed_rows') if top_consensus_rerank_policy_gate is not None else 0} / blocked {top_consensus_rerank_policy_gate.get('blocked_rows') if top_consensus_rerank_policy_gate is not None else 0}",
        f"Consensus rerank action queue: {len(consensus_rerank_action_queue_df)} rows / top {top_consensus_rerank_action.get('pocket_id') if top_consensus_rerank_action is not None else '-'} / issue {top_consensus_rerank_action.get('issue_type') if top_consensus_rerank_action is not None else '-'}",
        f"Consensus rerank action checklist: {'available' if consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty else 'not available'}",
        f"Consensus rerank apply simulation: {len(consensus_rerank_apply_simulation_df)} rows / top {top_consensus_rerank_apply.get('pocket_id') if top_consensus_rerank_apply is not None else '-'} / status {top_consensus_rerank_apply.get('apply_status') if top_consensus_rerank_apply is not None else '-'}",
        f"Consensus rerank simulation delta: {len(consensus_rerank_simulation_delta_df)} rows / top {top_consensus_rerank_delta.get('pocket_id') if top_consensus_rerank_delta is not None else '-'} / change {top_consensus_rerank_delta.get('change_type') if top_consensus_rerank_delta is not None else '-'}",
        f"Consensus rerank precision scorecard: {top_consensus_rerank_scorecard.get('scorecard_status') if top_consensus_rerank_scorecard is not None else '-'} / score {top_consensus_rerank_scorecard.get('precision_improvement_score') if top_consensus_rerank_scorecard is not None else 0} / blockers {top_consensus_rerank_scorecard.get('open_blocker_rows') if top_consensus_rerank_scorecard is not None else 0}",
        f"Consensus rerank precision guardrail: {top_consensus_rerank_guardrail.get('guardrail_status') if top_consensus_rerank_guardrail is not None else '-'} / decision {top_consensus_rerank_guardrail.get('guardrail_decision') if top_consensus_rerank_guardrail is not None else '-'} / mode {top_consensus_rerank_guardrail.get('apply_mode') if top_consensus_rerank_guardrail is not None else '-'}",
        f"Consensus rerank precision guardrail report: {'available' if consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty else 'not available'}",
        f"Consensus rerank guardrail handoff bundle: {'available' if consensus_rerank_guardrail_handoff_zip else 'not available'} / manifest {len(consensus_rerank_guardrail_artifact_manifest_df)} files",
        f"Consensus rerank guardrail bundle verification: {consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get('verification_status') if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else '-'} / failed {consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get('failed_files') if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else 0}",
        f"Consensus rerank guardrail handoff certificate: {'available' if consensus_rerank_guardrail_handoff_certificate_markdown else 'not available'}",
        f"Consensus rerank release decision template: {len(consensus_rerank_release_decision_template_df)} rows",
        f"Consensus rerank release decisions: {len(consensus_rerank_release_decision_df)} rows / status {consensus_rerank_release_decision_meta.get('status') or '-'}",
        f"Consensus rerank release decision validation: {len(consensus_rerank_release_decision_validation_df)} rows / blocked {int((consensus_rerank_release_decision_validation_df['validation_status'].astype(str) == 'blocked').sum()) if not consensus_rerank_release_decision_validation_df.empty and 'validation_status' in consensus_rerank_release_decision_validation_df.columns else 0}",
        f"Consensus rerank release review: {top_consensus_rerank_release_decision_summary.get('release_review_status') if top_consensus_rerank_release_decision_summary is not None else '-'} / allowed {'yes' if top_consensus_rerank_release_decision_summary is not None and bool(top_consensus_rerank_release_decision_summary.get('release_allowed')) else 'no'}",
        f"Consensus rerank release apply plan: {len(consensus_rerank_release_apply_plan_df)} rows / top {top_consensus_rerank_release_apply_plan.get('pocket_id') if top_consensus_rerank_release_apply_plan is not None else '-'} / status {top_consensus_rerank_release_apply_plan.get('release_apply_status') if top_consensus_rerank_release_apply_plan is not None else '-'}",
        f"Consensus rerank release apply report: {'available' if consensus_rerank_release_apply_report_markdown else 'not available'}",
        f"Consensus rerank release execution template: {len(consensus_rerank_release_execution_template_df)} rows",
        f"Consensus rerank release execution receipt: {len(consensus_rerank_release_execution_receipt_df)} rows / status {consensus_rerank_release_execution_receipt_meta.get('status') or '-'}",
        f"Consensus rerank release execution validation: {len(consensus_rerank_release_execution_validation_df)} rows / blocked {int((consensus_rerank_release_execution_validation_df['validation_status'].astype(str) == 'blocked').sum()) if not consensus_rerank_release_execution_validation_df.empty and 'validation_status' in consensus_rerank_release_execution_validation_df.columns else 0}",
        f"Consensus rerank release execution: {top_consensus_rerank_release_execution_summary.get('execution_review_status') if top_consensus_rerank_release_execution_summary is not None else '-'} / complete {'yes' if top_consensus_rerank_release_execution_summary is not None and bool(top_consensus_rerank_release_execution_summary.get('execution_complete')) else 'no'}",
        f"Consensus rerank release execution report: {'available' if consensus_rerank_release_execution_report_markdown else 'not available'}",
        f"Consensus rerank release closure certificate: {'available' if consensus_rerank_release_closure_certificate_markdown else 'not available'}",
        f"Consensus rerank release closure ledger: {len(consensus_rerank_release_closure_ledger_df)} rows / blocked {int(consensus_rerank_release_closure_ledger_df['closure_check'].astype(str).str.lower().isin({'blocked', 'missing'}).sum()) if not consensus_rerank_release_closure_ledger_df.empty and 'closure_check' in consensus_rerank_release_closure_ledger_df.columns else 0}",
        f"Consensus rerank release closure readiness: {top_consensus_rerank_release_closure_summary.get('closure_readiness_status') if top_consensus_rerank_release_closure_summary is not None else '-'} / closed {'yes' if top_consensus_rerank_release_closure_summary is not None and bool(top_consensus_rerank_release_closure_summary.get('release_closed')) else 'no'}",
        f"Consensus rerank release closure blockers: {len(consensus_rerank_release_closure_blocker_df)} rows / top {consensus_rerank_release_closure_blocker_df.iloc[0].get('blocker_type') if not consensus_rerank_release_closure_blocker_df.empty else '-'}",
        f"Consensus rerank release closure remediation checklist: {'available' if consensus_rerank_release_closure_remediation_checklist_markdown else 'not available'}",
        f"Consensus rerank release closure detached manifest: {len(consensus_rerank_release_closure_detached_manifest_df)} files",
        f"AI review decisions: {len(ai_review_decision_df)} rows / applied {ai_review_decision_meta.get('applied_rows') or '0'} / status {ai_review_decision_meta.get('status') or '-'}",
        f"AI review decision validation: {len(ai_review_decision_validation_df)} rows / blocked {int((ai_review_decision_validation_df['validation_status'].astype(str) == 'blocked').sum()) if not ai_review_decision_validation_df.empty and 'validation_status' in ai_review_decision_validation_df.columns else 0}",
        f"AI review round: {ai_review_round_summary_df.iloc[0].get('review_round_status') if not ai_review_round_summary_df.empty else '-'} / rankable {ai_review_round_summary_df.iloc[0].get('rankable_after_review_rows') if not ai_review_round_summary_df.empty else 0}",
        f"AI review ranking delta: {ai_review_ranking_delta_df.iloc[0].get('review_effect_status') if not ai_review_ranking_delta_df.empty else '-'} / +{ai_review_ranking_delta_df.iloc[0].get('promoted_rows') if not ai_review_ranking_delta_df.empty else 0} / -{ai_review_ranking_delta_df.iloc[0].get('removed_rows') if not ai_review_ranking_delta_df.empty else 0}",
        f"AI review artifact manifest: {len(ai_review_artifact_manifest_df)} files",
        f"AI review bundle README: {'available' if ai_review_bundle_readme_markdown else 'not available'}",
        f"AI review artifact bundle: {'available' if ai_review_artifact_bundle_zip else 'not available'}",
        f"AI review bundle verification: {len(ai_review_bundle_verification_df)} files / failed {int((ai_review_bundle_verification_df['verification_status'].astype(str) != 'verified').sum()) if not ai_review_bundle_verification_df.empty and 'verification_status' in ai_review_bundle_verification_df.columns else 0}",
        f"AI review bundle verification summary: {ai_review_bundle_verification_summary_df.iloc[0].get('verification_status') if not ai_review_bundle_verification_summary_df.empty else '-'}",
        f"AI review bundle certificate: {'available' if ai_review_bundle_certificate_markdown else 'not available'}",
        f"AI review decision outcomes: {len(ai_review_decision_outcome_df)} rows",
        f"AI review decision template: {len(ai_review_decision_template_df)} rows",
        f"AI 影响: {_localize_status_text(ai_ranking_impact_df.iloc[0].get('ai_influence_level') if not ai_ranking_impact_df.empty else '-')} / Top 口袋 AI 残基 {ai_ranking_impact_df.iloc[0].get('top_pocket_ai_residues') if not ai_ranking_impact_df.empty else '-'}",
        f"AI review queue: {len(ai_evidence_review_queue_df)} rows / top fix {ai_evidence_review_queue_df.iloc[0].get('fix_type') if not ai_evidence_review_queue_df.empty else '-'}",
        f"AI follow-up plan: {len(ai_followup_plan_df)} rows",
        "AI evidence audit: "
        + (
            ", ".join(f"{status}:{count}" for status, count in ai_evidence_audit_df["audit_status"].astype(str).value_counts().to_dict().items())
            if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns
            else "none"
        ),
    ]
    if top_joint_candidate is not None:
        report_lines.extend(
            [
                f"联合推荐 Top1: {top_joint_candidate['pocket_id']}",
                f"联合推荐等级: {top_joint_candidate['recommendation_label']}",
                f"联合推荐理由: {top_joint_candidate['recommendation_reason']}",
            ]
        )
    if top_pocket_decision is not None:
        reliability_pass_count = int((pocket_reliability_df["status"].astype(str) == "pass").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0
        reliability_review_count = int((pocket_reliability_df["status"].astype(str) == "review").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0
        reliability_missing_count = int((pocket_reliability_df["status"].astype(str) == "missing").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0
        report_lines.extend(
            [
                f"Top 活性位点决策: {top_pocket_decision.get('pocket_id')} / {_localize_pocket_decision_text(top_pocket_decision.get('decision_label'))}",
                f"Top 决策评分: {top_pocket_decision.get('decision_score')} / 审计 {_localize_pocket_decision_text(top_pocket_decision.get('audit_status'))}",
                f"精度分层: {_localize_pocket_decision_text(top_pocket_triage.get('precision_tier') if top_pocket_triage is not None else '-')}",
                f"分诊动作: {_localize_pocket_decision_text(top_pocket_triage.get('triage_action') if top_pocket_triage is not None else '-')}",
                f"可靠性检查: 通过 {reliability_pass_count}, 复核 {reliability_review_count}, 缺失 {reliability_missing_count}",
                f"可靠性缺口: {_localize_pocket_decision_text(top_reliability_gaps or 'none')}",
                f"下一步: {_localize_pocket_decision_text(top_pocket_decision.get('next_step'))}",
            ]
        )
    report_lines = [_localize_report_line(line) for line in report_lines]
    report_text = "\n".join(report_lines)
    if PDF_EXPORT_AVAILABLE:
        st.download_button(
            "导出 PDF 报告",
            data=build_simple_pdf(report_text, snapshot=snapshot),
            file_name="pocket_interface_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("当前环境未安装 reportlab，PDF 报告暂不可用。请安装依赖后重试。")

st.info("页面已补齐三类主链路：上传 Pocket 与自动口袋可切换或合并；未上传界面注释时可使用结构推断界面；交集分析、筛选结果和当前主分析结果都可以单独导出。")
