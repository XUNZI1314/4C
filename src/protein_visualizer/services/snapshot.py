from __future__ import annotations

import html
import json
import textwrap
from datetime import datetime
from typing import Any, Optional, Sequence

import pandas as pd

from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return value.item()  # numpy scalar
    except Exception:
        return str(value)


def _format_volume_text(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.1f} A³"
    except (TypeError, ValueError):
        return "-"


def _frame_preview(table: Optional[pd.DataFrame], columns: Optional[Sequence[str]] = None, max_rows: int = 8) -> list[dict[str, Any]]:
    if table is None or getattr(table, "empty", True):
        return []

    preview = table.copy()
    if columns:
        available_columns = [column for column in columns if column in preview.columns]
        preview = preview[available_columns]
    preview = preview.head(max_rows)

    records: list[dict[str, Any]] = []
    for row in preview.to_dict(orient="records"):
        records.append({key: _safe_value(value) for key, value in row.items()})
    return records


def build_analysis_snapshot(
    energy_table: Optional[pd.DataFrame],
    *,
    title: str = "ProteinInsight 分析快照",
    annotation_table: Optional[pd.DataFrame] = None,
    hotspot_df: Optional[pd.DataFrame] = None,
    pocket_summary: Optional[pd.DataFrame] = None,
    joint_candidate_df: Optional[pd.DataFrame] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    protein_volume: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    summary = build_analysis_summary(energy_table)
    if protein_volume is not None:
        summary["protein_volume"] = float(protein_volume)

    snapshot = {
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {key: _safe_value(value) for key, value in summary.items()},
        "protein_volume": _safe_value(protein_volume),
        "hotspot_count": int(len(hotspot_df)) if hotspot_df is not None else 0,
        "annotation_rows": int(len(annotation_table)) if annotation_table is not None else 0,
        "pocket_rows": int(len(pocket_summary)) if pocket_summary is not None else 0,
        "joint_candidate_rows": int(len(joint_candidate_df)) if joint_candidate_df is not None else 0,
        "comparison_rows": int(len(comparison_df)) if comparison_df is not None else 0,
        "top_hotspots": _frame_preview(hotspot_df, ["label", "chain", "resid", "resname", "delta_total", "hotspot_rank", "energy"]),
        "annotation_preview": _frame_preview(
            annotation_table,
            ["residue_label", "classification_label", "energy", "hotspot_rank", "is_hotspot", "is_pocket"],
            max_rows=12,
        ),
        "pocket_summary": _frame_preview(
            pocket_summary,
            [
                "pocket_id",
                "smart_rank_label",
                "smart_rank_score",
                "smart_rank_reason",
                "evidence_quality_label",
                "evidence_quality_score",
                "evidence_quality_warning",
                "detection_route",
                "consensus_methods",
                "method_vote_count",
                "consensus_score",
                "volume",
                "score",
                "residue_count",
                "hotspot_count",
                "residue_labels",
            ],
            max_rows=10,
        ),
        "joint_candidate_preview": _frame_preview(
            joint_candidate_df,
            [
                "recommendation_rank",
                "pocket_id",
                "recommendation_score",
                "recommendation_label",
                "recommendation_action",
                "evidence_quality_label",
                "recommendation_reason",
                "smart_rank_label",
                "hotspot_overlap_count",
                "interface_overlap_count",
                "triple_overlap_count",
            ],
            max_rows=10,
        ),
        "comparison_preview": _frame_preview(
            comparison_df,
            ["chain", "resid", "resname", "count", "label", "is_common"],
            max_rows=20,
        ),
        "extra": {key: _safe_value(value) for key, value in (extra or {}).items()},
    }
    return snapshot


def snapshot_to_json_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _reliability_status_counts(extra: dict[str, Any]) -> dict[str, int]:
    records = extra.get("pocket_reliability") or []
    counts = {"pass": 0, "review": 0, "missing": 0}
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            status = str(record.get("status") or "").strip().lower()
            if status in counts:
                counts[status] += 1
    for status in counts:
        key = f"pocket_reliability_{status}_count"
        if key in extra:
            try:
                counts[status] = int(extra.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return counts


SNAPSHOT_VALUE_LABELS = {
    "ok": "正常",
    "blocked": "阻断",
    "promoted": "提升",
    "removed": "移除",
    "review": "复核",
    "pass": "通过",
    "missing": "缺失",
    "supported": "已支持",
    "verified": "已校验",
    "available": "可用",
    "none": "无",
    "top-pocket-supported": "Top 口袋受支持",
    "missing-citation-or-snippet": "缺少引用或证据片段",
    "Review mapping before validation": "验证前复核映射",
    "mapping-review-needed": "映射需复核",
    "mapping-review": "映射需复核",
    "Review chain/numbering/mapping before validation.": "验证前复核链、编号和映射。",
}


SNAPSHOT_TEXT_REPLACEMENTS = [
    ("Functional anchors", "功能锚点"),
    ("Evidence mapping risk", "证据映射风险"),
    ("Geometry consensus", "几何共识"),
    ("Evidence A/B movement", "证据 A/B 变化"),
    ("Actionability", "可操作性"),
    ("validated-anchor", "已验证锚点"),
    ("consensus-validated-pocket", "共识已验证口袋"),
    ("keep-prioritized", "保持优先"),
    ("would-keep-priority", "将保持优先"),
    ("no-change-needed", "无需变更"),
    ("validation-anchor-ready", "验证锚点就绪"),
    ("keep-current-ready", "保留当前且就绪"),
    ("frozen-blocker", "冻结阻断"),
    ("likely-precision-gain", "可能提升精度"),
    ("manual-review-ready", "人工复核就绪"),
    ("allow-after-review", "复核后允许"),
    ("manual-consensus-rerank", "人工共识重排"),
    ("approved-for-manual-release", "批准人工发布"),
    ("ready-for-manual-apply", "可人工应用"),
    ("closed-and-verified", "已关闭并校验"),
    ("ledger-blocked", "台账阻断"),
    ("missing-evidence", "缺少证据"),
    ("not available", "不可用"),
    ("executed", "已执行"),
    ("available", "可用"),
    ("verified", "已校验"),
    ("ok", "正常"),
    (": pass", ": 通过"),
    (": review", ": 复核"),
    (": missing", ": 缺失"),
    (", pass", ", 通过"),
    (" yes", " 是"),
    (" no", " 否"),
]


SNAPSHOT_CONSENSUS_LINE_REPLACEMENTS = [
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
    ("Residue evidence consensus", "残基证据共识"),
    ("Pocket consensus coverage", "口袋共识覆盖"),
    ("ledger-blocked", "台账阻断"),
    ("rank delta ", "排名变化 "),
    ("source audit ", "来源审计 "),
    ("claim ", "结论 "),
    ("accepted ", "已接受 "),
    ("positive ", "正向 "),
    ("blockers ", "阻断项 "),
    ("blocked ", "阻断 "),
    ("changed ", "变化 "),
    ("decision ", "决策 "),
    ("status ", "状态 "),
    ("allowed ", "允许 "),
    ("complete ", "完成 "),
    ("closed ", "关闭 "),
    ("anchors ", "锚点 "),
    ("score ", "评分 "),
    ("mode ", "模式 "),
    ("manifest ", "清单 "),
    ("files ", "文件数 "),
    ("failed ", "失败 "),
    (" / top ", " / Top "),
    (" rows", " 行"),
    (" files", " 个文件"),
]


SNAPSHOT_AUTO_DETECTION_TEXT_REPLACEMENTS = [
    (":used", ":已使用"),
    (":single-method", ":单方法"),
    ("; consensus", "; 共识"),
    ("enabled", "已启用"),
    ("ok", "正常"),
]


SNAPSHOT_BENCHMARK_LINE_REPLACEMENTS = [
    ("Benchmark source-audit decision dataset impact action summary", "基准来源审计决策数据集影响行动汇总"),
    ("Benchmark source-audit decision dataset impact action queue", "基准来源审计决策数据集影响行动队列"),
    ("Benchmark source-audit decision dataset impact artifacts", "基准来源审计决策数据集影响产物"),
    ("Benchmark source-audit decision dataset impact cases", "基准来源审计决策数据集影响案例"),
    ("Benchmark source-audit decision dataset impact", "基准来源审计决策数据集影响"),
    ("Benchmark reference source audit case decision readiness impact summary", "基准参考来源审计案例决策就绪影响汇总"),
    ("Benchmark reference source audit case decision readiness impact", "基准参考来源审计案例决策就绪影响"),
    ("Benchmark reference source audit case decision closure checklist", "基准参考来源审计案例决策关闭清单"),
    ("Benchmark reference source audit case decision closure queue", "基准参考来源审计案例决策关闭队列"),
    ("Benchmark reference source audit case decision outcome summary", "基准参考来源审计案例决策结果汇总"),
    ("Benchmark reference source audit case decision outcomes", "基准参考来源审计案例决策结果"),
    ("Benchmark reference source audit case decision template", "基准参考来源审计案例决策模板"),
    ("Benchmark reference source audit case decisions", "基准参考来源审计案例决策"),
    ("Benchmark reference source audit action queue", "基准参考来源审计行动队列"),
    ("Benchmark reference source audit case checklist", "基准参考来源审计案例清单"),
    ("Benchmark reference source audit checklist", "基准参考来源审计清单"),
    ("Benchmark reference source audit summary", "基准参考来源审计汇总"),
    ("Benchmark reference source audit cases", "基准参考来源审计案例"),
    ("Benchmark reference source audit", "基准参考来源审计"),
    ("Benchmark reference candidate review decisions", "基准参考候选复核决策"),
    ("Benchmark reference candidate review", "基准参考候选复核"),
    ("Benchmark reference candidate", "基准参考候选"),
    ("Benchmark reference curation quality", "基准参考整理质量"),
    ("Benchmark reference structure validation", "基准参考结构校验"),
    ("Benchmark reference readiness cases", "基准参考就绪案例"),
    ("Benchmark reference readiness", "基准参考就绪"),
    ("Benchmark reference template", "基准参考模板"),
    ("Benchmark reference source", "基准参考来源"),
    ("Catalytic pocket benchmark", "催化口袋基准"),
    ("Benchmark case interpretation matrix summary", "基准案例解释矩阵汇总"),
    ("Benchmark case interpretation matrix queue", "基准案例解释矩阵队列"),
    ("Benchmark case interpretation matrix", "基准案例解释矩阵"),
    ("Benchmark case interpretation", "基准案例解释"),
    ("Benchmark dataset interpretation queue", "基准数据集解释队列"),
    ("Benchmark dataset interpretation report", "基准数据集解释报告"),
    ("Benchmark dataset interpretation", "基准数据集解释"),
    ("Benchmark interpretation", "基准解释"),
    ("Catalytic benchmark remediation queue", "催化基准修复队列"),
    ("Catalytic benchmark variant residues", "催化基准变体残基"),
    ("Catalytic benchmark variant cases", "催化基准变体案例"),
    ("Catalytic benchmark variants", "催化基准变体"),
    ("Catalytic benchmark dataset", "催化基准数据集"),
    ("provisional-external-evidence", "临时外部证据"),
    ("topn-complete-hit", "Top-N 完全命中"),
    ("top1-partial-hit", "Top-1 部分命中"),
    ("blocked-provisional", "临时参考阻断"),
    ("source-gate-mismatch", "来源门控不匹配"),
    ("review-needed", "需复核"),
    ("validation blocked", "校验阻断"),
    (" / top ", " / Top "),
    ("accepted references", "已接受参考"),
    ("accepted actions", "已接受动作"),
    ("provisional used", "使用临时参考"),
    ("reviewed candidate", "已复核候选"),
    ("independent claim", "独立结论"),
    ("claim status", "结论状态"),
    ("top status", "Top 状态"),
    ("provisional rows", "临时参考行"),
    ("reviewed rows", "已复核行"),
    ("net blocker delta", "阻断项净变化"),
    ("Top-1 claim", "Top-1 结论"),
    ("Top-3 claim", "Top-3 结论"),
    ("dataset rows", "数据集行"),
    ("summary rows", "汇总行"),
    ("P0 groups", "P0 组"),
    ("mismatches", "不匹配项"),
    ("mismatch", "不匹配"),
    ("references", "参考"),
    ("reference", "参考"),
    ("checklist", "清单"),
    ("report", "报告"),
    ("import", "导入"),
    ("notes", "备注"),
    ("issues", "问题"),
    ("issue", "问题"),
    ("summary", "汇总"),
    ("source audit", "来源审计"),
    ("queue", "队列"),
    ("actions", "动作"),
    ("action", "动作"),
    ("bytes", "字节"),
    ("hashes", "哈希"),
    ("cases", "案例"),
    ("case", "案例"),
    ("usable", "可用"),
    ("open", "未关闭"),
    ("pending", "待处理"),
    ("cleared", "已清除"),
    ("blocked", "阻断"),
    ("blockers", "阻断项"),
    ("review", "复核"),
    ("status", "状态"),
    ("provisional", "临时参考"),
]


def _snapshot_value_label(value: Any, *, default: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return SNAPSHOT_VALUE_LABELS.get(text, text)


def _snapshot_text_label(value: Any, *, default: str = "-") -> str:
    text = _snapshot_value_label(value, default=default)
    for source, target in SNAPSHOT_TEXT_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def _snapshot_auto_detection_label(value: Any, *, default: str = "-") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    for source, target in SNAPSHOT_AUTO_DETECTION_TEXT_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def _snapshot_summary_line_label(line: str) -> str:
    text = str(line)
    if text.startswith(("Benchmark ", "Catalytic ")):
        for source, target in SNAPSHOT_BENCHMARK_LINE_REPLACEMENTS:
            text = text.replace(source, target)
        text = text.replace(" rows", " 行").replace(" files", " 个文件")
        return _snapshot_text_label(text)
    if not text.startswith(("Residue evidence consensus", "Pocket consensus coverage", "Consensus rerank")):
        return text
    for source, target in SNAPSHOT_CONSENSUS_LINE_REPLACEMENTS:
        text = text.replace(source, target)
    return _snapshot_text_label(text)


def snapshot_to_summary_lines(snapshot: dict[str, Any]) -> list[str]:
    summary = snapshot.get("summary") or {}
    extra = snapshot.get("extra") or {}
    lines = [
        f"生成时间: {snapshot.get('generated_at', '-')}",
        f"标题: {snapshot.get('title', '-')}",
        f"残基总数: {summary.get('residue_count', '-')}",
        f"有效能量数: {summary.get('valid_energy_count', '-')}/{summary.get('residue_count', '-')}",
        f"能量来源: {summary.get('energy_source') or '未标注'}",
        f"平均能量: {format_energy_value(summary.get('mean_energy'))}",
        f"最低能量: {format_energy_value(summary.get('min_energy'))}",
        f"最高能量: {format_energy_value(summary.get('max_energy'))}",
    ]

    protein_volume = summary.get("protein_volume")
    if protein_volume is None:
        protein_volume = snapshot.get("protein_volume")
    volume_text = _format_volume_text(protein_volume)
    if volume_text != "-":
        lines.append(f"蛋白质体积: {volume_text}")

    if snapshot.get("hotspot_count") is not None:
        lines.append(f"热点残基数: {snapshot.get('hotspot_count')}")
    if snapshot.get("pocket_rows") is not None:
        lines.append(f"口袋条目数: {snapshot.get('pocket_rows')}")
    if snapshot.get("joint_candidate_rows") is not None:
        lines.append(f"联合推荐条目数: {snapshot.get('joint_candidate_rows')}")
    methods_used = str(extra.get("auto_detection_methods_used") or "").strip()
    if methods_used:
        lines.append(f"自动口袋方法: {methods_used}")
    status_summary = str(extra.get("auto_detection_status_summary") or "").strip()
    if status_summary:
        lines.append(f"检测状态: {_snapshot_auto_detection_label(status_summary)}")
    p2rank_status = str(extra.get("auto_detection_p2rank_status") or "").strip()
    if p2rank_status:
        lines.append(
            f"P2Rank: {_snapshot_auto_detection_label(p2rank_status)} / 预测 {int(extra.get('auto_detection_p2rank_prediction_rows') or 0)} / "
            f"残基 {int(extra.get('auto_detection_p2rank_residue_rows') or 0)}"
        )
    if bool(extra.get("p2rank_ab_enabled")):
        comparison_rows = len(extra.get("p2rank_ab_comparison") or [])
        lines.append(f"P2Rank A/B: 已启用 / 记录 {comparison_rows}")
    try:
        external_rows = int(extra.get("auto_detection_external_rows") or 0)
    except (TypeError, ValueError):
        external_rows = 0
    if external_rows > 0:
        source_text = str(extra.get("auto_detection_external_sources") or "external").strip()
        lines.append(f"外部位点证据: {external_rows} ({source_text})")
    manual_key_rows = int(extra.get("manual_key_residue_rows") or 0)
    if manual_key_rows > 0:
        manual_meta = extra.get("manual_key_residue_metadata") if isinstance(extra.get("manual_key_residue_metadata"), dict) else {}
        manual_status = str(extra.get("manual_key_residue_status") or manual_meta.get("status") or "").strip()
        lines.append(f"人工关键残基: {manual_key_rows} 行 / 状态 {_snapshot_value_label(manual_status)}")
    ai_rows = int(extra.get("ai_evidence_rows") or 0)
    ai_status = str(extra.get("ai_evidence_status") or "").strip()
    if ai_rows > 0 or ai_status:
        lines.append(f"AI 证据: {ai_rows} 行 / 状态 {_snapshot_value_label(ai_status)}")
    ai_ranked = int(extra.get("ai_evidence_ranked_rows") or 0)
    if ai_rows > 0 or ai_ranked > 0:
        lines.append(f"AI 排名可用证据: {ai_ranked} 行")
    ai_review_decisions = int(extra.get("ai_review_decision_rows") or 0)
    if ai_review_decisions > 0:
        ai_review_status = str(extra.get("ai_review_decision_status") or "-").strip()
        ai_review_applied = int(extra.get("ai_review_decision_applied_rows") or ai_review_decisions)
        lines.append(
            f"AI 复核决策: {ai_review_decisions} 行 / 已应用 {ai_review_applied} / 状态 {_snapshot_value_label(ai_review_status)}"
        )
    ai_review_validation_rows = int(extra.get("ai_review_decision_validation_rows") or 0)
    if ai_review_validation_rows > 0:
        blocked_rows = int(extra.get("ai_review_decision_validation_blocked_rows") or 0)
        lines.append(f"AI 复核决策校验: {ai_review_validation_rows} 行 / 阻断 {blocked_rows}")
    ai_review_round_status = str(extra.get("ai_review_round_status") or "").strip()
    if ai_review_round_status:
        rankable_rows = int(extra.get("ai_review_round_rankable_rows") or 0)
        lines.append(f"AI 复核轮次: {_snapshot_value_label(ai_review_round_status)} / 可排名 {rankable_rows}")
    ai_review_effect = str(extra.get("ai_review_ranking_effect_status") or "").strip()
    if ai_review_effect:
        promoted_rows = int(extra.get("ai_review_ranking_promoted_rows") or 0)
        removed_rows = int(extra.get("ai_review_ranking_removed_rows") or 0)
        lines.append(f"AI 复核排名变化: {_snapshot_value_label(ai_review_effect)} / 提升 {promoted_rows}, 移除 {removed_rows}")
    ai_review_manifest_rows = int(extra.get("ai_review_artifact_manifest_rows") or 0)
    if ai_review_manifest_rows > 0:
        lines.append(f"AI 复核产物清单: {ai_review_manifest_rows} 个文件")
    if bool(extra.get("ai_review_bundle_readme_available")):
        lines.append("AI 复核包 README: 可用")
    if bool(extra.get("ai_review_artifact_bundle_available")):
        lines.append("AI 复核产物包: 可用")
    ai_review_bundle_verification_rows = int(extra.get("ai_review_bundle_verification_rows") or 0)
    if ai_review_bundle_verification_rows > 0:
        failed_rows = int(extra.get("ai_review_bundle_verification_failed_rows") or 0)
        lines.append(f"AI 复核包校验: {ai_review_bundle_verification_rows} 个文件 / 失败 {failed_rows}")
        verification_status = str(extra.get("ai_review_bundle_verification_status") or "").strip()
        if verification_status:
            lines.append(f"AI 复核包校验汇总: {_snapshot_value_label(verification_status)}")
    if bool(extra.get("ai_review_bundle_certificate_available")):
        lines.append("AI 复核包证书: 可用")
    ai_review_outcomes = int(extra.get("ai_review_decision_outcome_rows") or 0)
    if ai_review_outcomes > 0:
        lines.append(f"AI 复核决策结果: {ai_review_outcomes} 行")
    ai_review_template_rows = int(extra.get("ai_review_decision_template_rows") or 0)
    if ai_review_template_rows > 0:
        lines.append(f"AI 复核决策模板行数: {ai_review_template_rows}")
    ai_influence = str(extra.get("ai_influence_level") or "").strip()
    if ai_influence:
        top_ai_residues = str(extra.get("top_pocket_ai_residues") or "none").strip()
        lines.append(f"AI 排名影响: {_snapshot_value_label(ai_influence)} / Top 口袋 AI 残基 {top_ai_residues or '无'}")
    ai_supported = int(extra.get("ai_evidence_audit_supported_count") or 0)
    ai_review = int(extra.get("ai_evidence_audit_review_count") or 0)
    if ai_supported > 0 or ai_review > 0:
        lines.append(f"AI 证据审计: 已支持 {ai_supported}, 复核 {ai_review}")
    ai_review_queue_rows = int(extra.get("ai_evidence_review_queue_rows") or 0)
    if ai_review_queue_rows > 0:
        top_fix = str(extra.get("top_ai_review_fix_type") or "-").strip()
        lines.append(f"AI 证据复核队列: {ai_review_queue_rows} 行 / Top 修复项 {_snapshot_value_label(top_fix)}")
    ai_followup_rows = int(extra.get("ai_followup_plan_rows") or 0)
    if ai_followup_rows > 0:
        lines.append(f"AI 后续取证计划: {ai_followup_rows} 行")
        top_query = str(extra.get("top_ai_followup_query") or "").strip()
        if top_query:
            lines.append(f"Top AI 后续检索词: {top_query}")
    residue_consensus_rows = int(extra.get("residue_evidence_consensus_rows") or 0)
    if residue_consensus_rows > 0:
        top_anchor = str(extra.get("top_residue_consensus_anchor") or "-").strip()
        top_tier = str(extra.get("top_residue_consensus_tier") or "-").strip()
        top_score = format_energy_value(extra.get("top_residue_consensus_score"))
        lines.append(f"Residue evidence consensus: {residue_consensus_rows} rows / top {top_anchor or '-'} ({top_tier or '-'}, score {top_score})")
    pocket_consensus_rows = int(extra.get("pocket_consensus_coverage_rows") or 0)
    if pocket_consensus_rows > 0:
        top_pocket = str(extra.get("top_pocket_consensus_coverage_id") or "-").strip()
        top_label = str(extra.get("top_pocket_consensus_label") or "-").strip()
        anchor_count = int(extra.get("top_pocket_consensus_anchor_count") or 0)
        best_score = format_energy_value(extra.get("top_pocket_consensus_best_score"))
        lines.append(
            f"Pocket consensus coverage: {pocket_consensus_rows} rows / top {top_pocket or '-'} ({top_label or '-'}, anchors {anchor_count}, score {best_score})"
        )
    benchmark_reference_candidate_rows = int(extra.get("pocket_benchmark_reference_candidate_rows") or 0)
    if benchmark_reference_candidate_rows > 0:
        import_status = str(extra.get("pocket_benchmark_reference_import_status") or "-").strip()
        provisional_status = "yes" if bool(extra.get("pocket_benchmark_reference_is_provisional")) else "no"
        lines.append(
            f"Benchmark reference candidate: {benchmark_reference_candidate_rows} rows / import {import_status or '-'} / provisional used {provisional_status}"
        )
    benchmark_reference_source_mode = str(extra.get("pocket_benchmark_reference_source_mode") or "").strip()
    benchmark_reference_rows_for_source = int(extra.get("pocket_benchmark_reference_rows") or 0)
    if benchmark_reference_source_mode or benchmark_reference_rows_for_source > 0:
        provisional_status = "yes" if bool(extra.get("pocket_benchmark_reference_is_provisional")) else "no"
        reviewed_candidate_status = "yes" if bool(extra.get("pocket_benchmark_reference_is_reviewed_candidate")) else "no"
        lines.append(
            f"Benchmark reference source: {benchmark_reference_source_mode or '-'} / provisional {provisional_status} / reviewed candidate {reviewed_candidate_status}"
        )
    benchmark_reference_source_audit_rows = int(extra.get("pocket_benchmark_reference_source_audit_rows") or 0)
    if benchmark_reference_source_audit_rows > 0:
        source_claim_status = str(extra.get("pocket_benchmark_reference_source_claim_status") or "-").strip()
        independent_claim_status = str(extra.get("pocket_benchmark_reference_source_independent_claim_status") or "-").strip()
        provisional_rows = int(extra.get("pocket_benchmark_reference_source_provisional_rows") or 0)
        reviewed_candidate_rows = int(extra.get("pocket_benchmark_reference_source_reviewed_candidate_rows") or 0)
        lines.append(
            f"Benchmark reference source audit: {benchmark_reference_source_audit_rows} rows / claim status {source_claim_status or '-'} / independent claim {independent_claim_status or '-'} / provisional rows {provisional_rows} / reviewed rows {reviewed_candidate_rows}"
        )
    benchmark_reference_source_audit_summary_rows = int(extra.get("pocket_benchmark_reference_source_audit_summary_rows") or 0)
    if benchmark_reference_source_audit_summary_rows > 0:
        source_summary_status = str(extra.get("pocket_benchmark_reference_source_audit_summary_status") or "-").strip()
        source_summary_independent_status = str(
            extra.get("pocket_benchmark_reference_source_audit_summary_independent_claim_status") or "-"
        ).strip()
        lines.append(
            f"Benchmark reference source audit summary: {benchmark_reference_source_audit_summary_rows} rows / top status {source_summary_status or '-'} / independent claim {source_summary_independent_status or '-'}"
        )
    source_audit_action_rows = int(extra.get("pocket_benchmark_reference_source_audit_action_queue_rows") or 0)
    if source_audit_action_rows > 0:
        source_audit_blockers = int(extra.get("pocket_benchmark_reference_source_audit_action_queue_blocker_rows") or 0)
        source_audit_review = int(extra.get("pocket_benchmark_reference_source_audit_action_queue_review_rows") or 0)
        lines.append(
            f"Benchmark reference source audit action queue: {source_audit_action_rows} rows / blockers {source_audit_blockers} / review {source_audit_review}"
        )
    source_audit_case_rows = int(extra.get("pocket_benchmark_reference_source_audit_case_summary_rows") or 0)
    if source_audit_case_rows > 0:
        source_audit_blocked_cases = int(extra.get("pocket_benchmark_reference_source_audit_case_summary_blocked_cases") or 0)
        source_audit_review_cases = int(extra.get("pocket_benchmark_reference_source_audit_case_summary_review_cases") or 0)
        lines.append(
            f"Benchmark reference source audit cases: {source_audit_case_rows} rows / blocked {source_audit_blocked_cases} / review {source_audit_review_cases}"
        )
    source_audit_case_decision_template_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_template_rows") or 0
    )
    if source_audit_case_decision_template_rows > 0:
        lines.append(
            f"Benchmark reference source audit case decision template: {source_audit_case_decision_template_rows} rows"
        )
    source_audit_case_decision_rows = int(extra.get("pocket_benchmark_reference_source_audit_case_decision_rows") or 0)
    if source_audit_case_decision_rows > 0:
        source_audit_case_decision_blocked = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_validation_blocked_rows") or 0
        )
        lines.append(
            f"Benchmark reference source audit case decisions: {source_audit_case_decision_rows} rows / validation blocked {source_audit_case_decision_blocked}"
        )
    source_audit_case_decision_outcome_summary_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_summary_rows") or 0
    )
    if source_audit_case_decision_outcome_summary_rows > 0:
        outcome_summary_status = str(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_summary_status") or "-"
        )
        outcome_summary_open_cases = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_summary_open_cases") or 0
        )
        lines.append(
            f"Benchmark reference source audit case decision outcome summary: {source_audit_case_decision_outcome_summary_rows} rows / status {outcome_summary_status} / open {outcome_summary_open_cases}"
        )
    source_audit_case_decision_closure_queue_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_closure_queue_rows") or 0
    )
    if source_audit_case_decision_closure_queue_rows > 0:
        closure_queue_blockers = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_closure_queue_blocker_rows") or 0
        )
        closure_queue_review = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_closure_queue_review_rows") or 0
        )
        closure_queue_top_status = str(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_closure_queue_top_status") or "-"
        )
        lines.append(
            f"Benchmark reference source audit case decision closure queue: {source_audit_case_decision_closure_queue_rows} rows / blockers {closure_queue_blockers} / review {closure_queue_review} / top {closure_queue_top_status}"
        )
    source_audit_case_decision_readiness_impact_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_rows") or 0
    )
    if source_audit_case_decision_readiness_impact_rows > 0:
        impact_cleared = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_cleared_rows") or 0
        )
        impact_open = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_open_rows") or 0
        )
        lines.append(
            f"Benchmark reference source audit case decision readiness impact: {source_audit_case_decision_readiness_impact_rows} rows / cleared {impact_cleared} / open {impact_open}"
        )
    source_audit_case_decision_readiness_impact_summary_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_rows") or 0
    )
    if source_audit_case_decision_readiness_impact_summary_rows > 0:
        impact_summary_status = str(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_status") or "-"
        )
        impact_summary_open = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases") or 0
        )
        impact_summary_delta = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_net_blocker_delta") or 0
        )
        lines.append(
            f"Benchmark reference source audit case decision readiness impact summary: {source_audit_case_decision_readiness_impact_summary_rows} rows / status {impact_summary_status} / open {impact_summary_open} / net blocker delta {impact_summary_delta}"
        )
    if bool(extra.get("pocket_benchmark_reference_source_audit_case_decision_closure_checklist_available")):
        lines.append("Benchmark reference source audit case decision closure checklist: available")
    source_audit_case_decision_outcome_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_rows") or 0
    )
    if source_audit_case_decision_outcome_rows > 0:
        outcome_blocked = int(extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_blocked_rows") or 0)
        outcome_pending = int(extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_pending_rows") or 0)
        outcome_cleared = int(extra.get("pocket_benchmark_reference_source_audit_case_decision_outcome_cleared_rows") or 0)
        lines.append(
            f"Benchmark reference source audit case decision outcomes: {source_audit_case_decision_outcome_rows} rows / blocked {outcome_blocked} / pending {outcome_pending} / cleared {outcome_cleared}"
        )
    if bool(extra.get("pocket_benchmark_reference_source_audit_case_checklist_available")):
        lines.append("Benchmark reference source audit case checklist: available")
    if bool(extra.get("pocket_benchmark_reference_source_audit_checklist_available")):
        lines.append("Benchmark reference source audit checklist: available")
    benchmark_reference_candidate_review_rows = int(extra.get("pocket_benchmark_reference_candidate_review_rows") or 0)
    if benchmark_reference_candidate_review_rows > 0:
        p1_rows = int(extra.get("pocket_benchmark_reference_candidate_review_p1_rows") or 0)
        p2_rows = int(extra.get("pocket_benchmark_reference_candidate_review_p2_rows") or 0)
        checklist_status = "available" if bool(extra.get("pocket_benchmark_reference_candidate_review_checklist_available")) else "not available"
        lines.append(
            f"Benchmark reference candidate review: {benchmark_reference_candidate_review_rows} rows / P1 {p1_rows} / P2 {p2_rows} / checklist {checklist_status}"
        )
    benchmark_reference_candidate_decisions = int(extra.get("pocket_benchmark_reference_candidate_review_decision_rows") or 0)
    if benchmark_reference_candidate_decisions > 0:
        blocked_rows = int(extra.get("pocket_benchmark_reference_candidate_review_decision_validation_blocked_rows") or 0)
        accepted_actions = int(extra.get("pocket_benchmark_reference_candidate_review_outcome_accepted_rows") or 0)
        accepted_references = int(extra.get("pocket_benchmark_reference_candidate_accepted_rows") or 0)
        lines.append(
            f"Benchmark reference candidate review decisions: {benchmark_reference_candidate_decisions} rows / validation blocked {blocked_rows} / accepted actions {accepted_actions} / accepted references {accepted_references}"
        )
    benchmark_reference_rows = int(extra.get("pocket_benchmark_reference_rows") or 0)
    if benchmark_reference_rows > 0:
        top1_coverage = format_energy_value(extra.get("pocket_benchmark_top1_coverage"))
        top3_coverage = format_energy_value(extra.get("pocket_benchmark_top3_coverage"))
        top1_status = str(extra.get("pocket_benchmark_top1_status") or "-").strip()
        top3_status = str(extra.get("pocket_benchmark_top3_status") or "-").strip()
        lines.append(
            f"Catalytic pocket benchmark: references {benchmark_reference_rows} / Top-1 {top1_coverage} ({top1_status or '-'}) / Top-3 {top3_coverage} ({top3_status or '-'})"
        )
    benchmark_template_rows = int(extra.get("pocket_benchmark_reference_template_rows") or 0)
    if benchmark_template_rows > 0:
        notes_status = "available" if bool(extra.get("pocket_benchmark_reference_template_notes_available")) else "not available"
        lines.append(f"Benchmark reference template: {benchmark_template_rows} rows / notes {notes_status}")
    benchmark_quality_rows = int(extra.get("pocket_benchmark_reference_quality_issue_rows") or 0)
    if benchmark_quality_rows > 0:
        quality_summary_rows = int(extra.get("pocket_benchmark_reference_quality_summary_rows") or 0)
        checklist_status = "available" if bool(extra.get("pocket_benchmark_reference_quality_checklist_available")) else "not available"
        lines.append(
            f"Benchmark reference curation quality: {benchmark_quality_rows} issues / summary {quality_summary_rows} rows / checklist {checklist_status}"
        )
    benchmark_structure_validation_rows = int(extra.get("pocket_benchmark_reference_structure_validation_issue_rows") or 0)
    if benchmark_structure_validation_rows > 0:
        validation_summary_rows = int(extra.get("pocket_benchmark_reference_structure_validation_summary_rows") or 0)
        checklist_status = "available" if bool(extra.get("pocket_benchmark_reference_structure_validation_checklist_available")) else "not available"
        lines.append(
            f"Benchmark reference structure validation: {benchmark_structure_validation_rows} issues / summary {validation_summary_rows} rows / checklist {checklist_status}"
        )
    benchmark_readiness_rows = int(extra.get("pocket_benchmark_reference_readiness_summary_rows") or 0)
    if benchmark_readiness_rows > 0:
        readiness_status = str(extra.get("pocket_benchmark_reference_readiness_status") or "-").strip()
        blocker_rows = int(extra.get("pocket_benchmark_reference_readiness_blocker_rows") or 0)
        review_rows = int(extra.get("pocket_benchmark_reference_readiness_review_rows") or 0)
        source_audit_rows = int(extra.get("pocket_benchmark_reference_readiness_source_audit_issue_rows") or 0)
        queue_rows = int(extra.get("pocket_benchmark_reference_readiness_queue_rows") or 0)
        checklist_status = "available" if bool(extra.get("pocket_benchmark_reference_readiness_checklist_available")) else "not available"
        lines.append(
            f"Benchmark reference readiness: {readiness_status or '-'} / blockers {blocker_rows} / review {review_rows} / source audit {source_audit_rows} / queue {queue_rows} / checklist {checklist_status}"
        )
    readiness_case_rows = int(extra.get("pocket_benchmark_reference_readiness_case_summary_rows") or 0)
    if readiness_case_rows > 0:
        blocked_cases = int(extra.get("pocket_benchmark_reference_readiness_blocked_cases") or 0)
        review_cases = int(extra.get("pocket_benchmark_reference_readiness_review_cases") or 0)
        lines.append(
            f"Benchmark reference readiness cases: {readiness_case_rows} rows / blocked {blocked_cases} / review {review_cases}"
        )
    benchmark_interpretation_rows = int(extra.get("pocket_benchmark_interpretation_rows") or 0)
    if benchmark_interpretation_rows > 0:
        top1_claim = str(extra.get("pocket_benchmark_top1_claim_status") or "-").strip()
        top3_claim = str(extra.get("pocket_benchmark_top3_claim_status") or "-").strip()
        lines.append(
            f"Benchmark interpretation: {benchmark_interpretation_rows} rows / Top-1 claim {top1_claim or '-'} / Top-3 claim {top3_claim or '-'}"
        )
    benchmark_case_interpretation_rows = int(extra.get("pocket_benchmark_case_interpretation_rows") or 0)
    if benchmark_case_interpretation_rows > 0:
        blocked_rows = int(extra.get("pocket_benchmark_case_interpretation_blocked_rows") or 0)
        review_rows = int(extra.get("pocket_benchmark_case_interpretation_review_rows") or 0)
        lines.append(
            f"Benchmark case interpretation: {benchmark_case_interpretation_rows} rows / blocked {blocked_rows} / review {review_rows}"
        )
    benchmark_case_interpretation_matrix_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_rows") or 0)
    if benchmark_case_interpretation_matrix_rows > 0:
        blocked_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_blocked_rows") or 0)
        review_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_review_rows") or 0)
        lines.append(
            f"Benchmark case interpretation matrix: {benchmark_case_interpretation_matrix_rows} rows / blocked {blocked_rows} / review {review_rows}"
        )
    benchmark_case_interpretation_matrix_summary_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_summary_rows") or 0)
    if benchmark_case_interpretation_matrix_summary_rows > 0:
        summary_status = str(extra.get("pocket_benchmark_case_interpretation_matrix_summary_status") or "-").strip()
        usable_cases = int(extra.get("pocket_benchmark_case_interpretation_matrix_summary_usable_cases") or 0)
        lines.append(
            f"Benchmark case interpretation matrix summary: {summary_status or '-'} / usable {usable_cases}"
        )
    benchmark_case_interpretation_matrix_queue_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_queue_rows") or 0)
    if benchmark_case_interpretation_matrix_queue_rows > 0:
        blocker_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_queue_blocker_rows") or 0)
        review_rows = int(extra.get("pocket_benchmark_case_interpretation_matrix_queue_review_rows") or 0)
        lines.append(
            f"Benchmark case interpretation matrix queue: {benchmark_case_interpretation_matrix_queue_rows} rows / blockers {blocker_rows} / review {review_rows}"
        )
    benchmark_dataset_interpretation_rows = int(extra.get("pocket_benchmark_dataset_interpretation_rows") or 0)
    if benchmark_dataset_interpretation_rows > 0:
        blocked_rows = int(extra.get("pocket_benchmark_dataset_interpretation_blocked_rows") or 0)
        review_rows = int(extra.get("pocket_benchmark_dataset_interpretation_review_rows") or 0)
        lines.append(
            f"Benchmark dataset interpretation: {benchmark_dataset_interpretation_rows} rows / blocked {blocked_rows} / review {review_rows}"
        )
    source_dataset_impact_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_rows") or 0
    )
    if source_dataset_impact_rows > 0:
        blocker_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_blocker_rows") or 0
        )
        review_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_review_rows") or 0
        )
        mismatch_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_mismatch_rows") or 0
        )
        lines.append(
            f"Benchmark source-audit decision dataset impact: {source_dataset_impact_rows} rows / blockers {blocker_rows} / review {review_rows} / mismatch {mismatch_rows}"
        )
    source_dataset_impact_case_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_rows") or 0
    )
    if source_dataset_impact_case_rows > 0:
        blocker_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_blocker_rows") or 0
        )
        review_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_review_rows") or 0
        )
        mismatch_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_mismatch_rows") or 0
        )
        checklist_status = (
            "available"
            if bool(extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_available"))
            else "not available"
        )
        report_status = (
            "available"
            if bool(extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_report_available"))
            else "not available"
        )
        lines.append(
            f"Benchmark source-audit decision dataset impact cases: {source_dataset_impact_case_rows} rows / blockers {blocker_rows} / review {review_rows} / mismatch {mismatch_rows} / checklist {checklist_status} / report {report_status}"
        )
    source_dataset_impact_action_queue_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_rows") or 0
    )
    if source_dataset_impact_action_queue_rows > 0:
        blocker_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_blocker_rows") or 0
        )
        review_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_review_rows") or 0
        )
        mismatch_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_mismatch_rows") or 0
        )
        lines.append(
            f"Benchmark source-audit decision dataset impact action queue: {source_dataset_impact_action_queue_rows} rows / blockers {blocker_rows} / review {review_rows} / mismatch {mismatch_rows}"
        )
    source_dataset_impact_action_summary_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_rows") or 0
    )
    if source_dataset_impact_action_summary_rows > 0:
        action_count = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_action_count") or 0
        )
        p0_rows = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_p0_rows") or 0
        )
        mismatch_count = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_mismatch_count") or 0
        )
        top_priority = str(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_priority") or "-"
        ).strip()
        top_source_impact = str(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_action_queue_summary_top_source_impact") or "-"
        ).strip()
        lines.append(
            f"Benchmark source-audit decision dataset impact action summary: {source_dataset_impact_action_summary_rows} rows / actions {action_count} / P0 groups {p0_rows} / mismatches {mismatch_count} / top {top_priority or '-'} {top_source_impact or '-'}"
        )
    source_dataset_impact_artifact_rows = int(
        extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_rows") or 0
    )
    if source_dataset_impact_artifact_rows > 0:
        artifact_bytes = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_bytes") or 0
        )
        artifact_hashes = int(
            extra.get("pocket_benchmark_reference_source_audit_case_decision_dataset_impact_artifact_manifest_hash_rows") or 0
        )
        lines.append(
            f"Benchmark source-audit decision dataset impact artifacts: {source_dataset_impact_artifact_rows} files / bytes {artifact_bytes} / hashes {artifact_hashes}"
        )
    benchmark_dataset_interpretation_queue_rows = int(extra.get("pocket_benchmark_dataset_interpretation_queue_rows") or 0)
    if benchmark_dataset_interpretation_queue_rows > 0:
        blocker_rows = int(extra.get("pocket_benchmark_dataset_interpretation_queue_blocker_rows") or 0)
        review_rows = int(extra.get("pocket_benchmark_dataset_interpretation_queue_review_rows") or 0)
        checklist_status = "available" if bool(extra.get("pocket_benchmark_dataset_interpretation_checklist_available")) else "not available"
        report_status = "available" if bool(extra.get("pocket_benchmark_dataset_interpretation_report_available")) else "not available"
        lines.append(
            f"Benchmark dataset interpretation queue: {benchmark_dataset_interpretation_queue_rows} rows / blockers {blocker_rows} / review {review_rows} / checklist {checklist_status} / report {report_status}"
        )
    elif bool(extra.get("pocket_benchmark_dataset_interpretation_report_available")):
        lines.append("Benchmark dataset interpretation report: available")
    benchmark_case_summary = extra.get("pocket_benchmark_case_summary") or []
    benchmark_dataset_rows = int(extra.get("pocket_benchmark_dataset_summary_rows") or 0)
    if benchmark_dataset_rows > 0 or benchmark_case_summary:
        case_ids = {
            str(row.get("benchmark_id") or "").strip()
            for row in benchmark_case_summary
            if isinstance(row, dict) and str(row.get("benchmark_id") or "").strip()
        }
        case_count = len(case_ids) if case_ids else int(extra.get("pocket_benchmark_case_summary_rows") or 0)
        lines.append(f"Catalytic benchmark dataset: cases {case_count} / summary rows {benchmark_dataset_rows}")
    benchmark_variant_rows = int(extra.get("pocket_benchmark_variant_comparison_rows") or 0)
    if benchmark_variant_rows > 0:
        lines.append(f"Catalytic benchmark variants: {benchmark_variant_rows} rows")
    benchmark_variant_case_rows = int(extra.get("pocket_benchmark_variant_case_comparison_rows") or 0)
    benchmark_variant_dataset_rows = int(extra.get("pocket_benchmark_variant_dataset_comparison_rows") or 0)
    if benchmark_variant_case_rows > 0 or benchmark_variant_dataset_rows > 0:
        lines.append(
            f"Catalytic benchmark variant cases: {benchmark_variant_case_rows} rows / dataset rows {benchmark_variant_dataset_rows}"
        )
    benchmark_variant_detail_rows = int(extra.get("pocket_benchmark_variant_detail_comparison_rows") or 0)
    if benchmark_variant_detail_rows > 0:
        lines.append(f"Catalytic benchmark variant residues: {benchmark_variant_detail_rows} rows")
    benchmark_remediation_rows = int(extra.get("pocket_benchmark_variant_remediation_rows") or 0)
    if benchmark_remediation_rows > 0:
        remediation_summary_rows = int(extra.get("pocket_benchmark_variant_remediation_summary_rows") or 0)
        checklist_status = "available" if bool(extra.get("pocket_benchmark_variant_remediation_checklist_available")) else "not available"
        lines.append(
            f"Catalytic benchmark remediation queue: {benchmark_remediation_rows} rows / summary {remediation_summary_rows} rows / checklist {checklist_status}"
        )
    consensus_rerank_rows = int(extra.get("consensus_rerank_suggestion_rows") or 0)
    if consensus_rerank_rows > 0:
        rerank_pocket = str(extra.get("top_consensus_rerank_pocket_id") or "-").strip()
        rerank_status = str(extra.get("top_consensus_rerank_status") or "-").strip()
        rerank_delta = int(extra.get("top_consensus_rerank_rank_delta") or 0)
        lines.append(
            f"Consensus rerank suggestions: {consensus_rerank_rows} rows / top {rerank_pocket or '-'} ({rerank_status or '-'}, rank delta {rerank_delta:+d})"
        )
    consensus_preview_rows = int(extra.get("consensus_rerank_preview_rows") or 0)
    if consensus_preview_rows > 0:
        preview_pocket = str(extra.get("top_consensus_preview_pocket_id") or "-").strip()
        preview_decision = str(extra.get("top_consensus_preview_decision") or "-").strip()
        preview_delta = int(extra.get("top_consensus_preview_rank_delta") or 0)
        preview_score = format_energy_value(extra.get("top_consensus_preview_score"))
        lines.append(
            f"Consensus rerank preview: {consensus_preview_rows} rows / top {preview_pocket or '-'} ({preview_decision or '-'}, rank delta {preview_delta:+d}, score {preview_score})"
        )
    rerank_policy = str(extra.get("consensus_rerank_policy_status") or "").strip()
    if rerank_policy:
        changed_rows = int(extra.get("consensus_rerank_policy_changed_rows") or 0)
        blocked_rows = int(extra.get("consensus_rerank_policy_blocked_rows") or 0)
        lines.append(f"Consensus rerank policy gate: {rerank_policy} / changed {changed_rows}, blocked {blocked_rows}")
    rerank_action_rows = int(extra.get("consensus_rerank_action_queue_rows") or 0)
    if rerank_action_rows > 0:
        action_pocket = str(extra.get("top_consensus_rerank_action_pocket_id") or "-").strip()
        issue_type = str(extra.get("top_consensus_rerank_issue_type") or "-").strip()
        issue_severity = str(extra.get("top_consensus_rerank_issue_severity") or "-").strip()
        lines.append(
            f"Consensus rerank action queue: {rerank_action_rows} rows / top {action_pocket or '-'} ({issue_type or '-'}, {issue_severity or '-'})"
        )
    if bool(extra.get("consensus_rerank_action_checklist_available")):
        lines.append("Consensus rerank action checklist: available")
    rerank_apply_rows = int(extra.get("consensus_rerank_apply_simulation_rows") or 0)
    if rerank_apply_rows > 0:
        apply_pocket = str(extra.get("top_consensus_rerank_apply_pocket_id") or "-").strip()
        apply_status = str(extra.get("top_consensus_rerank_apply_status") or "-").strip()
        apply_delta = int(extra.get("top_consensus_rerank_apply_rank_delta") or 0)
        lines.append(
            f"Consensus rerank apply simulation: {rerank_apply_rows} rows / top {apply_pocket or '-'} ({apply_status or '-'}, rank delta {apply_delta:+d})"
        )
    rerank_delta_rows = int(extra.get("consensus_rerank_simulation_delta_rows") or 0)
    if rerank_delta_rows > 0:
        delta_pocket = str(extra.get("top_consensus_rerank_delta_pocket_id") or "-").strip()
        delta_type = str(extra.get("top_consensus_rerank_delta_change_type") or "-").strip()
        delta_rank = int(extra.get("top_consensus_rerank_delta_rank_delta") or 0)
        lines.append(
            f"Consensus rerank simulation delta: {rerank_delta_rows} rows / top {delta_pocket or '-'} ({delta_type or '-'}, rank delta {delta_rank:+d})"
        )
    scorecard_rows = int(extra.get("consensus_rerank_precision_scorecard_rows") or 0)
    if scorecard_rows > 0:
        precision_status = str(extra.get("consensus_rerank_precision_status") or "-").strip()
        precision_score = int(extra.get("consensus_rerank_precision_score") or 0)
        positive_rows = int(extra.get("consensus_rerank_positive_signal_rows") or 0)
        blocker_rows = int(extra.get("consensus_rerank_open_blocker_rows") or 0)
        lines.append(
            f"Consensus rerank precision scorecard: {precision_status or '-'} / score {precision_score} / positive {positive_rows}, blockers {blocker_rows}"
        )
    guardrail_rows = int(extra.get("consensus_rerank_precision_guardrail_rows") or 0)
    if guardrail_rows > 0:
        guardrail_status = str(extra.get("consensus_rerank_guardrail_status") or "-").strip()
        guardrail_decision = str(extra.get("consensus_rerank_guardrail_decision") or "-").strip()
        apply_mode = str(extra.get("consensus_rerank_guardrail_apply_mode") or "-").strip()
        lines.append(
            f"Consensus rerank precision guardrail: {guardrail_status or '-'} / decision {guardrail_decision or '-'}, mode {apply_mode or '-'}"
        )
    if bool(extra.get("consensus_rerank_guardrail_report_available")):
        lines.append("Consensus rerank precision guardrail report: available")
    manifest_rows = int(extra.get("consensus_rerank_guardrail_artifact_manifest_rows") or 0)
    if manifest_rows > 0 or bool(extra.get("consensus_rerank_guardrail_handoff_zip_available")):
        bundle_status = "available" if bool(extra.get("consensus_rerank_guardrail_handoff_zip_available")) else "not available"
        lines.append(f"Consensus rerank guardrail handoff bundle: {bundle_status} / manifest {manifest_rows} files")
    verification_status = str(extra.get("consensus_rerank_guardrail_bundle_verification_status") or "").strip()
    if verification_status:
        verification_rows = int(extra.get("consensus_rerank_guardrail_bundle_verification_rows") or 0)
        failed_rows = int(extra.get("consensus_rerank_guardrail_bundle_verification_failed_rows") or 0)
        lines.append(
            f"Consensus rerank guardrail bundle verification: {verification_status} / files {verification_rows}, failed {failed_rows}"
        )
    if bool(extra.get("consensus_rerank_guardrail_handoff_certificate_available")):
        lines.append("Consensus rerank guardrail handoff certificate: available")
    release_template_rows = int(extra.get("consensus_rerank_release_decision_template_rows") or 0)
    if release_template_rows > 0:
        lines.append(f"Consensus rerank release decision template: {release_template_rows} rows")
    release_decision_rows = int(extra.get("consensus_rerank_release_decision_rows") or 0)
    release_decision_status = str(extra.get("consensus_rerank_release_decision_status") or "").strip()
    if release_decision_rows > 0 or (release_decision_status and release_decision_status != "not-uploaded"):
        lines.append(f"Consensus rerank release decisions: {release_decision_rows} rows / status {release_decision_status or '-'}")
    release_validation_rows = int(extra.get("consensus_rerank_release_decision_validation_rows") or 0)
    if release_validation_rows > 0:
        blocked_rows = int(extra.get("consensus_rerank_release_decision_blocked_rows") or 0)
        lines.append(f"Consensus rerank release decision validation: {release_validation_rows} rows / blocked {blocked_rows}")
    release_review_status = str(extra.get("consensus_rerank_release_review_status") or "").strip()
    if release_review_status:
        release_allowed = "yes" if bool(extra.get("consensus_rerank_release_allowed")) else "no"
        lines.append(f"Consensus rerank release review: {release_review_status} / allowed {release_allowed}")
    release_apply_plan_rows = int(extra.get("consensus_rerank_release_apply_plan_rows") or 0)
    if release_apply_plan_rows > 0:
        top_apply_pocket = str(extra.get("top_consensus_rerank_release_apply_pocket_id") or "-").strip()
        apply_status = str(extra.get("top_consensus_rerank_release_apply_status") or "-").strip()
        lines.append(
            f"Consensus rerank release apply plan: {release_apply_plan_rows} rows / top {top_apply_pocket or '-'} ({apply_status or '-'})"
        )
    if bool(extra.get("consensus_rerank_release_apply_report_available")):
        lines.append("Consensus rerank release apply report: available")
    execution_template_rows = int(extra.get("consensus_rerank_release_execution_template_rows") or 0)
    if execution_template_rows > 0:
        lines.append(f"Consensus rerank release execution template: {execution_template_rows} rows")
    execution_receipt_rows = int(extra.get("consensus_rerank_release_execution_receipt_rows") or 0)
    execution_receipt_status = str(extra.get("consensus_rerank_release_execution_receipt_status") or "").strip()
    if execution_receipt_rows > 0 or (execution_receipt_status and execution_receipt_status != "not-uploaded"):
        lines.append(f"Consensus rerank release execution receipt: {execution_receipt_rows} rows / status {execution_receipt_status or '-'}")
    execution_validation_rows = int(extra.get("consensus_rerank_release_execution_validation_rows") or 0)
    if execution_validation_rows > 0:
        blocked_rows = int(extra.get("consensus_rerank_release_execution_blocked_rows") or 0)
        lines.append(f"Consensus rerank release execution validation: {execution_validation_rows} rows / blocked {blocked_rows}")
    execution_review_status = str(extra.get("consensus_rerank_release_execution_review_status") or "").strip()
    if execution_review_status:
        execution_complete = "yes" if bool(extra.get("consensus_rerank_release_execution_complete")) else "no"
        lines.append(f"Consensus rerank release execution: {execution_review_status} / complete {execution_complete}")
    if bool(extra.get("consensus_rerank_release_execution_report_available")):
        lines.append("Consensus rerank release execution report: available")
    if bool(extra.get("consensus_rerank_release_closure_certificate_available")):
        lines.append("Consensus rerank release closure certificate: available")
    closure_ledger_rows = int(extra.get("consensus_rerank_release_closure_ledger_rows") or 0)
    if closure_ledger_rows > 0:
        blocked_rows = int(extra.get("consensus_rerank_release_closure_ledger_blocked_rows") or 0)
        lines.append(f"Consensus rerank release closure ledger: {closure_ledger_rows} rows / blocked {blocked_rows}")
    closure_readiness_status = str(extra.get("consensus_rerank_release_closure_readiness_status") or "").strip()
    if closure_readiness_status:
        release_closed = "yes" if bool(extra.get("consensus_rerank_release_closed")) else "no"
        lines.append(f"Consensus rerank release closure readiness: {closure_readiness_status} / closed {release_closed}")
    closure_blocker_rows = int(extra.get("consensus_rerank_release_closure_blocker_rows") or 0)
    if closure_blocker_rows > 0:
        blocker_type = str(extra.get("top_consensus_rerank_release_closure_blocker_type") or "-").strip()
        lines.append(f"Consensus rerank release closure blockers: {closure_blocker_rows} rows / top {blocker_type or '-'}")
    if bool(extra.get("consensus_rerank_release_closure_remediation_checklist_available")):
        lines.append("Consensus rerank release closure remediation checklist: available")
    detached_manifest_rows = int(extra.get("consensus_rerank_release_closure_detached_manifest_rows") or 0)
    if detached_manifest_rows > 0:
        lines.append(f"Consensus rerank release closure detached manifest: {detached_manifest_rows} files")
    pocket_preview = snapshot.get("pocket_summary") or []
    if pocket_preview:
        top_pocket = pocket_preview[0]
        pocket_id = top_pocket.get("pocket_id") or "-"
        rank_label = top_pocket.get("smart_rank_label") or "-"
        lines.append(f"Top 口袋: {pocket_id} ({rank_label})")
        evidence_quality = top_pocket.get("evidence_quality_label") or ""
        evidence_score = top_pocket.get("evidence_quality_score")
        if evidence_quality:
            lines.append(f"Top 口袋证据质量: {evidence_quality} ({format_energy_value(evidence_score)})")
    joint_preview = snapshot.get("joint_candidate_preview") or []
    if joint_preview:
        top_joint = joint_preview[0]
        joint_id = top_joint.get("pocket_id") or "-"
        joint_label = top_joint.get("recommendation_label") or "-"
        joint_action = top_joint.get("recommendation_action") or "-"
        if joint_action != "-":
            lines.append(f"Top 联合动作: {joint_action}")
        lines.append(f"Top 联合推荐: {joint_id} ({joint_label})")
    decision_label = str(extra.get("top_pocket_decision_label") or "").strip()
    decision_score = extra.get("top_pocket_decision_score")
    audit_status = str(extra.get("top_pocket_audit_status") or "").strip()
    if decision_label or audit_status:
        lines.append(
            f"Top 活性位点决策: {_snapshot_text_label(decision_label)} / 评分 {format_energy_value(decision_score)} / 审计 {_snapshot_text_label(audit_status)}"
        )
    precision_tier = str(extra.get("top_pocket_precision_tier") or "").strip()
    triage_action = str(extra.get("top_pocket_triage_action") or "").strip()
    if precision_tier or triage_action:
        lines.append(f"Top 口袋精度分层: {_snapshot_text_label(precision_tier)} / 动作 {_snapshot_text_label(triage_action)}")
    reliability_counts = _reliability_status_counts(extra)
    if any(reliability_counts.values()):
        lines.append(
            "口袋可靠性检查: "
            f"通过 {reliability_counts['pass']}, 复核 {reliability_counts['review']}, 缺失 {reliability_counts['missing']}"
        )
    reliability_gaps = str(extra.get("top_pocket_reliability_gaps") or "").strip()
    if reliability_gaps:
        lines.append(f"Top 口袋可靠性缺口: {_snapshot_text_label(reliability_gaps)}")
    return [_snapshot_summary_line_label(line) for line in lines]


def _svg_escape(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=False)


def _svg_multiline_text(x: int, y: int, text: str, *, font_size: int = 16, fill: str = "#0f1724", max_chars: int = 38) -> str:
    lines = textwrap.wrap(text, width=max_chars) or [""]
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else font_size + 4
        tspans.append(f'<tspan x="{x}" dy="{dy}">{_svg_escape(line)}</tspan>')
    return f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{font_size}" fill="{fill}">' + "".join(tspans) + "</text>"


def build_snapshot_svg(snapshot: dict[str, Any], *, width: int = 1280, height: int = 920) -> bytes:
    summary = snapshot.get("summary") or {}
    protein_volume = summary.get("protein_volume")
    if protein_volume is None:
        protein_volume = snapshot.get("protein_volume")
    cards = [
        ("残基总数", summary.get("residue_count", "-")),
        ("平均能量", format_energy_value(summary.get("mean_energy"))),
        ("蛋白质体积", _format_volume_text(protein_volume)),
        ("热点残基", snapshot.get("hotspot_count", 0)),
        ("口袋条目", snapshot.get("pocket_rows", 0)),
        ("能量来源", summary.get("energy_source") or "未标注"),
    ]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append(f'<rect width="100%" height="100%" fill="#f6f9fc"/>')
    parts.append(f'<defs><linearGradient id="headerGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#2b6cb0"/><stop offset="100%" stop-color="#805ad5"/></linearGradient></defs>')
    parts.append(f'<rect x="32" y="28" width="{width - 64}" height="130" rx="20" fill="url(#headerGrad)"/>')
    parts.append(f'<text x="58" y="74" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="28" fill="#ffffff" font-weight="700">{_svg_escape(snapshot.get("title", "ProteinInsight 分析快照"))}</text>')
    parts.append(f'<text x="58" y="104" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="14" fill="#eff6ff">{_svg_escape(snapshot.get("generated_at", ""))}</text>')
    parts.append(_svg_multiline_text(58, 132, f"能量来源：{summary.get('energy_source') or '未标注'}；平均能量：{format_energy_value(summary.get('mean_energy'))}", font_size=15, fill="#eff6ff", max_chars=80))

    card_width = (width - 84) / 3
    card_height = 84
    card_y = 186
    for index, (label, value) in enumerate(cards):
        row = index // 3
        col = index % 3
        x = 32 + col * (card_width + 10)
        y = card_y + row * (card_height + 12)
        parts.append(f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="16" fill="#ffffff" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 28}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="13" fill="#64748b">{_svg_escape(label)}</text>')
        parts.append(f'<text x="{x + 16}" y="{y + 58}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="22" fill="#0f1724" font-weight="700">{_svg_escape(value)}</text>')

    sections_y = 332
    section_width = (width - 84) / 2
    section_height = 260
    sections = [
        ("自动分析摘要", snapshot_to_summary_lines(snapshot)[:10]),
        ("热点与口袋预览", [
            *[f"热点: {item.get('label', '-')}; ΔG={item.get('delta_total', item.get('energy', '-'))}" for item in snapshot.get("top_hotspots", [])[:4]],
            *[
                "口袋: {pocket_id}{route_text}{method_text} | votes={vote_count}, volume={volume}, score={score}, hotspot={hotspot}".format(
                    pocket_id=item.get("pocket_id", "-"),
                    route_text=f" | route={item.get('detection_route')}" if item.get("detection_route") else "",
                    method_text=f" | methods={item.get('consensus_methods')}" if item.get("consensus_methods") else "",
                    vote_count=item.get("method_vote_count") if item.get("method_vote_count") not in (None, "") else "-",
                    volume=item.get("volume", "-"),
                    score=item.get("score", "-"),
                    hotspot=item.get("hotspot_count", "-"),
                )
                for item in snapshot.get("pocket_summary", [])[:4]
            ],
        ]),
    ]

    for index, (title, lines) in enumerate(sections):
        x = 32 + index * (section_width + 10)
        y = sections_y
        parts.append(f'<rect x="{x}" y="{y}" width="{section_width}" height="{section_height}" rx="18" fill="#ffffff" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x + 16}" y="{y + 30}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="16" fill="#0f1724" font-weight="700">{_svg_escape(title)}</text>')
        body_y = y + 58
        if not lines:
            lines = ["无可用数据"]
        for line_index, line in enumerate(lines[:10]):
            line_y = body_y + line_index * 22
            parts.append(_svg_multiline_text(x + 16, line_y, str(line), font_size=13, fill="#334155", max_chars=48))

    bottom_y = 614
    parts.append(f'<rect x="32" y="{bottom_y}" width="{width - 64}" height="250" rx="18" fill="#ffffff" stroke="#e2e8f0"/>')
    parts.append(f'<text x="48" y="{bottom_y + 28}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="16" fill="#0f1724" font-weight="700">结果快照预览</text>')

    preview_lines = snapshot_to_summary_lines(snapshot)
    preview_lines.extend([
        f"注释行数: {snapshot.get('annotation_rows', 0)}",
        f"比较条目: {snapshot.get('comparison_rows', 0)}",
    ])
    for index, line in enumerate(preview_lines[:8]):
        parts.append(_svg_multiline_text(48, bottom_y + 58 + index * 22, line, font_size=13, fill="#334155", max_chars=90))

    parts.append(f'<text x="32" y="{height - 20}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="11" fill="#94a3b8">ProteinInsight snapshot export</text>')
    parts.append("</svg>")
    return "".join(parts).encode("utf-8")
