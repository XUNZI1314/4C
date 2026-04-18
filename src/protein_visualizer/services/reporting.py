from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


def _get_energy_column(table) -> Tuple[Optional[str], pd.Series]:
    if table is None or getattr(table, "empty", True):
        return None, pd.Series(dtype=float)

    if "delta_total_raw" in table.columns:
        series = pd.to_numeric(table["delta_total_raw"], errors="coerce").dropna()
        return "delta_total_raw", series

    for column in ("energy", "delta_total"):
        if column in table.columns:
            series = pd.to_numeric(table[column], errors="coerce").dropna()
            if not series.empty:
                return column, series

    return None, pd.Series(dtype=float)


def _get_energy_source(table) -> Optional[str]:
    if table is None or getattr(table, "empty", True):
        return None
    if "energy_source" not in table.columns:
        return None

    series = table["energy_source"].dropna().astype(str).str.strip()
    series = series[series != ""]
    if series.empty:
        return None

    values = list(dict.fromkeys(series.tolist()))
    if len(values) == 1:
        return values[0]
    return "混合来源"


def _format_residue_label(row) -> str:
    for column in ("label", "residue_label"):
        value = getattr(row, column, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    resname = str(getattr(row, "resname", "")).strip()
    chain = str(getattr(row, "chain", "")).strip()
    resid = getattr(row, "resid", None)
    if chain or resid is not None or resname:
        resid_text = "" if resid is None else str(int(resid))
        return f"{resname} {chain}{resid_text}".strip()

    return "-"


def format_energy_value(value, digits: int = 3, placeholder: str = "-") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return placeholder

    if pd.isna(numeric):
        return placeholder

    return f"{numeric:.{digits}f}"


def build_analysis_summary(energy_table) -> dict:
    residue_count = int(len(energy_table)) if energy_table is not None else 0
    column_name, series = _get_energy_column(energy_table)
    energy_source = _get_energy_source(energy_table)

    if series.empty:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "residue_count": residue_count,
            "valid_energy_count": 0,
            "energy_coverage": 0.0,
            "energy_column": column_name,
            "energy_source": energy_source,
            "min_energy": None,
            "max_energy": None,
            "mean_energy": None,
            "lowest_residue": "-",
            "highest_residue": "-",
        }

    if energy_table is not None and column_name and column_name in energy_table.columns:
        valid_rows = energy_table[pd.to_numeric(energy_table[column_name], errors="coerce").notna()].copy()
    else:
        valid_rows = energy_table.copy() if energy_table is not None else pd.DataFrame()

    if valid_rows.empty:
        valid_rows = energy_table.copy() if energy_table is not None else pd.DataFrame()

    min_index = series.idxmin()
    max_index = series.idxmax()
    min_row = valid_rows.loc[min_index] if min_index in valid_rows.index else None
    max_row = valid_rows.loc[max_index] if max_index in valid_rows.index else None

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "residue_count": residue_count,
        "valid_energy_count": int(series.count()),
        "energy_coverage": float(series.count() / residue_count) if residue_count else 0.0,
        "energy_column": column_name,
        "energy_source": energy_source,
        "min_energy": float(series.min()),
        "max_energy": float(series.max()),
        "mean_energy": float(series.mean()),
        "lowest_residue": _format_residue_label(min_row) if min_row is not None else "-",
        "highest_residue": _format_residue_label(max_row) if max_row is not None else "-",
    }


def build_text_report(energy_table, pocket_summary=None, joint_candidate_table=None) -> str:
    summary = build_analysis_summary(energy_table)
    mean_text = format_energy_value(summary["mean_energy"])
    min_text = format_energy_value(summary["min_energy"])
    max_text = format_energy_value(summary["max_energy"])
    lines = [
        "蛋白质可视化分析报告",
        "=" * 30,
        f"生成时间: {summary['generated_at']}",
        f"残基总数: {summary['residue_count']}",
        f"有效能量数: {summary['valid_energy_count']}/{summary['residue_count']}",
        f"能量来源: {summary['energy_source'] or '未标注'}",
        f"最小能量: {min_text}",
        f"最大能量: {max_text}",
        f"平均能量: {mean_text}",
        f"最低能量残基: {summary['lowest_residue']}",
        f"最高能量残基: {summary['highest_residue']}",
        "",
        "残基能量明细:",
    ]

    if pocket_summary is not None and not getattr(pocket_summary, "empty", True):
        try:
            top_pocket = pocket_summary.iloc[0]
            top_pocket_id = top_pocket.get("pocket_id") or "-"
            top_rank_label = top_pocket.get("smart_rank_label") or "-"
            top_rank_score = format_energy_value(top_pocket.get("smart_rank_score"))
            top_hotspot_count = top_pocket.get("hotspot_count")
            top_hotspot_text = "-" if pd.isna(top_hotspot_count) else str(int(top_hotspot_count))
            top_reason = top_pocket.get("smart_rank_reason") or top_pocket.get("detection_route") or "-"
            top_external_exact = top_pocket.get("external_exact_match_count")
            top_external_support = top_pocket.get("external_support_mean")
            top_evidence_quality = top_pocket.get("evidence_quality_label") or "-"
            top_evidence_quality_score = top_pocket.get("evidence_quality_score")
            top_evidence_warning = top_pocket.get("evidence_quality_warning") or ""
            pocket_lines = [
                "",
                "智能口袋摘要:",
                f"- Top1 口袋: {top_pocket_id}",
                f"- 排序等级: {top_rank_label}",
                f"- 排序分数: {top_rank_score}",
                f"- 热点覆盖数: {top_hotspot_text}",
                f"- 排序理由: {top_reason}",
            ]
            if top_external_exact is not None and not pd.isna(top_external_exact):
                pocket_lines.append(f"- 外部关键位点命中: {int(top_external_exact)}")
            if top_external_support is not None and not pd.isna(top_external_support):
                pocket_lines.append(f"- 外部证据均值: {format_energy_value(top_external_support)}")
            if top_evidence_quality != "-":
                pocket_lines.append(
                    f"- 证据质量: {top_evidence_quality} ({format_energy_value(top_evidence_quality_score)})"
                )
            if top_evidence_warning:
                pocket_lines.append(f"- 证据提醒: {top_evidence_warning}")
            lines[12:12] = pocket_lines
        except Exception:
            pass

    if joint_candidate_table is not None and not getattr(joint_candidate_table, "empty", True):
        try:
            top_joint = joint_candidate_table.iloc[0]
            top_joint_id = top_joint.get("pocket_id") or "-"
            top_joint_label = top_joint.get("recommendation_label") or "-"
            top_joint_score = format_energy_value(top_joint.get("recommendation_score"))
            top_joint_action = top_joint.get("recommendation_action") or "-"
            top_joint_evidence = top_joint.get("evidence_quality_label") or "-"
            top_joint_reason = top_joint.get("recommendation_reason") or "-"
            joint_lines = [
                "",
                "联合推荐摘要:",
                f"- Top1 候选: {top_joint_id}",
                f"- 推荐等级: {top_joint_label}",
                f"- 推荐分数: {top_joint_score}",
                f"- 推荐理由: {top_joint_reason}",
            ]
            joint_lines.extend(
                [
                    f"- 推荐动作: {top_joint_action}",
                    f"- 证据质量: {top_joint_evidence}",
                ]
            )
            lines[12:12] = joint_lines
        except Exception:
            pass

    if summary["valid_energy_count"] < summary["residue_count"]:
        lines.append("* 带星号的值为显示补齐值，未计入平均能量。")

    if energy_table is None or getattr(energy_table, "empty", True):
        lines.append("- 无可用残基能量明细")
        return "\n".join(lines)

    for row in energy_table.itertuples(index=False):
        energy_value = getattr(row, "delta_total_raw", None)
        imputed = energy_value is None or pd.isna(energy_value)
        if imputed:
            energy_value = getattr(row, "delta_total", None)
        energy_text = format_energy_value(energy_value)
        if imputed and energy_text != "-":
            energy_text = f"{energy_text}*"
        lines.append(
            f"- {row.label}: DELTA TOTAL = {energy_text}, color = {row.heat_color}"
        )

    return "\n".join(lines)
