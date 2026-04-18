from typing import Dict

from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value


def explain_analysis(energy_table, hotspot_df, pocket_summary=None) -> str:
    """生成简短的自动分析解释文本，便于比赛展示与答辩。"""
    lines = []
    try:
        count = int(len(hotspot_df))
    except Exception:
        count = 0

    if count == 0:
        lines.append("未检测到显著低能量热点残基；建议调整阈值或使用更高分辨率的数据进行分析。")
    else:
        sample = ", ".join(hotspot_df["label"].astype(str).tolist()[:5])
        lines.append(f"检测到 {count} 个显著低能量热点残基，主要有：{sample}。")

    if pocket_summary is not None and not getattr(pocket_summary, "empty", True):
        try:
            top = pocket_summary.iloc[0]
            detection_route = str(top.get("detection_route") or "").strip()
            vote_count = top.get("method_vote_count")
            consensus_methods = str(top.get("consensus_methods") or "").strip()
            smart_rank_label = str(top.get("smart_rank_label") or "").strip()
            smart_rank_reason = str(top.get("smart_rank_reason") or "").strip()
            smart_rank_score = top.get("smart_rank_score")
            route_note = f"（{detection_route}）" if detection_route else ""
            try:
                vote_count_int = int(float(vote_count)) if vote_count not in (None, "") else 0
            except Exception:
                vote_count_int = 0
            if vote_count_int > 1:
                route_note = f"（{detection_route}，{vote_count_int}个方法）" if detection_route else f"（{vote_count_int}个方法）"
            lines.append(f"{top['pocket_id']}{route_note} 包含 {int(top['hotspot_count'])} 个热点残基，体积约 {float(top['volume']):.1f}，可能是优先候选口袋。")
            if vote_count_int > 1 and consensus_methods:
                lines.append(
                    f"当前口袋排序基于 {consensus_methods} 的共识结果，系统会自动综合 geometry、ligand 和 KVFinder 信号，无需手动微调 probe 或聚类参数。"
                )
            elif "multiscale" in detection_route.lower():
                lines.append("当前口袋来自多尺度 KVFinder 扫描后的稳定性合并，比单一 probe 参数更稳健。")
            if smart_rank_reason:
                try:
                    smart_score_text = f"{float(smart_rank_score):.2f}" if smart_rank_score is not None else "-"
                except Exception:
                    smart_score_text = "-"
                lines.append(
                    f"智能排序将 {top['pocket_id']} 评为 {smart_rank_label or '优先候选'}（得分 {smart_score_text}）：{smart_rank_reason}。"
                )
        except Exception:
            pass

    summary = build_analysis_summary(energy_table)
    mean_energy = summary.get("mean_energy")
    valid_energy_count = summary.get("valid_energy_count", 0)
    residue_count = summary.get("residue_count", 0)
    energy_source = summary.get("energy_source")
    if mean_energy is None:
        lines.append("当前没有足够的有效能量值来计算平均能量。")
    else:
        source_note = "（结构估算，不是标准 MMPBSA）" if energy_source == "结构估算" else ""
        if valid_energy_count and residue_count and valid_energy_count != residue_count:
            lines.append(
                f"总体平均残基能量为 {format_energy_value(mean_energy)}{source_note}（基于 {valid_energy_count}/{residue_count} 个有效值），建议优先关注低于平均值的聚集区进行后续验证。"
            )
        else:
            lines.append(f"总体平均残基能量为 {format_energy_value(mean_energy)}{source_note}，建议优先关注低于平均值的聚集区进行后续验证。")

    lines.append("说明：本结论为自动生成的初步提示，需要进一步生物学验证。")
    return "\n".join(lines)


def explain_comparison(comparison_result: Dict) -> str:
    n = comparison_result.get("total_conformations", 1)
    score = comparison_result.get("consistency_score", 0.0)
    common = comparison_result.get("intersection_size", 0)
    union = comparison_result.get("union_size", 0)
    lines = []
    lines.append(f"在 {n} 个构象中检测到 {union} 个候选热点残基，其中 {common} 个在所有构象中一致出现。")
    lines.append(f"热点一致性得分 (共同热点/并集) 为 {score:.2f}，得分越高表示热点在构象间越稳定。")
    if common > 0:
        sample = ", ".join(comparison_result.get("common_hotspots", [])[:6])
        lines.append(f"共同热点示例：{sample}。")
    lines.append("说明：本比较基于自动阈值判定，仅供演示与初筛参考。")
    return "\n".join(lines)
