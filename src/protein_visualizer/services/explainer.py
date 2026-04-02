from typing import Dict


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
            lines.append(f"{top['pocket_id']} 包含 {int(top['hotspot_count'])} 个热点残基，体积约 {float(top['volume']):.1f}，可能是优先候选口袋。")
        except Exception:
            pass

    try:
        mean_energy = float(energy_table["delta_total"].mean())
        lines.append(f"总体平均残基能量为 {mean_energy:.3f}，建议优先关注低于平均值的聚集区进行后续验证。")
    except Exception:
        pass

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
