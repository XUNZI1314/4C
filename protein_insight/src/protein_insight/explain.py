"""Generate short automatic explanations for detected hotspots."""
from typing import List, Any
from .analysis import ResidueInfo


def explain_results(hotspots: List[ResidueInfo], consistency: Any) -> str:
    """Return a short human-friendly explanation string summarizing hotspots.

    This is intentionally concise and aimed at presentation/答辩用途.
    """
    if not hotspots:
        return "未检测到显著热点，建议调整阈值或检查结构是否足够完整。"

    sents = []
    sents.append(f"检测到 {len(hotspots)} 个显著能量热点（按能量排序）。")
    topn = min(3, len(hotspots))
    for i in range(topn):
        h = hotspots[i]
        sents.append(f"第 {i+1} 位：残基 {h.chain}{h.resid} ({h.resname})，能量 {h.energy:.2f}，可能为结合热点。")

    if consistency is not None:
        sents.append("跨构象一致性分析显示若干残基在多个构象中重复出现，推荐优先验证这些位置。")
    else:
        sents.append("建议在后续分析中添加更多构象以评估热点稳定性。")

    return ' '.join(sents)
