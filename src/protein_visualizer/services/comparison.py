import pandas as pd
from typing import List, Dict, Any


def compare_hotspot_sets(hotspot_tables: List[pd.DataFrame]) -> Dict[str, Any]:
    """比较多个热点表，返回共同/并集与每个残基在多少个构象中被标为热点的统计信息。"""
    n = len(hotspot_tables)
    sets = []
    for df in hotspot_tables:
        s = set()
        for row in df.itertuples(index=False):
            try:
                s.add((row.chain, int(row.resid), row.resname))
            except Exception:
                continue
        sets.append(s)

    union = set().union(*sets) if sets else set()
    intersection = sets[0].intersection(*sets[1:]) if n > 1 else sets[0] if sets else set()
    consistency_score = float(len(intersection)) / float(len(union)) if len(union) > 0 else 0.0

    counts = {}
    for s in sets:
        for key in s:
            counts[key] = counts.get(key, 0) + 1

    rows = []
    for (chain, resid, resname), cnt in counts.items():
        rows.append(
            {
                "chain": chain,
                "resid": int(resid),
                "resname": resname,
                "count": int(cnt),
                "label": f"{resname} {chain}{resid}",
                "is_common": cnt == n,
            }
        )

    per_residue_df = pd.DataFrame(rows).sort_values(["count", "resid"], ascending=[False, True])
    common_hotspots = [f"{r[2]} {r[0]}{r[1]}" for r in intersection]

    return {
        "total_conformations": n,
        "consistency_score": consistency_score,
        "union_size": len(union),
        "intersection_size": len(intersection),
        "common_hotspots": common_hotspots,
        "per_residue_df": per_residue_df,
    }


# 兼容旧名称
compare_hotspot_sets.__name__ = "compare_hotspot_sets"