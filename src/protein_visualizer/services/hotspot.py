import pandas as pd


def identify_hotspots(energy_table: pd.DataFrame, energy_threshold: float = -1.0, top_n: int = 5) -> pd.DataFrame:
    table = energy_table.copy()
    table["is_hotspot"] = table["delta_total"] <= energy_threshold
    hotspot_df = table[table["is_hotspot"]].sort_values("delta_total", ascending=True)

    if hotspot_df.empty:
        hotspot_df = table.nsmallest(top_n, "delta_total").copy()
        hotspot_df["is_hotspot"] = True

    hotspot_df["hotspot_rank"] = range(1, len(hotspot_df) + 1)
    return hotspot_df


def summarize_hotspot_clusters(hotspot_df: pd.DataFrame) -> dict:
    if hotspot_df.empty:
        return {
            "count": 0,
            "lowest_hotspot": "-",
            "cluster_hint": "未发现明显热点残基。",
        }

    labels = hotspot_df["label"].tolist()
    return {
        "count": int(len(hotspot_df)),
        "lowest_hotspot": labels[0],
        "cluster_hint": f"共识别到 {len(labels)} 个热点残基，重点关注：{', '.join(labels[:3])}。",
    }
