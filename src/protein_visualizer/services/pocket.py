import pandas as pd


REQUIRED_COLUMNS = ["pocket_id", "chain", "resid", "resname", "volume", "score"]


def parse_pocket_table(text: str) -> pd.DataFrame:
    pocket_df = pd.read_csv(pd.io.common.StringIO(text.strip()))
    missing = [column for column in REQUIRED_COLUMNS if column not in pocket_df.columns]
    if missing:
        raise ValueError(f"口袋文件缺少必要列: {', '.join(missing)}")
    return pocket_df


def build_pocket_summary(pocket_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> pd.DataFrame:
    hotspot_keys = set(zip(hotspot_df["chain"], hotspot_df["resid"])) if not hotspot_df.empty else set()
    records = []
    for pocket_id, group in pocket_df.groupby("pocket_id"):
        hotspot_count = sum((row.chain, int(row.resid)) in hotspot_keys for row in group.itertuples(index=False))
        records.append(
            {
                "pocket_id": pocket_id,
                "volume": float(group["volume"].iloc[0]),
                "score": float(group["score"].iloc[0]),
                "residue_count": int(len(group)),
                "hotspot_count": int(hotspot_count),
                "residue_labels": ", ".join(
                    f"{row.resname} {row.chain}{int(row.resid)}" for row in group.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(records).sort_values(["hotspot_count", "score"], ascending=[False, False])
