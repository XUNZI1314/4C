import math

import numpy as np
import pandas as pd


def normalize_energy(values: pd.Series) -> pd.Series:
    vmin = values.min()
    vmax = values.max()
    if math.isclose(vmin, vmax):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - vmin) / (vmax - vmin)


def rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(round(v)))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def energy_to_hex_color(norm_value: float) -> str:
    red = np.array([220, 53, 69])
    blue = np.array([13, 110, 253])
    rgb = red + (blue - red) * float(norm_value)
    return rgb_to_hex(rgb)


def prepare_energy_table(atom_df: pd.DataFrame, energy_df: pd.DataFrame) -> pd.DataFrame:
    residue_df = atom_df[["chain", "resid", "resname"]].drop_duplicates()
    merged = residue_df.merge(energy_df, on=["chain", "resid", "resname"], how="left")
    merged["delta_total"] = merged["delta_total"].fillna(0.0)
    merged["norm_energy"] = normalize_energy(merged["delta_total"])
    merged["heat_color"] = merged["norm_energy"].map(energy_to_hex_color)
    merged["label"] = merged.apply(lambda r: f"{r['resname']} {r['chain']}{int(r['resid'])}", axis=1)
    return merged
