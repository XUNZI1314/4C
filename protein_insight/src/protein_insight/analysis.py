"""Analysis utilities: map energies, detect hotspots, color mapping."""
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd


@dataclass
class ResidueInfo:
    chain: str
    resid: int
    resname: str
    energy: float


def map_residue_energies(df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    """Return mapping (chain, resid) -> energy from normalized MMPBSA df."""
    out = {}
    for _, row in df.iterrows():
        chain = str(row['chain']) if 'chain' in row else 'A'
        resid = int(row['resid'])
        energy = float(row['energy'])
        out[(chain, resid)] = energy
    return out


def energy_color_map(val: float, vmin: float, vmax: float) -> str:
    """Map energy value to hex color. Negative (favorable) -> blue, positive -> red, center 0 -> white."""
    mid = 0.0
    if vmin == vmax:
        return '#ffffff'
    if val <= mid:
        # blue (cold) -> white
        denom = (mid - vmin) if (mid - vmin) != 0 else 1.0
        ratio = float(val - vmin) / denom
        r = int(255 * (1 - ratio))
        g = int(255 * (1 - ratio))
        b = 255
    else:
        denom = (vmax - mid) if (vmax - mid) != 0 else 1.0
        ratio = float(val - mid) / denom
        r = 255
        g = int(255 * (1 - ratio))
        b = int(255 * (1 - ratio))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return '#%02x%02x%02x' % (r, g, b)


def detect_hotspots(df: pd.DataFrame, top_k: int = 5, energy_threshold: float = None) -> List[ResidueInfo]:
    """Detect hotspots by lowest energy (most negative). Returns list of ResidueInfo sorted by energy asc."""
    if df is None or df.empty:
        return []
    dfc = df.copy()
    dfc = dfc.sort_values(by='energy', ascending=True)
    if energy_threshold is not None:
        dfc = dfc[dfc['energy'] <= energy_threshold]
    dfc = dfc.head(top_k)
    out = []
    for _, r in dfc.iterrows():
        out.append(ResidueInfo(chain=str(r['chain']), resid=int(r['resid']), resname=str(r.get('resname', '')), energy=float(r['energy'])))
    return out
