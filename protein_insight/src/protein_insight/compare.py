"""Multi-conformation comparison helpers (simple implementation).

This module provides functions to compare hotspot lists across multiple MMPBSA
result tables and compute simple consistency scores.
"""
from typing import List, Dict, Any, Tuple
import pandas as pd
from collections import Counter, defaultdict


def compare_hotspots(mmpbsa_dfs: List[pd.DataFrame], top_k: int = 5, energy_threshold: float = None) -> pd.DataFrame:
    """Compare hotspots across multiple MMPBSA DataFrames.

    Returns a DataFrame with columns: chain, resid, resname, count, consistency, mean_energy.
    - `mmpbsa_dfs` is a list of DataFrames, each with columns ['chain','resid','resname','energy'].
    - A residue is considered a hotspot in a conformation if it is among the top_k lowest-energy residues
      after applying optional `energy_threshold` filter.
    """
    if not mmpbsa_dfs:
        return pd.DataFrame(columns=['chain', 'resid', 'resname', 'count', 'consistency', 'mean_energy'])

    total = len(mmpbsa_dfs)
    hotspot_lists: List[List[Tuple[str, int, str, float]]] = []
    for df in mmpbsa_dfs:
        if df is None or df.empty:
            hotspot_lists.append([])
            continue
        d = df.copy()
        if energy_threshold is not None:
            d = d[d['energy'] <= energy_threshold]
        d = d.sort_values(by='energy', ascending=True).head(top_k)
        hotspots = []
        for _, r in d.iterrows():
            hotspots.append((str(r.get('chain', 'A')), int(r['resid']), str(r.get('resname', '')), float(r['energy'])))
        hotspot_lists.append(hotspots)

    # count occurrences
    counter = Counter()
    energies_map = defaultdict(list)
    resname_map = {}
    for hotspots in hotspot_lists:
        for chain, resid, resname, energy in hotspots:
            key = (chain, resid)
            counter[key] += 1
            energies_map[key].append(energy)
            if key not in resname_map:
                resname_map[key] = resname

    rows = []
    for key, cnt in counter.items():
        chain, resid = key
        mean_energy = float(sum(energies_map[key]) / len(energies_map[key])) if energies_map[key] else float('nan')
        rows.append({'chain': chain, 'resid': resid, 'resname': resname_map.get(key, ''), 'count': cnt, 'consistency': cnt / total, 'mean_energy': mean_energy})

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out
    df_out = df_out.sort_values(by=['consistency', 'mean_energy'], ascending=[False, True]).reset_index(drop=True)
    return df_out
