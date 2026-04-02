import re

import pandas as pd

from protein_visualizer.models import BACKBONE_ATOMS


def parse_pdb_atoms(pdb_text: str) -> pd.DataFrame:
    records = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        records.append(
            {
                "chain": (line[21].strip() or "A"),
                "resid": int(line[22:26].strip()),
                "resname": line[17:20].strip(),
                "atom_name": line[12:16].strip(),
                "atom_type": "backbone" if line[12:16].strip() in BACKBONE_ATOMS else "sidechain",
            }
        )
    return pd.DataFrame(records)


def parse_mmpbsa_delta_total(text: str) -> pd.DataFrame:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "DELTA TOTAL" in line.upper() and any(ch.isalpha() for ch in line):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 4:
            continue
        try:
            resid = int(parts[0])
            chain = parts[1]
            resname = parts[2]
            energy = float(parts[-1])
        except (ValueError, IndexError):
            continue
        rows.append({"chain": chain, "resid": resid, "resname": resname, "delta_total": energy})

    energy_df = pd.DataFrame(rows)
    if energy_df.empty:
        raise ValueError("未解析到有效的 MMPBSA DELTA TOTAL 数据")

    return energy_df.drop_duplicates(subset=["chain", "resid"], keep="last").sort_values(["chain", "resid"])
