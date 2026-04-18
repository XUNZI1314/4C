"""I/O utilities: read PDB, MMPBSA CSV, pocket JSON."""
from pathlib import Path
import pandas as pd
import json
from io import StringIO


def load_pdb(source):
    """Load PDB content from a path or file-like object and return string."""
    if hasattr(source, 'read'):
        data = source.read()
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return data
    else:
        p = Path(source)
        return p.read_text(encoding='utf-8')


def load_mmpbsa_csv(source):
    """Load MMPBSA CSV from path or uploaded file and return normalized DataFrame with columns ['chain','resid','resname','energy'].

    The function is permissive about column names and will try to auto-detect the energy column.
    """
    if hasattr(source, 'read'):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        df = pd.read_csv(StringIO(raw))
    else:
        df = pd.read_csv(source)

    # normalize columns
    cols = {c.lower(): c for c in df.columns}

    # detect chain
    chain_col = None
    for candidate in ('chain',):
        if candidate in cols:
            chain_col = cols[candidate]
            break

    # detect resid
    resid_col = None
    for candidate in ('resid', 'res_num', 'resnum', 'resi', 'residue'):
        if candidate in cols:
            resid_col = cols[candidate]
            break

    # detect resname
    resname_col = None
    for candidate in ('resname', 'res_name', 'residue_name'):
        if candidate in cols:
            resname_col = cols[candidate]
            break

    # detect energy column: first numeric column not chain/resid/resname
    energy_col = None
    for c in df.columns:
        if c in (chain_col, resid_col, resname_col):
            continue
        # try convert
        try:
            pd.to_numeric(df[c])
            energy_col = c
            break
        except Exception:
            continue

    if energy_col is None:
        raise ValueError('无法识别 MMPBSA 能量列，请提供包含数值的 CSV')

    out = pd.DataFrame()
    out['chain'] = df[chain_col] if chain_col else 'A'
    if resid_col:
        out['resid'] = df[resid_col].astype(int)
    else:
        out['resid'] = range(1, len(df) + 1)
    out['resname'] = df[resname_col] if resname_col else ''
    out['energy'] = pd.to_numeric(df[energy_col])
    return out


def load_pocket_json(source):
    """Load pocket JSON from path or uploaded file."""
    if hasattr(source, 'read'):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        return json.loads(raw)
    else:
        with open(source, 'r', encoding='utf-8') as fh:
            return json.load(fh)
