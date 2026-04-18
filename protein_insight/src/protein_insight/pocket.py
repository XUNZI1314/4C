"""Simple pocket detection helpers.

This module implements a lightweight approach: if a pocket JSON is provided, it is returned;
otherwise it will attempt to find HETATM atoms in the PDB and mark residues within a radius.
"""
from typing import Dict, Any, List
import math


def load_pocket_json(source):
    # source can be path or file-like; reuse IO responsibility in io.py; here we assume source is already parsed dict or path
    try:
        # if already a dict
        if isinstance(source, dict):
            return source
        # else try to read as path
        import json
        with open(source, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def detect_pocket_by_ligand(pdb_str: str, radius: float = 5.0) -> Dict[str, Any]:
    """Heuristic: find any HETATM lines (ligand) and mark residues with CA within radius.

    Returns dict: {'pocket_residues': [{'chain':..., 'resid':..., 'resname':...}, ...]}
    """
    ligand_atoms = []
    atom_lines = []
    for line in pdb_str.splitlines():
        if line.startswith('HETATM'):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ligand_atoms.append((x, y, z))
            except Exception:
                continue
        if line.startswith('ATOM'):
            atom_lines.append(line)

    if not ligand_atoms:
        return {}

    # collect CA atoms as residue representatives
    pocket = []
    for line in atom_lines:
        try:
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            resname = line[17:20].strip()
            chain = line[21].strip() or 'A'
            resid = int(line[22:26].strip())
            # distance to any ligand atom
            for lx, ly, lz in ligand_atoms:
                d2 = (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
                if d2 <= radius ** 2:
                    pocket.append({'chain': chain, 'resid': resid, 'resname': resname})
                    break
        except Exception:
            continue

    return {'pocket_residues': pocket}


def get_pocket_residues(pocket_dict: Dict[str, Any], pdb_str: str = None, radius: float = 5.0):
    """Return list of (chain, resid) tuples for pocket residues.

    If `pocket_dict` is provided and contains explicit residues, use them.
    Otherwise, if `pdb_str` is provided, attempt ligand-based detection.
    """
    if pocket_dict:
        if 'pockets' in pocket_dict:
            res = []
            for p in pocket_dict['pockets']:
                for r in p.get('residues', []):
                    try:
                        res.append((r.get('chain', 'A'), int(r.get('resid', 0))))
                    except Exception:
                        continue
            return res
        if 'pocket_residues' in pocket_dict:
            out = []
            for r in pocket_dict['pocket_residues']:
                try:
                    out.append((r.get('chain', 'A'), int(r.get('resid', 0))))
                except Exception:
                    continue
            return out

    if pdb_str:
        d = detect_pocket_by_ligand(pdb_str, radius=radius)
        if d and 'pocket_residues' in d:
            return [(r['chain'], int(r['resid'])) for r in d['pocket_residues']]

    return []
