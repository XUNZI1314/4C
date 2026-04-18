"""Structure-driven residue energy estimation helpers.

This module provides a structure-based proxy for residue-level energy when a
real MMPBSA table is unavailable. The proxy combines residue burial,
pairwise contacts, electrostatics, aromatic packing, hydrogen-bond propensity,
and B-factor, so it is suitable for visualization and hotspot ranking, but it
is not a substitute for a full MMPBSA workflow.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from protein_visualizer.services.parsers import parse_mmpbsa_delta_total


HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP"}
CHARGED_RESIDUES = {"ASP", "GLU", "LYS", "ARG", "HIS"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "CYS"}
FLEXIBLE_RESIDUES = {"GLY", "PRO"}
POSITIVE_RESIDUE_CHARGES = {"LYS": 1.0, "ARG": 1.0, "HIS": 0.35}
NEGATIVE_RESIDUE_CHARGES = {"ASP": -1.0, "GLU": -1.0}
DONOR_RESIDUES = {"LYS", "ARG", "HIS", "ASN", "GLN", "SER", "THR", "TYR", "TRP", "CYS"}
ACCEPTOR_RESIDUES = {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR", "CYS", "HIS"}


def _normalize_chain(chain: object) -> str:
    value = str(chain or "").strip()
    return value if value else "A"


def _normalize_resname(resname: object) -> str:
    return str(resname or "").strip().upper()


def _residue_group(resname: str) -> str:
    normalized = _normalize_resname(resname)
    if normalized in HYDROPHOBIC_RESIDUES:
        return "hydrophobic"
    if normalized in AROMATIC_RESIDUES:
        return "aromatic"
    if normalized in CHARGED_RESIDUES:
        return "charged"
    if normalized in POLAR_RESIDUES:
        return "polar"
    if normalized in FLEXIBLE_RESIDUES:
        return "flexible"
    return "other"


def _parse_pdb_atom_records(pdb_text: str) -> pd.DataFrame:
    records = []
    for raw_line in str(pdb_text).splitlines():
        if not raw_line.startswith("ATOM"):
            continue
        try:
            chain = _normalize_chain(raw_line[21:22])
            resid = int(raw_line[22:26].strip())
            resname = _normalize_resname(raw_line[17:20])
            atom_name = raw_line[12:16].strip()
            x = float(raw_line[30:38])
            y = float(raw_line[38:46])
            z = float(raw_line[46:54])
            b_factor_text = raw_line[60:66].strip() if len(raw_line) >= 66 else ""
            b_factor = float(b_factor_text) if b_factor_text else float("nan")
        except Exception:
            continue

        records.append(
            {
                "chain": chain,
                "resid": resid,
                "resname": resname,
                "atom_name": atom_name,
                "x": x,
                "y": y,
                "z": z,
                "b_factor": b_factor,
            }
        )

    return pd.DataFrame(records)


def _normalize_numeric_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)

    clean_values = pd.to_numeric(values, errors="coerce")
    if not clean_values.notna().any():
        return pd.Series(np.full(len(clean_values), 0.5), index=clean_values.index)

    fill_value = float(clean_values.dropna().median())
    filled = clean_values.fillna(fill_value)
    minimum = float(filled.min())
    maximum = float(filled.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.full(len(filled), 0.5), index=filled.index)
    return (filled - minimum) / (maximum - minimum)


def _normalized_pocket_set(pocket_residues: Optional[Sequence[Tuple[str, int]]]) -> set[Tuple[str, int]]:
    return {(_normalize_chain(chain), int(resid)) for chain, resid in (pocket_residues or [])}


def _residue_charge(resname: str) -> float:
    normalized = _normalize_resname(resname)
    if normalized in POSITIVE_RESIDUE_CHARGES:
        return float(POSITIVE_RESIDUE_CHARGES[normalized])
    if normalized in NEGATIVE_RESIDUE_CHARGES:
        return float(NEGATIVE_RESIDUE_CHARGES[normalized])
    return 0.0


def _residue_can_donate(resname: str) -> bool:
    return _normalize_resname(resname) in DONOR_RESIDUES


def _residue_can_accept(resname: str) -> bool:
    return _normalize_resname(resname) in ACCEPTOR_RESIDUES


def _pair_contact_weight(distance: float, *, midpoint: float = 6.0, steepness: float = 0.85) -> float:
    if not math.isfinite(distance):
        return 0.0
    return 1.0 / (1.0 + math.exp((float(distance) - float(midpoint)) / float(steepness)))


def _tight_contact_weight(distance: float) -> float:
    return _pair_contact_weight(distance, midpoint=3.4, steepness=0.45)


def _min_atom_distance(left_coords: np.ndarray, right_coords: np.ndarray) -> float:
    if left_coords.size == 0 or right_coords.size == 0:
        return float("inf")

    diff = left_coords[:, None, :] - right_coords[None, :, :]
    return float(np.sqrt(np.sum(diff * diff, axis=2)).min())


def _standardize_energy_table(energy_df: Optional[pd.DataFrame], energy_source: str) -> Optional[pd.DataFrame]:
    if energy_df is None or getattr(energy_df, "empty", True):
        return None

    normalized = energy_df.copy()
    if "delta_total" not in normalized.columns and "energy" in normalized.columns:
        normalized["delta_total"] = pd.to_numeric(normalized["energy"], errors="coerce")
    if "energy" not in normalized.columns and "delta_total" in normalized.columns:
        normalized["energy"] = pd.to_numeric(normalized["delta_total"], errors="coerce")

    normalized["energy_source"] = energy_source
    return normalized


def estimate_structure_energy_table(
    pdb_text: str,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    *,
    contact_cutoff: float = 8.0,
) -> pd.DataFrame:
    """Estimate residue energies directly from structure geometry.

    The returned table keeps the same core columns as the real MMPBSA flow so it
    can be passed through the existing coloring, hotspot, and reporting code.
    """

    atom_df = _parse_pdb_atom_records(pdb_text)
    if atom_df.empty:
        raise ValueError("未能从结构中解析到可用于估算的原子坐标")

    residue_rows = []
    residue_atom_coords: list[np.ndarray] = []
    for (chain, resid, resname), group in atom_df.groupby(["chain", "resid", "resname"], sort=True):
        atom_names = group["atom_name"].astype(str).str.upper()
        heavy_atoms = group[~atom_names.str.startswith("H")]
        atom_source = heavy_atoms if not heavy_atoms.empty else group
        ca_atoms = atom_source[atom_source["atom_name"] == "CA"]
        coordinate_source = ca_atoms if not ca_atoms.empty else atom_source
        centroid = coordinate_source[["x", "y", "z"]].mean().to_numpy(dtype=float)
        residue_rows.append(
            {
                "chain": _normalize_chain(chain),
                "resid": int(resid),
                "resname": _normalize_resname(resname),
                "group_type": _residue_group(resname),
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]),
                "b_factor": float(pd.to_numeric(group["b_factor"], errors="coerce").mean()),
            }
        )
        residue_atom_coords.append(atom_source[["x", "y", "z"]].to_numpy(dtype=float))

    residue_df = pd.DataFrame(residue_rows).sort_values(["chain", "resid"]).reset_index(drop=True)
    if residue_df.empty:
        raise ValueError("未能生成可用于估算的残基表")

    residue_df["chain_position"] = residue_df.groupby("chain").cumcount()

    coords = residue_df[["x", "y", "z"]].to_numpy(dtype=float)
    if len(residue_df) > 1:
        diff = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
    else:
        distances = np.zeros((1, 1), dtype=float)

    b_factor_norm = _normalize_numeric_series(residue_df["b_factor"])
    pocket_set = _normalized_pocket_set(pocket_residues)
    contact_cap = max(1, min(12, len(residue_df) - 1))

    neighbor_contact_scores = np.zeros(len(residue_df), dtype=float)
    interface_contact_scores = np.zeros(len(residue_df), dtype=float)
    contact_counts = np.zeros(len(residue_df), dtype=int)
    interface_counts = np.zeros(len(residue_df), dtype=int)
    pairwise_energy = np.zeros(len(residue_df), dtype=float)
    electrostatic_energy = np.zeros(len(residue_df), dtype=float)
    hydrophobic_energy = np.zeros(len(residue_df), dtype=float)
    aromatic_energy = np.zeros(len(residue_df), dtype=float)
    hbond_energy = np.zeros(len(residue_df), dtype=float)
    clash_energy = np.zeros(len(residue_df), dtype=float)

    if len(residue_df) > 1:
        pair_search_cutoff = max(float(contact_cutoff) + 4.0, 11.0)
        candidate_pairs = np.argwhere(np.triu(distances <= pair_search_cutoff, 1))
    else:
        candidate_pairs = np.empty((0, 2), dtype=int)

    for left_index, right_index in candidate_pairs:
        atom_distance = _min_atom_distance(residue_atom_coords[left_index], residue_atom_coords[right_index])
        if not math.isfinite(atom_distance):
            continue

        contact_weight = _pair_contact_weight(atom_distance)
        if contact_weight <= 0.0:
            continue

        tight_weight = _tight_contact_weight(atom_distance)
        same_chain = residue_df.at[left_index, "chain"] == residue_df.at[right_index, "chain"]
        chain_gap: Optional[int] = None
        pair_scale = 1.0
        if same_chain:
            chain_gap = abs(int(residue_df.at[left_index, "chain_position"]) - int(residue_df.at[right_index, "chain_position"]))
            if chain_gap <= 1:
                pair_scale = 0.30
            elif chain_gap == 2:
                pair_scale = 0.60
            else:
                pair_scale = 0.85

        contact_weight *= pair_scale
        tight_weight *= pair_scale
        if contact_weight <= 0.0:
            continue

        if atom_distance <= float(contact_cutoff):
            contact_counts[left_index] += 1
            contact_counts[right_index] += 1

        neighbor_contact_scores[left_index] += contact_weight
        neighbor_contact_scores[right_index] += contact_weight

        if not same_chain:
            interface_counts[left_index] += 1
            interface_counts[right_index] += 1
            interface_contact_scores[left_index] += contact_weight
            interface_contact_scores[right_index] += contact_weight

        left_resname = residue_df.at[left_index, "resname"]
        right_resname = residue_df.at[right_index, "resname"]
        left_group = residue_df.at[left_index, "group_type"]
        right_group = residue_df.at[right_index, "group_type"]

        left_charge = _residue_charge(left_resname)
        right_charge = _residue_charge(right_resname)
        charge_product = left_charge * right_charge

        electrostatic = 1.95 * charge_product * contact_weight
        if charge_product < 0.0:
            electrostatic -= 0.30 * tight_weight
        elif charge_product > 0.0:
            electrostatic += 0.12 * tight_weight

        hydrophobic_term = 0.0
        if left_group in {"hydrophobic", "aromatic"} and right_group in {"hydrophobic", "aromatic"}:
            hydrophobic_term = -1.15 * contact_weight
        elif left_group in {"hydrophobic", "aromatic"} or right_group in {"hydrophobic", "aromatic"}:
            hydrophobic_term = -0.22 * contact_weight

        aromatic_term = 0.0
        if left_group == "aromatic" and right_group == "aromatic":
            aromatic_term = -0.95 * tight_weight
        elif left_group == "aromatic" or right_group == "aromatic":
            aromatic_term = -0.25 * tight_weight

        hbond_term = 0.0
        if (_residue_can_donate(left_resname) and _residue_can_accept(right_resname)) or (
            _residue_can_donate(right_resname) and _residue_can_accept(left_resname)
        ):
            hbond_term = -1.10 * tight_weight

        if atom_distance < 2.35:
            clash_term = 1.8 * (2.35 - atom_distance) ** 2
        else:
            clash_term = 0.0

        if same_chain and chain_gap is not None and chain_gap <= 1:
            electrostatic *= 0.7
            hydrophobic_term *= 0.7
            aromatic_term *= 0.7
            hbond_term *= 0.7
            clash_term *= 0.7

        pair_total = electrostatic + hydrophobic_term + aromatic_term + hbond_term + clash_term
        if not same_chain:
            pair_total -= 0.12 * contact_weight

        half_pair_total = pair_total / 2.0
        pairwise_energy[left_index] += half_pair_total
        pairwise_energy[right_index] += half_pair_total
        electrostatic_energy[left_index] += electrostatic / 2.0
        electrostatic_energy[right_index] += electrostatic / 2.0
        hydrophobic_energy[left_index] += hydrophobic_term / 2.0
        hydrophobic_energy[right_index] += hydrophobic_term / 2.0
        aromatic_energy[left_index] += aromatic_term / 2.0
        aromatic_energy[right_index] += aromatic_term / 2.0
        hbond_energy[left_index] += hbond_term / 2.0
        hbond_energy[right_index] += hbond_term / 2.0
        clash_energy[left_index] += clash_term / 2.0
        clash_energy[right_index] += clash_term / 2.0

    burial_index = _normalize_numeric_series(pd.Series(neighbor_contact_scores, index=residue_df.index))
    interface_index = _normalize_numeric_series(pd.Series(interface_contact_scores, index=residue_df.index))
    surface_proxy = 1.0 - burial_index

    energies = []
    for index, row in residue_df.iterrows():
        burial_term = float(burial_index.iloc[index]) if not burial_index.empty else 0.5
        surface_term = float(surface_proxy.iloc[index]) if not surface_proxy.empty else 0.5
        interface_term = float(interface_index.iloc[index]) if not interface_index.empty else 0.0
        flex_term = float(b_factor_norm.iloc[index]) if not b_factor_norm.empty else 0.5

        if row["group_type"] in {"hydrophobic", "aromatic"}:
            self_energy = -2.15 * burial_term + 0.45 * surface_term + 0.25 * flex_term - 0.18 * interface_term
        elif row["group_type"] == "charged":
            self_energy = 1.20 * surface_term - 0.70 * burial_term + 0.40 * flex_term - 0.12 * interface_term
        elif row["group_type"] == "polar":
            self_energy = 0.82 * surface_term - 0.46 * burial_term + 0.32 * flex_term - 0.10 * interface_term
        elif row["group_type"] == "flexible":
            self_energy = 0.48 * surface_term + 0.55 * flex_term - 0.08 * burial_term
        else:
            self_energy = 0.28 * surface_term + 0.30 * flex_term - 0.06 * burial_term

        if (row["chain"], int(row["resid"])) in pocket_set:
            self_energy -= 0.50

        total_energy = float(self_energy + pairwise_energy[index])
        energies.append(float(max(-10.0, min(10.0, total_energy))))

    residue_df["contact_score"] = neighbor_contact_scores
    residue_df["interface_contact_score"] = interface_contact_scores
    residue_df["contact_count"] = contact_counts
    residue_df["interface_contact_count"] = interface_counts
    residue_df["contact_density"] = [min(count, contact_cap) / contact_cap for count in contact_counts]
    residue_df["interface_contact_density"] = [min(count, contact_cap) / contact_cap for count in interface_counts]
    residue_df["burial_index"] = burial_index
    residue_df["surface_proxy"] = surface_proxy
    residue_df["pairwise_energy"] = pairwise_energy
    residue_df["electrostatic_energy"] = electrostatic_energy
    residue_df["hydrophobic_energy"] = hydrophobic_energy
    residue_df["aromatic_energy"] = aromatic_energy
    residue_df["hbond_energy"] = hbond_energy
    residue_df["clash_energy"] = clash_energy
    residue_df["delta_total"] = energies
    residue_df["energy"] = residue_df["delta_total"]
    residue_df["energy_source"] = "结构估算"
    residue_df["estimate_method"] = "structure_contact_proxy_v2"

    return residue_df[[
        "chain",
        "resid",
        "resname",
        "delta_total",
        "energy",
        "energy_source",
        "estimate_method",
        "contact_score",
        "contact_count",
        "contact_density",
        "interface_contact_score",
        "interface_contact_count",
        "interface_contact_density",
        "pairwise_energy",
        "electrostatic_energy",
        "hydrophobic_energy",
        "aromatic_energy",
        "hbond_energy",
        "clash_energy",
        "burial_index",
        "surface_proxy",
    ]]


def estimate_protein_volume(pdb_text: str, *, padding: float = 2.0) -> float:
    """Estimate a protein's volume from its atomic bounding box.

    This is a lightweight geometric estimate suitable for display in the UI.
    The returned value is in cubic angstroms (A³).
    """

    atom_df = _parse_pdb_atom_records(pdb_text)
    if atom_df.empty:
        raise ValueError("未能从结构中解析到可用于估算体积的原子坐标")

    coords = atom_df[["x", "y", "z"]].to_numpy(dtype=float)
    min_corner = coords.min(axis=0)
    max_corner = coords.max(axis=0)
    extents = np.maximum(max_corner - min_corner, 0.0) + 2.0 * float(padding)
    return float(np.prod(extents))


def resolve_energy_table(
    pdb_text: str,
    *,
    energy_mode: str = "auto",
    mmpbsa_text: Optional[str] = None,
    mmpbsa_table: Optional[pd.DataFrame] = None,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
) -> tuple[Optional[pd.DataFrame], str]:
    """Resolve the residue energy table for the requested mode.

    Modes:
    - ``auto``: prefer uploaded MMPBSA data, otherwise fall back to structure estimation.
    - ``mmpbsa``: use uploaded MMPBSA data only.
    - ``estimate``: always use structure estimation.
    """

    mode = str(energy_mode or "auto").strip().lower()
    if mode not in {"auto", "mmpbsa", "estimate"}:
        mode = "auto"

    uploaded_table: Optional[pd.DataFrame] = None
    if mmpbsa_table is not None and not getattr(mmpbsa_table, "empty", True):
        uploaded_table = mmpbsa_table.copy()
    elif mmpbsa_text:
        uploaded_table = parse_mmpbsa_delta_total(mmpbsa_text)

    if mode == "estimate":
        return _standardize_energy_table(
            estimate_structure_energy_table(pdb_text, pocket_residues=pocket_residues),
            "结构估算",
        ), "结构估算"

    if uploaded_table is not None and not uploaded_table.empty:
        return _standardize_energy_table(uploaded_table, "MMPBSA数据"), "MMPBSA数据"

    if mode == "mmpbsa":
        return None, "无可用能量数据"

    return _standardize_energy_table(
        estimate_structure_energy_table(pdb_text, pocket_residues=pocket_residues),
        "结构估算",
    ), "结构估算"
