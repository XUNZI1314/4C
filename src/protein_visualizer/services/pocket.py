import math
import os
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from protein_visualizer.services.parsers import parse_pdb_atoms
from protein_visualizer.services.p2rank import run_p2rank
from protein_visualizer.services.pocket_ranker import SMART_POCKET_COLUMNS, rank_detected_pockets

try:
    import pyKVFinder

    PYKVFINDER_AVAILABLE = True
except ModuleNotFoundError:
    pyKVFinder = None
    PYKVFINDER_AVAILABLE = False


REQUIRED_COLUMNS = ["pocket_id", "chain", "resid", "resname", "volume", "score"]
AUTO_POCKET_COLUMNS = REQUIRED_COLUMNS + [
    "residue_score",
    "contact_count",
    "center_distance",
    "ligand_contact_count",
    "detection_method",
    "detection_route",
    "is_hotspot",
    "depth_avg",
    "depth_max",
    "overlap_ratio",
    "proximity_distance",
    "precision_score",
    "seed_support",
    "confidence_score",
    "external_support",
    "external_confidence",
    "external_evidence_count",
    "external_exact_match",
    "external_structure_verified",
    "external_mapping_quality",
    "external_direct_anchor",
    "evidence_route_anchor",
    "evidence_anchor_distance",
    "evidence_anchor_proximity",
    "evidence_anchor_residue",
    "external_direct_sources",
    "external_evidence_types",
    "external_evidence_notes",
    "conservation_support",
    "conservation_confidence",
    "conservation_evidence_count",
    "consensus_score",
    "consensus_methods",
    "method_vote_count",
    "consensus_overlap_ratio",
]

CONSENSUS_METHOD_ORDER = ["p2rank", "external-evidence", "kvfinder", "ligand-proximity", "geometry-cluster"]
DETECTION_DIAGNOSTIC_COLUMNS = [
    "method",
    "enabled",
    "available",
    "status",
    "pocket_count",
    "residue_rows",
    "note",
]
EXTERNAL_EVIDENCE_ROUTE_DEFAULTS = {
    "min_support": 0.58,
    "min_confidence": 0.55,
    "min_mapping_quality": 0.82,
    "min_consensus_support": 0.70,
}


def _detection_route_label(path: str) -> str:
    return f"precision-{path}"


def _canonical_method_name(value: object) -> str:
    text = str(value or "").strip().lower()
    if "p2rank" in text:
        return "p2rank"
    if "external" in text or "evidence" in text:
        return "external-evidence"
    if "kvfinder" in text:
        return "kvfinder"
    if "ligand" in text:
        return "ligand-proximity"
    if "geometry" in text or "cluster" in text:
        return "geometry-cluster"
    if "consensus" in text:
        return "consensus"
    return text.replace(" ", "-") if text else "unknown"


def _ordered_unique_methods(method_names: Sequence[object]) -> list[str]:
    ordered: list[str] = []
    for name in method_names:
        canonical = _canonical_method_name(name)
        if canonical not in ordered:
            ordered.append(canonical)

    return sorted(
        ordered,
        key=lambda name: (
            CONSENSUS_METHOD_ORDER.index(name) if name in CONSENSUS_METHOD_ORDER else len(CONSENSUS_METHOD_ORDER),
            name,
        ),
    )


def _display_method_token(value: object) -> str:
    name = _canonical_method_name(value)
    if name == "ligand-proximity":
        return "ligand"
    if name == "geometry-cluster":
        return "geometry"
    return name


def _join_methods(method_names: Sequence[object]) -> str:
    methods = [_display_method_token(name) for name in _ordered_unique_methods(method_names)]
    return "+".join(methods)


def _build_detection_diagnostic_entry(
    method: str,
    *,
    enabled: bool,
    available: bool,
    status: str,
    pocket_count: int = 0,
    residue_rows: int = 0,
    note: str = "",
) -> dict[str, object]:
    return {
        "method": str(method),
        "enabled": bool(enabled),
        "available": bool(available),
        "status": str(status),
        "pocket_count": int(pocket_count),
        "residue_rows": int(residue_rows),
        "note": str(note or "").strip(),
    }


def _summarize_external_evidence_table(external_site_df: Optional[pd.DataFrame]) -> dict[str, object]:
    if external_site_df is None or getattr(external_site_df, "empty", True):
        return {
            "evidence_rows": 0,
            "exact_rows": 0,
            "weak_rows": 0,
            "sources": "",
            "evidence_types": "",
        }

    working = external_site_df.copy()
    exact_rows = 0
    weak_rows = 0
    if "mapping_level" in working.columns:
        level_series = working["mapping_level"].astype(str).str.strip().str.lower()
        exact_rows = int((level_series == "exact").sum())
        weak_rows = int((level_series == "weak").sum())

    sources = ""
    if "evidence_source" in working.columns:
        sources = ",".join(
            sorted(
                {
                    str(value).strip()
                    for value in working["evidence_source"].dropna().tolist()
                    if str(value).strip()
                }
            )
        )

    evidence_types = ""
    if "evidence_type" in working.columns:
        evidence_types = ",".join(
            sorted(
                {
                    str(value).strip()
                    for value in working["evidence_type"].dropna().tolist()
                    if str(value).strip()
                }
            )
        )

    return {
        "evidence_rows": int(len(working)),
        "exact_rows": int(exact_rows),
        "weak_rows": int(weak_rows),
        "sources": sources,
        "evidence_types": evidence_types,
    }


def get_pocket_detection_metadata(pocket_df: Optional[pd.DataFrame]) -> dict[str, object]:
    if pocket_df is None:
        return {}
    attrs = getattr(pocket_df, "attrs", {}) or {}
    metadata = attrs.get("pocket_detection_metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def build_pocket_detection_diagnostics_table(pocket_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    metadata = get_pocket_detection_metadata(pocket_df)
    diagnostics = metadata.get("diagnostics") or []
    if not diagnostics:
        return pd.DataFrame(columns=DETECTION_DIAGNOSTIC_COLUMNS)
    return pd.DataFrame(diagnostics, columns=DETECTION_DIAGNOSTIC_COLUMNS)


def summarize_pocket_detection_metadata(metadata: Optional[dict[str, object]]) -> dict[str, object]:
    if not metadata:
        return {}

    diagnostics = metadata.get("diagnostics") or []
    status_parts: list[str] = []
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        method = str(item.get("method") or "").strip()
        status = str(item.get("status") or "").strip()
        if method and status:
            status_parts.append(f"{method}:{status}")

    external_summary = metadata.get("external_evidence") or {}
    if not isinstance(external_summary, dict):
        external_summary = {}
    conservation_summary = metadata.get("conservation_evidence") or {}
    if not isinstance(conservation_summary, dict):
        conservation_summary = {}

    requested_methods = metadata.get("requested_methods") or {}
    requested_method_names: list[str] = []
    if isinstance(requested_methods, dict):
        for method_name, enabled in requested_methods.items():
            if enabled:
                requested_method_names.append(str(method_name))

    p2rank_meta = metadata.get("p2rank_meta") or {}
    if not isinstance(p2rank_meta, dict):
        p2rank_meta = {}

    external_route_meta = metadata.get("external_evidence_route") or {}
    if not isinstance(external_route_meta, dict):
        external_route_meta = {}
    external_route_status = ""
    for item in diagnostics:
        if isinstance(item, dict) and str(item.get("method") or "").strip() == "external-evidence":
            external_route_status = str(item.get("status") or "").strip()
            break

    try:
        p2rank_return_code = int(p2rank_meta.get("return_code")) if p2rank_meta.get("return_code") is not None else None
    except Exception:
        p2rank_return_code = None

    try:
        p2rank_prediction_rows = int(p2rank_meta.get("prediction_rows")) if p2rank_meta.get("prediction_rows") is not None else 0
    except Exception:
        p2rank_prediction_rows = 0

    try:
        p2rank_residue_rows = int(p2rank_meta.get("residue_rows")) if p2rank_meta.get("residue_rows") is not None else 0
    except Exception:
        p2rank_residue_rows = 0

    p2rank_message = str(
        p2rank_meta.get("reason")
        or p2rank_meta.get("stderr")
        or p2rank_meta.get("stdout")
        or ""
    ).strip()

    return {
        "auto_detection_adaptive_profile": bool(metadata.get("adaptive_profile", False)),
        "auto_detection_requested_methods": ",".join(requested_method_names),
        "auto_detection_methods_used": str(metadata.get("methods_used") or ""),
        "auto_detection_status_summary": "; ".join(status_parts),
        "auto_detection_structure_atom_count": int(metadata.get("structure_atom_count", 0) or 0),
        "auto_detection_hotspot_seed_count": int(metadata.get("hotspot_seed_count", 0) or 0),
        "auto_detection_residue_candidates": int(metadata.get("residue_candidate_count", 0) or 0),
        "auto_detection_ligand_atom_count": int(metadata.get("ligand_atom_count", 0) or 0),
        "auto_detection_result_pocket_count": int(metadata.get("result_pocket_count", 0) or 0),
        "auto_detection_result_residue_rows": int(metadata.get("result_residue_rows", 0) or 0),
        "auto_detection_external_rows": int(external_summary.get("evidence_rows", 0) or 0),
        "auto_detection_external_exact_rows": int(external_summary.get("exact_rows", 0) or 0),
        "auto_detection_external_weak_rows": int(external_summary.get("weak_rows", 0) or 0),
        "auto_detection_external_sources": str(external_summary.get("sources") or ""),
        "auto_detection_external_types": str(external_summary.get("evidence_types") or ""),
        "auto_detection_external_route_enabled": bool(external_route_meta.get("enabled", False)),
        "auto_detection_external_route_status": external_route_status,
        "auto_detection_external_route_min_support": float(
            external_route_meta.get("min_support", EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_support"]) or 0.0
        ),
        "auto_detection_external_route_min_confidence": float(
            external_route_meta.get("min_confidence", EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_confidence"]) or 0.0
        ),
        "auto_detection_external_route_min_mapping_quality": float(
            external_route_meta.get("min_mapping_quality", EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_mapping_quality"]) or 0.0
        ),
        "auto_detection_external_route_radius": (
            float(external_route_meta.get("radius"))
            if external_route_meta.get("radius") not in {None, ""}
            else None
        ),
        "auto_detection_conservation_rows": int(conservation_summary.get("evidence_rows", 0) or 0),
        "auto_detection_conservation_sources": str(conservation_summary.get("sources") or ""),
        "auto_detection_conservation_types": str(conservation_summary.get("evidence_types") or ""),
        "auto_detection_p2rank_status": str(p2rank_meta.get("status") or ""),
        "auto_detection_p2rank_return_code": p2rank_return_code,
        "auto_detection_p2rank_prediction_rows": p2rank_prediction_rows,
        "auto_detection_p2rank_residue_rows": p2rank_residue_rows,
        "auto_detection_p2rank_message": p2rank_message[:240],
    }


def _finalize_pocket_detection_result(
    table: Optional[pd.DataFrame],
    *,
    metadata: dict[str, object],
) -> pd.DataFrame:
    if table is None or getattr(table, "empty", True):
        result = _empty_auto_pocket_table()
    else:
        result = rank_detected_pockets(_ensure_auto_pocket_columns(table))

    diagnostics = metadata.get("diagnostics") or []
    methods_used = [
        str(item.get("method"))
        for item in diagnostics
        if isinstance(item, dict)
        and str(item.get("status") or "").strip().lower() in {"used", "single-method", "consensus", "fallback"}
        and int(item.get("pocket_count", 0) or 0) > 0
    ]
    metadata = {
        **metadata,
        "methods_used": ",".join(dict.fromkeys(methods_used)),
        "result_pocket_count": int(result["pocket_id"].astype(str).nunique()) if not result.empty and "pocket_id" in result.columns else 0,
        "result_residue_rows": int(len(result)),
    }
    result.attrs["pocket_detection_metadata"] = metadata
    return result


def _empty_external_support_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chain",
            "resid",
            "external_support",
            "external_confidence",
            "external_evidence_count",
            "external_exact_match",
            "external_structure_verified",
            "external_mapping_quality",
            "external_direct_anchor",
            "external_direct_sources",
            "external_evidence_types",
            "external_evidence_notes",
        ]
    )


def _empty_conservation_support_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chain",
            "resid",
            "conservation_support",
            "conservation_confidence",
            "conservation_evidence_count",
        ]
    )


def _external_mapping_quality(method: object, mapping_level: object) -> float:
    method_text = str(method or "").strip().lower()
    level_text = str(mapping_level or "").strip().lower()

    if "structure-order" in method_text or "verified" in method_text:
        return 1.0
    if "structure-interpolated" in method_text:
        return 0.88
    if "linear-exact" in method_text:
        return 0.84
    if "linear-interpolated" in method_text:
        return 0.72
    if "gap-fallback" in method_text:
        return 0.34
    if "chain-fallback" in method_text or "unmapped-fallback" in method_text or "resid-fallback" in method_text:
        return 0.30
    if method_text == "mcsa-direct":
        return 0.55
    if method_text == "conservation-import":
        return 0.52
    if method_text == "uniprot-direct":
        return 0.45
    if level_text == "exact":
        return 0.78
    return 0.38


def _join_unique_text(values: object, *, max_items: int = 6, max_len: int = 360) -> str:
    if values is None:
        return ""
    try:
        iterable = values.tolist()
    except AttributeError:
        iterable = values if isinstance(values, (list, tuple, set)) else [values]

    items: list[str] = []
    for value in iterable:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in items:
            items.append(text)
        if len(items) >= int(max_items):
            break

    joined = "; ".join(items)
    if len(joined) > int(max_len):
        return joined[: int(max_len) - 3].rstrip() + "..."
    return joined


def _build_external_support_frame(
    residue_df: pd.DataFrame,
    external_site_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if residue_df is None or residue_df.empty or external_site_df is None or getattr(external_site_df, "empty", True):
        return _empty_external_support_frame()

    working = external_site_df.copy()
    if "resid" not in working.columns:
        return _empty_external_support_frame()

    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return _empty_external_support_frame()

    working["resid"] = working["resid"].astype(int)
    if "chain" not in working.columns:
        working["chain"] = ""
    working["chain"] = working["chain"].astype(str).map(_normalize_chain)
    if "mapping_level" not in working.columns:
        working["mapping_level"] = "weak"
    working["mapping_level"] = working["mapping_level"].astype(str).str.strip().str.lower()
    working.loc[~working["mapping_level"].isin({"exact", "weak"}), "mapping_level"] = "weak"
    if "mapping_confidence" not in working.columns:
        working["mapping_confidence"] = working["mapping_level"].map({"exact": 0.9, "weak": 0.3}).fillna(0.3)
    working["mapping_confidence"] = pd.to_numeric(working["mapping_confidence"], errors="coerce").fillna(0.3).clip(0.0, 1.0)
    if "mapping_method" not in working.columns:
        working["mapping_method"] = "unknown"
    working["mapping_quality"] = working.apply(
        lambda row: _external_mapping_quality(row.get("mapping_method"), row.get("mapping_level")),
        axis=1,
    )
    working["structure_verified"] = working["mapping_method"].astype(str).str.contains("structure|verified", case=False, regex=True, na=False)
    if "evidence_score" not in working.columns:
        working["evidence_score"] = 0.75
    working["evidence_score"] = pd.to_numeric(working["evidence_score"], errors="coerce").fillna(0.75).clip(0.0, 1.0)
    for column in ("evidence_source", "evidence_type", "evidence_note"):
        if column not in working.columns:
            working[column] = ""

    exact_df = working[working["mapping_level"] == "exact"].copy()
    weak_df = working[working["mapping_level"] != "exact"].copy()

    exact_score_map: Dict[Tuple[str, int], float] = {}
    exact_confidence_map: Dict[Tuple[str, int], float] = {}
    exact_count_map: Dict[Tuple[str, int], int] = {}
    exact_quality_map: Dict[Tuple[str, int], float] = {}
    exact_verified_map: Dict[Tuple[str, int], bool] = {}
    exact_source_map: Dict[Tuple[str, int], str] = {}
    exact_type_map: Dict[Tuple[str, int], str] = {}
    exact_note_map: Dict[Tuple[str, int], str] = {}
    if not exact_df.empty:
        exact_df["weighted_score"] = exact_df["evidence_score"] * (
            0.55 + 0.25 * exact_df["mapping_confidence"] + 0.20 * exact_df["mapping_quality"]
        )
        exact_grouped = exact_df.groupby(["chain", "resid"], sort=False)
        for (chain, resid), group in exact_grouped:
            residue_key = (_normalize_chain(chain), int(resid))
            exact_score_map[residue_key] = float(group["weighted_score"].max())
            exact_confidence_map[residue_key] = float(
                (0.55 * group["mapping_confidence"] + 0.45 * group["mapping_quality"]).mean()
            )
            exact_count_map[residue_key] = int(len(group))
            exact_quality_map[residue_key] = float(group["mapping_quality"].max())
            exact_verified_map[residue_key] = bool(group["structure_verified"].fillna(False).astype(bool).any())
            exact_source_map[residue_key] = _join_unique_text(group["evidence_source"], max_items=6, max_len=220)
            exact_type_map[residue_key] = _join_unique_text(group["evidence_type"], max_items=6, max_len=220)
            exact_note_map[residue_key] = _join_unique_text(group["evidence_note"], max_items=4, max_len=520)

    weak_score_map: Dict[int, float] = {}
    weak_confidence_map: Dict[int, float] = {}
    weak_count_map: Dict[int, int] = {}
    weak_quality_map: Dict[int, float] = {}
    weak_verified_map: Dict[int, bool] = {}
    weak_source_map: Dict[int, str] = {}
    weak_type_map: Dict[int, str] = {}
    weak_note_map: Dict[int, str] = {}
    if not weak_df.empty:
        weak_df["weighted_score"] = weak_df["evidence_score"] * (
            0.18 + 0.17 * weak_df["mapping_confidence"] + 0.15 * weak_df["mapping_quality"]
        )
        weak_grouped = weak_df.groupby("resid", sort=False)
        for resid, group in weak_grouped:
            weak_score_map[int(resid)] = float(group["weighted_score"].max())
            weak_confidence_map[int(resid)] = float(
                (0.55 * group["mapping_confidence"] + 0.45 * group["mapping_quality"]).mean()
            )
            weak_count_map[int(resid)] = int(len(group))
            weak_quality_map[int(resid)] = float(group["mapping_quality"].max())
            weak_verified_map[int(resid)] = bool(group["structure_verified"].fillna(False).astype(bool).any())
            weak_source_map[int(resid)] = _join_unique_text(group["evidence_source"], max_items=6, max_len=220)
            weak_type_map[int(resid)] = _join_unique_text(group["evidence_type"], max_items=6, max_len=220)
            weak_note_map[int(resid)] = _join_unique_text(group["evidence_note"], max_items=4, max_len=520)

    coords = residue_df[["x", "y", "z"]].to_numpy(dtype=float)
    exact_seed_coords = np.empty((0, 3), dtype=float)
    if exact_score_map:
        exact_seed_coords = residue_df[
            residue_df.apply(
                lambda row: (_normalize_chain(row["chain"]), int(row["resid"])) in exact_score_map,
                axis=1,
            )
        ][["x", "y", "z"]].to_numpy(dtype=float)

    if exact_seed_coords.size:
        exact_distance = _compute_min_distances_to_points(coords, exact_seed_coords)
        exact_norm = _normalize_numeric_series(pd.Series(exact_distance, index=residue_df.index))
        exact_seed_quality = float(max(exact_quality_map.values())) if exact_quality_map else 1.0
        exact_proximity = exact_seed_quality * (1.0 - exact_norm.to_numpy(dtype=float))
    else:
        exact_proximity = np.zeros(len(residue_df), dtype=float)

    rows = []
    for row_index, row in enumerate(residue_df.itertuples(index=False)):
        residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
        direct_exact_support = float(exact_score_map.get(residue_key, 0.0))
        direct_exact_confidence = float(exact_confidence_map.get(residue_key, 0.0))
        direct_exact_count = int(exact_count_map.get(residue_key, 0))
        direct_exact_quality = float(exact_quality_map.get(residue_key, 0.0))
        direct_exact_verified = bool(exact_verified_map.get(residue_key, False))

        weak_support = float(weak_score_map.get(residue_key[1], 0.0))
        weak_confidence = float(weak_confidence_map.get(residue_key[1], 0.0))
        weak_count = int(weak_count_map.get(residue_key[1], 0))
        weak_quality = float(weak_quality_map.get(residue_key[1], 0.0))
        weak_verified = bool(weak_verified_map.get(residue_key[1], False))

        proximity_support = 0.55 * float(exact_proximity[row_index]) if row_index < len(exact_proximity) else 0.0
        external_support = max(direct_exact_support, proximity_support, 0.35 * weak_support)
        external_confidence = max(direct_exact_confidence, 0.65 * weak_confidence)
        external_evidence_count = direct_exact_count + weak_count
        external_mapping_quality = max(direct_exact_quality, weak_quality)
        external_structure_verified = bool(direct_exact_verified or weak_verified)
        direct_evidence_support = max(direct_exact_support, 0.35 * weak_support)
        direct_evidence_confidence = max(direct_exact_confidence, 0.65 * weak_confidence)
        external_direct_anchor = bool(
            direct_exact_count > 0
            or (
                external_evidence_count > 0
                and direct_evidence_support >= EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_support"]
                and direct_evidence_confidence >= EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_confidence"]
            )
            or (
                external_evidence_count > 0
                and external_mapping_quality >= EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_mapping_quality"]
            )
        )
        external_direct_sources = _join_unique_text(
            [exact_source_map.get(residue_key, ""), weak_source_map.get(residue_key[1], "")],
            max_items=8,
            max_len=260,
        )
        external_evidence_types = _join_unique_text(
            [exact_type_map.get(residue_key, ""), weak_type_map.get(residue_key[1], "")],
            max_items=8,
            max_len=260,
        )
        external_evidence_notes = _join_unique_text(
            [exact_note_map.get(residue_key, ""), weak_note_map.get(residue_key[1], "")],
            max_items=6,
            max_len=640,
        )

        rows.append(
            {
                "chain": residue_key[0],
                "resid": residue_key[1],
                "external_support": round(float(min(1.0, external_support)), 3),
                "external_confidence": round(float(min(1.0, external_confidence)), 3),
                "external_evidence_count": int(external_evidence_count),
                "external_exact_match": bool(direct_exact_count > 0),
                "external_structure_verified": bool(external_structure_verified),
                "external_mapping_quality": round(float(min(1.0, external_mapping_quality)), 3),
                "external_direct_anchor": bool(external_direct_anchor),
                "external_direct_sources": external_direct_sources,
                "external_evidence_types": external_evidence_types,
                "external_evidence_notes": external_evidence_notes,
            }
        )

    if not rows:
        return _empty_external_support_frame()
    return pd.DataFrame(rows)


def _build_conservation_support_frame(
    residue_df: pd.DataFrame,
    conservation_site_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if residue_df is None or residue_df.empty or conservation_site_df is None or getattr(conservation_site_df, "empty", True):
        return _empty_conservation_support_frame()

    working = conservation_site_df.copy()
    if "resid" not in working.columns:
        return _empty_conservation_support_frame()

    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return _empty_conservation_support_frame()

    working["resid"] = working["resid"].astype(int)
    if "chain" not in working.columns:
        working["chain"] = ""
    working["chain"] = working["chain"].astype(str).map(_normalize_chain)
    if "mapping_level" not in working.columns:
        working["mapping_level"] = "weak"
    working["mapping_level"] = working["mapping_level"].astype(str).str.strip().str.lower()
    working.loc[~working["mapping_level"].isin({"exact", "weak"}), "mapping_level"] = "weak"
    if "mapping_confidence" not in working.columns:
        working["mapping_confidence"] = working["mapping_level"].map({"exact": 0.9, "weak": 0.3}).fillna(0.3)
    working["mapping_confidence"] = pd.to_numeric(working["mapping_confidence"], errors="coerce").fillna(0.3).clip(0.0, 1.0)
    if "mapping_method" not in working.columns:
        working["mapping_method"] = "unknown"
    working["mapping_quality"] = working.apply(
        lambda row: _external_mapping_quality(row.get("mapping_method"), row.get("mapping_level")),
        axis=1,
    )
    if "evidence_score" not in working.columns:
        working["evidence_score"] = 0.75
    working["evidence_score"] = pd.to_numeric(working["evidence_score"], errors="coerce").fillna(0.75).clip(0.0, 1.0)

    exact_df = working[working["mapping_level"] == "exact"].copy()
    weak_df = working[working["mapping_level"] != "exact"].copy()

    exact_score_map: Dict[Tuple[str, int], float] = {}
    exact_confidence_map: Dict[Tuple[str, int], float] = {}
    exact_count_map: Dict[Tuple[str, int], int] = {}
    if not exact_df.empty:
        exact_df["weighted_score"] = exact_df["evidence_score"] * (
            0.32 + 0.18 * exact_df["mapping_confidence"] + 0.16 * exact_df["mapping_quality"]
        )
        exact_grouped = exact_df.groupby(["chain", "resid"], sort=False)
        for (chain, resid), group in exact_grouped:
            residue_key = (_normalize_chain(chain), int(resid))
            exact_score_map[residue_key] = float(group["weighted_score"].max())
            exact_confidence_map[residue_key] = float(
                (0.55 * group["mapping_confidence"] + 0.45 * group["mapping_quality"]).mean()
            )
            exact_count_map[residue_key] = int(len(group))

    weak_score_map: Dict[int, float] = {}
    weak_confidence_map: Dict[int, float] = {}
    weak_count_map: Dict[int, int] = {}
    if not weak_df.empty:
        weak_df["weighted_score"] = weak_df["evidence_score"] * (
            0.08 + 0.10 * weak_df["mapping_confidence"] + 0.08 * weak_df["mapping_quality"]
        )
        weak_grouped = weak_df.groupby("resid", sort=False)
        for resid, group in weak_grouped:
            weak_score_map[int(resid)] = float(group["weighted_score"].max())
            weak_confidence_map[int(resid)] = float(
                (0.55 * group["mapping_confidence"] + 0.45 * group["mapping_quality"]).mean()
            )
            weak_count_map[int(resid)] = int(len(group))

    rows = []
    for row in residue_df.itertuples(index=False):
        residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
        direct_exact_support = float(exact_score_map.get(residue_key, 0.0))
        direct_exact_confidence = float(exact_confidence_map.get(residue_key, 0.0))
        direct_exact_count = int(exact_count_map.get(residue_key, 0))

        weak_support = float(weak_score_map.get(residue_key[1], 0.0))
        weak_confidence = float(weak_confidence_map.get(residue_key[1], 0.0))
        weak_count = int(weak_count_map.get(residue_key[1], 0))

        conservation_support = max(direct_exact_support, 0.22 * weak_support)
        conservation_confidence = max(direct_exact_confidence, 0.45 * weak_confidence)
        conservation_evidence_count = direct_exact_count + weak_count

        rows.append(
            {
                "chain": residue_key[0],
                "resid": residue_key[1],
                "conservation_support": round(float(min(1.0, conservation_support)), 3),
                "conservation_confidence": round(float(min(1.0, conservation_confidence)), 3),
                "conservation_evidence_count": int(conservation_evidence_count),
            }
        )

    if not rows:
        return _empty_conservation_support_frame()
    return pd.DataFrame(rows)


def _external_row_payload(row: object) -> dict[str, object]:
    evidence_anchor_distance = _optional_finite_float(getattr(row, "evidence_anchor_distance", None))
    return {
        "external_support": round(float(getattr(row, "external_support", 0.0) or 0.0), 3),
        "external_confidence": round(float(getattr(row, "external_confidence", 0.0) or 0.0), 3),
        "external_evidence_count": int(getattr(row, "external_evidence_count", 0) or 0),
        "external_exact_match": bool(getattr(row, "external_exact_match", False)),
        "external_structure_verified": bool(getattr(row, "external_structure_verified", False)),
        "external_mapping_quality": round(float(getattr(row, "external_mapping_quality", 0.0) or 0.0), 3),
        "external_direct_anchor": bool(getattr(row, "external_direct_anchor", False)),
        "evidence_route_anchor": bool(getattr(row, "evidence_route_anchor", False)),
        "evidence_anchor_distance": (
            round(evidence_anchor_distance, 3)
            if evidence_anchor_distance is not None
            else None
        ),
        "evidence_anchor_proximity": round(float(getattr(row, "evidence_anchor_proximity", 0.0) or 0.0), 3),
        "evidence_anchor_residue": str(getattr(row, "evidence_anchor_residue", "") or ""),
        "external_direct_sources": str(getattr(row, "external_direct_sources", "") or ""),
        "external_evidence_types": str(getattr(row, "external_evidence_types", "") or ""),
        "external_evidence_notes": str(getattr(row, "external_evidence_notes", "") or ""),
        "conservation_support": round(float(getattr(row, "conservation_support", 0.0) or 0.0), 3),
        "conservation_confidence": round(float(getattr(row, "conservation_confidence", 0.0) or 0.0), 3),
        "conservation_evidence_count": int(getattr(row, "conservation_evidence_count", 0) or 0),
    }


def _build_precision_residue_table(
    atom_df: pd.DataFrame,
    hotspot_set: set[Tuple[str, int]],
    *,
    pdb_text: str = "",
    contact_cutoff: float,
    ligand_radius: float,
    external_site_df: Optional[pd.DataFrame] = None,
    conservation_site_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if atom_df is None or atom_df.empty or not {"x", "y", "z"}.issubset(atom_df.columns):
        return _empty_auto_pocket_table(), pd.DataFrame(columns=getattr(atom_df, "columns", []))

    protein_atoms = atom_df.copy()
    if "record_type" in protein_atoms.columns:
        protein_atoms = protein_atoms[protein_atoms["record_type"].astype(str).str.upper() == "ATOM"].copy()
    if protein_atoms.empty:
        return _empty_auto_pocket_table(), pd.DataFrame(columns=atom_df.columns)

    ligand_atoms = atom_df.copy()
    if "record_type" in ligand_atoms.columns:
        ligand_atoms = ligand_atoms[ligand_atoms["record_type"].astype(str).str.upper() == "HETATM"].copy()
    if ligand_atoms.empty:
        ligand_atoms = pd.DataFrame(columns=protein_atoms.columns)

    residue_rows = []
    for (chain, resid, resname), group in protein_atoms.groupby(["chain", "resid", "resname"], sort=True):
        coordinate_source = group[group["atom_name"] == "CA"]
        if coordinate_source.empty:
            coordinate_source = group

        centroid = coordinate_source[["x", "y", "z"]].mean().to_numpy(dtype=float)
        residue_rows.append(
            {
                "chain": _normalize_chain(chain),
                "resid": int(resid),
                "resname": str(resname).strip().upper(),
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]),
                "is_hotspot": (_normalize_chain(chain), int(resid)) in hotspot_set,
                "b_factor": float(pd.to_numeric(group["b_factor"], errors="coerce").mean()),
            }
        )

    residue_df = pd.DataFrame(residue_rows).sort_values(["chain", "resid"]).reset_index(drop=True)
    if residue_df.empty:
        return _empty_auto_pocket_table(), ligand_atoms

    coords = residue_df[["x", "y", "z"]].to_numpy(dtype=float)
    distances = _pairwise_distances(coords)

    if len(residue_df) > 1:
        contact_mask = (distances <= float(contact_cutoff)) & (distances > 0)
        contact_count = contact_mask.sum(axis=1)
        nearest_count = min(4, len(residue_df) - 1)
        sorted_distances = np.sort(distances, axis=1)
        nearest_mean = sorted_distances[:, 1 : nearest_count + 1].mean(axis=1)
    else:
        contact_count = np.zeros(len(residue_df), dtype=int)
        nearest_mean = np.zeros(len(residue_df), dtype=float)

    protein_center = coords.mean(axis=0)
    center_distance = np.sqrt(np.sum((coords - protein_center) ** 2, axis=1))

    if not ligand_atoms.empty and {"x", "y", "z"}.issubset(ligand_atoms.columns):
        ligand_coords = ligand_atoms[["x", "y", "z"]].to_numpy(dtype=float)
        ligand_distances = np.sqrt(np.sum((coords[:, None, :] - ligand_coords[None, :, :]) ** 2, axis=2))
        ligand_contact_count = (ligand_distances <= float(ligand_radius)).sum(axis=1)
        ligand_min_distance = ligand_distances.min(axis=1)
        ligand_norm = _normalize_numeric_series(pd.Series(ligand_min_distance, index=residue_df.index))
        ligand_proximity = 1.0 - ligand_norm.to_numpy(dtype=float)
    else:
        ligand_contact_count = np.zeros(len(residue_df), dtype=int)
        ligand_proximity = np.zeros(len(residue_df), dtype=float)

    if residue_df["is_hotspot"].any():
        hotspot_coords = residue_df.loc[residue_df["is_hotspot"], ["x", "y", "z"]].to_numpy(dtype=float)
        hotspot_distance = _compute_min_distances_to_points(coords, hotspot_coords)
        hotspot_norm = _normalize_numeric_series(pd.Series(hotspot_distance, index=residue_df.index))
        hotspot_proximity = 1.0 - hotspot_norm
    else:
        hotspot_distance = np.full(len(residue_df), np.inf, dtype=float)
        hotspot_proximity = np.zeros(len(residue_df), dtype=float)

    external_support_df = _build_external_support_frame(residue_df, external_site_df)
    if not external_support_df.empty:
        residue_df = residue_df.merge(external_support_df, on=["chain", "resid"], how="left")
    else:
        residue_df["external_support"] = 0.0
        residue_df["external_confidence"] = 0.0
        residue_df["external_evidence_count"] = 0
        residue_df["external_exact_match"] = False
        residue_df["external_structure_verified"] = False
        residue_df["external_mapping_quality"] = 0.0
        residue_df["external_direct_anchor"] = False
        residue_df["external_direct_sources"] = ""
        residue_df["external_evidence_types"] = ""
        residue_df["external_evidence_notes"] = ""

    conservation_support_df = _build_conservation_support_frame(residue_df, conservation_site_df)
    if not conservation_support_df.empty:
        residue_df = residue_df.merge(conservation_support_df, on=["chain", "resid"], how="left")
    else:
        residue_df["conservation_support"] = 0.0
        residue_df["conservation_confidence"] = 0.0
        residue_df["conservation_evidence_count"] = 0

    contact_norm = _normalize_numeric_series(pd.Series(contact_count, index=residue_df.index))
    centrality_norm = 1.0 - _normalize_numeric_series(pd.Series(center_distance, index=residue_df.index))
    compactness_norm = 1.0 - _normalize_numeric_series(pd.Series(nearest_mean, index=residue_df.index))
    ligand_contact_norm = _normalize_numeric_series(pd.Series(ligand_contact_count, index=residue_df.index))
    confidence_score, confidence_weight, confidence_mode = _infer_structure_confidence_signal(
        pdb_text,
        residue_df["b_factor"],
    )
    weighted_confidence = confidence_score * float(confidence_weight)
    external_support = pd.to_numeric(residue_df["external_support"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    seed_support = np.maximum.reduce([hotspot_proximity, ligand_proximity, external_support])

    residue_df["contact_count"] = contact_count
    residue_df["center_distance"] = center_distance
    residue_df["ligand_contact_count"] = ligand_contact_count
    residue_df["hotspot_distance"] = hotspot_distance
    residue_df["external_support"] = pd.to_numeric(residue_df["external_support"], errors="coerce").fillna(0.0)
    residue_df["external_confidence"] = pd.to_numeric(residue_df["external_confidence"], errors="coerce").fillna(0.0)
    residue_df["external_evidence_count"] = pd.to_numeric(residue_df["external_evidence_count"], errors="coerce").fillna(0).astype(int)
    residue_df["external_exact_match"] = residue_df["external_exact_match"].fillna(False).astype(bool)
    residue_df["external_structure_verified"] = residue_df["external_structure_verified"].fillna(False).astype(bool)
    residue_df["external_mapping_quality"] = pd.to_numeric(residue_df["external_mapping_quality"], errors="coerce").fillna(0.0)
    if "external_direct_anchor" not in residue_df.columns:
        residue_df["external_direct_anchor"] = False
    residue_df["external_direct_anchor"] = residue_df["external_direct_anchor"].fillna(False).astype(bool)
    for column in ("external_direct_sources", "external_evidence_types", "external_evidence_notes"):
        if column not in residue_df.columns:
            residue_df[column] = ""
        residue_df[column] = residue_df[column].fillna("").astype(str)
    residue_df["conservation_support"] = pd.to_numeric(residue_df["conservation_support"], errors="coerce").fillna(0.0)
    residue_df["conservation_confidence"] = pd.to_numeric(residue_df["conservation_confidence"], errors="coerce").fillna(0.0)
    residue_df["conservation_evidence_count"] = pd.to_numeric(residue_df["conservation_evidence_count"], errors="coerce").fillna(0).astype(int)
    residue_df["seed_support"] = seed_support
    residue_df["confidence_score"] = confidence_score
    residue_df["confidence_mode"] = confidence_mode
    residue_df["confidence_weight"] = float(confidence_weight)
    residue_df["pocket_score"] = (
        0.31 * contact_norm
        + 0.20 * centrality_norm
        + 0.13 * compactness_norm
        + 0.10 * ligand_contact_norm
        + 0.18 * residue_df["seed_support"]
        + 0.08 * weighted_confidence
    ).fillna(0.0)
    residue_df["precision_score"] = (
        0.58 * residue_df["pocket_score"]
        + 0.28 * residue_df["seed_support"]
        + 0.14 * weighted_confidence
    ).fillna(0.0)

    return residue_df, ligand_atoms


def parse_pocket_table(text: str) -> pd.DataFrame:
    pocket_df = pd.read_csv(pd.io.common.StringIO(text.strip()))
    missing = [column for column in REQUIRED_COLUMNS if column not in pocket_df.columns]
    if missing:
        raise ValueError(f"口袋文件缺少必要列: {', '.join(missing)}")
    return pocket_df


def _evidence_quality_assessment(row: pd.Series) -> dict[str, object]:
    external_total = int(row.get("external_evidence_total", 0) or 0)
    external_supported = int(row.get("external_supported_residue_count", 0) or 0)
    direct_anchor_count = int(row.get("external_direct_anchor_count", 0) or 0)
    route_anchor_count = int(row.get("evidence_route_anchor_count", 0) or 0)
    exact_count = int(row.get("external_exact_match_count", 0) or 0)
    direct_anchor_count = max(direct_anchor_count, exact_count)
    verified_count = int(row.get("external_structure_verified_count", 0) or 0)
    anchor_support = float(row.get("smart_evidence_anchor_support", 0.0) or 0.0)
    anchor_risk = float(row.get("smart_evidence_anchor_risk", 0.0) or 0.0)
    exact_ratio = float(row.get("external_exact_match_ratio", 0.0) or 0.0)
    mapping_quality = float(row.get("external_mapping_quality_mean", 0.0) or 0.0)

    if external_total <= 0 and external_supported <= 0:
        return {
            "evidence_quality_label": "no-external-evidence",
            "evidence_quality_score": 0.0,
            "evidence_quality_warning": "No curated external residue evidence is mapped to this pocket.",
        }

    verified_ratio = min(1.0, float(verified_count) / max(1.0, float(external_supported or 1)))
    quality_score = (
        0.42 * anchor_support
        + 0.22 * exact_ratio
        + 0.18 * verified_ratio
        + 0.18 * mapping_quality
        - 0.28 * anchor_risk
    )
    quality_score = _clamp_float(float(quality_score), 0.0, 1.0)

    if direct_anchor_count > 0 and exact_count > 0 and mapping_quality >= 0.70:
        label = "strong-direct-anchor"
        warning = "Direct exact external residue anchors support this pocket."
    elif direct_anchor_count > 0:
        label = "direct-anchor"
        warning = "Direct external residue anchors are present; inspect mapping/source confidence."
    elif route_anchor_count > 0:
        label = "route-anchor"
        warning = "Evidence-route anchors are present but direct anchor support is weaker."
    elif external_supported > 0 and anchor_risk >= 0.35:
        label = "neighborhood-expanded"
        warning = "External support is mostly from residues near an anchor, not direct key-residue hits."
    elif external_supported > 0:
        label = "diffuse-external-support"
        warning = "External support exists, but no direct key-residue anchor is in this pocket."
    else:
        label = "geometry-only"
        warning = "Pocket ranking is mainly geometric or local-structure based."

    return {
        "evidence_quality_label": label,
        "evidence_quality_score": round(float(quality_score), 3),
        "evidence_quality_warning": warning,
    }


def build_pocket_summary(pocket_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> pd.DataFrame:
    detection_meta = get_pocket_detection_metadata(pocket_df)
    external_sources = ""
    conservation_sources = ""
    if detection_meta:
        external_summary = detection_meta.get("external_evidence") or {}
        if isinstance(external_summary, dict):
            external_sources = str(external_summary.get("sources") or "")
        conservation_summary = detection_meta.get("conservation_evidence") or {}
        if isinstance(conservation_summary, dict):
            conservation_sources = str(conservation_summary.get("sources") or "")

    if pocket_df is None or pocket_df.empty:
        return pd.DataFrame(
            columns=[
                "pocket_id",
                "volume",
                "score",
                "residue_count",
                "hotspot_count",
                "detection_route",
                "consensus_methods",
                "method_vote_count",
                "consensus_score",
                "consensus_overlap_ratio",
                "smart_rank_score",
                "smart_rank_order",
                "smart_rank_label",
                "smart_rank_reason",
                "evidence_quality_label",
                "evidence_quality_score",
                "evidence_quality_warning",
                "smart_external_support",
                "smart_external_exact_ratio",
                "smart_external_verified_ratio",
                "smart_external_mapping_quality",
                "smart_evidence_anchor_support",
                "smart_evidence_anchor_risk",
                "smart_conservation_support",
                "smart_burial_support",
                "smart_exposure_penalty",
                "external_supported_residue_count",
                "external_evidence_total",
                "external_exact_match_count",
                "external_exact_match_ratio",
                "external_structure_verified_count",
                "external_support_mean",
                "external_confidence_mean",
                "external_mapping_quality_mean",
                "external_direct_anchor_count",
                "evidence_route_anchor_count",
                "evidence_anchor_min_distance",
                "evidence_anchor_max_proximity",
                "evidence_anchor_residues",
                "external_direct_sources",
                "external_evidence_types",
                "external_evidence_notes",
                "conservation_supported_residue_count",
                "conservation_evidence_total",
                "conservation_support_mean",
                "conservation_confidence_mean",
                "conservation_sources",
                "external_sources",
                "residue_labels",
            ]
        )

    hotspot_keys = set(zip(hotspot_df["chain"], hotspot_df["resid"])) if not hotspot_df.empty else set()
    records = []
    for pocket_id, group in pocket_df.groupby("pocket_id"):
        hotspot_count = sum((row.chain, int(row.resid)) in hotspot_keys for row in group.itertuples(index=False))
        route_values = []
        if "detection_route" in group.columns:
            route_values = sorted(
                {
                    str(value).strip()
                    for value in group["detection_route"].dropna().tolist()
                    if str(value).strip()
                }
            )
        external_support_series = (
            pd.to_numeric(group["external_support"], errors="coerce").fillna(0.0)
            if "external_support" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        external_confidence_series = (
            pd.to_numeric(group["external_confidence"], errors="coerce").fillna(0.0)
            if "external_confidence" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        external_evidence_series = (
            pd.to_numeric(group["external_evidence_count"], errors="coerce").fillna(0.0)
            if "external_evidence_count" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        external_exact_series = (
            group["external_exact_match"].fillna(False).astype(bool)
            if "external_exact_match" in group.columns
            else pd.Series([False] * len(group), index=group.index, dtype=bool)
        )
        external_verified_series = (
            group["external_structure_verified"].fillna(False).astype(bool)
            if "external_structure_verified" in group.columns
            else pd.Series([False] * len(group), index=group.index, dtype=bool)
        )
        external_mapping_quality_series = (
            pd.to_numeric(group["external_mapping_quality"], errors="coerce").fillna(0.0)
            if "external_mapping_quality" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        external_direct_anchor_series = (
            group["external_direct_anchor"].fillna(False).astype(bool)
            if "external_direct_anchor" in group.columns
            else pd.Series([False] * len(group), index=group.index, dtype=bool)
        )
        evidence_route_anchor_series = (
            group["evidence_route_anchor"].fillna(False).astype(bool)
            if "evidence_route_anchor" in group.columns
            else pd.Series([False] * len(group), index=group.index, dtype=bool)
        )
        evidence_anchor_distance_values = (
            pd.to_numeric(group["evidence_anchor_distance"], errors="coerce").dropna()
            if "evidence_anchor_distance" in group.columns
            else pd.Series(dtype=float)
        )
        evidence_anchor_proximity_series = (
            pd.to_numeric(group["evidence_anchor_proximity"], errors="coerce").fillna(0.0)
            if "evidence_anchor_proximity" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        conservation_support_series = (
            pd.to_numeric(group["conservation_support"], errors="coerce").fillna(0.0)
            if "conservation_support" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        conservation_confidence_series = (
            pd.to_numeric(group["conservation_confidence"], errors="coerce").fillna(0.0)
            if "conservation_confidence" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        conservation_evidence_series = (
            pd.to_numeric(group["conservation_evidence_count"], errors="coerce").fillna(0.0)
            if "conservation_evidence_count" in group.columns
            else pd.Series([0.0] * len(group), index=group.index, dtype=float)
        )
        conservation_supported_mask = conservation_support_series.gt(0.0)
        external_supported_mask = external_support_series.gt(0.0) | external_exact_series
        records.append(
            {
                "pocket_id": pocket_id,
                "volume": float(group["volume"].iloc[0]),
                "score": float(group["score"].iloc[0]),
                "residue_count": int(len(group)),
                "hotspot_count": int(hotspot_count),
                "detection_route": ", ".join(route_values) if route_values else None,
                "consensus_methods": ", ".join(
                    sorted({str(value).strip() for value in group["consensus_methods"].dropna().tolist() if str(value).strip()})
                )
                if "consensus_methods" in group.columns
                else None,
                "method_vote_count": int(pd.to_numeric(group["method_vote_count"], errors="coerce").fillna(0).max())
                if "method_vote_count" in group.columns
                else None,
                "consensus_score": float(pd.to_numeric(group["consensus_score"], errors="coerce").fillna(0.0).mean())
                if "consensus_score" in group.columns
                else None,
                "consensus_overlap_ratio": float(pd.to_numeric(group["consensus_overlap_ratio"], errors="coerce").fillna(0.0).max())
                if "consensus_overlap_ratio" in group.columns
                else None,
                "smart_rank_score": float(pd.to_numeric(group["smart_rank_score"], errors="coerce").fillna(0.0).max())
                if "smart_rank_score" in group.columns
                else None,
                "smart_rank_order": int(pd.to_numeric(group["smart_rank_order"], errors="coerce").fillna(0).min())
                if "smart_rank_order" in group.columns
                else None,
                "smart_rank_label": str(group["smart_rank_label"].dropna().astype(str).iloc[0])
                if "smart_rank_label" in group.columns and group["smart_rank_label"].notna().any()
                else None,
                "smart_rank_reason": str(group["smart_rank_reason"].dropna().astype(str).iloc[0])
                if "smart_rank_reason" in group.columns and group["smart_rank_reason"].notna().any()
                else None,
                "smart_external_support": float(pd.to_numeric(group["smart_external_support"], errors="coerce").fillna(0.0).max())
                if "smart_external_support" in group.columns
                else None,
                "smart_external_exact_ratio": float(pd.to_numeric(group["smart_external_exact_ratio"], errors="coerce").fillna(0.0).max())
                if "smart_external_exact_ratio" in group.columns
                else None,
                "smart_external_verified_ratio": float(pd.to_numeric(group["smart_external_verified_ratio"], errors="coerce").fillna(0.0).max())
                if "smart_external_verified_ratio" in group.columns
                else None,
                "smart_external_mapping_quality": float(pd.to_numeric(group["smart_external_mapping_quality"], errors="coerce").fillna(0.0).max())
                if "smart_external_mapping_quality" in group.columns
                else None,
                "smart_evidence_anchor_support": float(pd.to_numeric(group["smart_evidence_anchor_support"], errors="coerce").fillna(0.0).max())
                if "smart_evidence_anchor_support" in group.columns
                else None,
                "smart_evidence_anchor_risk": float(pd.to_numeric(group["smart_evidence_anchor_risk"], errors="coerce").fillna(0.0).max())
                if "smart_evidence_anchor_risk" in group.columns
                else None,
                "smart_conservation_support": float(pd.to_numeric(group["smart_conservation_support"], errors="coerce").fillna(0.0).max())
                if "smart_conservation_support" in group.columns
                else None,
                "smart_burial_support": float(pd.to_numeric(group["smart_burial_support"], errors="coerce").fillna(0.0).max())
                if "smart_burial_support" in group.columns
                else None,
                "smart_exposure_penalty": float(pd.to_numeric(group["smart_exposure_penalty"], errors="coerce").fillna(0.0).max())
                if "smart_exposure_penalty" in group.columns
                else None,
                "external_supported_residue_count": int(external_supported_mask.sum()),
                "external_evidence_total": int(external_evidence_series.sum()),
                "external_exact_match_count": int(external_exact_series.sum()),
                "external_exact_match_ratio": float(external_exact_series.mean()) if len(group) else 0.0,
                "external_structure_verified_count": int(external_verified_series.sum()),
                "external_support_mean": float(external_support_series.mean()) if len(group) else 0.0,
                "external_confidence_mean": float(external_confidence_series.mean()) if len(group) else 0.0,
                "external_mapping_quality_mean": float(external_mapping_quality_series.mean()) if len(group) else 0.0,
                "external_direct_anchor_count": int(external_direct_anchor_series.sum()),
                "evidence_route_anchor_count": int(evidence_route_anchor_series.sum()),
                "evidence_anchor_min_distance": float(evidence_anchor_distance_values.min())
                if not evidence_anchor_distance_values.empty
                else None,
                "evidence_anchor_max_proximity": float(evidence_anchor_proximity_series.max())
                if len(evidence_anchor_proximity_series)
                else 0.0,
                "evidence_anchor_residues": _join_unique_text(group["evidence_anchor_residue"], max_items=8, max_len=260)
                if "evidence_anchor_residue" in group.columns
                else None,
                "external_direct_sources": _join_unique_text(group["external_direct_sources"], max_items=8, max_len=320)
                if "external_direct_sources" in group.columns
                else None,
                "external_evidence_types": _join_unique_text(group["external_evidence_types"], max_items=8, max_len=320)
                if "external_evidence_types" in group.columns
                else None,
                "external_evidence_notes": _join_unique_text(group["external_evidence_notes"], max_items=6, max_len=760)
                if "external_evidence_notes" in group.columns
                else None,
                "conservation_supported_residue_count": int(conservation_supported_mask.sum()),
                "conservation_evidence_total": int(conservation_evidence_series.sum()),
                "conservation_support_mean": float(conservation_support_series.mean()) if len(group) else 0.0,
                "conservation_confidence_mean": float(conservation_confidence_series.mean()) if len(group) else 0.0,
                "conservation_sources": conservation_sources or None,
                "external_sources": external_sources or None,
                "residue_labels": ", ".join(
                    f"{row.resname} {row.chain}{int(row.resid)}" for row in group.itertuples(index=False)
                ),
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "pocket_id",
                "volume",
                "score",
                "residue_count",
                "hotspot_count",
                "detection_route",
                "consensus_methods",
                "method_vote_count",
                "consensus_score",
                "consensus_overlap_ratio",
                "smart_rank_score",
                "smart_rank_order",
                "smart_rank_label",
                "smart_rank_reason",
                "evidence_quality_label",
                "evidence_quality_score",
                "evidence_quality_warning",
                "smart_external_support",
                "smart_external_exact_ratio",
                "smart_external_verified_ratio",
                "smart_external_mapping_quality",
                "smart_evidence_anchor_support",
                "smart_evidence_anchor_risk",
                "smart_conservation_support",
                "smart_burial_support",
                "smart_exposure_penalty",
                "external_supported_residue_count",
                "external_evidence_total",
                "external_exact_match_count",
                "external_exact_match_ratio",
                "external_structure_verified_count",
                "external_support_mean",
                "external_confidence_mean",
                "external_mapping_quality_mean",
                "external_direct_anchor_count",
                "evidence_route_anchor_count",
                "evidence_anchor_min_distance",
                "evidence_anchor_max_proximity",
                "evidence_anchor_residues",
                "external_direct_sources",
                "external_evidence_types",
                "external_evidence_notes",
                "conservation_supported_residue_count",
                "conservation_evidence_total",
                "conservation_support_mean",
                "conservation_confidence_mean",
                "conservation_sources",
                "external_sources",
                "residue_labels",
            ]
        )

    summary_df = pd.DataFrame(records)
    evidence_quality_df = pd.DataFrame(
        [_evidence_quality_assessment(row) for _, row in summary_df.iterrows()],
        index=summary_df.index,
    )
    if not evidence_quality_df.empty:
        for column in evidence_quality_df.columns:
            summary_df[column] = evidence_quality_df[column]
    for column in (
        "evidence_quality_score",
        "smart_external_support",
        "smart_external_exact_ratio",
        "smart_external_verified_ratio",
        "smart_external_mapping_quality",
        "smart_evidence_anchor_support",
        "smart_evidence_anchor_risk",
        "smart_conservation_support",
        "smart_burial_support",
        "smart_exposure_penalty",
        "external_exact_match_ratio",
        "external_support_mean",
        "external_confidence_mean",
        "external_mapping_quality_mean",
        "evidence_anchor_min_distance",
        "evidence_anchor_max_proximity",
        "conservation_support_mean",
        "conservation_confidence_mean",
    ):
        if column in summary_df.columns:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce").fillna(0.0).round(3)
    sort_columns = ["hotspot_count", "score"]
    sort_ascending = [False, False]
    if "smart_rank_order" in summary_df.columns and summary_df["smart_rank_order"].notna().any():
        sort_columns = ["smart_rank_order", "hotspot_count", "score"]
        sort_ascending = [True, False, False]
    elif "method_vote_count" in summary_df.columns:
        sort_columns.insert(1, "method_vote_count")
        sort_ascending.insert(1, False)
    return summary_df.sort_values(sort_columns, ascending=sort_ascending).reset_index(drop=True)


def build_pocket_summary_without_conservation_signal(pocket_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> pd.DataFrame:
    if pocket_df is None or getattr(pocket_df, "empty", True):
        return build_pocket_summary(pocket_df, hotspot_df)

    working = pocket_df.copy()
    drop_columns = [column for column in SMART_POCKET_COLUMNS if column in working.columns]
    if drop_columns:
        working = working.drop(columns=drop_columns)

    for column in ("conservation_support", "conservation_confidence"):
        if column in working.columns:
            working[column] = 0.0
    if "conservation_evidence_count" in working.columns:
        working["conservation_evidence_count"] = 0

    ranked = rank_detected_pockets(working)
    ranked.attrs.update(getattr(pocket_df, "attrs", {}) or {})
    return build_pocket_summary(ranked, hotspot_df)


def build_auto_pocket_display_table(
    atom_df: pd.DataFrame,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    *,
    hotspot_residues: Optional[Sequence[Tuple[str, int]]] = None,
    pocket_id: Optional[str] = None,
    limit_hotspots_to_pocket: bool = False,
    pocket_residue_layers: Optional[object] = None,
) -> pd.DataFrame:
    if atom_df is None or atom_df.empty:
        return pd.DataFrame(
            columns=[
                "chain",
                "resid",
                "resname",
                "label",
                "delta_total",
                "display_color",
                "classification_label",
                "classification_color",
                "classification_description",
                "heat_color",
                "is_pocket",
                "is_hotspot",
                "pocket_id",
                "pocket_layer",
            ]
        )

    residue_df = atom_df[["chain", "resid", "resname"]].drop_duplicates().sort_values(["chain", "resid"]).reset_index(drop=True)
    pocket_set = _normalized_residue_set(pocket_residues)
    hotspot_set = _normalized_residue_set(hotspot_residues)
    layer_map = _normalized_residue_layer_map(pocket_residue_layers)
    if limit_hotspots_to_pocket:
        # In scoped mode only keep hotspots that belong to the selected pocket;
        # if no pocket is selected, hotspot overlay should be empty.
        hotspot_set = hotspot_set.intersection(pocket_set)

    palette = {
        "background": "#d1d5db",
        "pocket": "#2563eb",
        "hotspot": "#f97316",
        "overlap": "#ef4444",
        "core": "#dc2626",
        "shell": "#2563eb",
        "rim": "#14b8a6",
        "core_hotspot": "#991b1b",
        "shell_hotspot": "#ea580c",
        "rim_hotspot": "#f59e0b",
    }

    records = []
    for row in residue_df.itertuples(index=False):
        chain = _normalize_chain(getattr(row, "chain", "A"))
        resid = int(getattr(row, "resid", 0))
        resname = str(getattr(row, "resname", "")).strip().upper()
        residue_key = (chain, resid)
        is_pocket = residue_key in pocket_set
        is_hotspot = residue_key in hotspot_set
        pocket_layer = str(layer_map.get(residue_key, "") or "").strip().lower() if is_pocket else ""

        if is_pocket and pocket_layer in {"core", "shell", "rim"}:
            if is_hotspot:
                label = f"pocket {pocket_layer} + hotspot"
                color = palette.get(f"{pocket_layer}_hotspot", palette["overlap"])
                description = f"Selected pocket {pocket_layer} residue with hotspot support."
            else:
                label = f"pocket {pocket_layer}"
                color = palette[pocket_layer]
                description = f"Selected pocket {pocket_layer} residue."
        elif is_pocket and is_hotspot:
            label = "自动口袋 + 热点"
            color = palette["overlap"]
            description = "同时属于自动口袋和热点残基，优先级最高。"
        elif is_pocket:
            label = "自动口袋"
            color = palette["pocket"]
            description = "自动识别出的口袋候选残基。"
        elif is_hotspot:
            label = "热点"
            color = palette["hotspot"]
            description = "当前能量阈值筛出的热点残基。"
        else:
            label = "背景"
            color = palette["background"]
            description = "非当前口袋和热点集合中的残基。"

        records.append(
            {
                "chain": chain,
                "resid": resid,
                "resname": resname,
                "label": f"{resname} {chain}{resid}".strip(),
                "delta_total": 1.0,
                "display_color": color,
                "classification_label": label,
                "classification_color": color,
                "classification_description": description,
                "heat_color": color,
                "is_pocket": is_pocket,
                "is_hotspot": is_hotspot,
                "pocket_id": pocket_id if is_pocket else None,
                "pocket_layer": pocket_layer if is_pocket else "",
            }
        )

    return pd.DataFrame(records)


def _normalize_chain(chain: object) -> str:
    value = str(chain or "").strip()
    return value if value else "A"


def _normalized_residue_set(residue_pairs: Optional[Sequence[Tuple[str, int]]]) -> set[Tuple[str, int]]:
    return {(_normalize_chain(chain), int(resid)) for chain, resid in (residue_pairs or [])}


def _normalized_residue_layer_map(pocket_residue_layers: Optional[object]) -> dict[Tuple[str, int], str]:
    if pocket_residue_layers is None:
        return {}

    if isinstance(pocket_residue_layers, pd.DataFrame):
        if pocket_residue_layers.empty or not {"chain", "resid", "pocket_layer"}.issubset(pocket_residue_layers.columns):
            return {}
        layer_map: dict[Tuple[str, int], str] = {}
        for row in pocket_residue_layers.itertuples(index=False):
            try:
                key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
            except (TypeError, ValueError):
                continue
            layer = str(getattr(row, "pocket_layer", "") or "").strip().lower()
            if layer in {"core", "shell", "rim"}:
                layer_map[key] = layer
        return layer_map

    if isinstance(pocket_residue_layers, dict):
        layer_map: dict[Tuple[str, int], str] = {}
        for key, value in pocket_residue_layers.items():
            try:
                if isinstance(key, tuple) and len(key) >= 2:
                    chain, resid = key[0], key[1]
                else:
                    chain, resid = "A", key
                normalized_key = (_normalize_chain(chain), int(resid))
            except (TypeError, ValueError):
                continue
            layer = str(value or "").strip().lower()
            if layer in {"core", "shell", "rim"}:
                layer_map[normalized_key] = layer
        return layer_map

    return {}


def _normalize_numeric_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)

    numeric_values = pd.to_numeric(values, errors="coerce")
    if not numeric_values.notna().any():
        return pd.Series(np.full(len(numeric_values), 0.5), index=numeric_values.index)

    fill_value = float(numeric_values.dropna().median())
    filled = numeric_values.fillna(fill_value)
    minimum = float(filled.min())
    maximum = float(filled.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.full(len(filled), 0.5), index=filled.index)
    return (filled - minimum) / (maximum - minimum)


def _numeric_frame_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _optional_finite_float(value: object) -> Optional[float]:
    try:
        numeric_value = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _boolean_frame_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=bool), index=frame.index)
    return frame[column].fillna(default).astype(bool)


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, float(value))))


def _infer_structure_confidence_signal(
    pdb_text: str,
    b_factors: pd.Series,
) -> tuple[pd.Series, float, str]:
    numeric_values = pd.to_numeric(b_factors, errors="coerce")
    if numeric_values.empty:
        return pd.Series(dtype=float), 0.0, "missing"

    if not numeric_values.notna().any():
        return pd.Series(np.full(len(numeric_values), 0.5), index=numeric_values.index), 0.0, "missing"

    normalized = _normalize_numeric_series(numeric_values)
    upper_text = str(pdb_text or "").upper()
    predicted_markers = (
        "ALPHAFOLD",
        "PLDDT",
        "PREDICTED MODEL",
        "MODEL GENERATED BY",
        "AFDB",
    )
    if any(marker in upper_text for marker in predicted_markers):
        return normalized.fillna(0.5), 1.0, "predicted-model"

    valid_values = numeric_values.dropna()
    interquartile_span = float(valid_values.quantile(0.75) - valid_values.quantile(0.25)) if not valid_values.empty else 0.0
    if int(valid_values.nunique()) <= 2 or interquartile_span < 1.5:
        return pd.Series(np.full(len(numeric_values), 0.5), index=numeric_values.index), 0.35, "weak-bfactor"

    return (1.0 - normalized).fillna(0.5), 1.0, "experimental-bfactor"


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 1:
        return np.zeros((len(coords), len(coords)), dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _estimate_bbox_volume(coords: np.ndarray, padding: float = 2.0) -> float:
    if coords.size == 0:
        return 0.0
    min_corner = coords.min(axis=0)
    max_corner = coords.max(axis=0)
    extents = np.maximum(max_corner - min_corner, 0.0) + 2.0 * float(padding)
    return float(np.prod(extents))


def _empty_auto_pocket_table() -> pd.DataFrame:
    return pd.DataFrame(columns=AUTO_POCKET_COLUMNS)


def _ensure_auto_pocket_columns(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return _empty_auto_pocket_table()

    normalized = table.copy()
    for column in AUTO_POCKET_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None

    ordered_columns = AUTO_POCKET_COLUMNS + [
        column for column in normalized.columns if column not in AUTO_POCKET_COLUMNS
    ]
    return normalized[ordered_columns]


def _build_residue_center_map(atom_df: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    if atom_df is None or atom_df.empty:
        return {}

    protein_atoms = atom_df.copy()
    if "record_type" in protein_atoms.columns:
        protein_atoms = protein_atoms[protein_atoms["record_type"].astype(str).str.upper() == "ATOM"].copy()
    if protein_atoms.empty:
        return {}

    residue_center_map: Dict[Tuple[str, int], np.ndarray] = {}
    group_columns = [column for column in ["chain", "resid", "resname"] if column in protein_atoms.columns]
    if len(group_columns) < 3:
        return {}

    for (chain, resid, _), group in protein_atoms.groupby(group_columns, sort=True):
        coordinate_source = pd.DataFrame()

        if "atom_type" in group.columns:
            coordinate_source = group[group["atom_type"].astype(str).str.lower() == "sidechain"]

        if coordinate_source.empty and "atom_name" in group.columns:
            coordinate_source = group[group["atom_name"].astype(str).str.upper() == "CA"]

        if coordinate_source.empty:
            coordinate_source = group

        if coordinate_source.empty or not {"x", "y", "z"}.issubset(coordinate_source.columns):
            continue

        center = coordinate_source[["x", "y", "z"]].mean().to_numpy(dtype=float)
        residue_center_map[(_normalize_chain(chain), int(resid))] = center

    return residue_center_map


def _build_detection_profile(
    *,
    contact_cutoff: float,
    cluster_cutoff: float,
    ligand_radius: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
    kv_step: float,
    kv_probe_in: float,
    kv_probe_out: float,
    kv_volume_cutoff: float,
) -> Dict[str, object]:
    return {
        "contact_cutoff": float(contact_cutoff),
        "cluster_cutoff": float(cluster_cutoff),
        "ligand_radius": float(ligand_radius),
        "top_fraction": float(top_fraction),
        "min_candidates": int(min_candidates),
        "max_candidates": int(max_candidates),
        "max_pockets": int(max_pockets),
        "kv_step": float(kv_step),
        "kv_probe_in": float(kv_probe_in),
        "kv_probe_out": float(kv_probe_out),
        "kv_volume_cutoff": float(kv_volume_cutoff),
        "kv_probe_profiles": [
            {
                "step": round(float(kv_step), 3),
                "probe_in": round(float(kv_probe_in), 3),
                "probe_out": round(float(kv_probe_out), 3),
                "volume_cutoff": round(float(kv_volume_cutoff), 3),
            }
        ],
    }


def _infer_adaptive_detection_profile(
    residue_df: pd.DataFrame,
    ligand_atoms: pd.DataFrame,
    *,
    hotspot_set: set[Tuple[str, int]],
    contact_cutoff: float,
    cluster_cutoff: float,
    ligand_radius: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
    kv_step: float,
    kv_probe_in: float,
    kv_probe_out: float,
    kv_volume_cutoff: float,
) -> Dict[str, object]:
    defaults = _build_detection_profile(
        contact_cutoff=contact_cutoff,
        cluster_cutoff=cluster_cutoff,
        ligand_radius=ligand_radius,
        top_fraction=top_fraction,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        max_pockets=max_pockets,
        kv_step=kv_step,
        kv_probe_in=kv_probe_in,
        kv_probe_out=kv_probe_out,
        kv_volume_cutoff=kv_volume_cutoff,
    )

    if residue_df is None or residue_df.empty or not {"x", "y", "z"}.issubset(residue_df.columns):
        return defaults

    residue_count = int(len(residue_df))
    if residue_count <= 1:
        return defaults

    coords = residue_df[["x", "y", "z"]].to_numpy(dtype=float)
    distances = _pairwise_distances(coords)
    sorted_distances = np.sort(distances, axis=1)
    neighbor_width = min(4, residue_count - 1)
    neighbor_slice = sorted_distances[:, 1 : neighbor_width + 1]
    if neighbor_slice.size == 0:
        return defaults

    nearest_neighbor = neighbor_slice[:, 0]
    neighbor_mean = neighbor_slice.mean(axis=1)
    protein_center = coords.mean(axis=0)
    protein_radius = np.sqrt(np.sum((coords - protein_center) ** 2, axis=1))
    radius_90 = float(np.percentile(protein_radius, 90)) if protein_radius.size else 0.0

    hotspot_ratio = float(len(hotspot_set)) / float(max(1, residue_count))
    ligand_present = not ligand_atoms.empty
    ligand_signal = 1.0 if ligand_present else 0.0

    nearest_median = float(np.percentile(nearest_neighbor, 50))
    neighbor_q60 = float(np.percentile(neighbor_mean, 60))
    neighbor_q75 = float(np.percentile(neighbor_mean, 75))

    adaptive_contact = _clamp_float(
        0.55 * float(contact_cutoff) + 0.45 * (neighbor_q60 + 1.0 + 0.45 * hotspot_ratio),
        5.8,
        8.8,
    )
    # Keep cluster threshold stable to avoid unexpected pocket-count drift.
    adaptive_cluster = float(cluster_cutoff)
    if adaptive_contact >= adaptive_cluster:
        adaptive_contact = max(5.8, adaptive_cluster - 0.05)
    if adaptive_contact >= adaptive_cluster:
        adaptive_cluster = min(10.5, adaptive_contact + 0.05)

    adaptive_ligand_radius = _clamp_float(
        0.45 * float(ligand_radius) + 0.55 * (nearest_median + 0.55 + 0.35 * ligand_signal),
        3.4,
        max(3.8, adaptive_contact - 0.45),
    )

    # Respect caller controls for candidate volume and pocket count.
    adaptive_top_fraction = _clamp_float(float(top_fraction), 0.16, 0.60)
    adaptive_min_candidates = max(int(min_candidates), 2)
    adaptive_max_candidates = max(adaptive_min_candidates + 1, int(max_candidates))
    adaptive_max_pockets = max(1, int(max_pockets))

    adaptive_kv_step = _clamp_float(
        0.44 + 0.06 * (radius_90 / 18.0) + 0.02 * ligand_signal,
        0.42,
        0.68,
    )
    adaptive_kv_probe_in = _clamp_float(
        0.85 + 0.12 * adaptive_ligand_radius + 0.03 * min(1.0, hotspot_ratio * 4.0),
        1.2,
        1.6,
    )
    adaptive_kv_probe_out = _clamp_float(
        1.75 + 0.30 * adaptive_contact + 0.20 * ligand_signal + 0.10 * (radius_90 / 15.0),
        adaptive_kv_probe_in + 1.8,
        5.6,
    )
    adaptive_kv_volume_cutoff = _clamp_float(
        float(kv_volume_cutoff)
        + max(0.0, (adaptive_kv_step - 0.5) * 12.0)
        + max(0.0, (adaptive_kv_probe_out - 4.0) * 1.5),
        4.0,
        12.0,
    )

    probe_profiles: list[Dict[str, float]] = []
    profile_offsets = (-0.45, 0.0, 0.55 if residue_count >= 120 else 0.4)
    step_offsets = (-0.04, 0.0, 0.05)
    seen_profiles: set[tuple[float, float, float, float]] = set()
    for probe_offset, step_offset in zip(profile_offsets, step_offsets):
        profile_step = _clamp_float(adaptive_kv_step + step_offset, 0.42, 0.70)
        profile_probe_out = _clamp_float(
            adaptive_kv_probe_out + probe_offset,
            adaptive_kv_probe_in + 1.7,
            5.8,
        )
        profile_volume_cutoff = _clamp_float(
            adaptive_kv_volume_cutoff * (0.94 + 0.07 * len(probe_profiles)),
            4.0,
            12.0,
        )
        profile_key = (
            round(profile_step, 3),
            round(adaptive_kv_probe_in, 3),
            round(profile_probe_out, 3),
            round(profile_volume_cutoff, 3),
        )
        if profile_key in seen_profiles:
            continue
        seen_profiles.add(profile_key)
        probe_profiles.append(
            {
                "step": profile_key[0],
                "probe_in": profile_key[1],
                "probe_out": profile_key[2],
                "volume_cutoff": profile_key[3],
            }
        )

    if not probe_profiles:
        probe_profiles = defaults["kv_probe_profiles"]

    return {
        "contact_cutoff": round(adaptive_contact, 3),
        "cluster_cutoff": round(adaptive_cluster, 3),
        "ligand_radius": round(adaptive_ligand_radius, 3),
        "top_fraction": round(adaptive_top_fraction, 3),
        "min_candidates": adaptive_min_candidates,
        "max_candidates": adaptive_max_candidates,
        "max_pockets": adaptive_max_pockets,
        "kv_step": round(adaptive_kv_step, 3),
        "kv_probe_in": round(adaptive_kv_probe_in, 3),
        "kv_probe_out": round(adaptive_kv_probe_out, 3),
        "kv_volume_cutoff": round(adaptive_kv_volume_cutoff, 3),
        "kv_probe_profiles": probe_profiles,
    }


def _build_kvfinder_name_label_map(cavities: np.ndarray, cavity_names: Sequence[str]) -> Dict[str, int]:
    labels = sorted(int(value) for value in np.unique(cavities) if int(value) >= 2)
    names = sorted(str(name) for name in cavity_names)
    if not labels or len(labels) != len(names):
        return {}

    return {name: int(label) for label, name in zip(labels, names)}


def _build_kvfinder_grid_axes(vertices: np.ndarray, shape: Tuple[int, int, int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        vertices_array = np.asarray(vertices, dtype=float)
    except Exception:
        return None

    if vertices_array.shape != (4, 3):
        return None

    nx, ny, nz = shape
    origin = vertices_array[0]
    axis_x = (vertices_array[1] - origin) / float(max(nx - 1, 1))
    axis_y = (vertices_array[2] - origin) / float(max(ny - 1, 1))
    axis_z = (vertices_array[3] - origin) / float(max(nz - 1, 1))
    return origin, axis_x, axis_y, axis_z


def _sample_kvfinder_cavity_points(
    cavities: np.ndarray,
    label: int,
    grid_axes: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    max_points: int = 2500,
) -> np.ndarray:
    indices = np.argwhere(cavities == int(label))
    if indices.size == 0:
        return np.empty((0, 3), dtype=float)

    if int(max_points) > 0 and len(indices) > int(max_points):
        sampled_index = np.linspace(0, len(indices) - 1, num=int(max_points), dtype=int)
        indices = indices[sampled_index]

    origin, axis_x, axis_y, axis_z = grid_axes
    points = (
        origin[None, :]
        + indices[:, [0]] * axis_x[None, :]
        + indices[:, [1]] * axis_y[None, :]
        + indices[:, [2]] * axis_z[None, :]
    )
    return points.astype(float)


def _compute_min_distances_to_points(
    centers: np.ndarray,
    points: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    if centers.size == 0 or points.size == 0:
        return np.array([], dtype=float)

    chunk = max(1, int(chunk_size))
    minimum_distances = np.full(len(centers), np.inf, dtype=float)
    for start in range(0, len(centers), chunk):
        stop = min(start + chunk, len(centers))
        center_batch = centers[start:stop]
        diff = center_batch[:, None, :] - points[None, :, :]
        batch_distances = np.sqrt(np.sum(diff * diff, axis=2))
        minimum_distances[start:stop] = batch_distances.min(axis=1)
    return minimum_distances


def _refine_kvfinder_residues(
    residues: list[tuple[str, int, str]],
    residue_center_map: Dict[Tuple[str, int], np.ndarray],
    cavity_points: np.ndarray,
    *,
    depth_avg: float,
    hotspot_set: Optional[set[Tuple[str, int]]] = None,
    support_score_map: Optional[Dict[Tuple[str, int], float]] = None,
    confidence_score_map: Optional[Dict[Tuple[str, int], float]] = None,
) -> tuple[list[tuple[str, int, str]], Dict[Tuple[str, int], float], Optional[float], Dict[Tuple[str, int], float]]:
    if not residues or not residue_center_map or cavity_points.size == 0:
        return residues, {}, None, {}

    known_pairs: list[tuple[tuple[str, int, str], np.ndarray]] = []
    for chain, resid, resname in residues:
        residue_key = (chain, resid)
        center = residue_center_map.get(residue_key)
        if center is None:
            continue
        known_pairs.append(((chain, resid, resname), center))

    if not known_pairs:
        return residues, {}, None, {}

    centers = np.vstack([center for _, center in known_pairs]).astype(float)
    distances = _compute_min_distances_to_points(centers, cavity_points)

    distance_map: Dict[Tuple[str, int], float] = {}
    for (chain, resid, _), distance in zip([item[0] for item in known_pairs], distances):
        distance_map[(chain, resid)] = float(distance)

    if not distance_map:
        return residues, {}, None, {}

    valid_distances = np.array(list(distance_map.values()), dtype=float)
    percentile_cutoff = float(np.percentile(valid_distances, 60))
    depth_bonus = max(0.0, min(1.0, float(depth_avg) * 0.2))
    proximity_cutoff = float(max(1.9, min(4.4, percentile_cutoff + depth_bonus)))

    scored_entries: list[tuple[float, float, float, float, tuple[str, int, str]]] = []
    for chain, resid, resname in residues:
        residue_key = (chain, resid)
        distance = distance_map.get(residue_key)
        if distance is None:
            continue

        proximity_score = max(0.0, 1.0 - (distance / max(proximity_cutoff, 1e-6)))
        support_score = float(support_score_map.get(residue_key, 0.0)) if support_score_map else 0.0
        confidence_score = float(confidence_score_map.get(residue_key, 0.5)) if confidence_score_map else 0.5
        precision_score = 0.56 * proximity_score + 0.26 * support_score + 0.18 * confidence_score
        if hotspot_set and residue_key in hotspot_set:
            precision_score += 0.06
        scored_entries.append((precision_score, proximity_score, support_score, confidence_score, (chain, resid, resname)))

    if not scored_entries:
        return residues, distance_map, proximity_cutoff, {}

    precision_values = np.array([item[0] for item in scored_entries], dtype=float)
    precision_cutoff = float(np.percentile(precision_values, 58))
    support_threshold = 0.18
    confidence_threshold = 0.42
    filtered_known = [
        entry
        for precision_score, proximity_score, support_score, confidence_score, entry in scored_entries
        if (
            precision_score >= precision_cutoff
            and confidence_score >= 0.28
        )
        or support_score >= support_threshold
        or confidence_score >= confidence_threshold
        or (hotspot_set is not None and (entry[0], entry[1]) in hotspot_set)
    ]

    minimum_keep = min(len(scored_entries), max(1, int(math.ceil(len(scored_entries) * 0.35))))
    if len(filtered_known) < minimum_keep:
        ranked_entries = sorted(scored_entries, key=lambda item: (item[0], item[2], item[3]), reverse=True)
        filtered_known = [entry for _, _, _, _, entry in ranked_entries[:minimum_keep]]

    filtered_keys = {(chain, resid) for chain, resid, _ in filtered_known}
    refined_residues: list[tuple[str, int, str]] = []
    for chain, resid, resname in residues:
        residue_key = (chain, resid)
        if residue_key not in distance_map or residue_key in filtered_keys:
            refined_residues.append((chain, resid, resname))

    precision_map = { (chain, resid): precision_score for precision_score, _, _, _, (chain, resid, _) in scored_entries }
    return refined_residues or residues, distance_map, proximity_cutoff, precision_map


def _detect_with_kvfinder(
    pdb_text: str,
    hotspot_set: set[Tuple[str, int]],
    *,
    step: float = 0.6,
    probe_in: float = 1.4,
    probe_out: float = 4.0,
    volume_cutoff: float = 5.0,
    residue_support_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if not PYKVFINDER_AVAILABLE:
        return _empty_auto_pocket_table()

    pdb_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as handle:
            handle.write(pdb_text)
            pdb_path = handle.name

        atomic = pyKVFinder.read_pdb(pdb_path)
        if atomic is None or len(atomic) == 0:
            return _empty_auto_pocket_table()

        vertices = pyKVFinder.get_vertices(atomic, probe_out=float(probe_out), step=float(step))
        cavity_count, cavities = pyKVFinder.detect(
            atomic,
            vertices,
            step=float(step),
            probe_in=float(probe_in),
            probe_out=float(probe_out),
            volume_cutoff=float(volume_cutoff),
            surface="SES",
        )

        if int(cavity_count) <= 0:
            return _empty_auto_pocket_table()

        _, volumes, _ = pyKVFinder.spatial(cavities, step=float(step))
        _, max_depths, avg_depths = pyKVFinder.depth(cavities, step=float(step))
        constitutional_sidechain = pyKVFinder.constitutional(
            cavities,
            atomic,
            vertices,
            step=float(step),
            probe_in=float(probe_in),
            ignore_backbone=True,
        )
        constitutional_all = pyKVFinder.constitutional(
            cavities,
            atomic,
            vertices,
            step=float(step),
            probe_in=float(probe_in),
            ignore_backbone=False,
        )

        atom_df = parse_pdb_atoms(pdb_text)
        residue_center_map = _build_residue_center_map(atom_df)
        seed_support_map: Dict[Tuple[str, int], float] = {}
        confidence_score_map: Dict[Tuple[str, int], float] = {}
        precision_score_map: Dict[Tuple[str, int], float] = {}
        external_row_map: Dict[Tuple[str, int], object] = {}
        if residue_support_df is not None and not residue_support_df.empty:
            for row in residue_support_df.itertuples(index=False):
                residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
                seed_support_map[residue_key] = float(getattr(row, "seed_support", 0.0) or 0.0)
                confidence_score_map[residue_key] = float(getattr(row, "confidence_score", 0.5) or 0.5)
                precision_score_map[residue_key] = float(getattr(row, "precision_score", getattr(row, "pocket_score", 0.0)) or 0.0)
                external_row_map[residue_key] = row
        name_to_label_map = _build_kvfinder_name_label_map(cavities, volumes.keys())
        grid_axes = _build_kvfinder_grid_axes(vertices, tuple(cavities.shape))
        cavity_point_cache: Dict[int, np.ndarray] = {}

        rows = []
        cavity_names = sorted(set(volumes.keys()) | set(constitutional_sidechain.keys()) | set(constitutional_all.keys()))
        for cavity_name in cavity_names:
            residues = constitutional_sidechain.get(cavity_name) or constitutional_all.get(cavity_name) or []
            if not residues:
                continue

            parsed_residues: list[tuple[str, int, str]] = []
            seen_keys: set[tuple[str, int]] = set()
            for residue in residues:
                try:
                    resid = int(residue[0])
                    chain = _normalize_chain(residue[1])
                    resname = str(residue[2]).strip().upper()
                except Exception:
                    continue

                residue_key = (chain, resid)
                if residue_key in seen_keys:
                    continue
                seen_keys.add(residue_key)
                parsed_residues.append((chain, resid, resname))

            if not parsed_residues:
                continue

            volume_value = float(volumes.get(cavity_name, 0.0))
            depth_avg = float(avg_depths.get(cavity_name, 0.0))
            depth_max = float(max_depths.get(cavity_name, 0.0))

            refined_residues = parsed_residues
            distance_map: Dict[Tuple[str, int], float] = {}
            proximity_cutoff: Optional[float] = None
            precision_map: Dict[Tuple[str, int], float] = {}
            cavity_label = name_to_label_map.get(cavity_name)
            if cavity_label is not None and grid_axes is not None:
                cavity_points = cavity_point_cache.get(cavity_label)
                if cavity_points is None:
                    cavity_points = _sample_kvfinder_cavity_points(
                        cavities,
                        cavity_label,
                        grid_axes,
                        max_points=2500,
                    )
                    cavity_point_cache[cavity_label] = cavity_points
                if cavity_points.size > 0:
                    refined_residues, distance_map, proximity_cutoff, precision_map = _refine_kvfinder_residues(
                        parsed_residues,
                        residue_center_map,
                        cavity_points,
                        depth_avg=depth_avg,
                        hotspot_set=hotspot_set,
                        support_score_map=seed_support_map,
                        confidence_score_map=confidence_score_map,
                    )

            hotspot_overlap = sum((chain, resid) in hotspot_set for chain, resid, _ in refined_residues)
            overlap_ratio = float(hotspot_overlap) / float(len(refined_residues)) if refined_residues else 0.0
            precision_values = [float(precision_map.get((chain, resid), precision_score_map.get((chain, resid), 0.0))) for chain, resid, _ in refined_residues]
            support_values = [float(seed_support_map.get((chain, resid), 0.0)) for chain, resid, _ in refined_residues]
            cavity_precision_mean = float(np.mean(precision_values)) if precision_values else 0.0
            cavity_support_mean = float(np.mean(support_values)) if support_values else 0.0

            cavity_score = (
                (volume_value / 110.0)
                + (1.05 * depth_avg)
                + (1.65 * overlap_ratio)
                + (0.45 * cavity_precision_mean)
                + (0.25 * cavity_support_mean)
            )
            route_suffix = "consensus" if (cavity_support_mean > 0.0 or hotspot_overlap > 0) else "kvfinder"
            detection_route = _detection_route_label(route_suffix)

            for chain, resid, resname in refined_residues:
                residue_key = (chain, resid)
                proximity_distance = distance_map.get(residue_key)
                if proximity_distance is None or proximity_cutoff is None:
                    proximity_score = 0.5
                else:
                    proximity_score = max(0.0, 1.0 - (proximity_distance / max(proximity_cutoff, 1e-6)))

                support_score = float(seed_support_map.get(residue_key, 0.0))
                confidence_score = float(confidence_score_map.get(residue_key, 0.5))
                precision_score = float(precision_map.get(residue_key, precision_score_map.get(residue_key, 0.0)))

                residue_score = cavity_score * (0.6 + 0.4 * proximity_score)
                residue_score += 0.2 * precision_score
                if residue_key in hotspot_set:
                    residue_score += 0.2

                rows.append(
                    {
                        "pocket_id": f"KVFinder-{cavity_name}",
                        "chain": chain,
                        "resid": resid,
                        "resname": resname,
                        "volume": round(volume_value, 3),
                        "score": round(cavity_score, 3),
                        "residue_score": round(residue_score, 3),
                        "contact_count": None,
                        "center_distance": None,
                        "ligand_contact_count": None,
                        "detection_method": "kvfinder-refined",
                        "detection_route": detection_route,
                        "is_hotspot": residue_key in hotspot_set,
                        "depth_avg": round(depth_avg, 3),
                        "depth_max": round(depth_max, 3),
                        "overlap_ratio": round(overlap_ratio, 3),
                        "proximity_distance": round(float(proximity_distance), 3) if proximity_distance is not None else None,
                        "precision_score": round(float(precision_score), 3),
                        "seed_support": round(float(support_score), 3),
                        "confidence_score": round(float(confidence_score), 3),
                        **_external_row_payload(external_row_map.get(residue_key, object())),
                    }
                )

        if not rows:
            return _empty_auto_pocket_table()

        table = pd.DataFrame(rows).sort_values(
            ["score", "residue_score", "volume", "pocket_id", "chain", "resid"],
            ascending=[False, False, False, True, True, True],
        ).reset_index(drop=True)
        return _ensure_auto_pocket_columns(table)
    except Exception:
        return _empty_auto_pocket_table()
    finally:
        if pdb_path and os.path.exists(pdb_path):
            try:
                os.remove(pdb_path)
            except OSError:
                pass


def _detect_with_p2rank(
    pdb_text: str,
    hotspot_set: set[Tuple[str, int]],
    *,
    residue_support_df: Optional[pd.DataFrame] = None,
    executable: Optional[str] = None,
    profile: str = "default",
    output_dir: Optional[str] = None,
    timeout_sec: float = 180.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pocket_predictions, residue_predictions, run_meta = run_p2rank(
        pdb_text,
        executable=executable,
        profile=profile,
        timeout_sec=timeout_sec,
        output_dir=output_dir,
    )
    if residue_predictions.empty:
        return _empty_auto_pocket_table(), run_meta

    atom_df = parse_pdb_atoms(pdb_text)
    residue_center_map = _build_residue_center_map(atom_df)
    residue_name_map: Dict[Tuple[str, int], str] = {}
    for row in atom_df.itertuples(index=False):
        residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
        if residue_key not in residue_name_map:
            residue_name_map[residue_key] = str(getattr(row, "resname", "")).strip().upper()

    support_row_map: Dict[Tuple[str, int], object] = {}
    if residue_support_df is not None and not residue_support_df.empty:
        for row in residue_support_df.itertuples(index=False):
            residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
            support_row_map[residue_key] = row

    pocket_meta_map: Dict[str, dict[str, float]] = {}
    if not pocket_predictions.empty:
        for row in pocket_predictions.itertuples(index=False):
            pocket_meta_map[str(getattr(row, "pocket_label", "")).strip()] = {
                "pocket_score": float(getattr(row, "pocket_score", 0.0) or 0.0),
                "pocket_probability": float(getattr(row, "pocket_probability", 0.0) or 0.0),
                "center_x": float(getattr(row, "center_x", 0.0) or 0.0),
                "center_y": float(getattr(row, "center_y", 0.0) or 0.0),
                "center_z": float(getattr(row, "center_z", 0.0) or 0.0),
            }

    rows = []
    for (pocket_label, pocket_rank), group in residue_predictions.groupby(["pocket_label", "pocket_rank"], sort=True):
        normalized_rows: list[dict[str, object]] = []
        fallback_center = np.array(
            [
                float(pocket_meta_map.get(str(pocket_label), {}).get("center_x", 0.0)),
                float(pocket_meta_map.get(str(pocket_label), {}).get("center_y", 0.0)),
                float(pocket_meta_map.get(str(pocket_label), {}).get("center_z", 0.0)),
            ],
            dtype=float,
        )
        for row in group.itertuples(index=False):
            residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
            support_row = support_row_map.get(residue_key, object())
            resname = str(getattr(row, "resname", "") or "").strip().upper()
            if not resname:
                resname = str(getattr(support_row, "resname", "") or residue_name_map.get(residue_key, "")).strip().upper()
            center = residue_center_map.get(residue_key)
            if center is None:
                center = fallback_center
            normalized_rows.append(
                {
                    "chain": residue_key[0],
                    "resid": residue_key[1],
                    "resname": resname,
                    "center": np.asarray(center, dtype=float),
                    "residue_probability": float(getattr(row, "residue_probability", 0.0) or 0.0),
                    "residue_score_raw": float(getattr(row, "residue_score", 0.0) or 0.0),
                    "support_row": support_row,
                }
            )

        if not normalized_rows:
            continue

        cluster_coords = np.vstack([entry["center"] for entry in normalized_rows]).astype(float)
        cluster_centroid = cluster_coords.mean(axis=0) if len(cluster_coords) else fallback_center
        cluster_distances = np.sqrt(np.sum((cluster_coords - cluster_centroid) ** 2, axis=1)) if len(cluster_coords) else np.array([], dtype=float)
        distance_map = {
            (str(entry["chain"]), int(entry["resid"])): float(cluster_distances[index])
            for index, entry in enumerate(normalized_rows)
            if index < len(cluster_distances)
        }
        cluster_volume = _estimate_bbox_volume(cluster_coords, padding=2.0)
        depth_avg = float(cluster_distances.mean()) if cluster_distances.size else 0.0
        depth_max = float(cluster_distances.max()) if cluster_distances.size else 0.0

        hotspot_ratio = float(
            np.mean([(entry["chain"], entry["resid"]) in hotspot_set for entry in normalized_rows])
        ) if normalized_rows else 0.0
        seed_mean = float(
            np.mean([float(getattr(entry["support_row"], "seed_support", 0.0) or 0.0) for entry in normalized_rows])
        ) if normalized_rows else 0.0
        confidence_mean = float(
            np.mean([float(getattr(entry["support_row"], "confidence_score", 0.5) or 0.5) for entry in normalized_rows])
        ) if normalized_rows else 0.5
        external_mean = float(
            np.mean([float(getattr(entry["support_row"], "external_support", 0.0) or 0.0) for entry in normalized_rows])
        ) if normalized_rows else 0.0
        residue_mean = float(
            np.mean(
                [
                    0.52 * float(entry["residue_probability"])
                    + 0.20 * float(entry["residue_score_raw"])
                    + 0.16 * float(getattr(entry["support_row"], "seed_support", 0.0) or 0.0)
                    + 0.08 * float(getattr(entry["support_row"], "external_support", 0.0) or 0.0)
                    + 0.04 * float(getattr(entry["support_row"], "confidence_score", 0.5) or 0.5)
                    for entry in normalized_rows
                ]
            )
        ) if normalized_rows else 0.0

        pocket_meta = pocket_meta_map.get(str(pocket_label), {})
        pocket_base = max(float(pocket_meta.get("pocket_probability", 0.0) or 0.0), float(pocket_meta.get("pocket_score", 0.0) or 0.0))
        cluster_score = 0.55 * pocket_base + 0.45 * residue_mean
        cluster_score += 0.08 * hotspot_ratio
        cluster_score += 0.06 * seed_mean
        cluster_score += 0.04 * external_mean
        cluster_score += 0.03 * confidence_mean
        route_suffix = "p2rank"
        cleaned_profile = str(profile or "").strip().lower()
        if cleaned_profile and cleaned_profile not in {"default", "auto"}:
            route_suffix = f"p2rank-{cleaned_profile}"
        detection_route = _detection_route_label(route_suffix)

        for entry in sorted(
            normalized_rows,
            key=lambda item: (
                float(item["residue_probability"]),
                float(item["residue_score_raw"]),
                float(getattr(item["support_row"], "seed_support", 0.0) or 0.0),
                float(getattr(item["support_row"], "external_support", 0.0) or 0.0),
            ),
            reverse=True,
        ):
            support_row = entry["support_row"]
            residue_key = (str(entry["chain"]), int(entry["resid"]))
            residue_score = (
                0.55 * float(entry["residue_probability"])
                + 0.20 * float(entry["residue_score_raw"])
                + 0.15 * float(getattr(support_row, "seed_support", 0.0) or 0.0)
                + 0.06 * float(getattr(support_row, "external_support", 0.0) or 0.0)
                + 0.04 * float(getattr(support_row, "confidence_score", 0.5) or 0.5)
            )
            proximity_distance = distance_map.get(residue_key)
            rows.append(
                {
                    "pocket_id": f"P2Rank-{int(pocket_rank)}",
                    "chain": entry["chain"],
                    "resid": int(entry["resid"]),
                    "resname": entry["resname"],
                    "volume": round(float(cluster_volume), 3),
                    "score": round(float(cluster_score), 3),
                    "residue_score": round(float(residue_score), 3),
                    "contact_count": int(getattr(support_row, "contact_count", 0) or 0),
                    "center_distance": round(float(getattr(support_row, "center_distance", 0.0) or 0.0), 3),
                    "ligand_contact_count": int(getattr(support_row, "ligand_contact_count", 0) or 0),
                    "detection_method": "p2rank",
                    "detection_route": detection_route,
                    "is_hotspot": residue_key in hotspot_set,
                    "depth_avg": round(depth_avg, 3),
                    "depth_max": round(depth_max, 3),
                    "overlap_ratio": round(hotspot_ratio, 3),
                    "proximity_distance": round(float(proximity_distance), 3) if proximity_distance is not None else None,
                    "precision_score": round(float(residue_score), 3),
                    "seed_support": round(float(getattr(support_row, "seed_support", 0.0) or 0.0), 3),
                    "confidence_score": round(float(getattr(support_row, "confidence_score", 0.5) or 0.5), 3),
                    **_external_row_payload(support_row),
                    "consensus_score": round(float(residue_score), 3),
                    "consensus_methods": "p2rank",
                    "method_vote_count": 1,
                    "consensus_overlap_ratio": 1.0,
                }
            )

    if not rows:
        return _empty_auto_pocket_table(), run_meta
    table = pd.DataFrame(rows).sort_values(
        ["score", "residue_score", "volume", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return _ensure_auto_pocket_columns(table), run_meta


def _cluster_residue_indices(candidate_df: pd.DataFrame, cluster_cutoff: float) -> list[list[int]]:
    if candidate_df.empty:
        return []

    coords = candidate_df[["x", "y", "z"]].to_numpy(dtype=float)
    distances = _pairwise_distances(coords)
    adjacency = distances <= float(cluster_cutoff)

    clusters: list[list[int]] = []
    visited: set[int] = set()
    for start_index in range(len(candidate_df)):
        if start_index in visited:
            continue

        stack = [start_index]
        visited.add(start_index)
        component: list[int] = []

        while stack:
            current_index = stack.pop()
            component.append(current_index)
            neighbor_indices = np.where(adjacency[current_index])[0]
            for neighbor_index in neighbor_indices:
                if int(neighbor_index) == current_index or int(neighbor_index) in visited:
                    continue
                visited.add(int(neighbor_index))
                stack.append(int(neighbor_index))

        clusters.append(component)

    return clusters


def _select_ranked_candidate_rows(
    candidate_df: pd.DataFrame,
    *,
    target_candidates: int,
    sort_columns: Sequence[str],
    ascending: Sequence[bool],
    anchor_mask: Optional[pd.Series] = None,
    hard_limit: Optional[int] = None,
) -> pd.DataFrame:
    if candidate_df is None or candidate_df.empty:
        return pd.DataFrame(columns=getattr(candidate_df, "columns", []))

    ordered = candidate_df.sort_values(list(sort_columns), ascending=list(ascending)).copy()
    candidate_limit = max(1, int(target_candidates))
    if hard_limit is not None:
        candidate_limit = min(candidate_limit, max(1, int(hard_limit)))

    if anchor_mask is None:
        return ordered.head(candidate_limit).reset_index(drop=True)

    aligned_anchor_mask = anchor_mask.reindex(ordered.index).fillna(False).astype(bool)
    anchored_rows = ordered[aligned_anchor_mask].copy()
    unanchored_rows = ordered[~aligned_anchor_mask].copy()
    if anchored_rows.empty:
        return ordered.head(candidate_limit).reset_index(drop=True)

    keep_limit = max(candidate_limit, len(anchored_rows))
    if hard_limit is not None:
        keep_limit = min(max(1, int(hard_limit)), keep_limit)

    if len(anchored_rows) >= keep_limit:
        return anchored_rows.head(keep_limit).reset_index(drop=True)

    remaining_slots = keep_limit - len(anchored_rows)
    selected = pd.concat([anchored_rows, unanchored_rows.head(remaining_slots)], axis=0)
    return selected.reset_index(drop=True)


def _rank_cluster_indices_by_score(
    candidate_df: pd.DataFrame,
    cluster_indices: Sequence[Sequence[int]],
    *,
    score_column: str,
) -> list[list[int]]:
    if candidate_df is None or candidate_df.empty:
        return []

    scores = pd.to_numeric(candidate_df.get(score_column), errors="coerce").fillna(0.0).reset_index(drop=True)
    ranked_clusters: list[tuple[float, int, list[int]]] = []
    for indices in cluster_indices:
        cleaned_indices = sorted({int(index) for index in indices if 0 <= int(index) < len(candidate_df)})
        if not cleaned_indices:
            continue
        cluster_score = float(scores.iloc[cleaned_indices].mean())
        ranked_clusters.append((cluster_score, len(cleaned_indices), cleaned_indices))

    ranked_clusters.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [indices for _, _, indices in ranked_clusters]


def _spatially_diversify_cluster_indices(
    candidate_df: pd.DataFrame,
    cluster_indices: Sequence[Sequence[int]],
    *,
    max_pockets: int,
    cluster_cutoff: float,
    score_column: str,
    min_group_size: int = 2,
) -> list[list[int]]:
    ranked_clusters = _rank_cluster_indices_by_score(candidate_df, cluster_indices, score_column=score_column)
    if candidate_df is None or candidate_df.empty or not ranked_clusters:
        return []

    pocket_limit = max(1, int(max_pockets))
    if len(ranked_clusters) >= min(pocket_limit, len(candidate_df)):
        return ranked_clusters[:pocket_limit]
    if len(candidate_df) < max(4, int(min_group_size) * 2):
        return ranked_clusters[:pocket_limit]
    if not {"x", "y", "z"}.issubset(candidate_df.columns):
        return ranked_clusters[:pocket_limit]

    coords = candidate_df[["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        return ranked_clusters[:pocket_limit]

    score_values = pd.to_numeric(candidate_df.get(score_column), errors="coerce").fillna(0.0)
    seed_support = pd.to_numeric(candidate_df.get("seed_support"), errors="coerce").fillna(0.0)
    confidence_score = pd.to_numeric(candidate_df.get("confidence_score"), errors="coerce").fillna(0.0)
    contact_density = (
        _normalize_numeric_series(candidate_df["contact_count"])
        if "contact_count" in candidate_df.columns
        else pd.Series(np.zeros(len(candidate_df)), index=candidate_df.index)
    )
    anchor_flags = (
        _boolean_frame_series(candidate_df, "is_hotspot", False)
        | _boolean_frame_series(candidate_df, "external_exact_match", False)
        | _boolean_frame_series(candidate_df, "external_direct_anchor", False)
        | _boolean_frame_series(candidate_df, "evidence_route_anchor", False)
    )
    priority_score = (
        score_values
        + 0.16 * seed_support
        + 0.08 * confidence_score
        + 0.06 * contact_density
        + 0.22 * anchor_flags.astype(float)
    ).reset_index(drop=True)
    anchor_values = anchor_flags.reset_index(drop=True).to_numpy(dtype=bool)

    min_seed_distance = _clamp_float(float(cluster_cutoff) * 0.78, 5.0, 10.0)
    assignment_radius = _clamp_float(float(cluster_cutoff) * 0.82, 4.8, 10.5)
    seed_positions: list[int] = []
    ordered_positions = sorted(
        range(len(candidate_df)),
        key=lambda position: (float(priority_score.iloc[position]), bool(anchor_values[position])),
        reverse=True,
    )

    for position in ordered_positions:
        if len(seed_positions) >= pocket_limit:
            break
        if not seed_positions:
            seed_positions.append(int(position))
            continue
        distances_to_existing = np.sqrt(np.sum((coords[seed_positions] - coords[position]) ** 2, axis=1))
        if float(distances_to_existing.min()) >= min_seed_distance:
            seed_positions.append(int(position))

    if len(seed_positions) < min(2, pocket_limit):
        relaxed_distance = max(3.8, min_seed_distance * 0.65)
        for position in ordered_positions:
            if len(seed_positions) >= pocket_limit:
                break
            if int(position) in seed_positions:
                continue
            distances_to_existing = np.sqrt(np.sum((coords[seed_positions] - coords[position]) ** 2, axis=1))
            if float(distances_to_existing.min()) >= relaxed_distance:
                seed_positions.append(int(position))

    if len(seed_positions) <= len(ranked_clusters):
        return ranked_clusters[:pocket_limit]

    seed_coords = coords[seed_positions]
    distance_to_seeds = np.sqrt(np.sum((coords[:, None, :] - seed_coords[None, :, :]) ** 2, axis=2))
    nearest_seed_positions = distance_to_seeds.argmin(axis=1)
    nearest_seed_distances = distance_to_seeds.min(axis=1)
    grouped_positions: dict[int, list[int]] = {int(seed): [int(seed)] for seed in seed_positions}

    for position in range(len(candidate_df)):
        if int(position) in seed_positions:
            continue
        nearest_seed = int(seed_positions[int(nearest_seed_positions[position])])
        current_group = grouped_positions.setdefault(nearest_seed, [nearest_seed])
        if float(nearest_seed_distances[position]) <= assignment_radius or len(current_group) < int(min_group_size):
            current_group.append(int(position))

    diversified_groups: list[list[int]] = []
    for seed_position, group_positions in grouped_positions.items():
        cleaned_group = sorted({int(position) for position in group_positions if 0 <= int(position) < len(candidate_df)})
        if not cleaned_group:
            continue
        has_anchor = bool(anchor_values[cleaned_group].any())
        if len(cleaned_group) >= int(min_group_size) or has_anchor:
            diversified_groups.append(cleaned_group)

    if len(diversified_groups) <= len(ranked_clusters):
        return ranked_clusters[:pocket_limit]
    return _rank_cluster_indices_by_score(candidate_df, diversified_groups, score_column=score_column)[:pocket_limit]


def _build_precision_method_table(
    residue_df: pd.DataFrame,
    hotspot_set: set[Tuple[str, int]],
    *,
    method_kind: str,
    detection_method: str,
    route_suffix: str,
    cluster_cutoff: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
) -> pd.DataFrame:
    if residue_df is None or residue_df.empty:
        return _empty_auto_pocket_table()

    working = residue_df.copy()
    if method_kind == "ligand":
        ligand_total = int(pd.to_numeric(working.get("ligand_contact_count"), errors="coerce").fillna(0).sum()) if "ligand_contact_count" in working.columns else 0
        if ligand_total <= 0:
            return _empty_auto_pocket_table()

    contact_density = _normalize_numeric_series(working["contact_count"]) if "contact_count" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    center_closeness = 1.0 - _normalize_numeric_series(working["center_distance"]) if "center_distance" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    ligand_density = _normalize_numeric_series(working["ligand_contact_count"]) if "ligand_contact_count" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    precision_score = pd.to_numeric(working["precision_score"], errors="coerce").fillna(0.0) if "precision_score" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    seed_support = pd.to_numeric(working["seed_support"], errors="coerce").fillna(0.0) if "seed_support" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    confidence_score = pd.to_numeric(working["confidence_score"], errors="coerce").fillna(0.5) if "confidence_score" in working.columns else pd.Series(np.full(len(working), 0.5), index=working.index)
    external_support = _numeric_frame_series(working, "external_support", 0.0)
    external_confidence = _numeric_frame_series(working, "external_confidence", 0.0)
    external_exact = _boolean_frame_series(working, "external_exact_match", False).astype(float)
    hotspot_flag = working["is_hotspot"].astype(float) if "is_hotspot" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)

    if method_kind == "ligand":
        method_score = (
            0.42 * ligand_density
            + 0.20 * center_closeness
            + 0.18 * seed_support
            + 0.10 * confidence_score
            + 0.08 * precision_score
            + 0.06 * external_support
            + 0.03 * external_confidence
            + 0.02 * contact_density
            + 0.04 * external_exact
            + 0.05 * hotspot_flag
        )
        score_threshold = float(method_score.quantile(0.55)) if len(method_score) > 1 else float(method_score.iloc[0])
        support_gate = (
            (working["ligand_contact_count"].fillna(0) > 0)
            | (method_score >= score_threshold)
            | (seed_support >= 0.15)
            | (external_support >= 0.24)
            | _boolean_frame_series(working, "external_exact_match", False)
            | (confidence_score >= 0.45)
            | working["is_hotspot"]
        )
    else:
        method_score = (
            0.46 * precision_score
            + 0.18 * contact_density
            + 0.16 * center_closeness
            + 0.12 * seed_support
            + 0.07 * external_support
            + 0.03 * external_confidence
            + 0.08 * confidence_score
            + 0.04 * external_exact
            + 0.05 * hotspot_flag
        )
        score_threshold = float(method_score.quantile(0.58)) if len(method_score) > 1 else float(method_score.iloc[0])
        support_gate = (
            (method_score >= score_threshold)
            | (seed_support >= 0.20)
            | (external_support >= 0.24)
            | _boolean_frame_series(working, "external_exact_match", False)
            | (confidence_score >= 0.45)
            | working["is_hotspot"]
        )

    candidate_df = working[support_gate].copy()
    if candidate_df.empty:
        candidate_df = working.copy()

    candidate_df = candidate_df.assign(method_score=method_score.loc[candidate_df.index].to_numpy(dtype=float))
    target_candidates = max(min_candidates, int(math.ceil(len(candidate_df) * float(top_fraction))))
    target_candidates = min(max_candidates, target_candidates, len(candidate_df))
    target_candidates = max(1, target_candidates)
    seed_support_values = pd.to_numeric(candidate_df.get("seed_support"), errors="coerce").fillna(0.0)
    external_support_values = _numeric_frame_series(candidate_df, "external_support", 0.0)
    hotspot_flags = candidate_df.get("is_hotspot", pd.Series(False, index=candidate_df.index)).fillna(False).astype(bool)
    external_exact_flags = _boolean_frame_series(candidate_df, "external_exact_match", False)
    anchor_mask = hotspot_flags | (seed_support_values >= 0.60) | (external_support_values >= 0.70) | external_exact_flags
    candidate_df = _select_ranked_candidate_rows(
        candidate_df,
        target_candidates=target_candidates,
        sort_columns=["method_score", "seed_support", "confidence_score", "contact_count", "center_distance"],
        ascending=[False, False, False, False, True],
        anchor_mask=anchor_mask if anchor_mask.any() else None,
        hard_limit=max_candidates,
    )

    if candidate_df.empty:
        return _empty_auto_pocket_table()

    cluster_indices = _cluster_residue_indices(candidate_df, cluster_cutoff)
    cluster_indices = _spatially_diversify_cluster_indices(
        candidate_df,
        cluster_indices,
        max_pockets=max_pockets,
        cluster_cutoff=cluster_cutoff,
        score_column="method_score",
    )
    if not cluster_indices:
        return _empty_auto_pocket_table()

    rows = []
    for pocket_rank, indices in enumerate(cluster_indices[:max_pockets], start=1):
        cluster_df = candidate_df.iloc[indices].copy().sort_values(
            ["method_score", "seed_support", "confidence_score", "resid"],
            ascending=[False, False, False, True],
        )
        cluster_coords = cluster_df[["x", "y", "z"]].to_numpy(dtype=float)
        cluster_centroid = cluster_coords.mean(axis=0) if len(cluster_coords) else np.zeros(3, dtype=float)
        cluster_distances = np.sqrt(np.sum((cluster_coords - cluster_centroid) ** 2, axis=1)) if len(cluster_coords) else np.array([], dtype=float)
        cluster_hotspot_overlap = int(cluster_df["is_hotspot"].sum()) if "is_hotspot" in cluster_df.columns else 0
        cluster_hotspot_ratio = float(cluster_hotspot_overlap) / float(len(cluster_df)) if len(cluster_df) else 0.0
        cluster_seed_mean = float(cluster_df["seed_support"].mean()) if "seed_support" in cluster_df.columns else 0.0
        cluster_confidence_mean = float(cluster_df["confidence_score"].mean()) if "confidence_score" in cluster_df.columns else 0.5
        cluster_score = float(cluster_df["method_score"].mean())
        cluster_score += 0.08 * cluster_hotspot_ratio
        cluster_score += 0.05 * cluster_seed_mean
        cluster_score += 0.04 * cluster_confidence_mean

        pocket_id = f"AutoPocket-{pocket_rank}"
        consensus_method_text = _join_methods([detection_method])
        detection_route = _detection_route_label(route_suffix if route_suffix else consensus_method_text)
        depth_avg = float(cluster_distances.mean()) if cluster_distances.size else 0.0
        depth_max = float(cluster_distances.max()) if cluster_distances.size else 0.0
        cluster_volume = _estimate_bbox_volume(cluster_coords, padding=2.0)

        for row_index, row in enumerate(cluster_df.itertuples(index=False)):
            proximity_distance = float(cluster_distances[row_index]) if row_index < len(cluster_distances) else None
            row_score = float(getattr(row, "method_score", 0.0))
            rows.append(
                {
                    "pocket_id": pocket_id,
                    "chain": row.chain,
                    "resid": int(row.resid),
                    "resname": row.resname,
                    "volume": round(float(cluster_volume), 3),
                    "score": round(float(cluster_score), 3),
                    "residue_score": round(row_score, 3),
                    "contact_count": int(getattr(row, "contact_count", 0) or 0),
                    "center_distance": round(float(getattr(row, "center_distance", 0.0)), 3),
                    "ligand_contact_count": int(getattr(row, "ligand_contact_count", 0) or 0),
                    "detection_method": detection_method,
                    "detection_route": detection_route,
                    "is_hotspot": bool(getattr(row, "is_hotspot", False)),
                    "depth_avg": round(depth_avg, 3),
                    "depth_max": round(depth_max, 3),
                    "overlap_ratio": round(cluster_hotspot_ratio, 3),
                    "proximity_distance": round(float(proximity_distance), 3) if proximity_distance is not None else None,
                    "precision_score": round(row_score, 3),
                    "seed_support": round(float(getattr(row, "seed_support", 0.0) or 0.0), 3),
                    "confidence_score": round(float(getattr(row, "confidence_score", 0.5) or 0.5), 3),
                    **_external_row_payload(row),
                    "consensus_score": round(row_score, 3),
                    "consensus_methods": consensus_method_text,
                    "method_vote_count": 1,
                    "consensus_overlap_ratio": 1.0,
                }
            )

    if not rows:
        return _empty_auto_pocket_table()

    table = pd.DataFrame(rows).sort_values(
        ["score", "residue_score", "volume", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return _ensure_auto_pocket_columns(table)


def _merge_multiscale_method_tables(
    profile_tables: Sequence[pd.DataFrame],
    residue_df: pd.DataFrame,
    *,
    hotspot_set: set[Tuple[str, int]],
    cluster_cutoff: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
    detection_method: str,
    detection_route: str,
    pocket_prefix: str,
    consensus_methods: str,
) -> pd.DataFrame:
    valid_tables = [table for table in profile_tables if table is not None and not table.empty]
    if not valid_tables:
        return _empty_auto_pocket_table()
    if len(valid_tables) == 1:
        single = valid_tables[0].copy()
        if "detection_method" in single.columns:
            single["detection_method"] = detection_method
        if "detection_route" in single.columns:
            single["detection_route"] = detection_route
        if "consensus_methods" in single.columns:
            single["consensus_methods"] = consensus_methods
        return _ensure_auto_pocket_columns(single)

    residue_support: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for table in valid_tables:
        best_rows: Dict[Tuple[str, int], object] = {}
        if table is None or table.empty:
            continue

        for row in table.itertuples(index=False):
            residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
            row_score = float(
                getattr(
                    row,
                    "consensus_score",
                    getattr(row, "residue_score", getattr(row, "precision_score", getattr(row, "score", 0.0))),
                )
                or 0.0
            )
            existing = best_rows.get(residue_key)
            if existing is None:
                best_rows[residue_key] = row
                continue
            existing_score = float(
                getattr(
                    existing,
                    "consensus_score",
                    getattr(existing, "residue_score", getattr(existing, "precision_score", getattr(existing, "score", 0.0))),
                )
                or 0.0
            )
            if row_score > existing_score:
                best_rows[residue_key] = row

        for residue_key, row in best_rows.items():
            bucket = residue_support.setdefault(
                (residue_key[0], residue_key[1], str(getattr(row, "resname", "")).strip().upper()),
                {
                    "scores": [],
                    "pocket_scores": [],
                    "seed_support": [],
                    "confidence_score": [],
                    "is_hotspot": False,
                },
            )
            score_value = float(
                getattr(
                    row,
                    "consensus_score",
                    getattr(row, "residue_score", getattr(row, "precision_score", getattr(row, "score", 0.0))),
                )
                or 0.0
            )
            bucket["scores"].append(score_value)
            bucket["pocket_scores"].append(float(getattr(row, "score", score_value) or score_value))
            bucket["seed_support"].append(float(getattr(row, "seed_support", 0.0) or 0.0))
            bucket["confidence_score"].append(float(getattr(row, "confidence_score", 0.5) or 0.5))
            bucket["is_hotspot"] = bool(bucket["is_hotspot"] or getattr(row, "is_hotspot", False))

    if not residue_support:
        return _empty_auto_pocket_table()

    profile_count = max(1, len(valid_tables))
    support_rows = []
    for (chain, resid, resname), bucket in residue_support.items():
        score_mean = float(np.mean(bucket["scores"])) if bucket["scores"] else 0.0
        score_max = float(np.max(bucket["scores"])) if bucket["scores"] else 0.0
        pocket_score_mean = float(np.mean(bucket["pocket_scores"])) if bucket["pocket_scores"] else 0.0
        seed_mean = float(np.mean(bucket["seed_support"])) if bucket["seed_support"] else 0.0
        confidence_mean = float(np.mean(bucket["confidence_score"])) if bucket["confidence_score"] else 0.5
        profile_overlap_ratio = float(len(bucket["scores"])) / float(profile_count)
        multiscale_score = (
            0.44 * score_mean
            + 0.24 * score_max
            + 0.18 * profile_overlap_ratio
            + 0.09 * seed_mean
            + 0.05 * confidence_mean
            + 0.04 * pocket_score_mean
        )
        if bucket["is_hotspot"] or (chain, resid) in hotspot_set:
            multiscale_score += 0.05

        support_rows.append(
            {
                "chain": chain,
                "resid": resid,
                "resname": resname,
                "profile_vote_count": int(len(bucket["scores"])),
                "profile_overlap_ratio": round(profile_overlap_ratio, 3),
                "seed_support": round(seed_mean, 3),
                "confidence_score": round(confidence_mean, 3),
                "precision_score": round(float(multiscale_score), 3),
                "consensus_score": round(float(multiscale_score), 3),
                "is_hotspot": bool(bucket["is_hotspot"] or (chain, resid) in hotspot_set),
            }
        )

    support_df = pd.DataFrame(support_rows)
    residue_base_df = residue_df.drop(
        columns=[column for column in ["seed_support", "confidence_score", "precision_score", "is_hotspot"] if column in residue_df.columns],
        errors="ignore",
    )
    merged = residue_base_df.merge(support_df, on=["chain", "resid", "resname"], how="inner")
    if merged.empty:
        return _empty_auto_pocket_table()

    score_threshold = float(merged["consensus_score"].quantile(0.55)) if len(merged) > 1 else float(merged["consensus_score"].iloc[0])
    external_support = _numeric_frame_series(merged, "external_support", 0.0)
    external_exact = _boolean_frame_series(merged, "external_exact_match", False)
    support_gate = (
        (merged["consensus_score"] >= score_threshold)
        | (merged["profile_overlap_ratio"] >= 0.67)
        | (merged["seed_support"] >= 0.60)
        | (external_support >= 0.24)
        | external_exact
        | merged["is_hotspot"]
    )
    candidate_df = merged[support_gate].copy()
    if candidate_df.empty:
        candidate_df = merged.copy()

    target_candidates = max(min_candidates, int(math.ceil(len(candidate_df) * float(top_fraction))))
    target_candidates = min(max_candidates, target_candidates, len(candidate_df))
    target_candidates = max(1, target_candidates)
    anchor_mask = (
        candidate_df["is_hotspot"].fillna(False).astype(bool)
        | (pd.to_numeric(candidate_df["seed_support"], errors="coerce").fillna(0.0) >= 0.60)
        | (_numeric_frame_series(candidate_df, "external_support", 0.0) >= 0.70)
        | _boolean_frame_series(candidate_df, "external_exact_match", False)
        | (pd.to_numeric(candidate_df["profile_vote_count"], errors="coerce").fillna(0).astype(int) >= 2)
    )
    candidate_df = _select_ranked_candidate_rows(
        candidate_df,
        target_candidates=target_candidates,
        sort_columns=["consensus_score", "profile_overlap_ratio", "seed_support", "confidence_score", "contact_count", "center_distance"],
        ascending=[False, False, False, False, False, True],
        anchor_mask=anchor_mask if anchor_mask.any() else None,
        hard_limit=max_candidates,
    )
    if candidate_df.empty:
        return _empty_auto_pocket_table()

    cluster_indices = _cluster_residue_indices(candidate_df, cluster_cutoff)
    cluster_indices = _spatially_diversify_cluster_indices(
        candidate_df,
        cluster_indices,
        max_pockets=max_pockets,
        cluster_cutoff=cluster_cutoff,
        score_column="consensus_score",
    )
    if not cluster_indices:
        return _empty_auto_pocket_table()

    pocket_rows = []
    for pocket_rank, indices in enumerate(cluster_indices[:max_pockets], start=1):
        cluster_df = candidate_df.iloc[indices].copy().sort_values(
            ["consensus_score", "profile_overlap_ratio", "seed_support", "confidence_score", "resid"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        cluster_coords = cluster_df[["x", "y", "z"]].to_numpy(dtype=float)
        cluster_centroid = cluster_coords.mean(axis=0) if len(cluster_coords) else np.zeros(3, dtype=float)
        cluster_distances = np.sqrt(np.sum((cluster_coords - cluster_centroid) ** 2, axis=1)) if len(cluster_coords) else np.array([], dtype=float)
        cluster_hotspot_overlap = int(cluster_df["is_hotspot"].sum()) if "is_hotspot" in cluster_df.columns else 0
        cluster_hotspot_ratio = float(cluster_hotspot_overlap) / float(len(cluster_df)) if len(cluster_df) else 0.0
        cluster_profile_overlap = float(cluster_df["profile_overlap_ratio"].mean()) if "profile_overlap_ratio" in cluster_df.columns else 0.0
        cluster_seed_mean = float(cluster_df["seed_support"].mean()) if "seed_support" in cluster_df.columns else 0.0
        cluster_confidence_mean = float(cluster_df["confidence_score"].mean()) if "confidence_score" in cluster_df.columns else 0.5
        cluster_score = float(cluster_df["consensus_score"].mean())
        cluster_score += 0.12 * cluster_profile_overlap
        cluster_score += 0.08 * cluster_hotspot_ratio
        cluster_score += 0.05 * cluster_seed_mean
        cluster_score += 0.04 * cluster_confidence_mean
        cluster_volume = _estimate_bbox_volume(cluster_coords, padding=2.0)
        depth_avg = float(cluster_distances.mean()) if cluster_distances.size else 0.0
        depth_max = float(cluster_distances.max()) if cluster_distances.size else 0.0
        pocket_id = f"{pocket_prefix}-{pocket_rank}"

        for row_index, row in enumerate(cluster_df.itertuples(index=False)):
            proximity_distance = float(cluster_distances[row_index]) if row_index < len(cluster_distances) else None
            residue_score = float(getattr(row, "consensus_score", 0.0) or 0.0)
            pocket_rows.append(
                {
                    "pocket_id": pocket_id,
                    "chain": row.chain,
                    "resid": int(row.resid),
                    "resname": row.resname,
                    "volume": round(float(cluster_volume), 3),
                    "score": round(float(cluster_score), 3),
                    "residue_score": round(float(residue_score), 3),
                    "contact_count": int(getattr(row, "contact_count", 0) or 0),
                    "center_distance": round(float(getattr(row, "center_distance", 0.0)), 3),
                    "ligand_contact_count": int(getattr(row, "ligand_contact_count", 0) or 0),
                    "detection_method": detection_method,
                    "detection_route": detection_route,
                    "is_hotspot": bool(getattr(row, "is_hotspot", False)),
                    "depth_avg": round(depth_avg, 3),
                    "depth_max": round(depth_max, 3),
                    "overlap_ratio": round(cluster_hotspot_ratio, 3),
                    "proximity_distance": round(float(proximity_distance), 3) if proximity_distance is not None else None,
                    "precision_score": round(float(residue_score), 3),
                    "seed_support": round(float(getattr(row, "seed_support", 0.0) or 0.0), 3),
                    "confidence_score": round(float(getattr(row, "confidence_score", 0.5) or 0.5), 3),
                    **_external_row_payload(row),
                    "consensus_score": round(float(residue_score), 3),
                    "consensus_methods": consensus_methods,
                    "method_vote_count": 1,
                    "consensus_overlap_ratio": 1.0,
                }
            )

    if not pocket_rows:
        return _empty_auto_pocket_table()

    table = pd.DataFrame(pocket_rows).sort_values(
        ["score", "residue_score", "volume", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return _ensure_auto_pocket_columns(table)


def _merge_multiscale_kvfinder_tables(
    profile_tables: Sequence[pd.DataFrame],
    residue_df: pd.DataFrame,
    *,
    hotspot_set: set[Tuple[str, int]],
    cluster_cutoff: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
) -> pd.DataFrame:
    return _merge_multiscale_method_tables(
        profile_tables,
        residue_df,
        hotspot_set=hotspot_set,
        cluster_cutoff=cluster_cutoff,
        top_fraction=top_fraction,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        max_pockets=max_pockets,
        detection_method="kvfinder-multiscale",
        detection_route=_detection_route_label("kvfinder-multiscale"),
        pocket_prefix="KVFinder-MS",
        consensus_methods="kvfinder",
    )


def _build_multiscale_precision_method_table(
    residue_df: pd.DataFrame,
    hotspot_set: set[Tuple[str, int]],
    *,
    method_kind: str,
    detection_method: str,
    route_suffix: str,
    cluster_cutoff: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
) -> pd.DataFrame:
    profiles = [
        (
            _clamp_float(float(cluster_cutoff) - 0.55, 5.8, 10.8),
            _clamp_float(float(top_fraction) - 0.04, 0.14, 0.36),
        ),
        (
            _clamp_float(float(cluster_cutoff), 5.8, 10.8),
            _clamp_float(float(top_fraction), 0.14, 0.36),
        ),
        (
            _clamp_float(float(cluster_cutoff) + 0.55, 5.8, 10.8),
            _clamp_float(float(top_fraction) + 0.04, 0.14, 0.36),
        ),
    ]

    profile_tables: list[pd.DataFrame] = []
    for profile_cluster_cutoff, profile_top_fraction in profiles:
        table = _build_precision_method_table(
            residue_df,
            hotspot_set,
            method_kind=method_kind,
            detection_method=detection_method,
            route_suffix=route_suffix,
            cluster_cutoff=profile_cluster_cutoff,
            top_fraction=profile_top_fraction,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            max_pockets=max_pockets,
        )
        if not table.empty:
            profile_tables.append(table)

    if not profile_tables:
        return _empty_auto_pocket_table()

    method_token = _display_method_token(detection_method)
    pocket_prefix = "LigandPocket" if method_kind == "ligand" else "AutoPocket"
    return _merge_multiscale_method_tables(
        profile_tables,
        residue_df,
        hotspot_set=hotspot_set,
        cluster_cutoff=cluster_cutoff,
        top_fraction=top_fraction,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        max_pockets=max_pockets,
        detection_method=detection_method,
        detection_route=_detection_route_label(f"{route_suffix}-multiscale"),
        pocket_prefix=pocket_prefix,
        consensus_methods=method_token,
    )


def _build_external_evidence_method_table(
    residue_df: pd.DataFrame,
    hotspot_set: set[Tuple[str, int]],
    *,
    cluster_cutoff: float,
    max_candidates: int,
    max_pockets: int,
    min_anchor_support: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_support"],
    min_anchor_confidence: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_confidence"],
    min_mapping_quality: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_mapping_quality"],
    evidence_radius: Optional[float] = None,
) -> pd.DataFrame:
    if residue_df is None or residue_df.empty:
        return _empty_auto_pocket_table()

    working = residue_df.copy()
    min_anchor_support = _clamp_float(float(min_anchor_support), 0.0, 1.2)
    min_anchor_confidence = _clamp_float(float(min_anchor_confidence), 0.0, 1.0)
    min_mapping_quality = _clamp_float(float(min_mapping_quality), 0.0, 1.0)
    external_support = _numeric_frame_series(working, "external_support", 0.0)
    external_confidence = _numeric_frame_series(working, "external_confidence", 0.0)
    external_quality = _numeric_frame_series(working, "external_mapping_quality", 0.0)
    external_count = pd.to_numeric(working.get("external_evidence_count"), errors="coerce").fillna(0.0) if "external_evidence_count" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    exact_flags = _boolean_frame_series(working, "external_exact_match", False)
    verified_flags = _boolean_frame_series(working, "external_structure_verified", False)
    direct_anchor_flags = _boolean_frame_series(working, "external_direct_anchor", False)

    anchor_mask = (
        direct_anchor_flags
        | ((external_count > 0) & exact_flags)
        | ((external_count > 0) & (external_support >= min_anchor_support) & (external_confidence >= min_anchor_confidence))
        | ((external_count > 0) & (external_quality >= min_mapping_quality))
    )
    anchor_df = working[anchor_mask].copy()
    if anchor_df.empty or not {"x", "y", "z"}.issubset(anchor_df.columns):
        return _empty_auto_pocket_table()

    coords = working[["x", "y", "z"]].to_numpy(dtype=float)
    anchor_coords = anchor_df[["x", "y", "z"]].to_numpy(dtype=float)
    anchor_distance_matrix = np.sqrt(np.sum((coords[:, None, :] - anchor_coords[None, :, :]) ** 2, axis=2))
    anchor_distances = anchor_distance_matrix.min(axis=1)
    nearest_anchor_positions = anchor_distance_matrix.argmin(axis=1)
    anchor_labels = [
        f"{_normalize_chain(getattr(row, 'chain', 'A'))}:{int(getattr(row, 'resid', 0))}"
        for row in anchor_df.itertuples(index=False)
    ]
    nearest_anchor_labels = [
        anchor_labels[int(position)] if anchor_labels else ""
        for position in nearest_anchor_positions
    ]
    resolved_evidence_radius = (
        _clamp_float(float(evidence_radius), 3.5, 12.0)
        if evidence_radius is not None
        else _clamp_float(float(cluster_cutoff) * 0.92, 5.4, 8.4)
    )
    proximity_score = np.clip(1.0 - (anchor_distances / max(resolved_evidence_radius, 1e-6)), 0.0, 1.0)
    anchor_distance_series = pd.Series(anchor_distances, index=working.index)
    proximity_series = pd.Series(proximity_score, index=working.index)
    nearest_anchor_label_series = pd.Series(nearest_anchor_labels, index=working.index)

    precision_score = _numeric_frame_series(working, "precision_score", 0.0)
    confidence_score = _numeric_frame_series(working, "confidence_score", 0.5)
    contact_density = _normalize_numeric_series(working["contact_count"]) if "contact_count" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    ligand_density = _normalize_numeric_series(working["ligand_contact_count"]) if "ligand_contact_count" in working.columns else pd.Series(np.zeros(len(working)), index=working.index)
    hotspot_flags = working.get("is_hotspot", pd.Series(False, index=working.index)).fillna(False).astype(bool)

    method_score = (
        0.31 * external_support
        + 0.18 * external_confidence
        + 0.15 * external_quality
        + 0.18 * proximity_series
        + 0.08 * precision_score
        + 0.05 * confidence_score
        + 0.04 * contact_density
        + 0.03 * ligand_density
        + 0.05 * exact_flags.astype(float)
        + 0.03 * verified_flags.astype(float)
        + 0.04 * hotspot_flags.astype(float)
    ).clip(lower=0.0, upper=1.2)

    candidate_mask = (
        (anchor_distances <= resolved_evidence_radius)
        | anchor_mask
        | ((external_count > 0) & (external_support >= max(min_anchor_support, EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_consensus_support"])))
    )
    candidate_df = working[candidate_mask].copy()
    if candidate_df.empty:
        return _empty_auto_pocket_table()
    candidate_df = candidate_df.assign(
        method_score=method_score.loc[candidate_df.index].to_numpy(dtype=float),
        evidence_anchor_distance=anchor_distance_series.loc[candidate_df.index].to_numpy(dtype=float),
        evidence_anchor_proximity=proximity_series.loc[candidate_df.index].to_numpy(dtype=float),
        evidence_route_anchor=anchor_mask.loc[candidate_df.index].to_numpy(dtype=bool),
        evidence_anchor_residue=nearest_anchor_label_series.loc[candidate_df.index].astype(str).tolist(),
    )

    candidate_anchor_mask = anchor_mask.reindex(candidate_df.index).fillna(False).astype(bool)
    evidence_seed_floor = min(
        len(candidate_df),
        max(len(anchor_df), len(anchor_df) * 4, len(anchor_df) + 3),
    )
    keep_limit = min(
        len(candidate_df),
        max(len(anchor_df), int(max_candidates), evidence_seed_floor),
    )
    candidate_df = _select_ranked_candidate_rows(
        candidate_df,
        target_candidates=keep_limit,
        sort_columns=["method_score", "external_support", "external_confidence", "precision_score", "contact_count", "center_distance"],
        ascending=[False, False, False, False, False, True],
        anchor_mask=candidate_anchor_mask if candidate_anchor_mask.any() else None,
        hard_limit=max(int(max_candidates), evidence_seed_floor),
    )
    if candidate_df.empty:
        return _empty_auto_pocket_table()

    cluster_indices = _cluster_residue_indices(candidate_df, cluster_cutoff=max(resolved_evidence_radius, float(cluster_cutoff) * 0.75))
    if not cluster_indices:
        return _empty_auto_pocket_table()

    cluster_infos = []
    for indices in cluster_indices:
        cluster_df = candidate_df.iloc[indices].copy()
        cluster_anchor_count = int(anchor_mask.reindex(cluster_df.index).fillna(False).astype(bool).sum())
        if cluster_anchor_count <= 0:
            continue
        cluster_score = float(cluster_df["method_score"].mean())
        cluster_score += 0.12 * float(_numeric_frame_series(cluster_df, "external_support", 0.0).max())
        cluster_score += 0.08 * min(1.0, float(cluster_anchor_count) / 3.0)
        cluster_score += 0.05 * float(cluster_df.get("is_hotspot", pd.Series(False, index=cluster_df.index)).fillna(False).astype(bool).mean())
        cluster_infos.append((cluster_score, cluster_anchor_count, cluster_df))

    if not cluster_infos:
        return _empty_auto_pocket_table()

    cluster_infos.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    pocket_rows = []
    for pocket_rank, (cluster_score, cluster_anchor_count, cluster_df) in enumerate(cluster_infos[:max_pockets], start=1):
        cluster_df = cluster_df.sort_values(
            ["method_score", "external_support", "external_confidence", "resid"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        cluster_coords = cluster_df[["x", "y", "z"]].to_numpy(dtype=float)
        cluster_centroid = cluster_coords.mean(axis=0) if len(cluster_coords) else np.zeros(3, dtype=float)
        cluster_distances = np.sqrt(np.sum((cluster_coords - cluster_centroid) ** 2, axis=1)) if len(cluster_coords) else np.array([], dtype=float)
        cluster_hotspot_ratio = float(cluster_df.get("is_hotspot", pd.Series(False, index=cluster_df.index)).fillna(False).astype(bool).mean()) if len(cluster_df) else 0.0
        cluster_volume = _estimate_bbox_volume(cluster_coords, padding=2.0)
        depth_avg = float(cluster_distances.mean()) if cluster_distances.size else 0.0
        depth_max = float(cluster_distances.max()) if cluster_distances.size else 0.0
        pocket_id = f"EvidencePocket-{pocket_rank}"

        for row_index, row in enumerate(cluster_df.itertuples(index=False)):
            residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
            proximity_distance = float(cluster_distances[row_index]) if row_index < len(cluster_distances) else None
            residue_score = float(getattr(row, "method_score", 0.0) or 0.0)
            if bool(getattr(row, "external_exact_match", False)):
                residue_score += 0.05
            pocket_rows.append(
                {
                    "pocket_id": pocket_id,
                    "chain": row.chain,
                    "resid": int(row.resid),
                    "resname": row.resname,
                    "volume": round(float(cluster_volume), 3),
                    "score": round(float(cluster_score), 3),
                    "residue_score": round(float(residue_score), 3),
                    "contact_count": int(getattr(row, "contact_count", 0) or 0),
                    "center_distance": round(float(getattr(row, "center_distance", 0.0) or 0.0), 3),
                    "ligand_contact_count": int(getattr(row, "ligand_contact_count", 0) or 0),
                    "detection_method": "external-evidence-seeded",
                    "detection_route": _detection_route_label("external-evidence-seeded"),
                    "is_hotspot": residue_key in hotspot_set or bool(getattr(row, "is_hotspot", False)),
                    "depth_avg": round(depth_avg, 3),
                    "depth_max": round(depth_max, 3),
                    "overlap_ratio": round(cluster_hotspot_ratio, 3),
                    "proximity_distance": round(float(proximity_distance), 3) if proximity_distance is not None else None,
                    "precision_score": round(float(residue_score), 3),
                    "seed_support": round(float(getattr(row, "seed_support", 0.0) or 0.0), 3),
                    "confidence_score": round(float(getattr(row, "confidence_score", 0.5) or 0.5), 3),
                    **_external_row_payload(row),
                    "consensus_score": round(float(residue_score), 3),
                    "consensus_methods": "external-evidence",
                    "method_vote_count": 1,
                    "consensus_overlap_ratio": 1.0,
                }
            )

    if not pocket_rows:
        return _empty_auto_pocket_table()

    table = pd.DataFrame(pocket_rows).sort_values(
        ["score", "residue_score", "volume", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return _ensure_auto_pocket_columns(table)


def _build_consensus_pocket_table(
    method_tables: Sequence[pd.DataFrame],
    residue_df: pd.DataFrame,
    *,
    hotspot_set: set[Tuple[str, int]],
    cluster_cutoff: float,
    top_fraction: float,
    min_candidates: int,
    max_candidates: int,
    max_pockets: int,
    allow_external_anchor_gate: bool = True,
    external_anchor_min_support: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_consensus_support"],
) -> pd.DataFrame:
    if residue_df is None or residue_df.empty:
        return _empty_auto_pocket_table()

    support_entries = []
    available_methods: list[str] = []

    for table in method_tables:
        if table is None or table.empty:
            continue

        canonical_method = _canonical_method_name(table["detection_method"].iloc[0]) if "detection_method" in table.columns and not table.empty else "unknown"
        if canonical_method not in available_methods:
            available_methods.append(canonical_method)

        for row in table.itertuples(index=False):
            residue_key = (_normalize_chain(getattr(row, "chain", "A")), int(getattr(row, "resid", 0)))
            method_score = float(
                getattr(
                    row,
                    "consensus_score",
                    getattr(row, "residue_score", getattr(row, "precision_score", getattr(row, "score", 0.0))),
                )
                or 0.0
            )
            pocket_score = float(getattr(row, "score", method_score) or 0.0)
            evidence_anchor_distance = _optional_finite_float(getattr(row, "evidence_anchor_distance", None))
            raw_anchor_proximity = getattr(row, "evidence_anchor_proximity", 0.0)
            try:
                evidence_anchor_proximity = float(raw_anchor_proximity)
                if not np.isfinite(evidence_anchor_proximity):
                    evidence_anchor_proximity = 0.0
            except Exception:
                evidence_anchor_proximity = 0.0
            support_entries.append(
                {
                    "method_name": canonical_method,
                    "pocket_id": str(getattr(row, "pocket_id", "")).strip(),
                    "chain": residue_key[0],
                    "resid": residue_key[1],
                    "resname": str(getattr(row, "resname", "")).strip().upper(),
                    "method_score": method_score,
                    "pocket_score": pocket_score,
                    "seed_support": float(getattr(row, "seed_support", 0.0) or 0.0),
                    "confidence_score": float(getattr(row, "confidence_score", 0.5) or 0.5),
                    "is_hotspot": bool(getattr(row, "is_hotspot", False)),
                    "detection_route": str(getattr(row, "detection_route", "")).strip(),
                    "evidence_route_anchor": bool(getattr(row, "evidence_route_anchor", False)),
                    "evidence_anchor_distance": evidence_anchor_distance,
                    "evidence_anchor_proximity": evidence_anchor_proximity,
                    "evidence_anchor_residue": str(getattr(row, "evidence_anchor_residue", "") or "").strip(),
                }
            )

    if not support_entries:
        return _empty_auto_pocket_table()

    best_entries: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for entry in sorted(
        support_entries,
        key=lambda item: (item["method_score"], item["pocket_score"], item["seed_support"], item["confidence_score"]),
        reverse=True,
    ):
        key = (str(entry["method_name"]), str(entry["chain"]), int(entry["resid"]))
        if key not in best_entries:
            best_entries[key] = entry

    residue_support: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for entry in best_entries.values():
        residue_key = (str(entry["chain"]), int(entry["resid"]), str(entry["resname"]))
        bucket = residue_support.setdefault(
            residue_key,
            {
                "methods": [],
                "pocket_ids": [],
                "method_scores": [],
                "pocket_scores": [],
                "seed_support": [],
                "confidence_score": [],
                "is_hotspot": False,
                "evidence_route_anchor": False,
                "evidence_anchor_distances": [],
                "evidence_anchor_proximities": [],
                "evidence_anchor_residues": [],
            },
        )
        bucket["methods"].append(entry["method_name"])
        pocket_id = str(entry["pocket_id"]).strip()
        if pocket_id:
            bucket["pocket_ids"].append(pocket_id)
        bucket["method_scores"].append(float(entry["method_score"]))
        bucket["pocket_scores"].append(float(entry["pocket_score"]))
        bucket["seed_support"].append(float(entry["seed_support"]))
        bucket["confidence_score"].append(float(entry["confidence_score"]))
        bucket["is_hotspot"] = bool(bucket["is_hotspot"] or entry["is_hotspot"])
        bucket["evidence_route_anchor"] = bool(bucket["evidence_route_anchor"] or entry["evidence_route_anchor"])
        if entry["evidence_anchor_distance"] is not None:
            bucket["evidence_anchor_distances"].append(float(entry["evidence_anchor_distance"]))
        bucket["evidence_anchor_proximities"].append(float(entry["evidence_anchor_proximity"]))
        evidence_anchor_residue = str(entry["evidence_anchor_residue"]).strip()
        if evidence_anchor_residue:
            bucket["evidence_anchor_residues"].append(evidence_anchor_residue)

    available_method_count = max(1, len(available_methods))
    support_rows = []
    for (chain, resid, resname), bucket in residue_support.items():
        methods = _ordered_unique_methods(bucket["methods"])
        method_text = _join_methods(methods)
        method_vote_count = len(methods)
        consensus_overlap_ratio = float(method_vote_count) / float(available_method_count)
        method_score_mean = float(np.mean(bucket["method_scores"])) if bucket["method_scores"] else 0.0
        method_score_max = float(np.max(bucket["method_scores"])) if bucket["method_scores"] else 0.0
        seed_mean = float(np.mean(bucket["seed_support"])) if bucket["seed_support"] else 0.0
        confidence_mean = float(np.mean(bucket["confidence_score"])) if bucket["confidence_score"] else 0.5
        pocket_score_mean = float(np.mean(bucket["pocket_scores"])) if bucket["pocket_scores"] else 0.0
        evidence_anchor_distance = (
            float(np.min(bucket["evidence_anchor_distances"]))
            if bucket["evidence_anchor_distances"]
            else None
        )
        evidence_anchor_proximity = (
            float(np.max(bucket["evidence_anchor_proximities"]))
            if bucket["evidence_anchor_proximities"]
            else 0.0
        )
        consensus_score = (
            0.42 * method_score_mean
            + 0.24 * method_score_max
            + 0.18 * consensus_overlap_ratio
            + 0.10 * seed_mean
            + 0.06 * confidence_mean
            + 0.04 * pocket_score_mean
        )
        if bucket["is_hotspot"] or (chain, resid) in hotspot_set:
            consensus_score += 0.05

        support_rows.append(
            {
                "chain": chain,
                "resid": resid,
                "resname": resname,
                "support_methods": tuple(methods),
                "method_vote_count": method_vote_count,
                "consensus_methods": method_text,
                "consensus_overlap_ratio": round(consensus_overlap_ratio, 3),
                "method_score_mean": round(method_score_mean, 3),
                "method_score_max": round(method_score_max, 3),
                "seed_support": round(seed_mean, 3),
                "confidence_score": round(confidence_mean, 3),
                "consensus_score": round(float(consensus_score), 3),
                "support_pocket_ids": tuple(dict.fromkeys(bucket["pocket_ids"])),
                "is_hotspot": bool(bucket["is_hotspot"] or (chain, resid) in hotspot_set),
                "evidence_route_anchor": bool(bucket["evidence_route_anchor"]),
                "evidence_anchor_distance": round(evidence_anchor_distance, 3) if evidence_anchor_distance is not None else None,
                "evidence_anchor_proximity": round(evidence_anchor_proximity, 3),
                "evidence_anchor_residue": ", ".join(dict.fromkeys(bucket["evidence_anchor_residues"])),
            }
        )

    if not support_rows:
        return _empty_auto_pocket_table()

    support_df = pd.DataFrame(support_rows)
    residue_base_df = residue_df.drop(columns=[column for column in ["seed_support", "confidence_score", "is_hotspot"] if column in residue_df.columns], errors="ignore")
    merged = residue_base_df.merge(support_df, on=["chain", "resid", "resname"], how="inner")
    if merged.empty:
        return _empty_auto_pocket_table()

    merged["consensus_score"] = pd.to_numeric(merged["consensus_score"], errors="coerce").fillna(0.0)
    merged["method_vote_count"] = pd.to_numeric(merged["method_vote_count"], errors="coerce").fillna(1).astype(int)
    merged["consensus_overlap_ratio"] = pd.to_numeric(merged["consensus_overlap_ratio"], errors="coerce").fillna(0.0)
    merged["external_support"] = _numeric_frame_series(merged, "external_support", 0.0)
    merged["external_confidence"] = _numeric_frame_series(merged, "external_confidence", 0.0)
    merged["external_exact_match"] = _boolean_frame_series(merged, "external_exact_match", False)
    merged["consensus_score"] = (
        merged["consensus_score"]
        + 0.08 * merged["external_support"]
        + 0.03 * merged["external_confidence"]
        + 0.05 * merged["external_exact_match"].astype(float)
    )

    consensus_threshold = float(merged["consensus_score"].quantile(0.62)) if len(merged) > 1 else float(merged["consensus_score"].iloc[0])
    strong_seed_support = merged["seed_support"] >= 0.60
    multi_method_gate = (
        (merged["method_vote_count"] >= 2)
        & (
            (merged["consensus_score"] >= consensus_threshold)
            | merged["is_hotspot"]
            | strong_seed_support
            | (merged["external_support"] >= 0.24)
            | merged["external_exact_match"]
        )
    )
    if allow_external_anchor_gate:
        external_anchor_min_support = _clamp_float(float(external_anchor_min_support), 0.0, 1.2)
        evidence_anchor_gate = (
            ((_numeric_frame_series(merged, "external_evidence_count", 0.0) > 0) & merged["external_exact_match"])
            | (
                (_numeric_frame_series(merged, "external_support", 0.0) >= external_anchor_min_support)
                & (_numeric_frame_series(merged, "external_evidence_count", 0.0) > 0)
            )
        )
        external_method_gate = merged["support_methods"].apply(
            lambda methods: "external-evidence" in set(_ordered_unique_methods(methods))
        )
        evidence_anchor_proximity = pd.to_numeric(merged["evidence_anchor_proximity"], errors="coerce").fillna(0.0)
        evidence_anchor_distance = pd.to_numeric(merged["evidence_anchor_distance"], errors="coerce")
        evidence_seed_distance_cutoff = _clamp_float(float(cluster_cutoff) * 0.92, 5.4, 8.4)
        evidence_neighborhood_gate = (
            external_method_gate
            & (
                (evidence_anchor_proximity >= 0.16)
                | evidence_anchor_distance.le(evidence_seed_distance_cutoff).fillna(False)
            )
            & (
                (_numeric_frame_series(merged, "external_support", 0.0) >= 0.18)
                | (pd.to_numeric(merged["seed_support"], errors="coerce").fillna(0.0) >= 0.18)
                | merged["external_exact_match"]
            )
        )
    else:
        evidence_anchor_gate = pd.Series(np.full(len(merged), False, dtype=bool), index=merged.index)
        evidence_neighborhood_gate = pd.Series(np.full(len(merged), False, dtype=bool), index=merged.index)
    support_gate = multi_method_gate | evidence_anchor_gate | evidence_neighborhood_gate

    candidate_df = merged[support_gate].copy()
    if candidate_df.empty:
        return _empty_auto_pocket_table()

    evidence_seed_candidate_count = int((evidence_anchor_gate | evidence_neighborhood_gate).reindex(candidate_df.index).fillna(False).astype(bool).sum())
    target_candidates = max(
        min_candidates,
        int(math.ceil(len(candidate_df) * float(top_fraction))),
        evidence_seed_candidate_count,
    )
    target_candidates = min(max(max_candidates, evidence_seed_candidate_count), target_candidates, len(candidate_df))
    target_candidates = max(1, target_candidates)
    consensus_anchor_mask = (
        candidate_df["is_hotspot"].fillna(False).astype(bool)
        | (pd.to_numeric(candidate_df["seed_support"], errors="coerce").fillna(0.0) >= 0.60)
        | (_numeric_frame_series(candidate_df, "external_support", 0.0) >= 0.70)
        | _boolean_frame_series(candidate_df, "external_exact_match", False)
        | (pd.to_numeric(candidate_df["method_vote_count"], errors="coerce").fillna(0).astype(int) >= 3)
        | (evidence_anchor_gate | evidence_neighborhood_gate).reindex(candidate_df.index).fillna(False).astype(bool)
    )
    candidate_df = _select_ranked_candidate_rows(
        candidate_df,
        target_candidates=target_candidates,
        sort_columns=["consensus_score", "method_vote_count", "seed_support", "confidence_score", "contact_count", "center_distance"],
        ascending=[False, False, False, False, False, True],
        anchor_mask=consensus_anchor_mask if consensus_anchor_mask.any() else None,
        hard_limit=max(max_candidates, evidence_seed_candidate_count),
    )

    if candidate_df.empty:
        return _empty_auto_pocket_table()

    cluster_indices = _cluster_residue_indices(candidate_df, cluster_cutoff)
    cluster_indices = _spatially_diversify_cluster_indices(
        candidate_df,
        cluster_indices,
        max_pockets=max_pockets,
        cluster_cutoff=cluster_cutoff,
        score_column="consensus_score",
    )
    if not cluster_indices:
        return _empty_auto_pocket_table()

    cluster_infos = []
    for indices in cluster_indices:
        cluster_df = candidate_df.iloc[indices].copy()
        cluster_methods = _ordered_unique_methods(method for methods in cluster_df["support_methods"] for method in methods)
        cluster_score = float(cluster_df["consensus_score"].mean())
        cluster_score += 0.12 * float(cluster_df["consensus_overlap_ratio"].mean())
        cluster_score += 0.08 * float(cluster_df["is_hotspot"].mean())
        cluster_score += 0.05 * float(cluster_df["seed_support"].mean())
        cluster_score += 0.04 * float(cluster_df["confidence_score"].mean())

        if cluster_score < 0.25 and len(cluster_methods) < 2 and not cluster_df["is_hotspot"].any():
            continue

        cluster_infos.append((cluster_score, len(cluster_methods), cluster_df))

    if not cluster_infos:
        return _empty_auto_pocket_table()

    cluster_infos.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)

    pocket_rows = []
    for pocket_rank, (_, method_count, cluster_df) in enumerate(cluster_infos[:max_pockets], start=1):
        cluster_df = cluster_df.sort_values(
            ["consensus_score", "method_vote_count", "seed_support", "confidence_score", "resid"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)
        cluster_coords = cluster_df[["x", "y", "z"]].to_numpy(dtype=float)
        cluster_centroid = cluster_coords.mean(axis=0) if len(cluster_coords) else np.zeros(3, dtype=float)
        cluster_distances = np.sqrt(np.sum((cluster_coords - cluster_centroid) ** 2, axis=1)) if len(cluster_coords) else np.array([], dtype=float)
        cluster_methods = _ordered_unique_methods(method for methods in cluster_df["support_methods"] for method in methods)
        if not cluster_methods:
            cluster_methods = ["geometry-cluster"]
        cluster_method_text = _join_methods(cluster_methods)
        cluster_overlap_ratio = float(len(cluster_methods)) / float(available_method_count)
        cluster_hotspot_overlap = int(cluster_df["is_hotspot"].sum()) if "is_hotspot" in cluster_df.columns else 0
        cluster_hotspot_ratio = float(cluster_hotspot_overlap) / float(len(cluster_df)) if len(cluster_df) else 0.0
        cluster_seed_mean = float(cluster_df["seed_support"].mean()) if "seed_support" in cluster_df.columns else 0.0
        cluster_confidence_mean = float(cluster_df["confidence_score"].mean()) if "confidence_score" in cluster_df.columns else 0.5
        cluster_score = float(cluster_df["consensus_score"].mean())
        cluster_score += 0.12 * cluster_overlap_ratio
        cluster_score += 0.08 * cluster_hotspot_ratio
        cluster_score += 0.05 * cluster_seed_mean
        cluster_score += 0.04 * cluster_confidence_mean
        if len(cluster_methods) > 1:
            cluster_score += 0.05 * (len(cluster_methods) - 1)

        pocket_id = f"AutoPocket-{pocket_rank}"
        detection_method = "consensus-rerank" if len(cluster_methods) > 1 else cluster_methods[0]
        route_prefix = "consensus" if len(cluster_methods) > 1 else f"fallback-{cluster_method_text}"
        detection_route = _detection_route_label(route_prefix)
        depth_avg = float(cluster_distances.mean()) if cluster_distances.size else 0.0
        depth_max = float(cluster_distances.max()) if cluster_distances.size else 0.0
        cluster_volume = _estimate_bbox_volume(cluster_coords, padding=2.0)

        for row_index, row in enumerate(cluster_df.itertuples(index=False)):
            proximity_distance = float(cluster_distances[row_index]) if row_index < len(cluster_distances) else None
            pocket_rows.append(
                {
                    "pocket_id": pocket_id,
                    "chain": row.chain,
                    "resid": int(row.resid),
                    "resname": row.resname,
                    "volume": round(float(cluster_volume), 3),
                    "score": round(float(cluster_score), 3),
                    "residue_score": round(float(getattr(row, "consensus_score", 0.0)), 3),
                    "contact_count": int(getattr(row, "contact_count", 0) or 0),
                    "center_distance": round(float(getattr(row, "center_distance", 0.0)), 3),
                    "ligand_contact_count": int(getattr(row, "ligand_contact_count", 0) or 0),
                    "detection_method": detection_method,
                    "detection_route": detection_route,
                    "is_hotspot": bool(getattr(row, "is_hotspot", False)),
                    "depth_avg": round(depth_avg, 3),
                    "depth_max": round(depth_max, 3),
                    "overlap_ratio": round(cluster_hotspot_ratio, 3),
                    "proximity_distance": round(float(proximity_distance), 3) if proximity_distance is not None else None,
                    "precision_score": round(float(getattr(row, "consensus_score", 0.0)), 3),
                    "seed_support": round(float(getattr(row, "seed_support", 0.0) or 0.0), 3),
                    "confidence_score": round(float(getattr(row, "confidence_score", 0.5) or 0.5), 3),
                    **_external_row_payload(row),
                    "consensus_score": round(float(getattr(row, "consensus_score", 0.0)), 3),
                    "consensus_methods": cluster_method_text,
                    "method_vote_count": int(len(cluster_methods)),
                    "consensus_overlap_ratio": round(cluster_overlap_ratio, 3),
                }
            )

    if not pocket_rows:
        return _empty_auto_pocket_table()

    table = pd.DataFrame(pocket_rows).sort_values(
        ["score", "residue_score", "volume", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)
    return _ensure_auto_pocket_columns(table)


def detect_auto_pocket_table(
    pdb_text: str,
    hotspot_residues: Optional[Sequence[Tuple[str, int]]] = None,
    external_site_df: Optional[pd.DataFrame] = None,
    conservation_site_df: Optional[pd.DataFrame] = None,
    *,
    adaptive_profile: bool = True,
    prefer_kvfinder: bool = True,
    prefer_p2rank: bool = False,
    prefer_ligand: bool = True,
    enable_external_evidence_route: bool = True,
    external_evidence_min_support: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_support"],
    external_evidence_min_confidence: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_confidence"],
    external_evidence_min_mapping_quality: float = EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_mapping_quality"],
    external_evidence_radius: Optional[float] = None,
    contact_cutoff: float = 7.2,
    cluster_cutoff: float = 7.5,
    ligand_radius: float = 4.5,
    top_fraction: float = 0.25,
    min_candidates: int = 2,
    max_candidates: int = 12,
    max_pockets: int = 4,
    kv_step: float = 0.5,
    kv_probe_in: float = 1.4,
    kv_probe_out: float = 4.0,
    kv_volume_cutoff: float = 5.0,
    p2rank_profile: str = "default",
    p2rank_executable: Optional[str] = None,
    p2rank_output_dir: Optional[str] = None,
    p2rank_timeout_sec: float = 180.0,
) -> pd.DataFrame:
    """Detect pocket candidates directly from a PDB structure.

    The detector uses a consensus-first strategy: a structure-adaptive
    multiscale KVFinder pass, optional P2Rank residue-level predictions,
    ligand-guided residue seeding, and a conservative geometry cluster are
    evaluated and merged by overlap-aware reranking.
    """

    hotspot_set = {(_normalize_chain(chain), int(resid)) for chain, resid in (hotspot_residues or [])}
    external_summary = _summarize_external_evidence_table(external_site_df)
    conservation_summary = _summarize_external_evidence_table(conservation_site_df)
    external_route_min_support = _clamp_float(float(external_evidence_min_support), 0.0, 1.2)
    external_route_min_confidence = _clamp_float(float(external_evidence_min_confidence), 0.0, 1.0)
    external_route_min_mapping_quality = _clamp_float(float(external_evidence_min_mapping_quality), 0.0, 1.0)
    external_route_radius = (
        _clamp_float(float(external_evidence_radius), 3.5, 12.0)
        if external_evidence_radius is not None
        else None
    )
    external_route_evidence_rows = int(external_summary.get("evidence_rows", 0) or 0)
    external_route_requested = bool(enable_external_evidence_route)
    external_route_runnable = bool(external_route_requested and external_route_evidence_rows > 0)
    external_consensus_min_support = max(
        EXTERNAL_EVIDENCE_ROUTE_DEFAULTS["min_consensus_support"],
        external_route_min_support,
    )
    diagnostics: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "adaptive_profile": bool(adaptive_profile),
        "requested_methods": {
            "kvfinder": bool(prefer_kvfinder),
            "p2rank": bool(prefer_p2rank),
            "external-evidence": bool(external_route_runnable),
            "ligand-proximity": bool(prefer_ligand),
            "geometry-cluster": True,
        },
        "external_evidence": external_summary,
        "external_evidence_route": {
            "enabled": bool(external_route_requested),
            "runnable": bool(external_route_runnable),
            "min_support": external_route_min_support,
            "min_confidence": external_route_min_confidence,
            "min_mapping_quality": external_route_min_mapping_quality,
            "min_consensus_support": external_consensus_min_support,
            "radius": external_route_radius,
        },
        "conservation_evidence": conservation_summary,
        "hotspot_seed_count": int(len(hotspot_set)),
        "diagnostics": diagnostics,
    }

    atom_df = parse_pdb_atoms(pdb_text)
    if atom_df.empty or not {"x", "y", "z"}.issubset(atom_df.columns):
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "input",
                enabled=True,
                available=False,
                status="invalid-structure",
                note="No valid atomic coordinates were parsed from the structure.",
            )
        )
        return _finalize_pocket_detection_result(None, metadata=metadata)

    protein_atoms = atom_df.copy()
    if "record_type" in protein_atoms.columns:
        protein_atoms = protein_atoms[protein_atoms["record_type"].astype(str).str.upper() == "ATOM"].copy()
    if protein_atoms.empty:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "input",
                enabled=True,
                available=False,
                status="no-protein-atoms",
                note="The structure does not contain standard protein ATOM records.",
            )
        )
        return _finalize_pocket_detection_result(None, metadata=metadata)
    metadata["structure_atom_count"] = int(len(protein_atoms))

    residue_df, ligand_atoms = _build_precision_residue_table(
        atom_df,
        hotspot_set,
        pdb_text=pdb_text,
        contact_cutoff=contact_cutoff,
        ligand_radius=ligand_radius,
        external_site_df=external_site_df,
        conservation_site_df=conservation_site_df,
    )
    if residue_df.empty:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "residue-support",
                enabled=True,
                available=False,
                status="empty",
                note="Residue support table could not be constructed from the structure.",
            )
        )
        return _finalize_pocket_detection_result(None, metadata=metadata)
    metadata["residue_candidate_count"] = int(len(residue_df))
    metadata["ligand_atom_count"] = int(len(ligand_atoms))

    profile = _build_detection_profile(
        contact_cutoff=contact_cutoff,
        cluster_cutoff=cluster_cutoff,
        ligand_radius=ligand_radius,
        top_fraction=top_fraction,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        max_pockets=max_pockets,
        kv_step=kv_step,
        kv_probe_in=kv_probe_in,
        kv_probe_out=kv_probe_out,
        kv_volume_cutoff=kv_volume_cutoff,
    )
    if adaptive_profile:
        profile = _infer_adaptive_detection_profile(
            residue_df,
            ligand_atoms,
            hotspot_set=hotspot_set,
            contact_cutoff=contact_cutoff,
            cluster_cutoff=cluster_cutoff,
            ligand_radius=ligand_radius,
            top_fraction=top_fraction,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            max_pockets=max_pockets,
            kv_step=kv_step,
            kv_probe_in=kv_probe_in,
            kv_probe_out=kv_probe_out,
            kv_volume_cutoff=kv_volume_cutoff,
        )
        residue_df, ligand_atoms = _build_precision_residue_table(
            atom_df,
            hotspot_set,
            pdb_text=pdb_text,
            contact_cutoff=float(profile["contact_cutoff"]),
            ligand_radius=float(profile["ligand_radius"]),
            external_site_df=external_site_df,
            conservation_site_df=conservation_site_df,
        )
        if residue_df.empty:
            diagnostics.append(
                _build_detection_diagnostic_entry(
                    "residue-support",
                    enabled=True,
                    available=False,
                    status="empty",
                    note="Adaptive profile recomputation removed all residue candidates.",
                )
            )
            return _finalize_pocket_detection_result(None, metadata=metadata)

    contact_cutoff = float(profile["contact_cutoff"])
    cluster_cutoff = float(profile["cluster_cutoff"])
    ligand_radius = float(profile["ligand_radius"])
    top_fraction = float(profile["top_fraction"])
    min_candidates = int(profile["min_candidates"])
    max_candidates = int(profile["max_candidates"])
    max_pockets = int(profile["max_pockets"])
    kv_step = float(profile["kv_step"])
    kv_probe_in = float(profile["kv_probe_in"])
    kv_probe_out = float(profile["kv_probe_out"])
    kv_volume_cutoff = float(profile["kv_volume_cutoff"])
    kv_probe_profiles = profile.get("kv_probe_profiles") or []
    metadata["adaptive_profile_params"] = {
        "contact_cutoff": contact_cutoff,
        "cluster_cutoff": cluster_cutoff,
        "ligand_radius": ligand_radius,
        "top_fraction": top_fraction,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates,
        "max_pockets": max_pockets,
        "kv_probe_profiles": int(len(kv_probe_profiles)),
    }

    method_tables: list[pd.DataFrame] = []

    if prefer_kvfinder:
        kvfinder_profile_tables: list[pd.DataFrame] = []
        for kv_profile in kv_probe_profiles:
            kvfinder_table = _detect_with_kvfinder(
                pdb_text,
                hotspot_set,
                step=float(kv_profile.get("step", kv_step)),
                probe_in=float(kv_profile.get("probe_in", kv_probe_in)),
                probe_out=float(kv_profile.get("probe_out", kv_probe_out)),
                volume_cutoff=float(kv_profile.get("volume_cutoff", kv_volume_cutoff)),
                residue_support_df=residue_df,
            )
            if not kvfinder_table.empty:
                kvfinder_profile_tables.append(kvfinder_table)
        kvfinder_table = _merge_multiscale_kvfinder_tables(
            kvfinder_profile_tables,
            residue_df,
            hotspot_set=hotspot_set,
            cluster_cutoff=cluster_cutoff,
            top_fraction=top_fraction,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            max_pockets=max_pockets,
        )
        if not kvfinder_table.empty:
            method_tables.append(kvfinder_table)
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "kvfinder",
                enabled=True,
                available=PYKVFINDER_AVAILABLE,
                status="used" if not kvfinder_table.empty else ("unavailable" if not PYKVFINDER_AVAILABLE else "empty"),
                pocket_count=int(kvfinder_table["pocket_id"].astype(str).nunique()) if not kvfinder_table.empty else 0,
                residue_rows=int(len(kvfinder_table)),
                note=f"profiles={len(kvfinder_profile_tables)}/{len(kv_probe_profiles)}",
            )
        )
    else:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "kvfinder",
                enabled=False,
                available=PYKVFINDER_AVAILABLE,
                status="disabled",
            )
        )

    if prefer_p2rank:
        p2rank_table, p2rank_meta = _detect_with_p2rank(
            pdb_text,
            hotspot_set,
            residue_support_df=residue_df,
            executable=p2rank_executable,
            profile=p2rank_profile,
            output_dir=p2rank_output_dir,
            timeout_sec=p2rank_timeout_sec,
        )
        if not p2rank_table.empty:
            method_tables.append(p2rank_table)
        p2rank_status = str(p2rank_meta.get("status") or "").strip().lower()
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "p2rank",
                enabled=True,
                available=p2rank_status not in {"unavailable", "failed"},
                status="used" if not p2rank_table.empty else (p2rank_status or "empty"),
                pocket_count=int(p2rank_table["pocket_id"].astype(str).nunique()) if not p2rank_table.empty else 0,
                residue_rows=int(len(p2rank_table)),
                note=str(
                    p2rank_meta.get("reason")
                    or p2rank_meta.get("stderr")
                    or p2rank_meta.get("stdout")
                    or ""
                )[:240],
            )
        )
        metadata["p2rank_meta"] = p2rank_meta
    else:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "p2rank",
                enabled=False,
                available=bool(p2rank_executable),
                status="disabled",
            )
        )

    evidence_table = (
        _build_external_evidence_method_table(
            residue_df,
            hotspot_set,
            cluster_cutoff=cluster_cutoff,
            max_candidates=max_candidates,
            max_pockets=max_pockets,
            min_anchor_support=external_route_min_support,
            min_anchor_confidence=external_route_min_confidence,
            min_mapping_quality=external_route_min_mapping_quality,
            evidence_radius=external_route_radius,
        )
        if external_route_runnable
        else _empty_auto_pocket_table()
    )
    if not evidence_table.empty:
        method_tables.append(evidence_table)
    if not external_route_requested:
        evidence_status = "disabled"
    elif external_route_evidence_rows <= 0:
        evidence_status = "no-evidence"
    else:
        evidence_status = "used" if not evidence_table.empty else "empty"
    diagnostics.append(
        _build_detection_diagnostic_entry(
            "external-evidence",
            enabled=bool(external_route_requested),
            available=not evidence_table.empty,
            status=evidence_status,
            pocket_count=int(evidence_table["pocket_id"].astype(str).nunique()) if not evidence_table.empty else 0,
            residue_rows=int(len(evidence_table)),
            note=(
                f"rows={external_route_evidence_rows}; "
                f"support>={external_route_min_support:.2f}; "
                f"confidence>={external_route_min_confidence:.2f}; "
                f"quality>={external_route_min_mapping_quality:.2f}; "
                f"radius={external_route_radius if external_route_radius is not None else 'auto'}"
            ),
        )
    )

    if prefer_ligand and not ligand_atoms.empty:
        ligand_table = _build_multiscale_precision_method_table(
            residue_df,
            hotspot_set,
            method_kind="ligand",
            detection_method="ligand-proximity",
            route_suffix="ligand",
            cluster_cutoff=cluster_cutoff,
            top_fraction=top_fraction,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            max_pockets=max_pockets,
        )
        if not ligand_table.empty:
            method_tables.append(ligand_table)
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "ligand-proximity",
                enabled=True,
                available=True,
                status="used" if not ligand_table.empty else "empty",
                pocket_count=int(ligand_table["pocket_id"].astype(str).nunique()) if not ligand_table.empty else 0,
                residue_rows=int(len(ligand_table)),
                note=f"ligand_atoms={len(ligand_atoms)}",
            )
        )
    elif prefer_ligand:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "ligand-proximity",
                enabled=True,
                available=False,
                status="no-ligand",
                note="No ligand HETATM records were available for ligand-guided detection.",
            )
        )
    else:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "ligand-proximity",
                enabled=False,
                available=not ligand_atoms.empty,
                status="disabled",
            )
        )

    geometry_table = _build_multiscale_precision_method_table(
        residue_df,
        hotspot_set,
        method_kind="geometry",
        detection_method="geometry-cluster",
        route_suffix="geometry",
        cluster_cutoff=cluster_cutoff,
        top_fraction=top_fraction,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        max_pockets=max_pockets,
    )
    if not geometry_table.empty:
        method_tables.append(geometry_table)
    diagnostics.append(
        _build_detection_diagnostic_entry(
            "geometry-cluster",
            enabled=True,
            available=True,
            status="used" if not geometry_table.empty else "empty",
            pocket_count=int(geometry_table["pocket_id"].astype(str).nunique()) if not geometry_table.empty else 0,
            residue_rows=int(len(geometry_table)),
        )
    )

    if not method_tables:
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "consensus",
                enabled=True,
                available=False,
                status="no-methods",
                note="No detection route produced a usable pocket table.",
            )
        )
        return _finalize_pocket_detection_result(None, metadata=metadata)

    if len(method_tables) == 1:
        single_table = method_tables[0]
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "consensus",
                enabled=True,
                available=True,
                status="single-method",
                pocket_count=int(single_table["pocket_id"].astype(str).nunique()) if not single_table.empty else 0,
                residue_rows=int(len(single_table)),
                note=str(single_table["detection_method"].iloc[0]) if "detection_method" in single_table.columns and not single_table.empty else "",
            )
        )
        return _finalize_pocket_detection_result(single_table, metadata=metadata)

    consensus_table = _build_consensus_pocket_table(
        method_tables,
        residue_df,
        hotspot_set=hotspot_set,
        cluster_cutoff=cluster_cutoff,
        top_fraction=top_fraction,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        max_pockets=max_pockets,
        allow_external_anchor_gate=external_route_requested,
        external_anchor_min_support=external_consensus_min_support,
    )

    if consensus_table.empty:
        fallback_table = max(method_tables, key=len)
        diagnostics.append(
            _build_detection_diagnostic_entry(
                "consensus",
                enabled=True,
                available=True,
                status="fallback",
                pocket_count=int(fallback_table["pocket_id"].astype(str).nunique()) if not fallback_table.empty else 0,
                residue_rows=int(len(fallback_table)),
                note="Consensus merge returned empty; using the largest individual method table.",
            )
        )
        return _finalize_pocket_detection_result(fallback_table, metadata=metadata)

    diagnostics.append(
        _build_detection_diagnostic_entry(
            "consensus",
            enabled=True,
            available=True,
            status="consensus",
            pocket_count=int(consensus_table["pocket_id"].astype(str).nunique()) if not consensus_table.empty else 0,
            residue_rows=int(len(consensus_table)),
            note=f"methods={len(method_tables)}",
        )
    )
    return _finalize_pocket_detection_result(consensus_table, metadata=metadata)
