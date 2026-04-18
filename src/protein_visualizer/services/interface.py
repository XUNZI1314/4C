from __future__ import annotations

from io import StringIO
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["chain", "resid", "resname", "annotation", "region_type"]
INFERRED_EXTRA_COLUMNS = ["annotation_source", "inference_basis"]


def parse_interface_annotation_table(text: str) -> pd.DataFrame:
    annotation_df = pd.read_csv(StringIO(text.strip()))
    missing = [column for column in REQUIRED_COLUMNS if column not in annotation_df.columns]
    if missing:
        raise ValueError(f"界面注释文件缺少必要列: {', '.join(missing)}")
    annotation_df = annotation_df.copy()
    annotation_df["chain"] = annotation_df["chain"].astype(str).str.strip().replace("", "A")
    annotation_df["resid"] = pd.to_numeric(annotation_df["resid"], errors="coerce").astype("Int64")
    annotation_df["resname"] = annotation_df["resname"].astype(str).str.strip().str.upper()
    annotation_df["annotation"] = annotation_df["annotation"].astype(str).str.strip()
    annotation_df["region_type"] = annotation_df["region_type"].astype(str).str.strip()
    annotation_df = annotation_df.dropna(subset=["resid"]).copy()
    annotation_df["resid"] = annotation_df["resid"].astype(int)
    return annotation_df.sort_values(["chain", "resid"]).reset_index(drop=True)


def _normalized_residue_set(residue_pairs: Optional[Sequence[Tuple[str, int]]]) -> set[Tuple[str, int]]:
    return {(str(chain).strip() or "A", int(resid)) for chain, resid in (residue_pairs or [])}


def _normalize_interface_table(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or getattr(table, "empty", True):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    normalized = table.copy()
    for column in REQUIRED_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    normalized["chain"] = normalized["chain"].astype(str).str.strip().replace("", "A")
    normalized["resid"] = pd.to_numeric(normalized["resid"], errors="coerce").astype("Int64")
    normalized = normalized.dropna(subset=["resid"]).copy()
    normalized["resid"] = normalized["resid"].astype(int)
    normalized["resname"] = normalized["resname"].astype(str).str.strip().str.upper()
    normalized["annotation"] = normalized["annotation"].astype(str).str.strip()
    normalized["region_type"] = normalized["region_type"].astype(str).str.strip().replace("", "interface")
    return normalized.reset_index(drop=True)


def _numeric_series(frame: Optional[pd.DataFrame], column: str, default: float = 0.0) -> pd.Series:
    if frame is None or getattr(frame, "empty", True):
        return pd.Series(dtype=float)
    if column not in frame.columns:
        return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def merge_interface_annotation_tables(
    primary_df: Optional[pd.DataFrame],
    secondary_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    primary = _normalize_interface_table(primary_df)
    secondary = _normalize_interface_table(secondary_df)
    if primary.empty:
        return secondary
    if secondary.empty:
        return primary

    combined = pd.concat([primary.assign(_source_rank=0), secondary.assign(_source_rank=1)], ignore_index=True)
    combined = combined.sort_values(["_source_rank", "chain", "resid", "resname"], ascending=[True, True, True, True])
    combined = combined.drop_duplicates(subset=["chain", "resid", "resname"], keep="first")
    return combined.drop(columns=["_source_rank"]).reset_index(drop=True)


def build_inferred_interface_annotations(
    energy_table: Optional[pd.DataFrame],
    *,
    min_density: float = 0.2,
    min_count: int = 1,
    top_n: int = 12,
) -> pd.DataFrame:
    empty_columns = REQUIRED_COLUMNS + INFERRED_EXTRA_COLUMNS
    if energy_table is None or getattr(energy_table, "empty", True):
        return pd.DataFrame(columns=empty_columns)

    required = {"chain", "resid", "resname"}
    if not required.issubset(energy_table.columns):
        return pd.DataFrame(columns=empty_columns)

    working = energy_table.copy()
    interface_count = _numeric_series(working, "interface_contact_count", 0.0).astype(int)
    interface_density = _numeric_series(working, "interface_contact_density", 0.0)
    interface_score = _numeric_series(working, "interface_contact_score", 0.0)

    has_interface_signal = not (
        (interface_count <= 0).all()
        and np.isclose(float(interface_density.max()), 0.0)
        and np.isclose(float(interface_score.max()), 0.0)
    )

    if has_interface_signal:
        support_count = interface_count
        support_density = interface_density
        support_score = interface_score
        exposure = _numeric_series(working, "surface_proxy", 0.5).clip(lower=0.0, upper=1.0)
        inference_basis = "inter-chain-contact"
        annotation_prefix = "结构推断界面候选"
        exposure_threshold = 0.0
    else:
        support_count = _numeric_series(working, "contact_count", 0.0).astype(int)
        support_density = _numeric_series(working, "contact_density", 0.0)
        if np.isclose(float(support_density.max()), 0.0):
            max_count = max(int(support_count.max()), 1)
            support_density = support_count.clip(upper=max_count).astype(float) / float(max_count)
        support_score = _numeric_series(working, "contact_score", 0.0)
        exposure = _numeric_series(working, "surface_proxy", 0.0).clip(lower=0.0, upper=1.0)
        if (support_count <= 0).all() and np.isclose(float(support_density.max()), 0.0) and np.isclose(float(support_score.max()), 0.0):
            return pd.DataFrame(columns=empty_columns)
        inference_basis = "surface-contact"
        annotation_prefix = "结构推断表面接触候选"
        exposure_threshold = max(0.35, float(exposure.quantile(0.55))) if len(exposure) > 1 else 0.35

    working["_support_count"] = support_count
    working["_support_density"] = support_density
    working["_support_score"] = support_score
    working["_exposure"] = exposure

    density_threshold = max(float(min_density), float(support_density.quantile(0.60))) if len(support_density) > 1 else float(min_density)
    score_threshold = float(support_score.quantile(0.60)) if len(support_score) > 1 else float(support_score.max())
    count_threshold = max(int(min_count), int(np.ceil(support_count.quantile(0.60)))) if len(support_count) > 1 else int(min_count)

    support_mask = (
        (working["_support_count"] >= count_threshold)
        | (working["_support_density"] >= density_threshold)
        | (working["_support_score"] >= score_threshold)
    )
    if exposure_threshold > 0.0:
        support_mask = support_mask & (working["_exposure"] >= exposure_threshold)

    candidate_df = working.loc[support_mask].copy()
    if candidate_df.empty:
        max_support_score = float(working["_support_score"].max())
        if max_support_score > 0.0:
            normalized_support_score = working["_support_score"] / max_support_score
        else:
            normalized_support_score = pd.Series(np.zeros(len(working), dtype=float), index=working.index)
        ranked = working.assign(
            _rank_score=(
                (0.30 * working["_support_density"])
                + (0.25 * normalized_support_score)
                + (0.20 * np.minimum(working["_support_count"], 6) / 6.0)
                + (0.25 * working["_exposure"])
            )
        ).sort_values("_rank_score", ascending=False)
        candidate_df = ranked.head(int(max(1, top_n))).drop(columns=["_rank_score"], errors="ignore").copy()

    if candidate_df.empty:
        return pd.DataFrame(columns=empty_columns)

    core_density = float(working["_support_density"].quantile(0.80)) if len(working) > 1 else float(working["_support_density"].max())
    rim_density = float(working["_support_density"].quantile(0.50)) if len(working) > 1 else float(working["_support_density"].max())
    core_count = int(np.ceil(working["_support_count"].quantile(0.80))) if len(working) > 1 else int(working["_support_count"].max())
    rim_count = int(np.ceil(working["_support_count"].quantile(0.50))) if len(working) > 1 else int(working["_support_count"].max())
    core_exposure = float(working["_exposure"].quantile(0.75)) if len(working) > 1 else float(working["_exposure"].max())
    rim_exposure = float(working["_exposure"].quantile(0.50)) if len(working) > 1 else float(working["_exposure"].max())

    def _region_type(row) -> str:
        density = float(pd.to_numeric(row.get("_support_density"), errors="coerce") or 0.0)
        count = int(pd.to_numeric(row.get("_support_count"), errors="coerce") or 0)
        exposure_value = float(pd.to_numeric(row.get("_exposure"), errors="coerce") or 0.0)
        if density >= core_density or count >= max(core_count, 2):
            return "interface-core"
        if exposure_value >= max(core_exposure, 0.35) and density >= rim_density:
            return "interface-core"
        if density >= rim_density or count >= max(rim_count, 1):
            return "interface-rim"
        if exposure_value >= max(rim_exposure, 0.25):
            return "interface-rim"
        return "contact-rim"

    def _annotation_text(row) -> str:
        count = int(pd.to_numeric(row.get("_support_count"), errors="coerce") or 0)
        density = float(pd.to_numeric(row.get("_support_density"), errors="coerce") or 0.0)
        score = float(pd.to_numeric(row.get("_support_score"), errors="coerce") or 0.0)
        exposure_value = float(pd.to_numeric(row.get("_exposure"), errors="coerce") or 0.0)
        if inference_basis == "inter-chain-contact":
            return f"{annotation_prefix}；跨链接触数={count}，界面密度={density:.2f}，得分={score:.2f}"
        return f"{annotation_prefix}；接触数={count}，表面暴露={exposure_value:.2f}，接触得分={score:.2f}"

    inferred = candidate_df[["chain", "resid", "resname"]].copy()
    inferred["annotation"] = candidate_df.apply(_annotation_text, axis=1)
    inferred["region_type"] = candidate_df.apply(_region_type, axis=1)
    inferred["annotation_source"] = "structure-inference"
    inferred["inference_basis"] = inference_basis
    return inferred.sort_values(["chain", "resid"]).reset_index(drop=True)


def enrich_interface_annotations(
    annotation_df: pd.DataFrame,
    *,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    hotspot_residues: Optional[Sequence[Tuple[str, int]]] = None,
) -> pd.DataFrame:
    if annotation_df is None or annotation_df.empty:
        return pd.DataFrame(
            columns=list(annotation_df.columns) if annotation_df is not None else REQUIRED_COLUMNS
        )

    enriched = annotation_df.copy()
    pocket_set = _normalized_residue_set(pocket_residues)
    hotspot_set = _normalized_residue_set(hotspot_residues)
    enriched["is_pocket"] = [(str(row.chain).strip() or "A", int(row.resid)) in pocket_set for row in enriched.itertuples(index=False)]
    enriched["is_hotspot"] = [(str(row.chain).strip() or "A", int(row.resid)) in hotspot_set for row in enriched.itertuples(index=False)]
    enriched["is_overlap"] = enriched["is_pocket"] & enriched["is_hotspot"]
    enriched["residue_label"] = enriched.apply(lambda row: f"{row.resname} {row.chain}{int(row.resid)}".strip(), axis=1)
    return enriched


def build_interface_summary(annotation_df: pd.DataFrame) -> pd.DataFrame:
    if annotation_df is None or annotation_df.empty:
        return pd.DataFrame(
            columns=["region_type", "residue_count", "pocket_count", "hotspot_count", "overlap_count", "residue_labels"]
        )

    records = []
    grouped = annotation_df.groupby("region_type", dropna=False)
    for region_type, group in grouped:
        records.append(
            {
                "region_type": str(region_type),
                "residue_count": int(len(group)),
                "pocket_count": int(group.get("is_pocket", pd.Series(dtype=bool)).sum()) if "is_pocket" in group.columns else 0,
                "hotspot_count": int(group.get("is_hotspot", pd.Series(dtype=bool)).sum()) if "is_hotspot" in group.columns else 0,
                "overlap_count": int(group.get("is_overlap", pd.Series(dtype=bool)).sum()) if "is_overlap" in group.columns else 0,
                "residue_labels": ", ".join(group["residue_label"].astype(str).tolist()[:6]) if "residue_label" in group.columns else ", ".join(
                    f"{row.resname} {row.chain}{int(row.resid)}" for row in group.itertuples(index=False)
                ),
            }
        )

    return pd.DataFrame(records).sort_values(["overlap_count", "residue_count"], ascending=[False, False]).reset_index(drop=True)


def build_interface_overlap_summary(
    annotation_df: Optional[pd.DataFrame],
    *,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    hotspot_residues: Optional[Sequence[Tuple[str, int]]] = None,
) -> pd.DataFrame:
    if annotation_df is None or getattr(annotation_df, "empty", True):
        return pd.DataFrame(columns=["category", "count"])

    pocket_set = _normalized_residue_set(pocket_residues)
    hotspot_set = _normalized_residue_set(hotspot_residues)
    residue_keys = {
        (str(row.chain).strip() or "A", int(row.resid))
        for row in annotation_df.itertuples(index=False)
    }
    interface_hotspots = residue_keys & hotspot_set
    interface_pockets = residue_keys & pocket_set
    triple_overlap = interface_hotspots & pocket_set
    pocket_hotspots = pocket_set & hotspot_set

    rows = [
        {"category": "interface_residues", "count": int(len(residue_keys))},
        {"category": "pocket_residues", "count": int(len(pocket_set))},
        {"category": "hotspot_residues", "count": int(len(hotspot_set))},
        {"category": "interface_and_pocket", "count": int(len(interface_pockets))},
        {"category": "interface_and_hotspot", "count": int(len(interface_hotspots))},
        {"category": "pocket_and_hotspot", "count": int(len(pocket_hotspots))},
        {"category": "triple_overlap", "count": int(len(triple_overlap))},
    ]
    return pd.DataFrame(rows)
