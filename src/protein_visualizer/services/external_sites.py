from __future__ import annotations

from io import StringIO
import json
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib import error, request

import pandas as pd


UNIPROT_FEATURE_SCORES: Dict[str, float] = {
    "Active site": 1.0,
    "Binding site": 0.88,
    "Metal binding": 0.9,
    "Site": 0.76,
    "Mutagenesis": 0.82,
    "DNA binding": 0.82,
    "Nucleotide binding": 0.84,
}

MCSA_ROLE_SCORES: Dict[str, float] = {
    "catalytic residue": 1.0,
    "nucleophile": 0.98,
    "proton donor": 0.96,
    "proton acceptor": 0.96,
    "electrostatic stabiliser": 0.92,
    "electrostatic stabilizer": 0.92,
    "metal ligand": 0.9,
    "metal binding": 0.9,
    "substrate positioning": 0.84,
}

EVIDENCE_COLUMNS = [
    "chain",
    "resid",
    "evidence_source",
    "evidence_type",
    "evidence_score",
    "evidence_note",
    "uniprot_resid",
    "mapping_level",
    "mapping_confidence",
    "mapping_method",
    "article_title",
    "pmid",
    "pmcid",
    "doi",
    "evidence_snippet",
    "sentence_index",
    "extraction_pattern",
    "requires_manual_review",
]

EVIDENCE_SOURCE_DETAIL_COLUMNS = [
    "article_title",
    "pmid",
    "pmcid",
    "doi",
    "evidence_snippet",
    "sentence_index",
    "extraction_pattern",
    "requires_manual_review",
]

EVIDENCE_COLUMN_DEFAULTS: Dict[str, object] = {
    "chain": "",
    "resid": 0,
    "evidence_source": "external",
    "evidence_type": "Functional site",
    "evidence_score": 0.75,
    "evidence_note": "",
    "uniprot_resid": 0,
    "mapping_level": "weak",
    "mapping_confidence": 0.3,
    "mapping_method": "unknown",
    "article_title": "",
    "pmid": "",
    "pmcid": "",
    "doi": "",
    "evidence_snippet": "",
    "sentence_index": "",
    "extraction_pattern": "",
    "requires_manual_review": False,
}

MANUAL_KEY_RESIDUE_TEMPLATE_COLUMNS = [
    "chain",
    "resid",
    "resname",
    "evidence_source",
    "evidence_type",
    "evidence_score",
    "evidence_note",
    "uniprot_resid",
    "mapping_level",
    "mapping_confidence",
    "pmid",
    "doi",
    "evidence_snippet",
    "requires_manual_review",
]

MANUAL_KEY_RESIDUE_COLUMN_ALIASES = {
    "chain": "chain",
    "chain_id": "chain",
    "auth_asym_id": "chain",
    "resid": "resid",
    "residue_id": "resid",
    "residue_number": "resid",
    "position": "resid",
    "pdb_resid": "resid",
    "resname": "resname",
    "residue_name": "resname",
    "aa": "resname",
    "source": "evidence_source",
    "reference_source": "evidence_source",
    "evidence_source": "evidence_source",
    "type": "evidence_type",
    "site_type": "evidence_type",
    "reference_type": "evidence_type",
    "evidence_type": "evidence_type",
    "score": "evidence_score",
    "confidence": "evidence_score",
    "evidence_score": "evidence_score",
    "note": "evidence_note",
    "notes": "evidence_note",
    "reference_note": "evidence_note",
    "evidence_note": "evidence_note",
    "uniprot_position": "uniprot_resid",
    "uniprot_resid": "uniprot_resid",
    "mapping_level": "mapping_level",
    "mapping_confidence": "mapping_confidence",
    "mapping_method": "mapping_method",
    "pmid": "pmid",
    "pmcid": "pmcid",
    "doi": "doi",
    "article_title": "article_title",
    "snippet": "evidence_snippet",
    "evidence_snippet": "evidence_snippet",
    "requires_manual_review": "requires_manual_review",
    "manual_review": "requires_manual_review",
}


def _empty_evidence_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def build_manual_key_residue_template() -> pd.DataFrame:
    """Return a CSV template for manually curated enzyme key residues."""

    return pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 195,
                "resname": "SER",
                "evidence_source": "manual",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.95,
                "evidence_note": "Example catalytic residue; replace with your curated note.",
                "uniprot_resid": 195,
                "mapping_level": "exact",
                "mapping_confidence": 0.95,
                "pmid": "",
                "doi": "",
                "evidence_snippet": "Paste the sentence or curator note supporting this residue.",
                "requires_manual_review": False,
            }
        ],
        columns=MANUAL_KEY_RESIDUE_TEMPLATE_COLUMNS,
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "review", "needs-review", "manual-review"}


def ensure_evidence_columns(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return a copy with the full residue-evidence schema.

    Several evidence sources are intentionally sparse. Keeping the schema
    compatible here prevents newer source-detail fields from breaking older
    UniProt, M-CSA, conservation or manually supplied evidence tables.
    """

    if table is None or getattr(table, "empty", True):
        return _empty_evidence_df()

    working = table.copy()
    for column in EVIDENCE_COLUMNS:
        if column in working.columns:
            continue
        if column == "uniprot_resid" and "resid" in working.columns:
            working[column] = working["resid"]
        else:
            working[column] = EVIDENCE_COLUMN_DEFAULTS.get(column, "")

    working["requires_manual_review"] = working["requires_manual_review"].map(_coerce_bool)
    for column in ["article_title", "pmid", "pmcid", "doi", "evidence_snippet", "sentence_index", "extraction_pattern"]:
        working[column] = working[column].fillna("").astype(str).str.strip()
    return working[EVIDENCE_COLUMNS].copy()


def _source_detail_from_row(row: object) -> dict[str, object]:
    return {
        column: getattr(row, column, EVIDENCE_COLUMN_DEFAULTS.get(column, ""))
        for column in EVIDENCE_SOURCE_DETAIL_COLUMNS
    }


def _as_int(value: object) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return float(default)
        return float(text)
    except Exception:
        return float(default)


def _read_loose_residue_table(text: str) -> pd.DataFrame:
    payload = str(text or "").strip()
    if not payload:
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(payload), sep=None, engine="python", comment="#")
    except Exception:
        return pd.read_csv(StringIO(payload), comment="#")


def parse_manual_key_residue_table(
    text: str | bytes | None,
    *,
    source_hint: str = "manual",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse manually supplied catalytic/key residues into external evidence rows.

    The parser intentionally accepts both the exported template and common
    minimal tables such as `chain,resid,evidence_note`.
    """

    raw_text = text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else str(text or "")
    if not raw_text.strip():
        return _empty_evidence_df(), {"status": "empty", "manual_key_residue_rows": "0"}

    raw = _read_loose_residue_table(raw_text)
    if raw.empty:
        return _empty_evidence_df(), {"status": "empty", "manual_key_residue_rows": "0"}

    normalized = raw.copy()
    normalized.columns = [
        MANUAL_KEY_RESIDUE_COLUMN_ALIASES.get(str(column).strip().lower().replace(" ", "_"), str(column).strip())
        for column in normalized.columns
    ]
    if "resid" not in normalized.columns:
        return _empty_evidence_df(), {"status": "missing-resid", "manual_key_residue_rows": "0"}

    normalized["resid"] = pd.to_numeric(normalized["resid"], errors="coerce")
    normalized = normalized[normalized["resid"].notna()].copy()
    if normalized.empty:
        return _empty_evidence_df(), {"status": "empty", "manual_key_residue_rows": "0"}

    rows: list[dict[str, object]] = []
    cleaned_source_hint = str(source_hint or "manual").strip() or "manual"
    for row in normalized.to_dict(orient="records"):
        resid = _as_int(row.get("resid"))
        if resid is None:
            continue
        mapping_level = str(row.get("mapping_level") or "exact").strip().lower()
        if mapping_level not in {"exact", "weak"}:
            mapping_level = "exact"
        mapping_confidence_default = 0.95 if mapping_level == "exact" else 0.35
        evidence_score = _as_float(row.get("evidence_score"), 0.95)
        uniprot_resid = _as_int(row.get("uniprot_resid"))
        if uniprot_resid is None:
            uniprot_resid = int(resid)
        evidence_note = str(row.get("evidence_note") or "").strip()
        resname = str(row.get("resname") or "").strip()
        if resname and "resname=" not in evidence_note:
            evidence_note = f"{evidence_note} resname={resname}".strip()
        rows.append(
            {
                "chain": str(row.get("chain") or "").strip(),
                "resid": int(resid),
                "evidence_source": str(row.get("evidence_source") or cleaned_source_hint).strip() or cleaned_source_hint,
                "evidence_type": str(row.get("evidence_type") or "Manual key residue").strip() or "Manual key residue",
                "evidence_score": max(0.0, min(1.0, evidence_score)),
                "evidence_note": evidence_note,
                "uniprot_resid": int(uniprot_resid),
                "mapping_level": mapping_level,
                "mapping_confidence": max(0.0, min(1.0, _as_float(row.get("mapping_confidence"), mapping_confidence_default))),
                "mapping_method": str(row.get("mapping_method") or "manual-structure-numbering").strip(),
                "article_title": str(row.get("article_title") or "").strip(),
                "pmid": str(row.get("pmid") or "").strip(),
                "pmcid": str(row.get("pmcid") or "").strip(),
                "doi": str(row.get("doi") or "").strip(),
                "evidence_snippet": str(row.get("evidence_snippet") or "").strip(),
                "sentence_index": str(row.get("sentence_index") or "").strip(),
                "extraction_pattern": str(row.get("extraction_pattern") or "manual").strip(),
                "requires_manual_review": _coerce_bool(row.get("requires_manual_review")),
            }
        )

    evidence_df = ensure_evidence_columns(pd.DataFrame(rows))
    sources = ",".join(
        dict.fromkeys(
            source
            for source in evidence_df["evidence_source"].astype(str).str.strip().tolist()
            if source
        )
    )
    return evidence_df, {
        "status": "ok" if not evidence_df.empty else "empty",
        "manual_key_residue_rows": str(len(evidence_df)),
        "sources": sources or cleaned_source_hint,
    }


def _first_text(payload: dict, *keys: str) -> str:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return ""


def _list_from_payload(payload: object, *keys: str) -> list:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _feature_positions(feature: dict, *, max_span: int = 20) -> List[int]:
    location = feature.get("location") or {}

    single_position = _as_int((location.get("position") or {}).get("value"))
    if single_position is not None:
        return [single_position]

    start_pos = _as_int((location.get("start") or {}).get("value"))
    end_pos = _as_int((location.get("end") or {}).get("value"))
    if start_pos is None and end_pos is None:
        return []
    if start_pos is None:
        start_pos = end_pos
    if end_pos is None:
        end_pos = start_pos
    if start_pos is None or end_pos is None:
        return []

    start_pos, end_pos = sorted((int(start_pos), int(end_pos)))
    span = end_pos - start_pos + 1
    if span <= 0:
        return []
    if span > int(max_span):
        return [start_pos, end_pos]
    return list(range(start_pos, end_pos + 1))


def _fetch_json(url: str, *, timeout_sec: float = 10.0) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=float(timeout_sec)) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def extract_pdb_id_from_text(pdb_text: str) -> Optional[str]:
    if not str(pdb_text or "").strip():
        return None

    lines = str(pdb_text).splitlines()
    for line in lines[:40]:
        if line.startswith("HEADER") and len(line) >= 66:
            candidate = line[62:66].strip().upper()
            if re.fullmatch(r"[0-9][A-Z0-9]{3}", candidate):
                return candidate

    pattern = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")
    for line in lines[:40]:
        matched = pattern.search(line)
        if matched:
            return matched.group(1).upper()
    return None


def _extract_structure_residue_map(pdb_text: Optional[str]) -> dict[str, dict[str, object]]:
    if not str(pdb_text or "").strip():
        return {}

    residue_map: dict[str, dict[str, object]] = {}
    seen_keys: set[tuple[str, int, str]] = set()
    for order_index, line in enumerate(str(pdb_text).splitlines()):
        if not line.startswith("ATOM"):
            continue
        try:
            chain = str(line[21].strip() or "A")
            resid = int(line[22:26].strip())
            insertion_code = str(line[26].strip() if len(line) > 26 else "").strip()
            resname = str(line[17:20].strip())
        except (ValueError, IndexError):
            continue

        residue_key = (chain, int(resid), insertion_code)
        if residue_key in seen_keys:
            continue
        seen_keys.add(residue_key)

        chain_bucket = residue_map.setdefault(chain, {"entries": [], "resids": set()})
        chain_bucket["entries"].append(
            {
                "chain": chain,
                "resid": int(resid),
                "insertion_code": insertion_code,
                "resname": resname,
                "order_index": int(order_index),
            }
        )
        chain_bucket["resids"].add(int(resid))

    return residue_map


def _block_pdb_bounds(block: dict) -> tuple[Optional[int], Optional[int]]:
    start_node = block.get("start") or {}
    end_node = block.get("end") or {}

    pdb_start = _as_int(start_node.get("author_residue_number"))
    if pdb_start is None:
        pdb_start = _as_int(start_node.get("residue_number"))
    pdb_end = _as_int(end_node.get("author_residue_number"))
    if pdb_end is None:
        pdb_end = _as_int(end_node.get("residue_number"))
    if pdb_start is None or pdb_end is None:
        return None, None
    return (min(int(pdb_start), int(pdb_end)), max(int(pdb_start), int(pdb_end)))


def _expected_block_residue_count(block: dict) -> int:
    unp_start = _as_int(block.get("unp_start"))
    unp_end = _as_int(block.get("unp_end"))
    if unp_start is None or unp_end is None:
        return 0
    start_value = min(int(unp_start), int(unp_end))
    end_value = max(int(unp_start), int(unp_end))
    return max(0, end_value - start_value + 1)


def _block_structure_entries(chain_entries: list[dict[str, object]], block: dict) -> list[dict[str, object]]:
    if not chain_entries:
        return []
    pdb_start, pdb_end = _block_pdb_bounds(block)
    if pdb_start is None or pdb_end is None:
        return []
    return [
        entry
        for entry in chain_entries
        if int(entry.get("resid", 0) or 0) >= int(pdb_start) and int(entry.get("resid", 0) or 0) <= int(pdb_end)
    ]


def merge_external_evidence_tables(*tables: Optional[pd.DataFrame]) -> pd.DataFrame:
    normalized_tables: list[pd.DataFrame] = []
    for table in tables:
        if table is None or getattr(table, "empty", True):
            continue
        working = table.copy()
        if "resid" not in working.columns:
            continue
        working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
        working = working[working["resid"].notna()].copy()
        if working.empty:
            continue
        working["resid"] = working["resid"].astype(int)
        if "chain" not in working.columns:
            working["chain"] = ""
        working["chain"] = working["chain"].astype(str).str.strip()
        if "mapping_level" not in working.columns:
            working["mapping_level"] = "weak"
        working["mapping_level"] = working["mapping_level"].astype(str).str.strip().str.lower()
        working.loc[~working["mapping_level"].isin({"exact", "weak"}), "mapping_level"] = "weak"
        if "mapping_confidence" not in working.columns:
            working["mapping_confidence"] = working["mapping_level"].map({"exact": 0.9, "weak": 0.3}).fillna(0.3)
        working["mapping_confidence"] = pd.to_numeric(working["mapping_confidence"], errors="coerce").fillna(0.3).clip(0.0, 1.0)
        if "evidence_score" not in working.columns:
            working["evidence_score"] = 0.75
        working["evidence_score"] = pd.to_numeric(working["evidence_score"], errors="coerce").fillna(0.75).clip(0.0, 1.0)
        if "evidence_source" not in working.columns:
            working["evidence_source"] = "external"
        if "evidence_type" not in working.columns:
            working["evidence_type"] = "Functional site"
        if "evidence_note" not in working.columns:
            working["evidence_note"] = ""
        if "uniprot_resid" not in working.columns:
            working["uniprot_resid"] = working["resid"]
        if "mapping_method" not in working.columns:
            working["mapping_method"] = "unknown"
        normalized_tables.append(ensure_evidence_columns(working))

    if not normalized_tables:
        return _empty_evidence_df()

    combined = pd.concat(normalized_tables, ignore_index=True)
    combined["_mapping_rank"] = combined["mapping_level"].map({"exact": 0, "weak": 1}).fillna(2)
    combined = combined.sort_values(
        ["_mapping_rank", "mapping_confidence", "evidence_score", "resid", "evidence_source", "evidence_type"],
        ascending=[True, False, False, True, True, True],
    ).drop_duplicates(
        subset=["chain", "resid", "evidence_source", "evidence_type", "mapping_level", "uniprot_resid", "evidence_note"],
        keep="first",
    )
    return ensure_evidence_columns(combined.drop(columns="_mapping_rank").reset_index(drop=True))


def _fetch_pdbe_uniprot_mappings(
    pdb_id: str,
    accession: str,
    *,
    timeout_sec: float = 10.0,
) -> list[dict]:
    cleaned_pdb = str(pdb_id or "").strip().lower()
    cleaned_accession = str(accession or "").strip().upper()
    if not cleaned_pdb or not cleaned_accession:
        return []

    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{cleaned_pdb}"
    payload = _fetch_json(url, timeout_sec=timeout_sec)

    root = payload.get(cleaned_pdb) or payload.get(cleaned_pdb.upper()) or {}
    uniprot_node = root.get("UniProt") or root.get("uniprot") or {}
    accession_entry = uniprot_node.get(cleaned_accession)
    if accession_entry is None:
        for key, value in uniprot_node.items():
            if str(key).strip().upper() == cleaned_accession:
                accession_entry = value
                break

    if not isinstance(accession_entry, dict):
        return []
    mappings = accession_entry.get("mappings") or []
    return [mapping for mapping in mappings if isinstance(mapping, dict)]


def _map_residue_with_block(
    uniprot_resid: int,
    block: dict,
    *,
    chain_entries: Optional[list[dict[str, object]]] = None,
) -> tuple[Optional[int], float, str]:
    unp_start = _as_int(block.get("unp_start"))
    unp_end = _as_int(block.get("unp_end"))
    pdb_start, pdb_end = _block_pdb_bounds(block)

    if None in (unp_start, unp_end, pdb_start, pdb_end):
        return None, 0.0, "sifts-block-missing"

    if not (int(unp_start) <= int(uniprot_resid) <= int(unp_end)):
        return None, 0.0, "sifts-out-of-range"

    unp_span = int(unp_end) - int(unp_start)
    pdb_span = int(pdb_end) - int(pdb_start)
    offset = int(uniprot_resid) - int(unp_start)

    mapped_resid: Optional[int] = None
    mapped_confidence = 0.0
    mapped_method = "sifts-linear-anchor"
    if unp_span == pdb_span:
        mapped_resid = int(pdb_start) + offset
        mapped_confidence = 0.95
        mapped_method = "sifts-linear-exact"
    elif unp_span > 0 and pdb_span > 0:
        mapped_resid = int(round(int(pdb_start) + float(offset) * float(pdb_span) / float(unp_span)))
        mapped_confidence = 0.68
        mapped_method = "sifts-linear-interpolated"
    else:
        mapped_resid = int(pdb_start)
        mapped_confidence = 0.52
        mapped_method = "sifts-linear-anchor"

    if mapped_resid is None or not chain_entries:
        return mapped_resid, mapped_confidence, mapped_method

    observed_resids = {
        int(entry.get("resid", 0) or 0)
        for entry in chain_entries
    }
    if mapped_resid in observed_resids:
        return mapped_resid, min(0.99, mapped_confidence + 0.02), f"{mapped_method}-verified"

    expected_count = _expected_block_residue_count(block)
    block_entries = _block_structure_entries(chain_entries, block)
    if block_entries and expected_count > 0 and 0 <= int(offset) < int(expected_count):
        if len(block_entries) == int(expected_count):
            aligned_resid = int(block_entries[int(offset)].get("resid", mapped_resid) or mapped_resid)
            return aligned_resid, max(mapped_confidence, 0.90), "sifts-structure-order"

        if len(block_entries) > int(expected_count):
            if int(expected_count) == 1:
                aligned_index = 0
            else:
                aligned_index = int(
                    round(float(offset) * float(len(block_entries) - 1) / float(int(expected_count) - 1))
                )
            aligned_index = max(0, min(aligned_index, len(block_entries) - 1))
            aligned_resid = int(block_entries[aligned_index].get("resid", mapped_resid) or mapped_resid)
            return aligned_resid, max(mapped_confidence, 0.76), "sifts-structure-interpolated"

    return mapped_resid, min(mapped_confidence, 0.58), "sifts-gap-fallback"


def _map_uniprot_sites_to_structure(
    evidence_df: pd.DataFrame,
    *,
    accession: str,
    pdb_id: str,
    chain_hint: Optional[str] = None,
    pdb_text: Optional[str] = None,
    timeout_sec: float = 10.0,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    if evidence_df is None or evidence_df.empty:
        return _empty_evidence_df(), {
            "pdb_id": str(pdb_id or "").upper(),
            "mapping_status": "no-evidence",
            "mapped_rows": "0",
        }

    cleaned_chain_hint = str(chain_hint or "").strip()
    try:
        mapping_blocks = _fetch_pdbe_uniprot_mappings(
            pdb_id,
            accession,
            timeout_sec=timeout_sec,
        )
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        mapping_blocks = []

    if not mapping_blocks:
        fallback = evidence_df.copy()
        fallback["mapping_level"] = "weak"
        fallback["mapping_confidence"] = 0.30 if cleaned_chain_hint else 0.20
        fallback["mapping_method"] = "uniprot-resid-fallback"
        fallback["evidence_source"] = "UniProt"
        if cleaned_chain_hint:
            fallback["chain"] = cleaned_chain_hint
        return ensure_evidence_columns(fallback), {
            "pdb_id": str(pdb_id or "").upper(),
            "mapping_status": "unavailable",
            "mapped_rows": "0",
        }

    structure_residue_map = _extract_structure_residue_map(pdb_text)

    normalized_blocks: list[dict] = []
    for block in mapping_blocks:
        chain_id = str(block.get("chain_id") or block.get("auth_asym_id") or block.get("struct_asym_id") or "").strip()
        normalized_blocks.append(
            {
                "chain": chain_id,
                "block": block,
            }
        )

    preferred_blocks = normalized_blocks
    if cleaned_chain_hint:
        matched_blocks = [item for item in normalized_blocks if str(item["chain"]).strip() == cleaned_chain_hint]
        if matched_blocks:
            preferred_blocks = matched_blocks

    rows: list[dict] = []
    exact_count = 0
    weak_count = 0
    structure_verified_count = 0
    for row in evidence_df.itertuples(index=False):
        uniprot_resid = _as_int(getattr(row, "uniprot_resid", None))
        if uniprot_resid is None:
            uniprot_resid = _as_int(getattr(row, "resid", None))
        if uniprot_resid is None:
            continue

        best_resid = None
        best_chain = ""
        best_confidence = 0.0
        best_chain_penalty = 1.0
        best_mapping_method = "sifts-unmapped-fallback"

        for item in preferred_blocks:
            chain_value = str(item["chain"]).strip()
            chain_entries = []
            if chain_value and chain_value in structure_residue_map:
                chain_entries = list(structure_residue_map[chain_value].get("entries") or [])
            elif cleaned_chain_hint and cleaned_chain_hint in structure_residue_map:
                chain_entries = list(structure_residue_map[cleaned_chain_hint].get("entries") or [])

            mapped_resid, mapped_confidence, mapped_method = _map_residue_with_block(
                int(uniprot_resid),
                item["block"],
                chain_entries=chain_entries,
            )
            if mapped_resid is None:
                continue
            chain_penalty = 1.0
            if cleaned_chain_hint and chain_value and chain_value != cleaned_chain_hint:
                chain_penalty = 0.75
            if structure_residue_map:
                if chain_value and chain_value in structure_residue_map:
                    chain_penalty *= 1.0
                elif chain_value:
                    chain_penalty *= 0.70

            effective_confidence = float(mapped_confidence) * chain_penalty
            if effective_confidence > best_confidence:
                best_resid = int(mapped_resid)
                best_chain = chain_value or cleaned_chain_hint
                best_confidence = effective_confidence
                best_chain_penalty = chain_penalty
                best_mapping_method = mapped_method

        if best_resid is None:
            weak_count += 1
            rows.append(
                {
                    "chain": cleaned_chain_hint,
                    "resid": int(uniprot_resid),
                    "evidence_source": "UniProt",
                    "evidence_type": str(getattr(row, "evidence_type", "") or "").strip(),
                    "evidence_score": float(getattr(row, "evidence_score", 0.75) or 0.75),
                    "evidence_note": str(getattr(row, "evidence_note", "") or "").strip(),
                    "uniprot_resid": int(uniprot_resid),
                    "mapping_level": "weak",
                    "mapping_confidence": 0.28,
                    "mapping_method": "sifts-unmapped-fallback",
                    **_source_detail_from_row(row),
                }
            )
            continue

        mapping_level = "exact"
        if best_chain_penalty < 0.99 or best_confidence < 0.72 or "gap-fallback" in best_mapping_method:
            mapping_level = "weak"
        if mapping_level == "exact":
            exact_count += 1
        else:
            weak_count += 1
        if "structure" in best_mapping_method or "verified" in best_mapping_method:
            structure_verified_count += 1
        score_scale = 0.92 + 0.08 * max(0.0, min(1.0, best_confidence))
        rows.append(
            {
                "chain": best_chain,
                "resid": int(best_resid),
                "evidence_source": "UniProt+SIFTS",
                "evidence_type": str(getattr(row, "evidence_type", "") or "").strip(),
                "evidence_score": round(float(getattr(row, "evidence_score", 0.75) or 0.75) * score_scale, 3),
                "evidence_note": str(getattr(row, "evidence_note", "") or "").strip(),
                "uniprot_resid": int(uniprot_resid),
                "mapping_level": mapping_level,
                "mapping_confidence": round(float(best_confidence), 3),
                "mapping_method": best_mapping_method,
                **_source_detail_from_row(row),
            }
        )

    if not rows:
        return _empty_evidence_df(), {
            "pdb_id": str(pdb_id or "").upper(),
            "mapping_status": "empty",
            "mapped_rows": "0",
        }

    mapped_df = pd.DataFrame(rows)
    mapped_df = mapped_df.drop_duplicates(
        subset=["chain", "resid", "evidence_type", "mapping_level", "uniprot_resid"],
        keep="first",
    )
    mapped_df = mapped_df.sort_values(
        ["mapping_level", "mapping_confidence", "resid", "evidence_score", "evidence_type"],
        ascending=[True, False, True, False, True],
    ).reset_index(drop=True)
    mapped_df = ensure_evidence_columns(mapped_df)

    metadata = {
        "pdb_id": str(pdb_id or "").upper(),
        "mapping_status": "ok",
        "mapped_rows": str(len(mapped_df)),
        "exact_rows": str(exact_count),
        "weak_rows": str(weak_count),
        "structure_verified_rows": str(structure_verified_count),
    }
    return mapped_df, metadata


def _mcsa_role_score(role_text: str) -> float:
    normalized = str(role_text or "").strip().lower()
    if not normalized:
        return 0.95
    for role, score in MCSA_ROLE_SCORES.items():
        if role in normalized:
            return float(score)
    return 0.95


def _mcsa_residue_positions(payload: dict) -> List[int]:
    if not isinstance(payload, dict):
        return []

    positions: list[int] = []
    for key in ("residue_numbers", "positions", "seq_nums"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                parsed = _as_int(item)
                if parsed is not None:
                    positions.append(int(parsed))

    for key in ("residue_number", "residue", "position", "seq_num", "sequence_number", "uniprot_resid"):
        parsed = _as_int(payload.get(key))
        if parsed is not None:
            positions.append(int(parsed))

    if positions:
        return sorted(dict.fromkeys(positions))

    location = payload.get("location") or {}
    if isinstance(location, dict):
        return _feature_positions({"location": location}, max_span=10)
    return []


def _extract_mcsa_entries(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("results", "data", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if all(isinstance(value, dict) for value in payload.values()):
        return [value for value in payload.values() if isinstance(value, dict)]
    return []


def fetch_mcsa_catalytic_sites(
    accession: str = "",
    *,
    ec_number: str = "",
    chain_hint: Optional[str] = None,
    timeout_sec: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cleaned_accession = str(accession or "").strip().upper()
    cleaned_ec = str(ec_number or "").strip()
    if not cleaned_accession and not cleaned_ec:
        return _empty_evidence_df(), {}

    params = {"format": "json"}
    if cleaned_accession:
        params["uniprot_ac"] = cleaned_accession
    if cleaned_ec:
        params["ec"] = cleaned_ec
    url = f"https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?{urlencode(params)}"

    try:
        payload = _fetch_json(url, timeout_sec=timeout_sec)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return _empty_evidence_df(), {
            "accession": cleaned_accession,
            "ec_number": cleaned_ec,
            "status": "unavailable",
        }

    rows: list[dict] = []
    entries = _extract_mcsa_entries(payload)
    normalized_chain = str(chain_hint or "").strip()
    for entry in entries:
        entry_accession = _first_text(entry, "uniprot_ac", "uniprot_accession", "uniprot", "accession").upper()
        entry_ec = _first_text(entry, "ec", "ec_number", "enzyme_ec")
        if cleaned_accession and entry_accession and entry_accession != cleaned_accession:
            continue
        if cleaned_ec and entry_ec and entry_ec != cleaned_ec:
            continue

        mcsa_id = _first_text(entry, "mcsa_id", "id", "entry_id")
        mechanism_name = _first_text(entry, "enzyme_name", "name", "mechanism_name", "reaction")
        residue_entries = _list_from_payload(
            entry,
            "residues",
            "catalytic_residues",
            "active_site_residues",
            "functional_residues",
        )

        for residue_entry in residue_entries:
            positions = _mcsa_residue_positions(residue_entry)
            if not positions:
                continue

            role_text = _first_text(
                residue_entry,
                "chemical_function",
                "role",
                "residue_function",
                "type",
                "function",
            )
            if not role_text:
                role_text = "Catalytic residue"
            evidence_type = "Catalytic residue"
            evidence_score = _mcsa_role_score(role_text)
            note_parts = [part for part in ["M-CSA", mcsa_id, mechanism_name, role_text] if part]
            note_text = " | ".join(note_parts)

            for resid in positions:
                rows.append(
                    {
                        "chain": normalized_chain,
                        "resid": int(resid),
                        "evidence_source": "M-CSA",
                        "evidence_type": evidence_type,
                        "evidence_score": round(float(evidence_score), 3),
                        "evidence_note": note_text,
                        "uniprot_resid": int(resid),
                        "mapping_level": "weak",
                        "mapping_confidence": 0.42 if normalized_chain else 0.32,
                        "mapping_method": "mcsa-direct",
                    }
                )

    if not rows:
        return _empty_evidence_df(), {
            "accession": cleaned_accession,
            "ec_number": cleaned_ec,
            "status": "ok",
            "evidence_rows": "0",
        }

    evidence_df = pd.DataFrame(rows)
    evidence_df = evidence_df.drop_duplicates(
        subset=["chain", "resid", "evidence_type", "evidence_note", "uniprot_resid"]
    ).sort_values(
        ["resid", "evidence_score", "mapping_confidence"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    evidence_df = ensure_evidence_columns(evidence_df)
    return evidence_df, {
        "accession": cleaned_accession,
        "ec_number": cleaned_ec,
        "status": "ok",
        "evidence_rows": str(len(evidence_df)),
    }


def fetch_mcsa_catalytic_sites_for_structure(
    accession: str = "",
    *,
    ec_number: str = "",
    chain_hint: Optional[str] = None,
    pdb_id: Optional[str] = None,
    pdb_text: Optional[str] = None,
    timeout_sec: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    evidence_df, metadata = fetch_mcsa_catalytic_sites(
        accession,
        ec_number=ec_number,
        chain_hint=chain_hint,
        timeout_sec=timeout_sec,
    )
    if evidence_df.empty:
        return evidence_df, metadata

    cleaned_accession = str(accession or "").strip().upper()
    resolved_pdb_id = str(pdb_id or "").strip().upper()
    if not resolved_pdb_id:
        resolved_pdb_id = str(extract_pdb_id_from_text(str(pdb_text or "")) or "").strip().upper()

    if not cleaned_accession or not resolved_pdb_id:
        metadata = {
            **metadata,
            "pdb_id": resolved_pdb_id,
            "mapping_status": "limited",
            "mapped_rows": "0",
            "exact_rows": "0",
            "weak_rows": str(len(evidence_df)),
        }
        return evidence_df, metadata

    mapped_df, mapping_meta = _map_uniprot_sites_to_structure(
        evidence_df,
        accession=cleaned_accession,
        pdb_id=resolved_pdb_id,
        chain_hint=chain_hint,
        pdb_text=pdb_text,
        timeout_sec=timeout_sec,
    )
    metadata = {**metadata, **mapping_meta}
    metadata["pdb_id"] = resolved_pdb_id
    metadata["evidence_rows"] = str(len(mapped_df))
    return mapped_df, metadata


def fetch_combined_functional_sites_for_structure(
    accession: str = "",
    *,
    ec_number: str = "",
    chain_hint: Optional[str] = None,
    pdb_id: Optional[str] = None,
    pdb_text: Optional[str] = None,
    enable_uniprot: bool = True,
    enable_mcsa: bool = True,
    timeout_sec: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    uniprot_df = _empty_evidence_df()
    mcsa_df = _empty_evidence_df()
    uniprot_meta: Dict[str, str] = {}
    mcsa_meta: Dict[str, str] = {}

    if enable_uniprot and str(accession or "").strip():
        uniprot_df, uniprot_meta = fetch_uniprot_functional_sites_for_structure(
            accession,
            chain_hint=chain_hint,
            pdb_id=pdb_id,
            pdb_text=pdb_text,
            timeout_sec=timeout_sec,
        )

    if enable_mcsa and (str(accession or "").strip() or str(ec_number or "").strip()):
        mcsa_df, mcsa_meta = fetch_mcsa_catalytic_sites_for_structure(
            accession,
            ec_number=ec_number,
            chain_hint=chain_hint,
            pdb_id=pdb_id,
            pdb_text=pdb_text,
            timeout_sec=timeout_sec,
        )

    combined = merge_external_evidence_tables(uniprot_df, mcsa_df)
    exact_rows = 0
    weak_rows = 0
    if not combined.empty and "mapping_level" in combined.columns:
        level_series = combined["mapping_level"].astype(str).str.lower()
        exact_rows = int((level_series == "exact").sum())
        weak_rows = int((level_series == "weak").sum())

    metadata: Dict[str, object] = {
        "status": "ok" if not combined.empty else "empty",
        "accession": str(accession or "").strip().upper(),
        "ec_number": str(ec_number or "").strip(),
        "pdb_id": str(pdb_id or extract_pdb_id_from_text(str(pdb_text or "")) or "").strip().upper(),
        "evidence_rows": str(len(combined)),
        "exact_rows": str(exact_rows),
        "weak_rows": str(weak_rows),
        "sources": ",".join(
            source
            for source, enabled in (("uniprot", enable_uniprot), ("mcsa", enable_mcsa))
            if enabled
        ),
        "uniprot": uniprot_meta,
        "mcsa": mcsa_meta,
    }
    if not combined.empty:
        metadata["status"] = "ok"
    elif (enable_uniprot and uniprot_meta.get("status") == "unavailable") or (enable_mcsa and mcsa_meta.get("status") == "unavailable"):
        metadata["status"] = "unavailable"
    return combined, metadata


def fetch_uniprot_functional_sites(
    accession: str,
    *,
    chain_hint: Optional[str] = None,
    timeout_sec: float = 10.0,
    max_span: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Fetch functional residue hints from UniProt feature annotations.

    The returned table is residue-level evidence suitable for joint pocket
    reranking. It does not change the pocket count, only contributes evidence.
    """

    cleaned = str(accession or "").strip().upper()
    if not cleaned:
        return _empty_evidence_df(), {}

    url = f"https://rest.uniprot.org/uniprotkb/{cleaned}.json"
    try:
        payload = _fetch_json(url, timeout_sec=timeout_sec)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return _empty_evidence_df(), {
            "accession": cleaned,
            "status": "unavailable",
        }

    features = payload.get("features") or []
    rows: List[dict] = []

    normalized_chain = str(chain_hint or "").strip()
    for feature in features:
        feature_type = str(feature.get("type") or "").strip()
        if feature_type not in UNIPROT_FEATURE_SCORES:
            continue

        positions = _feature_positions(feature, max_span=max_span)
        if not positions:
            continue

        description = str(feature.get("description") or "").strip()
        evidence_score = float(UNIPROT_FEATURE_SCORES.get(feature_type, 0.75))
        feature_id = str(feature.get("featureId") or feature.get("id") or "").strip()
        note_parts = [part for part in [feature_type, description, feature_id] if part]
        note_text = " | ".join(note_parts)

        for resid in positions:
            rows.append(
                {
                    "chain": normalized_chain,
                    "resid": int(resid),
                    "evidence_source": "UniProt",
                    "evidence_type": feature_type,
                    "evidence_score": round(evidence_score, 3),
                    "evidence_note": note_text,
                    "uniprot_resid": int(resid),
                    "mapping_level": "weak",
                    "mapping_confidence": 0.35 if normalized_chain else 0.20,
                    "mapping_method": "uniprot-direct",
                }
            )

    if not rows:
        return _empty_evidence_df(), {
            "accession": cleaned,
            "entry": str(payload.get("uniProtkbId") or "").strip(),
            "status": "ok",
            "evidence_rows": "0",
        }

    evidence_df = pd.DataFrame(rows).drop_duplicates(
        subset=["chain", "resid", "evidence_type", "evidence_note", "mapping_level"]
    )
    evidence_df = evidence_df.sort_values(
        ["resid", "evidence_score", "evidence_type"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    evidence_df = ensure_evidence_columns(evidence_df)

    metadata = {
        "accession": cleaned,
        "entry": str(payload.get("uniProtkbId") or "").strip(),
        "status": "ok",
        "evidence_rows": str(len(evidence_df)),
    }
    return evidence_df, metadata


def fetch_uniprot_functional_sites_for_structure(
    accession: str,
    *,
    chain_hint: Optional[str] = None,
    pdb_id: Optional[str] = None,
    pdb_text: Optional[str] = None,
    timeout_sec: float = 10.0,
    max_span: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    evidence_df, metadata = fetch_uniprot_functional_sites(
        accession,
        chain_hint=chain_hint,
        timeout_sec=timeout_sec,
        max_span=max_span,
    )
    if evidence_df.empty:
        return evidence_df, metadata

    resolved_pdb_id = str(pdb_id or "").strip().upper()
    if not resolved_pdb_id:
        resolved_pdb_id = str(extract_pdb_id_from_text(str(pdb_text or "")) or "").strip().upper()

    if not resolved_pdb_id:
        metadata = {
            **metadata,
            "mapping_status": "pdb-id-unavailable",
            "mapped_rows": "0",
            "exact_rows": "0",
            "weak_rows": str(len(evidence_df)),
        }
        return evidence_df, metadata

    mapped_df, mapping_meta = _map_uniprot_sites_to_structure(
        evidence_df,
        accession=str(accession or "").strip().upper(),
        pdb_id=resolved_pdb_id,
        chain_hint=chain_hint,
        pdb_text=pdb_text,
        timeout_sec=timeout_sec,
    )
    metadata = {**metadata, **mapping_meta}
    metadata["pdb_id"] = resolved_pdb_id
    metadata["evidence_rows"] = str(len(mapped_df))
    return mapped_df, metadata
