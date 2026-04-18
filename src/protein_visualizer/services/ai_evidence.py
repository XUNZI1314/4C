from __future__ import annotations

import json
import hashlib
import os
import re
import zipfile
from io import BytesIO, StringIO
from typing import Any, Optional
from urllib import error, request
from urllib.parse import quote_plus

import pandas as pd

from protein_visualizer.services.external_sites import EVIDENCE_COLUMNS, _extract_structure_residue_map, ensure_evidence_columns


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA1_TO_3 = {value: key for key, value in AA3_TO_1.items()}

AI_EVIDENCE_SCHEMA_HINT = {
    "residues": [
        {
            "resname": "SER",
            "position_text": "Ser195",
            "uniprot_position": 195,
            "structure_position": None,
            "chain": "",
            "evidence_type": "Catalytic residue",
            "confidence": 0.86,
            "pmid": "12345678",
            "doi": "",
            "source_title": "Article title",
            "evidence_snippet": "Short source sentence that supports this residue.",
            "requires_manual_review": False,
        }
    ]
}

AI_EVIDENCE_AUDIT_COLUMNS = [
    "chain",
    "resid",
    "evidence_type",
    "ai_score",
    "mapping_level",
    "mapping_confidence",
    "audit_status",
    "overlap_sources",
    "risk_flags",
    "audit_reason",
    "recommended_action",
]

AI_FOLLOWUP_PLAN_COLUMNS = [
    "pocket_id",
    "decision_rank",
    "precision_tier",
    "followup_priority",
    "evidence_gap",
    "search_query",
    "pubmed_url",
    "europepmc_url",
    "uniprot_url",
    "rcsb_url",
    "ai_task_prompt",
    "acceptance_criteria",
    "why_this_matters",
]

AI_RANKING_IMPACT_COLUMNS = [
    "ai_input_rows",
    "ai_ranked_rows",
    "ai_excluded_rows",
    "audit_supported_rows",
    "audit_structure_verified_rows",
    "audit_review_rows",
    "audit_conflicting_rows",
    "top_pocket_id",
    "top_pocket_precision_tier",
    "top_pocket_has_ai_support",
    "top_pocket_ai_residue_count",
    "top_pocket_ai_residues",
    "ai_influence_level",
    "ai_influence_reason",
    "recommended_action",
]

AI_REVIEW_QUEUE_COLUMNS = [
    "review_priority",
    "chain",
    "resid",
    "evidence_type",
    "audit_status",
    "fix_type",
    "problem",
    "required_evidence",
    "can_affect_ranking_after_fix",
    "suggested_next_action",
]

AI_REVIEW_DECISION_COLUMNS = [
    "chain",
    "resid",
    "evidence_type",
    "review_decision",
    "reviewer",
    "review_note",
    "verified_source",
    "verified_snippet",
]

AI_REVIEW_DECISION_TEMPLATE_COLUMNS = [
    *AI_REVIEW_DECISION_COLUMNS,
    "current_audit_status",
    "review_priority",
    "fix_type",
    "problem",
    "required_evidence",
    "suggested_next_action",
    "can_affect_ranking_after_fix",
]

AI_REVIEW_DECISION_OUTCOME_COLUMNS = [
    "chain",
    "resid",
    "evidence_type",
    "review_decision",
    "applied_status",
    "current_audit_status",
    "risk_flags",
    "outcome_reason",
    "next_action",
    "reviewer",
    "verified_source",
    "verified_snippet",
]

AI_REVIEW_DECISION_VALIDATION_COLUMNS = [
    "row_index",
    "chain",
    "resid",
    "evidence_type",
    "review_decision",
    "validation_status",
    "issue_flags",
    "can_apply",
    "matched_audit_status",
    "validation_reason",
    "required_fix",
]

AI_REVIEW_ROUND_SUMMARY_COLUMNS = [
    "decision_rows",
    "validation_ok_rows",
    "validation_warning_rows",
    "validation_blocked_rows",
    "outcome_accepted_rows",
    "outcome_rejected_rows",
    "outcome_review_pending_rows",
    "outcome_missing_source_rows",
    "outcome_conflict_blocked_rows",
    "outcome_unmatched_rows",
    "rankable_after_review_rows",
    "review_round_status",
    "review_round_reason",
    "recommended_action",
]

AI_REVIEW_RANKING_DELTA_COLUMNS = [
    "before_rankable_rows",
    "after_rankable_rows",
    "promoted_rows",
    "removed_rows",
    "unchanged_rows",
    "promoted_residues",
    "removed_residues",
    "unchanged_residues",
    "review_effect_status",
    "review_effect_reason",
    "recommended_action",
]

AI_REVIEW_ARTIFACT_MANIFEST_COLUMNS = [
    "artifact_name",
    "file_name",
    "artifact_type",
    "row_count",
    "byte_size",
    "sha256",
    "status",
    "purpose",
    "recommended_use",
]

AI_REVIEW_BUNDLE_VERIFICATION_COLUMNS = [
    "file_name",
    "expected_byte_size",
    "actual_byte_size",
    "expected_sha256",
    "actual_sha256",
    "verification_status",
    "verification_reason",
]

AI_REVIEW_BUNDLE_VERIFICATION_SUMMARY_COLUMNS = [
    "checked_files",
    "verified_files",
    "failed_files",
    "missing_files",
    "size_mismatch_files",
    "hash_mismatch_files",
    "extra_files",
    "invalid_zip_rows",
    "verification_status",
    "failed_file_names",
    "recommended_action",
]

RESIDUE_EVIDENCE_CONSENSUS_COLUMNS = [
    "chain",
    "resid",
    "residue_anchor",
    "uniprot_resid",
    "evidence_rows",
    "source_count",
    "functional_source_count",
    "ai_rows",
    "rankable_ai_rows",
    "non_ai_rows",
    "conservation_rows",
    "exact_rows",
    "weak_rows",
    "best_evidence_score",
    "best_mapping_confidence",
    "consensus_score",
    "consensus_tier",
    "ranking_status",
    "evidence_sources",
    "evidence_types",
    "ai_audit_statuses",
    "risk_flags",
    "recommended_action",
]


def _empty_ai_evidence_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def _empty_ai_audit_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_EVIDENCE_AUDIT_COLUMNS)


def _empty_ai_followup_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_FOLLOWUP_PLAN_COLUMNS)


def _empty_ai_impact_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_RANKING_IMPACT_COLUMNS)


def _empty_ai_review_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_QUEUE_COLUMNS)


def _empty_ai_review_decision_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_DECISION_COLUMNS)


def _empty_ai_review_decision_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_DECISION_TEMPLATE_COLUMNS)


def _empty_ai_review_decision_outcome_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_DECISION_OUTCOME_COLUMNS)


def _empty_ai_review_decision_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_DECISION_VALIDATION_COLUMNS)


def _empty_ai_review_round_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_ROUND_SUMMARY_COLUMNS)


def _empty_ai_review_ranking_delta_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_RANKING_DELTA_COLUMNS)


def _empty_ai_review_artifact_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_ARTIFACT_MANIFEST_COLUMNS)


def _empty_ai_review_bundle_verification_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_BUNDLE_VERIFICATION_COLUMNS)


def _empty_ai_review_bundle_verification_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=AI_REVIEW_BUNDLE_VERIFICATION_SUMMARY_COLUMNS)


def _empty_residue_evidence_consensus_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RESIDUE_EVIDENCE_CONSENSUS_COLUMNS)


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _append_flag(value: Any, flag: str) -> str:
    text = _safe_text(value)
    parts = [
        part.strip()
        for part in text.split(",")
        if part.strip() and part.strip().lower() != "none"
    ]
    if flag and flag not in parts:
        parts.append(flag)
    return ", ".join(parts) if parts else "none"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "review", "manual"}
    return bool(value)


def _normalize_resname(value: Any) -> str:
    text = _safe_text(value).upper()
    if len(text) == 1:
        return AA1_TO_3.get(text, "")
    return text[:3] if text[:3] in AA3_TO_1 else ""


def _position_from_text(value: Any) -> Optional[int]:
    text = _safe_text(value)
    matched = re.search(r"\b(?:[A-Za-z]{1,3})?\s*(\d{1,5})\b", text)
    if not matched:
        return None
    try:
        return int(matched.group(1))
    except ValueError:
        return None


def _as_position(*values: Any) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            text = str(value).strip()
            if text:
                return int(float(text))
        except (TypeError, ValueError):
            parsed = _position_from_text(value)
            if parsed is not None:
                return parsed
    return None


def _bounded_score(value: Any, default: float = 0.55) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("residues", "evidence", "sites", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _load_json_payload(payload_text: str) -> tuple[Optional[Any], str]:
    text = _safe_text(payload_text)
    if not text:
        return None, "empty"
    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError:
        matched = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if matched:
            try:
                return json.loads(matched.group(1)), "ok"
            except json.JSONDecodeError:
                pass
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if 0 <= first_brace < last_brace:
            try:
                return json.loads(text[first_brace : last_brace + 1]), "ok"
            except json.JSONDecodeError:
                pass
    return None, "invalid-json"


def _evidence_note(record: dict[str, Any], *, resname: str, confidence: float, manual_review: bool) -> str:
    parts = [
        f"pmid={_safe_text(record.get('pmid'))}" if _safe_text(record.get("pmid")) else "",
        f"doi={_safe_text(record.get('doi'))}" if _safe_text(record.get("doi")) else "",
        f"title={_safe_text(record.get('source_title') or record.get('title'))}" if _safe_text(record.get("source_title") or record.get("title")) else "",
        f"match={_safe_text(record.get('position_text') or record.get('residue_text'))}" if _safe_text(record.get("position_text") or record.get("residue_text")) else "",
        f"aa={resname}" if resname else "",
        f"snippet={_safe_text(record.get('evidence_snippet') or record.get('snippet'))}" if _safe_text(record.get("evidence_snippet") or record.get("snippet")) else "",
        f"ai_confidence={confidence:.3f}",
        f"manual_review={str(bool(manual_review)).lower()}",
    ]
    return " | ".join(part for part in parts if part)


def _verify_ai_structure_numbering(
    evidence_df: pd.DataFrame,
    *,
    chain_hint: str,
    pdb_text: Optional[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if evidence_df is None or getattr(evidence_df, "empty", True):
        return _empty_ai_evidence_df(), {
            "identity_checked_rows": "0",
            "identity_matched_rows": "0",
            "identity_mismatched_rows": "0",
            "identity_missing_rows": "0",
        }

    working = evidence_df.copy()
    structure_map = _extract_structure_residue_map(pdb_text)
    chain_entries = list((structure_map.get(str(chain_hint or "").strip()) or {}).get("entries") or [])
    observed = {
        int(entry.get("resid", 0) or 0): str(entry.get("resname", "") or "").upper()
        for entry in chain_entries
    }
    if not observed:
        return ensure_evidence_columns(working), {
            "identity_checked_rows": "0",
            "identity_matched_rows": "0",
            "identity_mismatched_rows": "0",
            "identity_missing_rows": "0",
        }

    checked = matched = mismatched = missing = 0
    for index, row in working.iterrows():
        try:
            resid = int(float(row.get("resid")))
        except (TypeError, ValueError):
            continue
        checked += 1
        expected = _normalize_resname(_safe_text(row.get("_ai_resname")))
        observed_resname = observed.get(resid, "")
        if not observed_resname:
            missing += 1
            working.at[index, "mapping_level"] = "weak"
            working.at[index, "mapping_confidence"] = min(_safe_float(row.get("mapping_confidence")), 0.26)
            working.at[index, "evidence_score"] = min(_safe_float(row.get("evidence_score")), 0.50)
            working.at[index, "mapping_method"] = "ai-structure-residue-missing"
            working.at[index, "evidence_note"] = f"{row.get('evidence_note')} | structure_residue_missing={chain_hint}:{resid}"
            continue
        if expected and observed_resname != expected:
            mismatched += 1
            working.at[index, "mapping_level"] = "weak"
            working.at[index, "mapping_confidence"] = min(_safe_float(row.get("mapping_confidence")), 0.24)
            working.at[index, "evidence_score"] = min(_safe_float(row.get("evidence_score")), 0.48)
            working.at[index, "mapping_method"] = "ai-structure-identity-mismatch"
            working.at[index, "evidence_note"] = f"{row.get('evidence_note')} | structure_residue_mismatch={expected}!={observed_resname}"
            continue
        matched += 1
        working.at[index, "mapping_level"] = "exact"
        working.at[index, "mapping_confidence"] = max(_safe_float(row.get("mapping_confidence")), 0.84 if expected else 0.76)
        working.at[index, "mapping_method"] = "ai-structure-numbering-verified" if expected else "ai-structure-numbering-assumed"
        if expected:
            working.at[index, "evidence_note"] = f"{row.get('evidence_note')} | structure_residue_match={observed_resname}"

    return ensure_evidence_columns(working.drop(columns=["_ai_resname"], errors="ignore")), {
        "identity_checked_rows": str(checked),
        "identity_matched_rows": str(matched),
        "identity_mismatched_rows": str(mismatched),
        "identity_missing_rows": str(missing),
    }


def parse_ai_residue_evidence_payload(
    payload_text: str,
    *,
    chain_hint: str = "",
    source_label: str = "AI-Literature",
    min_confidence: float = 0.35,
    assume_structure_numbering: bool = False,
    pdb_text: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload, status = _load_json_payload(payload_text)
    if status != "ok":
        return _empty_ai_evidence_df(), {"status": status, "evidence_rows": "0", "source": source_label}

    records = _records_from_payload(payload)
    rows: list[dict[str, Any]] = []
    skipped_low_confidence = 0
    manual_review_rows = 0
    for record in records:
        confidence = _bounded_score(record.get("confidence") or record.get("evidence_score"), 0.55)
        if confidence < float(min_confidence):
            skipped_low_confidence += 1
            continue

        resid = _as_position(
            record.get("structure_position"),
            record.get("resid"),
            record.get("pdb_resid"),
            record.get("uniprot_position"),
            record.get("uniprot_resid"),
            record.get("position_text"),
            record.get("residue_text"),
        )
        if resid is None:
            continue

        uniprot_resid = _as_position(
            record.get("uniprot_position"),
            record.get("uniprot_resid"),
            record.get("position_text"),
            record.get("residue_text"),
        )
        resname = _normalize_resname(record.get("resname") or record.get("aa") or record.get("amino_acid"))
        snippet = _safe_text(record.get("evidence_snippet") or record.get("snippet"))
        has_source = bool(_safe_text(record.get("pmid") or record.get("doi") or record.get("source_title") or record.get("title")))
        manual_review = _safe_bool(record.get("requires_manual_review")) or not snippet or not has_source
        if manual_review:
            manual_review_rows += 1
            confidence = min(confidence, 0.62)

        row_chain = _safe_text(record.get("chain"), chain_hint).strip() or str(chain_hint or "").strip()
        mapping_confidence = 0.44 if row_chain else 0.30
        evidence_score = min(1.0, max(0.0, confidence))
        rows.append(
            {
                "chain": row_chain,
                "resid": int(resid),
                "evidence_source": _safe_text(record.get("evidence_source"), source_label),
                "evidence_type": _safe_text(record.get("evidence_type"), "AI extracted residue"),
                "evidence_score": round(float(evidence_score), 3),
                "evidence_note": _evidence_note(record, resname=resname, confidence=confidence, manual_review=manual_review),
                "uniprot_resid": int(uniprot_resid if uniprot_resid is not None else resid),
                "mapping_level": "weak",
                "mapping_confidence": mapping_confidence,
                "mapping_method": "ai-literature-extraction-review" if manual_review else "ai-literature-extraction",
                "article_title": _safe_text(record.get("source_title") or record.get("title")),
                "pmid": _safe_text(record.get("pmid")),
                "pmcid": _safe_text(record.get("pmcid") or record.get("pmc_id")),
                "doi": _safe_text(record.get("doi")),
                "evidence_snippet": snippet,
                "sentence_index": _safe_text(record.get("sentence_index")),
                "extraction_pattern": _safe_text(record.get("extraction_pattern"), "ai-json"),
                "requires_manual_review": bool(manual_review),
                "_ai_resname": resname,
            }
        )

    if not rows:
        return _empty_ai_evidence_df(), {
            "status": "empty",
            "source": source_label,
            "parsed_records": str(len(records)),
            "evidence_rows": "0",
            "skipped_low_confidence": str(skipped_low_confidence),
            "manual_review_rows": str(manual_review_rows),
        }

    evidence_df = pd.DataFrame(rows).sort_values(
        ["evidence_score", "mapping_confidence", "resid", "evidence_type"],
        ascending=[False, False, True, True],
    ).drop_duplicates(
        subset=["chain", "resid", "evidence_type", "uniprot_resid"],
        keep="first",
    ).reset_index(drop=True)

    identity_meta: dict[str, str] = {}
    if assume_structure_numbering and str(chain_hint or "").strip():
        evidence_df, identity_meta = _verify_ai_structure_numbering(
            evidence_df,
            chain_hint=str(chain_hint or "").strip(),
            pdb_text=pdb_text,
        )
    else:
        evidence_df = ensure_evidence_columns(evidence_df.drop(columns=["_ai_resname"], errors="ignore"))

    exact_rows = int(evidence_df["mapping_level"].astype(str).str.lower().eq("exact").sum()) if not evidence_df.empty else 0
    return ensure_evidence_columns(evidence_df), {
        "status": "ok",
        "source": source_label,
        "parsed_records": str(len(records)),
        "evidence_rows": str(len(evidence_df)),
        "exact_rows": str(exact_rows),
        "weak_rows": str(max(0, len(evidence_df) - exact_rows)),
        "skipped_low_confidence": str(skipped_low_confidence),
        "manual_review_rows": str(manual_review_rows),
        **identity_meta,
    }


def build_ai_evidence_prompt(
    literature_or_notes: str,
    *,
    protein_name: str = "",
    accession: str = "",
    pdb_id: str = "",
    ec_number: str = "",
    triage_context: str = "",
) -> str:
    context_parts = [
        f"Protein name: {protein_name}" if protein_name else "",
        f"UniProt accession: {accession}" if accession else "",
        f"PDB ID: {pdb_id}" if pdb_id else "",
        f"EC number: {ec_number}" if ec_number else "",
        f"Current triage context:\n{triage_context}" if triage_context else "",
    ]
    schema = json.dumps(AI_EVIDENCE_SCHEMA_HINT, ensure_ascii=False, indent=2)
    return (
        "Extract only enzyme-relevant catalytic, binding, metal-binding, cofactor-binding, or activity-loss mutagenesis residues "
        "from the supplied text. Do not infer residues from general knowledge. Every residue must be supported by a source snippet. "
        "If numbering may be UniProt rather than PDB numbering, use uniprot_position and leave structure_position null. "
        "Return strict JSON only, with this shape:\n"
        f"{schema}\n\n"
        + "\n".join(part for part in context_parts if part)
        + "\n\nSource text:\n"
        + str(literature_or_notes or "")[:20000]
    )


def build_ai_triage_context(
    decision_df: Optional[pd.DataFrame],
    reliability_df: Optional[pd.DataFrame],
    triage_df: Optional[pd.DataFrame],
    *,
    max_rows: int = 3,
) -> str:
    parts: list[str] = []
    for name, table in [
        ("decision", decision_df),
        ("reliability", reliability_df),
        ("precision_triage", triage_df),
    ]:
        if table is None or getattr(table, "empty", True):
            continue
        parts.append(f"[{name}]\n{table.head(max(1, int(max_rows))).to_json(orient='records', force_ascii=False)}")
    return "\n\n".join(parts)


def _split_checks(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text or text.lower() == "none":
        return []
    return [part.strip() for part in text.split(",") if part.strip() and part.strip().lower() != "none"]


def _base_search_terms(*, protein_name: str = "", accession: str = "", pdb_id: str = "", ec_number: str = "") -> list[str]:
    terms: list[str] = []
    for value in (protein_name, accession, pdb_id):
        text = _safe_text(value)
        if text and text not in terms:
            terms.append(text)
    ec_text = _safe_text(ec_number)
    if ec_text:
        terms.append(f'"{ec_text}"')
    return terms


def _query_for_gap(gap: str, base_terms: list[str]) -> str:
    gap_lower = gap.lower()
    query_terms = list(base_terms)
    if "functional" in gap_lower:
        query_terms.extend(['"active site"', "catalytic residue", "mutagenesis", '"loss of activity"'])
    elif "mapping" in gap_lower:
        query_terms.extend(["residue numbering", "PDB chain", "UniProt", "SIFTS", "author numbering"])
    elif "geometry" in gap_lower:
        query_terms.extend(['"binding pocket"', "ligand", "substrate binding", "P2Rank", "fpocket"])
    elif "a/b" in gap_lower:
        query_terms.extend(["catalytic residue", "conservation", "mutagenesis", '"binding site"'])
    elif "actionability" in gap_lower:
        query_terms.extend(["catalytic mechanism", "substrate binding", "cofactor", "mutagenesis"])
    else:
        query_terms.extend(["active site", "catalytic", "binding residue"])
    return " ".join(dict.fromkeys(term for term in query_terms if term)).strip()


def _followup_links(query: str, *, accession: str = "", pdb_id: str = "") -> dict[str, str]:
    cleaned_query = _safe_text(query)
    encoded_query = quote_plus(cleaned_query)
    cleaned_accession = _safe_text(accession).upper()
    cleaned_pdb = _safe_text(pdb_id).upper()
    return {
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded_query}" if cleaned_query else "",
        "europepmc_url": f"https://europepmc.org/search?query={encoded_query}" if cleaned_query else "",
        "uniprot_url": f"https://www.uniprot.org/uniprotkb/{quote_plus(cleaned_accession)}/entry" if cleaned_accession else "",
        "rcsb_url": f"https://www.rcsb.org/structure/{quote_plus(cleaned_pdb)}" if cleaned_pdb else "",
    }


def _acceptance_for_gap(gap: str) -> str:
    gap_lower = gap.lower()
    base = "Require PMID/DOI/title, evidence_snippet, residue identity, confidence >= 0.65, and AI audit status supported or structure-verified."
    if "mapping" in gap_lower:
        return base + " Also require chain/residue-numbering notes and PDB residue identity verification before ranking."
    if "functional" in gap_lower:
        return base + " Prefer catalytic, binding, metal/cofactor, or activity-loss mutagenesis evidence over generic sequence mentions."
    if "geometry" in gap_lower:
        return base + " Use geometry evidence only as supporting context; do not promote without residue-level functional evidence."
    return base


AI_FOLLOWUP_TEXT_REPLACEMENTS = [
    ("Functional anchors are missing.", "缺少功能锚点。"),
    ("Functional anchors", "功能锚点"),
    ("Evidence mapping risk", "证据映射风险"),
    ("Geometry consensus", "几何共识"),
    ("Evidence A/B movement", "证据 A/B 变化"),
    ("Actionability", "可操作性"),
    ("Cross-source validation", "跨来源验证"),
    ("validation-ready", "可验证"),
    ("evidence-gap", "证据缺口"),
    ("mapping-review", "映射复核"),
    ("geometry-review", "几何复核"),
    ("evidence-review", "证据复核"),
    ("exploratory", "探索性"),
    (
        "Require PMID/DOI/title, evidence_snippet, residue identity, confidence >= 0.65, and AI audit status supported or structure-verified.",
        "需要 PMID/DOI/标题、证据片段、残基身份、confidence >= 0.65，且 AI 审计状态为已支持或结构已验证。",
    ),
    (
        "Require PMID/DOI/title and AI audit status supported or structure-verified.",
        "需要 PMID/DOI/标题，并且 AI 审计状态为已支持或结构已验证。",
    ),
    (
        "Also require chain/residue-numbering notes and PDB residue identity verification before ranking.",
        "排名前还需要链/残基编号说明和 PDB 残基身份核验。",
    ),
    (
        "Prefer catalytic, binding, metal/cofactor, or activity-loss mutagenesis evidence over generic sequence mentions.",
        "优先采用催化、结合、金属/辅因子或活性损失突变证据，而不是泛泛的序列提及。",
    ),
    (
        "Use geometry evidence only as supporting context; do not promote without residue-level functional evidence.",
        "几何证据只能作为支持上下文；没有残基层功能证据时不要提升排名。",
    ),
    (
        "Resolve this evidence gap before treating the pocket as a high-confidence enzyme active site.",
        "先补齐该证据缺口，再把口袋视为高置信度酶活性位点。",
    ),
]


def _localize_ai_followup_text(value: Any, default: str = "-") -> str:
    text = _safe_text(value, default)
    for source, target in AI_FOLLOWUP_TEXT_REPLACEMENTS:
        text = text.replace(source, target)
    return text


def build_ai_followup_evidence_plan(
    decision_df: Optional[pd.DataFrame],
    reliability_df: Optional[pd.DataFrame],
    triage_df: Optional[pd.DataFrame],
    *,
    protein_name: str = "",
    accession: str = "",
    pdb_id: str = "",
    ec_number: str = "",
    max_pockets: int = 3,
) -> pd.DataFrame:
    if triage_df is None or getattr(triage_df, "empty", True) or "pocket_id" not in triage_df.columns:
        return _empty_ai_followup_df()

    triage = triage_df.copy()
    sort_columns = [column for column in ("triage_priority", "decision_rank", "pocket_id") if column in triage.columns]
    if sort_columns:
        triage = triage.sort_values(sort_columns, ascending=[True] * len(sort_columns))
    triage = triage.head(max(1, int(max_pockets)))

    triage_context = build_ai_triage_context(decision_df, reliability_df, triage, max_rows=max_pockets)
    base_terms = _base_search_terms(
        protein_name=protein_name,
        accession=accession,
        pdb_id=pdb_id,
        ec_number=ec_number,
    )
    rows: list[dict[str, Any]] = []
    for _, triage_row in triage.iterrows():
        pocket_id = _safe_text(triage_row.get("pocket_id"))
        if not pocket_id:
            continue
        blocking_checks = _split_checks(triage_row.get("blocking_checks"))
        review_checks = _split_checks(triage_row.get("review_checks"))
        checks = blocking_checks + [check for check in review_checks if check not in blocking_checks]
        if not checks:
            tier = _safe_text(triage_row.get("precision_tier"))
            if tier == "validation-ready":
                checks = ["Cross-source validation"]
            else:
                checks = ["Functional anchors"]

        for order, check in enumerate(checks[:3], start=1):
            query = _query_for_gap(check, base_terms)
            links = _followup_links(query, accession=accession, pdb_id=pdb_id)
            extraction_prompt = build_ai_evidence_prompt(
                "Paste retrieved abstracts, full text snippets, UniProt notes, or curator notes here.",
                protein_name=protein_name,
                accession=accession,
                pdb_id=pdb_id,
                ec_number=ec_number,
                triage_context=triage_context,
            )
            rows.append(
                {
                    "pocket_id": pocket_id,
                    "decision_rank": int(_safe_float(triage_row.get("decision_rank"), 0.0)),
                    "precision_tier": _safe_text(triage_row.get("precision_tier"), "-"),
                    "followup_priority": int(_safe_float(triage_row.get("triage_priority"), 99.0)) * 10 + order,
                    "evidence_gap": check,
                    "search_query": query,
                    **links,
                    "ai_task_prompt": extraction_prompt,
                    "acceptance_criteria": _acceptance_for_gap(check),
                    "why_this_matters": _safe_text(
                        triage_row.get("triage_reason"),
                        "Resolve this evidence gap before treating the pocket as a high-confidence enzyme active site.",
                    ),
                }
            )

    if not rows:
        return _empty_ai_followup_df()
    return pd.DataFrame(rows, columns=AI_FOLLOWUP_PLAN_COLUMNS).sort_values(
        ["followup_priority", "pocket_id"],
        ascending=[True, True],
    ).reset_index(drop=True)


def build_ai_followup_prompt_bundle(
    followup_plan_df: Optional[pd.DataFrame],
    *,
    title: str = "AI 后续取证计划",
) -> str:
    header = [
        f"# {title}",
        "",
        "只使用检索到的来源文本，不允许模型凭记忆补全缺失残基。",
        "只有来源引用、证据片段、结构映射、AI 审计和排名门控全部通过后，才接受 AI 残基证据。",
        "",
    ]
    if followup_plan_df is None or getattr(followup_plan_df, "empty", True):
        return "\n".join(header + ["暂无需要后续取证的证据缺口。"])

    lines = list(header)
    table = followup_plan_df.copy()
    if "followup_priority" in table.columns:
        table = table.sort_values(["followup_priority", "pocket_id"], ascending=[True, True])
    for index, row in enumerate(table.itertuples(index=False), start=1):
        pocket_id = _safe_text(getattr(row, "pocket_id", ""), "-")
        evidence_gap = _localize_ai_followup_text(getattr(row, "evidence_gap", ""), "-")
        precision_tier = _localize_ai_followup_text(getattr(row, "precision_tier", ""), "-")
        search_query = _safe_text(getattr(row, "search_query", ""), "-")
        pubmed_url = _safe_text(getattr(row, "pubmed_url", ""))
        europepmc_url = _safe_text(getattr(row, "europepmc_url", ""))
        uniprot_url = _safe_text(getattr(row, "uniprot_url", ""))
        rcsb_url = _safe_text(getattr(row, "rcsb_url", ""))
        acceptance = _localize_ai_followup_text(getattr(row, "acceptance_criteria", ""), "-")
        why = _localize_ai_followup_text(getattr(row, "why_this_matters", ""), "-")
        prompt = _safe_text(getattr(row, "ai_task_prompt", ""), "-")
        link_lines = []
        if pubmed_url:
            link_lines.append(f"- PubMed: {pubmed_url}")
        if europepmc_url:
            link_lines.append(f"- Europe PMC: {europepmc_url}")
        if uniprot_url:
            link_lines.append(f"- UniProt: {uniprot_url}")
        if rcsb_url:
            link_lines.append(f"- RCSB PDB: {rcsb_url}")
        lines.extend(
            [
                f"## {index}. {pocket_id} - {evidence_gap}",
                "",
                f"- 精度分层: {precision_tier}",
                f"- 检索式: `{search_query}`",
                f"- 重要性: {why}",
                f"- 接受标准: {acceptance}",
                "",
                "### 来源链接",
                "",
                *(link_lines or ["- 未生成来源链接。"]),
                "",
                "### AI 抽取提示词",
                "",
                "```text",
                prompt,
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _residue_keys_from_evidence(table: Optional[pd.DataFrame]) -> set[tuple[str, int]]:
    if table is None or getattr(table, "empty", True) or "resid" not in table.columns:
        return set()
    keys: set[tuple[str, int]] = set()
    for _, row in table.iterrows():
        try:
            resid = int(float(row.get("resid")))
        except (TypeError, ValueError):
            continue
        chain = _safe_text(row.get("chain"), "")
        keys.add((chain, resid))
    return keys


def _residue_keys_from_anchor_text(value: Any) -> set[tuple[str, int]]:
    text = _safe_text(value)
    keys: set[tuple[str, int]] = set()
    for chain, resid_text in re.findall(r"\b([A-Za-z0-9])\s*:\s*(-?\d+)\b", text):
        try:
            keys.add((chain.strip(), int(resid_text)))
        except ValueError:
            pass
    if keys:
        return keys
    for resid_text in re.findall(r"\b(?:[A-Z][a-z]{2}|[A-Z])?\s*(-?\d{1,5})\b", text):
        try:
            keys.add(("", int(resid_text)))
        except ValueError:
            pass
    return keys


def _format_residue_keys(keys: set[tuple[str, int]]) -> str:
    if not keys:
        return "none"
    return ", ".join(
        f"{chain}:{resid}" if chain else str(resid)
        for chain, resid in sorted(keys, key=lambda item: (item[0], item[1]))
    )


def _residue_key_overlap(left: set[tuple[str, int]], right: set[tuple[str, int]]) -> set[tuple[str, int]]:
    if not left or not right:
        return set()
    exact = left & right
    if exact:
        return exact
    left_resids = {resid for _chain, resid in left}
    right_resids = {resid for _chain, resid in right}
    shared_resids = left_resids & right_resids
    return {key for key in left if key[1] in shared_resids}


def build_ai_ranking_impact_summary(
    ai_evidence_df: Optional[pd.DataFrame],
    rankable_ai_evidence_df: Optional[pd.DataFrame],
    audit_df: Optional[pd.DataFrame],
    decision_df: Optional[pd.DataFrame],
    triage_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    ai_input_rows = 0 if ai_evidence_df is None or getattr(ai_evidence_df, "empty", True) else int(len(ai_evidence_df))
    ai_ranked_rows = 0 if rankable_ai_evidence_df is None or getattr(rankable_ai_evidence_df, "empty", True) else int(len(rankable_ai_evidence_df))
    ai_excluded_rows = max(0, ai_input_rows - ai_ranked_rows)

    audit_supported = audit_structure_verified = audit_review = audit_conflicting = 0
    if audit_df is not None and not getattr(audit_df, "empty", True) and "audit_status" in audit_df.columns:
        status_counts = audit_df["audit_status"].astype(str).str.lower().value_counts().to_dict()
        audit_supported = int(status_counts.get("supported", 0))
        audit_structure_verified = int(status_counts.get("structure-verified", 0))
        audit_conflicting = int(status_counts.get("conflicting", 0))
        audit_review = int(
            status_counts.get("needs-review", 0)
            + status_counts.get("unsupported", 0)
            + status_counts.get("conflicting", 0)
        )

    top_pocket_id = ""
    top_precision_tier = ""
    top_anchor_keys: set[tuple[str, int]] = set()
    supporting_text = ""
    if decision_df is not None and not getattr(decision_df, "empty", True):
        decision_table = decision_df.copy()
        if "decision_rank" in decision_table.columns:
            decision_table = decision_table.sort_values(["decision_rank"], ascending=[True])
        top_decision = decision_table.iloc[0]
        top_pocket_id = _safe_text(top_decision.get("pocket_id"))
        top_anchor_keys = _residue_keys_from_anchor_text(top_decision.get("anchor_residues"))
        supporting_text = _safe_text(top_decision.get("supporting_evidence")).lower()

    if top_pocket_id and triage_df is not None and not getattr(triage_df, "empty", True) and "pocket_id" in triage_df.columns:
        matched = triage_df[triage_df["pocket_id"].astype(str) == str(top_pocket_id)]
        if not matched.empty:
            top_precision_tier = _safe_text(matched.iloc[0].get("precision_tier"))

    rankable_keys = _residue_keys_from_evidence(rankable_ai_evidence_df)
    overlap_keys = _residue_key_overlap(top_anchor_keys, rankable_keys)
    top_has_ai_support = bool(overlap_keys) or ("ai" in supporting_text and ai_ranked_rows > 0)

    if ai_input_rows == 0:
        influence = "none"
        reason = "No AI residue evidence was supplied."
        action = "Use the follow-up plan to collect literature or database evidence before enabling AI evidence."
    elif ai_ranked_rows == 0:
        influence = "blocked"
        reason = "AI residue evidence was supplied, but none passed the audit gate for ranking."
        action = "Review excluded AI rows; add source snippets, citations, or structure-numbering verification."
    elif top_has_ai_support:
        influence = "top-pocket-supported"
        reason = "At least one ranking-gated AI residue overlaps the Top pocket anchors or Top pocket support text."
        action = "Keep AI audit details with the report and verify the cited source before validation."
    else:
        influence = "background-support"
        reason = "Ranking-gated AI evidence exists, but it does not directly overlap the Top pocket anchors."
        action = "Inspect lower-ranked pockets and confirm whether AI evidence should change follow-up priorities."

    return pd.DataFrame(
        [
            {
                "ai_input_rows": ai_input_rows,
                "ai_ranked_rows": ai_ranked_rows,
                "ai_excluded_rows": ai_excluded_rows,
                "audit_supported_rows": audit_supported,
                "audit_structure_verified_rows": audit_structure_verified,
                "audit_review_rows": audit_review,
                "audit_conflicting_rows": audit_conflicting,
                "top_pocket_id": top_pocket_id or "-",
                "top_pocket_precision_tier": top_precision_tier or "-",
                "top_pocket_has_ai_support": bool(top_has_ai_support),
                "top_pocket_ai_residue_count": int(len(overlap_keys)),
                "top_pocket_ai_residues": _format_residue_keys(overlap_keys),
                "ai_influence_level": influence,
                "ai_influence_reason": reason,
                "recommended_action": action,
            }
        ],
        columns=AI_RANKING_IMPACT_COLUMNS,
    )


def _review_fix_for_audit(status: str, risk_flags: str) -> tuple[int, str, str, str, bool, str]:
    normalized_status = _safe_text(status).lower()
    flags = {flag.strip() for flag in _safe_text(risk_flags).split(",") if flag.strip() and flag.strip().lower() != "none"}
    if normalized_status == "conflicting" or "structure-conflict" in flags:
        return (
            1,
            "structure-conflict",
            "AI residue conflicts with PDB residue identity or structure numbering.",
            "Correct chain/residue numbering, verify residue identity in the structure, and rerun AI audit.",
            False,
            "Do not use this residue for ranking until mapping conflict is resolved.",
        )
    if "missing-source-snippet" in flags or "missing-source-id" in flags:
        return (
            2,
            "missing-citation-or-snippet",
            "AI residue lacks a citable source or supporting evidence sentence.",
            "Add PMID/DOI/title plus the exact sentence that states the catalytic/binding/mutagenesis evidence.",
            True,
            "Fetch source text, rerun AI extraction, and require audit status supported or structure-verified.",
        )
    if "low-mapping-confidence" in flags or "weak-mapping" in flags:
        return (
            3,
            "weak-mapping",
            "AI residue has weak residue mapping or low mapping confidence.",
            "Verify UniProt/PDB offset, chain ID, author numbering, insertion codes, or structure residue identity.",
            True,
            "Resolve mapping before allowing this evidence to influence ranking.",
        )
    if normalized_status == "needs-review" or "manual-review" in flags:
        return (
            4,
            "manual-review",
            "AI residue may be useful but still needs manual evidence review.",
            "Check citation, snippet, residue identity, evidence type, and whether a non-AI source supports the same residue.",
            True,
            "Accept only after manual review or independent source support.",
        )
    if normalized_status == "unsupported" or "no-independent-support" in flags:
        return (
            5,
            "independent-support-needed",
            "AI residue has no independent non-AI evidence support.",
            "Find M-CSA, UniProt, curated literature, mutagenesis, conservation, or structure-verified support.",
            True,
            "Keep as background evidence until another source supports the same residue.",
        )
    return (
        6,
        "review",
        "AI evidence requires review before ranking confidence can increase.",
        "Inspect audit reason and supporting evidence details.",
        True,
        "Review manually before accepting.",
    )


def build_ai_evidence_review_queue(
    audit_df: Optional[pd.DataFrame],
    *,
    include_supported: bool = False,
) -> pd.DataFrame:
    if audit_df is None or getattr(audit_df, "empty", True) or "audit_status" not in audit_df.columns:
        return _empty_ai_review_queue_df()

    rows: list[dict[str, Any]] = []
    for _, row in audit_df.iterrows():
        status = _safe_text(row.get("audit_status")).lower()
        if not include_supported and status in {"supported", "structure-verified", "manually-accepted", "manually-rejected"}:
            continue
        priority, fix_type, problem, required_evidence, can_rank_after_fix, suggested_action = _review_fix_for_audit(
            status,
            _safe_text(row.get("risk_flags")),
        )
        try:
            resid = int(float(row.get("resid")))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "review_priority": priority,
                "chain": _safe_text(row.get("chain")),
                "resid": resid,
                "evidence_type": _safe_text(row.get("evidence_type"), "AI extracted residue"),
                "audit_status": status or "-",
                "fix_type": fix_type,
                "problem": problem,
                "required_evidence": required_evidence,
                "can_affect_ranking_after_fix": bool(can_rank_after_fix),
                "suggested_next_action": suggested_action,
            }
        )

    if not rows:
        return _empty_ai_review_queue_df()
    return pd.DataFrame(rows, columns=AI_REVIEW_QUEUE_COLUMNS).sort_values(
        ["review_priority", "chain", "resid"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_ai_review_checklist_markdown(
    review_queue_df: Optional[pd.DataFrame],
    *,
    title: str = "AI evidence review checklist",
) -> str:
    lines = [
        f"# {title}",
        "",
        "Use this checklist to resolve AI evidence rows that cannot safely increase pocket-ranking confidence yet.",
        "A row should only move into ranking after its required evidence is supplied and the AI audit/ranking gate is rerun.",
        "",
    ]
    if review_queue_df is None or getattr(review_queue_df, "empty", True):
        return "\n".join(lines + ["No AI evidence review items are currently open."])

    queue = review_queue_df.copy()
    if "review_priority" in queue.columns:
        queue = queue.sort_values(["review_priority", "chain", "resid"], ascending=[True, True, True])
    for index, row in enumerate(queue.itertuples(index=False), start=1):
        chain = _safe_text(getattr(row, "chain", ""))
        resid = _safe_text(getattr(row, "resid", ""))
        residue_label = f"{chain}:{resid}" if chain else resid
        can_rank = bool(getattr(row, "can_affect_ranking_after_fix", False))
        lines.extend(
            [
                f"## {index}. {residue_label} - {_safe_text(getattr(row, 'fix_type', ''), 'review')}",
                "",
                f"- [ ] Problem: {_safe_text(getattr(row, 'problem', ''), '-')}",
                f"- [ ] Required evidence: {_safe_text(getattr(row, 'required_evidence', ''), '-')}",
                f"- [ ] Suggested next action: {_safe_text(getattr(row, 'suggested_next_action', ''), '-')}",
                f"- Audit status: `{_safe_text(getattr(row, 'audit_status', ''), '-')}`",
                f"- Evidence type: `{_safe_text(getattr(row, 'evidence_type', ''), '-')}`",
                f"- Can affect ranking after fix: {'yes' if can_rank else 'no'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_ai_review_decision_template(review_queue_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if review_queue_df is None or getattr(review_queue_df, "empty", True):
        return _empty_ai_review_decision_template_df()

    rows: list[dict[str, Any]] = []
    queue = review_queue_df.copy()
    if "review_priority" in queue.columns:
        queue = queue.sort_values(["review_priority", "chain", "resid"], ascending=[True, True, True])
    for _, row in queue.iterrows():
        resid = _as_position(row.get("resid"))
        if resid is None:
            continue
        rows.append(
            {
                "chain": _safe_text(row.get("chain")),
                "resid": int(resid),
                "evidence_type": _safe_text(row.get("evidence_type"), "AI extracted residue"),
                "review_decision": "review",
                "reviewer": "",
                "review_note": "",
                "verified_source": "",
                "verified_snippet": "",
                "current_audit_status": _safe_text(row.get("audit_status"), "-"),
                "review_priority": int(_safe_float(row.get("review_priority"), 99)),
                "fix_type": _safe_text(row.get("fix_type"), "review"),
                "problem": _safe_text(row.get("problem"), "-"),
                "required_evidence": _safe_text(row.get("required_evidence"), "-"),
                "suggested_next_action": _safe_text(row.get("suggested_next_action"), "-"),
                "can_affect_ranking_after_fix": bool(row.get("can_affect_ranking_after_fix")),
            }
        )

    if not rows:
        return _empty_ai_review_decision_template_df()
    return pd.DataFrame(rows, columns=AI_REVIEW_DECISION_TEMPLATE_COLUMNS).reset_index(drop=True)


def _normalize_column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _safe_text(value).lower()).strip("_")


def _pick_column(columns: list[str], aliases: set[str]) -> str:
    normalized = {_normalize_column_key(column): column for column in columns}
    for alias in aliases:
        column = normalized.get(alias)
        if column:
            return column
    return ""


def _normalize_review_decision(value: Any) -> str:
    text = _safe_text(value).lower()
    if text in {"1", "true", "yes", "y", "accept", "accepted", "approve", "approved"}:
        return "accept"
    if text in {"0", "false", "no", "n", "reject", "rejected", "exclude", "excluded"}:
        return "reject"
    if text in {"review", "defer", "deferred", "pending", "hold", "manual-review"}:
        return "review"
    return "review"


def parse_ai_review_decision_table(decision_text: str) -> tuple[pd.DataFrame, dict[str, str]]:
    text = _safe_text(decision_text)
    if not text:
        return _empty_ai_review_decision_df(), {
            "status": "empty",
            "input_rows": "0",
            "decision_rows": "0",
            "skipped_rows": "0",
        }

    try:
        raw = pd.read_csv(StringIO(text), sep=None, engine="python")
    except Exception as exc:
        return _empty_ai_review_decision_df(), {
            "status": "parse-error",
            "input_rows": "0",
            "decision_rows": "0",
            "skipped_rows": "0",
            "message": str(exc),
        }
    if raw.empty:
        return _empty_ai_review_decision_df(), {
            "status": "empty",
            "input_rows": "0",
            "decision_rows": "0",
            "skipped_rows": "0",
        }

    columns = [str(column) for column in raw.columns]
    selected = {
        "chain": _pick_column(columns, {"chain", "pdb_chain", "asym_id"}),
        "resid": _pick_column(columns, {"resid", "residue_id", "residue", "position", "residue_number", "pdb_resid", "pdb_position"}),
        "evidence_type": _pick_column(columns, {"evidence_type", "type", "site_type", "role"}),
        "review_decision": _pick_column(columns, {"review_decision", "decision", "status", "accept", "review_status"}),
        "reviewer": _pick_column(columns, {"reviewer", "curator", "user"}),
        "review_note": _pick_column(columns, {"review_note", "note", "notes", "comment", "comments"}),
        "verified_source": _pick_column(columns, {"verified_source", "source", "citation", "pmid", "doi", "reference"}),
        "verified_snippet": _pick_column(columns, {"verified_snippet", "snippet", "evidence_snippet", "quote", "sentence"}),
    }
    if not selected["resid"]:
        return _empty_ai_review_decision_df(), {
            "status": "missing-resid-column",
            "input_rows": str(len(raw)),
            "decision_rows": "0",
            "skipped_rows": str(len(raw)),
        }

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in raw.iterrows():
        resid = _as_position(row.get(selected["resid"]))
        if resid is None:
            skipped += 1
            continue
        rows.append(
            {
                "chain": _safe_text(row.get(selected["chain"])) if selected["chain"] else "",
                "resid": int(resid),
                "evidence_type": _safe_text(row.get(selected["evidence_type"])) if selected["evidence_type"] else "",
                "review_decision": _normalize_review_decision(row.get(selected["review_decision"])) if selected["review_decision"] else "review",
                "reviewer": _safe_text(row.get(selected["reviewer"])) if selected["reviewer"] else "",
                "review_note": _safe_text(row.get(selected["review_note"])) if selected["review_note"] else "",
                "verified_source": _safe_text(row.get(selected["verified_source"])) if selected["verified_source"] else "",
                "verified_snippet": _safe_text(row.get(selected["verified_snippet"])) if selected["verified_snippet"] else "",
            }
        )

    if not rows:
        return _empty_ai_review_decision_df(), {
            "status": "empty-after-normalization",
            "input_rows": str(len(raw)),
            "decision_rows": "0",
            "skipped_rows": str(skipped),
        }

    return pd.DataFrame(rows, columns=AI_REVIEW_DECISION_COLUMNS), {
        "status": "ok",
        "input_rows": str(len(raw)),
        "decision_rows": str(len(rows)),
        "skipped_rows": str(skipped),
        "accept_rows": str(sum(1 for item in rows if item["review_decision"] == "accept")),
        "reject_rows": str(sum(1 for item in rows if item["review_decision"] == "reject")),
        "review_rows": str(sum(1 for item in rows if item["review_decision"] == "review")),
    }


def _matching_review_decision(
    decisions_df: pd.DataFrame,
    *,
    chain: str,
    resid: int,
    evidence_type: str,
) -> Optional[pd.Series]:
    if decisions_df is None or getattr(decisions_df, "empty", True):
        return None
    working = decisions_df.copy()
    if "resid" not in working.columns or "review_decision" not in working.columns:
        return None
    working["_resid_numeric"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["_resid_numeric"].notna()].copy()
    if working.empty:
        return None

    chain_text = _safe_text(chain)
    type_text = _safe_text(evidence_type).lower()
    working["_chain_text"] = working.get("chain", pd.Series("", index=working.index)).astype(str).str.strip()
    working["_type_text"] = working.get("evidence_type", pd.Series("", index=working.index)).astype(str).str.strip().str.lower()
    mask = working["_resid_numeric"].astype(int).eq(int(resid))
    if chain_text:
        mask = mask & ((working["_chain_text"] == chain_text) | (working["_chain_text"] == ""))
    else:
        mask = mask & ((working["_chain_text"] == "") | working["_chain_text"].notna())
    if type_text:
        mask = mask & ((working["_type_text"] == type_text) | (working["_type_text"] == ""))

    candidates = working.loc[mask].copy()
    if candidates.empty:
        return None
    candidates["_specificity"] = candidates["_chain_text"].astype(bool).astype(int) + candidates["_type_text"].astype(bool).astype(int)
    candidates = candidates.sort_values("_specificity", ascending=False)
    return candidates.iloc[0]


def _manual_review_reason(row: pd.Series) -> str:
    parts = [
        f"manual_review_decision={_safe_text(row.get('review_decision'))}",
        f"reviewer={_safe_text(row.get('reviewer'))}" if _safe_text(row.get("reviewer")) else "",
        f"source={_safe_text(row.get('verified_source'))}" if _safe_text(row.get("verified_source")) else "",
        f"snippet={_safe_text(row.get('verified_snippet'))}" if _safe_text(row.get("verified_snippet")) else "",
        f"note={_safe_text(row.get('review_note'))}" if _safe_text(row.get("review_note")) else "",
    ]
    return " | ".join(part for part in parts if part)


def apply_ai_review_decisions_to_audit(
    audit_df: Optional[pd.DataFrame],
    decisions_df: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if audit_df is None or getattr(audit_df, "empty", True):
        return _empty_ai_audit_df(), {
            "status": "empty-audit",
            "applied_rows": "0",
            "accepted_rows": "0",
            "rejected_rows": "0",
            "review_rows": "0",
            "conflict_blocked_rows": "0",
        }
    if decisions_df is None or getattr(decisions_df, "empty", True):
        return audit_df.copy(), {
            "status": "empty-decisions",
            "applied_rows": "0",
            "accepted_rows": "0",
            "rejected_rows": "0",
            "review_rows": "0",
            "conflict_blocked_rows": "0",
        }

    working = audit_df.copy()
    for column in AI_EVIDENCE_AUDIT_COLUMNS:
        if column not in working.columns:
            working[column] = ""

    applied = accepted = rejected = review = conflict_blocked = 0
    for index, row in working.iterrows():
        resid = _as_position(row.get("resid"))
        if resid is None:
            continue
        decision = _matching_review_decision(
            decisions_df,
            chain=_safe_text(row.get("chain")),
            resid=int(resid),
            evidence_type=_safe_text(row.get("evidence_type")),
        )
        if decision is None:
            continue

        applied += 1
        decision_text = _safe_text(decision.get("review_decision")).lower()
        original_status = _safe_text(row.get("audit_status")).lower()
        flags = _safe_text(row.get("risk_flags")).lower()
        reason_suffix = _manual_review_reason(decision)
        existing_reason = _safe_text(row.get("audit_reason"))
        if reason_suffix:
            working.at[index, "audit_reason"] = f"{existing_reason} | {reason_suffix}" if existing_reason else reason_suffix

        if decision_text == "reject":
            rejected += 1
            working.at[index, "audit_status"] = "manually-rejected"
            working.at[index, "risk_flags"] = _append_flag(row.get("risk_flags"), "manual-rejected")
            working.at[index, "recommended_action"] = "Excluded by manual review; do not use for ranking unless the review decision is changed."
            continue

        if decision_text == "accept":
            has_verified_source = bool(_safe_text(decision.get("verified_source")))
            has_verified_snippet = bool(_safe_text(decision.get("verified_snippet")))
            if original_status == "conflicting" or "structure-conflict" in flags:
                conflict_blocked += 1
                working.at[index, "audit_status"] = "conflicting"
                working.at[index, "risk_flags"] = _append_flag(row.get("risk_flags"), "manual-accept-blocked-conflict")
                working.at[index, "recommended_action"] = "Manual accept is blocked until the structure conflict is resolved."
            elif has_verified_source and has_verified_snippet:
                accepted += 1
                working.at[index, "audit_status"] = "manually-accepted"
                working.at[index, "risk_flags"] = _append_flag(row.get("risk_flags"), "manual-accepted")
                working.at[index, "recommended_action"] = "Allowed through ranking gate with downgrade; keep reviewer, source, and snippet in validation notes."
            else:
                review += 1
                working.at[index, "audit_status"] = "needs-review"
                working.at[index, "risk_flags"] = _append_flag(row.get("risk_flags"), "manual-accept-missing-source")
                working.at[index, "recommended_action"] = "Manual accept requires verified_source and verified_snippet before ranking."
            continue

        review += 1
        working.at[index, "audit_status"] = "needs-review"
        working.at[index, "risk_flags"] = _append_flag(row.get("risk_flags"), "manual-review")
        working.at[index, "recommended_action"] = "Manual review is pending; keep out of default ranking until accepted with source and snippet."

    return working[AI_EVIDENCE_AUDIT_COLUMNS].reset_index(drop=True), {
        "status": "ok",
        "decision_rows": str(len(decisions_df)),
        "applied_rows": str(applied),
        "accepted_rows": str(accepted),
        "rejected_rows": str(rejected),
        "review_rows": str(review),
        "conflict_blocked_rows": str(conflict_blocked),
    }


def _matching_audit_row(
    audit_df: Optional[pd.DataFrame],
    *,
    chain: str,
    resid: int,
    evidence_type: str,
) -> Optional[pd.Series]:
    if audit_df is None or getattr(audit_df, "empty", True) or "resid" not in audit_df.columns:
        return None

    working = audit_df.copy()
    working["_resid_numeric"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["_resid_numeric"].notna()].copy()
    if working.empty:
        return None

    chain_text = _safe_text(chain)
    type_text = _safe_text(evidence_type).lower()
    working["_chain_text"] = working.get("chain", pd.Series("", index=working.index)).astype(str).str.strip()
    working["_type_text"] = working.get("evidence_type", pd.Series("", index=working.index)).astype(str).str.strip().str.lower()
    mask = working["_resid_numeric"].astype(int).eq(int(resid))
    if chain_text:
        mask = mask & ((working["_chain_text"] == chain_text) | (working["_chain_text"] == ""))
    if type_text:
        mask = mask & ((working["_type_text"] == type_text) | (working["_type_text"] == ""))

    candidates = working.loc[mask].copy()
    if candidates.empty:
        return None
    candidates["_specificity"] = candidates["_chain_text"].astype(bool).astype(int) + candidates["_type_text"].astype(bool).astype(int)
    return candidates.sort_values("_specificity", ascending=False).iloc[0]


def _decision_outcome_for_audit_row(decision: pd.Series, audit_row: Optional[pd.Series]) -> tuple[str, str, str]:
    decision_text = _safe_text(decision.get("review_decision")).lower()
    if audit_row is None:
        return (
            "unmatched",
            "No matching AI audit row was found for this chain/residue/evidence_type.",
            "Check chain ID, residue numbering, evidence_type, and whether the AI evidence row still exists.",
        )

    status = _safe_text(audit_row.get("audit_status")).lower()
    flags = _safe_text(audit_row.get("risk_flags")).lower()
    action = _safe_text(audit_row.get("recommended_action"), "Review the audit row before rerunning ranking.")

    if decision_text == "reject":
        if status == "manually-rejected":
            return "rejected", "Decision was applied and this AI residue is excluded from ranking.", action
        return "reject-not-reflected", "Reject decision matched a row but the audit status did not become manually-rejected.", action

    if decision_text == "accept":
        if status == "manually-accepted":
            return "accepted", "Decision was applied; row can pass the ranking gate with manual-review downgrade.", action
        if status == "conflicting" or "manual-accept-blocked-conflict" in flags:
            return "conflict-blocked", "Manual accept was blocked because structure numbering or residue identity still conflicts.", action
        if "manual-accept-missing-source" in flags:
            return "missing-source-or-snippet", "Manual accept was not applied because verified_source or verified_snippet is missing.", action
        return "not-accepted", "Accept decision matched a row but did not satisfy the audit gate.", action

    if status == "needs-review" or "manual-review" in flags:
        return "review-pending", "Decision leaves this row in manual review.", action
    return "applied-with-existing-status", "Decision matched an audit row; current audit status is unchanged or already resolved.", action


def _decision_key(row: pd.Series) -> tuple[str, Optional[int], str]:
    resid = _as_position(row.get("resid"))
    return (
        _safe_text(row.get("chain")),
        int(resid) if resid is not None else None,
        _safe_text(row.get("evidence_type")).lower(),
    )


def build_ai_review_decision_validation_table(
    decisions_df: Optional[pd.DataFrame],
    audit_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if decisions_df is None or getattr(decisions_df, "empty", True):
        return _empty_ai_review_decision_validation_df()

    duplicate_counts: dict[tuple[str, Optional[int], str], int] = {}
    duplicate_decisions: dict[tuple[str, Optional[int], str], set[str]] = {}
    for _, row in decisions_df.iterrows():
        key = _decision_key(row)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        duplicate_decisions.setdefault(key, set()).add(_safe_text(row.get("review_decision"), "review").lower())

    rows: list[dict[str, Any]] = []
    for row_index, decision in enumerate(decisions_df.itertuples(index=False), start=1):
        row = pd.Series(decision._asdict())
        resid = _as_position(row.get("resid"))
        chain = _safe_text(row.get("chain"))
        evidence_type = _safe_text(row.get("evidence_type"))
        decision_text = _safe_text(row.get("review_decision"), "review").lower()
        key = (chain, int(resid) if resid is not None else None, evidence_type.lower())
        audit_row = (
            _matching_audit_row(audit_df, chain=chain, resid=int(resid), evidence_type=evidence_type)
            if resid is not None
            else None
        )

        flags: list[str] = []
        reasons: list[str] = []
        fixes: list[str] = []
        can_apply = True

        if resid is None:
            flags.append("invalid-resid")
            reasons.append("Decision row has no valid residue number.")
            fixes.append("Fix the resid column.")
            can_apply = False
        if audit_row is None and resid is not None:
            flags.append("unmatched-audit")
            reasons.append("No matching AI audit row was found.")
            fixes.append("Check chain, residue numbering, evidence_type, and current AI evidence rows.")
            can_apply = False

        duplicate_count = duplicate_counts.get(key, 0)
        decision_values = duplicate_decisions.get(key, set())
        if duplicate_count > 1:
            if len(decision_values) > 1:
                flags.append("conflicting-duplicate")
                reasons.append("Multiple decisions for the same residue disagree.")
                fixes.append("Keep one decision row per chain/residue/evidence_type before upload.")
                can_apply = False
            else:
                flags.append("duplicate")
                reasons.append("Repeated identical decision for the same residue.")
                fixes.append("Remove duplicate rows to keep the audit trail clean.")

        if decision_text == "accept":
            if not _safe_text(row.get("verified_source")):
                flags.append("accept-missing-source")
                reasons.append("Accept decision lacks verified_source.")
                fixes.append("Add PMID, DOI, title, or another citable source.")
                can_apply = False
            if not _safe_text(row.get("verified_snippet")):
                flags.append("accept-missing-snippet")
                reasons.append("Accept decision lacks verified_snippet.")
                fixes.append("Add the exact source sentence supporting the residue.")
                can_apply = False
            if audit_row is not None:
                status = _safe_text(audit_row.get("audit_status")).lower()
                risks = _safe_text(audit_row.get("risk_flags")).lower()
                if status == "conflicting" or "structure-conflict" in risks:
                    flags.append("accept-blocked-by-structure-conflict")
                    reasons.append("Manual accept cannot override a structure numbering or residue-identity conflict.")
                    fixes.append("Resolve PDB chain/residue numbering or residue identity before accepting.")
                    can_apply = False

        validation_status = "ok"
        if any(flag in flags for flag in {"invalid-resid", "unmatched-audit", "conflicting-duplicate", "accept-blocked-by-structure-conflict"}):
            validation_status = "blocked"
        elif flags:
            validation_status = "warning"

        rows.append(
            {
                "row_index": row_index,
                "chain": chain,
                "resid": int(resid) if resid is not None else "",
                "evidence_type": evidence_type,
                "review_decision": decision_text,
                "validation_status": validation_status,
                "issue_flags": ", ".join(dict.fromkeys(flags)) if flags else "none",
                "can_apply": bool(can_apply),
                "matched_audit_status": _safe_text(audit_row.get("audit_status")) if audit_row is not None else "",
                "validation_reason": " ".join(reasons) if reasons else "Decision row is ready for audit application.",
                "required_fix": " ".join(dict.fromkeys(fixes)) if fixes else "none",
            }
        )

    if not rows:
        return _empty_ai_review_decision_validation_df()
    return pd.DataFrame(rows, columns=AI_REVIEW_DECISION_VALIDATION_COLUMNS).reset_index(drop=True)


def build_ai_review_decision_outcome_table(
    decisions_df: Optional[pd.DataFrame],
    audit_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if decisions_df is None or getattr(decisions_df, "empty", True):
        return _empty_ai_review_decision_outcome_df()

    rows: list[dict[str, Any]] = []
    for _, decision in decisions_df.iterrows():
        resid = _as_position(decision.get("resid"))
        if resid is None:
            rows.append(
                {
                    "chain": _safe_text(decision.get("chain")),
                    "resid": "",
                    "evidence_type": _safe_text(decision.get("evidence_type")),
                    "review_decision": _safe_text(decision.get("review_decision"), "review"),
                    "applied_status": "invalid-resid",
                    "current_audit_status": "",
                    "risk_flags": "",
                    "outcome_reason": "Decision row does not contain a valid residue number.",
                    "next_action": "Fix the resid column and upload the decision table again.",
                    "reviewer": _safe_text(decision.get("reviewer")),
                    "verified_source": _safe_text(decision.get("verified_source")),
                    "verified_snippet": _safe_text(decision.get("verified_snippet")),
                }
            )
            continue

        audit_row = _matching_audit_row(
            audit_df,
            chain=_safe_text(decision.get("chain")),
            resid=int(resid),
            evidence_type=_safe_text(decision.get("evidence_type")),
        )
        applied_status, reason, next_action = _decision_outcome_for_audit_row(decision, audit_row)
        rows.append(
            {
                "chain": _safe_text(decision.get("chain")),
                "resid": int(resid),
                "evidence_type": _safe_text(decision.get("evidence_type")),
                "review_decision": _safe_text(decision.get("review_decision"), "review"),
                "applied_status": applied_status,
                "current_audit_status": _safe_text(audit_row.get("audit_status")) if audit_row is not None else "",
                "risk_flags": _safe_text(audit_row.get("risk_flags")) if audit_row is not None else "",
                "outcome_reason": reason,
                "next_action": next_action,
                "reviewer": _safe_text(decision.get("reviewer")),
                "verified_source": _safe_text(decision.get("verified_source")),
                "verified_snippet": _safe_text(decision.get("verified_snippet")),
            }
        )

    if not rows:
        return _empty_ai_review_decision_outcome_df()
    return pd.DataFrame(rows, columns=AI_REVIEW_DECISION_OUTCOME_COLUMNS).reset_index(drop=True)


def _count_status(table: Optional[pd.DataFrame], column: str, status: str) -> int:
    if table is None or getattr(table, "empty", True) or column not in table.columns:
        return 0
    return int((table[column].astype(str).str.lower() == status).sum())


def build_ai_review_round_summary(
    decisions_df: Optional[pd.DataFrame],
    validation_df: Optional[pd.DataFrame],
    outcome_df: Optional[pd.DataFrame],
    rankable_ai_evidence_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    decision_rows = 0 if decisions_df is None or getattr(decisions_df, "empty", True) else int(len(decisions_df))
    if decision_rows == 0:
        return _empty_ai_review_round_summary_df()

    validation_ok = _count_status(validation_df, "validation_status", "ok")
    validation_warning = _count_status(validation_df, "validation_status", "warning")
    validation_blocked = _count_status(validation_df, "validation_status", "blocked")
    accepted = _count_status(outcome_df, "applied_status", "accepted")
    rejected = _count_status(outcome_df, "applied_status", "rejected")
    review_pending = _count_status(outcome_df, "applied_status", "review-pending")
    missing_source = _count_status(outcome_df, "applied_status", "missing-source-or-snippet")
    conflict_blocked = _count_status(outcome_df, "applied_status", "conflict-blocked")
    unmatched = _count_status(outcome_df, "applied_status", "unmatched")
    rankable_rows = 0 if rankable_ai_evidence_df is None or getattr(rankable_ai_evidence_df, "empty", True) else int(len(rankable_ai_evidence_df))

    if validation_blocked > 0:
        status = "blocked"
        reason = "At least one review decision failed validation before audit application."
        action = "Fix validation-blocked rows, especially conflicting duplicates, unmatched residues, or structure conflicts, then upload again."
    elif conflict_blocked > 0 or unmatched > 0 or missing_source > 0:
        status = "needs-review"
        reason = "Some decisions matched but still cannot safely promote AI evidence."
        action = "Resolve conflict-blocked, unmatched, or missing-source rows before treating the review round as complete."
    elif accepted > 0 or rejected > 0:
        status = "applied"
        reason = "Manual review decisions were applied without blocking issues."
        action = "Inspect ranking-gated AI evidence and AI ranking impact before using the updated pocket recommendation."
    else:
        status = "review-pending"
        reason = "Review rows exist, but no accept/reject outcome has been applied yet."
        action = "Fill accept/reject decisions with verified sources/snippets or keep rows in review."

    return pd.DataFrame(
        [
            {
                "decision_rows": decision_rows,
                "validation_ok_rows": validation_ok,
                "validation_warning_rows": validation_warning,
                "validation_blocked_rows": validation_blocked,
                "outcome_accepted_rows": accepted,
                "outcome_rejected_rows": rejected,
                "outcome_review_pending_rows": review_pending,
                "outcome_missing_source_rows": missing_source,
                "outcome_conflict_blocked_rows": conflict_blocked,
                "outcome_unmatched_rows": unmatched,
                "rankable_after_review_rows": rankable_rows,
                "review_round_status": status,
                "review_round_reason": reason,
                "recommended_action": action,
            }
        ],
        columns=AI_REVIEW_ROUND_SUMMARY_COLUMNS,
    )


def build_ai_review_ranking_delta(
    before_rankable_ai_evidence_df: Optional[pd.DataFrame],
    after_rankable_ai_evidence_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    before_keys = _residue_keys_from_evidence(before_rankable_ai_evidence_df)
    after_keys = _residue_keys_from_evidence(after_rankable_ai_evidence_df)

    promoted = after_keys - before_keys
    removed = before_keys - after_keys
    unchanged = before_keys & after_keys

    if not before_keys and not after_keys:
        status = "no-rankable-ai"
        reason = "No AI residue evidence passed the ranking gate before or after review."
        action = "Use validation/outcome tables to resolve source, mapping, or structure conflicts before relying on AI evidence."
    elif promoted and removed:
        status = "changed"
        reason = "Manual review both promoted and removed rankable AI residues."
        action = "Inspect promoted and removed residues before interpreting changes in pocket ranking."
    elif promoted:
        status = "promoted"
        reason = "Manual review allowed additional AI residues through the ranking gate."
        action = "Verify promoted residues are supported by source snippets before using them as active-site anchors."
    elif removed:
        status = "removed"
        reason = "Manual review removed AI residues from the ranking gate."
        action = "Check whether removed residues were rejected, conflicted, or under-sourced before comparing pocket ranks."
    else:
        status = "unchanged"
        reason = "Manual review did not change which AI residues passed the ranking gate."
        action = "Use audit/outcome details to decide whether more source evidence is needed."

    return pd.DataFrame(
        [
            {
                "before_rankable_rows": int(len(before_keys)),
                "after_rankable_rows": int(len(after_keys)),
                "promoted_rows": int(len(promoted)),
                "removed_rows": int(len(removed)),
                "unchanged_rows": int(len(unchanged)),
                "promoted_residues": _format_residue_keys(promoted),
                "removed_residues": _format_residue_keys(removed),
                "unchanged_residues": _format_residue_keys(unchanged),
                "review_effect_status": status,
                "review_effect_reason": reason,
                "recommended_action": action,
            }
        ],
        columns=AI_REVIEW_RANKING_DELTA_COLUMNS,
    )


def _markdown_table_preview(table: Optional[pd.DataFrame], columns: list[str], *, max_rows: int = 8) -> list[str]:
    if table is None or getattr(table, "empty", True):
        return ["No rows."]
    available = [column for column in columns if column in table.columns]
    if not available:
        return ["No displayable columns."]

    preview = table[available].head(max_rows).copy()
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for _, row in preview.iterrows():
        values = [
            _safe_text(row.get(column), "-").replace("|", "\\|").replace("\n", " ")
            for column in available
        ]
        lines.append("| " + " | ".join(values) + " |")
    if len(table) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(table)} rows.")
    return lines


def build_ai_review_round_report_markdown(
    summary_df: Optional[pd.DataFrame],
    validation_df: Optional[pd.DataFrame],
    outcome_df: Optional[pd.DataFrame],
    ranking_delta_df: Optional[pd.DataFrame],
    *,
    title: str = "AI review round report",
) -> str:
    lines = [
        f"# {title}",
        "",
        "This report summarizes one manual AI-evidence review round. Use it with the exported CSV files for audit and follow-up.",
        "",
    ]

    if summary_df is None or getattr(summary_df, "empty", True):
        return "\n".join(lines + ["No review round summary is available."])

    summary = summary_df.iloc[0]
    lines.extend(
        [
            "## Round status",
            "",
            f"- Status: `{_safe_text(summary.get('review_round_status'), '-')}`",
            f"- Decision rows: {_safe_text(summary.get('decision_rows'), '0')}",
            f"- Validation blocked rows: {_safe_text(summary.get('validation_blocked_rows'), '0')}",
            f"- Accepted rows: {_safe_text(summary.get('outcome_accepted_rows'), '0')}",
            f"- Rejected rows: {_safe_text(summary.get('outcome_rejected_rows'), '0')}",
            f"- Rankable after review: {_safe_text(summary.get('rankable_after_review_rows'), '0')}",
            f"- Reason: {_safe_text(summary.get('review_round_reason'), '-')}",
            f"- Recommended action: {_safe_text(summary.get('recommended_action'), '-')}",
            "",
        ]
    )

    if ranking_delta_df is not None and not getattr(ranking_delta_df, "empty", True):
        delta = ranking_delta_df.iloc[0]
        lines.extend(
            [
                "## Ranking delta",
                "",
                f"- Effect: `{_safe_text(delta.get('review_effect_status'), '-')}`",
                f"- Before rankable rows: {_safe_text(delta.get('before_rankable_rows'), '0')}",
                f"- After rankable rows: {_safe_text(delta.get('after_rankable_rows'), '0')}",
                f"- Promoted residues: {_safe_text(delta.get('promoted_residues'), 'none')}",
                f"- Removed residues: {_safe_text(delta.get('removed_residues'), 'none')}",
                f"- Unchanged residues: {_safe_text(delta.get('unchanged_residues'), 'none')}",
                "",
            ]
        )

    lines.extend(
        [
            "## Validation issues",
            "",
            *_markdown_table_preview(
                validation_df,
                ["row_index", "chain", "resid", "review_decision", "validation_status", "issue_flags", "required_fix"],
            ),
            "",
            "## Decision outcomes",
            "",
            *_markdown_table_preview(
                outcome_df,
                ["chain", "resid", "review_decision", "applied_status", "current_audit_status", "next_action"],
            ),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _table_len(table: Optional[pd.DataFrame]) -> int:
    return 0 if table is None or getattr(table, "empty", True) else int(len(table))


def _validation_manifest_status(validation_df: Optional[pd.DataFrame]) -> str:
    if validation_df is None or getattr(validation_df, "empty", True) or "validation_status" not in validation_df.columns:
        return "not-generated"
    statuses = validation_df["validation_status"].astype(str).str.lower()
    if (statuses == "blocked").any():
        return "blocked"
    if (statuses == "warning").any():
        return "warning"
    return "ok"


def _outcome_manifest_status(outcome_df: Optional[pd.DataFrame]) -> str:
    if outcome_df is None or getattr(outcome_df, "empty", True) or "applied_status" not in outcome_df.columns:
        return "not-generated"
    statuses = outcome_df["applied_status"].astype(str).str.lower()
    if statuses.isin(["conflict-blocked", "unmatched", "missing-source-or-snippet", "invalid-resid"]).any():
        return "needs-review"
    if statuses.isin(["accepted", "rejected"]).any():
        return "applied"
    return "review-pending"


def _artifact_integrity(data: bytes) -> tuple[int, str]:
    return len(data), hashlib.sha256(data).hexdigest()


def build_ai_review_artifact_manifest(
    *,
    review_queue_df: Optional[pd.DataFrame] = None,
    decision_template_df: Optional[pd.DataFrame] = None,
    normalized_decision_df: Optional[pd.DataFrame] = None,
    validation_df: Optional[pd.DataFrame] = None,
    round_summary_df: Optional[pd.DataFrame] = None,
    ranking_delta_df: Optional[pd.DataFrame] = None,
    outcome_df: Optional[pd.DataFrame] = None,
    round_report_markdown: str = "",
    bundle_readme_markdown: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(
        artifact_name: str,
        file_name: str,
        artifact_type: str,
        row_count: int,
        data: bytes,
        status: str,
        purpose: str,
        recommended_use: str,
    ) -> None:
        if row_count <= 0:
            return
        byte_size, digest = _artifact_integrity(data)
        rows.append(
            {
                "artifact_name": artifact_name,
                "file_name": file_name,
                "artifact_type": artifact_type,
                "row_count": int(row_count),
                "byte_size": int(byte_size),
                "sha256": digest,
                "status": status,
                "purpose": purpose,
                "recommended_use": recommended_use,
            }
        )

    add_row(
        "AI review queue",
        "ai_evidence_review_queue.csv",
        "csv",
        _table_len(review_queue_df),
        _csv_artifact_bytes(review_queue_df) if review_queue_df is not None and not getattr(review_queue_df, "empty", True) else b"",
        "pending-review",
        "Actionable list of AI evidence rows that still need curation.",
        "Start here to understand which residues need source, mapping, or structure fixes.",
    )
    add_row(
        "AI review decision template",
        "ai_review_decision_template.csv",
        "csv",
        _table_len(decision_template_df),
        _csv_artifact_bytes(decision_template_df) if decision_template_df is not None and not getattr(decision_template_df, "empty", True) else b"",
        "ready-to-fill",
        "Editable curator sheet generated from the review queue.",
        "Fill review_decision, reviewer, verified_source, and verified_snippet, then upload it back.",
    )
    add_row(
        "Normalized AI review decisions",
        "ai_review_decisions_normalized.csv",
        "csv",
        _table_len(normalized_decision_df),
        _csv_artifact_bytes(normalized_decision_df) if normalized_decision_df is not None and not getattr(normalized_decision_df, "empty", True) else b"",
        "uploaded",
        "Parser-normalized version of the uploaded curator decisions.",
        "Use it to confirm the upload was parsed with the expected chain, residue, and decision values.",
    )
    add_row(
        "AI review decision validation",
        "ai_review_decision_validation.csv",
        "csv",
        _table_len(validation_df),
        _csv_artifact_bytes(validation_df) if validation_df is not None and not getattr(validation_df, "empty", True) else b"",
        _validation_manifest_status(validation_df),
        "Pre-apply checks for duplicates, unmatched rows, missing evidence, and structure conflicts.",
        "Fix blocked rows before trusting ranking changes from the review round.",
    )
    summary_status = (
        _safe_text(round_summary_df.iloc[0].get("review_round_status"), "not-generated")
        if round_summary_df is not None and not getattr(round_summary_df, "empty", True)
        else "not-generated"
    )
    add_row(
        "AI review round summary",
        "ai_review_round_summary.csv",
        "csv",
        _table_len(round_summary_df),
        _csv_artifact_bytes(round_summary_df) if round_summary_df is not None and not getattr(round_summary_df, "empty", True) else b"",
        summary_status,
        "One-row decision summary for the uploaded review round.",
        "Use this as the primary go/no-go indicator for the manual review upload.",
    )
    delta_status = (
        _safe_text(ranking_delta_df.iloc[0].get("review_effect_status"), "not-generated")
        if ranking_delta_df is not None and not getattr(ranking_delta_df, "empty", True)
        else "not-generated"
    )
    add_row(
        "AI review ranking delta",
        "ai_review_ranking_delta.csv",
        "csv",
        _table_len(ranking_delta_df),
        _csv_artifact_bytes(ranking_delta_df) if ranking_delta_df is not None and not getattr(ranking_delta_df, "empty", True) else b"",
        delta_status,
        "Before/after comparison of ranking-gated AI residues.",
        "Use it to see which residues were promoted, removed, or unchanged by manual review.",
    )
    add_row(
        "AI review decision outcomes",
        "ai_review_decision_outcomes.csv",
        "csv",
        _table_len(outcome_df),
        _csv_artifact_bytes(outcome_df) if outcome_df is not None and not getattr(outcome_df, "empty", True) else b"",
        _outcome_manifest_status(outcome_df),
        "Per-row feedback after applying curator decisions.",
        "Use it to diagnose why individual decisions were accepted, blocked, unmatched, or left pending.",
    )
    report_text = "" if round_report_markdown is None else str(round_report_markdown)
    report_lines = len([line for line in report_text.splitlines() if line.strip()])
    add_row(
        "AI review round report",
        "ai_review_round_report.md",
        "markdown",
        report_lines,
        report_text.encode("utf-8") if _safe_text(report_text) else b"",
        "available",
        "Human-readable review-round report.",
        "Use it for team review, record keeping, or defense material.",
    )
    readme_text = "" if bundle_readme_markdown is None else str(bundle_readme_markdown)
    readme_lines = len([line for line in readme_text.splitlines() if line.strip()])
    add_row(
        "AI review bundle README",
        "ai_review_bundle_README.md",
        "markdown",
        readme_lines,
        readme_text.encode("utf-8") if _safe_text(readme_text) else b"",
        "available",
        "Offline guide for reading and verifying the AI review artifact bundle.",
        "Open this first after downloading the ZIP bundle.",
    )

    if not rows:
        return _empty_ai_review_artifact_manifest_df()
    return pd.DataFrame(rows, columns=AI_REVIEW_ARTIFACT_MANIFEST_COLUMNS).reset_index(drop=True)


def build_ai_review_bundle_readme_markdown(
    artifact_manifest_df: Optional[pd.DataFrame],
    *,
    title: str = "AI review artifact bundle README",
) -> str:
    lines = [
        f"# {title}",
        "",
        "This ZIP bundle contains AI evidence review artifacts for one review round.",
        "Recommended order: read this README, inspect the manifest, read the round report, then open detailed CSVs only as needed.",
        "",
        "## Recommended Reading Order",
        "",
        "1. `ai_review_artifact_manifest.csv` - artifact index, status, byte size, and SHA-256 hash.",
        "2. `ai_review_round_report.md` - human-readable review-round summary.",
        "3. `ai_review_round_summary.csv` - one-row go/no-go decision summary.",
        "4. `ai_review_ranking_delta.csv` - residues promoted, removed, or unchanged by manual review.",
        "5. `ai_review_decision_validation.csv` - pre-apply validation issues.",
        "6. `ai_review_decision_outcomes.csv` - per-row post-apply results.",
        "7. `ai_evidence_review_queue.csv` and `ai_review_decision_template.csv` - remaining work and next curation template.",
        "",
        "## Integrity Check",
        "",
        "Use `byte_size` and `sha256` in `ai_review_artifact_manifest.csv` to verify files after handoff or archival.",
        "The SHA-256 hash is computed from the exact exported bytes for each listed CSV/Markdown artifact.",
        "",
        "## Included Artifacts",
        "",
    ]
    if artifact_manifest_df is None or getattr(artifact_manifest_df, "empty", True):
        return "\n".join(lines + ["No artifact manifest rows are available."]) + "\n"

    manifest = artifact_manifest_df.copy()
    display_columns = ["file_name", "artifact_type", "row_count", "status", "purpose"]
    lines.extend(_markdown_table_preview(manifest, display_columns, max_rows=50))
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _csv_artifact_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode("utf-8-sig")


def build_ai_review_artifact_bundle_zip(
    *,
    review_queue_df: Optional[pd.DataFrame] = None,
    decision_template_df: Optional[pd.DataFrame] = None,
    normalized_decision_df: Optional[pd.DataFrame] = None,
    validation_df: Optional[pd.DataFrame] = None,
    round_summary_df: Optional[pd.DataFrame] = None,
    ranking_delta_df: Optional[pd.DataFrame] = None,
    outcome_df: Optional[pd.DataFrame] = None,
    artifact_manifest_df: Optional[pd.DataFrame] = None,
    round_report_markdown: str = "",
    bundle_readme_markdown: str = "",
) -> bytes:
    artifacts: list[tuple[str, bytes]] = []

    def add_csv(file_name: str, table: Optional[pd.DataFrame]) -> None:
        if table is not None and not getattr(table, "empty", True):
            artifacts.append((file_name, _csv_artifact_bytes(table)))

    add_csv("ai_evidence_review_queue.csv", review_queue_df)
    add_csv("ai_review_decision_template.csv", decision_template_df)
    add_csv("ai_review_decisions_normalized.csv", normalized_decision_df)
    add_csv("ai_review_decision_validation.csv", validation_df)
    add_csv("ai_review_round_summary.csv", round_summary_df)
    add_csv("ai_review_ranking_delta.csv", ranking_delta_df)
    add_csv("ai_review_decision_outcomes.csv", outcome_df)
    add_csv("ai_review_artifact_manifest.csv", artifact_manifest_df)
    if _safe_text(round_report_markdown):
        artifacts.append(("ai_review_round_report.md", round_report_markdown.encode("utf-8")))
    if _safe_text(bundle_readme_markdown):
        artifacts.append(("ai_review_bundle_README.md", bundle_readme_markdown.encode("utf-8")))

    if not artifacts:
        return b""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_name, data in artifacts:
            archive.writestr(file_name, data)
    return buffer.getvalue()


def verify_ai_review_artifact_bundle_zip(
    bundle_zip: bytes,
    artifact_manifest_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if not bundle_zip or artifact_manifest_df is None or getattr(artifact_manifest_df, "empty", True):
        return _empty_ai_review_bundle_verification_df()
    if not {"file_name", "byte_size", "sha256"}.issubset(set(artifact_manifest_df.columns)):
        return _empty_ai_review_bundle_verification_df()

    rows: list[dict[str, Any]] = []
    try:
        archive = zipfile.ZipFile(BytesIO(bundle_zip), mode="r")
    except zipfile.BadZipFile:
        return pd.DataFrame(
            [
                {
                    "file_name": "",
                    "expected_byte_size": "",
                    "actual_byte_size": "",
                    "expected_sha256": "",
                    "actual_sha256": "",
                    "verification_status": "invalid-zip",
                    "verification_reason": "Bundle bytes are not a readable ZIP archive.",
                }
            ],
            columns=AI_REVIEW_BUNDLE_VERIFICATION_COLUMNS,
        )

    with archive:
        zip_names = set(archive.namelist())
        expected_names: set[str] = set()
        for _, manifest_row in artifact_manifest_df.iterrows():
            file_name = _safe_text(manifest_row.get("file_name"))
            if not file_name:
                continue
            expected_names.add(file_name)
            expected_size = int(_safe_float(manifest_row.get("byte_size"), 0))
            expected_hash = _safe_text(manifest_row.get("sha256"))
            if file_name not in zip_names:
                rows.append(
                    {
                        "file_name": file_name,
                        "expected_byte_size": expected_size,
                        "actual_byte_size": "",
                        "expected_sha256": expected_hash,
                        "actual_sha256": "",
                        "verification_status": "missing",
                        "verification_reason": "File listed in manifest is missing from the ZIP bundle.",
                    }
                )
                continue
            data = archive.read(file_name)
            actual_size, actual_hash = _artifact_integrity(data)
            status = "verified"
            reason = "File size and SHA-256 match the manifest."
            if actual_size != expected_size:
                status = "size-mismatch"
                reason = "File byte size does not match the manifest."
            elif expected_hash and actual_hash != expected_hash:
                status = "hash-mismatch"
                reason = "File SHA-256 does not match the manifest."
            rows.append(
                {
                    "file_name": file_name,
                    "expected_byte_size": expected_size,
                    "actual_byte_size": actual_size,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "verification_status": status,
                    "verification_reason": reason,
                }
            )

        ignored_unlisted = {"ai_review_artifact_manifest.csv"}
        for file_name in sorted(zip_names - expected_names - ignored_unlisted):
            data = archive.read(file_name)
            actual_size, actual_hash = _artifact_integrity(data)
            rows.append(
                {
                    "file_name": file_name,
                    "expected_byte_size": "",
                    "actual_byte_size": actual_size,
                    "expected_sha256": "",
                    "actual_sha256": actual_hash,
                    "verification_status": "extra",
                    "verification_reason": "File is present in the ZIP bundle but is not listed in the manifest.",
                }
            )

    if not rows:
        return _empty_ai_review_bundle_verification_df()
    return pd.DataFrame(rows, columns=AI_REVIEW_BUNDLE_VERIFICATION_COLUMNS).reset_index(drop=True)


def build_ai_review_bundle_verification_summary(
    verification_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if verification_df is None or getattr(verification_df, "empty", True) or "verification_status" not in verification_df.columns:
        return _empty_ai_review_bundle_verification_summary_df()

    checked_files = int(len(verification_df))
    verified_files = _count_status(verification_df, "verification_status", "verified")
    missing_files = _count_status(verification_df, "verification_status", "missing")
    size_mismatch_files = _count_status(verification_df, "verification_status", "size-mismatch")
    hash_mismatch_files = _count_status(verification_df, "verification_status", "hash-mismatch")
    extra_files = _count_status(verification_df, "verification_status", "extra")
    invalid_zip_rows = _count_status(verification_df, "verification_status", "invalid-zip")
    failed_files = checked_files - verified_files

    failed_rows = verification_df[verification_df["verification_status"].astype(str).str.lower() != "verified"].copy()
    failed_names = [
        _safe_text(value)
        for value in failed_rows.get("file_name", pd.Series(dtype=object)).tolist()
        if _safe_text(value)
    ]

    if invalid_zip_rows > 0:
        status = "failed"
        action = "Regenerate the ZIP bundle; the current bytes are not a valid archive."
    elif failed_files > 0:
        status = "failed"
        action = "Do not hand off this bundle until missing, extra, size-mismatch, or hash-mismatch rows are resolved."
    else:
        status = "verified"
        action = "Bundle is ready for handoff; keep the manifest and verification summary with the archive."

    return pd.DataFrame(
        [
            {
                "checked_files": checked_files,
                "verified_files": verified_files,
                "failed_files": failed_files,
                "missing_files": missing_files,
                "size_mismatch_files": size_mismatch_files,
                "hash_mismatch_files": hash_mismatch_files,
                "extra_files": extra_files,
                "invalid_zip_rows": invalid_zip_rows,
                "verification_status": status,
                "failed_file_names": ", ".join(failed_names) if failed_names else "none",
                "recommended_action": action,
            }
        ],
        columns=AI_REVIEW_BUNDLE_VERIFICATION_SUMMARY_COLUMNS,
    )


def build_ai_review_bundle_certificate_markdown(
    bundle_zip: bytes,
    verification_summary_df: Optional[pd.DataFrame],
    artifact_manifest_df: Optional[pd.DataFrame],
    *,
    title: str = "AI review bundle handoff certificate",
) -> str:
    if not bundle_zip:
        return ""

    bundle_size, bundle_hash = _artifact_integrity(bundle_zip)
    manifest_rows = 0 if artifact_manifest_df is None or getattr(artifact_manifest_df, "empty", True) else int(len(artifact_manifest_df))

    summary_row = (
        verification_summary_df.iloc[0]
        if verification_summary_df is not None and not getattr(verification_summary_df, "empty", True)
        else pd.Series(dtype=object)
    )
    status = _safe_text(summary_row.get("verification_status"), "not-verified")
    checked_files = _safe_text(summary_row.get("checked_files"), "0")
    verified_files = _safe_text(summary_row.get("verified_files"), "0")
    failed_files = _safe_text(summary_row.get("failed_files"), "0")
    failed_names = _safe_text(summary_row.get("failed_file_names"), "none") or "none"
    action = _safe_text(
        summary_row.get("recommended_action"),
        "Run bundle verification before handoff.",
    )

    lines = [
        f"# {title}",
        "",
        "This certificate records the ZIP bundle identity and verification result for handoff/archive.",
        "",
        "## Bundle identity",
        "",
        "- File: `ai_review_artifacts.zip`",
        f"- Byte size: {bundle_size}",
        f"- SHA-256: `{bundle_hash}`",
        "",
        "## Verification summary",
        "",
        f"- Status: `{status}`",
        f"- Checked files: {checked_files}",
        f"- Verified files: {verified_files}",
        f"- Failed files: {failed_files}",
        f"- Failed file names: {failed_names}",
        f"- Manifest rows: {manifest_rows}",
        f"- Recommended action: {action}",
        "",
        "## How to use",
        "",
        "- Keep this certificate next to the ZIP bundle.",
        "- Recompute the ZIP SHA-256 before handoff or archival and compare it with this certificate.",
        "- Open `ai_review_bundle_README.md` inside the ZIP for artifact reading order.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _is_ai_source(row: pd.Series) -> bool:
    source = str(row.get("evidence_source") or "").lower()
    method = str(row.get("mapping_method") or "").lower()
    note = str(row.get("evidence_note") or "").lower()
    return "ai" in source or method.startswith("ai-") or "ai_confidence=" in note


def _is_conservation_source(row: pd.Series) -> bool:
    source = str(row.get("evidence_source") or "").lower()
    evidence_type = str(row.get("evidence_type") or "").lower()
    method = str(row.get("mapping_method") or "").lower()
    return "conservation" in source or "consurf" in source or "conservation" in evidence_type or "conservation" in method


def _normalize_consensus_evidence(table: Optional[pd.DataFrame], *, source_scope: str) -> pd.DataFrame:
    if table is None or getattr(table, "empty", True) or "resid" not in table.columns:
        return pd.DataFrame(columns=[*EVIDENCE_COLUMNS, "_source_scope"])

    working = table.copy()
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=[*EVIDENCE_COLUMNS, "_source_scope"])

    working["resid"] = working["resid"].astype(int)
    defaults = {
        "chain": "",
        "evidence_source": source_scope or "evidence",
        "evidence_type": "Functional site",
        "evidence_score": 0.55,
        "evidence_note": "",
        "uniprot_resid": working["resid"],
        "mapping_level": "weak",
        "mapping_confidence": 0.3,
        "mapping_method": source_scope or "unknown",
    }
    for column, value in defaults.items():
        if column not in working.columns:
            working[column] = value
    working["chain"] = working["chain"].astype(str).str.strip()
    working["evidence_source"] = working["evidence_source"].astype(str).str.strip()
    working["evidence_type"] = working["evidence_type"].astype(str).str.strip()
    working["evidence_note"] = working["evidence_note"].astype(str).str.strip()
    working["mapping_method"] = working["mapping_method"].astype(str).str.strip()
    working["mapping_level"] = working["mapping_level"].astype(str).str.strip().str.lower()
    working.loc[~working["mapping_level"].isin({"exact", "weak"}), "mapping_level"] = "weak"
    working["evidence_score"] = pd.to_numeric(working["evidence_score"], errors="coerce").fillna(0.55).clip(0.0, 1.0)
    working["mapping_confidence"] = pd.to_numeric(working["mapping_confidence"], errors="coerce").fillna(0.3).clip(0.0, 1.0)
    working["uniprot_resid"] = pd.to_numeric(working["uniprot_resid"], errors="coerce").fillna(working["resid"]).astype(int)
    working = ensure_evidence_columns(working)
    working["_source_scope"] = source_scope
    return working[[*EVIDENCE_COLUMNS, "_source_scope"]].copy()


def _has_residue_key(keys: set[tuple[str, int]], chain: str, resid: int) -> bool:
    cleaned_chain = str(chain or "").strip()
    if (cleaned_chain, int(resid)) in keys or ("", int(resid)) in keys:
        return True
    if not cleaned_chain:
        return any(key_resid == int(resid) for _key_chain, key_resid in keys)
    return False


def _matching_audit_rows_for_residue(
    audit_df: Optional[pd.DataFrame],
    *,
    chain: str,
    resid: int,
) -> pd.DataFrame:
    if audit_df is None or getattr(audit_df, "empty", True) or "resid" not in audit_df.columns:
        return pd.DataFrame(columns=AI_EVIDENCE_AUDIT_COLUMNS)
    working = audit_df.copy()
    working["_resid_numeric"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["_resid_numeric"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=AI_EVIDENCE_AUDIT_COLUMNS)
    working["_chain_text"] = working.get("chain", pd.Series("", index=working.index)).astype(str).str.strip()
    cleaned_chain = str(chain or "").strip()
    mask = working["_resid_numeric"].astype(int).eq(int(resid))
    if cleaned_chain:
        mask = mask & ((working["_chain_text"] == cleaned_chain) | (working["_chain_text"] == ""))
    return working.loc[mask].drop(columns=["_resid_numeric", "_chain_text"], errors="ignore").copy()


def _consensus_tier_and_action(
    *,
    non_ai_rows: int,
    conservation_rows: int,
    ai_rows: int,
    rankable_ai_rows: int,
    exact_rows: int,
    source_count: int,
    audit_statuses: set[str],
) -> tuple[str, str, str]:
    has_blocking_ai = bool(audit_statuses & {"conflicting", "unsupported", "manually-rejected"})
    has_review_ai = bool(audit_statuses & {"needs-review"})
    if non_ai_rows > 0 and source_count >= 2 and exact_rows > 0 and not has_blocking_ai:
        return (
            "validated-anchor",
            "ranked-functional",
            "Use as a high-priority pocket anchor; it has exact mapping and cross-source support.",
        )
    if non_ai_rows > 0 and exact_rows > 0:
        return (
            "supported-anchor",
            "ranked-functional",
            "Use as a functional pocket anchor, but keep source/mapping details visible in review.",
        )
    if rankable_ai_rows > 0 and not has_blocking_ai:
        return (
            "ai-supported-anchor",
            "ranked-ai",
            "AI evidence passed the audit gate; verify cited source before wet-lab or docking decisions.",
        )
    if conservation_rows > 0 and conservation_rows == (non_ai_rows + conservation_rows + ai_rows):
        return (
            "conservation-context",
            "context-only",
            "Use conservation as supporting context only; add catalytic/binding evidence before promoting.",
        )
    if ai_rows > 0 and rankable_ai_rows == 0 and non_ai_rows == 0:
        return (
            "blocked-ai",
            "not-ranked-ai-review",
            "Do not use for ranking until AI audit issues are resolved with citation, snippet, and mapping checks.",
        )
    if exact_rows == 0:
        return (
            "weak-mapping",
            "review-before-ranking",
            "Resolve UniProt/PDB numbering or chain mapping before treating this residue as a pocket anchor.",
        )
    if has_review_ai:
        return (
            "review-needed",
            "review-before-ranking",
            "Review AI evidence and independent support before increasing ranking confidence.",
        )
    return (
        "evidence-context",
        "context-only",
        "Keep as contextual evidence and collect an independent functional source before promotion.",
    )


def build_residue_evidence_consensus(
    evidence_df: Optional[pd.DataFrame],
    *,
    ai_evidence_df: Optional[pd.DataFrame] = None,
    ai_audit_df: Optional[pd.DataFrame] = None,
    rankable_ai_evidence_df: Optional[pd.DataFrame] = None,
    conservation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    frames = [
        _normalize_consensus_evidence(evidence_df, source_scope="ranking-evidence"),
        _normalize_consensus_evidence(ai_evidence_df, source_scope="ai-input"),
        _normalize_consensus_evidence(conservation_df, source_scope="conservation"),
    ]
    combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame(columns=[*EVIDENCE_COLUMNS, "_source_scope"])
    if not combined.empty:
        combined = combined.drop_duplicates(
            subset=["chain", "resid", "evidence_source", "evidence_type", "mapping_level", "uniprot_resid", "evidence_note"],
            keep="first",
        ).reset_index(drop=True)

    known_keys = {
        (_safe_text(row.get("chain")), int(row.get("resid")))
        for _, row in combined.iterrows()
        if pd.notna(row.get("resid"))
    }
    audit_rows: list[dict[str, Any]] = []
    if ai_audit_df is not None and not getattr(ai_audit_df, "empty", True) and "resid" in ai_audit_df.columns:
        for _, audit_row in ai_audit_df.iterrows():
            try:
                resid = int(float(audit_row.get("resid")))
            except (TypeError, ValueError):
                continue
            chain = _safe_text(audit_row.get("chain"))
            if _has_residue_key(known_keys, chain, resid):
                continue
            audit_rows.append(
                {
                    "chain": chain,
                    "resid": resid,
                    "evidence_source": "AI audit",
                    "evidence_type": _safe_text(audit_row.get("evidence_type"), "AI extracted residue"),
                    "evidence_score": _bounded_score(audit_row.get("ai_score"), 0.45),
                    "evidence_note": _safe_text(audit_row.get("audit_reason")),
                    "uniprot_resid": resid,
                    "mapping_level": _safe_text(audit_row.get("mapping_level"), "weak").lower(),
                    "mapping_confidence": _bounded_score(audit_row.get("mapping_confidence"), 0.3),
                    "mapping_method": "ai-audit-only",
                    "_source_scope": "ai-audit",
                }
            )
    if audit_rows:
        combined = pd.concat([combined, pd.DataFrame(audit_rows)], ignore_index=True)

    if combined.empty:
        return _empty_residue_evidence_consensus_df()

    rankable_ai_keys = _residue_keys_from_evidence(rankable_ai_evidence_df)
    rows: list[dict[str, Any]] = []
    group_columns = ["chain", "resid"]
    combined["_tier_mapping_rank"] = combined["mapping_level"].map({"exact": 0, "weak": 1}).fillna(2)
    combined = combined.sort_values(["_tier_mapping_rank", "evidence_score", "mapping_confidence"], ascending=[True, False, False])
    for (chain, resid), group in combined.groupby(group_columns, dropna=False):
        chain_text = _safe_text(chain)
        resid_int = int(resid)
        audit_matches = _matching_audit_rows_for_residue(ai_audit_df, chain=chain_text, resid=resid_int)
        audit_statuses = {
            _safe_text(value).lower()
            for value in audit_matches.get("audit_status", pd.Series(dtype=object)).tolist()
            if _safe_text(value)
        }
        risk_values = [
            _safe_text(value)
            for value in audit_matches.get("risk_flags", pd.Series(dtype=object)).tolist()
            if _safe_text(value) and _safe_text(value).lower() != "none"
        ]

        ai_mask = group.apply(_is_ai_source, axis=1)
        conservation_mask = group.apply(_is_conservation_source, axis=1)
        rankable_mask = group.apply(
            lambda row: bool(_is_ai_source(row) and _has_residue_key(rankable_ai_keys, _safe_text(row.get("chain")), int(row.get("resid")))),
            axis=1,
        )
        non_ai_mask = ~ai_mask & ~conservation_mask
        mapping_level = group["mapping_level"].astype(str).str.lower()
        evidence_score = pd.to_numeric(group["evidence_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        mapping_confidence = pd.to_numeric(group["mapping_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        sources = sorted({_safe_text(value) for value in group["evidence_source"].tolist() if _safe_text(value)})
        functional_sources = sorted(
            {
                _safe_text(row.get("evidence_source"))
                for _, row in group[~conservation_mask].iterrows()
                if _safe_text(row.get("evidence_source"))
            }
        )
        evidence_types = sorted({_safe_text(value) for value in group["evidence_type"].tolist() if _safe_text(value)})
        uniprot_values = [
            int(value)
            for value in pd.to_numeric(group["uniprot_resid"], errors="coerce").dropna().astype(int).tolist()
        ]
        uniprot_resid = uniprot_values[0] if uniprot_values else resid_int

        ai_rows = int(ai_mask.sum())
        rankable_ai_rows = int(rankable_mask.sum())
        non_ai_rows = int(non_ai_mask.sum())
        conservation_rows = int(conservation_mask.sum())
        exact_rows = int(mapping_level.eq("exact").sum())
        weak_rows = int(mapping_level.eq("weak").sum())
        source_count = int(len(sources))
        functional_source_count = int(len(functional_sources))

        consensus_tier, ranking_status, recommended_action = _consensus_tier_and_action(
            non_ai_rows=non_ai_rows,
            conservation_rows=conservation_rows,
            ai_rows=ai_rows,
            rankable_ai_rows=rankable_ai_rows,
            exact_rows=exact_rows,
            source_count=source_count,
            audit_statuses=audit_statuses,
        )

        source_bonus = min(0.18, max(0, functional_source_count - 1) * 0.08 + (0.03 if conservation_rows > 0 else 0.0))
        exact_bonus = 0.10 if exact_rows > 0 else 0.0
        rankable_ai_bonus = 0.05 if rankable_ai_rows > 0 else 0.0
        non_ai_bonus = 0.05 if non_ai_rows > 0 else 0.0
        audit_penalty = 0.0
        if audit_statuses & {"conflicting", "manually-rejected"}:
            audit_penalty = 0.20
        elif audit_statuses & {"unsupported", "needs-review"} and rankable_ai_rows == 0:
            audit_penalty = 0.12
        consensus_score = (
            0.45 * float(evidence_score.max())
            + 0.25 * float(mapping_confidence.max())
            + source_bonus
            + exact_bonus
            + rankable_ai_bonus
            + non_ai_bonus
            - audit_penalty
        )
        if consensus_tier == "conservation-context":
            consensus_score = min(consensus_score, 0.65)
        if consensus_tier == "blocked-ai":
            consensus_score = min(consensus_score, 0.45)
        if consensus_tier == "weak-mapping":
            consensus_score = min(consensus_score, 0.68)
        consensus_score = round(float(min(1.0, max(0.0, consensus_score))), 3)

        rows.append(
            {
                "chain": chain_text,
                "resid": resid_int,
                "residue_anchor": f"{chain_text}:{resid_int}" if chain_text else str(resid_int),
                "uniprot_resid": int(uniprot_resid),
                "evidence_rows": int(len(group)),
                "source_count": source_count,
                "functional_source_count": functional_source_count,
                "ai_rows": ai_rows,
                "rankable_ai_rows": rankable_ai_rows,
                "non_ai_rows": non_ai_rows,
                "conservation_rows": conservation_rows,
                "exact_rows": exact_rows,
                "weak_rows": weak_rows,
                "best_evidence_score": round(float(evidence_score.max()), 3),
                "best_mapping_confidence": round(float(mapping_confidence.max()), 3),
                "consensus_score": consensus_score,
                "consensus_tier": consensus_tier,
                "ranking_status": ranking_status,
                "evidence_sources": ", ".join(sources) if sources else "unknown",
                "evidence_types": ", ".join(evidence_types) if evidence_types else "unknown",
                "ai_audit_statuses": ", ".join(sorted(audit_statuses)) if audit_statuses else "none",
                "risk_flags": ", ".join(dict.fromkeys(risk_values)) if risk_values else "none",
                "recommended_action": recommended_action,
            }
        )

    if not rows:
        return _empty_residue_evidence_consensus_df()
    tier_rank = {
        "validated-anchor": 0,
        "supported-anchor": 1,
        "ai-supported-anchor": 2,
        "review-needed": 3,
        "weak-mapping": 4,
        "conservation-context": 5,
        "blocked-ai": 6,
        "evidence-context": 7,
    }
    consensus = pd.DataFrame(rows, columns=RESIDUE_EVIDENCE_CONSENSUS_COLUMNS)
    consensus["_tier_rank"] = consensus["consensus_tier"].map(tier_rank).fillna(99)
    consensus = consensus.sort_values(
        ["consensus_score", "_tier_rank", "resid", "chain"],
        ascending=[False, True, True, True],
    ).drop(columns="_tier_rank").reset_index(drop=True)
    return consensus[RESIDUE_EVIDENCE_CONSENSUS_COLUMNS]


def _matching_reference_rows(reference_df: Optional[pd.DataFrame], chain: str, resid: int) -> pd.DataFrame:
    if reference_df is None or getattr(reference_df, "empty", True) or "resid" not in reference_df.columns:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    working = reference_df.copy()
    working = working[~working.apply(_is_ai_source, axis=1)].copy()
    if working.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    working["_resid_numeric"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["_resid_numeric"].notna()].copy()
    working["_chain_text"] = working.get("chain", pd.Series("", index=working.index)).astype(str).str.strip()
    cleaned_chain = str(chain or "").strip()
    mask = working["_resid_numeric"].astype(int).eq(int(resid))
    if cleaned_chain:
        mask = mask & ((working["_chain_text"] == cleaned_chain) | (working["_chain_text"] == ""))
    return working.loc[mask].drop(columns=["_resid_numeric", "_chain_text"], errors="ignore").copy()


def build_ai_evidence_audit_table(
    ai_evidence_df: Optional[pd.DataFrame],
    reference_evidence_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if ai_evidence_df is None or getattr(ai_evidence_df, "empty", True) or "resid" not in ai_evidence_df.columns:
        return _empty_ai_audit_df()

    rows: list[dict[str, Any]] = []
    for _, row in ai_evidence_df.iterrows():
        try:
            resid = int(float(row.get("resid")))
        except (TypeError, ValueError):
            continue
        chain = str(row.get("chain") or "").strip()
        evidence_type = _safe_text(row.get("evidence_type"), "AI extracted residue")
        evidence_score = _bounded_score(row.get("evidence_score"), 0.0)
        mapping_level = _safe_text(row.get("mapping_level"), "weak").lower()
        mapping_confidence = _bounded_score(row.get("mapping_confidence"), 0.0)
        method = _safe_text(row.get("mapping_method")).lower()
        note = _safe_text(row.get("evidence_note"))
        lowered_note = note.lower()
        reference_rows = _matching_reference_rows(reference_evidence_df, chain, resid)
        overlap_sources = sorted(
            {
                str(value).strip()
                for value in reference_rows.get("evidence_source", pd.Series(dtype=object)).tolist()
                if str(value).strip()
            }
        )

        flags: list[str] = []
        has_snippet = "snippet=" in lowered_note and not lowered_note.rstrip().endswith("snippet=")
        has_source = any(token in lowered_note for token in ("pmid=", "doi=", "title="))
        manual_review = "manual_review=true" in lowered_note or "review" in method
        structure_conflict = "identity-mismatch" in method or "residue-missing" in method or "structure_residue_mismatch=" in lowered_note
        if not has_snippet:
            flags.append("missing-source-snippet")
        if not has_source:
            flags.append("missing-source-id")
        if manual_review:
            flags.append("manual-review")
        if mapping_level != "exact":
            flags.append("weak-mapping")
        if mapping_confidence < 0.55:
            flags.append("low-mapping-confidence")
        if evidence_score < 0.55:
            flags.append("low-ai-confidence")
        if reference_rows.empty:
            flags.append("no-independent-support")
        if structure_conflict:
            flags.append("structure-conflict")

        if structure_conflict:
            status = "conflicting"
            reason = "AI residue conflicts with structure numbering or residue identity."
            action = "Do not promote this residue; verify chain, residue identity, and numbering manually."
        elif not reference_rows.empty:
            status = "supported"
            reason = "AI residue overlaps non-AI evidence sources."
            action = "Keep as supporting evidence, but cite the non-AI source in validation notes."
        elif mapping_level == "exact" and not manual_review and has_snippet and has_source:
            status = "structure-verified"
            reason = "AI residue has source text and passed structure-numbering verification, but lacks independent database support."
            action = "Use as provisional evidence and seek M-CSA, UniProt, or literature cross-support."
        elif evidence_score < 0.55 or not has_snippet or not has_source:
            status = "unsupported"
            reason = "AI residue lacks enough source support for ranking confidence."
            action = "Keep out of high-confidence interpretation until a source sentence and citation are supplied."
        else:
            status = "needs-review"
            reason = "AI residue is potentially useful but still needs manual evidence or mapping review."
            action = "Review the evidence snippet, citation, chain, and numbering before accepting."

        rows.append(
            {
                "chain": chain,
                "resid": resid,
                "evidence_type": evidence_type,
                "ai_score": round(float(evidence_score), 3),
                "mapping_level": mapping_level,
                "mapping_confidence": round(float(mapping_confidence), 3),
                "audit_status": status,
                "overlap_sources": ", ".join(overlap_sources) if overlap_sources else "none",
                "risk_flags": ", ".join(dict.fromkeys(flags)) if flags else "none",
                "audit_reason": reason,
                "recommended_action": action,
            }
        )

    if not rows:
        return _empty_ai_audit_df()
    return pd.DataFrame(rows, columns=AI_EVIDENCE_AUDIT_COLUMNS).sort_values(
        ["audit_status", "chain", "resid"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def filter_ai_evidence_for_ranking(
    ai_evidence_df: Optional[pd.DataFrame],
    audit_df: Optional[pd.DataFrame],
    *,
    allow_review: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if ai_evidence_df is None or getattr(ai_evidence_df, "empty", True) or "resid" not in ai_evidence_df.columns:
        return _empty_ai_evidence_df(), {
            "status": "empty",
            "input_rows": "0",
            "accepted_rows": "0",
            "excluded_rows": "0",
        }
    if audit_df is None or getattr(audit_df, "empty", True) or "audit_status" not in audit_df.columns:
        return _empty_ai_evidence_df(), {
            "status": "no-audit",
            "input_rows": str(len(ai_evidence_df)),
            "accepted_rows": "0",
            "excluded_rows": str(len(ai_evidence_df)),
        }

    allowed_statuses = {"supported", "structure-verified", "manually-accepted"}
    if allow_review:
        allowed_statuses.add("needs-review")

    working = ai_evidence_df.copy()
    working["_row_order"] = range(len(working))
    working["_resid_numeric"] = pd.to_numeric(working["resid"], errors="coerce")
    working["_chain_text"] = working.get("chain", pd.Series("", index=working.index)).astype(str).str.strip()
    working["_type_text"] = working.get("evidence_type", pd.Series("", index=working.index)).astype(str).str.strip()

    audit = audit_df.copy()
    audit["_resid_numeric"] = pd.to_numeric(audit["resid"], errors="coerce")
    audit["_chain_text"] = audit.get("chain", pd.Series("", index=audit.index)).astype(str).str.strip()
    audit["_type_text"] = audit.get("evidence_type", pd.Series("", index=audit.index)).astype(str).str.strip()
    audit = audit[audit["_resid_numeric"].notna()].copy()
    audit["_audit_status"] = audit["audit_status"].astype(str).str.strip().str.lower()
    audit = audit[["_chain_text", "_resid_numeric", "_type_text", "_audit_status", "risk_flags", "audit_reason"]]

    merged = working.merge(
        audit,
        on=["_chain_text", "_resid_numeric", "_type_text"],
        how="left",
    )
    merged["_audit_status"] = merged["_audit_status"].fillna("no-audit")
    accepted = merged[merged["_audit_status"].isin(allowed_statuses)].copy()
    if accepted.empty:
        return _empty_ai_evidence_df(), {
            "status": "empty-after-audit",
            "input_rows": str(len(ai_evidence_df)),
            "accepted_rows": "0",
            "excluded_rows": str(len(ai_evidence_df)),
            "allow_review": str(bool(allow_review)).lower(),
        }

    accepted["evidence_note"] = accepted["evidence_note"].astype(str) + " | ai_audit_status=" + accepted["_audit_status"].astype(str)

    review_mask = accepted["_audit_status"] == "needs-review"
    manual_mask = accepted["_audit_status"] == "manually-accepted"
    accepted.loc[review_mask, "evidence_score"] = pd.to_numeric(
        accepted.loc[review_mask, "evidence_score"],
        errors="coerce",
    ).fillna(0.0).clip(upper=0.58)
    accepted.loc[review_mask, "mapping_confidence"] = pd.to_numeric(
        accepted.loc[review_mask, "mapping_confidence"],
        errors="coerce",
    ).fillna(0.0).clip(upper=0.54)
    accepted.loc[review_mask, "mapping_level"] = "weak"
    accepted.loc[review_mask, "mapping_method"] = (
        accepted.loc[review_mask, "mapping_method"].map(lambda value: _safe_text(value, "ai-evidence") + "-audit-review")
    )
    accepted.loc[manual_mask, "evidence_score"] = pd.to_numeric(
        accepted.loc[manual_mask, "evidence_score"],
        errors="coerce",
    ).fillna(0.0).clip(upper=0.68)
    accepted.loc[manual_mask, "mapping_confidence"] = pd.to_numeric(
        accepted.loc[manual_mask, "mapping_confidence"],
        errors="coerce",
    ).fillna(0.0).clip(upper=0.70)
    accepted.loc[manual_mask, "mapping_level"] = "weak"
    accepted.loc[manual_mask, "mapping_method"] = (
        accepted.loc[manual_mask, "mapping_method"].map(lambda value: _safe_text(value, "ai-evidence") + "-manual-review")
    )

    status_counts = accepted["_audit_status"].value_counts().to_dict()
    result = accepted.sort_values("_row_order").drop(
        columns=[
            "_row_order",
            "_resid_numeric",
            "_chain_text",
            "_type_text",
            "_audit_status",
            "risk_flags",
            "audit_reason",
        ],
        errors="ignore",
    )
    return ensure_evidence_columns(result).reset_index(drop=True), {
        "status": "ok",
        "input_rows": str(len(ai_evidence_df)),
        "accepted_rows": str(len(result)),
        "excluded_rows": str(max(0, len(ai_evidence_df) - len(result))),
        "allow_review": str(bool(allow_review)).lower(),
        "supported_rows": str(int(status_counts.get("supported", 0))),
        "structure_verified_rows": str(int(status_counts.get("structure-verified", 0))),
        "manually_accepted_rows": str(int(status_counts.get("manually-accepted", 0))),
        "review_rows": str(int(status_counts.get("needs-review", 0))),
    }


def _post_json(url: str, payload: dict[str, Any], *, api_key: str = "", timeout_sec: float = 30.0) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(req, timeout=float(timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def _message_text_from_response(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
        if isinstance(first.get("text"), str):
            return first["text"]
    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        if chunks:
            return "\n".join(chunks)
    return ""


def fetch_ai_residue_evidence(
    literature_or_notes: str,
    *,
    api_url: str = "",
    api_key: str = "",
    model: str = "",
    chain_hint: str = "",
    protein_name: str = "",
    accession: str = "",
    pdb_id: str = "",
    ec_number: str = "",
    triage_context: str = "",
    min_confidence: float = 0.35,
    assume_structure_numbering: bool = False,
    pdb_text: Optional[str] = None,
    timeout_sec: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned_url = _safe_text(api_url or os.getenv("AI_EVIDENCE_API_URL"))
    cleaned_model = _safe_text(model or os.getenv("AI_EVIDENCE_MODEL"))
    cleaned_key = _safe_text(api_key or os.getenv("AI_EVIDENCE_API_KEY"))
    if not cleaned_url or not cleaned_model:
        return _empty_ai_evidence_df(), {
            "status": "not-configured",
            "evidence_rows": "0",
            "message": "Set AI_EVIDENCE_API_URL and AI_EVIDENCE_MODEL, or paste AI JSON output.",
        }
    if not _safe_text(literature_or_notes):
        return _empty_ai_evidence_df(), {"status": "empty-input", "evidence_rows": "0"}

    prompt = build_ai_evidence_prompt(
        literature_or_notes,
        protein_name=protein_name,
        accession=accession,
        pdb_id=pdb_id,
        ec_number=ec_number,
        triage_context=triage_context,
    )
    payload = {
        "model": cleaned_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You extract enzyme residue evidence. Return strict JSON only. Never invent unsupported residues.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response_payload = _post_json(cleaned_url, payload, api_key=cleaned_key, timeout_sec=timeout_sec)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return _empty_ai_evidence_df(), {
            "status": "unavailable",
            "evidence_rows": "0",
            "message": str(exc),
        }

    response_text = _message_text_from_response(response_payload)
    evidence_df, metadata = parse_ai_residue_evidence_payload(
        response_text,
        chain_hint=chain_hint,
        source_label="AI-Literature",
        min_confidence=min_confidence,
        assume_structure_numbering=assume_structure_numbering,
        pdb_text=pdb_text,
    )
    metadata = {
        **metadata,
        "model": cleaned_model,
        "api_url_configured": "true",
        "raw_response_chars": str(len(response_text)),
    }
    return evidence_df, metadata
