from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from urllib import error, request
from urllib.parse import urlencode

import pandas as pd

from protein_visualizer.services.external_sites import (
    EVIDENCE_COLUMNS,
    _extract_structure_residue_map,
    _map_uniprot_sites_to_structure,
    merge_external_evidence_tables,
)


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

AA3_PATTERN = "|".join(
    [
        "Ala",
        "Arg",
        "Asn",
        "Asp",
        "Cys",
        "Gln",
        "Glu",
        "Gly",
        "His",
        "Ile",
        "Leu",
        "Lys",
        "Met",
        "Phe",
        "Pro",
        "Ser",
        "Thr",
        "Trp",
        "Tyr",
        "Val",
    ]
)

THREE_LETTER_RESIDUE_RE = re.compile(
    rf"\b(?P<aa>{AA3_PATTERN})\.?\s*[- ]?\s*(?P<resid>\d{{1,5}})(?:\s*(?P<mut>{AA3_PATTERN}|[ACDEFGHIKLMNPQRSTVWY]))?\b",
    flags=re.IGNORECASE,
)
ONE_LETTER_MUTATION_RE = re.compile(r"\b(?P<aa>[ACDEFGHIKLMNPQRSTVWY])\s*(?P<resid>\d{1,5})\s*(?P<mut>[ACDEFGHIKLMNPQRSTVWY])\b")

CATALYTIC_KEYWORDS = (
    "active site",
    "catalytic",
    "catalysis",
    "catalytic triad",
    "catalytic dyad",
    "nucleophile",
    "proton donor",
    "proton acceptor",
    "general acid",
    "general base",
    "acid/base",
)
BINDING_KEYWORDS = (
    "binding site",
    "substrate binding",
    "substrate-binding",
    "ligand binding",
    "cofactor",
    "metal binding",
    "zinc",
    "magnesium",
    "manganese",
    "calcium",
)
MUTATION_KEYWORDS = (
    "mutant",
    "mutation",
    "mutagenesis",
    "substitution",
    "variant",
    "alanine mutant",
)
ACTIVITY_KEYWORDS = (
    "abolished activity",
    "loss of activity",
    "inactive",
    "reduced activity",
    "impaired activity",
    "decreased activity",
    "kcat",
    "km",
    "turnover",
)

LOW_SIGNAL_CONTEXT = (
    "sequence alignment",
    "numbering scheme",
    "residue numbering",
    "supplementary table",
)


def _empty_literature_evidence_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def _literature_article_key(row: pd.Series) -> str:
    note_parts = [part.strip() for part in str(row.get("evidence_note", "") or "").split("|")]
    article_id = note_parts[0] if note_parts else ""
    article_title = note_parts[1] if len(note_parts) > 1 else ""
    normalized_title = re.sub(r"\W+", " ", article_title.lower()).strip()
    if normalized_title:
        return f"title:{normalized_title[:140]}"

    pmid_match = re.search(r"(?:PMID[:\s]*|MED[:\s]*)(\d+)", article_id, flags=re.IGNORECASE)
    if pmid_match:
        return f"pmid:{pmid_match.group(1)}"

    cleaned_id = re.sub(r"\s+", " ", article_id).strip().lower()
    if cleaned_id:
        return cleaned_id
    return f"row:{row.name}"


def _boost_replicated_literature_support(table: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    if table is None or getattr(table, "empty", True):
        return _empty_literature_evidence_df(), {
            "replicated_residue_groups": "0",
            "max_article_support": "0",
            "max_source_support": "0",
        }

    working = table.copy()
    working["evidence_score"] = pd.to_numeric(working.get("evidence_score"), errors="coerce").fillna(0.75).clip(0.0, 1.0)
    working["mapping_confidence"] = pd.to_numeric(working.get("mapping_confidence"), errors="coerce").fillna(0.34).clip(0.0, 1.0)
    working["_support_position"] = pd.to_numeric(working.get("uniprot_resid"), errors="coerce")
    missing_position = working["_support_position"].isna()
    working.loc[missing_position, "_support_position"] = pd.to_numeric(working.loc[missing_position, "resid"], errors="coerce")
    working["_article_key"] = working.apply(_literature_article_key, axis=1)
    working["_source_key"] = working.get("evidence_source", pd.Series("", index=working.index)).astype(str).str.strip()

    replicated_groups = 0
    max_article_support = 0
    max_source_support = 0
    for _position, group in working[working["_support_position"].notna()].groupby("_support_position"):
        article_count = int(group["_article_key"].replace("", pd.NA).dropna().nunique())
        source_count = int(group["_source_key"].replace("", pd.NA).dropna().nunique())
        max_article_support = max(max_article_support, article_count)
        max_source_support = max(max_source_support, source_count)
        if article_count < 2:
            continue

        replicated_groups += 1
        support_boost = min(0.16, 0.07 * float(article_count - 1) + 0.02 * float(max(0, source_count - 1)))
        support_note = f"literature_support={article_count} articles/{source_count} sources"
        row_indices = group.index
        working.loc[row_indices, "evidence_score"] = (working.loc[row_indices, "evidence_score"] + support_boost).clip(0.0, 1.0)
        working.loc[row_indices, "mapping_confidence"] = (working.loc[row_indices, "mapping_confidence"] + support_boost * 0.5).clip(0.0, 1.0)
        has_note = working.loc[row_indices, "evidence_note"].astype(str).str.contains("literature_support=", regex=False, na=False)
        append_indices = row_indices[~has_note.to_numpy()]
        if len(append_indices):
            working.loc[append_indices, "evidence_note"] = (
                working.loc[append_indices, "evidence_note"].astype(str).str.rstrip()
                + " | "
                + support_note
            )

    metadata = {
        "replicated_residue_groups": str(replicated_groups),
        "max_article_support": str(max_article_support),
        "max_source_support": str(max_source_support),
    }
    return working.drop(columns=["_support_position", "_article_key", "_source_key"])[EVIDENCE_COLUMNS].reset_index(drop=True), metadata


def merge_literature_evidence_tables(*tables: Optional[pd.DataFrame]) -> pd.DataFrame:
    combined = merge_external_evidence_tables(*tables)
    boosted, support_meta = _boost_replicated_literature_support(combined)
    boosted.attrs["literature_support"] = support_meta
    return boosted


def _literature_aa3_from_note(note: object) -> str:
    matched = re.search(r"(?:^|\|\s*)aa=([A-Za-z]{1,3})\b", str(note or ""))
    if not matched:
        return ""
    token = matched.group(1).upper()
    if len(token) == 1:
        return AA1_TO_3.get(token, "")
    return token[:3] if token[:3] in AA3_TO_1 else ""


def _append_note_once(value: object, note: str) -> str:
    cleaned_note = str(note or "").strip()
    if not cleaned_note:
        return str(value or "").strip()
    text = str(value or "").strip()
    if cleaned_note in text:
        return text
    return f"{text} | {cleaned_note}".strip(" |")


def _verify_assumed_structure_numbering(
    evidence_df: pd.DataFrame,
    *,
    chain: str,
    pdb_text: Optional[str],
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if evidence_df is None or getattr(evidence_df, "empty", True):
        return _empty_literature_evidence_df(), {
            "identity_checked_rows": "0",
            "identity_matched_rows": "0",
            "identity_mismatched_rows": "0",
            "identity_missing_rows": "0",
        }

    working = evidence_df.copy()
    structure_map = _extract_structure_residue_map(pdb_text)
    chain_entries = list((structure_map.get(str(chain or "").strip()) or {}).get("entries") or [])
    observed_resnames = {
        int(entry.get("resid", 0) or 0): str(entry.get("resname", "") or "").upper()
        for entry in chain_entries
    }

    if not observed_resnames:
        return working[EVIDENCE_COLUMNS], {
            "identity_checked_rows": "0",
            "identity_matched_rows": "0",
            "identity_mismatched_rows": "0",
            "identity_missing_rows": "0",
        }

    checked_rows = 0
    matched_rows = 0
    mismatched_rows = 0
    missing_rows = 0
    for index, row in working.iterrows():
        expected_aa = _literature_aa3_from_note(row.get("evidence_note"))
        if not expected_aa:
            continue
        try:
            resid = int(float(row.get("resid")))
        except (TypeError, ValueError):
            continue

        checked_rows += 1
        observed_aa = observed_resnames.get(resid, "")
        if not observed_aa:
            missing_rows += 1
            working.at[index, "mapping_level"] = "weak"
            working.at[index, "mapping_confidence"] = min(float(row.get("mapping_confidence") or 0.0), 0.30)
            working.at[index, "evidence_score"] = min(float(row.get("evidence_score") or 0.0), 0.58)
            working.at[index, "mapping_method"] = "literature-residue-missing"
            working.at[index, "evidence_note"] = _append_note_once(
                row.get("evidence_note"),
                f"structure_residue_missing={chain}:{resid}",
            )
            continue

        if observed_aa == expected_aa:
            matched_rows += 1
            working.at[index, "mapping_confidence"] = max(float(row.get("mapping_confidence") or 0.0), 0.88)
            working.at[index, "mapping_method"] = "literature-structure-numbering-verified"
            working.at[index, "evidence_note"] = _append_note_once(
                row.get("evidence_note"),
                f"structure_residue_match={observed_aa}",
            )
            continue

        mismatched_rows += 1
        working.at[index, "mapping_level"] = "weak"
        working.at[index, "mapping_confidence"] = min(float(row.get("mapping_confidence") or 0.0), 0.28)
        working.at[index, "evidence_score"] = min(float(row.get("evidence_score") or 0.0), 0.52)
        working.at[index, "mapping_method"] = "literature-residue-identity-mismatch"
        working.at[index, "evidence_note"] = _append_note_once(
            row.get("evidence_note"),
            f"structure_residue_mismatch={expected_aa}!={observed_aa}",
        )

    return working[EVIDENCE_COLUMNS], {
        "identity_checked_rows": str(checked_rows),
        "identity_matched_rows": str(matched_rows),
        "identity_mismatched_rows": str(mismatched_rows),
        "identity_missing_rows": str(missing_rows),
    }


def remove_literature_evidence(external_site_df: pd.DataFrame) -> pd.DataFrame:
    if external_site_df is None or getattr(external_site_df, "empty", True):
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    working = external_site_df.copy()
    literature_mask = pd.Series(False, index=working.index)
    for column in ("evidence_source", "mapping_method", "evidence_note"):
        if column in working.columns:
            literature_mask = literature_mask | working[column].astype(str).str.contains("literature", case=False, na=False)

    filtered = working[~literature_mask].copy().reset_index(drop=True)
    return filtered


def _fetch_json(url: str, *, timeout_sec: float = 10.0) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=float(timeout_sec)) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore"))


def _fetch_text(url: str, *, timeout_sec: float = 10.0) -> str:
    req = request.Request(url, headers={"Accept": "application/xml,text/xml,text/plain"})
    with request.urlopen(req, timeout=float(timeout_sec)) as response:
        return response.read().decode("utf-8", errors="ignore")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _context_window(text: str, start: int, end: int, *, width: int = 180) -> str:
    left = max(0, int(start) - int(width))
    right = min(len(text), int(end) + int(width))
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _clean_snippet(text: str, *, max_len: int = 280) -> str:
    snippet = html.unescape(re.sub(r"\s+", " ", str(text or "")).strip())
    if len(snippet) <= int(max_len):
        return snippet
    return snippet[: int(max_len) - 3].rstrip() + "..."


def _score_literature_context(context: str, *, mutation_pattern: bool) -> tuple[float, str]:
    lowered = str(context or "").lower()
    catalytic = _contains_any(lowered, CATALYTIC_KEYWORDS)
    binding = _contains_any(lowered, BINDING_KEYWORDS)
    mutation = mutation_pattern or _contains_any(lowered, MUTATION_KEYWORDS)
    activity = _contains_any(lowered, ACTIVITY_KEYWORDS)
    low_signal = _contains_any(lowered, LOW_SIGNAL_CONTEXT)

    score = 0.0
    if catalytic:
        score += 0.48
    if binding:
        score += 0.30
    if mutation:
        score += 0.24
    if activity:
        score += 0.24
    if "catalytic triad" in lowered or "catalytic dyad" in lowered:
        score += 0.12
    if low_signal:
        score -= 0.22

    if catalytic:
        evidence_type = "Catalytic residue"
    elif mutation and activity:
        evidence_type = "Activity-loss mutagenesis"
    elif mutation:
        evidence_type = "Mutagenesis"
    elif binding:
        evidence_type = "Binding site"
    else:
        evidence_type = "Literature residue"

    return max(0.0, min(1.0, 0.35 + score)), evidence_type


def _add_literature_row(
    rows: list[dict],
    *,
    chain_hint: str,
    resid: int,
    aa3: str,
    matched_text: str,
    context: str,
    source_label: str,
    article_id: str,
    article_title: str,
    mutation_pattern: bool,
) -> None:
    evidence_score, evidence_type = _score_literature_context(context, mutation_pattern=mutation_pattern)
    min_score = 0.66 if mutation_pattern else 0.58
    if evidence_score < min_score:
        return

    cleaned_chain = str(chain_hint or "").strip()
    note_parts = [
        str(article_id or "").strip(),
        str(article_title or "").strip(),
        f"match={matched_text}",
        f"aa={aa3}",
        _clean_snippet(context),
    ]
    rows.append(
        {
            "chain": cleaned_chain,
            "resid": int(resid),
            "evidence_source": str(source_label or "Literature").strip() or "Literature",
            "evidence_type": evidence_type,
            "evidence_score": round(float(evidence_score), 3),
            "evidence_note": " | ".join(part for part in note_parts if part),
            "uniprot_resid": int(resid),
            "mapping_level": "weak",
            "mapping_confidence": 0.46 if cleaned_chain else 0.34,
            "mapping_method": "literature-text-mining",
        }
    )


def extract_literature_residue_evidence(
    text: str,
    *,
    chain_hint: Optional[str] = None,
    source_label: str = "Literature",
    article_id: str = "",
    article_title: str = "",
) -> tuple[pd.DataFrame, Dict[str, object]]:
    cleaned_text = str(text or "")
    if not cleaned_text.strip():
        return _empty_literature_evidence_df(), {"status": "empty", "evidence_rows": "0"}

    rows: list[dict] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in THREE_LETTER_RESIDUE_RE.finditer(cleaned_text):
        resid = int(match.group("resid"))
        aa3 = str(match.group("aa") or "").upper()
        aa3 = aa3[:3]
        context = _context_window(cleaned_text, match.start(), match.end())
        matched_text = match.group(0)
        mutation_pattern = bool(match.group("mut")) or _contains_any(context, MUTATION_KEYWORDS)
        _add_literature_row(
            rows,
            chain_hint=str(chain_hint or ""),
            resid=resid,
            aa3=aa3,
            matched_text=matched_text,
            context=context,
            source_label=source_label,
            article_id=article_id,
            article_title=article_title,
            mutation_pattern=mutation_pattern,
        )
        consumed_spans.append((match.start(), match.end()))

    for match in ONE_LETTER_MUTATION_RE.finditer(cleaned_text):
        if any(match.start() >= start and match.end() <= end for start, end in consumed_spans):
            continue
        resid = int(match.group("resid"))
        aa1 = str(match.group("aa") or "").upper()
        context = _context_window(cleaned_text, match.start(), match.end())
        if not (_contains_any(context, MUTATION_KEYWORDS) or _contains_any(context, ACTIVITY_KEYWORDS)):
            continue
        _add_literature_row(
            rows,
            chain_hint=str(chain_hint or ""),
            resid=resid,
            aa3=AA1_TO_3.get(aa1, aa1),
            matched_text=match.group(0),
            context=context,
            source_label=source_label,
            article_id=article_id,
            article_title=article_title,
            mutation_pattern=True,
        )

    if not rows:
        return _empty_literature_evidence_df(), {
            "status": "ok",
            "evidence_rows": "0",
            "source": source_label,
        }

    evidence_df = pd.DataFrame(rows)
    evidence_df = evidence_df.sort_values(
        ["evidence_score", "mapping_confidence", "resid", "evidence_type"],
        ascending=[False, False, True, True],
    ).drop_duplicates(
        subset=["chain", "resid", "evidence_type", "uniprot_resid"],
        keep="first",
    ).reset_index(drop=True)
    evidence_df = evidence_df[EVIDENCE_COLUMNS]
    return evidence_df, {
        "status": "ok",
        "evidence_rows": str(len(evidence_df)),
        "source": source_label,
        "article_id": article_id,
    }


def _article_text_from_pubmed_xml(xml_text: str) -> list[dict[str, str]]:
    if not str(xml_text or "").strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    articles: list[dict[str, str]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = "".join(article.findtext(".//PMID") or "").strip()
        title = "".join(article.findtext(".//ArticleTitle") or "").strip()
        abstract_parts: list[str] = []
        for abstract_node in article.findall(".//Abstract/AbstractText"):
            label = str(abstract_node.attrib.get("Label") or "").strip()
            text = "".join(abstract_node.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        text = " ".join(part for part in [title, *abstract_parts] if part)
        if text.strip():
            articles.append(
                {
                    "pmid": pmid,
                    "title": html.unescape(title),
                    "text": html.unescape(text),
                }
            )
    return articles


def _plain_text_from_xml(xml_text: str, *, max_chars: int = 120_000) -> str:
    cleaned = str(xml_text or "").strip()
    if not cleaned:
        return ""
    try:
        root = ET.fromstring(cleaned)
        text = " ".join(part.strip() for part in root.itertext() if str(part or "").strip())
    except ET.ParseError:
        text = re.sub(r"<[^>]+>", " ", cleaned)
    text = html.unescape(re.sub(r"\s+", " ", text).strip())
    return text[: int(max_chars)]


def _europepmc_fulltext_ids(result: dict) -> list[str]:
    ids: list[str] = []
    for key in ("pmcid", "pmcId"):
        value = str(result.get(key) or "").strip()
        if value:
            ids.append(value)

    full_text_list = result.get("fullTextIdList") or result.get("fulltextIdList") or {}
    if isinstance(full_text_list, dict):
        values = full_text_list.get("fullTextId") or full_text_list.get("fulltextId") or []
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            ids.extend(str(value).strip() for value in values if str(value).strip())
    elif isinstance(full_text_list, list):
        ids.extend(str(value).strip() for value in full_text_list if str(value).strip())

    normalized: list[str] = []
    for value in ids:
        cleaned = value.strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _europepmc_article_records(payload: dict) -> list[dict[str, object]]:
    results = ((payload or {}).get("resultList") or {}).get("result") or []
    if not isinstance(results, list):
        return []

    records: list[dict[str, object]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title = html.unescape(str(result.get("title") or "").strip())
        abstract = html.unescape(str(result.get("abstractText") or "").strip())
        source = str(result.get("source") or "").strip()
        article_id = str(result.get("id") or result.get("pmid") or result.get("doi") or "").strip()
        pmid = str(result.get("pmid") or "").strip()
        text = " ".join(part for part in [title, abstract] if part)
        records.append(
            {
                "source": source,
                "id": article_id,
                "pmid": pmid,
                "title": title,
                "text": text,
                "full_text_ids": _europepmc_fulltext_ids(result),
            }
        )
    return records


def build_literature_query(
    *,
    accession: str = "",
    ec_number: str = "",
    pdb_id: str = "",
    protein_name: str = "",
    extra_query: str = "",
) -> str:
    terms: list[str] = []
    for value in [accession, pdb_id, protein_name]:
        text = str(value or "").strip()
        if text:
            terms.append(text)
    ec_text = str(ec_number or "").strip()
    if ec_text:
        terms.append(f'"{ec_text}"')
    if extra_query:
        terms.append(str(extra_query).strip())

    core = " ".join(terms).strip()
    residue_terms = '(active site OR catalytic OR catalysis OR mutagenesis OR "substrate binding" OR "binding site")'
    return f"{core} {residue_terms}".strip() if core else residue_terms


def fetch_pubmed_literature_evidence(
    query: str,
    *,
    chain_hint: Optional[str] = None,
    max_articles: int = 6,
    timeout_sec: float = 10.0,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return _empty_literature_evidence_df(), {"status": "empty", "evidence_rows": "0", "article_count": "0"}

    search_params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": max(1, int(max_articles)),
        "sort": "relevance",
        "term": cleaned_query,
    }
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{urlencode(search_params)}"
    try:
        search_payload = _fetch_json(search_url, timeout_sec=timeout_sec)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return _empty_literature_evidence_df(), {
            "status": "unavailable",
            "query": cleaned_query,
            "evidence_rows": "0",
            "article_count": "0",
        }

    ids = [
        str(value).strip()
        for value in ((search_payload.get("esearchresult") or {}).get("idlist") or [])
        if str(value).strip()
    ][: max(1, int(max_articles))]
    if not ids:
        return _empty_literature_evidence_df(), {
            "status": "ok",
            "query": cleaned_query,
            "evidence_rows": "0",
            "article_count": "0",
        }

    fetch_params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(ids),
    }
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{urlencode(fetch_params)}"
    try:
        xml_text = _fetch_text(fetch_url, timeout_sec=timeout_sec)
    except (error.URLError, TimeoutError, ValueError):
        return _empty_literature_evidence_df(), {
            "status": "unavailable",
            "query": cleaned_query,
            "evidence_rows": "0",
            "article_count": str(len(ids)),
        }

    evidence_tables: list[pd.DataFrame] = []
    articles = _article_text_from_pubmed_xml(xml_text)
    for article in articles:
        table, _ = extract_literature_residue_evidence(
            article.get("text", ""),
            chain_hint=chain_hint,
            source_label="Literature-PubMed",
            article_id=f"PMID:{article.get('pmid', '')}",
            article_title=article.get("title", ""),
        )
        if not table.empty:
            evidence_tables.append(table)

    evidence_df = merge_literature_evidence_tables(*evidence_tables)
    support_meta = evidence_df.attrs.get("literature_support", {})
    return evidence_df, {
        "status": "ok" if not evidence_df.empty else "empty",
        "query": cleaned_query,
        "evidence_rows": str(len(evidence_df)),
        "article_count": str(len(articles)),
        "pmids": ",".join(article.get("pmid", "") for article in articles if article.get("pmid")),
        "source": "Literature-PubMed",
        "literature_support": support_meta,
    }


def fetch_europepmc_literature_evidence(
    query: str,
    *,
    chain_hint: Optional[str] = None,
    max_articles: int = 6,
    include_open_fulltext: bool = True,
    max_fulltext_articles: int = 2,
    timeout_sec: float = 10.0,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return _empty_literature_evidence_df(), {"status": "empty", "evidence_rows": "0", "article_count": "0", "fulltext_count": "0"}

    search_params = {
        "query": cleaned_query,
        "format": "json",
        "resultType": "core",
        "pageSize": max(1, int(max_articles)),
    }
    search_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{urlencode(search_params)}"
    try:
        payload = _fetch_json(search_url, timeout_sec=timeout_sec)
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return _empty_literature_evidence_df(), {
            "status": "unavailable",
            "query": cleaned_query,
            "evidence_rows": "0",
            "article_count": "0",
            "fulltext_count": "0",
        }

    records = _europepmc_article_records(payload)[: max(1, int(max_articles))]
    evidence_tables: list[pd.DataFrame] = []
    fulltext_count = 0
    ids: list[str] = []
    for record in records:
        article_id = str(record.get("id") or record.get("pmid") or "").strip()
        source = str(record.get("source") or "EuropePMC").strip()
        if article_id:
            ids.append(f"{source}:{article_id}")

        table, _ = extract_literature_residue_evidence(
            str(record.get("text") or ""),
            chain_hint=chain_hint,
            source_label="Literature-EuropePMC",
            article_id=f"{source}:{article_id}" if article_id else "EuropePMC",
            article_title=str(record.get("title") or ""),
        )
        if not table.empty:
            evidence_tables.append(table)

        if not include_open_fulltext or fulltext_count >= int(max_fulltext_articles):
            continue
        for full_text_id in list(record.get("full_text_ids") or []):
            if fulltext_count >= int(max_fulltext_articles):
                break
            fulltext_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{full_text_id}/fullTextXML"
            try:
                xml_text = _fetch_text(fulltext_url, timeout_sec=timeout_sec)
            except (error.URLError, TimeoutError, ValueError):
                continue
            full_text = _plain_text_from_xml(xml_text)
            if not full_text:
                continue
            fulltext_count += 1
            full_table, _ = extract_literature_residue_evidence(
                full_text,
                chain_hint=chain_hint,
                source_label="Literature-EuropePMC-OA",
                article_id=str(full_text_id),
                article_title=str(record.get("title") or ""),
            )
            if not full_table.empty:
                evidence_tables.append(full_table)
            break

    evidence_df = merge_literature_evidence_tables(*evidence_tables)
    support_meta = evidence_df.attrs.get("literature_support", {})
    return evidence_df, {
        "status": "ok" if not evidence_df.empty else "empty",
        "query": cleaned_query,
        "evidence_rows": str(len(evidence_df)),
        "article_count": str(len(records)),
        "fulltext_count": str(fulltext_count),
        "ids": ",".join(ids),
        "source": "Literature-EuropePMC",
        "literature_support": support_meta,
    }


def map_literature_evidence_to_structure(
    evidence_df: pd.DataFrame,
    *,
    accession: str = "",
    pdb_id: str = "",
    chain_hint: Optional[str] = None,
    pdb_text: Optional[str] = None,
    timeout_sec: float = 10.0,
    assume_structure_numbering: bool = False,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    if evidence_df is None or getattr(evidence_df, "empty", True):
        return _empty_literature_evidence_df(), {"mapping_status": "no-evidence", "mapped_rows": "0"}

    cleaned_accession = str(accession or "").strip().upper()
    cleaned_pdb_id = str(pdb_id or "").strip().upper()
    cleaned_chain = str(chain_hint or "").strip()
    if assume_structure_numbering and cleaned_chain:
        direct = evidence_df.copy()
        direct["chain"] = cleaned_chain
        direct["mapping_level"] = "exact"
        direct["mapping_confidence"] = pd.to_numeric(direct["mapping_confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        direct["mapping_confidence"] = direct["mapping_confidence"].map(lambda value: max(float(value), 0.78))
        direct["mapping_method"] = "literature-structure-numbering"
        direct["evidence_source"] = direct["evidence_source"].astype(str).map(
            lambda value: value if "Structure" in value else f"{value}+Structure"
        )
        direct, identity_meta = _verify_assumed_structure_numbering(
            direct,
            chain=cleaned_chain,
            pdb_text=pdb_text,
        )
        exact_rows = int(direct["mapping_level"].astype(str).str.lower().eq("exact").sum())
        weak_rows = int(len(direct) - exact_rows)
        return direct[EVIDENCE_COLUMNS], {
            "mapping_status": "assumed-structure-numbering",
            "mapped_rows": str(len(direct)),
            "exact_rows": str(exact_rows),
            "weak_rows": str(weak_rows),
            **identity_meta,
        }

    if not cleaned_accession or not cleaned_pdb_id:
        fallback = evidence_df.copy()
        fallback["evidence_source"] = fallback["evidence_source"].astype(str).where(
            fallback["evidence_source"].astype(str).str.len().gt(0),
            "Literature",
        )
        return fallback[EVIDENCE_COLUMNS], {
            "mapping_status": "direct-weak",
            "mapped_rows": str(len(fallback)),
            "exact_rows": "0",
            "weak_rows": str(len(fallback)),
        }

    mapped_df, mapping_meta = _map_uniprot_sites_to_structure(
        evidence_df,
        accession=cleaned_accession,
        pdb_id=cleaned_pdb_id,
        chain_hint=chain_hint,
        pdb_text=pdb_text,
        timeout_sec=timeout_sec,
    )
    if mapped_df.empty:
        return mapped_df, mapping_meta
    mapped_df = mapped_df.copy()
    mapped_df["evidence_source"] = mapped_df["evidence_source"].astype(str).str.replace("UniProt+SIFTS", "Literature+SIFTS", regex=False)
    mapped_df["evidence_source"] = mapped_df["evidence_source"].astype(str).str.replace("UniProt", "Literature", regex=False)
    mapped_df["mapping_method"] = mapped_df["mapping_method"].astype(str).map(
        lambda value: value if value.startswith("literature-") else f"literature-{value}"
    )
    return mapped_df[EVIDENCE_COLUMNS], mapping_meta


def fetch_literature_residue_evidence_for_structure(
    *,
    query: str = "",
    manual_text: str = "",
    accession: str = "",
    ec_number: str = "",
    pdb_id: str = "",
    protein_name: str = "",
    chain_hint: Optional[str] = None,
    pdb_text: Optional[str] = None,
    max_articles: int = 6,
    timeout_sec: float = 10.0,
    enable_pubmed: bool = True,
    enable_europepmc: bool = False,
    include_europepmc_fulltext: bool = True,
    max_fulltext_articles: int = 2,
    assume_structure_numbering: bool = False,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    tables: list[pd.DataFrame] = []
    manual_meta: Dict[str, object] = {}
    pubmed_meta: Dict[str, object] = {}
    europepmc_meta: Dict[str, object] = {}

    if str(manual_text or "").strip():
        manual_df, manual_meta = extract_literature_residue_evidence(
            manual_text,
            chain_hint=chain_hint,
            source_label="Literature-Manual",
            article_id="manual",
        )
        if not manual_df.empty:
            tables.append(manual_df)

    search_query = build_literature_query(
        accession=accession,
        ec_number=ec_number,
        pdb_id=pdb_id,
        protein_name=protein_name,
        extra_query=query,
    )
    if enable_pubmed and str(query or accession or ec_number or pdb_id or protein_name or "").strip():
        pubmed_df, pubmed_meta = fetch_pubmed_literature_evidence(
            search_query,
            chain_hint=chain_hint,
            max_articles=max_articles,
            timeout_sec=timeout_sec,
        )
        if not pubmed_df.empty:
            tables.append(pubmed_df)
    if enable_europepmc and str(query or accession or ec_number or pdb_id or protein_name or "").strip():
        europepmc_df, europepmc_meta = fetch_europepmc_literature_evidence(
            search_query,
            chain_hint=chain_hint,
            max_articles=max_articles,
            include_open_fulltext=include_europepmc_fulltext,
            max_fulltext_articles=max_fulltext_articles,
            timeout_sec=timeout_sec,
        )
        if not europepmc_df.empty:
            tables.append(europepmc_df)

    combined = merge_literature_evidence_tables(*tables)
    support_meta = combined.attrs.get("literature_support", {})
    mapped_df, mapping_meta = map_literature_evidence_to_structure(
        combined,
        accession=accession,
        pdb_id=pdb_id,
        chain_hint=chain_hint,
        pdb_text=pdb_text,
        timeout_sec=timeout_sec,
        assume_structure_numbering=assume_structure_numbering,
    )
    metadata: Dict[str, object] = {
        "status": "ok" if not mapped_df.empty else "empty",
        "source": "Literature",
        "query": search_query,
        "evidence_rows": str(len(mapped_df)),
        "manual": manual_meta,
        "pubmed": pubmed_meta,
        "europepmc": europepmc_meta,
        "literature_support": support_meta,
        "mapping": mapping_meta,
    }
    return mapped_df, metadata
