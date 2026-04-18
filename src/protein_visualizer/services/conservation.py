from __future__ import annotations

import io
import re
from typing import Optional, Tuple

import pandas as pd

from protein_visualizer.services.external_sites import EVIDENCE_COLUMNS


GRADE_SCORE_ALIASES = {
    "conservationscore",
    "conservation",
    "grade",
    "color",
    "consurfgrade",
    "conservationgrade",
}
RATE_SCORE_ALIASES = {
    "rate4site",
    "rate4sitescore",
    "evolutionaryrate",
    "rawscore",
    "consurfscore",
}
GENERIC_SCORE_ALIASES = {
    "score",
    "normalizedscore",
    "importance",
}
CHAIN_ALIASES = {
    "chain",
    "chainid",
    "chain_id",
    "authasymid",
    "auth_asym_id",
    "asymid",
    "asym_id",
}
RESID_ALIASES = {
    "resid",
    "residue",
    "residuenumber",
    "residue_number",
    "residueid",
    "residue_id",
    "position",
    "seqnum",
    "seq_num",
    "seqnumber",
    "sequenceposition",
    "pdbresiduenumber",
}
RESIDUE_TOKEN_ALIASES = {
    "residuelabel",
    "residue_label",
    "residuecode",
    "residue_code",
    "residueidentifier",
    "residue_identifier",
    "atom",
}
CONFIDENCE_ALIASES = {
    "confidence",
    "reliability",
    "probability",
    "scoreconfidence",
    "score_confidence",
    "mappingconfidence",
    "mapping_confidence",
}
NOTE_ALIASES = {
    "note",
    "notes",
    "annotation",
    "description",
    "comment",
}
SOURCE_ALIASES = {
    "source",
    "dataset",
    "method",
}


def _empty_conservation_evidence_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def _simplify_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _find_matching_column(frame: pd.DataFrame, aliases: set[str]) -> Optional[str]:
    alias_map = {_simplify_column_name(column): column for column in frame.columns}
    for alias in aliases:
        if alias in alias_map:
            return alias_map[alias]
    return None


def _safe_int(value: object) -> Optional[int]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _read_delimited_table(text: str) -> pd.DataFrame:
    payload = str(text or "").strip()
    if not payload:
        return pd.DataFrame()

    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ","},
        {"delim_whitespace": True},
    )
    for kwargs in attempts:
        try:
            frame = pd.read_csv(io.StringIO(payload), comment="#", **kwargs)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        if len(frame.columns) == 1 and kwargs.get("sep") is None:
            continue
        return frame
    return pd.DataFrame()


def _extract_chain_and_resid(value: object) -> tuple[str, Optional[int]]:
    text = str(value or "").strip()
    if not text:
        return "", None

    patterns = (
        r"(?P<chain>[A-Za-z0-9])\s*[:/_-]\s*(?P<resid>-?\d+)",
        r"(?P<resid>-?\d+)\s*[:/_-]\s*(?P<chain>[A-Za-z0-9])",
        r"\b(?P<chain>[A-Za-z0-9])\b.*?\b(?P<resid>-?\d+)\b",
    )
    for pattern in patterns:
        matched = re.search(pattern, text)
        if matched:
            resid = _safe_int(matched.group("resid"))
            if resid is not None:
                return str(matched.group("chain") or "").strip(), resid

    resid = _safe_int(text)
    return "", resid


def _normalize_score_series(
    values: pd.Series,
    *,
    prefer_lower_scores: bool = False,
    prefer_grade_scale: bool = False,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=float)

    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([0.0] * len(numeric), index=numeric.index, dtype=float)

    minimum = float(valid.min())
    maximum = float(valid.max())

    if prefer_grade_scale and minimum >= 1.0 and maximum <= 9.0:
        normalized = (numeric - 1.0) / 8.0
    elif prefer_grade_scale and minimum >= 0.0 and maximum <= 9.0:
        normalized = numeric / 9.0
    elif minimum >= 0.0 and maximum <= 1.0:
        normalized = numeric
    elif maximum - minimum <= 1e-9:
        normalized = pd.Series([1.0 if maximum > 0.0 else 0.0] * len(numeric), index=numeric.index, dtype=float)
    else:
        normalized = (numeric - minimum) / (maximum - minimum)

    normalized = normalized.fillna(0.0).clip(lower=0.0, upper=1.0)
    if prefer_lower_scores:
        normalized = 1.0 - normalized
    return normalized.clip(lower=0.0, upper=1.0)


def parse_conservation_evidence_table(
    text: str,
    *,
    chain_hint: Optional[str] = None,
    source_hint: str = "ConSurf",
) -> Tuple[pd.DataFrame, dict[str, str]]:
    raw = _read_delimited_table(text)
    if raw.empty:
        return _empty_conservation_evidence_df(), {
            "status": "empty",
            "reason": "No parseable conservation table was found.",
        }

    chain_column = _find_matching_column(raw, CHAIN_ALIASES)
    resid_column = _find_matching_column(raw, RESID_ALIASES)
    residue_token_column = _find_matching_column(raw, RESIDUE_TOKEN_ALIASES)
    note_column = _find_matching_column(raw, NOTE_ALIASES)
    source_column = _find_matching_column(raw, SOURCE_ALIASES)
    confidence_column = _find_matching_column(raw, CONFIDENCE_ALIASES)

    grade_score_column = _find_matching_column(raw, GRADE_SCORE_ALIASES)
    rate_score_column = _find_matching_column(raw, RATE_SCORE_ALIASES)
    generic_score_column = _find_matching_column(raw, GENERIC_SCORE_ALIASES)
    score_column = grade_score_column or rate_score_column or generic_score_column

    if score_column is None:
        return _empty_conservation_evidence_df(), {
            "status": "invalid",
            "reason": "A conservation score column was not found.",
        }

    chain_hint_text = str(chain_hint or "").strip()
    score_series = _normalize_score_series(
        raw[score_column],
        prefer_lower_scores=rate_score_column is not None,
        prefer_grade_scale=grade_score_column is not None,
    )
    confidence_series = (
        _normalize_score_series(raw[confidence_column], prefer_lower_scores=False, prefer_grade_scale=False)
        if confidence_column is not None
        else pd.Series([None] * len(raw), index=raw.index, dtype=object)
    )

    rows: list[dict[str, object]] = []
    for row_idx, row in raw.iterrows():
        row_dict = row.to_dict()
        chain_value = str(row_dict.get(chain_column, "") if chain_column else "").strip()

        resid_value = _safe_int(row_dict.get(resid_column)) if resid_column else None
        parsed_chain = ""
        if resid_value is None and residue_token_column is not None:
            parsed_chain, resid_value = _extract_chain_and_resid(row_dict.get(residue_token_column))
        if resid_value is None:
            continue

        effective_chain = chain_value or parsed_chain or chain_hint_text
        mapping_level = "exact" if effective_chain else "weak"
        mapping_confidence = 0.92 if mapping_level == "exact" else 0.42
        evidence_score = float(score_series.loc[row_idx]) if row_idx in score_series.index else 0.0

        if row_idx in confidence_series.index and pd.notna(confidence_series.loc[row_idx]):
            evidence_confidence = float(confidence_series.loc[row_idx])
        else:
            evidence_confidence = 0.82 if mapping_level == "exact" else 0.45

        source_value = str(row_dict.get(source_column, "") if source_column else "").strip()
        note_value = str(row_dict.get(note_column, "") if note_column else "").strip()
        note_parts = [part for part in [note_value, f"score={evidence_score:.3f}"] if part]

        rows.append(
            {
                "chain": effective_chain,
                "resid": int(resid_value),
                "evidence_source": source_value or str(source_hint or "Conservation").strip() or "Conservation",
                "evidence_type": "Conservation",
                "evidence_score": round(float(evidence_score), 3),
                "evidence_note": " | ".join(note_parts),
                "uniprot_resid": int(resid_value),
                "mapping_level": mapping_level,
                "mapping_confidence": round(float(max(mapping_confidence, evidence_confidence)), 3),
                "mapping_method": "conservation-import",
            }
        )

    if not rows:
        return _empty_conservation_evidence_df(), {
            "status": "empty",
            "reason": "No residue rows could be normalized from the conservation table.",
        }

    evidence_df = pd.DataFrame(rows)
    evidence_df = evidence_df.drop_duplicates(
        subset=["chain", "resid", "evidence_source", "evidence_type", "mapping_level"],
        keep="first",
    ).sort_values(
        ["mapping_level", "evidence_score", "mapping_confidence", "resid"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    level_series = evidence_df["mapping_level"].astype(str).str.lower()
    metadata = {
        "status": "ok",
        "source": str(source_hint or "Conservation").strip() or "Conservation",
        "evidence_rows": str(len(evidence_df)),
        "exact_rows": str(int((level_series == "exact").sum())),
        "weak_rows": str(int((level_series == "weak").sum())),
        "score_mean": f"{float(pd.to_numeric(evidence_df['evidence_score'], errors='coerce').fillna(0.0).mean()):.3f}",
        "score_max": f"{float(pd.to_numeric(evidence_df['evidence_score'], errors='coerce').fillna(0.0).max()):.3f}",
    }
    return evidence_df[EVIDENCE_COLUMNS], metadata
