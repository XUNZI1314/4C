from __future__ import annotations

import io
import re
from typing import Optional, Sequence

import pandas as pd


BENCHMARK_REFERENCE_COLUMNS = [
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "reference_type",
    "reference_source",
    "reference_note",
    "expected_pocket_id",
]

BENCHMARK_REFERENCE_TEMPLATE_COLUMNS = [
    *BENCHMARK_REFERENCE_COLUMNS,
]

BENCHMARK_DETAIL_COLUMNS = [
    *BENCHMARK_REFERENCE_COLUMNS,
    "residue_label",
    "matched",
    "matched_pocket_id",
    "matched_rank",
    "matched_pocket_ids",
    "matched_top1",
    "matched_top3",
    "matched_top5",
    "expected_pocket_matched",
    "benchmark_warning",
]

BENCHMARK_SUMMARY_COLUMNS = [
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "any_hit",
    "all_hit",
    "best_rank",
    "best_pocket_id",
    "top_pocket_id",
    "top_pocket_hit",
    "matched_residues",
    "missed_residues",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_CASE_SUMMARY_COLUMNS = [
    "benchmark_id",
    *BENCHMARK_SUMMARY_COLUMNS,
]

BENCHMARK_DATASET_SUMMARY_COLUMNS = [
    "top_n",
    "case_count",
    "reference_residue_count",
    "matched_reference_count",
    "mean_coverage_ratio",
    "median_coverage_ratio",
    "min_coverage_ratio",
    "max_coverage_ratio",
    "any_hit_case_count",
    "all_hit_case_count",
    "miss_case_count",
    "any_hit_rate",
    "all_hit_rate",
    "mean_best_rank",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "reference_coverage_ratio",
    "coverage_delta_vs_reference",
    "coverage_loss_vs_reference",
    "best_rank",
    "reference_best_rank",
    "best_rank_delta_vs_reference",
    "best_pocket_id",
    "top_pocket_id",
    "top_pocket_hit",
    "matched_residues",
    "missed_residues",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_CASE_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "benchmark_id",
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "reference_coverage_ratio",
    "coverage_delta_vs_reference",
    "coverage_loss_vs_reference",
    "best_rank",
    "reference_best_rank",
    "best_rank_delta_vs_reference",
    "best_pocket_id",
    "top_pocket_id",
    "top_pocket_hit",
    "matched_residues",
    "missed_residues",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_DATASET_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "top_n",
    "case_count",
    "mean_coverage_ratio",
    "reference_mean_coverage_ratio",
    "mean_coverage_delta_vs_reference",
    "mean_coverage_loss_vs_reference",
    "any_hit_rate",
    "reference_any_hit_rate",
    "any_hit_rate_delta_vs_reference",
    "all_hit_rate",
    "reference_all_hit_rate",
    "all_hit_rate_delta_vs_reference",
    "mean_best_rank",
    "reference_mean_best_rank",
    "mean_best_rank_delta_vs_reference",
    "case_loss_count",
    "case_gain_count",
    "case_unchanged_count",
    "benchmark_status",
    "benchmark_warning",
]

BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS = [
    "variant_label",
    "reference_variant_label",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "reference_type",
    "reference_source",
    "expected_pocket_id",
    "variant_matched",
    "reference_matched",
    "match_delta",
    "variant_matched_rank",
    "reference_matched_rank",
    "rank_delta_vs_reference",
    "variant_matched_pocket_id",
    "reference_matched_pocket_id",
    "variant_matched_pocket_ids",
    "reference_matched_pocket_ids",
    "variant_expected_pocket_matched",
    "reference_expected_pocket_matched",
    "benchmark_warning",
]

BENCHMARK_VARIANT_REMEDIATION_COLUMNS = [
    "action_id",
    "priority",
    "issue_type",
    "variant_label",
    "reference_variant_label",
    "benchmark_id",
    "residue_label",
    "chain",
    "resid",
    "resname",
    "match_delta",
    "reference_matched_pocket_id",
    "variant_matched_pocket_id",
    "reference_matched_rank",
    "variant_matched_rank",
    "expected_pocket_id",
    "suggested_action",
    "benchmark_warning",
]

BENCHMARK_VARIANT_REMEDIATION_SUMMARY_COLUMNS = [
    "priority",
    "issue_type",
    "variant_label",
    "action_count",
    "affected_case_count",
    "affected_residue_count",
    "top_residues",
    "suggested_action",
    "summary_status",
    "summary_warning",
]

BENCHMARK_REFERENCE_QUALITY_COLUMNS = [
    "issue_id",
    "severity",
    "issue_type",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "reference_type",
    "reference_source",
    "reference_note",
    "suggested_action",
    "quality_warning",
]

BENCHMARK_REFERENCE_QUALITY_SUMMARY_COLUMNS = [
    "severity",
    "issue_type",
    "issue_count",
    "affected_case_count",
    "affected_residue_count",
    "suggested_action",
    "summary_status",
    "summary_warning",
]

BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS = [
    "issue_id",
    "severity",
    "issue_type",
    "benchmark_id",
    "chain",
    "resid",
    "resname",
    "residue_label",
    "structure_chains",
    "structure_resnames",
    "matched_chain",
    "matched_resname",
    "reference_source",
    "reference_note",
    "suggested_action",
    "validation_warning",
]

BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_SUMMARY_COLUMNS = [
    "severity",
    "issue_type",
    "issue_count",
    "affected_case_count",
    "affected_residue_count",
    "suggested_action",
    "summary_status",
    "summary_warning",
]

BENCHMARK_REFERENCE_READINESS_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "issue_source",
    "issue_type",
    "benchmark_id",
    "residue_label",
    "chain",
    "resid",
    "resname",
    "suggested_action",
    "readiness_warning",
]

BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS = [
    "readiness_status",
    "reference_residue_count",
    "curation_issue_count",
    "structure_validation_issue_count",
    "p0_p1_issue_count",
    "p2_issue_count",
    "blocking_issue_types",
    "review_issue_types",
    "recommended_action",
    "readiness_warning",
]

BENCHMARK_REFERENCE_READINESS_CASE_SUMMARY_COLUMNS = [
    "benchmark_id",
    *BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS,
]

BENCHMARK_INTERPRETATION_COLUMNS = [
    "top_n",
    "reference_residue_count",
    "matched_reference_count",
    "coverage_ratio",
    "benchmark_status",
    "readiness_status",
    "claim_status",
    "claim_ready",
    "best_rank",
    "best_pocket_id",
    "interpretation_label",
    "recommended_action",
    "interpretation_warning",
]

BENCHMARK_CASE_INTERPRETATION_COLUMNS = [
    "benchmark_id",
    *BENCHMARK_INTERPRETATION_COLUMNS,
]

BENCHMARK_CASE_INTERPRETATION_MATRIX_BASE_COLUMNS = [
    "benchmark_id",
    "top_n_count",
    "best_claim_ready_top_n",
    "best_claim_ready_coverage",
    "best_claim_ready_rank",
    "any_blocked",
    "any_review_needed",
    "any_readiness_unknown",
    "case_interpretation_status",
    "recommended_action",
]

BENCHMARK_DATASET_INTERPRETATION_COLUMNS = [
    "top_n",
    "case_count",
    "claim_ready_case_count",
    "blocked_case_count",
    "review_case_count",
    "unknown_case_count",
    "mean_claim_ready_coverage",
    "mean_all_case_coverage",
    "claim_ready_rate",
    "blocked_case_rate",
    "review_case_rate",
    "dataset_claim_status",
    "interpretation_label",
    "recommended_action",
    "interpretation_warning",
]

BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS = [
    "action_id",
    "priority",
    "action_status",
    "top_n",
    "benchmark_id",
    "claim_status",
    "coverage_ratio",
    "best_rank",
    "best_pocket_id",
    "benchmark_status",
    "readiness_status",
    "issue_type",
    "suggested_action",
    "interpretation_warning",
]

CHAIN_ALIASES = {"chain", "chainid", "chain_id", "authasymid", "auth_asym_id", "asymid", "asym_id"}
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
    "uniprotresid",
    "uniprot_resid",
}
RESNAME_ALIASES = {"resname", "residue_name", "aa", "aminoacid", "amino_acid"}
RESIDUE_TOKEN_ALIASES = {"residuelabel", "residue_label", "site", "positiontext", "position_text"}
TYPE_ALIASES = {"reference_type", "type", "evidence_type", "role", "annotation", "function"}
SOURCE_ALIASES = {"reference_source", "source", "dataset", "citation", "reference", "pmid", "doi"}
NOTE_ALIASES = {"reference_note", "note", "notes", "comment", "description", "evidence_note"}
EXPECTED_POCKET_ALIASES = {"expected_pocket_id", "pocket_id", "active_site_pocket", "validated_pocket_id"}
BENCHMARK_ID_ALIASES = {"benchmark_id", "case_id", "dataset_id", "enzyme_id", "pdb_id", "entry_id"}
GENERIC_REFERENCE_SOURCE_LABELS = {
    "",
    "source",
    "reference",
    "curatedbenchmark",
    "curatedcatalyticbenchmark",
    "curatedliterature",
    "mcsapmiddoi",
}
SUPPORTED_REFERENCE_TYPE_TOKENS = (
    "active",
    "binding",
    "catalytic",
    "cofactor",
    "ligand",
    "metal",
    "mutation",
    "mutagenesis",
    "substrate",
)
MAPPING_ASSUMPTION_TOKENS = ("uniprot", "mature", "isoform", "offset", "precursor")


def _empty_reference_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_COLUMNS)


def _empty_reference_template_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_TEMPLATE_COLUMNS)


def _empty_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DETAIL_COLUMNS)


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_SUMMARY_COLUMNS)


def _empty_case_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_CASE_SUMMARY_COLUMNS)


def _empty_dataset_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DATASET_SUMMARY_COLUMNS)


def _empty_variant_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_COMPARISON_COLUMNS)


def _empty_variant_case_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_CASE_COMPARISON_COLUMNS)


def _empty_variant_dataset_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_DATASET_COMPARISON_COLUMNS)


def _empty_variant_detail_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS)


def _empty_variant_remediation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_REMEDIATION_COLUMNS)


def _empty_variant_remediation_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_REMEDIATION_SUMMARY_COLUMNS)


def _empty_reference_quality_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_QUALITY_COLUMNS)


def _empty_reference_quality_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_QUALITY_SUMMARY_COLUMNS)


def _empty_reference_structure_validation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS)


def _empty_reference_structure_validation_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_SUMMARY_COLUMNS)


def _empty_reference_readiness_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_READINESS_QUEUE_COLUMNS)


def _empty_reference_readiness_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS)


def _empty_reference_readiness_case_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_READINESS_CASE_SUMMARY_COLUMNS)


def _empty_interpretation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_INTERPRETATION_COLUMNS)


def _empty_case_interpretation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_CASE_INTERPRETATION_COLUMNS)


def _case_interpretation_matrix_columns(top_ns: Sequence[int] = (1, 3, 5)) -> list[str]:
    columns = list(BENCHMARK_CASE_INTERPRETATION_MATRIX_BASE_COLUMNS)
    for top_n in top_ns:
        prefix = f"top{int(top_n)}"
        columns.extend(
            [
                f"{prefix}_claim_status",
                f"{prefix}_claim_ready",
                f"{prefix}_coverage_ratio",
                f"{prefix}_best_rank",
                f"{prefix}_best_pocket_id",
                f"{prefix}_benchmark_status",
            ]
        )
    return columns


def _empty_case_interpretation_matrix_df(top_ns: Sequence[int] = (1, 3, 5)) -> pd.DataFrame:
    return pd.DataFrame(columns=_case_interpretation_matrix_columns(top_ns))


def _empty_dataset_interpretation_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DATASET_INTERPRETATION_COLUMNS)


def _empty_dataset_interpretation_queue_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS)


def _simplify_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _find_column(frame: pd.DataFrame, aliases: set[str]) -> Optional[str]:
    simplified = {_simplify_column_name(column): column for column in frame.columns}
    for alias in aliases:
        normalized_alias = _simplify_column_name(alias)
        if normalized_alias in simplified:
            return simplified[normalized_alias]
    return None


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_int(value: object) -> Optional[int]:
    try:
        text = _safe_text(value)
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
        {"sep": r"\s+", "engine": "python"},
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


def build_pocket_benchmark_reference_template(*, include_examples: bool = True) -> pd.DataFrame:
    """Build an editable curated catalytic residue benchmark template."""

    if not include_examples:
        return _empty_reference_template_df()
    rows = [
        {
            "benchmark_id": "case-001",
            "chain": "A",
            "resid": 195,
            "resname": "SER",
            "reference_type": "Catalytic residue",
            "reference_source": "M-CSA / PMID / DOI",
            "reference_note": "Example nucleophile; replace with curated residue evidence.",
            "expected_pocket_id": "",
        },
        {
            "benchmark_id": "case-001",
            "chain": "A",
            "resid": 57,
            "resname": "HIS",
            "reference_type": "Catalytic residue",
            "reference_source": "M-CSA / PMID / DOI",
            "reference_note": "Example catalytic base; keep benchmark_id identical for the same enzyme/structure.",
            "expected_pocket_id": "",
        },
        {
            "benchmark_id": "case-002",
            "chain": "",
            "resid": 123,
            "resname": "",
            "reference_type": "Binding residue",
            "reference_source": "curated literature",
            "reference_note": "Blank chain acts as wildcard when source numbering is chain-agnostic.",
            "expected_pocket_id": "",
        },
    ]
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_TEMPLATE_COLUMNS)


def build_pocket_benchmark_reference_template_markdown() -> str:
    """Explain the curated benchmark reference template fields."""

    lines = [
        "# Pocket benchmark reference template",
        "",
        "Use this template to collect curated catalytic or binding residues for enzyme pocket accuracy checks.",
        "",
        "## Columns",
        "",
        "- `benchmark_id`: Case identifier. Use one ID per enzyme/structure; examples include a PDB ID, M-CSA entry, or local dataset case ID.",
        "- `chain`: PDB chain. Leave blank only when the curated source is chain-agnostic; blank chain is treated as wildcard.",
        "- `resid`: Required residue number in the numbering system you want to benchmark.",
        "- `resname`: Optional three-letter residue name such as `SER`, `HIS`, or `ASP`.",
        "- `reference_type`: Evidence role, for example `Catalytic residue`, `Binding residue`, `Metal binding`, or `Mutagenesis`.",
        "- `reference_source`: Stable source label such as `M-CSA`, `PMID:...`, `DOI:...`, or a curated dataset name.",
        "- `reference_note`: Free-text note for EC number, UniProt accession, mature-chain offset, paper quote, or curation caveat.",
        "- `expected_pocket_id`: Optional pocket ID if a validated pocket label is already known.",
        "",
        "## Rules",
        "",
        "- Keep all catalytic residues from the same enzyme/structure under the same `benchmark_id`.",
        "- Prefer structure author residue numbering when benchmarking uploaded PDB coordinates.",
        "- If using UniProt or mature-chain numbering, record the mapping assumption in `reference_note`.",
        "- Do not mix different structures under the same `benchmark_id` unless residue numbering is intentionally shared.",
    ]
    return "\n".join(lines).strip() + "\n"


def _extract_residue_token(value: object) -> tuple[str, str, Optional[int]]:
    text = _safe_text(value)
    if not text:
        return "", "", None

    aa3_match = re.search(
        r"\b(?P<resname>Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)\.?\s*[- ]?\s*(?P<resid>-?\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if aa3_match:
        return "", aa3_match.group("resname").upper()[:3], _safe_int(aa3_match.group("resid"))

    chain_first = re.search(r"\b(?P<chain>[A-Za-z0-9])\s*[:/_-]\s*(?P<resid>-?\d+)\b", text)
    if chain_first:
        return chain_first.group("chain").strip(), "", _safe_int(chain_first.group("resid"))

    resid_first = re.search(r"\b(?P<resid>-?\d+)\s*[:/_-]\s*(?P<chain>[A-Za-z0-9])\b", text)
    if resid_first:
        return resid_first.group("chain").strip(), "", _safe_int(resid_first.group("resid"))

    one_letter_mutation = re.search(r"\b[A-Z](?P<resid>-?\d+)[A-Z]\b", text)
    if one_letter_mutation:
        return "", "", _safe_int(one_letter_mutation.group("resid"))

    return "", "", _safe_int(text)


def _residue_label(chain: str, resid: int, resname: str = "") -> str:
    resname_text = _safe_text(resname).upper()
    prefix = f"{resname_text} " if resname_text else ""
    chain_text = _safe_text(chain)
    return f"{prefix}{chain_text}{int(resid)}".strip()


def parse_benchmark_reference_table(text: str, *, source_hint: str = "Curated benchmark") -> tuple[pd.DataFrame, dict[str, str]]:
    """Parse curated catalytic/reference residues for pocket accuracy checks.

    Accepted tables are CSV/TSV/whitespace-delimited. Required information is
    only a residue number; chain is optional and acts as a wildcard when blank.
    """

    frame = _read_delimited_table(text)
    if frame.empty:
        return _empty_reference_df(), {
            "status": "empty",
            "reference_rows": "0",
            "reason": "No benchmark rows could be parsed.",
        }

    resid_column = _find_column(frame, RESID_ALIASES)
    token_column = _find_column(frame, RESIDUE_TOKEN_ALIASES)
    chain_column = _find_column(frame, CHAIN_ALIASES)
    resname_column = _find_column(frame, RESNAME_ALIASES)
    type_column = _find_column(frame, TYPE_ALIASES)
    source_column = _find_column(frame, SOURCE_ALIASES)
    note_column = _find_column(frame, NOTE_ALIASES)
    expected_pocket_column = _find_column(frame, EXPECTED_POCKET_ALIASES)
    benchmark_id_column = _find_column(frame, BENCHMARK_ID_ALIASES)

    rows: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        row_dict = row.to_dict()
        token_chain = ""
        token_resname = ""
        token_resid = None
        if token_column:
            token_chain, token_resname, token_resid = _extract_residue_token(row_dict.get(token_column))

        resid = _safe_int(row_dict.get(resid_column)) if resid_column else None
        if resid is None:
            resid = token_resid
        if resid is None:
            continue

        chain = _safe_text(row_dict.get(chain_column, "") if chain_column else "") or token_chain
        resname = _safe_text(row_dict.get(resname_column, "") if resname_column else "").upper() or token_resname
        rows.append(
            {
                "benchmark_id": _safe_text(row_dict.get(benchmark_id_column, "") if benchmark_id_column else ""),
                "chain": chain,
                "resid": int(resid),
                "resname": resname,
                "reference_type": _safe_text(row_dict.get(type_column, "") if type_column else "Catalytic residue") or "Catalytic residue",
                "reference_source": _safe_text(row_dict.get(source_column, "") if source_column else source_hint) or source_hint,
                "reference_note": _safe_text(row_dict.get(note_column, "") if note_column else ""),
                "expected_pocket_id": _safe_text(row_dict.get(expected_pocket_column, "") if expected_pocket_column else ""),
            }
        )

    if not rows:
        return _empty_reference_df(), {
            "status": "empty",
            "reference_rows": "0",
            "reason": "No valid residue numbers were found.",
        }

    reference_df = pd.DataFrame(rows)
    reference_df = reference_df.drop_duplicates(subset=["benchmark_id", "chain", "resid", "reference_type"]).sort_values(
        ["benchmark_id", "chain", "resid", "reference_type"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return reference_df[BENCHMARK_REFERENCE_COLUMNS], {
        "status": "ok",
        "reference_rows": str(len(reference_df)),
        "source": source_hint,
        "chain_specific_rows": str(int(reference_df["chain"].astype(str).str.strip().ne("").sum())),
        "wildcard_chain_rows": str(int(reference_df["chain"].astype(str).str.strip().eq("").sum())),
    }


def _normalize_pocket_rows(pocket_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if pocket_df is None or getattr(pocket_df, "empty", True) or "pocket_id" not in pocket_df.columns or "resid" not in pocket_df.columns:
        return pd.DataFrame(columns=["benchmark_id", "pocket_id", "chain", "resid", "resname"])
    working = pocket_df.copy()
    benchmark_id_column = _find_column(working, BENCHMARK_ID_ALIASES)
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["benchmark_id", "pocket_id", "chain", "resid", "resname"])
    working["benchmark_id"] = working[benchmark_id_column].astype(str).str.strip() if benchmark_id_column else ""
    working["pocket_id"] = working["pocket_id"].astype(str).str.strip()
    working["chain"] = working["chain"].astype(str).str.strip() if "chain" in working.columns else ""
    working["resname"] = working["resname"].astype(str).str.strip().str.upper() if "resname" in working.columns else ""
    working["resid"] = working["resid"].astype(int)
    return working[["benchmark_id", "pocket_id", "chain", "resid", "resname"]].drop_duplicates().reset_index(drop=True)


def _filter_rows_for_benchmark_id(frame: Optional[pd.DataFrame], benchmark_id: str) -> Optional[pd.DataFrame]:
    if frame is None or getattr(frame, "empty", True):
        return frame
    benchmark_id_column = _find_column(frame, BENCHMARK_ID_ALIASES)
    if not benchmark_id_column:
        return frame
    working = frame.copy()
    expected = str(benchmark_id or "").strip()
    values = working[benchmark_id_column].astype(str).str.strip()
    return working[values.eq(expected) | values.eq("")].reset_index(drop=True)


def _ranked_pocket_ids(pocket_df: pd.DataFrame, pocket_summary_df: Optional[pd.DataFrame]) -> list[str]:
    if pocket_df.empty:
        return []

    if pocket_summary_df is not None and not getattr(pocket_summary_df, "empty", True) and "pocket_id" in pocket_summary_df.columns:
        summary = pocket_summary_df.copy()
        summary["pocket_id"] = summary["pocket_id"].astype(str).str.strip()
        if "smart_rank_order" in summary.columns:
            summary["_rank_value"] = pd.to_numeric(summary["smart_rank_order"], errors="coerce")
        elif "rank" in summary.columns:
            summary["_rank_value"] = pd.to_numeric(summary["rank"], errors="coerce")
        else:
            summary["_rank_value"] = range(1, len(summary) + 1)
        fallback_rank = pd.Series(range(1, len(summary) + 1), index=summary.index)
        summary["_rank_value"] = pd.to_numeric(summary["_rank_value"], errors="coerce").fillna(fallback_rank)
        if "smart_rank_score" in summary.columns:
            summary["_score_value"] = pd.to_numeric(summary["smart_rank_score"], errors="coerce").fillna(0.0)
        elif "score" in summary.columns:
            summary["_score_value"] = pd.to_numeric(summary["score"], errors="coerce").fillna(0.0)
        else:
            summary["_score_value"] = 0.0
        ranked = summary.sort_values(["_rank_value", "_score_value", "pocket_id"], ascending=[True, False, True])["pocket_id"].tolist()
    else:
        ranked = pocket_df["pocket_id"].dropna().astype(str).str.strip().drop_duplicates().tolist()

    pocket_ids = set(pocket_df["pocket_id"].astype(str).str.strip().tolist())
    ordered = [pocket_id for pocket_id in ranked if pocket_id in pocket_ids]
    for pocket_id in pocket_df["pocket_id"].astype(str).str.strip().drop_duplicates().tolist():
        if pocket_id not in ordered:
            ordered.append(pocket_id)
    return ordered


def _reference_rows(reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if reference_df is None or getattr(reference_df, "empty", True):
        return _empty_reference_df()
    working = reference_df.copy()
    if "resid" not in working.columns:
        return _empty_reference_df()
    for column in BENCHMARK_REFERENCE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return _empty_reference_df()
    working["resid"] = working["resid"].astype(int)
    for column in ("benchmark_id", "chain", "resname", "reference_type", "reference_source", "reference_note", "expected_pocket_id"):
        working[column] = working[column].map(_safe_text)
    working["resname"] = working["resname"].str.upper()
    return working[BENCHMARK_REFERENCE_COLUMNS].drop_duplicates(subset=["benchmark_id", "chain", "resid", "reference_type"]).reset_index(drop=True)


def _reference_quality_issue(
    issue_number: int,
    reference_row: pd.Series,
    *,
    severity: str,
    issue_type: str,
    suggested_action: str,
    quality_warning: str,
) -> dict[str, object]:
    chain = _safe_text(reference_row.get("chain"))
    resid = int(reference_row.get("resid"))
    resname = _safe_text(reference_row.get("resname")).upper()
    return {
        "issue_id": f"REFQ-{issue_number:03d}",
        "severity": severity,
        "issue_type": issue_type,
        "benchmark_id": _safe_text(reference_row.get("benchmark_id")),
        "chain": chain,
        "resid": resid,
        "resname": resname,
        "residue_label": _residue_label(chain, resid, resname),
        "reference_type": _safe_text(reference_row.get("reference_type")),
        "reference_source": _safe_text(reference_row.get("reference_source")),
        "reference_note": _safe_text(reference_row.get("reference_note")),
        "suggested_action": suggested_action,
        "quality_warning": quality_warning,
    }


def _source_is_generic(value: object) -> bool:
    return _simplify_column_name(value) in GENERIC_REFERENCE_SOURCE_LABELS


def build_pocket_benchmark_reference_quality_issues(reference_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Flag curation risks in uploaded benchmark reference residues."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_quality_df()

    issues: list[dict[str, object]] = []

    def add_issue(reference_row: pd.Series, severity: str, issue_type: str, suggested_action: str, quality_warning: str) -> None:
        issues.append(
            _reference_quality_issue(
                len(issues) + 1,
                reference_row,
                severity=severity,
                issue_type=issue_type,
                suggested_action=suggested_action,
                quality_warning=quality_warning,
            )
        )

    for _, reference in references.iterrows():
        benchmark_id = _safe_text(reference.get("benchmark_id"))
        chain = _safe_text(reference.get("chain"))
        resname = _safe_text(reference.get("resname"))
        reference_type = _safe_text(reference.get("reference_type"))
        reference_source = _safe_text(reference.get("reference_source"))
        note_text = f"{reference_source} {_safe_text(reference.get('reference_note'))}".lower()

        if not benchmark_id:
            add_issue(
                reference,
                "P1",
                "missing_benchmark_id",
                "Assign a stable benchmark_id/case_id before using this row in batch benchmark aggregation.",
                "Rows without benchmark_id are grouped as an unnamed case and can hide case-level errors.",
            )
        if _source_is_generic(reference_source):
            add_issue(
                reference,
                "P1",
                "generic_reference_source",
                "Replace the generic source with M-CSA ID, PDB ID, PMID, DOI, or another stable curated dataset label.",
                "Generic sources make benchmark residues difficult to audit or reproduce.",
            )
        if not chain:
            add_issue(
                reference,
                "P2",
                "wildcard_chain",
                "Map the residue to a PDB chain, or document why chain-agnostic matching is intended.",
                "Blank chain matches any pocket chain and can overestimate catalytic coverage.",
            )
        if not resname:
            add_issue(
                reference,
                "P2",
                "missing_resname",
                "Fill the expected three-letter residue name and verify it against the structure.",
                "Missing residue identity weakens numbering and mapping validation.",
            )
        if reference_type and not any(token in reference_type.lower() for token in SUPPORTED_REFERENCE_TYPE_TOKENS):
            add_issue(
                reference,
                "P2",
                "unsupported_reference_type",
                "Use a functional role such as Catalytic residue, Binding residue, Metal binding, Mutagenesis, Substrate, or Cofactor.",
                "Unrecognized reference_type values can make benchmark interpretation ambiguous.",
            )
        if any(token in note_text for token in MAPPING_ASSUMPTION_TOKENS):
            add_issue(
                reference,
                "P2",
                "mapping_assumption_note",
                "Record the exact UniProt/PDB numbering conversion and verify residue identity in the uploaded structure.",
                "UniProt, mature-chain, isoform, precursor, or offset assumptions can shift catalytic residue numbers.",
            )

    for _, group in references.groupby(["benchmark_id", "chain", "resid"], dropna=False):
        if len(group) <= 1:
            continue
        reference_types = group["reference_type"].map(_safe_text).str.lower().drop_duplicates()
        sources = group["reference_source"].map(_safe_text).str.lower().drop_duplicates()
        if len(reference_types) > 1 or len(sources) > 1:
            add_issue(
                group.iloc[0],
                "P3",
                "multi_role_or_source_residue",
                "Keep the duplicated evidence if intentional, but consolidate notes so reviewers understand why the same residue has multiple roles or sources.",
                "Multiple roles or sources for one residue are valid evidence, but should be explicit in curated benchmark records.",
            )

    if not issues:
        return _empty_reference_quality_df()
    return pd.DataFrame(issues, columns=BENCHMARK_REFERENCE_QUALITY_COLUMNS)


def build_pocket_benchmark_reference_quality_summary(quality_issue_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize benchmark reference curation issues by severity and type."""

    if quality_issue_df is None or getattr(quality_issue_df, "empty", True):
        return _empty_reference_quality_summary_df()
    working = quality_issue_df.copy()
    for column in BENCHMARK_REFERENCE_QUALITY_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    rows: list[dict[str, object]] = []
    severity_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    for (severity, issue_type), group in working.groupby(["severity", "issue_type"], dropna=False):
        severity_text = _safe_text(severity) or "P3"
        case_count = int(group["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().nunique())
        residue_count = int(
            group[["benchmark_id", "chain", "resid"]]
            .astype(str)
            .drop_duplicates()
            .shape[0]
        )
        if severity_text in {"P0", "P1"}:
            status = "needs-curation"
            warning = "Fix before using this reference set as a batch accuracy benchmark."
        elif severity_text == "P2":
            status = "review"
            warning = "Review before interpreting precision metrics."
        else:
            status = "informational"
            warning = "Document the curation decision for traceability."
        rows.append(
            {
                "severity": severity_text,
                "issue_type": _safe_text(issue_type),
                "issue_count": int(len(group)),
                "affected_case_count": case_count,
                "affected_residue_count": residue_count,
                "suggested_action": _safe_text(group.iloc[0].get("suggested_action")),
                "summary_status": status,
                "summary_warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_QUALITY_SUMMARY_COLUMNS).sort_values(
        ["severity", "issue_count", "issue_type"],
        key=lambda series: series.map(severity_rank).fillna(series) if series.name == "severity" else series,
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_pocket_benchmark_reference_quality_checklist_markdown(
    quality_issue_df: Optional[pd.DataFrame],
    quality_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render curation issues as a reviewer checklist."""

    if quality_issue_df is None or getattr(quality_issue_df, "empty", True):
        return ""
    issues = quality_issue_df.copy()
    summary = build_pocket_benchmark_reference_quality_summary(issues) if quality_summary_df is None else quality_summary_df
    lines = [
        "# Benchmark reference curation checklist",
        "",
        "Resolve or explicitly accept these issues before using the reference table as a precision benchmark.",
        "",
    ]
    if summary is not None and not getattr(summary, "empty", True):
        lines.extend(["## Summary", ""])
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.severity} `{row.issue_type}`: {int(row.issue_count)} issues / {int(row.affected_residue_count)} residues / {row.summary_status}."
            )
        lines.append("")
    lines.extend(["## Actions", ""])
    severity_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    issues["_severity_order"] = issues["severity"].map(severity_order).fillna(9)
    issues = issues.sort_values(["_severity_order", "issue_type", "benchmark_id", "chain", "resid"]).drop(columns=["_severity_order"])
    for row in issues.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        lines.append(f"- [ ] {row.severity} `{row.issue_type}` for {case_text}, residue `{row.residue_label}`: {row.suggested_action}")
    return "\n".join(lines).strip() + "\n"


def _structure_residue_rows(atom_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if atom_df is None or getattr(atom_df, "empty", True):
        return pd.DataFrame(columns=["chain", "resid", "resname"])
    working = atom_df.copy()
    if "resid" not in working.columns:
        return pd.DataFrame(columns=["chain", "resid", "resname"])
    for column in ("chain", "resname"):
        if column not in working.columns:
            working[column] = ""
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["chain", "resid", "resname"])
    working["resid"] = working["resid"].astype(int)
    working["chain"] = working["chain"].map(_safe_text)
    working["resname"] = working["resname"].map(_safe_text).str.upper()
    if "record_type" in working.columns:
        atom_rows = working[working["record_type"].astype(str).str.upper().eq("ATOM")].copy()
        if not atom_rows.empty:
            working = atom_rows
    return working[["chain", "resid", "resname"]].drop_duplicates().reset_index(drop=True)


def _structure_validation_issue(
    issue_number: int,
    reference_row: pd.Series,
    *,
    severity: str,
    issue_type: str,
    structure_chains: str,
    structure_resnames: str,
    matched_chain: str,
    matched_resname: str,
    suggested_action: str,
    validation_warning: str,
) -> dict[str, object]:
    chain = _safe_text(reference_row.get("chain"))
    resid = int(reference_row.get("resid"))
    resname = _safe_text(reference_row.get("resname")).upper()
    return {
        "issue_id": f"REFS-{issue_number:03d}",
        "severity": severity,
        "issue_type": issue_type,
        "benchmark_id": _safe_text(reference_row.get("benchmark_id")),
        "chain": chain,
        "resid": resid,
        "resname": resname,
        "residue_label": _residue_label(chain, resid, resname),
        "structure_chains": structure_chains,
        "structure_resnames": structure_resnames,
        "matched_chain": matched_chain,
        "matched_resname": matched_resname,
        "reference_source": _safe_text(reference_row.get("reference_source")),
        "reference_note": _safe_text(reference_row.get("reference_note")),
        "suggested_action": suggested_action,
        "validation_warning": validation_warning,
    }


def build_pocket_benchmark_reference_structure_validation(
    reference_df: Optional[pd.DataFrame],
    atom_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Validate benchmark reference residues against residues present in the uploaded structure."""

    references = _reference_rows(reference_df)
    structure_residues = _structure_residue_rows(atom_df)
    if references.empty or structure_residues.empty:
        return _empty_reference_structure_validation_df()

    issues: list[dict[str, object]] = []

    def add_issue(
        reference_row: pd.Series,
        *,
        severity: str,
        issue_type: str,
        structure_chains: str,
        structure_resnames: str,
        matched_chain: str,
        matched_resname: str,
        suggested_action: str,
        validation_warning: str,
    ) -> None:
        issues.append(
            _structure_validation_issue(
                len(issues) + 1,
                reference_row,
                severity=severity,
                issue_type=issue_type,
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain=matched_chain,
                matched_resname=matched_resname,
                suggested_action=suggested_action,
                validation_warning=validation_warning,
            )
        )

    for _, reference in references.iterrows():
        chain = _safe_text(reference.get("chain"))
        resid = int(reference.get("resid"))
        resname = _safe_text(reference.get("resname")).upper()
        same_resid = structure_residues[structure_residues["resid"].astype(int).eq(resid)]
        if chain:
            same_resid = same_resid[same_resid["chain"].map(_safe_text).eq(chain)]

        if same_resid.empty:
            all_same_number = structure_residues[structure_residues["resid"].astype(int).eq(resid)]
            structure_chains = ";".join(sorted(all_same_number["chain"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist()))
            structure_resnames = ";".join(sorted(all_same_number["resname"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist()))
            add_issue(
                reference,
                severity="P1",
                issue_type="reference_residue_absent",
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain="",
                matched_resname="",
                suggested_action="Check whether the reference uses UniProt, mature-chain, isoform, or another PDB author's numbering before interpreting coverage.",
                validation_warning="The curated reference residue was not found in the uploaded structure.",
            )
            continue

        chains = sorted(same_resid["chain"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist())
        resnames = sorted(same_resid["resname"].map(_safe_text).replace("", pd.NA).dropna().unique().tolist())
        structure_chains = ";".join(chains)
        structure_resnames = ";".join(resnames)

        if not chain and len(chains) > 1:
            add_issue(
                reference,
                severity="P2",
                issue_type="wildcard_chain_ambiguous_in_structure",
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain="",
                matched_resname=structure_resnames,
                suggested_action="Choose the intended PDB chain for this benchmark residue or split the reference into chain-specific rows.",
                validation_warning="Blank chain matches multiple chains in the uploaded structure and can inflate benchmark coverage.",
            )

        if resname and resname not in resnames:
            add_issue(
                reference,
                severity="P1",
                issue_type="reference_resname_mismatch",
                structure_chains=structure_chains,
                structure_resnames=structure_resnames,
                matched_chain=structure_chains,
                matched_resname=structure_resnames,
                suggested_action="Verify residue numbering and expected residue identity against the uploaded PDB before using this row for accuracy claims.",
                validation_warning="The reference residue name does not match the residue identity in the uploaded structure.",
            )

    if not issues:
        return _empty_reference_structure_validation_df()
    return pd.DataFrame(issues, columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS)


def build_pocket_benchmark_reference_structure_validation_summary(
    validation_issue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize structure validation issues for benchmark reference residues."""

    if validation_issue_df is None or getattr(validation_issue_df, "empty", True):
        return _empty_reference_structure_validation_summary_df()
    working = validation_issue_df.copy()
    for column in BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    rows: list[dict[str, object]] = []
    severity_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    for (severity, issue_type), group in working.groupby(["severity", "issue_type"], dropna=False):
        severity_text = _safe_text(severity) or "P3"
        if severity_text == "P1":
            status = "mapping-blocked"
            warning = "Fix before treating misses as pocket-detection errors."
        elif severity_text == "P2":
            status = "mapping-review"
            warning = "Review before trusting wildcard-chain coverage."
        else:
            status = "informational"
            warning = "Document the structure validation decision."
        rows.append(
            {
                "severity": severity_text,
                "issue_type": _safe_text(issue_type),
                "issue_count": int(len(group)),
                "affected_case_count": int(group["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().nunique()),
                "affected_residue_count": int(group[["benchmark_id", "chain", "resid"]].astype(str).drop_duplicates().shape[0]),
                "suggested_action": _safe_text(group.iloc[0].get("suggested_action")),
                "summary_status": status,
                "summary_warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_STRUCTURE_VALIDATION_SUMMARY_COLUMNS).sort_values(
        ["severity", "issue_count", "issue_type"],
        key=lambda series: series.map(severity_rank).fillna(series) if series.name == "severity" else series,
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_pocket_benchmark_reference_structure_validation_checklist_markdown(
    validation_issue_df: Optional[pd.DataFrame],
    validation_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render structure validation issues as a reviewer checklist."""

    if validation_issue_df is None or getattr(validation_issue_df, "empty", True):
        return ""
    issues = validation_issue_df.copy()
    summary = (
        build_pocket_benchmark_reference_structure_validation_summary(issues)
        if validation_summary_df is None
        else validation_summary_df
    )
    lines = [
        "# Benchmark reference structure validation checklist",
        "",
        "Resolve these structure-mapping issues before interpreting catalytic pocket benchmark misses as detection failures.",
        "",
    ]
    if summary is not None and not getattr(summary, "empty", True):
        lines.extend(["## Summary", ""])
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.severity} `{row.issue_type}`: {int(row.issue_count)} issues / {int(row.affected_residue_count)} residues / {row.summary_status}."
            )
        lines.append("")
    lines.extend(["## Actions", ""])
    severity_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    issues["_severity_order"] = issues["severity"].map(severity_order).fillna(9)
    issues = issues.sort_values(["_severity_order", "issue_type", "benchmark_id", "chain", "resid"]).drop(columns=["_severity_order"])
    for row in issues.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        structure_text = f"structure chains `{row.structure_chains or '-'}`, resnames `{row.structure_resnames or '-'}`"
        lines.append(
            f"- [ ] {row.severity} `{row.issue_type}` for {case_text}, residue `{row.residue_label}` ({structure_text}): {row.suggested_action}"
        )
    return "\n".join(lines).strip() + "\n"


def _reference_issue_queue_rows(issue_df: Optional[pd.DataFrame], *, issue_source: str, warning_column: str) -> list[dict[str, object]]:
    if issue_df is None or getattr(issue_df, "empty", True):
        return []
    working = issue_df.copy()
    for column in ("severity", "issue_type", "benchmark_id", "residue_label", "chain", "resid", "resname", "suggested_action", warning_column):
        if column not in working.columns:
            working[column] = ""
    rows: list[dict[str, object]] = []
    for _, issue in working.iterrows():
        priority = _safe_text(issue.get("severity")) or "P3"
        if priority not in {"P0", "P1", "P2"}:
            continue
        rows.append(
            {
                "priority": priority,
                "action_status": "blocker" if priority in {"P0", "P1"} else "review",
                "issue_source": issue_source,
                "issue_type": _safe_text(issue.get("issue_type")),
                "benchmark_id": _safe_text(issue.get("benchmark_id")),
                "residue_label": _safe_text(issue.get("residue_label")),
                "chain": _safe_text(issue.get("chain")),
                "resid": _safe_text(issue.get("resid")),
                "resname": _safe_text(issue.get("resname")).upper(),
                "suggested_action": _safe_text(issue.get("suggested_action")),
                "readiness_warning": _safe_text(issue.get(warning_column)),
            }
        )
    return rows


def build_pocket_benchmark_reference_readiness_queue(
    quality_issue_df: Optional[pd.DataFrame],
    structure_validation_issue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Combine reference curation and structure validation issues into one readiness queue."""

    rows = [
        *_reference_issue_queue_rows(quality_issue_df, issue_source="curation_quality", warning_column="quality_warning"),
        *_reference_issue_queue_rows(
            structure_validation_issue_df,
            issue_source="structure_validation",
            warning_column="validation_warning",
        ),
    ]
    if not rows:
        return _empty_reference_readiness_queue_df()

    severity_order = {"P0": 0, "P1": 1, "P2": 2}
    queue = pd.DataFrame(rows)
    queue["_priority_order"] = queue["priority"].map(severity_order).fillna(9)
    queue = queue.sort_values(
        ["_priority_order", "issue_source", "issue_type", "benchmark_id", "chain", "resid"],
        ascending=[True, True, True, True, True, True],
    ).drop(columns=["_priority_order"]).reset_index(drop=True)
    queue["action_id"] = [f"REFR-{index + 1:03d}" for index in range(len(queue))]
    return queue[BENCHMARK_REFERENCE_READINESS_QUEUE_COLUMNS]


def build_pocket_benchmark_reference_readiness_summary(
    reference_df: Optional[pd.DataFrame],
    quality_issue_df: Optional[pd.DataFrame],
    structure_validation_issue_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a one-row gate for whether benchmark reference rows are ready for accuracy claims."""

    references = _reference_rows(reference_df)
    queue = build_pocket_benchmark_reference_readiness_queue(quality_issue_df, structure_validation_issue_df)
    reference_count = int(len(references))
    curation_count = 0 if quality_issue_df is None or getattr(quality_issue_df, "empty", True) else int(len(quality_issue_df))
    structure_count = (
        0
        if structure_validation_issue_df is None or getattr(structure_validation_issue_df, "empty", True)
        else int(len(structure_validation_issue_df))
    )

    if reference_count <= 0:
        row = {
            "readiness_status": "no-reference",
            "reference_residue_count": 0,
            "curation_issue_count": curation_count,
            "structure_validation_issue_count": structure_count,
            "p0_p1_issue_count": 0,
            "p2_issue_count": 0,
            "blocking_issue_types": "",
            "review_issue_types": "",
            "recommended_action": "Upload curated catalytic residues before running benchmark readiness checks.",
            "readiness_warning": "Benchmark accuracy cannot be interpreted without reference residues.",
        }
        return pd.DataFrame([row], columns=BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS)

    p0_p1_count = (
        int(queue["priority"].astype(str).isin(["P0", "P1"]).sum())
        if not queue.empty and "priority" in queue.columns
        else 0
    )
    p2_count = int(queue["priority"].astype(str).eq("P2").sum()) if not queue.empty and "priority" in queue.columns else 0
    blocking_types = ""
    review_types = ""
    if not queue.empty:
        blocking_types = "; ".join(
            sorted(queue.loc[queue["priority"].astype(str).isin(["P0", "P1"]), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )
        review_types = "; ".join(
            sorted(queue.loc[queue["priority"].astype(str).eq("P2"), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )

    if p0_p1_count > 0:
        readiness_status = "blocked"
        recommended_action = "Resolve P0/P1 reference curation or structure-mapping blockers before using coverage as a precision claim."
        warning = "Current benchmark misses may reflect reference numbering or curation errors, not pocket-detection errors."
    elif p2_count > 0:
        readiness_status = "review-needed"
        recommended_action = "Review P2 issues, especially wildcard chain and residue identity assumptions, before publishing benchmark coverage."
        warning = "Benchmark coverage can be inspected, but should be labeled as reviewer-pending."
    else:
        readiness_status = "ready"
        recommended_action = "Reference residues are ready for catalytic pocket coverage interpretation."
        warning = "No benchmark reference readiness blockers detected."

    row = {
        "readiness_status": readiness_status,
        "reference_residue_count": reference_count,
        "curation_issue_count": curation_count,
        "structure_validation_issue_count": structure_count,
        "p0_p1_issue_count": p0_p1_count,
        "p2_issue_count": p2_count,
        "blocking_issue_types": blocking_types,
        "review_issue_types": review_types,
        "recommended_action": recommended_action,
        "readiness_warning": warning,
    }
    return pd.DataFrame([row], columns=BENCHMARK_REFERENCE_READINESS_SUMMARY_COLUMNS)


def _normalized_case_id(value: object, default_benchmark_id: str) -> str:
    return _safe_text(value) or _safe_text(default_benchmark_id) or "current"


def _readiness_issue_frame(issue_df: Optional[pd.DataFrame], *, issue_source: str, default_benchmark_id: str) -> pd.DataFrame:
    if issue_df is None or getattr(issue_df, "empty", True):
        return pd.DataFrame(columns=["benchmark_id", "severity", "issue_type", "issue_source"])
    working = issue_df.copy()
    for column in ("benchmark_id", "severity", "issue_type"):
        if column not in working.columns:
            working[column] = ""
    working["benchmark_id"] = working["benchmark_id"].map(lambda value: _normalized_case_id(value, default_benchmark_id))
    working["severity"] = working["severity"].map(_safe_text)
    working["issue_type"] = working["issue_type"].map(_safe_text)
    working["issue_source"] = issue_source
    return working[["benchmark_id", "severity", "issue_type", "issue_source"]].reset_index(drop=True)


def build_pocket_benchmark_reference_readiness_case_summary(
    reference_df: Optional[pd.DataFrame],
    quality_issue_df: Optional[pd.DataFrame],
    structure_validation_issue_df: Optional[pd.DataFrame],
    *,
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Build per-case benchmark reference readiness gates."""

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_reference_readiness_case_summary_df()

    fallback_id = _safe_text(default_benchmark_id) or "current"
    references = references.copy()
    references["benchmark_id"] = references["benchmark_id"].map(lambda value: _normalized_case_id(value, fallback_id))
    quality_issues = _readiness_issue_frame(quality_issue_df, issue_source="curation_quality", default_benchmark_id=fallback_id)
    structure_issues = _readiness_issue_frame(
        structure_validation_issue_df,
        issue_source="structure_validation",
        default_benchmark_id=fallback_id,
    )
    issue_frame = pd.concat([quality_issues, structure_issues], ignore_index=True)

    case_ids = sorted(
        set(references["benchmark_id"].map(_safe_text).tolist())
        | set(issue_frame["benchmark_id"].map(_safe_text).replace("", pd.NA).dropna().tolist())
    )
    rows: list[dict[str, object]] = []
    for benchmark_id in case_ids:
        case_references = references[references["benchmark_id"].astype(str).eq(benchmark_id)]
        case_issues = issue_frame[issue_frame["benchmark_id"].astype(str).eq(benchmark_id)]
        curation_count = int(case_issues["issue_source"].astype(str).eq("curation_quality").sum())
        structure_count = int(case_issues["issue_source"].astype(str).eq("structure_validation").sum())
        p0_p1_count = int(case_issues["severity"].astype(str).isin(["P0", "P1"]).sum())
        p2_count = int(case_issues["severity"].astype(str).eq("P2").sum())
        blocking_types = "; ".join(
            sorted(case_issues.loc[case_issues["severity"].astype(str).isin(["P0", "P1"]), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )
        review_types = "; ".join(
            sorted(case_issues.loc[case_issues["severity"].astype(str).eq("P2"), "issue_type"].map(_safe_text).drop_duplicates().tolist())
        )

        if p0_p1_count > 0:
            readiness_status = "blocked"
            recommended_action = "Resolve this case's P0/P1 reference curation or structure-mapping blockers before using its coverage in dataset-level claims."
            warning = "This case can distort dataset coverage until reference numbering and curation blockers are fixed."
        elif p2_count > 0:
            readiness_status = "review-needed"
            recommended_action = "Review this case's P2 assumptions before treating its coverage as publication-ready."
            warning = "This case can be inspected, but should be labeled reviewer-pending."
        else:
            readiness_status = "ready"
            recommended_action = "This case is ready for catalytic pocket coverage interpretation."
            warning = "No case-level readiness blockers detected."

        rows.append(
            {
                "benchmark_id": benchmark_id,
                "readiness_status": readiness_status,
                "reference_residue_count": int(len(case_references)),
                "curation_issue_count": curation_count,
                "structure_validation_issue_count": structure_count,
                "p0_p1_issue_count": p0_p1_count,
                "p2_issue_count": p2_count,
                "blocking_issue_types": blocking_types,
                "review_issue_types": review_types,
                "recommended_action": recommended_action,
                "readiness_warning": warning,
            }
        )

    if not rows:
        return _empty_reference_readiness_case_summary_df()
    return pd.DataFrame(rows, columns=BENCHMARK_REFERENCE_READINESS_CASE_SUMMARY_COLUMNS)


def build_pocket_benchmark_reference_readiness_checklist_markdown(
    readiness_queue_df: Optional[pd.DataFrame],
    readiness_summary_df: Optional[pd.DataFrame] = None,
) -> str:
    """Render the combined benchmark reference readiness gate as a checklist."""

    if readiness_queue_df is None or getattr(readiness_queue_df, "empty", True):
        return ""
    queue = readiness_queue_df.copy()
    summary = _empty_reference_readiness_summary_df() if readiness_summary_df is None else readiness_summary_df
    lines = [
        "# Benchmark reference readiness checklist",
        "",
        "Use this checklist before interpreting catalytic pocket benchmark coverage as an accuracy claim.",
        "",
    ]
    if summary is not None and not getattr(summary, "empty", True):
        row = summary.iloc[0]
        lines.extend(
            [
                "## Gate",
                "",
                f"- Status: `{_safe_text(row.get('readiness_status')) or '-'}`.",
                f"- P0/P1 blockers: {int(row.get('p0_p1_issue_count') or 0)}.",
                f"- P2 review items: {int(row.get('p2_issue_count') or 0)}.",
                f"- Recommended action: {_safe_text(row.get('recommended_action')) or '-'}",
                "",
            ]
        )
    lines.extend(["## Actions", ""])
    for row in queue.itertuples(index=False):
        case_text = f"case `{row.benchmark_id}`" if _safe_text(row.benchmark_id) else "unnamed case"
        lines.append(
            f"- [ ] {row.priority} `{row.issue_source}/{row.issue_type}` for {case_text}, residue `{row.residue_label or '-'}`: {row.suggested_action}"
        )
    return "\n".join(lines).strip() + "\n"


def _matches_reference(pocket_row: pd.Series, reference_row: pd.Series) -> bool:
    ref_benchmark_id = str(reference_row.get("benchmark_id") or "").strip()
    pocket_benchmark_id = str(pocket_row.get("benchmark_id") or "").strip()
    if ref_benchmark_id and pocket_benchmark_id and ref_benchmark_id != pocket_benchmark_id:
        return False
    if int(pocket_row.get("resid")) != int(reference_row.get("resid")):
        return False
    ref_chain = str(reference_row.get("chain") or "").strip()
    pocket_chain = str(pocket_row.get("chain") or "").strip()
    return not ref_chain or not pocket_chain or ref_chain == pocket_chain


def build_pocket_benchmark_details(
    reference_df: Optional[pd.DataFrame],
    pocket_df: Optional[pd.DataFrame],
    pocket_summary_df: Optional[pd.DataFrame] = None,
    *,
    top_thresholds: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    references = _reference_rows(reference_df)
    pockets = _normalize_pocket_rows(pocket_df)
    if references.empty:
        return _empty_detail_df()

    ranked_ids = _ranked_pocket_ids(pockets, pocket_summary_df)
    rank_map = {pocket_id: rank + 1 for rank, pocket_id in enumerate(ranked_ids)}
    rows: list[dict[str, object]] = []
    for reference in references.itertuples(index=False):
        ref_series = pd.Series(reference._asdict())
        matching_rows = []
        for pocket_row in pockets.itertuples(index=False):
            pocket_series = pd.Series(pocket_row._asdict())
            if _matches_reference(pocket_series, ref_series):
                matching_rows.append(pocket_series)
        matching_rows = sorted(matching_rows, key=lambda row: (rank_map.get(str(row.get("pocket_id")), 999999), str(row.get("pocket_id"))))
        matched_pocket_ids = [str(row.get("pocket_id")) for row in matching_rows if str(row.get("pocket_id"))]
        matched_rank = min([rank_map.get(pocket_id, 999999) for pocket_id in matched_pocket_ids], default=0)
        matched_pocket_id = matched_pocket_ids[0] if matched_pocket_ids else ""
        expected_pocket_id = str(ref_series.get("expected_pocket_id") or "").strip()
        rows.append(
            {
                **{column: ref_series.get(column, "") for column in BENCHMARK_REFERENCE_COLUMNS},
                "residue_label": _residue_label(str(ref_series.get("chain") or ""), int(ref_series.get("resid")), str(ref_series.get("resname") or "")),
                "matched": bool(matched_pocket_ids),
                "matched_pocket_id": matched_pocket_id,
                "matched_rank": int(matched_rank) if matched_rank and matched_rank < 999999 else 0,
                "matched_pocket_ids": ", ".join(dict.fromkeys(matched_pocket_ids)),
                "matched_top1": bool(matched_rank and matched_rank <= 1),
                "matched_top3": bool(matched_rank and matched_rank <= 3),
                "matched_top5": bool(matched_rank and matched_rank <= 5),
                "expected_pocket_matched": bool(expected_pocket_id and expected_pocket_id in matched_pocket_ids),
                "benchmark_warning": "" if matched_pocket_ids else "reference-residue-not-covered-by-any-pocket",
            }
        )

    detail_df = pd.DataFrame(rows)
    for threshold in top_thresholds:
        column = f"matched_top{int(threshold)}"
        if column not in detail_df.columns:
            detail_df[column] = pd.to_numeric(detail_df["matched_rank"], errors="coerce").fillna(0).between(1, int(threshold))
    return detail_df[BENCHMARK_DETAIL_COLUMNS]


def _summary_status(*, top_n: int, coverage_ratio: float, any_hit: bool, all_hit: bool, reference_count: int, pocket_count: int) -> tuple[str, str]:
    if reference_count <= 0:
        return "no-reference", "Upload curated catalytic residues to compute benchmark coverage."
    if pocket_count <= 0:
        return "no-pockets", "No pocket rows are available for benchmark comparison."
    if all_hit:
        return "topn-complete-hit", f"All reference residues are covered within Top-{top_n} pockets."
    if any_hit and top_n == 1:
        return "top1-partial-hit", "Top-1 covers at least one reference residue but misses part of the catalytic set."
    if any_hit:
        return "topn-partial-hit", f"Top-{top_n} covers some reference residues; inspect missed residues before claiming active-site precision."
    if coverage_ratio <= 0.0 and top_n == 1:
        return "top1-miss", "Top-1 does not cover the curated catalytic residues."
    return "topn-miss", f"Top-{top_n} does not cover the curated catalytic residues."


def build_pocket_benchmark_summary(
    reference_df: Optional[pd.DataFrame],
    pocket_df: Optional[pd.DataFrame],
    pocket_summary_df: Optional[pd.DataFrame] = None,
    *,
    top_ns: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    details = build_pocket_benchmark_details(reference_df, pocket_df, pocket_summary_df, top_thresholds=top_ns)
    references = _reference_rows(reference_df)
    pockets = _normalize_pocket_rows(pocket_df)
    ranked_ids = _ranked_pocket_ids(pockets, pocket_summary_df)
    reference_count = int(len(references))
    pocket_count = int(len(ranked_ids))
    if reference_count <= 0:
        return _empty_summary_df()

    rows: list[dict[str, object]] = []
    matched_rank_series = pd.to_numeric(details.get("matched_rank", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    best_rank_values = matched_rank_series[matched_rank_series > 0]
    best_rank = int(best_rank_values.min()) if not best_rank_values.empty else 0
    best_pocket_id = ""
    if best_rank:
        best_rows = details[matched_rank_series == best_rank]
        if not best_rows.empty:
            best_pocket_id = str(best_rows.iloc[0].get("matched_pocket_id") or "")

    for top_n in top_ns:
        top_n = max(1, int(top_n))
        matched_mask = matched_rank_series.between(1, top_n)
        matched_count = int(matched_mask.sum())
        coverage_ratio = round(float(matched_count) / float(reference_count), 3) if reference_count else 0.0
        any_hit = matched_count > 0
        all_hit = matched_count == reference_count and reference_count > 0
        matched_residues = ", ".join(details.loc[matched_mask, "residue_label"].astype(str).tolist()) if matched_count else "none"
        missed_residues = ", ".join(details.loc[~matched_mask, "residue_label"].astype(str).tolist()) if matched_count < reference_count else "none"
        top_pocket_id = ranked_ids[0] if ranked_ids else ""
        top_pocket_hit = bool((matched_rank_series == 1).any())
        status, warning = _summary_status(
            top_n=top_n,
            coverage_ratio=coverage_ratio,
            any_hit=any_hit,
            all_hit=all_hit,
            reference_count=reference_count,
            pocket_count=pocket_count,
        )
        rows.append(
            {
                "top_n": top_n,
                "reference_residue_count": reference_count,
                "matched_reference_count": matched_count,
                "coverage_ratio": coverage_ratio,
                "any_hit": any_hit,
                "all_hit": all_hit,
                "best_rank": best_rank,
                "best_pocket_id": best_pocket_id,
                "top_pocket_id": top_pocket_id,
                "top_pocket_hit": top_pocket_hit,
                "matched_residues": matched_residues,
                "missed_residues": missed_residues,
                "benchmark_status": status,
                "benchmark_warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_SUMMARY_COLUMNS)


def build_pocket_benchmark_interpretation_summary(
    benchmark_summary_df: Optional[pd.DataFrame],
    readiness_summary_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach reference readiness to each Top-N benchmark coverage row."""

    if benchmark_summary_df is None or getattr(benchmark_summary_df, "empty", True):
        return _empty_interpretation_df()

    benchmark = benchmark_summary_df.copy()
    for column in BENCHMARK_SUMMARY_COLUMNS:
        if column not in benchmark.columns:
            benchmark[column] = 0 if column in {"top_n", "reference_residue_count", "matched_reference_count", "best_rank"} else ""

    readiness_status = "unknown"
    readiness_action = "Run benchmark reference readiness checks before using coverage as a precision claim."
    readiness_warning = "Benchmark reference readiness was not available."
    blocker_count = 0
    review_count = 0
    if readiness_summary_df is not None and not getattr(readiness_summary_df, "empty", True):
        row = readiness_summary_df.iloc[0]
        readiness_status = _safe_text(row.get("readiness_status")) or "unknown"
        readiness_action = _safe_text(row.get("recommended_action")) or readiness_action
        readiness_warning = _safe_text(row.get("readiness_warning")) or readiness_warning
        blocker_count = int(row.get("p0_p1_issue_count") or 0)
        review_count = int(row.get("p2_issue_count") or 0)

    rows: list[dict[str, object]] = []
    for _, benchmark_row in benchmark.iterrows():
        top_n = int(pd.to_numeric(pd.Series([benchmark_row.get("top_n")]), errors="coerce").fillna(0).iloc[0] or 0)
        coverage_ratio = round(float(pd.to_numeric(pd.Series([benchmark_row.get("coverage_ratio")]), errors="coerce").fillna(0.0).iloc[0]), 3)
        matched_count = int(pd.to_numeric(pd.Series([benchmark_row.get("matched_reference_count")]), errors="coerce").fillna(0).iloc[0])
        reference_count = int(pd.to_numeric(pd.Series([benchmark_row.get("reference_residue_count")]), errors="coerce").fillna(0).iloc[0])
        best_rank = int(pd.to_numeric(pd.Series([benchmark_row.get("best_rank")]), errors="coerce").fillna(0).iloc[0])
        benchmark_status = _safe_text(benchmark_row.get("benchmark_status"))

        if readiness_status in {"blocked", "no-reference"} or blocker_count > 0:
            claim_status = "blocked"
            claim_ready = False
            interpretation_label = f"Top-{top_n} coverage is not claimable until benchmark reference blockers are fixed."
            recommended_action = readiness_action
            warning = readiness_warning
        elif readiness_status == "review-needed" or review_count > 0:
            claim_status = "review-needed"
            claim_ready = False
            interpretation_label = f"Top-{top_n} coverage is reviewer-pending because benchmark reference assumptions need sign-off."
            recommended_action = readiness_action
            warning = readiness_warning
        elif readiness_status == "ready":
            claim_status = "claim-ready"
            claim_ready = True
            if reference_count > 0 and matched_count == reference_count:
                interpretation_label = f"Top-{top_n} has complete curated residue coverage."
                recommended_action = "Use this coverage row as supported active-site pocket evidence, while still reporting the benchmark dataset size."
                warning = "Reference readiness passed; coverage can be interpreted directly."
            elif matched_count > 0:
                interpretation_label = f"Top-{top_n} has partial curated residue coverage."
                recommended_action = "Report partial coverage and inspect missed residues before claiming complete active-site localization."
                warning = "Reference readiness passed; incomplete coverage likely reflects detection/ranking limits or pocket boundary choices."
            else:
                interpretation_label = f"Top-{top_n} misses the curated residues."
                recommended_action = "Treat this as a pocket detection or ranking miss and inspect candidate generation, evidence routes, and thresholds."
                warning = "Reference readiness passed; this miss is more likely to reflect model behavior than reference curation."
        else:
            claim_status = "readiness-unknown"
            claim_ready = False
            interpretation_label = f"Top-{top_n} coverage cannot be interpreted as a precision claim until readiness is available."
            recommended_action = readiness_action
            warning = readiness_warning

        rows.append(
            {
                "top_n": top_n,
                "reference_residue_count": reference_count,
                "matched_reference_count": matched_count,
                "coverage_ratio": coverage_ratio,
                "benchmark_status": benchmark_status,
                "readiness_status": readiness_status,
                "claim_status": claim_status,
                "claim_ready": bool(claim_ready),
                "best_rank": best_rank,
                "best_pocket_id": _safe_text(benchmark_row.get("best_pocket_id")),
                "interpretation_label": interpretation_label,
                "recommended_action": recommended_action,
                "interpretation_warning": warning,
            }
        )

    return pd.DataFrame(rows, columns=BENCHMARK_INTERPRETATION_COLUMNS)


def build_pocket_benchmark_case_interpretation_summary(
    case_summary_df: Optional[pd.DataFrame],
    readiness_case_summary_df: Optional[pd.DataFrame] = None,
    *,
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Attach case-level readiness to each case-level Top-N benchmark coverage row."""

    if case_summary_df is None or getattr(case_summary_df, "empty", True) or "benchmark_id" not in case_summary_df.columns:
        return _empty_case_interpretation_df()

    case_summary = case_summary_df.copy()
    fallback_id = _safe_text(default_benchmark_id) or "current"
    case_summary["benchmark_id"] = case_summary["benchmark_id"].map(lambda value: _normalized_case_id(value, fallback_id))

    readiness_by_case: dict[str, pd.DataFrame] = {}
    if readiness_case_summary_df is not None and not getattr(readiness_case_summary_df, "empty", True) and "benchmark_id" in readiness_case_summary_df.columns:
        readiness = readiness_case_summary_df.copy()
        readiness["benchmark_id"] = readiness["benchmark_id"].map(lambda value: _normalized_case_id(value, fallback_id))
        for benchmark_id, group in readiness.groupby("benchmark_id", sort=False, dropna=False):
            readiness_by_case[_safe_text(benchmark_id)] = group.drop(columns=["benchmark_id"], errors="ignore").head(1).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for benchmark_id, group in case_summary.groupby("benchmark_id", sort=True, dropna=False):
        benchmark_id_text = _safe_text(benchmark_id) or fallback_id
        interpretation = build_pocket_benchmark_interpretation_summary(
            group.drop(columns=["benchmark_id"], errors="ignore"),
            readiness_by_case.get(benchmark_id_text),
        )
        for row in interpretation.to_dict(orient="records"):
            rows.append({"benchmark_id": benchmark_id_text, **row})

    if not rows:
        return _empty_case_interpretation_df()
    return pd.DataFrame(rows, columns=BENCHMARK_CASE_INTERPRETATION_COLUMNS)


def _claim_ready_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_text(value).strip().lower()
    return text in {"1", "true", "yes", "y", "ready", "claim-ready"}


def _normalized_claim_status(value: object) -> str:
    return re.sub(r"[\s_]+", "-", _safe_text(value).strip().lower())


def _case_matrix_status(
    has_claim_ready: bool,
    any_blocked: bool,
    any_review_needed: bool,
    any_readiness_unknown: bool,
) -> tuple[str, str]:
    if any_blocked:
        return "blocked", "Fix blocked Top-N case interpretation rows before using this case in dataset-level claims."
    if any_review_needed or any_readiness_unknown:
        return "review-needed", "Resolve reviewer-pending or unknown Top-N rows before using this case in dataset-level claims."
    if has_claim_ready:
        return "claim-ready", "Use the earliest claim-ready Top-N row as case-level benchmark support."
    return "no-claim-ready", "Inspect coverage misses, ranking thresholds or reference readiness before claiming this case."


def build_pocket_benchmark_case_interpretation_matrix(
    case_interpretation_df: Optional[pd.DataFrame],
    *,
    top_ns: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Pivot case-level Top-N interpretation rows into one row per benchmark case."""

    normalized_top_values: list[int] = []
    for value in top_ns or ():
        try:
            top_n_value = int(value)
        except (TypeError, ValueError):
            continue
        if top_n_value > 0:
            normalized_top_values.append(top_n_value)
    normalized_top_ns = tuple(sorted(set(normalized_top_values)))
    if (
        case_interpretation_df is None
        or getattr(case_interpretation_df, "empty", True)
        or "benchmark_id" not in case_interpretation_df.columns
        or "top_n" not in case_interpretation_df.columns
    ):
        return _empty_case_interpretation_matrix_df(normalized_top_ns or (1, 3, 5))

    working = case_interpretation_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_case_interpretation_matrix_df(normalized_top_ns or (1, 3, 5))

    working["top_n"] = working["top_n"].astype(int)
    if not normalized_top_ns:
        normalized_top_ns = tuple(sorted({int(value) for value in working["top_n"].dropna().tolist() if int(value) > 0}))
    if not normalized_top_ns:
        normalized_top_ns = (1, 3, 5)

    working = working[working["top_n"].isin(normalized_top_ns)].copy()
    if working.empty:
        return _empty_case_interpretation_matrix_df(normalized_top_ns)

    working["benchmark_id"] = working["benchmark_id"].map(lambda value: _safe_text(value) or "current")
    working["claim_status"] = working["claim_status"].map(_normalized_claim_status)
    working.loc[working["claim_status"].eq(""), "claim_status"] = "readiness-unknown"
    working["claim_ready"] = working["claim_ready"].map(_claim_ready_bool)
    working.loc[working["claim_ready"] & working["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, object]] = []
    for benchmark_id, group in working.groupby("benchmark_id", sort=True, dropna=False):
        group = group.sort_values("top_n")
        claim_ready_rows = group[group["claim_status"].eq("claim-ready")]
        best_claim = claim_ready_rows.iloc[0] if not claim_ready_rows.empty else None
        any_blocked = bool(group["claim_status"].eq("blocked").any())
        any_review_needed = bool(group["claim_status"].eq("review-needed").any())
        any_readiness_unknown = bool(group["claim_status"].eq("readiness-unknown").any())
        case_status, action = _case_matrix_status(
            best_claim is not None,
            any_blocked,
            any_review_needed,
            any_readiness_unknown,
        )
        row: dict[str, object] = {
            "benchmark_id": _safe_text(benchmark_id) or "current",
            "top_n_count": int(group["top_n"].nunique()),
            "best_claim_ready_top_n": int(best_claim.get("top_n") or 0) if best_claim is not None else 0,
            "best_claim_ready_coverage": round(float(best_claim.get("coverage_ratio") or 0.0), 3) if best_claim is not None else 0.0,
            "best_claim_ready_rank": int(best_claim.get("best_rank") or 0) if best_claim is not None else 0,
            "any_blocked": any_blocked,
            "any_review_needed": any_review_needed,
            "any_readiness_unknown": any_readiness_unknown,
            "case_interpretation_status": case_status,
            "recommended_action": action,
        }
        for top_n in normalized_top_ns:
            prefix = f"top{int(top_n)}"
            top_group = group[group["top_n"].eq(int(top_n))]
            top_row = top_group.iloc[0] if not top_group.empty else None
            row[f"{prefix}_claim_status"] = _safe_text(top_row.get("claim_status")) if top_row is not None else ""
            row[f"{prefix}_claim_ready"] = bool(top_row.get("claim_ready")) if top_row is not None else False
            row[f"{prefix}_coverage_ratio"] = round(float(top_row.get("coverage_ratio") or 0.0), 3) if top_row is not None else 0.0
            row[f"{prefix}_best_rank"] = int(top_row.get("best_rank") or 0) if top_row is not None else 0
            row[f"{prefix}_best_pocket_id"] = _safe_text(top_row.get("best_pocket_id")) if top_row is not None else ""
            row[f"{prefix}_benchmark_status"] = _safe_text(top_row.get("benchmark_status")) if top_row is not None else ""
        rows.append(row)

    if not rows:
        return _empty_case_interpretation_matrix_df(normalized_top_ns)
    return pd.DataFrame(rows, columns=_case_interpretation_matrix_columns(normalized_top_ns))


def _dataset_interpretation_status(
    case_count: int,
    claim_ready_count: int,
    blocked_count: int,
    review_count: int,
    unknown_count: int,
    top_n: int,
) -> tuple[str, str, str, str]:
    if case_count <= 0:
        return (
            "no-cases",
            f"Top-{top_n} dataset interpretation has no benchmark cases.",
            "Add benchmark cases before reporting dataset-level catalytic pocket coverage.",
            "No benchmark cases are available for dataset-level interpretation.",
        )
    if blocked_count > 0:
        return (
            "blocked",
            f"Top-{top_n} dataset interpretation is blocked because one or more benchmark cases are not claim-ready.",
            "Fix blocked cases before using dataset-level coverage as a precision claim.",
            "At least one benchmark case is blocked; dataset-level coverage is not claimable.",
        )
    if review_count > 0 or unknown_count > 0:
        return (
            "review-needed",
            f"Top-{top_n} dataset interpretation needs review because some cases are reviewer-pending or unknown.",
            "Review non-claim-ready cases before publishing dataset-level coverage.",
            "Some benchmark cases are not claim-ready; dataset-level coverage needs reviewer interpretation.",
        )
    if claim_ready_count == case_count:
        return (
            "claim-ready",
            f"Top-{top_n} dataset interpretation is claim-ready across all benchmark cases.",
            "Report dataset coverage with case count and mean claim-ready coverage.",
            "All benchmark cases are claim-ready for this Top-N dataset interpretation.",
        )
    return (
        "review-needed",
        f"Top-{top_n} dataset interpretation needs review because not all cases are claim-ready.",
        "Review non-claim-ready cases before publishing dataset-level coverage.",
        "Dataset-level readiness could not be fully resolved from case-level interpretation.",
    )


def build_pocket_benchmark_dataset_interpretation(case_interpretation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate case-level benchmark interpretations into dataset-level claim readiness."""

    if case_interpretation_df is None or getattr(case_interpretation_df, "empty", True) or "top_n" not in case_interpretation_df.columns:
        return _empty_dataset_interpretation_df()

    working = case_interpretation_df.copy()
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_dataset_interpretation_df()

    if "coverage_ratio" not in working.columns:
        working["coverage_ratio"] = 0.0
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)

    if "claim_ready" not in working.columns:
        working["claim_ready"] = False
    working["claim_ready"] = working["claim_ready"].map(_claim_ready_bool)

    if "claim_status" not in working.columns:
        working["claim_status"] = ""
    working["claim_status"] = working.apply(
        lambda row: _normalized_claim_status(row.get("claim_status"))
        or ("claim-ready" if bool(row.get("claim_ready")) else "readiness-unknown"),
        axis=1,
    )
    known_statuses = {"claim-ready", "blocked", "review-needed", "readiness-unknown"}
    working.loc[~working["claim_status"].isin(known_statuses), "claim_status"] = "readiness-unknown"
    working.loc[working["claim_ready"] & working["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"

    if "benchmark_id" not in working.columns:
        working["benchmark_id"] = [f"case-{index}" for index in working.index]
    else:
        working["benchmark_id"] = [
            _safe_text(value).strip() or f"case-{index}"
            for index, value in zip(working.index, working["benchmark_id"])
        ]

    rows: list[dict[str, object]] = []
    for top_n, group in working.groupby("top_n", sort=True, dropna=False):
        top_n_int = int(top_n)
        case_group = group.drop_duplicates(subset=["benchmark_id"], keep="first").copy()
        case_count = int(len(case_group))
        claim_ready_count = int(case_group["claim_status"].eq("claim-ready").sum())
        blocked_count = int(case_group["claim_status"].eq("blocked").sum())
        review_count = int(case_group["claim_status"].eq("review-needed").sum())
        unknown_count = int(case_group["claim_status"].eq("readiness-unknown").sum())
        claim_ready_rows = case_group[case_group["claim_status"].eq("claim-ready")]
        dataset_status, label, action, warning = _dataset_interpretation_status(
            case_count,
            claim_ready_count,
            blocked_count,
            review_count,
            unknown_count,
            top_n_int,
        )

        rows.append(
            {
                "top_n": top_n_int,
                "case_count": case_count,
                "claim_ready_case_count": claim_ready_count,
                "blocked_case_count": blocked_count,
                "review_case_count": review_count,
                "unknown_case_count": unknown_count,
                "mean_claim_ready_coverage": round(float(claim_ready_rows["coverage_ratio"].mean()), 3) if not claim_ready_rows.empty else 0.0,
                "mean_all_case_coverage": round(float(case_group["coverage_ratio"].mean()), 3) if case_count else 0.0,
                "claim_ready_rate": round(float(claim_ready_count) / float(case_count), 3) if case_count else 0.0,
                "blocked_case_rate": round(float(blocked_count) / float(case_count), 3) if case_count else 0.0,
                "review_case_rate": round(float(review_count) / float(case_count), 3) if case_count else 0.0,
                "dataset_claim_status": dataset_status,
                "interpretation_label": label,
                "recommended_action": action,
                "interpretation_warning": warning,
            }
        )

    if not rows:
        return _empty_dataset_interpretation_df()
    return pd.DataFrame(rows, columns=BENCHMARK_DATASET_INTERPRETATION_COLUMNS)


def _dataset_interpretation_queue_issue(claim_status: str) -> tuple[str, str, str]:
    if claim_status == "blocked":
        return "P0", "blocker", "blocked-case"
    if claim_status == "review-needed":
        return "P2", "review", "review-needed-case"
    if claim_status == "readiness-unknown":
        return "P2", "review", "readiness-unknown-case"
    return "", "", ""


def build_pocket_benchmark_dataset_interpretation_queue(case_interpretation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """List non-claim-ready benchmark cases that block or weaken dataset-level interpretation."""

    if case_interpretation_df is None or getattr(case_interpretation_df, "empty", True) or "top_n" not in case_interpretation_df.columns:
        return _empty_dataset_interpretation_queue_df()

    working = case_interpretation_df.copy()
    for column in BENCHMARK_CASE_INTERPRETATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_dataset_interpretation_queue_df()

    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)
    working["claim_status"] = working["claim_status"].map(_normalized_claim_status)
    working.loc[working["claim_status"].eq(""), "claim_status"] = "readiness-unknown"
    if "claim_ready" in working.columns:
        working["claim_ready"] = working["claim_ready"].map(_claim_ready_bool)
        working.loc[working["claim_ready"] & working["claim_status"].eq("readiness-unknown"), "claim_status"] = "claim-ready"

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        claim_status = _safe_text(row.get("claim_status"))
        priority, action_status, issue_type = _dataset_interpretation_queue_issue(claim_status)
        if not priority:
            continue
        rows.append(
            {
                "priority": priority,
                "action_status": action_status,
                "top_n": int(row.get("top_n") or 0),
                "benchmark_id": _safe_text(row.get("benchmark_id")) or "current",
                "claim_status": claim_status,
                "coverage_ratio": round(float(row.get("coverage_ratio") or 0.0), 3),
                "best_rank": int(row.get("best_rank") or 0),
                "best_pocket_id": _safe_text(row.get("best_pocket_id")),
                "benchmark_status": _safe_text(row.get("benchmark_status")),
                "readiness_status": _safe_text(row.get("readiness_status")),
                "issue_type": issue_type,
                "suggested_action": _safe_text(row.get("recommended_action")),
                "interpretation_warning": _safe_text(row.get("interpretation_warning")),
            }
        )

    if not rows:
        return _empty_dataset_interpretation_queue_df()

    frame = pd.DataFrame(rows)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["top_n", "_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"]).reset_index(drop=True)
    frame["action_id"] = [f"BDSI-{index + 1:03d}" for index in range(len(frame))]
    return frame[BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS]


def build_pocket_benchmark_dataset_interpretation_checklist_markdown(
    dataset_interpretation_queue_df: Optional[pd.DataFrame],
    *,
    title: str = "Benchmark dataset interpretation checklist",
    max_actions: int = 80,
) -> str:
    """Render dataset-level benchmark interpretation blockers as a reviewer checklist."""

    if dataset_interpretation_queue_df is None or getattr(dataset_interpretation_queue_df, "empty", True):
        return ""

    working = dataset_interpretation_queue_df.copy()
    for column in BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce").fillna(0).astype(int)
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    working["best_rank"] = pd.to_numeric(working["best_rank"], errors="coerce").fillna(0).astype(int)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    working["_priority_rank"] = working["priority"].map(priority_rank).fillna(99)
    working = working.sort_values(["top_n", "_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"])

    summary = (
        working.groupby(["priority", "action_status", "issue_type"], sort=False, dropna=False)
        .agg(action_count=("action_id", "count"), affected_case_count=("benchmark_id", "nunique"))
        .reset_index()
    )
    summary["_priority_rank"] = summary["priority"].map(priority_rank).fillna(99)
    summary = summary.sort_values(["_priority_rank", "issue_type"]).drop(columns=["_priority_rank"])

    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_dataset_interpretation_queue.csv`.",
        "",
        "Use this checklist before reporting dataset-level benchmark coverage as a precision claim.",
        "",
        "## Summary",
        "",
        "| Priority | Status | Issue | Actions | Cases |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {priority} | {status} | {issue} | {actions} | {cases} |".format(
                priority=_safe_text(row.get("priority")) or "-",
                status=_safe_text(row.get("action_status")) or "-",
                issue=_safe_text(row.get("issue_type")) or "-",
                actions=int(row.get("action_count") or 0),
                cases=int(row.get("affected_case_count") or 0),
            )
        )

    lines.extend(["", "## Actions", ""])
    for _, row in working.head(max_actions).iterrows():
        lines.append(
            "- [ ] `{priority}` Top-{top_n} `{issue}` case `{case}` coverage `{coverage}` best rank `{best_rank}` pocket `{pocket}`: {action}".format(
                priority=_safe_text(row.get("priority")) or "-",
                top_n=int(row.get("top_n") or 0),
                issue=_safe_text(row.get("issue_type")) or "-",
                case=_safe_text(row.get("benchmark_id")) or "current",
                coverage=round(float(row.get("coverage_ratio") or 0.0), 3),
                best_rank=int(row.get("best_rank") or 0),
                pocket=_safe_text(row.get("best_pocket_id")) or "-",
                action=_safe_text(row.get("suggested_action")) or "Review this benchmark case before reporting dataset-level coverage.",
            )
        )

    remaining = int(len(working) - max_actions)
    if remaining > 0:
        lines.append(f"- [ ] Review {remaining} additional queued actions in the CSV export.")
    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_dataset_interpretation_report_markdown(
    dataset_interpretation_df: Optional[pd.DataFrame],
    dataset_interpretation_queue_df: Optional[pd.DataFrame] = None,
    *,
    checklist_available: bool = False,
    title: str = "Benchmark dataset interpretation report",
    max_queue_actions: int = 20,
) -> str:
    """Render a compact report for dataset-level benchmark claim readiness."""

    has_interpretation = dataset_interpretation_df is not None and not getattr(dataset_interpretation_df, "empty", True)
    has_queue = dataset_interpretation_queue_df is not None and not getattr(dataset_interpretation_queue_df, "empty", True)
    if not has_interpretation and not has_queue:
        return ""

    interpretation = dataset_interpretation_df.copy() if has_interpretation else _empty_dataset_interpretation_df()
    for column in BENCHMARK_DATASET_INTERPRETATION_COLUMNS:
        if column not in interpretation.columns:
            interpretation[column] = ""
    if not interpretation.empty:
        interpretation["top_n"] = pd.to_numeric(interpretation["top_n"], errors="coerce").fillna(0).astype(int)
        for column in (
            "case_count",
            "claim_ready_case_count",
            "blocked_case_count",
            "review_case_count",
            "unknown_case_count",
        ):
            interpretation[column] = pd.to_numeric(interpretation[column], errors="coerce").fillna(0).astype(int)
        for column in ("mean_claim_ready_coverage", "mean_all_case_coverage", "claim_ready_rate"):
            interpretation[column] = pd.to_numeric(interpretation[column], errors="coerce").fillna(0.0)
        interpretation = interpretation.sort_values("top_n")

    queue = dataset_interpretation_queue_df.copy() if has_queue else _empty_dataset_interpretation_queue_df()
    for column in BENCHMARK_DATASET_INTERPRETATION_QUEUE_COLUMNS:
        if column not in queue.columns:
            queue[column] = ""
    if not queue.empty:
        queue["top_n"] = pd.to_numeric(queue["top_n"], errors="coerce").fillna(0).astype(int)
        queue["coverage_ratio"] = pd.to_numeric(queue["coverage_ratio"], errors="coerce").fillna(0.0)
        queue["best_rank"] = pd.to_numeric(queue["best_rank"], errors="coerce").fillna(0).astype(int)
        priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        queue["_priority_rank"] = queue["priority"].map(priority_rank).fillna(99)
        queue = queue.sort_values(["top_n", "_priority_rank", "benchmark_id", "issue_type"]).drop(columns=["_priority_rank"])

    blocked_rows = int(interpretation["dataset_claim_status"].astype(str).eq("blocked").sum()) if not interpretation.empty else 0
    review_rows = int(interpretation["dataset_claim_status"].astype(str).eq("review-needed").sum()) if not interpretation.empty else 0
    claim_ready_rows = int(interpretation["dataset_claim_status"].astype(str).eq("claim-ready").sum()) if not interpretation.empty else 0
    report_status = "blocked" if blocked_rows else "review-needed" if review_rows else "claim-ready" if claim_ready_rows else "no-dataset-interpretation"
    next_action = (
        "Fix P0 blocker cases before reporting dataset-level coverage."
        if blocked_rows
        else "Complete reviewer sign-off before publishing dataset-level coverage."
        if review_rows
        else "Report dataset coverage with case counts and Top-N status."
        if claim_ready_rows
        else "Generate dataset interpretation before reporting benchmark accuracy."
    )

    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_dataset_interpretation.csv` and `pocket_benchmark_dataset_interpretation_queue.csv`.",
        "",
        "## Gate",
        "",
        f"- Dataset claim status: `{report_status}`.",
        f"- Dataset interpretation rows: {int(len(interpretation))}.",
        f"- Queued case actions: {int(len(queue))}.",
        f"- Checklist: {'available' if checklist_available else 'not available'}.",
        f"- Recommended action: {next_action}",
        "",
        "## Top-N Interpretation",
        "",
    ]
    if interpretation.empty:
        lines.append("No dataset interpretation rows are available.")
    else:
        lines.append("| Top-N | Status | Cases | Claim-ready | Blocked | Review | Unknown | Mean claim-ready coverage | Mean all-case coverage |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in interpretation.iterrows():
            lines.append(
                "| {top_n} | {status} | {cases} | {ready} | {blocked} | {review} | {unknown} | {ready_cov:.3f} | {all_cov:.3f} |".format(
                    top_n=int(row.get("top_n") or 0),
                    status=_safe_text(row.get("dataset_claim_status")) or "-",
                    cases=int(row.get("case_count") or 0),
                    ready=int(row.get("claim_ready_case_count") or 0),
                    blocked=int(row.get("blocked_case_count") or 0),
                    review=int(row.get("review_case_count") or 0),
                    unknown=int(row.get("unknown_case_count") or 0),
                    ready_cov=float(row.get("mean_claim_ready_coverage") or 0.0),
                    all_cov=float(row.get("mean_all_case_coverage") or 0.0),
                )
            )

    lines.extend(["", "## Queued Actions", ""])
    if queue.empty:
        lines.append("No non-claim-ready benchmark cases are queued.")
    else:
        lines.append("| Priority | Top-N | Case | Issue | Coverage | Best rank | Action |")
        lines.append("| --- | ---: | --- | --- | ---: | ---: | --- |")
        for _, row in queue.head(max_queue_actions).iterrows():
            lines.append(
                "| {priority} | {top_n} | {case} | {issue} | {coverage:.3f} | {best_rank} | {action} |".format(
                    priority=_safe_text(row.get("priority")) or "-",
                    top_n=int(row.get("top_n") or 0),
                    case=_safe_text(row.get("benchmark_id")) or "current",
                    issue=_safe_text(row.get("issue_type")) or "-",
                    coverage=float(row.get("coverage_ratio") or 0.0),
                    best_rank=int(row.get("best_rank") or 0),
                    action=_safe_text(row.get("suggested_action")) or "Review this benchmark case.",
                )
            )
        remaining = int(len(queue) - max_queue_actions)
        if remaining > 0:
            lines.append(f"\n{remaining} additional queued actions are available in the CSV export.")

    return "\n".join(lines).strip() + "\n"


def build_pocket_benchmark_case_summary(
    reference_df: Optional[pd.DataFrame],
    pocket_df: Optional[pd.DataFrame],
    pocket_summary_df: Optional[pd.DataFrame] = None,
    *,
    top_ns: Sequence[int] = (1, 3),
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Build Top-N coverage summaries per benchmark case.

    A curated dataset can contain multiple `benchmark_id`/`case_id` values.
    Keeping case rows separate prevents enzymes with many curated residues from
    dominating the reported accuracy.
    """

    references = _reference_rows(reference_df)
    if references.empty:
        return _empty_case_summary_df()

    working = references.copy()
    fallback_id = str(default_benchmark_id or "current").strip() or "current"
    working["benchmark_id"] = working["benchmark_id"].astype(str).str.strip().replace("", fallback_id)

    rows: list[dict[str, object]] = []
    for benchmark_id, case_reference_df in working.groupby("benchmark_id", sort=True, dropna=False):
        case_pocket_df = _filter_rows_for_benchmark_id(pocket_df, str(benchmark_id or fallback_id))
        case_pocket_summary_df = _filter_rows_for_benchmark_id(pocket_summary_df, str(benchmark_id or fallback_id))
        summary = build_pocket_benchmark_summary(
            case_reference_df,
            case_pocket_df,
            case_pocket_summary_df,
            top_ns=top_ns,
        )
        for row in summary.to_dict(orient="records"):
            rows.append({"benchmark_id": str(benchmark_id or fallback_id), **row})

    if not rows:
        return _empty_case_summary_df()
    return pd.DataFrame(rows, columns=BENCHMARK_CASE_SUMMARY_COLUMNS)


def _dataset_summary_status(case_count: int, any_hit_count: int, all_hit_count: int) -> tuple[str, str]:
    if case_count <= 0:
        return "no-cases", "No benchmark cases are available."
    if all_hit_count == case_count:
        return "all-cases-complete-hit", "All benchmark cases have complete catalytic residue coverage."
    if any_hit_count == case_count:
        return "all-cases-any-hit", "Every benchmark case has at least one catalytic residue covered."
    if any_hit_count > 0:
        return "mixed-hit", "Some benchmark cases are hit while others miss; inspect case_summary before claiming dataset accuracy."
    return "all-cases-miss", "No benchmark case has catalytic residue coverage."


def build_pocket_benchmark_dataset_summary(case_summary_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-case benchmark coverage into dataset-level metrics."""

    if case_summary_df is None or getattr(case_summary_df, "empty", True) or "top_n" not in case_summary_df.columns:
        return _empty_dataset_summary_df()

    working = case_summary_df.copy()
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_dataset_summary_df()

    for column in ["reference_residue_count", "matched_reference_count", "best_rank"]:
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    if "coverage_ratio" not in working.columns:
        working["coverage_ratio"] = 0.0
    working["coverage_ratio"] = pd.to_numeric(working["coverage_ratio"], errors="coerce").fillna(0.0)
    if "any_hit" not in working.columns:
        working["any_hit"] = False
    if "all_hit" not in working.columns:
        working["all_hit"] = False
    working["any_hit"] = working["any_hit"].astype(bool)
    working["all_hit"] = working["all_hit"].astype(bool)

    rows: list[dict[str, object]] = []
    for top_n, group in working.groupby("top_n", sort=True, dropna=False):
        case_count = int(len(group))
        any_hit_count = int(group["any_hit"].sum())
        all_hit_count = int(group["all_hit"].sum())
        best_ranks = group.loc[group["best_rank"] > 0, "best_rank"]
        status, warning = _dataset_summary_status(case_count, any_hit_count, all_hit_count)
        rows.append(
            {
                "top_n": int(top_n),
                "case_count": case_count,
                "reference_residue_count": int(group["reference_residue_count"].sum()),
                "matched_reference_count": int(group["matched_reference_count"].sum()),
                "mean_coverage_ratio": round(float(group["coverage_ratio"].mean()), 3) if case_count else 0.0,
                "median_coverage_ratio": round(float(group["coverage_ratio"].median()), 3) if case_count else 0.0,
                "min_coverage_ratio": round(float(group["coverage_ratio"].min()), 3) if case_count else 0.0,
                "max_coverage_ratio": round(float(group["coverage_ratio"].max()), 3) if case_count else 0.0,
                "any_hit_case_count": any_hit_count,
                "all_hit_case_count": all_hit_count,
                "miss_case_count": int(case_count - any_hit_count),
                "any_hit_rate": round(float(any_hit_count) / float(case_count), 3) if case_count else 0.0,
                "all_hit_rate": round(float(all_hit_count) / float(case_count), 3) if case_count else 0.0,
                "mean_best_rank": round(float(best_ranks.mean()), 3) if not best_ranks.empty else 0.0,
                "benchmark_status": status,
                "benchmark_warning": warning,
            }
        )

    if not rows:
        return _empty_dataset_summary_df()
    return pd.DataFrame(rows, columns=BENCHMARK_DATASET_SUMMARY_COLUMNS)


def build_pocket_benchmark_variant_comparison(
    reference_df: Optional[pd.DataFrame],
    variants: Sequence[tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    *,
    reference_variant_label: str = "current",
    top_ns: Sequence[int] = (1, 3),
) -> pd.DataFrame:
    """Compare catalytic coverage across pocket-ranking variants.

    `variants` contains `(label, pocket_df, pocket_summary_df)` tuples. The
    reference variant is usually the active/current run; ablated variants then
    show coverage loss when literature, evidence-route or conservation support
    is removed.
    """

    references = _reference_rows(reference_df)
    if references.empty or not variants:
        return _empty_variant_comparison_df()

    summaries: dict[str, pd.DataFrame] = {}
    labels: list[str] = []
    for label, pocket_df, pocket_summary_df in variants:
        cleaned_label = str(label or "").strip()
        if not cleaned_label or cleaned_label in summaries:
            continue
        summary = build_pocket_benchmark_summary(
            references,
            pocket_df,
            pocket_summary_df,
            top_ns=top_ns,
        )
        if summary.empty:
            continue
        summaries[cleaned_label] = summary
        labels.append(cleaned_label)

    if not summaries:
        return _empty_variant_comparison_df()

    reference_label = str(reference_variant_label or "").strip()
    if reference_label not in summaries:
        reference_label = labels[0]
    reference_summary = summaries[reference_label].copy()
    reference_by_top_n = {
        int(row.top_n): row
        for row in reference_summary.itertuples(index=False)
        if pd.notna(getattr(row, "top_n", None))
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        summary = summaries[label]
        for row in summary.itertuples(index=False):
            top_n = int(getattr(row, "top_n"))
            reference_row = reference_by_top_n.get(top_n)
            reference_coverage = float(getattr(reference_row, "coverage_ratio", 0.0)) if reference_row is not None else 0.0
            coverage = float(getattr(row, "coverage_ratio", 0.0))
            reference_best_rank = int(getattr(reference_row, "best_rank", 0) or 0) if reference_row is not None else 0
            best_rank = int(getattr(row, "best_rank", 0) or 0)
            if best_rank <= 0 or reference_best_rank <= 0:
                rank_delta = 0
            else:
                rank_delta = int(best_rank - reference_best_rank)
            rows.append(
                {
                    "variant_label": label,
                    "reference_variant_label": reference_label,
                    "top_n": top_n,
                    "reference_residue_count": int(getattr(row, "reference_residue_count", 0) or 0),
                    "matched_reference_count": int(getattr(row, "matched_reference_count", 0) or 0),
                    "coverage_ratio": coverage,
                    "reference_coverage_ratio": reference_coverage,
                    "coverage_delta_vs_reference": round(float(coverage - reference_coverage), 3),
                    "coverage_loss_vs_reference": round(float(reference_coverage - coverage), 3),
                    "best_rank": best_rank,
                    "reference_best_rank": reference_best_rank,
                    "best_rank_delta_vs_reference": rank_delta,
                    "best_pocket_id": str(getattr(row, "best_pocket_id", "") or ""),
                    "top_pocket_id": str(getattr(row, "top_pocket_id", "") or ""),
                    "top_pocket_hit": bool(getattr(row, "top_pocket_hit", False)),
                    "matched_residues": str(getattr(row, "matched_residues", "") or ""),
                    "missed_residues": str(getattr(row, "missed_residues", "") or ""),
                    "benchmark_status": str(getattr(row, "benchmark_status", "") or ""),
                    "benchmark_warning": str(getattr(row, "benchmark_warning", "") or ""),
                }
            )

    if not rows:
        return _empty_variant_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_COMPARISON_COLUMNS)


def build_pocket_benchmark_variant_case_comparison(
    reference_df: Optional[pd.DataFrame],
    variants: Sequence[tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    *,
    reference_variant_label: str = "current",
    top_ns: Sequence[int] = (1, 3),
    default_benchmark_id: str = "current",
) -> pd.DataFrame:
    """Compare variant coverage per benchmark case."""

    references = _reference_rows(reference_df)
    if references.empty or not variants:
        return _empty_variant_case_comparison_df()

    case_summaries: dict[str, pd.DataFrame] = {}
    labels: list[str] = []
    for label, pocket_df, pocket_summary_df in variants:
        cleaned_label = str(label or "").strip()
        if not cleaned_label or cleaned_label in case_summaries:
            continue
        summary = build_pocket_benchmark_case_summary(
            references,
            pocket_df,
            pocket_summary_df,
            top_ns=top_ns,
            default_benchmark_id=default_benchmark_id,
        )
        if summary.empty:
            continue
        case_summaries[cleaned_label] = summary
        labels.append(cleaned_label)

    if not case_summaries:
        return _empty_variant_case_comparison_df()

    reference_label = str(reference_variant_label or "").strip()
    if reference_label not in case_summaries:
        reference_label = labels[0]
    reference_summary = case_summaries[reference_label].copy()
    reference_by_case_top_n = {
        (str(row.benchmark_id), int(row.top_n)): row
        for row in reference_summary.itertuples(index=False)
        if pd.notna(getattr(row, "top_n", None))
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        summary = case_summaries[label]
        for row in summary.itertuples(index=False):
            benchmark_id = str(getattr(row, "benchmark_id", "") or "")
            top_n = int(getattr(row, "top_n"))
            reference_row = reference_by_case_top_n.get((benchmark_id, top_n))
            reference_coverage = float(getattr(reference_row, "coverage_ratio", 0.0)) if reference_row is not None else 0.0
            coverage = float(getattr(row, "coverage_ratio", 0.0))
            reference_best_rank = int(getattr(reference_row, "best_rank", 0) or 0) if reference_row is not None else 0
            best_rank = int(getattr(row, "best_rank", 0) or 0)
            rank_delta = int(best_rank - reference_best_rank) if best_rank > 0 and reference_best_rank > 0 else 0
            rows.append(
                {
                    "variant_label": label,
                    "reference_variant_label": reference_label,
                    "benchmark_id": benchmark_id,
                    "top_n": top_n,
                    "reference_residue_count": int(getattr(row, "reference_residue_count", 0) or 0),
                    "matched_reference_count": int(getattr(row, "matched_reference_count", 0) or 0),
                    "coverage_ratio": coverage,
                    "reference_coverage_ratio": reference_coverage,
                    "coverage_delta_vs_reference": round(float(coverage - reference_coverage), 3),
                    "coverage_loss_vs_reference": round(float(reference_coverage - coverage), 3),
                    "best_rank": best_rank,
                    "reference_best_rank": reference_best_rank,
                    "best_rank_delta_vs_reference": rank_delta,
                    "best_pocket_id": str(getattr(row, "best_pocket_id", "") or ""),
                    "top_pocket_id": str(getattr(row, "top_pocket_id", "") or ""),
                    "top_pocket_hit": bool(getattr(row, "top_pocket_hit", False)),
                    "matched_residues": str(getattr(row, "matched_residues", "") or ""),
                    "missed_residues": str(getattr(row, "missed_residues", "") or ""),
                    "benchmark_status": str(getattr(row, "benchmark_status", "") or ""),
                    "benchmark_warning": str(getattr(row, "benchmark_warning", "") or ""),
                }
            )

    if not rows:
        return _empty_variant_case_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_CASE_COMPARISON_COLUMNS)


def build_pocket_benchmark_variant_dataset_comparison(variant_case_comparison_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate case-level variant comparison into dataset-level deltas."""

    if (
        variant_case_comparison_df is None
        or getattr(variant_case_comparison_df, "empty", True)
        or "variant_label" not in variant_case_comparison_df.columns
        or "top_n" not in variant_case_comparison_df.columns
    ):
        return _empty_variant_dataset_comparison_df()

    working = variant_case_comparison_df.copy()
    working["top_n"] = pd.to_numeric(working["top_n"], errors="coerce")
    working = working[working["top_n"].notna()].copy()
    if working.empty:
        return _empty_variant_dataset_comparison_df()

    for column in [
        "coverage_ratio",
        "reference_coverage_ratio",
        "coverage_delta_vs_reference",
        "coverage_loss_vs_reference",
        "best_rank",
        "reference_best_rank",
    ]:
        if column not in working.columns:
            working[column] = 0.0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    rows: list[dict[str, object]] = []
    group_columns = ["variant_label", "reference_variant_label", "top_n"]
    for (variant_label, reference_label, top_n), group in working.groupby(group_columns, sort=True, dropna=False):
        case_count = int(len(group))
        any_hit_rate = round(float((group["coverage_ratio"] > 0.0).sum()) / float(case_count), 3) if case_count else 0.0
        reference_any_hit_rate = round(float((group["reference_coverage_ratio"] > 0.0).sum()) / float(case_count), 3) if case_count else 0.0
        all_hit_rate = round(float((group["coverage_ratio"] >= 1.0).sum()) / float(case_count), 3) if case_count else 0.0
        reference_all_hit_rate = round(float((group["reference_coverage_ratio"] >= 1.0).sum()) / float(case_count), 3) if case_count else 0.0
        best_ranks = group.loc[group["best_rank"] > 0, "best_rank"]
        reference_best_ranks = group.loc[group["reference_best_rank"] > 0, "reference_best_rank"]
        case_loss_count = int((group["coverage_loss_vs_reference"] > 0.0).sum())
        case_gain_count = int((group["coverage_delta_vs_reference"] > 0.0).sum())
        case_unchanged_count = int((group["coverage_delta_vs_reference"].abs() <= 0.0005).sum())
        mean_coverage = round(float(group["coverage_ratio"].mean()), 3) if case_count else 0.0
        reference_mean_coverage = round(float(group["reference_coverage_ratio"].mean()), 3) if case_count else 0.0
        mean_best_rank = round(float(best_ranks.mean()), 3) if not best_ranks.empty else 0.0
        reference_mean_best_rank = round(float(reference_best_ranks.mean()), 3) if not reference_best_ranks.empty else 0.0
        status, warning = _dataset_summary_status(
            case_count,
            int((group["coverage_ratio"] > 0.0).sum()),
            int((group["coverage_ratio"] >= 1.0).sum()),
        )
        if case_loss_count > 0:
            warning = f"{case_loss_count} benchmark cases lost catalytic coverage versus {reference_label}."
        rows.append(
            {
                "variant_label": str(variant_label or ""),
                "reference_variant_label": str(reference_label or ""),
                "top_n": int(top_n),
                "case_count": case_count,
                "mean_coverage_ratio": mean_coverage,
                "reference_mean_coverage_ratio": reference_mean_coverage,
                "mean_coverage_delta_vs_reference": round(float(mean_coverage - reference_mean_coverage), 3),
                "mean_coverage_loss_vs_reference": round(float(reference_mean_coverage - mean_coverage), 3),
                "any_hit_rate": any_hit_rate,
                "reference_any_hit_rate": reference_any_hit_rate,
                "any_hit_rate_delta_vs_reference": round(float(any_hit_rate - reference_any_hit_rate), 3),
                "all_hit_rate": all_hit_rate,
                "reference_all_hit_rate": reference_all_hit_rate,
                "all_hit_rate_delta_vs_reference": round(float(all_hit_rate - reference_all_hit_rate), 3),
                "mean_best_rank": mean_best_rank,
                "reference_mean_best_rank": reference_mean_best_rank,
                "mean_best_rank_delta_vs_reference": round(float(mean_best_rank - reference_mean_best_rank), 3)
                if mean_best_rank > 0 and reference_mean_best_rank > 0
                else 0.0,
                "case_loss_count": case_loss_count,
                "case_gain_count": case_gain_count,
                "case_unchanged_count": case_unchanged_count,
                "benchmark_status": status,
                "benchmark_warning": warning,
            }
        )

    if not rows:
        return _empty_variant_dataset_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_DATASET_COMPARISON_COLUMNS)


def _detail_key(row: pd.Series) -> tuple[str, str, int, str]:
    return (
        str(row.get("benchmark_id") or "").strip(),
        str(row.get("chain") or "").strip(),
        int(row.get("resid") or 0),
        str(row.get("reference_type") or "").strip(),
    )


def _match_delta(variant_matched: bool, reference_matched: bool) -> str:
    if variant_matched and reference_matched:
        return "unchanged-hit"
    if not variant_matched and not reference_matched:
        return "unchanged-miss"
    if reference_matched and not variant_matched:
        return "lost"
    return "gained"


def build_pocket_benchmark_variant_detail_comparison(
    reference_df: Optional[pd.DataFrame],
    variants: Sequence[tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    *,
    reference_variant_label: str = "current",
    top_thresholds: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    """Compare exact catalytic residue matches across ranking variants."""

    references = _reference_rows(reference_df)
    if references.empty or not variants:
        return _empty_variant_detail_comparison_df()

    details_by_label: dict[str, pd.DataFrame] = {}
    labels: list[str] = []
    for label, pocket_df, pocket_summary_df in variants:
        cleaned_label = str(label or "").strip()
        if not cleaned_label or cleaned_label in details_by_label:
            continue
        details = build_pocket_benchmark_details(
            references,
            pocket_df,
            pocket_summary_df,
            top_thresholds=top_thresholds,
        )
        if details.empty:
            continue
        details_by_label[cleaned_label] = details
        labels.append(cleaned_label)

    if not details_by_label:
        return _empty_variant_detail_comparison_df()

    reference_label = str(reference_variant_label or "").strip()
    if reference_label not in details_by_label:
        reference_label = labels[0]
    reference_details = details_by_label[reference_label]
    reference_by_key = {
        _detail_key(row): row
        for _, row in reference_details.iterrows()
    }

    rows: list[dict[str, object]] = []
    for label in labels:
        details = details_by_label[label]
        for _, row in details.iterrows():
            key = _detail_key(row)
            reference_row = reference_by_key.get(key)
            variant_matched = bool(row.get("matched"))
            reference_matched = bool(reference_row.get("matched")) if reference_row is not None else False
            variant_rank = int(row.get("matched_rank") or 0)
            reference_rank = int(reference_row.get("matched_rank") or 0) if reference_row is not None else 0
            rank_delta = int(variant_rank - reference_rank) if variant_rank > 0 and reference_rank > 0 else 0
            delta = _match_delta(variant_matched, reference_matched)
            warning = str(row.get("benchmark_warning") or "")
            if delta == "lost":
                warning = "reference-residue-lost-vs-current"
            elif delta == "gained":
                warning = "reference-residue-gained-vs-current"
            rows.append(
                {
                    "variant_label": label,
                    "reference_variant_label": reference_label,
                    "benchmark_id": str(row.get("benchmark_id") or ""),
                    "chain": str(row.get("chain") or ""),
                    "resid": int(row.get("resid") or 0),
                    "resname": str(row.get("resname") or ""),
                    "residue_label": str(row.get("residue_label") or ""),
                    "reference_type": str(row.get("reference_type") or ""),
                    "reference_source": str(row.get("reference_source") or ""),
                    "expected_pocket_id": str(row.get("expected_pocket_id") or ""),
                    "variant_matched": variant_matched,
                    "reference_matched": reference_matched,
                    "match_delta": delta,
                    "variant_matched_rank": variant_rank,
                    "reference_matched_rank": reference_rank,
                    "rank_delta_vs_reference": rank_delta,
                    "variant_matched_pocket_id": str(row.get("matched_pocket_id") or ""),
                    "reference_matched_pocket_id": str(reference_row.get("matched_pocket_id") or "") if reference_row is not None else "",
                    "variant_matched_pocket_ids": str(row.get("matched_pocket_ids") or ""),
                    "reference_matched_pocket_ids": str(reference_row.get("matched_pocket_ids") or "") if reference_row is not None else "",
                    "variant_expected_pocket_matched": bool(row.get("expected_pocket_matched")),
                    "reference_expected_pocket_matched": bool(reference_row.get("expected_pocket_matched")) if reference_row is not None else False,
                    "benchmark_warning": warning,
                }
            )

    if not rows:
        return _empty_variant_detail_comparison_df()
    return pd.DataFrame(rows, columns=BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS)


def _variant_issue_action(variant_label: str, issue_type: str) -> str:
    variant = str(variant_label or "").strip().lower()
    if issue_type == "current-missed-residue":
        return "Check curated residue numbering, chain mapping and pocket detection thresholds; current ranking does not cover this catalytic residue."
    if "no-literature" in variant:
        return "Literature evidence appears to support this residue; verify citation/snippet quality, numbering assumptions and manual-review flags before weakening the literature route."
    if "no-evidence-route" in variant:
        return "External-evidence route appears to preserve this residue; review route thresholds, mapping quality and neighborhood radius before disabling the route."
    if "no-p2rank" in variant:
        return "P2Rank appears to preserve this residue; verify the local P2Rank install/profile and keep a fallback when P2Rank is unavailable."
    if "no-conservation" in variant:
        return "Conservation rerank appears to preserve this residue; review conservation score direction and keep it rerank-only unless benchmark loss persists."
    return "Inspect this ablation before accepting the rank change; removing the evidence path changed catalytic residue coverage."


def build_pocket_benchmark_variant_remediation_queue(variant_detail_comparison_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Turn residue-level variant errors into a review/remediation queue."""

    if (
        variant_detail_comparison_df is None
        or getattr(variant_detail_comparison_df, "empty", True)
        or "match_delta" not in variant_detail_comparison_df.columns
    ):
        return _empty_variant_remediation_df()

    working = variant_detail_comparison_df.copy()
    for column in BENCHMARK_VARIANT_DETAIL_COMPARISON_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["match_delta"] = working["match_delta"].astype(str).str.strip()
    working["variant_label"] = working["variant_label"].astype(str).str.strip()
    working["reference_variant_label"] = working["reference_variant_label"].astype(str).str.strip()
    reference_label = str(working["reference_variant_label"].replace("", pd.NA).dropna().iloc[0]) if working["reference_variant_label"].replace("", pd.NA).dropna().shape[0] else "current"

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        variant_label = str(row.get("variant_label") or "").strip()
        delta = str(row.get("match_delta") or "").strip()
        issue_type = ""
        priority = ""
        if delta == "lost" and variant_label != reference_label:
            issue_type = "ablation-lost-residue"
            priority = "P0"
        elif delta == "unchanged-miss" and variant_label == reference_label:
            issue_type = "current-missed-residue"
            priority = "P1"
        else:
            continue

        benchmark_id = str(row.get("benchmark_id") or "").strip() or "current"
        residue_label = str(row.get("residue_label") or "").strip() or f"{row.get('chain', '')}{row.get('resid', '')}"
        action_id = f"{issue_type}:{variant_label}:{benchmark_id}:{residue_label}".replace(" ", "-")
        rows.append(
            {
                "action_id": action_id,
                "priority": priority,
                "issue_type": issue_type,
                "variant_label": variant_label,
                "reference_variant_label": reference_label,
                "benchmark_id": benchmark_id,
                "residue_label": residue_label,
                "chain": str(row.get("chain") or ""),
                "resid": int(row.get("resid") or 0),
                "resname": str(row.get("resname") or ""),
                "match_delta": delta,
                "reference_matched_pocket_id": str(row.get("reference_matched_pocket_id") or ""),
                "variant_matched_pocket_id": str(row.get("variant_matched_pocket_id") or ""),
                "reference_matched_rank": int(row.get("reference_matched_rank") or 0),
                "variant_matched_rank": int(row.get("variant_matched_rank") or 0),
                "expected_pocket_id": str(row.get("expected_pocket_id") or ""),
                "suggested_action": _variant_issue_action(variant_label, issue_type),
                "benchmark_warning": str(row.get("benchmark_warning") or ""),
            }
        )

    if not rows:
        return _empty_variant_remediation_df()
    frame = pd.DataFrame(rows, columns=BENCHMARK_VARIANT_REMEDIATION_COLUMNS)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["_priority_rank", "variant_label", "benchmark_id", "resid", "residue_label"]).drop(columns=["_priority_rank"])
    return frame.reset_index(drop=True)


def build_pocket_benchmark_variant_remediation_summary(remediation_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Summarize remediation actions by priority, issue type and variant."""

    if remediation_df is None or getattr(remediation_df, "empty", True):
        return _empty_variant_remediation_summary_df()

    working = remediation_df.copy()
    for column in BENCHMARK_VARIANT_REMEDIATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["priority"] = working["priority"].astype(str).str.strip()
    working["issue_type"] = working["issue_type"].astype(str).str.strip()
    working["variant_label"] = working["variant_label"].astype(str).str.strip()
    working["benchmark_id"] = working["benchmark_id"].astype(str).str.strip()
    working["residue_label"] = working["residue_label"].astype(str).str.strip()
    working["suggested_action"] = working["suggested_action"].astype(str).str.strip()
    if working.empty:
        return _empty_variant_remediation_summary_df()

    rows: list[dict[str, object]] = []
    group_columns = ["priority", "issue_type", "variant_label"]
    for (priority, issue_type, variant_label), group in working.groupby(group_columns, sort=True, dropna=False):
        residue_keys = (
            group["benchmark_id"].astype(str).str.strip()
            + "|"
            + group["chain"].astype(str).str.strip()
            + "|"
            + group["resid"].astype(str).str.strip()
        )
        top_residues = ", ".join(group["residue_label"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().drop_duplicates().head(8).tolist())
        action_count = int(len(group))
        status = "review-required" if action_count else "no-actions"
        warning = ""
        if str(priority).upper() == "P0":
            warning = "Ablation removes residues that current covers; do not weaken this evidence path without review."
        elif str(issue_type) == "current-missed-residue":
            warning = "Current run misses curated residues; validate numbering, mapping and detection thresholds."
        rows.append(
            {
                "priority": str(priority or ""),
                "issue_type": str(issue_type or ""),
                "variant_label": str(variant_label or ""),
                "action_count": action_count,
                "affected_case_count": int(group["benchmark_id"].replace("", pd.NA).dropna().nunique()),
                "affected_residue_count": int(residue_keys.nunique()),
                "top_residues": top_residues or "none",
                "suggested_action": str(group["suggested_action"].dropna().astype(str).head(1).iloc[0]) if not group.empty else "",
                "summary_status": status,
                "summary_warning": warning,
            }
        )

    if not rows:
        return _empty_variant_remediation_summary_df()
    frame = pd.DataFrame(rows, columns=BENCHMARK_VARIANT_REMEDIATION_SUMMARY_COLUMNS)
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    frame["_priority_rank"] = frame["priority"].map(priority_rank).fillna(99)
    frame = frame.sort_values(["_priority_rank", "variant_label", "issue_type"]).drop(columns=["_priority_rank"])
    return frame.reset_index(drop=True)


def build_pocket_benchmark_variant_remediation_checklist_markdown(
    remediation_df: Optional[pd.DataFrame],
    summary_df: Optional[pd.DataFrame] = None,
    *,
    title: str = "Pocket benchmark remediation checklist",
    max_actions: int = 80,
) -> str:
    """Render a manual review checklist for benchmark remediation actions."""

    if remediation_df is None or getattr(remediation_df, "empty", True):
        return ""

    working = remediation_df.copy()
    for column in BENCHMARK_VARIANT_REMEDIATION_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    working["_priority_rank"] = working["priority"].map(priority_rank).fillna(99)
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce").fillna(0).astype(int)
    working = working.sort_values(["_priority_rank", "variant_label", "benchmark_id", "resid", "residue_label"]).drop(columns=["_priority_rank"])

    summary = summary_df.copy() if summary_df is not None and not getattr(summary_df, "empty", True) else build_pocket_benchmark_variant_remediation_summary(working)
    lines = [
        f"# {title}",
        "",
        "Generated from `pocket_benchmark_variant_remediation_queue.csv`.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No remediation actions are currently required.")
    else:
        lines.append("| Priority | Issue | Variant | Actions | Cases | Residues |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for _, row in summary.iterrows():
            lines.append(
                "| {priority} | {issue} | {variant} | {actions} | {cases} | {residues} |".format(
                    priority=str(row.get("priority") or "-"),
                    issue=str(row.get("issue_type") or "-"),
                    variant=str(row.get("variant_label") or "-"),
                    actions=int(row.get("action_count") or 0),
                    cases=int(row.get("affected_case_count") or 0),
                    residues=int(row.get("affected_residue_count") or 0),
                )
            )

    lines.extend(["", "## Actions", ""])
    for index, (_, row) in enumerate(working.head(max_actions).iterrows(), start=1):
        lines.append(
            "- [ ] `{priority}` `{issue}` `{variant}` {case} {residue}: {action} Reference pocket `{ref_pocket}` rank `{ref_rank}`, variant pocket `{variant_pocket}` rank `{variant_rank}`.".format(
                priority=str(row.get("priority") or "-"),
                issue=str(row.get("issue_type") or "-"),
                variant=str(row.get("variant_label") or "-"),
                case=str(row.get("benchmark_id") or "current"),
                residue=str(row.get("residue_label") or f"{row.get('chain', '')}{row.get('resid', '')}"),
                action=str(row.get("suggested_action") or "Review this benchmark action."),
                ref_pocket=str(row.get("reference_matched_pocket_id") or "-"),
                ref_rank=int(row.get("reference_matched_rank") or 0),
                variant_pocket=str(row.get("variant_matched_pocket_id") or "-"),
                variant_rank=int(row.get("variant_matched_rank") or 0),
            )
        )
    if len(working) > max_actions:
        lines.append("")
        lines.append(f"Additional actions omitted: {len(working) - max_actions}. Export the CSV for the full queue.")
    return "\n".join(lines).strip() + "\n"
