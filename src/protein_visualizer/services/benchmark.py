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


def _empty_reference_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_REFERENCE_COLUMNS)


def _empty_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_DETAIL_COLUMNS)


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_SUMMARY_COLUMNS)


def _empty_variant_comparison_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BENCHMARK_VARIANT_COMPARISON_COLUMNS)


def _simplify_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _find_column(frame: pd.DataFrame, aliases: set[str]) -> Optional[str]:
    simplified = {_simplify_column_name(column): column for column in frame.columns}
    for alias in aliases:
        normalized_alias = _simplify_column_name(alias)
        if normalized_alias in simplified:
            return simplified[normalized_alias]
    return None


def _safe_int(value: object) -> Optional[int]:
    try:
        text = str(value or "").strip()
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


def _extract_residue_token(value: object) -> tuple[str, str, Optional[int]]:
    text = str(value or "").strip()
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
    prefix = f"{str(resname or '').strip().upper()} " if str(resname or "").strip() else ""
    chain_text = str(chain or "").strip()
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

        chain = str(row_dict.get(chain_column, "") if chain_column else "").strip() or token_chain
        resname = str(row_dict.get(resname_column, "") if resname_column else "").strip().upper() or token_resname
        rows.append(
            {
                "benchmark_id": str(row_dict.get(benchmark_id_column, "") if benchmark_id_column else "").strip(),
                "chain": chain,
                "resid": int(resid),
                "resname": resname,
                "reference_type": str(row_dict.get(type_column, "") if type_column else "Catalytic residue").strip() or "Catalytic residue",
                "reference_source": str(row_dict.get(source_column, "") if source_column else source_hint).strip() or source_hint,
                "reference_note": str(row_dict.get(note_column, "") if note_column else "").strip(),
                "expected_pocket_id": str(row_dict.get(expected_pocket_column, "") if expected_pocket_column else "").strip(),
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
        return pd.DataFrame(columns=["pocket_id", "chain", "resid", "resname"])
    working = pocket_df.copy()
    working["resid"] = pd.to_numeric(working["resid"], errors="coerce")
    working = working[working["resid"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["pocket_id", "chain", "resid", "resname"])
    working["pocket_id"] = working["pocket_id"].astype(str).str.strip()
    working["chain"] = working["chain"].astype(str).str.strip() if "chain" in working.columns else ""
    working["resname"] = working["resname"].astype(str).str.strip().str.upper() if "resname" in working.columns else ""
    working["resid"] = working["resid"].astype(int)
    return working[["pocket_id", "chain", "resid", "resname"]].drop_duplicates().reset_index(drop=True)


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
    working["chain"] = working["chain"].astype(str).str.strip()
    working["resname"] = working["resname"].astype(str).str.strip().str.upper()
    return working[BENCHMARK_REFERENCE_COLUMNS].drop_duplicates(subset=["benchmark_id", "chain", "resid", "reference_type"]).reset_index(drop=True)


def _matches_reference(pocket_row: pd.Series, reference_row: pd.Series) -> bool:
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
