from __future__ import annotations

from datetime import datetime
import html
import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.config.settings import SETTINGS
from protein_visualizer.sample_data import ANNOTATION_TEXT, MMPBSA_TEXT, PDB_TEXT, POCKET_TEXT
from protein_visualizer.services.ai_evidence import (
    apply_ai_review_decisions_to_audit,
    build_ai_evidence_audit_table,
    build_ai_followup_evidence_plan,
    build_ai_followup_prompt_bundle,
    build_ai_evidence_review_queue,
    build_ai_review_artifact_manifest,
    build_ai_review_artifact_bundle_zip,
    build_ai_review_checklist_markdown,
    build_ai_review_bundle_certificate_markdown,
    build_ai_review_bundle_readme_markdown,
    build_ai_review_bundle_verification_summary,
    build_ai_review_decision_template,
    build_ai_review_decision_outcome_table,
    build_ai_review_decision_validation_table,
    build_ai_review_ranking_delta,
    build_ai_review_round_report_markdown,
    build_ai_review_round_summary,
    build_ai_ranking_impact_summary,
    build_residue_evidence_consensus,
    fetch_ai_residue_evidence,
    filter_ai_evidence_for_ranking,
    parse_ai_review_decision_table,
    parse_ai_residue_evidence_payload,
    verify_ai_review_artifact_bundle_zip,
)
from protein_visualizer.services.benchmark import (
    build_pocket_benchmark_case_interpretation_summary,
    build_pocket_benchmark_case_interpretation_matrix,
    build_pocket_benchmark_case_interpretation_matrix_queue,
    build_pocket_benchmark_case_interpretation_matrix_summary,
    build_pocket_benchmark_dataset_interpretation_checklist_markdown,
    build_pocket_benchmark_dataset_interpretation_report_markdown,
    build_pocket_benchmark_dataset_interpretation,
    build_pocket_benchmark_dataset_interpretation_queue,
    build_pocket_benchmark_interpretation_summary,
    build_pocket_benchmark_case_summary,
    build_pocket_benchmark_dataset_summary,
    build_pocket_benchmark_details,
    build_pocket_benchmark_reference_quality_checklist_markdown,
    build_pocket_benchmark_reference_quality_issues,
    build_pocket_benchmark_reference_quality_summary,
    build_pocket_benchmark_reference_candidate_accepted_reference,
    build_pocket_benchmark_reference_candidate_review_checklist_markdown,
    build_pocket_benchmark_reference_candidate_review_decision_template,
    build_pocket_benchmark_reference_candidate_review_decision_validation,
    build_pocket_benchmark_reference_candidate_review_outcomes,
    build_pocket_benchmark_reference_candidate_review_queue,
    build_pocket_benchmark_reference_from_external_evidence,
    build_pocket_benchmark_reference_import_summary,
    build_pocket_benchmark_reference_source_audit,
    build_pocket_benchmark_reference_source_audit_action_queue,
    build_pocket_benchmark_reference_source_audit_case_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_closure_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_closure_queue,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases,
    build_pocket_benchmark_reference_source_audit_case_decision_outcomes,
    build_pocket_benchmark_reference_source_audit_case_decision_outcome_summary,
    build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact,
    build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary,
    build_pocket_benchmark_reference_source_audit_case_decision_template,
    build_pocket_benchmark_reference_source_audit_case_decision_validation,
    build_pocket_benchmark_reference_source_audit_case_summary,
    build_pocket_benchmark_reference_source_audit_checklist_markdown,
    build_pocket_benchmark_reference_source_audit_summary,
    build_pocket_benchmark_reference_readiness_case_summary,
    build_pocket_benchmark_reference_readiness_checklist_markdown,
    build_pocket_benchmark_reference_readiness_queue,
    build_pocket_benchmark_reference_readiness_summary,
    build_pocket_benchmark_reference_structure_validation,
    build_pocket_benchmark_reference_structure_validation_checklist_markdown,
    build_pocket_benchmark_reference_structure_validation_summary,
    build_pocket_benchmark_reference_template,
    build_pocket_benchmark_reference_template_markdown,
    build_pocket_benchmark_summary,
    build_pocket_benchmark_variant_comparison,
    build_pocket_benchmark_variant_case_comparison,
    build_pocket_benchmark_variant_dataset_comparison,
    build_pocket_benchmark_variant_detail_comparison,
    build_pocket_benchmark_variant_remediation_checklist_markdown,
    build_pocket_benchmark_variant_remediation_queue,
    build_pocket_benchmark_variant_remediation_summary,
    parse_benchmark_reference_table,
    parse_pocket_benchmark_reference_candidate_review_decision_table,
    parse_pocket_benchmark_reference_source_audit_case_decision_table,
    select_pocket_benchmark_reference_source,
)
from protein_visualizer.services.candidate_fusion import build_joint_candidate_table, build_pocket_consensus_coverage
from protein_visualizer.services.comparison import compare_pocket_ranking_summaries
from protein_visualizer.services.conservation import parse_conservation_evidence_table
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.external_sites import (
    extract_pdb_id_from_text,
    fetch_combined_functional_sites_for_structure,
    merge_external_evidence_tables,
)
from protein_visualizer.services.explainer import explain_analysis
from protein_visualizer.services.hotspot import identify_hotspots
from protein_visualizer.services.interface import (
    build_inferred_interface_annotations,
    build_interface_overlap_summary,
    build_interface_summary,
    enrich_interface_annotations,
    merge_interface_annotation_tables,
    parse_interface_annotation_table,
)
from protein_visualizer.services.literature_sites import fetch_literature_residue_evidence_for_structure, remove_literature_evidence
from protein_visualizer.services.parsers import parse_pdb_atoms
from protein_visualizer.services.pdf_export import PDF_EXPORT_AVAILABLE, build_simple_pdf
from protein_visualizer.services.pocket_decision import (
    add_pocket_residue_layers,
    build_consensus_rerank_action_checklist_markdown,
    build_consensus_rerank_action_queue,
    build_consensus_rerank_apply_simulation,
    build_consensus_rerank_guardrail_artifact_manifest,
    build_consensus_rerank_guardrail_bundle_verification_summary,
    build_consensus_rerank_guardrail_handoff_certificate_markdown,
    build_consensus_rerank_guardrail_handoff_zip,
    build_consensus_rerank_policy_gate,
    build_consensus_rerank_preview,
    build_consensus_rerank_precision_guardrail,
    build_consensus_rerank_precision_guardrail_report_markdown,
    build_consensus_rerank_precision_scorecard,
    build_consensus_rerank_release_apply_plan,
    build_consensus_rerank_release_apply_report_markdown,
    build_consensus_rerank_release_closure_blocker_queue,
    build_consensus_rerank_release_closure_certificate_markdown,
    build_consensus_rerank_release_closure_detached_manifest,
    build_consensus_rerank_release_closure_ledger,
    build_consensus_rerank_release_closure_remediation_checklist_markdown,
    build_consensus_rerank_release_closure_summary,
    build_consensus_rerank_release_decision_summary,
    build_consensus_rerank_release_decision_template,
    build_consensus_rerank_release_execution_report_markdown,
    build_consensus_rerank_release_execution_template,
    build_consensus_rerank_release_execution_summary,
    build_consensus_rerank_simulation_delta,
    build_consensus_rerank_suggestion,
    build_pocket_decision_table,
    build_pocket_precision_triage,
    build_pocket_reliability_checklist,
    parse_consensus_rerank_release_decision_table,
    parse_consensus_rerank_release_execution_table,
    validate_consensus_rerank_release_execution_receipt,
    validate_consensus_rerank_release_decisions,
    verify_consensus_rerank_guardrail_handoff_zip,
)
from protein_visualizer.services.pocket import (
    PYKVFINDER_AVAILABLE,
    build_auto_pocket_display_table,
    build_pocket_detection_diagnostics_table,
    build_pocket_summary,
    build_pocket_summary_without_conservation_signal,
    detect_auto_pocket_table,
    get_pocket_detection_metadata,
    parse_pocket_table,
    summarize_pocket_detection_metadata,
)
from protein_visualizer.services.reporting import build_analysis_summary, format_energy_value
from protein_visualizer.services.session_state import (
    append_history_record,
    get_current_energy_table,
    get_current_mmpbsa_text,
    get_current_pdb_text,
    get_uploaded_inputs_cache,
    initialize_state,
    set_analysis_state,
    set_uploaded_inputs_cache,
)
from protein_visualizer.services.snapshot import build_analysis_snapshot, build_snapshot_svg, snapshot_to_json_bytes
from protein_visualizer.services.structure_energy import estimate_protein_volume, resolve_energy_table
from protein_visualizer.services.viewer import build_view


st.set_page_config(page_title="口袋与界面", layout="wide")
st.title("口袋 / 界面专页")
st.caption("聚焦口袋、界面注释和热点交集，可作为结构可视化页之外的独立补充分析页。")
initialize_state()
st.markdown(
    """
    <style>
    .decision-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 10px 0 16px 0; }
    .decision-card { border-radius: 16px; padding: 15px; border: 1px solid #dbe4f0; background: linear-gradient(145deg, #ffffff, #f8fbff); box-shadow: 0 8px 24px rgba(15,23,42,0.06); min-height: 220px; }
    .decision-card.ready { border-color: #34d399; background: linear-gradient(145deg, #ecfdf5, #ffffff); }
    .decision-card.review { border-color: #f59e0b; background: linear-gradient(145deg, #fffbeb, #ffffff); }
    .decision-card.explore { border-color: #93c5fd; background: linear-gradient(145deg, #eff6ff, #ffffff); }
    .decision-kicker { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; font-weight: 800; }
    .decision-title { font-size: 18px; font-weight: 800; color: #0f172a; margin: 6px 0; }
    .decision-score { font-size: 28px; font-weight: 900; color: #1d4ed8; margin: 4px 0; }
    .decision-meta { font-size: 12px; color: #334155; line-height: 1.55; margin-top: 8px; }
    .decision-pill { display: inline-block; border-radius: 999px; padding: 3px 8px; background: rgba(15,23,42,0.06); color: #0f172a; font-size: 11px; margin: 3px 4px 0 0; }
    @media (max-width: 980px) { .decision-grid { grid-template-columns: 1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入数据")
    uploaded_pdb = st.file_uploader("上传 PDB 文件（可选）", type=["pdb"], accept_multiple_files=False)
    uploaded_mmpbsa = st.file_uploader("上传 MMPBSA 文件（可选）", type=["txt", "dat", "out", "csv"], accept_multiple_files=False)
    uploaded_pocket = st.file_uploader("上传 Pocket CSV（可选）", type=["csv", "txt"], accept_multiple_files=False)
    uploaded_annotation = st.file_uploader("上传界面注释 CSV（可选）", type=["csv", "txt"], accept_multiple_files=False)
    use_examples = st.checkbox("使用示例数据", value=False)
    energy_mode = st.selectbox(
        "能量来源模式",
        ["auto", "mmpbsa", "estimate"],
        index=0,
        format_func=lambda x: {"auto": "自动", "mmpbsa": "上传 MMPBSA", "estimate": "结构估算"}[x],
    )

def _read_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def _uploaded_file_entry(uploaded_file) -> dict | None:
    if uploaded_file is None:
        return None
    text = _read_uploaded_text(uploaded_file)
    if not text:
        return None
    return {
        "name": str(getattr(uploaded_file, "name", "uploaded_file")),
        "text": text,
    }


def _to_csv_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode("utf-8")


benchmark_reference_template_df = build_pocket_benchmark_reference_template()
benchmark_reference_template_markdown = build_pocket_benchmark_reference_template_markdown()


def _external_evidence_counts(table: pd.DataFrame) -> dict[str, int]:
    if table is None or getattr(table, "empty", True):
        return {"rows": 0, "exact": 0, "weak": 0}
    exact_rows = 0
    weak_rows = 0
    if "mapping_level" in table.columns:
        level_series = table["mapping_level"].astype(str).str.lower()
        exact_rows = int((level_series == "exact").sum())
        weak_rows = int((level_series == "weak").sum())
    return {"rows": int(len(table)), "exact": exact_rows, "weak": weak_rows}


def _source_audit_case_summary_counts(table: pd.DataFrame) -> tuple[int, int]:
    if table is None or getattr(table, "empty", True) or not {"blocker_rows", "review_rows"}.issubset(table.columns):
        return 0, 0
    blocker_rows = pd.to_numeric(table["blocker_rows"], errors="coerce").fillna(0)
    review_rows = pd.to_numeric(table["review_rows"], errors="coerce").fillna(0)
    blocked_cases = int(blocker_rows.gt(0).sum())
    review_cases = int((blocker_rows.eq(0) & review_rows.gt(0)).sum())
    return blocked_cases, review_cases


@st.cache_data(show_spinner=False, ttl=3600)
def _load_external_evidence(
    accession: str,
    chain_hint: str,
    pdb_id: str,
    pdb_text: str,
    ec_number: str,
    enable_uniprot: bool,
    enable_mcsa: bool,
) -> tuple[pd.DataFrame, dict]:
    return fetch_combined_functional_sites_for_structure(
        accession,
        ec_number=ec_number,
        chain_hint=chain_hint,
        pdb_id=pdb_id,
        pdb_text=pdb_text,
        enable_uniprot=enable_uniprot,
        enable_mcsa=enable_mcsa,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _load_literature_evidence(
    query: str,
    manual_text: str,
    accession: str,
    ec_number: str,
    pdb_id: str,
    protein_name: str,
    chain_hint: str,
    pdb_text: str,
    max_articles: int,
    enable_pubmed: bool,
    enable_europepmc: bool,
    include_europepmc_fulltext: bool,
    max_fulltext_articles: int,
    assume_structure_numbering: bool,
) -> tuple[pd.DataFrame, dict]:
    return fetch_literature_residue_evidence_for_structure(
        query=query,
        manual_text=manual_text,
        accession=accession,
        ec_number=ec_number,
        pdb_id=pdb_id,
        protein_name=protein_name,
        chain_hint=chain_hint,
        pdb_text=pdb_text,
        max_articles=max_articles,
        enable_pubmed=enable_pubmed,
        enable_europepmc=enable_europepmc,
        include_europepmc_fulltext=include_europepmc_fulltext,
        max_fulltext_articles=max_fulltext_articles,
        assume_structure_numbering=assume_structure_numbering,
    )


def _normalize_pocket_table(table: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if table is None or getattr(table, "empty", True):
        return pd.DataFrame()

    normalized = table.copy()
    normalized["pocket_source"] = source_label
    if "residue_label" not in normalized.columns:
        normalized["residue_label"] = normalized.apply(
            lambda row: f"{row.resname} {row.chain}{int(row.resid)}",
            axis=1,
        )
    return normalized


def _merge_pocket_tables(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    valid_tables = [table for table in [primary_df, secondary_df] if table is not None and not getattr(table, "empty", True)]
    if not valid_tables:
        return pd.DataFrame()

    combined = pd.concat(valid_tables, ignore_index=True)
    for column in ("score", "residue_score", "consensus_score"):
        if column not in combined.columns:
            combined[column] = 0.0

    combined = combined.sort_values(
        ["score", "residue_score", "consensus_score", "pocket_id", "chain", "resid"],
        ascending=[False, False, False, True, True, True],
    )
    combined = combined.drop_duplicates(subset=["pocket_id", "chain", "resid", "resname"], keep="first")
    return combined.reset_index(drop=True)


def _resolve_pocket_source(manual_df: pd.DataFrame, auto_df: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    if source_mode == "uploaded":
        return manual_df.copy() if manual_df is not None else pd.DataFrame()
    if source_mode == "auto":
        return auto_df.copy() if auto_df is not None else pd.DataFrame()
    return _merge_pocket_tables(manual_df, auto_df)


def _resolve_annotation_source(uploaded_df: pd.DataFrame, inferred_df: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    if source_mode == "uploaded":
        return uploaded_df.copy() if uploaded_df is not None else pd.DataFrame()
    if source_mode == "inferred":
        return inferred_df.copy() if inferred_df is not None else pd.DataFrame()
    return merge_interface_annotation_tables(uploaded_df, inferred_df)


def _residue_pairs(table: pd.DataFrame) -> list[tuple[str, int]]:
    if table is None or getattr(table, "empty", True):
        return []
    return [(str(row.chain).strip() or "A", int(row.resid)) for row in table.itertuples(index=False)]


def _render_pocket_decision_panel(
    decision_df: pd.DataFrame,
    checklist_df: pd.DataFrame | None = None,
    triage_df: pd.DataFrame | None = None,
) -> None:
    st.subheader("Active-site decision panel")
    st.caption(
        "This panel turns pocket ranking into an auditable product view: functional evidence, geometry support, A/B movement, risk flags, and the next action."
    )
    if decision_df is None or getattr(decision_df, "empty", True):
        st.info("No decision panel is available yet. Run auto-pocket detection or add pocket evidence first.")
        return

    cards = []
    for _, row in decision_df.head(3).iterrows():
        audit_status = str(row.get("audit_status") or "")
        card_class = "ready" if audit_status == "ready-to-validate" else ("review" if "review" in audit_status or "needed" in audit_status else "explore")
        risk_flags = str(row.get("risk_flags") or "none")
        pills = "".join(
            f'<span class="decision-pill">{html.escape(flag.strip())}</span>'
            for flag in risk_flags.split(",")
            if flag.strip()
        )
        cards.append(
            """
            <div class="decision-card {card_class}">
              <div class="decision-kicker">Rank #{rank} | {audit_status}</div>
              <div class="decision-title">{pocket_id}</div>
              <div>{decision_label}</div>
              <div class="decision-score">{decision_score:.3f}</div>
              <div class="decision-meta">
                Functional {functional:.3f} / Geometry {geometry:.3f}<br/>
                Evidence: {quality}<br/>
                Action: {action}<br/>
                A/B: literature {lit_delta:+d}, route {route_delta:+d}, conservation {cons_delta:+d}
              </div>
              <div style="margin-top:8px;">{pills}</div>
              <div class="decision-meta"><strong>Next:</strong> {next_step}</div>
            </div>
            """.format(
                card_class=card_class,
                rank=int(row.get("decision_rank") or 0),
                audit_status=html.escape(audit_status or "-"),
                pocket_id=html.escape(str(row.get("pocket_id") or "-")),
                decision_label=html.escape(str(row.get("decision_label") or "-")),
                decision_score=float(row.get("decision_score") or 0.0),
                functional=float(row.get("functional_confidence") or 0.0),
                geometry=float(row.get("geometry_confidence") or 0.0),
                quality=html.escape(str(row.get("evidence_quality_label") or "-")),
                action=html.escape(str(row.get("recommended_action") or "-")),
                lit_delta=int(row.get("literature_rank_delta") or 0),
                route_delta=int(row.get("evidence_route_rank_delta") or 0),
                cons_delta=int(row.get("conservation_rank_delta") or 0),
                pills=pills or '<span class="decision-pill">none</span>',
                next_step=html.escape(str(row.get("next_step") or "-")),
            )
        )
    st.markdown(f'<div class="decision-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    if checklist_df is not None and not getattr(checklist_df, "empty", True):
        st.markdown("##### Reliability checklist")
        st.caption(
            "Pass = usable signal; Review = useful but needs inspection; Missing = precision gap before treating the pocket as an active-site call."
        )
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)

    if triage_df is not None and not getattr(triage_df, "empty", True):
        st.markdown("##### Precision triage")
        st.caption(
            "Triage compresses the checklist into a product-level action: validate, review mapping, add evidence, refine geometry, or keep exploratory."
        )
        st.dataframe(triage_df, use_container_width=True, hide_index=True)

    with st.expander("Decision audit table", expanded=False):
        st.dataframe(decision_df, use_container_width=True, hide_index=True)


POCKET_SOURCE_LABELS = {
    "uploaded": "上传 Pocket CSV",
    "auto": "自动识别口袋",
    "combined": "上传 + 自动合并",
}

ANNOTATION_SOURCE_LABELS = {
    "uploaded": "上传界面注释",
    "inferred": "结构推断界面",
    "combined": "上传 + 推断合并",
}

INFERENCE_BASIS_LABELS = {
    "inter-chain-contact": "跨链接触",
    "surface-contact": "表面接触",
}


cached_inputs = get_uploaded_inputs_cache()
cached_pdb_files = list(cached_inputs.get("pdb_files", []))
cached_mmpbsa_files = list(cached_inputs.get("mmpbsa_files", []))
cached_pocket_entry = cached_inputs.get("pocket_file")
shared_energy_table = get_current_energy_table()
shared_pdb_text = str(get_current_pdb_text() or "")
shared_mmpbsa_text = str(get_current_mmpbsa_text() or "")

if uploaded_pdb is not None:
    pdb_text = _read_uploaded_text(uploaded_pdb)
    pdb_input_note = "当前使用本页上传的 PDB。"
elif use_examples:
    pdb_text = PDB_TEXT
    pdb_input_note = "当前使用示例 PDB。"
elif shared_energy_table is not None and shared_pdb_text.strip():
    pdb_text = shared_pdb_text
    pdb_input_note = "当前复用结构可视化页最近一次分析的 PDB。"
elif cached_pdb_files:
    pdb_text = str(cached_pdb_files[0].get("text") or "")
    pdb_input_note = f"当前复用已缓存的 PDB：{cached_pdb_files[0].get('name', 'uploaded.pdb')}。"
else:
    pdb_text = ""
    pdb_input_note = ""

if uploaded_mmpbsa is not None:
    mmpbsa_text = _read_uploaded_text(uploaded_mmpbsa)
    mmpbsa_input_note = "当前使用本页上传的 MMPBSA。"
elif use_examples:
    mmpbsa_text = MMPBSA_TEXT
    mmpbsa_input_note = "当前使用示例 MMPBSA。"
elif shared_mmpbsa_text.strip() and "结构估算" not in shared_mmpbsa_text:
    mmpbsa_text = shared_mmpbsa_text
    mmpbsa_input_note = "当前复用结构可视化页最近一次分析的 MMPBSA。"
elif cached_mmpbsa_files:
    mmpbsa_text = str(cached_mmpbsa_files[0].get("text") or "")
    mmpbsa_input_note = f"当前复用已缓存的 MMPBSA：{cached_mmpbsa_files[0].get('name', 'uploaded_mmpbsa')}。"
else:
    mmpbsa_text = ""
    mmpbsa_input_note = ""

if uploaded_pocket is not None:
    pocket_text = _read_uploaded_text(uploaded_pocket)
    pocket_input_note = "当前使用本页上传的 Pocket。"
elif use_examples:
    pocket_text = POCKET_TEXT
    pocket_input_note = "当前使用示例 Pocket。"
elif cached_pocket_entry and cached_pocket_entry.get("text"):
    pocket_text = str(cached_pocket_entry.get("text") or "")
    pocket_input_note = f"当前复用已缓存的 Pocket：{cached_pocket_entry.get('name', 'uploaded_pocket.csv')}。"
else:
    pocket_text = ""
    pocket_input_note = ""

if uploaded_annotation is not None:
    annotation_text = _read_uploaded_text(uploaded_annotation)
elif use_examples:
    annotation_text = ANNOTATION_TEXT
else:
    annotation_text = ""

benchmark_reference_text = _read_uploaded_text(uploaded_benchmark_reference) if uploaded_benchmark_reference is not None else ""

if not use_examples:
    new_pdb_entry = _uploaded_file_entry(uploaded_pdb)
    new_mmpbsa_entry = _uploaded_file_entry(uploaded_mmpbsa)
    new_pocket_entry = _uploaded_file_entry(uploaded_pocket)
    if new_pdb_entry is not None or new_mmpbsa_entry is not None or new_pocket_entry is not None:
        set_uploaded_inputs_cache(
            pdb_files=[new_pdb_entry] if new_pdb_entry is not None else cached_pdb_files,
            mmpbsa_files=[new_mmpbsa_entry] if new_mmpbsa_entry is not None else cached_mmpbsa_files,
            pocket_file=new_pocket_entry if new_pocket_entry is not None else cached_pocket_entry,
        )

if not pdb_text:
    st.warning("请上传 PDB 文件或勾选使用示例数据。")
    st.stop()

for input_note in [pdb_input_note, mmpbsa_input_note, pocket_input_note]:
    if input_note:
        st.caption(input_note)

try:
    atom_df = parse_pdb_atoms(pdb_text)
except Exception as exc:
    st.error(f"PDB 解析失败：{exc}")
    st.stop()

try:
    energy_df, energy_source = resolve_energy_table(
        pdb_text,
        energy_mode=energy_mode,
        mmpbsa_text=mmpbsa_text or None,
    )
    if energy_df is not None and not energy_df.empty:
        energy_table = prepare_energy_table(atom_df, energy_df)
        energy_table["energy_source"] = energy_source
    else:
        energy_table = pd.DataFrame()
except Exception:
    energy_table = pd.DataFrame()

try:
    structure_energy_df, _ = resolve_energy_table(
        pdb_text,
        energy_mode="estimate",
        mmpbsa_text=None,
    )
    if structure_energy_df is not None and not structure_energy_df.empty:
        structure_energy_table = prepare_energy_table(atom_df, structure_energy_df)
        structure_energy_table["energy_source"] = "结构估算"
    else:
        structure_energy_table = pd.DataFrame()
except Exception:
    structure_energy_table = pd.DataFrame()

if not energy_table.empty and "delta_total" in energy_table.columns:
    energy_series = pd.to_numeric(energy_table["delta_total"], errors="coerce").dropna()
    energy_limit = float(max(abs(energy_series.min()), abs(energy_series.max()))) if not energy_series.empty else 0.1
else:
    energy_limit = 0.1

hotspot_slider_max = max(energy_limit, 0.1)
hotspot_slider_value = 0.0 if energy_limit <= 0 else min(1.0, hotspot_slider_max)

with st.sidebar:
    st.header("热点筛选")
    hotspot_threshold = st.slider(
        "MMPBSA |阈值| (绝对值)",
        0.0,
        hotspot_slider_max,
        hotspot_slider_value,
        0.1,
        disabled=energy_table.empty,
    )
    hotspot_top_n = st.slider("热点保底数量", 1, 20, 5, 1, disabled=energy_table.empty)

    with st.expander("自动口袋识别参数", expanded=False):
        auto_adaptive_profile = st.checkbox("启用结构自适应参数", value=True)
        auto_use_kvfinder = st.checkbox(
            "优先使用 pyKVFinder 口袋引擎（推荐）",
            value=True,
            disabled=not PYKVFINDER_AVAILABLE,
        )
        auto_use_p2rank = st.checkbox("Enable P2Rank boost when installed locally", value=False)
        p2rank_profile = st.selectbox("P2Rank profile", ["default", "alphafold"], index=0)
        p2rank_executable = st.text_input(
            "P2Rank executable path (optional)",
            value="",
            placeholder="e.g. C:\\tools\\p2rank\\prank.bat",
        )
        enable_p2rank_ab = st.checkbox(
            "Show P2Rank A/B pocket comparison",
            value=False,
            disabled=not auto_use_p2rank,
        )
        if PYKVFINDER_AVAILABLE:
            st.caption("当前环境已检测到 pyKVFinder，将优先使用库算法进行口袋识别。")
        else:
            st.caption("当前环境未检测到 pyKVFinder，将使用几何启发式检测。")

        auto_detection_mode = st.selectbox(
            "识别策略",
            ["auto", "geometry"],
            index=0,
            format_func=lambda x: {"auto": "配体优先", "geometry": "纯几何"}[x],
        )
        auto_ligand_radius = st.slider("配体邻域半径", 3.0, 8.0, 5.0, 0.5)
        auto_contact_cutoff = st.slider("残基接触半径", 6.0, 12.0, 8.0, 0.5)
        auto_cluster_cutoff = st.slider("聚类距离阈值", 4.0, 14.0, 8.5, 0.5)
        auto_candidate_fraction = st.slider("候选残基比例", 0.15, 0.60, 0.35, 0.05)
        auto_max_candidates = st.slider("最大候选残基数", 3, 24, 18, 1)
        auto_max_pockets = st.slider("最多口袋数", 1, 8, 6, 1)

    with st.expander("外部关键位点证据（可选）", expanded=False):
        enable_uniprot_evidence = st.checkbox("启用 UniProt 功能位点增强", value=False)
        enable_mcsa_evidence = st.checkbox("Enable M-CSA catalytic-site boost", value=False)
        uniprot_accession = st.text_input("UniProt Accession", value="", placeholder="例如: P00533")
        enzyme_ec_number = st.text_input("EC Number (optional)", value="", placeholder="e.g. 3.2.1.4")
        uniprot_chain_hint = st.text_input("链提示（可选）", value="", placeholder="例如: A")
        enable_literature_evidence = st.checkbox("Enable literature residue mining", value=False)
        literature_query = st.text_input("Literature query override (optional)", value="", placeholder="e.g. enzyme name catalytic residue")
        literature_protein_name = st.text_input("Protein name for literature search (optional)", value="")
        enable_europepmc_evidence = st.checkbox(
            "Enable Europe PMC open-text mining",
            value=False,
            disabled=not enable_literature_evidence,
        )
        include_europepmc_fulltext = st.checkbox(
            "Include Europe PMC open full text",
            value=True,
            disabled=not enable_literature_evidence or not enable_europepmc_evidence,
        )
        literature_max_articles = st.slider(
            "Literature articles to scan",
            1,
            12,
            6,
            1,
            disabled=not enable_literature_evidence,
        )
        literature_max_fulltext = st.slider(
            "Europe PMC full texts to scan",
            0,
            4,
            2,
            1,
            disabled=not enable_literature_evidence or not enable_europepmc_evidence or not include_europepmc_fulltext,
        )
        uploaded_literature = st.file_uploader(
            "Upload literature text / abstracts (optional)",
            type=["txt", "md", "xml"],
            accept_multiple_files=False,
        )
        literature_assume_structure_numbering = st.checkbox(
            "Assume literature residue numbers match the uploaded PDB chain",
            value=False,
            disabled=not bool(str(uniprot_chain_hint or "").strip()),
        )
        enable_ai_evidence = st.checkbox("Enable AI residue evidence assistant", value=False)
        ai_context_text = st.text_area(
            "AI source text / notes",
            value="",
            height=120,
            disabled=not enable_ai_evidence,
            placeholder="Paste abstracts, paper snippets, or reviewer notes. AI should extract only supported enzyme residues.",
        )
        ai_payload_text = st.text_area(
            "Paste AI residue JSON (optional)",
            value="",
            height=110,
            disabled=not enable_ai_evidence,
            placeholder='{"residues":[{"resname":"SER","position_text":"Ser195","confidence":0.86,"evidence_snippet":"..."}]}',
        )
        ai_api_url = st.text_input(
            "AI API URL (OpenAI-compatible, optional)",
            value=os.getenv("AI_EVIDENCE_API_URL", ""),
            disabled=not enable_ai_evidence or bool(str(ai_payload_text or "").strip()),
            placeholder="https://.../v1/chat/completions",
        )
        ai_model = st.text_input(
            "AI model",
            value=os.getenv("AI_EVIDENCE_MODEL", ""),
            disabled=not enable_ai_evidence or bool(str(ai_payload_text or "").strip()),
            placeholder="model name",
        )
        ai_api_key = st.text_input(
            "AI API key (optional; env AI_EVIDENCE_API_KEY is also supported)",
            value="",
            type="password",
            disabled=not enable_ai_evidence or bool(str(ai_payload_text or "").strip()),
        )
        ai_min_confidence = st.slider(
            "AI residue min confidence",
            0.20,
            0.90,
            0.35,
            0.05,
            disabled=not enable_ai_evidence,
        )
        ai_assume_structure_numbering = st.checkbox(
            "Assume AI residue numbers match the uploaded PDB chain",
            value=False,
            disabled=not enable_ai_evidence or not bool(str(uniprot_chain_hint or "").strip()),
        )
        ai_allow_review_ranking = st.checkbox(
            "Allow review-level AI evidence to influence ranking",
            value=False,
            disabled=not enable_ai_evidence,
        )
        uploaded_ai_review_decisions = st.file_uploader(
            "Upload AI review decisions CSV/TSV (optional)",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
            disabled=not enable_ai_evidence,
        )
        st.caption("Decision columns: chain,resid,evidence_type,review_decision,verified_source,verified_snippet,review_note.")
        uploaded_consensus_rerank_release_decisions = st.file_uploader(
            "Upload consensus rerank release decision CSV (optional)",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption("Release decision columns: decision_item_id,review_decision,reviewer,verified_anchor_residues,verified_sources,blocker_resolved.")
        uploaded_consensus_rerank_release_execution_receipt = st.file_uploader(
            "Upload consensus rerank release execution receipt CSV (optional)",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption("Execution receipt columns: execution_item_id,execution_decision,applied_rank,operator,executed_at,plan_sha256.")
        st.caption("AI evidence is audit-gated: conflicting/unsupported AI residues are kept for review/export but do not affect ranking.")
        enable_literature_ab = st.checkbox(
            "Show literature A/B pocket comparison",
            value=False,
            disabled=not enable_literature_evidence and uploaded_literature is None,
        )
        auto_external_evidence_route = st.checkbox("Enable external evidence-guided pocket route", value=True)
        external_route_min_support = st.slider(
            "Evidence route min support",
            0.30,
            1.00,
            0.58,
            0.02,
            disabled=not auto_external_evidence_route,
        )
        external_route_min_confidence = st.slider(
            "Evidence route min mapping confidence",
            0.30,
            1.00,
            0.55,
            0.02,
            disabled=not auto_external_evidence_route,
        )
        external_route_min_quality = st.slider(
            "Evidence route min mapping quality",
            0.50,
            1.00,
            0.82,
            0.02,
            disabled=not auto_external_evidence_route,
        )
        external_route_radius_mode = st.selectbox(
            "Evidence route neighborhood radius",
            ["auto", "manual"],
            index=0,
            disabled=not auto_external_evidence_route,
        )
        external_route_radius = (
            st.slider(
                "Manual evidence route radius",
                3.5,
                12.0,
                6.0,
                0.5,
                disabled=not auto_external_evidence_route or external_route_radius_mode != "manual",
            )
            if external_route_radius_mode == "manual"
            else None
        )
        enable_evidence_route_ab = st.checkbox(
            "Show evidence-route A/B pocket comparison",
            value=False,
            disabled=not auto_external_evidence_route,
        )
        st.caption("UniProt, M-CSA, and high-confidence literature residue evidence are used in auto-pocket detection and final reranking.")

    with st.expander("Conservation Evidence (optional)", expanded=False):
        uploaded_conservation = st.file_uploader(
            "Upload ConSurf / conservation table",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        conservation_source_name = st.text_input("Conservation source label", value="ConSurf")
        enable_conservation_ab = st.checkbox(
            "Show conservation A/B ranking comparison",
            value=False,
            disabled=uploaded_conservation is None,
        )
        st.caption("Imported conservation scores are kept as an independent rerank-only signal; they do not seed candidate generation.")

    with st.expander("Benchmark Reference (optional)", expanded=False):
        uploaded_benchmark_reference = st.file_uploader(
            "Upload curated catalytic residues CSV/TSV",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        benchmark_source_name = st.text_input("Benchmark source label", value="Curated catalytic benchmark")
        use_external_evidence_as_benchmark_reference = st.checkbox(
            "Use loaded external evidence as provisional benchmark reference when no curated file is uploaded",
            value=False,
        )
        use_reviewed_candidate_as_benchmark_reference = st.checkbox(
            "Use accepted reviewed candidate references as benchmark reference when no curated file is uploaded",
            value=True,
        )
        uploaded_benchmark_reference_candidate_review_decisions = st.file_uploader(
            "Upload benchmark reference candidate review decisions CSV/TSV",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        uploaded_benchmark_reference_source_audit_case_decisions = st.file_uploader(
            "Upload benchmark source audit case decisions CSV/TSV",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=False,
        )
        st.caption(
            "Columns can include chain,resid,resname,reference_type,reference_source,reference_note,expected_pocket_id. "
            "Blank chain is treated as wildcard. Provisional external-evidence references are useful for triage, "
            "but should not be treated as independent accuracy proof until curated separately."
        )
        st.caption("Candidate review decisions need action_id,review_decision,reviewer,verified_source,verified_mapping,review_note.")
        st.caption("Source audit case decisions need benchmark_id,source_decision,reviewer,verified_independence,decision_note.")
        st.download_button(
            "Download benchmark reference template CSV",
            data=_to_csv_bytes(benchmark_reference_template_df),
            file_name="pocket_benchmark_reference_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download benchmark reference template notes",
            data=benchmark_reference_template_markdown.encode("utf-8"),
            file_name="pocket_benchmark_reference_template.md",
            mime="text/markdown",
        )

hotspot_df = (
    identify_hotspots(
        energy_table,
        energy_threshold=-abs(hotspot_threshold) if hotspot_threshold > 0 else -1.0,
        top_n=hotspot_top_n,
    )
    if not energy_table.empty
    else pd.DataFrame()
)

try:
    uploaded_pocket_df = parse_pocket_table(pocket_text) if pocket_text else pd.DataFrame()
except Exception as exc:
    st.warning(f"Pocket 文件解析失败：{exc}")
    uploaded_pocket_df = pd.DataFrame()

try:
    uploaded_annotation_df = parse_interface_annotation_table(annotation_text) if annotation_text else pd.DataFrame()
except Exception as exc:
    st.warning(f"界面注释文件解析失败：{exc}")
    uploaded_annotation_df = pd.DataFrame()

hotspot_residues = _residue_pairs(hotspot_df)

external_site_df = pd.DataFrame()
external_site_meta: dict = {}
literature_site_df = pd.DataFrame()
literature_site_meta: dict = {}
ai_evidence_df = pd.DataFrame()
ai_evidence_meta: dict = {}
ai_evidence_audit_df = pd.DataFrame()
ai_evidence_review_queue_df = pd.DataFrame()
ai_review_decision_template_df = pd.DataFrame()
ai_review_decision_df = pd.DataFrame()
ai_review_decision_validation_df = pd.DataFrame()
ai_review_decision_outcome_df = pd.DataFrame()
ai_review_round_summary_df = pd.DataFrame()
ai_review_ranking_delta_df = pd.DataFrame()
ai_review_round_report_markdown = ""
ai_review_artifact_manifest_df = pd.DataFrame()
ai_review_bundle_readme_markdown = ""
ai_review_artifact_bundle_zip = b""
ai_review_bundle_verification_df = pd.DataFrame()
ai_review_bundle_verification_summary_df = pd.DataFrame()
ai_review_bundle_certificate_markdown = ""
ai_review_decision_meta: dict = {}
rankable_ai_evidence_df = pd.DataFrame()
rankable_ai_evidence_before_review_df = pd.DataFrame()
rankable_ai_evidence_meta: dict = {}
conservation_site_df = pd.DataFrame()
conservation_site_meta: dict = {}
residue_evidence_consensus_df = pd.DataFrame()
pocket_consensus_coverage_df = pd.DataFrame()
benchmark_reference_df = pd.DataFrame()
benchmark_reference_meta: dict = {}
benchmark_reference_candidate_df = pd.DataFrame()
benchmark_reference_candidate_meta: dict = {}
benchmark_reference_import_summary_df = pd.DataFrame()
benchmark_reference_candidate_review_queue_df = pd.DataFrame()
benchmark_reference_candidate_review_checklist_markdown = ""
benchmark_reference_candidate_review_decision_template_df = pd.DataFrame()
benchmark_reference_candidate_review_decision_df = pd.DataFrame()
benchmark_reference_candidate_review_decision_meta: dict = {}
benchmark_reference_candidate_review_decision_validation_df = pd.DataFrame()
benchmark_reference_candidate_review_outcome_df = pd.DataFrame()
benchmark_reference_candidate_accepted_df = pd.DataFrame()
benchmark_reference_is_provisional = False
benchmark_reference_is_reviewed_candidate = False
benchmark_reference_source_mode = ""
benchmark_reference_source_audit_df = pd.DataFrame()
benchmark_reference_source_audit_summary_df = pd.DataFrame()
benchmark_reference_source_audit_action_queue_df = pd.DataFrame()
benchmark_reference_source_audit_case_summary_df = pd.DataFrame()
benchmark_reference_source_audit_case_summary_blocked_cases = 0
benchmark_reference_source_audit_case_summary_review_cases = 0
benchmark_reference_source_audit_case_checklist_markdown = ""
benchmark_reference_source_audit_case_decision_template_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_meta: dict = {}
benchmark_reference_source_audit_case_decision_validation_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_outcome_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_outcome_summary_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_outcome_summary_status = ""
benchmark_reference_source_audit_case_decision_outcome_summary_open_cases = 0
benchmark_reference_source_audit_case_decision_closure_queue_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_closure_checklist_markdown = ""
benchmark_reference_source_audit_case_decision_readiness_impact_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_readiness_impact_summary_df = pd.DataFrame()
benchmark_reference_source_audit_case_decision_readiness_impact_summary_status = ""
benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases = 0
benchmark_reference_source_audit_checklist_markdown = ""
pocket_benchmark_reference_quality_issue_df = pd.DataFrame()
pocket_benchmark_reference_quality_summary_df = pd.DataFrame()
pocket_benchmark_reference_quality_checklist_markdown = ""
pocket_benchmark_reference_structure_validation_df = pd.DataFrame()
pocket_benchmark_reference_structure_validation_summary_df = pd.DataFrame()
pocket_benchmark_reference_structure_validation_checklist_markdown = ""
pocket_benchmark_reference_readiness_queue_df = pd.DataFrame()
pocket_benchmark_reference_readiness_summary_df = pd.DataFrame()
pocket_benchmark_reference_readiness_case_summary_df = pd.DataFrame()
pocket_benchmark_reference_readiness_checklist_markdown = ""
pocket_benchmark_interpretation_df = pd.DataFrame()
pocket_benchmark_case_interpretation_df = pd.DataFrame()
pocket_benchmark_case_interpretation_matrix_df = pd.DataFrame()
pocket_benchmark_case_interpretation_matrix_summary_df = pd.DataFrame()
pocket_benchmark_case_interpretation_matrix_queue_df = pd.DataFrame()
pocket_benchmark_dataset_interpretation_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df = pd.DataFrame()
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown = ""
pocket_benchmark_dataset_interpretation_queue_df = pd.DataFrame()
pocket_benchmark_dataset_interpretation_checklist_markdown = ""
pocket_benchmark_dataset_interpretation_report_markdown = ""
pocket_benchmark_summary_df = pd.DataFrame()
pocket_benchmark_details_df = pd.DataFrame()
pocket_benchmark_case_summary_df = pd.DataFrame()
pocket_benchmark_dataset_summary_df = pd.DataFrame()
pocket_benchmark_variant_comparison_df = pd.DataFrame()
pocket_benchmark_variant_case_comparison_df = pd.DataFrame()
pocket_benchmark_variant_dataset_comparison_df = pd.DataFrame()
pocket_benchmark_variant_detail_comparison_df = pd.DataFrame()
pocket_benchmark_variant_remediation_df = pd.DataFrame()
pocket_benchmark_variant_remediation_summary_df = pd.DataFrame()
pocket_benchmark_variant_remediation_checklist_markdown = ""
consensus_rerank_suggestion_df = pd.DataFrame()
consensus_rerank_preview_df = pd.DataFrame()
consensus_rerank_policy_gate_df = pd.DataFrame()
consensus_rerank_action_queue_df = pd.DataFrame()
consensus_rerank_action_checklist_markdown = ""
consensus_rerank_apply_simulation_df = pd.DataFrame()
consensus_rerank_simulation_delta_df = pd.DataFrame()
consensus_rerank_precision_scorecard_df = pd.DataFrame()
consensus_rerank_precision_guardrail_df = pd.DataFrame()
consensus_rerank_precision_guardrail_report_markdown = ""
consensus_rerank_guardrail_artifact_manifest_df = pd.DataFrame()
consensus_rerank_guardrail_handoff_zip = b""
consensus_rerank_guardrail_bundle_verification_df = pd.DataFrame()
consensus_rerank_guardrail_bundle_verification_summary_df = pd.DataFrame()
consensus_rerank_guardrail_handoff_certificate_markdown = ""
consensus_rerank_release_decision_template_df = pd.DataFrame()
consensus_rerank_release_apply_plan_df = pd.DataFrame()
consensus_rerank_release_apply_report_markdown = ""
consensus_rerank_release_execution_template_df = pd.DataFrame()
consensus_rerank_release_execution_receipt_df = pd.DataFrame()
consensus_rerank_release_execution_receipt_meta: dict = {}
consensus_rerank_release_execution_validation_df = pd.DataFrame()
consensus_rerank_release_execution_summary_df = pd.DataFrame()
consensus_rerank_release_execution_report_markdown = ""
consensus_rerank_release_closure_certificate_markdown = ""
consensus_rerank_release_closure_ledger_df = pd.DataFrame()
consensus_rerank_release_closure_summary_df = pd.DataFrame()
consensus_rerank_release_closure_blocker_df = pd.DataFrame()
consensus_rerank_release_closure_remediation_checklist_markdown = ""
consensus_rerank_release_closure_detached_manifest_df = pd.DataFrame()
structure_pdb_id = str(extract_pdb_id_from_text(pdb_text) or "").strip().upper()
if (enable_uniprot_evidence and str(uniprot_accession or "").strip()) or (
    enable_mcsa_evidence and (str(uniprot_accession or "").strip() or str(enzyme_ec_number or "").strip())
):
    with st.spinner("Loading external functional-site evidence..."):
        external_site_df, external_site_meta = _load_external_evidence(
            str(uniprot_accession or "").strip(),
            str(uniprot_chain_hint or "").strip(),
            structure_pdb_id,
            pdb_text,
            str(enzyme_ec_number or "").strip(),
            bool(enable_uniprot_evidence),
            bool(enable_mcsa_evidence),
        )
literature_manual_text = _read_uploaded_text(uploaded_literature) if uploaded_literature is not None else ""
if bool(enable_literature_evidence) or literature_manual_text.strip():
    with st.spinner("Loading literature residue evidence..."):
        literature_site_df, literature_site_meta = _load_literature_evidence(
            str(literature_query or "").strip(),
            literature_manual_text,
            str(uniprot_accession or "").strip(),
            str(enzyme_ec_number or "").strip(),
            structure_pdb_id,
            str(literature_protein_name or "").strip(),
            str(uniprot_chain_hint or "").strip(),
            pdb_text,
            int(literature_max_articles),
            bool(enable_literature_evidence),
            bool(enable_europepmc_evidence),
            bool(include_europepmc_fulltext),
            int(literature_max_fulltext),
            bool(literature_assume_structure_numbering),
        )
    if not literature_site_df.empty:
        external_site_df = merge_external_evidence_tables(external_site_df, literature_site_df)
        counts = _external_evidence_counts(external_site_df)
        source_values = []
        if str(external_site_meta.get("sources") or "").strip():
            source_values.extend(str(external_site_meta.get("sources")).split(","))
        source_values.append("literature")
        external_site_meta = {
            **external_site_meta,
            "status": "ok",
            "sources": ",".join(dict.fromkeys(source.strip() for source in source_values if source.strip())),
            "evidence_rows": str(counts["rows"]),
            "exact_rows": str(counts["exact"]),
            "weak_rows": str(counts["weak"]),
            "literature": literature_site_meta,
        }
if bool(enable_literature_evidence) or literature_manual_text.strip():
    if literature_site_df.empty:
        st.sidebar.caption("Literature residue mining did not produce usable high-confidence residues.")
    else:
        st.sidebar.caption(
            f"Literature evidence: {len(literature_site_df)} rows / "
            f"status {literature_site_meta.get('status') or '-'} / "
            f"query {literature_site_meta.get('query') or '-'}"
        )
if bool(enable_ai_evidence):
    ai_source_text = str(ai_context_text or literature_manual_text or "").strip()
    reference_evidence_before_ai_df = external_site_df.copy()
    with st.spinner("Loading AI residue evidence..."):
        if str(ai_payload_text or "").strip():
            ai_evidence_df, ai_evidence_meta = parse_ai_residue_evidence_payload(
                str(ai_payload_text or ""),
                chain_hint=str(uniprot_chain_hint or "").strip(),
                min_confidence=float(ai_min_confidence),
                assume_structure_numbering=bool(ai_assume_structure_numbering),
                pdb_text=pdb_text,
            )
        elif ai_source_text.strip():
            ai_evidence_df, ai_evidence_meta = fetch_ai_residue_evidence(
                ai_source_text,
                api_url=str(ai_api_url or "").strip(),
                api_key=str(ai_api_key or "").strip(),
                model=str(ai_model or "").strip(),
                chain_hint=str(uniprot_chain_hint or "").strip(),
                protein_name=str(literature_protein_name or "").strip(),
                accession=str(uniprot_accession or "").strip(),
                pdb_id=structure_pdb_id,
                ec_number=str(enzyme_ec_number or "").strip(),
                triage_context="",
                min_confidence=float(ai_min_confidence),
                assume_structure_numbering=bool(ai_assume_structure_numbering),
                pdb_text=pdb_text,
            )
        else:
            ai_evidence_df, ai_evidence_meta = pd.DataFrame(), {
                "status": "empty-input",
                "evidence_rows": "0",
                "message": "Paste source text, upload literature text, paste AI JSON, or configure an AI API call.",
            }
    ai_evidence_audit_df = build_ai_evidence_audit_table(ai_evidence_df, reference_evidence_before_ai_df)
    rankable_ai_evidence_before_review_df, _rankable_ai_evidence_before_review_meta = filter_ai_evidence_for_ranking(
        ai_evidence_df,
        ai_evidence_audit_df,
        allow_review=bool(ai_allow_review_ranking),
    )
    if uploaded_ai_review_decisions is not None:
        ai_review_decision_text = _read_uploaded_text(uploaded_ai_review_decisions)
        if ai_review_decision_text.strip():
            ai_review_decision_df, ai_review_decision_meta = parse_ai_review_decision_table(ai_review_decision_text)
            ai_review_decision_validation_df = build_ai_review_decision_validation_table(
                ai_review_decision_df,
                ai_evidence_audit_df,
            )
            decisions_for_apply_df = ai_review_decision_df
            if not ai_review_decision_validation_df.empty and "issue_flags" in ai_review_decision_validation_df.columns:
                blocked_duplicate_rows = ai_review_decision_validation_df[
                    ai_review_decision_validation_df["issue_flags"].astype(str).str.contains("conflicting-duplicate", case=False, na=False)
                ]
                if not blocked_duplicate_rows.empty and "row_index" in blocked_duplicate_rows.columns:
                    blocked_row_numbers = set(pd.to_numeric(blocked_duplicate_rows["row_index"], errors="coerce").dropna().astype(int).tolist())
                    decisions_for_apply_df = ai_review_decision_df.reset_index(drop=True).iloc[
                        [index for index in range(len(ai_review_decision_df)) if index + 1 not in blocked_row_numbers]
                    ].copy()
            ai_evidence_audit_df, ai_review_decision_apply_meta = apply_ai_review_decisions_to_audit(
                ai_evidence_audit_df,
                decisions_for_apply_df,
            )
            ai_review_decision_outcome_df = build_ai_review_decision_outcome_table(
                ai_review_decision_df,
                ai_evidence_audit_df,
            )
            ai_review_decision_meta = {
                **ai_review_decision_meta,
                "apply_status": str(ai_review_decision_apply_meta.get("status") or ""),
                "applied_rows": str(ai_review_decision_apply_meta.get("applied_rows") or "0"),
                "accepted_rows": str(ai_review_decision_apply_meta.get("accepted_rows") or "0"),
                "rejected_rows": str(ai_review_decision_apply_meta.get("rejected_rows") or "0"),
                "review_rows_after_apply": str(ai_review_decision_apply_meta.get("review_rows") or "0"),
                "conflict_blocked_rows": str(ai_review_decision_apply_meta.get("conflict_blocked_rows") or "0"),
                "validation_rows": str(len(ai_review_decision_validation_df)),
                "validation_blocked_rows": str(
                    int((ai_review_decision_validation_df["validation_status"].astype(str) == "blocked").sum())
                    if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns
                    else 0
                ),
                "outcome_rows": str(len(ai_review_decision_outcome_df)),
            }
    ai_evidence_review_queue_df = build_ai_evidence_review_queue(ai_evidence_audit_df)
    ai_review_decision_template_df = build_ai_review_decision_template(ai_evidence_review_queue_df)
    rankable_ai_evidence_df, rankable_ai_evidence_meta = filter_ai_evidence_for_ranking(
        ai_evidence_df,
        ai_evidence_audit_df,
        allow_review=bool(ai_allow_review_ranking),
    )
    ai_review_round_summary_df = build_ai_review_round_summary(
        ai_review_decision_df,
        ai_review_decision_validation_df,
        ai_review_decision_outcome_df,
        rankable_ai_evidence_df,
    )
    if not ai_review_decision_df.empty:
        ai_review_ranking_delta_df = build_ai_review_ranking_delta(
            rankable_ai_evidence_before_review_df,
            rankable_ai_evidence_df,
        )
        ai_review_round_report_markdown = build_ai_review_round_report_markdown(
            ai_review_round_summary_df,
            ai_review_decision_validation_df,
            ai_review_decision_outcome_df,
            ai_review_ranking_delta_df,
        )
    ai_review_artifact_manifest_df = build_ai_review_artifact_manifest(
        review_queue_df=ai_evidence_review_queue_df,
        decision_template_df=ai_review_decision_template_df,
        normalized_decision_df=ai_review_decision_df,
        validation_df=ai_review_decision_validation_df,
        round_summary_df=ai_review_round_summary_df,
        ranking_delta_df=ai_review_ranking_delta_df,
        outcome_df=ai_review_decision_outcome_df,
        round_report_markdown=ai_review_round_report_markdown,
    )
    if not ai_review_artifact_manifest_df.empty:
        ai_review_bundle_readme_markdown = build_ai_review_bundle_readme_markdown(ai_review_artifact_manifest_df)
        ai_review_artifact_manifest_df = build_ai_review_artifact_manifest(
            review_queue_df=ai_evidence_review_queue_df,
            decision_template_df=ai_review_decision_template_df,
            normalized_decision_df=ai_review_decision_df,
            validation_df=ai_review_decision_validation_df,
            round_summary_df=ai_review_round_summary_df,
            ranking_delta_df=ai_review_ranking_delta_df,
            outcome_df=ai_review_decision_outcome_df,
            round_report_markdown=ai_review_round_report_markdown,
            bundle_readme_markdown=ai_review_bundle_readme_markdown,
        )
    ai_review_artifact_bundle_zip = build_ai_review_artifact_bundle_zip(
        review_queue_df=ai_evidence_review_queue_df,
        decision_template_df=ai_review_decision_template_df,
        normalized_decision_df=ai_review_decision_df,
        validation_df=ai_review_decision_validation_df,
        round_summary_df=ai_review_round_summary_df,
        ranking_delta_df=ai_review_ranking_delta_df,
        outcome_df=ai_review_decision_outcome_df,
        artifact_manifest_df=ai_review_artifact_manifest_df,
        round_report_markdown=ai_review_round_report_markdown,
        bundle_readme_markdown=ai_review_bundle_readme_markdown,
    )
    ai_review_bundle_verification_df = verify_ai_review_artifact_bundle_zip(
        ai_review_artifact_bundle_zip,
        ai_review_artifact_manifest_df,
    )
    ai_review_bundle_verification_summary_df = build_ai_review_bundle_verification_summary(
        ai_review_bundle_verification_df
    )
    ai_review_bundle_certificate_markdown = build_ai_review_bundle_certificate_markdown(
        ai_review_artifact_bundle_zip,
        ai_review_bundle_verification_summary_df,
        ai_review_artifact_manifest_df,
    )
    if not rankable_ai_evidence_df.empty:
        external_site_df = merge_external_evidence_tables(external_site_df, rankable_ai_evidence_df)
        counts = _external_evidence_counts(external_site_df)
        source_values = []
        if str(external_site_meta.get("sources") or "").strip():
            source_values.extend(str(external_site_meta.get("sources")).split(","))
        source_values.append("AI")
        external_site_meta = {
            **external_site_meta,
            "status": "ok",
            "sources": ",".join(dict.fromkeys(source.strip() for source in source_values if source.strip())),
            "evidence_rows": str(counts["rows"]),
            "exact_rows": str(counts["exact"]),
            "weak_rows": str(counts["weak"]),
            "ai": ai_evidence_meta,
            "ai_ranking": rankable_ai_evidence_meta,
        }
    if ai_evidence_df.empty:
        st.sidebar.caption(f"AI evidence assistant: {ai_evidence_meta.get('status') or 'empty'}")
    else:
        audit_status_text = (
            ", ".join(f"{status}:{count}" for status, count in ai_evidence_audit_df["audit_status"].astype(str).value_counts().to_dict().items())
            if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns
            else "none"
        )
        st.sidebar.caption(
            f"AI evidence: {len(ai_evidence_df)} rows / status {ai_evidence_meta.get('status') or '-'} / "
            f"ranked {len(rankable_ai_evidence_df)} / manual review {ai_evidence_meta.get('manual_review_rows') or '0'} / audit {audit_status_text}"
        )
    if ai_review_decision_meta:
        st.sidebar.caption(
            f"AI review decisions: {len(ai_review_decision_df)} rows / "
            f"status {ai_review_decision_meta.get('status') or '-'} / "
            f"applied {ai_review_decision_meta.get('applied_rows') or '0'} / "
            f"accepted {ai_review_decision_meta.get('accepted_rows') or '0'} / "
            f"rejected {ai_review_decision_meta.get('rejected_rows') or '0'}"
        )
        if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns:
            validation_text = ", ".join(
                f"{status}:{count}"
                for status, count in ai_review_decision_validation_df["validation_status"].astype(str).value_counts().to_dict().items()
            )
            st.sidebar.caption(f"AI review validation: {validation_text}")
        if not ai_review_decision_outcome_df.empty and "applied_status" in ai_review_decision_outcome_df.columns:
            outcome_text = ", ".join(
                f"{status}:{count}"
                for status, count in ai_review_decision_outcome_df["applied_status"].astype(str).value_counts().to_dict().items()
            )
            st.sidebar.caption(f"AI review outcomes: {outcome_text}")
        if not ai_review_round_summary_df.empty:
            summary_row = ai_review_round_summary_df.iloc[0]
            st.sidebar.caption(
                f"AI review round: {summary_row.get('review_round_status') or '-'} / "
                f"rankable {summary_row.get('rankable_after_review_rows') or 0}"
            )
        if not ai_review_ranking_delta_df.empty:
            delta_row = ai_review_ranking_delta_df.iloc[0]
            st.sidebar.caption(
                f"AI review ranking delta: {delta_row.get('review_effect_status') or '-'} / "
                f"+{delta_row.get('promoted_rows') or 0} / -{delta_row.get('removed_rows') or 0}"
            )
        if not ai_review_artifact_manifest_df.empty:
            st.sidebar.caption(f"AI review artifact manifest: {len(ai_review_artifact_manifest_df)} files")
        if ai_review_artifact_bundle_zip:
            st.sidebar.caption("AI review artifact bundle: ready")
        if not ai_review_bundle_verification_df.empty and "verification_status" in ai_review_bundle_verification_df.columns:
            failed_verification = int((ai_review_bundle_verification_df["verification_status"].astype(str) != "verified").sum())
            st.sidebar.caption(f"AI review bundle verification: {len(ai_review_bundle_verification_df)} files / failed {failed_verification}")
        if not ai_review_bundle_verification_summary_df.empty:
            verify_row = ai_review_bundle_verification_summary_df.iloc[0]
            st.sidebar.caption(
                f"AI review bundle verification summary: {verify_row.get('verification_status') or '-'} / "
                f"failed {verify_row.get('failed_files') or 0}"
            )
        if ai_review_bundle_certificate_markdown:
            st.sidebar.caption("AI review bundle handoff certificate: ready")
if not external_site_df.empty:
    benchmark_reference_candidate_df, benchmark_reference_candidate_meta = build_pocket_benchmark_reference_from_external_evidence(
        external_site_df,
        default_benchmark_id=str(structure_pdb_id or uniprot_accession or enzyme_ec_number or "current-structure").strip(),
        source_hint=str(external_site_meta.get("sources") or "Loaded external evidence").strip() or "Loaded external evidence",
    )
    benchmark_reference_import_summary_df = build_pocket_benchmark_reference_import_summary(
        benchmark_reference_candidate_df,
        benchmark_reference_candidate_meta,
    )
    benchmark_reference_candidate_review_queue_df = build_pocket_benchmark_reference_candidate_review_queue(
        benchmark_reference_candidate_df
    )
    benchmark_reference_candidate_review_checklist_markdown = (
        build_pocket_benchmark_reference_candidate_review_checklist_markdown(
            benchmark_reference_candidate_review_queue_df
        )
    )
    benchmark_reference_candidate_review_decision_template_df = (
        build_pocket_benchmark_reference_candidate_review_decision_template(
            benchmark_reference_candidate_review_queue_df
        )
    )
    review_decision_text = (
        _read_uploaded_text(uploaded_benchmark_reference_candidate_review_decisions)
        if uploaded_benchmark_reference_candidate_review_decisions is not None
        else ""
    )
    if review_decision_text.strip():
        (
            benchmark_reference_candidate_review_decision_df,
            benchmark_reference_candidate_review_decision_meta,
        ) = parse_pocket_benchmark_reference_candidate_review_decision_table(review_decision_text)
        benchmark_reference_candidate_review_decision_validation_df = (
            build_pocket_benchmark_reference_candidate_review_decision_validation(
                benchmark_reference_candidate_review_decision_df,
                benchmark_reference_candidate_review_queue_df,
            )
        )
        benchmark_reference_candidate_review_outcome_df = (
            build_pocket_benchmark_reference_candidate_review_outcomes(
                benchmark_reference_candidate_review_queue_df,
                benchmark_reference_candidate_review_decision_df,
                benchmark_reference_candidate_review_decision_validation_df,
            )
        )
    else:
        benchmark_reference_candidate_review_outcome_df = (
            build_pocket_benchmark_reference_candidate_review_outcomes(
                benchmark_reference_candidate_review_queue_df,
                pd.DataFrame(),
            )
        )
    benchmark_reference_candidate_accepted_df = (
        build_pocket_benchmark_reference_candidate_accepted_reference(
            benchmark_reference_candidate_df,
            benchmark_reference_candidate_review_queue_df,
            benchmark_reference_candidate_review_outcome_df,
        )
    )
    if not benchmark_reference_import_summary_df.empty:
        import_row = benchmark_reference_import_summary_df.iloc[0]
        st.sidebar.caption(
            f"Benchmark reference candidate: {import_row.get('reference_rows') or 0} residues / "
            f"{import_row.get('import_status') or '-'}."
        )
    if not benchmark_reference_candidate_review_queue_df.empty:
        st.sidebar.caption(
            f"Benchmark reference candidate review: {len(benchmark_reference_candidate_review_queue_df)} actions."
        )
    if benchmark_reference_candidate_review_decision_meta:
        st.sidebar.caption(
            f"Benchmark reference candidate decisions: {benchmark_reference_candidate_review_decision_meta.get('decision_rows') or 0} rows / "
            f"status {benchmark_reference_candidate_review_decision_meta.get('status') or '-'}."
        )
if uploaded_conservation is not None:
    conservation_text = _read_uploaded_text(uploaded_conservation)
    if conservation_text.strip():
        with st.spinner("Loading conservation evidence..."):
            conservation_site_df, conservation_site_meta = parse_conservation_evidence_table(
                conservation_text,
                chain_hint=str(uniprot_chain_hint or "").strip(),
                source_hint=str(conservation_source_name or "").strip() or "ConSurf",
            )
benchmark_reference_loaded = False
benchmark_reference_uploaded = bool(benchmark_reference_text.strip())
if benchmark_reference_uploaded:
    benchmark_reference_df, benchmark_reference_meta = parse_benchmark_reference_table(
        benchmark_reference_text,
        source_hint=str(benchmark_source_name or "").strip() or "Curated catalytic benchmark",
    )
benchmark_reference_df, benchmark_reference_meta, benchmark_reference_selection = select_pocket_benchmark_reference_source(
    benchmark_reference_df,
    benchmark_reference_meta,
    curated_reference_uploaded=benchmark_reference_uploaded,
    external_candidate_df=benchmark_reference_candidate_df,
    external_candidate_meta=benchmark_reference_candidate_meta,
    accepted_candidate_df=benchmark_reference_candidate_accepted_df,
    prefer_reviewed_candidate=bool(use_reviewed_candidate_as_benchmark_reference),
    allow_provisional_candidate=bool(use_external_evidence_as_benchmark_reference),
)
benchmark_reference_loaded = bool(benchmark_reference_selection.get("loaded"))
benchmark_reference_is_provisional = bool(benchmark_reference_selection.get("is_provisional"))
benchmark_reference_is_reviewed_candidate = bool(benchmark_reference_selection.get("is_reviewed_candidate"))
benchmark_reference_source_mode = str(benchmark_reference_selection.get("source_mode") or "")
if benchmark_reference_selection.get("message"):
    st.sidebar.caption(str(benchmark_reference_selection.get("message")))
if benchmark_reference_loaded:
    if benchmark_reference_df.empty:
        st.sidebar.caption(
            f"Benchmark reference: no usable rows ({benchmark_reference_meta.get('reason') or benchmark_reference_meta.get('status') or '-'})."
        )
    else:
        benchmark_reference_source_audit_df = build_pocket_benchmark_reference_source_audit(
            benchmark_reference_df,
            source_mode=benchmark_reference_source_mode,
            is_provisional=benchmark_reference_is_provisional,
            is_reviewed_candidate=benchmark_reference_is_reviewed_candidate,
        )
        benchmark_reference_source_audit_summary_df = build_pocket_benchmark_reference_source_audit_summary(
            benchmark_reference_source_audit_df
        )
        benchmark_reference_source_audit_action_queue_df = build_pocket_benchmark_reference_source_audit_action_queue(
            benchmark_reference_source_audit_df
        )
        benchmark_reference_source_audit_case_summary_df = build_pocket_benchmark_reference_source_audit_case_summary(
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_action_queue_df,
        )
        (
            benchmark_reference_source_audit_case_summary_blocked_cases,
            benchmark_reference_source_audit_case_summary_review_cases,
        ) = _source_audit_case_summary_counts(benchmark_reference_source_audit_case_summary_df)
        benchmark_reference_source_audit_case_checklist_markdown = (
            build_pocket_benchmark_reference_source_audit_case_checklist_markdown(
                benchmark_reference_source_audit_case_summary_df,
                benchmark_reference_source_audit_action_queue_df,
            )
        )
        benchmark_reference_source_audit_case_decision_template_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_template(
                benchmark_reference_source_audit_case_summary_df
            )
        )
        source_audit_case_decision_text = (
            _read_uploaded_text(uploaded_benchmark_reference_source_audit_case_decisions)
            if uploaded_benchmark_reference_source_audit_case_decisions is not None
            else ""
        )
        if source_audit_case_decision_text.strip():
            (
                benchmark_reference_source_audit_case_decision_df,
                benchmark_reference_source_audit_case_decision_meta,
            ) = parse_pocket_benchmark_reference_source_audit_case_decision_table(
                source_audit_case_decision_text
            )
            benchmark_reference_source_audit_case_decision_validation_df = (
                build_pocket_benchmark_reference_source_audit_case_decision_validation(
                    benchmark_reference_source_audit_case_decision_df,
                    benchmark_reference_source_audit_case_summary_df,
                )
            )
        benchmark_reference_source_audit_case_decision_outcome_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_outcomes(
                benchmark_reference_source_audit_case_summary_df,
                benchmark_reference_source_audit_case_decision_df,
                benchmark_reference_source_audit_case_decision_validation_df,
            )
        )
        benchmark_reference_source_audit_case_decision_outcome_summary_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_outcome_summary(
                benchmark_reference_source_audit_case_decision_outcome_df
            )
        )
        if not benchmark_reference_source_audit_case_decision_outcome_summary_df.empty:
            _source_audit_outcome_summary_row = benchmark_reference_source_audit_case_decision_outcome_summary_df.iloc[0]
            benchmark_reference_source_audit_case_decision_outcome_summary_status = str(
                _source_audit_outcome_summary_row.get("closure_status") or ""
            )
            benchmark_reference_source_audit_case_decision_outcome_summary_open_cases = int(
                _source_audit_outcome_summary_row.get("open_actionable_case_count") or 0
            )
        benchmark_reference_source_audit_case_decision_closure_queue_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_closure_queue(
                benchmark_reference_source_audit_case_decision_outcome_df
            )
        )
        benchmark_reference_source_audit_case_decision_closure_checklist_markdown = (
            build_pocket_benchmark_reference_source_audit_case_decision_closure_checklist_markdown(
                benchmark_reference_source_audit_case_decision_outcome_summary_df,
                benchmark_reference_source_audit_case_decision_outcome_df,
            )
        )
        benchmark_reference_source_audit_case_decision_readiness_impact_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact(
                benchmark_reference_source_audit_df,
                benchmark_reference_source_audit_case_decision_outcome_df,
            )
        )
        benchmark_reference_source_audit_case_decision_readiness_impact_summary_df = (
            build_pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary(
                benchmark_reference_source_audit_case_decision_readiness_impact_df
            )
        )
        if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty:
            _source_audit_readiness_impact_summary_row = benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.iloc[0]
            benchmark_reference_source_audit_case_decision_readiness_impact_summary_status = str(
                _source_audit_readiness_impact_summary_row.get("readiness_impact_status") or ""
            )
            benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases = int(
                _source_audit_readiness_impact_summary_row.get("open_after_decision_cases") or 0
            )
        benchmark_reference_source_audit_checklist_markdown = build_pocket_benchmark_reference_source_audit_checklist_markdown(
            benchmark_reference_source_audit_summary_df,
            benchmark_reference_source_audit_df,
        )
        if benchmark_reference_is_provisional:
            st.sidebar.caption(
                "Benchmark reference warning: this candidate can overlap with detection inputs; curate separately before accuracy claims."
            )
        pocket_benchmark_reference_quality_issue_df = build_pocket_benchmark_reference_quality_issues(benchmark_reference_df)
        pocket_benchmark_reference_quality_summary_df = build_pocket_benchmark_reference_quality_summary(
            pocket_benchmark_reference_quality_issue_df
        )
        pocket_benchmark_reference_quality_checklist_markdown = build_pocket_benchmark_reference_quality_checklist_markdown(
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_quality_summary_df,
        )
        pocket_benchmark_reference_structure_validation_df = build_pocket_benchmark_reference_structure_validation(
            benchmark_reference_df,
            atom_df,
        )
        pocket_benchmark_reference_structure_validation_summary_df = build_pocket_benchmark_reference_structure_validation_summary(
            pocket_benchmark_reference_structure_validation_df
        )
        pocket_benchmark_reference_structure_validation_checklist_markdown = (
            build_pocket_benchmark_reference_structure_validation_checklist_markdown(
                pocket_benchmark_reference_structure_validation_df,
                pocket_benchmark_reference_structure_validation_summary_df,
            )
        )
        pocket_benchmark_reference_readiness_queue_df = build_pocket_benchmark_reference_readiness_queue(
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_structure_validation_df,
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_case_decision_outcome_df,
        )
        pocket_benchmark_reference_readiness_summary_df = build_pocket_benchmark_reference_readiness_summary(
            benchmark_reference_df,
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_structure_validation_df,
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_case_decision_outcome_df,
        )
        pocket_benchmark_reference_readiness_case_summary_df = build_pocket_benchmark_reference_readiness_case_summary(
            benchmark_reference_df,
            pocket_benchmark_reference_quality_issue_df,
            pocket_benchmark_reference_structure_validation_df,
            benchmark_reference_source_audit_df,
            benchmark_reference_source_audit_case_decision_outcome_df,
        )
        pocket_benchmark_reference_readiness_checklist_markdown = build_pocket_benchmark_reference_readiness_checklist_markdown(
            pocket_benchmark_reference_readiness_queue_df,
            pocket_benchmark_reference_readiness_summary_df,
        )
        p1_quality_issues = (
            int(pocket_benchmark_reference_quality_issue_df["severity"].astype(str).isin(["P0", "P1"]).sum())
            if not pocket_benchmark_reference_quality_issue_df.empty and "severity" in pocket_benchmark_reference_quality_issue_df.columns
            else 0
        )
        p1_structure_issues = (
            int(pocket_benchmark_reference_structure_validation_df["severity"].astype(str).isin(["P0", "P1"]).sum())
            if not pocket_benchmark_reference_structure_validation_df.empty and "severity" in pocket_benchmark_reference_structure_validation_df.columns
            else 0
        )
        st.sidebar.caption(
            f"Benchmark reference: {len(benchmark_reference_df)} residues / "
            f"chain-specific {benchmark_reference_meta.get('chain_specific_rows') or 0} / "
            f"wildcard {benchmark_reference_meta.get('wildcard_chain_rows') or 0}."
        )
        st.sidebar.caption(
            f"Benchmark reference curation: {len(pocket_benchmark_reference_quality_issue_df)} issues / P0-P1 {p1_quality_issues}."
        )
        st.sidebar.caption(
            f"Benchmark reference structure validation: {len(pocket_benchmark_reference_structure_validation_df)} issues / P0-P1 {p1_structure_issues}."
        )
        if not pocket_benchmark_reference_readiness_summary_df.empty:
            readiness_row = pocket_benchmark_reference_readiness_summary_df.iloc[0]
            st.sidebar.caption(
                f"Benchmark readiness: {readiness_row.get('readiness_status') or '-'} / "
                f"blockers {readiness_row.get('p0_p1_issue_count') or 0} / review {readiness_row.get('p2_issue_count') or 0}."
            )

residue_evidence_consensus_df = build_residue_evidence_consensus(
    external_site_df,
    ai_evidence_df=ai_evidence_df,
    ai_audit_df=ai_evidence_audit_df,
    rankable_ai_evidence_df=rankable_ai_evidence_df,
    conservation_df=conservation_site_df,
)

auto_pocket_df = detect_auto_pocket_table(
    pdb_text,
    hotspot_residues=hotspot_residues,
    external_site_df=external_site_df,
    conservation_site_df=conservation_site_df,
    adaptive_profile=auto_adaptive_profile,
    prefer_kvfinder=auto_use_kvfinder,
    prefer_p2rank=auto_use_p2rank,
    prefer_ligand=auto_detection_mode == "auto",
    enable_external_evidence_route=auto_external_evidence_route,
    external_evidence_min_support=external_route_min_support,
    external_evidence_min_confidence=external_route_min_confidence,
    external_evidence_min_mapping_quality=external_route_min_quality,
    external_evidence_radius=external_route_radius,
    contact_cutoff=auto_contact_cutoff,
    cluster_cutoff=auto_cluster_cutoff,
    ligand_radius=auto_ligand_radius,
    top_fraction=auto_candidate_fraction,
    min_candidates=3,
    max_candidates=auto_max_candidates,
    max_pockets=auto_max_pockets,
    p2rank_profile=p2rank_profile,
    p2rank_executable=str(p2rank_executable or "").strip() or None,
)
auto_detection_meta = get_pocket_detection_metadata(auto_pocket_df)
auto_detection_diag_df = build_pocket_detection_diagnostics_table(auto_pocket_df)
auto_detection_summary = summarize_pocket_detection_metadata(auto_detection_meta)
uploaded_pocket_df = _normalize_pocket_table(uploaded_pocket_df, "uploaded")
auto_pocket_df = _normalize_pocket_table(auto_pocket_df, "auto")
auto_pocket_summary = build_pocket_summary(auto_pocket_df, hotspot_df) if not auto_pocket_df.empty else pd.DataFrame()
uploaded_pocket_summary = build_pocket_summary(uploaded_pocket_df, hotspot_df) if not uploaded_pocket_df.empty else pd.DataFrame()
p2rank_ab_enabled = bool(enable_p2rank_ab and auto_use_p2rank and not auto_pocket_summary.empty)
auto_pocket_df_without_p2rank = pd.DataFrame()
auto_pocket_summary_without_p2rank = pd.DataFrame()
p2rank_ab_df = pd.DataFrame()
if p2rank_ab_enabled:
    auto_pocket_df_without_p2rank = detect_auto_pocket_table(
        pdb_text,
        hotspot_residues=hotspot_residues,
        external_site_df=external_site_df,
        conservation_site_df=conservation_site_df,
        adaptive_profile=auto_adaptive_profile,
        prefer_kvfinder=auto_use_kvfinder,
        prefer_p2rank=False,
        prefer_ligand=auto_detection_mode == "auto",
        enable_external_evidence_route=auto_external_evidence_route,
        external_evidence_min_support=external_route_min_support,
        external_evidence_min_confidence=external_route_min_confidence,
        external_evidence_min_mapping_quality=external_route_min_quality,
        external_evidence_radius=external_route_radius,
        contact_cutoff=auto_contact_cutoff,
        cluster_cutoff=auto_cluster_cutoff,
        ligand_radius=auto_ligand_radius,
        top_fraction=auto_candidate_fraction,
        min_candidates=3,
        max_candidates=auto_max_candidates,
        max_pockets=auto_max_pockets,
        p2rank_profile=p2rank_profile,
        p2rank_executable=str(p2rank_executable or "").strip() or None,
    )
    auto_pocket_df_without_p2rank = _normalize_pocket_table(auto_pocket_df_without_p2rank, "auto-no-p2rank")
    auto_pocket_summary_without_p2rank = (
        build_pocket_summary(auto_pocket_df_without_p2rank, hotspot_df)
        if not auto_pocket_df_without_p2rank.empty
        else pd.DataFrame()
    )
    p2rank_ab_df = compare_pocket_ranking_summaries(auto_pocket_summary_without_p2rank, auto_pocket_summary)
literature_ab_enabled = bool(enable_literature_ab and not literature_site_df.empty and not auto_pocket_summary.empty)
auto_pocket_df_without_literature = pd.DataFrame()
auto_pocket_summary_without_literature = pd.DataFrame()
literature_ab_df = pd.DataFrame()
if literature_ab_enabled:
    baseline_external_site_df = remove_literature_evidence(external_site_df)
    auto_pocket_df_without_literature = detect_auto_pocket_table(
        pdb_text,
        hotspot_residues=hotspot_residues,
        external_site_df=baseline_external_site_df,
        conservation_site_df=conservation_site_df,
        adaptive_profile=auto_adaptive_profile,
        prefer_kvfinder=auto_use_kvfinder,
        prefer_p2rank=auto_use_p2rank,
        prefer_ligand=auto_detection_mode == "auto",
        enable_external_evidence_route=auto_external_evidence_route,
        external_evidence_min_support=external_route_min_support,
        external_evidence_min_confidence=external_route_min_confidence,
        external_evidence_min_mapping_quality=external_route_min_quality,
        external_evidence_radius=external_route_radius,
        contact_cutoff=auto_contact_cutoff,
        cluster_cutoff=auto_cluster_cutoff,
        ligand_radius=auto_ligand_radius,
        top_fraction=auto_candidate_fraction,
        min_candidates=3,
        max_candidates=auto_max_candidates,
        max_pockets=auto_max_pockets,
        p2rank_profile=p2rank_profile,
        p2rank_executable=str(p2rank_executable or "").strip() or None,
    )
    auto_pocket_df_without_literature = _normalize_pocket_table(auto_pocket_df_without_literature, "auto-no-literature")
    auto_pocket_summary_without_literature = (
        build_pocket_summary(auto_pocket_df_without_literature, hotspot_df)
        if not auto_pocket_df_without_literature.empty
        else pd.DataFrame()
    )
    literature_ab_df = compare_pocket_ranking_summaries(auto_pocket_summary_without_literature, auto_pocket_summary)
evidence_route_ab_enabled = bool(
    enable_evidence_route_ab
    and auto_external_evidence_route
    and not external_site_df.empty
    and not auto_pocket_summary.empty
)
auto_pocket_df_without_evidence_route = pd.DataFrame()
auto_pocket_summary_without_evidence_route = pd.DataFrame()
evidence_route_ab_df = pd.DataFrame()
if evidence_route_ab_enabled:
    auto_pocket_df_without_evidence_route = detect_auto_pocket_table(
        pdb_text,
        hotspot_residues=hotspot_residues,
        external_site_df=external_site_df,
        conservation_site_df=conservation_site_df,
        adaptive_profile=auto_adaptive_profile,
        prefer_kvfinder=auto_use_kvfinder,
        prefer_p2rank=auto_use_p2rank,
        prefer_ligand=auto_detection_mode == "auto",
        enable_external_evidence_route=False,
        external_evidence_min_support=external_route_min_support,
        external_evidence_min_confidence=external_route_min_confidence,
        external_evidence_min_mapping_quality=external_route_min_quality,
        external_evidence_radius=external_route_radius,
        contact_cutoff=auto_contact_cutoff,
        cluster_cutoff=auto_cluster_cutoff,
        ligand_radius=auto_ligand_radius,
        top_fraction=auto_candidate_fraction,
        min_candidates=3,
        max_candidates=auto_max_candidates,
        max_pockets=auto_max_pockets,
        p2rank_profile=p2rank_profile,
        p2rank_executable=str(p2rank_executable or "").strip() or None,
    )
    auto_pocket_df_without_evidence_route = _normalize_pocket_table(auto_pocket_df_without_evidence_route, "auto-no-evidence-route")
    auto_pocket_summary_without_evidence_route = (
        build_pocket_summary(auto_pocket_df_without_evidence_route, hotspot_df)
        if not auto_pocket_df_without_evidence_route.empty
        else pd.DataFrame()
    )
    evidence_route_ab_df = compare_pocket_ranking_summaries(
        auto_pocket_summary_without_evidence_route,
        auto_pocket_summary,
    )
conservation_ab_enabled = bool(enable_conservation_ab and not conservation_site_df.empty and not auto_pocket_df.empty)
auto_pocket_summary_without_conservation = (
    build_pocket_summary_without_conservation_signal(auto_pocket_df, hotspot_df)
    if conservation_ab_enabled
    else pd.DataFrame()
)
conservation_ab_df = (
    compare_pocket_ranking_summaries(auto_pocket_summary_without_conservation, auto_pocket_summary)
    if conservation_ab_enabled and not auto_pocket_summary.empty
    else pd.DataFrame()
)

if not uploaded_annotation_df.empty:
    uploaded_annotation_df = uploaded_annotation_df.copy()
    uploaded_annotation_df["annotation_source"] = "uploaded"

inferred_annotation_df = build_inferred_interface_annotations(
    structure_energy_table if not structure_energy_table.empty else energy_table,
)

if (
    (enable_uniprot_evidence and str(uniprot_accession or "").strip())
    or enable_mcsa_evidence
    or enable_literature_evidence
    or not literature_site_df.empty
):
    if external_site_df.empty:
        st.sidebar.caption("未获取到可映射的外部功能位点证据，可能是编号不一致或网络不可用。")
    else:
        exact_rows = 0
        weak_rows = 0
        if "mapping_level" in external_site_df.columns:
            level_series = external_site_df["mapping_level"].astype(str).str.lower()
            exact_rows = int((level_series == "exact").sum())
            weak_rows = int((level_series == "weak").sum())
        mapping_status = str(external_site_meta.get("status") or external_site_meta.get("mapping_status") or "-")
        mapping_pdb = str(external_site_meta.get("pdb_id") or structure_pdb_id or "-")
        source_text = str(external_site_meta.get("sources") or "external")
        st.sidebar.caption(
            f"已加载 {source_text} 位点证据 {len(external_site_df)} 条（精确 {exact_rows} / 弱命中 {weak_rows}，PDB {mapping_pdb}，{mapping_status}）。"
        )
if uploaded_conservation is not None:
    if conservation_site_df.empty:
        st.sidebar.caption("Conservation evidence file did not produce usable residue rows.")
    else:
        st.sidebar.caption(
            f"Conservation import: {len(conservation_site_df)} rows / source {conservation_site_meta.get('source') or conservation_source_name} / "
            f"mean score {conservation_site_meta.get('score_mean') or '-'}"
        )
if not residue_evidence_consensus_df.empty:
    top_consensus = residue_evidence_consensus_df.iloc[0]
    st.sidebar.caption(
        f"Residue evidence consensus: {len(residue_evidence_consensus_df)} residues / "
        f"top {top_consensus.get('residue_anchor') or '-'} / {top_consensus.get('consensus_tier') or '-'}"
    )

pocket_source_options: list[str] = []
if not uploaded_pocket_df.empty:
    pocket_source_options.append("uploaded")
if not auto_pocket_df.empty:
    pocket_source_options.append("auto")
if not uploaded_pocket_df.empty and not auto_pocket_df.empty:
    pocket_source_options.append("combined")

annotation_source_options: list[str] = []
if not uploaded_annotation_df.empty:
    annotation_source_options.append("uploaded")
if not inferred_annotation_df.empty:
    annotation_source_options.append("inferred")
if not uploaded_annotation_df.empty and not inferred_annotation_df.empty:
    annotation_source_options.append("combined")

with st.sidebar:
    st.header("分析来源")
    if pocket_source_options:
        default_pocket_mode = "combined" if "combined" in pocket_source_options else pocket_source_options[0]
        effective_pocket_mode = st.radio(
            "界面分析使用的口袋来源",
            pocket_source_options,
            index=pocket_source_options.index(default_pocket_mode),
            format_func=lambda value: POCKET_SOURCE_LABELS.get(value, value),
        )
    else:
        effective_pocket_mode = "auto"
        st.caption("当前没有可用 Pocket 数据源。")

    if annotation_source_options:
        default_annotation_mode = "combined" if "combined" in annotation_source_options else annotation_source_options[0]
        effective_annotation_mode = st.radio(
            "界面分析使用的注释来源",
            annotation_source_options,
            index=annotation_source_options.index(default_annotation_mode),
            format_func=lambda value: ANNOTATION_SOURCE_LABELS.get(value, value),
        )
    else:
        effective_annotation_mode = "inferred"
        st.caption("当前没有可用界面注释来源。")

effective_pocket_df = _resolve_pocket_source(uploaded_pocket_df, auto_pocket_df, effective_pocket_mode)
effective_pocket_summary = build_pocket_summary(effective_pocket_df, hotspot_df) if not effective_pocket_df.empty else pd.DataFrame()
effective_pocket_residues = _residue_pairs(effective_pocket_df)
pocket_consensus_coverage_df = build_pocket_consensus_coverage(
    effective_pocket_df,
    residue_evidence_consensus_df,
)

effective_annotation_base_df = _resolve_annotation_source(
    uploaded_annotation_df,
    inferred_annotation_df,
    effective_annotation_mode,
)
enriched_annotations = enrich_interface_annotations(
    effective_annotation_base_df,
    pocket_residues=effective_pocket_residues,
    hotspot_residues=hotspot_residues,
)
interface_summary = build_interface_summary(enriched_annotations)
overlap_summary = build_interface_overlap_summary(
    enriched_annotations,
    pocket_residues=effective_pocket_residues,
    hotspot_residues=hotspot_residues,
)
joint_candidate_df = build_joint_candidate_table(
    effective_pocket_df,
    enriched_annotations,
    hotspot_df,
    external_site_df=external_site_df,
)
top_joint_candidate = joint_candidate_df.iloc[0] if not joint_candidate_df.empty else None
top_pocket_consensus_coverage = pocket_consensus_coverage_df.iloc[0] if not pocket_consensus_coverage_df.empty else None
pocket_benchmark_summary_df = build_pocket_benchmark_summary(
    benchmark_reference_df,
    effective_pocket_df,
    effective_pocket_summary,
    top_ns=(1, 3, 5),
) if not benchmark_reference_df.empty else pd.DataFrame()
pocket_benchmark_details_df = build_pocket_benchmark_details(
    benchmark_reference_df,
    effective_pocket_df,
    effective_pocket_summary,
    top_thresholds=(1, 3, 5),
) if not benchmark_reference_df.empty else pd.DataFrame()
pocket_benchmark_case_summary_df = build_pocket_benchmark_case_summary(
    benchmark_reference_df,
    effective_pocket_df,
    effective_pocket_summary,
    top_ns=(1, 3, 5),
) if not benchmark_reference_df.empty else pd.DataFrame()
pocket_benchmark_dataset_summary_df = (
    build_pocket_benchmark_dataset_summary(pocket_benchmark_case_summary_df)
    if not pocket_benchmark_case_summary_df.empty
    else pd.DataFrame()
)
top1_benchmark = (
    pocket_benchmark_summary_df[pocket_benchmark_summary_df["top_n"].astype(int) == 1].iloc[0]
    if not pocket_benchmark_summary_df.empty and "top_n" in pocket_benchmark_summary_df.columns and (pocket_benchmark_summary_df["top_n"].astype(int) == 1).any()
    else None
)
top3_benchmark = (
    pocket_benchmark_summary_df[pocket_benchmark_summary_df["top_n"].astype(int) == 3].iloc[0]
    if not pocket_benchmark_summary_df.empty and "top_n" in pocket_benchmark_summary_df.columns and (pocket_benchmark_summary_df["top_n"].astype(int) == 3).any()
    else None
)
pocket_benchmark_interpretation_df = build_pocket_benchmark_interpretation_summary(
    pocket_benchmark_summary_df,
    pocket_benchmark_reference_readiness_summary_df,
)
pocket_benchmark_case_interpretation_df = build_pocket_benchmark_case_interpretation_summary(
    pocket_benchmark_case_summary_df,
    pocket_benchmark_reference_readiness_case_summary_df,
)
pocket_benchmark_case_interpretation_matrix_df = build_pocket_benchmark_case_interpretation_matrix(
    pocket_benchmark_case_interpretation_df,
    top_ns=(1, 3, 5),
)
pocket_benchmark_case_interpretation_matrix_summary_df = (
    build_pocket_benchmark_case_interpretation_matrix_summary(
        pocket_benchmark_case_interpretation_matrix_df
    )
)
pocket_benchmark_case_interpretation_matrix_queue_df = (
    build_pocket_benchmark_case_interpretation_matrix_queue(
        pocket_benchmark_case_interpretation_matrix_df
    )
)
pocket_benchmark_dataset_interpretation_df = build_pocket_benchmark_dataset_interpretation(
    pocket_benchmark_case_interpretation_df
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact(
        pocket_benchmark_case_interpretation_df,
        benchmark_reference_source_audit_case_decision_readiness_impact_df,
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases(
        pocket_benchmark_case_interpretation_df,
        benchmark_reference_source_audit_case_decision_readiness_impact_df,
    )
)
pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown = (
    build_pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown(
        pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df
    )
)
pocket_benchmark_dataset_interpretation_queue_df = build_pocket_benchmark_dataset_interpretation_queue(
    pocket_benchmark_case_interpretation_df
)
pocket_benchmark_dataset_interpretation_checklist_markdown = (
    build_pocket_benchmark_dataset_interpretation_checklist_markdown(
        pocket_benchmark_dataset_interpretation_queue_df
    )
)
pocket_benchmark_dataset_interpretation_report_markdown = (
    build_pocket_benchmark_dataset_interpretation_report_markdown(
        pocket_benchmark_dataset_interpretation_df,
        pocket_benchmark_dataset_interpretation_queue_df,
        checklist_available=bool(pocket_benchmark_dataset_interpretation_checklist_markdown),
    )
)
benchmark_variants: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
if not benchmark_reference_df.empty:
    benchmark_variants.append(("current", effective_pocket_df, effective_pocket_summary))
    if p2rank_ab_enabled and not auto_pocket_df_without_p2rank.empty:
        benchmark_variants.append(("no-p2rank", auto_pocket_df_without_p2rank, auto_pocket_summary_without_p2rank))
    if literature_ab_enabled and not auto_pocket_df_without_literature.empty:
        benchmark_variants.append(("no-literature", auto_pocket_df_without_literature, auto_pocket_summary_without_literature))
    if evidence_route_ab_enabled and not auto_pocket_df_without_evidence_route.empty:
        benchmark_variants.append(("no-evidence-route", auto_pocket_df_without_evidence_route, auto_pocket_summary_without_evidence_route))
    if conservation_ab_enabled and not auto_pocket_summary_without_conservation.empty:
        benchmark_variants.append(("no-conservation-rerank", auto_pocket_df, auto_pocket_summary_without_conservation))
pocket_benchmark_variant_comparison_df = (
    build_pocket_benchmark_variant_comparison(
        benchmark_reference_df,
        benchmark_variants,
        reference_variant_label="current",
        top_ns=(1, 3, 5),
    )
    if len(benchmark_variants) > 1
    else pd.DataFrame()
)
pocket_benchmark_variant_case_comparison_df = (
    build_pocket_benchmark_variant_case_comparison(
        benchmark_reference_df,
        benchmark_variants,
        reference_variant_label="current",
        top_ns=(1, 3, 5),
    )
    if len(benchmark_variants) > 1
    else pd.DataFrame()
)
pocket_benchmark_variant_dataset_comparison_df = (
    build_pocket_benchmark_variant_dataset_comparison(pocket_benchmark_variant_case_comparison_df)
    if not pocket_benchmark_variant_case_comparison_df.empty
    else pd.DataFrame()
)
pocket_benchmark_variant_detail_comparison_df = (
    build_pocket_benchmark_variant_detail_comparison(
        benchmark_reference_df,
        benchmark_variants,
        reference_variant_label="current",
        top_thresholds=(1, 3, 5),
    )
    if len(benchmark_variants) > 1
    else pd.DataFrame()
)
pocket_benchmark_variant_remediation_df = build_pocket_benchmark_variant_remediation_queue(
    pocket_benchmark_variant_detail_comparison_df
)
pocket_benchmark_variant_remediation_summary_df = build_pocket_benchmark_variant_remediation_summary(
    pocket_benchmark_variant_remediation_df
)
pocket_benchmark_variant_remediation_checklist_markdown = build_pocket_benchmark_variant_remediation_checklist_markdown(
    pocket_benchmark_variant_remediation_df,
    pocket_benchmark_variant_remediation_summary_df,
)
pocket_decision_df = build_pocket_decision_table(
    effective_pocket_summary,
    joint_candidate_df,
    literature_ab_df=literature_ab_df,
    evidence_route_ab_df=evidence_route_ab_df,
    conservation_ab_df=conservation_ab_df,
)
consensus_rerank_suggestion_df = build_consensus_rerank_suggestion(
    pocket_decision_df,
    pocket_consensus_coverage_df,
)
consensus_rerank_preview_df = build_consensus_rerank_preview(
    pocket_decision_df,
    consensus_rerank_suggestion_df,
)
consensus_rerank_policy_gate_df = build_consensus_rerank_policy_gate(consensus_rerank_preview_df)
consensus_rerank_action_queue_df = build_consensus_rerank_action_queue(
    consensus_rerank_preview_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_action_checklist_markdown = build_consensus_rerank_action_checklist_markdown(
    consensus_rerank_action_queue_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_apply_simulation_df = build_consensus_rerank_apply_simulation(
    consensus_rerank_preview_df,
    consensus_rerank_action_queue_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_simulation_delta_df = build_consensus_rerank_simulation_delta(
    consensus_rerank_apply_simulation_df,
)
consensus_rerank_precision_scorecard_df = build_consensus_rerank_precision_scorecard(
    consensus_rerank_simulation_delta_df,
    consensus_rerank_apply_simulation_df,
    consensus_rerank_policy_gate_df,
)
consensus_rerank_precision_guardrail_df = build_consensus_rerank_precision_guardrail(
    consensus_rerank_precision_scorecard_df,
    consensus_rerank_policy_gate_df,
    consensus_rerank_action_queue_df,
)
consensus_rerank_precision_guardrail_report_markdown = build_consensus_rerank_precision_guardrail_report_markdown(
    consensus_rerank_precision_guardrail_df,
    consensus_rerank_precision_scorecard_df,
    consensus_rerank_action_queue_df,
    consensus_rerank_simulation_delta_df,
)
consensus_rerank_release_decision_template_df = build_consensus_rerank_release_decision_template(
    consensus_rerank_precision_guardrail_df,
    consensus_rerank_action_queue_df,
    consensus_rerank_simulation_delta_df,
)
if uploaded_consensus_rerank_release_decisions is not None:
    consensus_rerank_release_decision_df, consensus_rerank_release_decision_meta = parse_consensus_rerank_release_decision_table(
        _read_uploaded_text(uploaded_consensus_rerank_release_decisions)
    )
else:
    consensus_rerank_release_decision_df, consensus_rerank_release_decision_meta = pd.DataFrame(), {
        "status": "not-uploaded",
        "input_rows": "0",
        "decision_rows": "0",
        "skipped_rows": "0",
    }
consensus_rerank_release_decision_validation_df = validate_consensus_rerank_release_decisions(
    consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_template_df,
    consensus_rerank_precision_guardrail_df,
)
consensus_rerank_release_decision_summary_df = build_consensus_rerank_release_decision_summary(
    consensus_rerank_release_decision_validation_df,
    consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_template_df,
)
consensus_rerank_release_apply_plan_df = build_consensus_rerank_release_apply_plan(
    consensus_rerank_apply_simulation_df,
    consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_decision_validation_df,
)
consensus_rerank_release_apply_report_markdown = build_consensus_rerank_release_apply_report_markdown(
    consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_decision_summary_df,
)
consensus_rerank_release_execution_template_df = build_consensus_rerank_release_execution_template(
    consensus_rerank_release_apply_plan_df
)
if uploaded_consensus_rerank_release_execution_receipt is not None:
    consensus_rerank_release_execution_receipt_df, consensus_rerank_release_execution_receipt_meta = parse_consensus_rerank_release_execution_table(
        _read_uploaded_text(uploaded_consensus_rerank_release_execution_receipt)
    )
else:
    consensus_rerank_release_execution_receipt_df, consensus_rerank_release_execution_receipt_meta = pd.DataFrame(), {
        "status": "not-uploaded",
        "input_rows": "0",
        "receipt_rows": "0",
        "skipped_rows": "0",
    }
consensus_rerank_release_execution_validation_df = validate_consensus_rerank_release_execution_receipt(
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_template_df,
    consensus_rerank_release_apply_plan_df,
)
consensus_rerank_release_execution_summary_df = build_consensus_rerank_release_execution_summary(
    consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_template_df,
)
consensus_rerank_release_execution_report_markdown = build_consensus_rerank_release_execution_report_markdown(
    consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_receipt_df,
)
consensus_rerank_release_closure_certificate_markdown = build_consensus_rerank_release_closure_certificate_markdown(
    consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_report_markdown,
)
consensus_rerank_release_closure_ledger_df = build_consensus_rerank_release_closure_ledger(
    consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_report_markdown,
    consensus_rerank_release_closure_certificate_markdown,
)
consensus_rerank_guardrail_artifact_manifest_df = build_consensus_rerank_guardrail_artifact_manifest(
    consensus_rerank_suggestion_df=consensus_rerank_suggestion_df,
    consensus_rerank_preview_df=consensus_rerank_preview_df,
    consensus_rerank_policy_gate_df=consensus_rerank_policy_gate_df,
    consensus_rerank_action_queue_df=consensus_rerank_action_queue_df,
    consensus_rerank_action_checklist_markdown=consensus_rerank_action_checklist_markdown,
    consensus_rerank_apply_simulation_df=consensus_rerank_apply_simulation_df,
    consensus_rerank_simulation_delta_df=consensus_rerank_simulation_delta_df,
    consensus_rerank_precision_scorecard_df=consensus_rerank_precision_scorecard_df,
    consensus_rerank_precision_guardrail_df=consensus_rerank_precision_guardrail_df,
    consensus_rerank_precision_guardrail_report_markdown=consensus_rerank_precision_guardrail_report_markdown,
    consensus_rerank_release_decision_template_df=consensus_rerank_release_decision_template_df,
    consensus_rerank_release_decision_df=consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_validation_df=consensus_rerank_release_decision_validation_df,
    consensus_rerank_release_decision_summary_df=consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_apply_plan_df=consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_apply_report_markdown=consensus_rerank_release_apply_report_markdown,
    consensus_rerank_release_execution_template_df=consensus_rerank_release_execution_template_df,
    consensus_rerank_release_execution_receipt_df=consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_validation_df=consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_summary_df=consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_report_markdown=consensus_rerank_release_execution_report_markdown,
    consensus_rerank_release_closure_certificate_markdown=consensus_rerank_release_closure_certificate_markdown,
    consensus_rerank_release_closure_ledger_df=consensus_rerank_release_closure_ledger_df,
)
consensus_rerank_guardrail_handoff_zip = build_consensus_rerank_guardrail_handoff_zip(
    consensus_rerank_suggestion_df=consensus_rerank_suggestion_df,
    consensus_rerank_preview_df=consensus_rerank_preview_df,
    consensus_rerank_policy_gate_df=consensus_rerank_policy_gate_df,
    consensus_rerank_action_queue_df=consensus_rerank_action_queue_df,
    consensus_rerank_action_checklist_markdown=consensus_rerank_action_checklist_markdown,
    consensus_rerank_apply_simulation_df=consensus_rerank_apply_simulation_df,
    consensus_rerank_simulation_delta_df=consensus_rerank_simulation_delta_df,
    consensus_rerank_precision_scorecard_df=consensus_rerank_precision_scorecard_df,
    consensus_rerank_precision_guardrail_df=consensus_rerank_precision_guardrail_df,
    consensus_rerank_precision_guardrail_report_markdown=consensus_rerank_precision_guardrail_report_markdown,
    consensus_rerank_release_decision_template_df=consensus_rerank_release_decision_template_df,
    consensus_rerank_release_decision_df=consensus_rerank_release_decision_df,
    consensus_rerank_release_decision_validation_df=consensus_rerank_release_decision_validation_df,
    consensus_rerank_release_decision_summary_df=consensus_rerank_release_decision_summary_df,
    consensus_rerank_release_apply_plan_df=consensus_rerank_release_apply_plan_df,
    consensus_rerank_release_apply_report_markdown=consensus_rerank_release_apply_report_markdown,
    consensus_rerank_release_execution_template_df=consensus_rerank_release_execution_template_df,
    consensus_rerank_release_execution_receipt_df=consensus_rerank_release_execution_receipt_df,
    consensus_rerank_release_execution_validation_df=consensus_rerank_release_execution_validation_df,
    consensus_rerank_release_execution_summary_df=consensus_rerank_release_execution_summary_df,
    consensus_rerank_release_execution_report_markdown=consensus_rerank_release_execution_report_markdown,
    consensus_rerank_release_closure_certificate_markdown=consensus_rerank_release_closure_certificate_markdown,
    consensus_rerank_release_closure_ledger_df=consensus_rerank_release_closure_ledger_df,
    artifact_manifest_df=consensus_rerank_guardrail_artifact_manifest_df,
)
consensus_rerank_guardrail_bundle_verification_df = verify_consensus_rerank_guardrail_handoff_zip(
    consensus_rerank_guardrail_handoff_zip,
    consensus_rerank_guardrail_artifact_manifest_df,
)
consensus_rerank_guardrail_bundle_verification_summary_df = build_consensus_rerank_guardrail_bundle_verification_summary(
    consensus_rerank_guardrail_bundle_verification_df,
    consensus_rerank_guardrail_artifact_manifest_df,
)
consensus_rerank_release_closure_summary_df = build_consensus_rerank_release_closure_summary(
    consensus_rerank_release_closure_ledger_df,
    consensus_rerank_guardrail_bundle_verification_summary_df,
)
consensus_rerank_release_closure_blocker_df = build_consensus_rerank_release_closure_blocker_queue(
    consensus_rerank_release_closure_summary_df,
    consensus_rerank_release_closure_ledger_df,
    consensus_rerank_guardrail_bundle_verification_df,
    consensus_rerank_guardrail_bundle_verification_summary_df,
)
consensus_rerank_release_closure_remediation_checklist_markdown = build_consensus_rerank_release_closure_remediation_checklist_markdown(
    consensus_rerank_release_closure_blocker_df,
    consensus_rerank_release_closure_summary_df,
)
consensus_rerank_release_closure_detached_manifest_df = build_consensus_rerank_release_closure_detached_manifest(
    consensus_rerank_release_closure_summary_df,
    consensus_rerank_release_closure_blocker_df,
    consensus_rerank_release_closure_remediation_checklist_markdown,
)
consensus_rerank_guardrail_handoff_certificate_markdown = build_consensus_rerank_guardrail_handoff_certificate_markdown(
    consensus_rerank_guardrail_handoff_zip,
    consensus_rerank_guardrail_bundle_verification_summary_df,
    consensus_rerank_guardrail_artifact_manifest_df,
    consensus_rerank_precision_guardrail_df,
    consensus_rerank_release_decision_summary_df,
)
top_pocket_decision = pocket_decision_df.iloc[0] if not pocket_decision_df.empty else None
top_consensus_rerank_suggestion = consensus_rerank_suggestion_df.iloc[0] if not consensus_rerank_suggestion_df.empty else None
top_consensus_rerank_preview = consensus_rerank_preview_df.iloc[0] if not consensus_rerank_preview_df.empty else None
top_consensus_rerank_policy_gate = consensus_rerank_policy_gate_df.iloc[0] if not consensus_rerank_policy_gate_df.empty else None
top_consensus_rerank_action = consensus_rerank_action_queue_df.iloc[0] if not consensus_rerank_action_queue_df.empty else None
top_consensus_rerank_apply = consensus_rerank_apply_simulation_df.iloc[0] if not consensus_rerank_apply_simulation_df.empty else None
top_consensus_rerank_delta = consensus_rerank_simulation_delta_df.iloc[0] if not consensus_rerank_simulation_delta_df.empty else None
top_consensus_rerank_scorecard = consensus_rerank_precision_scorecard_df.iloc[0] if not consensus_rerank_precision_scorecard_df.empty else None
top_consensus_rerank_guardrail = consensus_rerank_precision_guardrail_df.iloc[0] if not consensus_rerank_precision_guardrail_df.empty else None
top_consensus_rerank_release_decision_summary = (
    consensus_rerank_release_decision_summary_df.iloc[0]
    if not consensus_rerank_release_decision_summary_df.empty
    else None
)
top_consensus_rerank_release_apply_plan = (
    consensus_rerank_release_apply_plan_df.iloc[0]
    if not consensus_rerank_release_apply_plan_df.empty
    else None
)
top_consensus_rerank_release_execution_summary = (
    consensus_rerank_release_execution_summary_df.iloc[0]
    if not consensus_rerank_release_execution_summary_df.empty
    else None
)
top_consensus_rerank_release_closure_summary = (
    consensus_rerank_release_closure_summary_df.iloc[0]
    if not consensus_rerank_release_closure_summary_df.empty
    else None
)
pocket_reliability_df = build_pocket_reliability_checklist(pocket_decision_df, max_pockets=3)
pocket_triage_df = build_pocket_precision_triage(pocket_decision_df, pocket_reliability_df, max_pockets=3)
ai_ranking_impact_df = build_ai_ranking_impact_summary(
    ai_evidence_df,
    rankable_ai_evidence_df,
    ai_evidence_audit_df,
    pocket_decision_df,
    pocket_triage_df,
)
ai_followup_plan_df = build_ai_followup_evidence_plan(
    pocket_decision_df,
    pocket_reliability_df,
    pocket_triage_df,
    protein_name=str(literature_protein_name or "").strip(),
    accession=str(uniprot_accession or "").strip(),
    pdb_id=structure_pdb_id,
    ec_number=str(enzyme_ec_number or "").strip(),
    max_pockets=3,
)
ai_followup_prompt_bundle = build_ai_followup_prompt_bundle(ai_followup_plan_df)
ai_review_checklist_markdown = build_ai_review_checklist_markdown(ai_evidence_review_queue_df)
top_precision_triage = pocket_triage_df.iloc[0] if not pocket_triage_df.empty else None
top_residue_consensus = residue_evidence_consensus_df.iloc[0] if not residue_evidence_consensus_df.empty else None
top_pocket_triage = None
if top_pocket_decision is not None and not pocket_reliability_df.empty:
    top_pocket_id = str(top_pocket_decision.get("pocket_id") or "")
    top_reliability_rows = pocket_reliability_df[pocket_reliability_df["pocket_id"].astype(str) == top_pocket_id]
    top_reliability_gaps = "; ".join(
        f"{row.check}: {row.status}"
        for row in top_reliability_rows.itertuples(index=False)
        if str(row.status) != "pass"
    )
    if not pocket_triage_df.empty:
        top_triage_rows = pocket_triage_df[pocket_triage_df["pocket_id"].astype(str) == top_pocket_id]
        if not top_triage_rows.empty:
            top_pocket_triage = top_triage_rows.iloc[0]
else:
    top_reliability_gaps = ""

try:
    protein_volume = estimate_protein_volume(pdb_text)
except Exception:
    protein_volume = None

summary = build_analysis_summary(energy_table)
analysis_text = explain_analysis(energy_table, hotspot_df, effective_pocket_summary)
try:
    stored_mmpbsa_text = mmpbsa_text or "结构估算（未上传 MMPBSA 文件）"
    top_pocket = effective_pocket_summary.iloc[0] if not effective_pocket_summary.empty else None
    set_analysis_state(
        pdb_text,
        stored_mmpbsa_text,
        atom_df,
        energy_df,
        energy_table,
        annotation_table=enriched_annotations,
        pocket_table=effective_pocket_df,
        pocket_summary=effective_pocket_summary,
        joint_candidate_table=joint_candidate_df,
    )
    append_history_record(
        {
            "generated_at": summary["generated_at"],
            "source_name": "口袋/界面专页",
            "energy_source_name": summary.get("energy_source") or energy_source or "未知",
            "residue_count": summary["residue_count"],
            "min_energy": summary["min_energy"],
            "max_energy": summary["max_energy"],
            "mean_energy": summary["mean_energy"],
            "lowest_residue": summary["lowest_residue"],
            "highest_residue": summary["highest_residue"],
            "valid_energy_count": summary["valid_energy_count"],
            "energy_coverage": summary["energy_coverage"],
            "protein_volume": protein_volume,
            "display_mode": "pocket-interface",
            "color_mode": "联合注释",
            "hotspot_count": int(len(hotspot_df)),
            "pocket_count": int(len(effective_pocket_df)),
            "annotation_rows": int(len(enriched_annotations)),
            **auto_detection_summary,
            "top_pocket_id": str(top_pocket.get("pocket_id")) if top_pocket is not None and pd.notna(top_pocket.get("pocket_id")) else None,
            "top_pocket_smart_rank_label": str(top_pocket.get("smart_rank_label")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_label")) else None,
            "top_pocket_smart_rank_score": float(top_pocket.get("smart_rank_score")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_score")) else None,
            "top_pocket_hotspot_count": int(top_pocket.get("hotspot_count")) if top_pocket is not None and pd.notna(top_pocket.get("hotspot_count")) else None,
            "top_pocket_detection_route": str(top_pocket.get("detection_route")) if top_pocket is not None and pd.notna(top_pocket.get("detection_route")) else None,
            "top_pocket_reason": str(top_pocket.get("smart_rank_reason")) if top_pocket is not None and pd.notna(top_pocket.get("smart_rank_reason")) else None,
            "top_pocket_evidence_quality_label": str(top_pocket.get("evidence_quality_label")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_label")) else None,
            "top_pocket_evidence_quality_score": float(top_pocket.get("evidence_quality_score")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_score")) else None,
            "top_pocket_evidence_quality_warning": str(top_pocket.get("evidence_quality_warning")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_warning")) else None,
            "top_pocket_decision_label": str(top_pocket_decision.get("decision_label")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_label")) else None,
            "top_pocket_decision_score": float(top_pocket_decision.get("decision_score")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_score")) else None,
            "top_pocket_audit_status": str(top_pocket_decision.get("audit_status")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("audit_status")) else None,
            "top_pocket_next_step": str(top_pocket_decision.get("next_step")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("next_step")) else None,
            "top_pocket_reliability_gaps": top_reliability_gaps or None,
            "pocket_reliability_pass_count": int((pocket_reliability_df["status"].astype(str) == "pass").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0,
            "pocket_reliability_missing_count": int((pocket_reliability_df["status"].astype(str) == "missing").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0,
            "top_pocket_precision_tier": str(top_pocket_triage.get("precision_tier")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("precision_tier")) else None,
            "top_pocket_triage_action": str(top_pocket_triage.get("triage_action")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("triage_action")) else None,
            "top_precision_triage_pocket_id": str(top_precision_triage.get("pocket_id")) if top_precision_triage is not None and pd.notna(top_precision_triage.get("pocket_id")) else None,
            "top_joint_pocket_id": str(top_joint_candidate.get("pocket_id")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("pocket_id")) else None,
            "top_joint_recommendation_label": str(top_joint_candidate.get("recommendation_label")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("recommendation_label")) else None,
            "top_joint_recommendation_score": float(top_joint_candidate.get("recommendation_score")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("recommendation_score")) else None,
            "top_joint_reason": str(top_joint_candidate.get("recommendation_reason")) if top_joint_candidate is not None and pd.notna(top_joint_candidate.get("recommendation_reason")) else None,
            "literature_site_rows": int(len(literature_site_df)),
            "literature_status": str(literature_site_meta.get("status") or ""),
            "literature_query": str(literature_site_meta.get("query") or ""),
            "ai_evidence_enabled": bool(enable_ai_evidence),
            "ai_evidence_rows": int(len(ai_evidence_df)),
            "ai_evidence_status": str(ai_evidence_meta.get("status") or ""),
            "ai_evidence_manual_review_rows": int(ai_evidence_meta.get("manual_review_rows") or 0)
            if str(ai_evidence_meta.get("manual_review_rows") or "").strip().isdigit()
            else 0,
            "ai_evidence_ranked_rows": int(len(rankable_ai_evidence_df)),
            "ai_evidence_ranking_status": str(rankable_ai_evidence_meta.get("status") or ""),
            "ai_review_decision_rows": int(len(ai_review_decision_df)),
            "ai_review_decision_status": str(ai_review_decision_meta.get("status") or ""),
            "ai_review_decision_applied_rows": int(ai_review_decision_meta.get("applied_rows") or 0)
            if str(ai_review_decision_meta.get("applied_rows") or "").strip().isdigit()
            else 0,
            "ai_review_decision_validation_rows": int(len(ai_review_decision_validation_df)),
            "ai_review_decision_validation_blocked_rows": int(
                (ai_review_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
            ) if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns else 0,
            "ai_review_round_status": str(ai_review_round_summary_df.iloc[0].get("review_round_status")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_status")) else "",
            "ai_review_round_reason": str(ai_review_round_summary_df.iloc[0].get("review_round_reason")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_reason")) else "",
            "ai_review_round_rankable_rows": int(ai_review_round_summary_df.iloc[0].get("rankable_after_review_rows") or 0) if not ai_review_round_summary_df.empty else 0,
            "ai_review_ranking_effect_status": str(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) if not ai_review_ranking_delta_df.empty and pd.notna(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) else "",
            "ai_review_ranking_promoted_rows": int(ai_review_ranking_delta_df.iloc[0].get("promoted_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
            "ai_review_ranking_removed_rows": int(ai_review_ranking_delta_df.iloc[0].get("removed_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
            "ai_review_artifact_manifest_rows": int(len(ai_review_artifact_manifest_df)),
            "ai_review_bundle_readme_available": bool(ai_review_bundle_readme_markdown),
            "ai_review_artifact_bundle_available": bool(ai_review_artifact_bundle_zip),
            "ai_review_bundle_verification_rows": int(len(ai_review_bundle_verification_df)),
            "ai_review_bundle_verification_failed_rows": int(
                (ai_review_bundle_verification_df["verification_status"].astype(str) != "verified").sum()
            ) if not ai_review_bundle_verification_df.empty and "verification_status" in ai_review_bundle_verification_df.columns else 0,
            "ai_review_bundle_verification_status": str(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) if not ai_review_bundle_verification_summary_df.empty and pd.notna(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) else "",
            "ai_review_bundle_certificate_available": bool(ai_review_bundle_certificate_markdown),
            "ai_review_decision_outcome_rows": int(len(ai_review_decision_outcome_df)),
            "ai_review_decision_template_rows": int(len(ai_review_decision_template_df)),
            "ai_evidence_supported_count": int((ai_evidence_audit_df["audit_status"].astype(str) == "supported").sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
            "ai_evidence_review_count": int((ai_evidence_audit_df["audit_status"].astype(str).isin(["needs-review", "unsupported", "conflicting"])).sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
            "ai_evidence_review_queue_rows": int(len(ai_evidence_review_queue_df)),
            "top_ai_review_fix_type": str(ai_evidence_review_queue_df.iloc[0].get("fix_type")) if not ai_evidence_review_queue_df.empty and pd.notna(ai_evidence_review_queue_df.iloc[0].get("fix_type")) else None,
            "ai_influence_level": str(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) else None,
            "top_pocket_has_ai_support": bool(ai_ranking_impact_df.iloc[0].get("top_pocket_has_ai_support")) if not ai_ranking_impact_df.empty else False,
            "top_pocket_ai_residues": str(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) else None,
            "ai_followup_plan_rows": int(len(ai_followup_plan_df)),
            "top_ai_followup_query": str(ai_followup_plan_df.iloc[0].get("search_query")) if not ai_followup_plan_df.empty and pd.notna(ai_followup_plan_df.iloc[0].get("search_query")) else None,
            "residue_evidence_consensus_rows": int(len(residue_evidence_consensus_df)),
            "top_residue_consensus_anchor": str(top_residue_consensus.get("residue_anchor")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("residue_anchor")) else None,
            "top_residue_consensus_tier": str(top_residue_consensus.get("consensus_tier")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_tier")) else None,
            "top_residue_consensus_score": float(top_residue_consensus.get("consensus_score")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_score")) else None,
            "top_residue_consensus_sources": str(top_residue_consensus.get("evidence_sources")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("evidence_sources")) else None,
            "pocket_consensus_coverage_rows": int(len(pocket_consensus_coverage_df)),
            "top_pocket_consensus_coverage_id": str(top_pocket_consensus_coverage.get("pocket_id")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_id")) else None,
            "top_pocket_consensus_label": str(top_pocket_consensus_coverage.get("pocket_consensus_label")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_consensus_label")) else None,
            "top_pocket_consensus_anchor_count": int(top_pocket_consensus_coverage.get("rank_safe_anchor_count") or 0) if top_pocket_consensus_coverage is not None else 0,
            "top_pocket_consensus_best_score": float(top_pocket_consensus_coverage.get("best_consensus_score")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("best_consensus_score")) else None,
            "pocket_benchmark_reference_candidate_rows": int(len(benchmark_reference_candidate_df)),
            "pocket_benchmark_reference_import_summary_rows": int(len(benchmark_reference_import_summary_df)),
            "pocket_benchmark_reference_import_status": str(benchmark_reference_import_summary_df.iloc[0].get("import_status") or "") if not benchmark_reference_import_summary_df.empty else "",
            "pocket_benchmark_reference_candidate_review_rows": int(len(benchmark_reference_candidate_review_queue_df)),
            "pocket_benchmark_reference_candidate_review_p1_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P1").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
            "pocket_benchmark_reference_candidate_review_p2_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P2").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
            "pocket_benchmark_reference_candidate_review_checklist_available": bool(benchmark_reference_candidate_review_checklist_markdown),
            "pocket_benchmark_reference_candidate_review_decision_template_rows": int(len(benchmark_reference_candidate_review_decision_template_df)),
            "pocket_benchmark_reference_candidate_review_decision_rows": int(len(benchmark_reference_candidate_review_decision_df)),
            "pocket_benchmark_reference_candidate_review_decision_status": str(benchmark_reference_candidate_review_decision_meta.get("status") or ""),
            "pocket_benchmark_reference_candidate_review_decision_validation_rows": int(len(benchmark_reference_candidate_review_decision_validation_df)),
            "pocket_benchmark_reference_candidate_review_decision_validation_blocked_rows": int(benchmark_reference_candidate_review_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_candidate_review_decision_validation_df.empty and "validation_status" in benchmark_reference_candidate_review_decision_validation_df.columns else 0,
            "pocket_benchmark_reference_candidate_review_outcome_rows": int(len(benchmark_reference_candidate_review_outcome_df)),
            "pocket_benchmark_reference_candidate_review_outcome_accepted_rows": int(benchmark_reference_candidate_review_outcome_df["applied_status"].astype(str).eq("accepted").sum()) if not benchmark_reference_candidate_review_outcome_df.empty and "applied_status" in benchmark_reference_candidate_review_outcome_df.columns else 0,
            "pocket_benchmark_reference_candidate_accepted_rows": int(len(benchmark_reference_candidate_accepted_df)),
            "pocket_benchmark_reference_is_provisional": bool(benchmark_reference_is_provisional),
            "pocket_benchmark_reference_is_reviewed_candidate": bool(benchmark_reference_is_reviewed_candidate),
            "pocket_benchmark_reference_source_mode": str(benchmark_reference_source_mode or ""),
            "pocket_benchmark_reference_source_audit_rows": int(len(benchmark_reference_source_audit_df)),
            "pocket_benchmark_reference_source_audit_summary_rows": int(len(benchmark_reference_source_audit_summary_df)),
            "pocket_benchmark_reference_source_audit_summary_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
            "pocket_benchmark_reference_source_audit_summary_independent_claim_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
            "pocket_benchmark_reference_source_audit_action_queue_rows": int(len(benchmark_reference_source_audit_action_queue_df)),
            "pocket_benchmark_reference_source_audit_action_queue_blocker_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_action_queue_review_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_summary_rows": int(len(benchmark_reference_source_audit_case_summary_df)),
            "pocket_benchmark_reference_source_audit_case_summary_blocked_cases": int(benchmark_reference_source_audit_case_summary_blocked_cases),
            "pocket_benchmark_reference_source_audit_case_summary_review_cases": int(benchmark_reference_source_audit_case_summary_review_cases),
            "pocket_benchmark_reference_source_audit_case_checklist_available": bool(benchmark_reference_source_audit_case_checklist_markdown),
            "pocket_benchmark_reference_source_audit_case_decision_template_rows": int(len(benchmark_reference_source_audit_case_decision_template_df)),
            "pocket_benchmark_reference_source_audit_case_decision_rows": int(len(benchmark_reference_source_audit_case_decision_df)),
            "pocket_benchmark_reference_source_audit_case_decision_status": str(benchmark_reference_source_audit_case_decision_meta.get("status") or ""),
            "pocket_benchmark_reference_source_audit_case_decision_validation_rows": int(len(benchmark_reference_source_audit_case_decision_validation_df)),
            "pocket_benchmark_reference_source_audit_case_decision_validation_blocked_rows": int(benchmark_reference_source_audit_case_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_validation_df.empty and "validation_status" in benchmark_reference_source_audit_case_decision_validation_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_df)),
            "pocket_benchmark_reference_source_audit_case_decision_outcome_blocked_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_pending_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("pending").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_cleared_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).isin(["cleared", "replaced", "source-ready"]).sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_summary_df)),
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_status": benchmark_reference_source_audit_case_decision_outcome_summary_status,
            "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_open_cases": int(benchmark_reference_source_audit_case_decision_outcome_summary_open_cases),
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_rows": int(len(benchmark_reference_source_audit_case_decision_closure_queue_df)),
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_blocker_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_review_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_closure_queue_top_status": str(benchmark_reference_source_audit_case_decision_closure_queue_df.iloc[0].get("applied_status") or "") if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty else "",
            "pocket_benchmark_reference_source_audit_case_decision_closure_checklist_available": bool(benchmark_reference_source_audit_case_decision_closure_checklist_markdown),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_df)),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_cleared_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).eq("cleared-by-decision").sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_open_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).isin(["decision-adjusted-open", "decision-open", "unchanged-open"]).sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df)),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_status": benchmark_reference_source_audit_case_decision_readiness_impact_summary_status,
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases),
            "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_net_blocker_delta": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.iloc[0].get("net_blocker_delta") or 0) if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty else 0,
            "pocket_benchmark_reference_source_audit_checklist_available": bool(benchmark_reference_source_audit_checklist_markdown),
            "pocket_benchmark_reference_source_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_df.empty else "",
            "pocket_benchmark_reference_source_independent_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_df.empty else "",
            "pocket_benchmark_reference_source_provisional_rows": int(benchmark_reference_source_audit_df["is_provisional"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_provisional" in benchmark_reference_source_audit_df.columns else 0,
            "pocket_benchmark_reference_source_reviewed_candidate_rows": int(benchmark_reference_source_audit_df["is_reviewed_candidate"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_reviewed_candidate" in benchmark_reference_source_audit_df.columns else 0,
            "pocket_benchmark_reference_rows": int(len(benchmark_reference_df)),
            "pocket_benchmark_reference_template_rows": int(len(benchmark_reference_template_df)),
            "pocket_benchmark_reference_template_notes_available": bool(benchmark_reference_template_markdown),
            "pocket_benchmark_reference_quality_issue_rows": int(len(pocket_benchmark_reference_quality_issue_df)),
            "pocket_benchmark_reference_quality_summary_rows": int(len(pocket_benchmark_reference_quality_summary_df)),
            "pocket_benchmark_reference_quality_checklist_available": bool(pocket_benchmark_reference_quality_checklist_markdown),
            "pocket_benchmark_reference_structure_validation_issue_rows": int(len(pocket_benchmark_reference_structure_validation_df)),
            "pocket_benchmark_reference_structure_validation_summary_rows": int(len(pocket_benchmark_reference_structure_validation_summary_df)),
            "pocket_benchmark_reference_structure_validation_checklist_available": bool(pocket_benchmark_reference_structure_validation_checklist_markdown),
            "pocket_benchmark_reference_readiness_queue_rows": int(len(pocket_benchmark_reference_readiness_queue_df)),
            "pocket_benchmark_reference_readiness_summary_rows": int(len(pocket_benchmark_reference_readiness_summary_df)),
            "pocket_benchmark_reference_readiness_case_summary_rows": int(len(pocket_benchmark_reference_readiness_case_summary_df)),
            "pocket_benchmark_reference_readiness_status": str(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("readiness_status") or "") if not pocket_benchmark_reference_readiness_summary_df.empty else "",
            "pocket_benchmark_reference_readiness_blocker_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p0_p1_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
            "pocket_benchmark_reference_readiness_review_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p2_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
            "pocket_benchmark_reference_readiness_source_audit_issue_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("source_audit_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
            "pocket_benchmark_reference_readiness_blocked_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
            "pocket_benchmark_reference_readiness_review_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
            "pocket_benchmark_reference_readiness_checklist_available": bool(pocket_benchmark_reference_readiness_checklist_markdown),
            "pocket_benchmark_interpretation_rows": int(len(pocket_benchmark_interpretation_df)),
            "pocket_benchmark_top1_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 1].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 1).any() else "",
            "pocket_benchmark_top3_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 3].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 3).any() else "",
            "pocket_benchmark_case_interpretation_rows": int(len(pocket_benchmark_case_interpretation_df)),
            "pocket_benchmark_case_interpretation_blocked_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
            "pocket_benchmark_case_interpretation_review_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_rows": int(len(pocket_benchmark_case_interpretation_matrix_df)),
            "pocket_benchmark_case_interpretation_matrix_blocked_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_review_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_summary_rows": int(len(pocket_benchmark_case_interpretation_matrix_summary_df)),
            "pocket_benchmark_case_interpretation_matrix_summary_status": str(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("summary_status") or "") if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else "",
            "pocket_benchmark_case_interpretation_matrix_summary_usable_cases": int(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("usable_claim_ready_case_count") or 0) if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else 0,
            "pocket_benchmark_case_interpretation_matrix_queue_rows": int(len(pocket_benchmark_case_interpretation_matrix_queue_df)),
            "pocket_benchmark_case_interpretation_matrix_queue_blocker_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
            "pocket_benchmark_case_interpretation_matrix_queue_review_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_rows": int(len(pocket_benchmark_dataset_interpretation_df)),
            "pocket_benchmark_dataset_interpretation_blocked_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_review_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-blocked").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-review-needed").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-gate-mismatch").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df)),
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_gate_mismatch"].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_gate_mismatch" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
            "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_available": bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown),
            "pocket_benchmark_dataset_interpretation_queue_rows": int(len(pocket_benchmark_dataset_interpretation_queue_df)),
            "pocket_benchmark_dataset_interpretation_queue_blocker_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_queue_review_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
            "pocket_benchmark_dataset_interpretation_checklist_available": bool(pocket_benchmark_dataset_interpretation_checklist_markdown),
            "pocket_benchmark_dataset_interpretation_report_available": bool(pocket_benchmark_dataset_interpretation_report_markdown),
            "pocket_benchmark_top1_coverage": float(top1_benchmark.get("coverage_ratio") or 0.0) if top1_benchmark is not None else None,
            "pocket_benchmark_top1_status": str(top1_benchmark.get("benchmark_status") or "") if top1_benchmark is not None else None,
            "pocket_benchmark_top3_coverage": float(top3_benchmark.get("coverage_ratio") or 0.0) if top3_benchmark is not None else None,
            "pocket_benchmark_top3_status": str(top3_benchmark.get("benchmark_status") or "") if top3_benchmark is not None else None,
            "pocket_benchmark_best_rank": int(top3_benchmark.get("best_rank") or 0) if top3_benchmark is not None else 0,
            "pocket_benchmark_best_pocket_id": str(top3_benchmark.get("best_pocket_id") or "") if top3_benchmark is not None else None,
            "pocket_benchmark_case_summary_rows": int(len(pocket_benchmark_case_summary_df)),
            "pocket_benchmark_dataset_summary_rows": int(len(pocket_benchmark_dataset_summary_df)),
            "pocket_benchmark_variant_comparison_rows": int(len(pocket_benchmark_variant_comparison_df)),
            "pocket_benchmark_variant_case_comparison_rows": int(len(pocket_benchmark_variant_case_comparison_df)),
            "pocket_benchmark_variant_dataset_comparison_rows": int(len(pocket_benchmark_variant_dataset_comparison_df)),
            "pocket_benchmark_variant_detail_comparison_rows": int(len(pocket_benchmark_variant_detail_comparison_df)),
            "pocket_benchmark_variant_remediation_rows": int(len(pocket_benchmark_variant_remediation_df)),
            "pocket_benchmark_variant_remediation_summary_rows": int(len(pocket_benchmark_variant_remediation_summary_df)),
            "pocket_benchmark_variant_remediation_checklist_available": bool(pocket_benchmark_variant_remediation_checklist_markdown),
            "p2rank_ab_enabled": bool(p2rank_ab_enabled),
            "p2rank_ab_changed_count": int((p2rank_ab_df["status"].astype(str) != "unchanged").sum())
            if not p2rank_ab_df.empty and "status" in p2rank_ab_df.columns
            else 0,
            "consensus_rerank_suggestion_rows": int(len(consensus_rerank_suggestion_df)),
            "top_consensus_rerank_pocket_id": str(top_consensus_rerank_suggestion.get("pocket_id")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("pocket_id")) else None,
            "top_consensus_rerank_status": str(top_consensus_rerank_suggestion.get("suggestion_status")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("suggestion_status")) else None,
            "top_consensus_rerank_rank_delta": int(top_consensus_rerank_suggestion.get("rank_delta") or 0) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("rank_delta")) else 0,
            "consensus_rerank_preview_rows": int(len(consensus_rerank_preview_df)),
            "top_consensus_preview_pocket_id": str(top_consensus_rerank_preview.get("pocket_id")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("pocket_id")) else None,
            "top_consensus_preview_decision": str(top_consensus_rerank_preview.get("preview_decision")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_decision")) else None,
            "top_consensus_preview_rank_delta": int(top_consensus_rerank_preview.get("preview_rank_delta") or 0) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_rank_delta")) else 0,
            "top_consensus_preview_score": float(top_consensus_rerank_preview.get("preview_score")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_score")) else None,
            "consensus_rerank_policy_status": str(top_consensus_rerank_policy_gate.get("policy_status")) if top_consensus_rerank_policy_gate is not None and pd.notna(top_consensus_rerank_policy_gate.get("policy_status")) else "",
            "consensus_rerank_policy_changed_rows": int(top_consensus_rerank_policy_gate.get("changed_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
            "consensus_rerank_policy_blocked_rows": int(top_consensus_rerank_policy_gate.get("blocked_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
            "consensus_rerank_action_queue_rows": int(len(consensus_rerank_action_queue_df)),
            "top_consensus_rerank_action_pocket_id": str(top_consensus_rerank_action.get("pocket_id")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("pocket_id")) else None,
            "top_consensus_rerank_issue_type": str(top_consensus_rerank_action.get("issue_type")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_type")) else None,
            "top_consensus_rerank_issue_severity": str(top_consensus_rerank_action.get("issue_severity")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_severity")) else None,
            "consensus_rerank_action_checklist_available": bool(consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty),
            "consensus_rerank_apply_simulation_rows": int(len(consensus_rerank_apply_simulation_df)),
            "top_consensus_rerank_apply_pocket_id": str(top_consensus_rerank_apply.get("pocket_id")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("pocket_id")) else None,
            "top_consensus_rerank_apply_status": str(top_consensus_rerank_apply.get("apply_status")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("apply_status")) else None,
            "top_consensus_rerank_apply_rank_delta": int(top_consensus_rerank_apply.get("simulated_rank_delta") or 0) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("simulated_rank_delta")) else 0,
            "consensus_rerank_simulation_delta_rows": int(len(consensus_rerank_simulation_delta_df)),
            "top_consensus_rerank_delta_pocket_id": str(top_consensus_rerank_delta.get("pocket_id")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("pocket_id")) else None,
            "top_consensus_rerank_delta_change_type": str(top_consensus_rerank_delta.get("change_type")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("change_type")) else None,
            "top_consensus_rerank_delta_rank_delta": int(top_consensus_rerank_delta.get("rank_delta") or 0) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("rank_delta")) else 0,
            "consensus_rerank_precision_scorecard_rows": int(len(consensus_rerank_precision_scorecard_df)),
            "consensus_rerank_precision_score": int(top_consensus_rerank_scorecard.get("precision_improvement_score") or 0) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("precision_improvement_score")) else 0,
            "consensus_rerank_precision_status": str(top_consensus_rerank_scorecard.get("scorecard_status")) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("scorecard_status")) else None,
            "consensus_rerank_positive_signal_rows": int(top_consensus_rerank_scorecard.get("positive_signal_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
            "consensus_rerank_open_blocker_rows": int(top_consensus_rerank_scorecard.get("open_blocker_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
            "consensus_rerank_precision_guardrail_rows": int(len(consensus_rerank_precision_guardrail_df)),
            "consensus_rerank_guardrail_status": str(top_consensus_rerank_guardrail.get("guardrail_status")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_status")) else None,
            "consensus_rerank_guardrail_decision": str(top_consensus_rerank_guardrail.get("guardrail_decision")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_decision")) else None,
            "consensus_rerank_guardrail_apply_mode": str(top_consensus_rerank_guardrail.get("apply_mode")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("apply_mode")) else None,
            "consensus_rerank_guardrail_can_apply_after_review": bool(top_consensus_rerank_guardrail.get("can_apply_after_manual_review")) if top_consensus_rerank_guardrail is not None else False,
            "consensus_rerank_guardrail_report_available": bool(consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty),
            "consensus_rerank_guardrail_artifact_manifest_rows": int(len(consensus_rerank_guardrail_artifact_manifest_df)),
            "consensus_rerank_guardrail_handoff_zip_available": bool(consensus_rerank_guardrail_handoff_zip),
            "consensus_rerank_guardrail_bundle_verification_rows": int(len(consensus_rerank_guardrail_bundle_verification_df)),
            "consensus_rerank_guardrail_bundle_verification_status": str(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty and pd.notna(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) else None,
            "consensus_rerank_guardrail_bundle_verification_failed_rows": int(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("failed_files") or 0) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else 0,
            "consensus_rerank_guardrail_handoff_certificate_available": bool(consensus_rerank_guardrail_handoff_certificate_markdown),
            "consensus_rerank_release_decision_template_rows": int(len(consensus_rerank_release_decision_template_df)),
            "consensus_rerank_release_decision_rows": int(len(consensus_rerank_release_decision_df)),
            "consensus_rerank_release_decision_status": str(consensus_rerank_release_decision_meta.get("status") or ""),
            "consensus_rerank_release_decision_validation_rows": int(len(consensus_rerank_release_decision_validation_df)),
            "consensus_rerank_release_decision_blocked_rows": int(
                (consensus_rerank_release_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
            ) if not consensus_rerank_release_decision_validation_df.empty and "validation_status" in consensus_rerank_release_decision_validation_df.columns else 0,
            "consensus_rerank_release_review_status": str(top_consensus_rerank_release_decision_summary.get("release_review_status")) if top_consensus_rerank_release_decision_summary is not None and pd.notna(top_consensus_rerank_release_decision_summary.get("release_review_status")) else "",
            "consensus_rerank_release_allowed": bool(top_consensus_rerank_release_decision_summary.get("release_allowed")) if top_consensus_rerank_release_decision_summary is not None else False,
            "consensus_rerank_release_apply_plan_rows": int(len(consensus_rerank_release_apply_plan_df)),
            "top_consensus_rerank_release_apply_pocket_id": str(top_consensus_rerank_release_apply_plan.get("pocket_id")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("pocket_id")) else None,
            "top_consensus_rerank_release_apply_status": str(top_consensus_rerank_release_apply_plan.get("release_apply_status")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("release_apply_status")) else None,
            "consensus_rerank_release_apply_report_available": bool(consensus_rerank_release_apply_report_markdown),
            "consensus_rerank_release_execution_template_rows": int(len(consensus_rerank_release_execution_template_df)),
            "consensus_rerank_release_execution_receipt_rows": int(len(consensus_rerank_release_execution_receipt_df)),
            "consensus_rerank_release_execution_receipt_status": str(consensus_rerank_release_execution_receipt_meta.get("status") or ""),
            "consensus_rerank_release_execution_validation_rows": int(len(consensus_rerank_release_execution_validation_df)),
            "consensus_rerank_release_execution_blocked_rows": int(
                (consensus_rerank_release_execution_validation_df["validation_status"].astype(str) == "blocked").sum()
            ) if not consensus_rerank_release_execution_validation_df.empty and "validation_status" in consensus_rerank_release_execution_validation_df.columns else 0,
            "consensus_rerank_release_execution_review_status": str(top_consensus_rerank_release_execution_summary.get("execution_review_status")) if top_consensus_rerank_release_execution_summary is not None and pd.notna(top_consensus_rerank_release_execution_summary.get("execution_review_status")) else "",
            "consensus_rerank_release_execution_complete": bool(top_consensus_rerank_release_execution_summary.get("execution_complete")) if top_consensus_rerank_release_execution_summary is not None else False,
            "consensus_rerank_release_execution_report_available": bool(consensus_rerank_release_execution_report_markdown),
            "consensus_rerank_release_closure_certificate_available": bool(consensus_rerank_release_closure_certificate_markdown),
            "consensus_rerank_release_closure_ledger_rows": int(len(consensus_rerank_release_closure_ledger_df)),
            "consensus_rerank_release_closure_ledger_blocked_rows": int(
                consensus_rerank_release_closure_ledger_df["closure_check"].astype(str).str.lower().isin({"blocked", "missing"}).sum()
            ) if not consensus_rerank_release_closure_ledger_df.empty and "closure_check" in consensus_rerank_release_closure_ledger_df.columns else 0,
            "consensus_rerank_release_closure_summary_rows": int(len(consensus_rerank_release_closure_summary_df)),
            "consensus_rerank_release_closure_readiness_status": str(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) if top_consensus_rerank_release_closure_summary is not None and pd.notna(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) else "",
            "consensus_rerank_release_closed": bool(top_consensus_rerank_release_closure_summary.get("release_closed")) if top_consensus_rerank_release_closure_summary is not None else False,
            "consensus_rerank_release_closure_blocker_rows": int(len(consensus_rerank_release_closure_blocker_df)),
            "top_consensus_rerank_release_closure_blocker_type": str(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) if not consensus_rerank_release_closure_blocker_df.empty and pd.notna(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) else "",
            "consensus_rerank_release_closure_remediation_checklist_available": bool(consensus_rerank_release_closure_remediation_checklist_markdown),
            "consensus_rerank_release_closure_detached_manifest_rows": int(len(consensus_rerank_release_closure_detached_manifest_df)),
            "literature_ab_enabled": bool(literature_ab_enabled),
            "literature_ab_changed_count": int((literature_ab_df["status"].astype(str) != "unchanged").sum())
            if not literature_ab_df.empty and "status" in literature_ab_df.columns
            else 0,
            "evidence_route_enabled": bool(auto_external_evidence_route),
            "evidence_route_status": str(auto_detection_summary.get("auto_detection_external_route_status") or ""),
            "evidence_route_ab_enabled": bool(evidence_route_ab_enabled),
            "evidence_route_ab_changed_count": int((evidence_route_ab_df["status"].astype(str) != "unchanged").sum())
            if not evidence_route_ab_df.empty and "status" in evidence_route_ab_df.columns
            else 0,
            "conservation_site_rows": int(len(conservation_site_df)),
            "conservation_source_name": str(conservation_site_meta.get("source") or ""),
            "conservation_score_mean": float(conservation_site_meta.get("score_mean")) if conservation_site_meta.get("score_mean") not in {None, ""} else None,
            "conservation_ab_enabled": bool(conservation_ab_enabled),
            "conservation_ab_changed_count": int((conservation_ab_df["status"].astype(str) != "unchanged").sum())
            if not conservation_ab_df.empty and "status" in conservation_ab_df.columns
            else 0,
        }
    )
except Exception:
    pass

snapshot = build_analysis_snapshot(
    energy_table,
    title="ProteinInsight 口袋 / 界面快照",
    annotation_table=enriched_annotations,
    hotspot_df=hotspot_df,
    pocket_summary=effective_pocket_summary,
    joint_candidate_df=joint_candidate_df,
    protein_volume=protein_volume,
    extra={
        **auto_detection_summary,
        "auto_detection_metadata": auto_detection_meta,
        "effective_pocket_source": effective_pocket_mode,
        "effective_annotation_source": effective_annotation_mode,
        "external_site_rows": int(len(external_site_df)),
        "external_site_accession": str(external_site_meta.get("accession") or ""),
        "external_site_pdb_id": str(external_site_meta.get("pdb_id") or structure_pdb_id),
        "external_mapping_status": str(external_site_meta.get("mapping_status") or ""),
        "literature_site_rows": int(len(literature_site_df)),
        "literature_status": str(literature_site_meta.get("status") or ""),
        "literature_query": str(literature_site_meta.get("query") or ""),
        "literature_metadata": literature_site_meta,
        "ai_evidence_enabled": bool(enable_ai_evidence),
        "ai_evidence_rows": int(len(ai_evidence_df)),
        "ai_evidence_status": str(ai_evidence_meta.get("status") or ""),
        "ai_evidence_metadata": ai_evidence_meta,
        "ai_evidence": ai_evidence_df.to_dict(orient="records"),
        "ai_evidence_ranked_rows": int(len(rankable_ai_evidence_df)),
        "ai_evidence_ranking_metadata": rankable_ai_evidence_meta,
        "ai_evidence_ranked": rankable_ai_evidence_df.to_dict(orient="records"),
        "ai_review_decision_rows": int(len(ai_review_decision_df)),
        "ai_review_decision_status": str(ai_review_decision_meta.get("status") or ""),
        "ai_review_decision_applied_rows": int(ai_review_decision_meta.get("applied_rows") or 0)
        if str(ai_review_decision_meta.get("applied_rows") or "").strip().isdigit()
        else 0,
        "ai_review_decision_validation_rows": int(len(ai_review_decision_validation_df)),
        "ai_review_decision_validation_blocked_rows": int(
            (ai_review_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
        ) if not ai_review_decision_validation_df.empty and "validation_status" in ai_review_decision_validation_df.columns else 0,
        "ai_review_decision_validation": ai_review_decision_validation_df.to_dict(orient="records"),
        "ai_review_round_status": str(ai_review_round_summary_df.iloc[0].get("review_round_status")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_status")) else "",
        "ai_review_round_reason": str(ai_review_round_summary_df.iloc[0].get("review_round_reason")) if not ai_review_round_summary_df.empty and pd.notna(ai_review_round_summary_df.iloc[0].get("review_round_reason")) else "",
        "ai_review_round_rankable_rows": int(ai_review_round_summary_df.iloc[0].get("rankable_after_review_rows") or 0) if not ai_review_round_summary_df.empty else 0,
        "ai_review_round_summary": ai_review_round_summary_df.to_dict(orient="records"),
        "ai_review_ranking_effect_status": str(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) if not ai_review_ranking_delta_df.empty and pd.notna(ai_review_ranking_delta_df.iloc[0].get("review_effect_status")) else "",
        "ai_review_ranking_promoted_rows": int(ai_review_ranking_delta_df.iloc[0].get("promoted_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
        "ai_review_ranking_removed_rows": int(ai_review_ranking_delta_df.iloc[0].get("removed_rows") or 0) if not ai_review_ranking_delta_df.empty else 0,
        "ai_review_ranking_delta": ai_review_ranking_delta_df.to_dict(orient="records"),
        "ai_review_round_report_available": bool(ai_review_round_report_markdown),
        "ai_review_artifact_manifest_rows": int(len(ai_review_artifact_manifest_df)),
        "ai_review_artifact_manifest": ai_review_artifact_manifest_df.to_dict(orient="records"),
        "ai_review_bundle_readme_available": bool(ai_review_bundle_readme_markdown),
        "ai_review_artifact_bundle_available": bool(ai_review_artifact_bundle_zip),
        "ai_review_bundle_verification_rows": int(len(ai_review_bundle_verification_df)),
        "ai_review_bundle_verification_failed_rows": int(
            (ai_review_bundle_verification_df["verification_status"].astype(str) != "verified").sum()
        ) if not ai_review_bundle_verification_df.empty and "verification_status" in ai_review_bundle_verification_df.columns else 0,
        "ai_review_bundle_verification": ai_review_bundle_verification_df.to_dict(orient="records"),
        "ai_review_bundle_verification_status": str(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) if not ai_review_bundle_verification_summary_df.empty and pd.notna(ai_review_bundle_verification_summary_df.iloc[0].get("verification_status")) else "",
        "ai_review_bundle_verification_summary": ai_review_bundle_verification_summary_df.to_dict(orient="records"),
        "ai_review_bundle_certificate_available": bool(ai_review_bundle_certificate_markdown),
        "ai_review_decision_outcome_rows": int(len(ai_review_decision_outcome_df)),
        "ai_review_decision_outcomes": ai_review_decision_outcome_df.to_dict(orient="records"),
        "ai_review_decision_template_rows": int(len(ai_review_decision_template_df)),
        "ai_review_decision_template": ai_review_decision_template_df.to_dict(orient="records"),
        "ai_review_decision_metadata": ai_review_decision_meta,
        "ai_review_decisions": ai_review_decision_df.to_dict(orient="records"),
        "ai_evidence_audit_supported_count": int((ai_evidence_audit_df["audit_status"].astype(str) == "supported").sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
        "ai_evidence_audit_review_count": int((ai_evidence_audit_df["audit_status"].astype(str).isin(["needs-review", "unsupported", "conflicting"])).sum()) if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns else 0,
        "ai_evidence_audit": ai_evidence_audit_df.to_dict(orient="records"),
        "ai_evidence_review_queue_rows": int(len(ai_evidence_review_queue_df)),
        "top_ai_review_fix_type": str(ai_evidence_review_queue_df.iloc[0].get("fix_type")) if not ai_evidence_review_queue_df.empty and pd.notna(ai_evidence_review_queue_df.iloc[0].get("fix_type")) else None,
        "ai_evidence_review_queue": ai_evidence_review_queue_df.to_dict(orient="records"),
        "ai_ranking_impact": ai_ranking_impact_df.to_dict(orient="records"),
        "ai_influence_level": str(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("ai_influence_level")) else None,
        "top_pocket_has_ai_support": bool(ai_ranking_impact_df.iloc[0].get("top_pocket_has_ai_support")) if not ai_ranking_impact_df.empty else False,
        "top_pocket_ai_residues": str(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) if not ai_ranking_impact_df.empty and pd.notna(ai_ranking_impact_df.iloc[0].get("top_pocket_ai_residues")) else None,
        "ai_followup_plan_rows": int(len(ai_followup_plan_df)),
        "top_ai_followup_query": str(ai_followup_plan_df.iloc[0].get("search_query")) if not ai_followup_plan_df.empty and pd.notna(ai_followup_plan_df.iloc[0].get("search_query")) else None,
        "ai_followup_plan": ai_followup_plan_df.to_dict(orient="records"),
        "residue_evidence_consensus_rows": int(len(residue_evidence_consensus_df)),
        "residue_evidence_consensus": residue_evidence_consensus_df.to_dict(orient="records"),
        "top_residue_consensus_anchor": str(top_residue_consensus.get("residue_anchor")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("residue_anchor")) else None,
        "top_residue_consensus_tier": str(top_residue_consensus.get("consensus_tier")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_tier")) else None,
        "top_residue_consensus_score": float(top_residue_consensus.get("consensus_score")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("consensus_score")) else None,
        "top_residue_consensus_sources": str(top_residue_consensus.get("evidence_sources")) if top_residue_consensus is not None and pd.notna(top_residue_consensus.get("evidence_sources")) else None,
        "pocket_consensus_coverage_rows": int(len(pocket_consensus_coverage_df)),
        "pocket_consensus_coverage": pocket_consensus_coverage_df.to_dict(orient="records"),
        "top_pocket_consensus_coverage_id": str(top_pocket_consensus_coverage.get("pocket_id")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_id")) else None,
        "top_pocket_consensus_label": str(top_pocket_consensus_coverage.get("pocket_consensus_label")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("pocket_consensus_label")) else None,
        "top_pocket_consensus_anchor_count": int(top_pocket_consensus_coverage.get("rank_safe_anchor_count") or 0) if top_pocket_consensus_coverage is not None else 0,
        "top_pocket_consensus_best_score": float(top_pocket_consensus_coverage.get("best_consensus_score")) if top_pocket_consensus_coverage is not None and pd.notna(top_pocket_consensus_coverage.get("best_consensus_score")) else None,
        "pocket_benchmark_reference_candidate_rows": int(len(benchmark_reference_candidate_df)),
        "pocket_benchmark_reference_candidate": benchmark_reference_candidate_df.to_dict(orient="records"),
        "pocket_benchmark_reference_import_summary_rows": int(len(benchmark_reference_import_summary_df)),
        "pocket_benchmark_reference_import_summary": benchmark_reference_import_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_import_status": str(benchmark_reference_import_summary_df.iloc[0].get("import_status") or "") if not benchmark_reference_import_summary_df.empty else "",
        "pocket_benchmark_reference_candidate_review_rows": int(len(benchmark_reference_candidate_review_queue_df)),
        "pocket_benchmark_reference_candidate_review_queue": benchmark_reference_candidate_review_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_p1_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P1").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_p2_rows": int(benchmark_reference_candidate_review_queue_df["priority"].astype(str).eq("P2").sum()) if not benchmark_reference_candidate_review_queue_df.empty and "priority" in benchmark_reference_candidate_review_queue_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_checklist_available": bool(benchmark_reference_candidate_review_checklist_markdown),
        "pocket_benchmark_reference_candidate_review_checklist": benchmark_reference_candidate_review_checklist_markdown,
        "pocket_benchmark_reference_candidate_review_decision_template_rows": int(len(benchmark_reference_candidate_review_decision_template_df)),
        "pocket_benchmark_reference_candidate_review_decision_template": benchmark_reference_candidate_review_decision_template_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_decision_rows": int(len(benchmark_reference_candidate_review_decision_df)),
        "pocket_benchmark_reference_candidate_review_decisions": benchmark_reference_candidate_review_decision_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_decision_status": str(benchmark_reference_candidate_review_decision_meta.get("status") or ""),
        "pocket_benchmark_reference_candidate_review_decision_metadata": benchmark_reference_candidate_review_decision_meta,
        "pocket_benchmark_reference_candidate_review_decision_validation_rows": int(len(benchmark_reference_candidate_review_decision_validation_df)),
        "pocket_benchmark_reference_candidate_review_decision_validation_blocked_rows": int(benchmark_reference_candidate_review_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_candidate_review_decision_validation_df.empty and "validation_status" in benchmark_reference_candidate_review_decision_validation_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_decision_validation": benchmark_reference_candidate_review_decision_validation_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_review_outcome_rows": int(len(benchmark_reference_candidate_review_outcome_df)),
        "pocket_benchmark_reference_candidate_review_outcome_accepted_rows": int(benchmark_reference_candidate_review_outcome_df["applied_status"].astype(str).eq("accepted").sum()) if not benchmark_reference_candidate_review_outcome_df.empty and "applied_status" in benchmark_reference_candidate_review_outcome_df.columns else 0,
        "pocket_benchmark_reference_candidate_review_outcomes": benchmark_reference_candidate_review_outcome_df.to_dict(orient="records"),
        "pocket_benchmark_reference_candidate_accepted_rows": int(len(benchmark_reference_candidate_accepted_df)),
        "pocket_benchmark_reference_candidate_accepted": benchmark_reference_candidate_accepted_df.to_dict(orient="records"),
        "pocket_benchmark_reference_is_provisional": bool(benchmark_reference_is_provisional),
        "pocket_benchmark_reference_is_reviewed_candidate": bool(benchmark_reference_is_reviewed_candidate),
        "pocket_benchmark_reference_source_mode": str(benchmark_reference_source_mode or ""),
        "pocket_benchmark_reference_source_audit_rows": int(len(benchmark_reference_source_audit_df)),
        "pocket_benchmark_reference_source_audit": benchmark_reference_source_audit_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_summary_rows": int(len(benchmark_reference_source_audit_summary_df)),
        "pocket_benchmark_reference_source_audit_summary": benchmark_reference_source_audit_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_summary_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
        "pocket_benchmark_reference_source_audit_summary_independent_claim_status": str(benchmark_reference_source_audit_summary_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_summary_df.empty else "",
        "pocket_benchmark_reference_source_audit_action_queue_rows": int(len(benchmark_reference_source_audit_action_queue_df)),
        "pocket_benchmark_reference_source_audit_action_queue": benchmark_reference_source_audit_action_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_action_queue_blocker_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_action_queue_review_rows": int(benchmark_reference_source_audit_action_queue_df["action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_action_queue_df.empty and "action_status" in benchmark_reference_source_audit_action_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_summary_rows": int(len(benchmark_reference_source_audit_case_summary_df)),
        "pocket_benchmark_reference_source_audit_case_summary": benchmark_reference_source_audit_case_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_summary_blocked_cases": int(benchmark_reference_source_audit_case_summary_blocked_cases),
        "pocket_benchmark_reference_source_audit_case_summary_review_cases": int(benchmark_reference_source_audit_case_summary_review_cases),
        "pocket_benchmark_reference_source_audit_case_checklist_available": bool(benchmark_reference_source_audit_case_checklist_markdown),
        "pocket_benchmark_reference_source_audit_case_checklist": benchmark_reference_source_audit_case_checklist_markdown,
        "pocket_benchmark_reference_source_audit_case_decision_template_rows": int(len(benchmark_reference_source_audit_case_decision_template_df)),
        "pocket_benchmark_reference_source_audit_case_decision_template": benchmark_reference_source_audit_case_decision_template_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_rows": int(len(benchmark_reference_source_audit_case_decision_df)),
        "pocket_benchmark_reference_source_audit_case_decision_status": str(benchmark_reference_source_audit_case_decision_meta.get("status") or ""),
        "pocket_benchmark_reference_source_audit_case_decisions": benchmark_reference_source_audit_case_decision_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_validation_rows": int(len(benchmark_reference_source_audit_case_decision_validation_df)),
        "pocket_benchmark_reference_source_audit_case_decision_validation_blocked_rows": int(benchmark_reference_source_audit_case_decision_validation_df["validation_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_validation_df.empty and "validation_status" in benchmark_reference_source_audit_case_decision_validation_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_validation": benchmark_reference_source_audit_case_decision_validation_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_df)),
        "pocket_benchmark_reference_source_audit_case_decision_outcomes": benchmark_reference_source_audit_case_decision_outcome_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_blocked_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("blocked").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_pending_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).eq("pending").sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_cleared_rows": int(benchmark_reference_source_audit_case_decision_outcome_df["applied_status"].astype(str).isin(["cleared", "replaced", "source-ready"]).sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and "applied_status" in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_rows": int(len(benchmark_reference_source_audit_case_decision_outcome_summary_df)),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_status": benchmark_reference_source_audit_case_decision_outcome_summary_status,
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary_open_cases": int(benchmark_reference_source_audit_case_decision_outcome_summary_open_cases),
        "pocket_benchmark_reference_source_audit_case_decision_outcome_summary": benchmark_reference_source_audit_case_decision_outcome_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_rows": int(len(benchmark_reference_source_audit_case_decision_closure_queue_df)),
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_blocker_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("blocker").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_review_rows": int(benchmark_reference_source_audit_case_decision_closure_queue_df["closure_action_status"].astype(str).eq("review").sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and "closure_action_status" in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue_top_status": str(benchmark_reference_source_audit_case_decision_closure_queue_df.iloc[0].get("applied_status") or "") if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty else "",
        "pocket_benchmark_reference_source_audit_case_decision_closure_queue": benchmark_reference_source_audit_case_decision_closure_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_closure_checklist_available": bool(benchmark_reference_source_audit_case_decision_closure_checklist_markdown),
        "pocket_benchmark_reference_source_audit_case_decision_closure_checklist": benchmark_reference_source_audit_case_decision_closure_checklist_markdown,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_df)),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_cleared_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).eq("cleared-by-decision").sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_open_rows": int(benchmark_reference_source_audit_case_decision_readiness_impact_df["readiness_impact"].astype(str).isin(["decision-adjusted-open", "decision-open", "unchanged-open"]).sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and "readiness_impact" in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact": benchmark_reference_source_audit_case_decision_readiness_impact_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_rows": int(len(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df)),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_status": benchmark_reference_source_audit_case_decision_readiness_impact_summary_status,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases),
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary_net_blocker_delta": int(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.iloc[0].get("net_blocker_delta") or 0) if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty else 0,
        "pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary": benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_checklist_available": bool(benchmark_reference_source_audit_checklist_markdown),
        "pocket_benchmark_reference_source_audit_checklist": benchmark_reference_source_audit_checklist_markdown,
        "pocket_benchmark_reference_source_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("source_claim_status") or "") if not benchmark_reference_source_audit_df.empty else "",
        "pocket_benchmark_reference_source_independent_claim_status": str(benchmark_reference_source_audit_df.iloc[0].get("can_support_independent_claim") or "") if not benchmark_reference_source_audit_df.empty else "",
        "pocket_benchmark_reference_source_provisional_rows": int(benchmark_reference_source_audit_df["is_provisional"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_provisional" in benchmark_reference_source_audit_df.columns else 0,
        "pocket_benchmark_reference_source_reviewed_candidate_rows": int(benchmark_reference_source_audit_df["is_reviewed_candidate"].astype(bool).sum()) if not benchmark_reference_source_audit_df.empty and "is_reviewed_candidate" in benchmark_reference_source_audit_df.columns else 0,
        "pocket_benchmark_reference_rows": int(len(benchmark_reference_df)),
        "pocket_benchmark_reference": benchmark_reference_df.to_dict(orient="records"),
        "pocket_benchmark_reference_template_rows": int(len(benchmark_reference_template_df)),
        "pocket_benchmark_reference_template": benchmark_reference_template_df.to_dict(orient="records"),
        "pocket_benchmark_reference_template_notes_available": bool(benchmark_reference_template_markdown),
        "pocket_benchmark_reference_quality_issue_rows": int(len(pocket_benchmark_reference_quality_issue_df)),
        "pocket_benchmark_reference_quality_issues": pocket_benchmark_reference_quality_issue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_quality_summary_rows": int(len(pocket_benchmark_reference_quality_summary_df)),
        "pocket_benchmark_reference_quality_summary": pocket_benchmark_reference_quality_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_quality_checklist_available": bool(pocket_benchmark_reference_quality_checklist_markdown),
        "pocket_benchmark_reference_quality_checklist": pocket_benchmark_reference_quality_checklist_markdown,
        "pocket_benchmark_reference_structure_validation_issue_rows": int(len(pocket_benchmark_reference_structure_validation_df)),
        "pocket_benchmark_reference_structure_validation_issues": pocket_benchmark_reference_structure_validation_df.to_dict(orient="records"),
        "pocket_benchmark_reference_structure_validation_summary_rows": int(len(pocket_benchmark_reference_structure_validation_summary_df)),
        "pocket_benchmark_reference_structure_validation_summary": pocket_benchmark_reference_structure_validation_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_structure_validation_checklist_available": bool(pocket_benchmark_reference_structure_validation_checklist_markdown),
        "pocket_benchmark_reference_structure_validation_checklist": pocket_benchmark_reference_structure_validation_checklist_markdown,
        "pocket_benchmark_reference_readiness_queue_rows": int(len(pocket_benchmark_reference_readiness_queue_df)),
        "pocket_benchmark_reference_readiness_queue": pocket_benchmark_reference_readiness_queue_df.to_dict(orient="records"),
        "pocket_benchmark_reference_readiness_summary_rows": int(len(pocket_benchmark_reference_readiness_summary_df)),
        "pocket_benchmark_reference_readiness_summary": pocket_benchmark_reference_readiness_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_readiness_case_summary_rows": int(len(pocket_benchmark_reference_readiness_case_summary_df)),
        "pocket_benchmark_reference_readiness_case_summary": pocket_benchmark_reference_readiness_case_summary_df.to_dict(orient="records"),
        "pocket_benchmark_reference_readiness_status": str(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("readiness_status") or "") if not pocket_benchmark_reference_readiness_summary_df.empty else "",
        "pocket_benchmark_reference_readiness_blocker_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p0_p1_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
        "pocket_benchmark_reference_readiness_review_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("p2_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
        "pocket_benchmark_reference_readiness_source_audit_issue_rows": int(pocket_benchmark_reference_readiness_summary_df.iloc[0].get("source_audit_issue_count") or 0) if not pocket_benchmark_reference_readiness_summary_df.empty else 0,
        "pocket_benchmark_reference_readiness_blocked_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
        "pocket_benchmark_reference_readiness_review_cases": int(pocket_benchmark_reference_readiness_case_summary_df["readiness_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and "readiness_status" in pocket_benchmark_reference_readiness_case_summary_df.columns else 0,
        "pocket_benchmark_reference_readiness_checklist_available": bool(pocket_benchmark_reference_readiness_checklist_markdown),
        "pocket_benchmark_reference_readiness_checklist": pocket_benchmark_reference_readiness_checklist_markdown,
        "pocket_benchmark_interpretation_rows": int(len(pocket_benchmark_interpretation_df)),
        "pocket_benchmark_interpretation": pocket_benchmark_interpretation_df.to_dict(orient="records"),
        "pocket_benchmark_top1_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 1].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 1).any() else "",
        "pocket_benchmark_top3_claim_status": str(pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df["top_n"].astype(int) == 3].iloc[0].get("claim_status") or "") if not pocket_benchmark_interpretation_df.empty and "top_n" in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df["top_n"].astype(int) == 3).any() else "",
        "pocket_benchmark_case_interpretation_rows": int(len(pocket_benchmark_case_interpretation_df)),
        "pocket_benchmark_case_interpretation": pocket_benchmark_case_interpretation_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_blocked_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
        "pocket_benchmark_case_interpretation_review_rows": int(pocket_benchmark_case_interpretation_df["claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_df.empty and "claim_status" in pocket_benchmark_case_interpretation_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_rows": int(len(pocket_benchmark_case_interpretation_matrix_df)),
        "pocket_benchmark_case_interpretation_matrix": pocket_benchmark_case_interpretation_matrix_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_matrix_blocked_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_review_rows": int(pocket_benchmark_case_interpretation_matrix_df["case_interpretation_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and "case_interpretation_status" in pocket_benchmark_case_interpretation_matrix_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_summary_rows": int(len(pocket_benchmark_case_interpretation_matrix_summary_df)),
        "pocket_benchmark_case_interpretation_matrix_summary": pocket_benchmark_case_interpretation_matrix_summary_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_matrix_summary_status": str(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("summary_status") or "") if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else "",
        "pocket_benchmark_case_interpretation_matrix_summary_usable_cases": int(pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get("usable_claim_ready_case_count") or 0) if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else 0,
        "pocket_benchmark_case_interpretation_matrix_queue_rows": int(len(pocket_benchmark_case_interpretation_matrix_queue_df)),
        "pocket_benchmark_case_interpretation_matrix_queue": pocket_benchmark_case_interpretation_matrix_queue_df.to_dict(orient="records"),
        "pocket_benchmark_case_interpretation_matrix_queue_blocker_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
        "pocket_benchmark_case_interpretation_matrix_queue_review_rows": int(pocket_benchmark_case_interpretation_matrix_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and "action_status" in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_rows": int(len(pocket_benchmark_dataset_interpretation_df)),
        "pocket_benchmark_dataset_interpretation": pocket_benchmark_dataset_interpretation_df.to_dict(orient="records"),
        "pocket_benchmark_dataset_interpretation_blocked_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("blocked").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_review_rows": int(pocket_benchmark_dataset_interpretation_df["dataset_claim_status"].astype(str).eq("review-needed").sum()) if not pocket_benchmark_dataset_interpretation_df.empty and "dataset_claim_status" in pocket_benchmark_dataset_interpretation_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-blocked").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-review-needed").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df["dataset_source_impact_status"].astype(str).eq("source-gate-mismatch").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and "dataset_source_impact_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_rows": int(len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df)),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.to_dict(orient="records"),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_blocker_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_review_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_action_status" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_mismatch_rows": int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df["source_gate_mismatch"].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and "source_gate_mismatch" in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0,
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_available": bool(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown),
        "pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist": pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown,
        "pocket_benchmark_dataset_interpretation_queue_rows": int(len(pocket_benchmark_dataset_interpretation_queue_df)),
        "pocket_benchmark_dataset_interpretation_queue": pocket_benchmark_dataset_interpretation_queue_df.to_dict(orient="records"),
        "pocket_benchmark_dataset_interpretation_queue_blocker_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("blocker").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_queue_review_rows": int(pocket_benchmark_dataset_interpretation_queue_df["action_status"].astype(str).eq("review").sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and "action_status" in pocket_benchmark_dataset_interpretation_queue_df.columns else 0,
        "pocket_benchmark_dataset_interpretation_checklist_available": bool(pocket_benchmark_dataset_interpretation_checklist_markdown),
        "pocket_benchmark_dataset_interpretation_checklist": pocket_benchmark_dataset_interpretation_checklist_markdown,
        "pocket_benchmark_dataset_interpretation_report_available": bool(pocket_benchmark_dataset_interpretation_report_markdown),
        "pocket_benchmark_dataset_interpretation_report": pocket_benchmark_dataset_interpretation_report_markdown,
        "pocket_benchmark_summary_rows": int(len(pocket_benchmark_summary_df)),
        "pocket_benchmark_summary": pocket_benchmark_summary_df.to_dict(orient="records"),
        "pocket_benchmark_details_rows": int(len(pocket_benchmark_details_df)),
        "pocket_benchmark_details": pocket_benchmark_details_df.to_dict(orient="records"),
        "pocket_benchmark_case_summary_rows": int(len(pocket_benchmark_case_summary_df)),
        "pocket_benchmark_case_summary": pocket_benchmark_case_summary_df.to_dict(orient="records"),
        "pocket_benchmark_dataset_summary_rows": int(len(pocket_benchmark_dataset_summary_df)),
        "pocket_benchmark_dataset_summary": pocket_benchmark_dataset_summary_df.to_dict(orient="records"),
        "pocket_benchmark_variant_comparison_rows": int(len(pocket_benchmark_variant_comparison_df)),
        "pocket_benchmark_variant_comparison": pocket_benchmark_variant_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_case_comparison_rows": int(len(pocket_benchmark_variant_case_comparison_df)),
        "pocket_benchmark_variant_case_comparison": pocket_benchmark_variant_case_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_dataset_comparison_rows": int(len(pocket_benchmark_variant_dataset_comparison_df)),
        "pocket_benchmark_variant_dataset_comparison": pocket_benchmark_variant_dataset_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_detail_comparison_rows": int(len(pocket_benchmark_variant_detail_comparison_df)),
        "pocket_benchmark_variant_detail_comparison": pocket_benchmark_variant_detail_comparison_df.to_dict(orient="records"),
        "pocket_benchmark_variant_remediation_rows": int(len(pocket_benchmark_variant_remediation_df)),
        "pocket_benchmark_variant_remediation": pocket_benchmark_variant_remediation_df.to_dict(orient="records"),
        "pocket_benchmark_variant_remediation_summary_rows": int(len(pocket_benchmark_variant_remediation_summary_df)),
        "pocket_benchmark_variant_remediation_summary": pocket_benchmark_variant_remediation_summary_df.to_dict(orient="records"),
        "pocket_benchmark_variant_remediation_checklist_available": bool(pocket_benchmark_variant_remediation_checklist_markdown),
        "pocket_benchmark_variant_remediation_checklist": pocket_benchmark_variant_remediation_checklist_markdown,
        "pocket_benchmark_top1_coverage": float(top1_benchmark.get("coverage_ratio") or 0.0) if top1_benchmark is not None else None,
        "pocket_benchmark_top1_status": str(top1_benchmark.get("benchmark_status") or "") if top1_benchmark is not None else None,
        "pocket_benchmark_top3_coverage": float(top3_benchmark.get("coverage_ratio") or 0.0) if top3_benchmark is not None else None,
        "pocket_benchmark_top3_status": str(top3_benchmark.get("benchmark_status") or "") if top3_benchmark is not None else None,
        "pocket_benchmark_best_rank": int(top3_benchmark.get("best_rank") or 0) if top3_benchmark is not None else 0,
        "pocket_benchmark_best_pocket_id": str(top3_benchmark.get("best_pocket_id") or "") if top3_benchmark is not None else None,
        "consensus_rerank_suggestion_rows": int(len(consensus_rerank_suggestion_df)),
        "consensus_rerank_suggestions": consensus_rerank_suggestion_df.to_dict(orient="records"),
        "top_consensus_rerank_pocket_id": str(top_consensus_rerank_suggestion.get("pocket_id")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("pocket_id")) else None,
        "top_consensus_rerank_status": str(top_consensus_rerank_suggestion.get("suggestion_status")) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("suggestion_status")) else None,
        "top_consensus_rerank_rank_delta": int(top_consensus_rerank_suggestion.get("rank_delta") or 0) if top_consensus_rerank_suggestion is not None and pd.notna(top_consensus_rerank_suggestion.get("rank_delta")) else 0,
        "consensus_rerank_preview_rows": int(len(consensus_rerank_preview_df)),
        "consensus_rerank_preview": consensus_rerank_preview_df.to_dict(orient="records"),
        "top_consensus_preview_pocket_id": str(top_consensus_rerank_preview.get("pocket_id")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("pocket_id")) else None,
        "top_consensus_preview_decision": str(top_consensus_rerank_preview.get("preview_decision")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_decision")) else None,
        "top_consensus_preview_rank_delta": int(top_consensus_rerank_preview.get("preview_rank_delta") or 0) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_rank_delta")) else 0,
        "top_consensus_preview_score": float(top_consensus_rerank_preview.get("preview_score")) if top_consensus_rerank_preview is not None and pd.notna(top_consensus_rerank_preview.get("preview_score")) else None,
        "consensus_rerank_policy_gate": consensus_rerank_policy_gate_df.to_dict(orient="records"),
        "consensus_rerank_policy_status": str(top_consensus_rerank_policy_gate.get("policy_status")) if top_consensus_rerank_policy_gate is not None and pd.notna(top_consensus_rerank_policy_gate.get("policy_status")) else "",
        "consensus_rerank_policy_changed_rows": int(top_consensus_rerank_policy_gate.get("changed_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
        "consensus_rerank_policy_blocked_rows": int(top_consensus_rerank_policy_gate.get("blocked_rows") or 0) if top_consensus_rerank_policy_gate is not None else 0,
        "consensus_rerank_action_queue_rows": int(len(consensus_rerank_action_queue_df)),
        "consensus_rerank_action_queue": consensus_rerank_action_queue_df.to_dict(orient="records"),
        "top_consensus_rerank_action_pocket_id": str(top_consensus_rerank_action.get("pocket_id")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("pocket_id")) else None,
        "top_consensus_rerank_issue_type": str(top_consensus_rerank_action.get("issue_type")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_type")) else None,
        "top_consensus_rerank_issue_severity": str(top_consensus_rerank_action.get("issue_severity")) if top_consensus_rerank_action is not None and pd.notna(top_consensus_rerank_action.get("issue_severity")) else None,
        "consensus_rerank_action_checklist_available": bool(consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty),
        "consensus_rerank_apply_simulation_rows": int(len(consensus_rerank_apply_simulation_df)),
        "consensus_rerank_apply_simulation": consensus_rerank_apply_simulation_df.to_dict(orient="records"),
        "top_consensus_rerank_apply_pocket_id": str(top_consensus_rerank_apply.get("pocket_id")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("pocket_id")) else None,
        "top_consensus_rerank_apply_status": str(top_consensus_rerank_apply.get("apply_status")) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("apply_status")) else None,
        "top_consensus_rerank_apply_rank_delta": int(top_consensus_rerank_apply.get("simulated_rank_delta") or 0) if top_consensus_rerank_apply is not None and pd.notna(top_consensus_rerank_apply.get("simulated_rank_delta")) else 0,
        "consensus_rerank_simulation_delta_rows": int(len(consensus_rerank_simulation_delta_df)),
        "consensus_rerank_simulation_delta": consensus_rerank_simulation_delta_df.to_dict(orient="records"),
        "top_consensus_rerank_delta_pocket_id": str(top_consensus_rerank_delta.get("pocket_id")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("pocket_id")) else None,
        "top_consensus_rerank_delta_change_type": str(top_consensus_rerank_delta.get("change_type")) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("change_type")) else None,
        "top_consensus_rerank_delta_rank_delta": int(top_consensus_rerank_delta.get("rank_delta") or 0) if top_consensus_rerank_delta is not None and pd.notna(top_consensus_rerank_delta.get("rank_delta")) else 0,
        "consensus_rerank_precision_scorecard_rows": int(len(consensus_rerank_precision_scorecard_df)),
        "consensus_rerank_precision_scorecard": consensus_rerank_precision_scorecard_df.to_dict(orient="records"),
        "consensus_rerank_precision_score": int(top_consensus_rerank_scorecard.get("precision_improvement_score") or 0) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("precision_improvement_score")) else 0,
        "consensus_rerank_precision_status": str(top_consensus_rerank_scorecard.get("scorecard_status")) if top_consensus_rerank_scorecard is not None and pd.notna(top_consensus_rerank_scorecard.get("scorecard_status")) else None,
        "consensus_rerank_positive_signal_rows": int(top_consensus_rerank_scorecard.get("positive_signal_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
        "consensus_rerank_open_blocker_rows": int(top_consensus_rerank_scorecard.get("open_blocker_rows") or 0) if top_consensus_rerank_scorecard is not None else 0,
        "consensus_rerank_precision_guardrail_rows": int(len(consensus_rerank_precision_guardrail_df)),
        "consensus_rerank_precision_guardrail": consensus_rerank_precision_guardrail_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_status": str(top_consensus_rerank_guardrail.get("guardrail_status")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_status")) else None,
        "consensus_rerank_guardrail_decision": str(top_consensus_rerank_guardrail.get("guardrail_decision")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("guardrail_decision")) else None,
        "consensus_rerank_guardrail_apply_mode": str(top_consensus_rerank_guardrail.get("apply_mode")) if top_consensus_rerank_guardrail is not None and pd.notna(top_consensus_rerank_guardrail.get("apply_mode")) else None,
        "consensus_rerank_guardrail_can_apply_after_review": bool(top_consensus_rerank_guardrail.get("can_apply_after_manual_review")) if top_consensus_rerank_guardrail is not None else False,
        "consensus_rerank_guardrail_report_available": bool(consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty),
        "consensus_rerank_guardrail_artifact_manifest_rows": int(len(consensus_rerank_guardrail_artifact_manifest_df)),
        "consensus_rerank_guardrail_artifact_manifest": consensus_rerank_guardrail_artifact_manifest_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_handoff_zip_available": bool(consensus_rerank_guardrail_handoff_zip),
        "consensus_rerank_guardrail_bundle_verification_rows": int(len(consensus_rerank_guardrail_bundle_verification_df)),
        "consensus_rerank_guardrail_bundle_verification": consensus_rerank_guardrail_bundle_verification_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_bundle_verification_summary": consensus_rerank_guardrail_bundle_verification_summary_df.to_dict(orient="records"),
        "consensus_rerank_guardrail_bundle_verification_status": str(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty and pd.notna(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("verification_status")) else None,
        "consensus_rerank_guardrail_bundle_verification_failed_rows": int(consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get("failed_files") or 0) if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else 0,
        "consensus_rerank_guardrail_handoff_certificate_available": bool(consensus_rerank_guardrail_handoff_certificate_markdown),
        "consensus_rerank_release_decision_template_rows": int(len(consensus_rerank_release_decision_template_df)),
        "consensus_rerank_release_decision_template": consensus_rerank_release_decision_template_df.to_dict(orient="records"),
        "consensus_rerank_release_decision_rows": int(len(consensus_rerank_release_decision_df)),
        "consensus_rerank_release_decision_status": str(consensus_rerank_release_decision_meta.get("status") or ""),
        "consensus_rerank_release_decision_metadata": consensus_rerank_release_decision_meta,
        "consensus_rerank_release_decisions": consensus_rerank_release_decision_df.to_dict(orient="records"),
        "consensus_rerank_release_decision_validation_rows": int(len(consensus_rerank_release_decision_validation_df)),
        "consensus_rerank_release_decision_blocked_rows": int(
            (consensus_rerank_release_decision_validation_df["validation_status"].astype(str) == "blocked").sum()
        ) if not consensus_rerank_release_decision_validation_df.empty and "validation_status" in consensus_rerank_release_decision_validation_df.columns else 0,
        "consensus_rerank_release_decision_validation": consensus_rerank_release_decision_validation_df.to_dict(orient="records"),
        "consensus_rerank_release_decision_summary": consensus_rerank_release_decision_summary_df.to_dict(orient="records"),
        "consensus_rerank_release_review_status": str(top_consensus_rerank_release_decision_summary.get("release_review_status")) if top_consensus_rerank_release_decision_summary is not None and pd.notna(top_consensus_rerank_release_decision_summary.get("release_review_status")) else "",
        "consensus_rerank_release_allowed": bool(top_consensus_rerank_release_decision_summary.get("release_allowed")) if top_consensus_rerank_release_decision_summary is not None else False,
        "consensus_rerank_release_apply_plan_rows": int(len(consensus_rerank_release_apply_plan_df)),
        "consensus_rerank_release_apply_plan": consensus_rerank_release_apply_plan_df.to_dict(orient="records"),
        "top_consensus_rerank_release_apply_pocket_id": str(top_consensus_rerank_release_apply_plan.get("pocket_id")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("pocket_id")) else None,
        "top_consensus_rerank_release_apply_status": str(top_consensus_rerank_release_apply_plan.get("release_apply_status")) if top_consensus_rerank_release_apply_plan is not None and pd.notna(top_consensus_rerank_release_apply_plan.get("release_apply_status")) else None,
        "consensus_rerank_release_apply_report_available": bool(consensus_rerank_release_apply_report_markdown),
        "consensus_rerank_release_execution_template_rows": int(len(consensus_rerank_release_execution_template_df)),
        "consensus_rerank_release_execution_template": consensus_rerank_release_execution_template_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_receipt_rows": int(len(consensus_rerank_release_execution_receipt_df)),
        "consensus_rerank_release_execution_receipt_status": str(consensus_rerank_release_execution_receipt_meta.get("status") or ""),
        "consensus_rerank_release_execution_receipt_metadata": consensus_rerank_release_execution_receipt_meta,
        "consensus_rerank_release_execution_receipt": consensus_rerank_release_execution_receipt_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_validation_rows": int(len(consensus_rerank_release_execution_validation_df)),
        "consensus_rerank_release_execution_blocked_rows": int(
            (consensus_rerank_release_execution_validation_df["validation_status"].astype(str) == "blocked").sum()
        ) if not consensus_rerank_release_execution_validation_df.empty and "validation_status" in consensus_rerank_release_execution_validation_df.columns else 0,
        "consensus_rerank_release_execution_validation": consensus_rerank_release_execution_validation_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_summary": consensus_rerank_release_execution_summary_df.to_dict(orient="records"),
        "consensus_rerank_release_execution_review_status": str(top_consensus_rerank_release_execution_summary.get("execution_review_status")) if top_consensus_rerank_release_execution_summary is not None and pd.notna(top_consensus_rerank_release_execution_summary.get("execution_review_status")) else "",
        "consensus_rerank_release_execution_complete": bool(top_consensus_rerank_release_execution_summary.get("execution_complete")) if top_consensus_rerank_release_execution_summary is not None else False,
        "consensus_rerank_release_execution_report_available": bool(consensus_rerank_release_execution_report_markdown),
        "consensus_rerank_release_closure_certificate_available": bool(consensus_rerank_release_closure_certificate_markdown),
        "consensus_rerank_release_closure_ledger_rows": int(len(consensus_rerank_release_closure_ledger_df)),
        "consensus_rerank_release_closure_ledger_blocked_rows": int(
            consensus_rerank_release_closure_ledger_df["closure_check"].astype(str).str.lower().isin({"blocked", "missing"}).sum()
        ) if not consensus_rerank_release_closure_ledger_df.empty and "closure_check" in consensus_rerank_release_closure_ledger_df.columns else 0,
        "consensus_rerank_release_closure_ledger": consensus_rerank_release_closure_ledger_df.to_dict(orient="records"),
        "consensus_rerank_release_closure_summary_rows": int(len(consensus_rerank_release_closure_summary_df)),
        "consensus_rerank_release_closure_summary": consensus_rerank_release_closure_summary_df.to_dict(orient="records"),
        "consensus_rerank_release_closure_readiness_status": str(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) if top_consensus_rerank_release_closure_summary is not None and pd.notna(top_consensus_rerank_release_closure_summary.get("closure_readiness_status")) else "",
        "consensus_rerank_release_closed": bool(top_consensus_rerank_release_closure_summary.get("release_closed")) if top_consensus_rerank_release_closure_summary is not None else False,
        "consensus_rerank_release_closure_blocker_rows": int(len(consensus_rerank_release_closure_blocker_df)),
        "top_consensus_rerank_release_closure_blocker_type": str(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) if not consensus_rerank_release_closure_blocker_df.empty and pd.notna(consensus_rerank_release_closure_blocker_df.iloc[0].get("blocker_type")) else "",
        "consensus_rerank_release_closure_blockers": consensus_rerank_release_closure_blocker_df.to_dict(orient="records"),
        "consensus_rerank_release_closure_remediation_checklist_available": bool(consensus_rerank_release_closure_remediation_checklist_markdown),
        "consensus_rerank_release_closure_detached_manifest_rows": int(len(consensus_rerank_release_closure_detached_manifest_df)),
        "consensus_rerank_release_closure_detached_manifest": consensus_rerank_release_closure_detached_manifest_df.to_dict(orient="records"),
        "p2rank_ab_enabled": bool(p2rank_ab_enabled),
        "p2rank_ab_comparison": p2rank_ab_df.to_dict(orient="records"),
        "literature_ab_enabled": bool(literature_ab_enabled),
        "literature_ab_comparison": literature_ab_df.to_dict(orient="records"),
        "evidence_route_enabled": bool(auto_external_evidence_route),
        "evidence_route_min_support": float(external_route_min_support),
        "evidence_route_min_confidence": float(external_route_min_confidence),
        "evidence_route_min_mapping_quality": float(external_route_min_quality),
        "evidence_route_radius": float(external_route_radius) if external_route_radius is not None else None,
        "evidence_route_ab_enabled": bool(evidence_route_ab_enabled),
        "evidence_route_ab_comparison": evidence_route_ab_df.to_dict(orient="records"),
        "conservation_site_rows": int(len(conservation_site_df)),
        "conservation_source_name": str(conservation_site_meta.get("source") or ""),
        "conservation_score_mean": str(conservation_site_meta.get("score_mean") or ""),
        "conservation_ab_enabled": bool(conservation_ab_enabled),
        "conservation_ab_comparison": conservation_ab_df.to_dict(orient="records"),
        "external_exact_rows": int(
            (external_site_df["mapping_level"].astype(str).str.lower() == "exact").sum()
        ) if not external_site_df.empty and "mapping_level" in external_site_df.columns else 0,
        "external_weak_rows": int(
            (external_site_df["mapping_level"].astype(str).str.lower() == "weak").sum()
        ) if not external_site_df.empty and "mapping_level" in external_site_df.columns else 0,
        "interface_rows": int(len(enriched_annotations)),
        "interface_summary": interface_summary.to_dict(orient="records"),
        "overlap_summary": overlap_summary.to_dict(orient="records"),
        "auto_detection_mode": auto_detection_mode,
        "uploaded_pocket_rows": int(len(uploaded_pocket_df)),
        "auto_pocket_rows": int(len(auto_pocket_df)),
        "inferred_annotation_rows": int(len(inferred_annotation_df)),
        "top_pocket_evidence_quality_label": str(top_pocket.get("evidence_quality_label")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_label")) else None,
        "top_pocket_evidence_quality_score": float(top_pocket.get("evidence_quality_score")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_score")) else None,
        "top_pocket_evidence_quality_warning": str(top_pocket.get("evidence_quality_warning")) if top_pocket is not None and pd.notna(top_pocket.get("evidence_quality_warning")) else None,
        "top_pocket_decision_label": str(top_pocket_decision.get("decision_label")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_label")) else None,
        "top_pocket_decision_score": float(top_pocket_decision.get("decision_score")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("decision_score")) else None,
        "top_pocket_audit_status": str(top_pocket_decision.get("audit_status")) if top_pocket_decision is not None and pd.notna(top_pocket_decision.get("audit_status")) else None,
        "top_pocket_reliability_gaps": top_reliability_gaps or None,
        "top_pocket_precision_tier": str(top_pocket_triage.get("precision_tier")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("precision_tier")) else None,
        "top_pocket_triage_action": str(top_pocket_triage.get("triage_action")) if top_pocket_triage is not None and pd.notna(top_pocket_triage.get("triage_action")) else None,
        "top_precision_triage_pocket_id": str(top_precision_triage.get("pocket_id")) if top_precision_triage is not None and pd.notna(top_precision_triage.get("pocket_id")) else None,
        "pocket_decision": pocket_decision_df.to_dict(orient="records"),
        "pocket_reliability": pocket_reliability_df.to_dict(orient="records"),
        "pocket_precision_triage": pocket_triage_df.to_dict(orient="records"),
        "auto_pocket_summary": auto_pocket_summary.to_dict(orient="records"),
        "joint_candidates": joint_candidate_df.to_dict(orient="records"),
    },
)

uploaded_annotation_summary = build_interface_summary(
    enrich_interface_annotations(
        uploaded_annotation_df,
        pocket_residues=effective_pocket_residues,
        hotspot_residues=hotspot_residues,
    )
) if not uploaded_annotation_df.empty else pd.DataFrame()
inferred_annotation_summary = build_interface_summary(
    enrich_interface_annotations(
        inferred_annotation_df,
        pocket_residues=effective_pocket_residues,
        hotspot_residues=hotspot_residues,
    )
) if not inferred_annotation_df.empty else pd.DataFrame()

if not inferred_annotation_df.empty and "inference_basis" in inferred_annotation_df.columns:
    inferred_basis_counts = inferred_annotation_df["inference_basis"].astype(str).value_counts().to_dict()
    inferred_basis_text = "；".join(
        f"{INFERENCE_BASIS_LABELS.get(basis, basis)}: {count}"
        for basis, count in inferred_basis_counts.items()
    )
else:
    inferred_basis_text = ""

pocket_hotspot_df = (
    effective_pocket_df[effective_pocket_df["is_hotspot"].fillna(False).astype(bool)].copy()
    if not effective_pocket_df.empty and "is_hotspot" in effective_pocket_df.columns
    else pd.DataFrame()
)
interface_hotspot_df = (
    enriched_annotations[enriched_annotations["is_hotspot"]].copy()
    if not enriched_annotations.empty and "is_hotspot" in enriched_annotations.columns
    else pd.DataFrame()
)
interface_pocket_df = (
    enriched_annotations[enriched_annotations["is_pocket"]].copy()
    if not enriched_annotations.empty and "is_pocket" in enriched_annotations.columns
    else pd.DataFrame()
)
triple_overlap_df = (
    enriched_annotations[enriched_annotations["is_overlap"]].copy()
    if not enriched_annotations.empty and "is_overlap" in enriched_annotations.columns
    else pd.DataFrame()
)

metric_cols = st.columns(6)
metric_cols[0].metric("有效口袋数", len(effective_pocket_summary) if not effective_pocket_summary.empty else 0)
metric_cols[1].metric("自动口袋数", len(auto_pocket_summary) if not auto_pocket_summary.empty else 0)
metric_cols[2].metric("界面注释数", len(enriched_annotations))
metric_cols[3].metric("三重交集", len(triple_overlap_df))
metric_cols[4].metric("热点数", len(hotspot_df))
metric_cols[5].metric("蛋白体积（估算）", f"{protein_volume:,.1f} A³" if protein_volume is not None else "-")

st.caption(
    f"当前主分析使用口袋来源：{POCKET_SOURCE_LABELS.get(effective_pocket_mode, effective_pocket_mode)}；"
    f"界面来源：{ANNOTATION_SOURCE_LABELS.get(effective_annotation_mode, effective_annotation_mode)}；"
    f"热点判定：ΔG ≤ -{hotspot_threshold:.1f}，至少保留 {hotspot_top_n} 个残基。"
)
if inferred_basis_text:
    st.caption(f"结构推断界面依据：{inferred_basis_text}")
if top_joint_candidate is not None:
    st.caption(
        f"联合推荐 Top1：{top_joint_candidate['pocket_id']} / {top_joint_candidate['recommendation_label']} / {top_joint_candidate['recommendation_reason']}"
    )
st.markdown(analysis_text)
_render_pocket_decision_panel(pocket_decision_df, pocket_reliability_df, pocket_triage_df)
if not ai_evidence_audit_df.empty:
    with st.expander("AI evidence audit", expanded=False):
        st.caption(
            "AI residues are audited against non-AI evidence, source snippets, mapping confidence, and structure identity before they should be trusted."
        )
        st.dataframe(ai_evidence_audit_df, use_container_width=True, hide_index=True)
if not residue_evidence_consensus_df.empty:
    with st.expander("Residue evidence consensus", expanded=False):
        st.caption(
            "Aggregates external, literature, AI, and conservation residue evidence into one anchor-level table for precision review."
        )
        st.dataframe(residue_evidence_consensus_df, use_container_width=True, hide_index=True)
if not pocket_consensus_coverage_df.empty:
    with st.expander("Pocket consensus coverage", expanded=False):
        st.caption(
            "Maps residue-level consensus anchors back onto each pocket without changing the pocket ranking."
        )
        st.dataframe(pocket_consensus_coverage_df, use_container_width=True, hide_index=True)
if not benchmark_reference_candidate_df.empty:
    with st.expander("Benchmark reference candidate from external evidence", expanded=False):
        st.caption(
            "Generated from currently loaded UniProt/M-CSA/literature/AI residue evidence. Curate this table before using it as an independent benchmark."
        )
        if not benchmark_reference_import_summary_df.empty:
            st.dataframe(benchmark_reference_import_summary_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_candidate_review_queue_df.empty:
            st.caption("Review queue: fix these items before promoting candidates into a curated benchmark reference.")
            st.dataframe(benchmark_reference_candidate_review_queue_df, use_container_width=True, hide_index=True)
            if benchmark_reference_candidate_review_checklist_markdown:
                with st.expander("Benchmark reference candidate review checklist", expanded=False):
                    st.markdown(benchmark_reference_candidate_review_checklist_markdown)
        if not benchmark_reference_candidate_review_decision_template_df.empty:
            st.caption("Decision template: fill review_decision, reviewer and verification evidence, then upload it back.")
            st.dataframe(benchmark_reference_candidate_review_decision_template_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_candidate_review_decision_validation_df.empty:
            st.caption("Decision validation: blocked rows must be fixed before actions can promote references.")
            st.dataframe(benchmark_reference_candidate_review_decision_validation_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_candidate_review_outcome_df.empty:
            st.caption("Decision outcomes: accepted actions can promote candidate residues only when every risk action for that residue is accepted.")
            st.dataframe(benchmark_reference_candidate_review_outcome_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_candidate_accepted_df.empty:
            st.caption("Accepted reference candidates: clean rows plus candidate residues whose review actions were fully accepted.")
            st.dataframe(benchmark_reference_candidate_accepted_df, use_container_width=True, hide_index=True)
        st.dataframe(benchmark_reference_candidate_df, use_container_width=True, hide_index=True)
if not benchmark_reference_source_audit_df.empty:
    with st.expander("Benchmark reference source audit", expanded=False):
        st.caption(
            "Audits the final reference rows used for benchmark scoring, including source mode and whether they can support independent precision claims."
        )
        if not benchmark_reference_source_audit_summary_df.empty:
            st.dataframe(benchmark_reference_source_audit_summary_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_summary_df.empty:
            st.caption("Source audit case summary: groups source-only blockers/review needs by benchmark_id.")
            st.dataframe(benchmark_reference_source_audit_case_summary_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_template_df.empty:
            st.caption("Source audit case decision template: fill one decision per affected benchmark_id before treating coverage as precision.")
            st.dataframe(benchmark_reference_source_audit_case_decision_template_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_df.empty:
            st.caption("Source audit case decisions: normalized reviewer decisions uploaded for source-risk closure.")
            st.dataframe(benchmark_reference_source_audit_case_decision_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_validation_df.empty:
            st.caption("Source audit case decision validation: blocked rows must be fixed before clearing source-risk cases.")
            st.dataframe(benchmark_reference_source_audit_case_decision_validation_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_outcome_summary_df.empty:
            st.caption("Source audit case decision outcome summary: closure status, open cases, and next action for source-risk decisions.")
            st.dataframe(benchmark_reference_source_audit_case_decision_outcome_summary_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty:
            st.caption("Source audit case decision closure queue: machine-readable open case actions after decision outcomes.")
            st.dataframe(benchmark_reference_source_audit_case_decision_closure_queue_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty:
            st.caption("Source audit case decision readiness impact summary: aggregate cleared/open readiness effects after decisions.")
            st.dataframe(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df, use_container_width=True, hide_index=True)
        if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty:
            st.caption("Source audit case decision readiness impact: original source issue versus decision-adjusted readiness issue by case.")
            st.dataframe(benchmark_reference_source_audit_case_decision_readiness_impact_df, use_container_width=True, hide_index=True)
        if benchmark_reference_source_audit_case_decision_closure_checklist_markdown:
            with st.expander("Benchmark reference source audit decision closure checklist", expanded=False):
                st.markdown(benchmark_reference_source_audit_case_decision_closure_checklist_markdown)
        if not benchmark_reference_source_audit_case_decision_outcome_df.empty:
            st.caption("Source audit case decision outcomes: applied status after validation for each source-risk case.")
            st.dataframe(benchmark_reference_source_audit_case_decision_outcome_df, use_container_width=True, hide_index=True)
        if benchmark_reference_source_audit_case_checklist_markdown:
            with st.expander("Benchmark reference source audit case checklist", expanded=False):
                st.markdown(benchmark_reference_source_audit_case_checklist_markdown)
        if not benchmark_reference_source_audit_action_queue_df.empty:
            st.caption("Source audit action queue: source-only remediation actions extracted from non-ready benchmark reference sources.")
            st.dataframe(benchmark_reference_source_audit_action_queue_df, use_container_width=True, hide_index=True)
        if benchmark_reference_source_audit_checklist_markdown:
            with st.expander("Benchmark reference source audit checklist", expanded=False):
                st.markdown(benchmark_reference_source_audit_checklist_markdown)
        st.dataframe(benchmark_reference_source_audit_df, use_container_width=True, hide_index=True)
if not pocket_benchmark_summary_df.empty:
    with st.expander("Catalytic pocket benchmark", expanded=True):
        st.caption(
            "Compares the current ranked pockets against uploaded curated catalytic residues. This is an evaluation layer only and does not change ranking."
        )
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Top-1 coverage",
            f"{float(top1_benchmark.get('coverage_ratio') or 0.0):.2f}" if top1_benchmark is not None else "-",
            str(top1_benchmark.get("benchmark_status") or "") if top1_benchmark is not None else "",
        )
        metric_cols[1].metric(
            "Top-3 coverage",
            f"{float(top3_benchmark.get('coverage_ratio') or 0.0):.2f}" if top3_benchmark is not None else "-",
            str(top3_benchmark.get("benchmark_status") or "") if top3_benchmark is not None else "",
        )
        metric_cols[2].metric(
            "Best hit rank",
            str(top3_benchmark.get("best_rank") or "-") if top3_benchmark is not None else "-",
            str(top3_benchmark.get("best_pocket_id") or "") if top3_benchmark is not None else "",
        )
        if not pocket_benchmark_reference_quality_issue_df.empty:
            st.caption("Benchmark reference curation quality: review P1/P2 rows before trusting benchmark coverage.")
            if not pocket_benchmark_reference_quality_summary_df.empty:
                st.dataframe(pocket_benchmark_reference_quality_summary_df, use_container_width=True, hide_index=True)
            st.dataframe(pocket_benchmark_reference_quality_issue_df, use_container_width=True, hide_index=True)
            if pocket_benchmark_reference_quality_checklist_markdown:
                with st.expander("Benchmark reference curation checklist", expanded=False):
                    st.markdown(pocket_benchmark_reference_quality_checklist_markdown)
        if not pocket_benchmark_reference_structure_validation_df.empty:
            st.caption(
                "Benchmark reference structure validation: verify reference residues against the uploaded PDB before treating misses as detection failures."
            )
            if not pocket_benchmark_reference_structure_validation_summary_df.empty:
                st.dataframe(pocket_benchmark_reference_structure_validation_summary_df, use_container_width=True, hide_index=True)
            st.dataframe(pocket_benchmark_reference_structure_validation_df, use_container_width=True, hide_index=True)
            if pocket_benchmark_reference_structure_validation_checklist_markdown:
                with st.expander("Benchmark reference structure validation checklist", expanded=False):
                    st.markdown(pocket_benchmark_reference_structure_validation_checklist_markdown)
        if not pocket_benchmark_reference_readiness_summary_df.empty:
            st.caption("Benchmark reference readiness gate: one combined decision before using coverage as an accuracy claim.")
            st.dataframe(pocket_benchmark_reference_readiness_summary_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_reference_readiness_case_summary_df.empty:
                st.dataframe(pocket_benchmark_reference_readiness_case_summary_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_reference_readiness_queue_df.empty:
                st.dataframe(pocket_benchmark_reference_readiness_queue_df, use_container_width=True, hide_index=True)
            if pocket_benchmark_reference_readiness_checklist_markdown:
                with st.expander("Benchmark reference readiness checklist", expanded=False):
                    st.markdown(pocket_benchmark_reference_readiness_checklist_markdown)
        if not pocket_benchmark_interpretation_df.empty:
            st.caption("Benchmark interpretation: combines Top-N coverage with reference readiness before any accuracy claim.")
            st.dataframe(pocket_benchmark_interpretation_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_case_interpretation_df.empty:
            st.caption("Benchmark case interpretation: combines case-level Top-N coverage with case-level readiness.")
            st.dataframe(pocket_benchmark_case_interpretation_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_case_interpretation_matrix_df.empty:
            st.caption("Benchmark case interpretation matrix: one row per benchmark_id with Top-1/Top-3/Top-5 claim status and coverage.")
            if not pocket_benchmark_case_interpretation_matrix_summary_df.empty:
                st.dataframe(pocket_benchmark_case_interpretation_matrix_summary_df, use_container_width=True, hide_index=True)
            if not pocket_benchmark_case_interpretation_matrix_queue_df.empty:
                st.caption("Benchmark case interpretation matrix queue: one action per non-claim-ready case.")
                st.dataframe(pocket_benchmark_case_interpretation_matrix_queue_df, use_container_width=True, hide_index=True)
            st.dataframe(pocket_benchmark_case_interpretation_matrix_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_dataset_interpretation_df.empty:
            st.caption("Benchmark dataset interpretation: aggregates claim-ready, blocked and review-needed cases per Top-N.")
            st.dataframe(pocket_benchmark_dataset_interpretation_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty:
            st.caption("Benchmark source-audit decision dataset impact: Top-N source-decision effects on dataset claim readiness.")
            st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty:
            st.caption("Benchmark source-audit decision dataset impact cases: per-case Top-N source-decision effects and gate mismatches.")
            st.dataframe(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df, use_container_width=True, hide_index=True)
            if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown:
                with st.expander("Benchmark source-audit decision dataset impact case checklist", expanded=False):
                    st.markdown(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown)
        if not pocket_benchmark_dataset_interpretation_queue_df.empty:
            st.caption("Benchmark dataset interpretation queue: non-claim-ready cases that block or weaken dataset-level precision claims.")
            st.dataframe(pocket_benchmark_dataset_interpretation_queue_df, use_container_width=True, hide_index=True)
            if pocket_benchmark_dataset_interpretation_checklist_markdown:
                with st.expander("Benchmark dataset interpretation checklist", expanded=False):
                    st.markdown(pocket_benchmark_dataset_interpretation_checklist_markdown)
        if pocket_benchmark_dataset_interpretation_report_markdown:
            with st.expander("Benchmark dataset interpretation report", expanded=False):
                st.markdown(pocket_benchmark_dataset_interpretation_report_markdown)
        st.dataframe(pocket_benchmark_summary_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_dataset_summary_df.empty:
            st.caption("Benchmark dataset summary: case-level aggregation prevents large catalytic sets from dominating accuracy.")
            st.dataframe(pocket_benchmark_dataset_summary_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_case_summary_df.empty:
            st.caption("Benchmark case summary: Top-N coverage is split by benchmark_id/case_id when provided.")
            st.dataframe(pocket_benchmark_case_summary_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_variant_comparison_df.empty:
            st.caption(
                "Benchmark variant comparison: negative coverage_delta or positive coverage_loss means removing that evidence path hurts catalytic coverage."
            )
            st.dataframe(pocket_benchmark_variant_comparison_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_variant_dataset_comparison_df.empty:
            st.caption("Benchmark variant dataset comparison: coverage loss aggregated across benchmark cases.")
            st.dataframe(pocket_benchmark_variant_dataset_comparison_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_variant_case_comparison_df.empty:
            st.caption("Benchmark variant case comparison: inspect which cases lose coverage when an evidence path is removed.")
            st.dataframe(pocket_benchmark_variant_case_comparison_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_variant_detail_comparison_df.empty:
            st.caption("Benchmark variant residue comparison: exact catalytic residues lost, gained or unchanged for each ablation.")
            st.dataframe(pocket_benchmark_variant_detail_comparison_df, use_container_width=True, hide_index=True)
        if not pocket_benchmark_variant_remediation_df.empty:
            st.caption("Benchmark remediation queue: lost/current-missed catalytic residues converted into review actions.")
            if not pocket_benchmark_variant_remediation_summary_df.empty:
                st.dataframe(pocket_benchmark_variant_remediation_summary_df, use_container_width=True, hide_index=True)
            st.dataframe(pocket_benchmark_variant_remediation_df, use_container_width=True, hide_index=True)
            if pocket_benchmark_variant_remediation_checklist_markdown:
                with st.expander("Benchmark remediation checklist", expanded=False):
                    st.markdown(pocket_benchmark_variant_remediation_checklist_markdown)
        if not pocket_benchmark_details_df.empty:
            st.dataframe(pocket_benchmark_details_df, use_container_width=True, hide_index=True)
if not consensus_rerank_suggestion_df.empty:
    with st.expander("Consensus rerank suggestions", expanded=False):
        st.caption(
            "Conservative suggestions for whether consensus evidence should promote, keep, demote, or review a pocket. This does not change ranking automatically."
        )
        st.dataframe(consensus_rerank_suggestion_df, use_container_width=True, hide_index=True)
if not consensus_rerank_preview_df.empty:
    with st.expander("Consensus rerank preview", expanded=False):
        st.caption(
            "Simulates conservative score adjustments from consensus suggestions. This is a preview only and does not replace the active ranking."
        )
        st.dataframe(consensus_rerank_preview_df, use_container_width=True, hide_index=True)
if not consensus_rerank_policy_gate_df.empty:
    with st.expander("Consensus rerank policy gate", expanded=False):
        st.caption(
            "One-row safety gate for deciding whether the consensus rerank preview is ready to apply, needs review, or should remain blocked."
        )
        st.dataframe(consensus_rerank_policy_gate_df, use_container_width=True, hide_index=True)
if not consensus_rerank_action_queue_df.empty:
    with st.expander("Consensus rerank action queue", expanded=False):
        st.caption(
            "Actionable fixes required before consensus rerank can be trusted or enabled."
        )
        st.dataframe(consensus_rerank_action_queue_df, use_container_width=True, hide_index=True)
if not consensus_rerank_apply_simulation_df.empty:
    with st.expander("Consensus rerank apply simulation", expanded=False):
        st.caption(
            "Non-destructive simulation of the ranking after conservative consensus rerank rules. Blocked evidence remains diagnostic until fixed."
        )
        st.dataframe(consensus_rerank_apply_simulation_df, use_container_width=True, hide_index=True)
if not consensus_rerank_simulation_delta_df.empty:
    with st.expander("Consensus rerank simulation delta", expanded=False):
        st.caption(
            "Explains why each pocket moved, stayed, or was frozen in the non-destructive apply simulation."
        )
        st.dataframe(consensus_rerank_simulation_delta_df, use_container_width=True, hide_index=True)
if not consensus_rerank_precision_scorecard_df.empty:
    with st.expander("Consensus rerank precision scorecard", expanded=False):
        st.caption(
            "One-row summary of whether the simulated rerank likely improves precision, remains blocked, or should stay diagnostic."
        )
        st.dataframe(consensus_rerank_precision_scorecard_df, use_container_width=True, hide_index=True)
if not consensus_rerank_precision_guardrail_df.empty:
    with st.expander("Consensus rerank precision guardrail", expanded=False):
        st.caption(
            "Go/no-go guardrail for whether consensus rerank can be applied, needs manual review, or must remain diagnostic."
        )
        st.dataframe(consensus_rerank_precision_guardrail_df, use_container_width=True, hide_index=True)
if consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty:
    with st.expander("Consensus rerank precision guardrail report", expanded=False):
        st.caption(
            "Markdown handoff report for guardrail decision, scorecard counters, clearances, and release checklist."
        )
        st.markdown(consensus_rerank_precision_guardrail_report_markdown)
if not consensus_rerank_guardrail_artifact_manifest_df.empty:
    with st.expander("Consensus rerank guardrail artifact manifest", expanded=False):
        st.caption(
            "Integrity manifest for the rerank guardrail handoff ZIP, including byte size and SHA-256 for each artifact."
        )
        st.dataframe(consensus_rerank_guardrail_artifact_manifest_df, use_container_width=True, hide_index=True)
if not consensus_rerank_guardrail_bundle_verification_summary_df.empty:
    with st.expander("Consensus rerank guardrail bundle verification summary", expanded=False):
        st.caption(
            "One-row integrity check for the handoff ZIP against the manifest."
        )
        st.dataframe(consensus_rerank_guardrail_bundle_verification_summary_df, use_container_width=True, hide_index=True)
if not consensus_rerank_guardrail_bundle_verification_df.empty:
    with st.expander("Consensus rerank guardrail bundle verification", expanded=False):
        st.caption(
            "Per-file ZIP verification using manifest byte size and SHA-256."
        )
        st.dataframe(consensus_rerank_guardrail_bundle_verification_df, use_container_width=True, hide_index=True)
if consensus_rerank_guardrail_handoff_certificate_markdown:
    with st.expander("Consensus rerank guardrail handoff certificate", expanded=False):
        st.caption(
            "Detached certificate for the handoff ZIP identity, verification result, and guardrail release decision."
        )
        st.markdown(consensus_rerank_guardrail_handoff_certificate_markdown)
if not consensus_rerank_release_decision_template_df.empty:
    with st.expander("Consensus rerank release decision template", expanded=False):
        st.caption(
            "Reviewer sign-off template for release decision, blocker clearance, and key simulated rank changes."
        )
        st.dataframe(consensus_rerank_release_decision_template_df, use_container_width=True, hide_index=True)
if uploaded_consensus_rerank_release_decisions is not None and consensus_rerank_release_decision_df.empty:
    st.warning(
        f"Consensus rerank release decision upload was not usable: {consensus_rerank_release_decision_meta.get('status') or 'unknown'}."
    )
if not consensus_rerank_release_decision_summary_df.empty:
    with st.expander("Consensus rerank release decision summary", expanded=False):
        st.caption(
            "One-row review result for the uploaded release decision CSV. Rerank is only releasable when this summary allows it."
        )
        st.dataframe(consensus_rerank_release_decision_summary_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_apply_plan_df.empty:
    with st.expander("Consensus rerank release apply plan", expanded=False):
        st.caption(
            "Approved manual rank order. This table is only generated when release review is approved and the current apply simulation is clean."
        )
        st.dataframe(consensus_rerank_release_apply_plan_df, use_container_width=True, hide_index=True)
if consensus_rerank_release_apply_report_markdown:
    with st.expander("Consensus rerank release apply report", expanded=False):
        st.caption(
            "Markdown execution worksheet for the approved manual rank order, including gate status, hash, order, and pre-apply checks."
        )
        st.markdown(consensus_rerank_release_apply_report_markdown)
if not consensus_rerank_release_execution_template_df.empty:
    with st.expander("Consensus rerank release execution template", expanded=False):
        st.caption(
            "Operator receipt template for recording whether each approved manual rank was actually applied."
        )
        st.dataframe(consensus_rerank_release_execution_template_df, use_container_width=True, hide_index=True)
if uploaded_consensus_rerank_release_execution_receipt is not None and consensus_rerank_release_execution_receipt_df.empty:
    st.warning(
        f"Consensus rerank release execution receipt upload was not usable: {consensus_rerank_release_execution_receipt_meta.get('status') or 'unknown'}."
    )
if not consensus_rerank_release_execution_summary_df.empty:
    with st.expander("Consensus rerank release execution summary", expanded=False):
        st.caption(
            "One-row operational receipt result. Execution is complete only when every row was applied exactly as approved."
        )
        st.dataframe(consensus_rerank_release_execution_summary_df, use_container_width=True, hide_index=True)
if consensus_rerank_release_execution_report_markdown:
    with st.expander("Consensus rerank release execution report", expanded=False):
        st.caption(
            "Markdown operational receipt report for execution status, receipt hash, operators, row outcomes, and archival checks."
        )
        st.markdown(consensus_rerank_release_execution_report_markdown)
if consensus_rerank_release_closure_certificate_markdown:
    with st.expander("Consensus rerank release closure certificate", expanded=False):
        st.caption(
            "Detached closure certificate tying approved apply plan, release review, execution receipt, execution report, and final closure status."
        )
        st.markdown(consensus_rerank_release_closure_certificate_markdown)
if not consensus_rerank_release_closure_summary_df.empty:
    with st.expander("Consensus rerank release closure readiness summary", expanded=False):
        st.caption(
            "One-row release closure gate combining ledger completeness with handoff ZIP verification."
        )
        st.dataframe(consensus_rerank_release_closure_summary_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_closure_blocker_df.empty:
    with st.expander("Consensus rerank release closure blocker queue", expanded=False):
        st.caption(
            "Actionable blockers extracted from closure ledger and handoff ZIP verification. Resolve these before release closure."
        )
        st.dataframe(consensus_rerank_release_closure_blocker_df, use_container_width=True, hide_index=True)
if consensus_rerank_release_closure_remediation_checklist_markdown:
    with st.expander("Consensus rerank release closure remediation checklist", expanded=False):
        st.caption(
            "Human repair checklist generated from closure blockers. Re-run closure readiness after every checked fix."
        )
        st.markdown(consensus_rerank_release_closure_remediation_checklist_markdown)
if not consensus_rerank_release_closure_detached_manifest_df.empty:
    with st.expander("Consensus rerank release closure detached manifest", expanded=False):
        st.caption(
            "Detached manifest for ZIP-external closure artifacts produced after handoff ZIP verification."
        )
        st.dataframe(consensus_rerank_release_closure_detached_manifest_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_closure_ledger_df.empty:
    with st.expander("Consensus rerank release closure ledger", expanded=False):
        st.caption(
            "Machine-readable closure evidence ledger with status, row count, byte size, SHA-256, and closure check for each release artifact."
        )
        st.dataframe(consensus_rerank_release_closure_ledger_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_execution_validation_df.empty:
    with st.expander("Consensus rerank release execution validation", expanded=False):
        st.caption(
            "Per-row execution receipt validation for rank match, operator, timestamp, template item, and approved apply-plan hash."
        )
        st.dataframe(consensus_rerank_release_execution_validation_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_execution_receipt_df.empty:
    with st.expander("Consensus rerank release execution receipt uploaded", expanded=False):
        st.caption(
            "Normalized operator-uploaded execution receipt rows parsed from CSV/TSV."
        )
        st.dataframe(consensus_rerank_release_execution_receipt_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_decision_validation_df.empty:
    with st.expander("Consensus rerank release decision validation", expanded=False):
        st.caption(
            "Per-row validation for reviewer, source evidence, anchor residues, blocker clearance, template matching, and guardrail permission."
        )
        st.dataframe(consensus_rerank_release_decision_validation_df, use_container_width=True, hide_index=True)
if not consensus_rerank_release_decision_df.empty:
    with st.expander("Consensus rerank release decisions uploaded", expanded=False):
        st.caption(
            "Normalized reviewer-uploaded release decisions parsed from CSV/TSV."
        )
        st.dataframe(consensus_rerank_release_decision_df, use_container_width=True, hide_index=True)
if consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty:
    with st.expander("Consensus rerank action checklist", expanded=False):
        st.caption(
            "Markdown handoff checklist for validating rerank blockers, anchor residues, and apply readiness."
        )
        st.markdown(consensus_rerank_action_checklist_markdown)
if not ai_review_decision_df.empty:
    with st.expander("AI review decisions applied", expanded=False):
        st.caption(
            "Manual decisions are applied conservatively: accepts need verified source and snippet, and structure conflicts remain blocked."
        )
        st.dataframe(ai_review_decision_df, use_container_width=True, hide_index=True)
if not ai_review_round_summary_df.empty:
    with st.expander("AI review round summary", expanded=False):
        st.caption(
            "One-row summary of this manual-review upload: whether it is blocked, needs more review, or was safely applied."
        )
        st.dataframe(ai_review_round_summary_df, use_container_width=True, hide_index=True)
if not ai_review_ranking_delta_df.empty:
    with st.expander("AI review ranking delta", expanded=False):
        st.caption(
            "Compares ranking-gated AI residues before and after the manual-review upload."
        )
        st.dataframe(ai_review_ranking_delta_df, use_container_width=True, hide_index=True)
if ai_review_round_report_markdown:
    with st.expander("AI review round report", expanded=False):
        st.markdown(ai_review_round_report_markdown)
if not ai_review_artifact_manifest_df.empty:
    with st.expander("AI review artifact manifest", expanded=False):
        st.caption(
            "Index of generated AI review artifacts, including file names, row counts, status, purpose, and recommended use."
        )
        st.dataframe(ai_review_artifact_manifest_df, use_container_width=True, hide_index=True)
if ai_review_bundle_readme_markdown:
    with st.expander("AI review bundle README", expanded=False):
        st.markdown(ai_review_bundle_readme_markdown)
if ai_review_artifact_bundle_zip:
    st.caption("AI review artifact bundle is ready for one-click export from the Export tab.")
if not ai_review_bundle_verification_df.empty:
    with st.expander("AI review bundle verification", expanded=False):
        st.caption(
            "Automatic ZIP self-check against manifest byte size and SHA-256 values."
        )
        st.dataframe(ai_review_bundle_verification_df, use_container_width=True, hide_index=True)
if not ai_review_bundle_verification_summary_df.empty:
    with st.expander("AI review bundle verification summary", expanded=False):
        st.caption(
            "One-row bundle integrity status derived from ZIP verification results."
        )
        st.dataframe(ai_review_bundle_verification_summary_df, use_container_width=True, hide_index=True)
if ai_review_bundle_certificate_markdown:
    with st.expander("AI review bundle handoff certificate", expanded=False):
        st.markdown(ai_review_bundle_certificate_markdown)
if not ai_review_decision_validation_df.empty:
    with st.expander("AI review decision validation", expanded=False):
        st.caption(
            "Pre-apply checks for duplicate, conflicting, unmatched, or under-sourced manual decisions. Conflicting duplicate rows are not applied."
        )
        st.dataframe(ai_review_decision_validation_df, use_container_width=True, hide_index=True)
if not ai_review_decision_outcome_df.empty:
    with st.expander("AI review decision outcomes", expanded=False):
        st.caption(
            "Per-row feedback for uploaded review decisions, including accepted, rejected, blocked, missing-source, and unmatched decisions."
        )
        st.dataframe(ai_review_decision_outcome_df, use_container_width=True, hide_index=True)
if not ai_evidence_review_queue_df.empty:
    with st.expander("AI evidence review queue", expanded=False):
        st.caption(
            "Actionable queue for AI residues that cannot safely increase ranking confidence yet."
        )
        st.dataframe(ai_evidence_review_queue_df, use_container_width=True, hide_index=True)
if not ai_ranking_impact_df.empty:
    with st.expander("AI ranking impact summary", expanded=False):
        st.caption(
            "This summary separates AI evidence collection from actual ranking influence, so excluded AI rows cannot silently affect Top pocket interpretation."
        )
        st.dataframe(ai_ranking_impact_df, use_container_width=True, hide_index=True)
if not ai_followup_plan_df.empty:
    with st.expander("AI follow-up evidence plan", expanded=False):
        st.caption(
            "Use this plan to collect the next round of literature/database evidence. The generated prompt expects retrieved source text, not unsupported model memory."
        )
        st.dataframe(ai_followup_plan_df, use_container_width=True, hide_index=True)

tab_auto, tab_overview, tab_annotations, tab_overlap, tab_export = st.tabs(["自动识别", "总览", "界面注释", "交集分析", "导出"])

with tab_auto:
    st.subheader("自动口袋识别")
    st.caption("无需上传 Pocket CSV，系统会基于 PDB 坐标自动筛入口袋候选；若结构中含有 HETATM，则会优先尝试配体邻域识别。")
    st.caption(f"当前自动口袋结果{'已' if effective_pocket_mode in {'auto', 'combined'} else '未'}接入界面主分析。")
    auto_metric_cols = st.columns(4)
    auto_metric_cols[0].metric("自动口袋数", len(auto_pocket_summary) if not auto_pocket_summary.empty else 0)
    auto_metric_cols[1].metric("候选残基数", len(auto_pocket_df) if not auto_pocket_df.empty else 0)
    auto_metric_cols[2].metric("热点重叠", int(auto_pocket_df["is_hotspot"].sum()) if not auto_pocket_df.empty and "is_hotspot" in auto_pocket_df.columns else 0)
    auto_metric_cols[3].metric("识别策略", "配体优先" if auto_detection_mode == "auto" else "纯几何")

    auto_overlay_hotspots = st.checkbox("叠加热点残基（仅当前口袋）", value=True, key="pocket_interface_auto_overlay_hotspots")
    auto_view_mode = st.radio(
        "自动口袋展示模式",
        ["surface", "cartoon", "sticks"],
        index=0,
        format_func=lambda x: {"surface": "表面", "cartoon": "卡通", "sticks": "球棍"}[x],
        key="pocket_interface_auto_view_mode",
    )
    auto_show_backbone = st.checkbox("显示主链", value=True, disabled=auto_view_mode != "cartoon", key="pocket_interface_auto_show_backbone")
    auto_surface_opacity = st.slider(
        "表面透明度",
        0.0,
        1.0,
        SETTINGS.default_opacity,
        0.05,
        disabled=auto_view_mode != "surface",
        key="pocket_interface_auto_surface_opacity",
    )

    selected_auto_pocket_id = None
    selected_auto_rows = pd.DataFrame()
    if not auto_pocket_summary.empty:
        auto_pocket_ids = auto_pocket_summary["pocket_id"].astype(str).tolist()
        selected_auto_pocket_id = st.selectbox("查看自动口袋", auto_pocket_ids, index=0, key="pocket_interface_selected_auto_pocket")
        if not auto_pocket_df.empty:
            selected_auto_rows = auto_pocket_df[auto_pocket_df["pocket_id"].astype(str) == selected_auto_pocket_id].copy()
            if selected_auto_rows.empty and auto_pocket_ids:
                selected_auto_pocket_id = auto_pocket_ids[0]
                selected_auto_rows = auto_pocket_df[auto_pocket_df["pocket_id"].astype(str) == selected_auto_pocket_id].copy()
        if not selected_auto_rows.empty:
            selected_auto_rows = add_pocket_residue_layers(selected_auto_rows)

    view_col, detail_col = st.columns([2.3, 1.0])

    with view_col:
        selected_pocket_residues = []
        selected_focus_row = None
        if selected_auto_pocket_id and not selected_auto_rows.empty:
            selected_pocket_residues = [(row.chain, int(row.resid)) for row in selected_auto_rows.itertuples(index=False)]
            selected_focus_row = selected_auto_rows.iloc[0]

        render_atom_df = atom_df
        if not render_atom_df.empty and "record_type" in render_atom_df.columns:
            protein_render_atom_df = render_atom_df[render_atom_df["record_type"].astype(str).str.upper() == "ATOM"].copy()
            if not protein_render_atom_df.empty:
                render_atom_df = protein_render_atom_df

        hotspot_residues_for_view = hotspot_residues if (auto_overlay_hotspots and selected_pocket_residues) else []
        auto_view_table = build_auto_pocket_display_table(
            render_atom_df,
            selected_pocket_residues,
            hotspot_residues=hotspot_residues_for_view,
            pocket_id=selected_auto_pocket_id,
            limit_hotspots_to_pocket=True,
            pocket_residue_layers=selected_auto_rows,
        )
        if auto_view_table.empty:
            st.info("当前没有可渲染的自动口袋结构。")
        else:
            focus_chain = None if auto_view_mode == "surface" else (getattr(selected_focus_row, "chain", None) if selected_focus_row is not None else None)
            focus_resid = None if auto_view_mode == "surface" else (int(getattr(selected_focus_row, "resid", 0)) if selected_focus_row is not None else None)
            viewer = build_view(
                pdb_text=pdb_text,
                energy_table=auto_view_table,
                threshold=0.0,
                display_mode=auto_view_mode,
                show_backbone=auto_show_backbone,
                opacity=auto_surface_opacity if auto_view_mode == "surface" else 0.85,
                selected_chain=focus_chain,
                selected_resid=focus_resid,
                color_mode="按口袋识别",
                surface_single_color=False,
                surface_uniform_color=SETTINGS.neutral_color,
                viewer_width=max(680, SETTINGS.viewer_width - 120),
                viewer_height=max(520, SETTINGS.viewer_height - 120),
            )
            st.components.v1.html(viewer._make_html(), height=max(520, SETTINGS.viewer_height - 120) + 20, scrolling=False)
            st.caption("仅当前选中口袋显色：蓝色为当前口袋，红色为当前口袋内热点，灰色为背景。")

    with detail_col:
        if auto_pocket_summary.empty:
            st.info("当前没有识别出自动口袋。你可以试着增大聚类距离、提高候选残基比例，或者上传含配体的结构。")
        else:
            selected_summary = auto_pocket_summary[auto_pocket_summary["pocket_id"].astype(str) == selected_auto_pocket_id].copy()
            st.subheader("当前口袋摘要")
            st.dataframe(selected_summary, use_container_width=True, hide_index=True)

            if not selected_auto_rows.empty:
                st.subheader("当前口袋残基")
                residue_columns = [
                    column
                    for column in [
                        "residue_label",
                        "pocket_layer",
                        "pocket_layer_score",
                        "pocket_layer_reason",
                        "residue_score",
                        "score",
                        "detection_route",
                        "consensus_methods",
                        "method_vote_count",
                        "smart_rank_score",
                        "smart_rank_label",
                        "smart_evidence_anchor_support",
                        "smart_evidence_anchor_risk",
                        "external_direct_anchor",
                        "evidence_route_anchor",
                        "evidence_anchor_distance",
                        "evidence_anchor_proximity",
                        "evidence_anchor_residue",
                        "external_structure_verified",
                        "external_mapping_quality",
                        "external_direct_sources",
                        "external_evidence_types",
                        "external_evidence_notes",
                        "conservation_support",
                        "conservation_confidence",
                        "contact_count",
                        "center_distance",
                        "ligand_contact_count",
                        "is_hotspot",
                    ]
                    if column in selected_auto_rows.columns
                ]
                if residue_columns:
                    st.dataframe(selected_auto_rows[residue_columns], use_container_width=True, hide_index=True)

                st.download_button(
                    "导出当前口袋残基 CSV",
                    data=_to_csv_bytes(selected_auto_rows),
                    file_name=f"{selected_auto_pocket_id}_residues.csv",
                    mime="text/csv",
                )

            st.download_button(
                "导出自动口袋汇总 CSV",
                data=_to_csv_bytes(auto_pocket_summary),
                file_name="auto_pocket_summary.csv",
                mime="text/csv",
            )
            st.download_button(
                "导出自动口袋明细 CSV",
                data=_to_csv_bytes(auto_pocket_df),
                file_name="auto_pocket_candidates.csv",
                mime="text/csv",
            )

        with st.expander("自动检测诊断", expanded=False):
            methods_used_text = str(auto_detection_summary.get("auto_detection_methods_used") or "-")
            status_text = str(auto_detection_summary.get("auto_detection_status_summary") or "-")
            st.caption(f"Methods: {methods_used_text}")
            st.caption(f"Status: {status_text}")
            p2rank_status_text = str(auto_detection_summary.get("auto_detection_p2rank_status") or "").strip()
            if p2rank_status_text:
                st.caption(
                    f"P2Rank: {p2rank_status_text} / pred {int(auto_detection_summary.get('auto_detection_p2rank_prediction_rows', 0) or 0)} / "
                    f"res {int(auto_detection_summary.get('auto_detection_p2rank_residue_rows', 0) or 0)}"
                )
            if int(auto_detection_summary.get("auto_detection_external_rows", 0) or 0) > 0:
                external_text = str(auto_detection_summary.get("auto_detection_external_sources") or "external")
                st.caption(
                    f"External evidence: {int(auto_detection_summary.get('auto_detection_external_rows', 0) or 0)} rows / "
                    f"exact {int(auto_detection_summary.get('auto_detection_external_exact_rows', 0) or 0)} / "
                    f"weak {int(auto_detection_summary.get('auto_detection_external_weak_rows', 0) or 0)} "
                    f"({external_text})"
                )
            if not external_site_df.empty:
                source_detail_columns = [
                    column
                    for column in [
                        "chain",
                        "resid",
                        "evidence_source",
                        "evidence_type",
                        "evidence_score",
                        "mapping_level",
                        "mapping_confidence",
                        "article_title",
                        "pmid",
                        "pmcid",
                        "doi",
                        "evidence_snippet",
                        "sentence_index",
                        "extraction_pattern",
                        "requires_manual_review",
                    ]
                    if column in external_site_df.columns
                ]
                if source_detail_columns:
                    st.caption("External evidence source details: structured citations, snippets and manual-review flags used by the pocket rerank.")
                    st.dataframe(
                        external_site_df[source_detail_columns].head(80),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.download_button(
                    "导出外部证据明细 CSV",
                    data=_to_csv_bytes(external_site_df),
                    file_name="external_residue_evidence_details.csv",
                    mime="text/csv",
                )
            if p2rank_ab_enabled:
                st.caption("P2Rank A/B: base = P2Rank disabled and auto detection rerun; positive rank_delta means P2Rank moved the pocket up.")
                if p2rank_ab_df.empty:
                    st.caption("P2Rank A/B: no comparable pocket ranking rows.")
                else:
                    st.dataframe(p2rank_ab_df, use_container_width=True, hide_index=True)
            route_status_text = str(auto_detection_summary.get("auto_detection_external_route_status") or "").strip()
            if route_status_text:
                st.caption(
                    f"Evidence route: {route_status_text} / "
                    f"support>={float(auto_detection_summary.get('auto_detection_external_route_min_support') or 0.0):.2f} / "
                    f"confidence>={float(auto_detection_summary.get('auto_detection_external_route_min_confidence') or 0.0):.2f} / "
                    f"quality>={float(auto_detection_summary.get('auto_detection_external_route_min_mapping_quality') or 0.0):.2f}"
                )
            if literature_ab_enabled:
                st.caption("Literature A/B: base = literature evidence removed and auto detection rerun; positive rank_delta means literature moved the pocket up.")
                if literature_ab_df.empty:
                    st.caption("Literature A/B: no comparable pocket ranking rows.")
                else:
                    st.dataframe(literature_ab_df, use_container_width=True, hide_index=True)
            if evidence_route_ab_enabled:
                st.caption("Evidence-route A/B: base = same external evidence but route disabled; positive rank_delta means the evidence route moved the pocket up.")
                if evidence_route_ab_df.empty:
                    st.caption("Evidence-route A/B: no comparable pocket ranking rows.")
                else:
                    st.dataframe(evidence_route_ab_df, use_container_width=True, hide_index=True)
            if int(auto_detection_summary.get("auto_detection_conservation_rows", 0) or 0) > 0:
                conservation_text = str(auto_detection_summary.get("auto_detection_conservation_sources") or "conservation")
                st.caption(
                    f"Conservation: {int(auto_detection_summary.get('auto_detection_conservation_rows', 0) or 0)} rows "
                    f"({conservation_text}, rerank-only)"
                )
            if conservation_ab_enabled:
                st.caption("Conservation A/B: base = conservation columns zeroed; positive rank_delta means the pocket moved up.")
                if conservation_ab_df.empty:
                    st.caption("Conservation A/B: no comparable pocket ranking rows.")
                else:
                    st.dataframe(conservation_ab_df, use_container_width=True, hide_index=True)
            if not auto_detection_diag_df.empty:
                st.dataframe(auto_detection_diag_df, use_container_width=True, hide_index=True)
            elif auto_detection_meta:
                st.json(auto_detection_meta)

with tab_overview:
    st.subheader("主分析总览")
    overview_cols = st.columns(2)
    with overview_cols[0]:
        st.markdown(f"**当前口袋来源**：{POCKET_SOURCE_LABELS.get(effective_pocket_mode, effective_pocket_mode)}")
        if effective_pocket_summary.empty:
            st.info("当前口袋来源下没有可用口袋。")
        else:
            st.dataframe(effective_pocket_summary, use_container_width=True, hide_index=True)
    with overview_cols[1]:
        st.markdown(f"**当前界面来源**：{ANNOTATION_SOURCE_LABELS.get(effective_annotation_mode, effective_annotation_mode)}")
        if interface_summary.empty:
            st.info("当前界面来源下没有可用注释。")
        else:
            st.dataframe(interface_summary, use_container_width=True, hide_index=True)

    if not interface_summary.empty:
        chart_df = interface_summary.set_index("region_type")[
            [column for column in ["residue_count", "pocket_count", "hotspot_count", "overlap_count"] if column in interface_summary.columns]
        ]
        st.subheader("region_type 计数对比")
        st.bar_chart(chart_df)

    st.subheader("联合推荐")
    if joint_candidate_df.empty:
        st.info("当前还没有可用的联合推荐结果。至少需要有效口袋，界面和热点证据会在存在时自动叠加。")
    else:
        recommendation_columns = [
            "recommendation_rank",
            "pocket_id",
            "recommendation_label",
            "recommendation_score",
            "recommendation_action",
            "evidence_quality_label",
            "evidence_anchor_support",
            "evidence_anchor_risk",
            "smart_rank_label",
            "smart_rank_score",
            "hotspot_overlap_count",
            "interface_overlap_count",
            "triple_overlap_count",
            "external_overlap_count",
            "external_exact_overlap_count",
            "external_weak_overlap_count",
            "external_overlap_ratio",
            "external_weighted_overlap_ratio",
            "external_mapping_confidence",
            "external_structure_verified_count",
            "external_structure_verified_ratio",
            "external_evidence_types",
            "method_vote_count",
            "recommendation_reason",
        ]
        recommendation_columns = [column for column in recommendation_columns if column in joint_candidate_df.columns]
        st.dataframe(joint_candidate_df[recommendation_columns], use_container_width=True, hide_index=True)

    with st.expander("数据源详情", expanded=False):
        source_cols = st.columns(2)
        with source_cols[0]:
            st.markdown("**上传 Pocket 汇总**")
            if uploaded_pocket_summary.empty:
                st.caption("无上传 Pocket 数据。")
            else:
                st.dataframe(uploaded_pocket_summary, use_container_width=True, hide_index=True)
            st.markdown("**自动 Pocket 汇总**")
            if auto_pocket_summary.empty:
                st.caption("无自动口袋结果。")
            else:
                st.dataframe(auto_pocket_summary, use_container_width=True, hide_index=True)
        with source_cols[1]:
            st.markdown("**上传界面汇总**")
            if uploaded_annotation_summary.empty:
                st.caption("无上传界面注释。")
            else:
                st.dataframe(uploaded_annotation_summary, use_container_width=True, hide_index=True)
            st.markdown("**结构推断界面汇总**")
            if inferred_annotation_summary.empty:
                st.caption("当前未生成结构推断界面。")
            else:
                st.dataframe(inferred_annotation_summary, use_container_width=True, hide_index=True)

with tab_annotations:
    st.subheader("界面注释明细")
    if enriched_annotations.empty:
        st.info("当前没有可显示的界面注释。你可以上传界面 CSV，或使用结构估算生成推断界面。")
    else:
        filter_cols = st.columns([1.1, 1.1, 1.1, 1.2, 1.5])
        only_overlap = filter_cols[0].checkbox("仅三重交集", value=False)
        only_pocket = filter_cols[1].checkbox("仅口袋命中", value=False)
        only_hotspot = filter_cols[2].checkbox("仅热点命中", value=False)
        only_interface_core = filter_cols[3].checkbox("仅核心界面", value=False)
        keyword = filter_cols[4].text_input("按残基/注释搜索", value="")

        region_options = sorted(
            value for value in enriched_annotations.get("region_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if value
        )
        selected_regions = st.multiselect("region_type 过滤", region_options, default=region_options)

        view = enriched_annotations.copy()
        if selected_regions:
            view = view[view["region_type"].astype(str).isin(selected_regions)]
        if only_overlap and "is_overlap" in view.columns:
            view = view[view["is_overlap"]]
        if only_pocket and "is_pocket" in view.columns:
            view = view[view["is_pocket"]]
        if only_hotspot and "is_hotspot" in view.columns:
            view = view[view["is_hotspot"]]
        if only_interface_core and "region_type" in view.columns:
            view = view[view["region_type"].astype(str).str.contains("core", case=False, na=False)]
        if keyword.strip():
            pattern = keyword.strip().lower()
            mask = pd.Series(False, index=view.index)
            for column in [col for col in ["residue_label", "annotation", "region_type"] if col in view.columns]:
                mask = mask | view[column].astype(str).str.lower().str.contains(pattern, na=False)
            view = view[mask]

        display_view = view.copy()
        if "inference_basis" in display_view.columns:
            display_view["inference_basis"] = (
                display_view["inference_basis"]
                .astype(str)
                .map(INFERENCE_BASIS_LABELS)
                .fillna(display_view["inference_basis"].astype(str))
            )

        display_columns = [
            "residue_label",
            "region_type",
            "annotation",
            "annotation_source",
            "inference_basis",
            "is_pocket",
            "is_hotspot",
            "is_overlap",
        ]
        display_columns = [column for column in display_columns if column in view.columns]
        st.dataframe(display_view[display_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "导出当前筛选结果 CSV",
            data=_to_csv_bytes(view),
            file_name="filtered_interface_annotations.csv",
            mime="text/csv",
        )

with tab_overlap:
    st.subheader("交集分析")
    overlap_metric_cols = st.columns(4)
    overlap_metric_cols[0].metric("口袋 ∩ 热点", len(pocket_hotspot_df))
    overlap_metric_cols[1].metric("界面 ∩ 热点", len(interface_hotspot_df))
    overlap_metric_cols[2].metric("界面 ∩ 口袋", len(interface_pocket_df))
    overlap_metric_cols[3].metric("界面 ∩ 口袋 ∩ 热点", len(triple_overlap_df))

    if not overlap_summary.empty:
        st.bar_chart(overlap_summary.set_index("category"))

    overlap_cols = st.columns(2)
    with overlap_cols[0]:
        st.markdown("**口袋与热点重叠残基**")
        if pocket_hotspot_df.empty:
            st.info("当前没有发现口袋与热点的重叠残基。")
        else:
            st.dataframe(
                pocket_hotspot_df[
                    [col for col in ["pocket_id", "residue_label", "score", "consensus_methods", "detection_route"] if col in pocket_hotspot_df.columns]
                ],
                use_container_width=True,
                hide_index=True,
            )
    with overlap_cols[1]:
        st.markdown("**界面、口袋、热点三重交集**")
        if triple_overlap_df.empty:
            st.info("当前没有发现三重交集残基。")
        else:
            st.dataframe(
                triple_overlap_df[[col for col in ["residue_label", "region_type", "annotation", "annotation_source"] if col in triple_overlap_df.columns]],
                use_container_width=True,
                hide_index=True,
            )

    if not hotspot_df.empty:
        st.subheader("热点列表")
        st.dataframe(hotspot_df[[col for col in ["label", "delta_total", "hotspot_rank"] if col in hotspot_df.columns]], use_container_width=True, hide_index=True)

with tab_export:
    st.subheader("导出")
    export_cols = st.columns(2)
    with export_cols[0]:
        if not effective_pocket_summary.empty:
            st.download_button(
                "导出当前口袋汇总 CSV",
                data=_to_csv_bytes(effective_pocket_summary),
                file_name="effective_pocket_summary.csv",
                mime="text/csv",
            )
        if not effective_pocket_df.empty:
            st.download_button(
                "导出当前口袋明细 CSV",
                data=_to_csv_bytes(effective_pocket_df),
                file_name="effective_pocket_candidates.csv",
                mime="text/csv",
            )
        if not enriched_annotations.empty:
            st.download_button(
                "导出当前界面注释 CSV",
                data=_to_csv_bytes(enriched_annotations),
                file_name="effective_interface_annotations.csv",
                mime="text/csv",
            )
        if not interface_summary.empty:
            st.download_button(
                "导出当前界面汇总 CSV",
                data=_to_csv_bytes(interface_summary),
                file_name="effective_interface_summary.csv",
                mime="text/csv",
            )
        if not joint_candidate_df.empty:
            st.download_button(
                "导出联合推荐 CSV",
                data=_to_csv_bytes(joint_candidate_df),
                file_name="joint_candidate_recommendations.csv",
                mime="text/csv",
            )
        if not pocket_decision_df.empty:
            st.download_button(
                "Export active-site decision CSV",
                data=_to_csv_bytes(pocket_decision_df),
                file_name="active_site_decision.csv",
                mime="text/csv",
            )
        if not pocket_reliability_df.empty:
            st.download_button(
                "Export reliability checklist CSV",
                data=_to_csv_bytes(pocket_reliability_df),
                file_name="pocket_reliability_checklist.csv",
                mime="text/csv",
            )
        if not pocket_triage_df.empty:
            st.download_button(
                "Export precision triage CSV",
                data=_to_csv_bytes(pocket_triage_df),
                file_name="pocket_precision_triage.csv",
                mime="text/csv",
            )
        if not ai_evidence_df.empty:
            st.download_button(
                "Export AI evidence CSV",
                data=_to_csv_bytes(ai_evidence_df),
                file_name="ai_residue_evidence.csv",
                mime="text/csv",
            )
        if not ai_evidence_audit_df.empty:
            st.download_button(
                "Export AI evidence audit CSV",
                data=_to_csv_bytes(ai_evidence_audit_df),
                file_name="ai_residue_evidence_audit.csv",
                mime="text/csv",
            )
        if not residue_evidence_consensus_df.empty:
            st.download_button(
                "Export residue evidence consensus CSV",
                data=_to_csv_bytes(residue_evidence_consensus_df),
                file_name="residue_evidence_consensus.csv",
                mime="text/csv",
            )
        if not pocket_consensus_coverage_df.empty:
            st.download_button(
                "Export pocket consensus coverage CSV",
                data=_to_csv_bytes(pocket_consensus_coverage_df),
                file_name="pocket_consensus_coverage.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_df.empty:
            st.download_button(
                "Export benchmark reference candidate CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_df),
                file_name="pocket_benchmark_reference_candidate.csv",
                mime="text/csv",
            )
        if not benchmark_reference_import_summary_df.empty:
            st.download_button(
                "Export benchmark reference import summary CSV",
                data=_to_csv_bytes(benchmark_reference_import_summary_df),
                file_name="pocket_benchmark_reference_import_summary.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_queue_df.empty:
            st.download_button(
                "Export benchmark reference candidate review queue CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_queue_df),
                file_name="pocket_benchmark_reference_candidate_review_queue.csv",
                mime="text/csv",
            )
        if benchmark_reference_candidate_review_checklist_markdown:
            st.download_button(
                "Export benchmark reference candidate review checklist MD",
                data=benchmark_reference_candidate_review_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_candidate_review_checklist.md",
                mime="text/markdown",
            )
        if not benchmark_reference_candidate_review_decision_template_df.empty:
            st.download_button(
                "Export benchmark reference candidate review decision template CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_decision_template_df),
                file_name="pocket_benchmark_reference_candidate_review_decision_template.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_decision_df.empty:
            st.download_button(
                "Export benchmark reference candidate review decisions normalized CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_decision_df),
                file_name="pocket_benchmark_reference_candidate_review_decisions_normalized.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_decision_validation_df.empty:
            st.download_button(
                "Export benchmark reference candidate review decision validation CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_decision_validation_df),
                file_name="pocket_benchmark_reference_candidate_review_decision_validation.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_review_outcome_df.empty:
            st.download_button(
                "Export benchmark reference candidate review outcomes CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_review_outcome_df),
                file_name="pocket_benchmark_reference_candidate_review_outcomes.csv",
                mime="text/csv",
            )
        if not benchmark_reference_candidate_accepted_df.empty:
            st.download_button(
                "Export accepted benchmark reference candidates CSV",
                data=_to_csv_bytes(benchmark_reference_candidate_accepted_df),
                file_name="pocket_benchmark_reference_candidate_accepted.csv",
                mime="text/csv",
            )
        if not benchmark_reference_df.empty:
            st.download_button(
                "Export benchmark reference CSV",
                data=_to_csv_bytes(benchmark_reference_df),
                file_name="pocket_benchmark_reference.csv",
                mime="text/csv",
            )
        if not benchmark_reference_source_audit_df.empty:
            if not benchmark_reference_source_audit_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_case_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_template_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision template CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_template_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_template.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decisions normalized CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decisions_normalized.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_validation_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision validation CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_validation_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_validation.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_outcome_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision outcome summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_outcome_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_outcome_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision closure queue CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_closure_queue_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_closure_queue.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_readiness_impact_summary_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision readiness impact summary CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary.csv",
                    mime="text/csv",
                )
            if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision readiness impact CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_readiness_impact_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_readiness_impact.csv",
                    mime="text/csv",
                )
            if benchmark_reference_source_audit_case_decision_closure_checklist_markdown:
                st.download_button(
                    "Export benchmark reference source audit case decision closure checklist MD",
                    data=benchmark_reference_source_audit_case_decision_closure_checklist_markdown.encode("utf-8"),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_closure_checklist.md",
                    mime="text/markdown",
                )
            if not benchmark_reference_source_audit_case_decision_outcome_df.empty:
                st.download_button(
                    "Export benchmark reference source audit case decision outcomes CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_case_decision_outcome_df),
                    file_name="pocket_benchmark_reference_source_audit_case_decision_outcomes.csv",
                    mime="text/csv",
                )
            if benchmark_reference_source_audit_case_checklist_markdown:
                st.download_button(
                    "Export benchmark reference source audit case checklist MD",
                    data=benchmark_reference_source_audit_case_checklist_markdown.encode("utf-8"),
                    file_name="pocket_benchmark_reference_source_audit_case_checklist.md",
                    mime="text/markdown",
                )
            if not benchmark_reference_source_audit_action_queue_df.empty:
                st.download_button(
                    "Export benchmark reference source audit action queue CSV",
                    data=_to_csv_bytes(benchmark_reference_source_audit_action_queue_df),
                    file_name="pocket_benchmark_reference_source_audit_action_queue.csv",
                    mime="text/csv",
                )
            if benchmark_reference_source_audit_checklist_markdown:
                st.download_button(
                    "Export benchmark reference source audit checklist MD",
                    data=benchmark_reference_source_audit_checklist_markdown.encode("utf-8"),
                    file_name="pocket_benchmark_reference_source_audit_checklist.md",
                    mime="text/markdown",
                )
            st.download_button(
                "Export benchmark reference source audit CSV",
                data=_to_csv_bytes(benchmark_reference_source_audit_df),
                file_name="pocket_benchmark_reference_source_audit.csv",
                mime="text/csv",
            )
        st.download_button(
            "Export benchmark reference template CSV",
            data=_to_csv_bytes(benchmark_reference_template_df),
            file_name="pocket_benchmark_reference_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Export benchmark reference template notes",
            data=benchmark_reference_template_markdown.encode("utf-8"),
            file_name="pocket_benchmark_reference_template.md",
            mime="text/markdown",
        )
        if not pocket_benchmark_reference_quality_issue_df.empty:
            st.download_button(
                "Export benchmark reference quality issues CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_quality_issue_df),
                file_name="pocket_benchmark_reference_quality_issues.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_quality_summary_df.empty:
            st.download_button(
                "Export benchmark reference quality summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_quality_summary_df),
                file_name="pocket_benchmark_reference_quality_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_quality_checklist_markdown:
            st.download_button(
                "Export benchmark reference curation checklist",
                data=pocket_benchmark_reference_quality_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_quality_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_reference_structure_validation_df.empty:
            st.download_button(
                "Export benchmark reference structure validation CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_structure_validation_df),
                file_name="pocket_benchmark_reference_structure_validation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_structure_validation_summary_df.empty:
            st.download_button(
                "Export benchmark reference structure validation summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_structure_validation_summary_df),
                file_name="pocket_benchmark_reference_structure_validation_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_structure_validation_checklist_markdown:
            st.download_button(
                "Export benchmark reference structure validation checklist",
                data=pocket_benchmark_reference_structure_validation_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_structure_validation_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_reference_readiness_summary_df.empty:
            st.download_button(
                "Export benchmark reference readiness summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_readiness_summary_df),
                file_name="pocket_benchmark_reference_readiness_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_readiness_case_summary_df.empty:
            st.download_button(
                "Export benchmark reference readiness case summary CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_readiness_case_summary_df),
                file_name="pocket_benchmark_reference_readiness_case_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_readiness_queue_df.empty:
            st.download_button(
                "Export benchmark reference readiness queue CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_readiness_queue_df),
                file_name="pocket_benchmark_reference_readiness_queue.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_readiness_checklist_markdown:
            st.download_button(
                "Export benchmark reference readiness checklist",
                data=pocket_benchmark_reference_readiness_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_readiness_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_interpretation_df.empty:
            st.download_button(
                "Export pocket benchmark interpretation CSV",
                data=_to_csv_bytes(pocket_benchmark_interpretation_df),
                file_name="pocket_benchmark_interpretation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_df),
                file_name="pocket_benchmark_case_interpretation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_matrix_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation matrix CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_matrix_df),
                file_name="pocket_benchmark_case_interpretation_matrix.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_matrix_summary_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation matrix summary CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_matrix_summary_df),
                file_name="pocket_benchmark_case_interpretation_matrix_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_interpretation_matrix_queue_df.empty:
            st.download_button(
                "Export pocket benchmark case interpretation matrix queue CSV",
                data=_to_csv_bytes(pocket_benchmark_case_interpretation_matrix_queue_df),
                file_name="pocket_benchmark_case_interpretation_matrix_queue.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_dataset_interpretation_df.empty:
            st.download_button(
                "Export pocket benchmark dataset interpretation CSV",
                data=_to_csv_bytes(pocket_benchmark_dataset_interpretation_df),
                file_name="pocket_benchmark_dataset_interpretation.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact cases CSV",
                data=_to_csv_bytes(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases.csv",
                mime="text/csv",
            )
        if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown:
            st.download_button(
                "Export pocket benchmark source audit case decision dataset impact case checklist MD",
                data=pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_dataset_interpretation_queue_df.empty:
            st.download_button(
                "Export pocket benchmark dataset interpretation queue CSV",
                data=_to_csv_bytes(pocket_benchmark_dataset_interpretation_queue_df),
                file_name="pocket_benchmark_dataset_interpretation_queue.csv",
                mime="text/csv",
            )
        if pocket_benchmark_dataset_interpretation_checklist_markdown:
            st.download_button(
                "Export pocket benchmark dataset interpretation checklist MD",
                data=pocket_benchmark_dataset_interpretation_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_dataset_interpretation_checklist.md",
                mime="text/markdown",
            )
        if pocket_benchmark_dataset_interpretation_report_markdown:
            st.download_button(
                "Export pocket benchmark dataset interpretation report MD",
                data=pocket_benchmark_dataset_interpretation_report_markdown.encode("utf-8"),
                file_name="pocket_benchmark_dataset_interpretation_report.md",
                mime="text/markdown",
            )
        if not pocket_benchmark_summary_df.empty:
            st.download_button(
                "Export pocket benchmark summary CSV",
                data=_to_csv_bytes(pocket_benchmark_summary_df),
                file_name="pocket_benchmark_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_dataset_summary_df.empty:
            st.download_button(
                "Export pocket benchmark dataset summary CSV",
                data=_to_csv_bytes(pocket_benchmark_dataset_summary_df),
                file_name="pocket_benchmark_dataset_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_case_summary_df.empty:
            st.download_button(
                "Export pocket benchmark case summary CSV",
                data=_to_csv_bytes(pocket_benchmark_case_summary_df),
                file_name="pocket_benchmark_case_summary.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_comparison_df),
                file_name="pocket_benchmark_variant_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_dataset_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant dataset comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_dataset_comparison_df),
                file_name="pocket_benchmark_variant_dataset_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_case_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant case comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_case_comparison_df),
                file_name="pocket_benchmark_variant_case_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_detail_comparison_df.empty:
            st.download_button(
                "Export pocket benchmark variant residue comparison CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_detail_comparison_df),
                file_name="pocket_benchmark_variant_residue_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_remediation_df.empty:
            st.download_button(
                "Export pocket benchmark remediation queue CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_remediation_df),
                file_name="pocket_benchmark_variant_remediation_queue.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_variant_remediation_summary_df.empty:
            st.download_button(
                "Export pocket benchmark remediation summary CSV",
                data=_to_csv_bytes(pocket_benchmark_variant_remediation_summary_df),
                file_name="pocket_benchmark_variant_remediation_summary.csv",
                mime="text/csv",
            )
        if pocket_benchmark_variant_remediation_checklist_markdown:
            st.download_button(
                "Export pocket benchmark remediation checklist",
                data=pocket_benchmark_variant_remediation_checklist_markdown.encode("utf-8"),
                file_name="pocket_benchmark_variant_remediation_checklist.md",
                mime="text/markdown",
            )
        if not p2rank_ab_df.empty:
            st.download_button(
                "Export P2Rank A/B comparison CSV",
                data=_to_csv_bytes(p2rank_ab_df),
                file_name="p2rank_ab_comparison.csv",
                mime="text/csv",
            )
        if not pocket_benchmark_details_df.empty:
            st.download_button(
                "Export pocket benchmark details CSV",
                data=_to_csv_bytes(pocket_benchmark_details_df),
                file_name="pocket_benchmark_details.csv",
                mime="text/csv",
            )
        if not consensus_rerank_suggestion_df.empty:
            st.download_button(
                "Export consensus rerank suggestions CSV",
                data=_to_csv_bytes(consensus_rerank_suggestion_df),
                file_name="consensus_rerank_suggestions.csv",
                mime="text/csv",
            )
        if not consensus_rerank_preview_df.empty:
            st.download_button(
                "Export consensus rerank preview CSV",
                data=_to_csv_bytes(consensus_rerank_preview_df),
                file_name="consensus_rerank_preview.csv",
                mime="text/csv",
            )
        if not consensus_rerank_policy_gate_df.empty:
            st.download_button(
                "Export consensus rerank policy gate CSV",
                data=_to_csv_bytes(consensus_rerank_policy_gate_df),
                file_name="consensus_rerank_policy_gate.csv",
                mime="text/csv",
            )
        if not consensus_rerank_action_queue_df.empty:
            st.download_button(
                "Export consensus rerank action queue CSV",
                data=_to_csv_bytes(consensus_rerank_action_queue_df),
                file_name="consensus_rerank_action_queue.csv",
                mime="text/csv",
            )
        if not consensus_rerank_apply_simulation_df.empty:
            st.download_button(
                "Export consensus rerank apply simulation CSV",
                data=_to_csv_bytes(consensus_rerank_apply_simulation_df),
                file_name="consensus_rerank_apply_simulation.csv",
                mime="text/csv",
            )
        if not consensus_rerank_simulation_delta_df.empty:
            st.download_button(
                "Export consensus rerank simulation delta CSV",
                data=_to_csv_bytes(consensus_rerank_simulation_delta_df),
                file_name="consensus_rerank_simulation_delta.csv",
                mime="text/csv",
            )
        if not consensus_rerank_precision_scorecard_df.empty:
            st.download_button(
                "Export consensus rerank precision scorecard CSV",
                data=_to_csv_bytes(consensus_rerank_precision_scorecard_df),
                file_name="consensus_rerank_precision_scorecard.csv",
                mime="text/csv",
            )
        if not consensus_rerank_precision_guardrail_df.empty:
            st.download_button(
                "Export consensus rerank precision guardrail CSV",
                data=_to_csv_bytes(consensus_rerank_precision_guardrail_df),
                file_name="consensus_rerank_precision_guardrail.csv",
                mime="text/csv",
            )
        if consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty:
            st.download_button(
                "Export consensus rerank precision guardrail report",
                data=consensus_rerank_precision_guardrail_report_markdown.encode("utf-8"),
                file_name="consensus_rerank_precision_guardrail_report.md",
                mime="text/markdown",
            )
        if not consensus_rerank_guardrail_artifact_manifest_df.empty:
            st.download_button(
                "Export consensus rerank guardrail artifact manifest CSV",
                data=_to_csv_bytes(consensus_rerank_guardrail_artifact_manifest_df),
                file_name="consensus_rerank_guardrail_artifact_manifest.csv",
                mime="text/csv",
            )
        if consensus_rerank_guardrail_handoff_zip:
            st.download_button(
                "Export consensus rerank guardrail handoff ZIP",
                data=consensus_rerank_guardrail_handoff_zip,
                file_name="consensus_rerank_guardrail_handoff.zip",
                mime="application/zip",
            )
        if not consensus_rerank_guardrail_bundle_verification_summary_df.empty:
            st.download_button(
                "Export consensus rerank guardrail bundle verification summary CSV",
                data=_to_csv_bytes(consensus_rerank_guardrail_bundle_verification_summary_df),
                file_name="consensus_rerank_guardrail_bundle_verification_summary.csv",
                mime="text/csv",
            )
        if not consensus_rerank_guardrail_bundle_verification_df.empty:
            st.download_button(
                "Export consensus rerank guardrail bundle verification CSV",
                data=_to_csv_bytes(consensus_rerank_guardrail_bundle_verification_df),
                file_name="consensus_rerank_guardrail_bundle_verification.csv",
                mime="text/csv",
            )
        if consensus_rerank_guardrail_handoff_certificate_markdown:
            st.download_button(
                "Export consensus rerank guardrail handoff certificate",
                data=consensus_rerank_guardrail_handoff_certificate_markdown.encode("utf-8"),
                file_name="consensus_rerank_guardrail_handoff_certificate.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_decision_template_df.empty:
            st.download_button(
                "Export consensus rerank release decision template CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_template_df),
                file_name="consensus_rerank_release_decision_template.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_decision_df.empty:
            st.download_button(
                "Export normalized consensus rerank release decisions CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_df),
                file_name="consensus_rerank_release_decisions_normalized.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_decision_validation_df.empty:
            st.download_button(
                "Export consensus rerank release decision validation CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_validation_df),
                file_name="consensus_rerank_release_decision_validation.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_decision_summary_df.empty:
            st.download_button(
                "Export consensus rerank release decision summary CSV",
                data=_to_csv_bytes(consensus_rerank_release_decision_summary_df),
                file_name="consensus_rerank_release_decision_summary.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_apply_plan_df.empty:
            st.download_button(
                "Export consensus rerank release apply plan CSV",
                data=_to_csv_bytes(consensus_rerank_release_apply_plan_df),
                file_name="consensus_rerank_release_apply_plan.csv",
                mime="text/csv",
            )
        if consensus_rerank_release_apply_report_markdown:
            st.download_button(
                "Export consensus rerank release apply report",
                data=consensus_rerank_release_apply_report_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_apply_report.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_execution_template_df.empty:
            st.download_button(
                "Export consensus rerank release execution template CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_template_df),
                file_name="consensus_rerank_release_execution_template.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_execution_receipt_df.empty:
            st.download_button(
                "Export normalized consensus rerank release execution receipt CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_receipt_df),
                file_name="consensus_rerank_release_execution_receipt_normalized.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_execution_validation_df.empty:
            st.download_button(
                "Export consensus rerank release execution validation CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_validation_df),
                file_name="consensus_rerank_release_execution_validation.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_execution_summary_df.empty:
            st.download_button(
                "Export consensus rerank release execution summary CSV",
                data=_to_csv_bytes(consensus_rerank_release_execution_summary_df),
                file_name="consensus_rerank_release_execution_summary.csv",
                mime="text/csv",
            )
        if consensus_rerank_release_execution_report_markdown:
            st.download_button(
                "Export consensus rerank release execution report",
                data=consensus_rerank_release_execution_report_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_execution_report.md",
                mime="text/markdown",
            )
        if consensus_rerank_release_closure_certificate_markdown:
            st.download_button(
                "Export consensus rerank release closure certificate",
                data=consensus_rerank_release_closure_certificate_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_closure_certificate.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_closure_ledger_df.empty:
            st.download_button(
                "Export consensus rerank release closure ledger CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_ledger_df),
                file_name="consensus_rerank_release_closure_ledger.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_closure_summary_df.empty:
            st.download_button(
                "Export consensus rerank release closure readiness summary CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_summary_df),
                file_name="consensus_rerank_release_closure_summary.csv",
                mime="text/csv",
            )
        if not consensus_rerank_release_closure_blocker_df.empty:
            st.download_button(
                "Export consensus rerank release closure blocker queue CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_blocker_df),
                file_name="consensus_rerank_release_closure_blocker_queue.csv",
                mime="text/csv",
            )
        if consensus_rerank_release_closure_remediation_checklist_markdown:
            st.download_button(
                "Export consensus rerank release closure remediation checklist",
                data=consensus_rerank_release_closure_remediation_checklist_markdown.encode("utf-8"),
                file_name="consensus_rerank_release_closure_remediation_checklist.md",
                mime="text/markdown",
            )
        if not consensus_rerank_release_closure_detached_manifest_df.empty:
            st.download_button(
                "Export consensus rerank release closure detached manifest CSV",
                data=_to_csv_bytes(consensus_rerank_release_closure_detached_manifest_df),
                file_name="consensus_rerank_release_closure_detached_manifest.csv",
                mime="text/csv",
            )
        if consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty:
            st.download_button(
                "Export consensus rerank action checklist",
                data=consensus_rerank_action_checklist_markdown.encode("utf-8"),
                file_name="consensus_rerank_action_checklist.md",
                mime="text/markdown",
            )
        if not ai_review_decision_df.empty:
            st.download_button(
                "Export normalized AI review decisions CSV",
                data=_to_csv_bytes(ai_review_decision_df),
                file_name="ai_review_decisions_normalized.csv",
                mime="text/csv",
            )
        if not ai_review_decision_validation_df.empty:
            st.download_button(
                "Export AI review decision validation CSV",
                data=_to_csv_bytes(ai_review_decision_validation_df),
                file_name="ai_review_decision_validation.csv",
                mime="text/csv",
            )
        if not ai_review_round_summary_df.empty:
            st.download_button(
                "Export AI review round summary CSV",
                data=_to_csv_bytes(ai_review_round_summary_df),
                file_name="ai_review_round_summary.csv",
                mime="text/csv",
            )
        if not ai_review_ranking_delta_df.empty:
            st.download_button(
                "Export AI review ranking delta CSV",
                data=_to_csv_bytes(ai_review_ranking_delta_df),
                file_name="ai_review_ranking_delta.csv",
                mime="text/csv",
            )
        if ai_review_round_report_markdown:
            st.download_button(
                "Export AI review round report",
                data=ai_review_round_report_markdown.encode("utf-8"),
                file_name="ai_review_round_report.md",
                mime="text/markdown",
            )
        if not ai_review_artifact_manifest_df.empty:
            st.download_button(
                "Export AI review artifact manifest CSV",
                data=_to_csv_bytes(ai_review_artifact_manifest_df),
                file_name="ai_review_artifact_manifest.csv",
                mime="text/csv",
            )
        if ai_review_bundle_readme_markdown:
            st.download_button(
                "Export AI review bundle README",
                data=ai_review_bundle_readme_markdown.encode("utf-8"),
                file_name="ai_review_bundle_README.md",
                mime="text/markdown",
            )
        if ai_review_artifact_bundle_zip:
            st.download_button(
                "Export AI review artifact bundle ZIP",
                data=ai_review_artifact_bundle_zip,
                file_name="ai_review_artifacts.zip",
                mime="application/zip",
            )
        if not ai_review_bundle_verification_df.empty:
            st.download_button(
                "Export AI review bundle verification CSV",
                data=_to_csv_bytes(ai_review_bundle_verification_df),
                file_name="ai_review_bundle_verification.csv",
                mime="text/csv",
            )
        if not ai_review_bundle_verification_summary_df.empty:
            st.download_button(
                "Export AI review bundle verification summary CSV",
                data=_to_csv_bytes(ai_review_bundle_verification_summary_df),
                file_name="ai_review_bundle_verification_summary.csv",
                mime="text/csv",
            )
        if ai_review_bundle_certificate_markdown:
            st.download_button(
                "Export AI review bundle handoff certificate",
                data=ai_review_bundle_certificate_markdown.encode("utf-8"),
                file_name="ai_review_bundle_certificate.md",
                mime="text/markdown",
            )
        if not ai_review_decision_outcome_df.empty:
            st.download_button(
                "Export AI review decision outcomes CSV",
                data=_to_csv_bytes(ai_review_decision_outcome_df),
                file_name="ai_review_decision_outcomes.csv",
                mime="text/csv",
            )
        if not ai_evidence_review_queue_df.empty:
            st.download_button(
                "Export AI evidence review queue CSV",
                data=_to_csv_bytes(ai_evidence_review_queue_df),
                file_name="ai_evidence_review_queue.csv",
                mime="text/csv",
            )
            if not ai_review_decision_template_df.empty:
                st.download_button(
                    "Export AI review decision template CSV",
                    data=_to_csv_bytes(ai_review_decision_template_df),
                    file_name="ai_review_decision_template.csv",
                    mime="text/csv",
                )
            st.download_button(
                "Export AI evidence review checklist",
                data=ai_review_checklist_markdown.encode("utf-8"),
                file_name="ai_evidence_review_checklist.md",
                mime="text/markdown",
            )
        if not rankable_ai_evidence_df.empty:
            st.download_button(
                "Export ranking-gated AI evidence CSV",
                data=_to_csv_bytes(rankable_ai_evidence_df),
                file_name="ai_residue_evidence_ranked.csv",
                mime="text/csv",
            )
        if not ai_ranking_impact_df.empty:
            st.download_button(
                "Export AI ranking impact CSV",
                data=_to_csv_bytes(ai_ranking_impact_df),
                file_name="ai_ranking_impact_summary.csv",
                mime="text/csv",
            )
        if not ai_followup_plan_df.empty:
            st.download_button(
                "Export AI follow-up evidence plan CSV",
                data=_to_csv_bytes(ai_followup_plan_df),
                file_name="ai_followup_evidence_plan.csv",
                mime="text/csv",
            )
            st.download_button(
                "Export AI follow-up prompt bundle",
                data=ai_followup_prompt_bundle.encode("utf-8"),
                file_name="ai_followup_prompt_bundle.md",
                mime="text/markdown",
            )
    with export_cols[1]:
        if not overlap_summary.empty:
            st.download_button(
                "导出交集摘要 CSV",
                data=_to_csv_bytes(overlap_summary),
                file_name="overlap_summary.csv",
                mime="text/csv",
            )
        if not triple_overlap_df.empty:
            st.download_button(
                "导出三重交集 CSV",
                data=_to_csv_bytes(triple_overlap_df),
                file_name="triple_overlap.csv",
                mime="text/csv",
            )
        st.download_button(
            "导出快照 JSON",
            data=snapshot_to_json_bytes(snapshot),
            file_name="pocket_interface_snapshot.json",
            mime="application/json",
        )
        st.download_button(
            "导出快照 SVG",
            data=build_snapshot_svg(snapshot),
            file_name="pocket_interface_snapshot.svg",
            mime="image/svg+xml",
        )

    report_lines = [
        "ProteinInsight 口袋 / 界面分析报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"能量来源: {summary.get('energy_source') or '未标注'}",
        f"平均能量: {format_energy_value(summary.get('mean_energy'))}",
        f"蛋白体积: {f'{protein_volume:,.1f} A³' if protein_volume is not None else '-'}",
        f"当前口袋来源: {POCKET_SOURCE_LABELS.get(effective_pocket_mode, effective_pocket_mode)}",
        f"当前界面来源: {ANNOTATION_SOURCE_LABELS.get(effective_annotation_mode, effective_annotation_mode)}",
        f"结构推断依据: {inferred_basis_text or '-'}",
        f"口袋数量: {len(effective_pocket_summary) if not effective_pocket_summary.empty else 0}",
        f"自动口袋数: {len(auto_pocket_summary) if not auto_pocket_summary.empty else 0}",
        f"界面注释数: {len(enriched_annotations)}",
        f"三重交集: {len(triple_overlap_df)}",
        f"AI evidence: {len(ai_evidence_df)} rows / status {ai_evidence_meta.get('status') or '-'}",
        f"AI evidence used for ranking: {len(rankable_ai_evidence_df)} rows / status {rankable_ai_evidence_meta.get('status') or '-'}",
        f"Residue evidence consensus: {len(residue_evidence_consensus_df)} rows / top {top_residue_consensus.get('residue_anchor') if top_residue_consensus is not None else '-'} / tier {top_residue_consensus.get('consensus_tier') if top_residue_consensus is not None else '-'}",
        f"Pocket consensus coverage: {len(pocket_consensus_coverage_df)} rows / top {top_pocket_consensus_coverage.get('pocket_id') if top_pocket_consensus_coverage is not None else '-'} / label {top_pocket_consensus_coverage.get('pocket_consensus_label') if top_pocket_consensus_coverage is not None else '-'}",
        f"Benchmark reference candidate: {len(benchmark_reference_candidate_df)} rows / import {benchmark_reference_import_summary_df.iloc[0].get('import_status') if not benchmark_reference_import_summary_df.empty else '-'} / provisional used {'yes' if benchmark_reference_is_provisional else 'no'}",
        f"Benchmark reference source: {benchmark_reference_source_mode or '-'} / provisional {'yes' if benchmark_reference_is_provisional else 'no'} / reviewed candidate {'yes' if benchmark_reference_is_reviewed_candidate else 'no'}",
        f"Benchmark reference source audit: {len(benchmark_reference_source_audit_df)} rows / claim status {benchmark_reference_source_audit_df.iloc[0].get('source_claim_status') if not benchmark_reference_source_audit_df.empty else '-'} / independent claim {benchmark_reference_source_audit_df.iloc[0].get('can_support_independent_claim') if not benchmark_reference_source_audit_df.empty else '-'}",
        f"Benchmark reference source audit summary: {len(benchmark_reference_source_audit_summary_df)} rows / top status {benchmark_reference_source_audit_summary_df.iloc[0].get('source_claim_status') if not benchmark_reference_source_audit_summary_df.empty else '-'} / independent claim {benchmark_reference_source_audit_summary_df.iloc[0].get('can_support_independent_claim') if not benchmark_reference_source_audit_summary_df.empty else '-'}",
        f"Benchmark reference source audit action queue: {len(benchmark_reference_source_audit_action_queue_df)} rows / blockers {int(benchmark_reference_source_audit_action_queue_df['action_status'].astype(str).eq('blocker').sum()) if not benchmark_reference_source_audit_action_queue_df.empty and 'action_status' in benchmark_reference_source_audit_action_queue_df.columns else 0} / review {int(benchmark_reference_source_audit_action_queue_df['action_status'].astype(str).eq('review').sum()) if not benchmark_reference_source_audit_action_queue_df.empty and 'action_status' in benchmark_reference_source_audit_action_queue_df.columns else 0}",
        f"Benchmark reference source audit cases: {len(benchmark_reference_source_audit_case_summary_df)} rows / blocked {benchmark_reference_source_audit_case_summary_blocked_cases} / review {benchmark_reference_source_audit_case_summary_review_cases}",
        f"Benchmark reference source audit case decision template: {len(benchmark_reference_source_audit_case_decision_template_df)} rows",
        f"Benchmark reference source audit case decisions: {len(benchmark_reference_source_audit_case_decision_df)} rows / validation blocked {int(benchmark_reference_source_audit_case_decision_validation_df['validation_status'].astype(str).eq('blocked').sum()) if not benchmark_reference_source_audit_case_decision_validation_df.empty and 'validation_status' in benchmark_reference_source_audit_case_decision_validation_df.columns else 0}",
        f"Benchmark reference source audit case decision outcome summary: {len(benchmark_reference_source_audit_case_decision_outcome_summary_df)} rows / status {benchmark_reference_source_audit_case_decision_outcome_summary_status or '-'} / open {benchmark_reference_source_audit_case_decision_outcome_summary_open_cases}",
        f"Benchmark reference source audit case decision closure queue: {len(benchmark_reference_source_audit_case_decision_closure_queue_df)} rows / blockers {int(benchmark_reference_source_audit_case_decision_closure_queue_df['closure_action_status'].astype(str).eq('blocker').sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and 'closure_action_status' in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0} / review {int(benchmark_reference_source_audit_case_decision_closure_queue_df['closure_action_status'].astype(str).eq('review').sum()) if not benchmark_reference_source_audit_case_decision_closure_queue_df.empty and 'closure_action_status' in benchmark_reference_source_audit_case_decision_closure_queue_df.columns else 0}",
        f"Benchmark reference source audit case decision readiness impact summary: {len(benchmark_reference_source_audit_case_decision_readiness_impact_summary_df)} rows / status {benchmark_reference_source_audit_case_decision_readiness_impact_summary_status or '-'} / open {benchmark_reference_source_audit_case_decision_readiness_impact_summary_open_cases}",
        f"Benchmark reference source audit case decision readiness impact: {len(benchmark_reference_source_audit_case_decision_readiness_impact_df)} rows / cleared {int(benchmark_reference_source_audit_case_decision_readiness_impact_df['readiness_impact'].astype(str).eq('cleared-by-decision').sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and 'readiness_impact' in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0} / open {int(benchmark_reference_source_audit_case_decision_readiness_impact_df['readiness_impact'].astype(str).isin(['decision-adjusted-open', 'decision-open', 'unchanged-open']).sum()) if not benchmark_reference_source_audit_case_decision_readiness_impact_df.empty and 'readiness_impact' in benchmark_reference_source_audit_case_decision_readiness_impact_df.columns else 0}",
        f"Benchmark reference source audit case decision closure checklist: {'available' if benchmark_reference_source_audit_case_decision_closure_checklist_markdown else 'not available'}",
        f"Benchmark reference source audit case decision outcomes: {len(benchmark_reference_source_audit_case_decision_outcome_df)} rows / blocked {int(benchmark_reference_source_audit_case_decision_outcome_df['applied_status'].astype(str).eq('blocked').sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and 'applied_status' in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0} / pending {int(benchmark_reference_source_audit_case_decision_outcome_df['applied_status'].astype(str).eq('pending').sum()) if not benchmark_reference_source_audit_case_decision_outcome_df.empty and 'applied_status' in benchmark_reference_source_audit_case_decision_outcome_df.columns else 0}",
        f"Benchmark reference source audit case checklist: {'available' if benchmark_reference_source_audit_case_checklist_markdown else 'not available'}",
        f"Benchmark reference source audit checklist: {'available' if benchmark_reference_source_audit_checklist_markdown else 'not available'}",
        f"Benchmark reference candidate review: {len(benchmark_reference_candidate_review_queue_df)} rows / P1 {int(benchmark_reference_candidate_review_queue_df['priority'].astype(str).eq('P1').sum()) if not benchmark_reference_candidate_review_queue_df.empty and 'priority' in benchmark_reference_candidate_review_queue_df.columns else 0} / P2 {int(benchmark_reference_candidate_review_queue_df['priority'].astype(str).eq('P2').sum()) if not benchmark_reference_candidate_review_queue_df.empty and 'priority' in benchmark_reference_candidate_review_queue_df.columns else 0} / checklist {'available' if benchmark_reference_candidate_review_checklist_markdown else 'not available'}",
        f"Benchmark reference candidate review decisions: {len(benchmark_reference_candidate_review_decision_df)} rows / validation blocked {int(benchmark_reference_candidate_review_decision_validation_df['validation_status'].astype(str).eq('blocked').sum()) if not benchmark_reference_candidate_review_decision_validation_df.empty and 'validation_status' in benchmark_reference_candidate_review_decision_validation_df.columns else 0} / accepted actions {int(benchmark_reference_candidate_review_outcome_df['applied_status'].astype(str).eq('accepted').sum()) if not benchmark_reference_candidate_review_outcome_df.empty and 'applied_status' in benchmark_reference_candidate_review_outcome_df.columns else 0} / accepted references {len(benchmark_reference_candidate_accepted_df)}",
        f"Catalytic pocket benchmark: references {len(benchmark_reference_df)} / Top-1 {top1_benchmark.get('coverage_ratio') if top1_benchmark is not None else '-'} / Top-3 {top3_benchmark.get('coverage_ratio') if top3_benchmark is not None else '-'} / best rank {top3_benchmark.get('best_rank') if top3_benchmark is not None else '-'}",
        f"Benchmark reference template: {len(benchmark_reference_template_df)} rows / notes {'available' if benchmark_reference_template_markdown else 'not available'}",
        f"Benchmark reference curation quality: {len(pocket_benchmark_reference_quality_issue_df)} issues / summary {len(pocket_benchmark_reference_quality_summary_df)} rows / checklist {'available' if pocket_benchmark_reference_quality_checklist_markdown else 'not available'}",
        f"Benchmark reference structure validation: {len(pocket_benchmark_reference_structure_validation_df)} issues / summary {len(pocket_benchmark_reference_structure_validation_summary_df)} rows / checklist {'available' if pocket_benchmark_reference_structure_validation_checklist_markdown else 'not available'}",
        f"Benchmark reference readiness: {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('readiness_status') if not pocket_benchmark_reference_readiness_summary_df.empty else '-'} / blockers {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('p0_p1_issue_count') if not pocket_benchmark_reference_readiness_summary_df.empty else 0} / review {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('p2_issue_count') if not pocket_benchmark_reference_readiness_summary_df.empty else 0} / source audit {pocket_benchmark_reference_readiness_summary_df.iloc[0].get('source_audit_issue_count') if not pocket_benchmark_reference_readiness_summary_df.empty else 0}",
        f"Benchmark reference readiness cases: {len(pocket_benchmark_reference_readiness_case_summary_df)} rows / blocked {int(pocket_benchmark_reference_readiness_case_summary_df['readiness_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and 'readiness_status' in pocket_benchmark_reference_readiness_case_summary_df.columns else 0} / review {int(pocket_benchmark_reference_readiness_case_summary_df['readiness_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_reference_readiness_case_summary_df.empty and 'readiness_status' in pocket_benchmark_reference_readiness_case_summary_df.columns else 0}",
        f"Benchmark interpretation: {len(pocket_benchmark_interpretation_df)} rows / Top-1 claim {pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df['top_n'].astype(int) == 1].iloc[0].get('claim_status') if not pocket_benchmark_interpretation_df.empty and 'top_n' in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df['top_n'].astype(int) == 1).any() else '-'} / Top-3 claim {pocket_benchmark_interpretation_df[pocket_benchmark_interpretation_df['top_n'].astype(int) == 3].iloc[0].get('claim_status') if not pocket_benchmark_interpretation_df.empty and 'top_n' in pocket_benchmark_interpretation_df.columns and (pocket_benchmark_interpretation_df['top_n'].astype(int) == 3).any() else '-'}",
        f"Benchmark case interpretation: {len(pocket_benchmark_case_interpretation_df)} rows / blocked {int(pocket_benchmark_case_interpretation_df['claim_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_case_interpretation_df.empty and 'claim_status' in pocket_benchmark_case_interpretation_df.columns else 0} / review {int(pocket_benchmark_case_interpretation_df['claim_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_case_interpretation_df.empty and 'claim_status' in pocket_benchmark_case_interpretation_df.columns else 0}",
        f"Benchmark case interpretation matrix: {len(pocket_benchmark_case_interpretation_matrix_df)} rows / blocked {int(pocket_benchmark_case_interpretation_matrix_df['case_interpretation_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and 'case_interpretation_status' in pocket_benchmark_case_interpretation_matrix_df.columns else 0} / review {int(pocket_benchmark_case_interpretation_matrix_df['case_interpretation_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_case_interpretation_matrix_df.empty and 'case_interpretation_status' in pocket_benchmark_case_interpretation_matrix_df.columns else 0}",
        f"Benchmark case interpretation matrix summary: {pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get('summary_status') if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else '-'} / usable {pocket_benchmark_case_interpretation_matrix_summary_df.iloc[0].get('usable_claim_ready_case_count') if not pocket_benchmark_case_interpretation_matrix_summary_df.empty else 0}",
        f"Benchmark case interpretation matrix queue: {len(pocket_benchmark_case_interpretation_matrix_queue_df)} rows / blockers {int(pocket_benchmark_case_interpretation_matrix_queue_df['action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and 'action_status' in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0} / review {int(pocket_benchmark_case_interpretation_matrix_queue_df['action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_case_interpretation_matrix_queue_df.empty and 'action_status' in pocket_benchmark_case_interpretation_matrix_queue_df.columns else 0}",
        f"Benchmark dataset interpretation: {len(pocket_benchmark_dataset_interpretation_df)} rows / blocked {int(pocket_benchmark_dataset_interpretation_df['dataset_claim_status'].astype(str).eq('blocked').sum()) if not pocket_benchmark_dataset_interpretation_df.empty and 'dataset_claim_status' in pocket_benchmark_dataset_interpretation_df.columns else 0} / review {int(pocket_benchmark_dataset_interpretation_df['dataset_claim_status'].astype(str).eq('review-needed').sum()) if not pocket_benchmark_dataset_interpretation_df.empty and 'dataset_claim_status' in pocket_benchmark_dataset_interpretation_df.columns else 0}",
        f"Benchmark source-audit decision dataset impact: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df)} rows / blockers {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df['dataset_source_impact_status'].astype(str).eq('source-blocked').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and 'dataset_source_impact_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0} / review {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df['dataset_source_impact_status'].astype(str).eq('source-review-needed').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and 'dataset_source_impact_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0} / mismatch {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df['dataset_source_impact_status'].astype(str).eq('source-gate-mismatch').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.empty and 'dataset_source_impact_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_df.columns else 0}",
        f"Benchmark source-audit decision dataset impact cases: {len(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df)} rows / blockers {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df['source_action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and 'source_action_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0} / review {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df['source_action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and 'source_action_status' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0} / mismatch {int(pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df['source_gate_mismatch'].map(bool).sum()) if not pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.empty and 'source_gate_mismatch' in pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_df.columns else 0} / checklist {'available' if pocket_benchmark_reference_source_audit_case_decision_dataset_impact_case_checklist_markdown else 'not available'}",
        f"Benchmark dataset interpretation queue: {len(pocket_benchmark_dataset_interpretation_queue_df)} rows / blockers {int(pocket_benchmark_dataset_interpretation_queue_df['action_status'].astype(str).eq('blocker').sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and 'action_status' in pocket_benchmark_dataset_interpretation_queue_df.columns else 0} / review {int(pocket_benchmark_dataset_interpretation_queue_df['action_status'].astype(str).eq('review').sum()) if not pocket_benchmark_dataset_interpretation_queue_df.empty and 'action_status' in pocket_benchmark_dataset_interpretation_queue_df.columns else 0} / checklist {'available' if pocket_benchmark_dataset_interpretation_checklist_markdown else 'not available'} / report {'available' if pocket_benchmark_dataset_interpretation_report_markdown else 'not available'}",
        f"Catalytic benchmark dataset: cases {int(pocket_benchmark_case_summary_df['benchmark_id'].nunique()) if not pocket_benchmark_case_summary_df.empty and 'benchmark_id' in pocket_benchmark_case_summary_df.columns else 0} / dataset rows {len(pocket_benchmark_dataset_summary_df)}",
        f"Catalytic benchmark variants: {len(pocket_benchmark_variant_comparison_df)} rows / current vs ablations {'available' if not pocket_benchmark_variant_comparison_df.empty else 'not available'}",
        f"Catalytic benchmark variant cases: {len(pocket_benchmark_variant_case_comparison_df)} rows / variant dataset rows {len(pocket_benchmark_variant_dataset_comparison_df)}",
        f"Catalytic benchmark variant residues: {len(pocket_benchmark_variant_detail_comparison_df)} rows",
        f"Catalytic benchmark remediation queue: {len(pocket_benchmark_variant_remediation_df)} rows / summary {len(pocket_benchmark_variant_remediation_summary_df)} rows / checklist {'available' if pocket_benchmark_variant_remediation_checklist_markdown else 'not available'}",
        f"P2Rank A/B: {len(p2rank_ab_df)} rows / {'enabled' if p2rank_ab_enabled else 'not enabled'}",
        f"Consensus rerank suggestions: {len(consensus_rerank_suggestion_df)} rows / top {top_consensus_rerank_suggestion.get('pocket_id') if top_consensus_rerank_suggestion is not None else '-'} / status {top_consensus_rerank_suggestion.get('suggestion_status') if top_consensus_rerank_suggestion is not None else '-'}",
        f"Consensus rerank preview: {len(consensus_rerank_preview_df)} rows / top {top_consensus_rerank_preview.get('pocket_id') if top_consensus_rerank_preview is not None else '-'} / decision {top_consensus_rerank_preview.get('preview_decision') if top_consensus_rerank_preview is not None else '-'}",
        f"Consensus rerank policy gate: {top_consensus_rerank_policy_gate.get('policy_status') if top_consensus_rerank_policy_gate is not None else '-'} / changed {top_consensus_rerank_policy_gate.get('changed_rows') if top_consensus_rerank_policy_gate is not None else 0} / blocked {top_consensus_rerank_policy_gate.get('blocked_rows') if top_consensus_rerank_policy_gate is not None else 0}",
        f"Consensus rerank action queue: {len(consensus_rerank_action_queue_df)} rows / top {top_consensus_rerank_action.get('pocket_id') if top_consensus_rerank_action is not None else '-'} / issue {top_consensus_rerank_action.get('issue_type') if top_consensus_rerank_action is not None else '-'}",
        f"Consensus rerank action checklist: {'available' if consensus_rerank_action_checklist_markdown and not consensus_rerank_action_queue_df.empty else 'not available'}",
        f"Consensus rerank apply simulation: {len(consensus_rerank_apply_simulation_df)} rows / top {top_consensus_rerank_apply.get('pocket_id') if top_consensus_rerank_apply is not None else '-'} / status {top_consensus_rerank_apply.get('apply_status') if top_consensus_rerank_apply is not None else '-'}",
        f"Consensus rerank simulation delta: {len(consensus_rerank_simulation_delta_df)} rows / top {top_consensus_rerank_delta.get('pocket_id') if top_consensus_rerank_delta is not None else '-'} / change {top_consensus_rerank_delta.get('change_type') if top_consensus_rerank_delta is not None else '-'}",
        f"Consensus rerank precision scorecard: {top_consensus_rerank_scorecard.get('scorecard_status') if top_consensus_rerank_scorecard is not None else '-'} / score {top_consensus_rerank_scorecard.get('precision_improvement_score') if top_consensus_rerank_scorecard is not None else 0} / blockers {top_consensus_rerank_scorecard.get('open_blocker_rows') if top_consensus_rerank_scorecard is not None else 0}",
        f"Consensus rerank precision guardrail: {top_consensus_rerank_guardrail.get('guardrail_status') if top_consensus_rerank_guardrail is not None else '-'} / decision {top_consensus_rerank_guardrail.get('guardrail_decision') if top_consensus_rerank_guardrail is not None else '-'} / mode {top_consensus_rerank_guardrail.get('apply_mode') if top_consensus_rerank_guardrail is not None else '-'}",
        f"Consensus rerank precision guardrail report: {'available' if consensus_rerank_precision_guardrail_report_markdown and not consensus_rerank_precision_guardrail_df.empty else 'not available'}",
        f"Consensus rerank guardrail handoff bundle: {'available' if consensus_rerank_guardrail_handoff_zip else 'not available'} / manifest {len(consensus_rerank_guardrail_artifact_manifest_df)} files",
        f"Consensus rerank guardrail bundle verification: {consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get('verification_status') if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else '-'} / failed {consensus_rerank_guardrail_bundle_verification_summary_df.iloc[0].get('failed_files') if not consensus_rerank_guardrail_bundle_verification_summary_df.empty else 0}",
        f"Consensus rerank guardrail handoff certificate: {'available' if consensus_rerank_guardrail_handoff_certificate_markdown else 'not available'}",
        f"Consensus rerank release decision template: {len(consensus_rerank_release_decision_template_df)} rows",
        f"Consensus rerank release decisions: {len(consensus_rerank_release_decision_df)} rows / status {consensus_rerank_release_decision_meta.get('status') or '-'}",
        f"Consensus rerank release decision validation: {len(consensus_rerank_release_decision_validation_df)} rows / blocked {int((consensus_rerank_release_decision_validation_df['validation_status'].astype(str) == 'blocked').sum()) if not consensus_rerank_release_decision_validation_df.empty and 'validation_status' in consensus_rerank_release_decision_validation_df.columns else 0}",
        f"Consensus rerank release review: {top_consensus_rerank_release_decision_summary.get('release_review_status') if top_consensus_rerank_release_decision_summary is not None else '-'} / allowed {'yes' if top_consensus_rerank_release_decision_summary is not None and bool(top_consensus_rerank_release_decision_summary.get('release_allowed')) else 'no'}",
        f"Consensus rerank release apply plan: {len(consensus_rerank_release_apply_plan_df)} rows / top {top_consensus_rerank_release_apply_plan.get('pocket_id') if top_consensus_rerank_release_apply_plan is not None else '-'} / status {top_consensus_rerank_release_apply_plan.get('release_apply_status') if top_consensus_rerank_release_apply_plan is not None else '-'}",
        f"Consensus rerank release apply report: {'available' if consensus_rerank_release_apply_report_markdown else 'not available'}",
        f"Consensus rerank release execution template: {len(consensus_rerank_release_execution_template_df)} rows",
        f"Consensus rerank release execution receipt: {len(consensus_rerank_release_execution_receipt_df)} rows / status {consensus_rerank_release_execution_receipt_meta.get('status') or '-'}",
        f"Consensus rerank release execution validation: {len(consensus_rerank_release_execution_validation_df)} rows / blocked {int((consensus_rerank_release_execution_validation_df['validation_status'].astype(str) == 'blocked').sum()) if not consensus_rerank_release_execution_validation_df.empty and 'validation_status' in consensus_rerank_release_execution_validation_df.columns else 0}",
        f"Consensus rerank release execution: {top_consensus_rerank_release_execution_summary.get('execution_review_status') if top_consensus_rerank_release_execution_summary is not None else '-'} / complete {'yes' if top_consensus_rerank_release_execution_summary is not None and bool(top_consensus_rerank_release_execution_summary.get('execution_complete')) else 'no'}",
        f"Consensus rerank release execution report: {'available' if consensus_rerank_release_execution_report_markdown else 'not available'}",
        f"Consensus rerank release closure certificate: {'available' if consensus_rerank_release_closure_certificate_markdown else 'not available'}",
        f"Consensus rerank release closure ledger: {len(consensus_rerank_release_closure_ledger_df)} rows / blocked {int(consensus_rerank_release_closure_ledger_df['closure_check'].astype(str).str.lower().isin({'blocked', 'missing'}).sum()) if not consensus_rerank_release_closure_ledger_df.empty and 'closure_check' in consensus_rerank_release_closure_ledger_df.columns else 0}",
        f"Consensus rerank release closure readiness: {top_consensus_rerank_release_closure_summary.get('closure_readiness_status') if top_consensus_rerank_release_closure_summary is not None else '-'} / closed {'yes' if top_consensus_rerank_release_closure_summary is not None and bool(top_consensus_rerank_release_closure_summary.get('release_closed')) else 'no'}",
        f"Consensus rerank release closure blockers: {len(consensus_rerank_release_closure_blocker_df)} rows / top {consensus_rerank_release_closure_blocker_df.iloc[0].get('blocker_type') if not consensus_rerank_release_closure_blocker_df.empty else '-'}",
        f"Consensus rerank release closure remediation checklist: {'available' if consensus_rerank_release_closure_remediation_checklist_markdown else 'not available'}",
        f"Consensus rerank release closure detached manifest: {len(consensus_rerank_release_closure_detached_manifest_df)} files",
        f"AI review decisions: {len(ai_review_decision_df)} rows / applied {ai_review_decision_meta.get('applied_rows') or '0'} / status {ai_review_decision_meta.get('status') or '-'}",
        f"AI review decision validation: {len(ai_review_decision_validation_df)} rows / blocked {int((ai_review_decision_validation_df['validation_status'].astype(str) == 'blocked').sum()) if not ai_review_decision_validation_df.empty and 'validation_status' in ai_review_decision_validation_df.columns else 0}",
        f"AI review round: {ai_review_round_summary_df.iloc[0].get('review_round_status') if not ai_review_round_summary_df.empty else '-'} / rankable {ai_review_round_summary_df.iloc[0].get('rankable_after_review_rows') if not ai_review_round_summary_df.empty else 0}",
        f"AI review ranking delta: {ai_review_ranking_delta_df.iloc[0].get('review_effect_status') if not ai_review_ranking_delta_df.empty else '-'} / +{ai_review_ranking_delta_df.iloc[0].get('promoted_rows') if not ai_review_ranking_delta_df.empty else 0} / -{ai_review_ranking_delta_df.iloc[0].get('removed_rows') if not ai_review_ranking_delta_df.empty else 0}",
        f"AI review artifact manifest: {len(ai_review_artifact_manifest_df)} files",
        f"AI review bundle README: {'available' if ai_review_bundle_readme_markdown else 'not available'}",
        f"AI review artifact bundle: {'available' if ai_review_artifact_bundle_zip else 'not available'}",
        f"AI review bundle verification: {len(ai_review_bundle_verification_df)} files / failed {int((ai_review_bundle_verification_df['verification_status'].astype(str) != 'verified').sum()) if not ai_review_bundle_verification_df.empty and 'verification_status' in ai_review_bundle_verification_df.columns else 0}",
        f"AI review bundle verification summary: {ai_review_bundle_verification_summary_df.iloc[0].get('verification_status') if not ai_review_bundle_verification_summary_df.empty else '-'}",
        f"AI review bundle certificate: {'available' if ai_review_bundle_certificate_markdown else 'not available'}",
        f"AI review decision outcomes: {len(ai_review_decision_outcome_df)} rows",
        f"AI review decision template: {len(ai_review_decision_template_df)} rows",
        f"AI influence: {ai_ranking_impact_df.iloc[0].get('ai_influence_level') if not ai_ranking_impact_df.empty else '-'} / Top pocket AI residues {ai_ranking_impact_df.iloc[0].get('top_pocket_ai_residues') if not ai_ranking_impact_df.empty else '-'}",
        f"AI review queue: {len(ai_evidence_review_queue_df)} rows / top fix {ai_evidence_review_queue_df.iloc[0].get('fix_type') if not ai_evidence_review_queue_df.empty else '-'}",
        f"AI follow-up plan: {len(ai_followup_plan_df)} rows",
        "AI evidence audit: "
        + (
            ", ".join(f"{status}:{count}" for status, count in ai_evidence_audit_df["audit_status"].astype(str).value_counts().to_dict().items())
            if not ai_evidence_audit_df.empty and "audit_status" in ai_evidence_audit_df.columns
            else "none"
        ),
    ]
    if top_joint_candidate is not None:
        report_lines.extend(
            [
                f"联合推荐 Top1: {top_joint_candidate['pocket_id']}",
                f"联合推荐等级: {top_joint_candidate['recommendation_label']}",
                f"联合推荐理由: {top_joint_candidate['recommendation_reason']}",
            ]
        )
    if top_pocket_decision is not None:
        reliability_pass_count = int((pocket_reliability_df["status"].astype(str) == "pass").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0
        reliability_review_count = int((pocket_reliability_df["status"].astype(str) == "review").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0
        reliability_missing_count = int((pocket_reliability_df["status"].astype(str) == "missing").sum()) if not pocket_reliability_df.empty and "status" in pocket_reliability_df.columns else 0
        report_lines.extend(
            [
                f"Top active-site decision: {top_pocket_decision.get('pocket_id')} / {top_pocket_decision.get('decision_label')}",
                f"Top decision score: {top_pocket_decision.get('decision_score')} / audit {top_pocket_decision.get('audit_status')}",
                f"Precision tier: {top_pocket_triage.get('precision_tier') if top_pocket_triage is not None else '-'}",
                f"Triage action: {top_pocket_triage.get('triage_action') if top_pocket_triage is not None else '-'}",
                f"Reliability checks: pass {reliability_pass_count}, review {reliability_review_count}, missing {reliability_missing_count}",
                f"Reliability gaps: {top_reliability_gaps or 'none'}",
                f"Next step: {top_pocket_decision.get('next_step')}",
            ]
        )
    report_text = "\n".join(report_lines)
    if PDF_EXPORT_AVAILABLE:
        st.download_button(
            "导出 PDF 报告",
            data=build_simple_pdf(report_text, snapshot=snapshot),
            file_name="pocket_interface_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("当前环境未安装 reportlab，PDF 报告暂不可用。请安装依赖后重试。")

st.info("页面已补齐三类主链路：上传 Pocket 与自动口袋可切换或合并；未上传界面注释时可使用结构推断界面；交集分析、筛选结果和当前主分析结果都可以单独导出。")
