# Pocket Detection Enhancement Notes

更新时间：2026-04-18

本文档记录 ProteinInsight 口袋检测、外部证据融合、文献残基抽取、P2Rank 接入、consensus rerank 和 release closure 审计链路，方便后续继续迭代。

## Scope

当前口袋管线包含以下增强路径：

- 将结构化 catalytic-site evidence 前移到检测阶段，而不只在最终推荐表中 rerank。
- 可选本地 `P2Rank`，与 `pyKVFinder`、ligand proximity、geometry clustering 和 external-evidence route 共同参与候选生成。
- Conservation 表格作为独立 rerank-only 信号，不污染外部功能位点通道。
- Literature residue mining 将 PubMed / Europe PMC / manual text 中的关键残基转换为 external-evidence schema。
- Pocket-level ranker 加入 shallow-exposure penalty，降低无热点、无 ligand、无外部功能位点支持的浅表 false positive。
- Evidence quality、mapping risk、A/B comparison、core/shell/rim layer 和 closure audit 都会进入导出和 snapshot。

## Key Files

- `src/protein_visualizer/services/pocket.py`
- `src/protein_visualizer/services/pocket_ranker.py`
- `src/protein_visualizer/services/pocket_decision.py`
- `src/protein_visualizer/services/external_sites.py`
- `src/protein_visualizer/services/literature_sites.py`
- `src/protein_visualizer/services/conservation.py`
- `src/protein_visualizer/services/p2rank.py`
- `src/protein_visualizer/services/candidate_fusion.py`
- `src/protein_visualizer/services/snapshot.py`
- `pages/6_口袋与界面.py`
- `tests/test_pocket_detection.py`
- `tests/test_pocket_ranker.py`
- `tests/test_pocket_decision.py`
- `tests/test_external_sites.py`
- `tests/test_literature_sites.py`
- `tests/test_conservation.py`
- `tests/test_p2rank.py`
- `tests/test_snapshot_export.py`

## Current Detection Flow

1. Parse uploaded PDB into atom table, residue centers and ligand atoms.
2. Build residue-level support signals in the precision residue table.
3. Load or derive external evidence:
   - UniProt functional sites.
   - PDBe/SIFTS mapping.
   - M-CSA catalytic residues.
   - PubMed / Europe PMC / manual literature residues.
   - Optional conservation table.
4. Run enabled detection methods:
   - `pyKVFinder`
   - optional `P2Rank`
   - `external-evidence`
   - ligand-guided clustering
   - geometry-guided clustering
5. Merge method outputs through consensus pocket construction.
6. Rank detected pockets with structural, evidence, hotspot, ligand, interface and conservation signals.
7. Build summary tables, A/B comparison tables, reliability checks and export artifacts.

## External Evidence Integration

External evidence affects detection through residue-level support, not only post-hoc annotation.

### Sources

- UniProt active site, binding site, metal-binding and modified residue annotations.
- PDBe/SIFTS UniProt/PDB residue mapping.
- M-CSA catalytic residues.
- Literature-mined residues from PubMed, Europe PMC and manual text.

Conservation is intentionally handled separately because broad conservation patches can overwhelm precise catalytic residue signals if used as candidate-generation seeds.

### Structure-Assisted Mapping

When `pdb_text` is available, mapping prefers chains and residues that actually exist in the uploaded structure.

Current behavior:

- Prefer chains present in the uploaded PDB.
- Verify whether mapped author residues exist in `ATOM` residue set.
- Downgrade missing, gap-crossing or residue-number-only mappings.
- Track mapping metadata such as verified rows, weak rows and identity mismatch rows.
- Propagate mapping quality into pocket ranking and recommendation actions.

### Evidence Features

Residue-level fields include:

- `external_support`
- `external_confidence`
- `external_mapping_quality`
- `external_evidence_count`
- `external_exact_match`
- `external_direct_anchor`
- `external_direct_sources`
- `external_evidence_types`
- `external_evidence_notes`
- `evidence_anchor_distance`
- `evidence_anchor_proximity`
- `evidence_anchor_residue`

Pocket-level summary fields include:

- `external_supported_residue_count`
- `external_evidence_total`
- `external_exact_match_count`
- `external_direct_anchor_count`
- `evidence_route_anchor_count`
- `evidence_anchor_residues`
- `evidence_quality_label`
- `evidence_quality_score`
- `evidence_quality_warning`

## Evidence-Guided Pocket Route

`external-evidence` is a standalone detection route.

It is intended for enzyme cases where a curated or literature-supported residue is more meaningful than a purely geometric cavity score.

Current behavior:

- Requires strong anchors such as exact mapped UniProt/M-CSA/literature residues or high-confidence structure-verified evidence.
- Expands a local residue neighborhood around anchors.
- Scores residues by external support, mapping confidence, mapping quality, anchor proximity, local structure score, contact density, ligand contact and hotspot overlap.
- Emits `EvidencePocket-*` rows with `detection_route = precision-external-evidence`.
- Participates in consensus merge alongside P2Rank, pyKVFinder, ligand and geometry routes.

The route is controllable in `detect_auto_pocket_table()` and the Streamlit sidebar:

- `enable_external_evidence_route`
- `external_evidence_min_support`
- `external_evidence_min_confidence`
- `external_evidence_min_mapping_quality`
- `external_evidence_radius`

## Literature Evidence Mining

`src/protein_visualizer/services/literature_sites.py` converts paper text and abstracts into the external evidence schema.

### Inputs

- Manual text upload.
- PubMed E-utilities search.
- Europe PMC search.
- Optional Europe PMC Open Access `fullTextXML` retrieval.

### Extraction Policy

The extraction is intentionally conservative.

Accepted patterns include:

- `Asp123`, `Asp-123`, `Glu35`
- mutation-style `D123A`
- triad/dyad-style residue lists such as `Ser195, His57 and Asp102`

Rows are retained only when nearby text contains functional terms such as:

- `active site`
- `catalytic`
- `nucleophile`
- `substrate binding`
- `metal binding`
- `mutation`
- `mutagenesis`
- `abolished activity`
- `reduced activity`
- `kcat`
- `Km`

Low-signal contexts such as generic sequence alignment are filtered out.

### Source Detail Fields

Literature and AI-assisted evidence now preserve citation and snippet metadata as first-class evidence columns:

- `article_title`
- `pmid`
- `pmcid`
- `doi`
- `evidence_snippet`
- `sentence_index`
- `extraction_pattern`
- `requires_manual_review`

These fields are part of the shared external evidence schema, so UniProt, M-CSA and conservation rows remain compatible with empty defaults while literature rows carry source details through merge, SIFTS mapping, pocket detection, page preview and CSV export.

`requires_manual_review` is set when the source lacks stable identifiers, when the evidence is manually supplied, when the context score is weak, or when the extracted row is only a generic literature residue rather than a strong catalytic/binding/mutagenesis signal.

### Mapping Policy

Literature numbering is risky, so rows are tiered:

- If UniProt accession and PDB ID are available, literature residues are treated as UniProt positions and mapped through the existing mapper.
- If the user explicitly assumes PDB structure numbering, rows can become exact structure-numbering evidence.
- Otherwise rows remain weak literature-text-mining evidence.

When PDB text is available, assumed structure numbering checks residue identity. Example: a paper `Asp10` will not become a high-confidence exact anchor if uploaded chain residue 10 is `Ala`.

### Cross-Article Support

Literature evidence is merged before structure mapping:

- Group by residue position.
- Normalize article identity by title, then PMID/PMCID/DOI when available.
- Boost support only when the same residue appears in multiple distinct articles.
- Avoid double-counting abstract and full text from the same article.

## Conservation Import

Conservation is parsed through `src/protein_visualizer/services/conservation.py`.

Current behavior:

- Supports ConSurf / Rate4Site / generic residue score tables.
- Normalizes score direction.
- Keeps conservation out of candidate generation.
- Applies a small rerank-only contribution.
- Provides conservation A/B comparison by recomputing ranks with conservation zeroed out.

This avoids letting broad conserved surfaces overpower precise functional residue evidence.

## Catalytic Pocket Benchmark

`src/protein_visualizer/services/benchmark.py` adds an evaluation-only benchmark layer. It does not change pocket ranking.

### Input

Users can upload a curated catalytic residue table in the pocket page. Accepted columns include:

- `chain`
- `resid`
- `resname`
- `reference_type`
- `reference_source`
- `reference_note`
- `expected_pocket_id`

The parser also accepts residue labels such as `Ser195`, `A:195` or `D123A`. Blank chain is treated as a wildcard, which is useful when the curated source does not specify a PDB chain.

### Metrics

The benchmark exports:

- `pocket_benchmark_reference.csv`
- `pocket_benchmark_summary.csv`
- `pocket_benchmark_variant_comparison.csv`
- `pocket_benchmark_details.csv`

Current summary metrics:

- Top-1 catalytic residue coverage.
- Top-3 catalytic residue coverage.
- Top-5 catalytic residue coverage.
- Best hit rank and best hit pocket.
- Matched and missed catalytic residues.
- Coverage delta / loss for current vs ablated variants.

This gives a concrete accuracy check before claiming that the top-ranked pocket is the active-site pocket.

### Variant Comparison

When the relevant A/B toggles are enabled, the benchmark also compares current ranking against ablated variants:

- `no-literature`
- `no-evidence-route`
- `no-conservation-rerank`

Positive `coverage_loss_vs_reference` means removing that evidence path reduced catalytic residue coverage compared with the current run. This converts A/B ranking changes into an active-site accuracy metric.

## P2Rank Integration

P2Rank support is optional and local-only.

Discovery order:

1. Explicit UI path.
2. `P2RANK_SCRIPT`.
3. `P2RANK_HOME`.
4. Common executable names such as `prank`, `prank.sh`, `prank.bat`, `p2rank.jar`.

Parsed outputs:

- Pocket-level prediction CSV.
- Residue-level prediction CSV.
- Fallback `residue_list` extraction from pocket CSV.

P2Rank output enters consensus as:

- `detection_method = "p2rank"`
- `detection_route = "precision-p2rank"` or profile-specific route.

## A/B Checks

Current A/B comparison tables:

- Literature evidence removal.
- External-evidence route on/off.
- Conservation rerank-only on/off.

Positive `rank_delta` means the enhanced condition moved the pocket upward. A/B rows also carry evidence-quality fields so a rank improvement can be judged against direct anchor support and mapping risk.

## Pocket Quality Labels

Pocket evidence quality labels:

- `strong-direct-anchor`
- `direct-anchor`
- `route-anchor`
- `neighborhood-expanded`
- `diffuse-external-support`
- `no-external-evidence`
- `geometry-only`

These labels are exported to tables, reports, snapshots and history so reviewers can distinguish true key-residue support from broad proximity support.

## Core / Shell / Rim Layers

Pocket residues can be layered:

- `core`: direct functional anchors, strong evidence residues or catalytic context.
- `shell`: residues supporting or surrounding the core.
- `rim`: entrance, border or low-support boundary residues.

This is important for enzyme use cases because the user often needs to know which residues deserve mutation, validation or presentation focus.

## Consensus Rerank Audit Chain

`src/protein_visualizer/services/pocket_decision.py` owns the high-risk rerank workflow.

### Diagnostic and review artifacts

- `consensus_rerank_suggestions.csv`
- `consensus_rerank_preview.csv`
- `consensus_rerank_policy_gate.csv`
- `consensus_rerank_action_queue.csv`
- `consensus_rerank_action_checklist.md`
- `consensus_rerank_apply_simulation.csv`
- `consensus_rerank_simulation_delta.csv`
- `consensus_rerank_precision_scorecard.csv`
- `consensus_rerank_precision_guardrail.csv`
- `consensus_rerank_precision_guardrail_report.md`

### Handoff artifacts

- `consensus_rerank_guardrail_handoff.zip`
- `consensus_rerank_guardrail_artifact_manifest.csv`
- `consensus_rerank_guardrail_bundle_verification.csv`
- `consensus_rerank_guardrail_bundle_verification_summary.csv`
- `consensus_rerank_guardrail_handoff_certificate.md`

### Release approval artifacts

- `consensus_rerank_release_decision_template.csv`
- `consensus_rerank_release_decisions_normalized.csv`
- `consensus_rerank_release_decision_validation.csv`
- `consensus_rerank_release_decision_summary.csv`

### Apply and execution artifacts

- `consensus_rerank_release_apply_plan.csv`
- `consensus_rerank_release_apply_report.md`
- `consensus_rerank_release_execution_template.csv`
- `consensus_rerank_release_execution_receipt_normalized.csv`
- `consensus_rerank_release_execution_validation.csv`
- `consensus_rerank_release_execution_summary.csv`
- `consensus_rerank_release_execution_report.md`

### Closure artifacts

Inside the handoff ZIP/manifest when available:

- `consensus_rerank_release_closure_certificate.md`
- `consensus_rerank_release_closure_ledger.csv`

Detached after ZIP verification:

- `consensus_rerank_release_closure_summary.csv`
- `consensus_rerank_release_closure_blocker_queue.csv`
- `consensus_rerank_release_closure_remediation_checklist.md`
- `consensus_rerank_release_closure_detached_manifest.csv`

Detached artifacts stay outside the ZIP because they depend on ZIP verification. Adding them back into the ZIP would create a hash cycle.

## Closure Readiness Rule

A release is considered `closed-and-verified` only when:

- Every required closure ledger row has `closure_check = ok`.
- Every required ledger artifact has a 64-character SHA-256.
- Handoff ZIP verification summary is `verified`.
- Handoff ZIP failed file count is 0.

If any condition fails:

- `closure_summary` reports a blocked state.
- `closure_blocker_queue` lists actionable blockers.
- `closure_remediation_checklist` turns blockers into human checkboxes.
- `closure_detached_manifest` hashes the detached closure artifacts.

## Snapshot and Persistence

Snapshot `extra` records:

- detection metadata
- evidence-route status
- literature/conservation A/B rows
- consensus rerank tables and top statuses
- handoff ZIP and manifest availability
- bundle verification status
- release decision/apply/execution/closure statuses
- detached closure manifest row count

This allows analysis history, JSON snapshot, SVG snapshot, TXT report and PDF report to summarize the same audit state.

## Tests Added

Coverage includes:

- external evidence parsing and merge
- structure-assisted mapping behavior
- P2Rank adapter
- conservation parsing and A/B rerank
- literature extraction and mocked API flows
- evidence route enable/disable behavior
- evidence quality labels
- pocket layering
- consensus rerank suggestions, preview, policy gate and action queue
- precision scorecard and guardrail
- handoff ZIP, manifest, verification and certificate
- release decision template/import/validation/summary
- apply plan/report
- execution template/receipt/validation/summary/report
- closure certificate, ledger, readiness summary, blocker queue, remediation checklist and detached manifest
- snapshot export summary lines

Latest full run:

```text
195 passed
```

## Known Limits

- P2Rank binary is not vendored; users must install it locally.
- Literature mining remains rule-based and conservative.
- Paywalled full text is not fetched.
- mmCIF, insertion code and mature-chain offset support still need deeper implementation.
- Benchmark accuracy is not yet proven on a curated enzyme set.
- Structure quality signals such as resolution, R-factor and AlphaFold pLDDT are not yet fully integrated.

## Recommended Next Iterations

1. Expand the benchmark set from M-CSA, holo PDB structures and literature-confirmed catalytic residues.
2. Add snippet-level literature evidence with PMID/PMCID/DOI and manual review flags.
3. Improve mmCIF and insertion-code mapping.
4. Add SEQRES/ATOM and UniProt canonical/mature-chain alignment.
5. Add family priors from Pfam, InterPro, PROSITE or EC motifs.
6. Improve visual rendering of core/shell/rim and evidence anchors.
7. Add structure quality warnings for low-confidence or incomplete regions.
8. Add PyMOL/ChimeraX session export for final presentation.

## Practical High-Accuracy Usage

1. Provide UniProt accession.
2. Provide EC number if known.
3. Enable M-CSA evidence.
4. Enable literature mining or upload paper text when catalytic residues are known.
5. Enable structure-numbering assumption only when the paper uses the same PDB chain numbering.
6. Keep external-evidence route enabled for curated exact sites.
7. Tighten evidence thresholds or disable the route if A/B shows weak evidence creating noisy new pockets.
8. Upload conservation table when available, but interpret it as rerank support rather than direct catalytic proof.
9. Enable P2Rank when installed locally.
10. Review closure blocker queue and remediation checklist before claiming a consensus rerank release is closed.
