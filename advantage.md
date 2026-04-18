# ProteinInsight 优势与改进分析

更新时间：2026-04-18

## 产品定位

ProteinInsight 当前应定位为“面向酶蛋白的关键残基证据驱动口袋定位与审计工具”，而不是普通的通用蛋白口袋检测器。

同类口袋工具通常回答的是：蛋白表面有哪些几何凹陷、通道或潜在 ligand-binding pocket。

ProteinInsight 更关注的问题是：对于一个酶蛋白，哪些口袋真正围绕催化残基、底物结合残基、金属/辅因子结合残基、文献突变位点或数据库功能位点形成，并且这个判断能否被复核、导出、审批、执行和归档。

## 同类产品常见缺陷

### 1. 几何口袋工具功能语义不足

代表工具包括 fpocket、CASTp、DoGSiteScorer 等。它们擅长发现凹陷、体积、深度和 druggability，但不足是：

- 几何最高分不一定等于酶活性口袋。
- 大而深的非活性凹陷容易排在前面。
- 对浅表 catalytic groove、开放底物槽、金属配位区和构象依赖口袋不稳定。
- 用户仍需人工查 UniProt、M-CSA、文献和 PDB 编号。

### 2. 机器学习 binding-site 工具缺少审计链

P2Rank / PrankWeb 等工具比纯几何更强，但仍偏向通用 ligand-binding site prediction：

- 输出不天然解释“为什么这是酶的关键活性口袋”。
- 对 catalytic residue、mutation evidence、EC family context 的利用有限。
- 通常缺少 residue-level source、mapping quality、review decision 和 release audit。
- 对 AlphaFold 低置信区、缺失 loop 和编号漂移仍需人工判断。

### 3. 数据库证据高质量但不是结构口袋生成器

UniProt、PDBe/SIFTS、M-CSA 能提供关键残基和映射，但缺口是：

- 它们本身不生成结构口袋。
- SIFTS 映射仍可能受链选择、缺失残基、insertion code、author numbering 和 isoform offset 影响。
- 数据库通常给关键点位，不给 pocket boundary。
- 多来源证据冲突时，需要质量分层和人工复核入口。

### 4. 文献检索无法直接进入结构流程

PubMed / Europe PMC 能找到催化残基、突变残基和功能片段，但实际使用中存在断点：

- `Asp10`、`D123A`、`Ser-His-Asp triad` 需要被结构化提取。
- 文献编号可能是 UniProt、成熟酶、前体蛋白或 PDB author numbering。
- 同一文章摘要和全文不能被重复当成独立证据。
- 如果不校验 PDB 实际残基类型，容易把文献点位映射错。

### 5. Docking / MD / 商业平台成本更高

更重的计算平台有更高上限，但对当前场景不一定合适：

- 需要候选配体、参数准备和更多算力。
- 解释门槛高，不适合快速展示、课程设计或初筛。
- 通常从“配体能否 dock 进去”出发，而不是从“酶关键残基在哪里”出发。

## 我们当前的优势

### 1. 垂直聚焦酶蛋白

ProteinInsight 不追求找出所有可能 druggable pocket，而是优先定位与酶功能相关的口袋。系统可以更积极地使用 catalytic residue、active site、substrate binding、metal binding、mutagenesis、activity-loss、EC number 等信号。

### 2. 多证据融合，不依赖单一算法

当前链路整合了：

- 几何聚类和 ligand proximity。
- 可选本地 P2Rank。
- UniProt 功能位点。
- PDBe/SIFTS 残基映射。
- M-CSA 催化残基。
- PubMed / Europe PMC 文献残基抽取。
- 手动文献文本。
- conservation 表格。
- hotspot、interface、joint recommendation。

核心价值不是“跑了更多算法”，而是把数据库、文献和结构证据统一到 residue-level evidence schema，再进入检测、排序、解释、导出和审计。

### 3. 外部证据前移到候选生成

系统不仅在预测后展示注释，而是允许外部功能位点参与口袋生成：

- `external_support`
- `external_confidence`
- `external_mapping_quality`
- `external_direct_anchor`
- `evidence_route_anchor`
- `evidence_anchor_distance`
- `evidence_anchor_residue`

这意味着当 UniProt / M-CSA / 文献明确指出关键残基时，即使几何法没有把对应浅表区域排到前面，系统也能通过 `external-evidence` route 生成候选口袋。

### 4. 映射质量进入排序和风险提示

系统区分 structure-verified、structure-assisted interpolation、linear SIFTS transfer、gap fallback 和 weak numbering assumption。映射质量会影响口袋排序和 recommendation action，而不是被一概当作 exact evidence。

### 5. 文献证据可控

文献模块具备基础保护：

- 只保留 active site、catalytic、mutagenesis、activity-loss、kcat、Km 等功能上下文。
- 过滤 sequence alignment 等低信号上下文。
- 支持 PubMed、Europe PMC 和手动文本。
- 对跨文章重复支持做聚合，而不是把同一文章摘要/全文重复计分。
- 在假设 PDB 编号时校验实际 residue identity，不匹配则降级。

### 6. A/B 审计，而不是盲目信任证据

系统提供多类 A/B 检查：

- literature evidence 移除前后排名变化。
- external-evidence route 开启/关闭对比。
- conservation rerank-only 对比。
- pocket comparison 中保留 evidence-quality delta。

这让用户看到证据是否真的改善排序，而不是只看一个最终分数。

### 7. 口袋精细分层

当前口袋残基可标注为：

- `core`：直接关键残基、高证据 anchor、金属/辅因子邻近或强功能支持。
- `shell`：围绕 core 的结构支撑残基。
- `rim`：入口、边缘和边界残基。

这让结果从“一个残基集合”提升为“核心、壳层、边界”的解释结构，更适合后续突变设计、验证和展示。

### 8. Consensus rerank 不直接自动应用

系统把 rerank 拆成可审计阶段：

- suggestion
- preview
- policy gate
- action queue
- scorecard
- guardrail
- handoff ZIP
- reviewer decision
- apply plan
- execution receipt
- closure certificate
- closure ledger
- detached readiness summary
- blocker queue
- remediation checklist
- detached manifest

这解决了一个常见问题：很多工具给出新排序，却没有说明这个排序是否应该进入正式结果，也没有审批、执行和关闭证据。

## 当前已落地的审计产物

### Guardrail handoff

- `consensus_rerank_guardrail_handoff.zip`
- `consensus_rerank_guardrail_artifact_manifest.csv`
- `consensus_rerank_guardrail_bundle_verification.csv`
- `consensus_rerank_guardrail_bundle_verification_summary.csv`
- `consensus_rerank_guardrail_handoff_certificate.md`

价值：形成可移交、可校验的 rerank 评审证据包。

### Release approval

- `consensus_rerank_release_decision_template.csv`
- `consensus_rerank_release_decisions_normalized.csv`
- `consensus_rerank_release_decision_validation.csv`
- `consensus_rerank_release_decision_summary.csv`

价值：把“人工审批”从口头动作变成结构化回传和验证。

### Apply and execution

- `consensus_rerank_release_apply_plan.csv`
- `consensus_rerank_release_apply_report.md`
- `consensus_rerank_release_execution_template.csv`
- `consensus_rerank_release_execution_receipt_normalized.csv`
- `consensus_rerank_release_execution_validation.csv`
- `consensus_rerank_release_execution_summary.csv`
- `consensus_rerank_release_execution_report.md`

价值：证明被批准的 manual rank order 是否真的按计划执行。

### Closure pack

- `consensus_rerank_release_closure_certificate.md`
- `consensus_rerank_release_closure_ledger.csv`
- `consensus_rerank_release_closure_summary.csv`
- `consensus_rerank_release_closure_blocker_queue.csv`
- `consensus_rerank_release_closure_remediation_checklist.md`
- `consensus_rerank_release_closure_detached_manifest.csv`

价值：形成最终关闭证据。`closure_summary` 只有在 ledger 全部 ok、必需证据都有 SHA-256、handoff ZIP verification 通过且失败文件数为 0 时，才会标记为 `closed-and-verified`。

## 当前短板

### 1. 缺真实 benchmark

目前已经有 catalytic pocket benchmark 框架，可以从已加载的 UniProt / M-CSA / 文献 / AI 外部证据生成 benchmark reference candidate、import summary、review queue、checklist、decision loop 和 accepted candidate export，也可以上传 curated catalytic residues 并输出 Top-1 / Top-3 / Top-5 coverage、case/dataset summary、case interpretation matrix/summary/queue、case/dataset-level readiness-aware interpretation、dataset claim readiness queue/checklist/report、best hit rank、missed residues、P2Rank on/off 对照和 current vs ablation variant comparison。variant comparison 已支持整体、case、dataset 和 residue 四层视图，并能生成 lost/current-missed residue remediation queue、summary 和 checklist。但还缺少批量真实数据集，尚未证明不同算法组合在大样本上的命中率提升。

当前 reference source 已加安全边界：如果没有上传 curated benchmark 文件，系统会优先使用 reviewer 接受后的 accepted candidate；未审核的 external-evidence candidate 只作为显式开启的 provisional fallback，并在 snapshot/report/source audit/summary/case summary/case decision import/validation/outcomes/outcome summary/closure queue/readiness impact/readiness impact summary/dataset impact/dataset impact cases/closure checklist/case checklist/action queue/checklist 中记录 source mode、provisional、reviewed-candidate 和 claim 安全状态。source audit 已接入 decision-adjusted readiness gate，provisional reference 会成为 P0 blocker，review-qualified candidate 会成为 P2 independence review item；上传并通过的 source-audit case decisions 会让 cleared/replaced/source-ready case 不再被原始 source audit 重复阻断，blocked/pending/held/unknown case 则继续作为 P0/P1/P2 readiness issue。readiness impact 会逐 case 对照原始 source issue 与 decision-adjusted issue，解释哪些 case 被 decision 清除、哪些仍然 open；summary 会汇总 cleared/open case、原始/调整后 P0/P1/P2 数和 net blocker delta；dataset impact 会按 Top-N 汇总这些来源 decision 对 dataset-level claim readiness 的阻断、复核和 gate mismatch，dataset impact cases 则把每个 Top-N/case 的 claim status、coverage、adjusted priority 和 mismatch 拆出来，方便直接定位被来源闭环影响的酶/结构 case。case summary、case decision import/validation/outcomes/outcome summary/closure queue/closure checklist 和 case checklist 会按 benchmark_id 定位被来源问题阻断或需要独立性复核的酶/结构 case，并汇总 blocked/pending/held/replaced/cleared/source-ready 后给出 closure status、open case 数、下一步建议、可筛选 CSV 队列和可勾选关闭清单。

建议指标：

- Top-1 是否覆盖任一 catalytic residue。
- Top-3 是否覆盖多数 catalytic residues。
- catalytic residue 到 pocket center 的距离。
- direct-anchor pocket precision。
- evidence route 开启/关闭的 rank delta。
- 文献证据加入前后的命中率变化。

已落地能力：

- `pocket_benchmark_reference.csv`
- `pocket_benchmark_reference_candidate.csv`
- `pocket_benchmark_reference_import_summary.csv`
- `pocket_benchmark_reference_candidate_review_queue.csv`
- `pocket_benchmark_reference_candidate_review_checklist.md`
- `pocket_benchmark_reference_candidate_review_decision_template.csv`
- `pocket_benchmark_reference_candidate_review_decisions_normalized.csv`
- `pocket_benchmark_reference_candidate_review_decision_validation.csv`
- `pocket_benchmark_reference_candidate_review_outcomes.csv`
- `pocket_benchmark_reference_candidate_accepted.csv`
- `pocket_benchmark_reference_source_audit_summary.csv`
- `pocket_benchmark_reference_source_audit_case_summary.csv`
- `pocket_benchmark_reference_source_audit_case_decision_template.csv`
- `pocket_benchmark_reference_source_audit_case_decisions_normalized.csv`
- `pocket_benchmark_reference_source_audit_case_decision_validation.csv`
- `pocket_benchmark_reference_source_audit_case_decision_outcome_summary.csv`
- `pocket_benchmark_reference_source_audit_case_decision_closure_queue.csv`
- `pocket_benchmark_reference_source_audit_case_decision_readiness_impact.csv`
- `pocket_benchmark_reference_source_audit_case_decision_readiness_impact_summary.csv`
- `pocket_benchmark_reference_source_audit_case_decision_dataset_impact.csv`
- `pocket_benchmark_reference_source_audit_case_decision_dataset_impact_cases.csv`
- `pocket_benchmark_reference_source_audit_case_decision_closure_checklist.md`
- `pocket_benchmark_reference_source_audit_case_decision_outcomes.csv`
- `pocket_benchmark_reference_source_audit_case_checklist.md`
- `pocket_benchmark_reference_source_audit_action_queue.csv`
- `pocket_benchmark_reference_source_audit_checklist.md`
- `pocket_benchmark_reference_source_audit.csv`
- `pocket_benchmark_reference_template.csv`
- `pocket_benchmark_reference_template.md`
- `pocket_benchmark_reference_quality_issues.csv`
- `pocket_benchmark_reference_quality_summary.csv`
- `pocket_benchmark_reference_quality_checklist.md`
- `pocket_benchmark_reference_structure_validation.csv`
- `pocket_benchmark_reference_structure_validation_summary.csv`
- `pocket_benchmark_reference_structure_validation_checklist.md`
- `pocket_benchmark_reference_readiness_summary.csv`
- `pocket_benchmark_reference_readiness_case_summary.csv`
- `pocket_benchmark_reference_readiness_queue.csv`
- `pocket_benchmark_reference_readiness_checklist.md`
- `pocket_benchmark_interpretation.csv`
- `pocket_benchmark_case_interpretation.csv`
- `pocket_benchmark_case_interpretation_matrix.csv`
- `pocket_benchmark_case_interpretation_matrix_summary.csv`
- `pocket_benchmark_case_interpretation_matrix_queue.csv`
- `pocket_benchmark_dataset_interpretation.csv`
- `pocket_benchmark_dataset_interpretation_queue.csv`
- `pocket_benchmark_dataset_interpretation_checklist.md`
- `pocket_benchmark_dataset_interpretation_report.md`
- `pocket_benchmark_summary.csv`
- `pocket_benchmark_case_summary.csv`
- `pocket_benchmark_dataset_summary.csv`
- `pocket_benchmark_variant_comparison.csv`
- `pocket_benchmark_variant_case_comparison.csv`
- `pocket_benchmark_variant_dataset_comparison.csv`
- `pocket_benchmark_variant_residue_comparison.csv`
- `pocket_benchmark_variant_remediation_queue.csv`
- `pocket_benchmark_variant_remediation_summary.csv`
- `pocket_benchmark_variant_remediation_checklist.md`
- `pocket_benchmark_details.csv`
- `p2rank_ab_comparison.csv`
- 页面中展示 Top-1 / Top-3 coverage、case-level hit rate、dataset mean coverage、best hit rank，以及 no-p2rank / no-literature / no-evidence-route / no-conservation-rerank 在整体、case、dataset 和 residue 层面的 coverage loss，并把 P0/P1 remediation actions 汇总为 CSV 和 Markdown checklist。

### 2. 文献抽取仍偏规则驱动

当前抽取已经保留 `PMID/PMCID/DOI`、标题、证据片段、句子序号、抽取模式和 `requires_manual_review`，便于用户回看证据来源。但规则驱动方案仍会漏掉跨句 catalytic triad、表格、补充材料、非标准突变描述和成熟酶 offset。

下一步可引入 LLM/NLP 辅助，但必须继续保留 source、snippet、PMID/PMCID/DOI、manual review flag，并且不能让无片段或无稳定 citation 的结果直接升级为高置信口袋 anchor。

### 3. UniProt/PDB 对齐还可增强

后续应继续支持：

- mmCIF author/asym chain 双编号。
- insertion code。
- SEQRES / ATOM pairwise alignment。
- isoform、signal peptide、propeptide、mature chain offset。
- 缺失片段和突变位点标记。

### 4. 口袋边界仍可更精细

可继续增强：

- 局部表面可达性。
- pocket residue contact graph。
- 金属、辅因子、底物邻近残基单独建模。
- core/shell/rim 的可视化层级和导出解释。

### 5. 结构质量感知不足

需要更系统读取：

- PDB resolution / R-factor。
- AlphaFold pLDDT。
- 缺失残基和断链。
- 关键残基是否处于低置信区。

## 下一步优先级

### P0：扩展酶口袋 benchmark 数据集

数据来源建议：

- M-CSA 中有 PDB 结构和 catalytic residues 的酶。
- 带天然底物、辅因子或抑制剂的 holo PDB。
- 文献明确报道 catalytic triad/dyad 的酶。

### P1：继续增强证据片段级文献抽取

已落地字段：

- `article_title`
- `pmid`
- `pmcid`
- `doi`
- `evidence_snippet`
- `sentence_index`
- `extraction_pattern`
- `requires_manual_review`

下一步重点：

- 跨句 catalytic triad / dyad 合并。
- 表格和补充材料抽取。
- mature chain / isoform offset 解释。
- snippet 级 false positive 统计。

### P1：增强结构编号校验

重点是 mmCIF、insertion code、SEQRES/ATOM 对齐和 mature enzyme offset。

### P2：加入 family prior

可考虑 Pfam、InterPro、PROSITE、EC catalytic motif、homolog active-site transfer。

### P2：结果可信度解释页

每个推荐口袋应能解释：

- 为什么排第一。
- 哪些关键残基支持它。
- 哪些证据是 direct anchor，哪些只是 neighborhood expansion。
- 是否存在编号、结构缺失或低置信风险。
- 与 baseline 相比提升在哪里。

## 可用于答辩的核心表述

ProteinInsight 的优势不是发明了一个全新的几何口袋算法，而是把酶活性口袋定位从“几何凹陷搜索”提升为“关键残基证据驱动的可审计判断”。

同类工具能快速告诉用户哪里像口袋，但往往不能回答为什么这个口袋是酶的关键活性口袋。ProteinInsight 把 UniProt、M-CSA、SIFTS、PubMed/Europe PMC 文献、P2Rank、几何聚类、热点残基、界面和 conservation 证据合并到同一流程，并保留每一步的证据质量、映射质量、A/B 对照和 release closure audit。

这使系统更适合酶蛋白场景：已知关键残基时，系统围绕关键残基定位口袋；只有文献线索时，系统把文献残基结构化并映射到 PDB；证据有风险时，系统降级、提示复核或阻断 release，而不是盲目把它当作高置信口袋。
