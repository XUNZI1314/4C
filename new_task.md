# 当前任务清单

更新时间：2026-04-18

## 总目标

在不重写整体架构的前提下，把 ProteinInsight 打造成面向酶蛋白的关键残基证据驱动口袋定位工具，并让高风险 rerank 具备可审批、可执行、可关闭、可归档的审计链。

## 已完成

- [x] 智能口袋排序：`pocket_ranker.py`
- [x] 口袋 / 界面 / 热点联合推荐：`candidate_fusion.py`
- [x] UniProt / M-CSA / SIFTS 外部证据接入。
- [x] 文献残基抽取：PubMed、Europe PMC、手动文本。
- [x] 文献证据片段级字段：`article_title`、`pmid`、`pmcid`、`doi`、`evidence_snippet`、`sentence_index`、`extraction_pattern`、`requires_manual_review`。
- [x] 结构辅助 UniProt/PDB 映射质量校验。
- [x] P2Rank 本地可选接入。
- [x] P2Rank on/off A/B 和 no-p2rank benchmark variant。
- [x] Conservation rerank-only 信号和 A/B 对比。
- [x] External-evidence route 候选口袋生成。
- [x] Literature A/B 和 evidence-route A/B。
- [x] Catalytic pocket benchmark：提供 reference template、external-evidence reference candidate/import summary/review queue/checklist/decision loop/accepted candidate export、reference curation quality check、PDB structure validation、case-level readiness gate 和 case/dataset-level readiness-aware interpretation，上传 curated catalytic residues，计算 Top-1 / Top-3 / Top-5 coverage、case/dataset summary、case interpretation matrix/summary/queue、dataset claim readiness queue/checklist/report、best hit rank、missed residues、整体/case/dataset/residue 四层 current vs ablation variant comparison，以及 remediation queue / summary / checklist。
- [x] Benchmark reference source control：无 curated 文件时优先 accepted reviewed candidate，未审核 external-evidence candidate 只作为显式 provisional fallback，并在 snapshot/report/source audit 记录 source mode 和 claim 安全状态。
- [x] Evidence quality label / score / warning。
- [x] Pocket core / shell / rim 分层。
- [x] Snapshot、report、history 中记录关键口袋证据。
- [x] Consensus rerank suggestion、preview、policy gate、action queue、checklist。
- [x] Precision scorecard、guardrail、guardrail report。
- [x] Handoff ZIP、artifact manifest、ZIP verification、handoff certificate。
- [x] Release decision template、decision import、validation、summary。
- [x] Release apply plan、apply report。
- [x] Release execution template、receipt import、validation、summary、report。
- [x] Release closure certificate、closure ledger。
- [x] Detached closure summary、blocker queue、remediation checklist、detached manifest。
- [x] 单元测试和全量测试覆盖，最近一次验证：`228 passed`。

## 当前导出重点

### ZIP 内证据包

- `consensus_rerank_guardrail_handoff.zip`
- `consensus_rerank_guardrail_artifact_manifest.csv`
- guardrail / scorecard / simulation / queue / release / execution / closure certificate / closure ledger 等文件。

### ZIP 外 detached closure pack

- `consensus_rerank_release_closure_summary.csv`
- `consensus_rerank_release_closure_blocker_queue.csv`
- `consensus_rerank_release_closure_remediation_checklist.md`
- `consensus_rerank_release_closure_detached_manifest.csv`

这些文件依赖 ZIP verification 输出，因此保持在 ZIP 外，避免哈希循环。

## 下一步建议

### P0：扩展真实 benchmark 数据集

- 已落地 benchmark 评估框架：上传 curated catalytic residues 后可输出 Top-1 / Top-3 / Top-5 catalytic coverage，并按 benchmark_id/case_id 输出 case summary 与 dataset summary。
- 已落地 current vs no-p2rank / no-literature / no-evidence-route / no-conservation-rerank 的整体、case、dataset 和 residue 四层 benchmark variant comparison，并生成 remediation queue / summary / checklist，用 coverage loss 与 lost/gained residue 定位证据路径贡献。
- 已落地 benchmark reference candidate/import summary/review queue/checklist/decision loop/accepted candidate export，可把已加载的 UniProt / M-CSA / 文献 / AI 残基证据转换成待复核的 benchmark reference candidate，再把 weak mapping、wildcard chain、missing resname 和 manual-review 风险拆成行动队列，由 reviewer 回填 accept/reject/hold 决策，最终只导出 clean 或全部风险 action 被接受的 reference candidate，并明确提示不能直接当作独立精度证明。
- 已落地 benchmark reference source control：在无 curated benchmark 文件时，accepted reviewed candidate 优先于 provisional external-evidence candidate；provisional 路径必须显式开启，并会在 snapshot/report/source audit 标记为 provisional 或 review-qualified。
- 已落地 benchmark reference template CSV/Markdown，方便后续收集 M-CSA + PDB + literature-confirmed catalytic residues。
- 已落地 benchmark reference curation quality issues / summary / checklist，先检查 benchmark_id、source、chain、resname、编号假设和重复角色，再解释 coverage。
- 已落地 benchmark reference structure validation，检查 curated residues 是否存在于当前 PDB、resname 是否匹配、空 chain 是否造成多链歧义。
- 已落地 benchmark reference readiness gate，把 curation quality 和 structure validation 合并为 blocked / review-needed / ready，并导出 summary、queue 和 checklist。
- 已落地 benchmark reference readiness case summary，按 benchmark_id/case_id 定位 blocked / review-needed / ready 的具体酶或结构 case。
- 已落地 readiness-aware benchmark interpretation，把 Top-N coverage 标记为 claim-ready / review-needed / blocked / readiness-unknown，避免把不可靠参考集上的 coverage 当作精度结论。
- 已落地 benchmark case interpretation，把每个 benchmark_id/case_id 的 Top-N coverage 和 case readiness 绑定，区分哪些 case 可声明、待复核或被阻断。
- 已落地 benchmark case interpretation matrix/summary/queue，把每个 benchmark_id 横向展开 Top-1 / Top-3 / Top-5 的 claim status、coverage、best rank 和 best pocket，并汇总 Top-1/Top-3/Top-5 最早可声明 case 数，再把非 claim-ready case 转成一行一个 case 的 triage queue。
- 已落地 benchmark dataset interpretation，按 Top-N 汇总 claim-ready、blocked、review-needed 和 unknown case 数量、比例与平均 coverage，避免单个 case 状态被总 coverage 掩盖。
- 已落地 benchmark dataset interpretation queue/checklist/report，把非 claim-ready case 拆成 P0 blocker / P2 review 队列、人工勾选清单和可归档报告，方便优先修复阻断 dataset-level claim 的 case。
- 下一步收集 M-CSA + PDB + catalytic residue 数据集，形成可重复运行的真实批量 benchmark。
- 下一步扩展批量数据集层面的独立贡献评估。

### P1：继续增强文献证据片段

- 已落地结构化 citation/snippet/manual-review 字段，并在口袋页提供外部证据明细预览和 CSV 导出。
- 下一步处理跨句 triad、表格、补充材料和非标准突变描述。
- 下一步把 snippet 级别证据纳入 benchmark 误差分析，区分真实 catalytic support 和弱上下文命中。

### P1：增强残基编号映射

- mmCIF。
- insertion code。
- SEQRES/ATOM alignment。
- isoform / mature chain offset。

### P1：增强可视化

- 在 3D 视图中突出 core/shell/rim。
- 支持 closure 状态和 blocker 数在页面顶部形成摘要卡片。
- 将 evidence anchors、mapping risk 和 A/B rank delta 放进更直观的口袋详情面板。

### P2：实验闭环

- 导出 PyMOL / ChimeraX session。
- 生成 mutation suggestion。
- 生成 docking preparation hint。
- 加入 substrate/cofactor template matching。

## 工作原则

- 不让弱证据自动覆盖几何和结构质量。
- 不默认自动应用 rerank。
- 所有高风险排序变化必须经过 manual review、apply plan、execution receipt 和 closure gate。
- 每个导出包必须能追溯来源、byte size 和 SHA-256。
