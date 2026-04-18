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
- [x] Catalytic pocket benchmark：上传 curated catalytic residues，计算 Top-1 / Top-3 / Top-5 coverage、case/dataset summary、best hit rank、missed residues，以及整体/case/dataset 三层 current vs ablation variant comparison。
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
- [x] 单元测试和全量测试覆盖，最近一次验证：`203 passed`。

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
- 已落地 current vs no-p2rank / no-literature / no-evidence-route / no-conservation-rerank 的整体、case 和 dataset 三层 benchmark variant comparison，用 coverage loss 量化证据路径贡献。
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
