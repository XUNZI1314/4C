# 当前任务清单

更新时间：2026-04-19

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
- [x] Evidence-seeded pocket route：关键残基直接生成邻域口袋，而不只是参与后处理加权。
- [x] 空间多样性拆分：避免 CD38 这类大酶的几何/共识候选被单一连通簇吞掉，只输出一个错误口袋。
- [x] CD38 活性位点模板：检测到 CD38/P28907 语境时，按结构实际匹配的人 CD38 编号残基 W125/R127/E146/D155/W189/S193/T221/E226 生成需复核证据锚点。
- [x] Literature A/B 和 evidence-route A/B。
- [x] Catalytic pocket benchmark：提供 reference template、external-evidence reference candidate/import summary/review queue/checklist/decision loop/accepted candidate export、reference curation quality check、PDB structure validation、case-level readiness gate 和 case/dataset-level readiness-aware interpretation，上传 curated catalytic residues，计算 Top-1 / Top-3 / Top-5 coverage、case/dataset summary、case interpretation matrix/summary/queue、dataset claim readiness queue/checklist/report、best hit rank、missed residues、整体/case/dataset/residue 四层 current vs ablation variant comparison，以及 remediation queue / summary / checklist。
- [x] Benchmark reference source control：无 curated 文件时优先 accepted reviewed candidate，未审核 external-evidence candidate 只作为显式 provisional fallback，并在 snapshot/report/source audit/summary/case summary/case decision import/validation/outcomes/outcome summary/closure queue/readiness impact/readiness impact summary/dataset impact/dataset impact cases/dataset impact case checklist/dataset impact action queue/action summary/artifact manifest/dataset impact report/closure checklist/case checklist/action queue/checklist 与 decision-adjusted readiness gate 记录 source mode 和 claim 安全状态。
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
- [x] 单元测试和全量测试覆盖，最近一次验证：`270 passed`。

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
- 已落地 benchmark reference source audit summary：按 source mode 和 claim status 汇总 reference 行数、case 数、provisional 行数和 reviewed candidate 行数，方便快速判断 reference 来源整体风险。
- 已落地 benchmark reference source audit action queue：把非 source-ready reference source risk 转成只针对来源风险的结构化 CSV 动作队列，便于单独安排人工复核。
- 已落地 benchmark reference source audit case summary：按 benchmark_id/case_id 汇总来源 blocker、review item、top issue 和 source mode，便于批量 benchmark 时优先修复受影响的酶或结构 case。
- 已落地 benchmark reference source audit case decision template：只把有 blocker/review/action 的 case 导出成 reviewer 可回填的 CSV 模板，为后续来源决策导入和验证做准备。
- 已落地 benchmark reference source audit case decision import/validation：回填 CSV 会被标准化，并校验未知 case、未知 decision、缺 reviewer、缺独立性证明、缺 replacement source 和 hold 备注。
- 已落地 benchmark reference source audit case decision outcomes：把校验后的 case 决策应用为 blocked / pending / held / replaced / cleared / source-ready，方便判断来源风险是否真正关闭。
- 已落地 benchmark reference source audit case decision outcome summary：按 applied status 汇总 closed/open source-risk cases，给出 closure status、open case 数和下一步建议，方便判断来源风险闭环是否可用于精度声明。
- 已落地 benchmark reference source audit case decision closure queue：把未关闭 outcomes 转成可筛选 CSV 行动队列，标记 P0/P1/P2、blocker/review、issue type 和 required action，便于批量指派整改。
- 已落地 benchmark reference source audit case decision readiness impact：逐 case 对照原始 source audit issue 与 decision-adjusted readiness issue，解释哪些来源风险被 decision 清除、哪些仍然 open。
- 已落地 benchmark reference source audit case decision readiness impact summary：汇总 cleared/open case、原始/调整后 P0/P1/P2 数和 net blocker delta，用一行判断 source decision 对 readiness gate 的净影响。
- 已落地 benchmark reference source audit case decision dataset impact：按 Top-N 汇总 source decision 对 dataset-level claim readiness 的阻断、复核和 source-gate mismatch，避免来源闭环状态只停留在 case 级。
- 已落地 benchmark reference source audit case decision dataset impact cases：按 Top-N/case 连接 claim status、coverage、source decision readiness impact、adjusted priority 和 mismatch，直接定位被来源闭环影响的酶/结构 case。
- 已落地 benchmark reference source audit case decision dataset impact case checklist：只把 blocker/review/mismatch case 转成 Markdown 勾选项，方便 reviewer 按 Top-N/case 指派和关闭。
- 已落地 benchmark reference source audit case decision dataset impact action queue：把 blocker/review/mismatch case 转成 machine-readable CSV 队列，便于按 priority/action_status/source_gate_mismatch 批量筛选和指派。
- 已落地 benchmark reference source audit case decision dataset impact action summary：按 priority/action_status/source_impact_status 汇总 action 数、受影响 case、Top-N、mismatch 和首要 action，方便批量 benchmark 时先看哪类来源问题最该处理。
- 已落地 benchmark reference source audit case decision dataset impact artifact manifest：给 dataset impact CSV、cases、action queue、action summary、checklist 和 report 记录 byte size 与 SHA-256，方便后续归档和交付完整性校验。
- 已落地 benchmark reference source audit case decision dataset impact report：把 source gate、Top-N impact 汇总、action summary 和 case actions 组合成可归档 Markdown 报告，便于随 benchmark 结果交付。
- 已落地 benchmark reference source audit case decision closure checklist：把 outcome summary 和未关闭 outcomes 转成 Markdown 勾选清单，方便 reviewer 按 blocked/pending/held/unknown case 继续整改并归档。
- 已落地 benchmark reference source audit case checklist：把 source audit case summary 和逐行来源动作组合成 case-first Markdown 清单，便于 reviewer 按酶/结构 case 逐个关闭来源风险。
- 已落地 benchmark reference source audit checklist：把 source audit summary 和非 source-ready reference 行转成可勾选 Markdown 动作，便于人工整改和归档。
- 已落地 decision-adjusted source-aware benchmark readiness gate：provisional reference source 进入 P0 blocker，review-qualified accepted candidate 进入 P2 independence review item；已 cleared/replaced/source-ready 的 source-audit case 不再被原始来源风险重复阻断，blocked/pending/held/unknown outcome 会继续进入 P0/P1/P2 readiness issue，避免 coverage 在 reference 来源未闭环时被解释为精度声明。
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
