# ProteinInsight

ProteinInsight 是一个面向酶蛋白的结构可视化、口袋定位、关键残基证据融合和审计导出的 Streamlit 工具。

项目目标不是只回答“结构表面哪里像口袋”，而是回答更适合酶蛋白场景的问题：哪些候选口袋真正围绕催化残基、底物/辅因子结合残基、金属配位残基或文献支持的关键位点形成，并且这些结论能否被复核、导出和归档。

## 当前能力

- PDB / MMPBSA 上传、示例数据、热点残基识别和结构可视化。
- 自动口袋识别：优先 `pyKVFinder`，可选本地 `P2Rank`，并保留 ligand proximity、geometry clustering 和 external-evidence route。
- 外部证据融合：UniProt、PDBe/SIFTS、M-CSA、PubMed、Europe PMC、手动文献文本、保守性表格、热点和界面信息。
- 强化 UniProt/PDB 残基对齐：优先结构中真实存在的链和残基，降低缺失、gap、弱映射和 residue identity mismatch 的置信度。
- 文献残基抽取：支持 catalytic、active site、mutagenesis、activity-loss 等上下文过滤，保留 `PMID/PMCID/DOI`、标题、证据片段、抽取模式和人工复核标记，并做跨文章支持统计。
- 证据驱动口袋排序：输出 evidence quality、direct anchor、route anchor、mapping risk、A/B rank delta 和 practical recommendation。
- Catalytic pocket benchmark：提供 reference template、external-evidence reference candidate/import summary/review queue/checklist/decision loop/accepted candidate export、reference curation quality check、PDB structure validation、case-level readiness gate 和 case/dataset-level readiness-aware interpretation，上传 curated catalytic residues 后输出 Top-1 / Top-3 / Top-5 coverage、case/dataset summary、case interpretation matrix/summary/queue、dataset claim readiness queue/checklist/report、best hit rank、missed residues，并可按整体、case、dataset 和 residue 层面对比 current 与 no-p2rank / no-literature / no-evidence-route / no-conservation-rerank 的 coverage loss，同时生成 remediation queue、summary 和 checklist。
- Benchmark reference source control：无 curated 文件时优先使用已审核 accepted candidate 作为 benchmark reference；未审核 external-evidence candidate 只能作为显式开启的 provisional fallback，source audit 会进入 readiness gate，并生成 summary/checklist 供整改归档；provisional reference 直接阻断精度声明，review-qualified candidate 需要独立性复核。
- 口袋分层：将口袋残基标注为 `core`、`shell`、`rim`，便于区分关键位点和边界残基。
- Consensus rerank 审计链：从建议、预览、policy gate、action queue、scorecard、guardrail 到 release approval、apply plan、execution receipt 和 closure。
- 导出：CSV、JSON snapshot、SVG snapshot、TXT report、PDF report、Markdown report、handoff ZIP、manifest 和 closure detached manifest。
- 分析历史持久化：默认写入 `data/analysis_history.json`。

## Release / Closure 审计链

精度敏感的 consensus rerank 不会默认自动应用。当前流程分为以下阶段：

1. `consensus_rerank_suggestions.csv`：证据驱动的 rerank 建议。
2. `consensus_rerank_preview.csv`：保守分数预览。
3. `consensus_rerank_policy_gate.csv`：全局安全 gate。
4. `consensus_rerank_action_queue.csv` 和 `consensus_rerank_action_checklist.md`：待修复问题和人工清单。
5. `consensus_rerank_precision_scorecard.csv`：预期精度收益与 blocker 汇总。
6. `consensus_rerank_precision_guardrail.csv` 和 `consensus_rerank_precision_guardrail_report.md`：是否允许进入人工 release review。
7. `consensus_rerank_guardrail_handoff.zip` 和 `consensus_rerank_guardrail_artifact_manifest.csv`：可归档的证据包。
8. `consensus_rerank_guardrail_bundle_verification.csv` 和 summary：校验 ZIP 中每个文件的 byte size 和 SHA-256。
9. `consensus_rerank_guardrail_handoff_certificate.md`：ZIP 交付证明。
10. `consensus_rerank_release_decision_template.csv`：人工审批表。
11. `consensus_rerank_release_decisions_normalized.csv`、validation、summary：审批回传解析和验证。
12. `consensus_rerank_release_apply_plan.csv` 和 apply report：只有审批通过且模拟干净时生成。
13. `consensus_rerank_release_execution_template.csv`：执行回执模板。
14. `consensus_rerank_release_execution_receipt_normalized.csv`、validation、summary、report：执行结果回传和验证。
15. `consensus_rerank_release_closure_certificate.md`：最终闭环证书。
16. `consensus_rerank_release_closure_ledger.csv`：闭环证据结构化 ledger。
17. `consensus_rerank_release_closure_summary.csv`：ZIP 外 detached readiness gate。
18. `consensus_rerank_release_closure_blocker_queue.csv`：闭环失败时的修复队列。
19. `consensus_rerank_release_closure_remediation_checklist.md`：人工修复清单。
20. `consensus_rerank_release_closure_detached_manifest.csv`：ZIP 外 closure 产物的 SHA-256 索引。

`closure_summary`、`closure_blocker_queue`、`closure_remediation_checklist` 和 `closure_detached_manifest` 是 ZIP 外产物，因为它们依赖 ZIP verification 输出，不能放回同一个 ZIP 里形成哈希循环。

## 快速开始

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 启动应用

```powershell
streamlit run app.py
```

默认访问地址：

```text
http://localhost:8501
```

### 3. 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

最近一次完整验证结果：

```text
231 passed
```

## 输入文件

- PDB：标准 `.pdb`，至少包含可解析的 `ATOM/HETATM` 行。
- MMPBSA：`.txt`、`.dat`、`.out`、`.csv`，需能解析 residue-level `DELTA TOTAL`。
- 口袋文件，可选：CSV，推荐包含 `pocket_id, chain, resid, resname, volume, score`。
- 界面注释，可选：CSV，推荐包含 `chain, resid, resname, annotation, region_type`。
- Conservation，可选：ConSurf / Rate4Site / generic residue score table。
- 文献文本，可选：abstract、full-text snippet、XML snippet 或人工整理的 residue evidence。

## 页面说明

- `pages/1_首页.py`：产品入口和总览。
- `pages/3_使用说明.py`：使用说明。
- `pages/4_多构象比较.py`：多构象比较。
- `pages/4_结果与导出.py`：结果查看和导出。
- `pages/5_分析历史.py`：历史记录。
- `pages/6_口袋与界面.py`：口袋、界面、证据融合、rerank 审计和 closure workflow。

## 关键目录

```text
.
├─ app.py
├─ pages/
├─ src/protein_visualizer/
│  ├─ services/
│  └─ ui/
├─ tests/
├─ docs/
├─ data/
├─ requirements.txt
├─ pyproject.toml
└─ Dockerfile
```

## 文档索引

- `advantage.md`：同类产品缺陷、我们的优势、当前能力和后续改进。
- `new_task.md`：当前实现状态和后续任务清单。
- `docs/pocket_detection_enhancement.md`：口袋检测、外部证据、文献、P2Rank、rerank closure 的技术记录。
- `docs/docker_local_setup.md`：本地 Docker 安装、构建和运行说明。
- `protein_insight/README.md`：子目录原型说明。

## Docker

```powershell
docker build -t protein-visualizer .
docker run -d --name protein-visualizer -p 8501:8501 -v ${PWD}\data:/app/data protein-visualizer
```

或：

```powershell
docker compose up --build -d
```

## 常见问题

- 页面没有数据：先完成一次结构/口袋分析，或启用示例数据。
- PDF 不可用：安装 `reportlab` 后重启应用。
- P2Rank 无结果：确认已安装 P2Rank，并设置 `P2RANK_HOME`、`P2RANK_SCRIPT` 或在页面中填写可执行文件路径。
- 文献证据过弱：优先提供 UniProt accession、EC number、PDB ID 和结构编号一致性信息。
- closure 不能关闭：查看 `consensus_rerank_release_closure_summary.csv`、blocker queue 和 remediation checklist。
- 历史无法写入：检查 `data/analysis_history.json` 和 `data/` 目录权限。
