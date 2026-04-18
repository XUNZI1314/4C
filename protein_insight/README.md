# ProteinInsight 子目录原型

`protein_insight/` 是早期任务型蛋白可视化原型目录，保留了独立的 `app.py`、`requirements.txt`、`src/`、`tests/` 和示例数据。

当前主线开发以仓库根目录的 `app.py`、`pages/`、`src/protein_visualizer/` 和 `tests/` 为准。除非需要复现早期原型，否则建议优先运行根目录应用。

## 原型能力

- 上传单个或多个 PDB。
- 上传 MMPBSA residue energy CSV。
- 将残基能量映射到 3D 结构。
- 自动识别热点残基。
- 基础口袋/界面展示。
- 多构象热点一致性分析。
- residue annotation table 和 CSV 导出。
- 可选 PyMOL 静态渲染后端；不可用时回退到 3Dmol。

## 运行方式

```powershell
cd protein_insight
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 测试

```powershell
cd protein_insight
pytest -q
```

## 与主线应用的区别

主线应用已经新增并强化：

- UniProt / M-CSA / SIFTS 外部证据。
- PubMed / Europe PMC 文献残基抽取。
- 可选 P2Rank。
- Conservation rerank-only。
- Evidence route、literature A/B、conservation A/B。
- Pocket evidence quality、core/shell/rim 分层。
- Consensus rerank guardrail、release approval、execution receipt 和 closure audit。
- Handoff ZIP、manifest、verification、certificate、closure detached manifest。

因此，比赛展示和后续产品化建议使用仓库根目录应用。

## PyMOL 说明

如果本机安装 PyMOL 或 `pymol2`，原型可以调用 PyMOL Python API 输出 PNG。若使用官方评估版，图片可能带有 “No License File - For Evaluation Only” 水印，这不是程序错误。

Smoke test 示例：

```powershell
& 'C:\Users\acd\AppData\Local\Schrodinger\PyMOL2\python.exe' .\scripts\pymol_smoke_test.py
```

输出位置：

```text
data/examples/pymol_smoke_test.png
```
