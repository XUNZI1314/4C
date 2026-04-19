from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PAGE = ROOT_DIR / "app.py"
HOME_PAGE = ROOT_DIR / "pages" / "1_首页.py"
HELP_PAGE = ROOT_DIR / "pages" / "3_使用说明.py"
STRUCTURE_LAYOUT = ROOT_DIR / "src" / "protein_visualizer" / "ui" / "layout.py"
POCKET_PAGE = ROOT_DIR / "pages" / "6_口袋与界面.py"
RESULTS_PAGE = ROOT_DIR / "pages" / "4_结果与导出.py"
MULTI_CONFORMATION_PAGE = ROOT_DIR / "pages" / "4_多构象比较.py"
HISTORY_PAGE = ROOT_DIR / "pages" / "5_分析历史.py"


def _app_source() -> str:
    return APP_PAGE.read_text(encoding="utf-8")


def _home_source() -> str:
    return HOME_PAGE.read_text(encoding="utf-8")


def _help_source() -> str:
    return HELP_PAGE.read_text(encoding="utf-8")


def _structure_layout_source() -> str:
    return STRUCTURE_LAYOUT.read_text(encoding="utf-8")


def _page_source() -> str:
    return POCKET_PAGE.read_text(encoding="utf-8")


def _results_page_source() -> str:
    return RESULTS_PAGE.read_text(encoding="utf-8")


def _multi_conformation_page_source() -> str:
    return MULTI_CONFORMATION_PAGE.read_text(encoding="utf-8")


def _history_page_source() -> str:
    return HISTORY_PAGE.read_text(encoding="utf-8")


def _visible_streamlit_lines(source: str) -> str:
    visible_tokens = (
        "st.caption",
        "st.subheader",
        "st.header",
        "st.markdown",
        "st.info",
        "st.warning",
        "st.error",
        "st.success",
        "st.expander",
        "st.spinner",
        "st.metric",
    )
    return "\n".join(
        line.strip()
        for line in source.splitlines()
        if any(token in line for token in visible_tokens)
    )


def test_pocket_page_keeps_visible_runtime_messages_localized():
    visible_source = _visible_streamlit_lines(_page_source())

    forbidden_visible_snippets = [
        'with st.spinner("Loading external functional-site evidence...',
        'with st.spinner("Loading literature residue evidence...',
        'with st.spinner("Loading AI residue evidence...',
        'st.subheader("region_type 计数对比")',
        "rank_delta 为",
        "Benchmark reference warning",
        "benchmark case",
        " case 决策",
        " case 汇总",
        " case 解释",
    ]

    for snippet in forbidden_visible_snippets:
        assert snippet not in visible_source


def test_pocket_page_keeps_display_localization_hooks_enabled():
    source = _page_source()

    required_snippets = [
        "st.download_button = _localized_download_button",
        "st.dataframe = _localized_dataframe",
        "def _localize_json_for_display",
        'st.json(_localize_json_for_display(auto_detection_meta))',
        '"rank_delta": "排名变化"',
        '"region_type": "区域类型"',
        '"category": "类别"',
        'st.subheader("区域类型计数对比")',
        "排名变化值为正",
        "基准案例",
        'f"AI 影响:',
        'f"Top 活性位点决策:',
        'f"可靠性检查: 通过',
        '"上传人工关键残基 CSV/TSV（可选）"',
        '"下载人工关键残基模板 CSV"',
        "manual_key_residue_evidence_template.csv",
        "manual_key_residue_evidence.csv",
        "parse_manual_key_residue_table",
        "人工关键残基：",
        '"导出人工关键残基证据 CSV"',
        "def _needs_manual_key_residue_evidence",
        "当前候选口袋仍缺少可审计功能残基证据",
        "下载人工关键残基补证模板 CSV",
        "decision_manual_key_residue_template",
        "def _build_manual_key_residue_followup_df",
        "def _manual_evidence_overlap_for_pocket",
        "人工关键残基补证任务",
        "人工证据状态",
        "已补人工证据（需确认链/编号）",
        "重新运行自动口袋识别",
        "def _summarize_manual_key_residue_followup_df",
        "补证闭环状态",
        "发布门控",
        "不可直接作为活性位点",
        "决策缺口任务数",
        "补证已上传，等待决策缺口复核",
        "def _build_manual_key_residue_followup_checklist_markdown",
        "补证复跑检查清单",
        "导出人工关键残基补证复跑检查清单 MD",
        "manual_key_residue_followup_checklist_markdown",
        "manual_key_residue_followup_checklist.md",
        "download_manual_key_residue_followup_checklist",
        "export_manual_key_residue_followup_checklist",
        "def _build_manual_key_residue_collection_template_df",
        "补证采集模板",
        "下载按口袋预填的人工关键残基补证采集模板 CSV",
        "导出按口袋预填的人工关键残基补证采集模板 CSV",
        "manual_key_residue_collection_template_df",
        "manual_key_residue_collection_template.csv",
        "download_manual_key_residue_collection_template",
        "export_manual_key_residue_collection_template",
        "导出人工关键残基补证闭环总览 CSV",
        "manual_key_residue_followup_summary_df",
        "manual_key_residue_followup_summary.csv",
        "download_manual_key_residue_followup_summary",
        "导出人工关键残基补证任务 CSV",
        "manual_key_residue_followup_df",
        "manual_key_residue_followup_tasks.csv",
        "download_manual_key_residue_followup_tasks",
    ]

    for snippet in required_snippets:
        assert snippet in source


def test_pocket_page_report_text_is_built_with_localized_precision_labels():
    source = _page_source()

    forbidden_snippets = [
        'f"AI influence:',
        'f"Top active-site decision:',
        'f"Top decision score:',
        'f"Precision tier:',
        'f"Triage action:',
        'f"Reliability checks:',
        'f"Reliability gaps:',
        'f"Next step:',
    ]
    required_snippets = [
        'f"AI 影响:',
        'f"Top 活性位点决策:',
        'f"Top 决策评分:',
        'f"精度分层:',
        'f"分诊动作:',
        'f"可靠性检查: 通过',
        'f"可靠性缺口:',
        'f"下一步:',
    ]

    for snippet in forbidden_snippets:
        assert snippet not in source
    for snippet in required_snippets:
        assert snippet in source


def test_pocket_page_ai_export_buttons_are_explicitly_localized():
    source = _page_source()

    forbidden_snippets = [
        '"Export AI evidence CSV"',
        '"Export AI evidence audit CSV"',
        '"Export normalized AI review decisions CSV"',
        '"Export AI review decision validation CSV"',
        '"Export AI review artifact bundle ZIP"',
        '"Export ranking-gated AI evidence CSV"',
        '"Export AI follow-up prompt bundle"',
    ]
    required_snippets = [
        '"导出 AI 残基证据 CSV"',
        '"导出 AI 证据审计 CSV"',
        '"导出规范化 AI 复核决策 CSV"',
        '"导出 AI 复核决策校验 CSV"',
        '"导出 AI 复核产物包 ZIP"',
        '"导出通过排名门控的 AI 证据 CSV"',
        '"导出 AI 后续提示词包"',
    ]

    for snippet in forbidden_snippets:
        assert snippet not in source
    for snippet in required_snippets:
        assert snippet in source


def test_results_page_keeps_pocket_export_labels_readable():
    source = _results_page_source()

    forbidden_mojibake_snippets = [
        "鏅鸿兘",
        "鍙ｈ",
        "鎽樿",
        "瀵煎嚭",
        "鏄庣粏",
        "锛?",
    ]
    required_snippets = [
        'st.subheader("智能口袋摘要")',
        'st.caption(f"Top1 口袋：',
        'label="导出智能口袋摘要 CSV"',
        'label="导出口袋明细 CSV"',
        "def localize_display_table",
        "def localize_column_name",
        '"pocket_id": "口袋 ID"',
        '"recommendation_reason": "推荐理由"',
        '"recommendation": "推荐"',
        '"overlap": "重叠"',
    ]

    for snippet in forbidden_mojibake_snippets:
        assert snippet not in source
    for snippet in required_snippets:
        assert snippet in source


def test_multi_conformation_page_localizes_display_tables():
    source = _multi_conformation_page_source()

    required_snippets = [
        "def localize_display_table",
        '"conformation": "构象"',
        '"mean_energy_delta_vs_reference": "相对参考平均能量差"',
        '"reference_overlap_ratio": "参考重叠率"',
        "localize_display_table(per_conformation_df)",
        "localize_display_table(reference_comparison_df)",
        "localize_display_table(stable_hotspots_df",
        "fallback_chart = chart_frame.rename",
    ]

    for snippet in required_snippets:
        assert snippet in source


def test_history_page_keeps_evidence_labels_localized():
    source = _history_page_source()

    forbidden_snippets = [
        "Top1 evidence quality",
        "Top1 evidence warning",
        "pred `",
        "res `",
        "exact `",
        "weak `",
    ]
    required_snippets = [
        "def localize_history_table",
        '"top_pocket_evidence_quality_label": "Top1 证据质量"',
        "Top1 证据质量",
        "Top1 证据提醒",
        "预测行",
        "弱匹配",
        "localize_history_table(history_df[available_columns])",
    ]

    for snippet in forbidden_snippets:
        assert snippet not in source
    for snippet in required_snippets:
        assert snippet in source


def test_structure_page_localizes_display_tables():
    source = _structure_layout_source()

    forbidden_snippets = [
        'st.dataframe(df_display, use_container_width=True',
        'st.dataframe(pocket_summary, use_container_width=True',
        'comparison.get("summary_table"',
        "hotspot_sets.append(set())",
    ]
    required_snippets = [
        "def localize_display_table",
        "def localize_column_name",
        '"delta_total": "总能量变化"',
        '"pocket_id": "口袋 ID"',
        '"is_common": "是否共同热点"',
        "localize_display_table(df_display)",
        "localize_display_table(hotspot_display)",
        "localize_display_table(pocket_summary)",
        'comparison.get("per_residue_df", pd.DataFrame())',
        "hotspot_tables.append(hs)",
    ]

    for snippet in forbidden_snippets:
        assert snippet not in source
    for snippet in required_snippets:
        assert snippet in source


def test_app_uses_chinese_first_font_stack():
    source = _app_source()

    assert "font-family: Inter, -apple-system" not in source
    assert "'Noto Sans SC', 'Source Han Sans SC', 'Microsoft YaHei', 'PingFang SC'" in source
    assert "cartoon / sticks / surface" not in source
    assert "卡通视图 / 球棍视图 / 分子表面" in source
    assert "结构输入 → 能量映射 → 热点/口袋定位 → 证据复核" in source


def test_home_page_keeps_visible_mode_labels_localized():
    source = _home_source()

    assert "cartoon / sticks / surface" not in source
    assert "卡通视图、球棍视图和分子表面" in source
    assert "推荐分析链路" in source
    assert "证据复核" in source


def test_help_page_documents_current_pocket_workflow():
    source = _help_source()

    required_snippets = [
        "推荐分析链路",
        "UniProt、M-CSA、文献、AI 提取残基、保守性表格或人工关键残基",
        "文献 A/B、证据路径 A/B、保守性 A/B 和 P2Rank on/off 对照",
        "AI 残基证据需要来源文本、片段或引用支持",
        "Top1 口袋证据质量",
        "PDB author numbering、插入码或成熟肽编号不一致",
    ]

    for snippet in required_snippets:
        assert snippet in source
