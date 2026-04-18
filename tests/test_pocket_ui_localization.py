from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
POCKET_PAGE = ROOT_DIR / "pages" / "6_口袋与界面.py"
RESULTS_PAGE = ROOT_DIR / "pages" / "4_结果与导出.py"


def _page_source() -> str:
    return POCKET_PAGE.read_text(encoding="utf-8")


def _results_page_source() -> str:
    return RESULTS_PAGE.read_text(encoding="utf-8")


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
    ]

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
        '"pocket_id": "口袋 ID"',
        '"recommendation_reason": "推荐理由"',
    ]

    for snippet in forbidden_mojibake_snippets:
        assert snippet not in source
    for snippet in required_snippets:
        assert snippet in source
