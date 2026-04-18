from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.config.settings import SETTINGS

st.set_page_config(page_title=f"{SETTINGS.page_title} - 首页", layout="wide")

st.markdown(
    """
    <style>
    .intro-box {
        background: linear-gradient(135deg, #eef4ff 0%, #f7f0ff 100%);
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #dbeafe;
        margin-bottom: 16px;
    }
    .hero-shell {
        background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(168,85,247,0.10));
        border: 1px solid rgba(37,99,235,0.14);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
    }
    .stat-card {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        margin: 8px 8px 0 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(15,23,42,0.08);
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-shell">
        <div style="letter-spacing:0.12em;text-transform:uppercase;font-size:12px;color:#1d4ed8;font-weight:700;">ProteinInsight</div>
        <h2 style="margin:10px 0 12px 0;">蛋白质结构、残基分类与能量分析的一体化展示。</h2>
        <p style="margin:0;line-height:1.8;color:#334155;">
            这是一个面向蛋白质结构展示与 MMPBSA 热力分析的交互式软件原型，
            现在进一步加入了残基分类图例、热点注释表、多构象比较、口袋/界面分析、结果快照和导出能力，适合比赛展示、课程设计和生物信息学可视化演示。
        </p>
        <div>
            <span class="pill">3D 可视化</span>
            <span class="pill">热点识别</span>
            <span class="pill">分类图例</span>
            <span class="pill">快照导出</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 你可以直接做什么")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        "<div class='stat-card'><strong>结构展示</strong><br><span style='color:#475569;'>cartoon / sticks / surface，适合从整体到局部逐层查看。</span></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        "<div class='stat-card'><strong>比较与分析</strong><br><span style='color:#475569;'>支持多构象比较、口袋/界面分析和历史记录查看。</span></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        "<div class='stat-card'><strong>结果导出</strong><br><span style='color:#475569;'>CSV、JSON 快照、SVG 快照、残基注释表和 PDF 都可以直接下载。</span></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    ### 功能入口
    - **首页**：查看软件简介与模块说明
    - **结构可视化**：加载 PDB、交互查看蛋白质结构、查看分类图例
    - **结果与导出**：查看当前分析结果并导出数据、快照和报告
    - **分析历史**：查看已保存的分析摘要
    - **多构象比较**：比较多个结构的热点与相似性
    - **口袋与界面**：查看口袋、界面和重叠注释
    - **使用说明**：查看软件操作步骤与输入格式要求
    """
)

col1, col2 = st.columns(2)
with col1:
    st.info("适用场景：蛋白质结构可视化、残基能量分析、比赛演示、课程设计。")
with col2:
    st.success("软件特点：多页面结构、模块化代码、可部署、可扩展、支持快照导出与分析历史持久化。")

st.markdown("---")
st.markdown("### 软件设计目标")
st.write("- 提供直观的 3D 蛋白质查看界面")
st.write("- 支持 MMPBSA DELTA TOTAL 残基能量热力映射")
st.write("- 提供易用的交互控件、分类图例与残基注释表")
st.write("- 形成可部署的软件原型系统")
