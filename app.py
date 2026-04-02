from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.config.settings import SETTINGS


if __name__ == "__main__":
    st.set_page_config(page_title=SETTINGS.page_title, layout=SETTINGS.layout)
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, #0d6efd 0%, #6f42c1 100%);
            color: white;
            padding: 24px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(13,110,253,0.25);
            margin-bottom: 18px;
        }
        .feature-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 16px;
            min-height: 140px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title(SETTINGS.page_title)
    st.markdown(
        """
        <div class="hero-card">
            <h2 style="margin-top:0;">蛋白质可视化与 MMPBSA 热力分析平台</h2>
            <p style="font-size:16px;margin-bottom:0;">
                面向比赛展示与课程设计的交互式软件，支持蛋白质三维结构浏览、残基能量映射、结果导出与多页面分析流程。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### 导航说明
        - `首页`：软件总览
        - `结构可视化`：蛋白质 3D 展示与 MMPBSA 热力分析
        - `结果与导出`：查看结果数据并导出
        - `使用说明`：查看操作帮助
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><h4>3D 结构可视化</h4><p>支持主链、侧链、球棒和表面模式，便于展示蛋白质空间结构。</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><h4>MMPBSA 热力分析</h4><p>将 DELTA TOTAL 映射为残基颜色，辅助定位关键能量区域。</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><h4>结果导出</h4><p>支持导出分析表格与文本报告，适合汇报展示与材料提交。</p></div>', unsafe_allow_html=True)
