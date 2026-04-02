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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("蛋白质可视化软件")
st.subheader("首页")

st.markdown(
    """
    <div class="intro-box">
        <h3 style="margin-top:0;">软件简介</h3>
        <p>
            这是一个面向蛋白质结构展示与 MMPBSA 能量热力分析的交互式软件原型，
            适用于比赛展示、课程设计和生物信息学可视化演示。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    ### 功能入口
    - **首页**：查看软件简介与模块说明
    - **结构可视化**：加载 PDB、交互查看蛋白质结构
    - **结果与导出**：查看当前分析结果并导出数据
    - **使用说明**：查看软件操作步骤与输入格式要求
    """
)

col1, col2 = st.columns(2)
with col1:
    st.info("适用场景：蛋白质结构可视化、残基能量分析、比赛演示、课程设计。")
with col2:
    st.success("软件特点：多页面结构、模块化代码、可部署、可扩展。")

st.markdown("---")
st.markdown("### 软件设计目标")
st.write("- 提供直观的 3D 蛋白质查看界面")
st.write("- 支持 MMPBSA DELTA TOTAL 残基能量热力映射")
st.write("- 提供易用的交互控件与结果展示")
st.write("- 形成可部署的软件原型系统")
