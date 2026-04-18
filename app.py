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
        <div class="header-card">
          <h1 style="margin:0;">ProteinInsight</h1>
          <div style="font-size:14px;opacity:0.9">蛋白质可视化与 MMPBSA 热力分析平台</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        html, body { background: #f6f9fc; color:#0f1724; font-family: 'Noto Sans SC', 'Source Han Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif; }
        .header-card { background: linear-gradient(90deg,#2b6cb0,#805ad5); color: #fff; padding: 14px; border-radius: 12px; box-shadow: 0 6px 18px rgba(11,38,77,0.08); margin-bottom: 12px; }
        .hero-shell { background: linear-gradient(135deg, rgba(43,108,176,0.10), rgba(128,90,213,0.10)); border: 1px solid rgba(43,108,176,0.16); border-radius: 18px; padding: 18px; margin: 14px 0 18px 0; }
        .hero-grid { display: grid; grid-template-columns: 1.3fr 0.9fr; gap: 14px; align-items: start; }
        .hero-aside { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .hero-title { font-size: 28px; line-height: 1.1; margin: 0 0 10px 0; color: #0f1724; }
        .hero-copy { font-size: 15px; line-height: 1.75; color: #334155; margin: 0; }
        .info-chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; margin: 8px 8px 0 0; border-radius: 999px; background: rgba(255,255,255,0.88); border: 1px solid rgba(15,23,42,0.08); font-size: 12px; color: #0f1724; }
        .feature-card { background: #fff; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px rgba(15,23,42,0.04); }
        .feature-card h4 { margin: 0 0 6px 0; }
        .feature-card p { margin: 0; line-height: 1.6; }
        .stButton>button { background-color: #2563eb; color:#fff; border-radius:8px; }
        .hero-card { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### 导航说明
        - `首页`：软件总览
        - `结构可视化`：蛋白质 3D 展示、残基分类与 MMPBSA 热力分析
        - `结果与导出`：查看结果、分类统计、快照导出与 PDF 报告
        - `分析历史`：查看已保存的分析记录
        - `多构象比较`：比较多个构象的热点与相似性
        - `口袋与界面`：查看口袋、界面和重叠注释
        - `使用说明`：查看操作帮助
        """
    )

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div style="letter-spacing:0.12em;text-transform:uppercase;font-size:12px;color:#1d4ed8;font-weight:700;">ProteinInsight</div>
                    <h2 class="hero-title">把结构、能量、热点和分类注释放在同一条分析链里。</h2>
                    <p class="hero-copy">
                        这个软件不只是展示蛋白质 3D 结构，还把残基分类、热点识别、口袋标记、多构象比较、结果快照和 PDF 报告连成一套工作流，
                        适合比赛展示和快速复核分析结果。
                    </p>
                    <div>
                        <span class="info-chip">结构可视化</span>
                        <span class="info-chip">残基分类图例</span>
                        <span class="info-chip">热点识别</span>
                        <span class="info-chip">快照导出</span>
                    </div>
                </div>
                <div>
                    <div class="hero-aside">
                    <div class="feature-card">
                        <strong>分析模式</strong>
                        <div style="margin-top:6px;color:#475569;">卡通视图 / 球棍视图 / 分子表面</div>
                    </div>
                    <div class="feature-card">
                        <strong>分析能力</strong>
                        <div style="margin-top:6px;color:#475569;">口袋分析、界面注释、多构象比较、历史记录</div>
                    </div>
                    <div class="feature-card">
                        <strong>分析链路</strong>
                        <div style="margin-top:6px;color:#475569;">结构输入 → 能量映射 → 热点/口袋定位 → 证据复核</div>
                    </div>
                    <div class="feature-card">
                        <strong>导出内容</strong>
                        <div style="margin-top:6px;color:#475569;">CSV、JSON 快照、SVG 快照、PDF 报告</div>
                    </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
                st.markdown('<div class="feature-card"><h4>三维结构可视化</h4><p>支持主链、侧链、球棍和表面模式，并兼容 PyMOL 与 3Dmol 渲染路径。</p></div>', unsafe_allow_html=True)
    with col2:
                st.markdown('<div class="feature-card"><h4>比较与口袋分析</h4><p>支持多构象热点对比、口袋/界面标注与重叠分析，便于复核关键位点。</p></div>', unsafe_allow_html=True)
    with col3:
                st.markdown('<div class="feature-card"><h4>结果导出</h4><p>支持导出分析表格、JSON/SVG 快照、残基注释 CSV、文本报告与 PDF 报告。</p></div>', unsafe_allow_html=True)
