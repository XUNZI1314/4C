from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.config.settings import SETTINGS

st.set_page_config(page_title=f"{SETTINGS.page_title} - 使用说明", layout="wide")

st.title("使用说明")

st.markdown(
    """
    ## 1. 输入数据
    软件支持两类输入：
    - `PDB` 文件：蛋白质三维结构
    - `MMPBSA` 输出文件：包含 `DELTA TOTAL` 残基能量

    ## 2. 操作流程
    1. 打开“结构可视化”页面
    2. 上传 PDB 文件
    3. 上传 MMPBSA 输出文件
    4. 调整阈值、显示模式、透明度
    5. 选择链与残基进行高亮查看
    6. 在“结果与导出”页面查看表格、快照与 PDF 报告
    7. 在“分析历史”页面查看已保存的分析记录
    8. 在“多构象比较”和“口袋与界面”页面做进一步分析

    ## 3. 显示模式说明
    - **球棒**：适合观察局部原子结构
    - **表面**：适合观察分子表面包络
    - **透明**：适合同时观察整体与局部

    ## 4. 颜色分类说明
    - 可以按 `DELTA TOTAL` 热度查看连续渐变
    - 也可以切换为按理化性质、电荷、极性、热点等级或按链着色
    - 结构页会同步显示颜色图例和残基注释表

    ## 5. 颜色含义
    - 低能量残基：红色
    - 高能量残基：蓝色
    - 未达到阈值：灰色

    ## 6. 分析历史
    - 历史记录会写入本地数据文件，刷新页面后仍可保留
    - 记录中会包含平均能量、蛋白质体积、热点数和口袋数等摘要

    ## 7. 注意事项
    - MMPBSA 文件需包含可解析的 `DELTA TOTAL`
    - 若格式不标准，可能需要后续扩展解析器
    - 大型 PDB 文件会增加浏览器渲染压力
    - 结构估算模式用于没有 MMPBSA 文件时的近似分析，不等同于真实 MMPBSA 结果
    """
)
