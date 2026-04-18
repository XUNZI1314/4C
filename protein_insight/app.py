from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_insight.analysis import detect_hotspots
from protein_insight.coloring import (
    CLASSIFICATION_THEME_COLORS,
    CLASSIFICATION_THEME_DESCRIPTIONS,
    CLASSIFICATION_THEME_OPTIONS,
    annotate_residue_table,
    build_chain_color_map,
    build_mode_legend,
    build_surface_block_color_map,
    resolve_residue_color,
)
from protein_insight.compare import compare_hotspots
from protein_insight.explain import explain_results
from protein_insight.io import load_mmpbsa_csv, load_pdb, load_pocket_json
from protein_insight.pocket import detect_pocket_by_ligand, get_pocket_residues
from protein_insight.pymol_colors import cycle_palette, resolve_color
from protein_insight.vis import render_pdb


st.set_page_config(page_title="ProteinInsight", layout="wide")


DATA_DIR = ROOT_DIR / "data" / "examples"
SAMPLE_PDB = DATA_DIR / "sample1.pdb"
SAMPLE_MMPBSA = DATA_DIR / "sample_mmpbsa.csv"
SAMPLE_POCKET = DATA_DIR / "sample_pocket.json"

COLOR_MODES = [
    "按氨基酸理化性质",
    "按电荷状态",
    "按侧链极性",
    "按MMPBSA等级",
    "按热点等级",
    "按链",
    "按能量连续梯度",
    "单色",
    "PyMOL 调色板",
]

PALETTES = ["PyMOL 经典", "PyMOL 高对比", "PyMOL 冷色", "PyMOL 暖色", "PyMOL 莫兰迪"]
SURFACE_THEME_ACCENT_PALETTES = {
    "清新海洋": "PyMOL 冷色",
    "专业高对比": "PyMOL 高对比",
    "柔和莫兰迪": "PyMOL 莫兰迪",
}
CLASSIFICATION_THEME_LABELS = {
    "清新海洋": "清新海洋 · 冷色清爽",
    "专业高对比": "专业高对比 · 强烈对撞",
    "柔和莫兰迪": "柔和莫兰迪 · 低饱和雾面",
}


def _theme_preview_html(colors: Sequence[str]) -> str:
    swatches = "".join(
        f"<span style='display:inline-block;width:12px;height:12px;border-radius:999px;background:{color};border:1px solid rgba(15,23,42,0.14);'></span>"
        for color in list(colors)[:8]
    )
    return f"<div style='display:flex;gap:4px;flex-wrap:wrap;margin:6px 0 2px 0;'>{swatches}</div>"


def header() -> None:
    st.markdown(
        """
        <div style="padding:14px 16px;border-radius:12px;background:linear-gradient(135deg,#1f4d8f,#2b7abf);color:#fff;margin-bottom:10px;">
          <h2 style="margin:0;">ProteinInsight</h2>
          <div style="opacity:0.95;font-size:14px;">任务型蛋白可视化：结构 + 能量 + 热点 + 口袋</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_inputs() -> Dict[str, object]:
    sidebar = st.sidebar
    sidebar.header("输入")
    pdb_upload = sidebar.file_uploader("上传 PDB（支持多个）", type=["pdb"], accept_multiple_files=True)
    mmpbsa_upload = sidebar.file_uploader("上传 MMPBSA CSV（可选，支持多个）", type=["csv"], accept_multiple_files=True)
    pocket_upload = sidebar.file_uploader("上传口袋 JSON（可选，支持多个）", type=["json"], accept_multiple_files=True)
    use_examples = sidebar.checkbox("使用示例数据", value=False)

    sidebar.header("可视化")
    display_mode = sidebar.selectbox(
        "显示模式",
        ["cartoon", "sticks", "surface"],
        format_func=lambda x: {"cartoon": "卡通", "sticks": "棒状", "surface": "表面"}[x],
    )
    surface_render_mode = "稳定单色（推荐）"
    if display_mode == "surface":
        surface_render_mode = sidebar.selectbox(
            "表面渲染策略",
            ["稳定单色（推荐）", "分类着色（实验）"],
            index=0,
            help="分类着色会直接使用当前配色模式的残基颜色；若加载较慢可切回稳定单色。",
        )
    cartoon_theme = sidebar.selectbox("卡通风格", ["PyMOL 风格", "链色卡通", "彩虹卡通", "简洁卡通"], index=0)
    surface_opacity = sidebar.slider("表面透明度", 0.05, 1.0, 0.8, 0.05)
    color_mode = sidebar.selectbox("配色模式", COLOR_MODES, index=0)
    classification_theme = sidebar.selectbox(
        "分类配色主题",
        CLASSIFICATION_THEME_OPTIONS,
        index=0,
        format_func=lambda theme: CLASSIFICATION_THEME_LABELS.get(theme, theme),
        help="影响理化、电荷、极性、MMPBSA、热点等分类模式的表面颜色。",
    )
    sidebar.markdown(
        _theme_preview_html(CLASSIFICATION_THEME_COLORS[classification_theme]["property"]),
        unsafe_allow_html=True,
    )
    sidebar.caption(CLASSIFICATION_THEME_DESCRIPTIONS[classification_theme])
    palette_name = sidebar.selectbox("调色板", PALETTES, index=0)
    mono_color = sidebar.color_picker("单色颜色", "#d9d9d9")
    surface_color = sidebar.color_picker("表面底色", "#d6e6ff")
    surface_colorize = display_mode == "surface" and surface_render_mode == "分类着色（实验）"
    mark_pocket_residues = sidebar.checkbox(
        "口袋残基覆盖着色",
        value=False,
        disabled=(display_mode == "surface" and not surface_colorize),
    )
    pocket_color = sidebar.color_picker("口袋颜色", "#ffd60a")

    sidebar.header("分析")
    energy_mode = sidebar.selectbox(
        "能量来源模式",
        ["auto", "mmpbsa", "estimate"],
        index=0,
        format_func=lambda x: {"auto": "自动", "mmpbsa": "上传 MMPBSA", "estimate": "结构估算"}[x],
    )
    energy_threshold = sidebar.slider("能量阈值（更低更重要）", -20.0, 10.0, -1.0, 0.1)
    top_k = int(sidebar.number_input("显示前 N 个热点", min_value=1, max_value=50, value=5))
    run_button = sidebar.button("运行分析", type="primary")

    return {
        "pdb_upload": pdb_upload,
        "mmpbsa_upload": mmpbsa_upload,
        "pocket_upload": pocket_upload,
        "use_examples": use_examples,
        "display_mode": display_mode,
        "surface_render_mode": surface_render_mode,
        "cartoon_theme": cartoon_theme,
        "surface_opacity": surface_opacity,
        "surface_color": surface_color,
        "surface_colorize": surface_colorize,
        "color_mode": color_mode,
        "classification_theme": classification_theme,
        "palette_name": palette_name,
        "mono_color": mono_color,
        "mark_pocket_residues": mark_pocket_residues,
        "pocket_color": pocket_color,
        "energy_mode": energy_mode,
        "energy_threshold": energy_threshold,
        "top_k": top_k,
        "run_button": run_button,
    }


def extract_residues_from_pdb(pdb_str: str) -> List[Tuple[str, int, str]]:
    residues: List[Tuple[str, int, str]] = []
    seen = set()
    for line in str(pdb_str).splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            chain = line[21].strip() or "A"
            resid = int(line[22:26].strip())
            resname = line[17:20].strip().upper()
        except Exception:
            continue
        key = (chain, resid)
        if key in seen:
            continue
        seen.add(key)
        residues.append((chain, resid, resname))
    return residues


def estimate_energy_table_from_structure(pdb_str: str) -> pd.DataFrame:
    residues = extract_residues_from_pdb(pdb_str)
    if not residues:
        return pd.DataFrame(columns=["chain", "resid", "resname", "energy"])

    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET"}
    aromatic = {"PHE", "TYR", "TRP"}
    charged = {"ASP", "GLU", "LYS", "ARG", "HIS"}
    polar = {"SER", "THR", "ASN", "GLN", "CYS"}
    flexible = {"GLY", "PRO"}

    rows = []
    total = max(1, len(residues) - 1)
    for index, (chain, resid, resname) in enumerate(residues):
        if resname in hydrophobic:
            base = -1.8
        elif resname in aromatic:
            base = -2.2
        elif resname in charged:
            base = 1.1
        elif resname in polar:
            base = 0.2
        elif resname in flexible:
            base = 0.7
        else:
            base = 0.0

        center = abs((index / total) - 0.5)
        smooth = math.sin(index * 0.33) * 0.25
        energy = base + center * 0.4 + smooth
        rows.append({"chain": chain, "resid": resid, "resname": resname, "energy": float(energy)})

    return pd.DataFrame(rows)


def resolve_energy_table(
    pdb_str: str,
    *,
    energy_mode: str,
    mmpbsa_table: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], str]:
    mode = (energy_mode or "auto").strip().lower()
    if mode not in {"auto", "mmpbsa", "estimate"}:
        mode = "auto"

    uploaded = None
    if mmpbsa_table is not None and not mmpbsa_table.empty:
        uploaded = mmpbsa_table.copy()

    if mode == "mmpbsa":
        return (uploaded, "MMPBSA数据") if uploaded is not None else (None, "无可用能量数据")

    if mode == "estimate":
        return estimate_energy_table_from_structure(pdb_str), "结构估算"

    if uploaded is not None:
        return uploaded, "MMPBSA数据"

    return estimate_energy_table_from_structure(pdb_str), "结构估算"


def _align_optional_inputs(items: Sequence[object], target_size: int) -> List[object]:
    if target_size <= 0:
        return []
    if not items:
        return [None for _ in range(target_size)]
    if len(items) == 1 and target_size > 1:
        return [items[0] for _ in range(target_size)]
    if len(items) >= target_size:
        return list(items[:target_size])
    return list(items) + [None for _ in range(target_size - len(items))]


def load_inputs(
    pdb_upload_list,
    mmpbsa_files,
    pocket_files,
    use_examples_flag: bool,
):
    pdbs: List[str] = []
    names: List[str] = []

    if pdb_upload_list and len(pdb_upload_list) > 0:
        for file_obj in pdb_upload_list:
            pdbs.append(load_pdb(file_obj))
            names.append(getattr(file_obj, "name", "uploaded.pdb"))
    elif use_examples_flag and SAMPLE_PDB.exists():
        pdbs.append(load_pdb(SAMPLE_PDB))
        names.append(SAMPLE_PDB.name)
    else:
        return None

    mmpbsa_list: List[Optional[pd.DataFrame]] = []
    if mmpbsa_files and len(mmpbsa_files) > 0:
        for file_obj in mmpbsa_files:
            try:
                mmpbsa_list.append(load_mmpbsa_csv(file_obj))
            except Exception as exc:
                st.warning(f"解析 MMPBSA 文件失败：{exc}")
                mmpbsa_list.append(None)
    elif use_examples_flag and SAMPLE_MMPBSA.exists():
        sample_df = load_mmpbsa_csv(SAMPLE_MMPBSA)
        mmpbsa_list = [sample_df for _ in pdbs]

    pocket_list: List[Optional[dict]] = []
    if pocket_files and len(pocket_files) > 0:
        for file_obj in pocket_files:
            try:
                pocket_list.append(load_pocket_json(file_obj))
            except Exception:
                pocket_list.append(None)
    elif use_examples_flag and SAMPLE_POCKET.exists():
        sample_pocket = load_pocket_json(SAMPLE_POCKET)
        pocket_list = [sample_pocket for _ in pdbs]

    return pdbs, names, mmpbsa_list, pocket_list


def build_display_residue_table(pdb_str: str, energy_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    residues = extract_residues_from_pdb(pdb_str)
    base_df = pd.DataFrame(residues, columns=["chain", "resid", "resname"]) if residues else pd.DataFrame()
    if energy_df is None or energy_df.empty:
        if base_df.empty:
            return pd.DataFrame(columns=["chain", "resid", "resname", "energy"])
        base_df["energy"] = pd.NA
        return base_df

    mmp_df = energy_df[["chain", "resid", "resname", "energy"]].copy()
    mmp_df["chain"] = mmp_df["chain"].astype(str).replace("", "A")
    mmp_df["resid"] = pd.to_numeric(mmp_df["resid"], errors="coerce").astype("Int64")
    mmp_df = mmp_df.dropna(subset=["resid"]).copy()
    mmp_df["resid"] = mmp_df["resid"].astype(int)

    if base_df.empty:
        return mmp_df.sort_values(by=["chain", "resid"]).reset_index(drop=True)

    merged = base_df.merge(mmp_df[["chain", "resid", "energy"]], on=["chain", "resid"], how="left")
    merged = merged.sort_values(by=["chain", "resid"]).reset_index(drop=True)
    return merged


def build_residue_color_map(
    table: pd.DataFrame,
    color_mode: str,
    classification_theme: str,
    palette_name: str,
    mono_color: str,
    hotspot_rank_map: Optional[Dict[Tuple[str, int], int]] = None,
) -> Dict[Tuple[str, int], str]:
    if table is None or table.empty:
        return {}

    ordered = table.sort_values(by=["chain", "resid"]).reset_index(drop=True)
    chains = []
    for _, row in ordered.iterrows():
        chain = str(row.get("chain", "A"))
        if chain not in chains:
            chains.append(chain)

    chain_color_map = build_chain_color_map(chains, palette_name)
    valid_energy = ordered["energy"].dropna()
    energy_bounds = None
    if not valid_energy.empty:
        energy_bounds = (float(valid_energy.min()), float(valid_energy.max()))

    cmap: Dict[Tuple[str, int], str] = {}
    for _, row in ordered.iterrows():
        chain = str(row.get("chain", "A"))
        resid = int(row["resid"])
        resname = str(row.get("resname", ""))
        energy = float(row["energy"]) if pd.notna(row["energy"]) else None
        hotspot_rank = hotspot_rank_map.get((chain, resid)) if hotspot_rank_map else None
        color = resolve_residue_color(
            color_mode,
            resname=resname,
            energy=energy,
            chain=chain,
            chain_color_map=chain_color_map,
            hotspot_rank=hotspot_rank,
            energy_bounds=energy_bounds,
            mono_color=mono_color,
            theme_name=classification_theme,
        )
        cmap[(chain, resid)] = resolve_color(color)

    return cmap


def _surface_accent_palette_name(classification_theme: str) -> str:
    return SURFACE_THEME_ACCENT_PALETTES.get(classification_theme, "PyMOL 冷色")


def _is_neutral_like_color(color: str) -> bool:
    value = str(color or "").strip().lower()
    if not value:
        return True

    neutral_palette = {
        "#b8c1cc",
        "#c7c7c7",
        "#cccccc",
        "#d0d0d0",
        "#d1d5db",
        "#d8dde6",
        "#e2e8f0",
        "#f8fafc",
        "#ffffff",
    }
    if value in neutral_palette:
        return True

    if value.startswith("#") and len(value) == 7:
        try:
            red = int(value[1:3], 16)
            green = int(value[3:5], 16)
            blue = int(value[5:7], 16)
        except Exception:
            return False

        spread = max(red, green, blue) - min(red, green, blue)
        mean = (red + green + blue) / 3.0
        return spread <= 18 and 108 <= mean <= 244

    return False


def apply_sparse_energy_surface_fallback(
    table: pd.DataFrame,
    color_map: Dict[Tuple[str, int], str],
    *,
    display_mode: str,
    color_mode: str,
    palette_name: str,
) -> Tuple[Dict[Tuple[str, int], str], bool, float]:
    if table is None or table.empty:
        return color_map, False, 1.0

    total = len(table)
    valid = int(table["energy"].notna().sum())
    coverage = valid / total if total else 1.0

    if display_mode != "surface":
        return color_map, False, coverage

    if color_mode not in {"按能量连续梯度", "按MMPBSA等级", "按热点等级"}:
        return color_map, False, coverage

    if coverage >= 0.65:
        return color_map, False, coverage

    fallback_count = max(48, min(512, total * 2))
    fallback_palette = cycle_palette("PyMOL 冷色", fallback_count)
    if not fallback_palette:
        fallback_palette = ["#d9d9d9"]

    chain_order: Dict[str, int] = {}
    patched = dict(color_map)
    for index, row in enumerate(table.itertuples(index=False)):
        chain = str(getattr(row, "chain", "A") or "A")
        resid = int(getattr(row, "resid"))
        if chain not in chain_order:
            chain_order[chain] = len(chain_order)
        if pd.notna(getattr(row, "energy", None)):
            continue

        palette_index = (chain_order[chain] * 29 + resid * 17 + index * 7 + total) % len(fallback_palette)
        patched[(chain, resid)] = resolve_color(fallback_palette[palette_index])

    return patched, True, coverage


def _blend_hex(color_a: str, color_b: str, ratio: float) -> str:
    first = str(color_a or "").strip().lower()
    second = str(color_b or "").strip().lower()
    if not (first.startswith("#") and len(first) == 7 and second.startswith("#") and len(second) == 7):
        return first or second

    try:
        a_red = int(first[1:3], 16)
        a_green = int(first[3:5], 16)
        a_blue = int(first[5:7], 16)
        b_red = int(second[1:3], 16)
        b_green = int(second[3:5], 16)
        b_blue = int(second[5:7], 16)
    except Exception:
        return first

    weight = max(0.0, min(1.0, float(ratio)))

    def _mix(channel_a: int, channel_b: int) -> int:
        return max(0, min(255, int(round(channel_a * (1.0 - weight) + channel_b * weight))))

    return f"#{_mix(a_red, b_red):02x}{_mix(a_green, b_green):02x}{_mix(a_blue, b_blue):02x}"


def _rgb_distance(color_a: str, color_b: str) -> float:
    first = str(color_a or "").strip().lower()
    second = str(color_b or "").strip().lower()
    if not (first.startswith("#") and len(first) == 7 and second.startswith("#") and len(second) == 7):
        return 999.0

    try:
        a_red = int(first[1:3], 16)
        a_green = int(first[3:5], 16)
        a_blue = int(first[5:7], 16)
        b_red = int(second[1:3], 16)
        b_green = int(second[3:5], 16)
        b_blue = int(second[5:7], 16)
    except Exception:
        return 999.0

    return float(abs(a_red - b_red) + abs(a_green - b_green) + abs(a_blue - b_blue))


def _boost_surface_contrast(
    table: pd.DataFrame,
    color_map: Dict[Tuple[str, int], str],
    *,
    palette_name: str,
) -> Dict[Tuple[str, int], str]:
    if table is None or table.empty or not color_map:
        return color_map

    ordered = table.sort_values(by=["chain", "resid"]).reset_index(drop=True)
    anchors = cycle_palette(palette_name, max(64, min(1400, len(ordered) * 3)))
    if not anchors:
        return color_map

    boosted = dict(color_map)
    previous_by_chain: Dict[str, str] = {}

    for row in ordered.itertuples(index=False):
        chain = str(getattr(row, "chain", "A") or "A")
        resid = int(getattr(row, "resid"))
        key = (chain, resid)
        if key not in boosted:
            continue

        seed = sum(ord(ch) for ch in chain) * 13 + resid * 17 + len(ordered) * 5
        anchor_primary = resolve_color(anchors[seed % len(anchors)])
        anchor_secondary = resolve_color(anchors[(seed * 3 + 7) % len(anchors)])
        anchor_tertiary = resolve_color(anchors[(seed * 5 + 11) % len(anchors)])
        base = resolve_color(boosted[key])
        mixed = anchor_primary if _is_neutral_like_color(base) else base

        previous_color = previous_by_chain.get(chain)
        if previous_color:
            candidates = [mixed, anchor_primary, anchor_secondary, anchor_tertiary]
            best = mixed
            best_distance = _rgb_distance(best, previous_color)
            for candidate in candidates[1:]:
                candidate_distance = _rgb_distance(candidate, previous_color)
                if candidate_distance > best_distance:
                    best = candidate
                    best_distance = candidate_distance
            if best_distance < 88 and _rgb_distance(anchor_secondary, previous_color) >= best_distance:
                best = anchor_secondary
            mixed = best

        boosted[key] = mixed
        previous_by_chain[chain] = mixed

    return boosted


def format_energy(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    return f"{float(value):.2f}"


def build_energy_summary(table: Optional[pd.DataFrame]) -> Dict[str, object]:
    if table is None or table.empty:
        return {
            "residue_count": 0,
            "valid_energy_count": 0,
            "mean_energy": None,
            "min_energy": None,
            "max_energy": None,
            "energy_coverage": 0.0,
        }

    values = pd.to_numeric(table["energy"], errors="coerce")
    valid = values.dropna()
    residue_count = int(len(table))
    valid_count = int(len(valid))
    return {
        "residue_count": residue_count,
        "valid_energy_count": valid_count,
        "mean_energy": float(valid.mean()) if valid_count else None,
        "min_energy": float(valid.min()) if valid_count else None,
        "max_energy": float(valid.max()) if valid_count else None,
        "energy_coverage": (valid_count / residue_count) if residue_count else 0.0,
    }


def main() -> None:
    header()
    inputs = sidebar_inputs()

    if not inputs["run_button"]:
        st.info("请在左侧上传 PDB/MMPBSA 文件，或勾选“使用示例数据”后点击“运行分析”。")
        return

    loaded = load_inputs(
        inputs["pdb_upload"],
        inputs["mmpbsa_upload"],
        inputs["pocket_upload"],
        bool(inputs["use_examples"]),
    )
    if not loaded:
        st.error("请上传至少一个 PDB 文件，或启用示例数据。")
        return

    pdbs, names, mmpbsa_list_raw, pocket_list_raw = loaded
    mmpbsa_list = _align_optional_inputs(mmpbsa_list_raw, len(pdbs))
    pocket_list = _align_optional_inputs(pocket_list_raw, len(pdbs))

    resolved_tables: List[Optional[pd.DataFrame]] = []
    resolved_sources: List[str] = []
    resolved_pocket_residues: List[List[Tuple[str, int]]] = []

    for index, pdb_str in enumerate(pdbs):
        pocket = pocket_list[index]
        pocket_res = get_pocket_residues(pocket, pdb_str)
        if not pocket_res:
            try:
                detected = detect_pocket_by_ligand(pdb_str)
                pocket_res = get_pocket_residues(detected, pdb_str)
            except Exception:
                pocket_res = []

        energy_df, source = resolve_energy_table(
            pdb_str,
            energy_mode=str(inputs["energy_mode"]),
            mmpbsa_table=mmpbsa_list[index],
        )

        resolved_tables.append(energy_df)
        resolved_sources.append(source)
        resolved_pocket_residues.append(pocket_res)

    valid_tables = [df for df in resolved_tables if df is not None and not df.empty]
    multi_compare_df = compare_hotspots(valid_tables, top_k=int(inputs["top_k"]), energy_threshold=float(inputs["energy_threshold"])) if len(valid_tables) > 1 else None

    left, right = st.columns([2.2, 1.05])

    with left:
        st.subheader("3D 结构视图")
        pdb_index = 0
        if len(pdbs) > 1:
            pdb_index = st.selectbox("选择构象", list(range(len(pdbs))), format_func=lambda i: names[i])

        pdb_str = pdbs[pdb_index]
        pocket = pocket_list[pdb_index]
        pocket_res = resolved_pocket_residues[pdb_index]
        energy_df = resolved_tables[pdb_index]
        energy_source_label = resolved_sources[pdb_index]

        display_table = build_display_residue_table(pdb_str, energy_df)
        hotspots = detect_hotspots(
            energy_df,
            top_k=int(inputs["top_k"]),
            energy_threshold=float(inputs["energy_threshold"]),
        )
        hotspot_rank_map = {(h.chain, h.resid): idx + 1 for idx, h in enumerate(hotspots)}

        display_mode = str(inputs["display_mode"])
        classification_theme = str(inputs["classification_theme"])
        surface_single_color_mode = (
            display_mode == "surface"
            and str(inputs.get("surface_render_mode", "稳定单色（推荐）")) == "稳定单色（推荐）"
        )
        surface_palette_name = str(inputs["palette_name"])
        surface_accent_palette_name = _surface_accent_palette_name(classification_theme)

        res_color_map = build_residue_color_map(
            display_table,
            color_mode=str(inputs["color_mode"]),
            classification_theme=classification_theme,
            palette_name=surface_palette_name,
            mono_color=str(inputs["mono_color"]),
            hotspot_rank_map=hotspot_rank_map,
        )

        res_color_map, fallback_applied, coverage = apply_sparse_energy_surface_fallback(
            display_table,
            res_color_map,
            display_mode=str(inputs["display_mode"]),
            color_mode=str(inputs["color_mode"]),
            palette_name=str(inputs["palette_name"]),
        )

        if bool(inputs.get("mark_pocket_residues")):
            pocket_hex = resolve_color(str(inputs.get("pocket_color", "#ffd60a")))
            for pr in pocket_res:
                res_color_map[(str(pr[0]), int(pr[1]))] = pocket_hex

        if display_mode == "surface":
            render_surface_colorize = not surface_single_color_mode
            if render_surface_colorize:
                render_color_map = dict(res_color_map)
                block_color_map = build_surface_block_color_map(
                    display_table,
                    palette_name=surface_accent_palette_name,
                    block_size=3,
                )
                detail_count = max(64, min(1024, max(1, len(display_table)) * 3))
                detail_palette = cycle_palette(surface_accent_palette_name, detail_count)
                if not detail_palette:
                    detail_palette = ["#d9d9d9"]
                for key, accent in block_color_map.items():
                    base = render_color_map.get(key, accent)
                    chain, resid = key
                    seed = sum(ord(ch) for ch in str(chain)) * 7 + int(resid) * 11
                    detail = resolve_color(detail_palette[seed % len(detail_palette)])
                    if _is_neutral_like_color(base):
                        render_color_map[key] = accent
                        continue
                    selector = seed % 6
                    if selector in {0, 1, 2, 3}:
                        render_color_map[key] = base
                    elif selector == 4:
                        render_color_map[key] = accent
                    else:
                        render_color_map[key] = detail
                if bool(inputs.get("mark_pocket_residues")):
                    pocket_hex = resolve_color(str(inputs.get("pocket_color", "#ffd60a")))
                    for pr in pocket_res:
                        render_color_map[(str(pr[0]), int(pr[1]))] = pocket_hex

                render_color_map = _boost_surface_contrast(
                    display_table,
                    render_color_map,
                    palette_name=surface_accent_palette_name,
                )
            else:
                render_color_map = {}
        else:
            render_surface_colorize = bool(inputs.get("surface_colorize", False))
            render_color_map = res_color_map

        annotated_df = annotate_residue_table(
            display_table,
            str(inputs["color_mode"]),
            palette_name=surface_palette_name,
            mono_color=str(inputs["mono_color"]),
            hotspot_rank_map=hotspot_rank_map,
            pocket_residues=pocket_res,
            theme_name=classification_theme,
        )

        legend_items, legend_note = build_mode_legend(
            str(inputs["color_mode"]),
            annotated_df,
            palette_name=surface_palette_name,
            hotspot_rank_map=hotspot_rank_map,
            theme_name=classification_theme,
        )

        selectable = ["None"]
        if hotspots:
            selectable += [f"{h.chain}:{h.resid} {h.resname} ({h.energy:.2f})" for h in hotspots]
        if energy_df is not None and not energy_df.empty:
            selectable += [
                f"{row.chain}:{int(row.resid)} {row.resname} ({row.energy:.2f})"
                for row in energy_df.itertuples()
            ]
        selectable = list(dict.fromkeys(selectable))
        selected = st.selectbox("选择残基高亮（或选择空）", selectable)

        highlight = None
        if st.session_state.get("highlight"):
            highlight = st.session_state["highlight"]
        elif selected and selected != "None":
            chain_res = selected.split()[0]
            chain, resid = chain_res.split(":")
            highlight = (chain, int(resid))

        st.info("当前使用 3Dmol 交互渲染，不依赖 PyMOL 本地环境。")

        if surface_single_color_mode:
            st.caption("表面模式已固定为单色渲染，不再按残基着色。")
        elif display_mode == "surface":
            st.caption("表面模式已启用分类着色（实验），直接使用当前残基配色。")
        elif fallback_applied:
            st.caption(f"能量数据覆盖率 {coverage:.0%}，表面模式已对缺失残基启用分段背景色，避免出现大片发白。")

        render_pdb(
            pdb_str,
            style=display_mode,
            cartoon_theme=str(inputs["cartoon_theme"]),
            residue_colors=render_color_map,
            highlight=highlight,
            height=640,
            surface_opacity=float(inputs.get("surface_opacity", 0.8)),
            surface_color=resolve_color(str(inputs["surface_color"])),
            surface_colorize=render_surface_colorize,
            palette_name=str(inputs["palette_name"]),
        )

        if energy_df is not None and not energy_df.empty:
            st.subheader("残基能量分布")
            plot_df = energy_df.copy()
            plot_df["label"] = plot_df["chain"].astype(str) + ":" + plot_df["resid"].astype(str)
            fig = px.bar(
                plot_df,
                x="label",
                y="energy",
                color="energy",
                color_continuous_scale="RdBu_r",
                hover_data=["resname"],
            )
            fig.update_layout(xaxis_title="残基", yaxis_title="能量")
            st.plotly_chart(fig, use_container_width=True)

        if multi_compare_df is not None and not multi_compare_df.empty:
            st.subheader("多构象热点一致性")
            st.dataframe(multi_compare_df, use_container_width=True, hide_index=True)
            csvc = multi_compare_df.to_csv(index=False).encode("utf-8")
            st.download_button("下载一致性结果 CSV", data=csvc, file_name="consistency.csv", mime="text/csv")

    with right:
        st.subheader("热点 & 统计")
        summary = build_energy_summary(display_table)
        st.caption(f"能量来源：{energy_source_label}")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("平均能量", format_energy(summary["mean_energy"]))
        metric_col2.metric("有效能量数", f"{summary['valid_energy_count']}/{summary['residue_count']}")
        st.metric("最低/最高", f"{format_energy(summary['min_energy'])} / {format_energy(summary['max_energy'])}")
        if summary["energy_coverage"] < 1.0:
            st.caption(f"当前能量覆盖率：{summary['energy_coverage']:.0%}（其余残基使用结构/分类背景色）")

    st.markdown("---")
    with st.expander("更多详细结果", expanded=False):
        detail_tabs = st.tabs(["概览", "图例", "注释", "热点", "口袋"])

        with detail_tabs[0]:
            st.markdown("### 自动解释")
            st.info(explain_results(hotspots, multi_compare_df))

        with detail_tabs[1]:
            if legend_note:
                st.caption(legend_note)
            if legend_items:
                for item in legend_items:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
                        f"<span style='width:12px;height:12px;border-radius:3px;background:{resolve_color(str(item['color']))};display:inline-block;border:1px solid rgba(0,0,0,0.15);'></span>"
                        f"<span>{item['label']}</span>"
                        f"<span style='opacity:0.6;'>({item['count']})</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if item.get("description"):
                        st.caption(str(item["description"]))
            else:
                st.info("当前没有可显示的图例。")

        with detail_tabs[2]:
            if annotated_df.empty:
                st.info("当前没有可显示的残基注释。")
            else:
                filter_cols = st.columns(3)
                only_hotspots = filter_cols[0].checkbox("仅热点", value=False, key=f"anno_hotspots_{pdb_index}")
                only_pocket = filter_cols[1].checkbox("仅口袋", value=False, key=f"anno_pocket_{pdb_index}")
                sort_by = filter_cols[2].selectbox(
                    "排序",
                    ["hotspot_rank", "energy", "resid", "classification_label"],
                    index=0,
                    key=f"anno_sort_{pdb_index}",
                )

                table_view = annotated_df.copy()
                if only_hotspots:
                    table_view = table_view[table_view["is_hotspot"]]
                if only_pocket:
                    table_view = table_view[table_view["is_pocket"]]

                ascending = sort_by not in {"energy", "hotspot_rank"}
                table_view = table_view.sort_values(by=sort_by, ascending=ascending, na_position="last")

                display_columns = [
                    "residue_label",
                    "classification_label",
                    "classification_color",
                    "energy",
                    "hotspot_rank",
                    "is_hotspot",
                    "is_pocket",
                ]
                display_columns = [col for col in display_columns if col in table_view.columns]

                if table_view.empty:
                    st.info("当前筛选条件下没有可显示的残基。")
                else:
                    st.dataframe(table_view[display_columns], use_container_width=True, hide_index=True)
                    st.download_button(
                        "下载注释 CSV",
                        data=table_view.to_csv(index=False).encode("utf-8"),
                        file_name="residue_annotations.csv",
                        mime="text/csv",
                    )

        with detail_tabs[3]:
            if hotspots:
                st.markdown("**热点列表（点击“高亮”按钮可在 3D 视图定位）**")
                df_hot = pd.DataFrame(
                    [{"chain": h.chain, "resid": h.resid, "resname": h.resname, "energy": h.energy} for h in hotspots]
                )
                for idx, hotspot in enumerate(hotspots):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    c1.write(f"{hotspot.chain}{hotspot.resid} {hotspot.resname}")
                    c2.write(f"{hotspot.energy:.2f}")
                    if c3.button("高亮", key=f"hot_btn_{idx}"):
                        st.session_state["highlight"] = (hotspot.chain, hotspot.resid)

                if st.button("清除高亮", key="clear_highlight"):
                    st.session_state["highlight"] = None

                st.download_button(
                    "下载热点 CSV",
                    data=df_hot.to_csv(index=False).encode("utf-8"),
                    file_name="hotspots.csv",
                    mime="text/csv",
                )
            else:
                st.info("没有可用的能量数据来识别热点。")

        with detail_tabs[4]:
            if pocket:
                st.write(pocket)
                if pocket_res:
                    st.write("口袋残基举例：", pocket_res[:10])
            else:
                st.info("未检测到上传口袋数据。")

    st.markdown("---")
    st.markdown("演示注：表面模式固定为单色渲染；如果需要按残基颜色观察，请切换到卡通或棒状模式。")


if __name__ == "__main__":
    main()