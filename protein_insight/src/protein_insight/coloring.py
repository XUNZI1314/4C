"""Residue classification and legend helpers for protein visualization."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from protein_insight.analysis import energy_color_map
from protein_insight.pymol_colors import cycle_palette, resolve_color


PROPERTY_SCHEME = [
    {
        "label": "酸性（Asp/Glu）",
        "residues": {"ASP", "GLU"},
        "color": "tv_red",
        "description": "通常带负电，常参与盐桥和氢键网络。",
    },
    {
        "label": "碱性（Lys/Arg）",
        "residues": {"LYS", "ARG"},
        "color": "tv_blue",
        "description": "通常带正电，常参与盐桥和界面锚定。",
    },
    {
        "label": "弱碱（His）",
        "residues": {"HIS"},
        "color": "purple",
        "description": "pH 相关，可质子化，常见于催化位点。",
    },
    {
        "label": "极性不带电（Ser/Thr/Asn/Gln/Cys）",
        "residues": {"SER", "THR", "ASN", "GLN", "CYS"},
        "color": "tv_cyan",
        "description": "偏向氢键和表面相互作用。",
    },
    {
        "label": "芳香族（Phe/Tyr/Trp）",
        "residues": {"PHE", "TYR", "TRP"},
        "color": "tv_orange",
        "description": "富含 π 体系，常参与疏水堆叠。",
    },
    {
        "label": "疏水脂肪族（Ala/Val/Leu/Ile/Met）",
        "residues": {"ALA", "VAL", "LEU", "ILE", "MET"},
        "color": "tv_green",
        "description": "常位于蛋白核心或疏水界面。",
    },
    {
        "label": "特殊/柔性（Gly/Pro）",
        "residues": {"GLY", "PRO"},
        "color": "lightgray",
        "description": "影响局部柔性，常出现在转角或折返处。",
    },
    {
        "label": "其他/未知",
        "residues": set(),
        "color": "gray",
        "description": "非标准或未识别的残基。",
    },
]


CHARGE_SCHEME = [
    {
        "label": "负电（Asp/Glu）",
        "residues": {"ASP", "GLU"},
        "color": "tv_red",
        "description": "侧链通常带负电。",
    },
    {
        "label": "正电（Lys/Arg）",
        "residues": {"LYS", "ARG"},
        "color": "tv_blue",
        "description": "侧链通常带正电。",
    },
    {
        "label": "可质子化（His）",
        "residues": {"HIS"},
        "color": "purple",
        "description": "电荷状态受 pH 影响较明显。",
    },
    {
        "label": "中性/其他",
        "residues": set(),
        "color": "lightgray",
        "description": "未归入上述电荷分类的残基。",
    },
]


POLARITY_SCHEME = [
    {
        "label": "疏水",
        "residues": {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"},
        "color": "tv_green",
        "description": "偏向蛋白核心和疏水接触。",
    },
    {
        "label": "极性不带电",
        "residues": {"SER", "THR", "ASN", "GLN", "CYS", "TYR"},
        "color": "tv_cyan",
        "description": "常参与氢键和界面识别。",
    },
    {
        "label": "带电",
        "residues": {"ASP", "GLU", "LYS", "ARG", "HIS"},
        "color": "tv_orange",
        "description": "带电或可质子化侧链。",
    },
    {
        "label": "特殊/柔性",
        "residues": {"GLY"},
        "color": "lightgray",
        "description": "柔性最高的常见残基。",
    },
    {
        "label": "其他/未知",
        "residues": set(),
        "color": "gray",
        "description": "非标准或未识别的残基。",
    },
]


ENERGY_BIN_SCHEME = [
    {
        "label": "强有利（≤ -5）",
        "min": float("-inf"),
        "max": -5.0,
        "color": "tv_blue",
        "description": "显著有利于结合。",
    },
    {
        "label": "有利（-5 ~ -1）",
        "min": -5.0,
        "max": -1.0,
        "color": "sky",
        "description": "偏有利的结合贡献。",
    },
    {
        "label": "近中性（-1 ~ 1）",
        "min": -1.0,
        "max": 1.0,
        "color": "lightgray",
        "description": "能量接近中性。",
    },
    {
        "label": "不利（1 ~ 5）",
        "min": 1.0,
        "max": 5.0,
        "color": "tv_orange",
        "description": "对结合略有不利。",
    },
    {
        "label": "强不利（> 5）",
        "min": 5.0,
        "max": float("inf"),
        "color": "tv_red",
        "description": "明显不利于结合。",
    },
]


HOTSPOT_SCHEME = [
    {
        "label": "核心热点（Rank 1-2）",
        "min_rank": 1,
        "max_rank": 2,
        "color": "tv_red",
        "description": "当前集合中最强的结合热点。",
    },
    {
        "label": "重要热点（Rank 3-5）",
        "min_rank": 3,
        "max_rank": 5,
        "color": "tv_orange",
        "description": "次一级但仍然值得重点关注。",
    },
    {
        "label": "次级热点（Rank > 5）",
        "min_rank": 6,
        "max_rank": None,
        "color": "tv_yellow",
        "description": "进入热点集合，但优先级较低。",
    },
    {
        "label": "非热点",
        "min_rank": None,
        "max_rank": None,
        "color": "lightgray",
        "description": "未进入当前热点集合。",
    },
]


POCKET_RECOGNITION_SCHEME = [
    {
        "label": "口袋+热点重叠",
        "is_pocket": True,
        "is_hotspot": True,
        "color": "#e85d75",
        "description": "同时属于口袋和热点，优先级最高。",
    },
    {
        "label": "口袋残基",
        "is_pocket": True,
        "is_hotspot": False,
        "color": "#2563eb",
        "description": "来自口袋识别或口袋文件的候选残基。",
    },
    {
        "label": "热点残基",
        "is_pocket": False,
        "is_hotspot": True,
        "color": "#f59e0b",
        "description": "满足热点阈值但未被标记为口袋。",
    },
    {
        "label": "背景残基",
        "is_pocket": False,
        "is_hotspot": False,
        "color": "#d8dde6",
        "description": "未归入口袋和热点集合。",
    },
]


DEFAULT_CLASSIFICATION_THEME = "清新海洋"
CLASSIFICATION_THEME_OPTIONS = ["清新海洋", "专业高对比", "柔和莫兰迪"]
CLASSIFICATION_THEME_DESCRIPTIONS = {
    "清新海洋": "冷色海蓝为主，整体更清爽，适合想要安静但有层次的表面。",
    "专业高对比": "高饱和原色对撞，类别边界最明显，适合演示和快速对照。",
    "柔和莫兰迪": "低饱和雾面感更强，整体更克制，适合论文风格和长时间查看。",
}
CLASSIFICATION_THEME_COLORS = {
    "清新海洋": {
        "property": ["#e76f51", "#2563eb", "#7c3aed", "#06b6d4", "#f59e0b", "#10b981", "#64748b", "#dbeafe"],
        "charge": ["#e76f51", "#2563eb", "#7c3aed", "#dbeafe"],
        "polarity": ["#10b981", "#06b6d4", "#2563eb", "#64748b", "#dbeafe"],
        "energy": ["#1d4ed8", "#38bdf8", "#e2e8f0", "#f59e0b", "#e76f51"],
        "hotspot": ["#e76f51", "#f59e0b", "#fde047", "#dbeafe"],
        "pocket": ["#e76f51", "#2563eb", "#f59e0b", "#dbeafe"],
    },
    "专业高对比": {
        "property": ["#e11d48", "#1d4ed8", "#f97316", "#06b6d4", "#a855f7", "#16a34a", "#f59e0b", "#1f2937"],
        "charge": ["#e11d48", "#1d4ed8", "#a855f7", "#f8fafc"],
        "polarity": ["#16a34a", "#06b6d4", "#f97316", "#1f2937", "#f8fafc"],
        "energy": ["#1d4ed8", "#0ea5e9", "#f8fafc", "#f97316", "#e11d48"],
        "hotspot": ["#e11d48", "#f97316", "#facc15", "#1f2937"],
        "pocket": ["#e11d48", "#1d4ed8", "#f97316", "#f8fafc"],
    },
    "柔和莫兰迪": {
        "property": ["#c47e67", "#6f8ea7", "#8ea37f", "#b4b0a1", "#c6a26f", "#8797a6", "#d2d7de", "#e6e0d6"],
        "charge": ["#c47e67", "#6f8ea7", "#b4b0a1", "#e6e0d6"],
        "polarity": ["#8ea37f", "#6f8ea7", "#c6a26f", "#aab4bf", "#e6e0d6"],
        "energy": ["#5f7fb0", "#8fb3da", "#e6e0d6", "#c68863", "#c47e67"],
        "hotspot": ["#c47e67", "#c68863", "#d8b26d", "#e6e0d6"],
        "pocket": ["#c47e67", "#6f8ea7", "#c68863", "#e6e0d6"],
    },
}


def _normalize_classification_theme(theme_name: Optional[str]) -> str:
    theme = str(theme_name or DEFAULT_CLASSIFICATION_THEME).strip()
    return theme if theme in CLASSIFICATION_THEME_COLORS else DEFAULT_CLASSIFICATION_THEME


def _apply_theme_colors(base_scheme: Sequence[Dict[str, object]], colors: Sequence[str]) -> List[Dict[str, object]]:
    palette = [str(color).strip() for color in colors if str(color).strip()]
    if not palette:
        palette = ["#d8dde6"]

    themed: List[Dict[str, object]] = []
    for index, item in enumerate(base_scheme):
        themed_item = dict(item)
        themed_item["color"] = palette[index % len(palette)]
        themed.append(themed_item)
    return themed


PROPERTY_SCHEME_BASE = [dict(item) for item in PROPERTY_SCHEME]
CHARGE_SCHEME_BASE = [dict(item) for item in CHARGE_SCHEME]
POLARITY_SCHEME_BASE = [dict(item) for item in POLARITY_SCHEME]
ENERGY_BIN_SCHEME_BASE = [dict(item) for item in ENERGY_BIN_SCHEME]
HOTSPOT_SCHEME_BASE = [dict(item) for item in HOTSPOT_SCHEME]
POCKET_RECOGNITION_SCHEME_BASE = [dict(item) for item in POCKET_RECOGNITION_SCHEME]


def _themed_scheme(base_scheme: Sequence[Dict[str, object]], theme_name: Optional[str], key: str) -> List[Dict[str, object]]:
    theme = CLASSIFICATION_THEME_COLORS[_normalize_classification_theme(theme_name)]
    return _apply_theme_colors(base_scheme, theme[key])


PROPERTY_SCHEME = _themed_scheme(PROPERTY_SCHEME_BASE, DEFAULT_CLASSIFICATION_THEME, "property")
CHARGE_SCHEME = _themed_scheme(CHARGE_SCHEME_BASE, DEFAULT_CLASSIFICATION_THEME, "charge")
POLARITY_SCHEME = _themed_scheme(POLARITY_SCHEME_BASE, DEFAULT_CLASSIFICATION_THEME, "polarity")
ENERGY_BIN_SCHEME = _themed_scheme(ENERGY_BIN_SCHEME_BASE, DEFAULT_CLASSIFICATION_THEME, "energy")
HOTSPOT_SCHEME = _themed_scheme(HOTSPOT_SCHEME_BASE, DEFAULT_CLASSIFICATION_THEME, "hotspot")
POCKET_RECOGNITION_SCHEME = _themed_scheme(POCKET_RECOGNITION_SCHEME_BASE, DEFAULT_CLASSIFICATION_THEME, "pocket")


def normalize_resname(resname: str) -> str:
    return (resname or "").strip().upper()


def normalize_chain(chain: str) -> str:
    value = (chain or "").strip()
    return value if value else "A"


def build_chain_color_map(chains: Sequence[str], palette_name: str) -> Dict[str, str]:
    normalized = [normalize_chain(chain) for chain in chains if normalize_chain(chain)]
    unique_chains: List[str] = []
    for chain in normalized:
        if chain not in unique_chains:
            unique_chains.append(chain)

    if not unique_chains:
        return {}

    colors = cycle_palette(palette_name, len(unique_chains))
    return dict(zip(unique_chains, colors))


def _scheme_for_mode(mode: str, theme_name: Optional[str]) -> Sequence[Dict[str, object]]:
    if mode == "按氨基酸理化性质":
        return _themed_scheme(PROPERTY_SCHEME_BASE, theme_name, "property")
    if mode == "按电荷状态":
        return _themed_scheme(CHARGE_SCHEME_BASE, theme_name, "charge")
    if mode == "按侧链极性":
        return _themed_scheme(POLARITY_SCHEME_BASE, theme_name, "polarity")
    if mode == "按MMPBSA等级":
        return _themed_scheme(ENERGY_BIN_SCHEME_BASE, theme_name, "energy")
    if mode == "按热点等级":
        return _themed_scheme(HOTSPOT_SCHEME_BASE, theme_name, "hotspot")
    if mode == "按口袋识别":
        return _themed_scheme(POCKET_RECOGNITION_SCHEME_BASE, theme_name, "pocket")
    return []


def build_surface_block_color_map(
    table: pd.DataFrame,
    palette_name: str = "PyMOL 经典",
    *,
    block_size: Optional[int] = None,
) -> Dict[Tuple[str, int], str]:
    if table is None or table.empty:
        return {}

    ordered = table.sort_values(by=["chain", "resid"]).reset_index(drop=True)
    colors = cycle_palette(palette_name, max(24, min(160, max(1, len(ordered) // 2 + 16))))
    if not colors:
        colors = [resolve_color("lightgray")]

    result: Dict[Tuple[str, int], str] = {}
    for chain, group in ordered.groupby("chain", sort=False):
        chain_key = normalize_chain(str(chain))
        residues = [int(value) for value in pd.to_numeric(group["resid"], errors="coerce").dropna().astype(int).tolist()]
        if not residues:
            continue

        local_block_size = int(block_size) if block_size else max(1, min(6, max(1, len(residues) // 24)))
        local_block_size = max(1, local_block_size)
        chain_shift = sum(ord(ch) for ch in chain_key) % len(colors)
        for index, resid in enumerate(residues):
            color = colors[(chain_shift + (index // local_block_size) * 3 + (index % 5) + int(resid) * 2) % len(colors)]
            result[(chain_key, int(resid))] = color

    return result


def _match_scheme(resname: str, scheme: Sequence[Dict[str, object]]) -> Tuple[str, str, str]:
    normalized = normalize_resname(resname)
    for item in scheme:
        residues = item.get("residues", set())
        if normalized in residues:
            return str(item["label"]), str(item["color"]), str(item["description"])

    fallback = scheme[-1]
    return str(fallback["label"]), str(fallback["color"]), str(fallback["description"])


def classify_property(resname: str, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    return _match_scheme(resname, _scheme_for_mode("按氨基酸理化性质", theme_name))


def classify_charge(resname: str, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    return _match_scheme(resname, _scheme_for_mode("按电荷状态", theme_name))


def classify_polarity(resname: str, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    return _match_scheme(resname, _scheme_for_mode("按侧链极性", theme_name))


def classify_energy_bin(energy: float, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    scheme = list(_scheme_for_mode("按MMPBSA等级", theme_name))
    if energy is None or energy != energy:
        fallback = scheme[2]
        return str(fallback["label"]), str(fallback["color"]), str(fallback["description"])

    for item in scheme:
        if float(item["min"]) < float(energy) <= float(item["max"]):
            return str(item["label"]), str(item["color"]), str(item["description"])

    fallback = scheme[-1]
    return str(fallback["label"]), str(fallback["color"]), str(fallback["description"])


def classify_hotspot(rank: Optional[int], theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    scheme = list(_scheme_for_mode("按热点等级", theme_name))
    if rank is None:
        fallback = scheme[-1]
        return str(fallback["label"]), str(fallback["color"]), str(fallback["description"])

    for item in scheme[:-1]:
        min_rank = item["min_rank"]
        max_rank = item["max_rank"]
        if min_rank is not None and rank >= int(min_rank) and (max_rank is None or rank <= int(max_rank)):
            return str(item["label"]), str(item["color"]), str(item["description"])

    fallback = scheme[-1]
    return str(fallback["label"]), str(fallback["color"]), str(fallback["description"])


def classify_pocket_recognition(is_pocket: bool, is_hotspot: bool, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    pocket_flag = bool(is_pocket)
    hotspot_flag = bool(is_hotspot)
    scheme = list(_scheme_for_mode("按口袋识别", theme_name))
    for item in scheme:
        if bool(item["is_pocket"]) == pocket_flag and bool(item["is_hotspot"]) == hotspot_flag:
            return str(item["label"]), str(item["color"]), str(item["description"])

    fallback = scheme[-1]
    return str(fallback["label"]), str(fallback["color"]), str(fallback["description"])


def resolve_residue_annotation(
    mode: str,
    *,
    resname: str,
    energy: Optional[float] = None,
    chain: str = "A",
    chain_color_map: Optional[Dict[str, str]] = None,
    hotspot_rank: Optional[int] = None,
    is_pocket: bool = False,
    energy_bounds: Optional[Tuple[float, float]] = None,
    mono_color: Optional[str] = None,
    theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME,
) -> Tuple[str, str, str]:
    if mode == "单色":
        color = resolve_color(mono_color or "lightgray")
        return "统一颜色", color, "所有残基使用同一颜色。"

    if mode in {"按链", "PyMOL 调色板"}:
        chain_key = normalize_chain(chain)
        color = resolve_color("lightgray")
        if chain_color_map:
            color = chain_color_map.get(chain_key, next(iter(chain_color_map.values())))
        return f"链 {chain_key}", color, "按链区分，适合多构象或多链复合物。"

    if mode == "按能量连续梯度":
        if energy is None or energy != energy or not energy_bounds:
            return "连续能量梯度", resolve_color("lightgray"), "能量缺失或范围不足，无法生成连续梯度。"
        vmin, vmax = energy_bounds
        return "连续能量梯度", energy_color_map(float(energy), float(vmin), float(vmax)), f"连续渐变：当前范围 {float(vmin):.2f} ~ {float(vmax):.2f}。"

    if mode == "按MMPBSA等级":
        return classify_energy_bin(energy, theme_name=theme_name)

    if mode == "按热点等级":
        return classify_hotspot(hotspot_rank, theme_name=theme_name)

    if mode == "按口袋识别":
        return classify_pocket_recognition(is_pocket=is_pocket, is_hotspot=hotspot_rank is not None, theme_name=theme_name)

    if mode == "按电荷状态":
        return classify_charge(resname, theme_name=theme_name)

    if mode == "按侧链极性":
        return classify_polarity(resname, theme_name=theme_name)

    if mode == "按氨基酸理化性质":
        return classify_property(resname, theme_name=theme_name)

    return "其他/未知", resolve_color("gray"), "未识别的分类模式。"


def resolve_residue_color(
    mode: str,
    *,
    resname: str,
    energy: Optional[float] = None,
    chain: str = "A",
    chain_color_map: Optional[Dict[str, str]] = None,
    hotspot_rank: Optional[int] = None,
    energy_bounds: Optional[Tuple[float, float]] = None,
    mono_color: Optional[str] = None,
    is_pocket: bool = False,
    theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME,
) -> str:
    _, color, _ = resolve_residue_annotation(
        mode,
        resname=resname,
        energy=energy,
        chain=chain,
        chain_color_map=chain_color_map,
        hotspot_rank=hotspot_rank,
        is_pocket=is_pocket,
        energy_bounds=energy_bounds,
        mono_color=mono_color,
        theme_name=theme_name,
    )
    return resolve_color(color)


def annotate_residue_table(
    table: pd.DataFrame,
    mode: str,
    *,
    palette_name: str = "PyMOL 经典",
    mono_color: Optional[str] = None,
    hotspot_rank_map: Optional[Dict[Tuple[str, int], int]] = None,
    pocket_residues: Optional[Sequence[Tuple[str, int]]] = None,
    theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME,
) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(
            columns=[
                "chain",
                "resid",
                "resname",
                "energy",
                "residue_label",
                "classification_label",
                "classification_color",
                "classification_description",
                "hotspot_rank",
                "is_hotspot",
                "is_pocket",
            ]
        )

    ordered = table.sort_values(by=["chain", "resid"]).reset_index(drop=True)
    chains: List[str] = []
    for _, r in ordered.iterrows():
        chain = normalize_chain(str(r.get("chain", "A")))
        if chain not in chains:
            chains.append(chain)

    chain_color_map = build_chain_color_map(chains, palette_name)
    energy_bounds = (float(ordered["energy"].min()), float(ordered["energy"].max()))
    pocket_set = {
        (normalize_chain(chain), int(resid))
        for chain, resid in (pocket_residues or [])
    }

    rows: List[Dict[str, object]] = []
    for _, r in ordered.iterrows():
        chain = normalize_chain(str(r.get("chain", "A")))
        resid = int(r["resid"])
        resname = str(r.get("resname", ""))
        energy = float(r["energy"]) if pd.notna(r["energy"]) else None
        hotspot_rank = hotspot_rank_map.get((chain, resid)) if hotspot_rank_map else None
        label, color, description = resolve_residue_annotation(
            mode,
            resname=resname,
            energy=energy,
            chain=chain,
            chain_color_map=chain_color_map,
            hotspot_rank=hotspot_rank,
            is_pocket=(chain, resid) in pocket_set,
            energy_bounds=energy_bounds,
            mono_color=mono_color,
            theme_name=theme_name,
        )
        rows.append(
            {
                "chain": chain,
                "resid": resid,
                "resname": resname,
                "energy": energy,
                "residue_label": f"{resname} {chain}{resid}".strip(),
                "classification_label": label,
                "classification_color": resolve_color(color),
                "classification_description": description,
                "hotspot_rank": hotspot_rank,
                "is_hotspot": hotspot_rank is not None,
                "is_pocket": (chain, resid) in pocket_set,
            }
        )

    return pd.DataFrame(rows)


def build_mode_legend(
    mode: str,
    table,
    *,
    palette_name: str = "PyMOL 经典",
    hotspot_rank_map: Optional[Dict[Tuple[str, int], int]] = None,
    theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME,
) -> Tuple[List[Dict[str, object]], str]:
    if table is None or getattr(table, "empty", True):
        return [], ""

    rows = list(table.itertuples(index=False))
    note = ""
    items: List[Dict[str, object]] = []

    if mode == "按能量连续梯度":
        energies = [float(getattr(row, "energy", 0.0)) for row in rows]
        if energies:
            note = f"连续渐变：当前范围 {min(energies):.2f} ~ {max(energies):.2f}，负值偏蓝、正值偏红。"
        return [
            {
                "label": "连续能量梯度",
                "color": "#5b8cff",
                "count": len(rows),
                "description": "低能量偏蓝，高能量偏红。",
            }
        ], note

    if mode in {"按氨基酸理化性质", "按电荷状态", "按侧链极性"}:
        scheme = list(_scheme_for_mode(mode, theme_name))
        counter: Counter[str] = Counter()
        for row in rows:
            label, color, description = _match_scheme(getattr(row, "resname", ""), scheme)
            counter[label] += 1

        for item in scheme:
            label = str(item["label"])
            if counter[label] > 0:
                color = str(item["color"])
                description = str(item["description"])
                items.append({"label": label, "color": color, "count": counter[label], "description": description})
        note = "基于标准氨基酸理化分组，不是随机配色。"
        return items, note

    if mode == "按MMPBSA等级":
        counter: Counter[str] = Counter()
        for row in rows:
            label, color, description = classify_energy_bin(float(getattr(row, "energy", 0.0)), theme_name=theme_name)
            counter[label] += 1

        for item in _scheme_for_mode("按MMPBSA等级", theme_name):
            label = str(item["label"])
            if counter[label] > 0:
                items.append(
                    {
                        "label": label,
                        "color": str(item["color"]),
                        "count": counter[label],
                        "description": str(item["description"]),
                    }
                )
        note = "MMPBSA 按固定能量区间分级，便于快速比较不同残基的贡献。"
        return items, note

    if mode == "按热点等级":
        counter: Counter[str] = Counter()
        for row in rows:
            chain = normalize_chain(getattr(row, "chain", "A"))
            resid = int(getattr(row, "resid", 0))
            rank = hotspot_rank_map.get((chain, resid)) if hotspot_rank_map else None
            label, color, description = classify_hotspot(rank, theme_name=theme_name)
            counter[label] += 1

        for item in _scheme_for_mode("按热点等级", theme_name):
            label = str(item["label"])
            if counter[label] > 0:
                items.append(
                    {
                        "label": label,
                        "color": str(item["color"]),
                        "count": counter[label],
                        "description": str(item["description"]),
                    }
                )
        note = "热点等级按当前热点列表排序，核心热点位于前几名。"
        return items, note

    if mode == "按口袋识别":
        counter: Counter[str] = Counter()
        for row in rows:
            chain = normalize_chain(getattr(row, "chain", "A"))
            resid = int(getattr(row, "resid", 0))
            rank = hotspot_rank_map.get((chain, resid)) if hotspot_rank_map else getattr(row, "hotspot_rank", None)
            is_hotspot = (rank is not None) or _as_bool(getattr(row, "is_hotspot", False))
            is_pocket = _as_bool(getattr(row, "is_pocket", False))
            label, _, _ = classify_pocket_recognition(is_pocket=is_pocket, is_hotspot=is_hotspot, theme_name=theme_name)
            counter[label] += 1

        for item in _scheme_for_mode("按口袋识别", theme_name):
            label = str(item["label"])
            if counter[label] > 0:
                items.append(
                    {
                        "label": label,
                        "color": _resolve_hex(str(item["color"])),
                        "count": counter[label],
                        "description": str(item["description"]),
                    }
                )
        note = "口袋识别模式优先显示口袋/热点交集，便于快速定位关键功能残基。"
        return items, note

    if mode in {"按链", "PyMOL 调色板"}:
        chains: List[str] = []
        for row in rows:
            chain = normalize_chain(getattr(row, "chain", "A"))
            if chain not in chains:
                chains.append(chain)

        chain_color_map = build_chain_color_map(chains, palette_name)
        counts: Counter[str] = Counter(normalize_chain(getattr(row, "chain", "A")) for row in rows)
        for chain in chains:
            color = chain_color_map.get(chain, resolve_color("lightgray"))
            items.append(
                {
                    "label": f"链 {chain}",
                    "color": color,
                    "count": counts[chain],
                    "description": "按链区分，适合多构象或多链复合物。",
                }
            )
        note = "按 chain 字段进行分组；多链时会自动生成不同颜色。"
        return items, note

    if mode == "单色":
        return [
            {
                "label": "统一颜色",
                "color": resolve_color("lightgray"),
                "count": len(rows),
                "description": "所有残基使用同一颜色。",
            }
        ], ""

    return [], ""
