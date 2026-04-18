from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from protein_visualizer.services.energy import energy_to_hex_color, normalize_energy


NAMED_COLORS = {
    "tv_red": "#ef4444",
    "tv_blue": "#2563eb",
    "tv_cyan": "#06b6d4",
    "tv_orange": "#f97316",
    "tv_green": "#10b981",
    "tv_yellow": "#f59e0b",
    "lightgray": "#d1d5db",
    "gray": "#6b7280",
    "purple": "#8b5cf6",
    "sky": "#38bdf8",
}

CHAIN_PALETTES = {
    "经典": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    "柔和": ["#6baed6", "#74c476", "#fd8d3c", "#9e9ac8", "#fdae6b", "#fdd0a2", "#bcbddc", "#969696"],
    "高对比": ["#1d4ed8", "#ea580c", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#e11d48", "#4b5563"],
}

RESNAME_EXACT_PALETTE = [
    "#2563eb",
    "#ef4444",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#06b6d4",
    "#f97316",
    "#84cc16",
    "#ec4899",
    "#14b8a6",
    "#3b82f6",
    "#eab308",
    "#22c55e",
    "#a855f7",
    "#0ea5e9",
    "#f43f5e",
    "#65a30d",
    "#c026d3",
    "#0891b2",
    "#d97706",
]

PROPERTY_SCHEME = [
    {"label": "酸性（Asp/Glu）", "residues": {"ASP", "GLU"}, "color": "tv_red", "description": "通常带负电，常参与盐桥和氢键网络。"},
    {"label": "碱性（Lys/Arg）", "residues": {"LYS", "ARG"}, "color": "tv_blue", "description": "通常带正电，常参与盐桥和界面锚定。"},
    {"label": "弱碱（His）", "residues": {"HIS"}, "color": "purple", "description": "pH 相关，可质子化，常见于催化位点。"},
    {"label": "极性不带电（Ser/Thr/Asn/Gln/Cys）", "residues": {"SER", "THR", "ASN", "GLN", "CYS"}, "color": "tv_cyan", "description": "偏向氢键和表面相互作用。"},
    {"label": "芳香族（Phe/Tyr/Trp）", "residues": {"PHE", "TYR", "TRP"}, "color": "tv_orange", "description": "富含 π 体系，常参与疏水堆叠。"},
    {"label": "疏水脂肪族（Ala/Val/Leu/Ile/Met）", "residues": {"ALA", "VAL", "LEU", "ILE", "MET"}, "color": "tv_green", "description": "常位于蛋白核心或疏水界面。"},
    {"label": "特殊/柔性（Gly/Pro）", "residues": {"GLY", "PRO"}, "color": "lightgray", "description": "影响局部柔性，常出现在转角或折返处。"},
    {"label": "其他/未知", "residues": set(), "color": "gray", "description": "非标准或未识别的残基。"},
]

CHARGE_SCHEME = [
    {"label": "负电（Asp/Glu）", "residues": {"ASP", "GLU"}, "color": "tv_red", "description": "侧链通常带负电。"},
    {"label": "正电（Lys/Arg）", "residues": {"LYS", "ARG"}, "color": "tv_blue", "description": "侧链通常带正电。"},
    {"label": "可质子化（His）", "residues": {"HIS"}, "color": "purple", "description": "电荷状态受 pH 影响较明显。"},
    {"label": "中性/其他", "residues": set(), "color": "lightgray", "description": "未归入上述电荷分类的残基。"},
]

POLARITY_SCHEME = [
    {"label": "疏水", "residues": {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}, "color": "tv_green", "description": "偏向蛋白核心和疏水接触。"},
    {"label": "极性不带电", "residues": {"SER", "THR", "ASN", "GLN", "CYS", "TYR"}, "color": "tv_cyan", "description": "常参与氢键和界面识别。"},
    {"label": "带电", "residues": {"ASP", "GLU", "LYS", "ARG", "HIS"}, "color": "tv_orange", "description": "带电或可质子化侧链。"},
    {"label": "特殊/柔性", "residues": {"GLY"}, "color": "lightgray", "description": "柔性最高的常见残基。"},
    {"label": "其他/未知", "residues": set(), "color": "gray", "description": "非标准或未识别的残基。"},
]

ENERGY_BIN_SCHEME = [
    {"label": "强有利（≤ -5）", "min": float("-inf"), "max": -5.0, "color": "tv_blue", "description": "显著有利于结合。"},
    {"label": "有利（-5 ~ -1）", "min": -5.0, "max": -1.0, "color": "sky", "description": "偏有利的结合贡献。"},
    {"label": "近中性（-1 ~ 1）", "min": -1.0, "max": 1.0, "color": "lightgray", "description": "能量接近中性。"},
    {"label": "不利（1 ~ 5）", "min": 1.0, "max": 5.0, "color": "tv_orange", "description": "对结合略有不利。"},
    {"label": "强不利（> 5）", "min": 5.0, "max": float("inf"), "color": "tv_red", "description": "明显不利于结合。"},
]

HOTSPOT_SCHEME = [
    {"label": "核心热点（Rank 1-2）", "min_rank": 1, "max_rank": 2, "color": "tv_red", "description": "当前集合中最强的结合热点。"},
    {"label": "重要热点（Rank 3-5）", "min_rank": 3, "max_rank": 5, "color": "tv_orange", "description": "次一级但仍然值得重点关注。"},
    {"label": "次级热点（Rank > 5）", "min_rank": 6, "max_rank": None, "color": "tv_yellow", "description": "进入热点集合，但优先级较低。"},
    {"label": "非热点", "min_rank": None, "max_rank": None, "color": "lightgray", "description": "未进入当前热点集合。"},
]

POCKET_RECOGNITION_SCHEME = [
    {
        "label": "口袋+热点重叠",
        "is_pocket": True,
        "is_hotspot": True,
        "color": "tv_red",
        "description": "同时属于口袋和热点，优先级最高。",
    },
    {
        "label": "口袋残基",
        "is_pocket": True,
        "is_hotspot": False,
        "color": "tv_blue",
        "description": "来自口袋识别或口袋文件的候选残基。",
    },
    {
        "label": "热点残基",
        "is_pocket": False,
        "is_hotspot": True,
        "color": "tv_orange",
        "description": "满足热点阈值但未被标记为口袋。",
    },
    {
        "label": "背景残基",
        "is_pocket": False,
        "is_hotspot": False,
        "color": "lightgray",
        "description": "未归入口袋和热点集合。",
    },
]


DEFAULT_CLASSIFICATION_THEME = "精准分类"
CLASSIFICATION_THEME_OPTIONS = [DEFAULT_CLASSIFICATION_THEME]
CLASSIFICATION_THEME_DESCRIPTIONS = {
    DEFAULT_CLASSIFICATION_THEME: "按分类固定配色，不做主题漂移；同一分类始终同色。",
}
CLASSIFICATION_THEME_COLORS = {
    DEFAULT_CLASSIFICATION_THEME: {
        "property": ["#ef4444", "#2563eb", "#8b5cf6", "#06b6d4", "#f97316", "#10b981", "#d1d5db", "#6b7280"],
        "charge": ["#ef4444", "#2563eb", "#8b5cf6", "#d1d5db"],
        "polarity": ["#10b981", "#06b6d4", "#f97316", "#d1d5db", "#6b7280"],
        "energy": ["#2563eb", "#38bdf8", "#d1d5db", "#f97316", "#ef4444"],
        "hotspot": ["#ef4444", "#f97316", "#f59e0b", "#d1d5db"],
        "pocket": ["#ef4444", "#2563eb", "#f97316", "#d1d5db"],
    },
}


def _normalize_classification_theme(theme_name: Optional[str]) -> str:
    theme = str(theme_name or DEFAULT_CLASSIFICATION_THEME).strip()
    return theme if theme in CLASSIFICATION_THEME_COLORS else DEFAULT_CLASSIFICATION_THEME


def _apply_theme_colors(base_scheme: Sequence[Dict[str, object]], colors: Sequence[str]) -> List[Dict[str, object]]:
    palette = [str(color).strip() for color in colors if str(color).strip()]
    if not palette:
        palette = ["#d1d5db"]

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


def _coerce_color(value: object) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _as_bool(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _resolve_hex(color: Optional[str], fallback: str = "#d1d5db") -> str:
    cleaned = _coerce_color(color)
    if cleaned is None:
        return fallback
    if cleaned.startswith("#") and len(cleaned) in {4, 7}:
        return cleaned
    return NAMED_COLORS.get(cleaned, fallback)


def resolve_color(color_name_or_hex: Optional[str]) -> str:
    return _resolve_hex(color_name_or_hex, fallback="#d1d5db")


def normalize_resname(resname: str) -> str:
    return (resname or "").strip().upper()


def normalize_chain(chain: str) -> str:
    value = (chain or "").strip()
    return value if value else "A"


def build_resname_color_map(resnames: Sequence[str]) -> Dict[str, str]:
    unique_names = sorted({normalize_resname(name) for name in resnames if normalize_resname(name)})
    if not unique_names:
        return {}

    return {
        name: RESNAME_EXACT_PALETTE[index % len(RESNAME_EXACT_PALETTE)]
        for index, name in enumerate(unique_names)
    }


def classify_resname_exact(
    resname: str,
    resname_color_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    normalized = normalize_resname(resname)
    if normalized and resname_color_map and normalized in resname_color_map:
        return normalized, _resolve_hex(resname_color_map[normalized]), "按氨基酸类型精确配色，不做分组。"

    if normalized:
        fallback_color = RESNAME_EXACT_PALETTE[sum(ord(ch) for ch in normalized) % len(RESNAME_EXACT_PALETTE)]
        return normalized, _resolve_hex(fallback_color), "按氨基酸类型精确配色，不做分组。"

    return "其他/未知", _resolve_hex("gray"), "非标准或未识别的残基。"


def build_chain_color_map(chains: Sequence[str], palette_name: str) -> Dict[str, str]:
    normalized = [normalize_chain(chain) for chain in chains if normalize_chain(chain)]
    unique_chains: List[str] = []
    for chain in normalized:
        if chain not in unique_chains:
            unique_chains.append(chain)

    if not unique_chains:
        return {}

    palette = CHAIN_PALETTES.get(palette_name, CHAIN_PALETTES["经典"])
    return {chain: palette[index % len(palette)] for index, chain in enumerate(unique_chains)}


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


def _match_scheme(resname: str, scheme: Sequence[Dict[str, object]]) -> Tuple[str, str, str]:
    normalized = normalize_resname(resname)
    for item in scheme:
        residues = item.get("residues", set())
        if normalized in residues:
            return str(item["label"]), _resolve_hex(str(item["color"])), str(item["description"])

    fallback = scheme[-1]
    return str(fallback["label"]), _resolve_hex(str(fallback["color"])), str(fallback["description"])


def classify_property(resname: str, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    return _match_scheme(resname, _scheme_for_mode("按氨基酸理化性质", theme_name))


def classify_charge(resname: str, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    return _match_scheme(resname, _scheme_for_mode("按电荷状态", theme_name))


def classify_polarity(resname: str, theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
    return _match_scheme(resname, _scheme_for_mode("按侧链极性", theme_name))


def classify_energy_bin(energy: Optional[float], theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME) -> Tuple[str, str, str]:
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


def resolve_legacy_residue_annotation(
    mode: str,
    *,
    resname: str,
    energy: Optional[float] = None,
    chain: str = "A",
    chain_color_map: Optional[Dict[str, str]] = None,
    hotspot_rank: Optional[int] = None,
    is_pocket: bool = False,
    mono_color: Optional[str] = None,
    heat_color: Optional[str] = None,
    theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME,
    resname_color_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str]:
    if mode in {"按DELTA TOTAL 热度", "按能量连续梯度"}:
        color = _resolve_hex(heat_color, fallback=_resolve_hex(mono_color, fallback="#d1d5db"))
        return "DELTA TOTAL 热度", color, "按 DELTA TOTAL 连续渐变，低能量偏红、高能量偏蓝。"

    if mode == "单色":
        color = _resolve_hex(mono_color, fallback="#d1d5db")
        return "统一颜色", color, "所有残基使用同一颜色。"

    if mode == "按链":
        chain_key = normalize_chain(chain)
        default_color = _resolve_hex(mono_color, fallback="#d1d5db")
        color = chain_color_map.get(chain_key, default_color) if chain_color_map else default_color
        return f"链 {chain_key}", color, "按链区分，适合多构象或多链复合物。"

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
        return classify_resname_exact(resname, resname_color_map=resname_color_map)

    return "其他/未知", _resolve_hex("gray"), "未识别的分类模式。"


def build_legacy_annotation_table(
    table: pd.DataFrame,
    mode: str,
    *,
    palette_name: str = "经典",
    mono_color: Optional[str] = None,
    hotspot_df: Optional[pd.DataFrame] = None,
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
                "display_color",
                "hotspot_rank",
                "is_hotspot",
                "is_pocket",
            ]
        )

    ordered = table.sort_values(by=["chain", "resid"]).reset_index(drop=True).copy()

    if "norm_energy" not in ordered.columns:
        ordered["norm_energy"] = normalize_energy(ordered["delta_total"])
    if "heat_color" not in ordered.columns:
        ordered["heat_color"] = ordered["norm_energy"].map(energy_to_hex_color)

    chains: List[str] = []
    for _, row in ordered.iterrows():
        chain = normalize_chain(str(row.get("chain", "A")))
        if chain not in chains:
            chains.append(chain)

    chain_color_map = build_chain_color_map(chains, palette_name)
    resname_color_map = build_resname_color_map(ordered["resname"].tolist()) if mode == "按氨基酸理化性质" else {}
    hotspot_rank_map: Dict[Tuple[str, int], int] = {}
    if hotspot_df is not None and not hotspot_df.empty:
        if "hotspot_rank" in hotspot_df.columns:
            hotspot_rows = hotspot_df.sort_values("hotspot_rank").itertuples(index=False)
        else:
            hotspot_rows = hotspot_df.itertuples(index=False)
        for index, row in enumerate(hotspot_rows, start=1):
            chain = normalize_chain(getattr(row, "chain", "A"))
            resid = int(getattr(row, "resid", 0))
            rank = getattr(row, "hotspot_rank", index)
            try:
                hotspot_rank_map[(chain, resid)] = int(rank)
            except Exception:
                hotspot_rank_map[(chain, resid)] = index

    pocket_set = {(normalize_chain(chain), int(resid)) for chain, resid in (pocket_residues or [])}

    rows: List[Dict[str, object]] = []
    for _, row in ordered.iterrows():
        chain = normalize_chain(str(row.get("chain", "A")))
        resid = int(row["resid"])
        resname = str(row.get("resname", ""))
        raw_energy = row.get("delta_total_raw")
        energy = float(raw_energy) if pd.notna(raw_energy) else (float(row["delta_total"]) if pd.notna(row.get("delta_total")) else None)
        hotspot_rank = hotspot_rank_map.get((chain, resid))
        is_pocket = (chain, resid) in pocket_set
        heat_color = _coerce_color(row.get("heat_color"))
        label, color, description = resolve_legacy_residue_annotation(
            mode,
            resname=resname,
            energy=energy,
            chain=chain,
            chain_color_map=chain_color_map,
            hotspot_rank=hotspot_rank,
            is_pocket=is_pocket,
            mono_color=mono_color,
            heat_color=heat_color,
            theme_name=theme_name,
            resname_color_map=resname_color_map,
        )
        display_color = heat_color if mode in {"按DELTA TOTAL 热度", "按能量连续梯度"} else color

        rows.append(
            {
                **row.to_dict(),
                "chain": chain,
                "resid": resid,
                "resname": resname,
                "energy": energy,
                "residue_label": f"{resname} {chain}{resid}".strip(),
                "classification_label": label,
                "classification_color": color,
                "classification_description": description,
                "display_color": _resolve_hex(display_color, fallback=heat_color or "#d1d5db"),
                "hotspot_rank": hotspot_rank,
                "is_hotspot": hotspot_rank is not None,
                "is_pocket": is_pocket,
            }
        )

    return pd.DataFrame(rows)


def build_legacy_legend(
    mode: str,
    table: pd.DataFrame,
    *,
    palette_name: str = "经典",
    hotspot_rank_map: Optional[Dict[Tuple[str, int], int]] = None,
    theme_name: Optional[str] = DEFAULT_CLASSIFICATION_THEME,
) -> Tuple[List[Dict[str, object]], str]:
    if table is None or table.empty:
        return [], ""

    rows = list(table.itertuples(index=False))
    note = ""
    items: List[Dict[str, object]] = []

    if mode in {"按DELTA TOTAL 热度", "按能量连续梯度"}:
        energies = [float(getattr(row, "delta_total", 0.0)) for row in rows]
        if energies:
            note = f"连续渐变：当前范围 {min(energies):.2f} ~ {max(energies):.2f}，低能量偏红、高能量偏蓝。"
        return [
            {"label": "DELTA TOTAL 热度", "color": "#ef4444", "count": len(rows), "description": "按残基自由能连续渐变。"}
        ], note

    if mode == "按氨基酸理化性质":
        counter: Counter[str] = Counter()
        label_to_color: Dict[str, str] = {}
        for row in rows:
            label = str(getattr(row, "classification_label", "其他/未知"))
            color = _resolve_hex(getattr(row, "classification_color", None), fallback=_resolve_hex("gray"))
            counter[label] += 1
            label_to_color.setdefault(label, color)

        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            items.append(
                {
                    "label": label,
                    "color": label_to_color.get(label, "#6b7280"),
                    "count": int(count),
                    "description": "按氨基酸类型精确配色，不做分组。",
                }
            )
        note = "按氨基酸类型精确配色，不做分组。"
        return items, note

    if mode in {"按电荷状态", "按侧链极性"}:
        scheme = list(_scheme_for_mode(mode, theme_name))
        counter: Counter[str] = Counter()
        for row in rows:
            label, _, _ = _match_scheme(getattr(row, "resname", ""), scheme)
            counter[label] += 1

        for item in scheme:
            label = str(item["label"])
            if counter[label] > 0:
                items.append({"label": label, "color": _resolve_hex(str(item["color"])), "count": counter[label], "description": str(item["description"])})
        note = "基于标准氨基酸理化分组，不是随机配色。"
        return items, note

    if mode == "按MMPBSA等级":
        counter: Counter[str] = Counter()
        for row in rows:
            label, _, _ = classify_energy_bin(float(getattr(row, "delta_total", 0.0)), theme_name=theme_name)
            counter[label] += 1

        for item in _scheme_for_mode("按MMPBSA等级", theme_name):
            label = str(item["label"])
            if counter[label] > 0:
                items.append({"label": label, "color": _resolve_hex(str(item["color"])), "count": counter[label], "description": str(item["description"])})
        note = "MMPBSA 按固定能量区间分级，便于快速比较不同残基的贡献。"
        return items, note

    if mode == "按热点等级":
        counter: Counter[str] = Counter()
        for row in rows:
            chain = normalize_chain(getattr(row, "chain", "A"))
            resid = int(getattr(row, "resid", 0))
            rank = hotspot_rank_map.get((chain, resid)) if hotspot_rank_map else getattr(row, "hotspot_rank", None)
            label, _, _ = classify_hotspot(rank, theme_name=theme_name)
            counter[label] += 1

        for item in _scheme_for_mode("按热点等级", theme_name):
            label = str(item["label"])
            if counter[label] > 0:
                items.append({"label": label, "color": _resolve_hex(str(item["color"])), "count": counter[label], "description": str(item["description"])})
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

    if mode == "按链":
        chains: List[str] = []
        for row in rows:
            chain = normalize_chain(getattr(row, "chain", "A"))
            if chain not in chains:
                chains.append(chain)

        chain_color_map = build_chain_color_map(chains, palette_name)
        counts: Counter[str] = Counter(normalize_chain(getattr(row, "chain", "A")) for row in rows)
        for chain in chains:
            items.append({"label": f"链 {chain}", "color": chain_color_map.get(chain, "#d1d5db"), "count": counts[chain], "description": "按链区分，适合多构象或多链复合物。"})
        note = "按 chain 字段进行分组；多链时会自动生成不同颜色。"
        return items, note

    if mode == "单色":
        return [{"label": "统一颜色", "color": "#d1d5db", "count": len(rows), "description": "所有残基使用同一颜色。"}], ""

    return [], ""