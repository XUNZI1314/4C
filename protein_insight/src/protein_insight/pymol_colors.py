"""PyMOL-like named colors and palettes for protein visualization."""

import colorsys
from typing import Dict, List, Tuple


PYMOL_NAMED_COLORS: Dict[str, str] = {
    "tv_red": "#ff3b30",
    "red": "#e31a1c",
    "firebrick": "#b22222",
    "salmon": "#fa8072",
    "deepsalmon": "#ff8c69",
    "tv_orange": "#ff9500",
    "orange": "#ff7f0e",
    "yelloworange": "#ffb347",
    "tv_yellow": "#ffd60a",
    "yellow": "#ffeb3b",
    "lime": "#b5e61d",
    "green": "#2ca02c",
    "forest": "#228b22",
    "tv_green": "#34c759",
    "cyan": "#00c7be",
    "tv_cyan": "#32ade6",
    "sky": "#7ec8e3",
    "blue": "#1f77b4",
    "tv_blue": "#0a84ff",
    "marine": "#2c3e73",
    "slate": "#708090",
    "violet": "#8a2be2",
    "purple": "#af52de",
    "magenta": "#ff2d55",
    "hotpink": "#ff69b4",
    "wheat": "#f5deb3",
    "lightblue": "#d6e6ff",
    "lightorange": "#ffd8b1",
    "gray": "#8a8a8a",
    "lightgray": "#d9d9d9",
    "white": "#ffffff",
    "black": "#000000",
}


PYMOL_PALETTES: Dict[str, List[str]] = {
    "PyMOL 经典": [
        "tv_red",
        "tv_orange",
        "tv_yellow",
        "tv_green",
        "tv_cyan",
        "tv_blue",
        "violet",
        "magenta",
        "salmon",
        "marine",
        "lime",
        "orange",
    ],
    "PyMOL 高对比": [
        "red",
        "orange",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "hotpink",
        "forest",
        "tv_blue",
        "tv_green",
        "deepsalmon",
    ],
    "PyMOL 冷色": [
        "marine",
        "tv_blue",
        "sky",
        "cyan",
        "tv_cyan",
        "slate",
        "blue",
        "violet",
        "gray",
        "lightgray",
    ],
    "PyMOL 暖色": [
        "tv_red",
        "salmon",
        "deepsalmon",
        "orange",
        "tv_orange",
        "yelloworange",
        "yellow",
        "hotpink",
        "purple",
        "magenta",
    ],
    "PyMOL 莫兰迪": [
        "#7f8ea3",
        "#c59673",
        "#93aa97",
        "#b4b0a1",
        "#6f8ea7",
        "#d2d7de",
        "#a9866e",
        "#8b98a6",
    ],
}


PYMOL_CARTOON_THEMES: Dict[str, Dict[str, object]] = {
    "PyMOL 风格": {
        "colorscheme": "ssPyMol",
        "opacity": 0.96,
        "arrows": True,
        "thickness": 0.48,
    },
    "链色卡通": {
        "colorscheme": "chain",
        "opacity": 0.96,
        "arrows": True,
        "thickness": 0.48,
    },
    "彩虹卡通": {
        "colorscheme": "spectrum",
        "opacity": 0.96,
        "arrows": True,
        "thickness": 0.48,
    },
    "简洁卡通": {
        "opacity": 0.96,
        "arrows": True,
        "thickness": 0.48,
    },
}


def get_named_color_options() -> List[str]:
    return list(PYMOL_NAMED_COLORS.keys())


def resolve_color(color_name_or_hex: str) -> str:
    value = (color_name_or_hex or "").strip()
    if not value:
        return PYMOL_NAMED_COLORS["lightgray"]
    if value.startswith("#"):
        return value
    return PYMOL_NAMED_COLORS.get(value, value)


def get_palette(palette_name: str) -> List[str]:
    palette = PYMOL_PALETTES.get(palette_name, PYMOL_PALETTES["PyMOL 经典"])
    return [resolve_color(name) for name in palette]


_PALETTE_VARIANTS: Tuple[Dict[str, float], ...] = (
    {
        "hue_shift": 0.0,
        "saturation_scale": 1.0,
        "lightness_scale": 1.0,
        "lightness_bias": 0.0,
        "neutral_hue_shift": 0.0,
        "neutral_saturation": 0.18,
        "neutral_lightness_scale": 1.0,
        "neutral_lightness_bias": 0.0,
    },
    {
        "hue_shift": 0.012,
        "saturation_scale": 0.96,
        "lightness_scale": 1.08,
        "lightness_bias": 0.0,
        "neutral_hue_shift": 0.08,
        "neutral_saturation": 0.20,
        "neutral_lightness_scale": 1.04,
        "neutral_lightness_bias": 0.01,
    },
    {
        "hue_shift": -0.012,
        "saturation_scale": 1.04,
        "lightness_scale": 0.92,
        "lightness_bias": 0.0,
        "neutral_hue_shift": 0.56,
        "neutral_saturation": 0.22,
        "neutral_lightness_scale": 0.94,
        "neutral_lightness_bias": -0.01,
    },
    {
        "hue_shift": 0.022,
        "saturation_scale": 0.92,
        "lightness_scale": 1.02,
        "lightness_bias": 0.0,
        "neutral_hue_shift": 0.14,
        "neutral_saturation": 0.24,
        "neutral_lightness_scale": 1.02,
        "neutral_lightness_bias": 0.0,
    },
    {
        "hue_shift": -0.022,
        "saturation_scale": 1.06,
        "lightness_scale": 0.88,
        "lightness_bias": 0.0,
        "neutral_hue_shift": 0.42,
        "neutral_saturation": 0.19,
        "neutral_lightness_scale": 0.90,
        "neutral_lightness_bias": 0.0,
    },
    {
        "hue_shift": 0.032,
        "saturation_scale": 0.94,
        "lightness_scale": 0.96,
        "lightness_bias": 0.0,
        "neutral_hue_shift": 0.20,
        "neutral_saturation": 0.26,
        "neutral_lightness_scale": 0.98,
        "neutral_lightness_bias": 0.01,
    },
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rgb_to_hex(red: float, green: float, blue: float) -> str:
    def _channel(value: float) -> int:
        return max(0, min(255, int(round(value * 255.0))))

    return f"#{_channel(red):02x}{_channel(green):02x}{_channel(blue):02x}"


def _variant_palette_color(color: str, *, variant_index: int, cycle_index: int) -> str:
    resolved = resolve_color(color).strip().lower()
    if not (resolved.startswith("#") and len(resolved) == 7):
        return resolved or "#d9d9d9"

    try:
        red = int(resolved[1:3], 16)
        green = int(resolved[3:5], 16)
        blue = int(resolved[5:7], 16)
    except Exception:
        return resolved

    hue, lightness, saturation = colorsys.rgb_to_hls(red / 255.0, green / 255.0, blue / 255.0)
    profile = _PALETTE_VARIANTS[cycle_index % len(_PALETTE_VARIANTS)]
    jitter = ((variant_index * 7 + cycle_index * 11) % 5) - 2

    if saturation <= 0.12:
        hue = (hue + profile["neutral_hue_shift"] + jitter * 0.014) % 1.0
        saturation = _clamp(
            profile["neutral_saturation"] + ((variant_index + cycle_index) % 3) * 0.025,
            0.14,
            0.35,
        )
        lightness = _clamp(
            lightness * profile["neutral_lightness_scale"] + profile["neutral_lightness_bias"],
            0.16,
            0.90,
        )
    else:
        hue = (hue + profile["hue_shift"] + jitter * 0.004) % 1.0
        saturation = _clamp(saturation * profile["saturation_scale"], 0.16, 1.0)
        lightness = _clamp(lightness * profile["lightness_scale"] + profile["lightness_bias"], 0.14, 0.90)

    red_f, green_f, blue_f = colorsys.hls_to_rgb(hue, lightness, saturation)
    return _rgb_to_hex(red_f, green_f, blue_f)


def _blend_hex(color_a: str, color_b: str, ratio: float) -> str:
    first = resolve_color(color_a).strip().lower()
    second = resolve_color(color_b).strip().lower()
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


def cycle_palette(palette_name: str, count: int) -> List[str]:
    if int(count) <= 0:
        return []

    palette = get_palette(palette_name)
    if not palette:
        palette = list(PYMOL_NAMED_COLORS.values())

    base = [resolve_color(value).lower() for value in palette]
    if len(base) == 1:
        return [base[0] for _ in range(int(count))]

    if int(count) <= len(base):
        return base[: int(count)]

    result: List[str] = []
    base_len = len(base)
    for index in range(int(count)):
        primary = base[index % base_len]
        cycle_index = index // base_len
        if cycle_index == 0:
            result.append(primary)
            continue
        result.append(_variant_palette_color(primary, variant_index=index, cycle_index=cycle_index))

    return result


def get_cartoon_style(theme_name: str) -> Dict[str, Dict[str, object]]:
    theme = PYMOL_CARTOON_THEMES.get(theme_name, PYMOL_CARTOON_THEMES["PyMOL 风格"])
    return {"cartoon": dict(theme)}