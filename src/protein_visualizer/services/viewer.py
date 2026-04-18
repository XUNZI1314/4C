from typing import Optional

import py3Dmol

from protein_visualizer.config.settings import SETTINGS


MAX_SURFACE_COLORING_RESIDUES = 480
MAX_SURFACE_COLORING_ATOMS = 6500
SURFACE_ISLAND_CHUNK_SIZE = 120
SURFACE_COLOR_PROP = "_pi_surface_color"
STABLE_CHAIN_SURFACE_PALETTE = [
    "#1e3a8a",
    "#2563eb",
    "#0ea5e9",
    "#06b6d4",
    "#14b8a6",
    "#10b981",
    "#5f7f9f",
    "#7796ad",
    "#4f7892",
    "#2f6f8f",
    "#4c8d83",
    "#739e77",
    "#c8d0d8",
    "#dce3ea",
]
BACKGROUND_SURFACE_PALETTE = [
    "#1f3f77",
    "#245c95",
    "#2d70b0",
    "#3b82c4",
    "#5aa3d8",
    "#8cc7eb",
    "#0d6f8c",
    "#148c97",
    "#1aa3a0",
    "#22b59d",
    "#31c9b1",
    "#4ed3b4",
    "#4b6b7d",
    "#617789",
    "#7a8694",
    "#929fac",
    "#aab4bf",
    "#c6cfd8",
    "#d8dfe5",
    "#e7ecf1",
    "#c98a6d",
    "#d39a70",
    "#dbb07a",
    "#c87867",
    "#b56a62",
]

SURFACE_COOL_PALETTE = [
    "#1d4f91",
    "#2267b3",
    "#2f7fc7",
    "#4498d9",
    "#5bb0e4",
    "#85c8ef",
    "#0d7f8e",
    "#10a0a3",
    "#17b7a8",
    "#29c8ac",
    "#48d3af",
    "#73dcb5",
    "#8fa4b8",
    "#a7b3c1",
    "#c2ccd6",
    "#d7dfe6",
    "#c87767",
    "#d98e69",
    "#e0af76",
    "#b36f62",
    "#6e7b8b",
    "#516477",
]

SURFACE_CONTRAST_PALETTE = [
    "#0b224d",
    "#12356f",
    "#1b4f95",
    "#2165b4",
    "#2d7ac8",
    "#3d8fda",
    "#57a7e5",
    "#0c7589",
    "#0f968f",
    "#16aa9c",
    "#1bbf9d",
    "#31cdab",
    "#2f5f55",
    "#3b735f",
    "#4e8567",
    "#6a9a72",
    "#93b07d",
    "#c59a6a",
    "#d57f67",
    "#e08d5b",
    "#626f82",
    "#4d5f74",
]


def _is_neutral_like_color(color: str) -> bool:
    value = str(color or "").strip().lower()
    if not value:
        return True

    neutral_palette = {
        str(SETTINGS.neutral_color).strip().lower(),
        "#b8c1cc",
        "#c7c7c7",
        "#cccccc",
        "#d0d0d0",
        "#b0b0b0",
        "#999999",
        "#808080",
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

        channel_spread = max(red, green, blue) - min(red, green, blue)
        channel_mean = (red + green + blue) / 3.0
        if channel_spread <= 14 and 110 <= channel_mean <= 240:
            return True

    return False


def _stable_chain_surface_color(chain: str) -> str:
    text = str(chain or "").strip() or "A"
    seed = sum(ord(ch) for ch in text)
    return STABLE_CHAIN_SURFACE_PALETTE[seed % len(STABLE_CHAIN_SURFACE_PALETTE)]


def _cool_surface_color(color: str) -> str:
    value = str(color or "").strip().lower()
    if not (value.startswith("#") and len(value) == 7):
        return value

    try:
        red = int(value[1:3], 16)
        green = int(value[3:5], 16)
        blue = int(value[5:7], 16)
    except Exception:
        return value

    # Keep pure reds/magentas, but remap orange/yellow-heavy tones into a cool PyMOL-like palette.
    if not (red >= 180 and green >= 80 and green > blue + 20):
        return value

    seed = red * 3 + green * 5 + blue * 7
    return SURFACE_COOL_PALETTE[seed % len(SURFACE_COOL_PALETTE)]


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


def _surface_anchor_color(chain: str, resid: int, total_residues: int, salt: int = 0) -> str:
    text = str(chain or "").strip() or "A"
    seed = sum(ord(ch) for ch in text) * 13 + int(resid) * 17 + max(1, int(total_residues)) * 7 + int(salt) * 31
    return SURFACE_CONTRAST_PALETTE[seed % len(SURFACE_CONTRAST_PALETTE)]


def _increase_surface_contrast(color: str, chain: str, resid: int, total_residues: int) -> str:
    value = str(color or "").strip().lower()
    if not (value.startswith("#") and len(value) == 7):
        return value

    primary = _surface_anchor_color(chain, resid, total_residues, salt=1)
    secondary = _surface_anchor_color(chain, resid, total_residues, salt=2)
    mixed = _blend_hex(value, primary, 0.24 + (int(resid) % 5) * 0.03)
    mixed = _blend_hex(mixed, secondary, 0.08 + ((int(resid) // 3) % 4) * 0.02)
    tone_target = "#f8fbff" if ((int(resid) + len(str(chain or "A"))) % 2 == 0) else "#111827"
    return _blend_hex(mixed, tone_target, 0.09)


def _background_surface_color(chain: str, resid: int, total_residues: int) -> str:
    text = str(chain or "").strip() or "A"
    residue_total = max(1, int(total_residues))
    segment_size = max(6, min(14, max(1, residue_total // 24)))
    segment_index = max(0, int(resid) - 1) // segment_size
    fine_index = max(0, int(resid) - 1) // max(2, segment_size // 3)
    seed = sum(ord(ch) for ch in text)
    palette_index = (
        seed * 7
        + segment_index * 11
        + fine_index * 5
        + int(resid) * 3
        + residue_total
    ) % len(BACKGROUND_SURFACE_PALETTE)
    return BACKGROUND_SURFACE_PALETTE[palette_index]


def _surface_detail_color(chain: str, resid: int, total_residues: int) -> str:
    text = str(chain or "").strip() or "A"
    seed = (
        sum(ord(ch) for ch in text) * 5
        + int(resid) * 9
        + max(1, int(total_residues)) * 3
        + (int(resid) // 8) * 3
    )
    return SURFACE_COOL_PALETTE[seed % len(SURFACE_COOL_PALETTE)]


def _pick_surface_variant(seed: int, *colors: str) -> str:
    options = [str(color or "").strip().lower() for color in colors if str(color or "").strip()]
    if not options:
        return ""
    return options[int(seed) % len(options)]


def _surface_pocket_color(base_color: str, chain: str, resid: int, total_residues: int) -> str:
    value = str(base_color or "").strip().lower()
    if value.startswith("#") and len(value) == 7:
        try:
            red = int(value[1:3], 16)
            green = int(value[3:5], 16)
            blue = int(value[5:7], 16)
        except Exception:
            red = green = blue = 0
    else:
        red = green = blue = 0

    chain_seed = sum(ord(ch) for ch in str(chain or "").strip() or "A")
    seed = red * 11 + green * 13 + blue * 17 + chain_seed * 19 + int(resid) * 23 + max(1, int(total_residues)) * 5
    palette_color = SURFACE_COOL_PALETTE[seed % len(SURFACE_COOL_PALETTE)]
    if int(resid) % 5 == 1:
        return palette_color

    accent_color = _background_surface_color(chain, resid, total_residues)
    detail_color = _surface_detail_color(chain, resid, total_residues)
    return _pick_surface_variant(seed, palette_color, accent_color, detail_color)


def _surface_micro_variation(base_color: str, chain: str, resid: int, total_residues: int) -> str:
    value = str(base_color or "").strip().lower()
    if not (value.startswith("#") and len(value) == 7):
        return value

    text = str(chain or "").strip() or "A"
    seed = sum(ord(ch) * 17 for ch in text) + int(resid) * 131 + max(1, int(total_residues)) * 19

    detail_color = _surface_detail_color(chain, resid, total_residues)
    accent_color = _background_surface_color(chain, resid, total_residues)
    return _pick_surface_variant(seed, value, detail_color, accent_color)


def _surface_rich_color(
    color: str,
    chain: str,
    resid: int,
    total_residues: int,
    *,
    is_heat_mode: bool,
    color_mode: Optional[str],
) -> str:
    base_color = _cool_surface_color(color)
    accent_color = _background_surface_color(chain, resid, total_residues)
    detail_color = _surface_detail_color(chain, resid, total_residues)
    anchor_color = _surface_anchor_color(chain, resid, total_residues, salt=3)
    seed = sum(ord(ch) for ch in str(chain or "").strip() or "A") * 13 + int(resid) * 17 + max(1, int(total_residues)) * 7

    if _is_neutral_like_color(base_color):
        if color_mode == "按口袋识别":
            return _surface_pocket_color(base_color, chain, resid, total_residues)
        return _pick_surface_variant(seed, anchor_color, accent_color, detail_color)

    if is_heat_mode:
        return _pick_surface_variant(seed, base_color, detail_color, accent_color)

    if color_mode == "按口袋识别":
        return _surface_pocket_color(base_color, chain, resid, total_residues)

    return _pick_surface_variant(seed, base_color, accent_color, detail_color)


def _pymol_cartoon_style():
    return {"cartoon": {"colorscheme": "ssPyMol", "opacity": 0.96, "arrows": True, "thickness": 0.48}}


def _normalize_opacity(opacity: float, fallback: float = SETTINGS.default_opacity) -> float:
    try:
        value = float(opacity)
    except Exception:
        value = float(fallback)

    if value != value:
        value = float(fallback)

    return max(0.0, min(1.0, value))


def _resolve_overlay_opacity(opacity: float, *, minimum: float, maximum: float) -> float:
    value = _normalize_opacity(opacity)
    lower_bound = max(0.0, min(1.0, float(minimum)))
    upper_bound = max(lower_bound, min(1.0, float(maximum)))
    return round(max(lower_bound, min(upper_bound, value)), 3)


def _resolve_surface_opacity(
    opacity: float,
    atom_count: int,
    *,
    surface_single_color: bool,
    color_mode: Optional[str] = None,
) -> float:
    slider_value = _normalize_opacity(opacity)
    atom_total = max(0, int(atom_count))

    if atom_total >= 900:
        minimum = 0.46
    elif atom_total >= 450:
        minimum = 0.38
    elif atom_total >= 180:
        minimum = 0.30
    else:
        minimum = 0.22

    if surface_single_color:
        minimum += 0.04
    elif color_mode == "按口袋识别":
        minimum += 0.02

    minimum = max(0.18, min(0.62, minimum))
    maximum = 0.96
    return round(minimum + (maximum - minimum) * slider_value, 3)


def _quantize_hex_color(color: str, step: int = 48) -> str:
    value = str(color or "").strip()
    if not (value.startswith("#") and len(value) == 7):
        return value

    try:
        red = int(value[1:3], 16)
        green = int(value[3:5], 16)
        blue = int(value[5:7], 16)
    except Exception:
        return value

    unit = max(4, int(step))

    def _q(channel: int) -> int:
        quantized = int(round(channel / unit) * unit)
        return max(0, min(255, quantized))

    return f"#{_q(red):02x}{_q(green):02x}{_q(blue):02x}"


def _chunked(values: list[int], size: int) -> list[list[int]]:
    chunk_size = max(1, int(size))
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _residue_range_expression(residue_ids: list[int]) -> str:
    if not residue_ids:
        return ""

    ordered = sorted(set(int(value) for value in residue_ids))
    ranges = []
    start = ordered[0]
    end = ordered[0]

    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        start = value
        end = value

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


def _split_surface_islands(residue_ids: list[int], chunk_size: int = SURFACE_ISLAND_CHUNK_SIZE) -> list[list[int]]:
    ordered = sorted(set(int(value) for value in residue_ids))
    if not ordered:
        return []

    island_size = max(1, int(chunk_size))
    if len(ordered) <= island_size:
        return [ordered]

    return _chunked(ordered, island_size)


def _build_surface_property_assignments(
    surface_groups: dict[tuple[str, str], list[int]],
    surface_chains: list[str],
    *,
    surface_single_color: bool,
    uniform_surface_color: str,
) -> tuple[list[dict], dict[str, str]]:
    assignments: list[dict] = [{"props": {SURFACE_COLOR_PROP: uniform_surface_color}}]
    color_map: dict[str, str] = {uniform_surface_color: uniform_surface_color}

    if surface_single_color:
        return assignments, color_map

    for (chain, color), residues in sorted(surface_groups.items(), key=lambda item: len(item[1]), reverse=True):
        residue_ids = sorted(set(int(value) for value in residues))
        for residue_id in residue_ids:
            assignments.append({"chain": chain, "resi": str(residue_id), "props": {SURFACE_COLOR_PROP: color}})
        color_map[color] = color

    return assignments, color_map


def _coarse_surface_type(atom_count: int):
    if atom_count <= 0:
        return py3Dmol.VDW
    if atom_count <= 5000:
        return getattr(py3Dmol, "SES", getattr(py3Dmol, "SAS", py3Dmol.VDW))
    if atom_count <= 12000:
        return getattr(py3Dmol, "SAS", py3Dmol.VDW)
    return py3Dmol.VDW


def _add_surface_with_fallback(viewer, surface_type, style: dict, residue_sel=None) -> bool:
    candidates = [surface_type, getattr(py3Dmol, "SAS", None), py3Dmol.VDW]
    tried = set()
    for candidate in candidates:
        if candidate is None or candidate in tried:
            continue
        tried.add(candidate)
        try:
            if residue_sel is None:
                viewer.addSurface(candidate, style)
            else:
                viewer.addSurface(candidate, style, residue_sel)
            return True
        except Exception:
            continue
    return False


def _reduce_surface_groups_for_stability(
    surface_groups: dict[tuple[str, str], list[int]],
    *,
    max_groups_per_chain: int,
    prefer_non_neutral: bool = False,
    ensure_non_neutral: bool = False,
    merge_overflow: bool = True,
) -> dict[tuple[str, str], list[int]]:
    if not surface_groups:
        return {}

    chain_buckets: dict[str, list[tuple[str, list[int]]]] = {}
    for (chain, color), residues in surface_groups.items():
        chain_buckets.setdefault(chain, []).append((color, sorted(set(int(value) for value in residues))))

    reduced: dict[tuple[str, str], list[int]] = {}
    for chain, items in chain_buckets.items():
        if prefer_non_neutral:
            ranked = sorted(
                items,
                key=lambda entry: (_is_neutral_like_color(entry[0]), -len(entry[1])),
            )
        else:
            ranked = sorted(items, key=lambda entry: len(entry[1]), reverse=True)

        keep_count = max(1, int(max_groups_per_chain))
        kept_items: list[tuple[str, list[int]]] = []

        if ensure_non_neutral:
            for color, residues in ranked:
                if not _is_neutral_like_color(color):
                    kept_items.append((color, residues))
                    break

        for color, residues in ranked:
            if len(kept_items) >= keep_count:
                break
            if any(color == kept_color for kept_color, _ in kept_items):
                continue
            kept_items.append((color, residues))

        if not kept_items:
            continue

        overflow_residues: list[int] = []

        for color, residues in ranked[keep_count:]:
            overflow_residues.extend(residues)

        for color, residues in kept_items:
            reduced[(chain, color)] = sorted(set(reduced.get((chain, color), []) + residues))

        if merge_overflow and overflow_residues:
            overflow_chunks = _split_surface_islands(overflow_residues, SURFACE_ISLAND_CHUNK_SIZE)
            for chunk in overflow_chunks:
                overflow_color = _background_surface_color(chain, chunk[0], len(overflow_residues))
                reduced[(chain, overflow_color)] = sorted(
                    set(reduced.get((chain, overflow_color), []) + chunk)
                )

    return reduced


def build_view(
    pdb_text: str,
    energy_table,
    threshold: float,
    display_mode: str,
    show_backbone: bool,
    opacity: float,
    selected_chain: Optional[str],
    selected_resid: Optional[int],
    color_mode: Optional[str] = None,
    surface_single_color: bool = False,
    surface_uniform_color: Optional[str] = None,
    viewer_width: Optional[int] = None,
    viewer_height: Optional[int] = None,
):
    resolved_width = int(viewer_width) if viewer_width is not None else int(SETTINGS.viewer_width)
    resolved_height = int(viewer_height) if viewer_height is not None else int(SETTINGS.viewer_height)
    viewer = py3Dmol.view(width=resolved_width, height=resolved_height)
    viewer.addModel(pdb_text, "pdb")
    viewer.setBackgroundColor(SETTINGS.background_color)

    display_mode = display_mode if display_mode in {"cartoon", "sticks", "surface"} else "cartoon"
    heat_modes = {"按DELTA TOTAL 热度", "按能量连续梯度"}
    use_threshold = color_mode in heat_modes if color_mode is not None else True
    is_heat_mode = color_mode in heat_modes if color_mode is not None else True
    force_surface_single_mode = display_mode == "surface" and bool(surface_single_color)

    def _row_color(row):
        for key in ("display_color", "classification_color", "heat_color"):
            value = getattr(row, key, None)
            if isinstance(value, str) and value.strip():
                return value
        return SETTINGS.neutral_color

    if display_mode == "cartoon":
        try:
            viewer.setStyle({}, _pymol_cartoon_style())
        except Exception:
            viewer.setStyle({}, {"cartoon": {"opacity": 0.96}})
    elif display_mode == "sticks":
        try:
            viewer.setStyle({}, {"stick": {}})
        except Exception:
            pass
    elif display_mode == "surface":
        try:
            viewer.setStyle({}, {})
        except Exception:
            pass

    surface_groups = {}
    surface_chains: list[str] = []
    for row in energy_table.itertuples(index=False):
        base_color = _row_color(row)
        if use_threshold:
            try:
                energy_value = abs(float(row.delta_total))
            except Exception:
                energy_value = 0.0
            color = base_color if energy_value >= float(threshold) else SETTINGS.neutral_color
        else:
            color = base_color

        chain = str(getattr(row, "chain", "")).strip() or "A"
        try:
            resid = int(getattr(row, "resid"))
        except Exception:
            continue

        if chain not in surface_chains:
            surface_chains.append(chain)

        residue_sel = {"chain": chain, "resi": resid}

        if display_mode == "cartoon" and show_backbone:
            viewer.setStyle(residue_sel, {"cartoon": {"color": color, "opacity": 0.95}})
        if display_mode == "sticks":
            stick_opacity = _resolve_overlay_opacity(opacity, minimum=0.2, maximum=0.92)
            viewer.addStyle(
                residue_sel,
                {
                    "stick": {"color": color, "radius": 0.22, "opacity": stick_opacity},
                },
            )
        if display_mode == "surface":
            if force_surface_single_mode:
                continue
            surface_color = str(color or SETTINGS.neutral_color).strip().lower() or SETTINGS.neutral_color
            surface_groups.setdefault((chain, surface_color), []).append(resid)

    if display_mode == "surface":
        atom_count = pdb_text.count("\nATOM") + (1 if pdb_text.startswith("ATOM") else 0)
        surface_type = _coarse_surface_type(atom_count)
        surface_opacity = _resolve_surface_opacity(
            opacity,
            atom_count,
            surface_single_color=bool(surface_single_color),
            color_mode=color_mode,
        )
        uniform_surface_color = str(surface_uniform_color or SETTINGS.neutral_color)

        surface_assignments, surface_color_map = _build_surface_property_assignments(
            surface_groups,
            surface_chains,
            surface_single_color=bool(surface_single_color),
            uniform_surface_color=uniform_surface_color,
        )

        try:
            if hasattr(viewer, "mapAtomProperties"):
                viewer.mapAtomProperties(surface_assignments)
        except Exception:
            pass

        surface_rendered = False
        try:
            surface_rendered = _add_surface_with_fallback(
                viewer,
                surface_type,
                {
                    "opacity": surface_opacity,
                    "colorscheme": {"prop": SURFACE_COLOR_PROP, "map": surface_color_map},
                },
            )
        except Exception:
            surface_rendered = False

        if not surface_rendered:
            _add_surface_with_fallback(
                viewer,
                surface_type,
                {
                    "opacity": surface_opacity,
                    "color": uniform_surface_color,
                },
            )

    if selected_chain is not None and selected_resid is not None:
        selection = {"chain": selected_chain, "resi": int(selected_resid)}
        viewer.addStyle(
            selection,
            {
                **({"cartoon": {"color": SETTINGS.highlight_color, "opacity": 0.98}} if display_mode == "cartoon" else {}),
                **({"stick": {"color": SETTINGS.highlight_color, "radius": 0.32}} if display_mode in {"sticks", "surface"} else {}),
            },
        )
        viewer.zoomTo(selection)
    else:
        viewer.zoomTo()
    viewer.spin(False)
    return viewer
