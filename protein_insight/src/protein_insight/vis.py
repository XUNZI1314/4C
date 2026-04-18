"""Visualization helpers using py3Dmol and Streamlit components."""
from typing import Dict, List, Tuple
import streamlit.components.v1 as components
import py3Dmol

from protein_insight.pymol_colors import cycle_palette, get_cartoon_style, resolve_color


SURFACE_COLOR_PROP = "_pi_surface_color"
MAX_SURFACE_ASSIGNMENTS = 1400


def _residue_range_expression(residue_ids: List[int]) -> str:
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


def _chunked(values: List[int], size: int) -> List[List[int]]:
    chunk_size = max(1, int(size))
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _extract_chains_from_pdb(pdb_str: str) -> List[str]:
    chains: List[str] = []
    for line in str(pdb_str).splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        chain = (line[21:22].strip() or 'A') if len(line) >= 22 else 'A'
        if chain not in chains:
            chains.append(chain)
    return chains


def _build_chain_color_map(chains: List[str], palette_name: str) -> Dict[str, str]:
    if not chains:
        return {}
    return dict(zip(chains, cycle_palette(palette_name, len(chains))))


def _build_surface_property_assignments(
    residue_colors: Dict[Tuple[str, int], str],
    chain_names: List[str],
    *,
    palette_name: str,
    fallback_color: str,
    max_assignments: int = MAX_SURFACE_ASSIGNMENTS,
) -> Tuple[List[dict], Dict[str, str]]:
    assignments: List[dict] = [{"props": {SURFACE_COLOR_PROP: fallback_color}}]
    color_map: Dict[str, str] = {fallback_color: fallback_color}
    chain_color_map = _build_chain_color_map(chain_names, palette_name)
    max_budget = max(1, int(max_assignments))
    residue_budget_target = len(residue_colors) + len(chain_names) + 1
    budget = max(64, min(max_budget, residue_budget_target * 2))

    for chain in chain_names:
        if len(assignments) >= budget:
            return assignments, color_map
        chain_color = chain_color_map.get(chain, fallback_color)
        assignments.append({"chain": chain, "props": {SURFACE_COLOR_PROP: chain_color}})
        color_map[chain_color] = chain_color

    grouped: Dict[Tuple[str, str], List[int]] = {}
    for (chain, resid), color in residue_colors.items():
        grouped.setdefault((str(chain), resolve_color(color)), []).append(int(resid))

    for (chain, color), residue_ids in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        for residue_chunk in _chunked(sorted(set(residue_ids)), 8):
            if len(assignments) >= budget:
                return assignments, color_map
            residue_expression = _residue_range_expression(residue_chunk)
            if not residue_expression:
                continue
            assignments.append({"chain": chain, "resi": residue_expression, "props": {SURFACE_COLOR_PROP: color}})
            color_map[color] = color

    return assignments, color_map


def _add_surface_with_fallback(view, surface_type, style: dict, residue_sel=None) -> bool:
    candidates = [surface_type, getattr(py3Dmol, "SAS", None), py3Dmol.VDW]
    tried = set()
    for candidate in candidates:
        if candidate is None or candidate in tried:
            continue
        tried.add(candidate)
        try:
            if residue_sel is None:
                view.addSurface(candidate, style)
            else:
                view.addSurface(candidate, style, residue_sel)
            return True
        except Exception:
            continue
    return False


def render_pdb(pdb_str: str,
               style: str = 'cartoon',
               cartoon_theme: str = 'PyMOL 风格',
               residue_colors: Dict[Tuple[str, int], str] = None,
               highlight: Tuple[str, int] = None,
               height: int = 480,
               width: int = 800,
               surface_opacity: float = 0.4,
               surface_color: str = '#c0c0c0',
               surface_colorize: bool = False,
               background_color: str = '#f6f9fc',
               palette_name: str = 'PyMOL 经典'):
    """Render PDB string in Streamlit using py3Dmol with improved defaults for visibility.

    Notes:
    - `surface_opacity` default lowered to 0.4 and `surface_color` set to light gray to avoid white-on-white.
    - `background_color` defaults to a very light page background; adjust from app UI if needed.
    """
    if style == 'stick':
        style = 'sticks'
    if style == 'ball_stick':
        style = 'sticks'

    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, 'pdb')
    resolved_surface_color = resolve_color(surface_color)
    surface_opacity_value = max(0.0, min(1.0, float(surface_opacity)))

    # base style applied to whole model (use empty selection to target model)
    try:
        if style == 'cartoon':
            view.setStyle({}, get_cartoon_style(cartoon_theme))
        elif style == 'sticks':
            view.setStyle({}, {'stick': {}})
    except Exception:
        pass

    # apply per-residue colors using the matching representation
    if residue_colors and style != 'surface':
        for (chain, resid), color in residue_colors.items():
            try:
                sel = {'chain': chain, 'resi': str(resid)}
                if style == 'cartoon':
                    view.setStyle(sel, {'cartoon': {'color': color}})
                elif style == 'sticks':
                    view.setStyle(sel, {'stick': {'color': color}})
            except Exception:
                continue

    # surface mode is a dedicated display mode; do not overlay it on other modes.
    if style == 'surface':
        try:
            view.setStyle({}, {})
            if surface_colorize and residue_colors:
                chain_names = _extract_chains_from_pdb(pdb_str)
                surface_assignments, surface_color_map = _build_surface_property_assignments(
                    residue_colors,
                    chain_names,
                    palette_name=palette_name,
                    fallback_color=resolved_surface_color,
                )
                if len(surface_assignments) > MAX_SURFACE_ASSIGNMENTS:
                    surface_assignments = surface_assignments[:MAX_SURFACE_ASSIGNMENTS]
                try:
                    if hasattr(view, "mapAtomProperties"):
                        view.mapAtomProperties(surface_assignments)
                except Exception:
                    surface_assignments = []
                    surface_color_map = {}

                surface_colorized = bool(surface_assignments) and _add_surface_with_fallback(
                    view,
                    py3Dmol.VDW,
                    {
                        'opacity': max(0.68, surface_opacity_value),
                        'colorscheme': {'prop': SURFACE_COLOR_PROP, 'map': surface_color_map},
                    },
                )
                if not surface_colorized:
                    _add_surface_with_fallback(
                        view,
                        py3Dmol.VDW,
                        {
                            'opacity': max(0.2, min(0.32, surface_opacity_value * 0.4)),
                            'color': resolved_surface_color,
                        },
                    )
            else:
                _add_surface_with_fallback(
                    view,
                    py3Dmol.VDW,
                    {
                        'opacity': max(0.2, min(0.32, surface_opacity_value * 0.4)),
                        'color': resolved_surface_color,
                    },
                )
        except Exception:
            pass

    # highlighted residue
    if highlight:
        ch, r = highlight
        try:
            if style == 'cartoon':
                view.setStyle({'chain': ch, 'resi': str(r)}, {'cartoon': {'color': '#ff5e57'}})
            elif style == 'sticks':
                view.setStyle({'chain': ch, 'resi': str(r)}, {'stick': {'color': '#ff5e57'}})
            elif style == 'surface' and surface_colorize:
                view.addSurface(
                    py3Dmol.VDW,
                    {'opacity': max(0.7, surface_opacity_value), 'color': '#ff5e57'},
                    {'chain': ch, 'resi': str(r)},
                )
        except Exception:
            pass

    # set a contrasting background and center the view
    try:
        view.setBackgroundColor(background_color)
    except Exception:
        view.setBackgroundColor('white')

    try:
        view.zoomTo()
    except Exception:
        pass

    html = view._make_html()
    components.html(html, height=height)
