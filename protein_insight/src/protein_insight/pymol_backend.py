"""Optional native PyMOL rendering backend.

This backend renders a static PNG via the PyMOL Python API when PyMOL is
installed locally. It is intentionally optional and falls back to the
interactive 3Dmol backend when unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from protein_insight.pymol_colors import resolve_color


def _candidate_install_roots() -> List[Path]:
    roots: List[Path] = []
    env_root = os.environ.get("PYMOL_HOME") or os.environ.get("PYMOL2_HOME")
    if env_root:
        roots.append(Path(env_root))

    home = Path.home()
    roots.extend(
        [
            home / "AppData" / "Local" / "Schrodinger" / "PyMOL2",
            home / "AppData" / "Local" / "Schrodinger" / "PyMOL",
        ]
    )
    return roots


def _find_local_pymol_python() -> Path | None:
    for root in _candidate_install_roots():
        python_exe = root / "python.exe"
        if python_exe.exists():
            return python_exe
    return None


def _find_local_pymol_exe() -> Path | None:
    for root in _candidate_install_roots():
        pymol_exe = root / "Scripts" / "pymol.exe"
        if pymol_exe.exists():
            return pymol_exe
    return None


def can_use_pymol() -> bool:
    return bool(
        importlib.util.find_spec("pymol2")
        or importlib.util.find_spec("pymol")
        or _find_local_pymol_python()
    )


def _hex_to_rgb01(hex_color: str):
    value = resolve_color(hex_color).lstrip("#")
    return [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _apply_background_color(cmd, color_value: str) -> None:
    resolved = resolve_color(color_value)
    if resolved.startswith("#"):
        try:
            cmd.set("bg_rgb", _hex_to_rgb01(resolved))
            try:
                cmd.set("ray_opaque_background", 0)
            except Exception:
                pass
            return
        except Exception:
            pass
    try:
        cmd.bg_color(color_value)
    except Exception:
        cmd.bg_color("white")


def _load_pymol_cmd():
    pymol2_spec = importlib.util.find_spec("pymol2")
    if pymol2_spec is not None:
        import pymol2

        context = pymol2.PyMOL()
        if hasattr(context, "start"):
            context.start()
        return context, context.cmd

    pymol_spec = importlib.util.find_spec("pymol")
    if pymol_spec is not None:
        import pymol

        if hasattr(pymol, "finish_launching"):
            try:
                pymol.finish_launching(["pymol", "-qc"])
            except Exception:
                pass
        return None, pymol.cmd

    raise RuntimeError("PyMOL is not available in the current environment.")


def _render_with_cmd(
    cmd,
    pdb_path: Path,
    png_path: Path,
    style: str,
    cartoon_theme: str,
    residue_colors: Dict[Tuple[str, int], str] | None,
    highlight: Tuple[str, int] | None,
    width: int,
    height: int,
    surface_color: str,
    surface_opacity: float,
    surface_colorize: bool,
    background_color: str,
) -> None:
    cmd.reinitialize()
    _apply_background_color(cmd, background_color)

    object_name = "mol"
    cmd.load(str(pdb_path), object_name)
    cmd.remove(f"{object_name} and hydro")

    normalized_style = "cartoon" if style not in {"cartoon", "sticks", "ball_stick", "surface"} else style
    if normalized_style == "cartoon":
        _apply_cartoon_style(cmd, cartoon_theme)
    elif normalized_style in {"sticks", "ball_stick"}:
        _apply_ball_stick_style(cmd)
    else:
        _apply_surface_style(cmd)

    color_cache: Dict[str, str] = {}
    if normalized_style == "surface":
        _color_selection(cmd, object_name, surface_color, color_cache)

    if residue_colors and (normalized_style != "surface" or surface_colorize):
        for (chain, resid), color in residue_colors.items():
            selection = f"{object_name} and chain {chain} and resi {int(resid)}"
            _color_selection(cmd, selection, color, color_cache)

    if normalized_style == "surface":
        try:
            cmd.set("surface_transparency", float(surface_opacity), object_name)
        except Exception:
            try:
                cmd.set("transparency", float(surface_opacity), object_name)
            except Exception:
                pass

    if highlight and (normalized_style != "surface" or surface_colorize):
        chain, resid = highlight
        highlight_sel = f"{object_name} and chain {chain} and resi {int(resid)}"
        try:
            _color_selection(cmd, highlight_sel, "hotpink", color_cache)
        except Exception:
            pass

    if normalized_style == "cartoon":
        _force_cartoon_only(cmd, object_name)

    try:
        cmd.orient(object_name)
    except Exception:
        pass

    try:
        cmd.png(str(png_path), width=width, height=height, ray=1)
    except TypeError:
        cmd.png(str(png_path))


EXTERNAL_RENDER_SCRIPT = r"""
from pathlib import Path
import json
import sys

from pymol import cmd, finish_launching


def hex_to_rgb01(hex_color):
    value = hex_color.lstrip('#')
    return [int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def color_selection(selection, hex_color, cache):
    if hex_color not in cache:
        cache[hex_color] = f"pi_{len(cache) + 1:03d}"
        cmd.set_color(cache[hex_color], hex_to_rgb01(hex_color))
    cmd.color(cache[hex_color], selection)


def apply_cartoon_style(theme_name):
    cmd.hide('everything', 'all')
    try:
        cmd.dss('all')
    except Exception:
        pass
    cmd.show('cartoon', 'all')
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('cartoon_smooth_loops', 1)
    cmd.set('cartoon_side_chain_helper', 0)
    if theme_name == '链色卡通':
        cmd.set('cartoon_sampling', 14)
    elif theme_name == '简洁卡通':
        cmd.set('cartoon_sampling', 10)


def apply_ball_stick_style():
    cmd.hide('everything', 'all')
    cmd.show('sticks', 'all')
    cmd.set('stick_radius', 0.22)


def apply_surface_style():
    cmd.hide('everything', 'all')
    cmd.show('surface', 'all')


def force_cartoon_only(object_name):
    try:
        cmd.set('cartoon_side_chain_helper', 0, object_name)
    except Exception:
        try:
            cmd.set('cartoon_side_chain_helper', 0)
        except Exception:
            pass

    for rep in ('sticks', 'lines', 'spheres', 'nonbonded'):
        try:
            cmd.hide(rep, object_name)
        except Exception:
            pass

    try:
        cmd.show('cartoon', object_name)
    except Exception:
        pass


def apply_surface_transparency(selection, opacity):
    try:
        cmd.set('surface_transparency', float(opacity), selection)
    except Exception:
        try:
            cmd.set('transparency', float(opacity), selection)
        except Exception:
            pass


def main():
    finish_launching(['pymol', '-cq'])
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    pdb_path = Path(payload['pdb_path'])
    png_path = Path(payload['png_path'])

    cmd.reinitialize()
    def apply_background_color(color_value):
        resolved = color_value
        if resolved.startswith('#'):
            try:
                cmd.set('bg_rgb', hex_to_rgb01(resolved))
                return
            except Exception:
                pass
        try:
            cmd.bg_color(color_value)
        except Exception:
            cmd.bg_color('white')

    apply_background_color(payload.get('background_color', 'white'))

    object_name = 'mol'
    cmd.load(str(pdb_path), object_name)
    cmd.remove(f'{object_name} and hydro')

    style = payload.get('style', 'cartoon')
    cartoon_theme = payload.get('cartoon_theme', 'PyMOL 风格')
    residue_colors = payload.get('residue_colors', [])
    highlight = payload.get('highlight')
    surface_color = payload.get('surface_color', '#d6e6ff')
    surface_opacity = float(payload.get('surface_opacity', 0.35))
    surface_colorize = bool(payload.get('surface_colorize', False))

    if style == 'cartoon':
        apply_cartoon_style(cartoon_theme)
    elif style in ('sticks', 'ball_stick'):
        apply_ball_stick_style()
    else:
        apply_surface_style()

    color_cache = {}
    if style == 'surface':
        color_selection(object_name, surface_color, color_cache)

    if style != 'surface' or surface_colorize:
        for item in residue_colors:
            selection = f"{object_name} and chain {item['chain']} and resi {int(item['resid'])}"
            color_selection(selection, item['color'], color_cache)

    if style == 'surface':
        apply_surface_transparency(object_name, surface_opacity)

    if highlight and (style != 'surface' or surface_colorize):
        selection = f"{object_name} and chain {highlight['chain']} and resi {int(highlight['resid'])}"
        try:
            color_selection(selection, '#ff69b4', color_cache)
        except Exception:
            pass

    if style == 'cartoon':
        force_cartoon_only(object_name)

    try:
        cmd.orient(object_name)
    except Exception:
        pass

    try:
        cmd.png(str(png_path), width=int(payload.get('width', 900)), height=int(payload.get('height', 700)), ray=1)
    except TypeError:
        cmd.png(str(png_path))


if __name__ == '__main__':
    main()
"""


def _render_with_external_python(
    python_exe: Path,
    pdb_path: Path,
    png_path: Path,
    style: str,
    cartoon_theme: str,
    residue_colors: Dict[Tuple[str, int], str] | None,
    highlight: Tuple[str, int] | None,
    width: int,
    height: int,
    surface_color: str,
    surface_opacity: float,
    surface_colorize: bool,
    background_color: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        payload_path = tmp_path / "payload.json"
        script_path = tmp_path / "render.py"

        payload = {
            "pdb_path": str(pdb_path),
            "png_path": str(png_path),
            "style": style,
            "cartoon_theme": cartoon_theme,
            "residue_colors": [
                {"chain": chain, "resid": int(resid), "color": resolve_color(color)}
                for (chain, resid), color in (residue_colors or {}).items()
            ],
            "highlight": None if highlight is None else {"chain": highlight[0], "resid": int(highlight[1])},
            "width": int(width),
            "height": int(height),
            "surface_color": resolve_color(surface_color),
            "surface_opacity": float(surface_opacity),
            "surface_colorize": bool(surface_colorize),
            "background_color": resolve_color(background_color),
        }
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        script_path.write_text(EXTERNAL_RENDER_SCRIPT, encoding="utf-8")

        completed = subprocess.run(
            [str(python_exe), str(script_path), str(payload_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=tmp_dir,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout or "PyMOL render failed")


def _apply_cartoon_style(cmd, theme_name: str) -> None:
    try:
        cmd.hide("everything", "all")
        try:
            cmd.dss("all")
        except Exception:
            pass
        cmd.show("cartoon", "all")
        cmd.set("cartoon_fancy_helices", 1)
        cmd.set("cartoon_smooth_loops", 1)
        cmd.set('cartoon_side_chain_helper', 0)
        if theme_name == "链色卡通":
            cmd.set("cartoon_sampling", 14)
        elif theme_name == "简洁卡通":
            cmd.set("cartoon_sampling", 10)
    except Exception:
        pass


def _apply_ball_stick_style(cmd) -> None:
    try:
        cmd.hide("everything", "all")
        cmd.show("sticks", "all")
        cmd.set("stick_radius", 0.22)
    except Exception:
        pass


def _apply_surface_style(cmd) -> None:
    try:
        cmd.hide("everything", "all")
        cmd.show("surface", "all")
    except Exception:
        pass


def _force_cartoon_only(cmd, object_name: str) -> None:
    try:
        cmd.set("cartoon_side_chain_helper", 0, object_name)
    except Exception:
        try:
            cmd.set("cartoon_side_chain_helper", 0)
        except Exception:
            pass

    for rep in ("sticks", "lines", "spheres", "nonbonded"):
        try:
            cmd.hide(rep, object_name)
        except Exception:
            pass

    try:
        cmd.show("cartoon", object_name)
    except Exception:
        pass


def _color_selection(cmd, selection: str, color_name: str, color_cache: Dict[str, str]) -> None:
    hex_value = resolve_color(color_name)
    if not hex_value.startswith("#"):
        try:
            cmd.color(hex_value, selection)
            return
        except Exception:
            hex_value = "#d9d9d9"

    if hex_value not in color_cache:
        color_cache[hex_value] = f"pi_{len(color_cache) + 1:03d}"
        cmd.set_color(color_cache[hex_value], _hex_to_rgb01(hex_value))
    cmd.color(color_cache[hex_value], selection)


def render_pdb_png(
    pdb_str: str,
    style: str = "cartoon",
    cartoon_theme: str = "PyMOL 风格",
    residue_colors: Dict[Tuple[str, int], str] | None = None,
    highlight: Tuple[str, int] | None = None,
    width: int = 900,
    height: int = 700,
    surface_color: str = "#d6e6ff",
    surface_opacity: float = 0.35,
    surface_colorize: bool = False,
    background_color: str = "#ffffff",
) -> bytes:
    """Render a PDB string to PNG bytes using the native PyMOL API."""
    normalized_style = "cartoon" if style not in {"cartoon", "sticks", "ball_stick", "surface"} else style
    direct_error: Exception | None = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pdb_path = tmp_path / "input.pdb"
        png_path = tmp_path / "render.png"
        pdb_path.write_text(pdb_str, encoding="utf-8")

        direct_import_available = bool(importlib.util.find_spec("pymol2") or importlib.util.find_spec("pymol"))
        if direct_import_available:
            context = None
            try:
                context, cmd = _load_pymol_cmd()
                _render_with_cmd(
                    cmd,
                    pdb_path,
                    png_path,
                    normalized_style,
                    cartoon_theme,
                    residue_colors,
                    highlight,
                    width,
                    height,
                    surface_color,
                    surface_opacity,
                    surface_colorize,
                    background_color,
                )
                return png_path.read_bytes()
            except Exception as exc:
                direct_error = exc
            finally:
                if context is not None and hasattr(context, "stop"):
                    try:
                        context.stop()
                    except Exception:
                        pass

        external_python = _find_local_pymol_python()
        if external_python is None:
            if direct_error is not None:
                raise direct_error
            raise RuntimeError("PyMOL is not available in the current environment.")

        _render_with_external_python(
            external_python,
            pdb_path,
            png_path,
            normalized_style,
            cartoon_theme,
            residue_colors,
            highlight,
            width,
            height,
            surface_color,
            surface_opacity,
            surface_colorize,
            background_color,
        )
        return png_path.read_bytes()