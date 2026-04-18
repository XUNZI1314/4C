from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_insight import pymol_backend


def test_render_pdb_png_falls_back_to_external_python(monkeypatch, tmp_path):
    real_find_spec = pymol_backend.importlib.util.find_spec

    def fake_find_spec(name):
        if name in {'pymol', 'pymol2'}:
            return object()
        return real_find_spec(name)

    monkeypatch.setattr(pymol_backend.importlib.util, 'find_spec', fake_find_spec)
    monkeypatch.setattr(pymol_backend, '_load_pymol_cmd', lambda: (_ for _ in ()).throw(RuntimeError('direct pymol failed')))
    monkeypatch.setattr(pymol_backend, '_find_local_pymol_python', lambda: tmp_path / 'python.exe')

    def fake_external_render(
        python_exe,
        pdb_path,
        png_path,
        style,
        cartoon_theme,
        residue_colors,
        highlight,
        width,
        height,
        surface_color,
        surface_opacity,
        surface_colorize,
        background_color,
    ):
        Path(png_path).write_bytes(b'fake-png')

    monkeypatch.setattr(pymol_backend, '_render_with_external_python', fake_external_render)

    png_bytes = pymol_backend.render_pdb_png('ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND')

    assert png_bytes == b'fake-png'
