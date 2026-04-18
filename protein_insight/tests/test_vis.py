from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_insight import vis


class _FakeView:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.calls = []

    def addModel(self, *args, **kwargs):
        self.calls.append(('addModel', args, kwargs))

    def setBackgroundColor(self, color):
        self.calls.append(('setBackgroundColor', (color,), {}))

    def setStyle(self, selection, style):
        self.calls.append(('setStyle', (selection, style), {}))

    def addSurface(self, *args, **kwargs):
        self.calls.append(('addSurface', args, kwargs))

    def mapAtomProperties(self, *args, **kwargs):
        self.calls.append(('mapAtomProperties', args, kwargs))

    def zoomTo(self, *args, **kwargs):
        self.calls.append(('zoomTo', args, kwargs))

    def _make_html(self):
        return '<div/>'


class _FlakySurfaceView(_FakeView):
    def addSurface(self, *args, **kwargs):
        self.calls.append(('addSurface', args, kwargs))
        if len(args) >= 2 and isinstance(args[1], dict) and 'colorscheme' in args[1]:
            raise RuntimeError('surface render failed')


def test_render_pdb_groups_surface_blocks(monkeypatch):
    created = {}
    html_calls = []

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created['view'] = fake
        return fake

    monkeypatch.setattr(vis.py3Dmol, 'view', fake_view)
    monkeypatch.setattr(vis.components, 'html', lambda html, height: html_calls.append((html, height)))

    residue_colors = {}
    palette = ['#0a84ff', '#34c759', '#ff9500', '#8a2be2']
    for index in range(1, 33):
        residue_colors[('A', index)] = palette[(index - 1) // 8]

    vis.render_pdb(
        'ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND',
        style='surface',
        residue_colors=residue_colors,
        surface_colorize=True,
        surface_color='#d6e6ff',
    )

    fake_view = created['view']
    surface_calls = [call for call in fake_view.calls if call[0] == 'addSurface']
    map_calls = [call for call in fake_view.calls if call[0] == 'mapAtomProperties']

    assert len(surface_calls) == 1
    assert map_calls
    surface_style = surface_calls[0][1][1]
    assert surface_style['colorscheme']['prop'] == '_pi_surface_color'
    assert set(surface_style['colorscheme']['map'].keys()).issuperset(set(palette))
    assert html_calls


def test_render_pdb_caps_surface_assignments_for_large_color_sets(monkeypatch):
    created = {}
    html_calls = []

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created['view'] = fake
        return fake

    monkeypatch.setattr(vis.py3Dmol, 'view', fake_view)
    monkeypatch.setattr(vis.components, 'html', lambda html, height: html_calls.append((html, height)))

    residue_colors = {}
    for index in range(1, 181):
        red = (index * 3) % 256
        green = (index * 7) % 256
        blue = (index * 11) % 256
        residue_colors[('A', index)] = f'#{red:02x}{green:02x}{blue:02x}'

    vis.render_pdb(
        'ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND',
        style='surface',
        residue_colors=residue_colors,
        surface_colorize=True,
        surface_color='#d6e6ff',
    )

    fake_view = created['view']
    map_calls = [call for call in fake_view.calls if call[0] == 'mapAtomProperties']

    assert map_calls
    assert len(map_calls[0][1][0]) <= vis.MAX_SURFACE_ASSIGNMENTS
    assert html_calls


def test_render_pdb_falls_back_when_colored_surface_creation_fails(monkeypatch):
    created = {}
    html_calls = []

    def fake_view(width, height):
        fake = _FlakySurfaceView(width, height)
        created['view'] = fake
        return fake

    monkeypatch.setattr(vis.py3Dmol, 'view', fake_view)
    monkeypatch.setattr(vis.components, 'html', lambda html, height: html_calls.append((html, height)))

    residue_colors = {('A', 1): '#0a84ff', ('A', 2): '#34c759'}

    vis.render_pdb(
        'ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND',
        style='surface',
        residue_colors=residue_colors,
        surface_colorize=True,
        surface_color='#d6e6ff',
    )

    fake_view = created['view']
    surface_calls = [call for call in fake_view.calls if call[0] == 'addSurface']

    assert surface_calls
    assert any('colorscheme' in call[1][1] for call in surface_calls)
    assert any('color' in call[1][1] for call in surface_calls)
    assert html_calls