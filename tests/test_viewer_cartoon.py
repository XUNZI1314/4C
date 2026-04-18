from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_visualizer.services import viewer


class _FakeView:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.calls = []

    def addModel(self, *args, **kwargs):
        self.calls.append(("addModel", args, kwargs))

    def setBackgroundColor(self, color):
        self.calls.append(("setBackgroundColor", (color,), {}))

    def setStyle(self, selection, style):
        self.calls.append(("setStyle", (selection, style), {}))

    def addStyle(self, selection, style):
        self.calls.append(("addStyle", (selection, style), {}))

    def mapAtomProperties(self, *args, **kwargs):
        self.calls.append(("mapAtomProperties", args, kwargs))

    def addSurface(self, *args, **kwargs):
        self.calls.append(("addSurface", args, kwargs))

    def zoomTo(self, *args, **kwargs):
        self.calls.append(("zoomTo", args, kwargs))

    def spin(self, value):
        self.calls.append(("spin", (value,), {}))

    def _make_html(self):
        return "<div/>"


def _surface_calls(fake_view):
    return [call for call in fake_view.calls if call[0] == "addSurface"]


def _surface_style(fake_view):
    surface_calls = _surface_calls(fake_view)
    assert surface_calls
    return surface_calls[0][1][1]


def _surface_assignments(fake_view):
    map_calls = [call for call in fake_view.calls if call[0] == "mapAtomProperties"]
    assert map_calls
    return map_calls[0][1][0]


def test_cartoon_mode_does_not_add_sticks(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "heat_color": "#ff0000", "display_color": "#00ff00", "delta_total": -2.0},
            {"chain": "A", "resid": 2, "heat_color": "#0000ff", "display_color": "#0000ff", "delta_total": -1.5},
        ]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=1.0,
        display_mode="cartoon",
        show_backbone=True,
        opacity=0.5,
        selected_chain="A",
        selected_resid=1,
    )

    fake_view = created["view"]
    add_style_calls = [call for call in fake_view.calls if call[0] == "addStyle"]

    assert add_style_calls, "cartoon mode should still highlight the selected residue"
    for _, (_, style), _ in add_style_calls:
        assert "stick" not in style

    cartoon_calls = [call for call in fake_view.calls if call[0] == "setStyle"]
    assert any(style.get("cartoon", {}).get("color") == "#00ff00" for _, (_, style), _ in cartoon_calls)


def test_cartoon_mode_does_not_highlight_by_default(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "heat_color": "#ff0000", "display_color": "#00ff00", "delta_total": -2.0},
            {"chain": "A", "resid": 2, "heat_color": "#0000ff", "display_color": "#0000ff", "delta_total": -1.5},
        ]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=1.0,
        display_mode="cartoon",
        show_backbone=True,
        opacity=0.5,
        selected_chain=None,
        selected_resid=None,
    )

    fake_view = created["view"]
    add_style_calls = [call for call in fake_view.calls if call[0] == "addStyle"]

    assert not add_style_calls


def test_sticks_mode_keeps_classification_color_when_not_heat_mode(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "display_color": "#123456", "delta_total": 0.2},
        ]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=5.0,
        display_mode="sticks",
        show_backbone=False,
        opacity=0.5,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    add_style_calls = [call for call in fake_view.calls if call[0] == "addStyle"]

    assert add_style_calls
    first_style = add_style_calls[0][1][1]
    assert first_style["stick"]["color"] == "#123456"


def test_surface_mode_adds_colored_surfaces(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "display_color": "#112233", "delta_total": 0.1},
            {"chain": "A", "resid": 2, "display_color": "#445566", "delta_total": 0.1},
        ]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=8.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)
    surface_colors = set(surface_style["colorscheme"]["map"].keys())
    non_base_colors = {color for color in surface_colors if color != viewer.SETTINGS.neutral_color}

    assert len(_surface_calls(fake_view)) == 1
    assert surface_style["colorscheme"]["prop"] == "_pi_surface_color"
    assert len(non_base_colors) >= 2
    assert any(entry.get("props", {}).get("_pi_surface_color") in non_base_colors for entry in assignments)


def test_surface_mode_uses_dynamic_opacity_scaling(monkeypatch):
    def render_surface(energy_table, pdb_text):
        created = {}

        def fake_view(width, height):
            fake = _FakeView(width, height)
            created["view"] = fake
            return fake

        monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

        viewer.build_view(
            pdb_text=pdb_text,
            energy_table=energy_table,
            threshold=8.0,
            display_mode="surface",
            show_backbone=False,
            opacity=0.0,
            selected_chain=None,
            selected_resid=None,
            color_mode="按口袋识别",
        )

        return _surface_style(created["view"])["opacity"]

    small_surface_opacity = render_surface(
        pd.DataFrame(
            [
                {"chain": "A", "resid": 1, "display_color": "#112233", "delta_total": 0.1},
                {"chain": "A", "resid": 2, "display_color": "#445566", "delta_total": 0.1},
            ]
        ),
        "ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
    )

    large_rows = [
        {"chain": "A", "resid": index, "display_color": "#112233", "delta_total": 0.1}
        for index in range(1, 601)
    ]
    large_pdb_text = "\n".join(
        [
            f"ATOM  {index:5d}  CA  ALA A{index:4d}    {float(index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            for index in range(1, 701)
        ]
    ) + "\nEND"
    large_surface_opacity = render_surface(pd.DataFrame(large_rows), large_pdb_text)

    single_color_opacity = viewer._resolve_surface_opacity(
        0.5,
        120,
        surface_single_color=True,
        color_mode="按口袋识别",
    )
    multi_color_opacity = viewer._resolve_surface_opacity(
        0.5,
        120,
        surface_single_color=False,
        color_mode="按口袋识别",
    )

    assert small_surface_opacity < 0.58
    assert large_surface_opacity > small_surface_opacity
    assert single_color_opacity > multi_color_opacity


def test_surface_mode_single_color_uses_uniform_surface(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "display_color": "#112233", "delta_total": 0.1},
            {"chain": "A", "resid": 2, "display_color": "#445566", "delta_total": 0.1},
        ]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=8.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按DELTA TOTAL 热度",
        surface_single_color=True,
        surface_uniform_color="#d1d5db",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)

    assert len(_surface_calls(fake_view)) == 1
    assert set(surface_style["colorscheme"]["map"].keys()) == {"#d1d5db"}
    assert len(assignments) == 1
    assert assignments[0]["props"]["_pi_surface_color"] == "#d1d5db"


def test_surface_mode_pocket_mode_keeps_neutral_background_color(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [{"chain": "A", "resid": idx, "display_color": "#c7c7c7", "delta_total": 0.1} for idx in range(1, 121)]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=8.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)
    surface_colors = set(surface_style["colorscheme"]["map"].keys())

    assert len(_surface_calls(fake_view)) == 1
    assert "#c7c7c7" in surface_colors
    assert any(entry.get("props", {}).get("_pi_surface_color") == "#c7c7c7" for entry in assignments)
    assert any("chain" in entry for entry in assignments)
    assert any("resi" in entry for entry in assignments)


def test_surface_mode_preserves_rich_heat_color_groups(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    rows = []
    for index in range(1, 81):
        red = (index * 3) % 256
        green = (index * 7) % 256
        blue = (index * 11) % 256
        rows.append(
            {
                "chain": "A",
                "resid": index,
                "display_color": f"#{red:02x}{green:02x}{blue:02x}",
                "delta_total": float(index),
            }
        )
    energy_table = pd.DataFrame(rows)

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=0.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按DELTA TOTAL 热度",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)

    assert len(_surface_calls(fake_view)) == 1
    assert len(assignments) >= 40
    assert len(surface_style["colorscheme"]["map"]) >= 30
    assert any("chain" in entry for entry in assignments)
    assert any("resi" in entry for entry in assignments)


def test_surface_mode_uses_coarse_color_groups_for_large_structure(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    rows = []
    for index in range(1, 601):
        rows.append(
            {
                "chain": "A",
                "resid": index,
                "display_color": "#123456",
                "delta_total": -1.0,
            }
        )
    energy_table = pd.DataFrame(rows)

    pdb_lines = [
        f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    {float(idx):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        for idx in range(1, 701)
    ]
    pdb_text = "\n".join(pdb_lines) + "\nEND"

    viewer.build_view(
        pdb_text=pdb_text,
        energy_table=energy_table,
        threshold=0.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)

    assert len(_surface_calls(fake_view)) == 1
    assert len(assignments) == len(energy_table) + 1
    assert "#123456" in surface_style["colorscheme"]["map"]
    assert any("chain" in entry for entry in assignments)
    assert any("resi" in entry for entry in assignments)


def test_surface_mode_large_structure_preserves_input_category_colors(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    rows = []
    for index in range(1, 601):
        rows.append(
            {
                "chain": "A",
                "resid": index,
                "display_color": "#c7c7c7" if index <= 500 else "#ff6a00",
                "delta_total": -1.0,
            }
        )
    energy_table = pd.DataFrame(rows)

    pdb_lines = [
        f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    {float(idx):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        for idx in range(1, 701)
    ]
    pdb_text = "\n".join(pdb_lines) + "\nEND"

    viewer.build_view(
        pdb_text=pdb_text,
        energy_table=energy_table,
        threshold=0.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)
    assigned_colors = [entry.get("props", {}).get("_pi_surface_color") for entry in assignments if entry.get("props")]

    assert len(_surface_calls(fake_view)) == 1
    assert "#c7c7c7" in surface_style["colorscheme"]["map"]
    assert "#ff6a00" in surface_style["colorscheme"]["map"]
    assert "#c7c7c7" in assigned_colors
    assert "#ff6a00" in assigned_colors
    assert any("resi" in entry for entry in assignments)


def test_surface_mode_large_structure_all_neutral_keeps_single_category_color(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    rows = []
    for index in range(1, 601):
        rows.append(
            {
                "chain": "A",
                "resid": index,
                "display_color": "#c7c7c7",
                "delta_total": -0.2,
            }
        )
    energy_table = pd.DataFrame(rows)

    pdb_lines = [
        f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    {float(idx):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
        for idx in range(1, 701)
    ]
    pdb_text = "\n".join(pdb_lines) + "\nEND"

    viewer.build_view(
        pdb_text=pdb_text,
        energy_table=energy_table,
        threshold=0.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    surface_style = _surface_style(fake_view)
    assignments = _surface_assignments(fake_view)
    surface_colors = set(surface_style["colorscheme"]["map"].keys())

    assert len(_surface_calls(fake_view)) == 1
    assert "#c7c7c7" in surface_colors
    assert len(surface_colors) <= 3
    assert any("chain" in entry for entry in assignments)
    assert any("resi" in entry for entry in assignments)


def test_surface_mode_does_not_force_cartoon_underlay(monkeypatch):
    created = {}

    def fake_view(width, height):
        fake = _FakeView(width, height)
        created["view"] = fake
        return fake

    monkeypatch.setattr(viewer.py3Dmol, "view", fake_view)

    energy_table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "display_color": "#112233", "delta_total": -1.0},
        ]
    )

    viewer.build_view(
        pdb_text="ATOM      1  CA  ALA A   1      0.0 0.0 0.0  1.00 20.00           C\nEND",
        energy_table=energy_table,
        threshold=0.0,
        display_mode="surface",
        show_backbone=False,
        opacity=0.6,
        selected_chain=None,
        selected_resid=None,
        color_mode="按口袋识别",
    )

    fake_view = created["view"]
    set_style_calls = [call for call in fake_view.calls if call[0] == "setStyle"]

    assert set_style_calls
    # surface 模式基底应为清空样式，不应强制加卡通底层条带
    assert set_style_calls[0][1][1] == {}
