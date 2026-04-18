"""Smoke test for the native PyMOL backend.

Run this script with the PyMOL Python interpreter if you want to verify that
the API is working end-to-end and produce a PNG on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_insight.pymol_backend import can_use_pymol, render_pdb_png


def main() -> int:
    if not can_use_pymol():
        print("PyMOL is not available in this environment.")
        return 1

    sample_pdb = ROOT_DIR / "data" / "examples" / "sample1.pdb"
    output_png = ROOT_DIR / "data" / "examples" / "pymol_smoke_test.png"

    if not sample_pdb.exists():
        print(f"Sample PDB not found: {sample_pdb}")
        return 1

    pdb_text = sample_pdb.read_text(encoding="utf-8")
    png_bytes = render_pdb_png(
        pdb_text,
        style="cartoon",
        cartoon_theme="PyMOL 风格",
        residue_colors={("A", 1): "tv_orange", ("A", 2): "tv_blue", ("A", 3): "tv_green"},
        highlight=("A", 2),
        width=900,
        height=700,
        surface_color="lightblue",
        background_color="white",
    )
    output_png.write_bytes(png_bytes)
    print(f"PyMOL smoke test saved: {output_png}")
    print(f"PNG bytes: {len(png_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())