from pathlib import Path
import sys


def test_annotate_residue_table_sample():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    from protein_insight.analysis import detect_hotspots
    from protein_insight.coloring import annotate_residue_table
    from protein_insight.io import load_mmpbsa_csv
    from protein_insight.pocket import get_pocket_residues, load_pocket_json

    sample_mmpbsa = repo_root / 'data' / 'examples' / 'sample_mmpbsa.csv'
    sample_pocket = repo_root / 'data' / 'examples' / 'sample_pocket.json'

    df = load_mmpbsa_csv(sample_mmpbsa)
    hotspot_rank_map = {
        (item.chain, item.resid): index + 1
        for index, item in enumerate(detect_hotspots(df, top_k=1, energy_threshold=-2.0))
    }
    pocket = load_pocket_json(sample_pocket)
    pocket_residues = get_pocket_residues(pocket)

    annotated = annotate_residue_table(
        df,
        '按氨基酸理化性质',
        hotspot_rank_map=hotspot_rank_map,
        pocket_residues=pocket_residues,
    )

    assert not annotated.empty
    assert {
        'classification_label',
        'classification_color',
        'classification_description',
        'hotspot_rank',
        'is_hotspot',
        'is_pocket',
    }.issubset(set(annotated.columns))

    ala = annotated.loc[annotated['resid'] == 2].iloc[0]
    assert ala['classification_label'] == '疏水脂肪族（Ala/Val/Leu/Ile/Met）'
    assert bool(ala['is_hotspot']) is True
    assert bool(ala['is_pocket']) is True
    assert int(ala['hotspot_rank']) == 1


def test_build_surface_block_color_map_uses_multiple_pymol_style_blocks():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    import pandas as pd

    from protein_insight.coloring import build_surface_block_color_map

    table = pd.DataFrame(
        [
            {"chain": "A", "resid": index, "resname": "ALA", "energy": float(index)}
            for index in range(1, 33)
        ]
    )

    color_map = build_surface_block_color_map(table, palette_name="PyMOL 经典")

    assert len(color_map) == 32
    assert len(set(color_map.values())) >= 4
    assert color_map[("A", 1)] != color_map[("A", 9)]
    assert color_map[("A", 9)] != color_map[("A", 17)]


def test_annotate_residue_table_theme_switch_changes_colors():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    import pandas as pd

    from protein_insight.coloring import annotate_residue_table

    table = pd.DataFrame(
        [
            {"chain": "A", "resid": 1, "resname": "ASP", "energy": -1.2},
            {"chain": "A", "resid": 2, "resname": "LYS", "energy": 0.2},
        ]
    )

    ocean = annotate_residue_table(table, '按氨基酸理化性质', theme_name='清新海洋')
    morandi = annotate_residue_table(table, '按氨基酸理化性质', theme_name='柔和莫兰迪')

    assert ocean.loc[ocean['resid'] == 1].iloc[0]['classification_color'] != morandi.loc[morandi['resid'] == 1].iloc[0]['classification_color']


def test_cycle_palette_preserves_base_palette_and_generates_clean_variants():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    from protein_insight.pymol_colors import cycle_palette, get_palette

    base = get_palette('PyMOL 经典')
    palette = cycle_palette('PyMOL 经典', len(base) * 2)

    assert palette[: len(base)] == base
    assert len(set(palette)) > len(base)
    assert palette[len(base)] != base[0]
    assert all(color.startswith('#') and len(color) == 7 for color in palette)
