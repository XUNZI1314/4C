from pathlib import Path
import sys

import pandas as pd


def test_legacy_annotation_table_sample():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT, POCKET_TEXT
    from protein_visualizer.services.coloring import build_legacy_annotation_table, build_legacy_legend
    from protein_visualizer.services.energy import prepare_energy_table
    from protein_visualizer.services.hotspot import identify_hotspots
    from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms
    from protein_visualizer.services.pocket import parse_pocket_table

    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)
    hotspot_df = identify_hotspots(energy_table, energy_threshold=-5.0)
    pocket_df = parse_pocket_table(POCKET_TEXT)
    pocket_residues = [(row.chain, int(row.resid)) for row in pocket_df.itertuples(index=False)]

    annotated = build_legacy_annotation_table(
        energy_table,
        '按氨基酸理化性质',
        hotspot_df=hotspot_df,
        pocket_residues=pocket_residues,
    )

    assert not annotated.empty
    assert {
        'classification_label',
        'classification_color',
        'classification_description',
        'display_color',
        'hotspot_rank',
        'is_hotspot',
        'is_pocket',
    }.issubset(set(annotated.columns))

    ala = annotated.loc[annotated['resid'] == 1].iloc[0]
    assert ala['classification_label'] == 'ALA'
    assert bool(ala['is_hotspot']) is True
    assert bool(ala['is_pocket']) is True
    assert int(ala['hotspot_rank']) == 1

    pocket_mode = build_legacy_annotation_table(
        energy_table,
        '按口袋识别',
        hotspot_df=hotspot_df,
        pocket_residues=pocket_residues,
    )

    assert not pocket_mode.empty
    allowed_labels = {'口袋+热点重叠', '口袋残基', '热点残基', '背景残基'}
    assert set(pocket_mode['classification_label']).issubset(allowed_labels)

    overlap_row = pocket_mode.loc[pocket_mode['resid'] == 1].iloc[0]
    assert overlap_row['classification_label'] == '口袋+热点重叠'
    assert overlap_row['classification_color'] == '#ef4444'

    legend_items, legend_note = build_legacy_legend('按口袋识别', pocket_mode)
    assert legend_note
    assert any(item['label'] == '口袋+热点重叠' for item in legend_items)


def test_legacy_annotation_uses_fixed_classification_colors():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
    from protein_visualizer.services.coloring import build_legacy_annotation_table
    from protein_visualizer.services.energy import prepare_energy_table
    from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms

    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    energy_table = prepare_energy_table(atom_df, energy_df)

    fixed_a = build_legacy_annotation_table(
        energy_table,
        '按氨基酸理化性质',
        theme_name='清新海洋',
    )
    fixed_b = build_legacy_annotation_table(
        energy_table,
        '按氨基酸理化性质',
        theme_name='专业高对比',
    )

    ala_a = fixed_a.loc[fixed_a['resid'] == 1].iloc[0]
    ala_b = fixed_b.loc[fixed_b['resid'] == 1].iloc[0]
    assert ala_a['classification_label'] == 'ALA'
    assert ala_a['classification_color'] == ala_b['classification_color']


def test_legacy_annotation_precise_color_mapping_across_modes():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    from protein_visualizer.services.coloring import build_legacy_annotation_table

    table = pd.DataFrame(
        [
            {'chain': 'A', 'resid': 1, 'resname': 'ASP', 'delta_total': -6.0},
            {'chain': 'A', 'resid': 2, 'resname': 'LYS', 'delta_total': -2.0},
            {'chain': 'A', 'resid': 3, 'resname': 'SER', 'delta_total': 0.0},
            {'chain': 'A', 'resid': 4, 'resname': 'ALA', 'delta_total': 6.0},
        ]
    )

    hotspot_df = pd.DataFrame(
        [
            {'chain': 'A', 'resid': 1, 'hotspot_rank': 1},
            {'chain': 'A', 'resid': 2, 'hotspot_rank': 4},
            {'chain': 'A', 'resid': 3, 'hotspot_rank': 8},
        ]
    )

    pocket_residues = [('A', 1), ('A', 4)]

    property_mode = build_legacy_annotation_table(table, '按氨基酸理化性质')
    assert property_mode['classification_label'].tolist() == ['ASP', 'LYS', 'SER', 'ALA']
    assert len(set(property_mode['classification_color'])) == 4

    charge = build_legacy_annotation_table(table, '按电荷状态')
    assert charge.loc[charge['resid'] == 1, 'classification_color'].iloc[0] == '#ef4444'
    assert charge.loc[charge['resid'] == 2, 'classification_color'].iloc[0] == '#2563eb'
    assert charge.loc[charge['resid'] == 3, 'classification_color'].iloc[0] == '#d1d5db'

    polarity = build_legacy_annotation_table(table, '按侧链极性')
    assert polarity.loc[polarity['resid'] == 1, 'classification_color'].iloc[0] == '#f97316'
    assert polarity.loc[polarity['resid'] == 3, 'classification_color'].iloc[0] == '#06b6d4'
    assert polarity.loc[polarity['resid'] == 4, 'classification_color'].iloc[0] == '#10b981'

    energy = build_legacy_annotation_table(table, '按MMPBSA等级')
    assert energy.loc[energy['resid'] == 1, 'classification_color'].iloc[0] == '#2563eb'
    assert energy.loc[energy['resid'] == 3, 'classification_color'].iloc[0] == '#d1d5db'
    assert energy.loc[energy['resid'] == 4, 'classification_color'].iloc[0] == '#ef4444'

    hotspot = build_legacy_annotation_table(table, '按热点等级', hotspot_df=hotspot_df)
    assert hotspot.loc[hotspot['resid'] == 1, 'classification_color'].iloc[0] == '#ef4444'
    assert hotspot.loc[hotspot['resid'] == 2, 'classification_color'].iloc[0] == '#f97316'
    assert hotspot.loc[hotspot['resid'] == 3, 'classification_color'].iloc[0] == '#f59e0b'
    assert hotspot.loc[hotspot['resid'] == 4, 'classification_color'].iloc[0] == '#d1d5db'

    pocket = build_legacy_annotation_table(
        table,
        '按口袋识别',
        hotspot_df=hotspot_df,
        pocket_residues=pocket_residues,
    )
    assert pocket.loc[pocket['resid'] == 1, 'classification_color'].iloc[0] == '#ef4444'
    assert pocket.loc[pocket['resid'] == 2, 'classification_color'].iloc[0] == '#f97316'
    assert pocket.loc[pocket['resid'] == 4, 'classification_color'].iloc[0] == '#2563eb'