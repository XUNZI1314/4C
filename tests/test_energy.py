from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
from protein_visualizer.services.energy import prepare_energy_table
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms


def test_prepare_energy_table_contains_heat_color():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    table = prepare_energy_table(atom_df, energy_df)

    assert not table.empty
    assert "heat_color" in table.columns
    assert table["heat_color"].str.startswith("#").all()
