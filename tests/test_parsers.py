from protein_visualizer.sample_data import MMPBSA_TEXT, PDB_TEXT
from protein_visualizer.services.parsers import parse_mmpbsa_delta_total, parse_pdb_atoms


def test_parse_pdb_atoms_returns_rows():
    atom_df = parse_pdb_atoms(PDB_TEXT)
    assert not atom_df.empty
    assert {"chain", "resid", "resname", "atom_name", "atom_type"}.issubset(atom_df.columns)


def test_parse_mmpbsa_delta_total_returns_rows():
    energy_df = parse_mmpbsa_delta_total(MMPBSA_TEXT)
    assert not energy_df.empty
    assert {"chain", "resid", "resname", "delta_total"}.issubset(energy_df.columns)
    assert len(energy_df) == 5
