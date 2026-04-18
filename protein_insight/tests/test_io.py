from pathlib import Path
import sys


def test_load_mmpbsa_csv_sample():
    # ensure package src is importable when running tests from project root
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / 'src'))

    from protein_insight.io import load_mmpbsa_csv

    sample = repo_root / 'data' / 'examples' / 'sample_mmpbsa.csv'
    assert sample.exists()
    df = load_mmpbsa_csv(sample)
    assert 'energy' in df.columns
    assert len(df) >= 1
    assert df['energy'].dtype.kind in 'fi'
