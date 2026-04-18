from protein_visualizer.services.p2rank import (
    parse_p2rank_predictions_csv,
    parse_p2rank_residues_csv,
)


def test_parse_p2rank_predictions_csv_extracts_pocket_rows():
    csv_text = """name,rank,score,probability,center_x,center_y,center_z,residues
Pocket-1,1,0.82,0.91,10.0,11.0,12.0,"ALA A 10; GLU A 12"
"""

    frame = parse_p2rank_predictions_csv(csv_text)

    assert not frame.empty
    assert frame.iloc[0]["pocket_label"] == "Pocket-1"
    assert int(frame.iloc[0]["pocket_rank"]) == 1
    assert float(frame.iloc[0]["pocket_probability"]) == 0.91
    assert "ALA A 10" in str(frame.iloc[0]["residue_list"])


def test_parse_p2rank_residues_csv_uses_residue_label_when_needed():
    csv_text = """pocket_rank,pocket_label,residue_label,score,probability
1,Pocket-1,"SER A 15",0.77,0.88
"""

    frame = parse_p2rank_residues_csv(csv_text)

    assert not frame.empty
    assert frame.iloc[0]["pocket_label"] == "Pocket-1"
    assert frame.iloc[0]["chain"] == "A"
    assert int(frame.iloc[0]["resid"]) == 15
    assert frame.iloc[0]["resname"] == "SER"
    assert float(frame.iloc[0]["residue_probability"]) == 0.88
