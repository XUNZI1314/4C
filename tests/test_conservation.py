import pandas as pd

from protein_visualizer.services.conservation import parse_conservation_evidence_table


def test_parse_conservation_evidence_table_normalizes_grade_scores():
    table_text = """chain,resid,grade,confidence,note
A,10,9,0.95,catalytic shell
A,12,5,0.70,near pocket rim
"""

    evidence_df, metadata = parse_conservation_evidence_table(
        table_text,
        source_hint="ConSurf",
    )

    assert metadata["status"] == "ok"
    assert metadata["source"] == "ConSurf"
    assert int(metadata["evidence_rows"]) == 2
    assert set(["chain", "resid", "evidence_source", "evidence_type", "evidence_score", "mapping_level"]).issubset(
        evidence_df.columns
    )
    assert set(evidence_df["evidence_source"].astype(str).tolist()) == {"ConSurf"}
    assert set(evidence_df["evidence_type"].astype(str).tolist()) == {"Conservation"}
    assert set(evidence_df["mapping_level"].astype(str).tolist()) == {"exact"}

    top_score = float(
        pd.to_numeric(
            evidence_df.loc[evidence_df["resid"].astype(int) == 10, "evidence_score"],
            errors="coerce",
        ).iloc[0]
    )
    lower_score = float(
        pd.to_numeric(
            evidence_df.loc[evidence_df["resid"].astype(int) == 12, "evidence_score"],
            errors="coerce",
        ).iloc[0]
    )
    assert top_score > lower_score


def test_parse_conservation_evidence_table_inverts_rate_scores_and_uses_chain_hint():
    table_text = """position,rate4site
10,-1.8
25,0.6
"""

    evidence_df, metadata = parse_conservation_evidence_table(
        table_text,
        chain_hint="B",
        source_hint="Rate4Site",
    )

    assert metadata["status"] == "ok"
    assert int(metadata["exact_rows"]) == 2
    assert set(evidence_df["chain"].astype(str).tolist()) == {"B"}

    top_score = float(
        pd.to_numeric(
            evidence_df.loc[evidence_df["resid"].astype(int) == 10, "evidence_score"],
            errors="coerce",
        ).iloc[0]
    )
    lower_score = float(
        pd.to_numeric(
            evidence_df.loc[evidence_df["resid"].astype(int) == 25, "evidence_score"],
            errors="coerce",
        ).iloc[0]
    )
    assert top_score > lower_score
