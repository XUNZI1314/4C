import pandas as pd

from protein_visualizer.services import literature_sites
from protein_visualizer.services.literature_sites import (
    build_literature_query,
    extract_literature_residue_evidence,
    fetch_europepmc_literature_evidence,
    fetch_literature_residue_evidence_for_structure,
    fetch_pubmed_literature_evidence,
    merge_literature_evidence_tables,
    remove_literature_evidence,
)


def test_extract_literature_residue_evidence_finds_catalytic_triad():
    text = (
        "Kinetic analysis showed that the active site catalytic triad Ser195, "
        "His57 and Asp102 is required for catalysis and substrate binding."
    )

    evidence_df, metadata = extract_literature_residue_evidence(
        text,
        chain_hint="A",
        article_id="PMID:1",
        article_title="Catalytic residues",
    )

    assert metadata["status"] == "ok"
    assert set(evidence_df["resid"].astype(int).tolist()) == {57, 102, 195}
    assert evidence_df["evidence_type"].eq("Catalytic residue").all()
    assert evidence_df["evidence_score"].astype(float).ge(0.8).all()
    assert evidence_df["mapping_level"].eq("weak").all()


def test_extract_literature_residue_evidence_requires_functional_context():
    weak_text = "The sequence alignment contains Asp123 and His124 in the supplementary table."
    strong_text = "The D123A mutation abolished activity, indicating that Asp123 is catalytic."

    weak_df, _ = extract_literature_residue_evidence(weak_text, chain_hint="A")
    strong_df, _ = extract_literature_residue_evidence(strong_text, chain_hint="A")

    assert weak_df.empty
    assert not strong_df.empty
    assert 123 in set(strong_df["resid"].astype(int).tolist())
    assert strong_df["evidence_type"].isin({"Catalytic residue", "Activity-loss mutagenesis"}).any()


def test_merge_literature_evidence_tables_boosts_cross_article_support():
    first_df, _ = extract_literature_residue_evidence(
        "The active site catalytic residue Ala1 is required for enzyme activity.",
        chain_hint="A",
        source_label="Literature-PubMed",
        article_id="PMID:1",
        article_title="Catalytic alanine evidence",
    )
    second_df, _ = extract_literature_residue_evidence(
        "Mutation of Ala1 to Gly abolished activity in the enzyme.",
        chain_hint="A",
        source_label="Literature-EuropePMC",
        article_id="PMC:2",
        article_title="Mutagenesis confirms alanine",
    )

    merged = merge_literature_evidence_tables(first_df, second_df)

    assert not merged.empty
    residue_rows = merged[merged["resid"].astype(int) == 1]
    assert len(residue_rows) == 2
    assert residue_rows["evidence_note"].astype(str).str.contains("literature_support=2 articles/2 sources").all()
    assert residue_rows["evidence_score"].astype(float).max() > first_df["evidence_score"].astype(float).max()
    assert merged.attrs["literature_support"]["replicated_residue_groups"] == "1"
    assert merged.attrs["literature_support"]["max_article_support"] == "2"


def test_fetch_pubmed_literature_evidence_extracts_from_mocked_abstract(monkeypatch):
    search_payload = {"esearchresult": {"idlist": ["123456"]}}
    xml_payload = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123456</PMID>
          <Article>
            <ArticleTitle>Enzyme catalytic residues</ArticleTitle>
            <Abstract>
              <AbstractText>The active site residue Glu35 acts as a general acid.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    monkeypatch.setattr(literature_sites, "_fetch_json", lambda *_args, **_kwargs: search_payload)
    monkeypatch.setattr(literature_sites, "_fetch_text", lambda *_args, **_kwargs: xml_payload)

    evidence_df, metadata = fetch_pubmed_literature_evidence(
        "P00001 catalytic residue",
        chain_hint="A",
        max_articles=2,
    )

    assert metadata["status"] == "ok"
    assert metadata["article_count"] == "1"
    assert not evidence_df.empty
    assert int(evidence_df.iloc[0]["resid"]) == 35
    assert "PMID:123456" in str(evidence_df.iloc[0]["evidence_note"])


def test_fetch_europepmc_literature_evidence_extracts_abstract_and_open_fulltext(monkeypatch):
    search_payload = {
        "resultList": {
            "result": [
                {
                    "id": "123456",
                    "source": "MED",
                    "pmid": "123456",
                    "title": "Open enzyme article",
                    "abstractText": "The active site residue Glu35 acts as a general acid.",
                    "fullTextIdList": {"fullTextId": ["PMC123456"]},
                }
            ]
        }
    }
    fulltext_xml = """
    <article>
      <body>
        <p>Mutation of D52A abolished activity, indicating that Asp52 is catalytic.</p>
      </body>
    </article>
    """

    monkeypatch.setattr(literature_sites, "_fetch_json", lambda *_args, **_kwargs: search_payload)
    monkeypatch.setattr(literature_sites, "_fetch_text", lambda *_args, **_kwargs: fulltext_xml)

    evidence_df, metadata = fetch_europepmc_literature_evidence(
        "P00001 catalytic residue",
        chain_hint="A",
        max_articles=2,
        include_open_fulltext=True,
        max_fulltext_articles=1,
    )

    assert metadata["status"] == "ok"
    assert metadata["article_count"] == "1"
    assert metadata["fulltext_count"] == "1"
    assert {35, 52}.issubset(set(evidence_df["resid"].astype(int).tolist()))
    assert evidence_df["evidence_source"].astype(str).str.contains("EuropePMC").any()


def test_fetch_literature_residue_evidence_for_structure_maps_when_alignment_available(monkeypatch):
    manual_text = "The active site residue Asp10 is essential for catalysis."
    mapped_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 101,
                "evidence_source": "UniProt+SIFTS",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.95,
                "evidence_note": "mapped",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.91,
                "mapping_method": "sifts-structure-verified",
            }
        ]
    )

    monkeypatch.setattr(
        literature_sites,
        "_map_uniprot_sites_to_structure",
        lambda *_args, **_kwargs: (mapped_df, {"mapping_status": "ok", "exact_rows": "1", "weak_rows": "0"}),
    )

    evidence_df, metadata = fetch_literature_residue_evidence_for_structure(
        manual_text=manual_text,
        accession="P00001",
        pdb_id="1ABC",
        chain_hint="A",
        enable_pubmed=False,
        enable_europepmc=False,
    )

    assert metadata["status"] == "ok"
    assert int(evidence_df.iloc[0]["resid"]) == 101
    assert str(evidence_df.iloc[0]["evidence_source"]) == "Literature+SIFTS"
    assert str(evidence_df.iloc[0]["mapping_method"]).startswith("literature-")


def test_fetch_literature_residue_evidence_can_assume_structure_numbering():
    evidence_df, metadata = fetch_literature_residue_evidence_for_structure(
        manual_text="The catalytic residue Asp10 is required for activity.",
        chain_hint="B",
        enable_pubmed=False,
        enable_europepmc=False,
        assume_structure_numbering=True,
    )

    assert metadata["mapping"]["mapping_status"] == "assumed-structure-numbering"
    assert not evidence_df.empty
    assert str(evidence_df.iloc[0]["chain"]) == "B"
    assert str(evidence_df.iloc[0]["mapping_level"]) == "exact"
    assert float(evidence_df.iloc[0]["mapping_confidence"]) >= 0.78


def test_assumed_structure_numbering_verifies_matching_residue_identity():
    pdb_text = """HEADER    TEST PDB                                  01-JAN-00   1ABC
ATOM      1  N   ASP B  10      11.111  12.222  13.333  1.00 10.00           N
ATOM      2  CA  ASP B  10      12.111  12.222  13.333  1.00 10.00           C
"""

    evidence_df, metadata = fetch_literature_residue_evidence_for_structure(
        manual_text="The catalytic residue Asp10 is required for activity.",
        chain_hint="B",
        pdb_text=pdb_text,
        enable_pubmed=False,
        enable_europepmc=False,
        assume_structure_numbering=True,
    )

    assert metadata["mapping"]["identity_checked_rows"] == "1"
    assert metadata["mapping"]["identity_matched_rows"] == "1"
    assert not evidence_df.empty
    assert str(evidence_df.iloc[0]["mapping_level"]) == "exact"
    assert str(evidence_df.iloc[0]["mapping_method"]) == "literature-structure-numbering-verified"
    assert float(evidence_df.iloc[0]["mapping_confidence"]) >= 0.88
    assert "structure_residue_match=ASP" in str(evidence_df.iloc[0]["evidence_note"])


def test_assumed_structure_numbering_downgrades_residue_identity_mismatch():
    pdb_text = """HEADER    TEST PDB                                  01-JAN-00   1ABC
ATOM      1  N   ALA B  10      11.111  12.222  13.333  1.00 10.00           N
ATOM      2  CA  ALA B  10      12.111  12.222  13.333  1.00 10.00           C
"""

    evidence_df, metadata = fetch_literature_residue_evidence_for_structure(
        manual_text="The catalytic residue Asp10 is required for activity.",
        chain_hint="B",
        pdb_text=pdb_text,
        enable_pubmed=False,
        enable_europepmc=False,
        assume_structure_numbering=True,
    )

    assert metadata["mapping"]["identity_checked_rows"] == "1"
    assert metadata["mapping"]["identity_mismatched_rows"] == "1"
    assert metadata["mapping"]["exact_rows"] == "0"
    assert metadata["mapping"]["weak_rows"] == "1"
    assert not evidence_df.empty
    assert str(evidence_df.iloc[0]["mapping_level"]) == "weak"
    assert str(evidence_df.iloc[0]["mapping_method"]) == "literature-residue-identity-mismatch"
    assert float(evidence_df.iloc[0]["mapping_confidence"]) <= 0.28
    assert float(evidence_df.iloc[0]["evidence_score"]) <= 0.52
    assert "structure_residue_mismatch=ASP!=ALA" in str(evidence_df.iloc[0]["evidence_note"])


def test_fetch_literature_residue_evidence_for_structure_can_use_europepmc(monkeypatch):
    europepmc_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 35,
                "evidence_source": "Literature-EuropePMC",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.92,
                "evidence_note": "Europe PMC",
                "uniprot_resid": 35,
                "mapping_level": "weak",
                "mapping_confidence": 0.46,
                "mapping_method": "literature-text-mining",
            }
        ]
    )

    monkeypatch.setattr(
        literature_sites,
        "fetch_europepmc_literature_evidence",
        lambda *_args, **_kwargs: (europepmc_df, {"status": "ok", "article_count": "1", "fulltext_count": "1"}),
    )

    evidence_df, metadata = fetch_literature_residue_evidence_for_structure(
        query="enzyme active site",
        chain_hint="A",
        enable_pubmed=False,
        enable_europepmc=True,
        assume_structure_numbering=True,
    )

    assert metadata["europepmc"]["status"] == "ok"
    assert not evidence_df.empty
    assert int(evidence_df.iloc[0]["resid"]) == 35
    assert str(evidence_df.iloc[0]["mapping_level"]) == "exact"


def test_build_literature_query_includes_enzyme_terms():
    query = build_literature_query(accession="P00001", ec_number="3.2.1.4", pdb_id="1ABC")

    assert "P00001" in query
    assert "1ABC" in query
    assert "3.2.1.4" in query
    assert "active site" in query


def test_remove_literature_evidence_keeps_non_literature_sources():
    evidence_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "Literature+SIFTS",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.9,
                "evidence_note": "PMID:1",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.9,
                "mapping_method": "literature-sifts-structure-verified",
            },
            {
                "chain": "A",
                "resid": 12,
                "evidence_source": "M-CSA",
                "evidence_type": "Catalytic residue",
                "evidence_score": 1.0,
                "evidence_note": "MCSA",
                "uniprot_resid": 12,
                "mapping_level": "exact",
                "mapping_confidence": 0.92,
                "mapping_method": "sifts-structure-verified",
            },
        ]
    )

    filtered = remove_literature_evidence(evidence_df)

    assert len(filtered) == 1
    assert str(filtered.iloc[0]["evidence_source"]) == "M-CSA"
