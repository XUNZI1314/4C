import pandas as pd

from protein_visualizer.services import external_sites


def test_fetch_uniprot_functional_sites_parses_feature_positions(monkeypatch):
    payload = {
        "uniProtkbId": "TEST_HUMAN",
        "features": [
            {
                "type": "Active site",
                "description": "Catalytic residue",
                "location": {"position": {"value": "123"}},
            },
            {
                "type": "Binding site",
                "description": "Ligand binding",
                "location": {"start": {"value": "200"}, "end": {"value": "201"}},
            },
            {
                "type": "Region",
                "description": "Should be ignored",
                "location": {"start": {"value": "1"}, "end": {"value": "10"}},
            },
        ],
    }

    monkeypatch.setattr(external_sites, "_fetch_json", lambda *_args, **_kwargs: payload)

    evidence_df, meta = external_sites.fetch_uniprot_functional_sites("P00001", chain_hint="A")

    assert not evidence_df.empty
    assert set(["chain", "resid", "evidence_source", "evidence_type", "evidence_score"]).issubset(evidence_df.columns)
    assert set(evidence_df["resid"].astype(int).tolist()) == {123, 200, 201}
    assert set(evidence_df["chain"].astype(str).tolist()) == {"A"}
    assert meta.get("status") == "ok"
    assert meta.get("accession") == "P00001"


def test_fetch_uniprot_functional_sites_returns_empty_when_unavailable(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise ValueError("network unavailable")

    monkeypatch.setattr(external_sites, "_fetch_json", _raise)

    evidence_df, meta = external_sites.fetch_uniprot_functional_sites("Q9TEST")

    assert isinstance(evidence_df, pd.DataFrame)
    assert evidence_df.empty
    assert meta.get("status") == "unavailable"


def test_extract_pdb_id_from_text_parses_header_code():
    pdb_text = """HEADER    TEST PDB                                  01-JAN-00   1ABC\nATOM      1  N   ALA A   1      11.111  12.222  13.333  1.00 10.00           N\n"""

    pdb_id = external_sites.extract_pdb_id_from_text(pdb_text)

    assert pdb_id == "1ABC"


def test_fetch_uniprot_functional_sites_for_structure_maps_with_sifts(monkeypatch):
    uniprot_payload = {
        "uniProtkbId": "TEST_HUMAN",
        "features": [
            {
                "type": "Active site",
                "description": "Catalytic residue",
                "location": {"position": {"value": "101"}},
            },
            {
                "type": "Binding site",
                "description": "Ligand binding",
                "location": {"position": {"value": "110"}},
            },
        ],
    }
    mapping_payload = {
        "1abc": {
            "UniProt": {
                "P00001": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 100,
                            "unp_end": 110,
                            "start": {"author_residue_number": 10},
                            "end": {"author_residue_number": 20},
                        }
                    ]
                }
            }
        }
    }

    def _fake_fetch(url, **_kwargs):
        if "rest.uniprot.org" in url:
            return uniprot_payload
        if "ebi.ac.uk/pdbe/api/mappings/uniprot" in url:
            return mapping_payload
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(external_sites, "_fetch_json", _fake_fetch)

    evidence_df, meta = external_sites.fetch_uniprot_functional_sites_for_structure(
        "P00001",
        chain_hint="A",
        pdb_id="1ABC",
    )

    assert not evidence_df.empty
    assert set(["mapping_level", "mapping_confidence", "uniprot_resid"]).issubset(evidence_df.columns)
    assert set(evidence_df["mapping_level"].astype(str).tolist()) == {"exact"}
    assert set(evidence_df["chain"].astype(str).tolist()) == {"A"}
    assert set(evidence_df["resid"].astype(int).tolist()) == {11, 20}
    assert meta.get("mapping_status") == "ok"
    assert meta.get("pdb_id") == "1ABC"
    assert int(meta.get("structure_verified_rows") or 0) >= 0


def test_fetch_uniprot_functional_sites_for_structure_prefers_chain_present_in_structure(monkeypatch):
    uniprot_payload = {
        "uniProtkbId": "TEST_HUMAN",
        "features": [
            {
                "type": "Active site",
                "description": "Catalytic residue",
                "location": {"position": {"value": "101"}},
            },
        ],
    }
    mapping_payload = {
        "1abc": {
            "UniProt": {
                "P00001": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 100,
                            "unp_end": 110,
                            "start": {"author_residue_number": 10},
                            "end": {"author_residue_number": 20},
                        },
                        {
                            "chain_id": "B",
                            "unp_start": 100,
                            "unp_end": 110,
                            "start": {"author_residue_number": 10},
                            "end": {"author_residue_number": 20},
                        },
                    ]
                }
            }
        }
    }
    pdb_text = """HEADER    TEST PDB                                  01-JAN-00   1ABC
ATOM      1  N   ALA B  11      11.111  12.222  13.333  1.00 10.00           N
ATOM      2  N   GLY B  12      12.111  12.222  13.333  1.00 10.00           N
"""

    def _fake_fetch(url, **_kwargs):
        if "rest.uniprot.org" in url:
            return uniprot_payload
        if "ebi.ac.uk/pdbe/api/mappings/uniprot" in url:
            return mapping_payload
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(external_sites, "_fetch_json", _fake_fetch)

    evidence_df, meta = external_sites.fetch_uniprot_functional_sites_for_structure(
        "P00001",
        pdb_id="1ABC",
        pdb_text=pdb_text,
    )

    assert not evidence_df.empty
    assert set(evidence_df["chain"].astype(str).tolist()) == {"B"}
    assert set(evidence_df["mapping_level"].astype(str).tolist()) == {"exact"}
    assert int(meta.get("structure_verified_rows") or 0) == 1


def test_fetch_uniprot_functional_sites_for_structure_downgrades_missing_author_residue(monkeypatch):
    uniprot_payload = {
        "uniProtkbId": "TEST_HUMAN",
        "features": [
            {
                "type": "Active site",
                "description": "Catalytic residue",
                "location": {"position": {"value": "102"}},
            },
        ],
    }
    mapping_payload = {
        "1abc": {
            "UniProt": {
                "P00001": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 100,
                            "unp_end": 104,
                            "start": {"author_residue_number": 10},
                            "end": {"author_residue_number": 14},
                        }
                    ]
                }
            }
        }
    }
    pdb_text = """HEADER    TEST PDB                                  01-JAN-00   1ABC
ATOM      1  N   ALA A  10      11.111  12.222  13.333  1.00 10.00           N
ATOM      2  N   GLY A  11      12.111  12.222  13.333  1.00 10.00           N
ATOM      3  N   SER A  13      13.111  12.222  13.333  1.00 10.00           N
ATOM      4  N   TYR A  14      14.111  12.222  13.333  1.00 10.00           N
"""

    def _fake_fetch(url, **_kwargs):
        if "rest.uniprot.org" in url:
            return uniprot_payload
        if "ebi.ac.uk/pdbe/api/mappings/uniprot" in url:
            return mapping_payload
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(external_sites, "_fetch_json", _fake_fetch)

    evidence_df, _meta = external_sites.fetch_uniprot_functional_sites_for_structure(
        "P00001",
        chain_hint="A",
        pdb_id="1ABC",
        pdb_text=pdb_text,
    )

    assert not evidence_df.empty
    row = evidence_df.iloc[0]
    assert int(row["resid"]) == 12
    assert str(row["mapping_level"]) == "weak"
    assert "gap-fallback" in str(row["mapping_method"])


def test_fetch_mcsa_catalytic_sites_parses_residue_entries(monkeypatch):
    payload = {
        "results": [
            {
                "mcsa_id": "MCSA-1",
                "uniprot_ac": "P00001",
                "ec": "3.2.1.4",
                "enzyme_name": "Example hydrolase",
                "residues": [
                    {
                        "residue_number": 101,
                        "chemical_function": "Proton donor",
                    },
                    {
                        "positions": [110, 111],
                        "role": "Metal ligand",
                    },
                ],
            }
        ]
    }

    monkeypatch.setattr(external_sites, "_fetch_json", lambda *_args, **_kwargs: payload)

    evidence_df, meta = external_sites.fetch_mcsa_catalytic_sites(
        "P00001",
        ec_number="3.2.1.4",
        chain_hint="A",
    )

    assert not evidence_df.empty
    assert set(evidence_df["resid"].astype(int).tolist()) == {101, 110, 111}
    assert set(evidence_df["chain"].astype(str).tolist()) == {"A"}
    assert set(evidence_df["evidence_source"].astype(str).tolist()) == {"M-CSA"}
    assert meta.get("status") == "ok"
    assert meta.get("accession") == "P00001"


def test_fetch_combined_functional_sites_for_structure_merges_sources(monkeypatch):
    uniprot_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 11,
                "evidence_source": "UniProt+SIFTS",
                "evidence_type": "Active site",
                "evidence_score": 0.96,
                "evidence_note": "UniProt",
                "uniprot_resid": 101,
                "mapping_level": "exact",
                "mapping_confidence": 0.95,
                "mapping_method": "sifts-chain-map",
            }
        ]
    )
    mcsa_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 11,
                "evidence_source": "M-CSA",
                "evidence_type": "Catalytic residue",
                "evidence_score": 1.0,
                "evidence_note": "M-CSA",
                "uniprot_resid": 101,
                "mapping_level": "exact",
                "mapping_confidence": 0.94,
                "mapping_method": "sifts-chain-map",
            },
            {
                "chain": "",
                "resid": 20,
                "evidence_source": "M-CSA",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.95,
                "evidence_note": "M-CSA weak",
                "uniprot_resid": 110,
                "mapping_level": "weak",
                "mapping_confidence": 0.32,
                "mapping_method": "mcsa-direct",
            },
        ]
    )

    monkeypatch.setattr(
        external_sites,
        "fetch_uniprot_functional_sites_for_structure",
        lambda *args, **kwargs: (uniprot_df, {"status": "ok", "mapping_status": "ok"}),
    )
    monkeypatch.setattr(
        external_sites,
        "fetch_mcsa_catalytic_sites_for_structure",
        lambda *args, **kwargs: (mcsa_df, {"status": "ok", "mapping_status": "ok"}),
    )

    evidence_df, meta = external_sites.fetch_combined_functional_sites_for_structure(
        "P00001",
        ec_number="3.2.1.4",
        pdb_id="1ABC",
        enable_uniprot=True,
        enable_mcsa=True,
    )

    assert not evidence_df.empty
    assert len(evidence_df) == 3
    assert meta.get("status") == "ok"
    assert "uniprot" in str(meta.get("sources"))
    assert "mcsa" in str(meta.get("sources"))
