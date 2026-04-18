import json
import hashlib
import zipfile
from io import BytesIO

import pandas as pd

from protein_visualizer.services import ai_evidence
from protein_visualizer.services.ai_evidence import (
    apply_ai_review_decisions_to_audit,
    build_ai_evidence_audit_table,
    build_ai_followup_evidence_plan,
    build_ai_followup_prompt_bundle,
    build_ai_evidence_prompt,
    build_ai_evidence_review_queue,
    build_ai_review_checklist_markdown,
    build_ai_review_decision_template,
    build_ai_review_decision_outcome_table,
    build_ai_review_decision_validation_table,
    build_ai_review_artifact_manifest,
    build_ai_review_artifact_bundle_zip,
    build_ai_review_ranking_delta,
    build_ai_review_bundle_certificate_markdown,
    build_ai_review_bundle_readme_markdown,
    build_ai_review_round_report_markdown,
    build_ai_review_round_summary,
    build_ai_ranking_impact_summary,
    build_residue_evidence_consensus,
    build_ai_triage_context,
    fetch_ai_residue_evidence,
    filter_ai_evidence_for_ranking,
    parse_ai_review_decision_table,
    parse_ai_residue_evidence_payload,
    build_ai_review_bundle_verification_summary,
    verify_ai_review_artifact_bundle_zip,
)


def test_parse_ai_residue_evidence_payload_builds_external_evidence_rows():
    payload = {
        "residues": [
            {
                "resname": "Ser",
                "position_text": "Ser195",
                "uniprot_position": 195,
                "evidence_type": "Catalytic residue",
                "confidence": 0.91,
                "pmid": "123456",
                "source_title": "Catalytic triad paper",
                "evidence_snippet": "The catalytic triad Ser195, His57 and Asp102 is essential.",
                "requires_manual_review": False,
            },
            {
                "resname": "His",
                "position_text": "His57",
                "confidence": 0.2,
                "pmid": "123456",
                "evidence_snippet": "The catalytic triad includes His57.",
            },
        ]
    }

    evidence_df, metadata = parse_ai_residue_evidence_payload(
        json.dumps(payload),
        chain_hint="A",
        min_confidence=0.35,
    )

    assert metadata["status"] == "ok"
    assert metadata["parsed_records"] == "2"
    assert metadata["skipped_low_confidence"] == "1"
    assert len(evidence_df) == 1
    assert int(evidence_df.iloc[0]["resid"]) == 195
    assert str(evidence_df.iloc[0]["chain"]) == "A"
    assert str(evidence_df.iloc[0]["mapping_level"]) == "weak"
    assert "pmid=123456" in str(evidence_df.iloc[0]["evidence_note"])
    assert "ai_confidence=0.910" in str(evidence_df.iloc[0]["evidence_note"])


def test_parse_ai_residue_evidence_requires_source_snippet_or_caps_as_review():
    payload = {
        "residues": [
            {
                "resname": "Asp",
                "position_text": "Asp102",
                "confidence": 0.95,
                "evidence_type": "Catalytic residue",
            }
        ]
    }

    evidence_df, metadata = parse_ai_residue_evidence_payload(json.dumps(payload), chain_hint="A")

    assert metadata["manual_review_rows"] == "1"
    assert float(evidence_df.iloc[0]["evidence_score"]) <= 0.62
    assert str(evidence_df.iloc[0]["mapping_method"]) == "ai-literature-extraction-review"
    assert "manual_review=true" in str(evidence_df.iloc[0]["evidence_note"])


def test_parse_ai_residue_evidence_verifies_assumed_structure_numbering():
    pdb_text = """HEADER    TEST PDB                                  01-JAN-00   1ABC
ATOM      1  N   SER A 195      11.111  12.222  13.333  1.00 10.00           N
ATOM      2  CA  SER A 195      12.111  12.222  13.333  1.00 10.00           C
ATOM      3  N   ALA A 102      11.111  12.222  13.333  1.00 10.00           N
ATOM      4  CA  ALA A 102      12.111  12.222  13.333  1.00 10.00           C
"""
    payload = {
        "residues": [
            {
                "resname": "Ser",
                "position_text": "Ser195",
                "confidence": 0.9,
                "pmid": "1",
                "source_title": "Verified",
                "evidence_snippet": "Ser195 is catalytic.",
            },
            {
                "resname": "Asp",
                "position_text": "Asp102",
                "confidence": 0.9,
                "pmid": "1",
                "source_title": "Mismatch",
                "evidence_snippet": "Asp102 is catalytic.",
            },
        ]
    }

    evidence_df, metadata = parse_ai_residue_evidence_payload(
        json.dumps(payload),
        chain_hint="A",
        assume_structure_numbering=True,
        pdb_text=pdb_text,
    )

    by_resid = evidence_df.set_index("resid")
    assert metadata["identity_checked_rows"] == "2"
    assert metadata["identity_matched_rows"] == "1"
    assert metadata["identity_mismatched_rows"] == "1"
    assert by_resid.loc[195, "mapping_level"] == "exact"
    assert by_resid.loc[195, "mapping_method"] == "ai-structure-numbering-verified"
    assert by_resid.loc[102, "mapping_level"] == "weak"
    assert by_resid.loc[102, "mapping_method"] == "ai-structure-identity-mismatch"
    assert float(by_resid.loc[102, "evidence_score"]) <= 0.48


def test_fetch_ai_residue_evidence_uses_openai_compatible_response(monkeypatch):
    response_text = json.dumps(
        {
            "residues": [
                {
                    "resname": "Glu",
                    "position_text": "Glu35",
                    "confidence": 0.88,
                    "pmid": "9",
                    "source_title": "AI extracted",
                    "evidence_snippet": "Glu35 acts as the general acid.",
                    "evidence_type": "Catalytic residue",
                }
            ]
        }
    )

    def fake_post_json(url, payload, *, api_key="", timeout_sec=30.0):
        assert url == "https://example.test/v1/chat/completions"
        assert payload["model"] == "test-model"
        assert "strict JSON" in payload["messages"][0]["content"]
        return {"choices": [{"message": {"content": response_text}}]}

    monkeypatch.setattr(ai_evidence, "_post_json", fake_post_json)

    evidence_df, metadata = fetch_ai_residue_evidence(
        "Glu35 acts as the general acid.",
        api_url="https://example.test/v1/chat/completions",
        api_key="secret",
        model="test-model",
        chain_hint="A",
    )

    assert metadata["status"] == "ok"
    assert metadata["model"] == "test-model"
    assert int(evidence_df.iloc[0]["resid"]) == 35


def test_build_ai_prompt_and_triage_context_are_structured():
    decision_df = pd.DataFrame([{"pocket_id": "Pocket-1", "decision_score": 0.5}])
    triage_df = pd.DataFrame([{"pocket_id": "Pocket-1", "precision_tier": "evidence-gap"}])

    triage_context = build_ai_triage_context(decision_df, None, triage_df)
    prompt = build_ai_evidence_prompt(
        "The active-site residue Asp10 is required.",
        protein_name="Example enzyme",
        accession="P00001",
        pdb_id="1ABC",
        ec_number="3.2.1.4",
        triage_context=triage_context,
    )

    assert "Example enzyme" in prompt
    assert "P00001" in prompt
    assert "evidence-gap" in prompt
    assert '"residues"' in prompt
    assert "Do not infer residues" in prompt


def test_build_ai_evidence_audit_table_marks_non_ai_overlap_as_supported():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 195,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.91,
                "evidence_note": "pmid=123 | title=Triad | snippet=Ser195 is catalytic. | ai_confidence=0.910 | manual_review=false",
                "uniprot_resid": 195,
                "mapping_level": "weak",
                "mapping_confidence": 0.44,
                "mapping_method": "ai-literature-extraction",
            }
        ]
    )
    reference_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 195,
                "evidence_source": "M-CSA",
                "evidence_type": "catalytic residue",
                "evidence_score": 1.0,
                "evidence_note": "curated catalytic residue",
                "uniprot_resid": 195,
                "mapping_level": "exact",
                "mapping_confidence": 0.95,
                "mapping_method": "sifts-structure-verified",
            }
        ]
    )

    audit = build_ai_evidence_audit_table(ai_df, reference_df)

    assert audit.iloc[0]["audit_status"] == "supported"
    assert audit.iloc[0]["overlap_sources"] == "M-CSA"
    assert "weak-mapping" in audit.iloc[0]["risk_flags"]
    assert "non-AI evidence" in audit.iloc[0]["audit_reason"]


def test_build_ai_evidence_audit_table_flags_conflicting_or_unsupported_ai_rows():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 102,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.48,
                "evidence_note": "ai_confidence=0.480 | manual_review=true | structure_residue_mismatch=ASP!=ALA",
                "uniprot_resid": 102,
                "mapping_level": "weak",
                "mapping_confidence": 0.24,
                "mapping_method": "ai-structure-identity-mismatch",
            },
            {
                "chain": "A",
                "resid": 57,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.62,
                "evidence_note": "ai_confidence=0.620 | manual_review=true",
                "uniprot_resid": 57,
                "mapping_level": "weak",
                "mapping_confidence": 0.44,
                "mapping_method": "ai-literature-extraction-review",
            },
        ]
    )

    audit = build_ai_evidence_audit_table(ai_df)
    by_resid = audit.set_index("resid")

    assert by_resid.loc[102, "audit_status"] == "conflicting"
    assert "structure-conflict" in by_resid.loc[102, "risk_flags"]
    assert by_resid.loc[57, "audit_status"] == "unsupported"
    assert "missing-source-snippet" in by_resid.loc[57, "risk_flags"]
    assert "no-independent-support" in by_resid.loc[57, "risk_flags"]


def test_filter_ai_evidence_for_ranking_accepts_only_audited_safe_rows_by_default():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.91,
                "evidence_note": "pmid=1 | snippet=supported | ai_confidence=0.910",
                "uniprot_resid": 10,
                "mapping_level": "weak",
                "mapping_confidence": 0.44,
                "mapping_method": "ai-literature-extraction",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.88,
                "evidence_note": "pmid=2 | snippet=verified | ai_confidence=0.880",
                "uniprot_resid": 20,
                "mapping_level": "exact",
                "mapping_confidence": 0.84,
                "mapping_method": "ai-structure-numbering-verified",
            },
            {
                "chain": "A",
                "resid": 30,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.72,
                "evidence_note": "pmid=3 | snippet=review | ai_confidence=0.720",
                "uniprot_resid": 30,
                "mapping_level": "weak",
                "mapping_confidence": 0.44,
                "mapping_method": "ai-literature-extraction",
            },
            {
                "chain": "A",
                "resid": 40,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.48,
                "evidence_note": "ai_confidence=0.480 | structure_residue_mismatch=ASP!=ALA",
                "uniprot_resid": 40,
                "mapping_level": "weak",
                "mapping_confidence": 0.24,
                "mapping_method": "ai-structure-identity-mismatch",
            },
        ]
    )
    audit_df = pd.DataFrame(
        [
            {"chain": "A", "resid": 10, "evidence_type": "Catalytic residue", "audit_status": "supported", "risk_flags": "none", "audit_reason": ""},
            {"chain": "A", "resid": 20, "evidence_type": "Catalytic residue", "audit_status": "structure-verified", "risk_flags": "none", "audit_reason": ""},
            {"chain": "A", "resid": 30, "evidence_type": "Catalytic residue", "audit_status": "needs-review", "risk_flags": "weak-mapping", "audit_reason": ""},
            {"chain": "A", "resid": 40, "evidence_type": "Catalytic residue", "audit_status": "conflicting", "risk_flags": "structure-conflict", "audit_reason": ""},
        ]
    )

    filtered, metadata = filter_ai_evidence_for_ranking(ai_df, audit_df)

    assert set(filtered["resid"].astype(int).tolist()) == {10, 20}
    assert metadata["accepted_rows"] == "2"
    assert metadata["excluded_rows"] == "2"
    assert metadata["supported_rows"] == "1"
    assert metadata["structure_verified_rows"] == "1"
    assert filtered["evidence_note"].astype(str).str.contains("ai_audit_status=").all()


def test_filter_ai_evidence_for_ranking_can_include_review_rows_but_downgrades_them():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 30,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.82,
                "evidence_note": "pmid=3 | snippet=review | ai_confidence=0.820",
                "uniprot_resid": 30,
                "mapping_level": "exact",
                "mapping_confidence": 0.82,
                "mapping_method": "ai-structure-numbering-verified",
            }
        ]
    )
    audit_df = pd.DataFrame(
        [
            {"chain": "A", "resid": 30, "evidence_type": "Catalytic residue", "audit_status": "needs-review", "risk_flags": "weak-mapping", "audit_reason": ""}
        ]
    )

    filtered, metadata = filter_ai_evidence_for_ranking(ai_df, audit_df, allow_review=True)

    assert len(filtered) == 1
    assert metadata["review_rows"] == "1"
    assert str(filtered.iloc[0]["mapping_level"]) == "weak"
    assert float(filtered.iloc[0]["evidence_score"]) <= 0.58
    assert float(filtered.iloc[0]["mapping_confidence"]) <= 0.54
    assert str(filtered.iloc[0]["mapping_method"]).endswith("-audit-review")


def test_parse_ai_review_decision_table_normalizes_decisions():
    text = "\n".join(
        [
            "Chain\tPosition\tdecision\tcurator\tcitation\tsentence\tnote",
            "A\tSer195\tapproved\treviewer-1\tPMID:123\tSer195 is catalytic.\tverified from abstract",
            "B\t57\tno\treviewer-2\t\t\twrong enzyme",
        ]
    )

    decisions, metadata = parse_ai_review_decision_table(text)

    assert metadata["status"] == "ok"
    assert metadata["decision_rows"] == "2"
    assert decisions["resid"].tolist() == [195, 57]
    assert decisions["review_decision"].tolist() == ["accept", "reject"]
    assert decisions.iloc[0]["verified_source"] == "PMID:123"
    assert decisions.iloc[0]["verified_snippet"] == "Ser195 is catalytic."


def test_apply_ai_review_decisions_accepts_only_with_verified_source_and_blocks_conflicts():
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "ai_score": 0.82,
                "mapping_level": "weak",
                "mapping_confidence": 0.46,
                "audit_status": "needs-review",
                "overlap_sources": "none",
                "risk_flags": "weak-mapping",
                "audit_reason": "Needs mapping review.",
                "recommended_action": "Review.",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "ai_score": 0.9,
                "mapping_level": "weak",
                "mapping_confidence": 0.2,
                "audit_status": "conflicting",
                "overlap_sources": "none",
                "risk_flags": "structure-conflict",
                "audit_reason": "Residue identity mismatch.",
                "recommended_action": "Block.",
            },
            {
                "chain": "A",
                "resid": 30,
                "evidence_type": "Catalytic residue",
                "ai_score": 0.7,
                "mapping_level": "weak",
                "mapping_confidence": 0.5,
                "audit_status": "unsupported",
                "overlap_sources": "none",
                "risk_flags": "missing-source-snippet",
                "audit_reason": "Missing snippet.",
                "recommended_action": "Review.",
            },
            {
                "chain": "A",
                "resid": 40,
                "evidence_type": "Catalytic residue",
                "ai_score": 0.7,
                "mapping_level": "weak",
                "mapping_confidence": 0.5,
                "audit_status": "needs-review",
                "overlap_sources": "none",
                "risk_flags": "manual-review",
                "audit_reason": "Reviewer requested.",
                "recommended_action": "Review.",
            },
        ]
    )
    decisions_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "source checked",
                "verified_source": "PMID:1",
                "verified_snippet": "Asp10 is catalytic.",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "PMID:2",
                "verified_snippet": "His20 is catalytic.",
            },
            {
                "chain": "A",
                "resid": 30,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "PMID:3",
                "verified_snippet": "",
            },
            {
                "chain": "A",
                "resid": 40,
                "evidence_type": "Catalytic residue",
                "review_decision": "reject",
                "reviewer": "curator",
                "review_note": "unsupported",
                "verified_source": "",
                "verified_snippet": "",
            },
        ]
    )

    updated, metadata = apply_ai_review_decisions_to_audit(audit_df, decisions_df)
    by_resid = updated.set_index("resid")

    assert metadata["applied_rows"] == "4"
    assert metadata["accepted_rows"] == "1"
    assert metadata["conflict_blocked_rows"] == "1"
    assert by_resid.loc[10, "audit_status"] == "manually-accepted"
    assert "manual-accepted" in by_resid.loc[10, "risk_flags"]
    assert by_resid.loc[20, "audit_status"] == "conflicting"
    assert "manual-accept-blocked-conflict" in by_resid.loc[20, "risk_flags"]
    assert by_resid.loc[30, "audit_status"] == "needs-review"
    assert "manual-accept-missing-source" in by_resid.loc[30, "risk_flags"]
    assert by_resid.loc[40, "audit_status"] == "manually-rejected"


def test_filter_ai_evidence_for_ranking_allows_manually_accepted_with_downgrade():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.93,
                "evidence_note": "pmid=1 | snippet=Asp10 is catalytic | ai_confidence=0.930",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.91,
                "mapping_method": "ai-structure-numbering-verified",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.9,
                "evidence_note": "pmid=2 | snippet=His20 is catalytic | ai_confidence=0.900",
                "uniprot_resid": 20,
                "mapping_level": "exact",
                "mapping_confidence": 0.9,
                "mapping_method": "ai-structure-numbering-verified",
            },
        ]
    )
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "audit_status": "manually-accepted",
                "risk_flags": "manual-accepted",
                "audit_reason": "source checked",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "audit_status": "manually-rejected",
                "risk_flags": "manual-rejected",
                "audit_reason": "unsupported",
            },
        ]
    )

    filtered, metadata = filter_ai_evidence_for_ranking(ai_df, audit_df)

    assert filtered["resid"].astype(int).tolist() == [10]
    assert metadata["manually_accepted_rows"] == "1"
    assert metadata["accepted_rows"] == "1"
    assert float(filtered.iloc[0]["evidence_score"]) <= 0.68
    assert float(filtered.iloc[0]["mapping_confidence"]) <= 0.70
    assert filtered.iloc[0]["mapping_level"] == "weak"
    assert str(filtered.iloc[0]["mapping_method"]).endswith("-manual-review")
    assert "ai_audit_status=manually-accepted" in filtered.iloc[0]["evidence_note"]


def test_build_ai_review_decision_outcome_table_reports_applied_and_unmatched_rows():
    decisions_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "PMID:1",
                "verified_snippet": "Asp10 is catalytic.",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "PMID:2",
                "verified_snippet": "His20 is catalytic.",
            },
            {
                "chain": "B",
                "resid": 99,
                "evidence_type": "Catalytic residue",
                "review_decision": "reject",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "",
                "verified_snippet": "",
            },
        ]
    )
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "audit_status": "manually-accepted",
                "risk_flags": "manual-accepted",
                "recommended_action": "Allowed through ranking gate with downgrade.",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "audit_status": "conflicting",
                "risk_flags": "structure-conflict, manual-accept-blocked-conflict",
                "recommended_action": "Manual accept is blocked until structure conflict is resolved.",
            },
        ]
    )

    outcomes = build_ai_review_decision_outcome_table(decisions_df, audit_df)

    assert outcomes["applied_status"].tolist() == ["accepted", "conflict-blocked", "unmatched"]
    assert "ranking gate" in outcomes.iloc[0]["outcome_reason"]
    assert "structure" in outcomes.iloc[1]["outcome_reason"]
    assert "chain/residue/evidence_type" in outcomes.iloc[2]["outcome_reason"]


def test_build_ai_review_decision_validation_table_flags_conflicts_before_apply():
    decisions_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "PMID:1",
                "verified_snippet": "Asp10 is catalytic.",
            },
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "review_decision": "reject",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "",
                "verified_snippet": "",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "PMID:2",
                "verified_snippet": "His20 is catalytic.",
            },
            {
                "chain": "B",
                "resid": 99,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
                "reviewer": "curator",
                "review_note": "",
                "verified_source": "",
                "verified_snippet": "",
            },
        ]
    )
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "audit_status": "needs-review",
                "risk_flags": "manual-review",
            },
            {
                "chain": "A",
                "resid": 20,
                "evidence_type": "Catalytic residue",
                "audit_status": "conflicting",
                "risk_flags": "structure-conflict",
            },
        ]
    )

    validation = build_ai_review_decision_validation_table(decisions_df, audit_df)

    assert validation["validation_status"].tolist() == ["blocked", "blocked", "blocked", "blocked"]
    assert "conflicting-duplicate" in validation.iloc[0]["issue_flags"]
    assert "conflicting-duplicate" in validation.iloc[1]["issue_flags"]
    assert "accept-blocked-by-structure-conflict" in validation.iloc[2]["issue_flags"]
    assert "unmatched-audit" in validation.iloc[3]["issue_flags"]
    assert "accept-missing-source" in validation.iloc[3]["issue_flags"]
    assert bool(validation.iloc[0]["can_apply"]) is False


def test_build_ai_review_round_summary_reports_blocked_or_applied_status():
    decisions_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "review_decision": "accept",
            }
        ]
    )
    blocked_validation = pd.DataFrame(
        [
            {
                "validation_status": "blocked",
                "issue_flags": "conflicting-duplicate",
            }
        ]
    )
    accepted_outcome = pd.DataFrame(
        [
            {
                "applied_status": "accepted",
            }
        ]
    )
    rankable_df = pd.DataFrame([{"resid": 10}])

    blocked_summary = build_ai_review_round_summary(decisions_df, blocked_validation, accepted_outcome, rankable_df)

    assert blocked_summary.iloc[0]["review_round_status"] == "blocked"
    assert blocked_summary.iloc[0]["validation_blocked_rows"] == 1
    assert blocked_summary.iloc[0]["rankable_after_review_rows"] == 1

    clean_validation = pd.DataFrame([{"validation_status": "ok", "issue_flags": "none"}])
    clean_summary = build_ai_review_round_summary(decisions_df, clean_validation, accepted_outcome, rankable_df)

    assert clean_summary.iloc[0]["review_round_status"] == "applied"
    assert clean_summary.iloc[0]["outcome_accepted_rows"] == 1
    assert "ranking-gated" in clean_summary.iloc[0]["recommended_action"]


def test_build_ai_review_ranking_delta_reports_promoted_and_removed_residues():
    before = pd.DataFrame(
        [
            {"chain": "A", "resid": 10},
            {"chain": "A", "resid": 20},
        ]
    )
    after = pd.DataFrame(
        [
            {"chain": "A", "resid": 20},
            {"chain": "A", "resid": 30},
        ]
    )

    delta = build_ai_review_ranking_delta(before, after)
    row = delta.iloc[0]

    assert row["before_rankable_rows"] == 2
    assert row["after_rankable_rows"] == 2
    assert row["promoted_rows"] == 1
    assert row["removed_rows"] == 1
    assert row["unchanged_rows"] == 1
    assert row["promoted_residues"] == "A:30"
    assert row["removed_residues"] == "A:10"
    assert row["unchanged_residues"] == "A:20"
    assert row["review_effect_status"] == "changed"


def test_build_ai_review_round_report_markdown_summarizes_status_and_delta():
    summary = pd.DataFrame(
        [
            {
                "decision_rows": 2,
                "validation_blocked_rows": 1,
                "outcome_accepted_rows": 1,
                "outcome_rejected_rows": 0,
                "rankable_after_review_rows": 1,
                "review_round_status": "blocked",
                "review_round_reason": "At least one review decision failed validation.",
                "recommended_action": "Fix blocked rows.",
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "row_index": 1,
                "chain": "A",
                "resid": 10,
                "review_decision": "accept",
                "validation_status": "blocked",
                "issue_flags": "conflicting-duplicate",
                "required_fix": "Keep one decision row.",
            }
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "review_decision": "accept",
                "applied_status": "accepted",
                "current_audit_status": "manually-accepted",
                "next_action": "Inspect ranking gate.",
            }
        ]
    )
    delta = pd.DataFrame(
        [
            {
                "review_effect_status": "promoted",
                "before_rankable_rows": 0,
                "after_rankable_rows": 1,
                "promoted_residues": "A:10",
                "removed_residues": "none",
                "unchanged_residues": "none",
            }
        ]
    )

    markdown = build_ai_review_round_report_markdown(summary, validation, outcomes, delta)

    assert markdown.startswith("# AI review round report")
    assert "Status: `blocked`" in markdown
    assert "Promoted residues: A:10" in markdown
    assert "| row_index | chain | resid | review_decision | validation_status | issue_flags | required_fix |" in markdown
    assert "conflicting-duplicate" in markdown
    assert "manually-accepted" in markdown


def test_build_ai_review_artifact_manifest_lists_available_review_exports():
    manifest = build_ai_review_artifact_manifest(
        review_queue_df=pd.DataFrame([{"resid": 10}]),
        decision_template_df=pd.DataFrame([{"resid": 10}]),
        normalized_decision_df=pd.DataFrame([{"resid": 10}]),
        validation_df=pd.DataFrame([{"validation_status": "blocked"}]),
        round_summary_df=pd.DataFrame([{"review_round_status": "blocked"}]),
        ranking_delta_df=pd.DataFrame([{"review_effect_status": "promoted"}]),
        outcome_df=pd.DataFrame([{"applied_status": "accepted"}]),
        round_report_markdown="# Report\n\nBody",
    )

    assert manifest["file_name"].tolist() == [
        "ai_evidence_review_queue.csv",
        "ai_review_decision_template.csv",
        "ai_review_decisions_normalized.csv",
        "ai_review_decision_validation.csv",
        "ai_review_round_summary.csv",
        "ai_review_ranking_delta.csv",
        "ai_review_decision_outcomes.csv",
        "ai_review_round_report.md",
    ]
    assert manifest.set_index("file_name").loc["ai_review_decision_validation.csv", "status"] == "blocked"
    assert manifest.set_index("file_name").loc["ai_review_ranking_delta.csv", "status"] == "promoted"
    assert manifest.set_index("file_name").loc["ai_review_round_report.md", "artifact_type"] == "markdown"
    report_row = manifest.set_index("file_name").loc["ai_review_round_report.md"]
    assert int(report_row["byte_size"]) == len("# Report\n\nBody".encode("utf-8"))
    assert report_row["sha256"] == hashlib.sha256("# Report\n\nBody".encode("utf-8")).hexdigest()


def test_build_ai_review_artifact_bundle_zip_contains_available_exports():
    manifest = pd.DataFrame(
        [
            {
                "artifact_name": "AI review round report",
                "file_name": "ai_review_round_report.md",
                "artifact_type": "markdown",
                "row_count": 2,
                "status": "available",
                "purpose": "Report.",
                "recommended_use": "Read first.",
            }
        ]
    )

    bundle = build_ai_review_artifact_bundle_zip(
        review_queue_df=pd.DataFrame([{"resid": 10}]),
        decision_template_df=pd.DataFrame([{"resid": 10, "review_decision": "review"}]),
        normalized_decision_df=pd.DataFrame([{"resid": 10, "review_decision": "accept"}]),
        validation_df=pd.DataFrame([{"validation_status": "ok"}]),
        round_summary_df=pd.DataFrame([{"review_round_status": "applied"}]),
        ranking_delta_df=pd.DataFrame([{"review_effect_status": "promoted"}]),
        outcome_df=pd.DataFrame([{"applied_status": "accepted"}]),
        artifact_manifest_df=manifest,
        round_report_markdown="# Report\n\nBody",
    )

    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        report_text = archive.read("ai_review_round_report.md").decode("utf-8")

    assert names == {
        "ai_evidence_review_queue.csv",
        "ai_review_decision_template.csv",
        "ai_review_decisions_normalized.csv",
        "ai_review_decision_validation.csv",
        "ai_review_round_summary.csv",
        "ai_review_ranking_delta.csv",
        "ai_review_decision_outcomes.csv",
        "ai_review_artifact_manifest.csv",
        "ai_review_round_report.md",
    }
    assert report_text.startswith("# Report")


def test_build_ai_review_bundle_readme_markdown_lists_manifest_and_integrity_guidance():
    manifest = pd.DataFrame(
        [
            {
                "file_name": "ai_review_round_report.md",
                "artifact_type": "markdown",
                "row_count": 2,
                "status": "available",
                "purpose": "Human-readable report.",
            }
        ]
    )

    readme = build_ai_review_bundle_readme_markdown(manifest)

    assert readme.startswith("# AI review artifact bundle README")
    assert "Recommended Reading Order" in readme
    assert "SHA-256" in readme
    assert "ai_review_round_report.md" in readme


def test_build_ai_review_artifact_bundle_zip_can_include_readme():
    manifest = build_ai_review_artifact_manifest(
        round_report_markdown="# Report\n\nBody",
    )
    readme = build_ai_review_bundle_readme_markdown(manifest)
    manifest = build_ai_review_artifact_manifest(
        round_report_markdown="# Report\n\nBody",
        bundle_readme_markdown=readme,
    )

    bundle = build_ai_review_artifact_bundle_zip(
        artifact_manifest_df=manifest,
        round_report_markdown="# Report\n\nBody",
        bundle_readme_markdown=readme,
    )

    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        readme_text = archive.read("ai_review_bundle_README.md").decode("utf-8")
        manifest_text = archive.read("ai_review_artifact_manifest.csv").decode("utf-8-sig")

    assert "ai_review_bundle_README.md" in names
    assert readme_text.startswith("# AI review artifact bundle README")
    assert "ai_review_bundle_README.md" in manifest_text


def test_verify_ai_review_artifact_bundle_zip_checks_manifest_hashes():
    base_manifest = build_ai_review_artifact_manifest(round_report_markdown="# Report\n\nBody")
    readme = build_ai_review_bundle_readme_markdown(base_manifest)
    manifest = build_ai_review_artifact_manifest(
        round_report_markdown="# Report\n\nBody",
        bundle_readme_markdown=readme,
    )
    bundle = build_ai_review_artifact_bundle_zip(
        artifact_manifest_df=manifest,
        round_report_markdown="# Report\n\nBody",
        bundle_readme_markdown=readme,
    )

    verification = verify_ai_review_artifact_bundle_zip(bundle, manifest)

    assert set(verification["verification_status"].tolist()) == {"verified"}
    assert set(verification["file_name"].tolist()) == {
        "ai_review_round_report.md",
        "ai_review_bundle_README.md",
    }

    tampered_manifest = manifest.copy()
    tampered_manifest.loc[
        tampered_manifest["file_name"] == "ai_review_round_report.md",
        "sha256",
    ] = "0" * 64
    tampered = verify_ai_review_artifact_bundle_zip(bundle, tampered_manifest)

    assert "hash-mismatch" in tampered["verification_status"].tolist()


def test_build_ai_review_bundle_verification_summary_reports_failed_files():
    verification = pd.DataFrame(
        [
            {
                "file_name": "ok.csv",
                "verification_status": "verified",
            },
            {
                "file_name": "bad.csv",
                "verification_status": "hash-mismatch",
            },
            {
                "file_name": "missing.csv",
                "verification_status": "missing",
            },
        ]
    )

    summary = build_ai_review_bundle_verification_summary(verification)
    row = summary.iloc[0]

    assert row["checked_files"] == 3
    assert row["verified_files"] == 1
    assert row["failed_files"] == 2
    assert row["missing_files"] == 1
    assert row["hash_mismatch_files"] == 1
    assert row["verification_status"] == "failed"
    assert row["failed_file_names"] == "bad.csv, missing.csv"


def test_build_ai_review_bundle_certificate_markdown_records_bundle_hash():
    manifest = build_ai_review_artifact_manifest(round_report_markdown="# Report\n\nBody")
    bundle = build_ai_review_artifact_bundle_zip(
        artifact_manifest_df=manifest,
        round_report_markdown="# Report\n\nBody",
    )
    verification = verify_ai_review_artifact_bundle_zip(bundle, manifest)
    summary = build_ai_review_bundle_verification_summary(verification)

    certificate = build_ai_review_bundle_certificate_markdown(bundle, summary, manifest)

    assert certificate.startswith("# AI review bundle handoff certificate")
    assert "ai_review_artifacts.zip" in certificate
    assert hashlib.sha256(bundle).hexdigest() in certificate
    assert "Status: `verified`" in certificate
    assert "Manifest rows: 1" in certificate


def test_build_residue_evidence_consensus_prioritizes_cross_source_anchors():
    external = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "M-CSA",
                "evidence_type": "Catalytic residue",
                "evidence_score": 1.0,
                "evidence_note": "curated catalytic residue",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.96,
                "mapping_method": "sifts",
            }
        ]
    )
    ai_rows = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.9,
                "evidence_note": "pmid=1 | snippet=Asp10 is catalytic | ai_confidence=0.900",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.9,
                "mapping_method": "ai-literature-extraction",
            },
            {
                "chain": "A",
                "resid": 30,
                "evidence_source": "AI-Literature",
                "evidence_type": "Binding residue",
                "evidence_score": 0.55,
                "evidence_note": "ai_confidence=0.550",
                "uniprot_resid": 30,
                "mapping_level": "weak",
                "mapping_confidence": 0.35,
                "mapping_method": "ai-literature-extraction-review",
            },
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "ai_score": 0.9,
                "mapping_level": "exact",
                "mapping_confidence": 0.9,
                "audit_status": "supported",
                "overlap_sources": "M-CSA",
                "risk_flags": "none",
                "audit_reason": "AI residue overlaps non-AI evidence.",
                "recommended_action": "Keep.",
            },
            {
                "chain": "A",
                "resid": 30,
                "evidence_type": "Binding residue",
                "ai_score": 0.55,
                "mapping_level": "weak",
                "mapping_confidence": 0.35,
                "audit_status": "unsupported",
                "overlap_sources": "none",
                "risk_flags": "missing-source-snippet",
                "audit_reason": "No independent support.",
                "recommended_action": "Review.",
            },
        ]
    )
    conservation = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "ConSurf",
                "evidence_type": "Conservation",
                "evidence_score": 0.82,
                "evidence_note": "score=0.820",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.82,
                "mapping_method": "conservation-import",
            }
        ]
    )

    consensus = build_residue_evidence_consensus(
        external,
        ai_evidence_df=ai_rows,
        ai_audit_df=audit,
        rankable_ai_evidence_df=ai_rows.iloc[[0]],
        conservation_df=conservation,
    )

    by_residue = consensus.set_index("residue_anchor")
    assert consensus.iloc[0]["residue_anchor"] == "A:10"
    assert by_residue.loc["A:10", "consensus_tier"] == "validated-anchor"
    assert by_residue.loc["A:10", "rankable_ai_rows"] == 1
    assert by_residue.loc["A:10", "functional_source_count"] == 2
    assert by_residue.loc["A:30", "consensus_tier"] == "blocked-ai"
    assert by_residue.loc["A:30", "ranking_status"] == "not-ranked-ai-review"


def test_build_ai_followup_evidence_plan_uses_triage_gaps_and_context():
    decision_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "decision_rank": 1,
                "decision_score": 0.42,
                "decision_label": "Review mapping before validation",
            }
        ]
    )
    reliability_df = pd.DataFrame(
        [
            {"pocket_id": "Pocket-1", "check": "Functional anchors", "status": "missing"},
            {"pocket_id": "Pocket-1", "check": "Evidence mapping risk", "status": "review"},
        ]
    )
    triage_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "decision_rank": 1,
                "precision_tier": "evidence-gap",
                "triage_priority": 2,
                "blocking_checks": "Functional anchors",
                "review_checks": "Evidence mapping risk",
                "triage_reason": "The candidate lacks residue-level enzyme evidence.",
            }
        ]
    )

    plan = build_ai_followup_evidence_plan(
        decision_df,
        reliability_df,
        triage_df,
        protein_name="Example enzyme",
        accession="P00001",
        pdb_id="1ABC",
        ec_number="3.2.1.4",
    )

    assert plan["evidence_gap"].tolist() == ["Functional anchors", "Evidence mapping risk"]
    assert "active site" in plan.iloc[0]["search_query"]
    assert "SIFTS" in plan.iloc[1]["search_query"]
    assert "pubmed.ncbi.nlm.nih.gov" in plan.iloc[0]["pubmed_url"]
    assert "europepmc.org/search" in plan.iloc[0]["europepmc_url"]
    assert plan.iloc[0]["uniprot_url"].endswith("/P00001/entry")
    assert plan.iloc[0]["rcsb_url"].endswith("/1ABC")
    assert "Example enzyme" in plan.iloc[0]["ai_task_prompt"]
    assert "evidence-gap" in plan.iloc[0]["ai_task_prompt"]
    assert "supported or structure-verified" in plan.iloc[0]["acceptance_criteria"]


def test_build_ai_followup_evidence_plan_handles_validation_ready_as_cross_check():
    triage_df = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-Ready",
                "decision_rank": 1,
                "precision_tier": "validation-ready",
                "triage_priority": 1,
                "blocking_checks": "none",
                "review_checks": "none",
                "triage_reason": "All gates pass.",
            }
        ]
    )

    plan = build_ai_followup_evidence_plan(None, None, triage_df, protein_name="Ready enzyme")

    assert len(plan) == 1
    assert plan.iloc[0]["evidence_gap"] == "Cross-source validation"
    assert "active site" in plan.iloc[0]["search_query"]


def test_build_ai_followup_prompt_bundle_exports_copyable_markdown():
    plan = pd.DataFrame(
        [
            {
                "pocket_id": "Pocket-1",
                "decision_rank": 1,
                "precision_tier": "evidence-gap",
                "followup_priority": 21,
                "evidence_gap": "Functional anchors",
                "search_query": "Example enzyme active site catalytic residue",
                "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/?term=Example+enzyme+active+site+catalytic+residue",
                "europepmc_url": "https://europepmc.org/search?query=Example+enzyme+active+site+catalytic+residue",
                "uniprot_url": "https://www.uniprot.org/uniprotkb/P00001/entry",
                "rcsb_url": "https://www.rcsb.org/structure/1ABC",
                "ai_task_prompt": "Extract residues. Do not infer residues from general knowledge.",
                "acceptance_criteria": "Require PMID/DOI/title and AI audit status supported or structure-verified.",
                "why_this_matters": "Functional anchors are missing.",
            }
        ]
    )

    markdown = build_ai_followup_prompt_bundle(plan)

    assert markdown.startswith("# AI follow-up evidence plan")
    assert "## 1. Pocket-1 - Functional anchors" in markdown
    assert "`Example enzyme active site catalytic residue`" in markdown
    assert "### Source links" in markdown
    assert "PubMed: https://pubmed.ncbi.nlm.nih.gov/" in markdown
    assert "UniProt: https://www.uniprot.org/uniprotkb/P00001/entry" in markdown
    assert "```text" in markdown
    assert "Do not infer residues" in markdown
    assert "supported or structure-verified" in markdown


def test_build_ai_ranking_impact_summary_detects_top_pocket_ai_support():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.9,
                "evidence_note": "pmid=1 | snippet=Asp10 catalytic | ai_confidence=0.900",
                "uniprot_resid": 10,
                "mapping_level": "exact",
                "mapping_confidence": 0.9,
                "mapping_method": "ai-structure-numbering-verified",
            }
        ]
    )
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "audit_status": "structure-verified",
                "risk_flags": "none",
                "audit_reason": "verified",
                "recommended_action": "use provisionally",
            }
        ]
    )
    decision_df = pd.DataFrame(
        [
            {
                "decision_rank": 1,
                "pocket_id": "Pocket-1",
                "anchor_residues": "A:10",
                "supporting_evidence": "direct: AI-Literature",
            }
        ]
    )
    triage_df = pd.DataFrame(
        [{"pocket_id": "Pocket-1", "precision_tier": "validation-ready"}]
    )

    summary = build_ai_ranking_impact_summary(ai_df, ai_df, audit_df, decision_df, triage_df)

    row = summary.iloc[0]
    assert row["ai_influence_level"] == "top-pocket-supported"
    assert bool(row["top_pocket_has_ai_support"]) is True
    assert row["top_pocket_ai_residue_count"] == 1
    assert row["top_pocket_ai_residues"] == "A:10"
    assert row["audit_structure_verified_rows"] == 1


def test_build_ai_ranking_impact_summary_flags_blocked_ai_evidence():
    ai_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 40,
                "evidence_source": "AI-Literature",
                "evidence_type": "Catalytic residue",
                "evidence_score": 0.4,
                "evidence_note": "manual_review=true",
                "uniprot_resid": 40,
                "mapping_level": "weak",
                "mapping_confidence": 0.2,
                "mapping_method": "ai-structure-identity-mismatch",
            }
        ]
    )
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 40,
                "evidence_type": "Catalytic residue",
                "audit_status": "conflicting",
                "risk_flags": "structure-conflict",
            }
        ]
    )

    summary = build_ai_ranking_impact_summary(ai_df, pd.DataFrame(), audit_df, pd.DataFrame(), pd.DataFrame())

    row = summary.iloc[0]
    assert row["ai_influence_level"] == "blocked"
    assert row["ai_input_rows"] == 1
    assert row["ai_ranked_rows"] == 0
    assert row["ai_excluded_rows"] == 1
    assert row["audit_conflicting_rows"] == 1


def test_build_ai_evidence_review_queue_prioritizes_actionable_ai_fixes():
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 40,
                "evidence_type": "Catalytic residue",
                "audit_status": "conflicting",
                "risk_flags": "structure-conflict, weak-mapping",
            },
            {
                "chain": "A",
                "resid": 57,
                "evidence_type": "Catalytic residue",
                "audit_status": "unsupported",
                "risk_flags": "missing-source-snippet, missing-source-id, no-independent-support",
            },
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "audit_status": "supported",
                "risk_flags": "none",
            },
        ]
    )

    queue = build_ai_evidence_review_queue(audit_df)

    assert queue["resid"].tolist() == [40, 57]
    assert queue.iloc[0]["fix_type"] == "structure-conflict"
    assert bool(queue.iloc[0]["can_affect_ranking_after_fix"]) is False
    assert queue.iloc[1]["fix_type"] == "missing-citation-or-snippet"
    assert "PMID" in queue.iloc[1]["required_evidence"]


def test_build_ai_evidence_review_queue_can_include_supported_rows_for_audit_export():
    audit_df = pd.DataFrame(
        [
            {
                "chain": "A",
                "resid": 10,
                "evidence_type": "Catalytic residue",
                "audit_status": "supported",
                "risk_flags": "none",
            }
        ]
    )

    queue = build_ai_evidence_review_queue(audit_df, include_supported=True)

    assert len(queue) == 1
    assert queue.iloc[0]["audit_status"] == "supported"
    assert queue.iloc[0]["review_priority"] == 6


def test_build_ai_review_checklist_markdown_exports_actionable_items():
    queue = pd.DataFrame(
        [
            {
                "review_priority": 1,
                "chain": "A",
                "resid": 40,
                "evidence_type": "Catalytic residue",
                "audit_status": "conflicting",
                "fix_type": "structure-conflict",
                "problem": "AI residue conflicts with PDB residue identity.",
                "required_evidence": "Correct chain/residue numbering and verify residue identity.",
                "can_affect_ranking_after_fix": False,
                "suggested_next_action": "Do not use this residue for ranking yet.",
            }
        ]
    )

    markdown = build_ai_review_checklist_markdown(queue)

    assert markdown.startswith("# AI evidence review checklist")
    assert "## 1. A:40 - structure-conflict" in markdown
    assert "- [ ] Problem:" in markdown
    assert "Required evidence" in markdown
    assert "Can affect ranking after fix: no" in markdown


def test_build_ai_review_decision_template_is_parseable_after_user_edits():
    queue = pd.DataFrame(
        [
            {
                "review_priority": 2,
                "chain": "A",
                "resid": 57,
                "evidence_type": "Catalytic residue",
                "audit_status": "unsupported",
                "fix_type": "missing-citation-or-snippet",
                "problem": "AI residue lacks source text.",
                "required_evidence": "Add PMID and exact supporting sentence.",
                "can_affect_ranking_after_fix": True,
                "suggested_next_action": "Fetch source text and rerun audit.",
            }
        ]
    )

    template = build_ai_review_decision_template(queue)
    template.loc[0, "review_decision"] = "accept"
    template.loc[0, "reviewer"] = "curator"
    template.loc[0, "verified_source"] = "PMID:123"
    template.loc[0, "verified_snippet"] = "His57 is part of the catalytic triad."

    parsed, metadata = parse_ai_review_decision_table(template.to_csv(index=False))

    assert template.iloc[0]["current_audit_status"] == "unsupported"
    assert template.iloc[0]["review_decision"] == "accept"
    assert metadata["status"] == "ok"
    assert parsed.iloc[0]["chain"] == "A"
    assert parsed.iloc[0]["resid"] == 57
    assert parsed.iloc[0]["review_decision"] == "accept"
    assert parsed.iloc[0]["verified_source"] == "PMID:123"
