from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_links_scientific_and_governance_documents() -> None:
    readme = read("README.md")
    for relative in (
        "docs/ANALYSIS_ROUTE_GUIDE.md",
        "docs/COMPATIBILITY_MATRIX.md",
        "docs/HUMAN_REVIEW_POLICY.md",
        "docs/PIPELINE_CARD.md",
        "docs/DATA_GOVERNANCE.md",
    ):
        assert (ROOT / relative).is_file()
        assert f"]({relative})" in readme


def test_route_guide_keeps_observation_units_and_claims_separate() -> None:
    guide = read("docs/ANALYSIS_ROUTE_GUIDE.md")
    assert "Nuclear instance" in guide
    assert "Contextual cell" in guide
    assert "Tissue-grid core" in guide
    assert "Virtual scores are measured protein abundance" in guide
    assert "grid route primary" in guide
    assert "second complete cell-centred route" in guide
    assert "Only the cell route performs sparse-mask tissue growth" in guide
    assert "does not establish that the two routes answer the same biological question" in guide


def test_compatibility_matrix_separates_execution_from_validation() -> None:
    matrix = read("docs/COMPATIBILITY_MATRIX.md")
    assert "Intake accepted" in matrix
    assert "Technically demonstrated" in matrix
    assert "Scientifically validated" in matrix
    assert "No current CellPhenotyper route has completed a multi-site external biological validation" in matrix
    assert "Tissue microarray H&E" in matrix
    assert "0.05 to 0.50" in matrix


def test_pipeline_card_preserves_model_scope_and_failure_limits() -> None:
    card = read("docs/PIPELINE_CARD.md")
    for component in (
        "GrandQC",
        "StarDist",
        "HoVer-Net",
        "CellViT++",
        "GigaTIME",
        "UNI-2",
        "KODAMA",
        "MedSAM",
        "TITAN",
        "PathoFMPred",
    ):
        assert component in card
    assert "Agreement is not accuracy" in card
    assert "No fairness claim is supported" in card
    assert "must be replaced before a formal release claim" in card


def test_human_review_template_is_nonclinical_and_pending_by_default() -> None:
    payload = json.loads(read("resources/human_review.template.json"))
    assert payload["overall"]["decision"] == "pending"
    assert (
        payload["overall"]["approved_claim_ceiling"]
        == "engineering_feasibility_and_exploratory_description_only"
    )
    assert payload["privacy_confirmation"]["contains_no_direct_patient_identifiers"] is False
    policy = read("docs/HUMAN_REVIEW_POLICY.md")
    assert "must not be used to select the most attractive result" in policy
    assert "This is not clinical approval" in policy


def test_governance_policy_treats_derived_outputs_and_scratch_as_sensitive() -> None:
    policy = read("docs/DATA_GOVERNANCE.md")
    assert "Sensitive derived data" in policy
    assert "Nextflow `work/`" in policy
    assert "is not a de-identification tool" in policy
    assert "A pipeline parameter override is not governance approval" in policy
