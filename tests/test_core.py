from pathlib import Path

from app.schemas import ClinicalCase, Differential, EvidencePassage, Hypothesis, LabResult, NextStep, VitalSign
from app.safety import evaluate_safety


def test_safety_escalates_configured_low_oxygen(tmp_path, monkeypatch):
    import app.config as config
    config.get_settings.cache_clear()
    config.load_json_file.cache_clear()
    monkeypatch.setenv("SAFETY_RULES_PATH", str(Path(__file__).resolve().parents[1] / "config" / "safety_rules.json"))
    case = ClinicalCase(
        case_id="CASE-001",
        patient_reference="synthetic",
        age=52,
        sex="male",
        chief_complaint="fever and cough",
        vitals=[VitalSign(test="oxygen saturation", value=88, unit="%")],
    )
    evidence = [EvidencePassage(passage_id="P1", source_id="S1", source_title="Reference", text="Respiratory review.", retrieval_score=1.0)]
    differential = Differential(hypotheses=[Hypothesis(rank=1,name="Respiratory infection",rationale="Possible pattern for review.",supporting_evidence=["Oxygen saturation 88%"],confidence="medium",citation_ids=["P1"])])
    result = evaluate_safety(case, differential, evidence, [NextStep(action="Chest imaging",kind="test",distinguishes_between=["Pneumonia","Bronchitis"],rationale="Clarifies focal disease.",citation_ids=["P1"])])
    assert result.status == "escalate"
    assert result.urgent_warning


def test_safety_blocks_unknown_citation(monkeypatch):
    case = ClinicalCase(case_id="CASE-002",patient_reference="synthetic",age=40,sex="unknown",chief_complaint="cough")
    evidence = [EvidencePassage(passage_id="P1",source_id="S1",source_title="Reference",text="Respiratory review.",retrieval_score=.5)]
    differential = Differential(hypotheses=[Hypothesis(rank=1,name="Respiratory illness",rationale="For review.",confidence="low",citation_ids=["UNKNOWN"])])
    result = evaluate_safety(case,differential,evidence,[])
    assert result.status == "block"
    assert any("citation" in reason.lower() for reason in result.blocked_reasons)


def test_problem_representation_accepts_numbers_from_original_note():
    from app.pipeline import _validate_problem_numbers

    note = (
        "A 68-year-old man with coronary artery disease, HFrEF (EF 30%), and hypertension "
        "presents with progressive dyspnea over 3 days, orthopnea requiring 3 pillows, and edema. "
        "Blood pressure 162/98 mmHg, heart rate 102 bpm, respiratory rate 24 breaths/min, "
        "oxygen saturation 87% on room air. WBC 7.1 and BNP 1,450 pg/mL."
    )
    case = ClinicalCase(
        case_id="CASE-HF",
        patient_reference="synthetic",
        age=68,
        sex="male",
        chief_complaint="progressive dyspnea",
        labs=[
            LabResult(test="WBC", value=7.1),
            LabResult(test="BNP", value=1450, unit="pg/mL"),
        ],
    )

    _validate_problem_numbers(
        case,
        note,
        "68-year-old man with HFrEF (EF 30%), 3 days of dyspnea, BP 162/98, "
        "heart rate 102, respiratory rate 24, oxygen saturation 87%, WBC 7.1 and BNP 1450.",
    )


def test_intake_normalizes_common_clinical_enum_synonyms():
    from app.pipeline import _normalize_intake_output

    raw = {
        "sex": "M",
        "symptoms": [{"name": "dyspnea", "severity": "Severe"}],
        "medications": [{"name": "furosemide", "status": "currently taking"}],
        "vitals": [{"test": "oxygen saturation", "value": 87, "flag": "decreased"}],
        "labs": [
            {"test": "BNP", "value": 1450, "flag": "elevated"},
            {"test": "Sodium", "value": 128, "flag": "decreased"},
            {"test": "WBC", "value": 7.1, "flag": "within normal limits"},
        ],
    }

    normalized = _normalize_intake_output(raw)

    assert normalized["sex"] == "male"
    assert normalized["symptoms"][0]["severity"] == "severe"
    assert normalized["medications"][0]["status"] == "active"
    assert normalized["vitals"][0]["flag"] == "low"
    assert normalized["labs"][0]["flag"] == "high"
    assert normalized["labs"][1]["flag"] == "low"
    assert normalized["labs"][2]["flag"] == "normal"


def test_confidence_normalization_maps_moderate_to_medium():
    from app.pipeline import _normalize_hypothesis_confidence

    raw = {
        "independent_hypotheses": [
            {"name": "A", "rationale": "x", "confidence": "moderate"},
            {"name": "B", "rationale": "y", "confidence": "intermediate"},
            {"name": "C", "rationale": "z", "confidence": "high"},
        ]
    }

    normalized = _normalize_hypothesis_confidence(raw, "independent_hypotheses")

    assert normalized["independent_hypotheses"][0]["confidence"] == "medium"
    assert normalized["independent_hypotheses"][1]["confidence"] == "medium"
    assert normalized["independent_hypotheses"][2]["confidence"] == "high"


def test_hypothesis_normalizer_accepts_stringified_json_object():
    import json
    from app.pipeline import _normalize_hypothesis_confidence

    raw = {
        "hypotheses": [
            json.dumps(
                {
                    "rank": 1,
                    "name": "Example",
                    "rationale": "For review.",
                    "confidence": "moderate",
                    "citation_ids": ["P1"],
                }
            )
        ]
    }

    normalized = _normalize_hypothesis_confidence(raw, "hypotheses")

    assert normalized["hypotheses"][0]["name"] == "Example"
    assert normalized["hypotheses"][0]["confidence"] == "medium"


def test_hypothesis_normalizer_rejects_plain_prose_string():
    import pytest
    from app.pipeline import _normalize_hypothesis_confidence

    with pytest.raises(ValueError, match="must be a JSON object"):
        _normalize_hypothesis_confidence(
            {"hypotheses": ["Acute decompensated heart failure"]},
            "hypotheses",
        )


def test_hypothesis_normalizer_ignores_out_of_contract_sixth_item():
    from app.pipeline import _normalize_hypothesis_confidence

    valid = [
        {
            "rank": i,
            "name": f"Hypothesis {i}",
            "rationale": "For review.",
            "confidence": "medium",
            "citation_ids": ["P1"],
        }
        for i in range(1, 6)
    ]
    raw = {
        "hypotheses": [
            *valid,
            "unresolved_questions [",
        ],
        "unresolved_questions": ["What changed from baseline?"],
    }

    normalized = _normalize_hypothesis_confidence(
        raw,
        "hypotheses",
        max_items=5,
    )

    assert len(normalized["hypotheses"]) == 5
    assert all(isinstance(item, dict) for item in normalized["hypotheses"])
    assert normalized["unresolved_questions"] == ["What changed from baseline?"]



def test_qualitative_lab_without_numeric_value_is_preserved():
    from app.pipeline import _normalize_intake_output, _timeline_and_findings
    from app.schemas import ClinicalCase

    raw = {
        "age": 24,
        "sex": "male",
        "chief_complaint": "vomiting and lethargy",
        "symptoms": [],
        "history": ["type 1 diabetes"],
        "medications": [],
        "vitals": [],
        "labs": [
            {
                "test": "serum ketones",
                "value": None,
                "unit": None,
                "reference_range": None,
                "flag": "unknown",
                "result": "positive",
                "observed_at": None,
            }
        ],
        "imaging": [],
        "notes": [],
    }

    normalized = _normalize_intake_output(raw)
    normalized["case_id"] = "CASE-QUAL"
    normalized["patient_reference"] = "synthetic"
    case = ClinicalCase.model_validate(normalized)

    assert case.labs[0].value is None
    assert case.labs[0].interpretation == "positive"

    timeline, _, _, _ = _timeline_and_findings(case)
    assert timeline[0].detail.startswith("positive;")
    assert "None" not in timeline[0].detail


def test_non_elevated_qualitative_lab_can_have_null_value():
    from app.schemas import LabResult

    lab = LabResult(
        test="troponin",
        value=None,
        flag="normal",
        interpretation="not elevated",
    )

    assert lab.value is None
    assert lab.flag == "normal"
    assert lab.interpretation == "not elevated"



def test_problem_representation_allows_grounded_interarm_bp_difference():
    from app.pipeline import _validate_problem_numbers

    note = (
        "A 64-year-old man with longstanding hypertension develops abrupt severe tearing chest pain "
        "radiating to the back. Blood pressure is 188/104 mmHg in the right arm and 158/92 mmHg "
        "in the left arm, heart rate 104 bpm, respiratory rate 22 breaths/min, and oxygen saturation "
        "97% on room air. Chest X-ray shows a widened mediastinum. The right radial pulse is weaker "
        "than the left."
    )
    case = ClinicalCase(
        case_id="CASE-AORTIC",
        patient_reference="synthetic",
        age=64,
        sex="male",
        chief_complaint="abrupt tearing chest pain",
        vitals=[
            VitalSign(test="right arm blood pressure", value="188/104", unit="mmHg"),
            VitalSign(test="left arm blood pressure", value="158/92", unit="mmHg"),
            VitalSign(test="heart rate", value=104, unit="bpm"),
            VitalSign(test="respiratory rate", value=22, unit="breaths/min"),
            VitalSign(test="oxygen saturation", value=97, unit="%"),
        ],
    )

    sanitized = _validate_problem_numbers(
        case,
        note,
        "64-year-old man with abrupt tearing chest pain, widened mediastinum, pulse asymmetry, "
        "and a 30 mmHg inter-arm systolic blood-pressure difference.",
    )

    assert "30 mmHg" not in sanitized
    assert "inter-arm systolic blood-pressure difference" in sanitized


def test_problem_representation_rejects_new_number_presented_as_patient_fact():
    import pytest
    from app.pipeline import _validate_problem_numbers

    note = "A 64-year-old man has blood pressure 188/104 mmHg."
    case = ClinicalCase(
        case_id="CASE-NUMERIC-BAD",
        patient_reference="synthetic",
        age=64,
        sex="male",
        chief_complaint="chest pain",
    )

    with pytest.raises(ValueError, match="unsupported patient number"):
        _validate_problem_numbers(
            case,
            note,
            "64-year-old man with blood pressure 31 mmHg.",
        )



def test_problem_representation_omits_generic_calculated_values():
    from app.pipeline import _validate_problem_numbers

    note = "A 50-year-old patient has sodium 140 mmol/L and sodium 132 mmol/L on repeat testing."
    case = ClinicalCase(
        case_id="CASE-CALC",
        patient_reference="synthetic",
        age=50,
        sex="unknown",
        chief_complaint="weakness",
        labs=[
            LabResult(test="sodium", value=140, unit="mmol/L"),
            LabResult(test="sodium repeat", value=132, unit="mmol/L"),
        ],
    )

    sanitized = _validate_problem_numbers(
        case,
        note,
        "50-year-old patient with an 8 mmol/L decrease in sodium.",
    )

    assert "8 mmol/L" not in sanitized
    assert "decrease in sodium" in sanitized


def test_semantic_similarity_handles_extended_diagnosis_names():
    from app.pipeline import _similar

    assert _similar(
        "Pulmonary embolism",
        "Pulmonary embolism as precipitant or co-existing process",
    )
    assert _similar(
        "Acute decompensated heart failure",
        "Acute decompensated heart failure (HFrEF exacerbation)",
    )
    assert not _similar("Pulmonary embolism", "Community-acquired pneumonia")


def test_next_step_kind_normalization_handles_common_synonyms():
    from app.pipeline import _normalize_next_step_kind

    assert _normalize_next_step_kind("diagnostic test") == "test"
    assert _normalize_next_step_kind("lab") == "test"
    assert _normalize_next_step_kind("physical examination") == "observation"
    assert _normalize_next_step_kind("history question") == "question"
