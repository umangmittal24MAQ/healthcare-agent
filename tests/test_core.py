from pathlib import Path

from app.schemas import ClinicalCase, Differential, EvidencePassage, Hypothesis, LabResult, NextStep
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
        labs=[LabResult(test="Oxygen saturation", value=88, unit="%")],
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
