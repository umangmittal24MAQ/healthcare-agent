from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.config as config
import app.pipeline as pipeline
import app.retrieval as retrieval
import app.store as store
from app.main import app
from app.schemas import (
    AnalysisResult,
    AnalyzeRequest,
    Challenge,
    ClinicalCase,
    Differential,
    EvidencePassage,
    Hypothesis,
    LabResult,
    NextStep,
    SafetyCheck,
    SafetyResult,
    Synthesis,
)


HF_NOTE = (
    "A 68-year-old man with a history of coronary artery disease, HFrEF (EF 30%), "
    "and hypertension presents with progressive dyspnea over 3 days, orthopnea requiring "
    "3 pillows to sleep, and bilateral lower extremity swelling. He reports a mild "
    "non-productive cough. Vitals: blood pressure 162/98 mmHg, heart rate 102 bpm, "
    "respiratory rate 24 breaths/min, and oxygen saturation 87% on room air. Labs reveal "
    "WBC 7.1 and elevated BNP at 1,450 pg/mL. Chest X-ray shows bilateral diffuse "
    "interstitial infiltrates, Kerley B lines, cardiomegaly, and small bilateral pleural effusions."
)


def _evidence():
    return [
        EvidencePassage(
            passage_id="P1",
            source_id="S1",
            source_title="Synthetic heart failure reference",
            text="Dyspnea, orthopnea, edema, cardiomegaly and interstitial edema may support heart failure assessment.",
            retrieval_score=1.0,
        ),
        EvidencePassage(
            passage_id="P2",
            source_id="S2",
            source_title="Synthetic respiratory reference",
            text="Hypoxemia and bilateral infiltrates warrant evaluation of pulmonary and cardiac causes.",
            retrieval_score=0.8,
        ),
    ]


def test_retrieval_uses_curated_corpus_and_validates_reranked_ids(monkeypatch):
    case = ClinicalCase(
        case_id="CASE-R",
        patient_reference="synthetic",
        age=68,
        sex="male",
        chief_complaint="progressive dyspnea",
        symptoms=[],
        history=["HFrEF"],
        labs=[LabResult(test="Oxygen saturation", value=87, unit="%", flag="low")],
    )
    synthesis = Synthesis(
        problem_representation="68-year-old man with HFrEF and progressive dyspnea with oxygen saturation 87%.",
        active_problems=["progressive dyspnea"],
        abnormal_findings=["Oxygen saturation: 87 % (low)"],
        timeline=[],
        missing_information=[],
    )

    seen = {}

    def fake_llm_json(**kwargs):
        candidates = kwargs["user_payload"]["candidates"]
        seen["candidate_count"] = len(candidates)
        assert candidates
        return {"ordered_ids": [candidates[0]["passage_id"]]}

    monkeypatch.setattr(retrieval, "llm_json", fake_llm_json)

    result = retrieval.retrieve_evidence(case, synthesis)

    assert seen["candidate_count"] >= 1
    assert len(result) == 1
    assert result[0].passage_id.startswith("SYN-GUIDE-")
    assert result[0].text


def test_full_pipeline_orchestration_with_blind_challenge(monkeypatch):
    request = AnalyzeRequest(note=HF_NOTE, patient_reference="ci-hf", confirm_synthetic=True)
    evidence = _evidence()
    calls = []
    differential_call = 0

    def fake_llm_json(**kwargs):
        nonlocal differential_call
        system = kwargs["system_prompt"]
        calls.append(system)

        if system.startswith("Extract explicit facts"):
            return {
                "age": 68,
                "sex": "M",
                "chief_complaint": "progressive dyspnea",
                "symptoms": [
                    {"name": "dyspnea", "onset": "3 days", "severity": "progressive", "notes": None},
                    {"name": "orthopnea", "onset": "3 days", "severity": None, "notes": "requires 3 pillows"},
                    {"name": "bilateral lower extremity swelling", "onset": "3 days", "severity": None, "notes": None},
                ],
                "history": ["coronary artery disease", "HFrEF EF 30%", "hypertension"],
                "medications": [],
                "labs": [
                    {"test": "Oxygen saturation", "value": 87, "unit": "%", "reference_range": None, "flag": "decreased", "observed_at": None},
                    {"test": "WBC", "value": 7.1, "unit": None, "reference_range": None, "flag": "within normal limits", "observed_at": None},
                    {"test": "BNP", "value": 1450, "unit": "pg/mL", "reference_range": None, "flag": "elevated", "observed_at": None},
                ],
                "imaging": [
                    {
                        "modality": "Chest X-ray",
                        "body_region": "chest",
                        "summary": "bilateral diffuse interstitial infiltrates, Kerley B lines, cardiomegaly, and small bilateral pleural effusions",
                        "observed_at": None,
                    }
                ],
                "notes": [],
            }

        if system.startswith("Write one concise clinical problem representation"):
            return {
                "problem_representation": (
                    "68-year-old man with HFrEF (EF 30%) and 3 days of progressive dyspnea, "
                    "orthopnea, edema, oxygen saturation 87%, BNP 1450, and bilateral interstitial "
                    "infiltrates with cardiomegaly and pleural effusions."
                )
            }

        if system.startswith("Generate a ranked differential diagnosis"):
            differential_call += 1
            if differential_call == 1:
                return {
                    "hypotheses": [
                        {
                            "rank": 1,
                            "name": "Acute decompensated heart failure",
                            "rationale": "Pattern is compatible and requires clinician review.",
                            "supporting_evidence": ["Oxygen saturation 87%", "BNP 1450"],
                            "opposing_evidence": [],
                            "missing_information": ["Current weight trend"],
                            "confidence": "high",
                            "citation_ids": ["P1"],
                        },
                        {
                            "rank": 2,
                            "name": "Pulmonary infection",
                            "rationale": "Cough and infiltrates are an alternative requiring review.",
                            "supporting_evidence": ["Bilateral infiltrates"],
                            "opposing_evidence": ["WBC 7.1"],
                            "missing_information": ["Temperature"],
                            "confidence": "low",
                            "citation_ids": ["P2"],
                        },
                    ],
                    "unresolved_questions": ["Volume status examination"],
                }
            return {
                "hypotheses": [
                    {
                        "rank": 1,
                        "name": "Acute decompensated heart failure",
                        "rationale": "Remains highest after independent challenge.",
                        "supporting_evidence": ["Oxygen saturation 87%", "BNP 1450"],
                        "opposing_evidence": [],
                        "missing_information": ["Current weight trend"],
                        "confidence": "high",
                        "citation_ids": ["P1"],
                    },
                    {
                        "rank": 2,
                        "name": "Pulmonary embolic disease",
                        "rationale": "Independent challenge identified a high-risk alternative for clinician review.",
                        "supporting_evidence": ["Oxygen saturation 87%"],
                        "opposing_evidence": ["Bilateral edema and cardiomegaly favor a cardiac explanation"],
                        "missing_information": ["Risk factors for venous thromboembolism"],
                        "confidence": "low",
                        "citation_ids": ["P2"],
                    },
                ],
                "unresolved_questions": ["Volume status examination", "Thromboembolic risk factors"],
            }

        if system.startswith("Act as an independent clinical challenge agent"):
            return {
                "independent_hypotheses": [
                    {
                        "name": "Acute decompensated heart failure",
                        "rationale": "Congestion pattern",
                        "confidence": "high",
                    },
                    {
                        "name": "Pulmonary embolic disease",
                        "rationale": "Hypoxemia requires an independent high-risk alternative check",
                        "confidence": "low",
                    },
                ],
                "contradictions": ["No fever is documented."],
                "missing_questions": ["Any venous thromboembolism risk factors?"],
                "high_risk_alternatives": ["Pulmonary embolic disease"],
                "summary": "Blind challenge adds a thromboembolic alternative.",
            }

        if system.startswith("Recommend only diagnostic information-gathering steps"):
            return {
                "suggestions": [
                    {
                        "action": "Clarify recent weight change and baseline functional status",
                        "kind": "question",
                        "distinguishes_between": ["Acute decompensated heart failure", "Pulmonary embolic disease"],
                        "rationale": "Helps characterize congestion and change from baseline.",
                        "citation_ids": ["P1"],
                    }
                ]
            }

        raise AssertionError(f"Unexpected LLM prompt: {system[:80]}")

    monkeypatch.setattr(pipeline, "llm_json", fake_llm_json)
    monkeypatch.setattr(pipeline, "retrieve_evidence", lambda case, synthesis: evidence)
    monkeypatch.setattr(pipeline, "save_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "get_llm_config", lambda: SimpleNamespace(id="qwen-3.8-27b"))

    events = []
    result = pipeline.run_analysis(request, emit=events.append)

    assert result.model == "qwen-3.8-27b"
    assert result.case.labs[2].flag == "high"
    assert result.case.labs[0].flag == "low"
    assert result.initial_differential.hypotheses[0].name == "Acute decompensated heart failure"
    assert result.revised_differential.hypotheses[1].name == "Pulmonary embolic disease"
    assert any("Pulmonary embolic disease" in item for item in result.challenge.disagreements)
    assert result.next_steps[0].distinguishes_between
    assert result.safety.status == "escalate"
    assert result.safety.urgent_warning

    starts = [e["stage"] for e in events if e["type"] == "stage_start"]
    completes = [e["stage"] for e in events if e["type"] == "stage_complete"]
    expected = [
        "Intake Agent",
        "Clinical Synthesis Agent",
        "Evidence Agent",
        "Reasoning Agent",
        "Challenge Agent",
        "Differential Revision",
        "Next-Best-Step Agent",
        "Safety Gate",
    ]
    assert starts == expected
    assert completes == expected
    assert differential_call == 2


def test_store_persists_run_and_human_review(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    config.get_settings.cache_clear()

    try:
        store.init_db()
        payload = {"run_id": "RUN-CI", "value": 1}
        store.save_run("RUN-CI", "2026-10-06T00:00:00+00:00", "qwen-3.8-27b", "pass", payload)
        assert store.get_run("RUN-CI") == payload

        store.save_review(
            "RUN-CI",
            "accepted",
            "Reviewed in CI",
            "2026-10-06T00:01:00+00:00",
        )

        with sqlite3.connect(db) as conn:
            review_count = conn.execute("SELECT COUNT(*) FROM reviews WHERE run_id='RUN-CI'").fetchone()[0]
            audit_count = conn.execute("SELECT COUNT(*) FROM audit_events WHERE run_id='RUN-CI'").fetchone()[0]

        assert review_count == 1
        assert audit_count == 2
    finally:
        config.get_settings.cache_clear()


def _api_result() -> AnalysisResult:
    case = ClinicalCase(
        case_id="CASE-API",
        patient_reference="synthetic",
        age=40,
        sex="unknown",
        chief_complaint="cough",
    )
    synthesis = Synthesis(
        problem_representation="40-year-old patient with cough.",
        active_problems=["cough"],
        abnormal_findings=[],
        timeline=[],
        missing_information=[],
    )
    differential = Differential(
        hypotheses=[
            Hypothesis(
                rank=1,
                name="Respiratory illness",
                rationale="Requires clinician review.",
                confidence="low",
                citation_ids=["P1"],
            )
        ]
    )
    evidence = [
        EvidencePassage(
            passage_id="P1",
            source_id="S1",
            source_title="Synthetic reference",
            text="Respiratory review.",
            retrieval_score=1.0,
        )
    ]
    challenge = Challenge(
        independent_hypotheses=[],
        contradictions=[],
        missing_questions=[],
        high_risk_alternatives=[],
        summary="No additional challenge in API test.",
        disagreements=[],
    )
    safety = SafetyResult(
        status="pass",
        checks=[SafetyCheck(name="ci", passed=True, detail="CI")],
    )
    return AnalysisResult(
        run_id="RUN-API",
        case=case,
        synthesis=synthesis,
        retrieved_evidence=evidence,
        initial_differential=differential,
        challenge=challenge,
        revised_differential=differential,
        next_steps=[],
        safety=safety,
        model="qwen-3.8-27b",
    )


def test_streaming_api_emits_progress_and_final_result(monkeypatch):
    import app.main as main

    def fake_run_analysis(request, emit=None):
        assert request.confirm_synthetic is True
        emit({"type": "stage_start", "stage": "Intake Agent"})
        emit({"type": "llm_connected", "stage": "Intake Agent", "status_code": 200})
        emit({"type": "llm_delta", "stage": "Intake Agent", "received_chars": 42})
        emit({"type": "stage_complete", "stage": "Intake Agent"})
        return _api_result()

    monkeypatch.setattr(main, "run_analysis", fake_run_analysis)

    with TestClient(app) as client:
        response = client.post(
            "/api/analyze/stream",
            json={
                "note": "Synthetic note long enough for API validation.",
                "patient_reference": "api-ci",
                "confirm_synthetic": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "stage_start"' in body
    assert '"type": "llm_delta"' in body
    assert '"type": "final"' in body
    assert '"run_id": "RUN-API"' in body


def test_configuration_is_single_maq_qwen_provider():
    config.get_settings.cache_clear()
    config.get_llm_config.cache_clear()
    model = config.get_llm_config()

    assert model.id == "qwen-3.8-27b"
    assert str(model.url) == "https://indiaai.maqsoftware.net/v1/chat/completions"
    assert model.maxOutputTokens == 8192

    requirements = open("requirements.txt", encoding="utf-8").read().lower()
    assert "groq" not in requirements
    assert "openai" not in requirements
