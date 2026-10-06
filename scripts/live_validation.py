from __future__ import annotations

import json

from app.llm import probe_llm
from app.pipeline import run_analysis
from app.schemas import AnalyzeRequest
from app.store import init_db


NOTE = (
    "A 68-year-old man with a history of coronary artery disease, HFrEF (EF 30%), "
    "and hypertension presents with progressive dyspnea over 3 days, orthopnea requiring "
    "3 pillows to sleep, and bilateral lower extremity swelling. He reports a mild "
    "non-productive cough. Vitals: blood pressure 162/98 mmHg, heart rate 102 bpm, "
    "respiratory rate 24 breaths/min, and oxygen saturation 87% on room air. Labs reveal "
    "WBC 7.1 and elevated BNP at 1,450 pg/mL. Chest X-ray shows bilateral diffuse "
    "interstitial infiltrates, Kerley B lines, cardiomegaly, and small bilateral pleural effusions."
)

EXPECTED_STAGES = [
    "Intake Agent",
    "Clinical Synthesis Agent",
    "Evidence Agent",
    "Reasoning Agent",
    "Challenge Agent",
    "Differential Revision",
    "Next-Best-Step Agent",
    "Safety Gate",
]


def main() -> None:
    init_db()
    print("1/2 Probing live MAQ IndiaAI Qwen endpoint...")
    probe = probe_llm()
    if probe.get("ok") is not True:
        raise AssertionError(f"Live LLM probe failed: {probe}")
    if probe.get("model") != "qwen-3.8-27b":
        raise AssertionError(f"Unexpected model: {probe.get('model')}")
    print("Live LLM probe: PASS")

    print("2/2 Running full live synthetic clinical pipeline...")
    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        event_type = event.get("type")
        stage = event.get("stage", "Pipeline")
        if event_type == "stage_start":
            print(f"[START] {stage}")
        elif event_type == "llm_connected":
            print(f"[LLM]   {stage}: connected HTTP {event.get('status_code')}")
        elif event_type == "llm_delta":
            chars = event.get("received_chars", 0)
            if chars and chars % 500 < 20:
                content_chars = event.get("content_chars", 0)
                reasoning_chars = event.get("reasoning_chars", 0)
                phase = "final JSON" if content_chars else "thinking"
                print(
                    f"[LLM]   {stage}: {phase} "
                    f"(content={content_chars}, reasoning={reasoning_chars})"
                )
        elif event_type == "llm_complete":
            print(
                f"[LLM]   {stage}: stream complete "
                f"(content={event.get('content_chars', 0)}, "
                f"reasoning={event.get('reasoning_chars', 0)}, "
                f"finish={event.get('finish_reason')})"
            )
        elif event_type == "stage_complete":
            print(f"[PASS]  {stage}")

    request = AnalyzeRequest(
        note=NOTE,
        patient_reference="github-actions-live-hf",
        confirm_synthetic=True,
    )
    result = run_analysis(request, emit=emit)

    starts = [e["stage"] for e in events if e.get("type") == "stage_start"]
    completes = [e["stage"] for e in events if e.get("type") == "stage_complete"]
    if starts != EXPECTED_STAGES:
        raise AssertionError(f"Unexpected stage order: {starts}")
    if completes != EXPECTED_STAGES:
        raise AssertionError(f"Not all stages completed: {completes}")

    if result.model != "qwen-3.8-27b":
        raise AssertionError(f"Unexpected result model: {result.model}")
    if result.case.age != 68:
        raise AssertionError(f"Intake failed to preserve age 68: {result.case.age}")
    if not result.synthesis.problem_representation:
        raise AssertionError("Clinical synthesis is empty.")
    if not result.retrieved_evidence:
        raise AssertionError("Evidence retrieval returned no passages.")
    if not result.initial_differential.hypotheses:
        raise AssertionError("Initial differential is empty.")
    if not result.challenge.summary:
        raise AssertionError("Challenge summary is empty.")
    if not result.revised_differential.hypotheses:
        raise AssertionError("Revised differential is empty.")
    if not result.next_steps:
        raise AssertionError("Next-best-step output is empty.")

    known_ids = {item.passage_id for item in result.retrieved_evidence}
    if not known_ids:
        raise AssertionError("No evidence IDs available for citation validation.")

    for label, differential in (
        ("initial", result.initial_differential),
        ("revised", result.revised_differential),
    ):
        for hypothesis in differential.hypotheses:
            if not hypothesis.citation_ids:
                raise AssertionError(f"{label} hypothesis has no citations: {hypothesis.name}")
            unknown = set(hypothesis.citation_ids) - known_ids
            if unknown:
                raise AssertionError(
                    f"{label} hypothesis contains unknown citations {sorted(unknown)}: {hypothesis.name}"
                )

    for step in result.next_steps:
        unknown = set(step.citation_ids) - known_ids
        if unknown:
            raise AssertionError(
                f"Next step contains unknown citations {sorted(unknown)}: {step.action}"
            )
        if not step.distinguishes_between:
            raise AssertionError(
                f"Next step does not explain what it distinguishes: {step.action}"
            )

    # This synthetic case explicitly contains SpO2 87%; the configured safety gate
    # must surface an urgent clinician-review warning when intake preserves it.
    if not result.safety.urgent_warning:
        raise AssertionError(
            "Expected urgent safety warning for oxygen saturation 87%, but none was produced."
        )
    if result.safety.status not in {"escalate", "block"}:
        raise AssertionError(
            f"Expected escalation/block for low oxygen case, got {result.safety.status}."
        )

    print(
        json.dumps(
            {
                "live_validation": "PASS",
                "run_id": result.run_id,
                "model": result.model,
                "evidence_passages": len(result.retrieved_evidence),
                "initial_hypotheses": len(result.initial_differential.hypotheses),
                "revised_hypotheses": len(result.revised_differential.hypotheses),
                "next_steps": len(result.next_steps),
                "safety_status": result.safety.status,
                "urgent_warning": bool(result.safety.urgent_warning),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
