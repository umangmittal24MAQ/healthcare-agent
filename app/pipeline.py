from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher

from app.config import get_llm_config
from app.llm import llm_json, reset_llm_event_callback, set_llm_event_callback
from app.retrieval import retrieve_evidence
from app.safety import evaluate_safety
from app.schemas import (
    AnalysisResult,
    AnalyzeRequest,
    Challenge,
    ClinicalCase,
    Differential,
    NextStep,
    Synthesis,
    TimelineEvent,
)
from app.store import save_run


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


def _stage(stage: str, fn, emit=None):
    if emit:
        emit({"type": "stage_start", "stage": stage})

    token = set_llm_event_callback(
        (lambda event: emit({**event, "stage": stage})) if emit else None
    )
    try:
        result = fn()
        if emit:
            emit({"type": "stage_complete", "stage": stage})
        return result
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError(stage, str(exc)) from exc
    finally:
        reset_llm_event_callback(token)


def _contract(model_cls) -> str:
    return json.dumps(model_cls.model_json_schema(), ensure_ascii=False)


def _derive_lab_flag(value: float | str | None, reference_range: str | None, supplied_flag: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return supplied_flag
    if not reference_range:
        return supplied_flag
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", reference_range.replace(",", ""))]
    if len(numbers) >= 2:
        low, high = numbers[0], numbers[1]
        return "low" if numeric < low else "high" if numeric > high else "normal"
    if len(numbers) == 1:
        boundary = numbers[0]
        if "<" in reference_range:
            return "high" if numeric >= boundary else "normal"
        if ">" in reference_range:
            return "low" if numeric <= boundary else "normal"
    return supplied_flag


def _timeline_and_findings(case: ClinicalCase) -> tuple[list[TimelineEvent], list[str], list[str], list[str]]:
    timeline: list[TimelineEvent] = []
    abnormal: list[str] = []
    active: list[str] = [s.name for s in case.symptoms]

    for symptom in case.symptoms:
        detail = symptom.name
        if symptom.severity:
            detail += f"; severity={symptom.severity}"
        if symptom.onset:
            detail += f"; onset={symptom.onset}"
        timeline.append(TimelineEvent(event_type="symptom", label=symptom.name, detail=detail))

    for vital in case.vitals:
        vital.flag = _derive_lab_flag(vital.value, vital.reference_range, vital.flag)
        detail = f"{vital.value}{(' ' + vital.unit) if vital.unit else ''}; flag={vital.flag}"
        timeline.append(
            TimelineEvent(
                event_type="vital",
                label=vital.test,
                detail=detail,
                observed_at=vital.observed_at,
            )
        )
        if vital.flag in {"low", "high", "critical"}:
            finding = f"{vital.test}: {vital.value}{(' ' + vital.unit) if vital.unit else ''} ({vital.flag})"
            abnormal.append(finding)
            active.append(f"abnormal vital - {finding}")

    for lab in case.labs:
        lab.flag = _derive_lab_flag(lab.value, lab.reference_range, lab.flag)
        if lab.value is not None:
            value_text = f"{lab.value}{(' ' + lab.unit) if lab.unit else ''}"
        elif lab.interpretation:
            value_text = lab.interpretation
        else:
            value_text = "not quantified"
        detail = f"{value_text}; flag={lab.flag}"
        timeline.append(TimelineEvent(event_type="lab", label=lab.test, detail=detail, observed_at=lab.observed_at))
        if lab.flag in {"low", "high", "critical"}:
            finding = f"{lab.test}: {value_text} ({lab.flag})"
            abnormal.append(finding)
            active.append(f"abnormal lab - {finding}")

    for imaging in case.imaging:
        timeline.append(
            TimelineEvent(
                event_type="imaging",
                label=imaging.modality,
                detail=imaging.summary,
                observed_at=imaging.observed_at,
            )
        )
        abnormal.append(f"{imaging.modality}: {imaging.summary}")

    timeline.sort(key=lambda x: (x.observed_at is None, x.observed_at or datetime.max.replace(tzinfo=timezone.utc)))
    missing: list[str] = []
    if not case.history:
        missing.append("Past medical history not supplied.")
    if not case.medications:
        missing.append("Medication history not supplied.")
    if not case.labs:
        missing.append("Laboratory data not supplied.")
    if not case.imaging:
        missing.append("Imaging data not supplied.")
    return timeline, abnormal, active or [case.chief_complaint], missing


def _canonical_number(token: str) -> str:
    value = float(token.replace(",", ""))
    return str(int(value)) if value.is_integer() else str(value).rstrip("0").rstrip(".")


def _record_numbers(case: ClinicalCase, source_note: str) -> set[str]:
    values = {_canonical_number(str(case.age))}
    normalized_note = source_note.replace(",", "")
    for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", normalized_note):
        values.add(_canonical_number(token))
    for measurement in [*case.vitals, *case.labs]:
        try:
            values.add(_canonical_number(str(measurement.value)))
        except (TypeError, ValueError):
            continue
    return values


def _validate_problem_numbers(case: ClinicalCase, source_note: str, text: str) -> None:
    allowed = _record_numbers(case, source_note)
    for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text.replace(",", "")):
        canonical = _canonical_number(token)
        if canonical not in allowed:
            raise ValueError(f"Problem representation introduced unsupported number: {token}")


def _normalize_enum(value, mapping: dict[str, str], default):
    if value is None:
        return default
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    return mapping.get(normalized, default)


def _normalize_confidence(value):
    if value is None:
        return value
    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    return {
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "intermediate": "medium",
        "high": "high",
    }.get(normalized, normalized)


def _normalize_hypothesis_confidence(raw: dict, key: str, *, max_items: int = 5) -> dict:
    normalized = dict(raw)
    raw_items = normalized.get(key) or []
    if not isinstance(raw_items, list):
        raise ValueError(f"{key} must be a JSON array of objects.")

    # The model contract explicitly caps hypothesis lists. Enforce that boundary
    # before validating individual items so any extra out-of-contract tail item
    # cannot invalidate an otherwise valid clinician-facing result.
    raw_items = raw_items[:max_items]

    items = []
    for index, item in enumerate(raw_items):
        if isinstance(item, dict):
            hypothesis = dict(item)
        elif isinstance(item, str):
            text = item.strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                preview = " ".join(text.split())[:220]
                raise ValueError(
                    f"{key}[{index}] must be a JSON object, but Qwen returned a string: {preview}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"{key}[{index}] decoded to {type(parsed).__name__}; expected a JSON object."
                )
            hypothesis = parsed
        else:
            raise ValueError(
                f"{key}[{index}] must be a JSON object; got {type(item).__name__}."
            )

        hypothesis["confidence"] = _normalize_confidence(hypothesis.get("confidence"))
        items.append(hypothesis)

    normalized[key] = items
    return normalized


def _normalize_intake_output(raw: dict) -> dict:
    normalized = dict(raw)

    normalized["sex"] = _normalize_enum(
        normalized.get("sex"),
        {
            "male": "male",
            "m": "male",
            "man": "male",
            "female": "female",
            "f": "female",
            "woman": "female",
            "other": "other",
            "non-binary": "other",
            "nonbinary": "other",
            "unknown": "unknown",
            "unspecified": "unknown",
        },
        "unknown",
    )

    symptoms = []
    for item in normalized.get("symptoms") or []:
        symptom = dict(item)
        symptom["severity"] = _normalize_enum(
            symptom.get("severity"),
            {
                "mild": "mild",
                "moderate": "moderate",
                "severe": "severe",
            },
            None,
        )
        symptoms.append(symptom)
    normalized["symptoms"] = symptoms

    medications = []
    for item in normalized.get("medications") or []:
        medication = dict(item)
        medication["status"] = _normalize_enum(
            medication.get("status"),
            {
                "active": "active",
                "current": "active",
                "currently taking": "active",
                "taking": "active",
                "stopped": "stopped",
                "discontinued": "stopped",
                "inactive": "stopped",
                "unknown": "unknown",
                "unspecified": "unknown",
            },
            "unknown",
        )
        medications.append(medication)
    normalized["medications"] = medications

    vitals = []
    for item in normalized.get("vitals") or []:
        vital = dict(item)
        vital["flag"] = _normalize_enum(
            vital.get("flag"),
            {
                "low": "low",
                "decreased": "low",
                "reduced": "low",
                "below normal": "low",
                "below range": "low",
                "high": "high",
                "elevated": "high",
                "raised": "high",
                "increased": "high",
                "above normal": "high",
                "above range": "high",
                "normal": "normal",
                "within normal limits": "normal",
                "within normal range": "normal",
                "wnl": "normal",
                "critical": "critical",
                "critically high": "critical",
                "critically low": "critical",
                "unknown": "unknown",
                "unspecified": "unknown",
                "not provided": "unknown",
            },
            "unknown",
        )
        vitals.append(vital)
    normalized["vitals"] = vitals

    labs = []
    for item in normalized.get("labs") or []:
        lab = dict(item)
        if lab.get("interpretation") is None:
            for alias in ("qualitative_value", "result", "result_text", "notes"):
                candidate = lab.get(alias)
                if candidate is not None and str(candidate).strip():
                    lab["interpretation"] = str(candidate).strip()
                    break
        lab["flag"] = _normalize_enum(
            lab.get("flag"),
            {
                "low": "low",
                "decreased": "low",
                "reduced": "low",
                "below normal": "low",
                "below range": "low",
                "high": "high",
                "elevated": "high",
                "raised": "high",
                "increased": "high",
                "above normal": "high",
                "above range": "high",
                "normal": "normal",
                "within normal limits": "normal",
                "within normal range": "normal",
                "wnl": "normal",
                "critical": "critical",
                "critically high": "critical",
                "critically low": "critical",
                "unknown": "unknown",
                "unspecified": "unknown",
                "not provided": "unknown",
            },
            "unknown",
        )
        labs.append(lab)
    normalized["labs"] = labs

    return normalized


def _parse_case(raw: dict, request: AnalyzeRequest) -> ClinicalCase:
    raw = _normalize_intake_output(raw)
    raw["case_id"] = f"CASE-{uuid.uuid4().hex[:12]}"
    raw["patient_reference"] = request.patient_reference or "synthetic-case"
    return ClinicalCase.model_validate(raw)


def _intake(request: AnalyzeRequest) -> ClinicalCase:
    raw = llm_json(
        system_prompt=(
            "Extract explicit facts from a synthetic/de-identified clinical note into structured fields. "
            "Do not infer diagnoses, treatments, normal values, or missing facts. Use empty arrays or nulls when absent. "
            "Use only these normalized enum values: sex=male|female|other|unknown; "
            "symptom severity=mild|moderate|severe|null; medication status=active|stopped|unknown; "
            "vital/lab flag=low|normal|high|critical|unknown. For example, map elevated/raised to high and decreased/reduced to low. "
            "Keep vital signs separate from laboratory tests. Put systolic blood pressure, diastolic blood pressure, heart rate, "
            "respiratory rate, oxygen saturation/SpO2, and temperature under vitals when explicitly present. "
            "Do not infer whether a numeric vital is normal or abnormal unless the note explicitly says so. "
            "For a qualitative lab result with no numeric value (for example 'ketones positive' or 'troponin not elevated'), "
            "set value to null and preserve the exact qualitative result in interpretation."
        ),
        user_payload=request.note,
        response_contract=json.dumps(
            {
                "age": 68,
                "sex": "male",
                "chief_complaint": "string",
                "symptoms": [
                    {
                        "name": "string",
                        "onset": None,
                        "severity": None,
                        "notes": None,
                    }
                ],
                "history": ["string"],
                "medications": [
                    {
                        "name": "string",
                        "dose": None,
                        "frequency": None,
                        "status": "unknown",
                    }
                ],
                "vitals": [
                    {
                        "test": "oxygen saturation",
                        "value": 87,
                        "unit": "%",
                        "reference_range": None,
                        "flag": "unknown",
                        "observed_at": None,
                    }
                ],
                "labs": [
                    {
                        "test": "string",
                        "value": 0,
                        "unit": None,
                        "reference_range": None,
                        "flag": "unknown",
                        "interpretation": None,
                        "observed_at": None,
                    }
                ],
                "imaging": [
                    {
                        "modality": "string",
                        "body_region": None,
                        "summary": "string",
                        "observed_at": None,
                    }
                ],
                "notes": ["string"],
            }
        ),
        max_output_tokens=4096,
    )
    return _parse_case(raw, request)


def _synthesize(case: ClinicalCase, source_note: str) -> Synthesis:
    timeline, abnormal, active, missing = _timeline_and_findings(case)
    raw = llm_json(
        system_prompt=(
            "Write one concise clinical problem representation for clinician decision support. "
            "Use only supplied facts. Do not make a diagnosis, recommend treatment, or invent numbers."
        ),
        user_payload={
            "case": case.model_dump(mode="json"),
            "code_derived_abnormal_findings": abnormal,
            "timeline": [event.model_dump(mode="json") for event in timeline],
        },
        response_contract='{"problem_representation":"string"}',
        max_output_tokens=3072,
    )
    representation = str(raw.get("problem_representation") or "").strip()
    if not representation:
        raise ValueError("Synthesis model returned an empty problem representation.")
    _validate_problem_numbers(case, source_note, representation)
    return Synthesis(
        problem_representation=representation,
        active_problems=active,
        abnormal_findings=abnormal,
        timeline=timeline,
        missing_information=missing,
    )


def _differential(case: ClinicalCase, synthesis: Synthesis, evidence, *, revision_context: dict | None = None) -> Differential:
    prompt = {
        "case": case.model_dump(mode="json"),
        "synthesis": synthesis.model_dump(mode="json"),
        "retrieved_evidence": [p.model_dump(mode="json") for p in evidence],
    }
    if revision_context:
        prompt["revision_context"] = revision_context
    raw = llm_json(
        system_prompt=(
            "Generate a ranked differential diagnosis for clinician review only. Evidence was retrieved before reasoning. "
            "Every hypothesis must cite one or more supplied passage_id values that actually influenced the claim. "
            "Include supporting evidence, opposing evidence, and missing information. Do not make a final diagnosis or prescribe treatment. "
            "Return at most 5 hypotheses. Keep each rationale to at most 2 short sentences, each evidence/missing list to at most 4 concise items, "
            "and unresolved_questions to at most 5 items. confidence must be low, medium, or high. "
            "hypotheses MUST be a JSON array of objects matching the contract; never return hypothesis names or prose as array strings."
        ),
        user_payload=prompt,
        response_contract=json.dumps(
            {
                "hypotheses": [
                    {
                        "rank": 1,
                        "name": "string",
                        "rationale": "string",
                        "supporting_evidence": ["string"],
                        "opposing_evidence": ["string"],
                        "missing_information": ["string"],
                        "confidence": "medium",
                        "citation_ids": ["passage_id"],
                    }
                ],
                "unresolved_questions": ["string"],
            }
        ),
        max_output_tokens=8192 if revision_context else 4096,
    )
    raw = _normalize_hypothesis_confidence(raw, "hypotheses", max_items=5)
    result = Differential.model_validate(raw)
    for index, hypothesis in enumerate(result.hypotheses, start=1):
        hypothesis.rank = index
    known = {p.passage_id for p in evidence}
    invalid = [(h.name, cid) for h in result.hypotheses for cid in h.citation_ids if cid not in known]
    if invalid:
        raise ValueError(f"Reasoning returned unknown citation IDs: {invalid}")
    if any(not h.citation_ids for h in result.hypotheses):
        raise ValueError("Every hypothesis must cite retrieved evidence.")
    return result


def _challenge(case: ClinicalCase, synthesis: Synthesis, evidence) -> Challenge:
    raw = llm_json(
        system_prompt=(
            "Act as an independent clinical challenge agent. Build your own differential from the raw case, synthesis, and retrieved evidence. "
            "You are blind to the primary reasoner's answer. Surface contradictions, missing discriminating questions, and high-risk alternatives. "
            "Do not diagnose or recommend treatment. Return at most 5 independent hypotheses and keep rationales concise. "
            "confidence must be low, medium, or high."
        ),
        user_payload={
            "case": case.model_dump(mode="json"),
            "synthesis": synthesis.model_dump(mode="json"),
            "retrieved_evidence": [p.model_dump(mode="json") for p in evidence],
        },
        response_contract=json.dumps(
            {
                "independent_hypotheses": [
                    {
                        "name": "string",
                        "rationale": "string",
                        "confidence": "low",
                    }
                ],
                "contradictions": ["string"],
                "missing_questions": ["string"],
                "high_risk_alternatives": ["string"],
                "summary": "string",
            }
        ),
        max_output_tokens=3072,
    )
    raw = _normalize_hypothesis_confidence(raw, "independent_hypotheses", max_items=5)
    return Challenge.model_validate({**raw, "disagreements": []})


def _similar(a: str, b: str) -> bool:
    a_norm = " ".join(re.findall(r"[a-z]+", a.lower()))
    b_norm = " ".join(re.findall(r"[a-z]+", b.lower()))
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= 0.68


def _compare(primary: Differential, challenge: Challenge) -> list[str]:
    disagreements: list[str] = []
    primary_names = [h.name for h in primary.hypotheses]
    challenge_names = [h.name for h in challenge.independent_hypotheses]
    for name in challenge_names:
        if not any(_similar(name, other) for other in primary_names):
            disagreements.append(f"Independent challenge surfaced '{name}', absent from the primary differential.")
    for name in primary_names:
        if not any(_similar(name, other) for other in challenge_names):
            disagreements.append(f"Primary differential contains '{name}', not independently reproduced by the challenge.")
    for name in challenge.high_risk_alternatives:
        if not any(_similar(name, other) for other in primary_names):
            disagreements.append(f"High-risk alternative '{name}' is not represented in the primary differential.")
    return list(dict.fromkeys(disagreements))


def _next_steps(case: ClinicalCase, synthesis: Synthesis, differential: Differential, evidence) -> list[NextStep]:
    raw = llm_json(
        system_prompt=(
            "Recommend only diagnostic information-gathering steps: questions, observations, or tests. "
            "For each step say which hypotheses it helps distinguish and why. Do not recommend treatment, medications, or dosing. "
            "Citations must use only supplied passage_id values. Return 3 to 5 concise suggestions. "
            "kind must be question, observation, or test."
        ),
        user_payload={
            "case": case.model_dump(mode="json"),
            "synthesis": synthesis.model_dump(mode="json"),
            "differential": differential.model_dump(mode="json"),
            "retrieved_evidence": [p.model_dump(mode="json") for p in evidence],
        },
        response_contract=json.dumps(
            {
                "suggestions": [
                    {
                        "action": "string",
                        "kind": "test",
                        "distinguishes_between": ["hypothesis A", "hypothesis B"],
                        "rationale": "string",
                        "citation_ids": ["passage_id"],
                    }
                ]
            }
        ),
        max_output_tokens=4096,
    )
    suggestions = [NextStep.model_validate(item) for item in raw.get("suggestions", [])]
    if not suggestions:
        raise ValueError("Next-step agent returned no suggestions.")
    known = {p.passage_id for p in evidence}
    invalid = [(step.action, cid) for step in suggestions for cid in step.citation_ids if cid not in known]
    if invalid:
        raise ValueError(f"Next-step agent returned unknown citation IDs: {invalid}")
    return suggestions


def run_analysis(request: AnalyzeRequest, emit=None) -> AnalysisResult:
    case = _stage("Intake Agent", lambda: _intake(request), emit)
    synthesis = _stage(
        "Clinical Synthesis Agent",
        lambda: _synthesize(case, request.note),
        emit,
    )
    evidence = _stage("Evidence Agent", lambda: retrieve_evidence(case, synthesis), emit)
    initial = _stage("Reasoning Agent", lambda: _differential(case, synthesis, evidence), emit)
    challenge = _stage("Challenge Agent", lambda: _challenge(case, synthesis, evidence), emit)
    challenge.disagreements = _compare(initial, challenge)
    revised = _stage(
        "Differential Revision",
        lambda: _differential(
            case,
            synthesis,
            evidence,
            revision_context={
                "initial_differential": initial.model_dump(mode="json"),
                "blind_challenge": challenge.model_dump(mode="json"),
                "instruction": "Re-rank or revise the differential in light of the independent challenge while staying grounded in the record and retrieved evidence.",
            },
        ),
        emit,
    )
    next_steps = _stage(
        "Next-Best-Step Agent",
        lambda: _next_steps(case, synthesis, revised, evidence),
        emit,
    )
    safety = _stage(
        "Safety Gate",
        lambda: evaluate_safety(case, revised, evidence, next_steps),
        emit,
    )
    model = get_llm_config().id
    result = AnalysisResult(
        run_id=f"RUN-{uuid.uuid4().hex[:12]}",
        case=case,
        synthesis=synthesis,
        retrieved_evidence=evidence,
        initial_differential=initial,
        challenge=challenge,
        revised_differential=revised,
        next_steps=next_steps,
        safety=safety,
        model=model,
    )
    save_run(
        result.run_id,
        result.created_at.isoformat(),
        result.model,
        result.safety.status,
        result.model_dump(mode="json"),
    )
    return result
