from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    note: str = Field(min_length=20, max_length=50000)
    patient_reference: str | None = Field(default=None, max_length=64)
    confirm_synthetic: Literal[True]


class Symptom(BaseModel):
    name: str
    onset: str | None = None
    severity: Literal["mild", "moderate", "severe"] | None = None
    notes: str | None = None


class Medication(BaseModel):
    name: str
    dose: str | None = None
    frequency: str | None = None
    status: Literal["active", "stopped", "unknown"] = "unknown"


class LabResult(BaseModel):
    test: str
    value: float | str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: Literal["low", "normal", "high", "critical", "unknown"] = "unknown"
    interpretation: str | None = None
    observed_at: datetime | None = None


class VitalSign(BaseModel):
    test: str
    value: float | str
    unit: str | None = None
    reference_range: str | None = None
    flag: Literal["low", "normal", "high", "critical", "unknown"] = "unknown"
    observed_at: datetime | None = None


class ImagingSummary(BaseModel):
    modality: str
    body_region: str | None = None
    summary: str
    observed_at: datetime | None = None


class ClinicalCase(BaseModel):
    case_id: str
    patient_reference: str
    age: int = Field(ge=0, le=120)
    sex: Literal["female", "male", "other", "unknown"] = "unknown"
    chief_complaint: str
    symptoms: list[Symptom] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    vitals: list[VitalSign] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    imaging: list[ImagingSummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    event_type: Literal["symptom", "vital", "lab", "imaging", "history", "medication"]
    label: str
    detail: str
    observed_at: datetime | None = None


class Synthesis(BaseModel):
    problem_representation: str
    active_problems: list[str]
    abnormal_findings: list[str]
    timeline: list[TimelineEvent]
    missing_information: list[str]


class EvidencePassage(BaseModel):
    passage_id: str
    source_id: str
    source_title: str
    text: str
    retrieval_score: float


class Hypothesis(BaseModel):
    rank: int = Field(ge=1)
    name: str
    rationale: str
    supporting_evidence: list[str] = Field(default_factory=list)
    opposing_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    citation_ids: list[str] = Field(default_factory=list)


class Differential(BaseModel):
    hypotheses: list[Hypothesis]
    unresolved_questions: list[str] = Field(default_factory=list)


class IndependentHypothesis(BaseModel):
    name: str
    rationale: str
    confidence: Literal["low", "medium", "high"]


class Challenge(BaseModel):
    independent_hypotheses: list[IndependentHypothesis]
    contradictions: list[str] = Field(default_factory=list)
    missing_questions: list[str] = Field(default_factory=list)
    high_risk_alternatives: list[str] = Field(default_factory=list)
    summary: str
    disagreements: list[str] = Field(default_factory=list)


class NextStep(BaseModel):
    action: str
    kind: Literal["question", "observation", "test"]
    distinguishes_between: list[str] = Field(default_factory=list)
    rationale: str
    citation_ids: list[str] = Field(default_factory=list)


class SafetyCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class SafetyResult(BaseModel):
    status: Literal["pass", "pass_with_warnings", "escalate", "block"]
    checks: list[SafetyCheck]
    warnings: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    urgent_warning: str | None = None


class ReviewRequest(BaseModel):
    decision: Literal["accepted", "rejected"]
    comment: str | None = Field(default=None, max_length=2000)


class AnalysisResult(BaseModel):
    run_id: str
    case: ClinicalCase
    synthesis: Synthesis
    retrieved_evidence: list[EvidencePassage]
    initial_differential: Differential
    challenge: Challenge
    revised_differential: Differential
    next_steps: list[NextStep]
    safety: SafetyResult
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
