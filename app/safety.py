from __future__ import annotations

import operator
import re
from typing import Iterable

from app.config import get_settings, load_json_file
from app.schemas import ClinicalCase, Differential, EvidencePassage, NextStep, SafetyCheck, SafetyResult

_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
}


def _numeric_values(case: ClinicalCase) -> set[float]:
    values = {float(case.age)}
    for measurement in [*case.vitals, *case.labs]:
        try:
            values.add(float(measurement.value))
        except (TypeError, ValueError):
            pass
    return values


def _numeric_grounding(case: ClinicalCase, differential: Differential) -> list[str]:
    allowed = _numeric_values(case)
    warnings: list[str] = []
    measurement_words = (
        "temperature", "heart rate", "respiratory", "oxygen", "saturation", "lactate", "wbc", "leuk", "crp", "blood pressure", "age"
    )
    for hypothesis in differential.hypotheses:
        for sentence in hypothesis.supporting_evidence:
            if not any(word in sentence.lower() for word in measurement_words):
                continue
            for token in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", sentence):
                value = float(token)
                if not any(abs(value - known) < 1e-9 for known in allowed):
                    warnings.append(f"{hypothesis.name}: numeric evidence {token} is not present in the record.")
    return warnings


def _contains_treatment_order(texts: Iterable[str]) -> bool:
    order = re.compile(r"\b(start|give|administer|prescribe|initiate|treat\s+with|increase|decrease)\b", re.I)
    dose = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|grams?|g|ml|units?)\b", re.I)
    negation = re.compile(r"\b(do not|don't|must not|should not|avoid|without)\b", re.I)
    for text in texts:
        if order.search(text) and not negation.search(text):
            return True
        if dose.search(text) and not negation.search(text):
            return True
    return False


def _red_flags(case: ClinicalCase) -> list[str]:
    rules = load_json_file(get_settings().safety_rules_path).get("red_flags", [])
    findings: list[str] = []
    for measurement in [*case.vitals, *case.labs]:
        try:
            value = float(measurement.value)
        except (TypeError, ValueError):
            continue
        normalized_name = measurement.test.strip().lower()
        for rule in rules:
            names = {str(x).strip().lower() for x in rule.get("test_names", [])}
            compare = _OPERATORS.get(str(rule.get("operator")))
            threshold = rule.get("threshold")
            if normalized_name in names and compare and isinstance(threshold, (int, float)) and compare(value, float(threshold)):
                findings.append(
                    f"{rule.get('message', rule.get('id', 'configured red flag'))}: "
                    f"{measurement.test}={measurement.value}{(' ' + measurement.unit) if measurement.unit else ''}"
                )
    return findings


def evaluate_safety(
    case: ClinicalCase,
    differential: Differential,
    evidence: list[EvidencePassage],
    next_steps: list[NextStep],
) -> SafetyResult:
    known_ids = {p.passage_id for p in evidence}
    invalid_citations = [
        h.name
        for h in differential.hypotheses
        if not h.citation_ids or any(citation not in known_ids for citation in h.citation_ids)
    ]
    ranking_ok = [h.rank for h in differential.hypotheses] == list(range(1, len(differential.hypotheses) + 1))
    definitive_terms = ("confirmed diagnosis", "definitively diagnosed", "diagnosis is", "certainly has")
    advisory_ok = all(not any(term in h.rationale.lower() for term in definitive_terms) for h in differential.hypotheses)
    numeric_warnings = _numeric_grounding(case, differential)

    all_text: list[str] = []
    for h in differential.hypotheses:
        all_text.extend([h.name, h.rationale, *h.supporting_evidence, *h.opposing_evidence, *h.missing_information])
    for step in next_steps:
        all_text.extend([step.action, step.rationale, *step.distinguishes_between])
    treatment_safe = not _contains_treatment_order(all_text)

    blockers: list[str] = []
    if not differential.hypotheses:
        blockers.append("No differential hypotheses were produced.")
    if not ranking_ok:
        blockers.append("Differential ranks are incomplete or non-consecutive.")
    if invalid_citations:
        blockers.append("Invalid or missing evidence citations: " + ", ".join(invalid_citations))
    if not advisory_ok:
        blockers.append("Definitive-diagnosis language was detected.")
    if not treatment_safe:
        blockers.append("Treatment or dosing language was detected.")

    flags = _red_flags(case)
    if blockers:
        status = "block"
    elif flags:
        status = "escalate"
    elif numeric_warnings:
        status = "pass_with_warnings"
    else:
        status = "pass"

    return SafetyResult(
        status=status,
        checks=[
            SafetyCheck(name="ranked_differential", passed=ranking_ok, detail="Ranks must be consecutive from 1."),
            SafetyCheck(name="citation_integrity", passed=not invalid_citations, detail="Every hypothesis must cite only retrieved passage IDs."),
            SafetyCheck(name="advisory_language", passed=advisory_ok, detail="Output must avoid definitive-diagnosis claims."),
            SafetyCheck(name="no_treatment_orders", passed=treatment_safe, detail="Output must not prescribe treatment or dosing."),
            SafetyCheck(name="numeric_grounding", passed=not numeric_warnings, detail="Quoted clinical measurements must exist in the source record."),
        ],
        warnings=numeric_warnings,
        blocked_reasons=blockers,
        urgent_warning=(
            "URGENT CLINICIAN REVIEW: configured high-risk finding(s) detected: " + "; ".join(flags)
            if flags
            else None
        ),
    )
