from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from app.pipeline import run_analysis
from app.schemas import AnalyzeRequest
from app.store import init_db


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "eval_cases.json"


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def diagnosis_matches(candidate: str, aliases: list[str]) -> bool:
    candidate_norm = _normalize(candidate)
    candidate_tokens = set(candidate_norm.split())

    for alias in aliases:
        alias_norm = _normalize(alias)
        if not alias_norm:
            continue
        if alias_norm in candidate_norm or candidate_norm in alias_norm:
            return True

        ratio = SequenceMatcher(None, candidate_norm, alias_norm).ratio()
        alias_tokens = set(alias_norm.split())
        union = candidate_tokens | alias_tokens
        jaccard = len(candidate_tokens & alias_tokens) / len(union) if union else 0.0

        if ratio >= 0.72 or jaccard >= 0.60:
            return True
    return False


def _citation_integrity(result) -> bool:
    known = {passage.passage_id for passage in result.retrieved_evidence}
    if not known:
        return False

    for differential in (result.initial_differential, result.revised_differential):
        for hypothesis in differential.hypotheses:
            if not hypothesis.citation_ids:
                return False
            if any(citation not in known for citation in hypothesis.citation_ids):
                return False

    for step in result.next_steps:
        if any(citation not in known for citation in step.citation_ids):
            return False

    return True


def evaluate_case(case_index: int) -> dict:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    case = cases[case_index]
    aliases = [
        case["must_consider"]["label"],
        *case["must_consider"].get("aliases", []),
    ]

    base = {
        "case_index": case_index,
        "case_id": case["id"],
        "category": case["category"],
        "must_consider": case["must_consider"]["label"],
        "trap": case["trap"],
        "expected_red_flag": bool(case["expected_red_flag"]),
        "expected_evidence_sources": case.get("expected_evidence_sources", []),
        "pipeline_completed": False,
        "initial_top3_hit": False,
        "revised_top3_hit": False,
        "red_flag_detected": False,
        "red_flag_correct": False,
        "challenge_adds_something": False,
        "citation_integrity": False,
        "evidence_domain_hit": False,
        "safety_status": None,
        "initial_top3": [],
        "revised_top3": [],
        "retrieved_source_ids": [],
        "error": None,
    }

    try:
        result = run_analysis(
            AnalyzeRequest(
                note=case["note"],
                patient_reference=f"eval-{case['id']}",
                confirm_synthetic=True,
            )
        )

        initial_names = [h.name for h in result.initial_differential.hypotheses[:3]]
        revised_names = [h.name for h in result.revised_differential.hypotheses[:3]]
        retrieved_sources = sorted({p.source_id for p in result.retrieved_evidence})

        red_flag_detected = bool(result.safety.urgent_warning)
        expected_sources = set(case.get("expected_evidence_sources", []))

        base.update(
            {
                "pipeline_completed": True,
                "initial_top3_hit": any(diagnosis_matches(name, aliases) for name in initial_names),
                "revised_top3_hit": any(diagnosis_matches(name, aliases) for name in revised_names),
                "red_flag_detected": red_flag_detected,
                "red_flag_correct": red_flag_detected == bool(case["expected_red_flag"]),
                "challenge_adds_something": bool(
                    result.challenge.high_risk_alternatives
                    or result.challenge.missing_questions
                    or result.challenge.contradictions
                ),
                "citation_integrity": _citation_integrity(result),
                "evidence_domain_hit": bool(expected_sources & set(retrieved_sources)),
                "safety_status": result.safety.status,
                "initial_top3": initial_names,
                "revised_top3": revised_names,
                "retrieved_source_ids": retrieved_sources,
            }
        )
    except Exception as exc:
        base["error"] = f"{type(exc).__name__}: {str(exc)[:1200]}"

    return base


def main() -> None:
    init_db()
    case_index = int(os.environ["EVAL_CASE_INDEX"])
    result = evaluate_case(case_index)

    output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", "eval_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"case_{case_index:02d}_{result['case_id']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    status = "PASS" if result["pipeline_completed"] else "PIPELINE_FAILED"
    print(
        json.dumps(
            {
                "case": result["case_id"],
                "status": status,
                "initial_top3_hit": result["initial_top3_hit"],
                "revised_top3_hit": result["revised_top3_hit"],
                "red_flag_correct": result["red_flag_correct"],
                "challenge_adds_something": result["challenge_adds_something"],
                "citation_integrity": result["citation_integrity"],
                "evidence_domain_hit": result["evidence_domain_hit"],
                "error": result["error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
