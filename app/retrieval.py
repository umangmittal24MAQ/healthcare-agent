from __future__ import annotations

import math
import re
from typing import Iterable

from app.config import get_settings, load_json_file
from app.llm import llm_json
from app.schemas import ClinicalCase, EvidencePassage, Synthesis

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


def _flatten_corpus() -> list[dict]:
    settings = get_settings()
    docs = load_json_file(settings.clinical_corpus_path)
    passages: list[dict] = []
    for doc in docs:
        for index, text in enumerate(doc.get("passages", []), start=1):
            passages.append(
                {
                    "passage_id": f"{doc['source_id']}-P{index}",
                    "source_id": doc["source_id"],
                    "source_title": doc["title"],
                    "text": text,
                    "tags": doc.get("tags", []),
                }
            )
    if not passages:
        raise ValueError("Clinical corpus is empty.")
    return passages


def _score(query: set[str], passage: dict) -> float:
    haystack = _tokens(passage["text"] + " " + " ".join(passage.get("tags", [])))
    if not query or not haystack:
        return 0.0
    overlap = len(query & haystack)
    coverage = overlap / len(query)
    density = overlap / len(haystack)
    return round(0.75 * coverage + 0.25 * math.sqrt(density), 6)


def _query_text(case: ClinicalCase, synthesis: Synthesis) -> str:
    fields: Iterable[str] = [
        case.chief_complaint,
        *(s.name for s in case.symptoms),
        *case.history,
        *(f"{vital.test} {vital.value} {vital.flag}" for vital in case.vitals),
        *(
            f"{lab.test} {lab.value if lab.value is not None else ''} "
            f"{lab.interpretation or ''} {lab.flag}"
            for lab in case.labs
        ),
        *(img.summary for img in case.imaging),
        synthesis.problem_representation,
        *synthesis.abnormal_findings,
    ]
    return " ".join(fields)


def retrieve_evidence(case: ClinicalCase, synthesis: Synthesis) -> list[EvidencePassage]:
    settings = get_settings()
    query = _tokens(_query_text(case, synthesis))
    scored = [(_score(query, item), item) for item in _flatten_corpus()]
    scored.sort(key=lambda item: (-item[0], item[1]["passage_id"]))
    candidates = scored[: settings.retrieval_candidates]

    candidate_payload = [
        {
            "passage_id": item["passage_id"],
            "source_title": item["source_title"],
            "text": item["text"],
            "keyword_score": score,
        }
        for score, item in candidates
    ]
    raw = llm_json(
        system_prompt=(
            "You rerank already-retrieved clinical reference passages for relevance to a synthetic patient record. "
            "You may only return passage_id values supplied in candidates. Do not add medical facts."
        ),
        user_payload={
            "problem_representation": synthesis.problem_representation,
            "abnormal_findings": synthesis.abnormal_findings,
            "candidates": candidate_payload,
            "top_k": settings.retrieval_top_k,
        },
        response_contract='{"ordered_ids": ["passage-id", "..."]}',
        max_output_tokens=2048,
    )
    ordered_ids = raw.get("ordered_ids")
    if not isinstance(ordered_ids, list) or not ordered_ids:
        raise ValueError("Evidence reranker returned no passage IDs.")

    by_id = {item["passage_id"]: (score, item) for score, item in candidates}
    requested_ids = [str(pid) for pid in ordered_ids]
    unknown = [pid for pid in requested_ids if pid not in by_id]

    # Never accept a hallucinated passage ID. If Qwen returns a mix of valid
    # candidate IDs and one malformed/unknown ID, keep only the verified IDs
    # instead of throwing away an otherwise usable reranking. If every returned
    # ID is invalid, fail visibly because there is no model-verified reranking.
    valid_ids: list[str] = []
    seen: set[str] = set()
    for pid in requested_ids:
        if pid in by_id and pid not in seen:
            valid_ids.append(pid)
            seen.add(pid)

    if not valid_ids:
        raise ValueError(
            "Evidence reranker returned no valid candidate IDs"
            + (f"; unknown IDs: {unknown}" if unknown else ".")
        )

    result: list[EvidencePassage] = []
    for pid in valid_ids[: settings.retrieval_top_k]:
        score, item = by_id[pid]
        result.append(
            EvidencePassage(
                passage_id=pid,
                source_id=item["source_id"],
                source_title=item["source_title"],
                text=item["text"],
                retrieval_score=score,
            )
        )
    return result
