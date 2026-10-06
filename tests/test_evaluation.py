from pathlib import Path

from scripts.aggregate_evaluation import aggregate
from scripts.evaluate_case import diagnosis_matches


def test_diagnosis_alias_matching_handles_common_variants():
    assert diagnosis_matches(
        "Acute decompensated heart failure (HFrEF exacerbation)",
        ["acute decompensated heart failure", "heart failure exacerbation"],
    )
    assert diagnosis_matches(
        "Pulmonary embolism as precipitant",
        ["pulmonary embolism", "pulmonary thromboembolism"],
    )
    assert not diagnosis_matches(
        "Community-acquired pneumonia",
        ["acute ischemic stroke", "cerebral infarction"],
    )


def test_eval_case_set_contains_twenty_unique_cases():
    import json

    path = Path(__file__).resolve().parents[1] / "data" / "eval_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))

    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == 20
    assert all(case["must_consider"]["label"] for case in cases)
    assert all(case["must_consider"]["aliases"] for case in cases)
    assert all(case["expected_evidence_sources"] for case in cases)


def test_aggregate_counts_failed_pipeline_as_metric_miss(tmp_path):
    import json

    completed = {
        "case_index": 0,
        "case_id": "hf_decompensation",
        "category": "cardiopulmonary",
        "pipeline_completed": True,
        "initial_top3_hit": True,
        "revised_top3_hit": True,
        "red_flag_correct": True,
        "challenge_adds_something": True,
        "citation_integrity": True,
        "evidence_domain_hit": True,
        "error": None,
    }
    failed = {
        "case_index": 1,
        "case_id": "pneumonia_sepsis",
        "category": "infectious-respiratory",
        "pipeline_completed": False,
        "initial_top3_hit": False,
        "revised_top3_hit": False,
        "red_flag_correct": False,
        "challenge_adds_something": False,
        "citation_integrity": False,
        "evidence_domain_hit": False,
        "error": "PipelineStageError: synthetic failure",
    }

    (tmp_path / "case_00_hf_decompensation.json").write_text(
        json.dumps(completed), encoding="utf-8"
    )
    (tmp_path / "case_01_pneumonia_sepsis.json").write_text(
        json.dumps(failed), encoding="utf-8"
    )

    summary, markdown = aggregate(tmp_path)

    assert summary["received_results"] == 2
    assert summary["pipeline_completed"] == 1
    assert "pneumonia_sepsis" in summary["failed_cases"]
    assert "Pipeline failures" in markdown
