from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "eval_cases.json"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def aggregate(results_dir: Path) -> tuple[dict, str]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    expected_count = len(cases)

    rows = []
    for path in sorted(results_dir.glob("case_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))

    completed = [row for row in rows if row.get("pipeline_completed")]
    completed_count = len(completed)

    summary = {
        "expected_cases": expected_count,
        "received_results": len(rows),
        "pipeline_completed": completed_count,
        "completion_rate": _rate(completed_count, expected_count),
        "initial_top3_hits": sum(bool(row.get("initial_top3_hit")) for row in rows),
        "initial_top3_rate": _rate(sum(bool(row.get("initial_top3_hit")) for row in rows), expected_count),
        "revised_top3_hits": sum(bool(row.get("revised_top3_hit")) for row in rows),
        "revised_top3_rate": _rate(sum(bool(row.get("revised_top3_hit")) for row in rows), expected_count),
        "red_flag_correct": sum(bool(row.get("red_flag_correct")) for row in rows),
        "red_flag_accuracy": _rate(sum(bool(row.get("red_flag_correct")) for row in rows), expected_count),
        "challenge_adds_something": sum(bool(row.get("challenge_adds_something")) for row in rows),
        "challenge_add_rate": _rate(sum(bool(row.get("challenge_adds_something")) for row in rows), expected_count),
        "citation_integrity_passes": sum(bool(row.get("citation_integrity")) for row in rows),
        "citation_integrity_rate": _rate(sum(bool(row.get("citation_integrity")) for row in rows), expected_count),
        "evidence_domain_hits": sum(bool(row.get("evidence_domain_hit")) for row in rows),
        "evidence_domain_hit_rate": _rate(sum(bool(row.get("evidence_domain_hit")) for row in rows), expected_count),
        "failed_cases": [row["case_id"] for row in rows if not row.get("pipeline_completed")],
    }

    lines = [
        "# Diagnostic Decision Support — 20-case evaluation",
        "",
        "This report is generated from synthetic cases and is a prototype engineering evaluation, not clinical validation.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Pipeline completion | {summary['pipeline_completed']}/{expected_count} ({summary['completion_rate']:.0%}) |",
        f"| Must-consider diagnosis in initial top 3 | {summary['initial_top3_hits']}/{expected_count} ({summary['initial_top3_rate']:.0%}) |",
        f"| Must-consider diagnosis in revised top 3 | {summary['revised_top3_hits']}/{expected_count} ({summary['revised_top3_rate']:.0%}) |",
        f"| Configured red-flag behavior correct | {summary['red_flag_correct']}/{expected_count} ({summary['red_flag_accuracy']:.0%}) |",
        f"| Challenge adds contradiction/missing question/high-risk alternative | {summary['challenge_adds_something']}/{expected_count} ({summary['challenge_add_rate']:.0%}) |",
        f"| Citation integrity | {summary['citation_integrity_passes']}/{expected_count} ({summary['citation_integrity_rate']:.0%}) |",
        f"| Expected evidence domain retrieved | {summary['evidence_domain_hits']}/{expected_count} ({summary['evidence_domain_hit_rate']:.0%}) |",
        "",
        "## Case-by-case",
        "",
        "| Case | Category | Complete | Initial top-3 | Revised top-3 | Red flag | Challenge adds | Citations | Evidence domain |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in sorted(rows, key=lambda x: x["case_index"]):
        mark = lambda value: "✅" if value else "❌"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["category"],
                    mark(row.get("pipeline_completed")),
                    mark(row.get("initial_top3_hit")),
                    mark(row.get("revised_top3_hit")),
                    mark(row.get("red_flag_correct")),
                    mark(row.get("challenge_adds_something")),
                    mark(row.get("citation_integrity")),
                    mark(row.get("evidence_domain_hit")),
                ]
            )
            + " |"
        )

    failures = [row for row in rows if row.get("error")]
    if failures:
        lines.extend(["", "## Pipeline failures", ""])
        for row in failures:
            lines.append(f"- **{row['case_id']}** — {row['error']}")

    if len(rows) != expected_count:
        lines.extend(
            [
                "",
                f"> Warning: expected {expected_count} result artifacts but received {len(rows)}.",
            ]
        )

    return summary, "\n".join(lines) + "\n"


def main() -> None:
    results_dir = Path(os.getenv("EVAL_RESULTS_DIR", "eval_results"))
    summary, markdown = aggregate(results_dir)

    Path("eval_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("eval_report.md").write_text(markdown, encoding="utf-8")

    print(markdown)

    github_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)

    enforce = os.getenv("ENFORCE_THRESHOLDS", "false").lower() == "true"
    if enforce:
        thresholds = {
            "completion_rate": 0.80,
            "revised_top3_rate": 0.75,
            "red_flag_accuracy": 0.90,
            "challenge_add_rate": 0.75,
            "citation_integrity_rate": 0.90,
            "evidence_domain_hit_rate": 0.75,
        }
        failures = [
            f"{key}={summary[key]:.2%} < {value:.0%}"
            for key, value in thresholds.items()
            if summary[key] < value
        ]
        if failures:
            raise SystemExit("Evaluation thresholds failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
