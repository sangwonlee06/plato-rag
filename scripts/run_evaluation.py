"""Run the curated evaluation set against a running Plato RAG service."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from plato_rag.api.contracts import QueryRequest, QueryResponse
from plato_rag.evaluation import evaluate_case_response, load_dataset


async def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/evaluation/public_seed.yaml",
        help="Path to the evaluation dataset YAML file.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the running Plato RAG service, for example http://localhost:8001.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only specific case ids. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Run only cases containing a given tag. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a JSON report.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout per request.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    selected_cases = [
        case
        for case in dataset.cases
        if (not args.case_id or case.id in args.case_id)
        and (not args.tag or any(tag in case.tags for tag in args.tag))
    ]
    if not selected_cases:
        print("No evaluation cases selected.")
        return 1

    base_url = args.base_url.rstrip("/")
    case_reports: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
        for case in selected_cases:
            request = QueryRequest(
                question=case.question,
                mode=case.mode,
                conversation_history=case.conversation_history,
                options=case.options.model_copy(update={"include_debug": True}),
            )
            try:
                response = await client.post(
                    f"{base_url}/v1/query",
                    json=request.model_dump(mode="json"),
                )
                response.raise_for_status()
                query_response = QueryResponse.model_validate(response.json())
                result = evaluate_case_response(case, query_response)
                report = {
                    "case_id": result.case_id,
                    "passed": result.passed,
                    "failures": result.failures,
                    "citation_count": result.citation_count,
                    "retrieved_chunk_count": result.retrieved_chunk_count,
                    "ungrounded_citation_count": result.ungrounded_citation_count,
                    "unsupported_claim_count": result.unsupported_claim_count,
                }
            except Exception as exc:
                report = {
                    "case_id": case.id,
                    "passed": False,
                    "failures": [f"request failed: {exc}"],
                    "citation_count": 0,
                    "retrieved_chunk_count": 0,
                    "ungrounded_citation_count": 0,
                    "unsupported_claim_count": 0,
                }
            case_reports.append(report)

    passed = sum(1 for report in case_reports if report["passed"])
    failed = len(case_reports) - passed
    summary = {
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "total_cases": len(case_reports),
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": passed / len(case_reports),
    }

    for report in case_reports:
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"{status} {report['case_id']} "
            f"(retrieved={report['retrieved_chunk_count']}, citations={report['citation_count']})"
        )
        for failure in report["failures"]:
            print(f"  - {failure}")

    print(
        "Summary: "
        f"{summary['passed_cases']}/{summary['total_cases']} passed "
        f"({summary['pass_rate']:.0%})"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            json.dumps({"summary": summary, "cases": case_reports}, indent=2),
            encoding="utf-8",
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
