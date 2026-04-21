"""Run the curated evaluation set against a running Plato RAG service."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

from plato_rag.api.contracts import QueryRequest, QueryResponse
from plato_rag.evaluation import evaluate_case_response, load_dataset
from plato_rag.evaluation.dataset import EvaluationCase
from plato_rag.generation.service import GenerationService


class _FixtureLLM:
    def __init__(self, raw_output: str) -> None:
        self._raw_output = raw_output

    async def generate(self, messages: list[object]) -> str:
        del messages
        return self._raw_output

    def model_name(self) -> str:
        return "evaluation-fixture-llm"


@asynccontextmanager
async def _in_process_client(timeout_seconds: float) -> tuple[httpx.AsyncClient, FastAPI]:
    from plato_rag.main import app

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://plato-rag.local",
            timeout=timeout_seconds,
        ) as client:
            yield client, app


def _request_for_case(case: EvaluationCase) -> QueryRequest:
    return QueryRequest(
        question=case.question,
        mode=case.mode,
        conversation_history=case.conversation_history,
        options=case.options.model_copy(update={"include_debug": True}),
    )


async def _execute_case_against_service(
    client: httpx.AsyncClient,
    case: EvaluationCase,
    *,
    base_url: str,
) -> QueryResponse:
    if case.generation_fixture is not None:
        msg = (
            f"case {case.id} requires an in-process evaluation run because it defines "
            "a generation fixture"
        )
        raise ValueError(msg)

    response = await client.post(
        f"{base_url.rstrip('/')}/v1/query",
        json=_request_for_case(case).model_dump(mode="json"),
    )
    response.raise_for_status()
    return QueryResponse.model_validate(response.json())


async def _execute_case_in_process(
    client: httpx.AsyncClient,
    app: FastAPI,
    case: EvaluationCase,
) -> QueryResponse:
    original_generation_service = app.state.generation_service
    if case.generation_fixture is not None:
        app.state.generation_service = GenerationService(
            llm=_FixtureLLM(case.generation_fixture.raw_output)
        )

    try:
        response = await client.post(
            "/v1/query",
            json=_request_for_case(case).model_dump(mode="json"),
        )
        response.raise_for_status()
        return QueryResponse.model_validate(response.json())
    finally:
        app.state.generation_service = original_generation_service


async def _run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/evaluation/public_seed.yaml",
        help="Path to the evaluation dataset YAML file.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--base-url",
        help="Base URL of the running Plato RAG service, for example http://localhost:8001.",
    )
    target_group.add_argument(
        "--in-process",
        action="store_true",
        help="Run against the local FastAPI app with lifespan enabled and optional fixtures.",
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

    case_reports: list[dict[str, Any]] = []

    if args.in_process:
        async with _in_process_client(args.timeout_seconds) as (client, app):
            for case in selected_cases:
                try:
                    query_response = await _execute_case_in_process(client, app, case)
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
    else:
        async with httpx.AsyncClient(timeout=args.timeout_seconds) as client:
            for case in selected_cases:
                try:
                    query_response = await _execute_case_against_service(
                        client,
                        case,
                        base_url=args.base_url,
                    )
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
