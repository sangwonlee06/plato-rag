from __future__ import annotations

import httpx
import pytest

from plato_rag.resilience import is_retryable_exception, retry_async


@pytest.mark.asyncio
async def test_retry_async_retries_transient_failures() -> None:
    attempts = 0

    async def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ReadTimeout("timed out")
        return "ok"

    result = await retry_async(
        "flaky operation",
        flaky_operation,
        max_attempts=3,
        initial_backoff_seconds=0.0,
        max_backoff_seconds=0.0,
    )

    assert result == "ok"
    assert attempts == 3


@pytest.mark.asyncio
async def test_retry_async_does_not_retry_non_transient_failures() -> None:
    attempts = 0

    async def bad_operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await retry_async(
            "bad operation",
            bad_operation,
            max_attempts=3,
            initial_backoff_seconds=0.0,
            max_backoff_seconds=0.0,
        )

    assert attempts == 1


def test_is_retryable_exception_accepts_server_http_status_errors() -> None:
    request = httpx.Request("GET", "https://example.test")
    response = httpx.Response(503, request=request)
    exc = httpx.HTTPStatusError("server error", request=request, response=response)

    assert is_retryable_exception(exc) is True
