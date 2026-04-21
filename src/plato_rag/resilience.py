"""Retry helpers for transient operational failures."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_exception(exc: BaseException) -> bool:
    """Return True when an exception looks transient and worth retrying."""

    if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
        return True

    name = type(exc).__name__.casefold()
    return any(
        token in name
        for token in (
            "timeout",
            "ratelimit",
            "apiconnection",
            "connectionerror",
            "overloaded",
            "serviceunavailable",
            "internalserver",
        )
    )


async def retry_async(
    operation_name: str,
    operation: Callable[[], Awaitable[object]],
    *,
    max_attempts: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    should_retry: Callable[[BaseException], bool] = is_retryable_exception,
) -> object:
    """Run an async operation with bounded retries on transient failures."""

    if max_attempts < 1:
        msg = "max_attempts must be at least 1"
        raise ValueError(msg)

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            is_last_attempt = attempt == max_attempts
            if is_last_attempt or not should_retry(exc):
                raise

            delay_seconds = min(
                initial_backoff_seconds * (2 ** (attempt - 1)),
                max_backoff_seconds,
            )
            logger.warning(
                "Transient failure during %s (attempt %d/%d): %s; retrying in %.2fs",
                operation_name,
                attempt,
                max_attempts,
                exc,
                delay_seconds,
            )
            await asyncio.sleep(delay_seconds)
