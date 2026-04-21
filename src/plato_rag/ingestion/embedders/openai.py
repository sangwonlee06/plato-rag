"""OpenAI embedding implementation."""

from __future__ import annotations

from typing import Any, cast

from openai import AsyncOpenAI

from plato_rag.resilience import retry_async


class OpenAIEmbedder:
    """Embeds text via OpenAI's text-embedding API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        *,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 4.0,
    ):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._max_attempts = max_attempts
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI supports batches up to ~8k tokens per text, 2048 texts per call
        response = cast(
            Any,
            await retry_async(
            "OpenAI embedding request",
            lambda: self._client.embeddings.create(
                input=texts,
                model=self._model,
                dimensions=self._dimensions,
            ),
            max_attempts=self._max_attempts,
            initial_backoff_seconds=self._initial_backoff_seconds,
            max_backoff_seconds=self._max_backoff_seconds,
            ),
        )
        # `response` is an SDK object with `.data`; the retry helper is intentionally generic.
        return [item.embedding for item in response.data]

    def model_name(self) -> str:
        return self._model

    def dimensions(self) -> int:
        return self._dimensions
