"""OpenAI embedding implementation."""

from __future__ import annotations

from openai import AsyncOpenAI


class OpenAIEmbedder:
    """Embeds text via OpenAI's text-embedding API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large", dimensions: int = 3072):
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI supports batches up to ~8k tokens per text, 2048 texts per call
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]

    def model_name(self) -> str:
        return self._model

    def dimensions(self) -> int:
        return self._dimensions
