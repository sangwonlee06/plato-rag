"""Embedding protocol.

Defines the contract for embedding text into vectors. The embedder
is used in two contexts:
1. Ingestion: embed chunk texts for storage in the vector store
2. Retrieval: embed the user's query for similarity search

Implementations must track their model name and dimensionality so
that chunks record which model produced their embedding. This is
critical for migration safety: if the model changes, stale embeddings
can be detected and re-embedded.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Embeds text into dense vectors for similarity search.

    Implementations:
    - OpenAIEmbedder: text-embedding-3-large (3072 dimensions)
    - Future: local sentence-transformer models

    The embed() method accepts a batch of texts and returns vectors
    in the same order. Implementations handle batching and rate
    limiting internally.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: One or more text strings to embed.

        Returns:
            Vectors in the same order as the input texts.
            Each vector has self.dimensions() elements.
        """
        ...

    def model_name(self) -> str:
        """The embedding model identifier, e.g. 'text-embedding-3-large'."""
        ...

    def dimensions(self) -> int:
        """The dimensionality of the output vectors, e.g. 3072."""
        ...
