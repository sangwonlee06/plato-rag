# Plato RAG — Development Notes

## Quick start
```
docker compose up db -d
cp .env.example .env
pip install -e ".[dev]"
alembic upgrade head
uvicorn plato_rag.main:app --reload --port 8001
```

## Package layout
- `domain/` — Pure dataclasses. No framework deps. Source of truth for types.
- `api/contracts/` — Pydantic models for the HTTP boundary.
- `api/v1/` — Endpoint handlers. Thin — delegate to services.
- `protocols/` — Python Protocol interfaces for service contracts.
- `retrieval/policy.py` — Declarative source-priority rules.
- `ingestion/` — Parse, chunk, embed, store.
- `retrieval/` — Staged search, rerank, grounding assessment.
- `generation/` — Prompt, LLM call, citation extraction.
- `db/` — SQLAlchemy ORM and repositories.

## Design rules
- trust_tier is derived from source_class via registry, never stored on chunks or documents
- CompatSourceType (PRIMARY/SECONDARY) lives in api/contracts, not in domain
- LocationRef is structured (system + value + range_end), not an opaque string
- Citations are verified against retrieved chunks, not trusted from LLM output
- RetrievalPolicy is a frozen dataclass
- IngestionService accepts Parser/Chunker via protocols, not hardcoded

## Testing
```
pytest -v
ruff check src/ tests/
```
