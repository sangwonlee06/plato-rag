# Plato RAG

Plato RAG is a separate FastAPI service that handles retrieval and answer grounding for Plato mode.

It sits beside the chatbot app, not inside it. The NestJS backend sends it a question, the service retrieves relevant chunks, asks the model for an answer, checks the citations it gets back, and returns a structured response.

The project is past the sketch phase. The API, retrieval policy, parsers, ingestion flow, and core domain model are in place. It is still early, though. The corpus is seed-sized, ingestion is manual, and the citation extractor is stricter than it is smart.

## Current status

What works:

- FastAPI app with versioned endpoints under `/v1`
- Retrieval pipeline with staged search and trust-tier reranking
- Source classification with separate source classes and derived trust tiers
- Seed corpus manifest and prepared primary-text inputs in the repo
- SEP ingestion path
- PostgreSQL + pgvector storage
- OpenAI embeddings
- Anthropic generation
- Citation extraction with post-generation verification
- Health and source-metadata endpoints
- 38 passing pytest tests

What is still rough:

- Fresh databases start empty until you run ingestion
- IEP parsing is not implemented yet
- Citation extraction is regex-based and still limited
- No proper evaluation harness yet
- Error handling and retry behavior need more work

## What this service is for

The chatbot repo has multiple answer modes. Plato mode is now handled here. That keeps retrieval logic, ingestion, and corpus management out of the main NestJS app.

The service is philosophy-oriented, not literally limited to Plato. The name comes from the product mode, not from a hard subject boundary.

## API

All endpoints live under `/v1`.

### `POST /v1/query`

Main query endpoint.

Request shape:

```json
{
  "question": "What is recollection in the Meno?",
  "mode": "PLATO",
  "conversation_history": [],
  "options": {
    "max_chunks": 5,
    "include_debug": false
  }
}
```

Response includes:

- `answer`
- `retrieved_chunks`
- `citations`
- `grounding`
- `request_id`
- `api_version`
- optional `debug`

### `GET /v1/health`

Returns service health plus corpus counts.

### `GET /v1/sources`

Returns source-class and collection metadata.

## Retrieval approach

The retrieval logic is built around source priority.

In plain terms, the service would rather answer from a primary text and a strong reference source than from a pile of weaker material.

Current order of preference:

1. primary philosophical texts
2. reference encyclopedias like SEP
3. broader secondary scholarship

That policy shows up in a few places:

- trust-tier boosts during reranking
- source quotas in staged retrieval
- grounding summaries in the final response

If the service cannot ground an answer well, it is supposed to say so.

## Repo layout

```text
.
├── src/plato_rag/
│   ├── api/
│   ├── db/
│   ├── domain/
│   ├── generation/
│   ├── ingestion/
│   ├── protocols/
│   └── retrieval/
├── data/
├── scripts/
└── tests/
```

## Local development

### Prerequisites

- Python 3.12+
- Docker
- PostgreSQL with pgvector
- OpenAI API key
- Anthropic API key

### Setup

```bash
cp .env.example .env
docker compose up db -d
pip install -e ".[dev]"
alembic upgrade head
```

Run the service:

```bash
uvicorn plato_rag.main:app --reload --port 8001
```

Docs will be available at:

- `http://localhost:8001/docs`
- `http://localhost:8001/redoc`

## Environment

The service uses `PLATO_RAG_`-prefixed environment variables.

The important ones are:

| Variable | Purpose |
| --- | --- |
| `PLATO_RAG_DATABASE_URL` | async Postgres connection string |
| `PLATO_RAG_OPENAI_API_KEY` | embedding API key |
| `PLATO_RAG_ANTHROPIC_API_KEY` | generation API key |

## Ingestion

The repo includes seed assets, but they are not loaded automatically.

That means a brand-new database will come up healthy and still have zero chunks until you ingest content.

Useful commands:

```bash
python scripts/ingest_corpus.py --dry-run
python scripts/ingest_corpus.py
```

There is also a one-off primary text ingestion script:

```bash
python scripts/ingest_primary.py \
  --file data/prepared/primary/meno.txt \
  --title "Meno" \
  --author "Plato" \
  --collection platonic_dialogues \
  --translation "W.R.M. Lamb"
```

## Testing and linting

Verified locally:

- `pytest -q` -> 38 passing tests

Common commands:

```bash
pytest -q
ruff check src tests scripts
```

## Integration with the chatbot

The chatbot backend calls this service through its `ExternalAiQueryProvider`.

At the moment the important connection point is:

- `PLATO_RAG_BASE_URL` in the chatbot backend env

Plato mode in the chatbot expects this service to be available at that base URL and to answer `POST /v1/query`.

## What is next

Short version:

- ingest more texts
- add IEP support
- improve citation matching
- build a real evaluation set
- tighten operational behavior around retries and failures

If you want the more detailed architecture story, see `ARCHITECTURE.md`.
