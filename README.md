# Plato RAG

Naming note: "Plato RAG" refers to the philosophy product context and the initial Plato-focused corpus. It does not imply any institutional affiliation.

## Important Stanford / SEP Notice

This project is an independent software project. It is not affiliated with, endorsed by, sponsored by, or otherwise connected to Stanford University or the Stanford Encyclopedia of Philosophy.

SEP-related components, if present anywhere in this repository, are not part of the default deployable service configuration and are not intended to be exposed through any public-facing deployment.

Any SEP-related code, parsers, manifests, ingestion logic, or experiments are retained solely for local/personal experimentation or internal development purposes.

Do not deploy SEP-related functionality as a public-facing service or expose it at any public URL.

Plato RAG is a separate FastAPI service that handles retrieval and answer grounding for Plato mode.

It sits beside the chatbot app, not inside it. The NestJS backend sends it a question, the service retrieves relevant chunks, asks the model for an answer, checks the citations it gets back, and returns a structured response.

The project is past the sketch phase. The API, retrieval policy, parsers, ingestion flow, and core domain model are in place. It is still early, though. The corpus is seed-sized, ingestion is manual, and the citation extractor is intentionally conservative, with stronger metadata-aware matching than the original MVP.

## Current status

What works:

- FastAPI app with versioned endpoints under `/v1`
- Startup bootstrap for the seed corpus on fresh databases
- Retrieval pipeline with staged search and trust-tier reranking
- Source classification with separate source classes and derived trust tiers
- Seed corpus manifest and prepared primary-text inputs in the repo
- public-safe seed corpus bootstrap
- manifest-driven Perseus TEI ingestion for Plato and Aristotle primary texts
- public IEP ingestion path with HTML parsing and URL-backed bootstrap
- expanded public-domain primary-text seed across Plato and Aristotle
- default public-safe seed corpus spans ancient, medieval, Islamic, Jewish, early modern, analytic, continental, political, Chinese, Buddhist, and African philosophy, with stronger topical IEP coverage in metaphysics, epistemology, ethics, language, mind, and logic
- local-only SEP ingestion path with deployment guardrails
- PostgreSQL + pgvector storage
- OpenAI embeddings
- Anthropic generation
- Structured JSON generation with claim-level citation binding
- Citation verification with location-aware, metadata-aware matching and claim-level bracket fallback for malformed model output
- Curated evaluation datasets and CLI runner for retrieval, citation, and grounding checks, including malformed-output fallback coverage
- Bounded retries for external embedding, generation, and bootstrap fetch operations
- Explicit 503 responses when retrieval or generation backends are unavailable
- Health and source-metadata endpoints
- public container builds exclude local-only SEP code via `.dockerignore`
- 105 passing pytest tests

What is still rough:

- Fallback claim reconstruction for malformed model output is still heuristic and less reliable than clean structured JSON responses
- The evaluation set is still seed-sized and hand-curated rather than statistically representative
- Operational behavior is tighter, but failures are still handled with simple bounded retries rather than circuit breakers or queue-backed recovery

## What this service is for

The chatbot repo has multiple answer modes. Plato mode is now handled here. That keeps retrieval logic, ingestion, and corpus management out of the main NestJS app.

The service is philosophy-oriented, not literally limited to Plato. The name comes from the product mode, not from a hard subject boundary.

The default public corpus is intentionally broader than Plato or ancient Greek philosophy. Plato remains an important primary-text layer, but the public-safe reference layer is meant to support academic philosophy more generally across major traditions and periods.

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
    "allowed_collections": ["platonic_dialogues"],
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
2. public-safe reference encyclopedias like IEP
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
│   ├── evaluation/
│   ├── guardrails/
│   ├── generation/
│   ├── ingestion/
│   ├── local_only/
│   ├── protocols/
│   └── retrieval/
├── data/
├── local_only/
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
| `PLATO_RAG_DEPLOYMENT_SCOPE` | `public`, `internal`, or `local` |
| `PLATO_RAG_PUBLIC_ALLOWED_COLLECTIONS` | Comma-separated public-safe collection allowlist |
| `PLATO_RAG_ENABLE_LOCAL_ONLY_SEP` | Enables local-only SEP bootstrap in non-public environments |
| `PLATO_RAG_LOCAL_ONLY_MANIFEST_PATH` | Local-only SEP manifest path |
| `PLATO_RAG_FAIL_START_ON_RESTRICTED_CONFIG` | Fails startup instead of warning on unsafe source configuration |
| `PLATO_RAG_EXTERNAL_REQUEST_MAX_ATTEMPTS` | Max attempts for transient external calls |
| `PLATO_RAG_EXTERNAL_RETRY_INITIAL_BACKOFF_SECONDS` | Initial retry backoff for transient external calls |
| `PLATO_RAG_EXTERNAL_RETRY_MAX_BACKOFF_SECONDS` | Max retry backoff for transient external calls |

## Ingestion

The repo includes a seed manifest, and the service now bootstraps it on startup when the database is empty or missing seed entries.

Default behavior is public-safe:

- `data/corpus_seed.json` contains only deployable public-safe entries
- SEP entries live under `local_only/sep/`
- SEP is never bootstrapped unless `PLATO_RAG_ENABLE_LOCAL_ONLY_SEP=true` in a non-public deployment
- default Docker builds exclude `local_only/` and `src/plato_rag/local_only/`

That only works after the schema is migrated. On a new environment, run migrations first and then start the service.

Useful commands:

```bash
python scripts/ingest_corpus.py --dry-run
python scripts/ingest_corpus.py
python scripts/ingest_corpus.py --replace-existing
```

`--replace-existing` is the reingestion path for seed entries whose stored documents and
chunks need to be rebuilt after metadata, parsing, or chunking changes. It requires the
normal database and embedding configuration because it performs a real delete-and-reingest
cycle, not a metadata-only patch.

The scalable path is manifest-driven:

1. add a new entry to `data/corpus_seed.json`
2. use `prepared_text` for local curated plaintext, `perseus_tei` for public-domain primary texts from Perseus, or `iep_url` for public IEP entries
3. for `perseus_tei`, set `source_config.text_id` when the TEI document contains multiple works, as Plato bundles often do
4. validate only the new entries before embedding anything:

```bash
python scripts/ingest_corpus.py --dry-run --only euthydemus nicomachean-ethics
```

5. ingest just those entries once the dry-run output looks right:

```bash
python scripts/ingest_corpus.py --only euthydemus nicomachean-ethics
```

6. when manifest metadata changes and stored seed chunks need to be refreshed, replace the
existing rows for the selected seed entries before reingesting:

```bash
python scripts/ingest_corpus.py --replace-existing --only epistemology-iep republic-vii
python scripts/ingest_corpus.py --replace-existing
```

Generation now asks the LLM for a JSON envelope with an `answer` plus explicit `claims`
and per-claim `citations`. That structured path is the primary claim-to-citation binding
mechanism. Legacy bracket parsing is retained only as a fallback when the model returns
malformed output.

The citation matcher is still heuristic, but it is no longer just raw title matching. It
uses normalized work and author matching, structured location overlap via `LocationRef`,
claim-to-chunk lexical and philosophy-profile alignment, and chunk metadata such as section
title or dialogue speaker to break otherwise ambiguous matches.

When `include_debug=true`, the query response now also exposes:

- `unsupported_claims`: claims the model made without a grounded citation match
- `citations[].claim_text`: the exact claim a citation is intended to support
- `citations[].match_score`: the extractor's chunk-level match score for that citation

There is also a one-off primary text ingestion script:

```bash
python scripts/ingest_primary.py \
  --file data/prepared/primary/meno.txt \
  --title "Meno" \
  --author "Plato" \
  --collection platonic_dialogues \
  --translation "W.R.M. Lamb"
```

## Evaluation

The repo now includes two curated evaluation datasets:

- `data/evaluation/public_seed.yaml` for normal end-to-end service evaluation
- `data/evaluation/malformed_output.yaml` for fixture-backed malformed-output cases that exercise bracket fallback through the same `/v1/query` path

Each case specifies:

- the user question and request options
- required retrieved works or collections
- required grounded citations
- answer anchor phrases
- caps on ungrounded citations and unsupported claims

Malformed-output cases can also define a deterministic generation fixture. Those
must be run in-process so the evaluation runner can swap in the fixture-backed
LLM while keeping the normal retrieval and API path intact.

Run it against a live service:

```bash
python scripts/run_evaluation.py --base-url http://localhost:8001
```

Useful filters:

```bash
python scripts/run_evaluation.py --base-url http://localhost:8001 --tag primary
python scripts/run_evaluation.py --base-url http://localhost:8001 --case-id meno_recollection_primary
python scripts/run_evaluation.py --base-url http://localhost:8001 --output eval-report.json
```

Run the malformed-output fixture set in-process:

```bash
python scripts/run_evaluation.py \
  --in-process \
  --dataset data/evaluation/malformed_output.yaml
```

## Testing and linting

Verified locally:

- `pytest -q`
- `python scripts/run_evaluation.py --help`

Common commands:

```bash
pytest -q
python scripts/run_evaluation.py --base-url http://localhost:8001
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
- expand public IEP coverage beyond the current seed set
- add more non-Platonic public-domain primary texts with citation-grade location systems
- expand the evaluation set and use it for regression gates
- further harden citation matching with evaluation-driven error analysis
- tighten operational behavior around retries and failures

If you want the more detailed architecture story, see `ARCHITECTURE.md`.
