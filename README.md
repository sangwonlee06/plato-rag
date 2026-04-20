# Plato RAG

A source-priority retrieval-augmented generation service for academic philosophy.

> **A note on naming.** "Plato" here is a product mode name, not a subject boundary. The parent application organizes its features into modes — Plato for broad philosophy, Frege for logic and analytic rigor. This repository implements the Plato mode's RAG service: the first corpora emphasize foundational and classical philosophy, but the architecture is designed for academic philosophy generally. It would be misleading to treat this as a system only about Plato the historical philosopher.

## What this project is

This is a standalone Python service that retrieves philosophical source material, generates grounded answers, and verifies that citations correspond to actual retrieved passages. It is called by a separate NestJS application over HTTP.

The central design idea is **source priority**: not all documents are treated equally. Sources are classified by epistemic role — a primary philosophical text is the thing being studied, a reference encyclopedia is an interpretation of it, a journal article is a narrower scholarly contribution — and the retrieval system uses these classifications to decide what the language model sees and what the user is told about the answer's grounding.

The project is early-stage. The domain models, API contracts, source-priority policy, and retrieval pipeline are working and tested. The retrieval quality depends on corpus coverage, which is currently empty. The citation extractor is regex-based and limited. The philosophy prompts are a single template that needs iteration. This is a serious foundation, not a finished product.

## Why this repository exists

The parent application is a NestJS backend that will support multiple modes:

- **Plato mode** — broad philosophy: foundational texts, scholarly reference, research-oriented queries
- **Frege mode** (future) — logic and analytic rigor: formal systems, proof structures, logical notation

Each mode has its own RAG service with its own corpus, retrieval policy, and domain-specific behavior. This repository is the Plato mode's RAG service. It is designed to be called from the NestJS app via HTTP, but it runs independently and owns its own database.

Frege is not implemented here. When it is built, it will be a separate service with the same API contract shape but different corpora and retrieval logic.

## Source-priority policy

The system classifies every document and chunk by source class. Source classes are ordered by epistemic role:

| Source class | Trust tier | What it represents | Initial examples |
|---|---|---|---|
| `primary_text` | 1 | Original philosophical works | Platonic dialogues, Aristotle |
| `reference_encyclopedia` | 2 | Peer-reviewed encyclopedic reference | SEP, IEP |
| `peer_reviewed` | 3 | Journal articles, scholarly monographs | (future) |
| `curated_bibliography` | 4 | Discovery and bibliography resources | (future) |

Trust tiers are not stored on individual chunks. They are derived from the source class at query time via a central registry. This avoids a data-update hazard if tier assignments ever change.

### SEP/IEP-first retrieval

For the initial implementation, the retrieval policy prioritizes:

1. **Primary philosophical texts** — the works themselves
2. **Stanford Encyclopedia of Philosophy (SEP) and Internet Encyclopedia of Philosophy (IEP)** — as the highest-trust secondary sources
3. **Broader scholarship** — only when primary and reference sources are insufficient

This is implemented as staged retrieval with trust-tier score boosts. Primary text chunks receive a 30% similarity score boost; reference encyclopedia chunks receive 15%. Source quotas guarantee at least one primary text chunk and one reference encyclopedia chunk appear in results when available. These are best-effort guarantees — if the corpus lacks relevant material from a required class, the quota is unmet and the grounding assessment reflects this honestly.

The bias toward narrower, more trustworthy grounding over broader but weaker grounding is a deliberate design choice.

## Current status

**What works:**

- Domain models with structured source classification, location references, and chunk metadata
- API contracts (Pydantic v2) with full request/response types and versioned endpoints
- Retrieval policy as a declarative, testable, frozen dataclass
- Source-priority reranker with trust-tier boosting
- Plaintext parser for prepared primary texts with `[SECTION]` markers
- Section-aware chunker that respects section boundaries and preserves location references
- SEP HTML parser that preserves entry metadata, numbered sections, and revision dates
- OpenAI embedding integration (`text-embedding-3-large`)
- pgvector similarity search with metadata filtering
- Retrieval service with staged search, quota enforcement, and grounding assessment
- Anthropic Claude integration for answer generation
- Citation extraction that verifies references against retrieved chunks
- Manifest-driven corpus ingestion for prepared primary texts and live SEP entry URLs
- Seed corpus assets for five Platonic dialogue texts plus a curated SEP entry manifest
- PostgreSQL schema with Alembic migrations
- Unit tests covering domain models, retrieval policy, API contracts, and ingestion parsers

**What is prototype-level:**

- **Citation extractor** — regex-based, handles `[Work Location]` and `[Author, SEP §Section]` formats only. No fuzzy matching, no range resolution, known substring false-positive risk. See the module docstring in `generation/citation_extractor.py` for the full limitations list.
- **Philosophy prompts** — single system prompt. Needs per-question-type prompt selection and iterative refinement.
- **IEP HTML parser** — not yet implemented. SEP is supported; IEP still needs its own parser and ingestion path.
- **Evaluation** — no automated evaluation suite. The system cannot yet measure citation fidelity or retrieval coverage against ground truth.
- **Corpus runtime state** — the repository now includes prepared seed assets, but a fresh database is still empty until you run ingestion.
- **Error handling** — basic. No retry logic for external API calls, no structured error responses.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design. The high-level structure:

```
NestJS app                                Plato RAG service
┌────────────────┐                        ┌─────────────────────────────┐
│  ChatService   │   POST /v1/query       │  api/v1/query.py            │
│  (mode=PLATO)  │ ────────────────────>  │    │                        │
│                │ <────────────────────  │    ├── retrieval/service.py  │
│                │   QueryResponse        │    │   (staged search,      │
└────────────────┘                        │    │    rerank, grounding)   │
                                          │    │                        │
                                          │    └── generation/service.py│
                                          │        (prompt, LLM, cite)  │
                                          └─────────────┬───────────────┘
                                                        │
                                          ┌─────────────┴───────────────┐
                                          │  PostgreSQL + pgvector       │
                                          │  (documents, chunks,         │
                                          │   embeddings)                │
                                          └─────────────────────────────┘
```

Key internal boundaries:

- `domain/` — pure Python dataclasses, no framework dependencies
- `api/contracts/` — Pydantic models defining the HTTP boundary
- `protocols/` — Python Protocol interfaces for internal service contracts
- `retrieval/policy.py` — declarative source-priority rules
- Services accept protocols, not concrete implementations (with the exception of the early-stage wiring in `dependencies.py`)

## API

All endpoints are versioned under `/v1`. The API is designed for the NestJS backend to call, but any HTTP client can use it.

### `POST /v1/query`

Submit a philosophy question and receive a grounded answer with citations.

```json
{
  "question": "What is Plato's theory of recollection?",
  "mode": "PLATO",
  "conversation_history": [],
  "options": { "max_chunks": 5, "include_debug": false }
}
```

The response includes:

- `answer` — generated text with inline citation markers
- `retrieved_chunks[]` — each chunk carries both `source_type` (`"PRIMARY"` / `"SECONDARY"` for NestJS compatibility) and `source_class` (the richer classification like `"primary_text"`, `"reference_encyclopedia"`)
- `citations[]` — references extracted from the answer and verified against retrieved chunks. Unverifiable citations are excluded and logged.
- `grounding` — an assessment of answer quality: `interpretation_level` (`DIRECT`, `INTERPRETIVE`, or `LOW_CONFIDENCE`), source coverage counts, and limitation notes when the system could not find adequate grounding
- `api_version`, `request_id`

### `GET /v1/health`

Service status and corpus statistics (total chunks, counts by source class).

### `GET /v1/sources`

Source class definitions and collection metadata. Useful for understanding what the corpus contains.

## NestJS integration

The NestJS backend's `AiQueryProvider` abstraction was designed for this. To connect:

1. Create an `ExternalAiQueryProvider` that HTTP-POSTs to `http://plato-rag:8001/v1/query`
2. Map the response to the existing `AiQueryResponse` interface:
   - `answer` maps directly
   - `retrieved_chunks[].source_type` is already `"PRIMARY"` / `"SECONDARY"` — compatible with the NestJS `CitationSourceType` enum
   - `retrieved_chunks[].id`, `.text`, `.work`, `.location` map to the existing `RetrievedChunk` type
   - `grounding.interpretation_level` maps to `interpretationLevel`
3. Bind the new provider in `AiModule`

The RAG service returns additional fields (`source_class`, `trust_tier`, `location_ref`, `chunk_metadata`, `grounding_notes`) that the NestJS app can adopt incrementally without breaking changes.

## Local development

### Prerequisites

- Python 3.12+
- Docker (for PostgreSQL with pgvector)
- An OpenAI API key (for embeddings)
- An Anthropic API key (for answer generation)

### Setup

```bash
git clone <this-repo>
cd plato-rag

# Start the database
docker compose up db -d

# Configure environment
cp .env.example .env
# Edit .env — at minimum, set PLATO_RAG_OPENAI_API_KEY and PLATO_RAG_ANTHROPIC_API_KEY

# Install
pip install -e ".[dev]"

# Run database migrations
alembic upgrade head

# Start the service
uvicorn plato_rag.main:app --reload --port 8001
```

The service will be available at `http://localhost:8001`. API docs are at `/docs` (Swagger) and `/redoc`.

### Environment variables

All variables are prefixed `PLATO_RAG_`. See `.env.example` for the full list. The required ones:

| Variable | Purpose |
|---|---|
| `PLATO_RAG_DATABASE_URL` | PostgreSQL connection string (async, `asyncpg`) |
| `PLATO_RAG_OPENAI_API_KEY` | OpenAI API key for embedding |
| `PLATO_RAG_ANTHROPIC_API_KEY` | Anthropic API key for answer generation |

### Running tests

```bash
pytest -v                     # All 38 tests
pytest tests/domain/          # Domain model and location reference tests
pytest tests/retrieval/       # Retrieval policy and reranker tests
pytest tests/api/             # API contract serialization tests
```

### Linting

```bash
ruff check src tests scripts
```

## Ingesting texts

The repository now includes a seed corpus manifest at `data/corpus_seed.json`, five prepared Platonic dialogue files under `data/prepared/primary/`, and a curated set of SEP entry URLs. A fresh database still starts empty; you need to run ingestion to load the seed corpus.

Primary texts use a `[SECTION]` marker format that preserves the metadata needed for academic citation:

```
[SECTION title="Recollection Argument" location="86b" speaker="Socrates"]
The soul, then, as being immortal, and having been born many times,
and having seen all things that exist, whether in this world or in
the world below, has knowledge of them all...

[SECTION title="Slave Boy Demonstration" location="82b-85b" speaker="Socrates" interlocutor="Slave Boy"]
Come here to me. Tell me, boy, do you know that a figure like this
is a square?
```

This format is a deliberate design choice. Rather than trying to parse arbitrary PDFs, the system requires a lightweight preparation step where human editorial judgment identifies section boundaries, location references (Stephanus numbers, Bekker numbers, etc.), and speaker attribution. This preparation is where source fidelity enters the pipeline.

To ingest one prepared file:

```bash
python scripts/ingest_primary.py \
  --file data/prepared/primary/meno.txt \
  --title "Meno" \
  --author "Plato" \
  --collection platonic_dialogues \
  --translation "W.R.M. Lamb"
```

To validate or ingest the full seed corpus manifest:

```bash
python scripts/ingest_corpus.py --dry-run
python scripts/ingest_corpus.py
```

## Scope and design boundaries

**Philosophy-capable, not Plato-only.** The domain models, source classifications, and location reference systems are designed for academic philosophy generally. `primary_text` can hold Descartes, Hume, or Kant. Location references support Stephanus (Plato), Bekker (Aristotle), DK (Presocratics), section numbers, page numbers, and more. Adding a new philosophical tradition requires a collection registry entry and a parser for the text format — not architectural changes.

**Source fidelity over breadth.** The system is biased toward narrower, more trustworthy grounding. If the corpus doesn't cover a topic, the system says so (`LOW_CONFIDENCE`) rather than hallucinating a plausible-sounding answer from general knowledge.

**Citation verification, not citation generation.** The LLM is instructed to cite sources using a specific format. The citation extractor then verifies each citation against the retrieved chunks. Citations that don't match a retrieved chunk are flagged as ungrounded and excluded from the response. The LLM cannot fabricate citations that survive this check — though the check itself is currently limited (see the prototype-level notes above).

## Roadmap

Near-term milestones, roughly in order:

1. Prepare and ingest 3–5 Platonic dialogues (Meno, Republic VII, Phaedo, Symposium, Apology)
2. Implement an HTML parser for SEP entries and ingest 5–10 core entries matching the primary text topics
3. Connect the NestJS app via `ExternalAiQueryProvider`
4. Build a 20–30 question evaluation set with expected source coverage and citation targets
5. Improve the citation extractor with fuzzy matching and range reference support
6. Iterate on philosophy prompts with evaluation feedback

Longer-term:

- IEP parser and ingestion
- Broader primary text coverage (Aristotle, Presocratics, modern philosophy)
- Cross-encoder reranking (model-based, not just trust-tier heuristics)
- Streaming answer generation
- Automated evaluation pipeline

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions and guidance on what kinds of contributions are most useful at this stage.

## License

This project does not yet have a license. If you are evaluating it for use or contribution, please contact the maintainer.
