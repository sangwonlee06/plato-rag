# Architecture

This document describes the internal structure of the Plato RAG service for contributors and maintainers. It covers what each module does, why the boundaries are drawn where they are, and which design rules should not be broken without good reason.

## Package layout

```
src/plato_rag/
├── domain/           Pure dataclasses. No framework dependencies.
│   ├── source.py       SourceClass enum, trust-tier registry, collection registry
│   ├── location.py     LocationRef: structured location references for citations
│   ├── document.py     DocumentMetadata
│   └── chunk.py        ChunkData, ScoredChunk
│
├── api/
│   ├── contracts/    Pydantic models defining the HTTP boundary.
│   │   ├── common.py     Shared enums, CompatSourceType (NestJS mapping)
│   │   ├── query.py      QueryRequest, QueryResponse
│   │   ├── chunks.py     RetrievedChunkResponse, LocationRefResponse
│   │   ├── citations.py  CitationResponse
│   │   └── grounding.py  GroundingResponse, SourceCoverageResponse
│   └── v1/           FastAPI endpoint handlers. Thin — delegate to services.
│       ├── query.py      POST /v1/query
│       ├── health.py     GET /v1/health
│       └── sources.py    GET /v1/sources
│
├── protocols/        Python Protocol interfaces for internal contracts.
│   ├── ingestion.py    Parser, Chunker, ParsedDocument, RawChunk
│   ├── embedding.py    Embedder
│   ├── retrieval.py    VectorStore, Reranker, SearchFilters
│   └── generation.py   LLM, CitationExtractor, ExtractedCitation
│
├── retrieval/
│   ├── policy.py       RetrievalPolicy (frozen dataclass), PLATO_RETRIEVAL_POLICY
│   ├── service.py      RetrievalService: staged search, quota enforcement, grounding
│   ├── vector_store/
│   │   └── pgvector.py   Thin wrapper over ChunkRepository
│   └── reranker/
│       └── source_priority.py  Trust-tier score multiplication
│
├── ingestion/
│   ├── service.py      IngestionService: parse → chunk → embed → store
│   ├── parsers/
│   │   ├── plaintext.py  [SECTION]-marker format for primary texts
│   │   └── sep_html.py   SEP entry HTML parser with numbered sections
│   ├── chunkers/
│   │   └── section.py    Section-aware splitting with location preservation
│   └── embedders/
│       └── openai.py     text-embedding-3-large
│
├── generation/
│   ├── service.py           GenerationService: prompt → LLM → citation extraction
│   ├── llm/
│   │   └── anthropic.py     Claude implementation
│   ├── prompts/
│   │   └── philosophy.py    System prompt and context formatting
│   └── citation_extractor.py  Regex-based citation parsing and verification
│
├── db/
│   ├── engine.py         Async SQLAlchemy engine setup
│   ├── models.py         ORM models (DocumentModel, ChunkModel)
│   └── repositories/
│       ├── document.py   Document CRUD
│       └── chunk.py      Chunk CRUD + pgvector similarity search
│
├── config.py           pydantic-settings: all config from env vars
├── dependencies.py     FastAPI DI wiring
└── main.py             App entry point and lifespan
```

## Design rules

These are the rules that protect the system's integrity. Breaking them creates debt that compounds.

**1. Trust tier is derived, never stored.** Every chunk has a `source_class`. The trust tier for that class is looked up from `SOURCE_CLASS_REGISTRY` at query time. It is not stored on `ChunkData`, `DocumentMetadata`, or in the database. This means changing a tier assignment is a registry edit, not a data migration.

**2. The domain layer does not know about NestJS.** `CompatSourceType` (`"PRIMARY"` / `"SECONDARY"`) exists only in `api/contracts/common.py`. The domain uses `SourceClass`. The mapping happens at the API boundary.

**3. Location references are structured.** `LocationRef` has `system`, `value`, and `range_end` — not an opaque string. The system field (`stephanus`, `bekker`, `section`, `page`, etc.) tells the citation extractor and formatter what kind of reference it is. The database stores these as three separate columns for queryability.

**4. Citations are verified, not generated.** The LLM produces text with citation markers. The `CitationExtractor` parses these markers and attempts to match each one against retrieved chunks. Only matched citations appear in the API response. Unmatched citations are logged and reported in the debug response. The LLM cannot add citations that survive this check. (The current check is simple regex matching with known limitations — see the module docstring.)

**5. The retrieval policy is declarative.** `PLATO_RETRIEVAL_POLICY` is a frozen dataclass. The retrieval service reads its fields to decide how to search, rank, and select. Policy logic is testable without a database. A different mode (future Frege) gets a different policy instance.

**6. Services accept protocols.** `IngestionService` takes `Parser`, `Chunker`, and an embedder as constructor arguments. The retrieval and generation services are moving in this direction. Concrete implementations are wired in `dependencies.py` and `main.py`, not in the service constructors.

## Pipeline flow

### Query (POST /v1/query)

```
Request
  → embed query (OpenAI)
  → staged vector search (pgvector, one stage per source class in priority order)
  → filter by similarity threshold
  → rerank with trust-tier boosts
  → enforce source quotas (best-effort)
  → assess grounding (interpretation level, limitations)
  → build LLM prompt (system instructions + retrieved chunks + conversation history)
  → generate answer (Anthropic Claude)
  → extract and verify citations (regex match against retrieved chunks)
  → assemble response
```

### Ingestion (CLI script)

```
Prepared text file or SEP HTML
  → parse (PlaintextParser or SepHtmlParser: extract sections, location refs, speakers/entry metadata)
  → chunk (SectionChunker: split at section boundaries, respect token limits)
  → embed (OpenAI: batch embed all chunks)
  → store (PostgreSQL: document metadata + chunks with embeddings)
```

## What is intentionally simple

The retrieval pipeline uses a score multiplier per trust tier, not a learned model. The citation extractor uses regex, not NLP. The grounding assessment uses threshold checks, not statistical analysis. These are reasonable starting points for a system whose quality is currently bottlenecked by corpus coverage, not by algorithmic sophistication. When the corpus is large enough for these to matter, they should be improved.
