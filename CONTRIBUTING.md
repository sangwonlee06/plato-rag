# Contributing

This is an early-stage project. Contributions that strengthen the foundation are more valuable than contributions that add features on top of a weak base.

## What is most useful right now

In roughly priority order:

1. **Prepared primary text files.** The system needs philosophy texts in the `[SECTION]` marker format with correct Stephanus/Bekker/DK references and speaker attribution. If you have editorial expertise in a philosophical text and can prepare it carefully, that is the single highest-value contribution.

2. **Evaluation data.** Questions with expected retrieval properties: which works should be cited, which SEP entries should appear, what interpretation level is appropriate. See `tests/` for the current test structure.

3. **HTML parser for SEP entries.** The `ingestion/parsers/` directory has a `plaintext.py` implementation. An `html.py` that handles SEP's page structure (section headings, author attribution, revision dates) while preserving section boundaries is the next parser needed.

4. **Citation extractor improvements.** The current extractor is regex-based with known limitations (documented in the module docstring). Fuzzy matching, range reference handling, and more citation format support would directly improve output quality.

5. **Bug reports and test cases.** If you run the system and a citation is wrong, a location reference is lost during chunking, or a grounding assessment is misleading, a clear bug report with the input that produced the problem is very helpful.

## What is less useful right now

- Feature additions that assume the retrieval quality is production-grade (it isn't — the corpus is empty)
- Framework migrations or dependency swaps without clear justification
- Abstractions for hypothetical future requirements

## Development setup

```bash
docker compose up db -d
cp .env.example .env        # Set your API keys
pip install -e ".[dev]"
alembic upgrade head
```

## Running tests

```bash
pytest -v
```

All tests should pass before submitting changes. Tests that require external API calls (OpenAI, Anthropic) or a running database are not yet in the suite — the current tests are all unit tests against domain models, policies, and contracts.

## Code style

- Python 3.12+, type annotations throughout
- `ruff` for linting, `mypy --strict` for type checking
- Docstrings where behavior is non-obvious. No docstrings on self-explanatory methods.
- Comments that explain *why*, not *what*. If a limitation exists, document it in the module docstring rather than hiding it.

## Design rules

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design rules that should not be broken without good reason. The most important ones:

- Trust tier is derived from source_class, never stored on chunks
- Location references are structured (`LocationRef`), not opaque strings
- Citations are verified against retrieved chunks, never trusted from LLM output
- The retrieval policy is a frozen dataclass, not scattered logic
