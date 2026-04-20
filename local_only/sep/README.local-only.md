# Local-Only SEP Components

This directory contains Stanford Encyclopedia of Philosophy related material that is retained only for local or internal experimentation.

Rules:
- Do not expose SEP-backed functionality through any public deployment.
- Do not add this manifest to a public bootstrap path.
- Do not enable `PLATO_RAG_ENABLE_LOCAL_ONLY_SEP` in a public environment.
- Use the default public manifest in `data/corpus_seed.json` for deployable builds.

The default Docker build and default seed manifest are intended to remain public-safe. SEP-related work should stay local, private, and non-deployed.
