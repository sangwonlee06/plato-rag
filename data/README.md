# Corpus Data

This directory holds corpus configuration and prepared text files for ingestion.

## Directory structure

```
data/
  corpus_manifest.yaml    # Declarative corpus description
  README.md               # This file
  prepared/               # Prepared text files ready for ingestion
    meno_sample.txt       # Meno excerpt (Jowett translation, public domain)
```

`data/raw/` and full `data/prepared/` contents are gitignored — prepared texts
can be large and are derived from source materials. Only sample/fixture files
are committed.

---

## Prepared text format

The plaintext parser expects files with `[SECTION]` markers. Every section must
be on its own line, immediately followed by the section text.

```
[SECTION title="Section Title" location="86b" speaker="Socrates" interlocutor="Meno"]
Text of the section goes here. This can span multiple lines.
A new paragraph within the same section is separated by a blank line.

More text continues here.

[SECTION title="Next Section" location="87a" speaker="Socrates"]
The next section begins here.
```

### Attribute reference

| Attribute | Required | Description |
|---|---|---|
| `title` | recommended | Section title (used in chunk metadata and context display) |
| `location` | strongly recommended | Stephanus/Bekker/DK reference. Range: `"82b-85b"`. |
| `speaker` | required for dialogues | The speaker in this section |
| `interlocutor` | recommended for dialogues | The primary interlocutor |

### Location systems by collection

| Collection | System | Example |
|---|---|---|
| `platonic_dialogues` | Stephanus | `86b`, `514a-520a` |
| `aristotle_corpus` | Bekker | `1094a1` |
| `presocratic_fragments` | DK | `22B30` |
| `sep` | Section | `2.1`, `§3` |
| `iep` | Section | `1`, `1.a`, `§1.a` |

---

## Manifest-driven remote sources

The corpus manifest also supports remote source kinds for sources that can be
parsed safely and reproducibly without a hand-prepared plaintext file.

### `perseus_tei`

Use for public-domain Plato texts available from the Perseus `dltext` TEI
endpoint.

```json
{
  "id": "protagoras",
  "kind": "perseus_tei",
  "collection": "platonic_dialogues",
  "title": "Protagoras",
  "author": "Plato",
  "translation": "W.R.M. Lamb",
  "source_url": "https://www.perseus.tufts.edu/hopper/dltext?doc=Perseus:text:1999.01.0178",
  "source_config": {
    "text_id": "Prot."
  }
}
```

`source_config.text_id` must match the TEI `<text n="...">` identifier inside
the downloaded document. The parser uses Perseus `milestone unit="section"`
markers as Stephanus references and preserves dialogue speaker changes in the
section text.

### `iep_url`

Use for public IEP entries. No additional `source_config` is required.

---

## What is manual

Preparing a text for ingestion requires human editorial judgment when you use
`prepared_text`:

1. **Obtain a public-domain translation.** The Jowett translations of Plato are
   in the public domain (Jowett died 1893). So are most pre-20th-century
   translations of Aristotle and the Presocratics. Modern scholarly translations
   (e.g., Grube/Cooper, Reeve) are under copyright and must not be ingested
   without permission.

2. **Segment into sections.** Each `[SECTION]` corresponds to one logical unit
   of argument or exchange. Typically this is a single Stephanus page or a
   named subsection (e.g., "Slave Boy Demonstration"). Sections should be
   200–500 words; the chunker will split larger sections automatically.

3. **Add location references.** Stephanus numbers must be verified against a
   reliable source. Do not guess or omit them.

4. **Add speaker attribution.** Required for dialogues. Check the source text.

5. **Verify the result.** After preparing, do a quick read-through to catch
   missed section breaks, wrong location numbers, or speaker errors. These
   errors will silently degrade citation quality.

---

## How to run ingestion

Prerequisites: the package is installed (`pip install -e ".[dev]"`) and a
PostgreSQL instance is running with the schema applied.

```bash
# Validate specific manifest entries before embedding
python scripts/ingest_corpus.py --dry-run --only protagoras socrates-iep

# Ingest only those entries
python scripts/ingest_corpus.py --only protagoras socrates-iep

# Run the database migration first (once)
PLATO_RAG_DATABASE_URL_SYNC=postgresql://postgres:postgres@localhost:5432/plato_rag \
  alembic upgrade head

# Ingest a prepared text
python scripts/ingest_primary.py \
  --file data/prepared/meno_sample.txt \
  --title "Meno" \
  --author "Plato" \
  --collection platonic_dialogues \
  --translation "Benjamin Jowett"

# Verify the corpus was updated
curl http://localhost:8001/v1/health | python -m json.tool
```

The ingestion script is idempotent — re-running it on the same file will
produce a "skipped (same hash)" result.

### Environment variables required for ingestion

```
PLATO_RAG_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/plato_rag
PLATO_RAG_OPENAI_API_KEY=sk-...
PLATO_RAG_EMBEDDING_MODEL=text-embedding-3-large
PLATO_RAG_EMBEDDING_DIMENSIONS=3072
```

---

## Current corpus status

| Text | Collection | Status | Translation | Sections |
|---|---|---|---|---|
| Meno (sample) | platonic_dialogues | committed | Jowett (public domain) | 10 |
| Republic | platonic_dialogues | planned | — | — |
| Phaedo | platonic_dialogues | planned | — | — |
| Perseus Plato texts | platonic_dialogues | supported | public-domain Perseus translations | varies by work |
| SEP entries | sep | local-only | — | — |
| IEP entries | iep | supported | live HTML bootstrap | varies by entry |

The sample covers the philosophically key passages but not the full text.
Full-text ingestion requires manual preparation (see above).
