# 🔎 Engineering Knowledge RAG — Local Vector DB

A **fully local, modular RAG system** for engineering knowledge: documentation, source code, configuration files and assembler — all searchable via semantic similarity, with optional LLM-grounded answers through [LiteLLM](https://docs.litellm.ai/).

> 📦 Works without any API key out of the box (retrieval-only mode).
> 🔌 Plug in any LLM later (OpenAI, Anthropic, Azure, Mistral, Groq, local Ollama, …) by setting one env var.
> 🧩 Add new data sources via a single YAML file — no code changes.
> 🤖 GitHub Actions pipeline re-indexes the knowledge base on every push.

---

## 📚 Table of Contents

- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Ingestion](#-ingestion)
- [Querying](#-querying)
- [FastAPI Server](#-fastapi-server)
- [Configuration (sources.yaml)](#-configuration-sourcesyaml)
- [Adding Your Own Sources](#-adding-your-own-sources)
- [LiteLLM Integration](#-litellm-integration)
- [Qdrant Local Mode Notes](#-qdrant-local-mode-notes)
- [Automated Knowledge Pipeline (CI)](#-automated-knowledge-pipeline-ci)
- [Testing](#-testing)
- [Example Commands](#-example-commands)

---

## 🏗 Architecture

```
                            ┌────────────────────┐
                            │  config/sources.yaml│
                            └─────────┬──────────┘
                                      │
                ┌─────────────────────▼─────────────────────┐
                │                SourceLoader               │
                │  walks each source path, picks files via  │
                │  include-globs, dispatches to a Parser    │
                └─────────┬─────────────────────────┬───────┘
                          │                         │
        ┌─────────────────▼─────┐         ┌─────────▼────────────┐
        │   ParserRegistry      │         │  ChunkerRegistry     │
        │  .md   → Markdown     │         │  documentation → MD  │
        │  .py   → Python       │         │  python_code   → AST │
        │  .yml  → YAML         │         │  yaml_config   → key │
        │  .asm  → Asm          │         │  assembler     → lbl │
        └─────────┬─────────────┘         └─────────┬────────────┘
                  │                                 │
                  │      ┌──────────────────────────▼──┐
                  │      │           Chunks            │
                  │      │ {text, source_file,         │
                  │      │  category, metadata, id}    │
                  │      └────────────┬────────────────┘
                  │                   │
                  │   ┌───────────────▼───────────────┐
                  │   │      EmbeddingService         │
                  │   │ sentence-transformers (local) │
                  │   └───────────────┬───────────────┘
                  │                   │ vectors
                  │   ┌───────────────▼───────────────┐
                  └──▶│   QdrantStore  (./vectorstore/qdrant)│
                      └───────────────┬───────────────┘
                                      │
              ┌───────────────────────┼─────────────────────────┐
              │                       │                         │
        ┌─────▼─────┐          ┌──────▼──────┐           ┌──────▼──────┐
        │ ingest.py │          │  query.py    │           │   app.py    │
        │  (CLI)    │          │   (CLI)      │           │ (FastAPI)   │
        └───────────┘          └──────┬───────┘           └──────┬──────┘
                                      │                          │
                                ┌─────▼──────────────────────────▼─────┐
                                │   LiteLLMClient (optional, default off)│
                                │   100+ providers via OpenAI-compat API │
                                └────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Ragsystem_vectorDB/
├── app.py                       # FastAPI server (/query, /health)
├── ingest.py                    # Ingestion CLI
├── query.py                     # Query CLI
├── config.py                    # Typed YAML config loader
├── requirements.txt
├── .env.example
├── pytest.ini
├── config/
│   └── sources.yaml             # All source definitions
├── data/
│   └── demo/                    # Demo data (docs, python, yaml, asm)
├── src/
│   ├── model/                   # Document & Chunk dataclasses
│   ├── ingestion/               # SourceLoader + registries
│   ├── parsers/                 # File-type parsers
│   ├── chunkers/                # Category-aware chunkers
│   ├── embeddings/              # sentence-transformers wrapper
│   ├── vectorstore/             # Qdrant local-mode wrapper
│   ├── retrieval/               # Embed → search composition
│   └── llm/                     # LiteLLM client
├── tests/                       # Unit + E2E tests
├── vectorstore/qdrant/          # Persistent Qdrant data (git-ignored)
└── .github/workflows/
    ├── ci.yml                   # Tests on PR/push
    └── ingest.yml               # Knowledge-base re-indexing pipeline
```

---

## ⚙️ Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/mt0726haw/Ragsystem_vectorDB.git
cd Ragsystem_vectorDB

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # optional - only needed for the LLM
```

> ℹ️ The first ingestion downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) to the HuggingFace cache. Offline runs after that.

---

## 📥 Ingestion

Build the local Qdrant vector store from the configured sources:

```bash
python ingest.py --config config/sources.yaml
```

Re-running is **idempotent**: chunk IDs are deterministic (`sha256(source_file + text)`), so upserts overwrite existing points instead of duplicating them.

Sample output:

```
============================================================
INGESTION SUMMARY
============================================================
Files processed       : 4
Chunks produced       : 28
Chunks upserted       : 28
Collection total      : 28
Collection            : engineering_knowledge
Vector store location : /…/vectorstore/qdrant

Chunks per category:
  - assembler             7
  - documentation         9
  - python_code           8
  - yaml_config           4
============================================================
```

---

## 🔍 Querying

```bash
python query.py --question "How is the pipeline built?" --top-k 5
```

Without an enabled LLM, the CLI prints the top-K retrieved chunks (score, source, category, metadata, excerpt). With `llm.enabled: true` in the config **and** a provider API key in `.env`, it also generates a grounded answer.

Override the LLM at the CLI:

```bash
python query.py -q "What handles errors?" --llm on
python query.py -q "How does process_data work?" --llm off
```

---

## 🌐 FastAPI Server

```bash
uvicorn app:app --reload --port 8000
```

**Endpoints:**

| Method | Path     | Description                                  |
|--------|----------|----------------------------------------------|
| GET    | `/health`| Status, indexed chunk count, LLM config      |
| POST   | `/query` | `{question, top_k?, use_llm?}` → hits + answer |
| GET    | `/docs`  | Interactive Swagger UI                        |

Example:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Wie funktioniert der error_handler?", "top_k": 3}' | jq
```

---

## 🛠 Configuration (`sources.yaml`)

```yaml
sources:
  - name: demo_sphinx_docs
    path: ./data/demo/sphinx_docs
    category: documentation
    include: ["*.md"]

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2

vectorstore:
  type: qdrant
  path: ./vectorstore/qdrant
  collection: engineering_knowledge

retrieval:
  top_k: 5

llm:
  provider: litellm
  model: openai/gpt-4o-mini
  enabled: false        # set to true once you provide an API key
```

**Categories → chunker mapping:**

| Category        | Chunker            | Splits by                                  |
|-----------------|--------------------|--------------------------------------------|
| `documentation` | `MarkdownChunker`  | ATX headings (`#`, `##`, …)                |
| `python_code`   | `PythonAstChunker` | module / class / function / method (AST)   |
| `yaml_config`   | `YamlChunker`      | top-level keys                             |
| `assembler`     | `AsmChunker`       | labels (`init:`, `main_loop:`, …)          |

---

## ➕ Adding Your Own Sources

The whole point of the registry architecture: **never edit Python to onboard data.** Just append to `sources.yaml`:

```yaml
sources:
  - name: internal_wiki
    path: /mnt/share/wiki
    category: documentation
    include:
      - "**/*.md"
      - "**/*.markdown"

  - name: firmware_repo
    path: ../firmware/src
    category: python_code
    include:
      - "*.py"
```

Then re-run `python ingest.py`. Deterministic IDs make this safe — unchanged content won't be re-embedded into duplicate rows.

**Need a new file type?**

1. Add a parser in `src/parsers/your_parser.py`, set `extensions = (".ext",)`, register it in `src/ingestion/parser_registry.py`.
2. Add a chunker in `src/chunkers/your_chunker.py` and register it in `src/ingestion/chunker_registry.py` under a new category name.
3. Use that category in `sources.yaml`. Done.

---

## 🤖 LiteLLM Integration

LiteLLM is a unified OpenAI-compatible gateway for 100+ LLM providers (OpenAI, Anthropic, Azure, Mistral, Groq, Bedrock, local Ollama, vLLM, …). To enable LLM-grounded answers:

1. Pick a model & set the matching API key in `.env`:
   ```env
   LITELLM_MODEL=anthropic/claude-3-5-sonnet-20241022
   ANTHROPIC_API_KEY=sk-ant-...
   ```
2. Flip the switch in `config/sources.yaml`:
   ```yaml
   llm:
     enabled: true
     model: anthropic/claude-3-5-sonnet-20241022
   ```
3. Re-run `python query.py -q "…"` — answers now come from the model, grounded in the retrieved chunks.

The system prompt forces the LLM to answer **only** from the retrieved context. If context is insufficient, it returns:

> "Die Wissensbasis enthält dazu keine ausreichende Antwort."

Source files used are cited at the end.

---

## 💾 Qdrant Local Mode Notes

We run Qdrant via `qdrant-client` in **embedded local mode**:

```python
QdrantClient(path="./vectorstore/qdrant")
```

- No server process needed — pure file-based persistence.
- One process at a time may hold the lock; close the CLI before starting the API.
- The `vectorstore/qdrant/` folder is **git-ignored** by default. Either ship it as an artifact (see the CI pipeline below) or regenerate it from sources via `python ingest.py`.
- To migrate to a hosted Qdrant later, swap `QdrantClient(path=…)` for `QdrantClient(url=…)` in `src/vectorstore/qdrant_store.py` — payload schema stays identical.

---

## 🚀 Automated Knowledge Pipeline (CI)

**`.github/workflows/ingest.yml`** is your "train the knowledge base" pipeline. It runs on:

- **Push to `main`** whenever `data/`, `config/sources.yaml`, `src/**`, `ingest.py` or `requirements.txt` change.
- **Manual dispatch** from the Actions tab — optionally pass a custom config path.
- **Nightly schedule** at 03:17 UTC to catch merges that didn't touch the trigger paths.

What the pipeline does:

1. Checks out the repo and sets up Python 3.11.
2. Caches the HuggingFace embedding model between runs.
3. `pip install -r requirements.txt`.
4. Runs `python ingest.py --config config/sources.yaml --verbose`.
5. Uploads the resulting `vectorstore/qdrant/` directory as a workflow **artifact** (retained 30 days).
6. Uploads the ingestion log as a separate artifact.
7. Writes an ingestion summary into the GitHub Actions job summary.

**Workflow:** add data → commit → push → CI re-indexes → download the latest `qdrant-vectorstore-<sha>` artifact and unzip it into `vectorstore/qdrant/`, or just let local users re-run `ingest.py`.

```yaml
# add a new source = one PR, no code:
sources:
  - name: confluence_export
    path: ./data/confluence
    category: documentation
    include: ["**/*.md"]
```

A second workflow, **`.github/workflows/ci.yml`**, runs the unit/E2E tests on every PR — fast, ML-free, network-free.

---

## 🧪 Testing

```bash
pip install pytest pyyaml qdrant-client
pytest -v
```

- `tests/test_chunkers.py` — parser/chunker/registry unit tests (no heavy deps).
- `tests/test_pipeline_e2e.py` — full ingest → Qdrant → retrieve loop with a deterministic stub embedder (no model download required).

---

## 📋 Example Commands

```bash
# 1. Build the vector DB from the demo data
python ingest.py --config config/sources.yaml

# 2. Pure retrieval (no LLM key needed)
python query.py -q "Wie wird die Pipeline gebaut?"
python query.py -q "Welche Funktion verarbeitet Daten?"
python query.py -q "Was macht der error_handler?"

# 3. Force the LLM on (requires API key + llm.enabled config)
python query.py -q "Erkläre den main_loop" --llm on

# 4. Run the API
uvicorn app:app --reload --port 8000

# 5. Tests
pytest -v
```

---

<p align="center">Built with Python · Qdrant · sentence-transformers · LiteLLM · FastAPI</p>
