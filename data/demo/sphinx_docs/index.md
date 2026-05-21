# Engineering Knowledge Base

Welcome to the demo knowledge base of the RAG system. This documentation
describes a small example pipeline and its API surface. It exists primarily
to demonstrate how Markdown content is chunked along headings.

## Overview

The system ingests engineering artifacts (documentation, source code,
configuration files, assembler) and exposes them via semantic search. It is
built around a local Qdrant vector database and pluggable parsers/chunkers.

## Architecture

The platform is composed of four major components:

1. **Source Loader** - discovers files described in `sources.yaml`.
2. **Parsers** - read raw bytes into normalized `Document` objects.
3. **Chunkers** - split documents into category-aware chunks.
4. **Vector Store** - persists embeddings in Qdrant for fast retrieval.

## Quick Start

To ingest the demo data and run a query:

```bash
python ingest.py --config config/sources.yaml
python query.py --question "How does the demo pipeline work?"
```

## Glossary

- **Chunk** - a small piece of text that fits into an embedding model.
- **Embedding** - a vector representation of a chunk's meaning.
- **Vector DB** - a database optimized for nearest-neighbor search.
