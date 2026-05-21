# 🔍 RAG System with Vector Database

A **Retrieval-Augmented Generation (RAG)** system that combines vector database search with large language models to answer questions based on your own documents.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

This project implements a RAG pipeline that:
1. **Ingests** documents (PDF, TXT, DOCX, etc.)
2. **Embeds** them into a vector database for semantic search
3. **Retrieves** the most relevant chunks for a given query
4. **Generates** accurate, context-aware answers using an LLM

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Embedding  │────▶│  Vector Database  │────▶│  Top-K Docs  │
│   Model     │     │  (e.g. ChromaDB,  │     │  Retrieved   │
└─────────────┘     │   Pinecone, FAISS)│     └──────┬───────┘
                    └──────────────────┘            │
                                                    ▼
                                          ┌──────────────────┐
                                          │   LLM (e.g.      │
                                          │  GPT-4, Claude)  │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                             Final Answer
```

---

## ✨ Features

- 📄 **Multi-format document ingestion** – PDF, TXT, DOCX, Markdown
- 🔎 **Semantic search** via vector embeddings
- 🗄️ **Vector DB support** – ChromaDB, FAISS, Pinecone (configurable)
- 🤖 **LLM-agnostic** – works with OpenAI, Anthropic, local models
- 🧩 **Modular architecture** – easily swap components
- 💬 **Conversational memory** – multi-turn Q&A support
- 🚀 **REST API** – ready for integration

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- An API key for your chosen LLM provider

### Installation

```bash
# Clone the repository
git clone https://github.com/mt0726haw/Ragsystem_vectorDB.git
cd Ragsystem_vectorDB

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
# LLM Provider
OPENAI_API_KEY=your_openai_key_here
# or
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vector Database
VECTOR_DB=chromadb          # Options: chromadb, faiss, pinecone
PINECONE_API_KEY=           # Only needed if using Pinecone
PINECONE_ENV=               # Only needed if using Pinecone

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

---

## 💻 Usage

### 1. Ingest Documents

```bash
python ingest.py --source ./docs/
```

### 2. Query the System

```bash
python query.py --question "What is the return policy?"
```

### 3. Start the API Server

```bash
uvicorn app:app --reload --port 8000
```

Then send a POST request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the main topics of the document."}'
```

---

## ⚙️ Configuration

| Parameter        | Default                    | Description                          |
|------------------|----------------------------|--------------------------------------|
| `VECTOR_DB`      | `chromadb`                 | Vector database backend              |
| `CHUNK_SIZE`     | `512`                      | Token size per document chunk        |
| `CHUNK_OVERLAP`  | `50`                       | Overlap between consecutive chunks   |
| `TOP_K`          | `5`                        | Number of retrieved chunks per query |
| `EMBEDDING_MODEL`| `text-embedding-3-small`   | Embedding model to use               |

---

## 📁 Project Structure

```
Ragsystem_vectorDB/
├── app.py                  # FastAPI application entry point
├── ingest.py               # Document ingestion pipeline
├── query.py                # Query & retrieval logic
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment variables
├── docs/                   # Sample documents for testing
├── vectorstore/            # Persisted vector database
└── utils/
    ├── chunker.py          # Text chunking utilities
    ├── embedder.py         # Embedding model wrapper
    └── retriever.py        # Retrieval logic
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with ❤️ using Python, LangChain & Vector Databases</p>
