# RAG Project: Document Splitting

This project is a Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It is designed to load, process, and query PDF documents using local LLMs.

---

## 🚀 Features

- **Local LLM & Embeddings**: Powered by Ollama (`llama3.1` and `nomic-embed-text`).
- **Async Processing**: High-performance asynchronous document loading using `alazy_load` and concurrent batch embeddings using `aembed_documents`.
- **Memory Efficiency**: Document splitting using `RecursiveCharacterTextSplitter` and lazy embedding chunks with via async generators.
- **Modern Tooling**: Managed by `uv` for lightning-fast dependency management and environment isolation.

---

## 🛠️ Setup

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- [uv](https://github.com/astral-sh/uv) installed.

### 2. Pull the Models
Ensure the `llama3.1` and `nomic-embed-text` models are available locally:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 3. Initialize & Install Dependencies
```bash
# Initialize project environment
uv venv
source .venv/bin/activate

# Install core and RAG-specific dependencies
uv add langchain langchain-ollama langchain-community langchain-text-splitters pypdf faiss-cpu
```

---

## 📂 Document Loading & Processing

The project supports asynchronous document loading, memory-efficient splitting, and async batch embedding.

- **Current Document**: `docs/constitution.pdf`
- **Logic**: 
  - Uses `PyPDFLoader` with `alazy_load` to stream pages.
  - Uses `RecursiveCharacterTextSplitter` to lazily yield 1000-character chunks with 200-character overlap.
  - Batches document chunks iteratively via `OllamaEmbeddings` to generate vectors using `nomic-embed-text` without overloading memory.
  - Builds an in-memory **FAISS** vector store iteratively from generated embeddings.

---

## 🏃 Running the Project

To verify the document loading, splitting, and environment setup, run:

```bash
uv run python main.py
```

### Expected Output
```text
Loading document from: .../rag-project/docs/constitution.pdf
Splitting document and generating embeddings in batches...
Processed batch 1 (10 chunks) -> Added to FAISS vector store. Total chunks embedded: 10
Example vector dimension for first chunk in batch 1: 768
Processed batch 2 (10 chunks) -> Added to FAISS vector store. Total chunks embedded: 20
...
Processed batch 12 (4 chunks) -> Added to FAISS vector store. Total chunks embedded: 114

Successfully created FAISS index with 114 chunks across 12 batches!
Final vector store doc count: 114
```

---

## 📈 Roadmap
- [x] Project Initialization
- [x] Basic LLM Connection
- [x] Asynchronous Document Loading
- [x] Memory-Efficient Document Splitting
- [x] Asynchronous Batch Embeddings
- [x] FAISS In-Memory Vector Store Integration
