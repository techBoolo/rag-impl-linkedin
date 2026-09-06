# RAG Project: Multi-Document Ingestion & Performance Tuning

This project is an end-to-end local Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It dynamically scans a `docs/` directory, tracks already-indexed files using cryptographic hashes to prevent duplicates, incrementally batch-embeds newly added PDFs into a persistent FAISS vector store, and provides an interactive terminal chatbot to answer questions across documents.

---

## 🚀 Features

- **Multi-Document Dynamic Scanning**: Automatically discovers all `.pdf` documents placed in the `docs/` directory.
- **Deduplication & Tracking**: Computes SHA-256 file hashes and maintains a tracking registry (`faiss_index/indexed_files.json`) to skip already-indexed documents.
- **Incremental Batch Ingestion**: Lazily streams, chunks, and batch-embeds *only* new or modified PDF documents.
- **FAISS Index Appending & Persistence**: Seamlessly creates or appends new embeddings to the existing FAISS vector store on disk.
- **Local LLM & Embeddings**: Powered by Ollama (`llama3.1` for conversational answers and `nomic-embed-text` for vector embeddings).
- **RAG Generation Chain**: Built with LangChain Expression Language (LCEL) connecting similarity retrieval, document-tagged context prompting, and Ollama.
- **Interactive Chatbot CLI**: Continuous interactive conversation loop in the terminal with status updates and graceful exit handling.
- **Memory Efficiency**: Asynchronous document streaming via `PyPDFLoader.alazy_load()` and generator-based text splitting with `RecursiveCharacterTextSplitter`.
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

## 📂 Multi-Document Ingestion Pipeline

1. **Scan & Deduplicate**: Scans `docs/` for `.pdf` files. Cross-references file names and SHA-256 hashes against `indexed_files.json` (or bootstrapped metadata from `vector_store.docstore`).
2. **Selective Processing**: Only unindexed or modified PDFs are queued for processing.
3. **Lazy Streaming & Splitting**: Loads PDF pages asynchronously with `alazy_load` and chunks them with `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap), tagging chunks with source document metadata.
4. **Batch Embedding**: Batches chunks (10 at a time) and embeds them concurrently using `aembed_documents` via Ollama's `nomic-embed-text`.
5. **Vector Store Update**: Appends new vector embeddings to the loaded FAISS index (or initializes it if none exists) and persists both the FAISS index and tracking registry to disk.
6. **Multi-Document Answering**: RAG pipeline formats retrieved context with source document identifiers (`[filename]: ...`) and generates concise answers using `llama3.1`.

---

## 🏃 Running the Project

To scan documents, ingest updates, and start the chatbot:

```bash
uv run python main.py
```

### Example Session Output

```text
==================================================
RAG PIPELINE: MULTI-DOCUMENT INGESTION & TRACKING
==================================================

Attempting to load existing index from 'faiss_index'...
Index 'faiss_index' loaded successfully.

Scanning documents in: /Users/tfa/Projects/ai/langchain/linkedin-rag/rag-project/docs
Document scan results: 2 total PDF(s) found.
 - Already indexed: 1 file(s)
   * constitution.pdf (Skipped - already up to date)
 - New or updated:  1 file(s)
   * sample_addendum.pdf (Queued for ingestion)

Starting batch ingestion of new document(s)...

[Ingestion] Loading new document: sample_addendum.pdf (.../docs/sample_addendum.pdf)
  Processed batch 1 (2 chunks) -> Total chunks embedded: 2
  Finished indexing sample_addendum.pdf (2 chunks).

Successfully saved updated FAISS index to 'faiss_index' with 2 new chunks added!

Verified vector store size: 116 document chunks.

==================================================
MULTI-DOCUMENT KNOWLEDGE ASSISTANT
Type your questions below. Type 'exit' or 'quit' to stop.
==================================================

You: What is the supreme law of the land?
Thinking...

AI: According to Article 9 of the Constitution, the supreme law of the land is the Constitution itself. This means that any law, customary practice, or decision of an organ of state that contravenes the Constitution shall be of no effect.
------------------------------
You: exit

Exiting conversation. Goodbye
```

---

## 📈 Roadmap
- [x] Project Initialization
- [x] Basic LLM Connection
- [x] Asynchronous Document Loading
- [x] Memory-Efficient Document Splitting
- [x] Asynchronous Batch Embeddings
- [x] FAISS Vector Store Integration
- [x] FAISS Index Persistence to Disk
- [x] FAISS Index Loading from Disk
- [x] RAG LCEL Question Answering Chain
- [x] Interactive Terminal Conversation Loop
- [x] Multi-Document Ingestion & Deduplication Tracking
