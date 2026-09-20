# RAG Project: Recursive Ingestion & Topic-Scoped Querying

This project is an end-to-end local Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It dynamically scans a `docs/` directory, tracks already-indexed files using cryptographic hashes to prevent duplicates, incrementally batch-embeds newly added PDFs into a persistent FAISS vector store, and provides an interactive terminal chatbot to answer questions across documents.

---

## 🚀 Features

- **Recursive Subfolder Scanning**: Recursively discovers all `.pdf` documents placed across `docs/` and its subdirectories.
- **Metadata Tagging**: Automatically tags each chunk with its topic/category (folder name) and source filename metadata.
- **Cryptographic Deduplication**: Computes SHA-256 file hashes and maintains a tracking registry (`faiss_index/indexed_files.json`) to skip already-indexed documents.
- **Incremental Batch Ingestion**: Lazily streams, chunks, and batch-embeds *only* new or modified PDF documents.
- **FAISS Index Appending & Persistence**: Seamlessly appends new embeddings to the existing persistent FAISS vector store on disk.
- **Interactive Topic & Document Selection**: Prompts user at startup to select retrieval scope (filter by topic/category or specific document, or search across all). Supports switching topics mid-session via `/filter` or `/topic`.
- **Local LLM & Embeddings**: Powered by Ollama (`llama3.1` for conversational answers and `nomic-embed-text` for vector embeddings).
- **RAG Generation Chain**: Built with LangChain Expression Language (LCEL) connecting similarity retrieval with metadata filters, document-tagged context prompting, and Ollama.
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

## 📂 Multi-Document Ingestion & Topic Pipeline

1. **Recursive Scan & Deduplicate**: Scans `docs/` recursively for `.pdf` files. Cross-references file relative paths and SHA-256 hashes against `indexed_files.json` (or bootstrapped metadata from `vector_store.docstore`).
2. **Category Extraction & Metadata Tagging**: Derives category names from relative subfolder paths (e.g. `docs/legal/contract.pdf` -> `legal`, root docs -> `general`). Tags each chunk with `category`, `filename`, `source`, and `file_hash`.
3. **Selective Processing**: Only unindexed or modified PDFs are queued for processing.
4. **Lazy Streaming & Splitting**: Loads PDF pages asynchronously with `alazy_load` and chunks them with `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap).
5. **Batch Embedding**: Batches chunks (10 at a time) and embeds them concurrently using `aembed_documents` via Ollama's `nomic-embed-text`.
6. **Vector Store Update**: Appends new vector embeddings to the loaded FAISS index (or initializes it if none exists) and persists both the FAISS index and tracking registry to disk.
7. **Topic / Document Filtered Retrieval**: User selects retrieval scope at startup or during chat (`/filter` or `/topic`). LangChain applies a callable metadata filter against the FAISS index.
8. **Multi-Document Answering**: RAG pipeline formats retrieved context with topic and source identifiers (`[category / filename]: ...`) and generates concise answers using `llama3.1`.

---

## 🏃 Running the Project

To scan documents, ingest updates, and start the chatbot:

```bash
uv run python main.py
```

### Example Session Output

```text
==================================================
RAG PIPELINE: RECURSIVE INGESTION & TOPIC SELECTION
==================================================

Attempting to load existing index from 'faiss_index'...
Index 'faiss_index' loaded successfully.

Recursively scanning documents in: /Users/tfa/Projects/ai/langchain/linkedin-rag/rag-project/docs
Document scan results: 2 total PDF(s) found.
 - Already indexed: 1 file(s)
   * [general] constitution.pdf (Skipped - already up to date)
 - New or updated:  1 file(s)
   * [legal] terms_of_service.pdf (Queued for ingestion)

Starting batch ingestion of new document(s)...

[Ingestion] Loading new document: terms_of_service.pdf [Category: legal] (.../docs/legal/terms_of_service.pdf)
  Processed batch 1 (4 chunks) -> Total chunks embedded: 4
  Finished indexing terms_of_service.pdf (4 chunks).

Successfully saved updated FAISS index to 'faiss_index' with 4 new chunks added!

Verified vector store size: 118 document chunks.

==================================================
TOPIC & DOCUMENT RETRIEVAL FILTER
==================================================
Available Topics (Categories):
  [1] All Topics & Documents (No filter)
  [2] Topic: general (1 document)
  [3] Topic: legal (1 document)

Available Documents:
  [4] Document: constitution.pdf [Topic: general]
  [5] Document: terms_of_service.pdf [Topic: legal]
--------------------------------------------------
Select retrieval scope [1-5] (Press Enter for 'All'): 3

==================================================
MULTI-DOCUMENT KNOWLEDGE ASSISTANT
Active Retrieval Scope: [Topic: legal]
Commands:
 - Type your question to query the assistant.
 - Type '/filter' or '/topic' to change retrieval scope.
 - Type 'exit' or 'quit' to stop.
==================================================

[Topic: legal] You: What is the termination policy?
Thinking...

AI: According to the terms of service, either party may terminate the agreement with thirty days written notice.
------------------------------
[Topic: legal] You: exit

Exiting conversation. Goodbye!
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
- [x] Recursive Subfolder Scanning & Metadata Tagging
- [x] Interactive Topic & Document Retrieval Filtering
