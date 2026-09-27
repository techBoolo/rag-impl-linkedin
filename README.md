# RAG Project: Drag-and-Drop PDF Upload & On-Demand Indexing

This project is an end-to-end local Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It dynamically scans a `docs/` directory, tracks already-indexed files using cryptographic hashes to prevent duplicates, incrementally batch-embeds newly added PDFs into a persistent FAISS vector store, and provides both an interactive terminal chatbot and a Streamlit web chat interface to answer questions across documents.

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
- **Streamlit Web Chat Interface**: A single-file browser UI (`app.py`) over the same pipeline, with live token-by-token streaming, per-session chat history, and a retrieval scope selector.
- **Document Category Sidebar**: Pick the topic or single document to search from directly in the web UI sidebar; switching scope clears the transcript so a conversation never mixes topics.
- **Grounded Source Citations**: Every answer carries an expandable `📚 View Sources` panel listing the exact document, page, and matched text the model used — so answers can be verified against the source.
- **Drag-and-Drop PDF Upload**: Add documents from the browser sidebar straight into a `docs/` category folder — no terminal commands.
- **On-Demand Indexing with Live Progress**: Uploads are chunked and batch-embedded on the spot, with a `st.status` step log and a chunk-level `st.progress` bar, then the FAISS index is reloaded so the new document is queryable immediately.
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

# Install core, RAG-specific, and UI dependencies
uv add langchain langchain-ollama langchain-community langchain-text-splitters pypdf faiss-cpu streamlit
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
9. **Source Attribution**: `retrieve_sources` returns the raw matched chunks so the UI can cite them. Those same chunks are passed into the chain via `build_rag_chain(docs=...)`, so retrieval happens once and the citations always match the passage the answer came from.

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

## 🖥️ Web Chat Interface

The same RAG pipeline is also available as a browser chat UI in `app.py`. Ingest your
documents once with the CLI (which creates `faiss_index/`), then launch:

```bash
uv run streamlit run app.py
```

Streamlit opens at `http://localhost:8501`.

- **Live streaming**: answers stream token by token via `generate_answer(..., stream=True)`.
- **Session history**: the transcript is kept in `st.session_state.messages` and replayed on every rerun.
- **Retrieval scope**: the sidebar mirrors the CLI's topic/document filter, so you can scope a
  conversation to one topic or a single document without restarting the app. Switching scope
  clears the transcript.
- **Source citations**: each answer has a `📚 View Sources` expander listing the document, page,
  and matched text used to produce it. Retrieval runs **once** per question and those exact chunks
  are handed to the model, so the citations cannot drift from the answer.
- **New chat**: clears the current session's transcript while keeping the loaded index warm.

The FAISS index is loaded once per server process via `@st.cache_resource`, so follow-up
questions are answered without reloading embeddings from disk.

### Verifying an Answer

```text
AI: To amend the constitution, Article 105 requires a two-thirds majority vote in the House of
    Peoples' Representatives or the House of the Federation...

> 📚 View Sources (3)
>    1. 📄 constitution.pdf — p. 40 · general
>    2. 📄 constitution.pdf — p. 24 · general
>    3. 📄 constitution.pdf — p. 32 · general
```

Page numbers come from the PDF's own page labels via `PyPDFLoader`, and citations are deduped per
page — four matching chunks across three distinct pages show as three sources.

### Adding a Document

1. Pick a **Target folder** in the sidebar — the `docs/` root or any existing subfolder.
2. Drop one or more PDFs onto the uploader and press **Index Document**.
3. Watch the status log and progress bar as the document is chunked and batch-embedded.
4. The new document is queryable straight away; the chunk counter and scope list update in place.

Uploads are deduplicated by SHA-256, so re-uploading unchanged content is a no-op. Both the
filename and the target folder are sanitized, so a crafted name cannot write outside `docs/`.

> **Note:** the index is append-only, so re-uploading a *modified* file leaves its old chunks in
> place. Delete and re-create `faiss_index/` (or add a new file) if you need a clean rebuild.

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
- [x] Streamlit Web Chat Interface with Live Token Streaming
- [x] Document Category Sidebar with History Reset on Scope Change
- [x] Grounded Source Citations with Page-Level References
- [x] Drag-and-Drop PDF Upload to Category Folders
- [x] On-Demand In-App Indexing with Live Progress
- [x] Hot-Reload of the FAISS Index After Ingestion
