# Multi-Document Ingestion & Performance Tuning

We have refactored the RAG pipeline to dynamically scan the `docs/` directory, track already-indexed documents to prevent duplicate chunks in FAISS, batch-ingest only new PDF files, persist the updated index, and provide a multi-document knowledge assistant chat loop.

## Highlights

### 1. Deduplication & Document Tracking
- **Content-Based Hashing:** Implemented `get_file_hash(file_path)` using SHA-256 to reliably detect file additions and content modifications regardless of path changes.
- **Registry Persistence (`indexed_files.json`):** Tracks ingested files (`faiss_index/indexed_files.json`), storing filenames, paths, and content hashes.
- **Backwards Compatibility:** If `indexed_files.json` is missing (such as when starting from an index created in earlier steps), `get_indexed_files()` automatically bootstraps tracking from existing `vector_store.docstore` metadata.
- **Dynamic Directory Scanning (`scan_docs_directory`):** Scans `docs/` for `.pdf` files and accurately separates them into already-indexed vs. new/unindexed files.

### 2. Incremental Batch Ingestion
- **Selective Processing (`ingest_new_documents`):** Only new or modified PDFs are processed, saving compute and preventing duplicate embeddings in the vector store.
- **Lazy Page Loading & Streaming:** Uses `PyPDFLoader.alazy_load()` and `split_document()` generator to stream chunks without loading entire PDFs into memory.
- **Chunk Metadata Tagging:** Injected `file_hash` and `filename` into each chunk's metadata.
- **Batch Embedding & FAISS Appending:** Asynchronously embeds chunks in batches (`process_embeddings`, batch size 10) and appends them to FAISS (`vector_store.add_embeddings()`), or creates the index if starting from scratch.
- **Atomic Persistence:** Saves both the updated FAISS vector store and `indexed_files.json` to disk.

### 3. Multi-Document Knowledge Assistant & Chat
- **Document-Aware Formatting:** Enhanced `generate_answer()` to prefix retrieved context with the document filename (`[filename]: content`), allowing the LLM to ground answers across multiple documents.
- **Interactive Session:** Updated conversation loop to greet users with `MULTI-DOCUMENT KNOWLEDGE ASSISTANT`.

---

## Verified Execution

Run the script to start the multi-document chatbot:

```bash
uv run python main.py
```

### Example Output: Incremental Ingestion

When a new PDF is placed into `docs/`:

```text
==================================================
RAG PIPELINE: MULTI-DOCUMENT INGESTION & TRACKING
==================================================

Attempting to load existing index from 'faiss_index'...
Index 'faiss_index' loaded successfully.

Scanning documents in: .../rag-project/docs
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

AI: According to the Ethiopian Constitution, the Constitution is the supreme law of the land. Any law, customary practice, or decision of an organ of state or public official that contravenes it shall be of no effect.
------------------------------
You: exit

Exiting conversation. Goodbye
```

### Example Output: Subsequent Runs (Skipping All Ingestion)

```text
Scanning documents in: .../rag-project/docs
Document scan results: 2 total PDF(s) found.
 - Already indexed: 2 file(s)
   * constitution.pdf (Skipped - already up to date)
   * sample_addendum.pdf (Skipped - already up to date)
 - New or updated:  0 file(s)

No new documents to ingest. Vector store is up to date.
Verified vector store size: 116 document chunks.
```
