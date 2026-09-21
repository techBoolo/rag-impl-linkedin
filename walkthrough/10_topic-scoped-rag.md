# Recursive Subfolder Indexing & Topic-Scoped Retrieval

We have extended the RAG pipeline to recursively scan subdirectories within `docs/`, automatically extract topic/category metadata based on directory structure, tag chunks during ingestion, and provide interactive topic- and document-scoped retrieval filtering during conversations.

---

## Key Highlights & Architecture

### 1. Recursive Subfolder Discovery & Category Extraction
- **Recursive Walk (`scan_docs_directory`):** Uses `os.walk` to traverse nested subdirectories inside `docs/` to discover all `.pdf` documents.
- **Hierarchical Category Tagging:** Automatically infers category names from relative folder paths:
  - Root documents (`docs/constitution.pdf`) $\rightarrow$ topic: `"general"`
  - Subfolder documents (`docs/legal/terms.pdf`) $\rightarrow$ topic: `"legal"`
  - Nested subfolder documents (`docs/finance/tax/filing.pdf`) $\rightarrow$ topic: `"finance/tax"`
- **Deduplication Persistence:** Tracks relative paths, content hashes (SHA-256), and assigned categories in `faiss_index/indexed_files.json`.

### 2. Topic-Aware Chunk Ingestion
- **Metadata Injection (`ingest_new_documents`):** When chunking documents with `RecursiveCharacterTextSplitter`, each chunk is injected with:
  - `category`: Inferred folder category
  - `filename`: Base name of the file
  - `source`: Full file path
  - `file_hash`: Content SHA-256 hash
- **Incremental & Safe Indexing:** Skips already-indexed files without duplicate chunking, appending only new or updated documents to the FAISS vector store.

### 3. Topic & Document Filtered Retrieval
- **Dynamic Scope Registry (`get_available_topics_and_docs`):** Aggregates all indexed topics and individual documents either from `indexed_files.json` or fallback `vector_store.docstore` metadata.
- **Interactive Scope Selection (`select_topic_filter`):** Presents an interactive menu allowing the user to select:
  - `[1] All Topics & Documents` (unfiltered search across the entire corpus)
  - `Topic: <category>` (restricts retrieval strictly to chunks in that category)
  - `Document: <filename>` (restricts retrieval to chunks from a specific document)
- **Metadata Filter in LangChain Retriever (`generate_answer`):**
  Uses FAISS retriever's callable filter mechanism:
  ```python
  if filter_type == "category":
      search_kwargs["filter"] = lambda m: m.get("category", "general").lower() == filter_val.lower()
  elif filter_type == "filename":
      search_kwargs["filter"] = lambda m: (
          m.get("filename", "").lower() == filter_val.lower()
          or os.path.basename(m.get("source", "")).lower() == filter_val.lower()
      )
  retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
  ```
- **Context Grounding:** The prompt formats retrieved context as `[{category} / {filename}]: content` to clearly inform the LLM of the document source and domain.

### 4. Interactive Conversation with Dynamic Filter Switching
- **Prompt Badge:** Displays the active retrieval scope in the CLI prompt: `[Topic: legal] You: `.
- **In-Session Topic Switching (`start_conversation`):** Users can switch topics at any time by typing `/filter` or `/topic` without restarting the application.

---

## Verified Execution

Run the application:

```bash
uv run python main.py
```

### Example Session

```text
==================================================
RAG PIPELINE: RECURSIVE INGESTION & TOPIC SELECTION
==================================================

Attempting to load existing index from 'faiss_index'...
Index 'faiss_index' loaded successfully.

Recursively scanning documents in: /path/to/rag-project/docs
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
[Topic: legal] You: /topic

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
Select retrieval scope [1-5] (Press Enter for 'All'): 1

Switched retrieval scope to: [All Topics & Documents]

[All Topics & Documents] You: exit

Exiting conversation. Goodbye!
```
