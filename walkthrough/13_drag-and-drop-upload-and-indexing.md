# Drag-and-Drop PDF Upload & On-Demand Indexing

The web interface can now ingest new documents itself. Previously the only way to add a PDF was
to drop it into `docs/` and run the CLI; the app assumed the index on disk was already complete.
This milestone adds a sidebar uploader, live indexing progress, and a hot reload of the vector
store so a freshly uploaded document is queryable in the same session.

---

## Key Highlights & Architecture

### 1. Reusing the Ingestion Pipeline
Rather than reimplement chunking and embedding for the browser, the UI drives the exact same
functions the CLI uses, so an uploaded file is indexed identically to one discovered on disk.
`main.py` gained only additive, optional parameters.

- **Progress Callback (`ingest_new_documents(..., on_batch=None)`):** Invoked as
  `on_batch(batches_done, chunks_done, filename)` after every batch. Defaults to `None`, so the
  terminal client behaves exactly as before.
- **Exact Progress Total (`count_document_chunks`):** Embedding a document is a streaming
  generator, so the total is unknown until the last batch. Rather than show a bar that guesses,
  the app pre-scans the document and counts chunks without embedding them, at the cost of parsing
  the PDF twice:

```python
async def count_document_chunks(file_record, chunk_size=1000, chunk_overlap=200):
    """
    Counts the chunks a document will produce without embedding them, so a caller can
    show an exact progress total. Streams the document instead of materializing it, at the
    cost of parsing the PDF twice.
    """
    doc_iterator = await load_document(file_record["path"])
    chunk_generator = split_document(
        doc_iterator,
        category=file_record.get("category", "general"),
        filename=file_record.get("filename"),
        file_hash=file_record.get("hash"),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    total = 0
    async for _ in chunk_generator:
        total += 1
    return total
```

- **Target Folders (`get_doc_categories`):** Returns the `docs/` root plus every existing
  subfolder, so the sidebar picker always reflects folders the pipeline can actually tag.

### 2. Safe Upload Persistence
- **`save_uploaded_pdf`** Writes the uploaded bytes into `docs/<category>/` and returns a file
  record in the same shape `ingest_new_documents` already consumes, so the SHA-256 hash is
  computed on the written file and deduplication works unchanged:

```python
def save_uploaded_pdf(uploaded_bytes, docs_dir, category, filename):
    """
    Writes an uploaded PDF into the docs/<category> folder and returns a file record
    in the same shape the ingestion pipeline expects. Rejects non-PDF files and
    flattens the filename so an upload cannot escape the docs directory.
    """
    if not filename.lower().endswith(".pdf"):
        raise ValueError(f"Only PDF files are supported (rejected: {filename})")

    safe_name = os.path.basename(filename)
    safe_category = os.path.basename(category) if category else "general"
    target_dir = os.path.join(docs_dir, safe_category) if safe_category != "general" else docs_dir
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_bytes)

    return {
        "path": file_path,
        "rel_path": os.path.relpath(file_path, docs_dir),
        "filename": safe_name,
        "category": safe_category if safe_category != "general" else "general",
        "hash": get_file_hash(file_path)
    }
```

- **Path Traversal Is Closed:** Both the filename and the category pass through `os.path.basename`,
  so `../../../etc/escaped.pdf` lands at `docs/escaped.pdf` instead of escaping the corpus.
- **Content Dedupe:** Re-uploading an identical file is detected by SHA-256 and reported as
  "Nothing new to index" rather than embedding duplicates.

### 3. Live Progress & Hot Reload
- **Two Indicators:** `st.status` provides a collapsible step log ("Saving...", "Scanned...", batch
  lines) while `st.progress` shows chunk-level completion against the exact pre-counted total.
- **Callback Writes From the Script Thread:** `run_async` drives the shared event loop on the same
  thread Streamlit renders on, so the `on_batch` callback can safely call `progress.progress()`.
- **Cache Invalidation:** Because the FAISS index is cached per process, the caches are dropped
  after ingestion so the next render deserializes the index from disk:

```python
def clear_caches():
    """Drops the cached index and registries so the next render reloads from disk."""
    get_vector_store.clear()
    get_index_registry.clear()
    get_scope_options.clear()
```

- **Surviving Confirmation:** `st.status` is cleared by the `st.rerun()` that follows ingestion,
  so the result is stashed in `st.session_state` and re-shown as a toast on the next render.

---

## Verified Execution

```bash
uv run streamlit run app.py
```

### Example Session

1. Choose a target folder (`docs/` or any existing subfolder) and drop a PDF on the sidebar uploader.
2. Press **Index Document**. The status log reports saving, the scanned chunk count, and then
   batch-by-batch embedding progress.
3. A toast confirms `Indexed 1 new chunks from 1 document(s) - now searchable`, the chunk counter
   in the sidebar increments, and the new document appears in the retrieval scope list.
4. Asking about the new document returns an answer citing it, with no restart required.

### Verification Results

Testing used a synthetic one-page PDF containing a single checkable fact ("47 credit units").

- **Hot reload:** upload moved the index from 114 to 115 chunks and the document appeared in the
  sidebar without a restart.
- **Immediately queryable:** "How many credit units is the compensation for a late Zephyr
  delivery?" answered *47 credit units*, citing `zephyr_handbook.pdf p. 1` as the first source.
- **Dedupe:** re-uploading the same bytes reported "Nothing new to index" and left the index at
  115 chunks.
- **Subfolder routing:** uploading into `docs/logistics/` created the folder, produced a
  `Topic: logistics` scope, and incrementing the index to 116 chunks.
- **Scoping is real:** restricted to `Topic: logistics`, a question about the Ethiopian
  constitution returned "I don't know." with zero citations, proving the filter excluded it.
- **Path traversal:** `../../../etc/escaped.pdf` resolved to `docs/escaped.pdf`; a traversal
  category resolved to a plain subfolder; `.txt` uploads raise `ValueError`.
- **Regression:** `uv run python main.py` still answers correctly through the modified ingestion
  function, and the server starts with no errors.

### Known Limitations

- **Re-uploading a modified file duplicates chunks.** The index is append-only
  (`add_embeddings` never deletes), so an updated PDF leaves its previous chunks behind. This
  predates the uploader but the uploader makes it easy to trigger.
- **New folders must exist on disk** to appear in the target picker; the UI lists folders rather
  than creating them.

### Testing Note

These tests write to `docs/` and `faiss_index/`. Back up both before running them, since FAISS
offers no API for removing individual vectors - restoring the files is the only way to undo a run.
