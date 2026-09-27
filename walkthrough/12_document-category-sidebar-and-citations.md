# Document Category Sidebar & Source Citations

We added two capabilities to the web interface: full control over which topic the assistant
retrieves from, and verifiable citations that expose the exact pages behind every answer.
The second one required reworking the RAG chain, because the `StrOutputParser` at the end of
the chain was discarding the retrieved chunks.

---

## Key Highlights & Architecture

### 1. Retrieval Split from Answer Generation
- **Retriever Extracted (`build_retriever`):** The `search_kwargs` construction and the callable
  metadata filter moved out of the chain builder into a reusable `build_retriever(vector_store,
  active_filter)`. The filter logic now exists in exactly one place instead of being inlined.
- **Context Formatter Promoted (`format_docs`):** Hoisted to module level so both the retriever
  pipeline and the pre-fetched path can use it.
- **Raw Chunks Exposed (`retrieve_sources`):** Runs the similarity search and returns the raw
  `Document` objects that the caller can turn into citations:

```python
async def retrieve_sources(vector_store, query, active_filter=None):
    """
    Runs similarity search and returns the raw retrieved chunks so callers can
    surface them as citations (document name, page, and matched text).
    """
    retriever = build_retriever(vector_store, active_filter)
    return await retriever.ainvoke(query)
```

- **Single Retrieval Pass (`build_rag_chain(..., docs=None)`):** When pre-fetched chunks are
  supplied, the chain substitutes a fixed-context step for the retriever step, so the answer is
  generated from *exactly* the chunks shown as citations. Running the retriever again would risk
  citing one set of passages while answering from another:

```python
    if docs is None:
        context_step = build_retriever(vector_store, active_filter) | format_docs
    else:
        context_step = RunnableLambda(lambda _query: format_docs(docs))
```

- **Unchanged Defaults:** Both parameters are optional and default to the previous behaviour, so
  the terminal chatbot is completely unaffected. Note the guard is `docs is None`, not a truthiness
  check - an empty list must still reach `format_docs` so the model is told no context was found.

### 2. Grounded Citations with Page References
- **Page Metadata Was Already There:** `PyPDFLoader` stamps each chunk with `page` (0-indexed) and
  `page_label` (the PDF's own page numbering), so page-accurate citations required **no re-indexing**.
- **Label Preference (`page_reference`):** Uses the PDF's own label so citations read `p. 40`
  rather than a possibly-mismatched derived offset, falling back to `page + 1`:

```python
def page_reference(doc):
    """Human-readable page label, preferring the PDF's own label over a 1-based index."""
    metadata = doc.metadata
    label = metadata.get("page_label")
    if label:
        return f"p. {label}"
    if metadata.get("page") is not None:
        return f"p. {int(metadata['page']) + 1}"
    return "page unknown"
```

- **Page-Level Dedupe (`to_sources`):** A single page can match several of the `k=4` chunks, so
  citations are deduped on `(filename, page)`. A typical answer collapses from 4 chunks to 3
  distinct pages instead of showing the same page twice.
- **Expandable Display (`render_sources`):** Each answer gets a `📚 View Sources (N)` expander
  containing one nested expander per source, labelled `📄 <filename> — p. <page> · <category>`,
  with the matched chunk text inside.

### 3. Session State & Topic Switching
- **Citations Persist Across Reruns:** Sources are stored on the message record
  (`{"role", "content", "sources"}`) and re-rendered by `render_message`, so citations do not
  disappear the moment the next question is asked.
- **History Reset on Scope Change:** The sidebar scope is mirrored into
  `st.session_state["active_scope"]`; when it differs from the current selection the transcript is
  cleared, because a conversation that mixes topics is misleading about what was searched:

```python
    if st.session_state.get("active_scope") != scope:
        st.session_state.messages = []
        st.session_state["active_scope"] = scope
```

---

## Verified Execution

```bash
uv run streamlit run app.py
```

### Example Session

Asking *"What is required to amend the constitution?"* returns an answer plus three citations
(the search matched four chunks, but two came from the same page):

```text
To amend the constitution, Article 105 requires a two-thirds majority vote in the House of
Peoples' Representatives or the House of the Federation...

📚 View Sources (3)
   1. 📄 constitution.pdf — p. 40 · general
   2. 📄 constitution.pdf — p. 24 · general
   3. 📄 constitution.pdf — p. 32 · general
```

Expanding source 1 shows `Article 104 Initiation of Amendments Any proposal for constitutional
amendment, if supported by two-thirds majority vote...`, which is the passage the answer is
actually grounded in.

Switching the sidebar to a single document then clearing the transcript leaves the empty state,
so a new conversation cannot mix scopes.

### Verification Results

- Citations grounded: the p. 40 excerpt contains the two-thirds rule the answer cites.
- Persistence: after a second turn the first answer's three expanders are still rendered (6 total).
- Dedupe: 4 retrieved chunks -> 3 citations, with the duplicate page collapsed.
- Topic switch: transcript cleared, welcome state restored, expanders removed.
- Empty retrieval (filter matching nothing): the chain still answers `I don't know.` instead of
  passing empty context to the model.
- Regression: `uv run python main.py` still answers correctly through the refactored chain.
- `streamlit.testing.v1.AppTest`: three-turn multi-scope session with no errors or exceptions.

### Known Limitation

Chunks indexed before metadata tagging carry no `category` key, so it resolves to `"general"` via
`.get()` and a category filter would match every document. This is invisible with a single
document, but adding a second folder requires re-ingestion to backfill `category`.
