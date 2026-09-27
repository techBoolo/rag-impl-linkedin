# Streamlit Web Chat Interface

We added `app.py`, a single-file Streamlit front end that reuses the existing ingestion and
query pipeline. The terminal chatbot and the browser UI now sit on top of the same
`load_index()` / `generate_answer()` functions.

---

## Key Highlights & Architecture

### 1. Shared Retrieval Chain
- **Chain Extraction (`build_rag_chain`):** The retriever, context formatter, prompt, and Ollama
  model were lifted out of `generate_answer` into a reusable `build_rag_chain(vector_store,
  active_filter)` so the CLI and the web UI build an identical chain.
- **Token Streaming (`stream_answer`):** A new async generator drives `rag_chain.astream(query)`,
  coercing each chunk to `str` so consumers can concatenate safely.
- **Backwards Compatible (`generate_answer`):** `generate_answer` keeps its previous signature and
  behaviour (`await rag_chain.ainvoke(query)` -> full string). The new `stream=True` flag returns
  the async token iterator instead, so no call site in `main.py` had to change.

```python
async def generate_answer(vector_store, query, active_filter=None, stream=False):
    if stream:
        return stream_answer(vector_store, query, active_filter=active_filter)

    rag_chain = build_rag_chain(vector_store, active_filter)
    response = await rag_chain.ainvoke(query)
    return response
```

### 2. Index Caching & the Shared Event Loop
- **Index Loaded Once (`@st.cache_resource`):** `get_vector_store()` deserializes `faiss_index/`
  a single time per server process and every browser session reuses that in-memory object.
  The ingestion registry (`get_indexed_files`) is cached the same way.
- **Persistent Event Loop (`get_event_loop`):** Ollama's async client is cached inside the
  vector store's embeddings and is bound to whichever event loop first used it. Calling
  `asyncio.run()` per query would close that loop and break every subsequent question with
  `RuntimeError: Event loop is closed`. A single loop is therefore created once and kept alive,
  with a `threading.Lock` serializing access because Streamlit runs each session on its own thread.
- **Graceful Degradation:** If no index exists yet, the app shows instructions to run
  `main.py` instead of raising.

### 3. Chat UI
- **Session History (`st.session_state.messages`):** Messages are stored as
  `{"role", "content"}` dicts, appended after a completed turn, and replayed on each rerun so
  the transcript survives Streamlit's script re-execution.
- **Rendering (`st.chat_message`):** History is drawn as alternating user/assistant bubbles, and
  the in-flight answer is written into an `st.empty()` placeholder.
- **Input (`st.chat_input`):** Returns a prompt that is immediately echoed as a user bubble before
  generation starts.
- **Incremental Painting:** Each token re-renders the accumulated text with a `▌` cursor, which
  gives the live typing effect; the cursor is dropped once the stream completes.

### 4. Retrieval Scope Selector
- The sidebar reuses `get_available_topics_and_docs` to list every topic and document in the index.
- The selection is encoded as `type::value` and converted back with `to_active_filter` into the
  exact dict shape `generate_answer` already expects, so scoping behaves identically to the CLI's
  startup menu and its `/topic` command.
- A **New chat** button clears the transcript via `st.rerun()` while leaving the warm index intact.

---

## Verified Execution

Run the CLI once to build the index, then start the UI:

```bash
uv run python main.py            # ingest docs/ into faiss_index/
uv run streamlit run app.py      # open http://localhost:8501
```

### Example Session

1. Sidebar shows `All Topics & Documents`, `Topic: general (1 document)`, and
   `Document: constitution.pdf [Topic: general]`, plus `114 indexed chunks`.
2. Asking *"What is required to amend the constitution?"* streams the answer token by token.
3. Switching the sidebar to `Document: constitution.pdf` and asking
   *"And how is the President replaced?"* scopes retrieval to that document.
4. **New chat** empties the transcript; the index stays loaded, so the next question starts instantly.

Multi-turn streaming was verified headlessly with `streamlit.testing.v1.AppTest`: three
consecutive turns across all three scope modes produced answers with no errors, and the
`generate_answer` non-streaming path still returns a full string for the CLI.
