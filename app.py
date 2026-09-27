import asyncio
import os
import threading

import streamlit as st

from main import (
    generate_answer,
    get_available_topics_and_docs,
    get_indexed_files,
    load_index,
    retrieve_sources,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(PROJECT_DIR, "faiss_index")

SCOPE_ALL = "__all__"
SCOPE_SEPARATOR = "::"
STREAM_CURSOR = "▌"

LOOP_LOCK = threading.Lock()

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource
def get_event_loop():
    """
    Keeps a single event loop alive for the whole server process.
    A fresh loop per call would strand the async HTTP client cached inside the
    vector store's embeddings, so every later query would fail with a closed loop.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def run_async(coroutine):
    """Runs a coroutine on the shared event loop, one query at a time."""
    loop = get_event_loop()
    with LOOP_LOCK:
        return loop.run_until_complete(coroutine)


@st.cache_resource(show_spinner="Loading FAISS index...")
def get_vector_store():
    """Loads the persisted FAISS index once per server process and reuses it across sessions."""
    if not os.path.exists(INDEX_DIR):
        return None
    return run_async(load_index(INDEX_DIR))


@st.cache_resource
def get_index_registry():
    """Reads the ingestion tracking registry, bootstrapping from the docstore when needed."""
    return get_indexed_files(get_vector_store(), INDEX_DIR)


@st.cache_data
def get_scope_options(indexed_files):
    """Builds the retrieval scope menu from the topics and documents already in the index."""
    categories, documents = get_available_topics_and_docs(indexed_files, get_vector_store())

    options = {SCOPE_ALL: "All Topics & Documents"}
    for category, count in sorted(categories.items()):
        plural = "document" if count == 1 else "documents"
        options[f"category{SCOPE_SEPARATOR}{category}"] = f"Topic: {category} ({count} {plural})"
    for document in documents:
        options[f"filename{SCOPE_SEPARATOR}{document['filename']}"] = (
            f"Document: {document['filename']} [Topic: {document['category']}]"
        )
    return options


def to_active_filter(scope):
    """Converts a sidebar scope choice into the filter dict expected by the RAG chain."""
    if scope == SCOPE_ALL:
        return None
    filter_type, _, filter_value = scope.partition(SCOPE_SEPARATOR)
    return {"type": filter_type, "value": filter_value}


def page_reference(doc):
    """Human-readable page label, preferring the PDF's own label over a 1-based index."""
    metadata = doc.metadata
    label = metadata.get("page_label")
    if label:
        return f"p. {label}"
    if metadata.get("page") is not None:
        return f"p. {int(metadata['page']) + 1}"
    return "page unknown"


def to_sources(docs):
    """Condenses retrieved chunks into citation records, one per document page."""
    sources = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata
        filename = metadata.get("filename") or os.path.basename(metadata.get("source", "Document"))
        key = (filename, metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "filename": filename,
                "category": metadata.get("category", "general"),
                "page": page_reference(doc),
                "excerpt": doc.page_content.strip(),
            }
        )
    return sources


def render_sources(sources):
    """Renders the retrieved chunks behind an answer as expandable citations."""
    if not sources:
        return
    with st.expander(f"📚 View Sources ({len(sources)})"):
        for position, source in enumerate(sources, start=1):
            label = f"{position}. 📄 {source['filename']} — {source['page']} · {source['category']}"
            with st.expander(label):
                st.text(source["excerpt"])


def stream_answer_into(placeholder, vector_store, query, active_filter, docs=None):
    """Streams the answer into a placeholder, rendering each token as it arrives."""
    async def render():
        answer = ""
        token_stream = await generate_answer(
            vector_store, query, active_filter=active_filter, stream=True, docs=docs
        )
        try:
            async for chunk in token_stream:
                answer += chunk
                placeholder.markdown(f"{answer}{STREAM_CURSOR}")
        finally:
            await token_stream.aclose()
        placeholder.markdown(answer)
        return answer

    return run_async(render())


def init_session_state():
    """Initializes the per-session chat transcript."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_message(message):
    """Renders a single stored chat message, including its citations."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))


def main():
    init_session_state()

    vector_store = get_vector_store()
    if vector_store is None:
        st.error(
            f"No FAISS index found at `{INDEX_DIR}`.\n\n"
            "Run `uv run python main.py` once to ingest the PDFs from `docs/`, then reload this page."
        )
        st.stop()

    indexed_files = get_index_registry()
    scope_options = get_scope_options(indexed_files)
    total_chunks = getattr(vector_store.index, "ntotal", 0)

    with st.sidebar:
        st.header("Retrieval Scope")
        st.caption("Restrict retrieval to a topic or a single document.")
        scope = st.selectbox(
            "Search",
            options=list(scope_options),
            format_func=lambda key: scope_options[key],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption(
            f"**{len(scope_options) - 1}** scoped option(s) · "
            f"**{total_chunks}** indexed chunks"
        )

        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.get("active_scope") != scope:
        st.session_state.messages = []
        st.session_state["active_scope"] = scope

    st.title("📚 Knowledge Assistant")
    st.caption(f"Retrieval scope: **{scope_options[scope]}**")

    if not st.session_state.messages:
        st.info("Ask a question about the indexed documents to get started.")

    for message in st.session_state.messages:
        render_message(message)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message(st.session_state.messages[-1])

        with st.chat_message("assistant"):
            answer, sources = None, []
            try:
                with st.spinner("Searching the index..."):
                    docs = run_async(
                        retrieve_sources(vector_store, prompt, to_active_filter(scope))
                    )
                    sources = to_sources(docs)
                placeholder = st.empty()
                answer = stream_answer_into(
                    placeholder, vector_store, prompt, to_active_filter(scope), docs=docs
                )
                render_sources(sources)
            except Exception as e:
                st.error(f"Failed to generate an answer: {e}")

        if answer is not None:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )


if __name__ == "__main__":
    main()
