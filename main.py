import asyncio
import hashlib
import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

INDEX_METADATA_FILE = "indexed_files.json"

def get_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash for a file to track file identity and modifications."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_indexed_files(vector_store, index_dir="faiss_index"):
    """
    Returns a dict of indexed files: {rel_path: {"path": ..., "category": ..., "filename": ..., "hash": ...}}
    Reads from faiss_index/indexed_files.json if available.
    Otherwise bootstraps tracking from existing docstore metadata in vector_store.
    """
    metadata_path = os.path.join(index_dir, INDEX_METADATA_FILE)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                normalized = {}
                for key, info in raw_data.items():
                    path = info.get("path", "")
                    filename = info.get("filename") or os.path.basename(path or key)
                    category = info.get("category", "general")
                    normalized[key] = {
                        "path": path,
                        "category": category,
                        "filename": filename,
                        "hash": info.get("hash")
                    }
                return normalized
        except Exception as e:
            print(f"Warning: Could not read {metadata_path}: {e}")

    # Fallback / bootstrap from existing vector_store docstore if index exists
    indexed = {}
    if vector_store and hasattr(vector_store, "docstore"):
        for doc in vector_store.docstore._dict.values():
            source = doc.metadata.get("source")
            if source:
                filename = doc.metadata.get("filename") or os.path.basename(source)
                category = doc.metadata.get("category", "general")
                file_hash = doc.metadata.get("file_hash")
                if not file_hash and os.path.exists(source):
                    try:
                        file_hash = get_file_hash(source)
                    except Exception:
                        file_hash = None
                indexed[filename] = {
                    "path": source,
                    "category": category,
                    "filename": filename,
                    "hash": file_hash
                }
    return indexed

def save_indexed_files(indexed_files, index_dir="faiss_index"):
    """Saves the tracking registry to disk inside the index directory."""
    os.makedirs(index_dir, exist_ok=True)
    metadata_path = os.path.join(index_dir, INDEX_METADATA_FILE)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(indexed_files, f, indent=2)

def scan_docs_directory(docs_dir, indexed_files):
    """
    Recursively scans docs_dir for PDF files across all subdirectories.
    Derives category from folder name (relative to docs_dir).
    Performs deduplication against already indexed files by relative path, filename, and content hash.
    """
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        return [], []

    pdf_files = []
    for root, dirs, files in os.walk(docs_dir):
        # Ignore hidden folders
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in sorted(files):
            if file.lower().endswith(".pdf") and not file.startswith("."):
                pdf_files.append(os.path.join(root, file))

    new_files = []
    existing_files = []

    for pdf_path in pdf_files:
        rel_path = os.path.relpath(pdf_path, docs_dir)
        filename = os.path.basename(pdf_path)
        rel_dir = os.path.dirname(rel_path)
        category = "general" if not rel_dir or rel_dir == "." else rel_dir
        current_hash = get_file_hash(pdf_path)

        # Check if already indexed by relative path, filename, or content hash
        is_already_indexed = False
        if rel_path in indexed_files:
            tracked_hash = indexed_files[rel_path].get("hash")
            if tracked_hash is None or tracked_hash == current_hash:
                is_already_indexed = True
        elif filename in indexed_files:
            tracked_hash = indexed_files[filename].get("hash")
            if tracked_hash is None or tracked_hash == current_hash:
                is_already_indexed = True

        if not is_already_indexed:
            for info in indexed_files.values():
                if info.get("hash") == current_hash:
                    is_already_indexed = True
                    break

        file_record = {
            "path": pdf_path,
            "rel_path": rel_path,
            "filename": filename,
            "category": category,
            "hash": current_hash
        }

        if is_already_indexed:
            existing_files.append(file_record)
        else:
            new_files.append(file_record)

    return existing_files, new_files

async def load_document(file_path):
    """Load the document lazily page by page."""
    loader = PyPDFLoader(file_path)
    return loader.alazy_load()

async def split_document(doc_iterator, category="general", filename=None, file_hash=None, chunk_size=1000, chunk_overlap=200):
    """
    Uses lazy splitting to avoid pulling the whole PDF into memory at once.
    Tags each chunk with category (subfolder), filename, source path, and content hash.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    async for doc in doc_iterator:
        source_path = doc.metadata.get("source", "")
        doc_filename = filename or os.path.basename(source_path)
        doc.metadata["category"] = category
        doc.metadata["filename"] = doc_filename
        if file_hash:
            doc.metadata["file_hash"] = file_hash

        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata["category"] = category
            chunk.metadata["filename"] = doc_filename
            if file_hash:
                chunk.metadata["file_hash"] = file_hash
            yield chunk

def get_embeddings_model():
    """Return the Ollama embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def get_chat_model(model="llama3.1"):
    """Return the Ollama chat model."""
    return ChatOllama(model=model)

async def create_faiss_index(batch, vectors, vector_store=None):
    """Adds pre-computed embeddings to the FAISS index."""
    embeddings = get_embeddings_model()
    text_embedding_pairs = list(zip([chunk.page_content for chunk in batch], vectors))
    metadatas = [chunk.metadata for chunk in batch]
    if vector_store is None:
        # Initial creation
        vector_store = await FAISS.afrom_embeddings(
            text_embedding_pairs, 
            embeddings, 
            metadatas
        )
    else:
        # Append to existing index
        vector_store.add_embeddings(
            text_embeddings=text_embedding_pairs, 
            metadatas=metadatas
        )

    return vector_store

async def process_embeddings(chunk_generator, batch_size=10):
    """Batches chunks and calls aembed_documents asynchronously."""
    embeddings_model = get_embeddings_model()
    batch = []
    
    async for chunk in chunk_generator:
        batch.append(chunk)
        if len(batch) >= batch_size:
            texts = [c.page_content for c in batch]
            vectors = await embeddings_model.aembed_documents(texts)
            yield batch, vectors
            batch = []
            
    # Process remaining chunks
    if batch:
        texts = [c.page_content for c in batch]
        vectors = await embeddings_model.aembed_documents(texts)
        yield batch, vectors

async def ingest_new_documents(new_files, vector_store=None, index_name="faiss_index", indexed_files=None):
    """
    Lazily loads, chunks, embeds, and indexes only the new PDF documents in batches.
    Updates the vector_store and saves both the FAISS index and tracking registry.
    """
    if indexed_files is None:
        indexed_files = {}

    total_chunks_processed = 0
    total_batches_processed = 0

    for file_record in new_files:
        file_path = file_record["path"]
        category = file_record["category"]
        filename = file_record["filename"]
        file_hash = file_record["hash"]
        rel_path = file_record["rel_path"]

        print(f"\n[Ingestion] Loading new document: {filename} [Category: {category}] ({file_path})")
        
        doc_iterator = await load_document(file_path)
        chunk_generator = split_document(
            doc_iterator, 
            category=category, 
            filename=filename, 
            file_hash=file_hash
        )

        doc_chunks = 0
        async for batch, vectors in process_embeddings(chunk_generator, batch_size=10):
            total_batches_processed += 1
            doc_chunks += len(batch)
            total_chunks_processed += len(batch)
            vector_store = await create_faiss_index(batch, vectors, vector_store)
            print(f"  Processed batch {total_batches_processed} ({len(batch)} chunks) -> Total chunks embedded: {total_chunks_processed}")

        # Update tracking registry for this file
        indexed_files[rel_path] = {
            "path": file_path,
            "category": category,
            "filename": filename,
            "hash": file_hash
        }
        print(f"  Finished indexing {filename} ({doc_chunks} chunks).")

    if vector_store:
        vector_store.save_local(index_name)
        save_indexed_files(indexed_files, index_name)
        print(f"\nSuccessfully saved updated FAISS index to '{index_name}' with {total_chunks_processed} new chunks added!")

    return vector_store

async def load_index(index_name="faiss_index"):
    """
    Loads the FAISS index from disk.
    Returns the vector_store object ready for similarity search.
    """
    if not os.path.exists(index_name):
        return None

    embeddings = get_embeddings_model()
    try:
        vector_store = FAISS.load_local(
            index_name, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"Index '{index_name}' loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading index '{index_name}': {e}")
        return None

def get_available_topics_and_docs(indexed_files, vector_store=None):
    """
    Discovers unique categories (topics) and documents from indexed_files registry
    and the loaded vector_store docstore.
    Returns:
      categories: dict of {category_name: count_of_docs}
      documents: list of dicts [{"filename": ..., "category": ..., "path": ...}]
    """
    docs_by_name = {}

    for key, info in indexed_files.items():
        fname = info.get("filename") or os.path.basename(info.get("path", key))
        cat = info.get("category", "general")
        docs_by_name[fname] = {
            "filename": fname,
            "category": cat,
            "path": info.get("path", "")
        }

    # Supplement from vector store docstore if available
    if vector_store and hasattr(vector_store, "docstore"):
        for doc in vector_store.docstore._dict.values():
            src = doc.metadata.get("source", "")
            fname = doc.metadata.get("filename") or (os.path.basename(src) if src else None)
            if fname and fname not in docs_by_name:
                cat = doc.metadata.get("category", "general")
                docs_by_name[fname] = {
                    "filename": fname,
                    "category": cat,
                    "path": src
                }

    categories = {}
    for doc_info in docs_by_name.values():
        cat = doc_info["category"]
        categories[cat] = categories.get(cat, 0) + 1

    return categories, sorted(docs_by_name.values(), key=lambda x: (x["category"], x["filename"]))

def select_topic_filter(categories, documents):
    """
    Interactive CLI selection for topic (category) or document filtering.
    Returns filter definition dict or None for no filter.
    """
    print("\n" + "=" * 50)
    print("TOPIC & DOCUMENT RETRIEVAL FILTER")
    print("=" * 50)

    options = []
    # Option 1: All Topics
    options.append({
        "label": "All Topics & Documents (No filter)",
        "filter": None
    })

    # Category options
    print("Available Topics (Categories):")
    idx = 1
    print(f"  [{idx}] All Topics & Documents (No filter)")
    for cat, count in sorted(categories.items()):
        idx += 1
        options.append({
            "label": f"Topic: {cat} ({count} document{'s' if count != 1 else ''})",
            "filter": {"type": "category", "value": cat}
        })
        print(f"  [{idx}] Topic: {cat} ({count} document{'s' if count != 1 else ''})")

    # Document options
    print("\nAvailable Documents:")
    for doc in documents:
        idx += 1
        fname = doc["filename"]
        cat = doc["category"]
        options.append({
            "label": f"Document: {fname} [Topic: {cat}]",
            "filter": {"type": "filename", "value": fname}
        })
        print(f"  [{idx}] Document: {fname} [Topic: {cat}]")

    print("-" * 50)
    user_choice = input(f"Select retrieval scope [1-{len(options)}] (Press Enter for 'All'): ").strip()

    if not user_choice or user_choice in ["1", "all", "none"]:
        return None

    # Check numeric selection
    if user_choice.isdigit():
        choice_idx = int(user_choice) - 1
        if 0 <= choice_idx < len(options):
            return options[choice_idx]["filter"]

    # Check matching category or filename by name
    lower_choice = user_choice.lower()
    for cat in categories:
        if cat.lower() == lower_choice:
            return {"type": "category", "value": cat}
    for doc in documents:
        if doc["filename"].lower() == lower_choice:
            return {"type": "filename", "value": doc["filename"]}

    print(f"Invalid selection '{user_choice}'. Defaulting to All Topics & Documents.")
    return None

async def generate_answer(vector_store, query, active_filter=None):
    """
    Takes a query, finds relevant context in the FAISS index with optional topic/doc filter,
    and generates an answer using an Ollama LLM.
    """
    llm = get_chat_model()

    template = """
    You are an assistant for question-answering tasks based on the provided documents.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Configure search kwargs with optional metadata filter
    search_kwargs = {"k": 4}
    if active_filter:
        filter_type = active_filter.get("type")
        filter_val = active_filter.get("value")

        if filter_type == "category":
            search_kwargs["filter"] = lambda m: m.get("category", "general").lower() == filter_val.lower()
        elif filter_type == "filename":
            search_kwargs["filter"] = lambda m: (
                m.get("filename", "").lower() == filter_val.lower()
                or os.path.basename(m.get("source", "")).lower() == filter_val.lower()
            )

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    def format_docs(docs):
        if not docs:
            return "No relevant context found matching the selected filter."
        formatted = []
        for doc in docs:
            src = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "Document"))
            cat = doc.metadata.get("category", "general")
            formatted.append(f"[{cat} / {src}]:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = await rag_chain.ainvoke(query)
    return response

async def start_conversation(vector_store, indexed_files=None):
    """
    Handles interactive topic selection and the continuous conversation loop.
    """
    if not vector_store:
        print("Error: Vector store not found. Cannot start conversation.")
        return

    categories, documents = get_available_topics_and_docs(indexed_files or {}, vector_store)
    active_filter = select_topic_filter(categories, documents)

    def get_filter_label(flt):
        if not flt:
            return "All Topics & Documents"
        if flt["type"] == "category":
            return f"Topic: {flt['value']}"
        return f"Document: {flt['value']}"

    print("\n" + "=" * 50)
    print("MULTI-DOCUMENT KNOWLEDGE ASSISTANT")
    print(f"Active Retrieval Scope: [{get_filter_label(active_filter)}]")
    print("Commands:")
    print(" - Type your question to query the assistant.")
    print(" - Type '/filter' or '/topic' to change retrieval scope.")
    print(" - Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        prompt_label = f"[{get_filter_label(active_filter)}] You: "
        query = input(prompt_label).strip()

        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nExiting conversation. Goodbye!")
            break
        if not query:
            continue
        if query.lower() in ['/filter', '/topic', 'filter', 'topic']:
            active_filter = select_topic_filter(categories, documents)
            print(f"\nSwitched retrieval scope to: [{get_filter_label(active_filter)}]\n")
            continue

        print("Thinking...")
        try:
            answer = await generate_answer(vector_store, query, active_filter=active_filter)
            print(f"\nAI: {answer}")
            print("-" * 30)
        except Exception as e:
            print(f"An error occurred while generating the answer: {e}")

async def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(project_dir, 'docs')
    index_name = "faiss_index"

    print("=" * 50)
    print("RAG PIPELINE: RECURSIVE INGESTION & TOPIC SELECTION")
    print("=" * 50)

    # 1. Attempt to load existing FAISS index
    vector_store = None
    if os.path.exists(index_name):
        print(f"\nAttempting to load existing index from '{index_name}'...")
        vector_store = await load_index(index_name)

    # 2. Inspect indexed files registry / docstore metadata
    indexed_files = get_indexed_files(vector_store, index_name)

    # 3. Recursively scan docs directory and perform deduplication
    print(f"\nRecursively scanning documents in: {docs_dir}")
    existing_files, new_files = scan_docs_directory(docs_dir, indexed_files)

    print(f"Document scan results: {len(existing_files) + len(new_files)} total PDF(s) found.")
    print(f" - Already indexed: {len(existing_files)} file(s)")
    for ef in existing_files:
        print(f"   * [{ef['category']}] {ef['filename']} (Skipped - already up to date)")

    print(f" - New or updated:  {len(new_files)} file(s)")
    for nf in new_files:
        print(f"   * [{nf['category']}] {nf['filename']} (Queued for ingestion)")

    # 4. Ingest new documents if present
    if new_files:
        print("\nStarting batch ingestion of new document(s)...")
        vector_store = await ingest_new_documents(
            new_files=new_files,
            vector_store=vector_store,
            index_name=index_name,
            indexed_files=indexed_files
        )
    else:
        # If we had an existing index and bootstrap metadata wasn't saved yet, save it now
        if vector_store and not os.path.exists(os.path.join(index_name, INDEX_METADATA_FILE)):
            save_indexed_files(indexed_files, index_name)
        print("\nNo new documents to ingest. Vector store is up to date.")

    # 5. Verify index state and launch conversation loop
    if vector_store:
        total_vectors = getattr(vector_store.index, "ntotal", 0)
        print(f"\nVerified vector store size: {total_vectors} document chunks.")
        await start_conversation(vector_store, indexed_files=indexed_files)
    else:
        print("\nError: No vector store available. Please add PDF documents to 'docs/' and re-run.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
