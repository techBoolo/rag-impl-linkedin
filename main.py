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
    Returns a dict of indexed files: {filename: {"path": ..., "hash": ...}}
    Reads from faiss_index/indexed_files.json if available.
    Otherwise bootstraps tracking from existing docstore metadata in vector_store.
    """
    metadata_path = os.path.join(index_dir, INDEX_METADATA_FILE)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {metadata_path}: {e}")

    # Fallback / bootstrap from existing vector_store docstore if index exists
    indexed = {}
    if vector_store and hasattr(vector_store, "docstore"):
        for doc in vector_store.docstore._dict.values():
            source = doc.metadata.get("source")
            if source:
                filename = os.path.basename(source)
                file_hash = doc.metadata.get("file_hash")
                if not file_hash and os.path.exists(source):
                    try:
                        file_hash = get_file_hash(source)
                    except Exception:
                        file_hash = None
                indexed[filename] = {
                    "path": source,
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
    Scans docs_dir for PDF files and classifies them into existing vs. new files.
    Deduplication checks filename and content hash.
    """
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        return [], []

    pdf_files = [
        os.path.join(docs_dir, f)
        for f in sorted(os.listdir(docs_dir))
        if f.lower().endswith(".pdf") and not f.startswith(".")
    ]

    new_files = []
    existing_files = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        current_hash = get_file_hash(pdf_path)

        # Check if already indexed by filename or hash
        is_already_indexed = False
        if filename in indexed_files:
            tracked_hash = indexed_files[filename].get("hash")
            if tracked_hash is None or tracked_hash == current_hash:
                is_already_indexed = True
        else:
            # Check by hash across all tracked files
            for info in indexed_files.values():
                if info.get("hash") == current_hash:
                    is_already_indexed = True
                    break

        if is_already_indexed:
            existing_files.append(pdf_path)
        else:
            new_files.append((pdf_path, current_hash))

    return existing_files, new_files

async def load_document(file_path):
    """Load the document lazily page by page."""
    loader = PyPDFLoader(file_path)
    return loader.alazy_load()

async def split_document(doc_iterator, file_hash=None, chunk_size=1000, chunk_overlap=200):
    """Uses lazy splitting to avoid pulling the whole PDF into memory at once."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    async for doc in doc_iterator:
        if file_hash:
            doc.metadata["file_hash"] = file_hash
        doc.metadata["filename"] = os.path.basename(doc.metadata.get("source", ""))
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
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

async def ingest_new_documents(new_files_with_hash, vector_store=None, index_name="faiss_index", indexed_files=None):
    """
    Lazily loads, chunks, embeds, and indexes only the new PDF documents in batches.
    Updates the vector_store and saves both the FAISS index and tracking registry.
    """
    if indexed_files is None:
        indexed_files = {}

    total_chunks_processed = 0
    total_batches_processed = 0

    for file_path, file_hash in new_files_with_hash:
        filename = os.path.basename(file_path)
        print(f"\n[Ingestion] Loading new document: {filename} ({file_path})")
        
        doc_iterator = await load_document(file_path)
        chunk_generator = split_document(doc_iterator, file_hash=file_hash)

        doc_chunks = 0
        async for batch, vectors in process_embeddings(chunk_generator, batch_size=10):
            total_batches_processed += 1
            doc_chunks += len(batch)
            total_chunks_processed += len(batch)
            vector_store = await create_faiss_index(batch, vectors, vector_store)
            print(f"  Processed batch {total_batches_processed} ({len(batch)} chunks) -> Total chunks embedded: {total_chunks_processed}")

        # Update tracking registry for this file
        indexed_files[filename] = {
            "path": file_path,
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

async def generate_answer(vector_store, query):
    """
    Takes a query, finds relevant context in the FAISS index across documents, 
    and generates an answer using an Ollama LLM.
    """
    # 1. Setup the LLM (Using Ollama)
    llm = get_chat_model()

    # 2. Define the Prompt Template
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

    # 3. Retrieve documents (using our search logic)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 4. Helper function to format the documents into a single string with source info
    def format_docs(docs):
        formatted = []
        for doc in docs:
            src = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "Document"))
            formatted.append(f"[{src}]:\n{doc.page_content}")
        return "\n\n".join(formatted)

    # 5. Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute the chain
    response = await rag_chain.ainvoke(query)
    return response

async def start_conversation(vector_store):
    """
    Handles the interactive loop between the user and the AI.
    """
    if not vector_store:
        print("Error: Vector store not found. Cannot start conversation.")
        return

    print("\n" + "="*50)
    print("MULTI-DOCUMENT KNOWLEDGE ASSISTANT")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nExiting conversation. Goodbye")
            break
        if not query:
            continue
        print("Thinking...")

        try:
            answer = await generate_answer(vector_store, query)
            print(f"\nAI: {answer}")
            print("-" * 30)
        except Exception as e:
            print(f"An error occurred while generating the answer: {e}")

async def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(project_dir, 'docs')
    index_name = "faiss_index"

    print("=" * 50)
    print("RAG PIPELINE: MULTI-DOCUMENT INGESTION & TRACKING")
    print("=" * 50)

    # 1. Attempt to load existing FAISS index
    vector_store = None
    if os.path.exists(index_name):
        print(f"\nAttempting to load existing index from '{index_name}'...")
        vector_store = await load_index(index_name)

    # 2. Inspect indexed files registry / docstore metadata
    indexed_files = get_indexed_files(vector_store, index_name)

    # 3. Scan docs directory and perform deduplication
    print(f"\nScanning documents in: {docs_dir}")
    existing_files, new_files = scan_docs_directory(docs_dir, indexed_files)

    print(f"Document scan results: {len(existing_files) + len(new_files)} total PDF(s) found.")
    print(f" - Already indexed: {len(existing_files)} file(s)")
    for ef in existing_files:
        print(f"   * {os.path.basename(ef)} (Skipped - already up to date)")

    print(f" - New or updated:  {len(new_files)} file(s)")
    for nf, _ in new_files:
        print(f"   * {os.path.basename(nf)} (Queued for ingestion)")

    # 4. Ingest new documents if present
    if new_files:
        print("\nStarting batch ingestion of new document(s)...")
        vector_store = await ingest_new_documents(
            new_files_with_hash=new_files,
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
        await start_conversation(vector_store)
    else:
        print("\nError: No vector store available. Please add PDF documents to 'docs/' and re-run.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
