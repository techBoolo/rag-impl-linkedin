import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

async def load_document(file_path):
    """Load the document"""
    loader = PyPDFLoader(file_path)
    return loader.alazy_load()

async def split_document(doc_iterator, chunk_size=1000, chunk_overlap=200):
    """Uses lazy splitting to avoid pulling the whole PDF into memory at once."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    async for doc in doc_iterator:
        # split the document into chunks and yield the chunk
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            yield chunk

def get_embeddings_model():
    """Return the Ollama embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text")

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

async def create_faiss_index_from_file(file_path, index_name="faiss_index"):
    """Orchestrates the lazy loading, splitting, and indexing process."""
    if os.path.exists(index_name):
        print("Index already exists. Please remove the index folder to reindex.")
        return
  
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    doc_iterator = await load_document(file_path)
    chunk_generator = split_document(doc_iterator)
    vector_store = None
    
    print(f"Loading document from: {file_path}")
    print("Splitting document and generating embeddings in batches...")
    
    batch_count = 0
    total_chunks_processed = 0

    async for batch, vectors in process_embeddings(chunk_generator, batch_size=10):
        batch_count += 1
        total_chunks_processed += len(batch)
        vector_store = await create_faiss_index(batch, vectors, vector_store)
        print(f"Processed batch {batch_count} ({len(batch)} chunks) -> Added to FAISS vector store. Total chunks embedded: {total_chunks_processed}")

    # Save the index to the local disk
    if vector_store:
        vector_store.save_local(index_name)
        print("\nSuccessfully created FAISS index with {} chunks across {} batches!".format(total_chunks_processed, batch_count))
        print("Index successfully saved to disk.")
    else:
        print("Error: No data was indexed.")

async def main():
    # Only if you have a get_chat_model function defined, otherwise skip or import it appropriately
    # llm = get_chat_model() 

    project_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(project_dir, 'docs', 'constitution.pdf')
    await create_faiss_index_from_file(doc_path)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
