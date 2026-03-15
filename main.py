import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

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

async def main():
    # Get the directory of the current script
    project_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the path to the docs folder and the PDF file
    doc_dir = os.path.join(project_dir, 'docs')
    doc_path = os.path.join(doc_dir, 'constitution.pdf')

    # Check if the file exists
    if not os.path.exists(doc_path):
        print(f"Error: Document not found at {doc_path}")
        return

    # Load the document asynchronously
    print(f"Loading document from: {doc_path}")
    doc_iterator = await load_document(doc_path)

    print("Splitting document and generating embeddings in batches...")
    batch_count = 0
    total_chunks_processed = 0

    async for batch, vectors in process_embeddings(split_document(doc_iterator), batch_size=10):
        batch_count += 1
        total_chunks_processed += len(batch)
        print(f"Processed batch {batch_count} ({len(batch)} chunks). Total chunks embedded: {total_chunks_processed}")
        
        if batch_count == 1:
            print(f"Example vector dimension for first chunk in batch 1: {len(vectors[0])}")
    
    print(f"\nSuccessfully generated embeddings for {total_chunks_processed} chunks across {batch_count} batches!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
