import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    # Test to show splitting works
    chunk_count = 0
    print("Splitting document into chunks...")

    async for chunk in split_document(doc_iterator):
        chunk_count += 1
        # To avoid printing too much, we just count them and maybe print the first one
        if chunk_count == 1:
            print(f"\nExample of first chunk:\n{'-'*40}\n{chunk.page_content}\n{'-'*40}")
    
    print(f"\nSuccessfully generated {chunk_count} chunks!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
