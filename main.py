import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader

async def load_document(file_path):
    """Load the document"""
    loader = PyPDFLoader(file_path)
    return loader.alazy_load()

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
    doc_iterator = await load_document(doc_path)
    print(f"Loading document from: {doc_path}")

    page_count = 0
    # Iterate through the pages and count them
    async for page in doc_iterator:
        page_count += 1
    
    print(f"\nSuccessfully loaded {page_count} pages.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
