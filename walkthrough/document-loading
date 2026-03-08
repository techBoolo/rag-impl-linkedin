# Document Loading Implementation

I have implemented the asynchronous document loading step for the RAG project. This allows the system to efficiently load large PDF documents without blocking the main execution thread.

## Changes Made

### 1. Dependencies Added
Installed `pypdf` and `langchain-community` to support PDF parsing and LangChain loaders.
```bash
uv add pypdf langchain-community
```

### 2. Code Update: `main.py`
Updated [main.py](file:///rag-project/main.py) to use `PyPDFLoader` with its asynchronous lazy loading capability (`alazy_load`).

```python
async def load_document(file_path):
    """Load the document"""
    loader = PyPDFLoader(file_path)
    return loader.alazy_load()

async def main():
    # ... directory setup ...
    doc_iterator = await load_document(doc_path)
    # ... iteration and counting ...
```

## Verification Results

I verified the implementation by running `main.py`, which successfully loaded the `constitution.pdf` file and counted **40 pages**.

```text
Loading document from: ./rag-project/docs/constitution.pdf

Successfully loaded 40 pages.
```
