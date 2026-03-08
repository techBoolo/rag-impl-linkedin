# Document Loading & Splitting Implementation

I have implemented the asynchronous document loading and efficient document splitting steps for the RAG project.

## Changes Made

### 1. Dependencies Added
Installed `pypdf`, `langchain-community`, and `langchain-text-splitters`.
```bash
uv add pypdf langchain-community langchain-text-splitters
```

### 2. Code Update: `main.py`
Updated [main.py](file:///rag-project/main.py) with the following features:
- **Async Loading**: Uses `PyPDFLoader.alazy_load()` to stream document pages.
- **Lazy Splitting**: Implemented `split_document` as an async generator that uses `RecursiveCharacterTextSplitter`. This allows processing large documents chunk-by-chunk without overwhelming memory.

```python
async def split_document(doc_iterator, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    async for doc in doc_iterator:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            yield chunk
```

## Verification Results

### Document Loading
Successfully loaded the `constitution.pdf` file with **40 pages**.

### Document Splitting
Successfully split the document into **114 chunks** (using 1000 char chunks with 200 char overlap).

```text
Loading document from: ./rag-project/docs/constitution.pdf
Splitting document into chunks...

Example of first chunk:
----------------------------------------
CONSTITUTION OF THE FEDERAL DEMOCRATIC 
REPUBLIC OF ETHIOPIA
...
----------------------------------------

Successfully generated 114 chunks!
```
