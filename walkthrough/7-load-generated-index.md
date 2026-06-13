# Loading the Persisted FAISS Index from Disk

We have introduced a loading mechanism to retrieve the previously generated and saved FAISS index from the local file system. This allows the application to utilize the index immediately without having to reload, chunk, and re-embed the source PDF on every start.

## Highlights of `load_index`

A new function `load_index` has been added to retrieve and deserialize the vector store:

- **Same Embedding Model:** It retrieves the exact same embedding model (Ollama `nomic-embed-text`) used during indexing to ensure vector compatibility.
- **Dangerous Deserialization Option:** It calls `FAISS.load_local` with `allow_dangerous_deserialization=True` to safely load the metadata (`index.pkl`) alongside the vector index (`index.faiss`).

## Refactoring of `main`

The orchestrator `main()` function has been updated to conditionalize index generation:

- **Conditional Generation:** It first checks if the index directory (`faiss_index`) exists. If it's missing, it calls `create_faiss_index_from_file` to construct it.
- **Verification and Size Check:** It then loads the index into memory using `load_index` and prints the verified count of documents stored (`vector_store.index.ntotal`).

## Verified Output

When you run the script, the index is loaded dynamically from the disk:

```bash
uv run python main.py
```

Output:
```text
Attempting to load index from disk...
Index 'faiss_index' loaded successfully.
Verified loaded store size: 114 documents
```
