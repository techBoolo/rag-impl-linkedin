# Batch Embedding Implementation

This document covers the recently implemented changes to support asynchronous batch embeddings generation in our RAG pipeline.

## Achieved Goals
- Pulled the `nomic-embed-text` model via Ollama to run high-quality text embeddings locally.
- Integrated `langchain_ollama.OllamaEmbeddings` into the `main.py` script.
- Adopted an asynchronous, memory-efficient batch processing technique instead of accumulating all embeddings in-memory or processing chunks one-by-one sequentially.

## Changes Made
```diff
+from langchain_ollama import OllamaEmbeddings

+def get_embeddings_model():
+    """Return the Ollama embeddings model."""
+    return OllamaEmbeddings(model="nomic-embed-text")

+async def process_embeddings(chunk_generator, batch_size=10):
+    """Batches chunks and calls aembed_documents asynchronously."""
+    embeddings_model = get_embeddings_model()
+    batch = []
+    
+    async for chunk in chunk_generator:
+        batch.append(chunk)
+        if len(batch) >= batch_size:
+            texts = [c.page_content for c in batch]
+            vectors = await embeddings_model.aembed_documents(texts)
+            yield batch, vectors
+            batch = []
+            
+    # Process remaining chunks
+    if batch:
+        texts = [c.page_content for c in batch]
+        vectors = await embeddings_model.aembed_documents(texts)
+        yield batch, vectors
```

The script `main.py` was updated to iterate over `process_embeddings` instead of directly consuming output from the splitter, ensuring a memory-friendly generator chain.

## Validation Results
We ran the script and successfully generated embedding vectors.
- Output demonstrated `114 chunks` encoded in `12 batches` (default batch size: 10).
- Validated standard model output size: `768` dimensions for the `nomic-embed-text` generated vector.
