# RAG Project: Conversation Loop & Answer Generation

This project is an end-to-end local Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It loads, processes, indexes PDF documents into a FAISS vector store, and provides an interactive terminal chatbot to answer questions based on the document.

---

## 🚀 Features

- **Local LLM & Embeddings**: Powered by Ollama (`llama3.1` for chat and `nomic-embed-text` for vector embeddings).
- **RAG Generation Chain**: Built with LangChain Expression Language (LCEL) connecting similarity retrieval, concise context-based prompting, and Ollama.
- **Interactive Chatbot CLI**: Continuous interactive conversation loop in the terminal with conversational feedback and graceful exit handling.
- **Async Processing**: High-performance asynchronous document loading using `alazy_load` and concurrent batch embeddings using `aembed_documents`.
- **Memory Efficiency**: Document splitting using `RecursiveCharacterTextSplitter` and lazy embedding chunks via async generators.
- **Index Reusability**: Dynamically loads the persisted FAISS index from disk (`faiss_index/`), skipping redundant PDF processing and embedding steps.
- **Modern Tooling**: Managed by `uv` for lightning-fast dependency management and environment isolation.

---

## 🛠️ Setup

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- [uv](https://github.com/astral-sh/uv) installed.

### 2. Pull the Models
Ensure the `llama3.1` and `nomic-embed-text` models are available locally:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### 3. Initialize & Install Dependencies
```bash
# Initialize project environment
uv venv
source .venv/bin/activate

# Install core and RAG-specific dependencies
uv add langchain langchain-ollama langchain-community langchain-text-splitters pypdf faiss-cpu
```

---

## 📂 Document Loading & Processing

The project supports asynchronous document loading, memory-efficient splitting, and async batch embedding.

- **Current Document**: `docs/constitution.pdf`
- **Logic**: 
  - Checks if a persisted FAISS index folder (`faiss_index`) exists.
  - If it exists, it loads the vector store directly from disk using `FAISS.load_local`.
  - If not, it uses `PyPDFLoader` with `alazy_load` to stream pages.
  - Uses `RecursiveCharacterTextSplitter` to lazily yield 1000-character chunks with 200-character overlap.
  - Batches document chunks iteratively via `OllamaEmbeddings` to generate vectors using `nomic-embed-text` without overloading memory.
  - Builds a **FAISS** vector store iteratively from generated embeddings and saves it to disk for persistence.
  - Passes the vector store to `start_conversation` to answer questions via `generate_answer`.

---

## 🏃 Running the Project

To start the chatbot and ask questions:

```bash
uv run python main.py
```

### Example Session Output
```text
Attempting to load index from disk...
Index 'faiss_index' loaded successfully.
Verified loaded store size: 114 documents

==================================================
ETHIOPIAN CONSTITUTION CHATBOT
Type your questions below. Type 'exit' or 'quit' to stop.
==================================================

You: What is the supreme law of the land?
Thinking...

AI: The Constitution is the supreme law of the land, as stated in Article 9 (1) of the Constitution. Any law, customary practice or a decision of an organ of state or a public official which contravenes this Constitution shall be of no effect.

------------------------------
You: exit

Exiting conversation. Goodbye
```

---

## 📈 Roadmap
- [x] Project Initialization
- [x] Basic LLM Connection
- [x] Asynchronous Document Loading
- [x] Memory-Efficient Document Splitting
- [x] Asynchronous Batch Embeddings
- [x] FAISS Vector Store Integration
- [x] FAISS Index Persistence to Disk
- [x] FAISS Index Loading from Disk
- [x] RAG LCEL Question Answering Chain
- [x] Interactive Terminal Conversation Loop
