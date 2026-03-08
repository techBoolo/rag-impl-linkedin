# RAG Project: Document Splitting

This project is a Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It is designed to load, process, and query PDF documents using local LLMs.

---

## 🚀 Features

- **Local LLM**: Powered by Ollama (`llama3.1`).
- **Async Processing**: High-performance asynchronous document loading using `alazy_load`.
- **Memory Efficiency**: Document splitting using `RecursiveCharacterTextSplitter` with async generators.
- **Modern Tooling**: Managed by `uv` for lightning-fast dependency management and environment isolation.

---

## 🛠️ Setup

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- [uv](https://github.com/astral-sh/uv) installed.

### 2. Pull the Model
Ensure the `llama3.1` model is available locally:
```bash
ollama pull llama3.1
```

### 3. Initialize & Install Dependencies
```bash
# Initialize project environment
uv venv
source .venv/bin/activate

# Install core and RAG-specific dependencies
uv add langchain langchain-ollama langchain-community langchain-text-splitters pypdf
```

---

## 📂 Document Loading & Splitting

The project supports asynchronous document loading and memory-efficient splitting.

- **Current Document**: `docs/constitution.pdf`
- **Logic**: 
  - Uses `PyPDFLoader` with `alazy_load` to stream pages.
  - Uses `RecursiveCharacterTextSplitter` to lazily yield 1000-character chunks with 200-character overlap.

---

## 🏃 Running the Project

To verify the document loading, splitting, and environment setup, run:

```bash
uv run python main.py
```

### Expected Output
```text
Loading document from: .../rag-project/docs/constitution.pdf
Splitting document into chunks...

Example of first chunk:
----------------------------------------
... (Preamble text) ...
----------------------------------------

Successfully generated 114 chunks!
```

---

## 📈 Roadmap
- [x] Project Initialization
- [x] Basic LLM Connection
- [x] Asynchronous Document Loading
- [x] Memory-Efficient Document Splitting
