# RAG Project: Document Loading

This project is a Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **Python**. It is designed to load, process, and query PDF documents using local LLMs.

---

## 🚀 Features

- **Local LLM**: Powered by Ollama (`llama3.1`).
- **Async Processing**: High-performance asynchronous document loading.
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
uv add langchain langchain-ollama langchain-community pypdf
```

---

## 📂 Document Loading

The project currently supports loading PDF documents asynchronously from the `docs/` directory.

- **Current Document**: `docs/constitution.pdf`
- **Logic**: Uses `PyPDFLoader` with `alazy_load` for memory-efficient iteration.

---

## 🏃 Running the Project

To verify the document loading and environment setup, run:

```bash
uv run python main.py
```

### Expected Output
```text
Loading document from: .../rag-project/docs/constitution.pdf

Successfully loaded 40 pages.
```

---

## 📈 Roadmap
- [x] Project Initialization
- [x] Basic LLM Connection
- [x] Asynchronous Document Loading

