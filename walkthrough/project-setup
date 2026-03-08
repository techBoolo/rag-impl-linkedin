# RAG Project Setup

This project is a setup for a Retrieval-Augmented Generation (RAG) system using LangChain and a local Ollama LLM.

## Setup Steps

### 1. Project Initialization
The project was initialized using `uv`.
```bash
mkdir rag-project && cd rag-project && uv init && uv venv
```

### 2. Dependency Installation
Required libraries for LangChain and Ollama integration:
```bash
uv add langchain langchain-ollama
```

### 3. Ollama Model
This project uses the `llama3.1` model. To ensure it is available locally:
```bash
ollama list
```
If you need to pull the model:
```bash
ollama pull llama3.1
```

### 4. Verification Script
A test script `main.py` is used to verify the connection to Ollama.

**Running the test:**
```bash
uv run python main.py
```

**What it does:**
It initializes `ChatOllama` with `llama3.1` and asks: *"What is the official name of Ethiopia?"*

## Expected Output
If correctly set up, you should see:
```text
Prompt: What is the official name of Ethiopia?

Response:
The official name of Ethiopia is the Federal Democratic Republic of Ethiopia (FDRE).
```
