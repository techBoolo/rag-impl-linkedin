# Conversation Loop: Answer Generation

We have implemented the final RAG generation step by integrating document retrieval with response generation using an Ollama LLM (`llama3.1`), wrapped in an interactive command-line conversation loop.

## Highlights

### 1. `get_chat_model`
Initializes and returns a `ChatOllama` model instance (defaulting to `llama3.1`).

### 2. `generate_answer`
Connects the FAISS vector store and the Ollama chat model via LangChain Expression Language (LCEL):
- **Retriever Setup:** Configures the vector store as a retriever to fetch the top 4 most relevant chunks (`k=4`).
- **Prompt Template:** Instructs the LLM to act as a specialized question-answering assistant for the Ethiopian Constitution, restricting answers strictly to retrieved context within at most three concise sentences.
- **LCEL RAG Chain:** 
  ```python
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )
  ```
- **Async Execution:** Asynchronously invokes `rag_chain.ainvoke(query)` to return the parsed string output.

### 3. `start_conversation`
Provides an interactive command-line interface:
- Continuously accepts user queries via `input("You: ")`.
- Gracefully handles exit commands (`exit`, `quit`, `bye`).
- Displays progress status (`Thinking...`) and prints the generated AI answer.
- Catches runtime exceptions to prevent session crashes.

---

## Verified Execution

Run the script to start the chatbot:

```bash
uv run python main.py
```

### Example Session

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

AI: According to the Ethiopian Constitution, the Constitution itself is the supreme law of the land. Any law, customary practice, or decision of an organ of state or a public official that contravenes it shall be of no effect. All citizens, organs of state, political organizations, and other associations are bound to obey the Constitution and ensure its observance.
------------------------------
You: exit

Exiting conversation. Goodbye
```
