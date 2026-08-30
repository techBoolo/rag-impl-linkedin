import asyncio
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

async def load_document(file_path):
    """Load the document"""
    loader = PyPDFLoader(file_path)
    return loader.alazy_load()

async def split_document(doc_iterator, chunk_size=1000, chunk_overlap=200):
    """Uses lazy splitting to avoid pulling the whole PDF into memory at once."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    async for doc in doc_iterator:
        # split the document into chunks and yield the chunk
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            yield chunk

def get_embeddings_model():
    """Return the Ollama embeddings model."""
    return OllamaEmbeddings(model="nomic-embed-text")

def get_chat_model(model="llama3.1"):
    """Return the Ollama chat model."""
    return ChatOllama(model=model)

async def create_faiss_index(batch, vectors, vector_store=None):
    """Adds pre-computed embeddings to the FAISS index."""
    embeddings = get_embeddings_model()
    text_embedding_pairs = list(zip([chunk.page_content for chunk in batch], vectors))
    metadatas = [chunk.metadata for chunk in batch]
    if vector_store is None:
        # Initial creation
        vector_store = await FAISS.afrom_embeddings(
            text_embedding_pairs, 
            embeddings, 
            metadatas
        )
    else:
        # Append to existing index
        vector_store.add_embeddings(
            text_embeddings=text_embedding_pairs, 
            metadatas=metadatas
        )

    return vector_store

async def process_embeddings(chunk_generator, batch_size=10):
    """Batches chunks and calls aembed_documents asynchronously."""
    embeddings_model = get_embeddings_model()
    batch = []
    
    async for chunk in chunk_generator:
        batch.append(chunk)
        if len(batch) >= batch_size:
            texts = [c.page_content for c in batch]
            vectors = await embeddings_model.aembed_documents(texts)
            yield batch, vectors
            batch = []
            
    # Process remaining chunks
    if batch:
        texts = [c.page_content for c in batch]
        vectors = await embeddings_model.aembed_documents(texts)
        yield batch, vectors

async def create_faiss_index_from_file(file_path, index_name="faiss_index"):
    """Orchestrates the lazy loading, splitting, and indexing process."""
    if os.path.exists(index_name):
        print("Index already exists. Please remove the index folder to reindex.")
        return
  
    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    doc_iterator = await load_document(file_path)
    chunk_generator = split_document(doc_iterator)
    vector_store = None
    
    print(f"Loading document from: {file_path}")
    print("Splitting document and generating embeddings in batches...")
    
    batch_count = 0
    total_chunks_processed = 0

    async for batch, vectors in process_embeddings(chunk_generator, batch_size=10):
        batch_count += 1
        total_chunks_processed += len(batch)
        vector_store = await create_faiss_index(batch, vectors, vector_store)
        print(f"Processed batch {batch_count} ({len(batch)} chunks) -> Added to FAISS vector store. Total chunks embedded: {total_chunks_processed}")

    # Save the index to the local disk
    if vector_store:
        vector_store.save_local(index_name)
        print("\nSuccessfully created FAISS index with {} chunks across {} batches!".format(total_chunks_processed, batch_count))
        print("Index successfully saved to disk.")
    else:
        print("Error: No data was indexed.")

async def load_index(index_name="faiss_index"):
    """
    Loads the FAISS index from disk.
    Returns the vector_store object ready for similarity search.
    """
    if not os.path.exists(index_name):
        print(f"Error: Index folder '{index_name}' not found.")
        return None

    # we must use the same embedding model used during creation
    embeddings = get_embeddings_model()

    # allow_dangerous_deserialization=True is required to load the metadata (index.pkl)
    vector_store = FAISS.load_local(
        index_name, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print(f"Index '{index_name}' loaded successfully.")
    return vector_store

async def generate_answer(vector_store, query):
    """
    Takes a query, finds relevant context in the FAISS index, 
    and generates an answer using an Ollama LLM.
    """
    # 1. Setup the LLM (Using Ollama)
    llm = get_chat_model()

    # 2. Define the Prompt Template
    template = """
    You are an assistant for question-answering tasks based specifically on the Ethiopian Constitution.
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # 3. Retrieve documents (using our search logic)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 4. Helper function to format the documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Create the RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Execute the chain
    response = await rag_chain.ainvoke(query)
    return response

async def start_conversation(vector_store):
    """
    Handles the interactive loop between the user and the AI.
    """
    if not vector_store:
        print("Error: Vector store not found. Cannot start conversation.")
        return

    print("\n" + "="*50)
    print("ETHIOPIAN CONSTITUTION CHATBOT")
    print("Type your questions below. Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in ['exit', 'quit', 'bye']:
            print("\nExiting conversation. Goodbye")
            break
        if not query:
            continue
        print("Thinking...")

        try:
            answer = await generate_answer(vector_store, query)
            print(f"\nAI: {answer}")
            print("-" * 30)
        except Exception as e:
            print(f"An error occurred while generating the answer: {e}")

async def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    doc_path = os.path.join(project_dir, 'docs', 'constitution.pdf')
    index_name = "faiss_index"

    # 1. Create index if it's missing
    if not os.path.exists(index_name):
        await create_faiss_index_from_file(doc_path, index_name)
    
    # 2. Load the index from local disk
    print("\nAttempting to load index from disk...")
    vector_store = await load_index(index_name)
    
    if vector_store:
        print(f"Verified loaded store size: {vector_store.index.ntotal} documents")
        # 3. Start conversation loop
        await start_conversation(vector_store)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
