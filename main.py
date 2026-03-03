from langchain_ollama import ChatOllama

def main():
    # Initialize ChatOllama with the llama3.1 model
    llm = ChatOllama(model="llama3.1")

    # Prompt specification
    prompt = "What is the official name of Ethiopia?"

    print(f"Prompt: {prompt}\n")

    # Invoke the model
    response = llm.invoke(prompt)

    # Print the content of the response
    print("Response:")
    print(response.content)

if __name__ == "__main__":
    main()
