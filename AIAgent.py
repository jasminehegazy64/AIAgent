import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings

def load_file_and_create_vectorstore(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return None

    file_loader = TextLoader(file_path)
    documents = file_loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="deepseek-r1")
    vectorstore = Chroma.from_documents(texts, embeddings)

    print(f"File '{file_path}' loaded and vector store created.")
    return vectorstore

def save_note(note_content, filename="notes.txt"):
    try:
        with open(filename, "a", encoding="utf-8") as file:
            file.write(f"{note_content}\n")
        return "Note saved successfully."
    except Exception as e:
        return f"Error saving note: {e}"

def query_agent(vectorstore, query):
    if not vectorstore:
        return "Vector store is not initialized."

    llm = Ollama(model="deepseek-r1")

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

    result = qa_chain({"query": query})
    return result["result"]

def main():
    print("Welcome to the LangChain Help File Agent!")
    
    file_path = input("Enter the file path: ").strip()
    vectorstore = load_file_and_create_vectorstore(file_path)
    if not vectorstore:
        return  # Exit if the file doesn't exist

   
    while True:
        user_input = input("\nWhat would you like to do? (Ask a question, save a note, or type 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

      
        if user_input.lower().startswith("save note:"):
            note_content = user_input.split("save note:", 1)[1].strip()
            response = save_note(note_content)
            print(f"Agent: {response}")
            continue

        
        try:
            response = query_agent(vectorstore, user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the program
if __name__ == "__main__":
    main()