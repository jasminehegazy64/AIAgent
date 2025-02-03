# AIAgent
In this python-based agent, LangChain is used to interact with Ollama "deepseek-r1" model to answer questions based on the content of an input document. The agent reads the file,process its content and respond to queries related to the content.
The agent can also save notes in a txt file as an additional action.

### Requirments 
To run this project, you need to install the following dependencies:

- `langchain`
- `ollama` (for interacting with the Ollama model)
- `pydantic` (for data validation)

You can install these dependencies using `pip`:

### Files
- Run AIAgent.py after installing the dependencies
- Random_file.txt is the file that will be used to answer the questions (it contains some random text that you can use, or you can use your own Help file)
- notes.txt is the file that will be used to save the notes (it will be created if it doesn't exist)





