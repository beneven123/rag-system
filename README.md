This application will create a vector store based on the documents you provide and will be able to answer question from the documents in a chat interface. The application currently
supports .md, .txt, .docx and .pdf files. You need to set below keys in a .env file in the applications root directory. 

**OPEN_API_KEY**
**GEMINI_API_KEY** (if you choose to use google's llm)

By default the application uses OPENAI embeddings and LLM. You have an option to use google llm or locally hosted ollama llms. The code
is already there, just have to comment/uncomment accordingly. If you are planning to use locally hosted ollama llms, make sure ollama is installed and the llms downloaded.

The application requires all your documents inside a parent folder "**knowledge-base**" in the root folder of the application. You can create sub-folders in any names based on the type of 
documents you have. The application currentlysupports .md, .txt, .docx and .pdf files. Files with other extension will be ignored.

Set below parameter to TRUE when you run the application for the first time to create the vector store. For subsequent runs, this can be updated as False to avoid recreating the vector store
unless you have added new documents to your knowledge-base folder and vector store need to be recreated.

`RECREATE_VECTORSTORE = True`
