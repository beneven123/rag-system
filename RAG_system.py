
import os
import glob
from dotenv import load_dotenv
import gradio as gr

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback

#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.schema import HumanMessage, AIMessage
# import numpy as np
# import plotly.graph_objects as go
# from langchain.schema import Document

# Load environment variables in a file called .env
load_dotenv(override=True)

# Define flag to control vectorstore regeneration
RECREATE_VECTORSTORE = False  # Set to True to force delete and rebuild

#Define Model and vector DB name
MODEL = "gpt-4o-mini"
# MODEL = "gpt-4o"
db_name = "vector_db"

# Load API key from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def add_metadata(doc, doc_type):
    doc.metadata["doc_type"] = doc_type
    return doc

def create_chunks(folders):
    text_loader_kwargs = {'encoding': 'utf-8'}
    # If above doesn't work, some Windows users might need to uncomment the next line instead
    # text_loader_kwargs={'autodetect_encoding': True}

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)

        md_loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        txt_loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        docx_loader = DirectoryLoader(folder, glob="**/*.docx", loader_cls=UnstructuredFileLoader)
        pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=UnstructuredFileLoader)

        folder_docs = md_loader.load() + txt_loader.load() + docx_loader.load() + pdf_loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Total number of documents: {len(documents)}")
    print(f"Total number of chunks: {len(chunks)}")
    print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")

    return chunks


#Create vectorstore, use OpenAI embedding
embeddings = OpenAIEmbeddings()


# Delete existing DB directory if recreating
if RECREATE_VECTORSTORE and os.path.exists(db_name):
    print(f"Deleting existing vectorstore at '{db_name}'...")
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Build or load the vectorstore
if RECREATE_VECTORSTORE or not os.path.exists(db_name):
    folders = glob.glob("knowledge-base/*") # This is the parent folder in your py file location where all documents reside. Vector store will be based out of this.
    print(f"folders : {folders}")

    print("Creating new vectorstore...")
    chunks = create_chunks(folders)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name
    )
else:
    print("Loading existing vectorstore...")
    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function=embeddings
    )

# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

#Create a Langchain conversation chain

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)


# llama3.2
# llm = ChatOpenAI(temperature=0.7, model_name='llama3.2', base_url='http://localhost:11434/v1', api_key='ollama')

# gemma3
# llm = ChatOpenAI(temperature=0.7, model_name='gemma3', base_url='http://localhost:11434/v1', api_key='ollama')

# deepseek-r1:1.5b
# llm = ChatOpenAI(temperature=0.7, model_name='deepseek_1b', base_url='http://localhost:11434/v1', api_key='ollama')

# gemini 2.5 flash
# llm = ChatOpenAI(temperature=0.7, model_name='gemini-2.5-flash', base_url='https://generativelanguage.googleapis.com/v1beta/openai', api_key=GEMINI_API_KEY)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True,output_key='answer')

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# putting it together: set up the conversation chain with the LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,return_source_documents=True)

# And in Gradio:

def chat(question, history):
    # converted_history = []

    # Convert Gradio message format to LangChain messages
    # for msg in history:
    #     if msg["role"] == "user":
    #         converted_history.append(HumanMessage(content=msg["content"]))
    #     elif msg["role"]== "assistant":
    #         converted_history.append(AIMessage(content=msg["content"]))

    # Inject into conversation memory
    #conversation_chain.memory.chat_memory.messages = converted_history

    # Track token usage during the model invocation
    with get_openai_callback() as cb:
        result = conversation_chain.invoke({"question": question})

    # Extract the LLM's response    
    answer = result["answer"]

    # Optional: Append source documents (if available)
    if "source_documents" in result:
        sources = "\n\n".join([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]])
        answer += f"\n\n**Sources:**\n{sources}"

    # Optional: Append token usage details
    answer += (
        f"\n\n**Token Usage:**\n"
        f"- Prompt (input + history): {cb.prompt_tokens} tokens\n"
        f"- Response: {cb.completion_tokens} tokens\n"
        f"- Total: {cb.total_tokens} tokens"
    )

    return answer


view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)