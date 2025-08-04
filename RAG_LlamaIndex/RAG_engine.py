from dotenv import load_dotenv
#from llama_index.llms.gemini import Gemini
#import google.generativeai as genai
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from IPython.display import Markdown, display
from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate,ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
import os

def rag_query(question):
    load_dotenv() 
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")  
 
    llm = Groq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_api_key)

    persist_directory = "./chroma_db"

    # Check if the directory exists
    if not os.path.exists(persist_directory):
        # If it doesn't exist, create it
        print(f"Creating directory: {persist_directory}")
        os.makedirs(persist_directory)

        documents=SimpleDirectoryReader(input_files=["Power+BI+Ebook.pdf"])
        doc=documents.load_data()   

        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        nodes = node_parser.get_nodes_from_documents(doc,show_progress=False)

        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        db = chromadb.PersistentClient(path=persist_directory)
        chroma_collection = db.get_or_create_collection ("mycollection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(nodes=nodes,embed_model=embed_model,storage_context=storage_context)
        
    else:
        print(f"Directory already exists: {persist_directory}")
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = chromadb.PersistentClient(path=persist_directory)
        chroma_collection = db.get_or_create_collection("mycollection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the VectorStoreIndex from the existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model # Ensure embed_model is passed
        )
        print("Index loaded from ChromaDB.")

    prompt_str= """
            "You are a helpful assistant that answers questions based on the provided context. "
            "If the context doesn't contain the answer, state that you don't know. "
            "Be concise and do not add extraneous information."
            Context information is below.
            ---------------------
            {context_str}
            ---------------------
            {query_str}
            """
    message_template = ChatMessage(content=prompt_str, role=MessageRole.USER)
    query_prompt=ChatPromptTemplate([message_template])

    query_engine = index.as_query_engine(llm=llm,query_template=query_prompt)
    answer = query_engine.query (question).response    
    return answer