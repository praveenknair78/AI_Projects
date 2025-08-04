from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os

def rag_query(question, chat_history=None):


    load_dotenv() 
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2, google_api_key=gemini_api_key)

    persist_directory = "./chroma_db"

    # Check if the directory exists
    if not os.path.exists(persist_directory):
        # If it doesn't exist, create it
        print(f"Creating directory: {persist_directory}")
        os.makedirs(persist_directory)

        pdf_reader = PyPDFLoader("Attention_is_all_you_need.pdf")
        documents = pdf_reader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,)
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        db = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory=persist_directory)

    else:
        print(f"Directory already exists: {persist_directory}")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Behave as a AI tutor, and respond the queries from the document trained, Given the following conversation and a follow question, rephrase the follow up question to be a standalone question.
                                                            Chat History:{chat_history}
                                                            Follow up Input: {question}
                                                            Standalone question:""")
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), condense_question_prompt=CONDENSE_QUESTION_PROMPT, return_source_documents=True,
                                            verbose=False)
    
    if chat_history is None:
        chat_history = []

    response = qa({"question": question, "chat_history": chat_history})
    answer = response['answer']
    source_documents = response.get('source_documents', [])
    
    return answer

