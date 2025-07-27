from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings

# --- Config
CHROMA_PATH = "chroma_db_store"
COLLECTION_NAME = "langchain"
MODEL_NAME = "llama3-70b-8192"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Initialize FastAPI
app = FastAPI()

# --- Embedding Model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Load Chroma Vector DB
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
vectordb = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
    persist_directory=CHROMA_PATH
)

# --- Load LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

# --- Retrieval QA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# --- Request Model
class QueryRequest(BaseModel):
    question: str

# --- API Endpoint
@app.post("/ask")
def ask_question(req: QueryRequest):
    response = qa_chain({"query": req.question})
    answer = response["result"]
    sources = [
        doc.metadata.get("source", "unknown") for doc in response["source_documents"]
    ]
    return {
        "answer": answer,
        "sources": sources
    }
