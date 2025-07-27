from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import chromadb
from chromadb.config import Settings

load_dotenv()

# === Configuration ===
CHROMA_PATH = "chroma_db_store"
COLLECTION_NAME = "langchain"
MODEL_NAME = "llama3-70b-8192"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === FastAPI App Initialization ===
app = FastAPI()

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Model ===
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    # === Lazy Load to Save Memory ===
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

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

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2}  # smaller k to reduce memory usage
    )

    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant with deep knowledge about Changi Airport and Jewel Singapore.

    Use ONLY the context below to answer the user's question. Do not make up answers.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    # === Perform QA ===
    result = qa_chain.invoke({"query": query.question})

    # === Extract top source for Read More link ===
    source_url = None
    if result["source_documents"]:
        top_doc = result["source_documents"][0]
        source_url = top_doc.metadata.get("source", "Unknown")

    answer = result["result"]
    if source_url and source_url != "Unknown":
        answer += f"\n\n🔗 [Read more here]({source_url})"

    return {
        "answer": answer,
        "source_url": source_url,
        "sources": [
            {
                "url": doc.metadata.get("source", "Unknown"),
                "excerpt": doc.page_content[:300].replace("\n", " ")
            } for doc in result["source_documents"]
        ]
    }
