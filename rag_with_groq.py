
import os
import chromadb
from chromadb.config import Settings

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# === CONFIG ===
CHROMA_DIR = "chroma_db_store"
COLLECTION_NAME = "langchain"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

# === Embedding Model ===
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Chroma Vector DB ===
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
vectordb = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
    persist_directory=CHROMA_DIR
)

# === Retriever (MMR with k=5) ===
retriever = vectordb.as_retriever(
    search_type="mmr",  # ✅ Improve precision and reduce repetition
    search_kwargs={"k": 5}
)

# === Custom Prompt ===
custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant with deep knowledge about Changi Airport and Jewel Singapore.

Use ONLY the context below to answer the user's question. Do not make up answers.

Context:
{context}

Question: {question}

Answer:""")

# === LLM (Groq) ===
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME
)

# === QA Chain with Prompt ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# === Chat CLI ===
print("🤖 Ask anything about Changi Airport (type 'exit' to quit)")
while True:
    query = input("\n🧠 Your question: ")
    if query.lower() in ("exit", "quit"):
        break

    result = qa_chain({"query": query})
    
    print("\n📝 Answer:")
    print(result["result"].strip())

    print("\n📄 Source Snippets:")
    for doc in result["source_documents"]:
        print(f"\n🔗 {doc.metadata.get('source', 'Unknown')}")
        print("→", doc.page_content[:300].replace("\n", " "), "...\n")
