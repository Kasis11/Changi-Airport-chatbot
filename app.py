import streamlit as st
import os
import chromadb
from chromadb.config import Settings

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_PATH = "chroma_db_store"
COLLECTION_NAME = "langchain"
MODEL_NAME = "llama3-70b-8192"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
    search_kwargs={"k": 5}
)


prompt_template = PromptTemplate.from_template("""
You are a helpful assistant with deep knowledge about Changi Airport and Jewel Singapore.

Use ONLY the context below to answer the user's question. Do not make up any answers.

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

st.set_page_config(page_title="Changi Airport Chatbot", page_icon="🛫")
st.title("Changi Airport AI Chatbot")
st.markdown("Ask anything based on Changi Airport and Jewel Changi website content.")

query = st.text_input("Ask your question:", placeholder="e.g. What are the facilities in Terminal 3?")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": query})
        # Extract the top source URL
        top_doc = response["source_documents"][0] if response["source_documents"] else None
        source_url = top_doc.metadata.get("source") if top_doc else None
        
        # Append the URL to the answer
        answer = response["result"]
        if source_url:
            answer += f"\n\n🔗 [Read more here]({source_url})"
        
        st.success(answer)

        with st.expander("Source Documents (Read More)"):
            for i, doc in enumerate(response["source_documents"]):
                source_url = doc.metadata.get("source", "Unknown")
                snippet = doc.page_content[:300].replace("\n", " ")
                st.markdown(f"""
                **🔗 Source {i+1}**
                [{source_url}]({source_url})  
                _Excerpt:_  
                > {snippet} ...
                """)
