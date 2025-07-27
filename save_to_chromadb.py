
import json
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

JSONL_FILE = "changi_airport_content.jsonl"
CHROMA_DIR = "chroma_db_store"

documents = []
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        content = "\n".join(item.get("h1", []) + item.get("h2", []) + item.get("paragraphs", []) + item.get("lists", []))
        if content.strip():
            documents.append(Document(
                page_content=content.strip(),
                metadata={"source": item["url"], "title": item["title"]}
            ))

print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Optionally reduce loaded documents
documents = documents[:30]  # only keep top 30

chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR) 



vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)
vectordb.persist()

print(f"Successfully stored in ChromaDB at {CHROMA_DIR}")
