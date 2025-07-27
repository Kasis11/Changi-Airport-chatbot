# import os
# import json
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import chromadb
# from chromadb.config import Settings

# # ✅ Step 1: Load raw text
# with open("output.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# # ✅ Step 2: Split into chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
# chunks = splitter.split_text(raw_text)

# # (Optional) Save chunks
# with open("chunks.json", "w", encoding="utf-8") as f:
#     json.dump(chunks, f, indent=2)

# # ✅ Step 3: Generate embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(chunks, show_progress_bar=True, batch_size=16)

# # ✅ Step 4: Store in ChromaDB
# persist_dir = "chroma_db"
# os.makedirs(persist_dir, exist_ok=True)

# chroma_client = chromadb.PersistentClient(
#     path=persist_dir,
#     settings=Settings(anonymized_telemetry=False)
# )
# collection = chroma_client.get_or_create_collection(name="changi")

# # ✅ Step 5: Batch insert
# def batch_insert(texts, embeddings, batch_size=100):
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         batch_embeds = embeddings[i:i+batch_size].tolist()
#         batch_ids = [f"doc_{i+j}" for j in range(len(batch_texts))]
#         batch_meta = [{"source": "output.txt"} for _ in batch_texts]

#         collection.add(
#             documents=batch_texts,
#             embeddings=batch_embeds,
#             ids=batch_ids,
#             metadatas=batch_meta
#         )
#         print(f"✅ Added batch {i} to {i + len(batch_texts) - 1}")

# batch_insert(chunks, embeddings)

# print(f"\n📊 Stored {collection.count()} documents in ChromaDB.")


# ingest_to_chromadb.py

import json
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

# === CONFIG ===
JSONL_FILE = "changi_airport_content.jsonl"
CHROMA_DIR = "chroma_db_store"

# ✅ Load JSONL
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

print(f"✅ Loaded {len(documents)} documents")

# ✅ Split into Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
print(f"✅ Total chunks: {len(chunks)}")
import json


# ✅ Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Store in ChromaDB
if os.path.exists(CHROMA_DIR):
    import shutil
    shutil.rmtree(CHROMA_DIR)  # clear old db

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)
vectordb.persist()

print(f"✅ Successfully stored in ChromaDB at {CHROMA_DIR}")
