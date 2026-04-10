import os
from openai import OpenAI
from pinecone import Pinecone
from metadata import extract_metadata
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def load_documents(data_path):
    docs = []
    for root, _, files in os.walk(data_path):
        for file in files:
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in res.data]

def ingest(data_path="data"):
    docs = load_documents(data_path)
    all_chunks = []
    metadatas = []

    for doc in docs:
        chunks = chunk_text(doc)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadatas.append({**extract_metadata(doc), "text": chunk})

    vectors = embed(all_chunks)

    to_upsert = [
        {
            "id": str(i),
            "values": vectors[i],
            "metadata": metadatas[i]
        }
        for i in range(len(vectors))
    ]

    index.upsert(to_upsert)
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest()
