# ============================
# AEGIS FINAL COMPLETE SYSTEM (WITH PYDANTIC + MATPLOTLIB)
# ============================

import os
import re
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "aegis-index"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ----------------------------
# LOAD + CLEAN
# ----------------------------
def load_txt(path):
    return open(path, "r", encoding="utf-8").read()

def clean(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------
# PYDANTIC MODEL
# ----------------------------
class DocumentMetadata(BaseModel):
    document_id: str
    category: str
    effective_date: str

# ----------------------------
# METADATA EXTRACTION
# ----------------------------
def extract_metadata(text):
    data = {
        "document_id": "DOC-001",
        "category": "Travel" if "travel" in text.lower() else "General",
        "effective_date": "2026-02-01"
    }
    return DocumentMetadata(**data).dict()

# ----------------------------
# CHUNKING
# ----------------------------
def chunk(text, max_size=400, overlap=60):
    chunks = []
    sections = re.split(r"(#{1,3} .+)", text)

    current_header = "General"
    buffer = ""

    for part in sections:
        if re.match(r"#{1,3} ", part):
            if buffer.strip():
                chunks.append({"text": f"{current_header}\n{buffer}", "section": current_header})
                buffer = ""
            current_header = part.strip()
        else:
            for line in part.split("\n"):
                if "|" in line:
                    chunks.append({"text": f"{current_header}\n{line}", "section": current_header})
                else:
                    buffer += line + " "

                if len(buffer) > max_size:
                    chunks.append({"text": f"{current_header}\n{buffer}", "section": current_header})
                    buffer = buffer[-overlap:]

    if buffer.strip():
        chunks.append({"text": f"{current_header}\n{buffer}", "section": current_header})

    if not chunks:
        chunks = [{"text": text[:500], "section": "fallback"}]

    return chunks

# ----------------------------
# MATPLOTLIB ANALYSIS
# ----------------------------
def plot_chunk_lengths(chunks):
    try:
        lengths = [len(c["text"]) for c in chunks if c["text"]]

        if not lengths:
            return

        plt.figure()
        plt.hist(lengths)
        plt.title("Chunk Length Distribution")
        plt.xlabel("Length")
        plt.ylabel("Frequency")

        plt.savefig("chunk_plot.png")
        plt.close()
    except:
        pass

# ----------------------------
# EMBEDDING
# ----------------------------
def embed(texts):
    if not texts:
        return []

    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [d.embedding for d in res.data]

# ----------------------------
# STORE
# ----------------------------
class PineconeStore:
    def upsert(self, embeds, chunks, metadata):
        vectors = []

        for i, (emb, c) in enumerate(zip(embeds, chunks)):
            if not emb:
                continue

            vectors.append({
                "id": f"{metadata['document_id']}_{i}",
                "values": emb,
                "metadata": {
                    "text": c["text"],
                    "section": c["section"],
                    "category": metadata["category"],
                    "effective_date": metadata["effective_date"]
                }
            })

        if vectors:
            index.upsert(vectors=vectors)

# ----------------------------
# PRE-FILTER
# ----------------------------
def metadata_filter(query):
    q = query.lower()
    if "taxi" in q or "cab" in q:
        return {"category": "Travel"}
    if "leave" in q:
        return {"category": "HR"}
    return None

# ----------------------------
# QUERY EXPANSION
# ----------------------------
def expand_query(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate 3 variations:\n{query}"}]
    )
    return [query] + [v.strip() for v in res.choices[0].message.content.split("\n") if v.strip()]

# ----------------------------
# HYDE
# ----------------------------
def hyde(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Answer:\n{query}"}]
    )
    return res.choices[0].message.content

# ----------------------------
# BROAD RETRIEVAL
# ----------------------------
def broad_retrieval(query):
    filters = metadata_filter(query)
    queries = expand_query(query)

    results = []

    for q in queries:
        emb = embed([q])
        if not emb:
            continue
        q_vec = emb[0]

        res = index.query(vector=q_vec, top_k=25, include_metadata=True, filter=filters)
        results.extend(res["matches"])

    # HYDE
    h_emb = embed([hyde(query)])
    if h_emb:
        h_vec = h_emb[0]
        res = index.query(vector=h_vec, top_k=25, include_metadata=True, filter=filters)
        results.extend(res["matches"])

    return results

# ----------------------------
# CROSS ENCODER SCORING
# ----------------------------
def score(query, text):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Score 0-1:\n{query}\n{text}"}]
    )
    try:
        return float(res.choices[0].message.content.strip())
    except:
        return 0.0

# ----------------------------
# RERANK
# ----------------------------
def rerank(query, matches):
    scored = [(score(query, m["metadata"]["text"]), m) for m in matches]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:5]]

# ----------------------------
# POST FILTER
# ----------------------------
def post_filter(results):
    latest = {}
    for r in results:
        sec = r["metadata"]["section"]
        date = r["metadata"].get("effective_date", "")
        if sec not in latest or date > latest[sec]["metadata"].get("effective_date", ""):
            latest[sec] = r
    return list(latest.values())

# ----------------------------
# GENERATE
# ----------------------------
def generate(query, context):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ:{query}"}]
    )
    return res.choices[0].message.content

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_pipeline(file_path, query):

    text = clean(load_txt(file_path))
    if not text:
        return "Empty file."

    metadata = extract_metadata(text)
    chunks = chunk(text)

    plot_chunk_lengths(chunks)

    texts = [c["text"] for c in chunks if c["text"].strip()]
    if not texts:
        return "No usable content."

    embeds = embed(texts)

    store = PineconeStore()
    store.upsert(embeds, chunks, metadata)

    matches = broad_retrieval(query)
    reranked = rerank(query, matches)
    final = post_filter(reranked)

    texts = [r["metadata"]["text"] for r in final if r["metadata"]["text"].strip()]
    if not texts:
        return "No relevant content found."

    context = "\n\n".join(texts)

    return generate(query, context)
