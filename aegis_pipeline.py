# ============================
# AEGIS FINAL PIPELINE (UPGRADED FAISS)
# ============================

import os
import re
import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# CONFIG
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOP_K = 5
CANDIDATES = 20

# LOAD
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# CLEAN
def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# STRUCTURE-AWARE CHUNKING (PANDAS)
def chunk(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    data = []
    current_section = "General"
    buffer = []

    for line in lines:
        if line.isupper() or (len(line.split()) < 6 and line.istitle()):
            if buffer:
                data.append({
                    "section": current_section,
                    "content": " ".join(buffer)
                })
                buffer = []
            current_section = line
        else:
            buffer.append(line)

    if buffer:
        data.append({
            "section": current_section,
            "content": " ".join(buffer)
        })

    df = pd.DataFrame(data)
    df["chunk"] = df["section"] + "\n" + df["content"]

    return df["chunk"].tolist()

# EMBEDDING
def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

# FAISS STORE (COSINE SIMILARITY)
class Store:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.data = []

    def add(self, embeds, chunks):
        arr = np.array(embeds).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(arr)

        self.index.add(arr)

        for c in chunks:
            self.data.append({
                "text": c,
                "length": len(c)
            })

    def search(self, q_vec, k=CANDIDATES):
        q = np.array([q_vec]).astype("float32")

        faiss.normalize_L2(q)

        D, I = self.index.search(q, k)

        results = []
        for idx in I[0]:
            if idx < len(self.data):
                results.append(self.data[idx])

        return results

# RE-RANKING
def rerank(query, results, top_k=TOP_K):
    scored = []

    for r in results:
        text = r["text"]
        score = sum(word in text.lower() for word in query.lower().split())
        scored.append((score, text))

    scored.sort(reverse=True)

    return [text for _, text in scored[:top_k]]

# GENERATION
def generate(query, context):
    prompt = f"""
Answer ONLY from context.
If not found say NOT FOUND.

Context:
{context}

Question:
{query}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# MATPLOTLIB GRAPH
def plot_chunk_lengths(chunks):
    lengths = [len(c) for c in chunks]

    plt.figure()
    plt.hist(lengths)
    plt.title("Chunk Length Distribution")
    plt.xlabel("Chunk Length")
    plt.ylabel("Frequency")
    plt.tight_layout()

    return plt

# MAIN PIPELINE
def run_pipeline(file_path, query):
    text = clean(load_txt(file_path))
    chunks = chunk(text)

    if not chunks:
        return "No readable content found", None

    embeds = embed(chunks)

    store = Store(len(embeds[0]))
    store.add(embeds, chunks)

    # QUERY
    q_vec = embed([query])[0]

    # STEP 1: Retrieve candidates
    candidates = store.search(q_vec, k=CANDIDATES)

    # STEP 2: Keyword filter (boost)
    if "leave" in query.lower():
        candidates = [c for c in candidates if "leave" in c["text"].lower()]

    # STEP 3: Re-rank
    results = rerank(query, candidates, top_k=TOP_K)

    # Fallback
    if not results:
        results = chunks[:TOP_K]

    context = "\n\n".join(results)

    answer = generate(query, context)

    # Graph
    fig = plot_chunk_lengths(chunks)

    return answer, fig
