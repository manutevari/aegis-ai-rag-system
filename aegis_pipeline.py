# ============================
# AEGIS FINAL PIPELINE (WORKING)
# ============================
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import faiss
import numpy as np
from openai import OpenAI

# ---------- CONFIG ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TOP_K = 5

# ---------- LOAD ----------
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------- CLEAN ----------
def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# ---------- CHUNK (IMPROVED) ----------
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

    return df, df["chunk"].tolist()
------------MATPLOLIB FUNCTION-----------
def plot_chunk_lengths(chunks):
    lengths = [len(c) for c in chunks]

    plt.figure()
    plt.hist(lengths)
    plt.title("Chunk Length Distribution")
    plt.xlabel("Chunk Length")
    plt.ylabel("Frequency")
    plt.tight_layout()

    return plt
# ---------- EMBEDDING ----------
def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

# ---------- VECTOR STORE ----------
class Store:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embeds, chunks):
        self.index.add(np.array(embeds).astype("float32"))
        self.data.extend(chunks)

    def search(self, q, k):
        D, I = self.index.search(np.array([q]).astype("float32"), k)
        return [self.data[i] for i in I[0] if i < len(self.data)]

# ---------- GENERATION ----------
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

# ---------- MAIN PIPELINE ----------
def run_pipeline(file_path, query):
    # Load + clean
    text = clean(load_txt(file_path))

    # Chunk
    chunks = chunk(text)

    if not chunks:
        return "No readable content found in file."

    # Embed
    embeds = embed(chunks)

    # Store
    store = Store(len(embeds[0]))
    store.add(embeds, chunks)

    # Query embedding
    q_vec = embed([query])[0]

    # Search
    results = store.search(q_vec, TOP_K)

    print("Retrieved:", results)  # Debug

    # 🔥 Keyword boost (important)
    if "leave" in query.lower():
        keyword_hits = [c for c in chunks if "leave" in c.lower()]
        if keyword_hits:
            results = keyword_hits[:TOP_K]

    # Fallback
    if not results:
        results = chunks[:TOP_K]

    # Context
    context = "\n\n".join(results)

    # Generate answer
    return generate(query, context)
