import os, re, faiss
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOP_K = 5

def load_txt(path):
    return open(path, "r", encoding="utf-8").read()

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk(text):
    return [text[i:i+400] for i in range(0, len(text), 400)]

def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

class Store:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embeds, chunks):
        self.index.add(np.array(embeds).astype("float32"))
        self.data.extend(chunks)

    def search(self, q, k):
        D, I = self.index.search(np.array([q]).astype("float32"), k)
        return [self.data[i] for i in I[0]]

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

def run_pipeline(file_path, query):
    text = clean(load_txt(file_path))
    chunks = chunk(text)

    embeds = embed(chunks)

    store = Store(len(embeds[0]))
    store.add(embeds, chunks)

    q_vec = embed([query])[0]
    results = store.search(q_vec, TOP_K)

    context = "\n\n".join(results)

    return generate(query, context)
