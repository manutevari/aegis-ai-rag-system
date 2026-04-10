import os
import re
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("aegis-index")

def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [d.embedding for d in res.data]

def classify_intent(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Classify as HR or Travel:\n{query}"}]
    )
    return res.choices[0].message.content.strip()

def expand(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Generate 3 variations:\n{query}"}]
    )
    return [query] + res.choices[0].message.content.split("\n")

def hyde(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Answer:\n{query}"}]
    )
    return res.choices[0].message.content

def search(query, category):
    vec = embed([query])[0]
    return index.query(
        vector=vec,
        top_k=25,
        include_metadata=True,
        filter={"policy_category": category}
    )["matches"]

def post_filter(results):
    if not results:
        return results
    latest = max(r["metadata"]["effective_date"] for r in results)
    return [r for r in results if r["metadata"]["effective_date"] == latest]

def retrieve_candidates(query):
    category = classify_intent(query)

    queries = expand(query)
    queries.append(hyde(query))

    results = []
    for q in queries:
        results.extend(search(q, category))

    results = post_filter(results)

    return [
        r["metadata"]["text"]
        for r in results
        if r.get("metadata") and r["metadata"].get("text")
    ]
