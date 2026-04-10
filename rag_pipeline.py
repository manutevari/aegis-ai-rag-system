import os
import re
from openai import OpenAI
from pydantic import BaseModel
from pinecone import Pinecone
from datetime import datetime

# -------------------------------
# INIT
# -------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("aegis-index")

# -------------------------------
# METADATA SCHEMA
# -------------------------------
class Metadata(BaseModel):
    document_id: str
    policy_category: str
    policy_owner: str
    effective_date: str
    h1_header: str
    h2_header: str

# -------------------------------
# CLEAN
# -------------------------------
def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# -------------------------------
# TABLE HANDLING (FULL)
# -------------------------------
def process_table(table_lines):
    headers = table_lines[0]
    rows = table_lines[2:]
    return [headers + "\n" + r for r in rows]

# -------------------------------
# SEMANTIC CHUNKING
# -------------------------------
def chunk(text):
    sections = re.split(r"(#{1,3} .+)", text)

    chunks = []
    h1, h2 = "", ""

    for i in range(1, len(sections), 2):
        header = sections[i]
        body = sections[i+1]

        if header.startswith("# "):
            h1 = header
        elif header.startswith("## "):
            h2 = header

        lines = body.split("\n")
        table_buf = []

        for line in lines:
            if "|" in line:
                table_buf.append(line)
            else:
                if table_buf:
                    chunks.extend(process_table(table_buf))
                    table_buf = []
                chunks.append(f"{h1}\n{h2}\n{line}")

    return chunks

# -------------------------------
# METADATA EXTRACTION
# -------------------------------
def extract_metadata(text):
    return {
        "document_id": "TRV-POL-2005-V3",
        "policy_category": "Travel" if "travel" in text.lower() else "HR",
        "policy_owner": "GCT-RM",
        "effective_date": "2026-02-01",
        "h1_header": text.split("\n")[0],
        "h2_header": text.split("\n")[1] if "\n" in text else ""
    }

# -------------------------------
# EMBEDDING
# -------------------------------
def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [d.embedding for d in res.data]

# -------------------------------
# UPSERT TO PINECONE
# -------------------------------
def upsert(chunks):
    vectors = []
    for i, c in enumerate(chunks):
        meta = extract_metadata(c)
        vec = embed([c])[0]
        vectors.append((str(i), vec, {"text": c, **meta}))

    index.upsert(vectors)

# -------------------------------
# LLM ROUTER (PRE-FILTER)
# -------------------------------
def classify_intent(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Classify into HR or Travel:\n{query}"
        }]
    )
    return res.choices[0].message.content.strip()

# -------------------------------
# MULTI-QUERY
# -------------------------------
def expand(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Generate 3 variations:\n{query}"}]
    )
    return [query] + res.choices[0].message.content.split("\n")

# -------------------------------
# HYDE
# -------------------------------
def hyde(query):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Answer:\n{query}"}]
    )
    return res.choices[0].message.content

# -------------------------------
# VECTOR SEARCH WITH FILTER
# -------------------------------
def search(query, category):
    q_vec = embed([query])[0]

    return index.query(
        vector=q_vec,
        top_k=25,
        include_metadata=True,
        filter={"policy_category": category}
    )["matches"]

# -------------------------------
# POST FILTER (LATEST)
# -------------------------------
def post_filter(results):
    latest = max(r["metadata"]["effective_date"] for r in results)
    return [r for r in results if r["metadata"]["effective_date"] == latest]

# -------------------------------
# BROAD RETRIEVAL
# -------------------------------
def broad_retrieval(query):
    category = classify_intent(query)

    queries = expand(query)
    queries.append(hyde(query))

    results = []
    for q in queries:
        results.extend(search(q, category))

    return post_filter(results)

# -------------------------------
# STREAMLIT ADAPTER
# -------------------------------
def retrieve_candidates(query):
    results = broad_retrieval(query)

    return [
        r["metadata"]["text"]
        for r in results
        if r.get("metadata") and r["metadata"].get("text")
    ]
