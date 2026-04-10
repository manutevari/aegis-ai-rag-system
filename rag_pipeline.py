import re
import os
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
# TABLE HANDLING (FIXED ✅)
# -------------------------------
def process_table(table_lines):
    headers = table_lines[0]
    rows = table_lines[2:]

    chunks = []
    for row in rows:
        chunks.append(headers + "\n" + row)
    return chunks

# -------------------------------
# SEMANTIC CHUNKING (FULL AEGIS)
# -------------------------------
def chunk(text, overlap=80):
    sections = re.split(r"(#{1,3} .+)", text)

    chunks = []
    current_h1 = ""
    current_h2 = ""

    for i in range(1, len(sections), 2):
        header = sections[i]
        body = sections[i+1]

        if header.startswith("# "):
            current_h1 = header
        elif header.startswith("## "):
            current_h2 = header

        # table detection
        lines = body.split("\n")
        table_buffer = []

        for line in lines:
            if "|" in line:
                table_buffer.append(line)
            else:
                if table_buffer:
                    chunks.extend(process_table(table_buffer))
                    table_buffer = []

                chunks.append(f"{current_h1}\n{current_h2}\n{line}")

    # overlap
    final_chunks = []
    for i in range(len(chunks)):
        start = max(0, i-1)
        combined = " ".join(chunks[start:i+1])
        final_chunks.append(combined[-500:])

    return final_chunks

# -------------------------------
# METADATA EXTRACTION (FULL)
# -------------------------------
def extract_metadata(text):
    return {
        "document_id": "DOC-001",
        "policy_category": "Travel" if "travel" in text.lower() else "HR",
        "policy_owner": "GCT-RM",
        "effective_date": "2026-02-01",
        "h1_header": text.split("\n")[0] if "\n" in text else "",
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
# MULTI QUERY
# -------------------------------
def expand_query(query):
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
# PRE-FILTER
# -------------------------------
def pre_filter(query):
    if "leave" in query.lower():
        return "HR"
    return "Travel"

# -------------------------------
# POST FILTER (GLOBAL FIX ✅)
# -------------------------------
def post_filter(results):
    if not results:
        return results

    latest_date = max(r["metadata"]["effective_date"] for r in results)

    return [
        r for r in results
        if r["metadata"]["effective_date"] == latest_date
    ]

# -------------------------------
# MOCK VECTOR SEARCH (Replace with Pinecone)
# -------------------------------
def vector_search(query):
    # replace with Pinecone
    return []

# -------------------------------
# BROAD RETRIEVAL
# -------------------------------
def broad_retrieval(query):
    queries = expand_query(query)
    queries.append(hyde(query))

    results = []
    for q in queries:
        results.extend(vector_search(q))

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
