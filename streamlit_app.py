import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import time
from openai import OpenAI

# SAFE IMPORT (won’t crash if pipeline fails)
try:
    from rag_pipeline import retrieve_candidates
except Exception as e:
    def retrieve_candidates(q):
        return [f"Pipeline not loaded: {e}"]

st.set_page_config(page_title="AEGIS RAG", layout="wide")

st.title("🚀 AEGIS Enterprise RAG System")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return tokenizer, model

tokenizer, model = load_model()

def rerank(query, candidates):
    results = []
    for text in candidates[:25]:
        inputs = tokenizer(query, text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            score = model(**inputs).logits.squeeze().item()
        results.append((text, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def generate_answer(query, context):
    prompt = f"""
You are an enterprise policy assistant.
Answer ONLY using the provided context.
Cite sources like [Source 1].

Context:
{context}

Question:
{query}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask your question...")

if query:
    start = time.time()

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            candidates = retrieve_candidates(query)
            reranked = rerank(query, candidates)

            top_chunks = reranked[:5]
            context = "\n\n".join([c for c, _ in top_chunks])

            answer = generate_answer(query, context)

            st.write(answer)

            with st.expander("📌 Sources"):
                for i, (text, score) in enumerate(top_chunks):
                    st.write(f"Source {i+1} | Score: {score:.3f}")
                    st.write(text)
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})

    latency = round(time.time() - start, 2)

    st.sidebar.header("📊 Metrics")
    st.sidebar.metric("Latency", latency)
    st.sidebar.metric("Candidates", len(candidates))
    st.sidebar.metric("Used Chunks", len(top_chunks))
