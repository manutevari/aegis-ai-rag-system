import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rag_pipeline import retrieve_candidates
from openai import OpenAI
import os
import time

st.set_page_config(page_title="Aegis RAG Chat", layout="wide")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return tokenizer, model

tokenizer, model = load_model()

def rerank(query, candidates):
    scores = []
    for passage in candidates[:25]:
        inputs = tokenizer(query, passage, return_tensors="pt", truncation=True)
        with torch.no_grad():
            score = model(**inputs).logits.squeeze().item()
        scores.append((passage, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def generate(query, context):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":f"Context:\n{context}\nQ:{query}"}]
    )
    return res.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🚀 AEGIS FINAL RAG SYSTEM")

query = st.chat_input("Ask...")

if query:
    start = time.time()

    st.session_state.messages.append({"role":"user","content":query})

    with st.chat_message("assistant"):
        candidates = retrieve_candidates(query)
        reranked = rerank(query, candidates)

        top = reranked[:5]
        context = "\n\n".join([t for t,_ in top])

        answer = generate(query, context)
        st.write(answer)

        with st.expander("Sources"):
            for i,(t,s) in enumerate(top):
                st.write(f"{i+1}. Score {s:.3f}")
                st.write(t)

    st.session_state.messages.append({"role":"assistant","content":answer})

    st.sidebar.metric("Latency", round(time.time()-start,2))
