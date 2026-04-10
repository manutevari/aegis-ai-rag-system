import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from rag_pipeline import retrieve_candidates

st.set_page_config(page_title="Aegis RAG", layout="wide")

st.title("🚀 Aegis RAG System with Cross-Encoder Reranking")

# ✅ Cache model (VERY IMPORTANT for deployment)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return tokenizer, model

tokenizer, model = load_model()

def rerank(query, candidates):
    scores = []

    for passage in candidates[:25]:  # 🔥 limit for speed
        inputs = tokenizer(query, passage, return_tensors="pt", truncation=True)

        with torch.no_grad():
            score = model(**inputs).logits.squeeze().item()

        scores.append((passage, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# ---------------- UI ----------------
query = st.text_input("🔍 Enter your query:")

if query:
    with st.spinner("Retrieving candidates..."):
        candidates = retrieve_candidates(query)

    if not candidates:
        st.warning("⚠️ No relevant documents found.")
    else:
        with st.spinner("Reranking results..."):
            reranked = rerank(query, candidates)

        st.subheader("📌 Top Results")

        for i, (text, score) in enumerate(reranked[:5]):
            st.markdown(f"**Rank {i+1} | Score: {score:.4f}**")
            st.write(text)
            st.divider()
