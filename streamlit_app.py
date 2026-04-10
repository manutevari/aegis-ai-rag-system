import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from rag_pipeline import retrieve_candidates  # 👈 connect backend

# Load model once
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates):
    scores = []
    for passage in candidates:
        inputs = tokenizer(query, passage, return_tensors="pt", truncation=True)
        with torch.no_grad():
            score = model(**inputs).logits.squeeze().item()
        scores.append((passage, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

# UI
st.title("🚀 Aegis RAG System with Reranking")

query = st.text_input("Enter your query:")

if query:
    candidates = retrieve_candidates(query)  # ✅ real data
    reranked = rerank(query, candidates)

    st.write("### Top Results")
    for text, score in reranked[:3]:
        st.write(f"Score: {score:.4f}")
        st.write(text)
        st.write("---")
