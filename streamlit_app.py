import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rag_pipeline import retrieve_candidates # Correct import, no indent
st.set_page_config(page_title="Aegis RAG Chat", layout="wide")
# Cache model load for performance
@st.cache_resource
def load_model():
tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-
v2")
model = AutoModelForSequenceClassification.from_pretrained("crossencoder/ms-marco-MiniLM-L-6-v2")
return tok, model
1.
6
tokenizer, model = load_model()
# Rerank function using cross-encoder
def rerank(query, candidates):
scores = []
for passage in candidates[:25]: # limit for speed
inputs = tokenizer(query, passage, return_tensors="pt",
truncation=True)
with torch.no_grad():
score = model(**inputs).logits.squeeze().item()
scores.append((passage, score))
return sorted(scores, key=lambda x: x[1], reverse=True)
st.title("🤖 Aegis RAG Chat")
query = st.text_input("Enter your query:")
if query:
candidates = retrieve_candidates(query) # from rag_pipeline
if not candidates:
st.warning("No candidates found.")
else:
reranked = rerank(query, candidates)
top5 = reranked[:5]
st.subheader("Top Context Chunks:")
for i, (text, score) in enumerate(top5, 1):
st.markdown(f"**Rank {i} (score {score:.3f})**")
st.write(text)
st.divider()
# Combine context and ask LLM
context = "\n\n".join([t for t,_ in top5])
res = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role":"user", "content": f"Context:\n{context}
\n\nQuestion:\n{query}"}]
)
answer = res.choices[0].message.content
st.write(answer)
# Show metrics in sidebar
st.sidebar.metric("Candidates Retrieved", len(candidates))
st.sidebar.metric("Chunks Used", len(top5))
