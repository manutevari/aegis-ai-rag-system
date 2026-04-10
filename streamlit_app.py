import streamlit as st
import os
import time
from openai import OpenAI
from sentence_transformers import CrossEncoder

# SAFE IMPORT
try:
    from rag_pipeline import retrieve_candidates
except Exception as e:
    def retrieve_candidates(q):
        return [f"Pipeline error: {e}"]

st.set_page_config(page_title="AEGIS RAG", layout="wide")
st.title("🚀 AEGIS Enterprise RAG System")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LOAD LIGHTWEIGHT CROSS-ENCODER
@st.cache_resource
def load_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

model = load_model()

# RERANK
def rerank(query, candidates):
    pairs = [[query, c] for c in candidates[:25]]
    scores = model.predict(pairs)

    results = list(zip(candidates[:25], scores))
    return sorted(results, key=lambda x: x[1], reverse=True)

# GENERATE ANSWER
def generate_answer(query, context):
    prompt = f"""
Answer ONLY from context.
Give clear, factual answer.

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

# SESSION
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
                    st.write(f"{i+1}. Score: {score:.3f}")
                    st.write(text)
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.sidebar.metric("Latency", round(time.time() - start, 2))
    st.sidebar.metric("Candidates", len(candidates))
    st.sidebar.metric("Chunks Used", len(top_chunks))
