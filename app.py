import streamlit as st
from rag_pipeline import ask

st.set_page_config(page_title="AEGIS RAG Assistant")

st.title("AEGIS Policy Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.chat_input("Ask your question...")

if query:
    st.session_state.chat.append(("user", query))
    answer, docs = ask(query)
    st.session_state.chat.append(("bot", answer))

for role, msg in st.session_state.chat:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
