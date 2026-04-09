import streamlit as st
from aegis_pipeline import run_pipeline
import tempfile

st.title("Aegis AI Assistant")

file = st.file_uploader("Upload file")
query = st.text_input("Ask something")

if st.button("Run"):
    if file and query:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name
        result = run_pipeline(file_path, query)

st.markdown("### 📌 Answer")
st.write(result)

# 👇 STEP 4 (DEBUG CONTEXT)
st.markdown("### 🧠 Debug Context")
st.write("Check terminal logs for retrieved chunks")
