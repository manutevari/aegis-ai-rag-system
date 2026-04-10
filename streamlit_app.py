import streamlit as st
import tempfile
from aegis_pipeline import run_pipeline

st.title("Aegis AI Assistant")

file = st.file_uploader("Upload file", type=["txt"])
query = st.text_input("Ask something")

if st.button("Run"):
    if file and query:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.getvalue())
            path = tmp.name

        result = run_pipeline(path, query)
        st.write(result)
    else:
        st.warning("Upload file and enter query")
