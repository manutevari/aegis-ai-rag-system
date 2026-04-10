import streamlit as st
import tempfile
from aegis_pipeline import run_pipeline

st.title("Aegis AI Assistant")

uploaded_file = st.file_uploader("Upload file", type=["txt"])
query = st.text_input("Ask something")

if st.button("Run"):
    if uploaded_file and query:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name

        result, fig = run_pipeline(file_path, query)

        st.markdown("### 📌 Answer")
        st.write(result)

        st.markdown("### 📊 Chunk Analysis")
        st.pyplot(fig)

    else:
        st.warning("Upload file and enter query")

