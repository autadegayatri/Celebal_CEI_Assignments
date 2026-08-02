

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Question Answering (RAG)")
st.caption(
    "Upload a PDF or text file, then ask questions about it. "
    "Answers are grounded in the retrieved content, not the model's general knowledge."
)

# ---- Sidebar: backend configuration --------------------------------------
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Answer generation backend",
        options=["local", "anthropic", "openai", "cohere"],
        index=0,
        help=(
            "'local' runs a small free model on your machine, no API key needed. "
            "Others need the matching API key set as an environment variable "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / COHERE_API_KEY)."
        ),
    )
    top_k = st.slider("Chunks to retrieve per question", 1, 10, 4)
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, step=100)

# ---- Session state ---------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Document upload + ingestion -------------------------------------------
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

if uploaded_file is not None:
    ingest_clicked = st.button("Process document", type="primary")
    if ingest_clicked:
        with st.spinner("Loading, chunking, and embedding your document..."):
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            pipeline = RAGPipeline(backend=backend, chunk_size=chunk_size)
            try:
                n_chunks = pipeline.ingest(tmp_path)
                st.session_state.pipeline = pipeline
                st.session_state.chat_history = []
                st.success(f"Document processed into {n_chunks} chunks. Ask away!")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
            finally:
                os.unlink(tmp_path)

# ---- Q&A --------------------------------------------------------------------
if st.session_state.pipeline is not None:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask") and question.strip():
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                answer, results = st.session_state.pipeline.ask(question, top_k=top_k)
                st.session_state.chat_history.append((question, answer, results))
            except Exception as e:
                st.error(f"Failed to generate an answer: {e}")

    for q, a, results in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.markdown(a)
        with st.expander("Show retrieved context"):
            for chunk, score in results:
                st.markdown(f"*Score: {score:.3f}*")
                st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
                st.divider()
        st.divider()
else:
    st.info("Upload and process a document to start asking questions.")
