"""
Streamlit chat interface for the Memory-Augmented Chatbot.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.knowledge_graph.graph_store import get_graph_store
from src.memory.memory_store import get_memory_store
from src.orchestration.graph import ChatbotGraph
from src.rag.retriever import Retriever
from ui.style import inject_custom_css, ROUTE_LABELS

st.set_page_config(page_title="Memory-Augmented Chatbot", page_icon="🧠", layout="wide")
inject_custom_css()



@st.cache_resource
def load_chatbot() -> ChatbotGraph:
    try:
        retriever = Retriever.load()
    except (FileNotFoundError, EOFError, AttributeError, ImportError, ValueError, pickle.UnpicklingError):
        st.warning("Knowledge base files were missing or incompatible. Rebuilding them from the sample corpus...")
        # Prefer importing existing scraped artifacts if available.
        from src import config as _config
        scraped_path = _config.ARTIFACTS_DIR / "scraped.jsonl"
        try:
            if scraped_path.exists():
                from scripts.import_scraped import import_and_build

                import_and_build(scraped_path)
            else:
                from scripts.build_knowledge_base import build

                build()
        finally:
            retriever = Retriever.load()

    graph_store = get_graph_store()
    if hasattr(graph_store, "load") and config.GRAPH_STORE_PATH.exists():
        graph_store.load()

    if len(retriever.vector_store) == 0 or not hasattr(graph_store, "stats") or graph_store.stats().get("num_entities", 0) == 0:
        st.warning("The knowledge base is still empty. Rebuilding the index and graph now...")
        # If we have scraped artifacts, import them; otherwise rebuild from sample corpus
        scraped_path = config.ARTIFACTS_DIR / "scraped.jsonl"
        if scraped_path.exists():
            from scripts.import_scraped import import_and_build

            import_and_build(scraped_path)
        else:
            from scripts.build_knowledge_base import build

            build()

        retriever = Retriever.load()
        graph_store = get_graph_store()
        if hasattr(graph_store, "load") and config.GRAPH_STORE_PATH.exists():
            graph_store.load()

    memory_store = get_memory_store()
    return ChatbotGraph(retriever=retriever, graph_store=graph_store, memory_store=memory_store)


chatbot = load_chatbot()

st.title("🧠 Memory-Augmented Chatbot")
st.caption("RAG + Knowledge Graph + Long-Term Memory, orchestrated with LangGraph")

with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value="demo_user")

    st.divider()
    st.subheader("Long-term memory")
    st.caption(f"Database: **{config.MEMORY_BACKEND}**" + (
        "  ·  connected" if config.MEMORY_BACKEND == "sqlite"
        else f"  ·  {config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
    ))
    facts = chatbot.memory_store.get_facts(user_id, limit=20)
    if facts:
        for f in facts:
            st.markdown(f"- {f.fact}")
    else:
        st.caption("No facts stored yet for this user.")

    if st.button("Forget this user"):
        chatbot.memory_store.forget(user_id)
        st.success("Memory cleared.")
        st.rerun()

    st.divider()
    st.subheader("Knowledge base stats")
    st.write(f"Chunks indexed: {len(chatbot.retriever.vector_store)}")
    if hasattr(chatbot.graph_store, "stats"):
        gstats = chatbot.graph_store.stats()
        st.write(f"KG entities: {gstats['num_entities']}")
        st.write(f"KG relations: {gstats['num_relations']}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chatbot.chat(user_id, prompt)
            answer = result.get("answer", "")
            st.markdown(answer)

            with st.expander("🔍 Debug: routing & retrieved context"):
                st.write(f"**Route:** {result.get('route')}")
                if result.get("rag_context"):
                    st.markdown("**RAG context:**")
                    st.code(result["rag_context"])
                if result.get("kg_context"):
                    st.markdown("**Knowledge Graph context:**")
                    st.code(result["kg_context"])
                if result.get("tool_output"):
                    st.markdown("**Tool output:**")
                    st.code(result["tool_output"])
                if result.get("memory_facts"):
                    st.markdown("**Memory used:**")
                    st.code(result["memory_facts"])

    st.session_state.messages.append({"role": "assistant", "content": answer})
