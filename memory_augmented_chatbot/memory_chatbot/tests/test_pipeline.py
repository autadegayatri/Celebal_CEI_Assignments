"""
Basic tests covering the ingestion, RAG, knowledge graph, memory, tools,
and orchestration layers. Not exhaustive, but exercises the full pipeline
end-to-end so a broken wiring change fails loudly.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.ingestion.chunker import chunk_document
from src.ingestion.cleaner import clean_text
from src.knowledge_graph.entity_extractor import extract_from_text
from src.knowledge_graph.graph_store import NetworkXGraphStore
from src.memory.memory_store import MemoryStore, extract_candidate_facts, get_memory_store
from src.rag.embeddings import TfidfEmbedder
from src.rag.vector_store import VectorStore
from src.tools import tool_registry


def test_clean_text_strips_urls_and_whitespace():
    dirty = "Check this out:   https://example.com   \n\n\n extra   spaces"
    cleaned = clean_text(dirty)
    assert "https://" not in cleaned
    assert "   " not in cleaned


def test_chunk_document_respects_min_length_and_overlap():
    text = " ".join([f"Sentence number {i} about testing." for i in range(30)])
    chunks = chunk_document("doc1", "Test Doc", "unit_test", text, chunk_size=200, overlap=40, min_length=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) >= 20
        assert c.doc_id == "doc1"


def test_tfidf_embedder_roundtrip(tmp_path):
    texts = ["Machine learning is great", "Neural networks power deep learning", "The cat sat on the mat"]
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(texts)
    assert vectors.shape[0] == 3

    save_path = tmp_path / "embedder.pkl"
    embedder.save(save_path)

    loaded = TfidfEmbedder()
    loaded.load(save_path)
    new_vectors = loaded.transform(["deep learning models"])
    assert new_vectors.shape[1] == vectors.shape[1]


def test_vector_store_search_returns_relevant_chunk():
    from src.ingestion.chunker import Chunk

    embedder = TfidfEmbedder()
    texts = ["Python is a programming language.", "Bananas are a type of fruit."]
    vectors = embedder.fit_transform(texts)

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", doc_title="Doc1", source="s1", text=texts[0]),
        Chunk(chunk_id="c2", doc_id="d1", doc_title="Doc1", source="s1", text=texts[1]),
    ]
    store = VectorStore()
    store.add(chunks, vectors)

    query_vec = embedder.transform(["What programming language is easy to learn?"])[0]
    results = store.search(query_vec, top_k=1)
    assert len(results) == 1
    assert "Python" in results[0].chunk.text


def test_entity_and_triple_extraction():
    text = "Python was developed by Guido van Rossum. Python is a subfield of programming languages."
    entities, triples = extract_from_text(text)
    assert "Python" in entities
    assert any(t.relation == "was developed by" for t in triples)


def test_extract_query_entities_ignores_question_words():
    from src.knowledge_graph.graph_query import extract_query_entities

    entities = extract_query_entities("Who created Python?")
    assert entities == ["Python"]


def test_graph_store_neighbors_and_search():
    from src.knowledge_graph.entity_extractor import Triple

    store = NetworkXGraphStore()
    store.add_triple(Triple(subject="PyTorch", relation="developed by", obj="Meta", sentence="PyTorch developed by Meta."))

    neighbors = store.neighbors("PyTorch")
    assert len(neighbors) == 1
    assert neighbors[0]["object"] == "Meta"

    matches = store.search_entities("pytorch")
    assert "PyTorch" in matches


def test_memory_store_add_and_get_facts(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=db_path)
    store.add_fact("user1", "User likes Python", category="preference")
    store.add_fact("user1", "User likes Python", category="preference")  # duplicate, should be ignored

    facts = store.get_facts("user1")
    assert len(facts) == 1
    assert facts[0].fact == "User likes Python"
    store.close()


def test_memory_store_turns_persist_and_round_trip(tmp_path):
    """Long-term memory: conversation history survives across a fresh
    connection to the same database file (i.e. a process restart)."""
    db_path = tmp_path / "test_turns.db"

    store = MemoryStore(db_path=db_path)
    store.add_turn("user1", "user", "Hello there")
    store.add_turn("user1", "assistant", "Hi! How can I help?")
    store.close()

    # Re-open a brand new connection to the same file, simulating a restart.
    store2 = MemoryStore(db_path=db_path)
    turns = store2.get_recent_turns("user1")
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello there"
    assert turns[1].role == "assistant"
    store2.close()


def test_memory_store_forget_clears_facts_and_turns(tmp_path):
    db_path = tmp_path / "test_forget.db"
    store = MemoryStore(db_path=db_path)
    store.add_fact("user1", "User is a final-year IT student")
    store.add_turn("user1", "user", "hi")

    store.forget("user1")

    assert store.get_facts("user1") == []
    assert store.get_recent_turns("user1") == []
    store.close()


def test_get_memory_store_defaults_to_sqlite(tmp_path):
    db_path = tmp_path / "test_factory.db"
    store = get_memory_store(backend="sqlite", db_path=db_path)
    assert isinstance(store, MemoryStore)
    store.close()


def test_get_memory_store_postgres_backend_is_selected_by_config():
    """Selecting the postgres backend should try to build a
    PostgresMemoryStore (and fail on the missing/unreachable connection
    rather than silently falling back to sqlite), proving MEMORY_BACKEND
    actually drives which database connection is used."""
    with pytest.raises(Exception):
        get_memory_store(backend="postgres")


def test_extract_candidate_facts():
    facts = extract_candidate_facts("Hi, I am a final year student and I like machine learning.")
    assert any("final year student" in f for f in facts)


def test_calculator_tool():
    tool = tool_registry.get_tool("calculator")
    result = tool.func("What is 12 * (3 + 4)?")
    assert "84" in result


def test_datetime_tool_returns_string():
    tool = tool_registry.get_tool("datetime")
    result = tool.func("what time is it")
    assert "Current date and time" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
