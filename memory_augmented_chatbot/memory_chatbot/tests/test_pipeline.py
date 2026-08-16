"""
Tests covering ingestion, RAG, knowledge graph (Neo4j), long-term memory
(PostgreSQL), and tools.

Pure-logic tests (cleaning, chunking, embeddings, vector search, entity
extraction, fact extraction, tools) always run. Tests that need a live
PostgreSQL or Neo4j connection skip themselves (with a clear reason)
rather than failing if those services aren't reachable in the current
environment -- run `docker-compose up -d` first to exercise them for real.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from memory_augmented_chatbot.memory_chatbot.src.ingestion.chunker import chunk_document
from memory_augmented_chatbot.memory_chatbot.src.ingestion.cleaner import clean_text
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.entity_extractor import extract_from_text
from memory_augmented_chatbot.memory_chatbot.src.memory.memory_store import extract_candidate_facts
from memory_augmented_chatbot.memory_chatbot.src.rag.embeddings import TfidfEmbedder
from memory_augmented_chatbot.memory_chatbot.src.rag.vector_store import VectorStore
from memory_augmented_chatbot.memory_chatbot.src.tools import tool_registry


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# RAG (embeddings + vector store)
# ---------------------------------------------------------------------------
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
    from memory_augmented_chatbot.memory_chatbot.src.ingestion.chunker import Chunk

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


# ---------------------------------------------------------------------------
# Knowledge graph -- entity/triple extraction (pure logic, no DB needed)
# ---------------------------------------------------------------------------
def test_entity_and_triple_extraction():
    text = "Python was developed by Guido van Rossum. Python is a subfield of programming languages."
    entities, triples = extract_from_text(text)
    assert "Python" in entities
    assert any(t.relation == "was developed by" for t in triples)


def test_extract_query_entities_ignores_question_words():
    from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_query import extract_query_entities

    entities = extract_query_entities("Who created Python?")
    assert entities == ["Python"]


# ---------------------------------------------------------------------------
# Knowledge graph -- Neo4j (skips if no live instance is reachable)
# ---------------------------------------------------------------------------
@pytest.fixture
def neo4j_store():
    from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_store import Neo4jGraphStore

    try:
        store = Neo4jGraphStore()
        store.driver.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable ({exc}) -- run `docker-compose up -d neo4j` to test this.")
    yield store
    store.close()


def test_graph_store_add_and_neighbors(neo4j_store):
    from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.entity_extractor import Triple

    # unique names per test run so repeated runs don't collide
    tag = uuid.uuid4().hex[:8]
    subj, obj = f"PyTorch-{tag}", f"Meta-{tag}"

    neo4j_store.add_triple(Triple(subject=subj, relation="developed by", obj=obj, sentence=f"{subj} developed by {obj}."))

    neighbors = neo4j_store.neighbors(subj)
    assert any(n["object"] == obj and n["relation"] == "developed by" for n in neighbors)

    matches = neo4j_store.search_entities(subj)
    assert subj in matches


def test_graph_store_stats(neo4j_store):
    stats = neo4j_store.stats()
    assert "num_entities" in stats
    assert "num_relations" in stats


# ---------------------------------------------------------------------------
# Long-term memory -- PostgreSQL (skips if no live instance is reachable)
# ---------------------------------------------------------------------------
@pytest.fixture
def memory_store():
    from memory_augmented_chatbot.memory_chatbot.src.memory.memory_store import MemoryStore

    try:
        store = MemoryStore()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc}) -- run `docker-compose up -d postgres` to test this.")
    yield store
    store.close()


def test_memory_store_add_and_get_facts(memory_store):
    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    memory_store.add_fact(user_id, "User likes Python", category="preference")
    memory_store.add_fact(user_id, "User likes Python", category="preference")  # duplicate, should be ignored

    facts = memory_store.get_facts(user_id)
    assert len(facts) == 1
    assert facts[0].fact == "User likes Python"
    memory_store.forget(user_id)


def test_memory_store_turns_persist_across_connections(memory_store):
    """Long-term memory: conversation history survives a fresh connection
    to the same PostgreSQL database (i.e. a process restart)."""
    from memory_augmented_chatbot.memory_chatbot.src.memory.memory_store import MemoryStore

    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    memory_store.add_turn(user_id, "user", "Hello there")
    memory_store.add_turn(user_id, "assistant", "Hi! How can I help?")

    # a brand-new connection, simulating a process restart
    store2 = MemoryStore(dsn=memory_store.dsn)
    turns = store2.get_recent_turns(user_id)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "Hello there"
    assert turns[1].role == "assistant"

    store2.forget(user_id)
    store2.close()


def test_memory_store_forget_clears_facts_and_turns(memory_store):
    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    memory_store.add_fact(user_id, "User is a final-year IT student")
    memory_store.add_turn(user_id, "user", "hi")

    memory_store.forget(user_id)

    assert memory_store.get_facts(user_id) == []
    assert memory_store.get_recent_turns(user_id) == []


# ---------------------------------------------------------------------------
# Memory -- fact extraction (pure logic, no DB needed)
# ---------------------------------------------------------------------------
def test_extract_candidate_facts():
    facts = extract_candidate_facts("Hi, I am a final year student and I like machine learning.")
    assert any("final year student" in f for f in facts)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
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
