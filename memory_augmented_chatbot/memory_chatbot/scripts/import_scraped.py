"""
Import a JSONL file of scraped documents into the RAG index and knowledge graph.

Each line in the JSONL should be an object with keys: `source`, `title`, `text`.

Usage:
    python -m scripts.import_scraped --in artifacts/scraped.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src import config
from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_text
from src.knowledge_graph.entity_extractor import extract_from_text
from src.knowledge_graph.graph_store import get_graph_store
from src.rag.embeddings import get_embedder
from src.rag.vector_store import VectorStore


def read_jsonl(path: Path) -> List[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs


def import_and_build(input_path: Path) -> dict:
    raw = read_jsonl(input_path)
    if not raw:
        raise RuntimeError(f"No documents found in {input_path}")

    documents = []
    for i, d in enumerate(raw):
        text = d.get("text", "")
        cleaned = clean_text(text)
        documents.append({"doc_id": f"doc_{i}", "title": d.get("title", "Untitled"), "source": d.get("source", ""), "text": cleaned})

    chunks = chunk_documents(documents)
    texts = [c.text for c in chunks]

    embedder = get_embedder()
    if hasattr(embedder, "fit_transform"):
        embeddings = embedder.fit_transform(texts)
    else:
        embedder.fit(texts)
        embeddings = embedder.transform(texts)

    vector_store = VectorStore()
    vector_store.add(chunks, embeddings)
    vector_store.save(config.VECTOR_STORE_PATH)
    embedder.save(config.VECTOR_STORE_PATH.parent / "embedder.pkl")

    # Try to get the configured graph store; fall back to NetworkX if Neo4j is not reachable
    try:
        graph_store = get_graph_store()
    except Exception:
        from src.knowledge_graph.graph_store import NetworkXGraphStore

        print("Warning: could not connect to Neo4j; falling back to NetworkXGraphStore (local file).")
        graph_store = NetworkXGraphStore()
    total_entities, total_triples = 0, 0
    all_entities = []
    all_triples = []
    for doc in documents:
        entities, triples = extract_from_text(doc["text"])
        all_entities.extend(entities)
        all_triples.extend(triples)
        total_entities += len(entities)
        total_triples += len(triples)

    # Try to write to the configured graph store; on any error, fall back to NetworkX
    try:
        for e in all_entities:
            graph_store.add_entity(e)
        for t in all_triples:
            graph_store.add_triple(t)
    except Exception:
        from src.knowledge_graph.graph_store import NetworkXGraphStore

        print("Warning: graph backend write failed; falling back to NetworkXGraphStore and saving locally.")
        nx_store = NetworkXGraphStore()
        for e in all_entities:
            nx_store.add_entity(e)
        for t in all_triples:
            nx_store.add_triple(t)
        nx_store.save(config.GRAPH_STORE_PATH)
        graph_store = nx_store
    # Persist graph store if it supports saving (NetworkX); Neo4j is already persistent.
    try:
        if config.GRAPH_BACKEND == "networkx":
            graph_store.save(config.GRAPH_STORE_PATH)
    except Exception:
        pass

    # Optionally, store chunk metadata in Postgres when MEMORY_BACKEND=postgres
    if config.MEMORY_BACKEND == "postgres":
        try:
            import psycopg2

            conn = None
            if config.DATABASE_URL:
                conn = psycopg2.connect(config.DATABASE_URL)
            else:
                conn = psycopg2.connect(
                    dbname=config.POSTGRES_DB,
                    user=config.POSTGRES_USER,
                    password=config.POSTGRES_PASSWORD,
                    host=config.POSTGRES_HOST,
                    port=config.POSTGRES_PORT,
                )
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_id TEXT UNIQUE,
                    doc_id TEXT,
                    doc_title TEXT,
                    source TEXT,
                    text TEXT
                );
                """
            )
            # Insert chunks
            insert_sql = "INSERT INTO chunks (chunk_id, doc_id, doc_title, source, text) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (chunk_id) DO NOTHING"
            records = [(c.chunk_id, c.doc_id, c.doc_title, c.source, c.text) for c in chunks]
            if records:
                cur.executemany(insert_sql, records)
                conn.commit()
            cur.close()
            conn.close()
        except Exception:
            # If Postgres isn't reachable or psycopg2 missing, skip silently
            pass

    stats = {
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "vector_store_size": len(vector_store),
    }
    # Add graph stats when available
    try:
        gstats = graph_store.stats()
        stats["graph_entities"] = gstats.get("num_entities")
        stats["graph_relations"] = gstats.get("num_relations")
    except Exception:
        # Neo4j-backed store may not implement stats()
        stats["graph_entities"] = None
        stats["graph_relations"] = None
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, default=Path("artifacts/scraped.jsonl"))
    args = parser.parse_args()

    stats = import_and_build(args.input)
    print("Import complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
