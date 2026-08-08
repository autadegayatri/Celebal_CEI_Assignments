"""
End-to-end knowledge-base build script.

Runs the full Static Knowledge + Knowledge Graph pipeline described in the
methodology:

    1. Data pipeline      : scrape (or load local corpus) -> clean
    2. Embedding & storage: chunk -> embed -> vector store
    3. KG construction    : entity/relation extraction -> graph store

Usage:
    python -m scripts.build_knowledge_base                  # sample corpus
    python -m scripts.build_knowledge_base --urls URL1 URL2  # live scrape
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import config
from src.ingestion.chunker import chunk_documents
from src.ingestion.cleaner import clean_text
from src.knowledge_graph.entity_extractor import extract_from_text
from src.knowledge_graph.graph_store import NetworkXGraphStore
from src.rag.embeddings import get_embedder
from src.rag.vector_store import VectorStore
from src.scraping.scraper import WebScraper, load_local_corpus


def load_default_urls(path: Path | str) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def build(urls: list[str] | None = None) -> dict:
    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # 1. Data pipeline: scrape (or load local corpus) -> clean
    # -------------------------------------------------------------
    if urls is None:
        urls = load_default_urls(config.DEFAULT_SOURCE_URLS_PATH)

    if urls:
        scraper = WebScraper()
        scraped = scraper.scrape_urls(urls)
    else:
        scraped = load_local_corpus(config.SAMPLE_CORPUS_DIR)

    if not scraped:
        raise RuntimeError("No documents were scraped/loaded -- nothing to build.")

    documents = []
    for i, doc in enumerate(scraped):
        cleaned = clean_text(doc.text)
        documents.append({"doc_id": f"doc_{i}", "title": doc.title, "source": doc.source, "text": cleaned})

    # -------------------------------------------------------------
    # 2. Embedding & storage: chunk -> embed -> vector store
    # -------------------------------------------------------------
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

    # -------------------------------------------------------------
    # 3. Knowledge graph construction: entities + relations -> graph store
    # -------------------------------------------------------------
    graph_store = NetworkXGraphStore()
    total_entities, total_triples = 0, 0
    for doc in documents:
        entities, triples = extract_from_text(doc["text"])
        for e in entities:
            graph_store.add_entity(e)
        for t in triples:
            graph_store.add_triple(t)
        total_entities += len(entities)
        total_triples += len(triples)
    graph_store.save(config.GRAPH_STORE_PATH)

    elapsed = time.perf_counter() - t0

    stats = {
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "vector_store_size": len(vector_store),
        "graph_entities": graph_store.stats()["num_entities"],
        "graph_relations": graph_store.stats()["num_relations"],
        "build_time_seconds": round(elapsed, 2),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build the RAG index and knowledge graph.")
    parser.add_argument("--urls", nargs="*", default=None, help="URLs to scrape instead of the sample corpus.")
    args = parser.parse_args()

    print("Building knowledge base...")
    stats = build(urls=args.urls)
    print("\nBuild complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.path.insert(0, str(config.BASE_DIR))
    main()
