"""
End-to-end knowledge-base build pipeline.

    1. Data pipeline      : scrape URLs (data/source_urls.txt or --urls),
                             or fall back to the bundled sample corpus -> clean
    2. Embedding & storage: chunk -> embed -> vector store (artifacts/vector_store.pkl)
    3. KG construction    : entity/relation extraction -> Neo4j

Usage:
    python -m scripts.build_knowledge_base                   # data/source_urls.txt, or sample corpus if empty
    python -m scripts.build_knowledge_base --urls URL1 URL2  # explicit URLs
    python -m scripts.build_knowledge_base --sample-only      # force the bundled sample corpus
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from memory_augmented_chatbot.memory_chatbot.src import config
from memory_augmented_chatbot.memory_chatbot.src.ingestion.chunker import chunk_documents
from memory_augmented_chatbot.memory_chatbot.src.ingestion.cleaner import clean_text
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.entity_extractor import extract_from_text
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_store import get_graph_store
from memory_augmented_chatbot.memory_chatbot.src.rag.embeddings import get_embedder
from memory_augmented_chatbot.memory_chatbot.src.rag.vector_store import VectorStore
from memory_augmented_chatbot.memory_chatbot.src.scraping.scraper import WebScraper, load_local_corpus


def _load_default_urls(path: Path) -> list[str]:
    if not path.exists():
        return []
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def build(urls: list[str] | None = None, sample_only: bool = False) -> dict:
    t0 = time.perf_counter()

    # -------------------------------------------------------------
    # 1. Data pipeline: scrape (or load local corpus) -> clean
    # -------------------------------------------------------------
    if sample_only:
        urls = []
    elif urls is None:
        urls = _load_default_urls(config.DEFAULT_SOURCE_URLS_PATH)

    if urls:
        scraped = WebScraper().scrape_urls(urls)
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
    # 3. Knowledge graph construction: entities + relations -> Neo4j
    # -------------------------------------------------------------
    graph_store = get_graph_store()
    graph_store.clear()  # fresh build -- avoid stacking duplicate data across runs

    total_entities, total_triples = 0, 0
    for doc in documents:
        entities, triples = extract_from_text(doc["text"])
        for e in entities:
            graph_store.add_entity(e)
        for t in triples:
            graph_store.add_triple(t)
        total_entities += len(entities)
        total_triples += len(triples)

    graph_stats = graph_store.stats()
    graph_store.close()

    elapsed = time.perf_counter() - t0

    return {
        "documents_processed": len(documents),
        "chunks_created": len(chunks),
        "vector_store_size": len(vector_store),
        "graph_entities": graph_stats["num_entities"],
        "graph_relations": graph_stats["num_relations"],
        "build_time_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Build the RAG index and Neo4j knowledge graph.")
    parser.add_argument("--urls", nargs="*", default=None, help="URLs to scrape instead of data/source_urls.txt.")
    parser.add_argument("--sample-only", action="store_true", help="Force the bundled sample corpus, ignore URLs.")
    args = parser.parse_args()

    print("Building knowledge base...")
    stats = build(urls=args.urls, sample_only=args.sample_only)
    print("\nBuild complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
