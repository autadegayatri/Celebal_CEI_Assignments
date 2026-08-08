"""
Retriever: the query-time face of the Static Knowledge (RAG) layer.
Combines an `Embedder` with a `VectorStore` to turn a natural-language
query into ranked, relevant context chunks.
"""

from __future__ import annotations

from pathlib import Path

from src import config
from src.rag.embeddings import Embedder, get_embedder
from src.rag.vector_store import SearchResult, VectorStore


class Retriever:
    def __init__(self, embedder: Embedder | None = None, vector_store: VectorStore | None = None):
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> list[SearchResult]:
        if len(self.vector_store) == 0:
            return []
        query_vec = self.embedder.transform([query])[0]
        return self.vector_store.search(query_vec, top_k=top_k)

    def retrieve_context_text(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> str:
        results = self.retrieve(query, top_k=top_k)
        return "\n\n".join(f"[{r.chunk.doc_title}] {r.chunk.text}" for r in results)

    def save(self, vector_store_path: Path = config.VECTOR_STORE_PATH, embedder_path: Path | None = None):
        self.vector_store.save(vector_store_path)
        if embedder_path is None:
            embedder_path = vector_store_path.parent / "embedder.pkl"
        self.embedder.save(embedder_path)

    @classmethod
    def load(cls, vector_store_path: Path = config.VECTOR_STORE_PATH, embedder_path: Path | None = None) -> "Retriever":
        if embedder_path is None:
            embedder_path = vector_store_path.parent / "embedder.pkl"
        embedder = get_embedder()
        embedder.load(embedder_path)
        store = VectorStore()
        store.load(vector_store_path)
        return cls(embedder=embedder, vector_store=store)
