"""
Vector Database
---------------
Stage 4 of the RAG pipeline: store chunk embeddings in a FAISS index
for fast similarity search, and Stage 6 (Context Retrieval): pull back
the top-k most relevant chunks for a query embedding.
"""

import pickle
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        # Inner product search on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        assert len(embeddings) == len(chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        """Return the top_k (chunk, score) pairs most similar to the query."""
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path: str) -> None:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorStore":
        path = Path(dir_path)
        index = faiss.read_index(str(path / "index.faiss"))
        store = cls(dim=index.d)
        store.index = index
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store
