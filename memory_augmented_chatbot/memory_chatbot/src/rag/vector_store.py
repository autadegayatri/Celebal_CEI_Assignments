"""
In-memory vector store with cosine-similarity search and disk persistence.

This plays the role FAISS/Chroma would play in a heavier deployment. The
interface (`add`, `search`, `save`, `load`) is intentionally the same shape
a FAISS/Chroma-backed implementation would expose, so swapping in a real
ANN index later is a drop-in change -- see `FaissVectorStore` stub at the
bottom for how that swap would look.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> list[SearchResult]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_embedding = query_embedding.reshape(1, -1)
        sims = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [SearchResult(chunk=self.chunks[i], score=float(sims[i])) for i in top_indices if sims[i] > 0]

    def __len__(self) -> int:
        return len(self.chunks)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.chunks = state["chunks"]
        self.embeddings = state["embeddings"]
