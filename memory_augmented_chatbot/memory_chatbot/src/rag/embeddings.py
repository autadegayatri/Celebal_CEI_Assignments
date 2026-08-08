"""
Embedding backends for the Static Knowledge (RAG) layer.

Two interchangeable backends are provided behind a common `Embedder`
interface:

- `TfidfEmbedder` (default): scikit-learn TF-IDF vectors. Zero downloads,
  fast, and a good fit for a keyword-rich domain corpus. This is what ships
  by default so the whole system runs offline / anywhere.
- `SentenceTransformerEmbedder` (optional): dense semantic embeddings via
  `sentence-transformers`. Enable by installing the package and setting
  `EMBEDDING_BACKEND=sentence-transformers` in the environment.

Swapping backends does not require touching any other module -- everything
downstream talks to the `Embedder` interface.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None: ...

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...


class TfidfEmbedder(Embedder):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50_000,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before transform().")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "fitted": self._fitted}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self._fitted = state["fitted"]


class SentenceTransformerEmbedder(Embedder):
    """Optional dense-embedding backend. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str = config.SENTENCE_TRANSFORMER_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` or use EMBEDDING_BACKEND=tfidf."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        # Pretrained model -- nothing to fit.
        pass

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, show_progress_bar=False), dtype=np.float32)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model_name": self.model_name}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.model_name = state["model_name"]
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)


def get_embedder(backend: str | None = None) -> Embedder:
    backend = backend or config.EMBEDDING_BACKEND
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return TfidfEmbedder()
