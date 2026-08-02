"""
Embedding Creation
------------------
Stage 3 of the RAG pipeline: turn text chunks into dense vector
representations using a local sentence-transformers model. This runs
fully offline/free (no API key required) after the model is downloaded
the first time.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, fast, good quality for QA retrieval


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings into an (N, dim) float32 numpy array."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # so cosine similarity == dot product
        )
        return embeddings.astype("float32")

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
