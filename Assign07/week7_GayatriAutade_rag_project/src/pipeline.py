"""
RAG Pipeline
------------
Orchestrates all 7 stages described in the project spec:
  1. Document Ingestion   (loader.py)
  2. Text Chunking        (chunker.py)
  3. Embedding Creation   (embeddings.py)
  4. Vector Database      (vectorstore.py)
  5. Query Processing     (embeddings.py, reused for the question)
  6. Context Retrieval    (vectorstore.py)
  7. Answer Generation    (generator.py)
"""

from .loader import load_document
from .chunker import chunk_text
from .embeddings import Embedder
from .vectorstore import VectorStore
from .generator import Generator


class RAGPipeline:
    def __init__(self, backend: str = "local", chunk_size: int = 800, overlap: int = 150):
        self.embedder = Embedder()
        self.generator = Generator(backend=backend)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store: VectorStore | None = None

    def ingest(self, file_path: str) -> int:
        """Load a document, chunk it, embed the chunks, and index them.
        Returns the number of chunks created."""
        raw_text = load_document(file_path)
        chunks = chunk_text(
            raw_text, source=file_path,
            chunk_size=self.chunk_size, overlap=self.overlap,
        )
        if not chunks:
            raise ValueError("No text could be extracted from this document.")

        embeddings = self.embedder.embed([c.text for c in chunks])

        self.store = VectorStore(dim=embeddings.shape[1])
        self.store.add(embeddings, chunks)
        return len(chunks)

    def ask(self, question: str, top_k: int = 4):
        """Answer a question using the ingested document(s).
        Returns (answer, retrieved_chunks_with_scores)."""
        if self.store is None:
            raise RuntimeError("No document has been ingested yet. Call ingest() first.")

        query_embedding = self.embedder.embed_one(question)
        results = self.store.search(query_embedding, top_k=top_k)

        context_chunks = [chunk.text for chunk, _score in results]
        answer = self.generator.generate(question, context_chunks)
        return answer, results
