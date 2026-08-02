"""
Text Chunking
-------------
Stage 2 of the RAG pipeline: split raw document text into overlapping
chunks so retrieval can operate on small, semantically coherent pieces
instead of the whole document at once.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    text: str
    source: str


def _split_into_sentences(text: str):
    # Lightweight sentence splitter (no heavy NLP dependency needed).
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in sentences if s]


def chunk_text(text: str, source: str = "document", chunk_size: int = 800,
               overlap: int = 150) -> list[Chunk]:
    """
    Split text into overlapping chunks of ~chunk_size characters,
    breaking on sentence boundaries where possible so chunks stay
    coherent instead of cutting words/sentences in half.

    Args:
        text: raw document text
        source: label for where this text came from (filename, etc.)
        chunk_size: target size of each chunk in characters
        overlap: how many characters of overlap between consecutive chunks
    """
    sentences = _split_into_sentences(text)
    chunks: list[Chunk] = []
    current = ""
    chunk_id = 0

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(Chunk(id=chunk_id, text=current, source=source))
                chunk_id += 1
                # carry the tail of the previous chunk forward for overlap
                current = current[-overlap:] + " " + sentence
            else:
                # single sentence longer than chunk_size: hard-split it
                for i in range(0, len(sentence), chunk_size - overlap):
                    piece = sentence[i:i + chunk_size]
                    chunks.append(Chunk(id=chunk_id, text=piece, source=source))
                    chunk_id += 1
                current = ""

    if current.strip():
        chunks.append(Chunk(id=chunk_id, text=current.strip(), source=source))

    return chunks
