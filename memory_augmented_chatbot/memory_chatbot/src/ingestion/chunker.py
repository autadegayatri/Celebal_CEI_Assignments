"""
Chunking utilities: split cleaned documents into overlapping, retrieval-sized
passages. Chunking is sentence-aware so passages don't split mid-sentence
where avoidable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip())
    return sentences


def chunk_document(
    doc_id: str,
    doc_title: str,
    source: str,
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    min_length: int = config.MIN_CHUNK_LENGTH,
) -> list[Chunk]:
    """
    Greedily pack sentences into chunks of roughly `chunk_size` characters,
    carrying `overlap` characters of context into the next chunk so
    retrieval doesn't lose context at chunk boundaries.
    """
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            if len(current) >= min_length:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::chunk_{idx}",
                        doc_id=doc_id,
                        doc_title=doc_title,
                        source=source,
                        text=current.strip(),
                    )
                )
                idx += 1
            # carry the tail of the current chunk forward as overlap
            current = current[-overlap:] if overlap > 0 else ""
        current = f"{current} {sentence}".strip()

    if len(current) >= min_length:
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk_{idx}",
                doc_id=doc_id,
                doc_title=doc_title,
                source=source,
                text=current.strip(),
            )
        )

    return chunks


def chunk_documents(documents: list[dict]) -> list[Chunk]:
    """
    documents: list of dicts with keys {doc_id, title, source, text}
    Returns the flattened list of Chunk objects across all documents.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(
                doc_id=doc["doc_id"],
                doc_title=doc["title"],
                source=doc["source"],
                text=doc["text"],
            )
        )
    return all_chunks
