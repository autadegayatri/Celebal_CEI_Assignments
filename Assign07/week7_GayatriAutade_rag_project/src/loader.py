"""
Document Ingestion
-------------------
Stage 1 of the RAG pipeline: load raw text out of PDFs / .txt files.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file, page by page."""
    reader = PdfReader(path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def load_txt(path: str) -> str:
    """Load raw text from a plain text file."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_document(path: str) -> str:
    """Dispatch to the right loader based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md"):
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")
