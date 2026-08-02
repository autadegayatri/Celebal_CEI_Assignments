"""
Command-line interface for quick testing.

Usage:
    python cli.py path/to/document.pdf
    (then type questions at the prompt; Ctrl+C or 'exit' to quit)

Set BACKEND env var to anthropic / openai / cohere / local (default: local)
"""

import sys
import os

from src.pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]
    backend = os.getenv("BACKEND", "local")

    print(f"Loading and indexing '{file_path}' (backend={backend})...")
    pipeline = RAGPipeline(backend=backend)
    n_chunks = pipeline.ingest(file_path)
    print(f"Indexed {n_chunks} chunks. Ask questions below (type 'exit' to quit).\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break

        answer, results = pipeline.ask(question)
        print(f"\nA: {answer}\n")
        print("Sources:")
        for chunk, score in results:
            preview = chunk.text[:120].replace("\n", " ")
            print(f"  [{score:.3f}] {preview}...")
        print()


if __name__ == "__main__":
    main()
