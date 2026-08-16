"""
Central configuration for the Memory-Augmented Chatbot system.

All paths, model names, and tunable parameters live here so every module
(ingestion, RAG, knowledge graph, memory, orchestration, evaluation, API)
reads from a single source of truth.

Backends used by this project (fixed, not switchable -- one clean path):
    LLM             Groq API              (src/llm/client.py)
    Knowledge Graph  Neo4j                  (src/knowledge_graph/graph_store.py)
    Long-term memory PostgreSQL             (src/memory/memory_store.py)
    Vector store     in-process cosine index (src/rag/vector_store.py)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
SAMPLE_CORPUS_DIR = DATA_DIR / "sample_corpus"
DEFAULT_SOURCE_URLS_PATH = DATA_DIR / "source_urls.txt"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
VECTOR_STORE_PATH = ARTIFACTS_DIR / "vector_store.pkl"
EVAL_REPORTS_DIR = ARTIFACTS_DIR / "eval_reports"

for _dir in (SAMPLE_CORPUS_DIR, ARTIFACTS_DIR, EVAL_REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 400       # characters per chunk (approx)
CHUNK_OVERLAP = 80     # character overlap between consecutive chunks
MIN_CHUNK_LENGTH = 40  # discard chunks shorter than this

# ---------------------------------------------------------------------------
# RAG / retrieval
# ---------------------------------------------------------------------------
# "tfidf" (default, lightweight, no downloads) or "sentence-transformers"
# (semantic embeddings; requires `pip install sentence-transformers`).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))

# Below this retrieval-confidence score, the router treats the query as
# "general" (not covered by the knowledge base) instead of forcing a
# low-quality RAG answer -- this is what routes open-domain questions to
# the LLM's own knowledge instead of irrelevant retrieved context.
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.08"))

# ---------------------------------------------------------------------------
# Knowledge graph -- Neo4j (required)
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ---------------------------------------------------------------------------
# LLM -- Groq (required for real generation)
# ---------------------------------------------------------------------------
# "groq" (default) calls the real Groq API and needs GROQ_API_KEY.
# "mock" is an offline, template-based fallback used only for tests / CI
# environments with no API key or network access.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Long-term memory -- PostgreSQL (required)
# ---------------------------------------------------------------------------
# Either set DATABASE_URL directly, or set the individual POSTGRES_* fields
# and it will be assembled for you.
DATABASE_URL = os.getenv("DATABASE_URL", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "memory_chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

MAX_SHORT_TERM_TURNS = int(os.getenv("MAX_SHORT_TERM_TURNS", "8"))
MAX_MEMORY_FACTS_IN_PROMPT = int(os.getenv("MAX_MEMORY_FACTS_IN_PROMPT", "6"))

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
