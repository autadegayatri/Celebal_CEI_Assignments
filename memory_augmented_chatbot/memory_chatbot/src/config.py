"""
Central configuration for the Memory-Augmented Chatbot system.

All paths, model names, and tunable parameters live here so that every
module (ingestion, RAG, knowledge graph, memory, orchestration, evaluation,
API) reads from a single source of truth.
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
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_CORPUS_DIR = DATA_DIR / "sample_corpus"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
VECTOR_STORE_PATH = ARTIFACTS_DIR / "vector_store.pkl"
GRAPH_STORE_PATH = ARTIFACTS_DIR / "knowledge_graph.gpickle"
MEMORY_DB_PATH = ARTIFACTS_DIR / "memory.db"
EVAL_REPORTS_DIR = ARTIFACTS_DIR / "eval_reports"
DEFAULT_SOURCE_URLS_PATH = BASE_DIR / "data" / "source_urls.txt"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_CORPUS_DIR, ARTIFACTS_DIR, EVAL_REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 400          # characters per chunk (approx)
CHUNK_OVERLAP = 80        # character overlap between consecutive chunks
MIN_CHUNK_LENGTH = 40     # discard chunks shorter than this

# ---------------------------------------------------------------------------
# RAG / retrieval
# ---------------------------------------------------------------------------
# Embedding backend: "tfidf" (lightweight, default, no downloads) or
# "sentence-transformers" (requires the optional `sentence-transformers`
# package + a model download; enable if disk/network allow it).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))

# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------
# Graph backend: "networkx" (default, in-process, zero setup) or "neo4j"
# (requires a running Neo4j instance + credentials below).
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "networkx")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# "anthropic" uses the real Claude API (requires ANTHROPIC_API_KEY).
# "mock" uses a deterministic, extractive fallback generator so the whole
# pipeline runs end-to-end offline / without any API key.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "groq-model")

# ---------------------------------------------------------------------------
# Memory (long-term, persistent user facts + conversation history)
# ---------------------------------------------------------------------------
# Memory backend: "sqlite" (default, zero setup, file-based at
# artifacts/memory.db) or "postgres" (real client/server database
# connection, matching the problem statement's tech stack -- requires a
# running PostgreSQL instance + `pip install psycopg2-binary`).
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "sqlite")

# Used only when MEMORY_BACKEND=postgres. Either set DATABASE_URL directly
# (e.g. "postgresql://user:password@localhost:5432/memory_chatbot") or set
# the individual POSTGRES_* fields below and it will be assembled for you.
DATABASE_URL = os.getenv("DATABASE_URL", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "memory_chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "root")

MAX_SHORT_TERM_TURNS = int(os.getenv("MAX_SHORT_TERM_TURNS", "8"))
MAX_MEMORY_FACTS_IN_PROMPT = int(os.getenv("MAX_MEMORY_FACTS_IN_PROMPT", "6"))

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true"
