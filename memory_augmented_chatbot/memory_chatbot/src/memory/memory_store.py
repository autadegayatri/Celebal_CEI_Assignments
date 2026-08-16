"""
Long-term user memory -- PostgreSQL only.

Stores, per user:
  - conversation turns (short-term history, capped)
  - durable "facts" / preferences extracted from conversation (long-term
    memory that persists across sessions and is re-injected into future
    turns for personalization)

Backed by a real PostgreSQL connection via `psycopg2`, matching the
problem statement's tech stack ("Database: MongoDB / PostgreSQL"). Requires
a running PostgreSQL instance (see docker-compose.yml) and DATABASE_URL
(or POSTGRES_* fields) in the environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import psycopg2

from memory_augmented_chatbot.memory_chatbot.src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at DOUBLE PRECISION NOT NULL,
    UNIQUE(user_id, fact)
);

CREATE TABLE IF NOT EXISTS turns (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
"""


@dataclass
class MemoryFact:
    user_id: str
    fact: str
    category: str
    created_at: float


@dataclass
class ConversationTurn:
    user_id: str
    role: str  # "user" | "assistant"
    content: str
    created_at: float


def _connection_string() -> str:
    if config.DATABASE_URL:
        return config.DATABASE_URL
    return (
        f"host={config.POSTGRES_HOST} port={config.POSTGRES_PORT} "
        f"dbname={config.POSTGRES_DB} user={config.POSTGRES_USER} "
        f"password={config.POSTGRES_PASSWORD}"
    )


class MemoryStore:
    """PostgreSQL-backed long-term memory + conversation history."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or _connection_string()
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)

    # ------------------------------------------------------------------
    # Conversation history (short-term, capped)
    # ------------------------------------------------------------------
    def add_turn(self, user_id: str, role: str, content: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO turns (user_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                (user_id, role, content, time.time()),
            )

    def get_recent_turns(self, user_id: str, limit: int = config.MAX_SHORT_TERM_TURNS) -> list[ConversationTurn]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, role, content, created_at FROM turns WHERE user_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [ConversationTurn(*row) for row in reversed(rows)]

    # ------------------------------------------------------------------
    # Long-term facts / preferences
    # ------------------------------------------------------------------
    def add_fact(self, user_id: str, fact: str, category: str = "general") -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO facts (user_id, fact, category, created_at) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (user_id, fact) DO NOTHING",
                (user_id, fact, category, time.time()),
            )

    def get_facts(self, user_id: str, limit: int = config.MAX_MEMORY_FACTS_IN_PROMPT) -> list[MemoryFact]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, fact, category, created_at FROM facts WHERE user_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [MemoryFact(*row) for row in rows]

    def get_facts_text(self, user_id: str, limit: int = config.MAX_MEMORY_FACTS_IN_PROMPT) -> str:
        facts = self.get_facts(user_id, limit=limit)
        if not facts:
            return ""
        return "\n".join(f"- {f.fact}" for f in facts)

    def forget(self, user_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM facts WHERE user_id = %s", (user_id,))
            cur.execute("DELETE FROM turns WHERE user_id = %s", (user_id,))

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Lightweight fact extraction: pulls durable preference/identity statements
# out of a user message ("I am a...", "I like...", "remember that...", etc.)
# so the orchestration layer can decide what's worth persisting long-term.
# ---------------------------------------------------------------------------
_FACT_TRIGGERS = [
    "i am ", "i'm ", "my name is ", "i like ", "i prefer ", "i work ",
    "i live ", "remember that ", "remember ", "i study ", "i'm interested in ",
    "i love ", "i hate ", "i dislike ", "call me ",
]


def extract_candidate_facts(user_message: str) -> list[str]:
    lower = user_message.lower()
    facts = []
    for trigger in _FACT_TRIGGERS:
        idx = lower.find(trigger)
        if idx != -1:
            end = lower.find(".", idx)
            snippet = user_message[idx : end if end != -1 else len(user_message)].strip()
            if snippet and len(snippet) < 200:
                facts.append(snippet.rstrip("."))
    return facts


def get_memory_store() -> MemoryStore:
    return MemoryStore()
