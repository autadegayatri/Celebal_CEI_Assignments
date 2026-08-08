"""
PostgreSQL-backed long-term memory store.

Real client/server database connection for the memory layer, matching the
problem statement's tech stack ("Database: MongoDB / PostgreSQL"). Implements
the exact same interface as the default SQLite `MemoryStore`
(`src.memory.memory_store.BaseMemoryStore`), so `ChatbotGraph`, the FastAPI
app, and the Streamlit UI all work unchanged regardless of which backend is
active.

Enable with:
    MEMORY_BACKEND=postgres
    DATABASE_URL=postgresql://user:password@host:5432/memory_chatbot
    (or POSTGRES_HOST / POSTGRES_PORT / POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD)

Requires: pip install psycopg2-binary
"""

from __future__ import annotations

import time

from src import config
from src.memory.memory_store import BaseMemoryStore, ConversationTurn, MemoryFact

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


def _connection_string() -> str:
    if config.DATABASE_URL:
        return config.DATABASE_URL
    return (
        f"host={config.POSTGRES_HOST} port={config.POSTGRES_PORT} "
        f"dbname={config.POSTGRES_DB} user={config.POSTGRES_USER} "
        f"password={config.POSTGRES_PASSWORD}"
    )


class PostgresMemoryStore(BaseMemoryStore):
    """PostgreSQL implementation of the long-term memory interface."""

    def __init__(self, dsn: str | None = None):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise ImportError(
                "MEMORY_BACKEND=postgres requires the `psycopg2` driver. "
                "Run `pip install psycopg2-binary` and make sure a PostgreSQL "
                "instance is reachable at DATABASE_URL / POSTGRES_* settings."
            ) from exc

        self._psycopg2 = psycopg2
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

    def close(self):
        self.conn.close()
