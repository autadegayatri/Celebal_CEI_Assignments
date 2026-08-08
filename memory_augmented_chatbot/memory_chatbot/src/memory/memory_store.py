"""
Long-term user memory.

Stores, per user:
  - conversation turns (short-term history, capped)
  - durable "facts" / preferences extracted from conversation (long-term
    memory that persists and is re-injected into future sessions)

Two backends behind a common interface (see `BaseMemoryStore`), selected via
config.MEMORY_BACKEND / the MEMORY_BACKEND env var:

  - "sqlite" (default): `MemoryStore`, a local file-based database at
    artifacts/memory.db. Survives process restarts with zero external
    dependencies -- what the system uses out of the box.
  - "postgres": `PostgresMemoryStore` (src/memory/postgres_store.py), a real
    client/server database connection via `psycopg2`, matching the
    problem statement's tech stack. Requires a running PostgreSQL instance
    and `DATABASE_URL` (or `POSTGRES_*`) to be configured.

Callers should generally use `get_memory_store()` rather than instantiating
a backend class directly, so the active backend is driven entirely by
config with no code changes elsewhere.
"""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src import config


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


class BaseMemoryStore(ABC):
    """Common interface every memory backend (SQLite, Postgres, ...) implements."""

    @abstractmethod
    def add_turn(self, user_id: str, role: str, content: str) -> None: ...

    @abstractmethod
    def get_recent_turns(self, user_id: str, limit: int = config.MAX_SHORT_TERM_TURNS) -> list[ConversationTurn]: ...

    @abstractmethod
    def add_fact(self, user_id: str, fact: str, category: str = "general") -> None: ...

    @abstractmethod
    def get_facts(self, user_id: str, limit: int = config.MAX_MEMORY_FACTS_IN_PROMPT) -> list[MemoryFact]: ...

    def get_facts_text(self, user_id: str, limit: int = config.MAX_MEMORY_FACTS_IN_PROMPT) -> str:
        facts = self.get_facts(user_id, limit=limit)
        if not facts:
            return ""
        return "\n".join(f"- {f.fact}" for f in facts)

    @abstractmethod
    def forget(self, user_id: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at REAL NOT NULL,
    UNIQUE(user_id, fact)
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns(user_id);
"""


class MemoryStore(BaseMemoryStore):
    """SQLite-backed implementation. Default, zero-setup memory backend."""

    def __init__(self, db_path: Path = config.MEMORY_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Conversation history (short-term, capped)
    # ------------------------------------------------------------------
    def add_turn(self, user_id: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO turns (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, content, time.time()),
        )
        self.conn.commit()

    def get_recent_turns(self, user_id: str, limit: int = config.MAX_SHORT_TERM_TURNS) -> list[ConversationTurn]:
        rows = self.conn.execute(
            "SELECT user_id, role, content, created_at FROM turns WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [ConversationTurn(*row) for row in reversed(rows)]

    # ------------------------------------------------------------------
    # Long-term facts / preferences
    # ------------------------------------------------------------------
    def add_fact(self, user_id: str, fact: str, category: str = "general") -> None:
        try:
            self.conn.execute(
                "INSERT INTO facts (user_id, fact, category, created_at) VALUES (?, ?, ?, ?)",
                (user_id, fact, category, time.time()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # fact already stored for this user

    def get_facts(self, user_id: str, limit: int = config.MAX_MEMORY_FACTS_IN_PROMPT) -> list[MemoryFact]:
        rows = self.conn.execute(
            "SELECT user_id, fact, category, created_at FROM facts WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [MemoryFact(*row) for row in rows]

    def get_facts_text(self, user_id: str, limit: int = config.MAX_MEMORY_FACTS_IN_PROMPT) -> str:
        facts = self.get_facts(user_id, limit=limit)
        if not facts:
            return ""
        return "\n".join(f"- {f.fact}" for f in facts)

    def forget(self, user_id: str) -> None:
        self.conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
        self.conn.execute("DELETE FROM turns WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def close(self):
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
            # take the sentence containing the trigger
            end = lower.find(".", idx)
            snippet = user_message[idx : end if end != -1 else len(user_message)].strip()
            if snippet and len(snippet) < 200:
                facts.append(snippet.rstrip("."))
    return facts


# ---------------------------------------------------------------------------
# Backend factory -- callers should prefer this over instantiating a
# concrete class directly, so MEMORY_BACKEND fully controls which database
# connection is used with no changes needed at call sites.
# ---------------------------------------------------------------------------
def get_memory_store(backend: str | None = None, db_path: Path = config.MEMORY_DB_PATH) -> BaseMemoryStore:
    """Return a long-term memory store for the configured backend.

    backend="sqlite" (default) -> local file DB, no setup required.
    backend="postgres"         -> real PostgreSQL connection (DATABASE_URL /
                                   POSTGRES_* env vars); requires
                                   `pip install psycopg2-binary` and a
                                   reachable Postgres instance.
    """
    backend = backend or config.MEMORY_BACKEND
    if backend == "postgres":
        from src.memory.postgres_store import PostgresMemoryStore

        return PostgresMemoryStore()
    return MemoryStore(db_path=db_path)
