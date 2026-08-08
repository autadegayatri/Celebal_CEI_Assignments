"""
Shared state schema passed between LangGraph nodes.

LangGraph threads a single mutable state dict through every node in the
graph. Each node reads what it needs and returns a partial update that gets
merged back in.
"""

from __future__ import annotations

from typing import TypedDict


class ChatState(TypedDict, total=False):
    # ---- input ----
    user_id: str
    query: str

    # ---- routing ----
    route: str              # "rag" | "kg" | "tool" | "memory_only"
    tool_name: str | None    # which tool to call, if route == "tool"

    # ---- retrieved context (populated by whichever node(s) run) ----
    memory_facts: str
    conversation_history: str
    rag_context: str
    kg_context: str
    tool_output: str

    # ---- generation ----
    system_prompt: str
    final_prompt: str
    answer: str

    # ---- bookkeeping / evaluation hooks ----
    retrieved_chunks: list  # raw SearchResult objects, for evaluation
    new_facts: list[str]    # facts extracted this turn, for memory write-back
