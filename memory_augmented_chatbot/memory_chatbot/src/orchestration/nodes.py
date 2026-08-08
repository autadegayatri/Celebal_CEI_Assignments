"""
LangGraph node implementations.

Each node is a plain function `(state) -> partial_state_update`, matching
the four node types called for in the problem statement's architecture:

  - Memory node   -> pulls long-term facts + recent history for this user
  - Router (Model) node -> decides RAG vs Knowledge Graph vs Tool
  - RAG node      -> static knowledge retrieval (vector search)
  - KG node       -> structured relationship lookup
  - Tool node     -> real-time / dynamic API & tool calls
  - Generation (Model) node -> composes the final answer via the LLM client
  - Memory-write node -> persists any new durable facts from this turn
"""

from __future__ import annotations

import re

from src import config
from src.knowledge_graph.graph_query import extract_query_entities, query_graph_for_entity
from src.knowledge_graph.graph_store import GraphStore
from src.llm.client import LLMClient
from src.memory.memory_store import MemoryStore, extract_candidate_facts
from src.orchestration.state import ChatState
from src.rag.retriever import Retriever
from src.tools import tool_registry

SYSTEM_PROMPT = (
    "You are a helpful, memory-augmented AI assistant. You have access to "
    "a static knowledge base (RAG), a knowledge graph of entities and "
    "relationships, real-time tools, and long-term memory about the user. "
    "Answer using only the CONTEXT provided. If the context doesn't contain "
    "the answer, say so honestly instead of guessing. Be concise and direct. "
    "If relevant memory facts about the user are present, personalize your "
    "answer to them."
)

# Keywords that suggest the query wants live/real-time/computed info rather
# than static knowledge.
_TOOL_KEYWORDS = {
    "datetime": ["what time", "current date", "today's date", "what day is it", "current time"],
    "calculator": ["calculate", "what is", "+", "-", "*", "/", "sum of", "multiply", "divide"],
    "web_search": ["latest", "current news", "today", "right now", "this week", "recent"],
}

# Keywords that suggest the query is about structured relationships
# ("who made X", "what is X part of") -> route to the knowledge graph.
_KG_KEYWORDS = [
    "who developed", "who created", "who invented", "who founded",
    "related to", "relationship between", "connected to", "part of",
    "developed by", "founded by", "invented by", "who made",
]

# Titles that commonly indicate a question about an entity/person relationship
_KG_TITLES = [
    "ceo",
    "founder",
    "co-founder",
    "cofounder",
    "president",
    "chair",
    "cto",
    "chief",
    "director",
]


def router_node(state: ChatState) -> dict:
    """Decide which retrieval path(s) this query needs."""
    query_lower = state["query"].lower()

    for tool_name, keywords in _TOOL_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            # avoid false-triggering calculator on ordinary "what is X" questions
            if tool_name == "calculator" and not re.search(r"\d", query_lower):
                continue
            return {"route": "tool", "tool_name": tool_name}

    if any(kw in query_lower for kw in _KG_KEYWORDS):
        return {"route": "kg"}

    # route person/role questions (e.g. "who is the CEO of OpenAI") to the KG
    if query_lower.startswith("who") and any(title in query_lower for title in _KG_TITLES):
        return {"route": "kg"}

    return {"route": "rag"}


def memory_node(state: ChatState, memory_store: MemoryStore) -> dict:
    """Load long-term facts + recent conversation history for this user."""
    user_id = state["user_id"]
    facts_text = memory_store.get_facts_text(user_id)
    recent_turns = memory_store.get_recent_turns(user_id, limit=config.MAX_SHORT_TERM_TURNS)
    history_text = "\n".join(f"{t.role}: {t.content}" for t in recent_turns)
    return {"memory_facts": facts_text, "conversation_history": history_text}


def rag_node(state: ChatState, retriever: Retriever) -> dict:
    results = retriever.retrieve(state["query"])
    context = "\n\n".join(f"[{r.chunk.doc_title}] {r.chunk.text}" for r in results)
    return {"rag_context": context, "retrieved_chunks": results}


def kg_node(state: ChatState, graph_store: GraphStore) -> dict:
    entities = extract_query_entities(state["query"])
    contexts = []
    for entity in entities[:3]:
        result = query_graph_for_entity(graph_store, entity)
        if result:
            contexts.append(result)
    if not contexts and entities:
        for entity in entities[:3]:
            fallback = graph_store.search_entities(entity, limit=1)
            if fallback:
                contexts.append(query_graph_for_entity(graph_store, fallback[0]))
    return {"kg_context": "\n".join(contexts)}


def tool_node(state: ChatState) -> dict:
    tool = tool_registry.get_tool(state.get("tool_name", ""))
    if tool is None:
        return {"tool_output": ""}
    output = tool.func(state["query"])
    return {"tool_output": output}


def build_prompt_node(state: ChatState) -> dict:
    """Assemble the final grounded prompt for the generation node."""
    context_parts = []
    if state.get("memory_facts"):
        context_parts.append(f"USER MEMORY:\n{state['memory_facts']}")
    if state.get("conversation_history"):
        context_parts.append(f"RECENT CONVERSATION:\n{state['conversation_history']}")
    if state.get("rag_context"):
        context_parts.append(f"KNOWLEDGE BASE:\n{state['rag_context']}")
    if state.get("kg_context"):
        context_parts.append(f"KNOWLEDGE GRAPH FACTS:\n{state['kg_context']}")
    if state.get("tool_output"):
        context_parts.append(f"TOOL RESULT:\n{state['tool_output']}")

    context = "\n\n".join(context_parts)
    final_prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{state['query']}"
    return {"system_prompt": SYSTEM_PROMPT, "final_prompt": final_prompt}


def generation_node(state: ChatState, llm_client: LLMClient) -> dict:
    answer = llm_client.generate(state["system_prompt"], state["final_prompt"])
    return {"answer": answer}


def memory_write_node(state: ChatState, memory_store: MemoryStore) -> dict:
    """Persist the turn, and any durable facts detected in the user message."""
    user_id = state["user_id"]
    memory_store.add_turn(user_id, "user", state["query"])
    memory_store.add_turn(user_id, "assistant", state["answer"])

    new_facts = extract_candidate_facts(state["query"])
    for fact in new_facts:
        memory_store.add_fact(user_id, fact)

    return {"new_facts": new_facts}


def route_decision(state: ChatState) -> str:
    """Conditional-edge function: map router output to the next node name."""
    return state.get("route", "rag")
