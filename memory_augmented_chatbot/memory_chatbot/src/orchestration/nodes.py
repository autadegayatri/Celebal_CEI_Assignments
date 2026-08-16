"""
LangGraph node implementations.

Each node is a plain function `(state) -> partial_state_update`. Together
they implement the architecture from the problem statement (Memory, Model
"router", RAG, Knowledge Graph, Tool nodes), plus a General node for
open-domain questions the knowledge base doesn't cover:

  - Memory node      -> pulls long-term facts + recent history for this user
  - Router node       -> decides RAG vs Knowledge Graph vs Tool vs General
  - RAG node          -> static knowledge retrieval (vector search)
  - KG node           -> structured relationship lookup (Neo4j)
  - Tool node         -> real-time / dynamic API & tool calls
  - General node      -> open question, not covered by KB/KG/tools --
                          answered directly by the LLM's own knowledge
  - Build-prompt node -> assembles the grounded/open prompt for generation
  - Generation node   -> calls the LLM (Groq) to produce the final answer
  - Memory-write node -> persists the turn + any durable facts detected
"""

from __future__ import annotations

import re

from memory_augmented_chatbot.memory_chatbot.src import config
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_query import extract_query_entities, query_graph_for_entity
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_store import Neo4jGraphStore
from memory_augmented_chatbot.memory_chatbot.src.llm.client import LLMClient
from memory_augmented_chatbot.memory_chatbot.src.memory.memory_store import MemoryStore, extract_candidate_facts
from memory_augmented_chatbot.memory_chatbot.src.orchestration.state import ChatState
from memory_augmented_chatbot.memory_chatbot.src.rag.retriever import Retriever
from memory_augmented_chatbot.memory_chatbot.src.tools import tool_registry

GROUNDED_SYSTEM_PROMPT = (
    "You are a helpful, memory-augmented AI assistant. You have access to a "
    "static knowledge base (RAG), a knowledge graph of entities and "
    "relationships, and real-time tools. Prioritize the information in "
    "CONTEXT when answering -- it comes from the assistant's own knowledge "
    "base and is usually the most accurate source for this question. You "
    "may also draw on your own general knowledge to explain, elaborate, or "
    "fill small gaps, but don't contradict the provided context. Answer "
    "naturally and conversationally, not as a list of copied facts. If "
    "relevant memory facts about the user are present, personalize your "
    "answer to them."
)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, memory-augmented AI assistant. This question is "
    "outside your knowledge base and knowledge graph, so answer it using "
    "your own general knowledge, the same way a capable AI assistant "
    "would. Be direct, accurate, and conversational. If relevant memory "
    "facts about the user are present, personalize your answer to them."
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

_KG_TITLES = ["ceo", "founder", "co-founder", "cofounder", "president", "chair", "cto", "chief", "director"]


def router_node(state: ChatState, retriever: Retriever) -> dict:
    """Decide which path best answers this query: a real-time Tool, the
    Knowledge Graph, the static knowledge base (RAG), or -- if none of
    those are a good fit -- General open-domain Q&A answered directly by
    the LLM's own knowledge."""
    query_lower = state["query"].lower()

    for tool_name, keywords in _TOOL_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            if tool_name == "calculator" and not re.search(r"\d", query_lower):
                continue
            return {"route": "tool", "tool_name": tool_name}

    if any(kw in query_lower for kw in _KG_KEYWORDS):
        return {"route": "kg"}
    if query_lower.startswith("who") and any(title in query_lower for title in _KG_TITLES):
        return {"route": "kg"}

    # Neither a tool nor a KG-style relationship question -- check whether
    # the knowledge base actually has relevant material before committing
    # to RAG. A low best-match score means this is a general question the
    # corpus doesn't cover, so it goes to the General node instead of
    # forcing an answer out of irrelevant retrieved chunks.
    results = retriever.retrieve(state["query"])
    best_score = results[0].score if results else 0.0
    if best_score < config.RAG_CONFIDENCE_THRESHOLD:
        return {"route": "general"}

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


def kg_node(state: ChatState, graph_store: Neo4jGraphStore) -> dict:
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


def general_node(state: ChatState) -> dict:
    """No specialized retrieval for this route -- the question is
    open-domain and gets answered directly from the LLM's own knowledge in
    the generation node. This node exists explicitly (rather than routing
    straight to build_prompt) to keep the graph's four answer paths
    symmetric and easy to reason about: RAG / KG / Tool / General."""
    return {}


def build_prompt_node(state: ChatState) -> dict:
    """Assemble the final prompt for the generation node. Grounded routes
    (rag/kg/tool) get the retrieved context plus grounded instructions;
    the general route gets an open system prompt and no forced context."""
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

    system_prompt = GENERAL_SYSTEM_PROMPT if state.get("route") == "general" else GROUNDED_SYSTEM_PROMPT
    return {"system_prompt": system_prompt, "final_prompt": final_prompt}


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
