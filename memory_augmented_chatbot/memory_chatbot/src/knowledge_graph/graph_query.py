"""
High-level query helpers over a Neo4jGraphStore, used by the LangGraph KG node
to turn a natural-language query into a formatted context string.
"""

from __future__ import annotations

import re

from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_store import Neo4jGraphStore


def format_neighbors(neighbors: list[dict]) -> str:
    if not neighbors:
        return ""
    lines = []
    for n in neighbors:
        lines.append(f"{n['subject']} --[{n['relation']}]--> {n['object']}")
    return "\n".join(lines)


def query_graph_for_entity(store: Neo4jGraphStore, entity_query: str, max_hops: int = 2) -> str:
    """
    Resolve an entity mention (possibly fuzzy/partial) to graph nodes and
    return a formatted string of the relationships found around it.
    """
    matches = store.search_entities(entity_query, limit=3)
    if not matches:
        return ""

    # Prefer an exact entity match when available, then fall back to partials.
    exact_matches = [m for m in matches if m.lower() == entity_query.lower()]
    candidate_entities = exact_matches or matches

    all_neighbors: list[dict] = []
    for entity in candidate_entities:
        all_neighbors.extend(store.neighbors(entity, max_hops=max_hops))

    # de-duplicate
    seen = set()
    unique = []
    for n in all_neighbors:
        key = (n["subject"], n["relation"], n["object"])
        if key not in seen:
            seen.add(key)
            unique.append(n)

    return format_neighbors(unique[:20])


def extract_query_entities(query: str) -> list[str]:
    """Pull the most relevant entity candidates out of a user query for KG lookup."""
    from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.entity_extractor import extract_entities

    entities = extract_entities(query)
    if entities:
        return entities

    # Fall back to simple token-based extraction for queries that do not use
    # capitalization, while avoiding common question words and punctuation.
    words = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query):
        cleaned = token.strip("?.,!:")
        if not cleaned:
            continue
        if cleaned.lower() in {"who", "what", "when", "where", "why", "how", "do", "does", "did", "is", "are", "was", "were", "can", "could", "would", "should", "tell", "please"}:
            continue
        if cleaned[0].isupper() or any(c.isupper() for c in cleaned):
            words.append(cleaned)

    if words:
        return words

    tokens = [token.strip("?.,!:") for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]*", query) if token.strip("?.,!:")]
    return [max(tokens, key=len)] if tokens else []
