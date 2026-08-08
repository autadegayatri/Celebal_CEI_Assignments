"""
Entity & relation extraction for Knowledge Graph construction.

This uses a lightweight, dependency-free rule-based approach (capitalized
proper-noun span detection + a curated relation-phrase lexicon) rather than
a heavyweight NLP model such as spaCy's transformer pipelines. That keeps
the whole system installable in constrained / offline environments while
still demonstrating the full entity-extraction -> relationship-mapping ->
graph-storage pipeline described in the methodology.

To upgrade extraction quality in a real deployment, swap this module for a
spaCy NER + dependency-parse pipeline or an LLM-based extractor -- the
`Triple` output shape stays the same, so `graph_store.py` doesn't need to
change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered by specificity: longer / more specific phrases first so
# "is a subfield of" matches before the generic "is a".
_RELATION_PHRASES = [
    "is a subfield of",
    "is a subset of",
    "is part of",
    "is used for",
    "is used by",
    "is widely used for",
    "was developed by",
    "developed by",
    "was invented by",
    "invented by",
    "was introduced by",
    "introduced by",
    "was popularized by",
    "popularized by",
    "was pioneered by",
    "pioneered by",
    "was launched by",
    "launched by",
    "was founded by",
    "founded by",
    "was created by",
    "created by",
    "was coined by",
    "manufactured by",
    "operated by",
    "trained using",
    "runs on",
    "based on",
    "powers",
    "includes",
    "consists of",
    "uses",
    "requires",
    "replaced",
    "is a",
    "is an",
]

# A capitalized-span regex: one or more capitalized "words" (allowing
# internal lowercase connector words like "of"/"the" and digits/acronyms).
_ENTITY_TOKEN = r"[A-Z][A-Za-z0-9\-]*(?:\s+(?:of|the|for|and)\s+[A-Z][A-Za-z0-9\-]*|\s+[A-Z][A-Za-z0-9\-]*)*"
_ENTITY_RE = re.compile(_ENTITY_TOKEN)

_STOPWORD_ENTITIES = {
    "The",
    "This",
    "It",
    "They",
    "These",
    "Those",
    "A",
    "An",
    "Who",
    "What",
    "When",
    "Where",
    "Why",
    "How",
    "Do",
    "Does",
    "Did",
    "Is",
    "Are",
    "Was",
    "Were",
    "Can",
    "Could",
    "Would",
    "Should",
    "Tell",
    "Please",
}


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    obj: str
    sentence: str


def extract_entities(sentence: str) -> list[str]:
    """Return candidate proper-noun entity spans found in a sentence."""
    candidates = _ENTITY_RE.findall(sentence)
    entities = []
    for c in candidates:
        c = c.strip().rstrip(".,;:")
        if c and c not in _STOPWORD_ENTITIES and len(c) > 1:
            entities.append(c)
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def extract_triples(sentence: str) -> list[Triple]:
    """
    Extract (subject, relation, object) triples from a sentence by matching
    a known relation-phrase lexicon between two capitalized entity spans.
    """
    triples: list[Triple] = []
    lower = sentence.lower()

    for phrase in _RELATION_PHRASES:
        start = 0
        while True:
            idx = lower.find(f" {phrase} ", start)
            if idx == -1:
                break
            left_text = sentence[:idx]
            right_text = sentence[idx + len(phrase) + 2 :]

            left_entities = extract_entities(left_text)
            right_entities = extract_entities(right_text)

            if left_entities and right_entities:
                subj = left_entities[-1]  # entity closest to the relation phrase
                obj = right_entities[0]
                if subj.lower() != obj.lower():
                    triples.append(Triple(subject=subj, relation=phrase, obj=obj, sentence=sentence.strip()))
            start = idx + len(phrase)
    return triples


def extract_from_text(text: str) -> tuple[list[str], list[Triple]]:
    """Run extraction over a full (multi-sentence) text blob."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    all_entities: list[str] = []
    all_triples: list[Triple] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        all_entities.extend(extract_entities(sentence))
        all_triples.extend(extract_triples(sentence))

    seen = set()
    unique_entities = []
    for e in all_entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)

    return unique_entities, all_triples
