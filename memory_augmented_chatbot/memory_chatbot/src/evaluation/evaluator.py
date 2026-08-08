"""
Evaluation framework.

Implements the three metrics called for in the problem statement:

  - Context Relevance: how relevant is the retrieved context to the query?
    (cosine similarity between query embedding and retrieved-context
    embedding, using the same embedder as the RAG layer)
  - Faithfulness: is the generated answer actually grounded in the
    retrieved context, or does it hallucinate beyond it? (token-overlap /
    entailment-style heuristic between answer and context)
  - Answer Correctness: how close is the generated answer to a reference
    ("gold") answer, when one is available (token-overlap F1)

Also tracks latency per turn. Runs over a batch of test cases and produces
a JSON report plus an aggregate summary printed to stdout -- this is the
"evaluation framework to measure response quality" deliverable.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.rag.embeddings import Embedder, get_embedder


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 1}


def token_overlap_f1(a: str, b: str) -> float:
    """Symmetric token-overlap F1 between two texts -- a lightweight,
    dependency-free stand-in for semantic-similarity/entailment scoring."""
    tokens_a, tokens_b = _tokenize(a), _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    precision = len(overlap) / len(tokens_a)
    recall = len(overlap) / len(tokens_b)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class TestCase:
    query: str
    reference_answer: str | None = None
    user_id: str = "eval_user"


@dataclass
class EvalResult:
    query: str
    answer: str
    route: str
    context_relevance: float
    faithfulness: float
    answer_correctness: float | None
    latency_seconds: float
    reference_answer: str | None = None


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.results:
            return {}
        n = len(self.results)
        correctness_scores = [r.answer_correctness for r in self.results if r.answer_correctness is not None]
        return {
            "num_cases": n,
            "avg_context_relevance": round(sum(r.context_relevance for r in self.results) / n, 4),
            "avg_faithfulness": round(sum(r.faithfulness for r in self.results) / n, 4),
            "avg_answer_correctness": round(sum(correctness_scores) / len(correctness_scores), 4)
            if correctness_scores
            else None,
            "avg_latency_seconds": round(sum(r.latency_seconds for r in self.results) / n, 4),
            "route_distribution": {
                route: sum(1 for r in self.results if r.route == route)
                for route in {r.route for r in self.results}
            },
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "results": [asdict(r) for r in self.results]}

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class Evaluator:
    def __init__(self, chatbot_graph, embedder: Embedder | None = None):
        """
        chatbot_graph: an instance of src.orchestration.graph.ChatbotGraph
        """
        self.chatbot_graph = chatbot_graph
        # a fresh embedder fit on-the-fly per evaluation for context-relevance
        # scoring (independent of whatever backend the RAG layer itself uses)
        self.embedder = embedder or get_embedder()

    def _context_relevance(self, query: str, context: str) -> float:
        if not context.strip():
            return 0.0
        try:
            self.embedder.fit([query, context])
            vecs = self.embedder.transform([query, context])
            return float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])
        except Exception:
            return token_overlap_f1(query, context)

    def evaluate_case(self, case: TestCase) -> EvalResult:
        start = time.perf_counter()
        state = self.chatbot_graph.chat(case.user_id, case.query)
        latency = time.perf_counter() - start

        context = "\n".join(
            filter(
                None,
                [state.get("rag_context", ""), state.get("kg_context", ""), state.get("tool_output", "")],
            )
        )
        answer = state.get("answer", "")

        relevance = self._context_relevance(case.query, context)
        faithfulness = token_overlap_f1(answer, context) if context else 0.0
        correctness = token_overlap_f1(answer, case.reference_answer) if case.reference_answer else None

        return EvalResult(
            query=case.query,
            answer=answer,
            route=state.get("route", "unknown"),
            context_relevance=round(relevance, 4),
            faithfulness=round(faithfulness, 4),
            answer_correctness=round(correctness, 4) if correctness is not None else None,
            latency_seconds=round(latency, 4),
            reference_answer=case.reference_answer,
        )

    def evaluate_batch(self, cases: list[TestCase]) -> EvalReport:
        report = EvalReport()
        for case in cases:
            report.results.append(self.evaluate_case(case))
        return report
