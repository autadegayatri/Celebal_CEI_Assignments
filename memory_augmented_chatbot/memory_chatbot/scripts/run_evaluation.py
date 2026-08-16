"""
Runs the evaluation framework over a sample QA test set (grounded RAG/KG/
tool questions plus a couple of open-domain "general" questions) and
writes a JSON report to artifacts/eval_reports/.

Usage:
    python -m scripts.run_evaluation
"""

from __future__ import annotations

import time

from memory_augmented_chatbot.memory_chatbot.src import config
from memory_augmented_chatbot.memory_chatbot.src.evaluation.evaluator import Evaluator, TestCase
from memory_augmented_chatbot.memory_chatbot.src.knowledge_graph.graph_store import get_graph_store
from memory_augmented_chatbot.memory_chatbot.src.memory.memory_store import get_memory_store
from memory_augmented_chatbot.memory_chatbot.src.orchestration.graph import ChatbotGraph
from memory_augmented_chatbot.memory_chatbot.src.rag.retriever import Retriever

SAMPLE_TEST_CASES = [
    TestCase(
        query="What is Retrieval-Augmented Generation?",
        reference_answer="RAG combines retrieval of relevant documents with a language model to generate grounded answers.",
    ),
    TestCase(query="Who developed the Transformer architecture used in LLMs?"),
    TestCase(query="What is a Knowledge Graph used for?"),
    TestCase(query="What is the current date?"),
    TestCase(query="Calculate 25 * 4 + 10"),
    # Open-domain question the corpus doesn't cover -- exercises the general route.
    TestCase(query="What's a good way to explain recursion to a beginner?"),
]


def main():
    retriever = Retriever.load()
    graph_store = get_graph_store()
    memory_store = get_memory_store()

    chatbot = ChatbotGraph(retriever=retriever, graph_store=graph_store, memory_store=memory_store)
    evaluator = Evaluator(chatbot)

    print(f"Running evaluation over {len(SAMPLE_TEST_CASES)} test cases...\n")
    report = evaluator.evaluate_batch(SAMPLE_TEST_CASES)

    for r in report.results:
        print(f"Q: {r.query}")
        print(
            f"  route={r.route}  relevance={r.context_relevance}  faithfulness={r.faithfulness}  "
            f"correctness={r.answer_correctness}  latency={r.latency_seconds}s"
        )
        print(f"  A: {r.answer[:200]}{'...' if len(r.answer) > 200 else ''}\n")

    summary = report.summary()
    print("=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    report_path = config.EVAL_REPORTS_DIR / f"eval_report_{int(time.time())}.json"
    report.save(report_path)
    print(f"\nFull report saved to {report_path}")

    graph_store.close()
    memory_store.close()


if __name__ == "__main__":
    main()
