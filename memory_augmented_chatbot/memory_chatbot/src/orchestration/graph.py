"""
Builds the LangGraph `StateGraph` that orchestrates the whole system.

Flow:

    START -> memory -> router -> [rag_node | kg_node | tool_node] -> build_prompt -> generate -> memory_write -> END

    The router picks exactly one of rag_node / kg_node / tool_node per turn.

The router (a lightweight rule-based "model" node here, easily swappable
for an LLM-based classifier) decides per-query whether static knowledge
(RAG), structured relationships (Knowledge Graph), or a real-time tool call
best answers the question -- this is the "dynamically fetch real-time
information via tools and APIs" requirement from the problem statement.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from src.knowledge_graph.graph_store import GraphStore, get_graph_store
from src.llm.client import LLMClient, get_llm_client
from src.memory.memory_store import BaseMemoryStore, get_memory_store
from src.orchestration import nodes
from src.orchestration.state import ChatState
from src.rag.retriever import Retriever


class ChatbotGraph:
    """
    Thin wrapper that owns the shared resources (retriever, graph store,
    memory store, LLM client) and compiles a LangGraph app bound to them.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        graph_store: GraphStore | None = None,
        memory_store: BaseMemoryStore | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.graph_store = graph_store or get_graph_store()
        self.memory_store = memory_store or get_memory_store()
        self.llm_client = llm_client or get_llm_client()
        self.app = self._build()

    def _build(self):
        graph = StateGraph(ChatState)

        graph.add_node("memory", partial(nodes.memory_node, memory_store=self.memory_store))
        graph.add_node("router", nodes.router_node)
        graph.add_node("rag", partial(nodes.rag_node, retriever=self.retriever))
        graph.add_node("kg", partial(nodes.kg_node, graph_store=self.graph_store))
        graph.add_node("tool", nodes.tool_node)
        graph.add_node("build_prompt", nodes.build_prompt_node)
        graph.add_node("generate", partial(nodes.generation_node, llm_client=self.llm_client))
        graph.add_node("memory_write", partial(nodes.memory_write_node, memory_store=self.memory_store))

        graph.add_edge(START, "memory")
        graph.add_edge("memory", "router")

        graph.add_conditional_edges(
            "router",
            nodes.route_decision,
            {"rag": "rag", "kg": "kg", "tool": "tool"},
        )

        graph.add_edge("rag", "build_prompt")
        graph.add_edge("kg", "build_prompt")
        graph.add_edge("tool", "build_prompt")

        graph.add_edge("build_prompt", "generate")
        graph.add_edge("generate", "memory_write")
        graph.add_edge("memory_write", END)

        return graph.compile()

    def chat(self, user_id: str, query: str) -> ChatState:
        initial_state: ChatState = {"user_id": user_id, "query": query}
        return self.app.invoke(initial_state)

    def save_artifacts(self):
        self.retriever.save()
        if hasattr(self.graph_store, "save"):
            self.graph_store.save()
