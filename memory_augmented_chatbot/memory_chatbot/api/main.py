"""
FastAPI application exposing the Memory-Augmented Chatbot over HTTP.

Endpoints:
    POST /chat                 - send a message, get a grounded, personalized reply
    GET  /memory/{user_id}     - inspect a user's stored long-term facts
    DELETE /memory/{user_id}   - forget a user's memory (facts + history)
    POST /kb/build              - (re)build the RAG index + knowledge graph from the corpus
    POST /evaluate               - run the evaluation framework over a test set
    GET  /health                 - liveness check
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config
from src.evaluation.evaluator import EvalReport, Evaluator, TestCase
from src.knowledge_graph.graph_store import NetworkXGraphStore
from src.memory.memory_store import BaseMemoryStore, get_memory_store
from src.orchestration.graph import ChatbotGraph
from src.rag.retriever import Retriever

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load previously built artifacts; fall back to an empty index
    # (callers should hit /kb/build first) so the API still boots cleanly.
    try:
        retriever = Retriever.load()
    except FileNotFoundError:
        retriever = Retriever()

    graph_store = NetworkXGraphStore()
    if config.GRAPH_STORE_PATH.exists():
        graph_store.load()

    memory_store = get_memory_store()
    chatbot = ChatbotGraph(retriever=retriever, graph_store=graph_store, memory_store=memory_store)

    _state["chatbot"] = chatbot
    _state["memory_store"] = memory_store
    yield
    memory_store.close()


app = FastAPI(
    title="Memory-Augmented Chatbot API",
    description=(
        "RAG + Knowledge Graph + Long-Term Memory chatbot, orchestrated with "
        "LangGraph, with a built-in evaluation framework."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    used_memory_facts: str
    used_rag_context: str
    used_kg_context: str
    used_tool_output: str


class TestCaseModel(BaseModel):
    query: str
    reference_answer: str | None = None
    user_id: str = "eval_user"


class EvalRequest(BaseModel):
    cases: list[TestCaseModel]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "memory_backend": config.MEMORY_BACKEND,
        "graph_backend": config.GRAPH_BACKEND,
        "llm_provider": config.LLM_PROVIDER,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chatbot: ChatbotGraph = _state["chatbot"]
    result = chatbot.chat(req.user_id, req.message)
    return ChatResponse(
        answer=result.get("answer", ""),
        route=result.get("route", ""),
        used_memory_facts=result.get("memory_facts", ""),
        used_rag_context=result.get("rag_context", ""),
        used_kg_context=result.get("kg_context", ""),
        used_tool_output=result.get("tool_output", ""),
    )


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    memory_store: BaseMemoryStore = _state["memory_store"]
    facts = memory_store.get_facts(user_id, limit=100)
    turns = memory_store.get_recent_turns(user_id, limit=50)
    return {
        "facts": [f.fact for f in facts],
        "recent_turns": [{"role": t.role, "content": t.content} for t in turns],
    }


@app.delete("/memory/{user_id}")
def delete_memory(user_id: str):
    memory_store: BaseMemoryStore = _state["memory_store"]
    memory_store.forget(user_id)
    return {"status": "forgotten", "user_id": user_id}


@app.post("/kb/build")
def build_kb():
    from scripts.build_knowledge_base import build

    stats = build()
    # reload the freshly built artifacts into the running chatbot
    chatbot: ChatbotGraph = _state["chatbot"]
    chatbot.retriever = Retriever.load()
    if hasattr(chatbot.graph_store, "load"):
        chatbot.graph_store.load()
    return stats


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    if not req.cases:
        raise HTTPException(status_code=400, detail="Provide at least one test case.")
    chatbot: ChatbotGraph = _state["chatbot"]
    evaluator = Evaluator(chatbot)
    cases = [TestCase(query=c.query, reference_answer=c.reference_answer, user_id=c.user_id) for c in req.cases]
    report: EvalReport = evaluator.evaluate_batch(cases)
    return report.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
