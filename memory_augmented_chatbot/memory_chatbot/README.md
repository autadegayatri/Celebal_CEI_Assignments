# Memory-Augmented Chatbot

A chatbot system that combines **Retrieval-Augmented Generation (RAG)**, a
**Knowledge Graph**, and **long-term user memory** to deliver context-aware,
personalized responses — orchestrated with **LangGraph**, with a built-in
**evaluation framework** to measure response quality.

This implements the full architecture end-to-end and runs completely
offline out of the box (no API keys, no external services required), while
every component is built to swap in production-grade backends
(sentence-transformer embeddings, Neo4j, the real Claude API, live web
search) via config flags with no code changes elsewhere.

---

## Architecture

```
                                   ┌─────────────────┐
                                   │   User Query     │
                                   └────────┬─────────┘
                                            │
                                    ┌───────▼────────┐
                                    │  Memory Node    │  loads long-term facts
                                    │                 │  + recent conversation
                                    └───────┬────────┘
                                            │
                                    ┌───────▼────────┐
                                    │  Router Node    │  decides: RAG / KG / Tool
                                    └───┬───────┬────┘
                       ┌────────────────┘       └────────────────┐
                       │                                          │
              ┌────────▼────────┐                        ┌───────▼────────┐
              │                 │                        │                │
      ┌───────▼──────┐  ┌───────▼──────┐          ┌──────▼───────┐        │
      │   RAG Node    │  │   KG Node    │          │  Tool Node    │       │
      │ (vector search│  │ (graph query)│          │ (datetime,    │       │
      │  over chunks) │  │              │          │  calculator,  │       │
      └───────┬──────┘  └───────┬──────┘          │  web search)  │       │
              │                 │                  └──────┬───────┘       │
              └────────┬────────┴──────────────────────────┘              │
                       │                                                  │
               ┌───────▼────────┐                                        │
               │ Build Prompt   │◄───────────────────────────────────────┘
               │  (merge all    │
               │   context)     │
               └───────┬────────┘
                       │
               ┌───────▼────────┐
               │ Generation     │  LLM (Claude API or offline mock)
               │     Node       │
               └───────┬────────┘
                       │
               ┌───────▼────────┐
               │ Memory Write    │  persists turn + extracted facts
               │     Node        │
               └───────┬────────┘
                       │
                 ┌─────▼─────┐
                 │  Answer    │
                 └───────────┘
```

**Static Knowledge Layer (RAG):** web scraper (`requests` + `BeautifulSoup`)
→ cleaning → sentence-aware chunking → TF-IDF (or optional
sentence-transformer) embeddings → in-memory cosine-similarity vector store.

**Knowledge Graph Layer:** rule-based entity + relation ("triple")
extraction → stored in a `networkx` directed multigraph by default, or a
real **Neo4j** instance if configured — same interface either way.

**Dynamic Intelligence Layer (LangGraph):** a `StateGraph` with Memory,
Router, RAG, KG, Tool, Generation, and Memory-Write nodes. The router
inspects each query and picks RAG vs. KG vs. a real-time Tool
(date/time, calculator, pluggable web search).

**Long-Term Memory:** persistent store of durable user facts ("I'm a
final-year IT student", "I like machine learning") extracted from
conversation and re-injected into future turns, plus a rolling window of
recent conversation history. Backed by **SQLite** by default (zero setup,
`artifacts/memory.db`), or a real **PostgreSQL** client/server connection
(`MEMORY_BACKEND=postgres`) — matching the problem statement's tech stack —
with no code changes needed anywhere else in the system. Memory survives
process restarts either way; `GET /memory/{user_id}` and
`DELETE /memory/{user_id}` let you inspect or wipe a user's memory.

**Evaluation Framework:** measures **context relevance** (query↔context
cosine similarity), **faithfulness** (answer↔context grounding), and
**answer correctness** (answer↔reference-answer overlap) per turn, plus
latency, with an aggregate JSON report.

---

## Project structure

```
memory_chatbot/
├── src/
│   ├── config.py                  # all paths & tunables in one place
│   ├── scraping/scraper.py        # requests+BeautifulSoup web scraper (+ offline corpus loader)
│   ├── ingestion/
│   │   ├── cleaner.py             # text cleaning
│   │   └── chunker.py             # sentence-aware overlapping chunking
│   ├── rag/
│   │   ├── embeddings.py          # TF-IDF (default) / sentence-transformers backends
│   │   ├── vector_store.py        # cosine-similarity vector index
│   │   └── retriever.py           # embedder + vector store, save/load
│   ├── knowledge_graph/
│   │   ├── entity_extractor.py    # rule-based entity/relation ("triple") extraction
│   │   ├── graph_store.py         # networkx (default) / Neo4j backend
│   │   └── graph_query.py         # NL query -> graph lookup helpers
│   ├── memory/memory_store.py     # SQLite long-term facts + conversation history
│   ├── tools/tool_registry.py     # datetime, calculator, web_search (pluggable)
│   ├── llm/client.py              # Anthropic client / offline mock generator
│   ├── orchestration/
│   │   ├── state.py               # LangGraph shared state schema
│   │   ├── nodes.py               # node functions (memory/router/rag/kg/tool/generate)
│   │   └── graph.py               # StateGraph wiring -> ChatbotGraph
│   └── evaluation/evaluator.py    # context relevance / faithfulness / correctness metrics
├── api/main.py                    # FastAPI app: /chat /memory /kb/build /evaluate
├── ui/app.py                      # Streamlit chat UI
├── scripts/
│   ├── build_knowledge_base.py    # run the full ingestion + KG-build pipeline
│   └── run_evaluation.py          # run the evaluation framework, save a report
├── data/sample_corpus/            # offline demo corpus (stand-in for scraped pages)
├── tests/test_pipeline.py         # unit tests across every layer
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # optional -- defaults work out of the box
```

## Quickstart

**1. Build the knowledge base** (RAG index + knowledge graph) from the
bundled sample corpus (swap in real URLs any time with `--urls`):

```bash
python -m scripts.build_knowledge_base
# or, to scrape live pages instead of the sample corpus:
python -m scripts.build_knowledge_base --urls https://example.com/article1 https://example.com/article2
```

**2. Chat via the Streamlit UI:**

```bash
streamlit run ui/app.py
```

**3. Or run the API server:**

```bash
uvicorn api.main:app --reload
```

## Running with real databases (PostgreSQL + Neo4j)

The project supports running the long-term memory in PostgreSQL and the
knowledge graph in Neo4j. Use the Docker Compose setup to bring up both
services locally, then set environment variables (or copy `.env.example` to
`.env`) to point the app at the services.

1. Start the databases:

```bash
docker compose up -d
```

2. Copy the example environment file and (optionally) tweak credentials:

```bash
cp .env.example .env
```

3. Install Postgres driver (if not installed):

```bash
pip install psycopg2-binary neo4j
```

4. Import your scraped data or rebuild the knowledge base (this will write
        the vector store and populate the Neo4j graph):

```bash
# scrape (if you haven't already)
python -m scripts.run_scraper --urls data/source_urls.txt --out artifacts/scraped.jsonl

# import into the vector store and Neo4j
python -m scripts.import_scraped --in artifacts/scraped.jsonl
```

5. Run the UI or API as usual; the system will use Postgres/Neo4j when
        `MEMORY_BACKEND=postgres` and `GRAPH_BACKEND=neo4j` are set in the
        environment (see `.env.example`).


```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "gayatri", "message": "What frameworks are used for Machine Learning?"}'

curl http://localhost:8000/memory/gayatri
```

**4. Run the evaluation framework:**

```bash
python -m scripts.run_evaluation
```

This prints per-query metrics and an aggregate summary, and saves a full
JSON report to `artifacts/eval_reports/`.

**5. Run tests:**

```bash
python -m pytest tests/ -v
```

---

## Design decisions & how to upgrade each layer

Every "heavy" dependency has a lightweight default so the whole system
installs and runs **anywhere**, plus a documented one-line swap to the
production-grade version:

| Layer | Default (ships as-is) | Production upgrade |
|---|---|---|
| Embeddings | TF-IDF (scikit-learn, no downloads) | `EMBEDDING_BACKEND=sentence-transformers` in `.env` |
| Vector store | In-memory cosine-similarity index | Swap `VectorStore` for a FAISS/Chroma-backed class (same interface) |
| Knowledge graph | `networkx` in-process graph | `GRAPH_BACKEND=neo4j` + running Neo4j instance |
| LLM | Deterministic offline extractive generator | `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` |
| Entity extraction | Rule-based capitalized-span + relation-phrase lexicon | Swap `entity_extractor.py` for a spaCy NER + dependency-parse pipeline |
| Web search tool | Disabled stub | `WEB_SEARCH_ENABLED=true` + plug in a real search API (SerpAPI/Tavily/Bing) |
| Memory storage | SQLite | Point `MemoryStore` at MongoDB/PostgreSQL instead |

This mirrors a real engineering tradeoff: ship something that **runs and is
gradeable everywhere on day one**, with every seam already in place to
swap in the heavier, more powerful component later without touching
callers.

## Known limitations (by design, given the above tradeoffs)

- The offline **mock LLM** answers extractively (it selects and stitches
  together the most relevant retrieved sentences) rather than truly
  generating novel prose — set `LLM_PROVIDER=anthropic` for real generation.
- The **TF-IDF** retriever matches on lexical overlap, not deep semantic
  meaning — set `EMBEDDING_BACKEND=sentence-transformers` for semantic
  retrieval on larger/less keyword-overlapping corpora.
- The **rule-based entity extractor** catches common "X was developed by Y"
  style sentence patterns well but will miss more complex relationships —
  a spaCy or LLM-based extractor will generalize further.
- The bundled **sample corpus** (`data/sample_corpus/`) stands in for
  scraped web pages so the pipeline is fully reproducible offline; pass
  `--urls` to `build_knowledge_base.py` to scrape real pages instead.
