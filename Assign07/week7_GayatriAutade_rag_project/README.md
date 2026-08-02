# Document Question Answering System (RAG)

A Retrieval-Augmented Generation pipeline that answers questions grounded in your own PDFs or text files, instead of relying only on a language model's internal knowledge.

## How it works (matches the 7-stage pipeline in the project brief)

1. **Document Ingestion** — `src/loader.py` extracts raw text from PDF/txt/md files.
2. **Text Chunking** — `src/chunker.py` splits the text into overlapping, sentence-aware chunks.
3. **Embedding Creation** — `src/embeddings.py` turns each chunk into a vector using a local `sentence-transformers` model (free, no API key).
4. **Vector Database** — `src/vectorstore.py` stores the vectors in a FAISS index for fast similarity search.
5. **Query Processing** — the user's question is embedded with the same model.
6. **Context Retrieval** — the top-k most similar chunks are pulled from FAISS.
7. **Answer Generation** — `src/generator.py` sends the question + retrieved chunks to an LLM to produce a grounded answer.

`src/pipeline.py` wires all of this together into a single `RAGPipeline` class used by both the Streamlit app and the CLI.

## Project structure

```
rag_project/
├── app.py                 # Streamlit UI
├── cli.py                 # Command-line interface for quick testing
├── requirements.txt
├── src/
│   ├── loader.py           # Stage 1: Document Ingestion
│   ├── chunker.py          # Stage 2: Text Chunking
│   ├── embeddings.py       # Stage 3 & 5: Embedding creation
│   ├── vectorstore.py      # Stage 4 & 6: Vector DB + retrieval
│   ├── generator.py        # Stage 7: Answer generation (pluggable backend)
│   └── pipeline.py         # Orchestrates all stages
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You don't need every dependency in `requirements.txt` — install only what your chosen generation backend needs (see below).

## Choosing a generation backend

Retrieval (embeddings + FAISS) is always free and local. For the final answer-generation step, pick one:

| Backend      | Needs               | Install                          |
|--------------|---------------------|-----------------------------------|
| `local`      | nothing (default)   | `pip install transformers torch` |
| `anthropic`  | `ANTHROPIC_API_KEY` | `pip install anthropic`          |
| `openai`     | `OPENAI_API_KEY`    | `pip install openai`             |
| `cohere`     | `COHERE_API_KEY`    | `pip install cohere`             |

Set the API key as an environment variable, e.g.:

```bash
export ANTHROPIC_API_KEY=your_key_here
export BACKEND=anthropic
```

(Or put these in a `.env` file — `python-dotenv` loads it automatically for the Streamlit app.)

## Running the app

**Streamlit UI** (recommended — lets you pick the backend and upload files from the browser):

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, upload a PDF, click "Process document", and start asking questions.

**Command line:**

```bash
export BACKEND=local   # or anthropic / openai / cohere
python cli.py path/to/your_document.pdf
```

## Notes on the `local` backend

It uses `google/flan-t5-base` via Hugging Face `transformers`, which downloads automatically the first time you run it and works fully offline afterward — good for testing without any API keys. Answer quality is noticeably lower than the hosted models, so switch to `anthropic`/`openai`/`cohere` for better results once you have a key.

## Ideas for extending this project (from the brief)

- Swap in hybrid search (BM25 keyword + vector) instead of pure vector similarity.
- Add a re-ranking step (e.g. a cross-encoder) after initial retrieval.
- Try different embedding models (`all-mpnet-base-v2` for higher quality, at the cost of speed).
- Support multiple documents at once with per-source filtering.
- Add chat-style memory so follow-up questions can reference earlier ones.
