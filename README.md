# LangGraph RAG Bot

Minimal Retrieval-Augmented Generation (RAG) chatbot built on the LangChain 1.x + LangGraph 1.x stack.

中文说明见 `README.zh-CN.md`，新手上手指南见 `docs/PROJECT_GUIDE.zh-CN.md`.

## Flowchart (Mermaid)

```mermaid
flowchart TD
    A[Docs in data/] --> B["services/ingest/processor.py<br>Markdown header split + recursive chunks"]
    B --> C[ModelScope Embedding API]
    C --> D[Weaviate Vector DB RAGChunk]

    E[User Question] --> F["workflows/rag_bot/graph.py<br>retriever.invoke()"]
    F --> D
    F --> G[Context + Prompt]
    G --> H[DeepSeek LLM API]
    H --> I[Answer]

    J[Streamlit UI] --> E
    J --> B
```

## Prerequisites
- Python 3.10+
- OpenAI API key in environment variable `OPENAI_API_KEY`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ingest your knowledge base
1) Drop `.md` or `.txt` files into `data/`.
2) Build the vector store:
```bash
PYTHONPATH=src python -m services.ingest.processor
```
This writes embeddings into a Weaviate class (default `RAGChunk`).

Notes:
- Markdown files are first split by headings (H1/H2/H3), then chunked with a recursive splitter.

## Run the chat loop
```bash
PYTHONPATH=src python -m workflows.rag_bot.graph
```
Ask questions; answers are grounded on the ingested context. If the answer is outside the context, the bot will say it does not know.

## Run the web UI (Streamlit)
```bash
streamlit run app.py
```
If you see `streamlit: command not found`, activate the venv first (`source .venv/bin/activate`) or run:
```bash
.venv/bin/streamlit run app.py
```

## Run Agent Chat UI (LangGraph-compatible)
This repo includes a minimal LangGraph-compatible HTTP API (for Agent Chat UI) at `http://localhost:2024`.

1) Start the API server:
```bash
./scripts/run_langgraph_api.sh
```

2) Create an API key (token) by logging in:
```bash
curl -sS -X POST http://localhost:2024/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"YOUR_USER","password":"YOUR_PASS"}'
```
Copy the returned `api_key`.

3) Run Agent Chat UI (separate Next.js app) and set:
- `Deployment URL`: `http://localhost:2024`
- `Assistant / Graph ID`: `agent`
- `LangSmith API Key`: paste the `api_key` from step 2 (we use it as `x-api-key`)

## Configuration
- Use `.env` (see `.env.example`).
- LLM:
  - `LLM_PROVIDER=deepseek` with `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL`, `DEEPSEEK_TEMPERATURE`
  - or `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, optional `OPENAI_MODEL`, `OPENAI_TEMPERATURE`
- Embeddings (retrieval, API-only):
  - `EMBED_PROVIDER=modelscope` with `MODELSCOPE_API_TOKEN`, `MODELSCOPE_BASE_URL`, `MODELSCOPE_EMBED_MODEL`
  - or `EMBED_PROVIDER=openai` with `OPENAI_API_KEY`, `OPENAI_EMBED_MODEL`
- Vector DB:
  - `WEAVIATE_URL`, optional `WEAVIATE_API_KEY`, `WEAVIATE_CLASS`, `WEAVIATE_REBUILD`
- Database (PostgreSQL):
  - `DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/chatbot_rag`
  - Run schema migration: `alembic upgrade head`
  - Auto-create tables (dev only): `APP_AUTO_CREATE_SCHEMA=1`
  - Optional pool tuning: `APP_DB_POOL_SIZE`, `APP_DB_MAX_OVERFLOW`, `APP_DB_POOL_RECYCLE`
  - Department bootstrap: `APP_DEFAULT_DEPARTMENT` (default `default`)

## What’s inside
- `src/services/ingest/processor.py`: loads files from `data/`, chunks them, and writes to Weaviate.
- `src/workflows/rag_bot/graph.py`: LangGraph pipeline (retrieve → generate) with ChatOpenAI and Weaviate retriever.
