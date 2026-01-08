# LangGraph RAG Bot

Minimal Retrieval-Augmented Generation (RAG) chatbot built on the LangChain 1.x + LangGraph 1.x stack.

中文说明见 `README.zh-CN.md`.

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
python src/rag/ingest.py
```
This writes embeddings into a Weaviate class (default `RAGChunk`).

Notes:
- Markdown files are first split by headings (H1/H2/H3), then chunked with a recursive splitter.

### Optional: local embeddings
```bash
export USE_LOCAL_EMBEDDER=1
export LOCAL_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
python src/rag/ingest.py
```

## Run the chat loop
```bash
python src/rag/rag_graph.py
```
Ask questions; answers are grounded on the ingested context. If the answer is outside the context, the bot will say it does not know.

## Run the web UI (Streamlit)
```bash
streamlit run app.py
```

## Configuration
- Use `.env` (see `.env.example`).
- LLM:
  - `LLM_PROVIDER=deepseek` with `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL`, `DEEPSEEK_TEMPERATURE`
  - or `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, optional `OPENAI_MODEL`, `OPENAI_TEMPERATURE`
- Embeddings (retrieval, API-only):
  - `EMBED_PROVIDER=dashscope` with `DASHSCOPE_API_KEY`, optional `DASHSCOPE_EMBED_MODEL`
  - or `EMBED_PROVIDER=openai` with `OPENAI_API_KEY`, `OPENAI_EMBED_MODEL`
- Vector DB:
  - `WEAVIATE_URL`, optional `WEAVIATE_API_KEY`, `WEAVIATE_CLASS`, `WEAVIATE_REBUILD`

## What’s inside
- `src/rag/ingest.py`: loads files from `data/`, chunks them, and builds a Chroma vector store.
- `src/rag/rag_graph.py`: LangGraph pipeline (retrieve → generate) with ChatOpenAI and the persisted retriever.
