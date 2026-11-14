"""Inventory & Sales RAG pipeline (Groq + Jina + FAISS), packaged.

Modules, one per pipeline stage:
- `config`      — settings, paths, API-key checks
- `ingest`      — load documents and split into chunks (stages 1-2)
- `embeddings`  — Jina-backed LangChain `Embeddings` (stage 3)
- `vectorstore` — FAISS index (stage 4)
- `llm`         — Groq chat model (stage 5)
- `tools`       — the `search_policy` retrieval tool (stage 6)
- `agent`       — the assistant built with `create_agent` (stage 7)
- `pipeline`    — assembly + ask/stream helpers used by both entry points
"""

from .pipeline import (
    Pipeline,
    ask_assistant,
    build_pipeline,
    stream_answer,
)

__all__ = ["Pipeline", "ask_assistant", "build_pipeline", "stream_answer"]
