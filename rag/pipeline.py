"""Assembles the full RAG pipeline: load -> chunk -> embed -> index -> agent.

This module ties all the stages together and exposes the two things entry
points (CLI and web UI) actually need:

- `build_pipeline()` — one call that returns a ready-to-use `Pipeline`
- `ask_assistant()` / `stream_answer()` — ask one question, get one answer
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph.state import CompiledStateGraph

from rag.agent import build_agent
from rag.config import JINA_MODEL, require_key
from rag.embeddings import JinaEmbeddings
from rag.ingest import chunk_documents, load_documents
from rag.llm import build_llm
from rag.tools import make_search_policy_tool
from rag.vectorstore import build_vector_store


@dataclass
class Pipeline:
    """Everything needed to answer questions about the policies."""

    documents: list[Document]
    chunks: list[Document]
    vector_store: FAISS
    llm: ChatGroq
    agent: CompiledStateGraph
    # One entry per `search_policy` call: the list of retrieved chunk texts.
    retrieval_log: list[list[str]] = field(default_factory=list)

    @property
    def last_sources(self) -> list[str]:
        """Chunk texts retrieved for the current question, deduplicated."""
        sources: list[str] = []
        for texts in self.retrieval_log:
            for text in texts:
                if text not in sources:
                    sources.append(text)
        return sources


def build_pipeline() -> Pipeline:
    """Build the complete pipeline. Raises `ConfigError` if API keys are missing."""
    groq_key = require_key("GROQ_API_KEY")
    jina_key = require_key("JINA_API_KEY")

    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings_model = JinaEmbeddings(api_key=jina_key)
    vector_store = build_vector_store(chunks, embeddings_model)
    llm = build_llm(api_key=groq_key)

    pipeline = Pipeline(
        documents=documents,
        chunks=chunks,
        vector_store=vector_store,
        llm=llm,
        agent=None,  # filled in below; the tool needs the pipeline's log
        retrieval_log=[],
    )
    search_policy = make_search_policy_tool(
        vector_store, retrieval_log=pipeline.retrieval_log
    )
    pipeline.agent = build_agent(llm, [search_policy])
    return pipeline


def ask_assistant(
    pipeline: Pipeline, question: str, history: list | None = None
) -> str:
    """Ask a single question and return the full answer.

    If `history` is given, it is used as prior conversation context
    (list of {'role', 'content'} dicts). The retrieval log is reset so
    `pipeline.last_sources` describes this question only.
    """
    pipeline.retrieval_log.clear()
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    response = pipeline.agent.invoke({"messages": messages})
    answer = response["messages"][-1].content
    return answer


def stream_answer(
    pipeline: Pipeline, question: str, history: list | None = None
) -> Iterator[str]:
    """Stream the agent's answer token by token.

    Yields text pieces. Streams only tokens from the agent's `model` node,
    skipping tool-result messages. If nothing streams (empty yield), the
    caller should fall back to `ask_assistant`.
    """
    pipeline.retrieval_log.clear()
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    for chunk, metadata in pipeline.agent.stream(
        {"messages": messages}, stream_mode="messages"
    ):
        if metadata.get("langgraph_node") == "tools":
            continue
        content = chunk.content
        if isinstance(content, str):
            if content:
                yield content
        elif isinstance(content, list):
            # Some models return a list of content blocks.
            for part in content:
                if isinstance(part, str):
                    yield part
                elif (
                    isinstance(part, dict)
                    and part.get("type", "text") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    yield part["text"]


# Re-exported so callers can show which embedding model is in use.
EMBEDDINGS_MODEL_NAME = JINA_MODEL
