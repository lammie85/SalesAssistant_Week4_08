"""Streamlit entry point — UI wiring only.

Run with:  streamlit run streamlit_app.py

Everything interesting lives in ``core/``. This file reads settings from the
sidebar, builds (or reuses) the index, and hands it to the tab renderers.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Acme Policy BOT",
    page_icon=":material/policy:",
    layout="centered",
)



# Import project modules dynamically so the workspace does not trigger static
# "missing import" warnings for namespace packages while still loading the same
# modules at runtime.
config_module = importlib.import_module("rag.config")
load_environment = config_module.load_env
groq_api_key = config_module.require_key("GROQ_API_KEY")
MODE_LABELS = config_module.MODE_LABELS
documents_module = importlib.import_module("core.documents")
load_documents = documents_module.load_documents
embeddings_module = importlib.import_module("core.embeddings")
get_embeddings = embeddings_module.get_embeddings
engine_module = importlib.import_module("core.engine")
PolicyEngine = engine_module.PolicyEngine
vectorstore_module = importlib.import_module("core.vectorstore")
build_vector_store = vectorstore_module.build_vector_store

about_module = importlib.import_module("ui.about")
render_about = about_module.render_about
chat_module = importlib.import_module("ui.chat")
render_chat = chat_module.render_chat
evaluation_module = importlib.import_module("ui.evaluation")
render_evaluation = evaluation_module.render_evaluation
inspector_module = importlib.import_module("ui.inspector")
render_inspector = inspector_module.render_inspector
sidebar_module = importlib.import_module("ui.sidebar")
render_index_stats = sidebar_module.render_index_stats
render_sidebar = sidebar_module.render_sidebar

load_environment()


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
# Cache the *index* only, keyed on the chunking settings. The engine itself is
# cheap to build (it holds no network state) so it is recreated per rerun —
# which keeps model/temperature changes instant instead of re-embedding 6
# chunks on every slider tick.
@st.cache_resource(max_entries=4, show_spinner=False)
def load_index(chunk_size: int, chunk_overlap: int):
    documents = load_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store = build_vector_store(documents, get_embeddings())
    return store, documents


def _hydrate_secrets():
    """Populate the process environment from Streamlit secrets if available."""
    try:
        value = st.secrets.get("GROQ_API_KEY")
    except Exception:
        return

    if value:
        os.environ["GROQ_API_KEY"] = str(value)


def groq_api_key():
    load_environment()
    _hydrate_secrets()

    try:
        value = st.secrets.get("GROQ_API_KEY")
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


_hydrate_secrets()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
config = render_sidebar()

st.title("Acme inventory & sales policy assistant", anchor=False)
st.caption(
    "Groq (LLM) · local MiniLM embeddings · FAISS · LangChain. "
    "Every answer is grounded in `data/policies.txt` — and the assistant is "
    "not allowed to approve refunds."
)

try:
    with st.spinner("Loading the embedding model and building the index…"):
        store, documents = load_index(config.chunk_size, config.chunk_overlap)
except Exception as exc:
    st.error(
        f"Could not build the policy index: `{type(exc).__name__}: {exc}`",
        icon=":material/database_off:",
    )
    st.stop()

engine = PolicyEngine(config, store, documents)

with st.sidebar:
    st.divider()
    render_index_stats(engine.stats, config)

chat_tab, inspector_tab, evaluation_tab, about_tab = st.tabs(
    ["Chat", "Retrieval inspector", "Evaluation", "How it works"],
    default="Chat",
)

with chat_tab:
    if groq_api_key():
        render_chat(engine)
    else:
        st.info(
            "Add a **Groq API key** in the sidebar to start chatting. "
            "It is free at console.groq.com.",
            icon=":material/key:",
        )
        st.markdown(
            "Meanwhile, the **Retrieval inspector** tab works without any API "
            "key — retrieval runs entirely on your machine."
        )
        st.caption(f"Current pipeline: {MODE_LABELS[config.mode]} · {config.describe()}")

with inspector_tab:
    render_inspector(engine, config)

with evaluation_tab:
    if groq_api_key():
        render_evaluation(engine, config)
    else:
        st.info("An API key is required to run the evaluation.", icon=":material/key:")

with about_tab:
    render_about(config)
