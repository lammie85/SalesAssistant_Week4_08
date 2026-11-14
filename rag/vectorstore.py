"""STAGE 4: the FAISS vector store.

Turns chunked documents + an embeddings model into a local, in-memory
similarity-search index.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_vector_store(
    chunks: list[Document], embeddings_model: Embeddings
) -> FAISS:
    """Index the chunks with the given embeddings model."""
    return FAISS.from_documents(chunks, embeddings_model)

