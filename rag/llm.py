"""STAGE 5: the LLM (Groq, free tier)."""

from langchain_groq import ChatGroq

from rag.config import GROQ_MODEL


def build_llm(api_key: str) -> ChatGroq:
    """Create the chat model used by the agent."""
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)
