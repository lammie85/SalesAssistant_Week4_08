"""Central configuration: paths, model names, and environment checks.

Everything the pipeline needs to know about *settings* lives here, so the
other modules stay focused on behavior.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains the `rag` package and `data/`.
# Resolving from this file (instead of the current working directory) means
# the pipeline works no matter where you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the policy documents live.
DATA_FILE_PATH = PROJECT_ROOT / "data" / "policies.txt"

# Model settings.
GROQ_MODEL = "openai/gpt-oss-120b"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"

# Chunking settings.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings: how many chunks the search_policy tool fetches.
RETRIEVER_K = 3


class ConfigError(RuntimeError):
    """Raised when the environment is not usable (e.g. a missing API key)."""


def load_env() -> None:
    """Load variables from the project's .env file (no-op if it is absent)."""
    load_dotenv(PROJECT_ROOT / ".env")


def require_key(name: str) -> str:
    """Return the API key stored under `name`, or raise a clear ConfigError."""
    load_env()
    value = os.getenv(name)
    if not value:
        raise ConfigError(
            f"Missing {name}. Add it to {PROJECT_ROOT / '.env'} as "
            f"`{name}=your_key`, then restart the app."
        )
    return value
