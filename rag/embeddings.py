"""STAGE 3: embeddings, backed by the Jina REST API.

`JinaEmbeddings` is a minimal LangChain `Embeddings` implementation: it is
just an HTTP call — POST text to https://api.jina.ai/v1/embeddings, get
vectors back. No local model, no GPU, no sentence-transformers.

It talks to the REST API directly (instead of depending on the third-party
`langchain-jina` package) because that package currently pins old
`langchain-core` versions.
"""

import requests
from langchain_core.embeddings import Embeddings

try:
    from rag.config import JINA_API_URL, JINA_MODEL  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for non-package execution
    from rag.config import JINA_API_URL, JINA_MODEL


class JinaEmbeddings(Embeddings):
    """Minimal LangChain `Embeddings` implementation backed by Jina.

    LangChain only needs two methods:
      - embed_documents(texts) -> list of vectors (used when indexing)
      - embed_query(text)      -> one vector   (used when searching)
    """

    def __init__(self, api_key: str, model: str = JINA_MODEL) -> None:
        self.model = model
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            }
        )

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        response = self.session.post(
            JINA_API_URL,
            json={
                "model": self.model,
                "task": task,
                "normalized": True,  # unit vectors -> cosine == dot product
                "input": texts,
            },
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Jina API error {response.status_code}: {response.text}"
            )

        # The API returns one object per input; sort by index just in case.
        data = response.json()["data"]
        return [
            item["embedding"]
            for item in sorted(data, key=lambda item: item["index"])
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task="retrieval.query")[0]
