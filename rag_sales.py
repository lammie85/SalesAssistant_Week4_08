# Imports and initialization

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import requests
import sys
import warnings


# getting standard out put from windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# silence noise from langchain
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain_community"
)

# load env
load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY loaded: ", bool(GROQ_KEY))

if not GROQ_KEY:
    raise SystemExit("Missing GROQ_API_KEY in .env")

JINA_KEY = os.getenv("JINA_API_KEY")
print("JINA_API_KEY loaded: ", bool(JINA_KEY))

if not JINA_KEY:
    raise SystemError("Missing JINA_API_KEY in .env")


# STAGE 1: DATA INGESTION
DATA_FILE_PATH = os.path.join("data", "policies.txt")
loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s), "
      f"{len(documents[0].page_content)} characters total")


# STAGE 2:  CHUNCKING or splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")


# Embedding(JINA API)
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"


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


embeddings_model = JinaEmbeddings(api_key=JINA_KEY)
print("Embeddings model ready (Jina API):", embeddings_model.model)


# STAGE 4: vector store
vector_store = FAISS.from_documents(chunks, embeddings_model)
print(f"Indexed {vector_store.index.ntotal} chunks in FAISS")

test_query = "How many days do you have for refund"
top_matches = vector_store.similarity_search(test_query, k=2)

print(f"\nTop matches for: '{test_query}'")
for i, match in enumerate(top_matches, start=1):
    print(f"---match {i} ---\n{match.page_content}\n")


# STAGE 5 - THE LLM(Groq)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
print("LLM ready: ", llm.model_name)


# ---------------------------------------------------------------------------
# STAGE 6 — THE TOOL
# ---------------------------------------------------------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


def search_policy(question: str) -> str:
    """Search the company's Inventory & Sales policy documents for information
    about returns, bulk discounts, supplier terms, restocking, complaints,
    shipping/delivery, warranties and repairs, customer credit terms, price
    matching, damaged or lost shipments, stocktakes and shrinkage, gift cards
    and store credit, or special orders and backorders."""
    matching_chunks = retriever.invoke(question)
    return "\n\n".join(chunk.page_content for chunk in matching_chunks)


# ---------------------------------------------------------------------------
# STAGE 7 — THE AGENT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a friendly Inventory & Sales policy assistant. Always use the "
    "search_policy tool to look up facts before answering. If the answer "
    "isn't in the search results, say you don't know. Never promise a "
    "refund directly — always say it needs human approval, per policy."
)

inventory_assistant = create_agent(
    model=llm,
    tools=[search_policy],
    system_prompt=SYSTEM_PROMPT,
)

print("Inventory & Sales assistant is ready.\n")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def ask_assistant(question: str, history: list | None = None) -> str:
    """Ask a single question. If `history` is given, it is used as prior
    conversation context (list of {'role', 'content'} dicts)."""
    print("=" * 60)
    print("Q:", question)
    print("-" * 60)

    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    response = inventory_assistant.invoke({"messages": messages})
    answer = response["messages"][-1].content
    print("A:", answer)
    return answer


def chat_loop():
    """Interactive REPL. Keeps rolling conversation history so follow-up
    questions like 'and for opened electronics?' work."""
    print("Type your question, or 'quit'/'exit' to stop.")
    print("(Type 'reset' to clear conversation memory.)\n")

    history: list[dict] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if question.lower() == "reset":
            history.clear()
            print("[memory cleared]\n")
            continue

        try:
            answer = ask_assistant(question, history)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"[error] {e.__class__.__name__}: {e}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Quick smoke test — proves the pipeline works end-to-end.
    ask_assistant("How many units do I need to order to get a bulk discount?")
    ask_assistant(
        "My order arrived late, can I get a refund on the whole order?")
    ask_assistant("What are the supplier payment terms?")

    # 2) Drop into an interactive chat.
    print()
    chat_loop()
