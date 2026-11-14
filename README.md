# Inventory & Sales RAG Practice — Groq + Jina + FAISS

Practice build following the free-tier stack: **Groq** (LLM, free tier),
**Jina** (embeddings, free tier), **FAISS** (local vector store, free).
Same workflow as the reference video — rebuilt on Inventory & Sales
policy data instead of HR, so this is genuinely yours to teach from.

## What's been verified already
- ✅ Project scaffolded, dependencies installed (`requirements.txt` pinned)
- ✅ `.env` / `.gitignore` wired correctly — secrets never get tracked by git
- ✅ Data loading + chunking confirmed working (13 clean chunks from
  `data/policies.txt` — one chunk per policy)
- ✅ **Jina embeddings** verified live: `jina-embeddings-v5-text-small` turns the
  chunks into vectors over the real API (no Hugging Face, no local model download,
  no PyTorch)
- ✅ FAISS index built from those Jina vectors, similarity search returns the
  right return-policy chunk
- ✅ Groq agent verified end-to-end — all three sample questions answered from
  `data/policies.txt` via the `search_policy` tool

## What you need to do on your own machine

### 1. Get free API keys
- **Groq**: console.groq.com → sign in → API Keys → Create API Key
- **Jina**: jina.ai → sign in → get API key (generous free tier)

### 2. Fill in `.env`
Replace the placeholders in `.env` with your real keys:
```
GROQ_API_KEY=gsk_your_real_key
JINA_API_KEY=jina_your_real_key
```

### 3. Run it
```bash
cd groq_rag_practice
source venv/bin/activate        # Windows: venv\Scripts\activate
python rag_practice.py
```

You should see: environment loaded → document loaded → 13 chunks →
Jina embeddings ready → FAISS indexed with Jina vectors → a raw similarity
search result → the LLM ready → three sample questions asked and answered by
the agent, using the `search_policy` tool to ground its answers in
`data/policies.txt`.

## Try asking it yourself
Add your own questions at the bottom of `rag_practice.py`, e.g.:
- "Can I return an opened electronics item after 10 days?"
- "How is the reorder quantity calculated?"
- "What happens if a supplier pays late?"
- "My laptop stopped working after 8 months — is it still covered?"
- "Do you match a cheaper price I saw at another store?"
- "A box arrived with two items missing. What do I do?"
- "How long is a gift card valid, and can I get cash for it?"

## What to notice while you run this (teaching points)
- **Jina embeddings are just an HTTP call.** `JinaEmbeddings` is ~30 lines:
  POST text to `https://api.jina.ai/v1/embeddings`, get vectors back. No local
  model, no GPU, no `sentence-transformers`. That demystifies "embedding model"
  for trainees — it's an API, same as the LLM.
- **Retrieval is asymmetric.** Jina gives documents the `retrieval.passage`
  adapter and questions the `retrieval.query` adapter. LangChain's `Embeddings`
  interface already splits these into `embed_documents()` and `embed_query()`,
  which is exactly why those are two methods instead of one. Swap the tasks
  around and show retrieval quality drop — great live demo.
- **The system prompt explicitly forbids** the assistant from promising a
  refund directly — this is the safety-guardrail pattern from Phase 2,
  now doing real work in an agent's instructions, not just a lecture point.
- `create_agent` is LangChain's newer, higher-level way to build an agent —
  compare this to the raw `tools_schema` + manual loop from the Build
  Guide's Phase 5. Teach the raw version first so trainees understand
  what `create_agent` is doing for them underneath.
- `langchain-community` is flagged as being sunset in favor of standalone
  integration packages. Jina's third-party `langchain-jina` package currently
  pins old `langchain-core` versions, which is why this file talks to the Jina
  REST API directly instead of depending on it. Worth mentioning to trainees
  that the ecosystem moves fast — checking a package's current compatibility is
  a real, ongoing AI-engineering skill, not a one-time lesson.

## Where this plugs into the Django app
This exact pattern — `TextLoader` → `RecursiveCharacterTextSplitter` →
embeddings → `FAISS`/Chroma → `create_agent` with a `search_policy` tool —
is the LangChain-framework version of what the Build Guide's Step 8-9
built with raw OpenAI calls and Chroma. Once you're comfortable with both,
decide which one to actually teach: raw code teaches the concepts more
transparently; `create_agent` ships faster. Many instructors teach raw
first, then show this as "the shortcut, now that you know what's really
happening."
