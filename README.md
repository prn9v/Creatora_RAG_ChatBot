# Creatora RAG Chatbot

A free, lightweight Retrieval-Augmented Generation (RAG) chatbot backend for **Creatora**, built with **FastAPI** and **Google's Gemini API** (free tier — no billing required).

The bot answers user questions about the Creatora product by retrieving relevant chunks from a knowledge-base document (`docs/project_details.txt`) and grounding Gemini's answer in that context — so it doesn't hallucinate features that don't exist.

---

## How it works

1. On startup, the app reads `docs/project_details.txt`, splits it into overlapping word-chunks, and embeds each chunk using Gemini's embedding model.
2. Embeddings are cached to `docs/embeddings_cache.json` (keyed by a hash of the document's content) so the app doesn't re-embed on every restart — only when the source document actually changes.
3. On every `/chat` request, the user's question is embedded and compared (cosine similarity) against the cached chunk embeddings to find the most relevant pieces of context.
4. Those chunks are inserted into a system prompt, and Gemini generates an answer grounded strictly in that context.

No paid vector database, no paid LLM API — everything runs on Gemini's free tier plus an in-memory/JSON-file vector store.

---

## Tech stack

| Component | Choice |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM (chat) | `gemini-2.5-flash-lite` (Gemini API, free tier) |
| Embeddings | `gemini-embedding-001` (Gemini API, free tier) |
| Vector store | In-memory NumPy array, cached to a local JSON file |
| Config | `python-dotenv` (`.env` file) |

---

## Project structure

```
.
├── main.py                      # FastAPI app: chunking, embedding, retrieval, chat endpoint
├── requirements.txt
├── .env.example                 # copy to .env and fill in your key
├── .gitignore
└── docs/
    ├── project_details.txt      # your knowledge base source document
    └── embeddings_cache.json    # auto-generated cache (git-ignored)
```

---

## Setup (local development)

### 1. Clone and install dependencies

```bash
git clone https://github.com/prn9v/Creatora_RAG_ChatBot.git
cd Creatora_RAG_ChatBot
pip install -r requirements.txt
```

### 2. Get a free Gemini API key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **Create API key** (choose "Create in new project" if asked)
4. Copy the key — no credit card or billing setup is required for the free tier

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_key_here
```

### 4. Add your knowledge base

Place your product documentation as plain text at:

```
docs/project_details.txt
```

### 5. Run the server

```bash
python main.py
```

The first run will chunk and embed your document (this consumes a small amount of free-tier embedding quota, paced at ~1 request/second to avoid rate limits) and cache the result. Subsequent restarts reuse the cache as long as `project_details.txt` hasn't changed.

The API will be live at `http://localhost:8000`.

---

## API Reference

### `GET /health`

Basic health check.

**Response**
```json
{ "status": "ok", "chunks_loaded": 32 }
```

### `POST /chat`

Ask the chatbot a question.

**Request body**
```json
{
  "question": "Can I auto-post to Instagram from Creatora?",
  "history": [
    { "role": "user", "content": "What is Creatora?" },
    { "role": "assistant", "content": "Creatora is an AI content creation platform..." }
  ]
}
```

- `question` (string, required)
- `history` (array, optional) — prior conversation turns for context continuity. Only the last 6 turns are used.

**Response**
```json
{
  "answer": "No — Creatora doesn't auto-publish to Instagram...",
  "sources_used": 4
}
```

If the knowledge base has no relevant content for the question, the bot will say it's unsure rather than guessing.

---

## Deploying to Render (free tier)

1. Push this repo to GitHub.
2. On [Render](https://render.com), click **New → Web Service** and connect the repo.
3. Configure:

| Field | Value |
|---|---|
| Language | Python 3 |
| Branch | `main` |
| Root Directory | leave blank if `main.py` is at repo root |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Instance Type | Free |

4. Under **Environment**, add:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | your Gemini API key |

5. Deploy. Your service will be live at `https://<your-service-name>.onrender.com`.

### Free-tier caveats

- Render's free instances **spin down after ~15 minutes of inactivity** and have **no persistent disk**. This means `embeddings_cache.json` won't survive a restart, so the app will re-embed the knowledge base on the first request after a cold start (roughly 30–40 seconds for ~30 chunks). This only affects the first request after idle time, not every request.
- If this cold-start delay is a problem, upgrade to Render's Starter tier ($7/mo), which includes persistent disks and no spin-down.

---

## Rate limits & quota

This project runs entirely on Gemini's **free tier**:
- `gemini-embedding-001` and `gemini-2.5-flash-lite` both have free daily/per-minute request quotas.
- The app automatically retries with exponential backoff if it hits a `429 RESOURCE_EXHAUSTED` error.
- You can check your current usage and limits at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit).

If you see errors mentioning a model has `limit: 0`, that model has likely been deprecated/shut down by Google — check [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for the current recommended replacement and update `EMBEDDING_MODEL` / `CHAT_MODEL` in `main.py`.

---

## Updating the knowledge base

Just edit or replace `docs/project_details.txt` and restart the app. The content hash will no longer match the cache, so it will automatically re-chunk and re-embed the new content.

---

## License

For personal/educational use.
