import os
import json
import logging
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import time
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing. Add it to your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "models/gemini-embedding-001"  # current embedding model (text-embedding-004 was shut down)
CHAT_MODEL = "gemini-2.5-flash-lite"             # current free-tier model (gemini-2.0-flash was shut down)

DOC_PATH = Path("docs/project_details.txt")
CACHE_PATH = Path("docs/embeddings_cache.json")

CHUNK_SIZE_WORDS = 220     # ~ words per chunk
CHUNK_OVERLAP_WORDS = 40   # overlap so context isn't cut mid-thought
TOP_K = 4                  # how many chunks to retrieve per question

# ---------------------------------------------------------------------------
# In-memory "vector store"
# ---------------------------------------------------------------------------
_chunks: list[str] = []
_embeddings: np.ndarray | None = None


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE_WORDS])
        if chunk.strip():
            chunks.append(chunk)
        if i + CHUNK_SIZE_WORDS >= len(words):
            break
    return chunks


def _embed_one_with_retry(text: str, task_type: str, max_retries: int = 5) -> list[float]:
    """Embed a single string, retrying with backoff if the free tier rate-limits us."""
    delay = 3
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return result.embeddings[0].values
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                logger.warning(f"Rate limited on embedding, retrying in {delay}s…")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise RuntimeError("Embedding failed after retries")


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    """Embed a list of texts using Gemini's embedding model, one call at a time
    (free tier allows a limited number of requests per minute, so we pace it)."""
    vectors = []
    for i, t in enumerate(texts):
        vectors.append(_embed_one_with_retry(t, task_type))
        # small pause to be gentle on free-tier RPM limits when embedding many chunks
        if i < len(texts) - 1:
            time.sleep(1)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def build_or_load_index():
    """Chunk the doc + embed it, caching to disk so we don't re-embed on every restart.
    Uses a content hash (not file mtime) as the cache key, since git checkouts reset
    file mtimes on every deploy — a hash lets the cache survive redeploys if you
    commit embeddings_cache.json and the doc content hasn't changed."""
    global _chunks, _embeddings

    if not DOC_PATH.exists():
        logger.warning(f"⚠️ {DOC_PATH} not found — chatbot will have no knowledge base.")
        return

    doc_text = DOC_PATH.read_text(encoding="utf-8")
    doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()

    # Try cache first (avoids re-calling the embedding API every restart/redeploy)
    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if cached.get("doc_hash") == doc_hash:
                _chunks = cached["chunks"]
                _embeddings = np.array(cached["embeddings"], dtype=np.float32)
                logger.info(f"♻️ Loaded {len(_chunks)} cached chunks/embeddings.")
                return
        except Exception as e:
            logger.warning(f"Cache read failed, rebuilding: {e}")

    logger.info("📄 Chunking document…")
    _chunks = chunk_text(doc_text)
    logger.info(f"🔢 Embedding {len(_chunks)} chunks with Gemini (one-time cost)…")
    _embeddings = embed_texts(_chunks, task_type="RETRIEVAL_DOCUMENT")

    CACHE_PATH.write_text(
        json.dumps(
            {
                "doc_hash": doc_hash,
                "chunks": _chunks,
                "embeddings": _embeddings.tolist(),
            }
        ),
        encoding="utf-8",
    )
    logger.info("✅ Knowledge base ready and cached.")


def retrieve_context(question: str, top_k: int = TOP_K) -> list[str]:
    if _embeddings is None or len(_chunks) == 0:
        return []
    query_vec = embed_texts([question], task_type="RETRIEVAL_QUERY")[0]
    scores = cosine_similarity(query_vec, _embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [_chunks[i] for i in top_idx]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Creatora RAG Chatbot (Gemini, free tier)…")
    try:
        build_or_load_index()
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        logger.warning("⚠️ Continuing without a knowledge base — chat will still work but ungrounded.")
    yield
    logger.info("👋 Shutting down Creatora RAG Chatbot")


app = FastAPI(lifespan=lifespan, title="Creatora RAG Chatbot (Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []  # e.g. [{"role": "user"/"assistant", "content": "..."}]


@app.get("/health")
async def health():
    return {"status": "ok", "chunks_loaded": len(_chunks)}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, detail="Question cannot be empty.")

    try:
        context_chunks = retrieve_context(req.question)
        context_block = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

        system_instruction = (
            "You are a helpful support assistant for Creatora, an AI content-creation platform. "
            "Answer the user's question using ONLY the CONTEXT provided below. "
            "If the answer isn't in the context, say you're not sure and suggest they contact support. "
            "Be concise, friendly, and accurate. Do not make up features that aren't in the context.\n\n"
            f"CONTEXT:\n{context_block}"
        )

        # Fold prior turns into the conversation for continuity
        history_text = ""
        for turn in req.history[-6:]:  # keep last few turns to save tokens
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

        full_prompt = f"{history_text}USER: {req.question}"

        delay = 5
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    ),
                )
                return {
                    "answer": response.text,
                    "sources_used": len(context_chunks),
                }
            except Exception as e:
                last_error = e
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if is_rate_limit and attempt < max_retries - 1:
                    logger.warning(f"Rate limited on chat, retrying in {delay}s…")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise last_error

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(500, detail=f"Failed to generate response: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)