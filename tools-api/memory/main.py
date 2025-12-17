"""
Jeeves Memory Service - Main Entry Point

This is my semantic memory layer for Open-WebUI. It captures conversations,
extracts facts, generates embeddings, and stores everything in Qdrant so I 
can recall relevant context later.

Stack:
    - FastAPI with CORS for Open-WebUI integration
    - Qdrant vector DB for semantic storage
    - SentenceTransformers (all-mpnet-base-v2) for 768-dim embeddings
    - Pattern-based fact extraction for structured data
    - My pragmatics classifier to decide what's worth saving

Endpoints:
    POST /api/memory/save      - Save a conversation
    POST /api/memory/search    - Search memories
    POST /api/memory/summaries - Get memory summaries
    GET  /api/memory/filter    - Serve the filter plugin for Open-WebUI
    GET  /.well-known/*        - Static plugin manifest files

Env vars:
    QDRANT_HOST, QDRANT_PORT, INDEX_NAME, PRAGMATICS_HOST, PRAGMATICS_PORT
"""

import inspect
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from pathlib import Path
from api import memory

print("[main] Starting OpenWebUI Memory Service...")

app = FastAPI(
    title="OpenWebUI Memory Service",
    version="1.0",
    description="Semantic memory layer for Open-WebUI conversations",
)


@app.on_event("startup")
async def preload_models() -> None:
    """Load the embedding model at startup so the first request isn't slow."""
    print("[main] Preloading embedding model...")
    from services.embedder import embed
    embed("warmup")
    print("[main] Embedding model loaded and ready.")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Simple request/response logging for debugging."""
    print(f"[main] Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"[main] Response status: {response.status_code}")
    return response


# CORS - allowing my local dev origins
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://localhost:8080",
    "http://open-webui:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Plugin manifest files for Open-WebUI discovery
static_dir = Path(__file__).resolve().parent / "static"
print(f"[main] Mounting static files from {static_dir}")
app.mount("/.well-known", StaticFiles(directory=str(static_dir)), name="static")

# Memory API routes
print("[main] Including memory router at /api/memory")
app.include_router(memory.router, prefix="/api/memory")


@app.get("/api/memory/filter", response_class=PlainTextResponse)
def get_memory_filter() -> str:
    """
    Serve the filter plugin source for Open-WebUI.
    Open-WebUI parses this to find the Filter class and wire it up.
    """
    print("[main] GET /api/memory/filter called")
    functions_module_path = Path(__file__).resolve().parent / "memory.filter.py"
    with open(functions_module_path, "r", encoding="utf-8") as f:
        return f.read()


print("[main] Memory service initialized and ready")