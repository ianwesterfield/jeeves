"""
OpenWebUI Memory Service - Main Application Entry Point

This FastAPI application provides a semantic memory layer for Open-WebUI conversations.
It captures user messages, extracts structured facts, generates embeddings, and stores
them in Qdrant for later retrieval via semantic search.

Architecture:
    - FastAPI app with CORS middleware for Open-WebUI integration
    - Qdrant vector database for semantic storage and retrieval
    - SentenceTransformers (all-mpnet-base-v2) for 768-dim embeddings
    - Pattern-based fact extraction for structured data
    - Pragmatics classifier for save/skip decisions

Endpoints:
    POST /api/memory/save    - Save conversation with semantic embedding
    POST /api/memory/search  - Search memories by query text
    POST /api/memory/summaries - Get memory summaries for a user
    GET  /api/memory/filter  - Serve filter plugin source for Open-WebUI
    GET  /.well-known/*      - Static plugin manifest files

Environment Variables:
    QDRANT_HOST      - Qdrant server hostname (default: localhost)
    QDRANT_PORT      - Qdrant server port (default: 6333)
    INDEX_NAME       - Collection name (default: user_memory_collection)
    PRAGMATICS_HOST  - Pragmatics classifier host (default: pragmatics_api)
    PRAGMATICS_PORT  - Pragmatics classifier port (default: 8001)

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
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
    """
    Pre-warm the embedding model on startup.
    
    This prevents timeout on the first real request by loading the
    SentenceTransformer model into memory before any API calls.
    """
    print("[main] Preloading embedding model...")
    from services.embedder import embed
    embed("warmup")  # Trigger model load
    print("[main] Embedding model loaded and ready.")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    HTTP middleware for request/response logging.
    
    Logs all incoming requests and their response status codes
    for debugging and monitoring purposes.
    
    Args:
        request: The incoming HTTP request.
        call_next: The next middleware/handler in the chain.
        
    Returns:
        The HTTP response from downstream handlers.
    """
    print(f"[main] Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"[main] Response status: {response.status_code}")
    return response


# CORS Configuration
# These origins are allowed to make cross-origin requests to the API.
# Add additional origins as needed for your deployment.
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

# Static Files
# Serve plugin manifest files (ai-plugin.json, openapi.yaml) under /.well-known
# These are required for Open-WebUI to discover and integrate the memory service.
static_dir = Path(__file__).resolve().parent / "static"
print(f"[main] Mounting static files from {static_dir}")
app.mount("/.well-known", StaticFiles(directory=str(static_dir)), name="static")

# API Router
# The memory router handles /save, /search, and /summaries endpoints.
print("[main] Including memory router at /api/memory")
app.include_router(memory.router, prefix="/api/memory")


@app.get("/api/memory/filter", response_class=PlainTextResponse)
def get_memory_filter() -> str:
    """
    Serve the memory filter plugin source code for Open-WebUI.
    
    Open-WebUI parses this Python source to discover the Filter class
    and its inlet/outlet methods. The filter intercepts conversations,
    sends them to the memory API, and injects retrieved context.
    
    Returns:
        str: The complete source code of memory.filter.py as plain text.
        
    Note:
        This endpoint is called by Open-WebUI when loading filters.
        The returned code must be valid Python with a Filter class.
    """
    print("[main] GET /api/memory/filter called")
    functions_module_path = Path(__file__).resolve().parent / "memory.filter.py"
    with open(functions_module_path, "r", encoding="utf-8") as f:
        return f.read()


print("[main] Memory service initialized and ready")