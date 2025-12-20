import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api import memory
from services.embedder import embed


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("memory.main")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


# ============================================================================
# Application Setup
# ============================================================================

logger.info("Initializing Memory Service...")

app = FastAPI(
    title="Memory Service",
    description="Semantic memory layer for Open-WebUI (save/search conversations)",
    version="1.0.0",
)


# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all HTTP requests for debugging."""
    logger.debug(f"→ {request.method:6s} {request.url.path}")
    response = await call_next(request)
    logger.debug(f"← {response.status_code} {request.url.path}")
    return response


# ============================================================================
# CORS Configuration
# ============================================================================

# Allow local development origins and docker services
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://localhost:8080",
    "http://open-webui:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for: {', '.join(ALLOWED_ORIGINS)}")


# ============================================================================
# Static Files (Plugin Manifests)
# ============================================================================

static_dir = Path(__file__).resolve().parent / "static"
logger.info(f"Mounting static files from: {static_dir}")

app.mount("/.well-known", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# Startup Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Preload models at startup.
    
    - Loads embedding model (all-mpnet-base-v2) for sub-second first request
    - Connects to Qdrant and initializes collection if needed
    """
    logger.info("[startup] Preloading embedding model...")
    try:
        embed("warmup")  # Warmup call to load model
        logger.info("[startup] ✓ Embedding model ready")
    except Exception as e:
        logger.warning(f"[startup] ⚠ Failed to preload embedding model: {e}")


# ============================================================================
# Routers
# ============================================================================

logger.info("Registering memory router at /api/memory")
app.include_router(memory.router, prefix="/api/memory")


# ============================================================================
# Filter Plugin Endpoint
# ============================================================================

@app.get("/api/memory/filter", response_class=PlainTextResponse)
def get_memory_filter() -> str:
    """
    Serve the Open-WebUI filter plugin source code.
    
    Open-WebUI periodically calls this endpoint to fetch the filter plugin
    source, parses it to find the Filter class, and instantiates it to
    intercept conversations for memory operations.
    
    Returns: Complete Python source of memory.filter.py
    """
    logger.debug("[filter] Serving filter plugin source")
    filter_path = Path(__file__).resolve().parent / "memory.filter.py"
    return filter_path.read_text(encoding="utf-8")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for orchestration."""
    return {"status": "healthy"}


# ============================================================================
# Ready
# ============================================================================

logger.info("✓ Memory Service initialized and ready")
