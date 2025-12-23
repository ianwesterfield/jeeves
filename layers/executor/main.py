"""
Executor Service - Unified Polyglot Code Execution Engine

Executes code in multiple languages with sandboxing and security:
  - Python (RestrictedPython)
  - PowerShell (Constrained Language Mode)
  - Node.js / JavaScript / TypeScript
  - Bash (command tokenization)
  - Go, Rust (Docker compilation)
  - Ruby, R, Julia (subprocess)

Endpoints:
  POST /api/execute/tool   - Execute any registered tool
  POST /api/execute/code   - Execute code in specified language
  POST /api/execute/shell  - Execute shell command
  POST /api/execute/file   - File operations (read/write/list)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.executor import router


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("executor.main")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


# ============================================================================
# Lifespan (startup/shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Executor Service...")
    logger.info("Initializing language handlers...")
    
    # Pre-check available runtimes
    from services.polyglot_handler import PolyglotHandler
    handler = PolyglotHandler()
    available = await handler.check_available_runtimes()
    logger.info(f"Available runtimes: {', '.join(available)}")
    
    logger.info("✓ Executor ready on port 8005")
    yield
    logger.info("Shutting down Executor Service...")


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Executor Service",
    description="Unified polyglot code execution with sandboxing",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# CORS Configuration
# ============================================================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://localhost:8180",
    "http://open-webui:8080",
    "http://orchestrator_api:8004",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Routers
# ============================================================================

app.include_router(router, prefix="/api/execute")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "executor"}
