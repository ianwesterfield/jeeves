"""
Orchestrator Service - Multi-Turn Agentic Reasoning Engine

The brain of Jeeves: receives user intents, decomposes them into steps,
detects parallelization opportunities, and coordinates execution.

Endpoints:
  POST /api/orchestrate/set-workspace  - Set active directory context
  POST /api/orchestrate/next-step      - Generate next reasoning step
  POST /api/orchestrate/execute-batch  - Execute parallel batch of steps

Architecture:
  - Reasoning Engine: LLM calls + JSON parsing for step generation
  - Task Planner: Decomposes intent → independent or sequential steps
  - Parallel Executor: asyncio.gather for batch execution
  - Memory Connector: Pattern retrieval + storage for learning
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.orchestrator import router


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("orchestrator.main")
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
    logger.info("Starting Orchestrator Service...")
    logger.info("✓ Orchestrator ready on port 8004")
    yield
    logger.info("Shutting down Orchestrator Service...")


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Orchestrator Service",
    description="Multi-turn agentic reasoning engine with parallelization detection",
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

app.include_router(router, prefix="/api/orchestrate")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "orchestrator"}
