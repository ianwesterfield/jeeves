"""
Extractor Service - Media to Text Conversion

Converts various media types to searchable text chunks:
  - Text/Markdown: Chunk by headings or fixed size
  - Images: LLaVA/Florence-2 vision model descriptions
  - Audio: Whisper speech-to-text transcription
  - PDF: PyMuPDF text extraction + chunking

Each extraction produces chunks with metadata for storage/embedding
(typically in Qdrant vector database).

Startup: Preloads vision model for sub-second first-request latency.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.extractor import router
from services.image_extractor import load_model_at_startup


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Extractor Service",
    description="Media to text extraction and chunking (text, images, audio, PDF)",
    version="1.0.0",
)

# Enable CORS for all origins (safe in local network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount extraction router at /api/extract
app.include_router(router, prefix="/api/extract")


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup hook.
    
    Preloads vision model (LLaVA/Florence) at startup to:
      1. Catch model loading errors early
      2. Eliminate ~8-12s latency from first image extraction request
      3. Ensure model stays in GPU memory during runtime
    """
    print("[extractor] Starting Extractor Service...")
    print("[extractor] Preloading vision model (LLaVA/Florence)...")
    
    try:
        load_model_at_startup()
        print("[extractor] ✓ Vision model preloaded successfully")
    except Exception as e:
        print(f"[extractor] ⚠ Failed to preload vision model: {e}")
        print("[extractor] Note: Model will be loaded on first request")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns: {"status": "healthy"} if service is running.
    Used by orchestration/load balancers for liveness probes.
    """
    return {"status": "healthy"}
