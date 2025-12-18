"""
Extractor Service - Media to Text Extraction + Chunking

Converts various media types to searchable text chunks:
- Text/Markdown: Chunk by headings or fixed size
- Images: Generate descriptions via Florence-2
- Audio: Transcribe via Whisper
- PDF: Extract text via PyMuPDF, then chunk

Each chunk is returned with metadata for storage in Qdrant.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.extractor import router

app = FastAPI(
    title="Extractor Service",
    description="Media to text extraction and chunking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/extract")


@app.on_event("startup")
async def startup():
    print("[extractor] Starting Extractor Service...")
    print("[extractor] Models will be loaded on first use")


@app.get("/health")
async def health():
    return {"status": "healthy"}
