"""
Extractor API Router

Endpoints:
- POST /extract: Extract text chunks from content
- POST /extract/file: Extract from uploaded file
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import base64
import io

from services.chunker import chunk_text, chunk_markdown
from services.image_extractor import extract_from_image
from services.audio_extractor import extract_from_audio
from services.pdf_extractor import extract_from_pdf

router = APIRouter(tags=["extractor"])


class Chunk(BaseModel):
    """A single extracted text chunk."""
    content: str
    chunk_index: int
    chunk_type: str  # "text", "heading", "image_description", "transcript"
    section_title: Optional[str] = None
    metadata: Optional[dict] = None


class ExtractRequest(BaseModel):
    """Request to extract text from content."""
    content: str  # Raw text or base64-encoded binary
    content_type: str  # "text/plain", "text/markdown", "image/png", etc.
    source_name: Optional[str] = None
    chunk_size: int = 500  # Target tokens per chunk
    chunk_overlap: int = 50  # Overlap between chunks


class ExtractResponse(BaseModel):
    """Response containing extracted chunks."""
    chunks: List[Chunk]
    source_name: Optional[str] = None
    source_type: str  # "text", "markdown", "image", "audio", "pdf"
    total_chunks: int


@router.post("/", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    """
    Extract text chunks from content.
    
    Content type detection:
    - text/plain, text/markdown -> Text chunking
    - image/* -> Florence-2 description
    - audio/* -> Whisper transcription
    - application/pdf -> PyMuPDF extraction
    """
    content_type = req.content_type.lower()
    chunks = []
    source_type = "text"
    
    try:
        if content_type in ("text/plain", "text/x-python", "application/json"):
            # Plain text chunking
            source_type = "text"
            chunks = chunk_text(
                req.content,
                chunk_size=req.chunk_size,
                overlap=req.chunk_overlap
            )
            
        elif content_type == "text/markdown" or (req.source_name and req.source_name.endswith(".md")):
            # Markdown with heading-aware chunking
            source_type = "markdown"
            chunks = chunk_markdown(
                req.content,
                chunk_size=req.chunk_size,
                overlap=req.chunk_overlap
            )
            
        elif content_type.startswith("image/"):
            # Image description
            source_type = "image"
            image_data = base64.b64decode(req.content)
            description = await extract_from_image(image_data)
            chunks = [Chunk(
                content=description,
                chunk_index=0,
                chunk_type="image_description",
                section_title=req.source_name
            )]
            
        elif content_type.startswith("audio/"):
            # Audio transcription
            source_type = "audio"
            audio_data = base64.b64decode(req.content)
            transcript = await extract_from_audio(audio_data)
            # Chunk the transcript
            text_chunks = chunk_text(transcript, chunk_size=req.chunk_size, overlap=req.chunk_overlap)
            chunks = [
                Chunk(
                    content=c.content,
                    chunk_index=c.chunk_index,
                    chunk_type="transcript",
                    section_title=req.source_name
                )
                for c in text_chunks
            ]
            
        elif content_type == "application/pdf":
            # PDF extraction
            source_type = "pdf"
            pdf_data = base64.b64decode(req.content)
            pdf_text = await extract_from_pdf(pdf_data)
            chunks = chunk_text(pdf_text, chunk_size=req.chunk_size, overlap=req.chunk_overlap)
            
        else:
            # Unknown type - try as plain text
            source_type = "text"
            chunks = chunk_text(
                req.content,
                chunk_size=req.chunk_size,
                overlap=req.chunk_overlap
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    return ExtractResponse(
        chunks=chunks,
        source_name=req.source_name,
        source_type=source_type,
        total_chunks=len(chunks)
    )


@router.post("/file", response_model=ExtractResponse)
async def extract_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50)
):
    """Extract text chunks from an uploaded file."""
    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    
    # For text files, decode to string
    if content_type.startswith("text/") or content_type == "application/json":
        content_str = content.decode("utf-8", errors="ignore")
    else:
        # Binary content - base64 encode for processing
        content_str = base64.b64encode(content).decode("utf-8")
    
    req = ExtractRequest(
        content=content_str,
        content_type=content_type,
        source_name=file.filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return await extract(req)
