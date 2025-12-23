"""
Extractor API Router

Endpoints:
  POST /extract       - Extract text chunks from content (base64 or text)
  POST /extract/file  - Extract text chunks from uploaded file

Supported formats:
  - Text/Markdown: Heading-aware or fixed-size chunking
  - Images: LLaVA/Florence vision model descriptions
  - Audio: Whisper transcription
  - PDF: PyMuPDF text extraction + chunking
"""

import base64
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from services.chunker import chunk_text, chunk_markdown
from services.image_extractor import extract_from_image
from services.audio_extractor import extract_from_audio
from services.pdf_extractor import extract_from_pdf


# ============================================================================
# Schemas
# ============================================================================

class Chunk(BaseModel):
    """
    A single extracted text chunk with metadata.
    
    Attributes:
        content: The actual text content
        chunk_index: Sequential index within the source
        chunk_type: Type of chunk (text, heading, image_description, transcript, etc.)
        section_title: Optional parent section/image name
        metadata: Optional additional metadata
    """
    content: str
    chunk_index: int
    chunk_type: str = Field(..., description="text|heading|image_description|transcript|pdf_page")
    section_title: Optional[str] = None
    metadata: Optional[dict] = None


class ExtractRequest(BaseModel):
    """
    Request to extract text from content.
    
    Attributes:
        content: Raw text or base64-encoded binary data
        content_type: MIME type (text/plain, image/png, application/pdf, etc.)
        source_name: Optional name for tracking source
        chunk_size: Target tokens per chunk
        chunk_overlap: Token overlap between chunks (for context)
        prompt: Optional guided prompt for image descriptions
    """
    content: str
    content_type: str
    source_name: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    prompt: Optional[str] = None


class ExtractResponse(BaseModel):
    """
    Response containing extracted chunks.
    
    Attributes:
        chunks: List of extracted chunks
        source_name: Echo of input source_name
        source_type: Detected type (text, markdown, image, audio, pdf)
        total_chunks: Total number of chunks extracted
    """
    chunks: List[Chunk]
    source_name: Optional[str] = None
    source_type: str
    total_chunks: int


# ============================================================================
# Router
# ============================================================================

router = APIRouter(tags=["extractor"])


# ============================================================================
# Content Type Handlers
# ============================================================================

def _handle_text_content(
    content: str,
    content_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list, str]:
    """Handle plain text, Python, and JSON content."""
    is_markdown = content_type == "text/markdown"
    
    if is_markdown:
        chunks = chunk_markdown(content, chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks, "markdown"
    else:
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks, "text"


async def _handle_image_content(
    content: str,
    source_name: Optional[str],
    prompt: Optional[str],
) -> tuple[list, str]:
    """Handle image content - decode and extract description."""
    image_data = base64.b64decode(content)
    description = await extract_from_image(image_data, prompt=prompt)
    
    chunk = Chunk(
        content=description,
        chunk_index=0,
        chunk_type="image_description",
        section_title=source_name,
    )
    
    return [chunk], "image"


async def _handle_audio_content(
    content: str,
    source_name: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list, str]:
    """Handle audio content - transcribe and chunk transcript."""
    audio_data = base64.b64decode(content)
    transcript = await extract_from_audio(audio_data)
    
    text_chunks = chunk_text(transcript, chunk_size=chunk_size, overlap=chunk_overlap)
    chunks = [
        Chunk(
            content=c.content,
            chunk_index=c.chunk_index,
            chunk_type="transcript",
            section_title=source_name,
        )
        for c in text_chunks
    ]
    
    return chunks, "audio"


async def _handle_pdf_content(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list, str]:
    """Handle PDF content - extract text and chunk."""
    pdf_data = base64.b64decode(content)
    pdf_text = await extract_from_pdf(pdf_data)
    chunks = chunk_text(pdf_text, chunk_size=chunk_size, overlap=chunk_overlap)
    
    return chunks, "pdf"


def _detect_markdown(content_type: str, source_name: Optional[str]) -> bool:
    """Detect if content should be treated as markdown."""
    if content_type == "text/markdown":
        return True
    if source_name and source_name.endswith(".md"):
        return True
    return False


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    """
    Extract text chunks from content.
    
    Routes to appropriate handler based on content_type:
      text/plain, text/python, application/json -> Text chunking
      text/markdown -> Heading-aware markdown chunking
      image/* -> Vision model (LLaVA/Florence) description extraction
      audio/* -> Whisper audio transcription + chunking
      application/pdf -> PyMuPDF text extraction + chunking
    
    Returns chunks with source tracking metadata.
    """
    content_type = req.content_type.lower().strip()
    
    try:
        # Text content (plain, JSON, Python code)
        if content_type in ("text/plain", "text/x-python", "application/json"):
            chunks, source_type = _handle_text_content(
                req.content,
                content_type,
                req.chunk_size,
                req.chunk_overlap,
            )
        
        # Markdown content
        elif _detect_markdown(content_type, req.source_name):
            chunks, source_type = _handle_text_content(
                req.content,
                "text/markdown",
                req.chunk_size,
                req.chunk_overlap,
            )
        
        # Image content
        elif content_type.startswith("image/"):
            chunks, source_type = await _handle_image_content(
                req.content,
                req.source_name,
                req.prompt,
            )
        
        # Audio content
        elif content_type.startswith("audio/"):
            chunks, source_type = await _handle_audio_content(
                req.content,
                req.source_name,
                req.chunk_size,
                req.chunk_overlap,
            )
        
        # PDF content
        elif content_type == "application/pdf":
            chunks, source_type = await _handle_pdf_content(
                req.content,
                req.chunk_size,
                req.chunk_overlap,
            )
        
        # Unknown content type - default to plain text
        else:
            chunks, source_type = _handle_text_content(
                req.content,
                "text/plain",
                req.chunk_size,
                req.chunk_overlap,
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    
    return ExtractResponse(
        chunks=chunks,
        source_name=req.source_name,
        source_type=source_type,
        total_chunks=len(chunks),
    )


@router.post("/file", response_model=ExtractResponse)
async def extract_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
):
    """
    Extract text chunks from an uploaded file.
    
    Handles file reading, content type detection, and delegates to extract().
    Text files are decoded as strings; binary files are base64-encoded.
    """
    content = await file.read()
    content_type = file.content_type or "application/octet-stream"
    
    # Decode text files; base64-encode binary
    if content_type.startswith("text/") or content_type == "application/json":
        content_str = content.decode("utf-8", errors="ignore")
    else:
        content_str = base64.b64encode(content).decode("utf-8")
    
    # Build request and delegate to main extract endpoint
    req = ExtractRequest(
        content=content_str,
        content_type=content_type,
        source_name=file.filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return await extract(req)
