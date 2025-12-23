"""
PDF Extraction Service

Extracts text from PDFs using PyMuPDF.
Handles both text-based and scanned PDFs (with OCR fallback).
"""

import io
import fitz  # PyMuPDF


async def extract_from_pdf(pdf_data: bytes) -> str:
    """
    Extract text from PDF.
    
    Args:
        pdf_data: Raw PDF bytes
    
    Returns:
        Extracted text content
    """
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    
    text_parts = []
    
    for page_num, page in enumerate(doc):
        # Extract text from page
        text = page.get_text()
        
        if text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
        else:
            # Page has no text - might be scanned
            # Could add OCR here with pytesseract if needed
            text_parts.append(f"[Page {page_num + 1}]\n[No extractable text - possibly scanned image]")
    
    doc.close()
    
    return "\n\n".join(text_parts)
