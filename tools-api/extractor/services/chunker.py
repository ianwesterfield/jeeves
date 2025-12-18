"""
Text Chunking Service

Splits text into chunks for embedding. Two strategies:
1. Fixed-size chunking with overlap (for plain text)
2. Heading-aware chunking (for markdown)
"""

import re
from typing import List, Dict, Any


def _count_tokens_approx(text: str) -> int:
    """Approximate token count (rough: ~4 chars per token)."""
    return len(text) // 4


def _make_chunk(content: str, chunk_index: int, chunk_type: str, section_title: str = None) -> Dict[str, Any]:
    """Create a chunk dict."""
    return {
        "content": content,
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
        "section_title": section_title,
        "metadata": None
    }


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size chunks with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Target tokens per chunk
        overlap: Tokens to overlap between chunks
    
    Returns:
        List of chunk dicts
    """
    if not text or not text.strip():
        return []
    
    # Convert token targets to char targets (approx 4 chars per token)
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + char_size
        
        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start + char_size // 2, end + 100)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence break
                sentence_break = text.rfind(". ", start + char_size // 2, end + 50)
                if sentence_break > start:
                    end = sentence_break + 1
        
        chunk_content = text[start:end].strip()
        
        if chunk_content:
            chunks.append(_make_chunk(
                content=chunk_content,
                chunk_index=chunk_index,
                chunk_type="text"
            ))
            chunk_index += 1
        
        # Move start with overlap
        start = end - char_overlap
        if start <= 0 and chunk_index > 0:
            break  # Avoid infinite loop
    
    return chunks


def chunk_markdown(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Split markdown into chunks, respecting heading structure.
    
    Strategy:
    1. Split by top-level headings (## or #)
    2. If a section is too long, sub-chunk it
    3. Preserve heading context in each chunk
    
    Args:
        text: Markdown text to chunk
        chunk_size: Target tokens per chunk
        overlap: Tokens to overlap (within sections)
    
    Returns:
        List of chunk dicts with section_title set
    """
    if not text or not text.strip():
        return []
    
    # Pattern to split on headings (capture the heading)
    heading_pattern = r'^(#{1,3}\s+.+)$'
    
    # Split into sections
    lines = text.split('\n')
    sections = []
    current_section = {"title": None, "content": []}
    
    for line in lines:
        if re.match(heading_pattern, line):
            # Save previous section if it has content
            if current_section["content"]:
                sections.append(current_section)
            # Start new section
            current_section = {
                "title": line.strip().lstrip('#').strip(),
                "content": [line]
            }
        else:
            current_section["content"].append(line)
    
    # Don't forget the last section
    if current_section["content"]:
        sections.append(current_section)
    
    # Now chunk each section
    chunks = []
    chunk_index = 0
    char_size = chunk_size * 4
    
    for section in sections:
        section_text = '\n'.join(section["content"])
        section_title = section["title"]
        
        # If section fits in one chunk, keep it whole
        if len(section_text) <= char_size * 1.2:  # Allow 20% overflow
            if section_text.strip():
                chunks.append(_make_chunk(
                    content=section_text.strip(),
                    chunk_index=chunk_index,
                    chunk_type="heading" if section_title else "text",
                    section_title=section_title
                ))
                chunk_index += 1
        else:
            # Section too long - sub-chunk it
            sub_chunks = chunk_text(section_text, chunk_size, overlap)
            for sub in sub_chunks:
                # Prepend section context if not first chunk
                content = sub["content"]
                if sub["chunk_index"] > 0 and section_title:
                    content = f"[Section: {section_title}]\n\n{content}"
                
                chunks.append(_make_chunk(
                    content=content,
                    chunk_index=chunk_index,
                    chunk_type="text",
                    section_title=section_title
                ))
                chunk_index += 1
    
    return chunks
