"""
Embedding Service

Generates 768-dim semantic embeddings using SentenceTransformers.
I use all-mpnet-base-v2 - good balance of quality and speed.

The embeddings are normalized, so COSINE = DOT product (same ordering).

Key functions:
    embed(text) - embed a single string
    embed_messages(messages) - embed a full conversation with role prefixes
"""

from sentence_transformers import SentenceTransformer
import json
from typing import List, Any

# Load model once at import time
_model = SentenceTransformer("all-mpnet-base-v2")


def _extract_text_from_content(content: List[dict] | str) -> str:
    """
    Extract text from message content. Handles both plain strings
    and multi-modal arrays (text + images). Images become "[image]".
    """
    if isinstance(content, str):
        return content
    
    text_parts = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif item.get("type") == "image_url":
                    # For images, add a placeholder
                    text_parts.append("[image]")
    
    return " ".join(text_parts)

def embed_messages(messages: List[dict]) -> list[float]:
    """Turn a full conversation into a single 768-dim vector.
    
    I join all the messages with their roles as prefixes (like "[user] ..."),
    then embed that. Handles both dicts and Pydantic models.
    """
    text_parts = []
    
    for msg in messages:
        if hasattr(msg, 'model_dump'):
            msg = msg.model_dump()
        
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role:
                text_parts.append(f"[{role}]")
            
            extracted = _extract_text_from_content(content)
            if extracted:
                text_parts.append(extracted)
    
    combined_text = " ".join(text_parts)
    
    if not combined_text.strip():
        combined_text = "[empty conversation]"
    
    vec = _model.encode(combined_text, normalize_embeddings=True)
    return vec.tolist()

def embed(text: str) -> list[float]:
    """Embed a single string into 768 dimensions. Low-level version—use embed_messages() for conversations."""
    vec = _model.encode(text, normalize_embeddings=True)
    return vec.tolist()