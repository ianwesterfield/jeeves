from sentence_transformers import SentenceTransformer
import json
from typing import List, Any

_model = SentenceTransformer("all-mpnet-base-v2")

def _extract_text_from_content(content: List[dict] | str) -> str:
    """
    Extract all text from a content array (handling mixed text/images).
    Falls back to string if content is already a string.
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
    """
    Generate embedding from full messages array by concatenating all text content.
    Returns a 768-dim vector.
    """
    print(f"[embedder] embed_messages called with {len(messages)} messages")
    text_parts = []
    
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Add role to context
            if role:
                text_parts.append(f"[{role}]")
            
            # Extract text from content
            extracted = _extract_text_from_content(content)
            if extracted:
                text_parts.append(extracted)
    
    combined_text = " ".join(text_parts)
    
    # Fallback if no text extracted
    if not combined_text.strip():
        combined_text = "[empty conversation]"
    
    print(f"[embedder] Combined text ({len(combined_text)} chars): {combined_text[:100]}..." if len(combined_text) > 100 else f"[embedder] Combined text: {combined_text}")
    vec = _model.encode(combined_text, normalize_embeddings=True)
    print(f"[embedder] Generated embedding, shape={vec.shape}")
    return vec.tolist()

def embed(text: str) -> list[float]:
    """
    Simple text embedding (kept for backwards compatibility).
    Returns a 768-dim vector.
    """
    print(f"[embedder] embed() called with text ({len(text)} chars)")
    vec = _model.encode(text, normalize_embeddings=True)
    print(f"[embedder] Generated embedding, shape={vec.shape}")
    return vec.tolist()