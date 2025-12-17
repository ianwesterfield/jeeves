"""
Embedding Service

Generates 768-dimensional semantic embeddings using SentenceTransformers.
Uses the 'all-mpnet-base-v2' model, which provides excellent quality/speed
tradeoff for semantic similarity tasks.

The model produces normalized embeddings, making COSINE similarity equivalent
to DOT product - both methods return the same ordering.

Key Functions:
    embed(text: str) -> list[float]
        Generate embedding for a single text string.
    
    embed_messages(messages: list[dict]) -> list[float]
        Generate embedding from a full conversation, concatenating all
        message content with role prefixes.

Notes:
    - The model is loaded once at module import time (~2-3 seconds)
    - Embeddings are normalized (unit vectors) for efficient similarity
    - Multi-modal content (text + images) is handled - images become [image] tags
    - Both Pydantic models and plain dicts are accepted as messages
"""

from sentence_transformers import SentenceTransformer
import json
from typing import List, Any

# Load model once at module import - this is the embedding bottleneck
_model = SentenceTransformer("all-mpnet-base-v2")


def _extract_text_from_content(content: List[dict] | str) -> str:
    """
    Extract text from a message content field.
    
    Handles both simple string content and multi-modal content arrays
    from LLM providers like OpenAI (text + images).
    
    Args:
        content: Either a plain string or a list of content items,
                 where each item is a dict with "type" and "text"/"image_url".
    
    Returns:
        A single string with all text content concatenated.
        Images are represented as "[image]" placeholders.
    
    Example:
        >>> _extract_text_from_content([
        ...     {"type": "text", "text": "What is this?"},
        ...     {"type": "image_url", "image_url": {...}}
        ... ])
        "What is this? [image]"
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
    Generate a semantic embedding from a full conversation.
    
    Concatenates all message content with role prefixes (e.g., "[user]", "[assistant]")
    to create a single text representation, then embeds it.
    
    Args:
        messages: List of message dictionaries or Pydantic Message models.
                  Each message should have "role" and "content" fields.
    
    Returns:
        A 768-dimensional normalized embedding vector as a list of floats.
    
    Example:
        >>> embed_messages([
        ...     {"role": "user", "content": "My name is Ian"},
        ...     {"role": "assistant", "content": "Nice to meet you!"}
        ... ])
        [0.0234, -0.0156, 0.0789, ...]  # 768 floats
    
    Note:
        Pydantic models are automatically converted to dicts.
        Empty conversations return an embedding for "[empty conversation]".
    """
    print(f"[embedder] embed_messages called with {len(messages)} messages")
    text_parts = []
    
    for msg in messages:
        # Handle both Pydantic models and plain dicts
        if hasattr(msg, 'model_dump'):
            msg = msg.model_dump()
        
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
    Generate a semantic embedding for a single text string.
    
    This is the low-level embedding function. For conversations,
    use embed_messages() instead which handles role prefixes.
    
    Args:
        text: The text to embed. Can be any length (will be truncated
              by the model if needed, typically at ~512 tokens).
    
    Returns:
        A 768-dimensional normalized embedding vector as a list of floats.
    
    Example:
        >>> embed("My name is Ian and I work at Satcom Direct")
        [0.0234, -0.0156, 0.0789, ...]  # 768 floats
    """
    print(f"[embedder] embed() called with text ({len(text)} chars)")
    vec = _model.encode(text, normalize_embeddings=True)
    print(f"[embedder] Generated embedding, shape={vec.shape}")
    return vec.tolist()