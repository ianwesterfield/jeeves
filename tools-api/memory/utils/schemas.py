"""
Pydantic Schemas for Memory API

Defines the request/response models for the memory service API.
These schemas provide validation, serialization, and documentation
for the FastAPI endpoints.

Models:
    Message: A single message in a conversation (user/assistant/system)
    SaveRequest: Request to save a conversation to memory
    SearchRequest: Request to search memories by query
    MemoryResult: A single memory search result
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union


class Message(BaseModel):
    """
    A single message in a conversation.
    
    Attributes:
        role: The speaker role - "user", "assistant", or "system".
        content: Message content - either a string or a list of content items
                 (for multi-modal messages with text and images).
    
    Note:
        Extra fields are allowed to support various LLM provider formats.
    """
    role: str
    content: Union[str, List[Any]]
    model_config = {"extra": "allow"}


class SaveRequest(BaseModel):
    """
    Request to save a conversation to memory.
    
    Attributes:
        user_id: Unique identifier for the user (required).
        messages: Full conversation history to store (required).
        model: The LLM model used (optional, for metadata).
        metadata: Additional metadata to store (optional).
        source_type: Content source type - "document", "prompt", "url", "image".
        source_name: Human-readable source identifier (filename, URL, snippet).
    """
    user_id: str = Field(..., description="Unique identifier for the user")
    messages: List[Union[Message, dict]] = Field(..., description="Full messages array to store")
    model: Optional[str] = Field(None, description="LLM model identifier")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    source_type: Optional[str] = Field(None, description="Source type: document, prompt, url, image")
    source_name: Optional[str] = Field(None, description="Source name: filename, URL, or text preview")


class SearchRequest(BaseModel):
    """
    Request to search memories by semantic similarity.
    
    Attributes:
        user_id: User ID to scope the search (required).
        query_text: Natural language query to match against memories.
        top_k: Maximum number of results to return (1-20, default 5).
    """
    user_id: str = Field(..., description="User ID to scope the search")
    query_text: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of memories to return")


class MemoryResult(BaseModel):
    """
    A single memory search result.
    
    Attributes:
        user_text: The stored user message text or extracted facts.
        messages: Deprecated - kept for backward compatibility.
        score: Similarity score (0-1, higher is more similar).
        source_type: Content source type (document, prompt, url, image).
        source_name: Human-readable source identifier.
    """
    user_text: Optional[str] = Field(None, description="Stored message text or facts")
    messages: Optional[List[dict]] = Field(None, description="Deprecated: raw messages")
    score: float = Field(..., description="Similarity score (0-1)")
    source_type: Optional[str] = Field(None, description="Source type")
    source_name: Optional[str] = Field(None, description="Source name")