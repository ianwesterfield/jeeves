"""
Schemas for the Memory API

Pydantic models for request/response validation. FastAPI uses these
for automatic docs and payload checking.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union


class Message(BaseModel):
    """One message in a conversation. Content can be string or list (for multi-modal)."""
    role: str
    content: Union[str, List[Any]]
    model_config = {"extra": "allow"}


class SaveRequest(BaseModel):
    """Request to save a conversation. user_id and messages are required."""
    user_id: str = Field(..., description="Unique identifier for the user")
    messages: List[Union[Message, dict]] = Field(..., description="Full messages array to store")
    model: Optional[str] = Field(None, description="LLM model identifier")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    source_type: Optional[str] = Field(None, description="Source type: document, prompt, url, image")
    source_name: Optional[str] = Field(None, description="Source name: filename, URL, or text preview")


class SearchRequest(BaseModel):
    """Search request—scoped by user_id, query text, and how many results you want."""
    user_id: str = Field(..., description="User ID to scope the search")
    query_text: str = Field(..., description="Natural language search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of memories to return")


class MemoryResult(BaseModel):
    """A single search result with the stored text, score, and source info."""
    user_text: Optional[str] = Field(None, description="Stored message text or facts")
    messages: Optional[List[dict]] = Field(None, description="Deprecated: raw messages")
    score: float = Field(..., description="Similarity score (0-1)")
    source_type: Optional[str] = Field(None, description="Source type")
    source_name: Optional[str] = Field(None, description="Source name")