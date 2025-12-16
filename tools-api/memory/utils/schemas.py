from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union

class Message(BaseModel):
    """Message object from the request"""
    role: str  # "user", "assistant", "system"
    content: Union[str, List[Any]]  # Can be simple string or list of content items
    model_config = {"extra": "allow"}  # Allow additional fields

class SaveRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    messages: List[Union[Message, dict]] = Field(..., description="Full messages array to store")
    model: Optional[str] = None
    metadata: Optional[dict] = None

class SearchRequest(BaseModel):
    user_id: str = Field(..., description="User ID to scope the search")
    query_text: str = Field(..., description="Prompt text to retrieve relevant memories")
    top_k: int = Field(5, ge=1, le=20, description="Number of memories to return")

class MemoryResult(BaseModel):
    messages: Optional[List[dict]] = None
    score: float