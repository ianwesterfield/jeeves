"""
Memory API Router

FastAPI router implementing the core memory service endpoints:
- /save: Store conversations with semantic embeddings and extracted facts
- /search: Retrieve relevant memories via semantic similarity
- /summaries: Get summarized memory entries for a user

The save endpoint implements a "search-first" pattern:
1. Generate embedding from conversation
2. Search for similar existing memories
3. Extract structured facts (names, dates, preferences, etc.)
4. Use pragmatics classifier to decide if worth saving
5. Store with both original text and extracted facts
6. Return any found context for injection into the conversation

This allows the system to both recall relevant context AND learn new information
in a single API call, minimizing latency for the Open-WebUI filter.
"""

import os
from typing import Dict, Any, Optional, List, Union
import uuid
import json
import hashlib
import requests
from fastapi import APIRouter, HTTPException
from services.qdrant_client import _client, _ensure_collection
from services.embedder import embed_messages, embed
from services.fact_extractor import (
    extract_facts,
    extract_facts_from_document,
    format_facts_for_storage,
    facts_to_embedding_text,
)
from utils.schemas import SaveRequest, SearchRequest, MemoryResult
from qdrant_client.http import models

router = APIRouter(tags=["memory"])
collection_name = os.getenv("INDEX_NAME", "user_memory_collection")

# Pragmatics service endpoint for save/skip classification
PRAGMATICS_HOST = os.getenv("PRAGMATICS_HOST", "pragmatics_api")
PRAGMATICS_PORT = os.getenv("PRAGMATICS_PORT", "8001")

# Keywords that indicate document/reference material being shared (should always save)
# These phrases indicate PROVIDING content, not asking about it.
# Used as a heuristic fallback when the ML classifier might miss document uploads.
DOCUMENT_SHARE_PHRASES = [
    "please parse", "here's my", "here is my", "this is my", "store this",
    "keep this", "save this", "remember this document", "for future reference",
    "here's a document", "this document", "attached document", "below is",
]


def _has_document_intent(text: str) -> bool:
    """
    Detect if the message indicates document/reference material sharing.
    
    This heuristic fallback helps catch document uploads that the ML classifier
    might miss. It looks for explicit sharing phrases and long-form content.
    
    Args:
        text: The user message text to analyze.
        
    Returns:
        True if the message appears to be sharing a document/reference,
        False if it looks like a question or casual message.
        
    Note:
        Questions are explicitly excluded to avoid saving recall requests.
        Long messages (>500 chars) without question marks are assumed to be documents.
    """
    text_lower = text.lower().strip()
    
    # Exclude questions - they're recalls, not saves
    if text_lower.startswith(("do you", "can you", "what is", "what's", "where is", 
                              "how do", "have you", "did you", "does ", "is there",
                              "are there", "could you", "would you", "will you")):
        return False
    if "?" in text_lower[:100]:  # Question mark early in text suggests a question
        return False
    
    # Check for sharing phrases
    for phrase in DOCUMENT_SHARE_PHRASES:
        if phrase in text_lower:
            return True
    
    # Long messages (>500 chars) that don't look like questions are likely documents
    if len(text) > 500:
        return True
    
    return False

def _is_worth_saving(messages: list, extracted_facts: Optional[List[Dict]] = None) -> bool:
    """
    Determine if a conversation is worth persisting to memory.
    
    Uses a multi-layer decision process:
    1. If facts were extracted, always save (valuable structured data)
    2. If document intent detected, always save (explicit sharing)
    3. Call pragmatics classifier for ML-based decision
    4. On any error, fail-open (save to avoid losing data)
    
    Args:
        messages: List of conversation messages (dicts or Pydantic models).
        extracted_facts: Pre-extracted facts from the text, if any.
        
    Returns:
        True if the conversation should be saved, False otherwise.
        
    Note:
        The pragmatics classifier is a DistilBERT model trained to distinguish
        "save" intent ("My name is Ian") from "skip" intent ("What time is it?").
        See pragmatics/ service for details.
    """
    # If we already extracted facts, definitely save
    if extracted_facts and len(extracted_facts) >= 1:
        print(f"[_is_worth_saving] Extracted {len(extracted_facts)} facts, forcing save")
        return True
    
    # Extract the last user message to send to pragmatics
    user_text = None
    for msg in reversed(messages):
        msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else msg
        if msg_dict.get("role") == "user":
            user_text = _get_text_content(msg_dict.get("content", ""))
            break
    
    if not user_text:
        print("[_is_worth_saving] No user message found, defaulting to save")
        return True
    
    # Heuristic check: document/reference sharing should always save
    if _has_document_intent(user_text):
        print(f"[_is_worth_saving] Document intent detected, forcing save")
        return True
    
    try:
        resp = requests.post(
            f"http://{PRAGMATICS_HOST}:{PRAGMATICS_PORT}/api/pragmatic",
            json={"text": user_text},
            timeout=3
        )
        resp.raise_for_status()
        result = resp.json()
        is_save = result.get("is_save_request", True)
        confidence = result.get("confidence", 0.0)
        print(f"[_is_worth_saving] Pragmatics response: is_save_request={is_save}, confidence={confidence:.4f}")
        
        # Low confidence = uncertain, so fail-open to avoid losing important info
        # If classifier isn't confident (< 0.60), default to save
        if confidence < 0.60:
            print(f"[_is_worth_saving] Low confidence ({confidence:.2f} < 0.60), defaulting to save")
            return True
        
        return is_save
    except Exception as e:
        # Fail open: if pragmatics is down, save anyway
        print(f"[_is_worth_saving] Pragmatics call failed ({e}), defaulting to save")
        return True

def _get_text_content(content) -> str:
    """
    Extract plain text from message content.
    
    Handles both simple string content and multi-modal content arrays
    (text + images) as used by providers like OpenAI.
    
    Args:
        content: Either a string or a list of content items.
                 List items should be dicts with "type" and "text" keys.
    
    Returns:
        Concatenated text content, or empty string if no text found.
    
    Example:
        >>> _get_text_content([{"type": "text", "text": "Hello"}, {"type": "image_url", ...}])
        "Hello"
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    return ""


def _make_uuid(user_id: str, content_hash: str) -> int:
    """
    Generate a deterministic 64-bit integer ID for Qdrant storage.
    
    Uses UUID5 (SHA-1 based) to create a reproducible ID from user_id
    and content hash. This ensures the same content always gets the
    same ID, enabling upsert semantics (update if exists, insert if not).
    
    Args:
        user_id: The user's unique identifier.
        content_hash: Hash of the content being stored.
    
    Returns:
        A 64-bit signed integer suitable for Qdrant point IDs.
    
    Note:
        Qdrant requires integer IDs. We use modulo to stay within
        the 64-bit signed integer range.
    """
    name = f"{user_id}:{content_hash}"
    uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, name)
    
    # Convert to integer, keeping within 64-bit signed range
    return int(uuid_obj.int) % (2**63)

@router.post("/save", status_code=200)
def save_memory(req: SaveRequest) -> Dict[str, Any]:
    """
    Save a conversation to memory with semantic embedding.
    
    Implements a "search-first" pattern:
    1. Generate embedding from conversation messages
    2. Search Qdrant for similar existing memories
    3. Extract structured facts (names, preferences, etc.)
    4. Use pragmatics classifier to decide if worth saving
    5. If worth saving, upsert to Qdrant with facts
    6. Return any found context for injection
    
    Args:
        req: SaveRequest containing:
            - user_id: User identifier for scoping
            - messages: Conversation messages to process
            - model: Optional LLM model identifier
            - source_type: Optional source type (document, prompt, url, image)
            - source_name: Optional source name (filename, URL, snippet)
    
    Returns:
        Dict with:
            - status: "saved", "saved_with_context", "context_found", or "skipped"
            - reason: Human-readable explanation
            - existing_context: List of relevant memories (if found)
    
    Raises:
        HTTPException: On storage or embedding errors.
    """
    print(f"[/save] Received request: user_id={req.user_id}, messages={len(req.messages)}, model={req.model}")
    
    # Ensure collection exists
    print("[/save] Ensuring collection exists...")
    _ensure_collection()
    
    # Generate embedding from all messages
    print("[/save] Generating embedding from messages...")
    vector = embed_messages(req.messages)
    print(f"[/save] Embedding generated, dim={len(vector)}")
    
    # Extract the last user message text for storage
    user_text = None
    for msg in reversed(req.messages):
        msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else msg
        if msg_dict.get("role") == "user":
            user_text = _get_text_content(msg_dict.get("content", ""))
            break
    
    if not user_text:
        user_text = "[empty message]"
    
    print(f"[/save] User text: {user_text[:100]}..." if len(user_text) > 100 else f"[/save] User text: {user_text}")
    
    # Always search for similar memories first (even if we might skip saving)
    client = _client()
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    existing_context = None
    
    try:
        # Use REST API directly for search
        search_response = requests.post(
            f"http://{qdrant_host}:{qdrant_port}/collections/{collection_name}/points/search",
            json={
                "vector": vector,
                "filter": {
                    "must": [
                        {
                            "key": "user_id",
                            "match": {"value": req.user_id}
                        }
                    ]
                },
                "limit": 10,  # Increased to capture more potential matches
                "score_threshold": 0.05,  # Lowered to include document content
                "with_payload": True,
            },
            timeout=5
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        search_results = search_data.get("result", []) if "result" in search_data else []
        
        print(f"Search for user {req.user_id}: found {len(search_results)} results")
        
        # Store context if found (we'll return it regardless of whether we save)
        # Prefer facts_text over user_text for cleaner context injection
        if search_results:
            existing_context = []
            for pt in search_results:
                payload = pt.get("payload", {})
                # Prefer structured facts if available, fall back to user_text
                facts_text = payload.get("facts_text")
                stored_text = payload.get("user_text", "")
                
                existing_context.append({
                    "user_text": facts_text if facts_text else stored_text,
                    "facts": payload.get("facts"),  # Include raw facts for advanced use
                    "score": pt.get("score", 0),
                    "source_type": payload.get("source_type", "prompt"),
                    "source_name": payload.get("source_name"),
                })
            
            print(f"Context extracted: {len(existing_context)} items")
    
    except Exception as e:
        # If search fails, just continue
        print(f"Search failed: {e}")
    
    # Extract structured facts from the text BEFORE checking if worth saving
    # This allows us to save messages that contain valuable facts even if
    # the pragmatics classifier thinks it's not a save request
    is_document = len(user_text) > 500
    if is_document:
        facts = extract_facts_from_document(user_text)
        print(f"[/save] Extracted {len(facts)} facts from document")
    else:
        facts = extract_facts(user_text)
        print(f"[/save] Extracted {len(facts)} facts from message")
    
    # Check if this conversation is worth saving (pass facts for override)
    worth_saving = _is_worth_saving(req.messages, extracted_facts=facts)
    
    print(f"DEBUG: _is_worth_saving returned {worth_saving}")
    
    if not worth_saving:
        print("Skipping save: conversation not worth saving (too trivial)")
    
        # Return context if found, even though we're not saving
        if existing_context:
            return {
                "status": "context_found",
                "point_id": None,
                "content_hash": None,
                "existing_context": existing_context,
                "reason": "Conversation too trivial to save, but context found"
            }
        return {
            "status": "skipped",
            "point_id": None,
            "content_hash": None,
            "existing_context": None,
            "reason": "Conversation too trivial to save"
        }
    
    # Always save the new memory (even if similar context was found)
    content_hash = hashlib.md5(user_text.encode()).hexdigest()
    
    # Format facts for storage
    facts_text = format_facts_for_storage(facts)
    
    # Create embedding-optimized text
    # For documents with facts, embed the facts for better semantic matching
    # For short messages without facts, use original text
    if facts:
        embedding_text = facts_to_embedding_text(facts, user_text)
        print(f"[/save] Using fact-based embedding: {embedding_text[:200]}..." if len(embedding_text) > 200 else f"[/save] Using fact-based embedding: {embedding_text}")
        # Re-embed with fact-focused text
        vector = embed(embedding_text)
    
    # Build payload with both original text and extracted facts
    payload: Dict[str, Any] = {
        "user_id": req.user_id,
        "user_text": user_text,
    }
    
    # Add source information
    if req.source_type:
        payload["source_type"] = req.source_type
    if req.source_name:
        payload["source_name"] = req.source_name
    
    # Add facts if extracted
    if facts:
        payload["facts"] = facts
        payload["facts_text"] = facts_text
    
    point_id = _make_uuid(req.user_id, content_hash)
    
    point = models.PointStruct(
        id=point_id,
        vector=vector,
        payload=payload,
    )
    print(f"[/save] Upserting point_id={point_id} to collection={collection_name}")
    client.upsert(collection_name=collection_name, points=[point])
    print(f"[/save] Upsert complete")
    
    # Return status with both saved info and any context that was found
    result = {
        "status": "saved",
        "point_id": point_id,
        "content_hash": content_hash,
        "existing_context": existing_context
    }
    
    # If we found context, also indicate that
    if existing_context:
        result["status"] = "saved_with_context"
    
    return result

@router.post("/search", response_model=list[MemoryResult])
def search_memory(req: SearchRequest) -> List[MemoryResult]:
    """
    Search for relevant memories by semantic similarity.
    
    Generates an embedding from the query text and searches Qdrant
    for similar stored memories, filtered by user_id.
    
    Args:
        req: SearchRequest containing:
            - user_id: User identifier for scoping
            - query_text: Natural language search query
            - top_k: Maximum number of results (1-20)
    
    Returns:
        List of MemoryResult objects with:
            - user_text: The stored message text or extracted facts
            - score: Similarity score (0-1, higher is more similar)
            - source_type: Content source type
            - source_name: Human-readable source identifier
    
    Note:
        Results are filtered by user_id using Qdrant payload filters.
        A score threshold of 0.05 is applied to filter out irrelevant results.
    """
    print(f"[/search] Received request: user_id={req.user_id}, query_text={req.query_text[:60]}..., top_k={req.top_k}")
    query_vec = embed(req.query_text)
    print(f"[/search] Query embedding generated, dim={len(query_vec)}")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    
    try:
        search_response = requests.post(
            f"http://{qdrant_host}:{qdrant_port}/collections/{collection_name}/points/search",
            json={
                "vector": query_vec,
                "filter": {
                    "must": [
                        {
                            "key": "user_id",
                            "match": {"value": req.user_id}
                        }
                    ]
                },
                "limit": req.top_k,
                "with_payload": True,
            },
            timeout=5
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        results = search_data.get("result", []) if "result" in search_data else []
        
        print(f"[/search] Qdrant returned {len(results)} results")
        
        if not results:
            print("[/search] No memories found, returning 404")
            raise HTTPException(status_code=404, detail="No memories found")
        
        return [
            MemoryResult(
                user_text=pt.get("payload", {}).get("user_text", ""),
                messages=None,  # Deprecated
                score=pt.get("score", 0)
            )
            for pt in results
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/summaries")
def list_summaries(req: SearchRequest) -> List[Dict[str, Any]]:
    """
    Return memory summaries matching a query for a user.
    
    Similar to search, but returns only summary text rather than
    full memory content. Useful for getting a quick overview of
    stored information without loading full documents.
    
    Args:
        req: SearchRequest containing:
            - user_id: User identifier for scoping
            - query_text: Search query (defaults to general terms if empty)
            - top_k: Maximum number of summaries (1-20)
    
    Returns:
        List of dicts with:
            - summary: The summarized memory text
            - score: Similarity score (0-1)
    
    Raises:
        HTTPException 404: If no summaries found for the user.
        HTTPException 500: On search errors.
    
    Note:
        If query_text is empty, searches using general terms like
        "personal facts dates names preferences" to get diverse results.
    """
    print(f"[/summaries] Received request: user_id={req.user_id}, query_text={req.query_text}, top_k={req.top_k}")
    query = req.query_text or "personal facts dates names preferences"
    query_vec = embed(query)
    print(f"[/summaries] Query embedding generated, dim={len(query_vec)}")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")

    try:
        search_response = requests.post(
            f"http://{qdrant_host}:{qdrant_port}/collections/{collection_name}/points/search",
            json={
                "vector": query_vec,
                "filter": {
                    "must": [
                        {"key": "user_id", "match": {"value": req.user_id}}
                    ]
                },
                "limit": req.top_k,
                "with_payload": True,
            },
            timeout=5,
        )
        search_response.raise_for_status()
        data = search_response.json()
        results = data.get("result", [])

        print(f"[/summaries] Qdrant returned {len(results)} results")
        
        # Extract summaries only
        summaries = []
        for pt in results:
            payload = pt.get("payload", {})
            summary = payload.get("summary")
            if summary:
                summaries.append({"summary": summary, "score": pt.get("score", 0)})

        print(f"[/summaries] Extracted {len(summaries)} summaries")
        
        if not summaries:
            raise HTTPException(status_code=404, detail="No summaries found")

        return summaries
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summaries failed: {str(e)}")