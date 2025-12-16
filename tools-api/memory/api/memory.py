import os
from typing import Dict, Any
import uuid
import json
import hashlib
import requests
from fastapi import APIRouter, HTTPException
from services.qdrant_client import _client, _ensure_collection
from services.embedder import embed_messages, embed
from utils.schemas import SaveRequest, SearchRequest, MemoryResult
from qdrant_client.http import models

router = APIRouter()
collection_name = os.getenv("INDEX_NAME", "user_memory_collection")

# Pragmatics service endpoint for save/skip classification
PRAGMATICS_HOST = os.getenv("PRAGMATICS_HOST", "pragmatics_api")
PRAGMATICS_PORT = os.getenv("PRAGMATICS_PORT", "8001")

# Keywords that indicate document/reference material being shared (should always save)
# These are phrases that indicate PROVIDING content, not asking about it
DOCUMENT_SHARE_PHRASES = [
    "please parse", "here's my", "here is my", "this is my", "store this",
    "keep this", "save this", "remember this document", "for future reference",
    "here's a document", "this document", "attached document", "below is",
]

def _has_document_intent(text: str) -> bool:
    """
    Check if the message contains document/reference sharing intent.
    This is a fallback heuristic for when the classifier misses document uploads.
    Only triggers on SHARING phrases, not questions about documents.
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

def _is_worth_saving(messages: list) -> bool:
    """
    Determine if a conversation is worth saving to memory.
    Calls the pragmatics classifier service to decide if the user's message
    indicates intent to remember/save information.
    Returns True if pragmatics says save, or on any error (fail-open).
    Also returns True for document/reference sharing (heuristic fallback).
    """
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
        return is_save
    except Exception as e:
        # Fail open: if pragmatics is down, save anyway
        print(f"[_is_worth_saving] Pragmatics call failed ({e}), defaulting to save")
        return True

def _get_text_content(content) -> str:
    """
    Extract text from content (handles both string and list formats).
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
    Deterministic integer ID based on user_id + content hash.
    Qdrant requires integer IDs.
    """
    name = f"{user_id}:{content_hash}"
    uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, name)
    
    # Convert to integer, keeping within 64-bit signed range
    return int(uuid_obj.int) % (2**63)

@router.post("/save", status_code=200)
def save_memory(req: SaveRequest):
    """
    Save entire messages array as memory with semantic embedding.
    First searches for similar context - if found, returns existing memories
    instead of saving new ones.
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
                "limit": 5,
                "score_threshold": 0.3,
                "with_payload": True,
            },
            timeout=5
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        search_results = search_data.get("result", []) if "result" in search_data else []
        
        print(f"Search for user {req.user_id}: found {len(search_results)} results")
        
        # Store context if found (we'll return it regardless of whether we save)
        if search_results:
            existing_context = [
                {
                    "user_text": pt.get("payload", {}).get("user_text", ""),
                    "score": pt.get("score", 0),
                }
                for pt in search_results
            ]
            
            print(f"Context extracted: {len(existing_context)} items")
    
    except Exception as e:
        # If search fails, just continue
        print(f"Search failed: {e}")
    
    # Check if this conversation is worth saving
    worth_saving = _is_worth_saving(req.messages)
    
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
    
    # Build payload with user text
    payload: Dict[str, Any] = {
        "user_id": req.user_id,
        "user_text": user_text,
    }
    
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
def search_memory(req: SearchRequest):
    """
    Search for relevant messages by query text.
    Returns full message objects that match the query.
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
                messages=pt.get("payload", {}).get("messages", []),
                score=pt.get("score", 0)
            )
            for pt in results
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/summaries")
def list_summaries(req: SearchRequest):
    """
    Return top-k summaries for a user's memories matching the query text.
    If query_text is empty, returns recent/top summaries (by similarity to a neutral vector).
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