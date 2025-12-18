"""
Memory API Router

Core endpoints for my memory service:
- /save: Store conversations with embeddings and extracted facts
- /search: Find relevant memories via semantic similarity
- /summaries: Get memory summaries for a user

The save endpoint does a "search-first" pattern:
1. Generate embedding from the conversation
2. Search for similar existing memories
3. Extract structured facts (names, dates, preferences, etc.)
4. Ask my pragmatics classifier if it's worth saving
5. Store with both original text and extracted facts
6. Return any found context so the filter can inject it

This way I can recall context AND learn new info in one API call.
"""

import os
import time
from typing import Dict, Any, Optional, List, Union
import uuid
import json
import hashlib
import requests
import base64
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import APIRouter, HTTPException
from services.qdrant_client import _client, _ensure_collection
from services.embedder import embed_messages, embed
# Fact extractor disabled for now - was causing false positives
# from services.fact_extractor import (
#     extract_facts,
#     extract_facts_from_document,
#     format_facts_for_storage,
#     facts_to_embedding_text,
# )
from utils.schemas import SaveRequest, SearchRequest, MemoryResult
from qdrant_client.http import models

router = APIRouter(tags=["memory"])
collection_name = os.getenv("INDEX_NAME", "user_memory_collection")

# Pragmatics classifier endpoint
PRAGMATICS_HOST = os.getenv("PRAGMATICS_HOST", "pragmatics_api")
PRAGMATICS_PORT = os.getenv("PRAGMATICS_PORT", "8001")

# HTTP session with retry for Qdrant REST calls
_http_session = None

def _get_http_session() -> requests.Session:
    """Get a requests session with automatic retry on connection errors."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        _http_session.mount("http://", adapter)
        _http_session.mount("https://", adapter)
    return _http_session

def _is_worth_saving(messages: list) -> tuple[bool, str]:
    """
    Ask the pragmatics classifier if this conversation is worth saving.
    
    Returns:
        (should_save, error_message) - error_message is empty string if no error
    
    Raises no exceptions - returns (False, error) if classifier is unavailable.
    """
    # Extract the last user message to send to pragmatics
    user_text = None
    for msg in reversed(messages):
        msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else msg
        if msg_dict.get("role") == "user":
            user_text = _get_text_content(msg_dict.get("content", ""))
            break
    
    if not user_text:
        return True, ""
    
    try:
        resp = requests.post(
            f"http://{PRAGMATICS_HOST}:{PRAGMATICS_PORT}/api/pragmatic",
            json={"text": user_text},
            timeout=5
        )
        resp.raise_for_status()
        result = resp.json()
        is_save = result.get("is_save_request", False)
        confidence = result.get("confidence", 0.0)
        print(f"[memory] Classifier: is_save={is_save} confidence={confidence:.2f}")
        return is_save, ""
    except requests.exceptions.ConnectionError:
        return False, "Pragmatics classifier unavailable"
    except requests.exceptions.Timeout:
        return False, "Pragmatics classifier timeout"
    except Exception as e:
        return False, f"Pragmatics error: {str(e)[:50]}"

def _get_text_content(content) -> str:
    """Extract plain text from message content (handles strings and multi-modal arrays)."""
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


def _serialize_full_content(content: Union[str, List[Dict[str, Any]]], max_size_mb: int = 100) -> Optional[str]:
    """
    Serialize full message content for storage, including multi-modal data.
    Converts images to base64, checks total size limit.
    
    Returns JSON string of the content, or None if too large.
    """
    if isinstance(content, str):
        # Plain text - check size
        size_bytes = len(content.encode('utf-8'))
        if size_bytes > max_size_mb * 1024 * 1024:
            return None
        return content
    
    elif isinstance(content, list):
        serialized_items = []
        total_size = 0
        
        for item in content:
            if not isinstance(item, dict):
                continue
                
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                text_bytes = text.encode('utf-8')
                total_size += len(text_bytes)
                if total_size > max_size_mb * 1024 * 1024:
                    return None
                serialized_items.append({"type": "text", "text": text})
                
            elif item_type == "image_url":
                # Convert image to base64
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = str(image_url)
                
                try:
                    # For now, assume data URLs or fetch if needed
                    # In practice, Open-WebUI sends base64 data URLs
                    if url.startswith("data:image/"):
                        # Already base64, extract
                        header, data = url.split(",", 1)
                        image_bytes = base64.b64decode(data)
                        total_size += len(image_bytes)
                        if total_size > max_size_mb * 1024 * 1024:
                            return None
                        serialized_items.append({
                            "type": "image_url",
                            "image_url": {"url": url}  # Keep original
                        })
                    else:
                        # External URL - could fetch, but for now skip or placeholder
                        # To keep simple, just store the URL
                        serialized_items.append({
                            "type": "image_url", 
                            "image_url": {"url": url}
                        })
                except Exception:
                    # If conversion fails, skip this item
                    continue
        
        if total_size > max_size_mb * 1024 * 1024:
            return None
            
        return json.dumps(serialized_items)
    
    return None


def _make_uuid(user_id: str, content_hash: str) -> int:
    """
    Generate a deterministic ID for Qdrant from user_id + content hash.
    Same content always gets same ID, so upserts work correctly.
    """
    name = f"{user_id}:{content_hash}"
    uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, name)
    return int(uuid_obj.int) % (2**63)  # Keep within 64-bit signed range

@router.post("/save", status_code=200)
def save_memory(req: SaveRequest) -> Dict[str, Any]:
    """
    Save a conversation to memory.
    
    Does search-first: looks for similar memories, extracts facts, 
    asks my classifier if it is worth saving, then stores it.
    Returns any found context so the filter can inject it.
    """
    _ensure_collection()
    
    # Extract the last user message for storage (full content + text)
    user_content = None
    user_text = None
    for msg in reversed(req.messages):
        msg_dict = msg.model_dump() if hasattr(msg, 'model_dump') else msg
        if msg_dict.get("role") == "user":
            user_content = msg_dict.get("content", "")
            user_text = _get_text_content(user_content)
            break
    
    # For SEARCH: embed only the last user message (not full conversation)
    # This prevents prior context from polluting search results
    if user_text:
        search_vector = embed_messages([{"role": "user", "content": user_text}])
    else:
        search_vector = embed_messages(req.messages)
    
    # For STORAGE: embed the full conversation for richer context
    storage_vector = embed_messages(req.messages)
    
    if not user_text:
        user_text = "[empty message]"
    
    # Serialize full content for storage (with size limit)
    serialized_content = _serialize_full_content(user_content)
    if serialized_content is None:
        # Content too large, skip saving
        return {
            "status": "skipped",
            "point_id": None,
            "content_hash": None,
            "existing_context": None,
            "reason": "Content too large (>100MB)"
        }
    
    # Always search for similar memories first (even if we might skip saving)
    client = _client()
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    existing_context = None
    
    try:
        # Use REST API directly for search with retry
        session = _get_http_session()
        search_response = session.post(
            f"http://{qdrant_host}:{qdrant_port}/collections/{collection_name}/points/search",
            json={
                "vector": search_vector,  # Use search_vector (last user message only) for retrieval
                "filter": {
                    "must": [
                        {
                            "key": "user_id",
                            "match": {"value": req.user_id}
                        }
                    ]
                },
                "limit": 5,  # Get more results to find relevant docs
                "score_threshold": 0.45,  # Lower threshold to catch document chunks
                "with_payload": True,
            },
            timeout=5
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        search_results = search_data.get("result", []) if "result" in search_data else []
        
        # Apply score gap filter: if there's a big drop between top result and others,
        # only keep the highly relevant ones (prevents "wife's name" when asking "my name")
        if len(search_results) > 1:
            top_score = search_results[0].get("score", 0)
            filtered_results = [search_results[0]]  # Always keep top result
            for pt in search_results[1:]:
                pt_score = pt.get("score", 0)
                # Keep if within 15% of top score (relative gap)
                # OR if absolute score is very high (>0.7)
                if pt_score >= top_score * 0.85 or pt_score > 0.7:
                    filtered_results.append(pt)
            search_results = filtered_results
        
        # Store context if found
        if search_results:
            existing_context = []
            for pt in search_results:
                payload = pt.get("payload", {})
                facts_text = payload.get("facts_text")
                stored_text = payload.get("user_text", "")
                full_content = payload.get("full_content")
                
                # Use full content if available, otherwise fall back to text
                context_text = stored_text
                if full_content:
                    try:
                        # If it's JSON array, extract text parts
                        content_data = json.loads(full_content)
                        if isinstance(content_data, list):
                            text_parts = []
                            for item in content_data:
                                if item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            if text_parts:
                                context_text = " ".join(text_parts)
                    except:
                        pass  # Fall back to stored_text
                
                existing_context.append({
                    "user_text": facts_text if facts_text else context_text,
                    "full_content": full_content,  # Include full content for multi-modal
                    "facts": payload.get("facts"),
                    "score": pt.get("score", 0),
                    "source_type": payload.get("source_type", "prompt"),
                    "source_name": payload.get("source_name"),
                })
    
    except Exception:
        pass  # Search failure is non-fatal
    
    # Check if this conversation is worth saving via pragmatics classifier
    # Skip classifier for document chunks (always save those)
    if req.skip_classifier:
        worth_saving = True
        classifier_error = ""
    else:
        worth_saving, classifier_error = _is_worth_saving(req.messages)
    
    if classifier_error:
        # Classifier is down - report the error
        return {
            "status": "error",
            "point_id": None,
            "content_hash": None,
            "existing_context": existing_context if existing_context else None,
            "reason": classifier_error
        }
    
    if not worth_saving:
    
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
    content_hash = hashlib.md5(serialized_content.encode()).hexdigest()
    
    # Build payload with full content
    payload: Dict[str, Any] = {
        "user_id": req.user_id,
        "user_text": user_text,  # Keep for compatibility/search
        "full_content": serialized_content,  # New: full multi-modal content
    }
    
    # Add source information
    if req.source_type:
        payload["source_type"] = req.source_type
    if req.source_name:
        payload["source_name"] = req.source_name
    
    point_id = _make_uuid(req.user_id, content_hash)
    point = models.PointStruct(id=point_id, vector=storage_vector, payload=payload)  # Use storage_vector (full convo) for storage
    client.upsert(collection_name=collection_name, points=[point])
    
    # Single consolidated log
    ctx_count = len(existing_context) if existing_context else 0
    print(f"[/save] user={req.user_id} saved={True} context={ctx_count}")
    
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
    Embeds the query, searches Qdrant, returns matches filtered by user_id.
    """
    query_vec = embed(req.query_text)
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    
    try:
        session = _get_http_session()
        search_response = session.post(
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
        
        if not results:
            raise HTTPException(status_code=404, detail="No memories found")
        
        print(f"[/search] user={req.user_id} results={len(results)}")
        
        return [
            MemoryResult(
                user_text=pt.get("payload", {}).get("user_text", ""),
                messages=None,
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
    Get memory summaries matching a query.
    Like search but returns just the summary text, not full content.
    """
    query = req.query_text or "personal facts dates names preferences"
    query_vec = embed(query)
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")

    try:
        session = _get_http_session()
        search_response = session.post(
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
            timeout=10,
        )
        search_response.raise_for_status()
        data = search_response.json()
        results = data.get("result", [])

        summaries = [
            {"summary": pt.get("payload", {}).get("summary"), "score": pt.get("score", 0)}
            for pt in results
            if pt.get("payload", {}).get("summary")
        ]
        
        print(f"[/summaries] user={req.user_id} results={len(summaries)}")
        
        if not summaries:
            raise HTTPException(status_code=404, detail="No summaries found")

        return summaries
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summaries failed: {str(e)}")