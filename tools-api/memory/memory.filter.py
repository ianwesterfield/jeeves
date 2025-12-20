"""
Open-WebUI Memory Filter Plugin

Intercepts Open-WebUI conversations to:
  1. Extract and chunk uploaded documents (files, images, audio)
  2. Send chunks to memory service for embedding + storage
  3. Search for relevant past context
  4. Inject context into conversation before LLM sees it

Data Flow:
  User uploads file/image → Extract → Chunk → Save to Qdrant
           ↓
  User sends message → Search Qdrant for context → Inject if found
           ↓
  Modified conversation → LLM


Status Icons:
  ✨ Searching     🧠 Found/saved  📄 Document  💬 Prompt
  🖼 Image         ⏭ Skipped       ⚠ Error      ✔ Done
"""

import os
import re
import json
import base64
from typing import Optional, List, Tuple

import requests
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

MEMORY_API_URL = "http://memory_api:8000"
EXTRACTOR_API_URL = "http://extractor_api:8002"

# File type to MIME type mapping
CONTENT_TYPE_MAP = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".json": "application/json",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
}


# ============================================================================
# Content Extraction Layer
# ============================================================================

def _extract_images_from_messages(
    messages: List[dict],
    user_prompt: Optional[str] = None
) -> List[dict]:
    """
    Extract image URLs from message content and generate descriptions.
    
    Handles:
      - Data URLs (base64 inline): data:image/png;base64,XXX
      - HTTP URLs: https://example.com/image.png
      - Local file paths: /path/to/image.png
    
    Args:
        messages: List of message dicts from Open-WebUI
        user_prompt: Optional user text for context-aware descriptions
    
    Returns:
        List of image chunks with descriptions from extractor service
    """
    image_chunks = []
    
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        
        for idx, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_url":
                continue
            
            image_url = item.get("image_url", {})
            if isinstance(image_url, str):
                url = image_url
            else:
                url = image_url.get("url", "")
            
            if not url:
                continue
            
            try:
                content_type, image_data = _load_image_url(url)
                if not content_type or not image_data:
                    continue
                
                # Call extractor service for image description
                chunks = _call_extractor(
                    content=image_data,
                    content_type=content_type,
                    source_name=f"image_{idx}",
                    prompt=user_prompt,
                    overlap=0,
                )
                
                # Tag chunks with image metadata
                for chunk in chunks:
                    chunk["source_type"] = "image"
                    chunk["source_name"] = f"uploaded_image_{idx}"
                    image_chunks.append(chunk)
                
                if chunks:
                    desc_preview = chunks[0].get("content", "")[:100]
                    print(f"[filter] ✓ Image {idx}: {desc_preview}...")
                
            except Exception as e:
                print(f"[filter] ⚠ Image {idx} error: {e}")
                continue
    
    return image_chunks


def _load_image_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Load image from various sources (data URL, HTTP, local file).
    
    Returns:
        (content_type, base64_data) tuple or (None, None) if failed
    """
    
    # Data URL: data:image/png;base64,XXX
    if url.startswith("data:"):
        match = re.match(r'data:([^;]+);base64,(.+)', url)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    # HTTP URL: fetch remote image
    if url.startswith("http"):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/png")
            image_data = base64.b64encode(resp.content).decode("utf-8")
            return content_type, image_data
        except Exception as e:
            print(f"[filter] Failed to fetch {url}: {e}")
            return None, None
    
    # Local file path
    if os.path.exists(url):
        try:
            with open(url, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            ext = os.path.splitext(url)[1].lower()
            content_type = CONTENT_TYPE_MAP.get(ext, "image/png")
            return content_type, image_data
        except Exception as e:
            print(f"[filter] Failed to read {url}: {e}")
            return None, None
    
    print(f"[filter] Unknown image URL format: {url[:50]}...")
    return None, None


def _extract_and_chunk_file(file_path: str, filename: str) -> List[dict]:
    """
    Extract text from file and split into chunks.
    
    Uses extractor service to handle:
      - Text files: markdown, plain text, code
      - Documents: PDF
      - Media: images, audio
    
    Falls back to single-chunk on extraction failure.
    
    Args:
        file_path: Absolute path to file
        filename: Original filename for source tracking
    
    Returns:
        List of chunks or fallback single-chunk list
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Detect content type from extension
        ext = os.path.splitext(filename)[1].lower()
        content_type = CONTENT_TYPE_MAP.get(ext, "text/plain")
        
        # Text files: send as decoded string; binary: base64 encode
        if content_type.startswith("text/") or content_type == "application/json":
            content_str = content.decode("utf-8", errors="ignore")
        else:
            content_str = base64.b64encode(content).decode("utf-8")
        
        # Call extractor
        chunks = _call_extractor(
            content=content_str,
            content_type=content_type,
            source_name=filename,
            overlap=50,
        )
        
        return chunks
    
    except Exception as e:
        print(f"[filter] ⚠ Extractor error for {filename}: {e}")
        
        # Fallback: read whole file as single chunk
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return [{"content": f.read(), "chunk_index": 0, "chunk_type": "text"}]
        except Exception:
            return []


def _extract_file_contents(body: dict) -> Tuple[str, List[str], List[dict]]:
    """
    Extract and chunk all uploaded files from request.
    
    Files can be in body["files"] or body["metadata"]["files"].
    
    Args:
        body: Request body from Open-WebUI
    
    Returns:
        (combined_text, filenames, all_chunks) tuple
    """
    file_contents = []
    filenames = []
    all_chunks = []
    
    # Check both locations for files
    files = body.get("files") or []
    if not files:
        files = body.get("metadata", {}).get("files") or []
    
    for file_info in files:
        try:
            if not isinstance(file_info, dict):
                continue
            
            file_data = file_info.get("file", file_info)
            file_path = file_data.get("path")
            filename = file_data.get("filename", file_data.get("name", "unknown"))
            
            if not file_path or not os.path.exists(file_path):
                continue
            
            # Extract chunks from file
            chunks = _extract_and_chunk_file(file_path, filename)
            
            if chunks:
                filenames.append(filename)
                
                # Tag each chunk with source filename
                for chunk in chunks:
                    chunk["source_name"] = filename
                    all_chunks.append(chunk)
                
                # Collect combined text (first 3 chunks for embedding context)
                chunk_texts = [c.get("content", "") for c in chunks[:3]]
                file_contents.append(
                    f"[Document: {filename}]\n" + "\n\n".join(chunk_texts)
                )
        
        except Exception as e:
            print(f"[filter] ⚠ Error processing file: {e}")
            continue
    
    return "\n\n".join(file_contents), filenames, all_chunks


def _call_extractor(
    content: str,
    content_type: str,
    source_name: Optional[str],
    prompt: Optional[str] = None,
    overlap: int = 50,
) -> List[dict]:
    """
    Call extractor service to chunk content.
    
    Args:
        content: Text or base64-encoded content
        content_type: MIME type
        source_name: For source tracking
        prompt: Optional guided prompt for images
        overlap: Token overlap between chunks
    
    Returns:
        List of chunks from extractor response
    """
    try:
        resp = requests.post(
            f"{EXTRACTOR_API_URL}/api/extract/",
            json={
                "content": content,
                "content_type": content_type,
                "source_name": source_name,
                "chunk_size": 500,
                "chunk_overlap": overlap,
                "prompt": prompt,
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("chunks", [])
    
    except Exception as e:
        print(f"[filter] Extractor error: {e}")
        return []


# ============================================================================
# Context Formatting Layer
# ============================================================================

def _format_source(ctx: dict) -> str:
    """
    Format a memory source as a brief status line with icon.
    
    Returns icons: 📄 (doc), 🔗 (url), 🖼 (image), 💬 (prompt), 📝 (other)
    """
    source_type = ctx.get("source_type")
    source_name = ctx.get("source_name")
    
    if source_type == "document" and source_name:
        return f"\U0001F4C4 {source_name}"  # 📄
    
    if source_type == "url" and source_name:
        display = (source_name[:30] + "...") if len(source_name) > 30 else source_name
        return f"\U0001F517 {display}"  # 🔗
    
    if source_type == "image":
        return f"\U0001F5BC {source_name or 'image'}"  # 🖼
    
    if source_type == "prompt" and source_name:
        snippet = source_name[:40].replace("\n", " ")
        if len(source_name) > 40:
            snippet += "..."
        return f"\U0001F4AC \"{snippet}\""  # 💬
    
    # Fallback: extract from user_text
    user_text = ctx.get("user_text", "")
    if user_text:
        snippet = user_text[:60].replace("\n", " ")
        if len(user_text) > 60:
            snippet += "..."
        return f"\U0001F4AC \"{snippet}\""  # 💬
    
    return "\U0001F4DD memory"  # 📝


# ============================================================================
# API Communication Layer
# ============================================================================

async def _save_chunk_to_memory(
    user_id: str,
    chunk: dict,
    model: str,
    metadata: dict,
    source_name: str,
) -> bool:
    """
    Save a single chunk to memory service.
    
    Always saves documents/images (skips classifier).
    Returns True if successful.
    """
    try:
        chunk_content = chunk.get("content", "")
        chunk_idx = chunk.get("chunk_index", 0)
        section = chunk.get("section_title", "")
        chunk_type = chunk.get("source_type", "document_chunk")
        
        if not chunk_content.strip():
            return False
        
        payload = {
            "user_id": user_id,
            "messages": [{"role": "user", "content": chunk_content}],
            "model": model,
            "metadata": {**metadata, "chunk_index": chunk_idx, "section_title": section},
            "source_type": chunk_type,
            "source_name": f"{source_name}#{chunk_idx}" + (f" ({section})" if section else ""),
            "skip_classifier": True,  # Always save documents
        }
        
        resp = requests.post(f"{MEMORY_API_URL}/api/memory/save", json=payload, timeout=30)
        return resp.status_code == 200
    
    except Exception as e:
        print(f"[filter] Failed to save chunk: {e}")
        return False


async def _search_memory(
    user_id: str,
    messages: List[dict],
    model: str,
    metadata: dict,
    source_type: Optional[str],
    source_name: Optional[str],
) -> Tuple[str, List[dict]]:
    """
    Search memory and potentially save conversation.
    
    Returns:
        (status, existing_context) tuple
        Status: "saved", "saved_with_context", "context_found", "skipped", "error"
    """
    try:
        payload = {
            "user_id": user_id,
            "messages": messages,
            "model": model,
            "metadata": metadata,
            "source_type": source_type,
            "source_name": source_name,
        }
        
        resp = requests.post(f"{MEMORY_API_URL}/api/memory/save", json=payload, timeout=60)
        resp.raise_for_status()
        
        result = resp.json()
        status = result.get("status", "unknown")
        existing_context = result.get("existing_context") or []
        
        return status, existing_context
    
    except Exception as e:
        print(f"[filter] Search error: {e}")
        return "error", []


def _extract_user_text_prompt(messages: List[dict]) -> Optional[str]:
    """
    Extract user's text prompt from last user message.
    
    Handles both string and multi-modal (list) content.
    """
    if not messages:
        return None
    
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        
        content = msg.get("content")
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts) if text_parts else None
    
    return None


# ============================================================================
# Filter Plugin Class
# ============================================================================

class Filter:
    """
    Open-WebUI filter plugin for semantic memory.
    
    Implements inlet/outlet hooks to intercept conversations,
    extract context from attachments, and inject relevant memories.
    """
    
    class Valves(BaseModel):
        """Configuration (currently unused)."""
        pass
    
    def __init__(self):
        """Initialize filter with defaults."""
        self.valves = self.Valves()
        self.toggle = True
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICAKICAKICAKICAKICAGPHBHDGGGC3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBkPSJNMTIgMTh2LTUuMjVtMCAwYTYuMDEgNi4wMSAwIDAgMCAxLjUtLjE4OW0tMS41LjE4OWE2LjAxIDYuMDEgMCAwIDEtMS41LS4xODltMy43NSA3LjQ3OGExMi4wNiAxMi4wNiAwIDAgMS00LjUgMG0zLjc1IDIuMzgzYTE0LjQwNiAxNC40MDYgMCAwIDEtMyAwTTE0LjI1IDE4di0uMTkyYzAtLjk4My42NTgtMS44MjMgMS41MDgtMi4zMTZhNy41IDcuNSAwIDEgMC03LjUxNyAwYy44NS40OTMgMS41MDkgMS4zMzMgMS41MDkgMi4zMTZWMTgiIC8+Cgo8L3N2Zz4K"
        )
    
    def _inject_context(self, body: dict, context_items: List[dict]) -> dict:
        """
        Inject retrieved memories into conversation.
        
        Prepends context block before the last user message so LLM can see it.
        
        Args:
            body: Request body
            context_items: List of retrieved memory dicts
        
        Returns:
            Modified body with injected context
        """
        if not context_items:
            return body
        
        # Build context block from memories
        context_lines = []
        for ctx in context_items:
            user_text = ctx.get("user_text")
            if user_text:
                context_lines.append(f"- {user_text}")
        
        if not context_lines:
            return body
        
        context_block = (
            "### Previous conversation context ###\n"
            + "\n".join(context_lines)
            + "\n### End of context ###\n\n"
        )
        
        # Find last user message and prepend context
        messages = body.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                
                if isinstance(content, str):
                    messages[i]["content"] = context_block + content
                elif isinstance(content, list):
                    messages[i]["content"] = (
                        [{"type": "text", "text": context_block}] + content
                    )
                break
        
        body["messages"] = messages
        return body
    
    async def inlet(
        self,
        body: dict,
        __event_emitter__,
        __user__: Optional[dict] = None
    ) -> dict:
        """
        Main filter entry point. Intercepts conversation to save/search memories.
        
        Process:
          1. Extract uploaded files/images
          2. Save chunks to memory
          3. Search for relevant context
          4. Inject context if found
          5. Emit status updates
        
        Never blocks conversation even on errors.
        """
        try:
            user_id = __user__["id"] if __user__ and "id" in __user__ else "anonymous"
            messages = body.get("messages", [])
            model = body.get("model", "unknown")
            metadata = body.get("metadata", {})
            
            if not messages or not isinstance(messages, list):
                return body
            
            # Extract files and images
            await __event_emitter__({
                "type": "status",
                "data": {"description": "✨ Processing...", "done": False, "hidden": False}
            })
            
            file_content, filenames, chunks = _extract_file_contents(body)
            user_text_prompt = _extract_user_text_prompt(messages)
            image_chunks = _extract_images_from_messages(messages, user_prompt=user_text_prompt)
            immediate_image_context = [
                {
                    "user_text": f"[Image]: {img.get('content', '')}",
                    "source_type": "image",
                    "source_name": img.get("source_name", "image"),
                }
                for img in image_chunks
                if img.get("content")
            ]
            
            chunks.extend(image_chunks)
            
            # Save chunks
            chunks_saved = 0
            if chunks:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"📄 Saving {len(chunks)} chunks...",
                        "done": False,
                        "hidden": False,
                    }
                })
                
                source_name = filenames[0] if filenames else "attachment"
                for chunk in chunks:
                    if await _save_chunk_to_memory(
                        user_id, chunk, model, metadata, source_name
                    ):
                        chunks_saved += 1
            
            # Search memory
            source_type = "document" if chunks else "prompt"
            if chunks:
                source_name = filenames[0] if filenames else "attachment"
            else:
                content = messages[-1].get("content", "") if messages else ""
                source_name = (
                    (content[:50] + "...") if len(str(content)) > 50 else str(content)
                )
            
            status, context = await _search_memory(
                user_id, messages, model, metadata, source_type, source_name
            )
            
            # Merge immediate image context with retrieved
            if immediate_image_context:
                context = immediate_image_context + context
                if status == "skipped":
                    status = "context_found"
                elif status == "saved":
                    status = "saved_with_context"
            
            # Emit status and inject if needed
            if status == "saved_with_context":
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"🧠 Saved + found {len(context)} memories",
                        "done": False,
                        "hidden": False,
                    }
                })
                for ctx in context:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"  • {_format_source(ctx)}", "done": False, "hidden": False}
                    })
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✔ Context injected", "done": True, "hidden": False}
                })
                body = self._inject_context(body, context)
            
            elif status == "context_found":
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"🧠 Found {len(context)} memories",
                        "done": False,
                        "hidden": False,
                    }
                })
                for ctx in context:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"  • {_format_source(ctx)}", "done": False, "hidden": False}
                    })
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✔ Context injected", "done": True, "hidden": False}
                })
                body = self._inject_context(body, context)
            
            elif status == "saved":
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "🧠 Memory saved", "done": True, "hidden": False}
                })
            
            elif status == "skipped":
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "⏭ Skipped", "done": True, "hidden": True}
                })
            
            else:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Memory: {status}", "done": True, "hidden": False}
                })
            
            print(f"[filter] user={user_id} status={status} context={len(context)} chunks={chunks_saved}")
            return body
        
        except Exception as e:
            print(f"[filter] Error: {e}")
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"⚠ Error: {str(e)[:40]}", "done": True, "hidden": False}
            })
            return body
    
    async def outlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> None:
        """Post-response hook. Currently no-op, could save assistant responses."""
        pass
