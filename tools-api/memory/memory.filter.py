"""
Open-WebUI Memory Filter

This is my filter plugin for Open-WebUI. It intercepts every conversation,
sends it to my memory API for storage and retrieval, then injects any
relevant context back in before the model sees it.

The flow:
  1. User sends message
  2. inlet() grabs it
  3. Reads any attached files
  4. For documents: calls extractor to chunk, then saves each chunk
  5. For prompts: hits /api/memory/save for search + storage
  6. If we found context, inject it
  7. Return modified body to Open-WebUI

Status icons:
  ✨ - Searching  🧠 - Found/saved  📄 - Doc source  💬 - Prompt source
  ⏭ - Skipped    ⚠ - Error
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
import requests
import json
import os
import hashlib
import base64
import re

# Service URLs
MEMORY_API_URL = "http://memory_api:8000"
EXTRACTOR_API_URL = "http://extractor_api:8002"


def _extract_images_from_messages(messages: List[dict], user_prompt: Optional[str] = None) -> List[dict]:
    """
    Extract images embedded in message content (image_url type items).
    Returns list of image chunks with descriptions from extractor.
    
    Args:
        messages: List of message dicts
        user_prompt: The user's text prompt for context-aware image descriptions
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
                # Handle data URLs (base64 encoded)
                if url.startswith("data:"):
                    # Parse data URL: data:image/png;base64,XXXXXX
                    match = re.match(r'data:([^;]+);base64,(.+)', url)
                    if match:
                        content_type = match.group(1)
                        image_data = match.group(2)
                    else:
                        continue
                elif url.startswith("http"):
                    # Fetch remote image
                    resp = requests.get(url, timeout=30)
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "image/png")
                    image_data = base64.b64encode(resp.content).decode("utf-8")
                elif os.path.exists(url):
                    # Local file path
                    with open(url, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(url)[1].lower()
                    content_type = {"png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}.get(ext, "image/png")
                else:
                    print(f"[filter] Skipping unknown image URL format: {url[:50]}...")
                    continue
                
                # Call extractor to get description
                resp = requests.post(
                    f"{EXTRACTOR_API_URL}/api/extract/",
                    json={
                        "content": image_data,
                        "content_type": content_type,
                        "source_name": f"image_{idx}",
                        "chunk_size": 500,
                        "chunk_overlap": 0,
                        "prompt": user_prompt  # Pass user's question for context-aware description
                    },
                    timeout=120
                )
                resp.raise_for_status()
                result = resp.json()
                
                chunks = result.get("chunks", [])
                for chunk in chunks:
                    chunk["source_type"] = "image"
                    chunk["source_name"] = f"uploaded_image_{idx}"
                    image_chunks.append(chunk)
                    
                print(f"[filter] Extracted description from image {idx}: {chunks[0].get('content', '')[:100] if chunks else 'no content'}...")
                
            except Exception as e:
                print(f"[filter] Error extracting image {idx}: {e}")
                continue
    
    return image_chunks


def _extract_and_chunk_file(file_path: str, filename: str) -> List[dict]:
    """
    Extract text and chunk a file using the extractor service.
    Returns list of chunks with content and metadata.
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Determine content type from extension
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
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
        content_type = content_type_map.get(ext, "text/plain")
        
        # For text files, send as string; for binary, base64 encode
        if content_type.startswith("text/") or content_type == "application/json":
            content_str = content.decode("utf-8", errors="ignore")
        else:
            import base64
            content_str = base64.b64encode(content).decode("utf-8")
        
        # Call extractor
        resp = requests.post(
            f"{EXTRACTOR_API_URL}/api/extract/",
            json={
                "content": content_str,
                "content_type": content_type,
                "source_name": filename,
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            timeout=120  # Allow time for image/audio processing
        )
        resp.raise_for_status()
        result = resp.json()
        
        return result.get("chunks", [])
        
    except Exception as e:
        print(f"[filter] Extractor error for {filename}: {e}")
        # Fallback: return whole file as single chunk
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return [{"content": content, "chunk_index": 0, "chunk_type": "text"}]
        except:
            return []


def _extract_file_contents(body: dict) -> Tuple[str, List[str], List[dict]]:
    """
    Read file contents from Open-WebUI's uploads.
    Returns (combined_content_string, filenames, all_chunks).
    
    Files can be in body["files"] or body["metadata"]["files"]
    """
    file_contents = []
    filenames = []
    all_chunks = []
    
    # Check both locations for files (handle None explicitly)
    files = body.get("files") or []
    if not files:
        files = body.get("metadata", {}).get("files") or []
    
    for file_info in files:
        try:
            if isinstance(file_info, dict):
                file_data = file_info.get("file", file_info)
                file_path = file_data.get("path")
                filename = file_data.get("filename", file_data.get("name", "unknown"))
                
                if file_path and os.path.exists(file_path):
                    # Get chunks from extractor
                    chunks = _extract_and_chunk_file(file_path, filename)
                    
                    if chunks:
                        filenames.append(filename)
                        # Add filename to each chunk for tracking
                        for chunk in chunks:
                            chunk["source_name"] = filename
                            all_chunks.append(chunk)
                        
                        # Also build combined content for embedding context
                        chunk_texts = [c.get("content", "") for c in chunks]
                        file_contents.append(f"[Document: {filename}]\n" + "\n\n".join(chunk_texts[:3]))  # First 3 chunks for context
                        
        except Exception as e:
            print(f"[filter] Error processing file: {e}")
            pass
    
    return "\n\n".join(file_contents), filenames, all_chunks


def _format_single_source(ctx: dict) -> str:
    """
    Format a single memory source for its own status line.
    Returns something like: 📄 runbook.md  or  💬 "My name is Ian..."
    """
    source_type = ctx.get("source_type")
    source_name = ctx.get("source_name")
    
    if source_type == "document" and source_name:
        return f"\U0001F4C4 {source_name}"  # 📄
    elif source_type == "url" and source_name:
        display_url = source_name[:50] + "..." if len(source_name) > 50 else source_name
        return f"\U0001F517 {display_url}"  # 🔗
    elif source_type == "image":
        return f"\U0001F5BC {source_name or 'image'}"  # 🖼
    elif source_type == "prompt" and source_name:
        snippet = source_name[:60].replace("\n", " ")
        if len(source_name) > 60:
            snippet += "..."
        return f"\U0001F4AC \"{snippet}\""  # 💬
    else:
        # Fallback: show user_text
        user_text = ctx.get("user_text", "")
        if user_text:
            snippet = user_text[:60].replace("\n", " ")
            if len(user_text) > 60:
                snippet += "..."
            return f"\U0001F4AC \"{snippet}\""  # 💬
        return "\U0001F4DD memory"  # 📝


def _format_source_description(context_items: list) -> str:
    """
    Format sources for UI status display. Uses icons:
      📄 doc | 🔗 url | 🖼 image | 💬 prompt | 📝 memory
    """
    sources = []
    for ctx in context_items[:3]:  # Limit to first 3 for brevity
        source_type = ctx.get("source_type")
        source_name = ctx.get("source_name")
        
        if source_type == "document" and source_name:
            sources.append(f"\U0001F4C4 {source_name}")  # 📄
        elif source_type == "url" and source_name:
            # Truncate long URLs
            display_url = source_name[:30] + "..." if len(source_name) > 30 else source_name
            sources.append(f"\U0001F517 {display_url}")  # 🔗
        elif source_type == "image":
            sources.append(f"\U0001F5BC {source_name or 'image'}")  # 🖼
        elif source_type == "prompt" and source_name:
            # source_name already contains a snippet for prompts
            snippet = source_name[:40].replace("\n", " ")
            if len(source_name) > 40:
                snippet += "..."
            sources.append(f"\U0001F4AC \"{snippet}\"")  # 💬
        else:
            # Fallback: try to show something useful from user_text
            user_text = ctx.get("user_text", "")
            if user_text:
                # Skip if it looks like extracted facts (starts with "key: value" pattern)
                if ": " in user_text[:50] and not user_text.startswith(("[", "http", "My ", "I ")):
                    sources.append("\U0001F4DD memory")  # 📝
                else:
                    snippet = user_text[:40].replace("\n", " ")
                    if len(user_text) > 40:
                        snippet += "..."
                    sources.append(f"\U0001F4AC \"{snippet}\"")  # 💬
            else:
                sources.append("\U0001F4DD memory")  # 📝
    
    return " | ".join(sources) if sources else "memory"


class Filter:
    """
    My Open-WebUI filter. Implements inlet/outlet to intercept conversations
    and talk to the memory API.
    """
    
    class Valves(BaseModel):
        """Config valves (unused for now)."""
        pass

    def __init__(self):
        """Set up defaults."""
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICAKICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1OC0xLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+Cg=="""

    def _inject_context(self, body: dict, existing_context: list) -> dict:
        """
        Inject retrieved memories into the conversation. Prepends a context block
        before the last user message so the model can see it.
        """
        if not existing_context:
            return body
        
        # Build context string from retrieved memories
        context_parts = []
        for ctx in existing_context:
            user_text = ctx.get("user_text")
            if user_text:
                context_parts.append(f"- {user_text}")
        
        if not context_parts:
            return body
        
        context_block = (
            "### Relevant information from previous conversations ###\n"
            + "\n".join(context_parts)
            + "\n### End of context ###\n\n"
        )
        
        messages = body.get("messages", [])
        
        # Find the last user message and prepend context to it
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                original_content = messages[i].get("content", "")
                if isinstance(original_content, str):
                    messages[i]["content"] = context_block + original_content
                elif isinstance(original_content, list):
                    messages[i]["content"] = [{"type": "text", "text": context_block}] + original_content
                break
        
        body["messages"] = messages
        return body

    async def inlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> dict:
        """
        Main entry point. Grabs the conversation, reads any attached files,
        hits the memory API, and injects context if we found any. Errors are
        caught and logged but never block the conversation.
        """
        try:
            user_id = __user__["id"] if __user__ and "id" in __user__ else "anonymous"
            messages = body.get("messages", [])
            model = body.get("model", "unknown")
            metadata = body.get("metadata", {})
            
            # Debug: log what we received
            files_in_body = body.get("files")
            files_in_meta = metadata.get("files")
            print(f"[filter] DEBUG body keys: {list(body.keys())}")
            print(f"[filter] DEBUG files in body: {files_in_body}")
            print(f"[filter] DEBUG files in metadata: {files_in_meta}")
            if messages:
                last_msg = messages[-1]
                print(f"[filter] DEBUG last message role: {last_msg.get('role')}")
                content = last_msg.get("content")
                if isinstance(content, list):
                    print(f"[filter] DEBUG content is list with {len(content)} items, types: {[c.get('type') for c in content]}")
            
            # Extract file contents and get chunks
            file_content, filenames, chunks = _extract_file_contents(body)
            
            # Extract user's text prompt from the last user message (for context-aware image description)
            user_text_prompt = None
            if messages:
                last_msg = messages[-1]
                if last_msg.get("role") == "user":
                    content = last_msg.get("content")
                    if isinstance(content, str):
                        user_text_prompt = content
                    elif isinstance(content, list):
                        # Extract text items from multi-modal content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        user_text_prompt = " ".join(text_parts) if text_parts else None
            
            # Also extract images embedded in messages (pass user prompt for context)
            image_chunks = _extract_images_from_messages(messages, user_prompt=user_text_prompt)
            immediate_image_context = []  # For injecting image descriptions immediately
            if image_chunks:
                print(f"[filter] Found {len(image_chunks)} image chunks from messages")
                chunks.extend(image_chunks)
                if not filenames:
                    filenames = ["uploaded_image"]
                # Collect image descriptions for immediate injection
                for img_chunk in image_chunks:
                    desc = img_chunk.get("content", "")
                    if desc:
                        immediate_image_context.append({
                            "user_text": f"[Image description]: {desc}",
                            "source_type": "image",
                            "source_name": img_chunk.get("source_name", "uploaded_image")
                        })
            
            source_type = None
            source_name = None
            
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "\U00002728 Searching memory...",  # ✨
                        "done": False,
                        "hidden": False,
                    },
                }
            )
            
            # If we have document chunks, save each one
            chunks_saved = 0
            if chunks:
                source_type = "document"
                source_name = filenames[0] if filenames else "attachment"
                
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"\U0001F4C4 Processing {len(chunks)} chunks from {source_name}...",  # 📄
                            "done": False,
                            "hidden": False,
                        },
                    }
                )
                
                # Save each chunk to memory
                for chunk in chunks:
                    chunk_content = chunk.get("content", "")
                    chunk_idx = chunk.get("chunk_index", 0)
                    section = chunk.get("section_title", "")
                    chunk_source = chunk.get("source_name", source_name)
                    chunk_type = chunk.get("source_type", "document_chunk")  # Preserve image type
                    
                    if not chunk_content.strip():
                        continue
                    
                    # Create a minimal message for this chunk
                    chunk_messages = [{
                        "role": "user",
                        "content": chunk_content
                    }]
                    
                    # Save to memory (skip classifier for documents/images - always save)
                    try:
                        chunk_payload = {
                            "user_id": user_id,
                            "messages": chunk_messages,
                            "model": model,
                            "metadata": {
                                **metadata,
                                "chunk_index": chunk_idx,
                                "section_title": section,
                                "document_name": chunk_source,
                            },
                            "source_type": chunk_type,  # Use actual chunk type (document_chunk or image)
                            "source_name": f"{chunk_source}#{chunk_idx}" + (f" ({section})" if section else ""),
                            "skip_classifier": True,  # Documents always get saved
                        }
                        resp = requests.post(f"{MEMORY_API_URL}/api/memory/save", json=chunk_payload, timeout=30)
                        if resp.status_code == 200:
                            chunks_saved += 1
                    except Exception as e:
                        print(f"[filter] Chunk save error: {e}")
                
                print(f"[filter] Saved {chunks_saved}/{len(chunks)} chunks from {source_name}")
            
            # Now do the regular search for context (whether or not we had docs)
            # Get the user's actual query for searching
            if not chunks:
                source_type = "prompt"
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            source_name = content[:50] + "..." if len(content) > 50 else content
                        break
            
            if not messages or not isinstance(messages, list):
                raise ValueError("Body must contain 'messages' array")
            
            # Send to memory API for search (and possibly save if it's a prompt worth saving)
            url = f"{MEMORY_API_URL}/api/memory/save"
            payload = {
                "user_id": user_id,
                "messages": messages,
                "model": model,
                "metadata": metadata,
                "source_type": source_type,
                "source_name": source_name,
            }
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            status = result.get("status", "unknown")
            existing_context = result.get("existing_context") or []
            
            # Merge immediate image context with retrieved context
            # Image descriptions should come first so the model sees them
            if immediate_image_context:
                existing_context = immediate_image_context + existing_context
                # Update status if we have image context
                if status == "skipped":
                    status = "context_found"
                elif status == "saved":
                    status = "saved_with_context"
            
            ctx_count = len(existing_context) if existing_context else 0
            
            # Single consolidated log line
            print(f"[filter] inlet: user={user_id} msgs={len(messages)} files={len(filenames)} chunks={chunks_saved} status={status} context={ctx_count} img_ctx={len(immediate_image_context)}")
            
            # Update status based on what happened
            if status == "saved_with_context":
                # Emit one status line per memory found
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"\U0001F9E0 Saved + Found {len(existing_context)} memories:",  # 🧠
                            "done": False,
                            "hidden": False,
                        },
                    }
                )
                for ctx in existing_context:
                    desc = _format_single_source(ctx)
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"  \u2022 {desc}",
                                "done": False,
                                "hidden": False,
                            },
                        }
                    )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "\u2714 Context injected",  # ✔
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
                # Inject context into conversation
                body = self._inject_context(body, existing_context)
                
            elif status == "context_found":
                # Emit one status line per memory found
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"\U0001F9E0 Found {len(existing_context)} memories:",  # 🧠
                            "done": False,
                            "hidden": False,
                        },
                    }
                )
                for ctx in existing_context:
                    desc = _format_single_source(ctx)
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"  \u2022 {desc}",
                                "done": False,
                                "hidden": False,
                            },
                        }
                    )
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "\u2714 Context injected",  # ✔
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
                # Inject context into conversation
                body = self._inject_context(body, existing_context)
                
            elif status == "saved":
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "\U0001F9E0 Memory saved",  # 🧠
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
                
            elif status == "skipped":
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "\U000023ED Skipped",  # ⏭
                            "done": True,
                            "hidden": True,
                        },
                    }
                )
            else:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Memory: {status}",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
            
            return body
            
        except Exception as e:
            print(f"[filter] Error: {e}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"\U000026A0 Memory error: {str(e)[:50]}",  # ⚠
                        "done": True,
                        "hidden": False,
                    },
                }
            )
            return body

    async def outlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> None:
        """Post-response hook. Currently a no-op but could save assistant responses later."""
        pass
