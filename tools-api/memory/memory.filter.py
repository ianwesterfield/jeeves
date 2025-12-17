"""
Open-WebUI Memory Filter Plugin

This filter integrates with Open-WebUI to provide semantic memory capabilities.
It intercepts all conversations, sends them to the memory API for storage and
retrieval, then injects any relevant context back into the conversation.

Workflow:
    1. User sends message to Open-WebUI
    2. inlet() intercepts the request
    3. Extracts any file attachments and reads their content
    4. Sends to memory API (/api/memory/save) for:
       - Semantic search for related memories
       - Fact extraction (names, preferences, etc.)
       - Storage if deemed worth saving by pragmatics classifier
    5. If context found, injects it into the conversation
    6. Returns modified body to Open-WebUI
    7. Model responds with full context available

Status Icons:
    ✨ (U+2728)  - Processing/searching memory
    🧠 (U+1F9E0) - Qdrant access (found/saved)
    📄 (U+1F4C4) - Document source
    💬 (U+1F4AC) - Prompt/conversation source
    ⏭ (U+23ED)  - Skipped (not worth saving)
    ⚠ (U+26A0)  - Error occurred

Configuration:
    The filter connects to memory_api:8000 (docker network hostname).
    No additional configuration required.

Attributes:
    title: Memory Service Tools
    version: 0.3
    author: open-webui
"""

from pydantic import BaseModel, Field
from typing import Optional
import requests
import json
import os

def _extract_file_contents(body: dict) -> tuple:
    """
    Extract file contents from Open-WebUI's file attachments.
    
    Open-WebUI stores uploaded files on disk and passes file metadata
    in the request body. This function reads the actual file content.
    
    Args:
        body: The Open-WebUI request body containing a "files" array.
              Each file entry has structure:
              {"type": "file", "file": {"path": "/app/.../file.txt", "filename": "doc.txt"}}
    
    Returns:
        tuple: (content_string, list_of_filenames)
            - content_string: All file contents concatenated with headers
            - list_of_filenames: Names of successfully read files
    
    Note:
        Files that cannot be read (missing, binary, etc.) are silently skipped.
        Content is prefixed with "[Document: filename]" for context.
    """
    file_contents = []
    filenames = []
    files = body.get("files", [])
    
    for file_info in files:
        try:
            # Open-WebUI structure: {'type': 'file', 'file': {..., 'path': '/app/backend/data/uploads/...'}}
            if isinstance(file_info, dict):
                file_data = file_info.get("file", file_info)
                file_path = file_data.get("path")
                filename = file_data.get("filename", file_data.get("name", "unknown"))
                
                if file_path and os.path.exists(file_path):
                    print(f"[filter] Reading file content from: {file_path}")
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        file_contents.append(f"[Document: {filename}]\n{content}")
                        filenames.append(filename)
                        print(f"[filter] Extracted {len(content)} chars from {filename}")
                else:
                    print(f"[filter] File path not found or missing: {file_path}")
        except Exception as e:
            print(f"[filter] Error reading file: {e}")
    
    return "\n\n".join(file_contents), filenames


def _format_source_description(context_items: list) -> str:
    """
    Format source descriptions for UI status display.
    
    Creates a human-readable summary of where retrieved memories came from,
    using icons and truncated identifiers.
    
    Args:
        context_items: List of memory context dicts with source_type,
                       source_name, and user_text fields.
    
    Returns:
        A formatted string like:
        "📄 runbook.md | 💬 'My name is Ian...'"
        
    Icons:
        📄 - Document (PDF, markdown, text file)
        🔗 - URL/link
        🖼 - Image
        💬 - Conversation/prompt
        📝 - Generic memory
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
    Open-WebUI Filter for Memory Service Integration.
    
    This class implements the Open-WebUI filter interface with inlet/outlet methods.
    It intercepts conversations, communicates with the memory API, and injects
    retrieved context back into the conversation flow.
    
    Attributes:
        valves: Configuration valves (unused, but required by interface).
        toggle: Whether the filter is enabled (always True).
        icon: Base64-encoded SVG icon for the UI.
    
    Methods:
        inlet: Process incoming messages before model inference.
        outlet: Process after model response (currently no-op).
        _inject_context: Insert retrieved memories into conversation.
    """
    
    class Valves(BaseModel):
        """Filter configuration (currently unused)."""
        pass

    def __init__(self):
        """Initialize the filter with default settings."""
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICAKICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1OC0xLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+Cg=="""

    def _inject_context(self, body: dict, existing_context: list) -> dict:
        """
        Inject retrieved memory context into the conversation.
        
        Prepends context as a formatted block before the last user message.
        This makes the relevant memories available to the model without
        modifying the visible conversation history.
        
        Args:
            body: The Open-WebUI request body with "messages" array.
            existing_context: List of memory dicts with "user_text" fields.
        
        Returns:
            Modified body with context injected, or original body if no context.
        
        Format:
            ### Relevant information from previous conversations ###
            - [memory 1]
            - [memory 2]
            ### End of context ###
            
            [original user message]
        """
        print(f"[filter] _inject_context called with {len(existing_context) if existing_context else 0} items")
        
        if not existing_context:
            print("[filter] No existing_context, returning body unchanged")
            return body
        
        # Debug: print what we received
        for i, ctx in enumerate(existing_context):
            print(f"[filter] Context item {i}: {ctx}")
        
        # Build context string from retrieved memories
        context_parts = []
        for ctx in existing_context:
            user_text = ctx.get("user_text")
            if user_text:
                context_parts.append(f"- {user_text}")
        
        if not context_parts:
            print("[filter] No context_parts extracted, returning body unchanged")
            return body
        
        context_block = (
            "### Relevant information from previous conversations ###\n"
            + "\n".join(context_parts)
            + "\n### End of context ###\n\n"
        )
        print(f"[filter] Injecting context: {context_block[:200]}...")
        
        messages = body.get("messages", [])
        
        # Find the last user message and prepend context to it
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                original_content = messages[i].get("content", "")
                # Handle content that might be a list (multi-modal)
                if isinstance(original_content, str):
                    messages[i]["content"] = context_block + original_content
                elif isinstance(original_content, list):
                    # For multi-modal, prepend as text item
                    messages[i]["content"] = [{"type": "text", "text": context_block}] + original_content
                print(f"[filter] Injected into user message at index {i}")
                break
        
        body["messages"] = messages
        return body

    async def inlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> dict:
        """
        Process incoming messages before model inference.
        
        This is the main entry point for the filter. It:
        1. Extracts user ID and messages from the request
        2. Reads any attached file contents from disk
        3. Sends to memory API for search and storage
        4. Injects found context back into the conversation
        5. Emits status updates for the UI
        
        Args:
            body: The Open-WebUI request body containing:
                  - messages: List of conversation messages
                  - model: The LLM model ID
                  - metadata: Additional request metadata
                  - files: Optional file attachments
            __event_emitter__: Async function to emit status updates to UI.
            __user__: Optional user info dict with "id" field.
        
        Returns:
            Modified body with injected context, or original on error.
        
        Note:
            Errors are caught and logged, but never block the conversation.
            On error, the original body is returned unchanged.
        """
        try:
            print("[filter] inlet called")
            user_id = __user__["id"] if __user__ and "id" in __user__ else "anonymous"
            
            # Extract messages array
            messages = body.get("messages", [])
            model = body.get("model", "unknown")
            metadata = body.get("metadata", {})
            print(f"[filter] user_id={user_id}, model={model}, messages={len(messages)}")
            
            # Debug: Log full body structure to understand what Open-WebUI sends
            import json
            print(f"[filter] DEBUG body keys: {list(body.keys())}")
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                if isinstance(content, str):
                    print(f"[filter] msg[{i}] role={msg.get('role')} content_len={len(content)} preview={content[:100]}...")
                elif isinstance(content, list):
                    print(f"[filter] msg[{i}] role={msg.get('role')} content is list with {len(content)} items")
                    for j, item in enumerate(content):
                        if isinstance(item, dict):
                            print(f"[filter]   item[{j}] type={item.get('type')} keys={list(item.keys())}")
            
            # Check for file attachments in metadata or elsewhere
            if "files" in body:
                print(f"[filter] DEBUG: body has 'files' key with {len(body['files'])} items")
            if "files" in metadata:
                print(f"[filter] DEBUG: metadata has 'files' key")
            
            # Extract file contents and append to messages
            file_content, filenames = _extract_file_contents(body)
            source_type = None
            source_name = None
            
            if file_content:
                print(f"[filter] Extracted file content ({len(file_content)} chars), adding to messages")
                # Add file content as a supplementary user message
                messages = messages.copy()  # Don't modify original
                messages.append({
                    "role": "user",
                    "content": f"[Attached Document Content]\n{file_content}"
                })
                # Set source info for documents
                source_type = "document"
                source_name = filenames[0] if filenames else "attachment"
            else:
                # For regular prompts, create a snippet as source_name
                source_type = "prompt"
                last_user_msg = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            last_user_msg = content
                        break
                if last_user_msg:
                    source_name = last_user_msg[:50] + "..." if len(last_user_msg) > 50 else last_user_msg
            
            if not messages or not isinstance(messages, list):
                raise ValueError("Body must contain 'messages' array")
            
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
            
            # Send to memory API
            url = "http://memory_api:8000/api/memory/save"
            payload = {
                "user_id": user_id,
                "messages": messages,
                "model": model,
                "metadata": metadata,
                "source_type": source_type,
                "source_name": source_name,
            }
            print(f"[filter] POST {url}")
            response = requests.post(url, json=payload, timeout=60)
            print(f"[filter] memory_api response status: {response.status_code}")
            response.raise_for_status()
            
            # Parse response to check for context
            result = response.json()
            status = result.get("status", "unknown")
            existing_context = result.get("existing_context")
            reason = result.get("reason", "")
            
            print(f"[filter] memory_api status={status}, context_items={len(existing_context) if existing_context else 0}")
            
            # Update status based on what happened
            if status == "saved_with_context":
                source_desc = _format_source_description(existing_context)
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"\U0001F9E0 Saved + Found {len(existing_context)}: {source_desc}",  # 🧠
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
                # Inject context into conversation
                body = self._inject_context(body, existing_context)
                
            elif status == "context_found":
                source_desc = _format_source_description(existing_context)
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"\U0001F9E0 Found {len(existing_context)}: {source_desc}",  # 🧠
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
        """
        Process after model response (outlet filter).
        
        Currently a no-op. Could be extended to:
        - Save assistant responses to memory
        - Extract facts from model output
        - Update conversation summaries
        
        Args:
            body: The response body from the model.
            __event_emitter__: Async function to emit status updates.
            __user__: Optional user info dict.
        """
        pass
