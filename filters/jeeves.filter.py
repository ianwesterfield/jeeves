"""
Jeeves Filter - Agentic Reasoning & Memory Integration

Open-WebUI filter that provides:
  - Intent classification (recall, save, task)
  - Multi-step reasoning via orchestrator
  - Code execution via executor
  - Semantic memory retrieval and storage
"""

import os
import re
import json
import base64
from typing import Optional, List, Tuple, Dict, Any

import requests
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://jeeves:8000")
EXTRACTOR_API_URL = os.getenv("EXTRACTOR_API_URL", "http://extractor_api:8002")
ORCHESTRATOR_API_URL = os.getenv("ORCHESTRATOR_API_URL", "http://orchestrator_api:8004")
EXECUTOR_API_URL = os.getenv("EXECUTOR_API_URL", "http://executor_api:8005")
PRAGMATICS_API_URL = os.getenv("PRAGMATICS_API_URL", "http://pragmatics_api:8001")

# System prompt describing Jeeves capabilities
JEEVES_SYSTEM_PROMPT = """You are Jeeves, an AI assistant with access to the user's workspace and semantic memory.

**Your Capabilities:**
- **Workspace Access**: You can read, list, and edit files in the user's workspace
- **File Editing**: You can make surgical edits (replace text, insert at position, append)
- **Memory**: You remember information about the user across conversations
- **Code Execution**: When enabled, you can run Python, PowerShell, and Node.js code

**When the user asks to edit/modify files:**
- If you see "### File Operation Result ###" below, the edit has ALREADY BEEN EXECUTED
- Report the result to the user - don't claim you'll do it, confirm it's done
- If the operation failed, explain why and suggest alternatives

**When the user asks about files or workspace:**
- If workspace context is provided below, use that information to answer
- Be specific about file names, paths, and contents
- If no workspace context is provided, explain you need workspace access configured

**Important**: When Jeeves Analysis or Workspace Results are included in the conversation, use that real data to answer the user's question directly.
"""

# File type to MIME type mapping
CONTENT_TYPE_MAP = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
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
# Intent Classification
# ============================================================================

def _classify_intent(text: str) -> Dict[str, Any]:
    """
    Classify user intent using pragmatics 4-class model.
    
    Returns:
        {"intent": "recall"|"save"|"task"|"casual", "confidence": float}
    """
    try:
        resp = requests.post(
            f"{PRAGMATICS_API_URL}/api/pragmatics/classify",
            json={"text": text},
            timeout=5,
        )
        if resp.status_code == 200:
            result = resp.json()
            intent = result.get("intent", "casual")
            confidence = result.get("confidence", 0.5)
            print(f"[jeeves] Pragmatics: {intent} ({confidence:.2f})")
            return {"intent": intent, "confidence": confidence}
    except Exception as e:
        print(f"[jeeves] Pragmatics API error: {e}")
    
    # Fallback if pragmatics unavailable - very basic heuristics
    print("[jeeves] Warning: Pragmatics unavailable, using basic fallback")
    return {"intent": "casual", "confidence": 0.3}


# ============================================================================
# Workspace Operations (Direct Execution)
# ============================================================================

def _list_workspace_files(workspace_root: str, max_depth: int = 2) -> Optional[str]:
    """
    List files in workspace via executor API.
    
    Returns formatted file listing or None on error.
    """
    try:
        # First try file listing endpoint
        resp = requests.post(
            f"{EXECUTOR_API_URL}/api/execute/file",
            json={
                "operation": "list",
                "path": workspace_root,
            },
            timeout=15,
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                items = result.get("data", [])
                if items:
                    lines = []
                    for item in items[:100]:
                        prefix = "📁 " if item.get("type") == "dir" else "📄 "
                        lines.append(f"{prefix}{item.get('name', '?')}")
                    return "\n".join(lines)
        
        # Fallback: use Python code execution for recursive listing
        resp = requests.post(
            f"{EXECUTOR_API_URL}/api/execute/code",
            json={
                "language": "python",
                "code": f'''
import os
from pathlib import Path

workspace = Path("{workspace_root}")
if workspace.exists():
    files = []
    for item in sorted(workspace.rglob("*")):
        try:
            rel = item.relative_to(workspace)
            if len(rel.parts) <= {max_depth}:
                prefix = "📁 " if item.is_dir() else "📄 "
                files.append(prefix + str(rel))
        except:
            pass
    print("\\n".join(files[:100]) if files else "No files found")
else:
    print(f"Workspace not found: {workspace_root}")
''',
                "timeout": 10,
            },
            timeout=15,
        )
        
        if resp.status_code == 200:
            result = resp.json()
            return result.get("stdout", result.get("output", ""))
            
    except Exception as e:
        print(f"[jeeves] File listing error: {e}")
    
    return None


def _read_workspace_file(workspace_root: str, filepath: str, max_lines: int = 100) -> Optional[str]:
    """
    Read a file from the workspace.
    
    Returns file content or None on error.
    """
    try:
        resp = requests.post(
            f"{EXECUTOR_API_URL}/api/execute/file",
            json={
                "operation": "read",
                "path": filepath,
            },
            timeout=10,
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                content = result.get("data", "")
                # Truncate if too long
                lines = content.split("\n")
                if len(lines) > max_lines:
                    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
                return content
            
    except Exception as e:
        print(f"[jeeves] File read error: {e}")
    
    return None


def _write_workspace_file(workspace_root: str, filepath: str, content: str) -> Tuple[bool, str]:
    """
    Write content to a file in the workspace (full replacement).
    
    Returns (success, message).
    """
    try:
        resp = requests.post(
            f"{EXECUTOR_API_URL}/api/execute/tool",
            json={
                "tool": "write_file",
                "params": {
                    "path": filepath,
                    "content": content,
                },
                "workspace_context": {
                    "workspace_root": workspace_root,
                    "cwd": workspace_root,
                    "allow_file_write": True,
                    "allow_shell_commands": False,
                    "allowed_languages": [],
                },
            },
            timeout=15,
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                return True, result.get("output", "File written successfully")
            else:
                return False, result.get("error", "Write failed")
                
    except Exception as e:
        print(f"[jeeves] File write error: {e}")
        return False, str(e)
    
    return False, "Unknown error"


def _replace_in_workspace_file(
    workspace_root: str, 
    filepath: str, 
    old_text: str, 
    new_text: str
) -> Tuple[bool, str]:
    """
    Surgical replace: find old_text and replace with new_text.
    
    Returns (success, message).
    """
    try:
        resp = requests.post(
            f"{EXECUTOR_API_URL}/api/execute/tool",
            json={
                "tool": "replace_in_file",
                "params": {
                    "path": filepath,
                    "old_text": old_text,
                    "new_text": new_text,
                },
                "workspace_context": {
                    "workspace_root": workspace_root,
                    "cwd": workspace_root,
                    "allow_file_write": True,
                    "allow_shell_commands": False,
                    "allowed_languages": [],
                },
            },
            timeout=15,
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                return True, result.get("output", "Replacement successful")
            else:
                return False, result.get("error", "Replace failed")
                
    except Exception as e:
        print(f"[jeeves] File replace error: {e}")
        return False, str(e)
    
    return False, "Unknown error"


def _insert_in_workspace_file(
    workspace_root: str,
    filepath: str,
    position: str,
    text: str,
    anchor: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Insert text at position in file.
    
    position: "start", "end", "before", "after"
    anchor: Required for "before"/"after" - the text to search for
    
    Returns (success, message).
    """
    try:
        params = {
            "path": filepath,
            "position": position,
            "text": text,
        }
        if anchor:
            params["anchor"] = anchor
            
        resp = requests.post(
            f"{EXECUTOR_API_URL}/api/execute/tool",
            json={
                "tool": "insert_in_file",
                "params": params,
                "workspace_context": {
                    "workspace_root": workspace_root,
                    "cwd": workspace_root,
                    "allow_file_write": True,
                    "allow_shell_commands": False,
                    "allowed_languages": [],
                },
            },
            timeout=15,
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if result.get("success"):
                return True, result.get("output", "Insert successful")
            else:
                return False, result.get("error", "Insert failed")
                
    except Exception as e:
        print(f"[jeeves] File insert error: {e}")
        return False, str(e)
    
    return False, "Unknown error"


def _append_to_workspace_file(workspace_root: str, filepath: str, content: str) -> Tuple[bool, str]:
    """
    Append content to end of file.
    
    Returns (success, message).
    """
    return _insert_in_workspace_file(workspace_root, filepath, "end", content)


# ============================================================================
# Orchestrator Integration
# ============================================================================

async def _orchestrate_task(
    user_id: str,
    messages: List[dict],
    workspace_root: Optional[str],
    __event_emitter__,
) -> Optional[str]:
    """
    Handle task intents with direct execution for common operations.
    
    For simple file operations, executes directly.
    For complex tasks, delegates to orchestrator for planning.
    
    Returns context string to inject into conversation.
    """
    user_text = _extract_user_text_prompt(messages) or ""
    original_user_text = user_text  # Keep original case for content extraction
    user_text_lower = user_text.lower()
    
    # Direct execution for common workspace operations
    if workspace_root:
        
        # ================================================================
        # EDIT OPERATIONS - Check these FIRST before read/list!
        # ================================================================
        # Detect edit intent: add/insert/append/update/edit + file reference
        edit_verbs = r'(?:add|insert|append|put|write|include|create)'
        file_refs = r'(?:readme|architecture|docker-compose|\w+\.\w+)'
        
        edit_intent_match = re.search(
            rf'{edit_verbs}\s+(?:a\s+)?(.+?)\s+(?:to|in|into)\s+(?:the\s+)?(?:file\s+)?({file_refs})',
            original_user_text,
            re.IGNORECASE
        )
        
        if edit_intent_match:
            content_description = edit_intent_match.group(1).strip()
            file_ref = edit_intent_match.group(2).strip()
            
            # Normalize file references
            file_ref_lower = file_ref.lower()
            if "readme" in file_ref_lower:
                target_file = "README.md"
            elif "architecture" in file_ref_lower:
                target_file = "ARCHITECTURE.md"
            elif "docker-compose" in file_ref_lower or "docker compose" in file_ref_lower:
                target_file = "docker-compose.yaml"
            elif "gitignore" in file_ref_lower:
                target_file = ".gitignore"
            else:
                target_file = file_ref
            
            target_path = f"{workspace_root}/{target_file}"
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"✏️ Editing {target_file}...", "done": False, "hidden": False}
            })
            
            # For natural language requests, construct the content
            if "credit" in content_description.lower():
                # Get user's name from memories if available
                user_name = "Ian"  # TODO: extract from memory context
                content_to_add = f"\n\n## Credits\n\n- **{user_name}** - Creator and maintainer\n"
            elif "contributor" in content_description.lower():
                user_name = "Ian"
                content_to_add = f"\n\n## Contributors\n\n- **{user_name}**\n"
            else:
                # Use the description as content
                content_to_add = f"\n{content_description}\n"
            
            success, msg = _append_to_workspace_file(workspace_root, target_path, content_to_add)
            
            if success:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"✅ Updated {target_file}", "done": True, "hidden": False}
                })
                
                # Read back to show result
                new_content = _read_workspace_file(workspace_root, target_path, max_lines=50)
                
                return f"""### File Operation Result ###
**Operation:** Append to {target_file}
**Status:** ✅ Success
**Added:**
```
{content_to_add}
```

**Updated file (last 50 lines):**
```
{new_content or "(unable to read back)"}
```
### End File Operation ###

"""
            else:
                return f"""### File Operation Result ###
**Operation:** Append to {target_file}
**Status:** ❌ Failed
**Error:** {msg}
### End File Operation ###

"""
        
        # Pattern: Replace "X" with "Y" in file.txt
        replace_match = re.search(
            r'replace\s+["\'](.+?)["\']\s+with\s+["\'](.+?)["\']\s+in\s+(?:the\s+)?(?:file\s+)?["\']?(\w+\.\w+)["\']?',
            original_user_text,
            re.IGNORECASE | re.DOTALL
        )
        
        if replace_match:
            old_text = replace_match.group(1)
            new_text = replace_match.group(2)
            target_file = replace_match.group(3)
            target_path = f"{workspace_root}/{target_file}"
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"✏️ Replacing in {target_file}...", "done": False, "hidden": False}
            })
            
            success, msg = _replace_in_workspace_file(workspace_root, target_path, old_text, new_text)
            
            if success:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"✅ Updated {target_file}", "done": True, "hidden": False}
                })
                
                return f"""### File Operation Result ###
**Operation:** Replace in {target_file}
**Status:** ✅ Success
**Changed:** `{old_text}` → `{new_text}`
### End File Operation ###

"""
            else:
                return f"""### File Operation Result ###
**Operation:** Replace in {target_file}
**Status:** ❌ Failed
**Error:** {msg}
### End File Operation ###

"""
        
        # ================================================================
        # READ OPERATIONS - Only if not an edit request
        # ================================================================
        # List files request - but exclude edit verbs
        list_keywords = [
            "list files", "show files", "what files", "list the files", "files in",
            "summarize", "analyze", "read the", "look at the", "check the",
            "what's in", "whats in", "contents of", "structure of",
            "show me the readme", "show me the architecture"
        ]
        if any(kw in user_text_lower for kw in list_keywords):
            await __event_emitter__({
                "type": "status",
                "data": {"description": "📂 Listing workspace files...", "done": False, "hidden": False}
            })
            
            file_listing = _list_workspace_files(workspace_root)
            
            if file_listing:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Workspace scanned", "done": False, "hidden": False}
                })
                
                context_parts = [f"""### Workspace Files ###
**Workspace:** `{workspace_root}`

```
{file_listing}
```
### End Workspace Files ###
"""]
                
                # If user asked to read/show specific files, also read key documentation files
                if any(kw in user_text_lower for kw in ["summarize", "analyze", "read the", "look at", "show me", "print the", "display the", "readme", "architecture"]):
                    doc_files = ["README.md", "ARCHITECTURE.md", "docker-compose.yaml"]
                    for doc in doc_files:
                        doc_path = f"{workspace_root}/{doc}"
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"📖 Reading {doc}...", "done": False, "hidden": False}
                        })
                        content = _read_workspace_file(workspace_root, doc_path, max_lines=200)
                        if content:
                            context_parts.append(f"""### File: {doc} ###
```
{content}
```
### End {doc} ###
""")
                
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Context gathered", "done": True, "hidden": False}
                })
                
                return "\n".join(context_parts)
            else:
                return f"""### Workspace Error ###
Could not list files in workspace: `{workspace_root}`
The executor service may not be running or the path may not exist.
### End Workspace Error ###

"""
        
        # Read file request - require explicit file path patterns
        # Must have file extension or path separator to avoid matching phrases like "read each"
        file_read_match = re.search(r'(?:read|show|display|cat|open)\s+(?:the\s+)?(?:file\s+)?["\']?([^\s"\']+\.[a-zA-Z0-9]+|/[^\s"\']+)["\']?', user_text_lower)
        if file_read_match:
            filepath = file_read_match.group(1)
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"📖 Reading {filepath}...", "done": False, "hidden": False}
            })
            
            content = _read_workspace_file(workspace_root, filepath)
            
            if content:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ File read", "done": True, "hidden": False}
                })
                
                return f"""### File Content: {filepath} ###
```
{content}
```
### End File Content ###

"""
    
    # For other tasks, try the orchestrator planning service
    try:
        if workspace_root:
            requests.post(
                f"{ORCHESTRATOR_API_URL}/api/orchestrator/set-workspace",
                json={"cwd": workspace_root, "user_id": user_id},
                timeout=10,
            )
        
        resp = requests.post(
            f"{ORCHESTRATOR_API_URL}/api/orchestrator/next-step",
            json={
                "task": user_text,
                "user_id": user_id,
                "conversation_history": messages[-10:],
            },
            timeout=30,
        )
        
        if resp.status_code != 200:
            return None
        
        result = resp.json()
        steps = result.get("steps", [])
        reasoning = result.get("reasoning", "")
        
        if not steps and not reasoning:
            return None
        
        await __event_emitter__({
            "type": "status",
            "data": {"description": f"🤔 Planning: {len(steps)} step(s)", "done": False, "hidden": False}
        })
        
        step_descriptions = [f"{i+1}. [{s.get('tool', '?')}] {s.get('description', '')}" 
                           for i, s in enumerate(steps)]
        
        return f"""### Jeeves Analysis ###
{reasoning}

Suggested approach:
{chr(10).join(step_descriptions) if step_descriptions else "Direct response recommended."}
### End Analysis ###

"""
        
    except Exception as e:
        print(f"[jeeves] Orchestrator error: {e}")
        return None


# ============================================================================
# Content Extraction Layer
# ============================================================================

def _extract_images_from_messages(
    messages: List[dict],
    user_prompt: Optional[str] = None
) -> List[dict]:
    """
    Extract image URLs from message content and generate descriptions.
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
                
                chunks = _call_extractor(
                    content=image_data,
                    content_type=content_type,
                    source_name=f"image_{idx}",
                    prompt=user_prompt,
                    overlap=0,
                )
                
                for chunk in chunks:
                    chunk["source_type"] = "image"
                    chunk["source_name"] = f"uploaded_image_{idx}"
                    image_chunks.append(chunk)
                
            except Exception as e:
                print(f"[jeeves] Image {idx} error: {e}")
                continue
    
    return image_chunks


def _load_image_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Load image from various sources (data URL, HTTP, local file)."""
    
    if url.startswith("data:"):
        match = re.match(r'data:([^;]+);base64,(.+)', url)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    if url.startswith("http"):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/png")
            image_data = base64.b64encode(resp.content).decode("utf-8")
            return content_type, image_data
        except Exception as e:
            print(f"[jeeves] Failed to fetch {url}: {e}")
            return None, None
    
    if os.path.exists(url):
        try:
            with open(url, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(url)[1].lower()
            content_type = CONTENT_TYPE_MAP.get(ext, "image/png")
            return content_type, image_data
        except Exception as e:
            print(f"[jeeves] Failed to read {url}: {e}")
            return None, None
    
    return None, None


def _extract_and_chunk_file(file_path: str, filename: str) -> List[dict]:
    """Extract text from file and split into chunks."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
        
        ext = os.path.splitext(filename)[1].lower()
        content_type = CONTENT_TYPE_MAP.get(ext, "text/plain")
        
        if content_type.startswith("text/") or content_type == "application/json":
            content_str = content.decode("utf-8", errors="ignore")
        else:
            content_str = base64.b64encode(content).decode("utf-8")
        
        chunks = _call_extractor(
            content=content_str,
            content_type=content_type,
            source_name=filename,
            overlap=50,
        )
        
        return chunks
    
    except Exception as e:
        print(f"[jeeves] Extractor error for {filename}: {e}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return [{"content": f.read(), "chunk_index": 0, "chunk_type": "text"}]
        except Exception:
            return []


def _extract_file_contents(body: dict) -> Tuple[str, List[str], List[dict]]:
    """Extract and chunk all uploaded files from request."""
    file_contents = []
    filenames = []
    all_chunks = []
    
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
            
            chunks = _extract_and_chunk_file(file_path, filename)
            
            if chunks:
                filenames.append(filename)
                
                for chunk in chunks:
                    chunk["source_name"] = filename
                    all_chunks.append(chunk)
                
                chunk_texts = [c.get("content", "") for c in chunks[:3]]
                file_contents.append(
                    f"[Document: {filename}]\n" + "\n\n".join(chunk_texts)
                )
        
        except Exception as e:
            print(f"[jeeves] Error processing file: {e}")
            continue
    
    return "\n\n".join(file_contents), filenames, all_chunks


def _call_extractor(
    content: str,
    content_type: str,
    source_name: Optional[str],
    prompt: Optional[str] = None,
    overlap: int = 50,
) -> List[dict]:
    """Call extractor service to chunk content."""
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
        print(f"[jeeves] Extractor error: {e}")
        return []


# ============================================================================
# Memory Layer
# ============================================================================

async def _save_chunk_to_memory(
    user_id: str,
    chunk: dict,
    model: str,
    metadata: dict,
    source_name: str,
) -> bool:
    """Save a single chunk to memory service."""
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
            "skip_classifier": True,
        }
        
        resp = requests.post(f"{MEMORY_API_URL}/api/memory/save", json=payload, timeout=30)
        return resp.status_code == 200
    
    except Exception as e:
        print(f"[jeeves] Failed to save chunk: {e}")
        return False


async def _search_memory(
    user_id: str,
    messages: List[dict],
    model: str,
    metadata: dict,
    source_type: Optional[str],
    source_name: Optional[str],
) -> Tuple[str, List[dict]]:
    """Search memory and potentially save conversation."""
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
        print(f"[jeeves] Search error: {e}")
        return "error", []


def _extract_user_text_prompt(messages: List[dict]) -> Optional[str]:
    """Extract user's text prompt from last user message."""
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
# Context Formatting
# ============================================================================

def _format_source(ctx: dict) -> str:
    """Format a memory source as a brief status line with icon."""
    source_type = ctx.get("source_type")
    source_name = ctx.get("source_name")
    
    if source_type == "document" and source_name:
        return f"📄 {source_name}"
    
    if source_type == "url" and source_name:
        display = (source_name[:30] + "...") if len(source_name) > 30 else source_name
        return f"🔗 {display}"
    
    if source_type == "image":
        return f"🖼 {source_name or 'image'}"
    
    if source_type == "prompt" and source_name:
        snippet = source_name[:40].replace("\n", " ")
        if len(source_name) > 40:
            snippet += "..."
        return f'💬 "{snippet}"'
    
    user_text = ctx.get("user_text", "")
    if user_text:
        snippet = user_text[:60].replace("\n", " ")
        if len(user_text) > 60:
            snippet += "..."
        return f'💬 "{snippet}"'
    
    return "📝 memory"


# ============================================================================
# Filter Plugin Class
# ============================================================================

class Filter:
    """
    Jeeves - Agentic Assistant Filter for Open-WebUI.
    
    Provides:
      - Intent classification (recall, save, task, casual)
      - Multi-step reasoning via orchestrator
      - Code execution via executor
      - Semantic memory retrieval and storage
      - Document/image extraction
    """
    
    class UserValves(BaseModel):
        """User configuration for Jeeves."""
        enable_orchestrator: bool = Field(
            default=True,
            description="Enable multi-step reasoning for task intents"
        )
        enable_code_execution: bool = Field(
            default=False,
            description="Allow code execution (requires explicit enable)"
        )
        max_context_items: int = Field(
            default=5,
            description="Maximum memory items to inject as context"
        )
        workspace_root: str = Field(
            default="/workspace/jeeves",
            description="Workspace root inside container (e.g., /workspace/myproject). "
                        "Maps to HOST_WORKSPACE_PATH on host machine."
        )
        host_workspace_hint: str = Field(
            default="C:/Code",
            description="Display hint: what HOST_WORKSPACE_PATH is set to on host. "
                        "This is informational only - actual mapping is in docker-compose."
        )
    
    def __init__(self):
        """Initialize Jeeves filter."""
        self.user_valves = self.UserValves()
        self.toggle = True
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik05LjgxMyAxNS45MDRMOSAxOC43NWwtLjgxMy0yLjg0NmE0LjUgNC41IDAgMDAtMy4wOS0zLjA5TDIuMjUgMTJsODQ2LS44MTNhNC41IDQuNSAwIDAwMy4wOS0zLjA5TDE1IDUuMjVsLjgxMyAyLjg0NmE0LjUgNC41IDAgMDAzLjA5IDMuMDlMMjEuNzUgMTJsLTIuODQ2LjgxM2E0LjUgNC41IDAgMDAtMy4wOSAzLjA5ek0xOC4yNTkgOC43MTVMMTggOS43NWwtLjI1OS0xLjAzNWE0LjUgNC41IDAgMDAtMi41LTIuNUwxNCA1Ljk1NmwxLjAzNS0uMjU5YTQuNSA0LjUgMCAwMDIuNS0yLjVMMTggMi4xNjJsLjI1OSAxLjAzNWE0LjUgNC41IDAgMDAyLjUgMi41TDIxLjc5NCA2bC0xLjAzNS4yNTlhNC41IDQuNSAwIDAwLTIuNSAyLjVaIiAvPgo8L3N2Zz4K"
        )
    
    def _inject_context(self, body: dict, context_items: List[dict], orchestrator_context: Optional[str] = None) -> dict:
        """Inject system prompt, retrieved memories, and orchestrator analysis into conversation."""
        messages = body.get("messages", [])
        
        # Inject Jeeves system prompt if not already present
        has_jeeves_system = any(
            m.get("role") == "system" and "Jeeves" in m.get("content", "")
            for m in messages
        )
        
        if not has_jeeves_system:
            # Add Jeeves system prompt at the beginning
            jeeves_system = {
                "role": "system",
                "content": JEEVES_SYSTEM_PROMPT
            }
            # Insert after any existing system messages, or at start
            insert_idx = 0
            for i, m in enumerate(messages):
                if m.get("role") == "system":
                    insert_idx = i + 1
                else:
                    break
            messages.insert(insert_idx, jeeves_system)
        
        if not context_items and not orchestrator_context:
            body["messages"] = messages
            return body
        
        # Build context block
        context_lines = []
        
        # Add orchestrator analysis first
        if orchestrator_context:
            context_lines.append(orchestrator_context)
        
        # Add memory context
        if context_items:
            context_lines.append("### Retrieved Memories ###")
            for ctx in context_items[:self.user_valves.max_context_items]:
                user_text = ctx.get("user_text")
                if user_text:
                    context_lines.append(f"- {user_text}")
            context_lines.append("### End Memories ###\n")
        
        if not context_lines:
            body["messages"] = messages
            return body
        
        context_block = "\n".join(context_lines) + "\n"
        
        # Find last user message and prepend context
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
        Main filter entry point.
        
        Process:
          1. Extract uploaded files/images
          2. Classify intent
          3. For tasks: engage orchestrator
          4. Search/save to memory
          5. Inject context
        """
        try:
            user_id = __user__["id"] if __user__ and "id" in __user__ else "anonymous"
            messages = body.get("messages", [])
            model = body.get("model", "unknown")
            metadata = body.get("metadata", {})
            
            if not messages or not isinstance(messages, list):
                return body
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": "🎩 Jeeves processing...", "done": False, "hidden": False}
            })
            
            # Extract user text for classification
            user_text = _extract_user_text_prompt(messages)
            
            # Extract files and images
            file_content, filenames, chunks = _extract_file_contents(body)
            image_chunks = _extract_images_from_messages(messages, user_prompt=user_text)
            
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
            
            # Save document chunks
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
            
            # Classify intent
            orchestrator_context = None
            intent_result = {"intent": "casual", "confidence": 0.5}
            
            if user_text:
                intent_result = _classify_intent(user_text)
                intent = intent_result.get("intent", "casual")
                confidence = intent_result.get("confidence", 0.5)
                
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"🎯 Intent: {intent} ({confidence:.0%})",
                        "done": False,
                        "hidden": False,
                    }
                })
                
                # For task intents, engage orchestrator
                if intent == "task" and self.user_valves.enable_orchestrator:
                    orchestrator_context = await _orchestrate_task(
                        user_id,
                        messages,
                        self.user_valves.workspace_root if self.user_valves.workspace_root else None,
                        __event_emitter__,
                    )
            
            # Search memory
            source_type = "document" if chunks else "prompt"
            
            if chunks:
                source_name = filenames[0] if filenames else "attachment"
            else:
                content = messages[-1].get("content", "") if messages else ""
                source_name = (
                    (str(content)[:50] + "...") if len(str(content)) > 50 else str(content)
                )
            
            status, context = await _search_memory(
                user_id, messages, model, metadata, source_type, source_name
            )
            
            # Merge immediate image context
            if immediate_image_context:
                context = immediate_image_context + context
                if status == "skipped":
                    status = "context_found"
                elif status == "saved":
                    status = "saved_with_context"
            
            # Determine if we have context to inject
            has_context = bool(context) or bool(orchestrator_context)
            
            # Emit status and inject
            if has_context:
                memory_count = len(context) if context else 0
                
                if orchestrator_context:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🧠 Analysis ready + {memory_count} memories",
                            "done": False,
                            "hidden": False,
                        }
                    })
                else:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🧠 Found {memory_count} memories",
                            "done": False,
                            "hidden": False,
                        }
                    })
                
                for ctx in (context or [])[:3]:
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": f"  • {_format_source(ctx)}", "done": False, "hidden": False}
                    })
                
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "🎩 Ready to assist", "done": True, "hidden": False}
                })
                
                body = self._inject_context(body, context or [], orchestrator_context)
            
            elif status == "saved":
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "🧠 Memory saved", "done": True, "hidden": False}
                })
            
            elif status == "skipped":
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "🎩 Ready", "done": True, "hidden": True}
                })
            
            else:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"🎩 {status}", "done": True, "hidden": False}
                })
            
            print(f"[jeeves] user={user_id} intent={intent_result.get('intent')} status={status} context={len(context or [])} chunks={chunks_saved}")
            return body
        
        except Exception as e:
            print(f"[jeeves] Error: {e}")
            
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"⚠ Error: {str(e)[:40]}", "done": True, "hidden": False}
            })
            
            return body
    
    async def outlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> None:
        """Post-response hook. Could save assistant responses to memory."""
        pass
