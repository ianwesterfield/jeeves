# Awesome Task
"""
Jeeves Filter - Agentic Reasoning & Memory

Open-WebUI filter that handles:
  - Intent classification (recall, save, task)
  - Multi-step reasoning via orchestrator
  - Code execution via executor
  - Semantic memory storage and retrieval
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

**CRITICAL RULES - FOLLOW EXACTLY:**

1. **DON'T MAKE UP FILE LISTINGS OR CONTENTS**
   - ONLY show files that appear in "### Workspace Files ###" blocks
   - ONLY show content that appears in "### File Content ###" blocks  
   - If you don't see actual data in these blocks, say "no data available"
   - DO NOT guess file names - EVER

2. **You don't call tools directly**
   - The Jeeves filter pre-processes requests and injects real results
   - NEVER output tool call syntax like [TOOL_CALLS] - it won't work

3. **How to handle injected context:**
   - "### Workspace Files ###" → Real file listing - show ONLY these files
   - "### File Content ###" → Real file contents - quote from this ONLY
   - "### File Operation Result ###" → An edit was executed - report the result
   - "### Retrieved Memories ###" → Relevant memories about the user

4. **If context blocks are empty or missing:**
   - Say you couldn't retrieve the information
   - DON'T make up a response
   - Ask if the user wants to try again

**Example of CORRECT behavior:**
If you see: "📁 filters\n📁 layers\n📄 README.md"
Say: "The workspace contains 2 directories (filters, layers) and 1 file (README.md)"
DON'T add files that aren't in the list!

**Your Capabilities (handled by Jeeves filter):**
- Workspace file listing and reading
- Surgical file edits (replace, insert, append)
- Semantic memory across conversations
- Code execution (when enabled)
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
    Classify user intent via pragmatics 4-class model.
    
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
    
    # Fallback if pragmatics is down - very basic heuristics
    print("[jeeves] Warning: Pragmatics unavailable, using basic fallback")
    return {"intent": "casual", "confidence": 0.3}


def _detect_task_continuation(user_text: str, messages: List[dict], confidence: float) -> bool:
    """
    Detect if a 'casual' response is actually continuing a task.
    
    Uses pragmatics context-aware classification to determine if short
    responses like "yes", "ok", "do it" are task continuations.
    
    Args:
        user_text: Current user message
        messages: Conversation history
        confidence: Initial classification confidence
    
    Returns True if we should upgrade intent to 'task'.
    """
    # Get recent assistant message for context
    context = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Take last ~500 chars of assistant message as context
                context = content[-500:] if len(content) > 500 else content
                break
    
    # If no context, can't determine continuation
    if not context:
        print(f"[jeeves] Task continuation: no assistant context")
        return False
    
    # Call pragmatics context-aware endpoint
    try:
        resp = requests.post(
            f"{PRAGMATICS_API_URL}/api/pragmatics/classify-with-context",
            json={"text": user_text, "context": context},
            timeout=5,
        )
        if resp.status_code == 200:
            result = resp.json()
            intent = result.get("intent", "casual")
            ctx_confidence = result.get("confidence", 0.5)
            
            if intent == "task":
                print(f"[jeeves] Task continuation: YES via pragmatics (conf={ctx_confidence:.2f})")
                return True
            else:
                print(f"[jeeves] Task continuation: NO via pragmatics (intent={intent}, conf={ctx_confidence:.2f})")
                return False
    except Exception as e:
        print(f"[jeeves] Task continuation: pragmatics error: {e}")
    
    # Fallback: low confidence + short message suggests continuation
    if confidence < 0.7 and len(user_text.strip()) < 50:
        print(f"[jeeves] Task continuation: MAYBE (low conf + short msg)")
        return True
    
    return False


# ============================================================================
# Orchestrator Integration
# ============================================================================

def _build_task_description(messages: List[dict]) -> str:
    """
    Build a complete task description from conversation context.
    
    For continuation requests like "Please do" or "yes", we need to include
    the original task context from previous messages so the orchestrator
    knows what to do.
    """
    user_text = _extract_user_text_prompt(messages) or ""
    
    # Check if this is a short continuation response
    short_continuations = [
        "please do", "yes", "yeah", "sure", "ok", "okay", "go ahead",
        "do it", "proceed", "make the changes", "sounds good",
        "that works", "perfect", "great", "continue", "next"
    ]
    
    user_lower = user_text.lower().strip()
    is_continuation = len(user_text) < 100 and any(
        user_lower.startswith(cont) or user_lower == cont
        for cont in short_continuations
    )
    
    if not is_continuation:
        return user_text
    
    # This is a continuation - find the original task from conversation history
    # Look for the most recent substantial user message
    for msg in reversed(messages[:-1]):  # Skip current message
        if msg.get("role") != "user":
            continue
        
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract text from multi-part content
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(texts)
        
        if not isinstance(content, str):
            continue
            
        # Skip short messages that are also continuations
        if len(content) < 50:
            continue
            
        # Skip messages that look like injected context
        if content.startswith("### ") or "End Workspace Files" in content:
            continue
        
        # Found the original task
        print(f"[jeeves] Continuation detected - using original task: {content[:100]}...")
        return f"User originally asked: {content}\n\nUser now confirms: {user_text}"
    
    # Fallback to current text if no original task found
    return user_text


async def _orchestrate_task(
    user_id: str,
    messages: List[dict],
    workspace_root: Optional[str],
    __event_emitter__,
    memory_context: Optional[List[dict]] = None,
) -> Optional[str]:
    """
    Handle task intents via orchestrator streaming endpoint.
    
    Flow:
      1. POST /run-task to orchestrator (starts SSE stream)
      2. Forward status events to __event_emitter__
      3. Return final context when complete
    
    The agentic loop now runs in the orchestrator, not here.
    """
    if not workspace_root:
        return None
    
    # Build complete task description (handles continuations)
    user_text = _build_task_description(messages)
    
    try:
        await __event_emitter__({
            "type": "status",
            "data": {"description": "✨ Thinking...", "done": False, "hidden": False}
        })
        
        # Stream task execution from orchestrator
        import httpx
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{ORCHESTRATOR_API_URL}/api/orchestrate/run-task",
                json={
                    "task": user_text,
                    "workspace_root": workspace_root,
                    "user_id": user_id,
                    "memory_context": memory_context,
                    "max_steps": 100,
                },
            ) as response:
                final_context = None
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    try:
                        event = json.loads(line[6:])  # Strip "data: " prefix
                    except json.JSONDecodeError:
                        continue
                    
                    event_type = event.get("event_type", "")
                    status = event.get("status", "")
                    done = event.get("done", False)
                    
                    # Forward status events to UI
                    if event_type == "status" and status:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": status, "done": done, "hidden": False}
                        })
                    
                    elif event_type == "error":
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"❌ {status}", "done": done, "hidden": False}
                        })
                    
                    elif event_type == "complete":
                        result = event.get("result", {})
                        final_context = result.get("context")
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "✅ Ready", "done": True, "hidden": False}
                        })
                
                return final_context
                
    except Exception as e:
        print(f"[jeeves] Orchestrator streaming error: {e}")
        await __event_emitter__({
            "type": "status",
            "data": {"description": f"❌ Error: {str(e)[:30]}", "done": True, "hidden": False}
        })
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
        # Butler icon - person with bow tie silhouette
        self.icon = (
            "data:image/svg+xml;base64,"
            "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iY3VycmVudENvbG9yIj4KICA8IS0tIEhlYWQgLS0+CiAgPGNpcmNsZSBjeD0iMTIiIGN5PSI2IiByPSIzLjUiLz4KICA8IS0tIEJvdyB0aWUgLS0+CiAgPHBhdGggZD0iTTkgMTJsLTIgMS41TDkgMTVoNmwyLTEuNUwxNSAxMkg5eiIvPgogIDwhLS0gQm9keSAtLT4KICA8cGF0aCBkPSJNNiAxNWMwLTEgMS41LTIgNi0yIDQuNSAwIDYgMSA2IDJ2NmMwIC41LS41IDEtMSAxSDdjLS41IDAtMS0uNS0xLTF2LTZ6Ii8+Cjwvc3ZnPgo="
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
                "data": {"description": "✨ Thinking...", "done": False, "hidden": False}
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
                        "description": f"💾 Saving {len(chunks)} chunk(s)...",
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
            
            # Search memory FIRST - we need user context (name, etc.) for tasks
            status = "skipped"
            context = []
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
            
            if user_text:
                intent_result = _classify_intent(user_text)
                intent = intent_result.get("intent", "casual")
                confidence = intent_result.get("confidence", 0.5)
                
                # SMART TASK CONTINUATION DETECTION
                # Check if user is confirming/continuing a previous task proposal
                if intent == "casual":
                    should_upgrade = _detect_task_continuation(user_text, messages, confidence)
                    if should_upgrade:
                        print(f"[jeeves] Upgrading intent from casual to task (continuation detected)")
                        intent = "task"
                
                # For task intents, engage orchestrator (pass memory context for user info)
                if intent == "task" and self.user_valves.enable_orchestrator:
                    orchestrator_context = await _orchestrate_task(
                        user_id,
                        messages,
                        self.user_valves.workspace_root if self.user_valves.workspace_root else None,
                        __event_emitter__,
                        memory_context=context,  # Pass user context (name, preferences)
                    )
            
            # Merge immediate image context (memory already searched above)
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
                
                if memory_count > 0:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"📚 {memory_count} relevant memories",
                            "done": False,
                            "hidden": False,
                        }
                    })
                
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Ready", "done": True, "hidden": False}
                })
                
                body = self._inject_context(body, context or [], orchestrator_context)
            
            elif status == "saved":
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "💾 Saved to memory", "done": True, "hidden": False}
                })
            
            elif status == "skipped":
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Ready", "done": True, "hidden": True}
                })
            
            else:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "✅ Ready", "done": True, "hidden": False}
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
    
    async def outlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> dict:
        """
        Post-response hook. Formats assistant responses for consistency.
        
        Ensures:
        - File/folder names are in `backticks`
        - Code references are in `backticks`
        - Lists and code blocks use fenced markdown
        """
        messages = body.get("messages", [])
        if not messages:
            return body
        
        # Find last assistant message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                if isinstance(content, str):
                    messages[i]["content"] = self._format_response(content)
                break
        
        body["messages"] = messages
        return body
    
    def _format_response(self, text: str) -> str:
        """
        Format response text for better markdown rendering.
        
        - Remove leaked tool call syntax (LLM hallucinations)
        - Wrap file/folder names in backticks
        - Wrap code references in backticks
        - Ensure code blocks are fenced
        """
        import re
        
        # Skip if empty
        if not text:
            return text
        
        # SANITIZATION: Remove any leaked tool call syntax
        # Some models output [TOOL_CALLS] when they shouldn't
        tool_call_pattern = r'\[TOOL_CALLS\][^\[]*(?:\[ARGS\][^\[]*)?'
        if '[TOOL_CALLS]' in text:
            text = re.sub(tool_call_pattern, '', text)
            # Clean up any orphaned text that looks like tool calls
            text = re.sub(r'\{"file_path":\s*"[^"]*"\}', '', text)
            text = re.sub(r'\{"path":\s*"[^"]*"\}', '', text)
            # Remove multiple newlines created by cleanup
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()
            # If we removed everything, provide a fallback
            if not text or len(text) < 20:
                text = "I'll help you with that task. Let me process the files in your workspace."
        
        # Skip further formatting if already well-formatted
        if text.startswith("```"):
            return text
        
        # Pattern for file paths and names (with extensions or path separators)
        # Match paths like: file.py, /path/to/file, ./folder, layers/executor/main.py
        file_pattern = r'(?<![`\w])([./\\]?[\w\-]+(?:[/\\][\w\-\.]+)+\.?\w*|[\w\-]+\.\w{1,10})(?![`\w])'
        
        def wrap_if_not_wrapped(match):
            path = match.group(1)
            # Don't wrap if it looks like a URL or already wrapped
            if path.startswith('http') or path.startswith('`'):
                return match.group(0)
            return f'`{path}`'
        
        # Apply file path formatting (but not inside code blocks)
        lines = text.split('\n')
        formatted_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                formatted_lines.append(line)
            elif in_code_block:
                formatted_lines.append(line)
            else:
                # Format file paths outside code blocks
                formatted_line = re.sub(file_pattern, wrap_if_not_wrapped, line)
                formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)
