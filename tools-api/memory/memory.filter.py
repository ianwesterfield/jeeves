"""
title: Memory Service Tools
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.3
"""

from pydantic import BaseModel, Field
from typing import Optional
import requests
import json

class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBjbGFzcz0ic2l6ZS02Ij4KICAKICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xMiAxOHYtNS4yNW0wIDBhNi4wMSA2LjAxIDAgMCAwIDEuNS0uMTg5bS0xLjUuMTg5YTYuMDEgNi4wMSAwIDAgMS0xLjUtLjE4OW0zLjc1IDcuNDc4YTEyLjA2IDEyLjA2IDAgMCAxLTQuNSAwbTMuNzUgMi4zODNhMTQuNDA2IDE0LjQwNiAwIDAgMS0zIDBNMTQuMjUgMTh2LS4xOTJjMC0uOTgzLjY1OC0xLjgyMyAxLjUwOC0yLjMxNmE3LjUgNy41IDAgMSAwLTcuNTE3IDBjLjg1LjQ5MyAxLjUwOSAxLjMzMyAxLjUwOSAyLjMxNlYxOCIgLz4KPC9zdmc+Cg=="""

    def _inject_context(self, body: dict, existing_context: list) -> dict:
        """
        Inject retrieved memory context into the conversation.
        Prepends context to the last user message for better model compatibility.
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
        Process incoming messages:
        1. Send to memory API for save/search
        2. If context found, inject it into conversation
        """
        try:
            print("[filter] inlet called")
            user_id = __user__["id"] if __user__ and "id" in __user__ else "anonymous"
            
            # Extract messages array
            messages = body.get("messages", [])
            model = body.get("model", "unknown")
            metadata = body.get("metadata", {})
            print(f"[filter] user_id={user_id}, model={model}, messages={len(messages)}")
            
            if not messages or not isinstance(messages, list):
                raise ValueError("Body must contain 'messages' array")
            
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "🔍 Searching memory...",
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
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"💾 Saved + 📚 Found {len(existing_context)} memories",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
                # Inject context into conversation
                body = self._inject_context(body, existing_context)
                
            elif status == "context_found":
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"📚 Found {len(existing_context)} memories",
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
                            "description": "💾 Memory saved",
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
                            "description": "⏭️ No relevant memory",
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
                        "description": f"⚠️ Memory error: {str(e)[:50]}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
            return body

    async def outlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> None:
        """
        Outlet filter - called after model response.
        Currently a no-op, but could be used to save assistant responses.
        """
        pass
