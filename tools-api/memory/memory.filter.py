"""
title: Memory Service Tools
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.2
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

    async def inlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> dict:
        """
        Save the entire messages array as memory.
        Called on every request to store conversation history.
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
            
            # Log what we're saving
            message_preview = f"{len(messages)} messages"
            if messages:
                first_role = messages[0].get("role", "unknown")
                message_preview += f" (first: {first_role})"
            
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Saving {message_preview} for user {user_id}",
                        "done": True,
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
            
            return body
            
        except Exception as e:
            print(f"[filter] Error: {e}")
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Memory save failed: {str(e)}",
                        "done": True,
                        "hidden": False,
                    },
                }
            )
            return body

    async def outlet(self, body: dict, __event_emitter__, __user__: Optional[dict] = None) -> None:
        """
        Outlet filter - called after model response.
        Can be used to save assistant responses.
        """
        print("[filter] outlet called")
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Memory service processed response",
                    "done": True,
                    "hidden": False,
                },
            }
        )
