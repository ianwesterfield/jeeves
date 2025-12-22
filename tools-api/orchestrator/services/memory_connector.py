"""
Memory Connector - Pattern Retrieval and Execution Trace Storage

Connects to the Memory service for:
  - Retrieving relevant past patterns (similar tasks)
  - Storing execution traces for learning
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from schemas.models import BatchResult


logger = logging.getLogger("orchestrator.memory")

# Memory service URL
MEMORY_BASE_URL = os.getenv("MEMORY_BASE_URL", "http://memory_api:8000")


class MemoryConnector:
    """
    Interface to the Memory service for pattern learning.
    
    Retrieves relevant past execution patterns and stores new traces.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        self.memory_url = MEMORY_BASE_URL
    
    async def search_patterns(
        self,
        query: str,
        user_id: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant past execution patterns.
        
        Args:
            query: Task description to search for
            user_id: User ID for filtering
            top_k: Number of results to return
            
        Returns:
            List of pattern dictionaries with description and approach
        """
        try:
            response = await self.client.post(
                f"{self.memory_url}/api/memory/search",
                json={
                    "user_id": user_id,
                    "query_text": query,
                    "top_k": top_k,
                },
            )
            response.raise_for_status()
            
            results = response.json()
            
            # Transform memory results into pattern format
            patterns = []
            for result in results:
                # Check if this was an agentic execution trace
                source_type = result.get("source_type", "")
                if source_type == "agentic_execution":
                    patterns.append({
                        "description": result.get("user_text", ""),
                        "approach": result.get("messages", [{}])[0].get("content", ""),
                        "score": result.get("score", 0),
                    })
            
            return patterns
            
        except httpx.HTTPError as e:
            logger.warning(f"Memory search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Memory connector error: {e}")
            return []
    
    async def store_execution_trace(
        self,
        user_id: str,
        batch_id: str,
        result: BatchResult,
    ) -> bool:
        """
        Store an execution trace for future pattern learning.
        
        Args:
            user_id: User who executed the task
            batch_id: Batch identifier
            result: Execution result with success/failure info
            
        Returns:
            True if stored successfully
        """
        try:
            # Build trace document
            trace = {
                "batch_id": batch_id,
                "successful_count": result.successful_count,
                "failed_count": result.failed_count,
                "duration": result.duration,
                "steps": [
                    {"step_id": s.step_id, "status": s.status.value}
                    for s in result.successful
                ],
                "errors": [
                    {"step_id": e.step_id, "error_type": e.error_type.value}
                    for e in result.failed
                ],
            }
            
            # Store as agentic execution memory
            response = await self.client.post(
                f"{self.memory_url}/api/memory/save",
                json={
                    "user_id": user_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"Execution trace for batch {batch_id}",
                        },
                        {
                            "role": "assistant",
                            "content": str(trace),
                        },
                    ],
                    "source_type": "agentic_execution",
                    "source_name": batch_id,
                    "skip_classifier": True,  # Always save execution traces
                },
            )
            response.raise_for_status()
            
            logger.info(f"Stored execution trace for batch {batch_id}")
            return True
            
        except httpx.HTTPError as e:
            logger.warning(f"Failed to store execution trace: {e}")
            return False
        except Exception as e:
            logger.error(f"Memory connector store error: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
