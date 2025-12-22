"""
Reasoning Engine - LLM Coordination for Step Generation

Uses Ollama to generate structured reasoning steps from user intents.
Parses LLM output into validated Step objects.
"""

import os
import json
import logging
import httpx
from typing import Any, Dict, List, Optional

from schemas.models import Step, StepResult, WorkspaceContext


logger = logging.getLogger("orchestrator.reasoning")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


SYSTEM_PROMPT = """You are Jeeves, an agentic AI assistant that breaks down user tasks into executable steps.

Given a task, you must respond with a JSON object containing the next step to execute.

Available tools:
- scan_workspace: List files in a directory. Params: {path: string}
- read_file: Read file contents. Params: {path: string}
- write_file: Write to a file. Params: {path: string, content: string}
- execute_code: Run code. Params: {language: "python"|"powershell"|"node"|"bash", code: string}
- execute_shell: Run shell command. Params: {command: string}
- analyze_code: Analyze code semantics. Params: {path: string}
- batch: Execute multiple independent steps in parallel. Params: {steps: Step[]}

Response format (JSON only, no markdown):
{
  "tool": "tool_name",
  "params": {...},
  "reasoning": "Why this step is needed",
  "batch_id": null or "batch_identifier" if parallelizable
}

For parallelizable tasks (e.g., "analyze all Python files"), use the batch tool:
{
  "tool": "batch",
  "params": {"pattern": "*.py", "operation": "analyze_code"},
  "reasoning": "Multiple independent files can be analyzed in parallel",
  "batch_id": "analyze_py_files"
}

Rules:
1. Only suggest ONE step at a time (unless it's a batch)
2. Consider the history of previous steps
3. If the task is complete, respond with: {"tool": "complete", "params": {}, "reasoning": "Task completed"}
4. Always validate paths are within the workspace
5. Prefer reading/scanning before writing/executing
"""


class ReasoningEngine:
    """
    LLM-powered reasoning engine for step generation.
    
    Coordinates with Ollama to generate structured steps from natural language tasks.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
    
    async def generate_next_step(
        self,
        task: str,
        history: List[StepResult],
        memory_context: List[Dict[str, Any]],
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Step:
        """
        Generate the next step for a task using LLM reasoning.
        
        Args:
            task: User's task description
            history: Previous step results
            memory_context: Relevant patterns from memory
            workspace_context: Current workspace settings
            
        Returns:
            Step object with tool, params, and reasoning
        """
        # Build context prompt
        context_parts = []
        
        if workspace_context:
            context_parts.append(
                f"Workspace: {workspace_context.cwd}\n"
                f"Available languages: {', '.join(workspace_context.allowed_languages)}\n"
                f"Parallel enabled: {workspace_context.parallel_enabled}"
            )
        
        if memory_context:
            patterns = "\n".join([
                f"- {p.get('description', 'Similar task')}: {p.get('approach', '')}"
                for p in memory_context[:3]
            ])
            context_parts.append(f"Relevant patterns from memory:\n{patterns}")
        
        if history:
            history_text = "\n".join([
                f"Step {i+1}: {r.status.value} - {r.output[:100] if r.output else r.error or 'No output'}"
                for i, r in enumerate(history[-5:])  # Last 5 steps
            ])
            context_parts.append(f"Previous steps:\n{history_text}")
        
        context = "\n\n".join(context_parts)
        
        # Build user message
        user_message = f"Task: {task}"
        if context:
            user_message = f"{context}\n\n{user_message}"
        
        # Call Ollama
        try:
            response = await self._call_ollama(user_message)
            step = self._parse_response(response, task)
            return step
        except Exception as e:
            logger.error(f"Reasoning engine error: {e}")
            # Return a safe fallback step
            return Step(
                step_id="error_fallback",
                tool="complete",
                params={"error": str(e)},
                reasoning=f"Error during reasoning: {e}",
            )
    
    async def _call_ollama(self, user_message: str) -> str:
        """Call Ollama API and return the response text."""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "format": "json",  # Request JSON output
        }
        
        logger.debug(f"Calling Ollama: {self.model}")
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get("message", {}).get("content", "{}")
    
    def _parse_response(self, response: str, task: str) -> Step:
        """Parse LLM response into a Step object."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Generate step ID
            import uuid
            step_id = f"step_{uuid.uuid4().hex[:8]}"
            
            return Step(
                step_id=step_id,
                tool=data.get("tool", "unknown"),
                params=data.get("params", {}),
                reasoning=data.get("reasoning", ""),
                batch_id=data.get("batch_id"),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            
            # Return error step
            import uuid
            return Step(
                step_id=f"parse_error_{uuid.uuid4().hex[:8]}",
                tool="complete",
                params={"error": "Failed to parse LLM response"},
                reasoning=f"Parse error: {e}",
            )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
