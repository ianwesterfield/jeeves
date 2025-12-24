"""
Reasoning Engine - LLM Coordination for Step Generation

Uses Ollama to generate structured reasoning steps from user intents.
Parses LLM output into validated Step objects.

Architecture:
- State is maintained EXTERNALLY by WorkspaceState (not by the LLM)
- LLM receives state as context, only outputs next step
- This prevents state drift and reduces token cost
"""

import os
import json
import logging
import httpx
from typing import Any, Dict, List, Optional

from schemas.models import Step, StepResult, WorkspaceContext
from services.workspace_state import WorkspaceState, get_workspace_state


logger = logging.getLogger("orchestrator.reasoning")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


SYSTEM_PROMPT = """You are Jeeves, an agentic AI that executes repository tasks step-by-step.

Respond with JSON for the NEXT step only. No markdown, no extra text.

=== OUTPUT FORMAT (JSON ONLY) ===

{"tool": "tool_name", "params": {...}, "note": "Brief status for user"}

- "note" is a SHORT operator-facing status (not reasoning/chain-of-thought)
- Do NOT include state tracking - the orchestrator maintains state externally

=== MISSING INFO RULE ===

If the task references info you don't have (e.g. "my name" but not provided):
{"tool": "complete", "params": {"error": "MISSING_INFO: I don't know your name."}, "note": "Need info"}

NEVER guess or fabricate user information.

=== AVAILABLE TOOLS ===

- scan_workspace: List files. Params: {"path": "string"}
- read_file: Read file. Params: {"path": "string"}
- write_file: Create/overwrite file. Params: {"path": "string", "content": "string"}
- replace_in_file: Find/replace EXISTING text. Params: {"path": "string", "old_text": "string", "new_text": "string"}
- insert_in_file: ADD NEW text at start or end. Params: {"path": "string", "position": "start"|"end", "text": "string"}
- append_to_file: Append to end. Params: {"path": "string", "content": "string"}
- execute_shell: Run shell command. Params: {"command": "string"}
- none: Skip (change already present). Params: {"reason": "string"}
- complete: Done. Params: {} OR {"error": "reason"}

⚠️ CRITICAL - INSERT vs REPLACE:
- To ADD NEW text (comment, header, etc.) → use insert_in_file with position="start" or "end"
- To CHANGE EXISTING text you SAW in the file → use replace_in_file
- replace_in_file WILL FAIL if old_text is not found - it cannot add new content!
- If you need to add a comment/header that DOES NOT EXIST, you MUST use insert_in_file
- ALWAYS include a trailing newline in inserted text: "<!-- Comment -->\\n" not "<!-- Comment -->"
- For Markdown files (.md), use HTML comment syntax: <!-- Author: Name -->

=== CRITICAL RULES ===

1. PATH DISCIPLINE: Use EXACT paths from workspace state. Never invent paths.
   - Correct: ".github/copilot-instructions.md"
   - WRONG: "copilot-instructions.md" (missing directory)

2. NO RESCANS: If workspace state shows files, don't scan again.

3. NO RE-READS: If state shows "Already read" a file, DO NOT read it again. Move to editing.

3. MINIMAL EDITS: Prefer insert_in_file/replace_in_file over write_file.

4. ONE STEP: Return exactly one action. Don't bundle multiple operations.

5. PRIORITY ORDER:
   a) If no scan exists → scan_workspace(".")
   b) If task mentions file → locate in state, read if needed, then edit
   c) After all edits → complete

=== SHELL EXAMPLES ===

Git: {"tool": "execute_shell", "params": {"command": "git status"}}
     {"tool": "execute_shell", "params": {"command": "git checkout -b feature/x"}}
     {"tool": "execute_shell", "params": {"command": "git add . && git commit -m 'msg'"}}

=== LOOP DETECTION ===

If you see a step you just completed in "Completed steps", DO NOT repeat it.
If task is "list/scan files" and you already did scan_workspace → {"tool": "complete", "params": {}, "note": "Done"}
If task is "read file X" and X is in "Already read" → {"tool": "complete", "params": {}, "note": "Already read"}
If you can't make progress → {"tool": "complete", "params": {"error": "Cannot complete task"}, "note": "Stuck"}

=== COMPLETION ===

When finished: {"tool": "complete", "params": {}, "note": "Done"}
On error: {"tool": "complete", "params": {"error": "reason"}, "note": "Failed"}
"""


class ReasoningEngine:
    """
    LLM-powered reasoning engine for step generation.
    
    Architecture:
    - State is maintained externally by WorkspaceState
    - LLM receives state as context, only outputs the next step
    - This prevents hallucination and state drift
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=300.0)  # DEV MODE: Extended timeout
        self.model = OLLAMA_MODEL
        self.base_url = OLLAMA_BASE_URL
    
    async def generate_next_step(
        self,
        task: str,
        history: List[StepResult],
        memory_context: List[Dict[str, Any]],
        workspace_context: Optional[WorkspaceContext] = None,
        workspace_state: Optional[WorkspaceState] = None,
    ) -> Step:
        """
        Generate the next step for a task using LLM reasoning.
        
        Args:
            task: User's task description
            history: Previous step results (for backward compat, prefer workspace_state)
            memory_context: Relevant patterns from memory
            workspace_context: Current workspace settings
            workspace_state: External state (ground truth from actual tool outputs)
            
        Returns:
            Step object with tool, params, and reasoning
        """
        # Use external state if provided, otherwise fall back to old approach
        if workspace_state is None:
            workspace_state = get_workspace_state()
        
        context_parts = []
        
        # 1. Inject external state (the key change - LLM doesn't maintain this)
        state_context = workspace_state.format_for_prompt()
        if state_context:
            context_parts.append(state_context)
        
        # 2. Workspace permissions
        if workspace_context:
            context_parts.append(
                f"Workspace: {workspace_context.cwd}\n"
                f"Permissions: write={workspace_context.allow_file_write}, "
                f"shell={workspace_context.allow_shell_commands}"
            )
        
        # 3. User info from state (name, etc.)
        if workspace_state.user_info:
            user_info_text = "\n".join(f"  {k}: {v}" for k, v in workspace_state.user_info.items())
            context_parts.append(f"User info (use this, don't guess):\n{user_info_text}")
        
        # 4. Memory patterns (optional)
        if memory_context:
            patterns = "\n".join([
                f"- {p.get('description', 'Similar task')}: {p.get('approach', '')}"
                for p in memory_context[:3]
            ])
            context_parts.append(f"Relevant patterns from memory:\n{patterns}")
        
        # 5. GUARDRAIL: If too many steps without progress, force completion
        if len(workspace_state.completed_steps) >= 15:
            logger.warning(f"GUARDRAIL: {len(workspace_state.completed_steps)} steps - forcing completion check")
            # Check if we're making progress
            recent_edits = sum(1 for s in workspace_state.completed_steps[-5:] 
                             if s.tool in ("write_file", "insert_in_file", "replace_in_file", "append_to_file")
                             and s.success)
            if recent_edits == 0:
                # Not making progress - complete
                return Step(
                    step_id="guardrail_complete",
                    tool="complete",
                    params={"error": "Too many steps without progress"},
                    reasoning="Forced completion after 15 steps without recent edits",
                )
        
        context = "\n\n".join(context_parts)
        
        # Build user message
        user_message = f"Task: {task}"
        if context:
            user_message = f"{context}\n\n{user_message}"
        
        # Log prompt size (helpful for debugging)
        logger.debug(f"Prompt size: {len(user_message)} chars")
        
        # Call Ollama
        try:
            response = await self._call_ollama(user_message)
            step = self._parse_response(response, task)
            
            # GUARDRAIL: Detect repeated replace_in_file failures on same file
            if step.tool == "replace_in_file":
                path = step.params.get("path", "")
                # Count recent failures on this file
                recent_failures = sum(
                    1 for s in workspace_state.completed_steps[-5:]
                    if s.tool == "replace_in_file" 
                    and s.params.get("path") == path
                    and not s.success
                )
                if recent_failures >= 2:
                    logger.warning(f"GUARDRAIL: {recent_failures} replace failures on {path} - switching to insert_in_file")
                    # Convert to insert_in_file at start
                    return Step(
                        step_id="guardrail_use_insert",
                        tool="insert_in_file",
                        params={
                            "path": path,
                            "position": "start",
                            "text": step.params.get("new_text", ""),
                        },
                        reasoning=f"Auto-corrected: replace_in_file failed {recent_failures}x, using insert_in_file instead",
                    )
            
            # GUARDRAIL: Prevent re-reading already read files
            if step.tool == "read_file":
                path = step.params.get("path", "")
                if path and path in workspace_state.read_files:
                    logger.warning(f"GUARDRAIL: Blocking re-read of '{path}'")
                    # Skip this step - return a prompt to move on
                    return Step(
                        step_id="guardrail_no_reread",
                        tool="complete",
                        params={"error": f"Already read {path}. Move to editing or complete."},
                        reasoning=f"Blocked re-read of already-read file: {path}",
                    )
            
            # GUARDRAIL: Validate paths exist in state
            if step.tool in ("read_file", "write_file", "insert_in_file", "replace_in_file", "append_to_file"):
                path = step.params.get("path", "")
                if path and workspace_state.files and path not in workspace_state.files:
                    # Path doesn't exist in scanned files - could be a hallucination
                    # Check if it's a new file being created
                    if step.tool != "write_file":
                        logger.warning(f"GUARDRAIL: Path '{path}' not in scanned files")
                        # Try to find similar path
                        similar = [f for f in workspace_state.files if f.endswith(path) or path in f]
                        if similar:
                            # Suggest the correct path
                            correct_path = similar[0]
                            logger.info(f"GUARDRAIL: Correcting path to '{correct_path}'")
                            step.params["path"] = correct_path
            
            return step
            
        except Exception as e:
            logger.error(f"Reasoning engine error: {e}")
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
            
            # Accept either "note" (new) or "reasoning" (backward compat)
            note = data.get("note", "") or data.get("reasoning", "")
            
            return Step(
                step_id=step_id,
                tool=data.get("tool", "unknown"),
                params=data.get("params", {}),
                reasoning=note,  # Store note in reasoning field for compat
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
