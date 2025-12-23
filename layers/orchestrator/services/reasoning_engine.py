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


SYSTEM_PROMPT = """You are Jeeves, an agentic AI assistant that executes tasks step by step.

You must respond with JSON containing the NEXT step to execute.

CRITICAL: If the task references information you don't have (like "my name" but no name was provided),
return: {"tool": "complete", "params": {"error": "I don't know your name. Please tell me your name first."}}
NEVER make up or guess user information - only use what is explicitly provided.

Available tools:
- scan_workspace: List files. Params: {"path": "string"}
- read_file: Read file. Params: {"path": "string"}  
- write_file: Overwrite file. Params: {"path": "string", "content": "string"}
- replace_in_file: Find/replace. Params: {"path": "string", "old_text": "string", "new_text": "string"}
- insert_in_file: Insert at position. Params: {"path": "string", "position": "start"|"end", "text": "string"}
- append_to_file: Add to end. Params: {"path": "string", "content": "string"}
- execute_shell: Run shell command. Params: {"command": "string"}
- complete: Done. Params: {} or {"error": "reason"}

Response format (JSON only):
{"tool": "tool_name", "params": {...}, "reasoning": "why"}

=== CRITICAL RULES ===

1. NEVER scan the same path twice. Check COMPLETED STEPS first.
2. If scan_workspace already succeeded for a path, USE THAT OUTPUT.
3. Process files in order - don't skip or re-scan.
4. Work silently - the user sees status updates, not your reasoning.
5. When done, return {"tool": "complete", "params": {}, "reasoning": "done"}
6. If you cannot perform a task with available tools, use complete with {"error": "reason"}.
7. **PATHS MUST BE EXACT** - Use the FULL path from scan results. If scan shows ".github/file.md", use ".github/file.md" NOT "file.md".

=== SHELL COMMAND EXAMPLES ===

Git operations:
- Create branch: {"tool": "execute_shell", "params": {"command": "git checkout -b feature/new-feature"}}
- Switch branch: {"tool": "execute_shell", "params": {"command": "git checkout main"}}
- Check status: {"tool": "execute_shell", "params": {"command": "git status"}}
- Stage files: {"tool": "execute_shell", "params": {"command": "git add ."}}
- Commit: {"tool": "execute_shell", "params": {"command": "git commit -m 'description'"}}
- Undo unstaged changes: {"tool": "execute_shell", "params": {"command": "git checkout -- ."}}
  Note: NEVER use "git checkout HEAD" as it discards staged work.
- View diff: {"tool": "execute_shell", "params": {"command": "git diff"}}
- List branches: {"tool": "execute_shell", "params": {"command": "git branch -a"}}

=== FILE OPERATION EXAMPLES ===

IMPORTANT: Always use the FULL relative path including directories.
- Correct: {"tool": "insert_in_file", "params": {"path": ".github/file.md", "position": "start", "text": "# Comment\\n"}}
- WRONG: {"tool": "insert_in_file", "params": {"path": "file.md", ...}}  ← Missing directory!
"""


class ReasoningEngine:
    """
    LLM-powered reasoning engine for step generation.
    
    Coordinates with Ollama to generate structured steps from natural language tasks.
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
        
        # DEBUG: Log history summary (not individual steps)
        if history:
            success_count = sum(1 for h in history if h.status.value == "success")
            logger.info(f"History: {len(history)} steps ({success_count} successful)")
        
        # GUARDRAIL: Detect repeated scanning and force completion
        if history:
            scan_count = sum(1 for h in history if h.tool == "scan_workspace")
            
            # Track which files have already been edited
            already_edited = set()
            for h in history:
                if h.tool in ("insert_in_file", "replace_in_file", "write_file", "append_to_file"):
                    path = h.params.get("path", "") if h.params else ""
                    if path:
                        already_edited.add(path)
            
            if scan_count >= 3:
                # Model is stuck in a scan loop - extract file list and force next action
                logger.warning(f"GUARDRAIL: {scan_count} scans, {len(already_edited)} files edited")
                
                # Collect all files found from scans
                all_files = set()
                for h in history:
                    if h.tool == "scan_workspace" and h.output:
                        # Extract file paths from scan output
                        # Format: NAME  TYPE  SIZE  MODIFIED
                        for line in h.output.split("\n"):
                            line = line.strip()
                            # Skip headers, separators, and metadata
                            if not line or line.startswith("PATH:") or line.startswith("TOTAL:"):
                                continue
                            if line.startswith("-") or line.startswith("NAME"):
                                continue
                            if line.startswith("..."):
                                continue
                            
                            # Parse the columns: NAME  TYPE  SIZE  MODIFIED
                            parts = line.split()
                            if len(parts) >= 2:
                                name = parts[0]
                                entry_type = parts[1] if len(parts) > 1 else ""
                                if entry_type == "file":
                                    all_files.add(name)
                
                # Filter to editable files only
                editable_extensions = ['.md', '.py', '.js', '.ts', '.yaml', '.yml', '.json', '.txt', '.sh', '.ps1', '.mmd']
                binary_extensions = ['.png', '.jpg', '.gif', '.bin', '.exe', '.dll', '.pyc', '.pth', '.safetensors']
                
                editable_files = set()
                for f in all_files:
                    if any(f.endswith(ext) for ext in binary_extensions):
                        continue
                    if any(f.endswith(ext) for ext in editable_extensions):
                        editable_files.add(f)
                
                # Remove already-edited files
                remaining_files = editable_files - already_edited
                
                logger.info(f"GUARDRAIL: {len(editable_files)} editable, {len(already_edited)} done, {len(remaining_files)} remaining")
                
                # Extract user's name from task if present
                # Look for patterns like "by Name", "name: Name", or just "FirstName LastName"
                import re
                
                # Common words that look like names but aren't
                not_names = {'Add', 'Edit', 'Update', 'Change', 'Remove', 'Delete', 'Create', 
                             'Make', 'Fix', 'Set', 'Get', 'Put', 'Can', 'Could', 'Would', 
                             'Should', 'Please', 'Each', 'Every', 'All', 'The', 'This', 'That',
                             'Updated', 'As', 'To', 'For', 'From', 'With', 'Of', 'No', 'Yes',
                             'Ok', 'Okay', 'Sure', 'Thanks', 'Hello', 'Hi', 'Hey', 'Let', 'Me',
                             'My', 'Your', 'Name', 'User', 'File', 'Files', 'Comment', 'Comments'}
                
                name_match = re.search(r'(?:name[:\s]+|by\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', task)
                if name_match:
                    author_name = name_match.group(1)
                else:
                    # Find all capitalized words, filter out non-names, take first consecutive pair
                    cap_words = re.findall(r'\b([A-Z][a-z]+)\b', task)
                    name_words = [w for w in cap_words if w not in not_names]
                    if len(name_words) >= 2:
                        author_name = f"{name_words[0]} {name_words[1]}"
                    elif len(name_words) == 1:
                        author_name = name_words[0]
                    else:
                        author_name = "Author"
                
                logger.info(f"GUARDRAIL: Extracted author name: {author_name}")
                
                # If we have files and it's an edit task, force an edit on NEXT unedited file
                task_lower = task.lower()
                if remaining_files and ("add" in task_lower or "edit" in task_lower or "comment" in task_lower or "name" in task_lower):
                    next_file = sorted(remaining_files)[0]
                    
                    # Determine comment style based on file extension
                    if next_file.endswith('.py') or next_file.endswith('.ps1') or next_file.endswith('.sh') or next_file.endswith('.yaml') or next_file.endswith('.yml') or next_file.endswith('.txt'):
                        comment_text = f"# {author_name}\n"
                    elif next_file.endswith('.md') or next_file.endswith('.mmd'):
                        comment_text = f"<!-- {author_name} -->\n"
                    elif next_file.endswith('.js') or next_file.endswith('.ts') or next_file.endswith('.json'):
                        comment_text = f"// {author_name}\n"
                    else:
                        comment_text = f"# {author_name}\n"
                    
                    logger.info(f"GUARDRAIL: Editing {next_file} with '{comment_text.strip()}'")
                    return Step(
                        step_id="guardrail_edit",
                        tool="insert_in_file",
                        params={
                            "path": next_file,
                            "position": "start",
                            "text": comment_text
                        },
                        reasoning=f"Editing file {len(already_edited)+1}/{len(editable_files)}: {next_file}",
                    )
                
                # All files edited or no editable files - complete!
                if already_edited:
                    logger.info(f"GUARDRAIL: All {len(already_edited)} files edited - completing")
                    return Step(
                        step_id="guardrail_complete",
                        tool="complete",
                        params={},
                        reasoning=f"Completed editing {len(already_edited)} files",
                    )
                
                # No actionable files found
                logger.info("GUARDRAIL: No actionable files found - completing")
                return Step(
                    step_id="guardrail_complete",
                    tool="complete",
                    params={"error": "No editable files found matching task"},
                    reasoning="No editable files found in scanned directories",
                )
        
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
            history_lines = []
            for i, r in enumerate(history[-10:]):  # Show more history (was 5)
                status_icon = "✓" if r.status.value == "success" else "✗"
                tool_info = f"{r.tool}({r.params})" if r.tool else "unknown"
                output_preview = ""
                if r.output:
                    # Show MORE output for scan results so LLM sees file list
                    preview_len = 2000 if r.tool == "scan_workspace" else 500
                    output_preview = r.output[:preview_len]
                    if len(r.output) > preview_len:
                        output_preview += f"... ({len(r.output) - preview_len} more chars)"
                elif r.error:
                    output_preview = f"ERROR: {r.error[:300]}"
                history_lines.append(f"{status_icon} {tool_info}\n   Output: {output_preview}")
            
            history_text = "\n".join(history_lines)
            context_parts.append(f"COMPLETED STEPS (don't repeat these):\n{history_text}")
        
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
