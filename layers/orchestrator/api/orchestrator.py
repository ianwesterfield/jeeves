"""
Orchestrator API Router

Core endpoints for the agentic reasoning engine:
  - /set-workspace: Set active directory context for execution
  - /clone-workspace: Clone a git repo and set as workspace
  - /next-step: Generate single next step (may detect parallelization)
  - /run-task: Execute full task with SSE streaming (agentic loop)
  - /execute-batch: Execute parallel batch; continue on individual failures
  - /update-state: Update workspace state after tool execution
  - /reset-state: Reset state for new task
"""

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, AsyncGenerator

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas.models import (
    WorkspaceContext,
    SetWorkspaceRequest,
    CloneWorkspaceRequest,
    CloneWorkspaceResponse,
    NextStepRequest,
    NextStepResponse,
    ExecuteBatchRequest,
    BatchResult,
    RunTaskRequest,
    TaskEvent,
)
from services.reasoning_engine import ReasoningEngine
from services.task_planner import TaskPlanner
from services.parallel_executor import ParallelExecutor
from services.memory_connector import MemoryConnector
from services.workspace_state import WorkspaceState, get_workspace_state, reset_workspace_state
from utils.workspace_context import WorkspaceContextManager


logger = logging.getLogger("orchestrator.api")
router = APIRouter(tags=["orchestrator"])

# Configuration
EXECUTOR_API_URL = os.getenv("EXECUTOR_API_URL", "http://executor_api:8005")

# Service instances (singleton pattern)
_workspace_manager: Optional[WorkspaceContextManager] = None
_reasoning_engine: Optional[ReasoningEngine] = None
_task_planner: Optional[TaskPlanner] = None
_parallel_executor: Optional[ParallelExecutor] = None
_memory_connector: Optional[MemoryConnector] = None


def _get_workspace_manager() -> WorkspaceContextManager:
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceContextManager()
    return _workspace_manager


def _get_reasoning_engine() -> ReasoningEngine:
    global _reasoning_engine
    if _reasoning_engine is None:
        _reasoning_engine = ReasoningEngine()
    return _reasoning_engine


def _get_task_planner() -> TaskPlanner:
    global _task_planner
    if _task_planner is None:
        _task_planner = TaskPlanner()
    return _task_planner


def _get_parallel_executor() -> ParallelExecutor:
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelExecutor()
    return _parallel_executor


def _get_memory_connector() -> MemoryConnector:
    global _memory_connector
    if _memory_connector is None:
        _memory_connector = MemoryConnector()
    return _memory_connector


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/set-workspace", response_model=WorkspaceContext)
async def set_workspace(request: SetWorkspaceRequest) -> WorkspaceContext:
    """
    Set the active workspace directory for execution.
    
    All subsequent operations will be scoped to this directory.
    Returns the workspace context with available paths and permissions.
    """
    logger.info(f"Setting workspace to: {request.cwd}")
    
    try:
        manager = _get_workspace_manager()
        context = await manager.set_workspace(
            cwd=request.cwd,
            user_id=request.user_id,
        )
        logger.info(f"Workspace set: {context.cwd} ({len(context.available_paths)} paths)")
        return context
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/clone-workspace", response_model=CloneWorkspaceResponse)
async def clone_workspace(request: CloneWorkspaceRequest) -> CloneWorkspaceResponse:
    """
    Clone a git repository and set it as the active workspace.
    
    Useful for working with remote codebases. The repo is cloned into
    /workspace/<repo_name> and automatically set as the active workspace.
    
    If the directory already exists, it will pull latest changes instead.
    """
    workspace_root = os.getenv("WORKSPACE_ROOT", "/workspace")
    
    # Extract repo name from URL
    # Handles: https://github.com/user/repo.git, git@github.com:user/repo.git
    repo_name = request.target_name
    if not repo_name:
        match = re.search(r"[/:]([^/:]+?)(?:\.git)?$", request.repo_url)
        if match:
            repo_name = match.group(1)
        else:
            repo_name = "cloned_repo"
    
    target_path = Path(workspace_root) / repo_name
    
    try:
        if target_path.exists():
            # Directory exists - pull latest
            logger.info(f"Workspace exists, pulling latest: {target_path}")
            result = subprocess.run(
                ["git", "pull"],
                cwd=str(target_path),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return CloneWorkspaceResponse(
                    success=False,
                    workspace_path=str(target_path),
                    message=f"Git pull failed: {result.stderr}",
                )
            message = f"Pulled latest changes: {result.stdout.strip()}"
        else:
            # Clone the repository
            logger.info(f"Cloning {request.repo_url} to {target_path}")
            cmd = ["git", "clone", request.repo_url, str(target_path)]
            if request.branch:
                cmd.extend(["--branch", request.branch])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min timeout for large repos
            )
            if result.returncode != 0:
                return CloneWorkspaceResponse(
                    success=False,
                    workspace_path=str(target_path),
                    message=f"Git clone failed: {result.stderr}",
                )
            message = f"Cloned successfully"
        
        # Set as active workspace
        manager = _get_workspace_manager()
        context = await manager.set_workspace(
            cwd=str(target_path),
            user_id=request.user_id,
        )
        
        return CloneWorkspaceResponse(
            success=True,
            workspace_path=str(target_path),
            message=message,
            context=context,
        )
        
    except subprocess.TimeoutExpired:
        return CloneWorkspaceResponse(
            success=False,
            workspace_path=str(target_path),
            message="Clone operation timed out",
        )
    except Exception as e:
        logger.error(f"Clone failed: {e}")
        return CloneWorkspaceResponse(
            success=False,
            workspace_path=str(target_path),
            message=f"Clone failed: {str(e)}",
        )


@router.post("/next-step", response_model=NextStepResponse)
async def next_step(request: NextStepRequest) -> NextStepResponse:
    """
    Generate the next reasoning step for a task.
    
    Uses external WorkspaceState for ground-truth tracking (not LLM memory).
    The state is injected into the prompt so LLM doesn't need to track it.
    
    The response includes:
      - tool: The tool to execute
      - params: Parameters for the tool
      - batch_id: If parallelizable, a batch identifier
      - reasoning: Short status note for the user
    """
    logger.info(f"Generating next step for task: {request.task[:50]}...")
    
    reasoning_engine = _get_reasoning_engine()
    task_planner = _get_task_planner()
    memory_connector = _get_memory_connector()
    
    # Get or create workspace state
    workspace_state = get_workspace_state()
    
    # 1. Retrieve relevant patterns from memory
    memory_context = []
    if request.user_id:
        memory_context = await memory_connector.search_patterns(
            query=request.task,
            user_id=request.user_id,
            top_k=3,
        )
        
        # Extract user info from memory results and store in state
        for mem in memory_context:
            payload = mem.get("payload", {})
            if payload.get("facts"):
                facts = payload["facts"]
                if facts.get("names"):
                    workspace_state.user_info["name"] = facts["names"][0]
                if facts.get("emails"):
                    workspace_state.user_info["email"] = facts["emails"][0]
    
    # 2. Generate next step using reasoning engine with external state
    step = await reasoning_engine.generate_next_step(
        task=request.task,
        history=request.history or [],
        memory_context=memory_context,
        workspace_context=request.workspace_context,
        workspace_state=workspace_state,  # Pass external state
    )
    
    # 3. Check if step can be parallelized
    if step.tool == "batch":
        # Task planner detected parallel opportunity
        batch_steps = await task_planner.expand_batch(step, request.workspace_context)
        return NextStepResponse(
            tool="batch",
            params={"steps": [s.model_dump() for s in batch_steps]},
            batch_id=step.batch_id,
            reasoning=step.reasoning,
            is_batch=True,
        )
    
    return NextStepResponse(
        tool=step.tool,
        params=step.params,
        batch_id=None,
        reasoning=step.reasoning,
        is_batch=False,
    )


@router.post("/execute-batch", response_model=BatchResult)
async def execute_batch(request: ExecuteBatchRequest) -> BatchResult:
    """
    Execute a batch of steps in parallel.
    
    Uses asyncio.gather to run all steps concurrently.
    Individual failures do not cascade - other steps continue.
    Returns aggregated results with success/failure counts.
    """
    logger.info(f"Executing batch {request.batch_id} with {len(request.steps)} steps")
    
    executor = _get_parallel_executor()
    memory_connector = _get_memory_connector()
    
    # Execute batch
    result = await executor.execute_batch(
        steps=request.steps,
        batch_id=request.batch_id,
        workspace_context=request.workspace_context,
    )
    
    # Store execution trace for learning
    if request.user_id:
        await memory_connector.store_execution_trace(
            user_id=request.user_id,
            batch_id=request.batch_id,
            result=result,
        )
    
    logger.info(
        f"Batch {request.batch_id} complete: "
        f"{result.successful_count} OK, {result.failed_count} failed"
    )
    
    return result

# ============================================================================
# State Management Endpoints
# ============================================================================

from pydantic import BaseModel, Field
from typing import Dict, Any


class UpdateStateRequest(BaseModel):
    """Request to update workspace state after tool execution."""
    tool: str = Field(..., description="Tool that was executed")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    output: str = Field(default="", description="Tool output")
    success: bool = Field(..., description="Whether execution succeeded")


class StateResponse(BaseModel):
    """Response with current workspace state summary."""
    scanned_paths: list
    total_files: int
    total_dirs: int
    edited_files: list
    completed_steps: int
    user_info: Dict[str, str]


@router.post("/update-state", response_model=StateResponse)
async def update_state(request: UpdateStateRequest) -> StateResponse:
    """
    Update workspace state after a tool execution.
    
    Called by the filter after each tool is executed to keep
    the external state in sync with reality.
    """
    state = get_workspace_state()
    
    # Update state based on tool result
    state.update_from_step(
        tool=request.tool,
        params=request.params,
        output=request.output,
        success=request.success,
    )
    
    logger.debug(f"State updated: {len(state.completed_steps)} steps, {len(state.files)} files")
    
    return StateResponse(
        scanned_paths=list(state.scanned_paths),
        total_files=len(state.files),
        total_dirs=len(state.dirs),
        edited_files=list(state.edited_files),
        completed_steps=len(state.completed_steps),
        user_info=state.user_info,
    )


@router.post("/reset-state", response_model=StateResponse)
async def reset_state() -> StateResponse:
    """
    Reset workspace state for a new task.
    
    Called at the start of a new task to clear previous state.
    User info is preserved across resets.
    """
    state = reset_workspace_state()
    
    logger.info("Workspace state reset")
    
    return StateResponse(
        scanned_paths=[],
        total_files=0,
        total_dirs=0,
        edited_files=[],
        completed_steps=0,
        user_info=state.user_info,
    )


@router.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """
    Get current workspace state.
    
    Returns the current external state for debugging/inspection.
    """
    state = get_workspace_state()
    
    return StateResponse(
        scanned_paths=list(state.scanned_paths),
        total_files=len(state.files),
        total_dirs=len(state.dirs),
        edited_files=list(state.edited_files),
        completed_steps=len(state.completed_steps),
        user_info=state.user_info,
    )


class SetUserInfoRequest(BaseModel):
    """Request to set user info in workspace state."""
    name: str = Field(None, description="User's name")
    email: str = Field(None, description="User's email")


@router.post("/set-user-info")
async def set_user_info(request: SetUserInfoRequest) -> Dict[str, str]:
    """
    Set user info directly in workspace state.
    
    Called when user info is extracted from the task or conversation.
    """
    state = get_workspace_state()
    
    if request.name:
        state.user_info["name"] = request.name
    if request.email:
        state.user_info["email"] = request.email
    
    logger.info(f"User info set: {state.user_info}")
    
    return state.user_info


# ============================================================================
# Streaming Task Execution (Agentic Loop)
# ============================================================================

async def _execute_tool(workspace_root: str, tool: str, params: dict) -> dict:
    """Execute a tool via the executor API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Resolve relative paths in params
            resolved_params = params.copy()
            if "path" in resolved_params:
                path = resolved_params["path"]
                project_name = workspace_root.rstrip("/").split("/")[-1].lower()
                
                if path == "." or path == "" or path.lower() == project_name:
                    resolved_params["path"] = workspace_root
                elif not path.startswith("/"):
                    resolved_params["path"] = f"{workspace_root}/{path}"
            
            resp = await client.post(
                f"{EXECUTOR_API_URL}/api/execute/tool",
                json={
                    "tool": tool,
                    "params": resolved_params,
                    "workspace_context": {
                        "workspace_root": "/workspace",
                        "cwd": workspace_root,
                        "allow_file_write": True,
                        "allow_shell_commands": True,
                        "allowed_languages": ["python"],
                    },
                },
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "output": None, "error": f"HTTP {resp.status_code}"}
                
        except Exception as e:
            return {"success": False, "output": None, "error": str(e)}


def _format_result_block(tool: str, params: dict, output: str, reasoning: str) -> str:
    """Format tool result as context block for LLM."""
    if tool == "scan_workspace":
        return f"""### Workspace Files ###
⚠️ THIS IS THE COMPLETE FILE LIST. Do not invent files not shown below.

```
{output or "(no files found)"}
```

### End Workspace Files ###
"""
    
    elif tool == "read_file":
        path = params.get("path", "file")
        return f"""### File Content: {path} ###
**ACTUAL CONTENT (do not invent or paraphrase):**

```
{output or "(empty)"}
```
⚠️ The above is the COMPLETE file content. Do not add anything not shown above.
### End File Content ###
"""
    
    elif tool in ("write_file", "append_to_file", "replace_in_file", "insert_in_file"):
        path = params.get("path", "file")
        content = params.get("content") or params.get("text") or params.get("new_text") or ""
        content_snippet = content[:200] + "..." if len(content) > 200 else content
        return f"""### File Operation Result ###
**Operation:** {tool} on {path}
**Status:** ✅ Success
**EXACT content written:** `{content_snippet}`
⚠️ Report ONLY this exact content. Do not embellish or reformat.
### End File Operation ###
"""
    
    elif tool == "execute_shell":
        cmd = params.get("command", "")
        return f"""### Shell Command Result ###
**Command:** `{cmd}`
**Status:** ✅ Success
**Output:** {output or "(no output)"}
### End Shell Command ###
"""
    
    else:
        return f"""### {tool} Result ###
**Reasoning:** {reasoning}

```
{output or "(no output)"}
```
### End Result ###
"""


async def _run_task_generator(request: RunTaskRequest) -> AsyncGenerator[str, None]:
    """
    Generator that yields SSE events during task execution.
    
    This is the agentic loop moved from the filter to the orchestrator.
    """
    reasoning_engine = _get_reasoning_engine()
    workspace_state = get_workspace_state()
    
    # Reset state for new task
    reset_workspace_state()
    workspace_state = get_workspace_state()
    
    # Build task with memory context
    user_text = request.task
    if request.memory_context:
        user_info = []
        for item in request.memory_context[:3]:
            facts = item.get('facts')
            if facts and isinstance(facts, dict):
                for fact_type, fact_value in facts.items():
                    user_info.append(f"{fact_type}: {fact_value}")
            else:
                text = item.get('user_text', '')
                if text:
                    user_info.append(text)
        
        if user_info:
            info_block = "\n".join(f"- {info}" for info in user_info)
            user_text = f"User information from memory:\n{info_block}\n\nTask: {user_text}"
    
    # Set workspace
    try:
        manager = _get_workspace_manager()
        await manager.set_workspace(cwd=request.workspace_root, user_id=request.user_id)
    except Exception as e:
        yield f"data: {json.dumps({'event_type': 'error', 'status': f'Failed to set workspace: {e}', 'done': True})}\n\n"
        return
    
    # Emit initial status
    yield f"data: {json.dumps({'event_type': 'status', 'status': '✨ Thinking...', 'done': False})}\n\n"
    
    step_history = []
    all_results = []
    edit_tools = ("write_file", "replace_in_file", "insert_in_file", "append_to_file")
    
    for step_num in range(1, request.max_steps + 1):
        try:
            # Get next step from reasoning engine
            step = await reasoning_engine.generate_next_step(
                task=user_text,
                history=[],  # History is now in workspace_state
                memory_context=[],
                workspace_context=None,
                workspace_state=workspace_state,
            )
            
            tool = step.tool
            params = step.params
            reasoning = step.reasoning or ""
            
            # Handle completion
            if tool == "complete":
                if params.get("error"):
                    all_results.append(f"""### Orchestrator Error ###
{params.get('error')}
### End Error ###
""")
                break
            
            # Build status message
            tool_path = params.get("path", "")
            short_path = tool_path.split("/")[-1].split("\\")[-1] if tool_path else ""
            reasoning_snippet = reasoning[:40] + "..." if len(reasoning) > 40 else reasoning
            
            status_map = {
                "scan_workspace": f"✨ {reasoning_snippet}, scanning 🔍 {short_path or 'the workspace'}",
                "read_file": f"✨ {reasoning_snippet}, reading 📖 {short_path}",
                "execute_shell": f"✨ {reasoning_snippet}, running 🖥️ {params.get('command', '')[:30]}...",
            }
            
            if tool in edit_tools:
                edit_count = sum(1 for h in step_history if h["tool"] in edit_tools and h.get("success"))
                if edit_count == 0:
                    status = f"✨ {reasoning_snippet}, editing ✏️ {short_path}"
                else:
                    status = None  # Silent for batch edits
            else:
                status = status_map.get(tool, f"✨ {reasoning_snippet}")
            
            if status:
                yield f"data: {json.dumps({'event_type': 'status', 'step_num': step_num, 'tool': tool, 'status': status, 'done': False})}\n\n"
            
            # Execute tool
            result = await _execute_tool(request.workspace_root, tool, params)
            success = result.get("success", False)
            output = result.get("output")
            error = result.get("error")
            
            # Update workspace state
            workspace_state.update_from_step(
                tool=tool,
                params={k: v for k, v in params.items() if k not in ("content", "text", "new_text")},
                output=output[:50000] if output else "",
                success=success,
            )
            
            # Record step
            step_history.append({
                "step_id": f"step_{step_num}",
                "tool": tool,
                "params": {k: v for k, v in params.items() if k != "content"},
                "success": success,
                "output": output[:10000] if output else None,
                "error": error,
            })
            
            # Format result for context
            if success:
                result_block = _format_result_block(tool, params, output, reasoning)
                all_results.append(result_block)
                
                # Yield result event
                yield f"data: {json.dumps({'event_type': 'result', 'step_num': step_num, 'tool': tool, 'result': {'success': True, 'output_preview': (output[:500] if output else None)}, 'done': False})}\n\n"
            else:
                error_snippet = (error[:40] + "...") if error and len(error) > 40 else (error or "unknown")
                yield f"data: {json.dumps({'event_type': 'status', 'step_num': step_num, 'tool': tool, 'status': f'⚠️ Failed: {error_snippet}', 'done': False})}\n\n"
                
                all_results.append(f"""### Step {step_num} Error ###
**Tool:** {tool}
**Reasoning:** {reasoning}
**Error:** {error or "Unknown error"}
### End Error ###
""")
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Step {step_num} error: {e}")
            yield f"data: {json.dumps({'event_type': 'error', 'step_num': step_num, 'status': f'Error: {str(e)[:50]}', 'done': False})}\n\n"
    
    # Build final context
    if all_results:
        file_ops = sum(1 for r in all_results if "File Operation Result" in r)
        
        header = """### TASK ALREADY COMPLETED ###
**CRITICAL:** The actions below have ALREADY been executed. Do NOT hallucinate details.
- Speak in PAST TENSE: "I added...", "I updated..."
- Report EXACTLY what the logs show - do NOT invent content or formats
- The "Content added" field shows the EXACT text that was written
"""
        if file_ops > 0:
            header += f"**Files modified:** {file_ops}\n"
        header += "### Action Log ###\n"
        
        footer = """### End Action Log ###

⚠️ FINAL INSTRUCTION: Summarize ONLY what is shown in the logs above.
- Do NOT invent file names, paths, or content
- Do NOT add files that aren't listed
- Do NOT make up file contents
- If you don't see data above, say "I couldn't complete that"
"""
        final_context = header + "\n".join(all_results) + footer
    else:
        final_context = None
    
    # Emit completion with final context
    yield f"data: {json.dumps({'event_type': 'complete', 'status': '✅ Ready', 'result': {'context': final_context, 'steps_executed': len(step_history)}, 'done': True})}\n\n"


@router.post("/run-task")
async def run_task(request: RunTaskRequest):
    """
    Execute a complete task with Server-Sent Events (SSE) streaming.
    
    This moves the entire agentic loop from the filter to the orchestrator.
    The filter can now simply:
      1. POST /run-task
      2. Stream SSE events
      3. Forward status events to __event_emitter__
      4. Inject final context into conversation
    
    Event types:
      - status: UI status update (step_num, tool, status, done)
      - result: Step result (step_num, tool, result, done)
      - error: Error occurred (status, done)
      - complete: Task finished (result.context, result.steps_executed, done)
    """
    return StreamingResponse(
        _run_task_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )