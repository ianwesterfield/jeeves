"""
Orchestrator API Router

Core endpoints for the agentic reasoning engine:
  - /set-workspace: Set active directory context for execution
  - /clone-workspace: Clone a git repo and set as workspace
  - /next-step: Generate single next step (may detect parallelization)
  - /execute-batch: Execute parallel batch; continue on individual failures
  - /update-state: Update workspace state after tool execution
  - /reset-state: Reset state for new task
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from schemas.models import (
    WorkspaceContext,
    SetWorkspaceRequest,
    CloneWorkspaceRequest,
    CloneWorkspaceResponse,
    NextStepRequest,
    NextStepResponse,
    ExecuteBatchRequest,
    BatchResult,
)
from services.reasoning_engine import ReasoningEngine
from services.task_planner import TaskPlanner
from services.parallel_executor import ParallelExecutor
from services.memory_connector import MemoryConnector
from services.workspace_state import WorkspaceState, get_workspace_state, reset_workspace_state
from utils.workspace_context import WorkspaceContextManager


logger = logging.getLogger("orchestrator.api")
router = APIRouter(tags=["orchestrator"])

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