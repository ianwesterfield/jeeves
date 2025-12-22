"""
Orchestrator API Router

Core endpoints for the agentic reasoning engine:
  - /set-workspace: Set active directory context for execution
  - /next-step: Generate single next step (may detect parallelization)
  - /execute-batch: Execute parallel batch; continue on individual failures
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from schemas.models import (
    WorkspaceContext,
    SetWorkspaceRequest,
    NextStepRequest,
    NextStepResponse,
    ExecuteBatchRequest,
    BatchResult,
)
from services.reasoning_engine import ReasoningEngine
from services.task_planner import TaskPlanner
from services.parallel_executor import ParallelExecutor
from services.memory_connector import MemoryConnector
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


@router.post("/next-step", response_model=NextStepResponse)
async def next_step(request: NextStepRequest) -> NextStepResponse:
    """
    Generate the next reasoning step for a task.
    
    Uses the LLM to analyze the task and history, then produces
    either a single step or a batch of parallelizable steps.
    
    The response includes:
      - tool: The tool to execute
      - params: Parameters for the tool
      - batch_id: If parallelizable, a batch identifier
      - reasoning: Explanation of why this step was chosen
    """
    logger.info(f"Generating next step for task: {request.task[:50]}...")
    
    reasoning_engine = _get_reasoning_engine()
    task_planner = _get_task_planner()
    memory_connector = _get_memory_connector()
    
    # 1. Retrieve relevant patterns from memory
    memory_context = []
    if request.user_id:
        memory_context = await memory_connector.search_patterns(
            query=request.task,
            user_id=request.user_id,
            top_k=3,
        )
    
    # 2. Generate next step using reasoning engine
    step = await reasoning_engine.generate_next_step(
        task=request.task,
        history=request.history or [],
        memory_context=memory_context,
        workspace_context=request.workspace_context,
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
