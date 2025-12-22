"""
Parallel Executor - Batch Execution with Partial Failure Handling

Executes multiple steps concurrently using asyncio.gather.
Individual failures do not cascade - sibling tasks continue.
"""

import asyncio
import logging
import os
import time
from typing import List, Optional

import httpx

from schemas.models import (
    Step,
    StepResult,
    StepStatus,
    BatchResult,
    ErrorMetadata,
    ErrorType,
    WorkspaceContext,
)


logger = logging.getLogger("orchestrator.executor")

# Executor service URL
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL", "http://executor_api:8005")


class ParallelExecutor:
    """
    Parallel batch executor with partial failure handling.
    
    Uses asyncio.gather to execute steps concurrently.
    Failures are captured but don't stop sibling executions.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.executor_url = EXECUTOR_BASE_URL
    
    async def execute_batch(
        self,
        steps: List[Step],
        batch_id: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> BatchResult:
        """
        Execute a batch of steps in parallel.
        
        Args:
            steps: List of steps to execute
            batch_id: Batch identifier for logging
            workspace_context: Execution context (limits, permissions)
            
        Returns:
            BatchResult with successful and failed step lists
        """
        start_time = time.time()
        
        # Determine concurrency limit
        max_concurrent = 4
        if workspace_context:
            max_concurrent = workspace_context.max_parallel_tasks
        
        logger.info(f"Executing batch {batch_id}: {len(steps)} steps, max {max_concurrent} concurrent")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(step: Step) -> tuple[Step, StepResult | Exception]:
            async with semaphore:
                try:
                    result = await self._execute_single_step(step, workspace_context)
                    return (step, result)
                except Exception as e:
                    return (step, e)
        
        # Execute all steps concurrently
        tasks = [execute_with_semaphore(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        successful = []
        failed = []
        
        for item in results:
            if isinstance(item, Exception):
                # asyncio.gather caught an exception
                logger.error(f"Batch task exception: {item}")
                continue
            
            step, result = item
            
            if isinstance(result, Exception):
                # Step execution raised an exception
                error_meta = self._classify_error(step, result)
                failed.append(error_meta)
            elif isinstance(result, StepResult):
                if result.status == StepStatus.SUCCESS:
                    successful.append(result)
                else:
                    error_meta = ErrorMetadata(
                        step_id=result.step_id,
                        error=result.error or "Unknown error",
                        error_type=ErrorType.EXECUTION_ERROR,
                        recoverable=True,
                    )
                    failed.append(error_meta)
        
        duration = time.time() - start_time
        
        return BatchResult(
            batch_id=batch_id,
            successful=successful,
            failed=failed,
            duration=duration,
        )
    
    async def _execute_single_step(
        self,
        step: Step,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> StepResult:
        """
        Execute a single step via the Executor service.
        
        Args:
            step: Step to execute
            workspace_context: Execution context
            
        Returns:
            StepResult with status and output
        """
        start_time = time.time()
        
        # Build request for executor
        request_body = {
            "tool": step.tool,
            "params": step.params,
        }
        
        if workspace_context:
            request_body["workspace_context"] = workspace_context.model_dump()
        
        try:
            # Call executor service
            response = await self.client.post(
                f"{self.executor_url}/api/execute/tool",
                json=request_body,
                timeout=workspace_context.max_execution_time if workspace_context else 30,
            )
            response.raise_for_status()
            
            data = response.json()
            execution_time = time.time() - start_time
            
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.SUCCESS if data.get("success") else StepStatus.FAILED,
                output=data.get("output"),
                error=data.get("error"),
                execution_time=execution_time,
            )
            
        except httpx.TimeoutException:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error="Execution timeout",
                execution_time=time.time() - start_time,
            )
        except httpx.HTTPStatusError as e:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=f"HTTP error: {e.response.status_code}",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _classify_error(self, step: Step, error: Exception) -> ErrorMetadata:
        """Classify an exception into an ErrorMetadata object."""
        error_str = str(error)
        
        # Classify error type
        if "timeout" in error_str.lower():
            error_type = ErrorType.TIMEOUT
            recoverable = True
        elif "permission" in error_str.lower():
            error_type = ErrorType.PERMISSION_DENIED
            recoverable = False
        elif "sandbox" in error_str.lower():
            error_type = ErrorType.SANDBOX_VIOLATION
            recoverable = False
        elif "resource" in error_str.lower() or "memory" in error_str.lower():
            error_type = ErrorType.RESOURCE_LIMIT
            recoverable = True
        else:
            error_type = ErrorType.EXECUTION_ERROR
            recoverable = True
        
        return ErrorMetadata(
            step_id=step.step_id,
            error=error_str[:200],  # Truncate long errors
            error_type=error_type,
            recoverable=recoverable,
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
