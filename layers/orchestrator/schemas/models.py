"""
Orchestrator Data Models

Core schemas for the agentic reasoning engine:
  - WorkspaceContext: Directory scoping and permissions
  - Step: Single tool invocation
  - Batch: Group of parallelizable steps
  - Results: Execution outcomes
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class StepStatus(str, Enum):
    """Status of a step execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorType(str, Enum):
    """Categories of execution errors."""
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    INVALID_PARAMS = "invalid_params"
    EXECUTION_ERROR = "execution_error"
    SANDBOX_VIOLATION = "sandbox_violation"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN = "unknown"


# ============================================================================
# Workspace Context
# ============================================================================

class WorkspaceContext(BaseModel):
    """
    Workspace scoping model.
    
    Defines the active directory and permissions for all operations.
    Set via /set-workspace endpoint before executing steps.
    """
    cwd: str = Field(..., description="Current working directory")
    workspace_root: str = Field(..., description="Root directory (sandbox boundary)")
    available_paths: List[str] = Field(default_factory=list, description="Cached directory listing")
    parallel_enabled: bool = Field(default=False, description="Admin valve: allow parallel execution")
    max_parallel_tasks: int = Field(default=4, description="Max concurrent tasks")
    
    # Permissions
    allowed_languages: List[str] = Field(
        default_factory=lambda: ["python", "powershell", "node"],
        description="Languages user can execute"
    )
    allow_code_execution: bool = Field(default=False, description="Can run arbitrary code")
    allow_file_write: bool = Field(default=False, description="Can write files")
    allow_shell_commands: bool = Field(default=False, description="Can run shell commands")
    max_execution_time: int = Field(default=30, description="Timeout per task (seconds)")


class SetWorkspaceRequest(BaseModel):
    """Request to set the active workspace."""
    cwd: str = Field(..., description="Directory to set as workspace")
    user_id: Optional[str] = Field(None, description="User ID for permission lookup")


class CloneWorkspaceRequest(BaseModel):
    """Request to clone a git repository as workspace."""
    repo_url: str = Field(..., description="Git repository URL to clone")
    branch: Optional[str] = Field(None, description="Branch to checkout (default: main)")
    target_name: Optional[str] = Field(None, description="Target directory name (default: repo name)")
    user_id: Optional[str] = Field(None, description="User ID for permission lookup")


class CloneWorkspaceResponse(BaseModel):
    """Response after cloning a workspace."""
    success: bool
    workspace_path: str = Field(..., description="Path to cloned workspace")
    message: str
    context: Optional[WorkspaceContext] = None


# ============================================================================
# Steps
# ============================================================================

class Step(BaseModel):
    """
    Single tool invocation step.
    
    Represents one atomic operation in a task decomposition.
    """
    step_id: str = Field(..., description="Unique step identifier")
    tool: str = Field(..., description="Tool to invoke (e.g., 'execute_code', 'read_file')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    batch_id: Optional[str] = Field(None, description="Batch ID if part of parallel group")
    reasoning: str = Field(default="", description="Why this step was chosen")
    
    # Execution state
    status: StepStatus = Field(default=StepStatus.PENDING)
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this depends on")


class StepResult(BaseModel):
    """Result of executing a single step."""
    step_id: str
    tool: Optional[str] = None  # Tool that was executed
    params: Optional[Dict[str, Any]] = None  # Params used (excluding large content)
    status: StepStatus
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    
    
class ErrorMetadata(BaseModel):
    """Detailed error information for failed steps."""
    step_id: str
    error: str
    error_type: ErrorType = ErrorType.UNKNOWN
    recoverable: bool = False
    suggestion: Optional[str] = None


# ============================================================================
# Requests / Responses
# ============================================================================

class NextStepRequest(BaseModel):
    """Request to generate the next reasoning step."""
    task: str = Field(..., description="User's task or intent")
    history: List[StepResult] = Field(default_factory=list, description="Previous step results")
    workspace_context: Optional[WorkspaceContext] = None
    user_id: Optional[str] = None


class NextStepResponse(BaseModel):
    """Response with the next step to execute."""
    tool: str = Field(..., description="Tool to invoke")
    params: Dict[str, Any] = Field(default_factory=dict)
    batch_id: Optional[str] = Field(None, description="Batch ID if parallelizable")
    reasoning: str = Field(..., description="Explanation of the step")
    is_batch: bool = Field(default=False, description="Whether this is a batch of steps")


# ============================================================================
# Streaming Task Execution
# ============================================================================

class RunTaskRequest(BaseModel):
    """Request to run a complete task with streaming updates."""
    task: str = Field(..., description="User's task description")
    workspace_root: str = Field(..., description="Workspace root path")
    user_id: Optional[str] = Field(None, description="User ID")
    memory_context: Optional[List[Dict[str, Any]]] = Field(None, description="User context from memory")
    max_steps: int = Field(default=100, description="Maximum steps before forced completion")


class TaskEvent(BaseModel):
    """Server-sent event during task execution."""
    event_type: str = Field(..., description="Event type: status, result, complete, error")
    step_num: int = Field(default=0, description="Current step number")
    tool: Optional[str] = Field(None, description="Tool being executed")
    status: str = Field(default="", description="Status message for UI")
    result: Optional[Dict[str, Any]] = Field(None, description="Step result data")
    done: bool = Field(default=False, description="Whether task is complete")


class ExecuteBatchRequest(BaseModel):
    """Request to execute a batch of steps in parallel."""
    steps: List[Step] = Field(..., description="Steps to execute")
    batch_id: str = Field(..., description="Batch identifier")
    workspace_context: Optional[WorkspaceContext] = None
    user_id: Optional[str] = None


class BatchResult(BaseModel):
    """Result of executing a batch of steps."""
    batch_id: str
    successful: List[StepResult] = Field(default_factory=list)
    failed: List[ErrorMetadata] = Field(default_factory=list)
    duration: float = 0.0
    
    @property
    def successful_count(self) -> int:
        return len(self.successful)
    
    @property
    def failed_count(self) -> int:
        return len(self.failed)
    
    def format_for_chat(self) -> str:
        """Format batch result for chat display."""
        total = self.successful_count + self.failed_count
        
        if self.failed_count == 0:
            return f"✓ {total} completed in {self.duration:.1f}s\n\nAll tasks successful."
        
        # Show failures (condensed)
        failures = "\n".join([
            f"  • {f.step_id}: {f.error_type.value}"
            for f in self.failed[:5]
        ])
        
        if len(self.failed) > 5:
            failures += f"\n  ... and {len(self.failed) - 5} more"
        
        return (
            f"✓ {total} completed in {self.duration:.1f}s "
            f"({self.successful_count} OK, {self.failed_count} failed)\n\n"
            f"**Failures:**\n{failures}\n\n"
            f"Continuing with next step using successful results..."
        )
