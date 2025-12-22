"""
Executor Data Models

Schemas for code execution requests and results.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    POWERSHELL = "powershell"
    PWSH = "pwsh"
    NODE = "node"
    JAVASCRIPT = "javascript"
    JS = "js"
    TYPESCRIPT = "typescript"
    TS = "ts"
    BASH = "bash"
    SH = "sh"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    R = "r"
    JULIA = "julia"


class FileOperation(str, Enum):
    """File operation types."""
    READ = "read"
    WRITE = "write"
    LIST = "list"
    DELETE = "delete"


# ============================================================================
# Workspace Context (shared with Orchestrator)
# ============================================================================

class WorkspaceContext(BaseModel):
    """Workspace scoping model (matches Orchestrator)."""
    cwd: str = Field(..., description="Current working directory")
    workspace_root: str = Field(..., description="Root directory (sandbox boundary)")
    available_paths: List[str] = Field(default_factory=list)
    parallel_enabled: bool = Field(default=False)
    max_parallel_tasks: int = Field(default=4)
    allowed_languages: List[str] = Field(default_factory=lambda: ["python", "powershell", "node"])
    allow_code_execution: bool = Field(default=False)
    allow_file_write: bool = Field(default=False)
    allow_shell_commands: bool = Field(default=False)
    max_execution_time: int = Field(default=30)


# ============================================================================
# Tool Execution
# ============================================================================

class ToolCallRequest(BaseModel):
    """Request to execute a tool (polymorphic)."""
    tool: str = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    workspace_context: Optional[WorkspaceContext] = None


class ToolResult(BaseModel):
    """Result of tool execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


# ============================================================================
# Code Execution
# ============================================================================

class CodeExecutionRequest(BaseModel):
    """Request to execute code."""
    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="Code to execute")
    timeout: int = Field(default=30, description="Timeout in seconds")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    workspace_context: Optional[WorkspaceContext] = None


class CodeExecutionResult(BaseModel):
    """Result of code execution."""
    success: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: int = 0
    execution_time: float = 0.0


# ============================================================================
# Shell Execution
# ============================================================================

class ShellCommandRequest(BaseModel):
    """Request to execute a shell command."""
    command: str = Field(..., description="Command to execute")
    timeout: int = Field(default=30, description="Timeout in seconds")
    workspace_context: Optional[WorkspaceContext] = None


class ShellCommandResult(BaseModel):
    """Result of shell command execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    exit_code: int = 0


# ============================================================================
# File Operations
# ============================================================================

class FileOperationRequest(BaseModel):
    """Request for file operations."""
    operation: FileOperation = Field(..., description="Operation type")
    path: str = Field(..., description="File path")
    content: Optional[str] = Field(None, description="Content for write operations")
    workspace_context: Optional[WorkspaceContext] = None


class FileOperationResult(BaseModel):
    """Result of file operation."""
    success: bool
    data: Optional[Any] = None  # File content or directory listing
    error: Optional[str] = None
