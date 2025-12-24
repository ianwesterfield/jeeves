"""
Executor API Router

Core endpoints for code execution:
  - /tool: Execute any registered tool (polymorphic)
  - /code: Execute code in a specific language
  - /shell: Execute shell command
  - /file: File operations
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from schemas.models import (
    ToolCallRequest,
    ToolResult,
    CodeExecutionRequest,
    CodeExecutionResult,
    ShellCommandRequest,
    ShellCommandResult,
    FileOperationRequest,
    FileOperationResult,
)
from services.polyglot_handler import PolyglotHandler
from services.shell_handler import ShellHandler
from services.file_handler import FileHandler
from utils.permissions import PermissionChecker


logger = logging.getLogger("executor.api")
router = APIRouter(tags=["executor"])

# Service instances (singleton pattern)
_polyglot_handler: Optional[PolyglotHandler] = None
_shell_handler: Optional[ShellHandler] = None
_file_handler: Optional[FileHandler] = None
_permission_checker: Optional[PermissionChecker] = None


def _get_polyglot_handler() -> PolyglotHandler:
    global _polyglot_handler
    if _polyglot_handler is None:
        _polyglot_handler = PolyglotHandler()
    return _polyglot_handler


def _get_shell_handler() -> ShellHandler:
    global _shell_handler
    if _shell_handler is None:
        _shell_handler = ShellHandler()
    return _shell_handler


def _get_file_handler() -> FileHandler:
    global _file_handler
    if _file_handler is None:
        _file_handler = FileHandler()
    return _file_handler


def _get_permission_checker() -> PermissionChecker:
    global _permission_checker
    if _permission_checker is None:
        _permission_checker = PermissionChecker()
    return _permission_checker


# ============================================================================
# Tool Registry
# ============================================================================


async def _handle_none(req) -> dict:
    """
    Handle 'none' action - no change needed (idempotent skip).
    
    Returns success immediately without any file operations.
    Used when the code planner determines the change is already present.
    """
    reason = req.params.get("reason", "no change needed")
    path = req.params.get("path", "")
    return {
        "success": True,
        "output": f"Skipped: {reason}" + (f" ({path})" if path else ""),
        "skipped": True,
        "reason": reason,
    }


TOOL_HANDLERS = {
    # No-op handler for idempotent operations
    "none": _handle_none,
    
    "execute_code": lambda req: _get_polyglot_handler().execute(
        language=req.params.get("language", "python"),
        code=req.params.get("code", ""),
        timeout=req.params.get("timeout", 30),
        workspace_context=req.workspace_context,
    ),
    "execute_shell": lambda req: _get_shell_handler().execute(
        command=req.params.get("command", ""),
        timeout=req.params.get("timeout", 30),
        workspace_context=req.workspace_context,
    ),
    "read_file": lambda req: _get_file_handler().read(
        path=req.params.get("path", ""),
        workspace_context=req.workspace_context,
    ),
    "write_file": lambda req: _get_file_handler().write(
        path=req.params.get("path", ""),
        content=req.params.get("content", ""),
        workspace_context=req.workspace_context,
    ),
    "replace_in_file": lambda req: _get_file_handler().replace_in_file(
        path=req.params.get("path", ""),
        old_text=req.params.get("old_text", ""),
        new_text=req.params.get("new_text", ""),
        workspace_context=req.workspace_context,
    ),
    "insert_in_file": lambda req: _get_file_handler().insert_in_file(
        path=req.params.get("path", ""),
        position=req.params.get("position", "end"),
        text=req.params.get("text", ""),
        anchor=req.params.get("anchor"),
        workspace_context=req.workspace_context,
    ),
    "append_to_file": lambda req: _get_file_handler().append_to_file(
        path=req.params.get("path", ""),
        content=req.params.get("content", ""),
        workspace_context=req.workspace_context,
    ),
    "list_files": lambda req: _get_file_handler().list_dir(
        path=req.params.get("path", "."),
        workspace_context=req.workspace_context,
    ),
    "scan_workspace": lambda req: _get_file_handler().scan_workspace(
        path=req.params.get("path", "."),
        pattern=req.params.get("pattern", "*"),
        workspace_context=req.workspace_context,
    ),
    "analyze_code": lambda req: _get_file_handler().analyze(
        path=req.params.get("path", ""),
        workspace_context=req.workspace_context,
    ),
}


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/tool", response_model=ToolResult)
async def execute_tool(request: ToolCallRequest) -> ToolResult:
    """
    Execute any registered tool (polymorphic endpoint).
    
    This is the main entry point for the orchestrator.
    Routes to the appropriate handler based on tool name.
    """
    logger.info(f"Executing tool: {request.tool}")
    
    # Check permissions
    checker = _get_permission_checker()
    allowed, reason = checker.check_tool_permission(
        tool=request.tool,
        params=request.params,
        workspace_context=request.workspace_context,
    )
    
    if not allowed:
        return ToolResult(
            success=False,
            output=None,
            error=f"Permission denied: {reason}",
            execution_time=0.0,
        )
    
    # Get handler
    handler = TOOL_HANDLERS.get(request.tool)
    if not handler:
        return ToolResult(
            success=False,
            output=None,
            error=f"Unknown tool: {request.tool}",
            execution_time=0.0,
        )
    
    # Execute
    try:
        import time
        import asyncio
        import inspect
        import json
        start = time.time()
        
        # Call handler - may return coroutine or result directly
        handler_result = handler(request)
        
        # If handler returns a coroutine (from async methods), await it
        if inspect.iscoroutine(handler_result):
            result = await handler_result
        else:
            result = handler_result
        
        execution_time = time.time() - start
        
        # Get output - try 'output' first, then 'data'
        output_data = result.get("output") or result.get("data")
        
        # Serialize complex outputs to JSON string
        if output_data is not None and not isinstance(output_data, str):
            output_data = json.dumps(output_data, indent=2, default=str)
        
        return ToolResult(
            success=result.get("success", True),
            output=output_data,
            error=result.get("error"),
            execution_time=execution_time,
        )
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return ToolResult(
            success=False,
            output=None,
            error=str(e),
            execution_time=0.0,
        )


@router.post("/code", response_model=CodeExecutionResult)
async def execute_code(request: CodeExecutionRequest) -> CodeExecutionResult:
    """
    Execute code in a specific language.
    
    Supports: python, powershell, node, javascript, typescript, bash
    """
    logger.info(f"Executing {request.language} code")
    
    # Check permissions
    checker = _get_permission_checker()
    allowed, reason = checker.check_language_permission(
        language=request.language,
        workspace_context=request.workspace_context,
    )
    
    if not allowed:
        return CodeExecutionResult(
            success=False,
            stdout=None,
            stderr=f"Permission denied: {reason}",
            exit_code=-1,
            execution_time=0.0,
        )
    
    # Execute
    handler = _get_polyglot_handler()
    result = await handler.execute(
        language=request.language,
        code=request.code,
        timeout=request.timeout,
        workspace_context=request.workspace_context,
    )
    
    return CodeExecutionResult(**result)


@router.post("/shell", response_model=ShellCommandResult)
async def execute_shell(request: ShellCommandRequest) -> ShellCommandResult:
    """
    Execute a shell command.
    
    Commands are tokenized (not run through shell interpreter)
    to prevent injection attacks.
    """
    logger.info(f"Executing shell command: {request.command[:50]}...")
    
    # Check permissions
    checker = _get_permission_checker()
    allowed, reason = checker.check_shell_permission(
        command=request.command,
        workspace_context=request.workspace_context,
    )
    
    if not allowed:
        return ShellCommandResult(
            success=False,
            output=None,
            error=f"Permission denied: {reason}",
            exit_code=-1,
        )
    
    # Execute
    handler = _get_shell_handler()
    result = await handler.execute(
        command=request.command,
        timeout=request.timeout,
        workspace_context=request.workspace_context,
    )
    
    return ShellCommandResult(**result)


@router.post("/file", response_model=FileOperationResult)
async def file_operation(request: FileOperationRequest) -> FileOperationResult:
    """
    Perform file operations (read, write, list, delete).
    """
    logger.info(f"File operation: {request.operation} on {request.path}")
    
    handler = _get_file_handler()
    
    if request.operation == "read":
        result = await handler.read(
            path=request.path,
            workspace_context=request.workspace_context,
        )
    elif request.operation == "write":
        result = await handler.write(
            path=request.path,
            content=request.content or "",
            workspace_context=request.workspace_context,
        )
    elif request.operation == "list":
        result = await handler.list_dir(
            path=request.path,
            workspace_context=request.workspace_context,
        )
    elif request.operation == "delete":
        result = await handler.delete(
            path=request.path,
            workspace_context=request.workspace_context,
        )
    else:
        result = {"success": False, "error": f"Unknown operation: {request.operation}"}
    
    return FileOperationResult(**result)
