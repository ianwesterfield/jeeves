"""
Permission Checker - Security Validation for Executor Operations

Validates that operations are allowed based on workspace context permissions.
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from schemas.models import WorkspaceContext


logger = logging.getLogger("executor.permissions")


class PermissionChecker:
    """
    Permission validation for executor operations.
    
    Checks tool, language, and command permissions against workspace context.
    """
    
    # Tools that require specific permissions
    WRITE_TOOLS = {"write_file", "delete_file", "replace_in_file", "insert_in_file", "append_to_file", "execute_code", "execute_shell"}
    CODE_TOOLS = {"execute_code"}
    SHELL_TOOLS = {"execute_shell"}
    FILE_WRITE_TOOLS = {"write_file", "delete_file", "replace_in_file", "insert_in_file", "append_to_file"}
    
    def check_tool_permission(
        self,
        tool: str,
        params: Dict[str, Any],
        workspace_context: Optional[WorkspaceContext],
    ) -> Tuple[bool, str]:
        """
        Check if a tool invocation is allowed.
        
        Args:
            tool: Tool name
            params: Tool parameters
            workspace_context: Execution context
            
        Returns:
            Tuple of (allowed, reason)
        """
        if workspace_context is None:
            # No context = allow read-only operations only
            if tool in self.WRITE_TOOLS:
                return False, "No workspace context provided"
            return True, ""
        
        # Check code execution permission
        if tool in self.CODE_TOOLS:
            if not workspace_context.allow_code_execution:
                return False, "Code execution not allowed"
        
        # Check shell permission
        if tool in self.SHELL_TOOLS:
            if not workspace_context.allow_shell_commands:
                return False, "Shell commands not allowed"
        
        # Check write permission
        if tool in self.FILE_WRITE_TOOLS:
            if not workspace_context.allow_file_write:
                return False, "File write not allowed"
        
        return True, ""
    
    def check_language_permission(
        self,
        language: str,
        workspace_context: Optional[WorkspaceContext],
    ) -> Tuple[bool, str]:
        """
        Check if a language is allowed.
        
        Args:
            language: Programming language
            workspace_context: Execution context
            
        Returns:
            Tuple of (allowed, reason)
        """
        if workspace_context is None:
            return False, "No workspace context provided"
        
        if not workspace_context.allow_code_execution:
            return False, "Code execution not allowed"
        
        # Normalize language
        lang = language.lower()
        allowed = [l.lower() for l in workspace_context.allowed_languages]
        
        # Check aliases
        aliases = {
            "py": "python",
            "pwsh": "powershell",
            "nodejs": "node",
            "js": "node",
            "javascript": "node",
            "ts": "node",
            "typescript": "node",
            "sh": "bash",
        }
        
        normalized = aliases.get(lang, lang)
        
        if normalized in allowed or lang in allowed:
            return True, ""
        
        return False, f"Language '{language}' not in allowed list: {workspace_context.allowed_languages}"
    
    def check_shell_permission(
        self,
        command: str,
        workspace_context: Optional[WorkspaceContext],
    ) -> Tuple[bool, str]:
        """
        Check if a shell command is allowed.
        
        Args:
            command: Shell command
            workspace_context: Execution context
            
        Returns:
            Tuple of (allowed, reason)
        """
        if workspace_context is None:
            return False, "No workspace context provided"
        
        if not workspace_context.allow_shell_commands:
            return False, "Shell commands not allowed"
        
        # Additional command validation could go here
        # (e.g., checking for dangerous patterns)
        
        return True, ""
