"""
Workspace Context Manager - Directory Scoping and Validation

Manages the active workspace context for agentic operations.
Validates paths, enforces sandbox boundaries, and caches directory listings.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from schemas.models import WorkspaceContext


logger = logging.getLogger("orchestrator.workspace")

# Default workspace root (can be overridden by env)
DEFAULT_WORKSPACE_ROOT = os.getenv("WORKSPACE_ROOT", "/workspace")


class WorkspaceContextManager:
    """
    Manages workspace context for agentic operations.
    
    Handles:
      - Setting and validating workspace directories
      - Enforcing sandbox boundaries
      - Caching directory listings
      - Permission lookups
    """
    
    def __init__(self):
        self.workspace_root = DEFAULT_WORKSPACE_ROOT
        self._context_cache: dict[str, WorkspaceContext] = {}
    
    async def set_workspace(
        self,
        cwd: str,
        user_id: Optional[str] = None,
    ) -> WorkspaceContext:
        """
        Set the active workspace directory.
        
        Args:
            cwd: Directory to set as workspace
            user_id: User ID for permission lookup
            
        Returns:
            WorkspaceContext with validated settings
            
        Raises:
            ValueError: If path is invalid
            PermissionError: If path is outside sandbox
        """
        # Normalize and validate path
        cwd_path = Path(cwd).resolve()
        root_path = Path(self.workspace_root).resolve()
        
        # Check sandbox boundary
        try:
            cwd_path.relative_to(root_path)
        except ValueError:
            # Path is outside workspace root
            raise PermissionError(
                f"Path {cwd} is outside workspace root {self.workspace_root}"
            )
        
        # Verify directory exists
        if not cwd_path.exists():
            raise ValueError(f"Directory does not exist: {cwd}")
        
        if not cwd_path.is_dir():
            raise ValueError(f"Path is not a directory: {cwd}")
        
        # Get directory listing
        available_paths = await self._list_directory(cwd_path)
        
        # Look up user permissions (TODO: integrate with permission service)
        permissions = await self._get_user_permissions(user_id)
        
        # Build context
        context = WorkspaceContext(
            cwd=str(cwd_path),
            workspace_root=str(root_path),
            available_paths=available_paths,
            parallel_enabled=permissions.get("parallel_enabled", False),
            max_parallel_tasks=permissions.get("max_parallel_tasks", 4),
            allowed_languages=permissions.get("allowed_languages", ["python", "powershell", "node"]),
            allow_code_execution=permissions.get("allow_code_execution", False),
            allow_file_write=permissions.get("allow_file_write", False),
            allow_shell_commands=permissions.get("allow_shell_commands", False),
            max_execution_time=permissions.get("max_execution_time", 30),
        )
        
        # Cache context
        cache_key = f"{user_id or 'anonymous'}:{cwd}"
        self._context_cache[cache_key] = context
        
        return context
    
    async def _list_directory(
        self,
        path: Path,
        max_depth: int = 2,
        max_files: int = 100,
    ) -> List[str]:
        """
        List directory contents up to a certain depth.
        
        Args:
            path: Directory to list
            max_depth: Maximum recursion depth
            max_files: Maximum files to return
            
        Returns:
            List of relative file paths
        """
        files = []
        
        def _walk(current: Path, depth: int):
            if depth > max_depth or len(files) >= max_files:
                return
            
            try:
                for item in current.iterdir():
                    if len(files) >= max_files:
                        break
                    
                    # Skip hidden files and common ignore patterns
                    if item.name.startswith(".") or item.name in {
                        "node_modules", "__pycache__", ".git", "venv", ".venv"
                    }:
                        continue
                    
                    rel_path = str(item.relative_to(path))
                    
                    if item.is_dir():
                        files.append(rel_path + "/")
                        _walk(item, depth + 1)
                    else:
                        files.append(rel_path)
                        
            except PermissionError:
                logger.warning(f"Permission denied: {current}")
        
        _walk(path, 0)
        return sorted(files)
    
    async def _get_user_permissions(
        self,
        user_id: Optional[str],
    ) -> dict:
        """
        Look up user permissions.
        
        TODO: Integrate with actual permission/admin service.
        For now, returns conservative defaults.
        
        Args:
            user_id: User to look up
            
        Returns:
            Permission dictionary
        """
        # Default permissions (conservative)
        default_perms = {
            "parallel_enabled": False,
            "max_parallel_tasks": 4,
            "allowed_languages": ["python", "powershell", "node"],
            "allow_code_execution": False,
            "allow_file_write": False,
            "allow_shell_commands": False,
            "max_execution_time": 30,
        }
        
        # TODO: Look up actual user permissions
        # For now, allow slightly more for authenticated users
        if user_id:
            default_perms["allow_code_execution"] = True
            default_perms["parallel_enabled"] = True
        
        return default_perms
    
    def validate_path(
        self,
        path: str,
        context: WorkspaceContext,
    ) -> bool:
        """
        Validate that a path is within the workspace.
        
        Args:
            path: Path to validate
            context: Current workspace context
            
        Returns:
            True if path is valid
        """
        try:
            resolved = Path(path).resolve()
            root = Path(context.workspace_root).resolve()
            resolved.relative_to(root)
            return True
        except ValueError:
            return False
