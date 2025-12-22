"""
File Handler - File Operations with Path Validation

Provides file read/write/list/delete operations with sandbox enforcement.
All paths are validated against the workspace root.
"""

import ast
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas.models import WorkspaceContext


logger = logging.getLogger("executor.file")


class FileHandler:
    """
    File operations handler with sandbox validation.
    
    All paths are validated to ensure they're within the workspace root.
    """
    
    async def read(
        self,
        path: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Read file contents.
        
        Args:
            path: File path to read
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data (content), error
        """
        # Validate path
        validated_path = self._validate_path(path, workspace_context)
        if validated_path is None:
            return {
                "success": False,
                "data": None,
                "error": "Path is outside workspace",
            }
        
        try:
            with open(validated_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {
                "success": True,
                "data": content,
                "error": None,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "data": None,
                "error": f"File not found: {path}",
            }
        except PermissionError:
            return {
                "success": False,
                "data": None,
                "error": f"Permission denied: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def write(
        self,
        path: str,
        content: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            path: File path to write
            content: Content to write
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data, error
        """
        # Check if write is allowed
        if workspace_context and not workspace_context.allow_file_write:
            return {
                "success": False,
                "data": None,
                "error": "File write not allowed",
            }
        
        # Validate path
        validated_path = self._validate_path(path, workspace_context)
        if validated_path is None:
            return {
                "success": False,
                "data": None,
                "error": "Path is outside workspace",
            }
        
        try:
            # Create parent directories if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(validated_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {
                "success": True,
                "data": f"Written {len(content)} bytes to {path}",
                "error": None,
            }
        except PermissionError:
            return {
                "success": False,
                "data": None,
                "error": f"Permission denied: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def list_dir(
        self,
        path: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        List directory contents.
        
        Args:
            path: Directory path
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data (file list), error
        """
        # Validate path
        validated_path = self._validate_path(path, workspace_context)
        if validated_path is None:
            return {
                "success": False,
                "data": None,
                "error": "Path is outside workspace",
            }
        
        try:
            if not validated_path.is_dir():
                return {
                    "success": False,
                    "data": None,
                    "error": f"Not a directory: {path}",
                }
            
            items = []
            for item in validated_path.iterdir():
                item_type = "dir" if item.is_dir() else "file"
                items.append({
                    "name": item.name,
                    "type": item_type,
                    "size": item.stat().st_size if item.is_file() else 0,
                })
            
            return {
                "success": True,
                "data": items,
                "error": None,
            }
        except PermissionError:
            return {
                "success": False,
                "data": None,
                "error": f"Permission denied: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def delete(
        self,
        path: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Delete a file.
        
        Args:
            path: File path to delete
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data, error
        """
        # Check if write is allowed (delete requires write permission)
        if workspace_context and not workspace_context.allow_file_write:
            return {
                "success": False,
                "data": None,
                "error": "File delete not allowed",
            }
        
        # Validate path
        validated_path = self._validate_path(path, workspace_context)
        if validated_path is None:
            return {
                "success": False,
                "data": None,
                "error": "Path is outside workspace",
            }
        
        try:
            if validated_path.is_file():
                validated_path.unlink()
                return {
                    "success": True,
                    "data": f"Deleted: {path}",
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Not a file: {path}",
                }
        except PermissionError:
            return {
                "success": False,
                "data": None,
                "error": f"Permission denied: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def scan_workspace(
        self,
        path: str,
        pattern: str = "*",
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Scan workspace for files matching a pattern.
        
        Args:
            path: Starting directory
            pattern: Glob pattern
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data (matching files), error
        """
        # Validate path
        validated_path = self._validate_path(path, workspace_context)
        if validated_path is None:
            return {
                "success": False,
                "data": None,
                "error": "Path is outside workspace",
            }
        
        try:
            matches = list(validated_path.glob(f"**/{pattern}"))
            
            # Limit results
            max_results = 100
            truncated = len(matches) > max_results
            matches = matches[:max_results]
            
            result = {
                "files": [str(m.relative_to(validated_path)) for m in matches if m.is_file()],
                "dirs": [str(m.relative_to(validated_path)) for m in matches if m.is_dir()],
                "truncated": truncated,
            }
            
            return {
                "success": True,
                "data": result,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def analyze(
        self,
        path: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Analyze code file (Python only for now).
        
        Args:
            path: File path to analyze
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data (analysis result), error
        """
        # Read file first
        read_result = await self.read(path, workspace_context)
        if not read_result["success"]:
            return read_result
        
        content = read_result["data"]
        
        # Only analyze Python files for now
        if not path.endswith(".py"):
            return {
                "success": True,
                "data": {
                    "type": "text",
                    "lines": len(content.splitlines()),
                    "size": len(content),
                },
                "error": None,
            }
        
        try:
            tree = ast.parse(content)
            
            # Extract basic info
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return {
                "success": True,
                "data": {
                    "type": "python",
                    "lines": len(content.splitlines()),
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                },
                "error": None,
            }
        except SyntaxError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Syntax error: {e}",
            }
    
    def _validate_path(
        self,
        path: str,
        workspace_context: Optional[WorkspaceContext],
    ) -> Optional[Path]:
        """
        Validate and resolve a path within the workspace.
        
        Returns None if path is outside workspace.
        """
        try:
            # Resolve relative to cwd if workspace context provided
            if workspace_context:
                if not os.path.isabs(path):
                    path = os.path.join(workspace_context.cwd, path)
                
                resolved = Path(path).resolve()
                root = Path(workspace_context.workspace_root).resolve()
                
                # Check if within workspace
                try:
                    resolved.relative_to(root)
                    return resolved
                except ValueError:
                    return None
            else:
                # No context - just resolve
                return Path(path).resolve()
        except Exception:
            return None
