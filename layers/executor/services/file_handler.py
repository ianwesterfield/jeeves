"""
File Handler - File Operations with Path Validation

Provides file read/write/list/delete operations with sandbox enforcement.
All paths are validated against the workspace root.
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pathspec

from schemas.models import WorkspaceContext


logger = logging.getLogger("executor.file")


# ============================================================================
# Pretty Listing Utilities
# ============================================================================

@dataclass(frozen=True)
class _Entry:
    name: str
    is_dir: bool
    size: Optional[int]
    mtime: Optional[datetime]


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    f = float(n)
    i = 0
    while f >= 1024.0 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    if i == 0:
        return f"{int(f)} {units[i]}"
    if f >= 10:
        return f"{f:.1f} {units[i]}"
    return f"{f:.2f} {units[i]}"


def _truncate_preserve_ext(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    p = Path(s)
    stem = p.stem
    suf = p.suffix
    if not suf or len(suf) >= max_len - 1:
        return s[: max_len - 1] + "…"
    keep = max_len - 1 - len(suf)
    if keep <= 1:
        return "…" + suf[-(max_len - 1) :]
    return stem[:keep] + "…" + suf


def _safe_stat(path: Path) -> Tuple[Optional[int], Optional[datetime]]:
    try:
        st = path.stat()
    except OSError:
        return None, None
    size = None
    if path.is_file():
        try:
            size = int(st.st_size)
        except Exception:
            size = None
    mtime = None
    try:
        mtime = datetime.fromtimestamp(st.st_mtime)
    except Exception:
        mtime = None
    return size, mtime


def pretty_ls(
    base: Path | str,
    *,
    pattern: str = "*",
    recursive: bool = False,
    show_hidden: bool = False,
    limit: int = 500,
    name_width: int = 60,
    human: bool = True,
    ignore_spec: Optional[pathspec.PathSpec] = None,
) -> str:
    """
    Returns a pretty monospace directory listing as a string.

    Notes (behavior):
    - Directories first, then files; alphabetical (case-insensitive).
    - Columns included only if available.
    - Truncates long names with "…" while preserving extension.
    - If > limit entries, prints first `limit` plus a summary line.
    """
    base_path = Path(base).expanduser().resolve()
    it: Iterable[Path]
    if recursive:
        it = base_path.rglob(pattern)
    else:
        it = base_path.glob(pattern)

    entries: list[_Entry] = []
    for p in it:
        try:
            # Get relative path for filtering
            try:
                rel_path = str(p.relative_to(base_path))
            except ValueError:
                rel_path = p.name
            
            # Skip hidden files unless requested
            if not show_hidden and p.name.startswith("."):
                continue
            
            # Always skip .git
            if rel_path == ".git" or rel_path.startswith(".git/") or rel_path.startswith(".git\\"):
                continue
            
            is_dir = p.is_dir()
            
            # Check gitignore - for directories, also check with trailing slash
            if ignore_spec:
                if ignore_spec.match_file(rel_path):
                    continue
                if is_dir and ignore_spec.match_file(rel_path + "/"):
                    continue
        except OSError:
            continue

        size, mtime = _safe_stat(p)
        
        # Use relative path for display in recursive mode
        display_name = rel_path if recursive else p.name
        
        entries.append(
            _Entry(
                name=display_name,
                is_dir=is_dir,
                size=size,
                mtime=mtime,
            )
        )

    entries.sort(key=lambda e: (not e.is_dir, e.name.casefold()))

    shown = entries[: max(0, limit)]
    remaining = max(0, len(entries) - len(shown))

    have_size = any(e.size is not None for e in shown)
    have_mtime = any(e.mtime is not None for e in shown)

    def fmt_size(sz: Optional[int]) -> str:
        if sz is None:
            return ""
        if human:
            return _human_bytes(sz)
        return str(sz)

    def fmt_mtime(ts: Optional[datetime]) -> str:
        if ts is None:
            return ""
        return ts.isoformat(sep=" ", timespec="seconds")

    size_vals = [fmt_size(e.size) for e in shown] if have_size else []
    mtime_vals = [fmt_mtime(e.mtime) for e in shown] if have_mtime else []

    size_w = max([len("SIZE"), *(len(v) for v in size_vals)]) if have_size else 0
    mtime_w = max([len("MODIFIED"), *(len(v) for v in mtime_vals)]) if have_mtime else 0

    name_w = max(name_width, len("NAME"))
    type_w = len("TYPE")

    header_cols = ["NAME".ljust(name_w), "TYPE".ljust(type_w)]
    if have_size:
        header_cols.append("SIZE".rjust(size_w))
    if have_mtime:
        header_cols.append("MODIFIED".ljust(mtime_w))

    lines: list[str] = []
    lines.append(f"PATH: {base_path}")
    lines.append(f"TOTAL: {len(entries)} items ({len([e for e in entries if e.is_dir])} dirs, {len([e for e in entries if not e.is_dir])} files)")
    lines.append("")
    lines.append("  ".join(header_cols))
    lines.append("-" * len(lines[-1]))

    for idx, e in enumerate(shown):
        nm = _truncate_preserve_ext(e.name, name_w).ljust(name_w)
        ty = ("dir" if e.is_dir else "file").ljust(type_w)
        row = [nm, ty]
        if have_size:
            row.append(fmt_size(e.size).rjust(size_w))
        if have_mtime:
            row.append(fmt_mtime(e.mtime).ljust(mtime_w))
        lines.append("  ".join(row))

    if remaining:
        lines.append(f"\n... +{remaining} more items")

    return "\n".join(lines)


# ============================================================================
# Gitignore Support  
# ============================================================================

def _load_gitignore_spec(root_path: Path) -> Optional[pathspec.PathSpec]:
    """
    Load .gitignore from root path and return a PathSpec matcher.
    
    Returns None if no .gitignore found.
    """
    gitignore_path = root_path / ".gitignore"
    if not gitignore_path.exists():
        return None
    
    try:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            patterns = f.read().splitlines()
        
        # Always add .git to ignored patterns
        patterns.append(".git")
        patterns.append(".git/**")
        
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    except Exception as e:
        logger.warning(f"Failed to parse .gitignore: {e}")
        return None


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
    
    async def replace_in_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Surgical replace: find old_text and replace with new_text.
        
        Args:
            path: File path
            old_text: Text to find (exact match)
            new_text: Replacement text
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data (replacement count), error
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
            # Read current content
            with open(validated_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Count occurrences
            count = content.count(old_text)
            if count == 0:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Text not found in file. Searched for: {old_text[:100]}...",
                }
            
            # Replace
            new_content = content.replace(old_text, new_text)
            
            # Write back
            with open(validated_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return {
                "success": True,
                "data": f"Replaced {count} occurrence(s) in {path}",
                "error": None,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "data": None,
                "error": f"File not found: {path}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def insert_in_file(
        self,
        path: str,
        position: str,
        text: str,
        anchor: Optional[str] = None,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Insert text at a position in file.
        
        Args:
            path: File path
            position: "start", "end", "before", "after"
            text: Text to insert
            anchor: For before/after, the text to search for
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
            # Read current content (or empty for new file)
            if validated_path.exists():
                with open(validated_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = ""
                # Create parent dirs
                validated_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Insert based on position
            if position == "start":
                new_content = text + content
            elif position == "end":
                new_content = content + text
            elif position in ("before", "after") and anchor:
                idx = content.find(anchor)
                if idx == -1:
                    return {
                        "success": False,
                        "data": None,
                        "error": f"Anchor text not found: {anchor[:100]}...",
                    }
                if position == "before":
                    new_content = content[:idx] + text + content[idx:]
                else:  # after
                    end_idx = idx + len(anchor)
                    new_content = content[:end_idx] + text + content[end_idx:]
            else:
                return {
                    "success": False,
                    "data": None,
                    "error": f"Invalid position: {position}. Use 'start', 'end', 'before', or 'after'",
                }
            
            # Write back
            with open(validated_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return {
                "success": True,
                "data": f"Inserted text at {position} in {path}",
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
            }
    
    async def append_to_file(
        self,
        path: str,
        content: str,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Append content to end of file (convenience method).
        
        Args:
            path: File path
            content: Content to append
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, data, error
        """
        return await self.insert_in_file(
            path=path,
            position="end",
            text=content,
            workspace_context=workspace_context,
        )
    
    async def scan_workspace(
        self,
        path: str,
        pattern: str = "*",
        workspace_context: Optional[WorkspaceContext] = None,
        respect_gitignore: bool = True,
        pretty: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan workspace for files matching a pattern.
        
        Args:
            path: Starting directory
            pattern: Glob pattern
            workspace_context: Execution context
            respect_gitignore: If True, exclude files matching .gitignore patterns
            pretty: If True, return formatted table string; if False, return raw lists
            
        Returns:
            Dictionary with success, data (formatted string or file lists), error
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
            # Load .gitignore if present
            ignore_spec = None
            if respect_gitignore:
                ignore_spec = _load_gitignore_spec(validated_path)
            
            if pretty:
                # Use pretty_ls for formatted output
                formatted = pretty_ls(
                    validated_path,
                    pattern=pattern,
                    recursive=True,
                    show_hidden=False,
                    limit=500,
                    name_width=60,
                    human=True,
                    ignore_spec=ignore_spec,
                )
                return {
                    "success": True,
                    "data": formatted,
                    "error": None,
                }
            
            # Raw mode: return file/dir lists
            all_matches = validated_path.glob(f"**/{pattern}")
            
            files = []
            dirs = []
            
            for m in all_matches:
                rel_path = str(m.relative_to(validated_path))
                
                # Always skip .git directory (even if not in .gitignore)
                if rel_path == ".git" or rel_path.startswith(".git/") or rel_path.startswith(".git\\"):
                    continue
                
                is_dir = m.is_dir()
                
                # Check against .gitignore patterns (dirs need trailing slash check too)
                if ignore_spec:
                    if ignore_spec.match_file(rel_path):
                        continue
                    if is_dir and ignore_spec.match_file(rel_path + "/"):
                        continue
                
                if m.is_file():
                    files.append(rel_path)
                elif is_dir:
                    dirs.append(rel_path)
            
            result = {
                "files": files,
                "dirs": dirs,
                "truncated": False,
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
