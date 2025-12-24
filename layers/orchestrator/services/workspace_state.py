"""
Workspace State Manager - External State Tracking

Maintains ground-truth state from actual tool outputs rather than
asking the LLM to track state (which causes drift/hallucination).

The orchestrator updates this after each tool execution, then
injects it as context into the LLM prompt.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import re
import logging

logger = logging.getLogger("orchestrator.workspace_state")


@dataclass
class CompletedStep:
    """Record of a completed step."""
    step_id: str
    tool: str
    params: Dict[str, Any]
    output_summary: str  # Brief summary, not full output
    success: bool
    timestamp: Optional[str] = None


@dataclass
class WorkspaceState:
    """
    External state maintained by orchestrator, NOT the LLM.
    
    This is ground-truth state built from actual tool outputs.
    Injected into LLM context so it doesn't need to track state.
    """
    # Files discovered via scan_workspace
    scanned_paths: Set[str] = field(default_factory=set)
    files: List[str] = field(default_factory=list)
    dirs: List[str] = field(default_factory=list)
    
    # Files already edited (to avoid re-editing)
    edited_files: Set[str] = field(default_factory=set)
    
    # Files already read (to avoid re-reading)
    read_files: Set[str] = field(default_factory=set)
    
    # Completed steps (compact log)
    completed_steps: List[CompletedStep] = field(default_factory=list)
    
    # User info extracted from memory (name, preferences, etc.)
    user_info: Dict[str, str] = field(default_factory=dict)
    
    def update_from_step(self, tool: str, params: Dict[str, Any], output: str, success: bool) -> None:
        """
        Update state based on a completed step.
        
        Called by orchestrator after each tool execution.
        """
        step_id = f"S{len(self.completed_steps) + 1:03d}"
        
        # Build compact summary based on tool type
        if tool == "scan_workspace":
            scan_path = params.get("path", ".")
            self.scanned_paths.add(scan_path)
            self._parse_scan_output(output, scan_path)
            summary = f"scan({scan_path}): {len(self.files)} files, {len(self.dirs)} dirs"
            
        elif tool == "read_file":
            path = params.get("path", "")
            self.read_files.add(path)
            char_count = len(output) if output else 0
            summary = f"read({path}): {char_count:,} chars"
            
        elif tool in ("write_file", "replace_in_file", "insert_in_file", "append_to_file"):
            path = params.get("path", "")
            # Only mark as edited if successful - failed edits should be retried
            if success:
                self.edited_files.add(path)
            summary = f"{tool}({path}): {'OK' if success else 'FAILED'}"
            
        elif tool == "execute_shell":
            cmd = params.get("command", "")[:40]
            summary = f"shell({cmd}): {'OK' if success else 'FAILED'}"
        
        elif tool == "none":
            # Idempotent skip - change already present
            reason = params.get("reason", "already present")
            path = params.get("path", "")
            summary = f"skipped({path}): {reason}" if path else f"skipped: {reason}"
            
        else:
            summary = f"{tool}: {'OK' if success else 'FAILED'}"
        
        self.completed_steps.append(CompletedStep(
            step_id=step_id,
            tool=tool,
            params={k: v for k, v in params.items() if k not in ("content", "text", "new_text")},
            output_summary=summary,
            success=success,
            timestamp=datetime.utcnow().isoformat() + "Z",
        ))
        
        logger.debug(f"State updated: {summary}")
    
    def _parse_scan_output(self, output: str, base_path: str) -> None:
        """
        Parse scan_workspace output to extract file/dir lists.
        
        Expected format (from executor):
        PATH: .
        NAME          TYPE   SIZE      MODIFIED
        ─────────────────────────────────────────
        .github/      dir    -         ...
        README.md     file   1.2 KB    ...
        """
        if not output:
            return
        
        for line in output.split("\n"):
            line = line.strip()
            
            # Skip headers and separators
            if not line or line.startswith("PATH:") or line.startswith("TOTAL:"):
                continue
            if line.startswith("-") or line.startswith("─") or line.startswith("NAME"):
                continue
            if line.startswith("..."):
                continue
            
            # Parse: NAME  TYPE  SIZE  MODIFIED
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                entry_type = parts[1] if len(parts) > 1 else ""
                
                # Build full path
                if base_path and base_path != ".":
                    full_path = f"{base_path.rstrip('/')}/{name.rstrip('/')}"
                else:
                    full_path = name.rstrip("/")
                
                if entry_type == "dir" or name.endswith("/"):
                    if full_path not in self.dirs:
                        self.dirs.append(full_path)
                elif entry_type == "file":
                    if full_path not in self.files:
                        self.files.append(full_path)
    
    def get_editable_files(self) -> List[str]:
        """Get files that can be edited (not binary, not already edited)."""
        editable_extensions = {'.md', '.py', '.js', '.ts', '.yaml', '.yml', '.json', '.txt', '.sh', '.ps1', '.mmd', '.html', '.css', '.env', '.toml', '.ini', '.cfg'}
        binary_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bin', '.exe', '.dll', '.pyc', '.pth', '.safetensors', '.ico', '.woff', '.woff2', '.ttf', '.eot'}
        
        editable = []
        for f in self.files:
            # Skip already edited
            if f in self.edited_files:
                continue
            
            # Skip binary
            if any(f.endswith(ext) for ext in binary_extensions):
                continue
            
            # Only include known editable extensions
            if any(f.endswith(ext) for ext in editable_extensions):
                editable.append(f)
        
        return sorted(editable)
    
    def get_unread_files(self) -> List[str]:
        """Get files that haven't been read yet."""
        return [f for f in self.files if f not in self.read_files]
    
    def format_for_prompt(self) -> str:
        """
        Format state as context for LLM prompt injection.
        
        This replaces asking the LLM to maintain state.
        """
        lines = ["=== WORKSPACE STATE (do not repeat, use this info) ===\n"]
        
        # Completed steps
        if self.completed_steps:
            lines.append("Completed steps:")
            for step in self.completed_steps[-15:]:  # Last 15 steps
                status = "✓" if step.success else "✗"
                lines.append(f"  {status} {step.output_summary}")
            if len(self.completed_steps) > 15:
                lines.append(f"  ... ({len(self.completed_steps) - 15} earlier steps)")
            lines.append("")
        
        # Workspace index
        if self.scanned_paths:
            lines.append(f"Scanned paths: {', '.join(sorted(self.scanned_paths))}")
            lines.append(f"Total files: {len(self.files)}")
            lines.append(f"Total dirs: {len(self.dirs)}")
            
            # Show editable files remaining
            editable = self.get_editable_files()
            if editable:
                lines.append(f"Editable files remaining: {len(editable)}")
                # Show first few
                preview = editable[:5]
                lines.append(f"  Next: {', '.join(preview)}")
                if len(editable) > 5:
                    lines.append(f"  ... and {len(editable) - 5} more")
            lines.append("")
        
        # Read files - critical to prevent re-reading
        if self.read_files:
            lines.append(f"Already read ({len(self.read_files)}) - DO NOT read again:")
            for f in sorted(self.read_files)[:10]:
                lines.append(f"  📖 {f}")
            if len(self.read_files) > 10:
                lines.append(f"  ... and {len(self.read_files) - 10} more")
            lines.append("")
        
        # Edited files
        if self.edited_files:
            lines.append(f"Already edited ({len(self.edited_files)}):")
            for f in sorted(self.edited_files)[:10]:
                lines.append(f"  ✓ {f}")
            if len(self.edited_files) > 10:
                lines.append(f"  ... and {len(self.edited_files) - 10} more")
            lines.append("")
        
        # User info
        if self.user_info:
            lines.append("User info:")
            for key, value in self.user_info.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append("=== END STATE ===")
        return "\n".join(lines)
    
    def has_scanned(self, path: str = ".") -> bool:
        """Check if a path has already been scanned."""
        return path in self.scanned_paths
    
    def has_edited(self, path: str) -> bool:
        """Check if a file has already been edited."""
        return path in self.edited_files
    
    def has_read(self, path: str) -> bool:
        """Check if a file has already been read."""
        return path in self.read_files
    
    def reset(self) -> None:
        """Reset state for a new task."""
        self.scanned_paths.clear()
        self.files.clear()
        self.dirs.clear()
        self.edited_files.clear()
        self.read_files.clear()
        self.completed_steps.clear()
        # Keep user_info across tasks


# Singleton for session state
_current_state: Optional[WorkspaceState] = None


def get_workspace_state() -> WorkspaceState:
    """Get or create the current workspace state."""
    global _current_state
    if _current_state is None:
        _current_state = WorkspaceState()
    return _current_state


def reset_workspace_state() -> WorkspaceState:
    """Reset and return fresh workspace state."""
    global _current_state
    _current_state = WorkspaceState()
    return _current_state
