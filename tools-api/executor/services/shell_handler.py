"""
Shell Handler - Command Execution with Tokenization

Executes shell commands safely using subprocess with shell=False.
Commands are tokenized to prevent injection attacks.
"""

import asyncio
import logging
import shlex
import shutil
from typing import Any, Dict, Optional, Set

from schemas.models import WorkspaceContext


logger = logging.getLogger("executor.shell")


# Commands that are blocked by default
BLOCKED_COMMANDS: Set[str] = {
    # Destructive
    "rm", "rmdir", "del", "erase",
    "format", "mkfs", "dd",
    # System
    "sudo", "su", "runas",
    "systemctl", "service",
    "shutdown", "reboot", "halt",
    # Network
    "nc", "netcat", "ncat",
    # Docker (prevent escape)
    "docker", "podman",
    # Package managers (could install malware)
    "pip", "npm", "gem", "cargo",  # Require explicit permission
}


class ShellHandler:
    """
    Shell command execution with safety tokenization.
    
    Commands are parsed into tokens (not passed to shell interpreter)
    to prevent command injection attacks.
    """
    
    def __init__(self):
        self.blocked_commands = BLOCKED_COMMANDS.copy()
    
    async def execute(
        self,
        command: str,
        timeout: int = 30,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute a shell command.
        
        Args:
            command: Command string to execute
            timeout: Execution timeout in seconds
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, output, error, exit_code
        """
        # Check if shell commands are allowed
        if workspace_context and not workspace_context.allow_shell_commands:
            return {
                "success": False,
                "output": None,
                "error": "Shell commands not allowed",
                "exit_code": -1,
            }
        
        # Tokenize command
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return {
                "success": False,
                "output": None,
                "error": f"Invalid command syntax: {e}",
                "exit_code": -1,
            }
        
        if not tokens:
            return {
                "success": False,
                "output": None,
                "error": "Empty command",
                "exit_code": -1,
            }
        
        # Check for blocked commands
        base_command = tokens[0].lower()
        if base_command in self.blocked_commands:
            return {
                "success": False,
                "output": None,
                "error": f"Command '{base_command}' is blocked for security reasons",
                "exit_code": -1,
            }
        
        # Get working directory
        cwd = None
        if workspace_context:
            cwd = workspace_context.cwd
        
        # Execute (shell=False for safety)
        try:
            process = await asyncio.create_subprocess_exec(
                *tokens,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            return {
                "success": process.returncode == 0,
                "output": stdout.decode("utf-8", errors="replace"),
                "error": stderr.decode("utf-8", errors="replace") if stderr else None,
                "exit_code": process.returncode,
            }
            
        except asyncio.TimeoutError:
            process.kill()
            return {
                "success": False,
                "output": None,
                "error": f"Command timeout ({timeout}s)",
                "exit_code": -1,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": None,
                "error": f"Command not found: {tokens[0]}",
                "exit_code": -1,
            }
        except Exception as e:
            logger.error(f"Shell execution error: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "exit_code": -1,
            }
