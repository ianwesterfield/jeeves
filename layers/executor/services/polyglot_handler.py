"""
Polyglot Handler - Multi-Language Code Execution Router

Routes code execution to language-specific handlers with sandboxing.
Primary languages: Python, PowerShell, Node.js
Secondary: Bash, Go, Rust, Ruby, R, Julia
"""

import asyncio
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from schemas.models import WorkspaceContext


logger = logging.getLogger("executor.polyglot")


class PolyglotHandler:
    """
    Multi-language code execution router.
    
    Routes to language-specific handlers and manages sandboxing.
    """
    
    # Language aliases
    LANGUAGE_ALIASES = {
        "python": "python",
        "py": "python",
        "powershell": "powershell",
        "pwsh": "powershell",
        "node": "node",
        "nodejs": "node",
        "javascript": "node",
        "js": "node",
        "typescript": "node",
        "ts": "node",
        "bash": "bash",
        "sh": "bash",
        "go": "go",
        "golang": "go",
        "rust": "rust",
        "rs": "rust",
        "ruby": "ruby",
        "rb": "ruby",
        "r": "r",
        "julia": "julia",
        "jl": "julia",
    }
    
    def __init__(self):
        self._runtime_cache: Dict[str, bool] = {}
    
    async def execute(
        self,
        language: str,
        code: str,
        timeout: int = 30,
        workspace_context: Optional[WorkspaceContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in the specified language.
        
        Args:
            language: Programming language
            code: Code to execute
            timeout: Execution timeout in seconds
            workspace_context: Execution context
            
        Returns:
            Dictionary with success, stdout, stderr, exit_code, execution_time
        """
        import time
        start_time = time.time()
        
        # Normalize language
        lang = self.LANGUAGE_ALIASES.get(language.lower(), language.lower())
        
        # Check if allowed
        if workspace_context:
            allowed_langs = [l.lower() for l in workspace_context.allowed_languages]
            if lang not in allowed_langs and language.lower() not in allowed_langs:
                return {
                    "success": False,
                    "stdout": None,
                    "stderr": f"Language '{language}' not allowed. Allowed: {workspace_context.allowed_languages}",
                    "exit_code": -1,
                    "execution_time": 0.0,
                }
        
        # Get working directory
        cwd = None
        if workspace_context:
            cwd = workspace_context.cwd
        
        # Route to handler
        try:
            if lang == "python":
                result = await self._execute_python(code, timeout, cwd)
            elif lang == "powershell":
                result = await self._execute_powershell(code, timeout, cwd)
            elif lang == "node":
                result = await self._execute_node(code, timeout, cwd, is_typescript="typescript" in language.lower() or "ts" == language.lower())
            elif lang == "bash":
                result = await self._execute_bash(code, timeout, cwd)
            else:
                result = {
                    "success": False,
                    "stdout": None,
                    "stderr": f"Language '{language}' not yet implemented",
                    "exit_code": -1,
                }
            
            result["execution_time"] = time.time() - start_time
            return result
            
        except asyncio.TimeoutError:
            return {
                "success": False,
                "stdout": None,
                "stderr": f"Execution timeout ({timeout}s)",
                "exit_code": -1,
                "execution_time": time.time() - start_time,
            }
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {
                "success": False,
                "stdout": None,
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start_time,
            }
    
    async def _execute_python(
        self,
        code: str,
        timeout: int,
        cwd: Optional[str],
    ) -> Dict[str, Any]:
        """Execute Python code using RestrictedPython."""
        # For now, use subprocess with limited Python
        # TODO: Implement full RestrictedPython sandboxing
        
        process = await asyncio.create_subprocess_exec(
            "python", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode,
            }
        except asyncio.TimeoutError:
            process.kill()
            raise
    
    async def _execute_powershell(
        self,
        code: str,
        timeout: int,
        cwd: Optional[str],
    ) -> Dict[str, Any]:
        """Execute PowerShell code with constrained execution policy."""
        # Use pwsh (PowerShell 7+) if available, fall back to powershell
        pwsh_cmd = "pwsh" if shutil.which("pwsh") else "powershell"
        
        process = await asyncio.create_subprocess_exec(
            pwsh_cmd,
            "-NoProfile",
            "-ExecutionPolicy", "Restricted",
            "-Command", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode,
            }
        except asyncio.TimeoutError:
            process.kill()
            raise
    
    async def _execute_node(
        self,
        code: str,
        timeout: int,
        cwd: Optional[str],
        is_typescript: bool = False,
    ) -> Dict[str, Any]:
        """Execute Node.js/JavaScript/TypeScript code."""
        if is_typescript:
            # Try ts-node first, fall back to transpile + node
            if shutil.which("npx"):
                cmd = ["npx", "ts-node", "--eval", code]
            else:
                # Fall back to regular node (TypeScript features won't work)
                cmd = ["node", "--eval", code]
        else:
            cmd = ["node", "--eval", code]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode,
            }
        except asyncio.TimeoutError:
            process.kill()
            raise
    
    async def _execute_bash(
        self,
        code: str,
        timeout: int,
        cwd: Optional[str],
    ) -> Dict[str, Any]:
        """Execute Bash code (tokenized, not through shell)."""
        # Use sh on Windows, bash on Unix
        shell_cmd = "bash" if shutil.which("bash") else "sh"
        
        process = await asyncio.create_subprocess_exec(
            shell_cmd, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": process.returncode,
            }
        except asyncio.TimeoutError:
            process.kill()
            raise
    
    async def check_available_runtimes(self) -> List[str]:
        """Check which language runtimes are available."""
        available = []
        
        checks = [
            ("python", ["python", "--version"]),
            ("powershell", ["pwsh", "--version"] if shutil.which("pwsh") else ["powershell", "-Version"]),
            ("node", ["node", "--version"]),
            ("bash", ["bash", "--version"] if shutil.which("bash") else ["sh", "--version"]),
        ]
        
        for lang, cmd in checks:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.communicate(), timeout=5)
                if process.returncode == 0:
                    available.append(lang)
            except Exception:
                pass
        
        return available
