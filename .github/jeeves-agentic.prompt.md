# Plan: Jeeves Redesign – Multi-Faceted AI Agentic & Reasoning Engine (REFINED)

Jeeves transforms from a passive memory layer into an active reasoning and execution engine with **contextual parallelization**. Workspace-scoped, permission-gated, batch-wise parallel execution for independent tasks.

## Key Refinements from Original Plan

1. **Workspace Context**: User sets active directory via chat tools menu. All operations scoped to that context.
2. **Contextual Parallelization**: LLM detects independent tasks (e.g., "analyze all Python files"). Auto-batches them.
3. **Admin Valve**: Parallel execution disabled by default; admin enables per-user via permissions.
4. **Batch Execution**: All parallel tasks complete; failures don't cascade. Aggregated error reporting.
5. **Progress Reporting**: One message per batch (not per task). Succinct, non-spammy, streaming-friendly.

---

## Steps

### 1. Simplified Service Architecture

| Service                | Port        | Tier      | Purpose                                                                          |
| ---------------------- | ----------- | --------- | -------------------------------------------------------------------------------- |
| **Orchestrator**       | 8004        | Reasoning | Multi-turn reasoning, parallelization detection, state machine, LLM coordination |
| **Executor** (unified) | 8005        | Execution | Shell, code (RestrictedPython), file ops, APIs—single sandbox                    |
| **Code Analyzer**      | (ext. 8002) | Analysis  | Workspace scanning, code semantics, dependency mapping                           |
| **Memory**             | 8000        | Context   | Context provider + execution trace learning                                      |
| **Pragmatics**         | 8001        | Analysis  | Intent classification                                                            |
| **Extractor**          | 8002        | Analysis  | Document/image/audio extraction + code analysis                                  |

---

### 2. Orchestrator Service (Port 8004)

**File Structure:**

```
tools-api/orchestrator/
├── Dockerfile
├── requirements.txt
├── main.py
├── api/
│   └── orchestrator.py              # Routers
├── services/
│   ├── reasoning_engine.py          # LLM calls + JSON parsing
│   ├── task_planner.py              # Decomposition + parallel detection
│   ├── parallel_executor.py         # asyncio.gather + error handling
│   └── memory_connector.py          # Pattern retrieval + storage
├── schemas/
│   ├── models.py                    # Task, Step, Batch, Result
│   └── tool_manifest.json           # Tool definitions (JSON Schema)
└── utils/
    └── workspace_context.py         # Workspace scoping
```

**Workspace Context Model:**

```python
class WorkspaceContext:
    cwd: str                    # Current directory (/workspace/src)
    workspace_root: str         # Sandbox root (/workspace)
    available_paths: List[str]  # Cached directory listing
    parallel_enabled: bool      # Admin valve (default: False)
    max_parallel_tasks: int     # Max concurrent (default: 4)
```

**Endpoints:**

```python
# POST /api/orchestrate/set-workspace
# Set active directory for execution
Request: { cwd: str }
Response: WorkspaceContext { cwd, available_paths, parallel_enabled }

# POST /api/orchestrate/next-step
# Generate single next step (may detect parallelization)
Request: { task: str, memory_context: List, history: List[StepResult] }
Response: NextStep { tool, params, batch_id?: str, reasoning: str }

# POST /api/orchestrate/execute-batch
# Execute parallel batch; continue on individual failures
Request: { steps: List[Step], batch_id: str }
Response: BatchResult { successful: int, failed: int, errors: List[ErrorMetadata] }
```

**Contextual Parallelization Detection:**

```python
# Examples the reasoning engine detects:

1. "Analyze all Python files in src/"
   → LLM recognizes independent analysis tasks
   → Batch: [analyze(file1), analyze(file2), analyze(file3), ...]
   → batch_id: "analyze_py_files"

2. "List contents of config/ and data/ and docs/"
   → LLM recognizes independent directory reads
   → Batch: [list(config/), list(data/), list(docs/)]
   → batch_id: "list_directories"

3. "Format and lint all Python files"
   → Independent transformation tasks
   → Batch: [format(file1), lint(file1), format(file2), ...]
   → batch_id: "format_and_lint"

# Non-parallelizable:
"Read config, then apply settings"
→ Sequential: read_file → execute_code
```

**Multi-Turn State Machine:**

```
USER: @jeeves autonomous: analyze all Python files in src/

1. PLANNING
   - Set workspace context (cwd=/workspace/src)
   - LLM generates STEP 1: scan_workspace
   - Show to user

2. USER APPROVES
   - Execute: scan_workspace → finds 12 Python files

3. PLANNING (next step)
   - LLM detects parallelizable pattern
   - Creates batch: [analyze(file1), analyze(file2), ..., analyze(file12)]
   - batch_id: "analyze_py_files"
   - Show to user: "Ready to analyze 12 files in parallel"

4. USER APPROVES
   - EXECUTING_BATCH: asyncio.gather(12 tasks)
   - File 5 fails with SyntaxError (others continue)
   - File 12 times out (others continue)
   - Return: successful=10, failed=2

5. REPORTING_COMPLETE
   - Output: "✓ 12/12 completed: 10 OK, 2 failed"
   - Show error metadata: File5 SyntaxError, File12 timeout
   - Relinquish logic flow (message sent once per batch)

6. NEXT_STEP
   - Continue with synthesis step using 10 successful results
```

---

### 3. Parallel Executor (Integrated into Orchestrator)

**Batch Execution with Partial Failure Handling:**

```python
class ParallelExecutor:
    async def execute_batch(self, batch: TaskBatch, context: WorkspaceContext):
        """
        Execute multiple independent steps concurrently.
        Failures don't stop others.
        Return aggregated results once per batch.
        """

        # Create async task for each step
        tasks = [
            asyncio.create_task(self.execute_single_step(step, context))
            for step in batch.steps
        ]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        batch_result = BatchResult(batch_id=batch.id)

        for step, result in zip(batch.steps, results):
            if isinstance(result, Exception):
                batch_result.failed.append({
                    "step_id": step.id,
                    "error": str(result),
                    "error_type": type(result).__name__,
                    "recoverable": is_recoverable(result)
                })
            else:
                batch_result.successful.append({
                    "step_id": step.id,
                    "result": result
                })

        # Relinquish logic flow ONCE (not per task)
        return batch_result
```

**User-Friendly Progress Reporting:**

```python
def format_batch_for_chat(batch_result: BatchResult) -> str:
    success = len(batch_result.successful)
    failed = len(batch_result.failed)
    total = success + failed

    # Succinct summary
    if failed == 0:
        return f"✓ {total} completed in {batch_result.duration:.1f}s\n\nAll tasks successful."

    # Show failures (condensed)
    failures = "\n".join([
        f"  • {f['step_id']}: {f['error_type']}"
        for f in batch_result.failed[:5]  # First 5
    ])

    if len(batch_result.failed) > 5:
        failures += f"\n  ... and {len(batch_result.failed) - 5} more"

    return f"""✓ {total} completed in {batch_result.duration:.1f}s ({success} OK, {failed} failed)

**Failures:**
{failures}

Continuing with next step using successful results..."""
```

---

### 4. Unified Executor Service (Port 8005)

**File Structure:**

```
tools-api/executor/
├── Dockerfile
├── requirements.txt
├── main.py
├── api/
│   └── executor.py                 # Single polymorphic router
├── services/
│   ├── shell_handler.py            # Command execution
│   ├── code_handler.py             # RestrictedPython execution
│   ├── file_handler.py             # File operations
│   ├── workspace_handler.py        # Code analysis
│   └── permissions.py              # Validation
└── utils/
    ├── sandbox.py                  # RestrictedPython setup
    ├── injection_detection.py      # Security scanning
    └── resource_limits.py          # Timeouts + memory limits
```

**Single Polymorphic Endpoint:**

```python
# POST /api/execute/tool
Request: { tool: str, params: dict }
Response: ToolResult { success: bool, output: str, stderr: str, execution_time: float }

@router.post("/api/execute/tool")
async def execute_tool(req: ToolCallRequest) -> ToolResult:
    handler = TOOL_HANDLERS[req.tool]
    return await handler.execute(req.params)
```

**Security: 4-Layer Defense**

```
1. Input Validation
   ✓ Tool in manifest
   ✓ Params match JSON schema
   ✓ Path in allowed roots

2. Injection Prevention
   ✓ Command tokenization (not shell=True)
   ✓ AST scan for forbidden imports
   ✓ Regex scan for dangerous patterns

3. Sandboxing
   ✓ RestrictedPython for code
   ✓ Whitelisted builtins only
   ✓ subprocess with shell=False
   ✓ Resource limits (time, memory)

4. Auditing
   ✓ All tool calls logged
   ✓ Execution outcome stored
   ✓ Results in memory for learning
```

---

### 5. Polyglot Code Execution (Multi-Language Support)

**Requirement**: Execute code in multiple languages (Python, JavaScript, Go, Rust, Bash, etc.), not just Python.

**Supported Languages & Runtimes:**

| Language       | Runtime     | Handler           | Sandbox Method           | Notes                                              |
| -------------- | ----------- | ----------------- | ------------------------ | -------------------------------------------------- |
| **Python**     | 3.10+       | RestrictedPython  | AST + restricted globals | Primary; fully sandboxed                           |
| **PowerShell** | 7.0+        | subprocess        | shell=False, policy      | Primary (Windows-first); constrained execution     |
| **Node.JS**    | 18.0+       | subprocess        | shell=False              | Primary (Windows-native); Angular, TypeScript, npm |
| **JavaScript** | Node.js 18+ | subprocess        | shell=False              | Alias for Node.JS; npm available in workspace      |
| **Bash**       | /bin/bash   | subprocess        | shell=False              | Workspace-scoped commands                          |
| **Go**         | Go 1.20+    | compile + execute | Docker container         | Pre-compile for safety                             |
| **Rust**       | Cargo       | compile + execute | Docker container         | Pre-compile for safety                             |
| **Ruby**       | Ruby 3+     | subprocess        | shell=False              | If installed on system                             |
| **R**          | R 4.0+      | Rscript           | subprocess               | For data analysis workflows                        |
| **Julia**      | Julia 1.8+  | subprocess        | shell=False              | Scientific computing                               |

**Language Detection & Routing:**

```python
# In code_handler.py (renamed to polyglot_handler.py)
LANGUAGE_RUNTIMES = {
    "python": PythonHandler,          # RestrictedPython AST
    "powershell": PowerShellHandler,  # Constrained execution policy
    "pwsh": PowerShellHandler,        # Alias for PowerShell
    "node": NodeJSHandler,            # Node.JS subprocess
    "js": NodeJSHandler,              # Alias for JavaScript
    "javascript": NodeJSHandler,      # Alias for Node.JS
    "typescript": NodeJSHandler,      # Transpiled via ts-node or tsc
    "ts": NodeJSHandler,              # Alias for TypeScript
    "bash": BashHandler,              # shell=False, command tokenization
    "sh": BashHandler,                # Alias
    "go": GoHandler,                  # Docker container (compile-time safety)
    "rust": RustHandler,              # Docker container (compile-time safety)
    "ruby": RubyHandler,              # subprocess
    "r": RHandler,                    # subprocess (Rscript)
    "julia": JuliaHandler,            # subprocess
}

async def execute_code(language: str, code: str, context: WorkspaceContext) -> ExecutionResult:
    handler = LANGUAGE_RUNTIMES.get(language)
    if not handler:
        raise UnsupportedLanguageError(f"Language {language} not supported")

    # Permission check
    if not context.permissions.allow_code_execution:
        raise PermissionDenied("Code execution not allowed for this user")

    return await handler.execute(code, context)
```

**Per-Language Sandbox Strategies:**

**Python (RestrictedPython):**

```python
# Whitelist only safe builtins: print, len, range, dict, list, etc.
# Deny: open, exec, eval, __import__, subprocess
safe_globals = {
    "__builtins__": {
        "print": print,
        "len": len,
        "range": range,
        "dict": dict,
        "list": list,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "sum": sum,
        "max": max,
        "min": min,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        # ... approved functions
    },
    "__name__": "__main__",
}

# Execute compiled code in restricted context
exec(compile(code, "<string>", "exec"), safe_globals)
```

**PowerShell (Constrained Execution Policy):**

```powershell
# Execution: pwsh -NoProfile -ExecutionPolicy Restricted -Command "<code>"
# Safety: shell=False, command tokenization, timeout
# Constrained Language Mode: Deny dangerous operations
# Access to: Safe cmdlets, basic types, no file I/O by default
# Deny: file deletion, registry access, process termination (unless whitelisted)

# PowerShell 7+ Constrained Language Mode enables:
# - Safe cmdlets only (e.g., Get-Content with allowed paths, Write-Output)
# - No dynamic code generation (Invoke-Expression, ScriptBlock creation)
# - No direct .NET reflection (limited to safe APIs)
# - Path validation: CWD-scoped only

# Example: User request "run PowerShell to list files"
user_code = """
Get-ChildItem -Path . -Recurse -Filter "*.ps1" | Select-Object FullName
"""

result = subprocess.run(
    ["pwsh", "-NoProfile", "-ExecutionPolicy", "Restricted", "-Command", user_code],
    capture_output=True,
    timeout=30,
    shell=False,
    cwd=context.cwd,
    env={**os.environ, "POWERSHELL_EXECUTION_POLICY": "Restricted"}  # Enforce policy
)

# Blocked by default (require explicit permission):
# - Remove-Item, Remove-Variable
# - Set-ItemProperty (registry)
# - New-Service, Stop-Process
# - Invoke-Expression, Invoke-Command
```

**Node.JS / JavaScript / TypeScript (Primary, Windows-native):**

```bash
# Execution: node --input-type=module --eval "<code>"
# Safety: shell=False, command tokenization, timeout
# Access to: Node.js stdlib, npm modules (whitelisted)
# Deny: fs (without permission), child_process, eval()
# TypeScript support: via ts-node or tsc transpilation

# Example 1: JavaScript - parse JSON
js_code = """
const data = JSON.parse(input);
console.log(data.name);
"""

result = subprocess.run(
    ["node", "--eval", js_code],
    capture_output=True,
    timeout=30,
    shell=False,
    cwd=context.cwd
)

# Example 2: TypeScript (via ts-node for immediate execution)
ts_code = """
interface User { name: string; age: number; }
const user: User = { name: "Alice", age: 30 };
console.log(`${user.name} is ${user.age}`);
"""

# Requires ts-node in workspace or use tsc + node:
result = subprocess.run(
    ["npx", "ts-node", "--eval", ts_code],
    capture_output=True,
    timeout=30,
    shell=False,
    cwd=context.cwd
)

# Example 3: Angular CLI / Build tools (workspace-scoped)
cli_command = "ng generate component my-component"
result = subprocess.run(
    ["npx", "ng", "generate", "component", "my-component"],
    capture_output=True,
    timeout=60,
    shell=False,
    cwd=context.cwd
)

# Allowed by default:
# - JSON parsing, file reading (via fs, with permission)
# - npm/npx execution (whitelisted packages)
# - TypeScript transpilation
# - Angular CLI commands (ng)
# - Build tools (webpack, tsc, prettier, eslint)

# Blocked by default:
# - child_process (spawn, exec, fork)
# - fs (delete, write without permission)
# - eval(), Function() constructor
# - require() of blacklisted modules
```

**Bash (subprocess tokenization):**

```python
# Tokenize command to prevent injection
# e.g., "ls -la /workspace/src" → ["ls", "-la", "/workspace/src"]
# Sandbox: CWD = workspace root, PATH limited, dangerous commands blocked

blocked_commands = {"rm", "dd", "mkfs", "sudo", "systemctl", "docker"}

tokens = shlex.split(command)
if tokens[0] in blocked_commands:
    raise BlockedCommand(f"{tokens[0]} not allowed")

result = subprocess.run(
    tokens,
    capture_output=True,
    timeout=30,
    shell=False,
    cwd=context.cwd
)
```

**Go/Rust (Docker containers):**

```dockerfile
# Dockerfile for Go/Rust compilation
# Compile code in isolation, execute binary in limited container
# Safety: No direct shell access, resource limits (CPU, RAM), timeout

FROM golang:1.20-alpine AS compiler
COPY code.go /build/code.go
RUN go build -o /build/app /build/code.go

FROM alpine:latest
COPY --from=compiler /build/app /app
RUN chmod 555 /app  # Read-only executable
CMD ["/app"]
```

**Comparison: Jupyter-style Polyglot Execution**

| Approach         | Pros                                                    | Cons                                                | Use Case                             |
| ---------------- | ------------------------------------------------------- | --------------------------------------------------- | ------------------------------------ |
| **Jupyter**      | Interactive notebooks, rich outputs, multi-kernel       | UI-bound, stateful, harder to embed in microservice | Exploratory analysis                 |
| **Quarto**       | Markdown + code polyglot, render to HTML/PDF            | Batch-oriented, not interactive                     | Documentation + reproducible reports |
| **Our approach** | Ephemeral, stateless, RESTful, isolated, parallelizable | No interactive REPL, no rich outputs (yet)          | Agentic automation                   |

We adopt an **ephemeral execution model**: each code call is stateless (no kernel persistence), fully sandboxed, and parallelizable. This aligns with agentic reasoning (single-shot tool calls) rather than interactive exploration.

**File Structure Update:**

```
tools-api/executor/
├── Dockerfile
├── requirements.txt
├── main.py
├── api/
│   └── executor.py                     # Single polymorphic router
├── services/
│   ├── polyglot_handler.py             # Language router (Python, JS, Bash, Go, Rust, etc.)
│   ├── handlers/
│   │   ├── python_handler.py           # RestrictedPython
│   │   ├── javascript_handler.py       # Node.js + subprocess
│   │   ├── bash_handler.py             # Bash tokenization
│   │   ├── compiled_handler.py         # Go/Rust (Docker)
│   │   └── scripting_handler.py        # Ruby, R, Julia
│   ├── shell_handler.py                # Command execution (old sh_handler compatibility)
│   ├── file_handler.py                 # File operations
│   ├── workspace_handler.py            # Code analysis
│   └── permissions.py                  # Validation
└── utils/
    ├── sandbox.py                      # RestrictedPython + per-language setup
    ├── injection_detection.py          # Security scanning (all languages)
    ├── resource_limits.py              # Timeouts + memory (all languages)
    └── docker_compiler.py              # Go/Rust container compilation
```

**Tool Manifest Extension:**

```json
{
  "tools": [
    {
      "name": "execute_code",
      "description": "Execute code in various languages",
      "parameters": {
        "type": "object",
        "properties": {
          "language": {
            "type": "string",
            "enum": [
              "python",
              "powershell",
              "pwsh",
              "node",
              "javascript",
              "js",
              "typescript",
              "ts",
              "bash",
              "go",
              "rust",
              "ruby",
              "r",
              "julia"
            ],
            "description": "Programming language"
          },
          "code": {
            "type": "string",
            "description": "Code to execute"
          },
          "timeout": {
            "type": "integer",
            "default": 30,
            "description": "Execution timeout in seconds"
          },
          "environment": {
            "type": "object",
            "description": "Environment variables (whitelist-gated)"
          }
        },
        "required": ["language", "code"]
      }
    }
  ]
}
```

**Permission Model Extension:**

```python
class UserPermissions:
    allowed_languages: List[str] = ["python", "powershell", "node"]  # Primary languages by default
    # Primary: Python (AST), PowerShell (Constrained), Node.JS (npm/TypeScript)
    # Admin can add: ["python", "powershell", "node", "bash", "go"]
    # Primary: Python (safe AST), PowerShell (Constrained Language Mode), Node.JS (Windows-native, TypeScript/Angular)
    # Secondary: JavaScript (alias), Bash (opt-in)
    # Tertiary: Go, Rust, Ruby, R, Julia (opt-in + admin review)

    allow_interpreted_languages: bool = False  # Ruby, R, Julia require opt-in
    allow_compiled_languages: bool = False     # Go, Rust require opt-in + admin review
```

---

### 6. Permission Model with Admin Valve

```python
class UserPermissions:
    allowed_workspace_dirs: List[str]  # Whitelist paths
    allow_code_execution: bool         # Run Python?
    allow_external_apis: bool          # Call external services?

    # ADMIN VALVE FOR PARALLELIZATION
    parallel_execution: bool = False   # Enable parallel tasks?
    max_parallel_tasks: int = 4        # Max concurrent

    max_execution_time: int = 300      # Seconds per task
    max_file_size: int = 100           # MB per file

# Admin enables for power user:
admin.set_permission(user_id, "parallel_execution", True)
admin.set_permission(user_id, "max_parallel_tasks", 8)

# User attempts parallel task → checked against permission
if not user.permissions.parallel_execution:
    → Fall back to sequential execution (transparent to user)
```

---

### 7. Progress Reporting Strategy

**Principles:**

1. **Workspace-wide visibility**: All progress tied to WorkspaceContext.cwd
2. **Succinct updates**: Batch-level, not per-task
3. **Intelligible format**: Success/failure counts, actionable errors
4. **Non-spammy**: One message per batch; one per sequential step
5. **Streaming-friendly**: Aggregated output for chat

**Message Templates:**

**Parallel Batch Complete:**

```
✓ 12/12 completed in 8.3s (11 OK, 1 failed)

Failures (1):
  • src/broken.py: SyntaxError - invalid syntax at line 5

Ready for next step...
```

**Sequential Step Complete:**

```
✓ Step 1/5: Scan workspace
Found 42 files in /workspace/src

Ready for step 2...
```

**Task Summary (End):**

```
✓ Complete: Analyze & Document Codebase

Summary:
  • Scanned: 1 directory (42 files)
  • Analyzed: 12 Python files (11 OK, 1 failed)
  • Generated: /workspace/ANALYSIS.md
  • Total time: 24.5s

Stored in memory for future reference.
```

---

### 8. Error Handling & Recovery

**Batch Failure Matrix:**

| Scenario                 | Behavior                                               |
| ------------------------ | ------------------------------------------------------ |
| **All succeeded**        | Proceed to next step; report counts                    |
| **Partial failure**      | Continue with successful results; report errors        |
| **All failed**           | Escalate to user with error metadata; offer retry/skip |
| **Some non-recoverable** | Report, offer user choice: retry/skip/abort            |
| **Injection detected**   | Block single task; escalate for approval               |
| **Timeout**              | Task fails; others continue                            |

**Retry Logic (Sequential Steps Only):**

```python
for attempt in range(1, 3):  # Max 2 retries
    try:
        result = await executor.execute_tool(step["tool"], step["params"])
        if result["success"]:
            return result
    except TimeoutError:
        if attempt < 2:
            step["params"]["timeout"] *= 2  # Double timeout
        else:
            return {"success": False, "escalate": True}
    except Exception as e:
        if attempt < 2:
            # Regenerate with error feedback
            step = await reasoning_engine.regenerate_step(step, str(e))
        else:
            return {"success": False, "escalate": True}
```

**Batch Retry Logic:**

```python
# Batches execute once; no automatic retry
# Failed tasks reported in aggregated output
# User can request retry of failed tasks if recoverable

if batch_result.failed:
    recoverable = sum(1 for f in batch_result.failed if f["recoverable"])
    if recoverable > 0:
        # Suggest retry to user
        message += f"\n\n{recoverable} recoverable failures. Retry? (yes/no)"
```

---

### 9. Memory Integration for Learning

**Store Execution Patterns:**

```python
execution_memory = {
    "user_id": user_id,
    "task": "analyze all Python files in src/",
    "workspace_context": {
        "cwd": "/workspace/src",
        "file_count": 12
    },
    "decomposition": [
        {"step": 1, "tool": "scan_workspace"},
        {"step": 2, "tool": "analyze_code", "batch_id": "analyze_py_files", "parallel": True, "count": 12}
    ],
    "outcome": "success",
    "total_time": 24.5,
    "source_type": "agentic_execution",
    "timestamp": 1703260800
}

await memory_service.save(execution_memory)
```

**Retrieve & Use Similar Patterns:**

```python
# When planning new task:
similar = await memory_service.search(
    query="analyze Python files",
    filter={"source_type": "agentic_execution"},
    top_k=3
)

# Inject into reasoning prompt:
"Past similar tasks (consider reusing their approach):\n"
"1. Analyze Python files in src/ → scan → parallel analyze → success\n"
"2. ..."

# LLM can recognize pattern and propose batch earlier
```

---

### 10. Implementation Roadmap

**Phase 1 (Weeks 1-2): Foundation**

- [ ] Orchestrator service skeleton
- [ ] WorkspaceContext model
- [ ] Reasoning engine + JSON parsing
- [ ] Task planner (detect independent tasks)
- [ ] Unified Executor (shell + file handlers)
- [ ] Tool manifest (JSON Schema)

**Phase 2 (Weeks 3-4): Parallelization & Polyglot Support**

- [ ] ParallelExecutor (asyncio.gather)
- [ ] Error handling + partial failure logic
- [ ] Python code handler (RestrictedPython)
- [ ] Polyglot language routing (JavaScript, Bash, Go, Rust, etc.)
- [ ] Per-language sandboxing (subprocess, Docker for compiled languages)
- [ ] Progress reporting (batch-wise)
- [ ] Memory integration
- [ ] Language permission model (user-scoped language access)

**Phase 3 (Week 5): Integration**

- [ ] Filter plugin integration (agentic mode)
- [ ] Workspace context menu (set CWD)
- [ ] Admin valve for permissions
- [ ] End-to-end test (simple → complex tasks)

**Phase 4 (Week 6): Polish**

- [ ] Safety hardening review
- [ ] Documentation + examples
- [ ] Optional: Dry-run mode

---

### 11. Success Criteria

✅ **Functional**

- User sets workspace via chat menu
- User requests parallel task ("analyze all Python files")
- LLM detects parallelizable pattern, creates batch
- Orchestrator executes all in parallel
- Failed tasks don't cascade
- Batch results reported once per batch

✅ **Safe**

- No command injection
- No unauthorized paths
- Admin valve prevents parallel unless enabled
- Per-language sandboxing (RestrictedPython, subprocess, Docker containers)
- Language permissions respected (user can only execute allowed languages)
- All operations logged and auditable

✅ **Polyglot**

- Python code execution (RestrictedPython, primary)
- PowerShell code execution (Constrained Language Mode, primary)
- Node.JS / JavaScript / TypeScript execution (primary, Windows-native; Angular CLI, npm, ts-node)
- Bash command execution
- Compiled languages (Go, Rust) via Docker
- Scripting languages (Ruby, R, Julia) via subprocess
- Seamless language detection + routing (node, js, typescript, ts aliases)

✅ **Observable**

- Progress reported workspace-wide
- Error metadata user-friendly
- Streaming-friendly output (succinct, non-spammy)
- Execution traces stored in memory

✅ **Learnable**

- Similar tasks retrieve past patterns
- Parallelization patterns reused

---

## Next Steps

1. Review this refined plan — any concerns?
2. Validate tool manifest for Phase 1
3. Begin Phase 1 implementation (Orchestrator skeleton)
