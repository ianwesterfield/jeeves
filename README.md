# Jeeves

An agentic AI assistant for Open-WebUI with semantic memory, intent classification, workspace operations, and file editing capabilities.

## What It Does

Jeeves acts as an intelligent filter between you and your LLM. It:

1. **Classifies Intent** — Determines if you're asking casually, saving info, recalling memories, or requesting a task
2. **Manages Memory** — Stores facts, documents, and image descriptions; retrieves relevant context automatically
3. **Executes Workspace Operations** — Can read, list, and **edit** files in your workspace
4. **Surgical File Editing** — Supports append, replace, and insert operations on files

**Key Capabilities:**

- `read` — View file contents
- `list` — Browse workspace structure
- `append` — Add content to end of files
- `replace` — Find and replace text in files
- `insert` — Insert text at specific positions

**Example Commands:**

```
"Show me the readme"
"Insert a credit to me in the readme file"
"Replace 'old text' with 'new text' in config.yaml"
"Add a contributors section to README.md"
```

## Architecture

```
User message → Jeeves Filter → Intent Classification
                    ↓
              ┌─────┴─────┐
              │           │
         Edit Request?   Read/Recall?
              │           │
              ↓           ↓
         Executor API   Memory Search
              │           │
              ↓           ↓
         File Modified   Context Injected
              │           │
              └─────┬─────┘
                    ↓
              LLM Response
```

## Services

| Service            | Port  | Purpose                                |
| ------------------ | ----- | -------------------------------------- |
| `jeeves`           | 8000  | Agent core + semantic memory           |
| `pragmatics_api`   | 8001  | 4-class intent classifier (DistilBERT) |
| `extractor_api`    | 8002  | Image/audio/PDF extraction (GPU)       |
| `orchestrator_api` | 8004  | Multi-step reasoning engine            |
| `executor_api`     | 8005  | File operations + code execution       |
| `qdrant`           | 6333  | Vector database                        |
| `ollama`           | 11434 | Local LLM inference                    |
| `open-webui`       | 8180  | Chat UI (filter runs here)             |

## Intent Classification

The pragmatics service uses a fine-tuned DistilBERT model (98% accuracy) to classify user intent:

| Intent   | Description                    | Example                     |
| -------- | ------------------------------ | --------------------------- |
| `casual` | General chat, no action needed | "How are you?"              |
| `save`   | User sharing info to remember  | "My name is Ian"            |
| `recall` | User asking about past info    | "What's my email?"          |
| `task`   | User requesting an action      | "Add credits to the readme" |

## File Operations

The filter detects edit patterns and executes them via the Executor API:

### Supported Patterns

```python
# Append/Insert patterns
"insert a credit to me in the readme"     → Appends to README.md
"add my name to the readme file"          → Appends to README.md
"put a section in ARCHITECTURE.md"        → Appends to ARCHITECTURE.md

# Replace patterns
"replace 'X' with 'Y' in file.txt"        → Surgical text replacement
```

### Executor API Tools

| Tool              | Description                             |
| ----------------- | --------------------------------------- |
| `read_file`       | Read file contents                      |
| `write_file`      | Overwrite entire file                   |
| `replace_in_file` | Find and replace text (surgical)        |
| `insert_in_file`  | Insert at start/end/before/after anchor |
| `append_to_file`  | Append to end of file                   |
| `list_files`      | List directory contents                 |
| `scan_workspace`  | Recursive file search with glob         |

## Running It

### Quick Start

```powershell
# Start everything
docker compose up -d --build

# Check status
docker ps

# View logs
docker logs jeeves -f
docker logs executor_api -f
```

### Environment Variables

| Variable              | Default             | Purpose                              |
| --------------------- | ------------------- | ------------------------------------ |
| `HOST_WORKSPACE_PATH` | `C:/Code`           | Host directory mounted to /workspace |
| `QDRANT_HOST`         | `qdrant`            | Vector database host                 |
| `OLLAMA_MODEL`        | `llama3.2`          | Default LLM for orchestration        |
| `CLASSIFIER_MODEL`    | `distilbert_intent` | Intent classifier model              |

### Filter Sync

The Jeeves filter runs inside Open-WebUI. To sync changes:

```powershell
# Sync filter to Open-WebUI
python -c "
import requests
f = open('filters/jeeves.filter.py', encoding='utf-8').read()
r = requests.post(
    'http://localhost:8180/api/v1/functions/id/api/update',
    headers={'Authorization':'Bearer YOUR_API_KEY','Content-Type':'application/json'},
    json={'id':'api','name':'Jeeves Agentic','content':f,'meta':{'description':'Agent','toggle':True}},
    timeout=30
)
print(f'Status: {r.status_code}')
"
```

## Project Structure

```
jeeves/
├── docker-compose.yaml       # Full stack definition
├── filters/
│   └── jeeves.filter.py      # Open-WebUI filter (edit detection, workspace ops)
├── layers/
│   ├── memory/               # Semantic memory service (port 8000)
│   │   ├── api/memory.py     # /save, /search endpoints
│   │   └── services/         # Embedder, Qdrant, Summarizer
│   ├── pragmatics/           # Intent classifier (port 8001)
│   │   └── services/classifier.py  # 4-class DistilBERT
│   ├── extractor/            # Media extraction (port 8002)
│   │   └── services/         # Image (LLaVA), Audio (Whisper), PDF
│   ├── orchestrator/         # Reasoning engine (port 8004)
│   │   └── services/         # Task planning, parallel execution
│   └── executor/             # Code/file execution (port 8005)
│       └── services/
│           ├── file_handler.py      # read, write, replace, insert, append
│           ├── polyglot_handler.py  # Python, Node, PowerShell
│           └── shell_handler.py     # Shell commands
└── .github/
    └── copilot-instructions.md  # AI coding guidelines
```

## How It Works

### 1. Filter Inlet (jeeves.filter.py)

```python
# User says: "insert a credit to me in the readme file"
#
# 1. Intent classified as "task" (99% confidence)
# 2. Edit pattern detected: insert + credit + readme
# 3. Executor API called: append_to_file(README.md, "## Credits\n...")
# 4. Result injected into context
# 5. LLM confirms the edit was made
```

### 2. Pattern Priority

Edit patterns are checked **FIRST**, before read/list patterns:

```python
# Check order in _orchestrate_task():
1. Edit patterns (insert/add/replace/append)  → Execute write
2. Read patterns (show/read/display)          → Return content
3. List patterns (list files/summarize)       → Return listing
4. Orchestrator (complex tasks)               → Multi-step planning
```

### 3. Memory Integration

- **Save**: Facts extracted from conversation, embedded, stored in Qdrant
- **Search**: Query embedded, similar memories retrieved (cosine similarity > 0.35)
- **Inject**: Relevant memories prepended to LLM context

## Quick Test

```powershell
# Test executor file append
Invoke-RestMethod -Uri 'http://localhost:8005/api/execute/tool' -Method Post `
  -ContentType 'application/json' `
  -Body (@{
    tool='append_to_file'
    params=@{path='/workspace/test.txt'; content='Hello from Jeeves'}
    workspace_context=@{workspace_root='/workspace'; cwd='/workspace'; allow_file_write=$true}
  } | ConvertTo-Json -Depth 5)

# Test memory search
Invoke-RestMethod -Uri 'http://localhost:8000/api/memory/search' -Method Post `
  -ContentType 'application/json' `
  -Body (@{user_id='test'; query_text='my name'; top_k=5} | ConvertTo-Json)

# Test intent classification
Invoke-RestMethod -Uri 'http://localhost:8001/api/pragmatics/classify' -Method Post `
  -ContentType 'application/json' `
  -Body (@{text='Add a credit to the readme'} | ConvertTo-Json)
```

## License

MIT

## Credits

- **Ian Westerfield** - Creator and maintainer
