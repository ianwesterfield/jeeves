# Jeeves

My agentic AI assistant for Open-WebUI. Handles semantic memory, intent classification, workspace ops, and surgical file editing.

## What It Does

Jeeves sits between you and your LLM as an intelligent filter. It:

1. **Classifies Intent** — Figures out if you're chatting casually, saving info, recalling memories, or asking for a task
2. **Manages Memory** — Stores facts, docs, and image descriptions; pulls relevant context automatically
3. **Runs Workspace Ops** — Can read, list, and **edit** files in your workspace
4. **Surgical Edits** — Supports append, replace, and insert operations on files

**What it can do:**

- `read` — View file contents
- `list` — Browse workspace structure
- `append` — Add content to end of files
- `replace` — Find and replace text in files
- `insert` — Insert text at specific positions

**Example prompts:**

```
"Show me the readme"
"Insert a credit to me in the readme file"
"Replace 'old text' with 'new text' in config.yaml"
"Add a contributors section to README.md"
```

## Architecture

```
User message → Jeeves Filter → Intent Classification (Pragmatics)
                    ↓
              ┌─────┴─────┐
              │           │
           task?       recall/save/casual?
              │           │
              ↓           ↓
         Orchestrator   Memory Search
         (reasoning)       │
              │           ↓
              ↓      Context Injected
         Executor API       │
         (file ops)         │
              │             │
              └─────┬───────┘
                    ↓
              LLM Response
```

**Key principle:** All task intents go to the Orchestrator for reasoning—no shortcut patterns in the filter.

## Services

| Service            | Port  | What it does                           |
| ------------------ | ----- | -------------------------------------- |
| `jeeves`           | 8000  | Agent core + semantic memory           |
| `pragmatics_api`   | 8001  | 4-class intent classifier (DistilBERT) |
| `extractor_api`    | 8002  | Image/audio/PDF extraction (GPU)       |
| `orchestrator_api` | 8004  | Multi-step reasoning engine            |
| `executor_api`     | 8005  | File ops + code execution              |
| `qdrant`           | 6333  | Vector database                        |
| `ollama`           | 11434 | Local LLM inference                    |
| `open-webui`       | 8180  | Chat UI (filter runs here)             |

## Intent Classification

Pragmatics uses a fine-tuned DistilBERT (98% accuracy) to classify what you want:

| Intent   | What it means            | Example                     |
| -------- | ------------------------ | --------------------------- |
| `casual` | Just chatting, no action | "How are you?"              |
| `save`   | Sharing info to remember | "My name is Ian"            |
| `recall` | Asking about past info   | "What's my email?"          |
| `task`   | Requesting an action     | "Add credits to the readme" |

## File Operations

All task requests go to the Orchestrator, which reasons about what you want and picks the right tool. The Executor then does the actual work.

### Executor Tools

| Tool              | What it does                                         |
| ----------------- | ---------------------------------------------------- |
| `read_file`       | Read file contents                                   |
| `write_file`      | Overwrite entire file                                |
| `replace_in_file` | Find and replace text (surgical)                     |
| `insert_in_file`  | Insert at start/end/before/after anchor              |
| `append_to_file`  | Append to end of file                                |
| `list_files`      | List directory contents                              |
| `scan_workspace`  | Recursive search with gitignore, pretty table output |
| `none`            | Skip (change already present)                        |

### scan_workspace Output

```
PATH: /workspace/jeeves
TOTAL: 105 items (27 dirs, 78 files)

NAME                                      TYPE    SIZE      MODIFIED
----------------------------------------------------------------------
filters                                   dir               2025-12-23 03:10:27
layers                                    dir               2025-12-22 23:36:53
README.md                                 file    8.32 KiB  2025-12-23 03:17:10
docker-compose.yaml                       file    6.36 KiB  2025-12-23 03:55:25
...
```

## Running It

### Quick Start

```powershell
# Spin everything up
docker compose up -d --build

# Check status
docker ps

# Tail logs
docker logs jeeves -f
docker logs executor_api -f
```

### Environment Variables

| Variable              | Default             | What it's for                        |
| --------------------- | ------------------- | ------------------------------------ |
| `HOST_WORKSPACE_PATH` | `C:/Code`           | Host directory mounted to /workspace |
| `QDRANT_HOST`         | `qdrant`            | Vector database host                 |
| `OLLAMA_MODEL`        | `qwen2.5:14b`       | LLM for orchestration                |
| `CLASSIFIER_MODEL`    | `distilbert_intent` | Intent classifier model              |

### Filter Sync

The Jeeves filter runs inside Open-WebUI (stored in DB, not mounted). To sync changes:

```powershell
# Push filter to Open-WebUI (utf-8-sig strips BOM)
$apiKey = (Get-Content "secrets/webui_admin_api_key.txt" -Raw).Trim()
python -c "import requests; f=open('filters/jeeves.filter.py',encoding='utf-8-sig').read(); r=requests.post('http://localhost:8180/api/v1/functions/id/api/update', headers={'Authorization':'Bearer $apiKey'}, json={'id':'api','name':'Jeeves','content':f,'meta':{'toggle':True}}, timeout=10); print(r.status_code)"
```

## Project Structure

```
jeeves/
├── docker-compose.yaml       # Full stack
├── filters/
│   └── jeeves.filter.py      # Open-WebUI filter (intent routing, workspace ops)
├── layers/
│   ├── memory/               # Semantic memory (port 8000)
│   │   ├── api/memory.py     # /save, /search endpoints
│   │   └── services/         # Embedder, Qdrant, Summarizer
│   ├── pragmatics/           # Intent classifier (port 8001)
│   │   └── services/
│   │       ├── classifier.py       # 4-class DistilBERT
│   │       └── entity_extractor.py # spaCy NER
│   ├── extractor/            # Media extraction (port 8002)
│   │   └── services/         # LLaVA, Whisper, PDF
│   ├── orchestrator/         # Reasoning engine (port 8004)
│   │   └── services/
│   │       ├── reasoning_engine.py # LLM step generation
│   │       ├── code_planner.py     # Specialist for code edits
│   │       └── workspace_state.py  # External state tracking
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
# You say: "list the files in this workspace"
#
# 1. Intent classified as "task" (99% confidence)
# 2. Task goes to Orchestrator for reasoning
# 3. Orchestrator picks: scan_workspace tool
# 4. Executor runs scan with gitignore support
# 5. Pretty-formatted table injected into context
# 6. LLM presents the results
```

### 2. Always Delegate (No Shortcuts)

All task intents go through the Orchestrator:

```python
# Flow in _orchestrate_task():
1. Set workspace context on Orchestrator
2. Get next step (tool + params) from Orchestrator
3. Execute tool via Executor API
4. Return formatted results for context injection

# No hardcoded patterns—Orchestrator decides everything
```

### 3. Status Messages

Live feedback while processing:

| Icon | What's happening   |
| ---- | ------------------ |
| ✨   | Thinking           |
| 🔍   | Scanning workspace |
| 📖   | Reading files      |
| ✏️   | Editing files      |
| ⚙️   | Running code       |
| 💾   | Saving to memory   |
| 📚   | Found memories     |
| ✅   | Done               |
| ❌   | Something broke    |

### 4. Memory Integration

- **Save**: Extracts facts from the conversation, embeds them, stores in Qdrant
- **Search**: Embeds your query, finds similar memories (cosine > 0.35)
- **Inject**: Prepends relevant memories to LLM context

## Quick Test

```powershell
# Test file append via Executor
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

```

## License

MIT

## Credits

- **Ian Westerfield** - Creator and maintainer
```
