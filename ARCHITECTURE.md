# Jeeves Architecture Documentation

## Overview

Jeeves is an agentic AI assistant for Open-WebUI that provides:

- **Semantic Memory** — Stores and retrieves conversation context via Qdrant
- **Intent Classification** — 4-class DistilBERT model (casual/save/recall/task)
- **Workspace Operations** — Read, list, and edit files in mounted workspace
- **Surgical File Editing** — Replace, insert, append operations

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Open-WebUI (8180)                             │
│                                  │                                      │
│                    ┌─────────────┴─────────────┐                        │
│                    │    jeeves.filter.py       │                        │
│                    │  (inlet/outlet hooks)     │                        │
│                    └─────────────┬─────────────┘                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
    │ Pragmatics  │       │   Jeeves    │       │  Executor   │
    │    8001     │       │    8000     │       │    8005     │
    │             │       │             │       │             │
    │ 4-class     │       │ Memory API  │       │ File Ops    │
    │ Intent      │       │ /save       │       │ /tool       │
    │ Classifier  │       │ /search     │       │ /file       │
    └─────────────┘       └──────┬──────┘       └─────────────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │   Qdrant    │
                          │    6333     │
                          │             │
                          │ Vector DB   │
                          │ 768-dim     │
                          └─────────────┘
```

---

## Services

### 1. Jeeves Filter (filters/jeeves.filter.py)

The main entry point running inside Open-WebUI.

**Responsibilities:**

- Classify user intent via Pragmatics API
- Detect edit patterns (insert/add/replace) and execute via Executor
- Search memory for relevant context
- Inject context and results into LLM conversation

**Pattern Priority (checked in order):**

```python
1. Edit patterns     → Execute write via Executor API
2. Read patterns     → Read file, return content
3. List patterns     → Scan workspace, return listing
4. Orchestrator      → Multi-step task planning
5. Memory search     → Retrieve relevant memories
```

**Edit Pattern Detection:**

```python
# Supported patterns:
"insert X in readme"          → append_to_file(README.md)
"add X to README.md"          → append_to_file(README.md)
"replace 'X' with 'Y' in F"   → replace_in_file(F, X, Y)
```

---

### 2. Pragmatics API (Port 8001)

4-class intent classification using fine-tuned DistilBERT.

**Model:** `distilbert_intent` (98% validation accuracy)

**Classes:**

| Class    | ID  | Description                   | Example                 |
| -------- | --- | ----------------------------- | ----------------------- |
| `casual` | 0   | General chat, no action       | "How are you?"          |
| `save`   | 1   | User sharing info to remember | "My name is Ian"        |
| `recall` | 2   | User asking about past info   | "What's my email?"      |
| `task`   | 3   | User requesting action        | "Add credits to readme" |

**Endpoint:**

```
POST /api/pragmatics/classify
Input:  { "text": "insert a credit in the readme" }
Output: { "intent": "task", "confidence": 0.99, "label": 3 }
```

---

### 3. Executor API (Port 8005)

Polyglot code execution and file operations with sandbox enforcement.

**File Operations:**

| Tool              | Method                    | Description           |
| ----------------- | ------------------------- | --------------------- |
| `read_file`       | `read(path)`              | Read file contents    |
| `write_file`      | `write(path, content)`    | Overwrite entire file |
| `replace_in_file` | `replace(path, old, new)` | Surgical find/replace |
| `insert_in_file`  | `insert(path, pos, text)` | Insert at position    |
| `append_to_file`  | `append(path, content)`   | Add to end of file    |
| `list_files`      | `list_dir(path)`          | Directory listing     |
| `scan_workspace`  | `scan(path, pattern)`     | Recursive glob search |

**Positions for insert_in_file:**

- `start` — Beginning of file
- `end` — End of file
- `before` — Before anchor text
- `after` — After anchor text

**Endpoint:**

```
POST /api/execute/tool
Input: {
  "tool": "replace_in_file",
  "params": {
    "path": "/workspace/README.md",
    "old_text": "TODO",
    "new_text": "DONE"
  },
  "workspace_context": {
    "workspace_root": "/workspace",
    "cwd": "/workspace",
    "allow_file_write": true
  }
}
Output: { "success": true, "output": "Replaced 3 occurrence(s)" }
```

**Permission Checks:**

- `allow_file_write` — Required for write/replace/insert/append/delete
- `allow_shell_commands` — Required for shell execution
- `allow_code_execution` — Required for Python/Node/PowerShell

---

### 4. Memory API (Port 8000)

Semantic memory storage and retrieval.

**Components:**

- **Embedder:** SentenceTransformer `all-mpnet-base-v2` (768-dim, L2-normalized)
- **Storage:** Qdrant vector database
- **Summarizer:** DistilBART (optional)
- **Fact Extractor:** KeyBERT + regex for importance filtering

**Endpoints:**

```
POST /api/memory/save
  - Check importance (skip casual queries)
  - Embed with SentenceTransformer
  - Upsert to Qdrant
  - Return existing_context

POST /api/memory/search
  - Embed query
  - Search Qdrant (threshold: 0.35)
  - Return matching memories

GET /api/jeeves/filter
  - Serve filter source code
```

---

### 5. Extractor API (Port 8002)

Media-to-text extraction (GPU-accelerated).

**Models:**

- **Image:** LLaVA-1.5-7B (4-bit) or Florence-2 fallback
- **Audio:** Whisper (base model)
- **PDF:** PyMuPDF

---

### 6. Orchestrator API (Port 8004)

Multi-step reasoning and task planning (under development).

**Endpoints:**

```
POST /api/orchestrator/set-workspace
POST /api/orchestrator/next-step
POST /api/orchestrator/execute-batch
```

---

## Data Flow

### Edit Request Flow

```
User: "insert a credit to me in the readme"
                    │
                    ▼
┌─────────────────────────────────────────┐
│        jeeves.filter.py inlet()         │
│                                         │
│  1. Classify intent → "task" (99%)      │
│  2. Match edit pattern:                 │
│     "insert" + "credit" + "readme"      │
│  3. Normalize file → README.md          │
│  4. Build content:                      │
│     "## Credits\n- Ian - Creator"       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Executor API (8005)             │
│                                         │
│  POST /api/execute/tool                 │
│  tool: "append_to_file"                 │
│  path: "/workspace/jeeves/README.md"    │
│  content: "## Credits\n- Ian..."        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│        Filter injects result            │
│                                         │
│  "### File Operation Result ###         │
│   Operation: Append to README.md        │
│   Status: ✅ Success                    │
│   Added: ## Credits..."                 │
└─────────────────────────────────────────┘
                    │
                    ▼
         LLM confirms edit done
```

### Memory Search Flow

```
User: "What's my name?"
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Classify intent → "recall"          │
│  2. Search memory API                   │
│     query: "what's my name"             │
│     top_k: 5                            │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Qdrant Search                   │
│                                         │
│  Embed query → 768-dim vector           │
│  Cosine similarity > 0.35               │
│  Filter by user_id                      │
│  Return: "My name is Ian" (score: 0.82) │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Context injected:                      │
│  "### Memories ###                      │
│   - My name is Ian                      │
│   ### End Memories ###"                 │
└─────────────────────────────────────────┘
         │
         ▼
    LLM: "Your name is Ian"
```

---

## Configuration

### Environment Variables

| Variable                      | Service               | Default             | Purpose                       |
| ----------------------------- | --------------------- | ------------------- | ----------------------------- |
| `HOST_WORKSPACE_PATH`         | Orchestrator/Executor | `C:/Code`           | Host directory for /workspace |
| `QDRANT_HOST`                 | Jeeves                | `qdrant`            | Vector DB hostname            |
| `QDRANT_PORT`                 | Jeeves                | `6333`              | Vector DB port                |
| `CLASSIFIER_MODEL`            | Pragmatics            | `distilbert_intent` | Intent model name             |
| `INTENT_CONFIDENCE_THRESHOLD` | Pragmatics            | `0.50`              | Min confidence for intent     |
| `IMAGE_MODEL`                 | Extractor             | `llava-4bit`        | Vision model                  |
| `WHISPER_MODEL`               | Extractor             | `base`              | Audio transcription model     |

### Docker Compose Services

```yaml
services:
  jeeves: 8000 # Memory + Filter serving
  pragmatics_api: 8001 # Intent classification
  extractor_api: 8002 # Media extraction (GPU)
  orchestrator_api: 8004 # Task planning
  executor_api: 8005 # File ops + code execution
  qdrant: 6333 # Vector database
  ollama: 11434 # LLM inference
  open-webui: 8180 # Chat UI
```

---

## File Structure

```
jeeves/
├── docker-compose.yaml
├── filters/
│   └── jeeves.filter.py          # Main Open-WebUI filter
├── layers/
│   ├── memory/
│   │   ├── main.py               # FastAPI app
│   │   ├── api/memory.py         # /save, /search endpoints
│   │   └── services/
│   │       ├── embedder.py       # SentenceTransformer
│   │       ├── qdrant_client.py  # Qdrant connection
│   │       └── summarizer.py     # DistilBART
│   ├── pragmatics/
│   │   ├── server.py             # FastAPI app
│   │   ├── services/classifier.py # DistilBERT 4-class
│   │   └── static/distilbert_intent/  # Trained model
│   ├── extractor/
│   │   ├── main.py
│   │   └── services/
│   │       ├── image_extractor.py  # LLaVA/Florence
│   │       └── audio_extractor.py  # Whisper
│   ├── orchestrator/
│   │   ├── main.py
│   │   └── services/
│   │       ├── reasoning_engine.py
│   │       └── task_planner.py
│   └── executor/
│       ├── main.py
│       ├── api/executor.py       # /tool endpoint
│       └── services/
│           ├── file_handler.py   # read/write/replace/insert/append
│           ├── polyglot_handler.py # Python/Node/PowerShell
│           └── shell_handler.py
└── .github/
    └── copilot-instructions.md
```

---

## Security

### Workspace Sandbox

- All file operations validate paths against `workspace_root`
- Paths outside workspace are rejected
- Write operations require explicit `allow_file_write: true`

### Permission Flags

```python
class WorkspaceContext:
    workspace_root: str       # Sandbox boundary
    cwd: str                  # Current directory
    allow_file_write: bool    # Enable write operations
    allow_shell_commands: bool # Enable shell execution
    allow_code_execution: bool # Enable code runners
    allowed_languages: List[str] # Permitted languages
```

---

## Development

### Filter Sync

After editing `filters/jeeves.filter.py`, sync to Open-WebUI:

```powershell
python -c "
import requests
f = open('filters/jeeves.filter.py', encoding='utf-8').read()
r = requests.post(
    'http://localhost:8180/api/v1/functions/id/api/update',
    headers={'Authorization':'Bearer YOUR_API_KEY'},
    json={'id':'api','name':'Jeeves','content':f,'meta':{'toggle':True}},
    timeout=30
)
print(f'Status: {r.status_code}')
"
```

### Rebuild Services

```powershell
# Rebuild executor after code changes
docker compose build --no-cache executor_api
docker compose up -d executor_api

# Rebuild all
docker compose up -d --build
```

### Test Endpoints

```powershell
# Intent classification
Invoke-RestMethod -Uri 'http://localhost:8001/api/pragmatics/classify' `
  -Method Post -ContentType 'application/json' `
  -Body '{"text":"add credits to readme"}'

# File append
Invoke-RestMethod -Uri 'http://localhost:8005/api/execute/tool' `
  -Method Post -ContentType 'application/json' `
  -Body (@{
    tool='append_to_file'
    params=@{path='/workspace/test.txt';content='Hello'}
    workspace_context=@{workspace_root='/workspace';cwd='/workspace';allow_file_write=$true}
  } | ConvertTo-Json -Depth 5)

# Memory search
Invoke-RestMethod -Uri 'http://localhost:8000/api/memory/search' `
  -Method Post -ContentType 'application/json' `
  -Body '{"user_id":"test","query_text":"my name","top_k":5}'
```
