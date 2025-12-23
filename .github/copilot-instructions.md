# AI Coding Agent Instructions for this Repo

## Command Execution Guidelines

**For the user's workflow:**

- ❌ Do NOT suggest Docker/compose rebuild commands in prose instructions
- ✅ DO suggest PowerShell automations as runnable code blocks
- ✅ DO suggest supporting commands (logs, status checks, etc.) as runnable code blocks
- User handles all docker/build commands manually

**Code block format for supporting commands:**

```powershell
docker logs extractor_api
docker ps -a
```

## Big Picture

- **Purpose:** Jeeves is an agentic assistant with semantic memory, intent classification, multi-step reasoning, and polyglot code execution.
- **Key services:**
  - `jeeves` on port `8000` - agent core + semantic memory (filter endpoint: `/api/agent`)
  - `pragmatics_api` on port `8001` - intent classification
  - `extractor_api` on port `8002` - document/image/audio extraction
  - `orchestrator_api` on port `8004` - multi-step reasoning engine
  - `executor_api` on port `8005` - polyglot code execution
  - `qdrant` vector DB on `6333` (HTTP) and `5100` (UI)
  - `ollama` on port `11434` - local LLM inference
- **Data flow:** Open‑WebUI → `jeeves.filter.py` → classify intent → orchestrate if task → search/save memory → inject context

## Where Things Live

### Directory Structure

```
jeeves/
├── filters/
│   └── jeeves.filter.py      # Main Open-WebUI filter (mounted read-only)
├── layers/
│   ├── memory/               # Semantic memory service (port 8000)
│   ├── pragmatics/           # Intent classification (port 8001)
│   ├── extractor/            # Media extraction (port 8002)
│   ├── orchestrator/         # Reasoning engine (port 8004)
│   └── executor/             # Code execution (port 8005)
└── docker-compose.yaml
```

### Memory Service (layers/memory/)

- **FastAPI app:** `main.py` serves filter at `/api/jeeves/filter`
- **Router:** `api/memory.py` implements `/save` and `/search`
- **Embeddings:** `services/embedder.py` uses `SentenceTransformer("all-mpnet-base-v2")`, 768-dim
- **Qdrant client:** `services/qdrant_client.py` lazy singleton

### Orchestrator Service (layers/orchestrator/)

- **FastAPI app:** `main.py` on port 8004
- **Router:** `api/orchestrator.py` - set-workspace, next-step, execute-batch
- **Services:** reasoning_engine.py, task_planner.py, parallel_executor.py, memory_connector.py

### Executor Service (layers/executor/)

- **FastAPI app:** `main.py` on port 8005
- **Router:** `api/executor.py` - tool, code, shell, file endpoints
- **Services:** polyglot_handler.py (Python/PowerShell/Node), shell_handler.py, file_handler.py

### Extractor Service (layers/extractor/)

- **Image extraction:** Uses LLaVA-1.5-7B (4-bit) or Florence-2 fallback
- **Audio extraction:** Whisper for transcription
- **PDF extraction:** PyMuPDF

## Conventions & Patterns

### Jeeves Filter (filters/jeeves.filter.py)

- **Intent classification:** Uses pragmatics_api or falls back to regex heuristics
- **Orchestrator integration:** For task intents, engages orchestrator for multi-step planning
- **Memory operations:** Saves documents/images, searches for relevant context
- **Context injection:** Prepends memories and analysis to user message

### Memory Service

- **Message content:** `content` may be `str` or `List[dict]` with items like `{type: "text", text: "..."}` or `{type: "image_url", ...}`
- **Deterministic IDs:** Points use `_make_uuid(user_id, content_hash)` producing 64‑bit int
- **Importance gating:** `_is_worth_saving(messages)` uses KeyBERT to skip casual queries

### Extractor Service

- **Image model routing:** Checks `_model_type in ("llava", "llava-4bit")` to route correctly
- **Lazy model loading:** Models load on first request and persist for container lifetime

## Build, Run, Debug

- **Docker Compose (Windows PowerShell):**
  ```powershell
  docker compose -f 'docker-compose.yaml' up -d --build
  ```
- **Environment:** Services configured via env vars in docker-compose.yaml
- **Volumes:**
  - Qdrant: `C:/docker-data/qdrant/storage`
  - Models: `C:/docker-data/models`
  - Filters: `./filters:/filters:ro` (read-only mount)
- **Local API test:**
  ```powershell
  Invoke-RestMethod -Uri 'http://localhost:8000/api/memory/search' -Method Post -Body (@{ user_id='anonymous'; query_text='test'; top_k=5 } | ConvertTo-Json) -ContentType 'application/json'
  ```

## Integration Notes

- **Jeeves filter:** `GET /api/jeeves/filter` returns filter source for Open-WebUI
- **Legacy endpoint:** `/api/memory/filter` redirects to Jeeves filter
- **Network:** All services join `webtools_network` (external)
- **Filter editing:** Edit `filters/jeeves.filter.py` directly - no rebuild needed (mounted volume)

## Guardrails for Agents

### Memory Service

- Do not change collection params unless migrating existing data
- Keep embedding model and dim in sync (768-dim)
- Preserve `Message` shape and mixed `content` handling

### Orchestrator/Executor

- Respect workspace_root boundaries for file operations
- Check allowed_languages before code execution
- Honor allow_file_write and allow_shell_commands flags
