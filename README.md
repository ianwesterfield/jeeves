# Jeeves

My semantic memory layer for Open-WebUI. It remembers what you tell it—names, preferences, facts from documents, even descriptions of images you upload—and brings that context back when it's relevant.

## What It Does

Every conversation flows through a filter plugin that decides what's worth remembering. Things like "My name is Ian" or "Jesus is Christ" get extracted and stored. Documents get chunked and indexed. Images get described by vision models. When you ask something later, Jeeves searches for related memories and injects them into the conversation so the model has context.

**Endpoints:**

- `POST /api/memory/save` — Store a conversation (checks for duplicates, extracts facts)
- `POST /api/memory/search` — Find related memories by query
- `POST /api/memory/summaries` — Get summarized versions of memories
- `GET /api/memory/filter` — Serves the Open-WebUI filter code

## How It Works

```
User message → Filter plugin → Memory API → Qdrant
       ↓              ↓
   [images]    Search for context
       ↓              ↓
  Extractor    Inject into conversation
       ↓              ↓
  Describe     Model responds with full context
```

**Key pieces:**

- **Embeddings**: `all-mpnet-base-v2` from SentenceTransformers, 768 dimensions, normalized
- **Storage**: Qdrant vector DB with COSINE similarity
- **Pragmatics Classifier**: DistilBERT model (96% accuracy) that decides if something's worth saving
- **Fact Extractor**: Regex patterns that pull out names, employers, VLANs, nicknames, etc.
- **Summarizer**: DistilBART for concise summaries, falls back to first few sentences if needed
- **Image Captioning**: Florence-2 for detailed image descriptions, BLIP as fallback
- **Document Chunking**: Markdown-aware chunking with section headers preserved
- **Score Gap Filter**: Prevents retrieving loosely related memories (15% relative threshold)
- **Separate Search/Storage Vectors**: Search uses last user message; storage uses full conversation

## Services

| Service          | Port  | What it does                      |
| ---------------- | ----- | --------------------------------- |
| `memory_api`     | 8000  | Main FastAPI service              |
| `qdrant`         | 6333  | Vector database                   |
| `pragmatics_api` | 8001  | Classifier for save decisions     |
| `extractor_api`  | 8002  | Image/audio/PDF extraction (GPU)  |
| `open-webui`     | 3000  | Chat UI (filter plugin runs here) |
| `ollama`         | 11434 | Local LLM inference               |

## Running It

### Quick Start

```powershell
# Start everything (pulls base images from Docker Hub, builds app code)
docker compose up -d --build

# Check status
docker ps

# View logs
docker logs memory_api -f
```

Build time: **~30 seconds** (base images are pre-built on Docker Hub)

### When to Use `--build`

| Scenario                    | Command                           |
| --------------------------- | --------------------------------- |
| Code changes (Python files) | `docker compose up -d --build`    |
| Just restart containers     | `docker compose up -d`            |
| Full rebuild from scratch   | `docker compose build --no-cache` |

### Building Base Images (Maintainers Only)

Base images contain system deps + pip packages. Only rebuild when `requirements.txt` changes:

```powershell
# Build and push all base images to Docker Hub
.\build-base-images.ps1 -Push

# Build specific service only
.\build-base-images.ps1 -Services "memory" -Push

# Build locally without pushing
.\build-base-images.ps1
```

### Using Different Registry

Create a `.env` file:

```env
BASE_REGISTRY=yourusername
```

Or set inline:

```powershell
$env:BASE_REGISTRY = "yourusername"
docker compose up -d --build
```

## Build Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Base Images (Docker Hub: ianwesterfield/jeeves-*-base)         │
│  Built once when requirements.txt changes (~35 min)              │
│                                                                  │
│  ├─ jeeves-memory-base:latest     (python + sentence-transformers) │
│  ├─ jeeves-extractor-base:latest  (python + torch + whisper)       │
│  └─ jeeves-pragmatics-base:latest (python + transformers)          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ docker compose up --build
┌──────────────────────────────────────────────────────────────────┐
│  App Images (built locally on each code change, ~30 sec)        │
│                                                                  │
│  ├─ FROM jeeves-memory-base + COPY memory/*.py                   │
│  ├─ FROM jeeves-extractor-base + COPY extractor/*.py             │
│  └─ FROM jeeves-pragmatics-base + COPY pragmatics/*.py           │
└──────────────────────────────────────────────────────────────────┘
```

### File Structure

```
tools-api/
├── memory/
│   ├── Dockerfile          ← FROM jeeves-memory-base + app code
│   ├── Dockerfile.base     ← System deps + pip (for maintainers)
│   ├── requirements.txt    ← Python dependencies
│   └── *.py                ← Application code
├── extractor/
│   ├── Dockerfile
│   ├── Dockerfile.base
│   └── ...
└── pragmatics/
    ├── Dockerfile
    ├── Dockerfile.base
    └── ...
```

### Why This Architecture?

| Scenario          | Old (single Dockerfile) | New (base + app)             |
| ----------------- | ----------------------- | ---------------------------- |
| First build       | 35 min                  | 35 min (base) + 30 sec (app) |
| Code change       | 2 sec (cached)          | 30 sec                       |
| New machine       | 35 min                  | 30 sec (pulls from Hub)      |
| Team member clone | 35 min                  | 30 sec                       |
| CI/CD pipeline    | 35 min                  | 30 sec                       |

**Regular development:** Just `docker compose up -d --build` (~30 seconds)

## Environment Variables

| Variable          | Default                         | What it does                          |
| ----------------- | ------------------------------- | ------------------------------------- |
| `QDRANT_HOST`     | `qdrant`                        | Qdrant server                         |
| `QDRANT_PORT`     | `6333`                          | Qdrant port                           |
| `INDEX_NAME`      | `user_memory_collection`        | Collection name                       |
| `SUMMARY_MODEL`   | `sshleifer/distilbart-cnn-12-6` | Summarization model                   |
| `SUMMARY_DEVICE`  | `cpu`                           | CPU or GPU index                      |
| `STORE_VERBATIM`  | `true`                          | Store full messages or just summaries |
| `PRAGMATICS_HOST` | `pragmatics_api`                | Classifier service                    |
| `PRAGMATICS_PORT` | `8001`                          | Classifier port                       |
| `WHISPER_MODEL`   | `base`                          | Whisper model for audio transcription |

## Quick Test

```powershell
# Search memories
Invoke-RestMethod -Uri 'http://localhost:8000/api/memory/search' `
  -Method Post `
  -Body (@{ user_id='test'; query_text='project settings'; top_k=5 } | ConvertTo-Json) `
  -ContentType 'application/json'

# List all memories for a user
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:6333/collections/user_memory_collection/points/scroll" `
  -ContentType "application/json" `
  -Body '{"limit": 20, "with_payload": true, "with_vector": false}'
```

## Project Structure

```
jeeves/
├── docker-compose.yaml      # Full stack definition
├── build-base-images.ps1    # Script to build/push base images
├── .env                     # Registry config (BASE_REGISTRY=...)
├── ARCHITECTURE.md          # Detailed technical docs
└── tools-api/
    ├── memory/
    │   ├── main.py          # FastAPI entry point
    │   ├── memory.filter.py # Open-WebUI filter plugin
    │   ├── api/memory.py    # REST endpoints
    │   ├── services/        # Embedder, Qdrant, Summarizer
    │   ├── Dockerfile       # App image (FROM base)
    │   ├── Dockerfile.base  # Base image (deps + pip)
    │   └── requirements.txt
    ├── extractor/
    │   ├── main.py          # FastAPI with model preloading
    │   ├── api/extractor.py # /extract endpoint
    │   ├── services/        # Image, Audio, PDF extractors
    │   ├── Dockerfile
    │   ├── Dockerfile.base
    │   └── requirements.txt
    └── pragmatics/
        ├── server.py        # FastAPI classifier service
        ├── services/        # DistilBERT classifier
        ├── static/          # Trained model weights
        ├── Dockerfile
        ├── Dockerfile.base
        └── requirements.txt
```

## Filter Plugin

The filter (`memory.filter.py`) must be installed in Open-WebUI:

1. Go to **Admin Panel → Functions**
2. Create a new filter, paste the contents of `memory.filter.py`
3. Enable it globally or per-model

The filter:

- Extracts images from messages and sends to extractor for captioning
- Chunks uploaded documents and saves each chunk
- Searches for relevant context before every response
- Injects found memories as a context block

## Source Types

Memories are tagged by source:

| Type             | Description                       |
| ---------------- | --------------------------------- |
| `prompt`         | User statements worth remembering |
| `document_chunk` | Sections from uploaded documents  |
| `image`          | Descriptions of uploaded images   |

## License

MIT—do whatever you want with it.
