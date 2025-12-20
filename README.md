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

```powershell
# Full stack with Open-WebUI
docker compose up -d --build

# Rebuild specific service
docker compose build memory_api && docker compose up -d memory_api
```

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
tools-api/
├── <tool>/
│   ├── main.py                 # FastAPI entry point
│   ├── filter.py               # Open-WebUI filter plugin
│   ├── api/                    # REST endpoints (search/save with score gap filter)
│   ├── services/               # Service layer
│   ├── utils/                  # Shared logic
│   └── static/
│       ├── ai-plugin.json      # Plugin manifest
│       └── openapi.yaml        # API spec
├── <tool 2>/
│   ├── ...
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
