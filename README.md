# Jeeves

My semantic memory layer for Open-WebUI. It remembers what you tell it—names, preferences, facts from documents—and brings that context back when it's relevant.

## What It Does

Every conversation flows through a filter plugin that decides what's worth remembering. Things like "My name is Ian" or "I work at Satcom Direct" get extracted and stored. When you ask something later, Jeeves searches for related memories and injects them into the conversation so the model has context.

**Endpoints:**

- `POST /api/memory/save` — Store a conversation (checks for duplicates, extracts facts)
- `POST /api/memory/search` — Find related memories by query
- `POST /api/memory/summaries` — Get summarized versions of memories
- `GET /api/memory/filter` — Serves the Open-WebUI filter code

## How It Works

```
User message → Filter plugin → Memory API → Qdrant
                    ↓
              Search for context
                    ↓
              Inject into conversation
                    ↓
              Model responds with full context
```

**Key pieces:**

- **Embeddings**: `all-mpnet-base-v2` from SentenceTransformers, 768 dimensions, normalized
- **Storage**: Qdrant vector DB with COSINE similarity
- **Pragmatics Classifier**: DistilBERT model that decides if something's worth saving. If it's uncertain (confidence < 0.60), I save anyway—better to have it than not.
- **Fact Extractor**: Regex patterns that pull out names, employers, VLANs, nicknames, etc.
- **Summarizer**: DistilBART for concise summaries, falls back to first few sentences if needed

## Services

| Service          | Port | What it does                  |
| ---------------- | ---- | ----------------------------- |
| `memory_api`     | 8000 | Main FastAPI service          |
| `qdrant`         | 6333 | Vector database               |
| `pragmatics_api` | 8001 | Classifier for save decisions |

## Running It

```powershell
# Full stack with Open-WebUI
docker compose -f docker-compose.yaml up -d --build

# Just the memory service
docker compose -f docker/tools-api/memory/docker-compose.yaml up -d --build
```

## Environment Variables

| Variable          | Default                         | What it does                          |
| ----------------- | ------------------------------- | ------------------------------------- |
| `QDRANT_HOST`     | `localhost`                     | Qdrant server                         |
| `QDRANT_PORT`     | `6333`                          | Qdrant port                           |
| `INDEX_NAME`      | `user_memory_collection`        | Collection name                       |
| `SUMMARY_MODEL`   | `sshleifer/distilbart-cnn-12-6` | Summarization model                   |
| `SUMMARY_DEVICE`  | `cpu`                           | CPU or GPU index                      |
| `STORE_VERBATIM`  | `true`                          | Store full messages or just summaries |
| `PRAGMATICS_HOST` | `pragmatics_api`                | Classifier service                    |
| `PRAGMATICS_PORT` | `8001`                          | Classifier port                       |

## Quick Test

```powershell
Invoke-RestMethod -Uri 'http://localhost:8000/api/memory/search' `
  -Method Post `
  -Body (@{ user_id='test'; query_text='project settings'; top_k=5 } | ConvertTo-Json) `
  -ContentType 'application/json'
```

## Project Structure

```
tools-api/memory/
├── main.py                 # FastAPI entry point
├── memory.filter.py        # Open-WebUI filter plugin
├── api/
│   └── memory.py           # REST endpoints
├── services/
│   ├── embedder.py         # SentenceTransformer embeddings
│   ├── qdrant_client.py    # Qdrant singleton client
│   ├── fact_extractor.py   # Regex-based fact extraction
│   └── summarizer.py       # Text summarization
├── utils/
│   └── schemas.py          # Pydantic models
└── static/
    ├── ai-plugin.json      # Plugin manifest
    └── openapi.yaml        # API spec
```

## License

MIT—do whatever you want with it.
