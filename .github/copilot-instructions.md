# AI Coding Agent Instructions for this Repo

## Big Picture

- **Purpose:** FastAPI microservice that captures conversation "memories", embeds them with `sentence-transformers`, and stores/searches them in **Qdrant**. It exposes endpoints used by Open-WebUI via a filter plugin.
- **Key services:**
  - `memory_api` (this app) on port `8000`.
  - `qdrant` vector DB on `6333` (HTTP) and optional `5100` UI.
- **Data flow:** Open‑WebUI → `Filter.inlet` (`app/memory.filter.py`) → POST `/api/memory/save` → embed → Qdrant upsert → optional similar-context search → response includes `existing_context` injected into the next model call.

## Where Things Live

- **FastAPI app:** `docker/tools-api/app/main.py` mounts static plugin at `/.well-known` and includes router at `/api/memory`.
- **Router:** `docker/tools-api/app/api/memory.py` implements `/save` and `/search`.
- **Embeddings:** `docker/tools-api/app/services/embedder.py` uses `SentenceTransformer("all-mpnet-base-v2")`, normalized, 768‑dim.
- **Qdrant client:** `docker/tools-api/app/services/qdrant_client.py` lazy singleton; collection created as needed (COSINE, size 768).
- **Schemas:** `docker/tools-api/app/utils/schemas.py` (`Message`, `SaveRequest`, `SearchRequest`, `MemoryResult`).
- **Open‑WebUI Filter:** `docker/tools-api/app/memory.filter.py` sends to `http://memory_api:8000/api/memory/save` and injects context.
- **Plugin files:** `docker/tools-api/app/static/` contains `ai-plugin.json` and `openapi.yaml` served under `/.well-known`.

## Conventions & Patterns

- **Message content:** `content` may be `str` or `List[dict]` with items like `{type: "text", text: "..."}` or `{type: "image_url", ...}`. Embedding extracts text and tags images as `[image]`.
- **Deterministic IDs:** Points use `_make_uuid(user_id, content_hash)` producing 64‑bit int for Qdrant `id`.
- **Importance gating:** `_is_worth_saving(messages)` uses **KeyBERT** (`all-mpnet-base-v2`) and heuristics to skip casual/musing queries (e.g., "just curious"). It saves only when strong keywords or anchor terms (names, dates, contacts) are present.
- **Search first, save always (if worth):** `/save` runs REST search against Qdrant for existing context, then upserts current memory if worth saving. If context exists, `status` becomes `saved_with_context`.
- **Direct Qdrant REST for search:** Although the Python client exists, searches in `memory.py` use `requests` against Qdrant HTTP API for speed/control. Results are limited and gated by a high `score_threshold` to avoid injecting unrelated context.
- **CORS/Static:** `main.py` whitelists common local origins, and mounts plugin files at `/.well-known`.

## Build, Run, Debug

- **Docker Compose (Windows PowerShell):**
  ```powershell
  & 'C:\Program Files\Docker\Docker\resources\bin\docker.EXE' compose -f 'docker/tools-api/docker-compose.yaml' up -d --build
  ```
- **Environment:** `memory_api` relies on env vars: `QDRANT_HOST`, `QDRANT_PORT`, `INDEX_NAME` (default `user_memory_collection`). `EMBEDDING_PROVIDER` currently informational; embeddings are via `SentenceTransformer`.
- **Summarization:** Set `SUMMARY_MODEL` (default `sshleifer/distilbart-cnn-12-6`) and optionally `SUMMARY_DEVICE` (`cpu` or GPU index). If `HF_TOKEN` is set, summarization can use HuggingFace Inference API; otherwise, a local `transformers` pipeline is used. Use `STORE_VERBATIM=true|false` to control whether full message bodies are stored alongside summaries.
- **Volumes:** Qdrant persists data under `C:/docker-data/qdrant/storage` mapped to `/qdrant/storage`.
- **Local API test:** After containers are up, call:
  ```powershell
  Invoke-RestMethod -Uri 'http://localhost:8000/api/memory/search' -Method Post -Body (@{ user_id='anonymous'; query_text='project settings'; top_k=5 } | ConvertTo-Json) -ContentType 'application/json'
  ```
- **Hotspots for debugging:**
  - Keyword extraction and gating in `_is_worth_saving`.
  - Embedding text extraction in `embedder._extract_text_from_content`.
  - Qdrant REST search payloads in `/save` and `/search`.

## Integration Notes

- **Open‑WebUI filter contract:** `GET /api/memory/filter` returns the raw source of `memory.filter.py` so Open‑WebUI can parse tools. The filter calls `/api/memory/save` with `user_id`, `messages`, `model`.
- **Context injection:** `_inject_context(body, existing_context)` inserts a system marker `[Retrieved from memory]` and prefers injecting `summary` (as `[Summary] ...`) plus a few non‑system messages. This keeps context succinct and relevant.
- **Network:** Both services join `webtools_network` (external). In‑container hostnames: `memory_api` and `qdrant`.
- **Summaries endpoint:** `POST /api/memory/summaries` accepts `SearchRequest` and returns top‑k `{ summary, score }` items for a user to enable lightweight retrieval.

## Examples

- **Embedding combined conversation:**
  ```python
  vec = embed_messages([
    {"role":"system","content":"You are helpful"},
    {"role":"user","content":[{"type":"text","text":"Order status for #123"}]}
  ])
  ```
- **Qdrant REST search (by user):** filters on `{ key: 'user_id', match: { value: req.user_id } }` and sets `score_threshold` in `/save`.
- **Summarize and save:** payloads include `summary` generated from messages; set `STORE_VERBATIM=false` to store summaries only.

## Guardrails for Agents

- Do not change collection params unless migrating existing data; `_ensure_collection` purposely avoids modifying existing collections.
- Preserve `Message` shape and mixed `content` handling to avoid losing multi‑modal context.
- Keep embedding model and dim in sync across `embedder.py` and Qdrant `VectorParams(size=768)`.
- When adding endpoints, follow router style in `app/api/memory.py` and use schemas in `app/utils/schemas.py`.
