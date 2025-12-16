# Jeeves: Conversational Memory Service

This repository implements a FastAPI-based microservice for managing conversational memories, inspired by the classic "Ask Jeeves" query service. It captures conversation threads, embeds them using semantic vectors, and stores them in a Qdrant vector database for efficient retrieval. Designed to integrate with Open-WebUI via a filter plugin, it provides context injection to enhance AI chat continuity.

## Purpose and Functionality

Jeeves serves as a memory API that processes incoming message arrays from conversations. It evaluates whether the content merits storage—filtering out trivial exchanges—then generates embeddings using the `sentence-transformers/all-mpnet-base-v2` model. These are persisted in Qdrant for vector similarity searches. Upon query, it retrieves relevant past messages or summaries, ensuring your AI retains context without redundancy.

Core endpoints:
- `POST /api/memory/save`: Accepts a message array, checks for existing similar context, and stores if appropriate.
- `POST /api/memory/search`: Searches for matching message sets by query text, scoped to the user.
- `POST /api/memory/summaries`: Provides summarized overviews of relevant memories.
- `GET /api/memory/filter`: Supplies the Open-WebUI filter code for integration.

Summarization leverages `sshleifer/distilbart-cnn-12-6` (or HuggingFace Inference API), with deterministic IDs preventing duplicates. A pragmatic classifier service determines save-worthiness, defaulting to preservation on uncertainty.

## Technical Architecture

- **Framework**: FastAPI application on port 8000, with CORS support for local development.
- **Embeddings**: 768-dimensional normalized vectors from SentenceTransformer, accommodating text and image content.
- **Database**: Qdrant collection (default `user_memory_collection`) using COSINE similarity; direct REST API calls for performance.
- **Integration**: Open-WebUI filter appends retrieved context as `[Retrieved from memory]` blocks, prioritizing summaries.
- **Deployment**: Docker Compose setup with `memory_api` and `qdrant` services on `webtools_network`.

## Setup and Usage

Clone the repository and run the services using the root `docker-compose.yaml` for full stack (including Open-WebUI) or `docker/tools-api/memory/docker-compose.yaml` for memory service only:

```bash
git clone https://github.com/ianwesterfield/jeeves.git
cd jeeves
# For full stack with Open-WebUI:
docker compose -f docker-compose.yaml up -d --build
# For memory service only:
docker compose -f docker/tools-api/memory/docker-compose.yaml up -d --build
```

## Environment Variables

Configure the following environment variables as needed (referenced in the root `docker-compose.yaml` for Open-WebUI integration):

- `QDRANT_HOST`: Qdrant server hostname (default: `localhost`)
- `QDRANT_PORT`: Qdrant server port (default: `6333`)
- `INDEX_NAME`: Vector collection name (default: `user_memory_collection`)
- `SUMMARY_MODEL`: Summarization model (default: `sshleifer/distilbart-cnn-12-6`)
- `SUMMARY_DEVICE`: Device for summarization (default: `cpu`)
- `STORE_VERBATIM`: Toggle full message storage (default: `true`)
- `PRAGMATICS_HOST`: Pragmatics service hostname (default: `pragmatics_api`)
- `PRAGMATICS_PORT`: Pragmatics service port (default: `8001`)
- `WEBUI_ADMIN_USERNAME`: `$(cat webui_username.txt)`  # Reference to local file for admin username
- `WEBUI_ADMIN_PASSWORD`: `$(cat webui_password.txt)`  # Reference to local file for admin password
- `OLLAMA_BASE_URL`: Ollama service URL (default: `http://ollama:11434`)
- `MEMORY_API_URL`: Memory service URL (default: `http://memory_api:8000`)

Example search query:
```bash
curl -X POST http://localhost:8000/api/memory/search \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "query_text": "project details", "top_k": 5}'
```

## License

Licensed under the MIT License—free to use, modify, and distribute. No restrictions beyond the license terms.
