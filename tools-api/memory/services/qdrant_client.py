"""
Qdrant Client

My singleton client for Qdrant. Connects lazily on first use and keeps
the connection alive. The collection is 768-dim COSINE (matches my
all-mpnet-base-v2 embeddings).

Includes retry logic for transient connection failures.

Payload fields I store:
  - user_id, user_text, facts, facts_text, source_type, source_name

Env vars:
  - QDRANT_HOST (default: localhost)
  - QDRANT_PORT (default: 6333)
  - INDEX_NAME (default: user_memory_collection)
"""

import os
import time
from functools import wraps
from qdrant_client import QdrantClient
from qdrant_client.http import models

_client_instance: QdrantClient | None = None

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, doubles each retry


def with_retry(func):
    """Decorator that retries a function on connection errors with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        delay = RETRY_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                # Only retry on connection-related errors
                if any(x in err_str for x in ["connection", "timeout", "refused", "unavailable"]):
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise
        raise last_error
    return wrapper


@with_retry
def _client() -> QdrantClient:
    """Get the singleton Qdrant client. Creates and caches the connection on first call."""
    global _client_instance
    if _client_instance is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        _client_instance = QdrantClient(host=host, port=port, timeout=10)
    return _client_instance


@with_retry
def _ensure_collection() -> None:
    """
    Make sure the collection exists. Creates it if missing, but leaves
    existing collections alone to avoid data loss. Idempotent and safe to
    call repeatedly.
    """
    collection_name = os.getenv("INDEX_NAME", "user_memory_collection")
    client = _client()

    try:
        client.get_collection(collection_name=collection_name)
    except Exception:
        _create_collection(client, collection_name)


def _create_collection(client: QdrantClient, collection_name: str) -> None:
    """Create a new collection with 768-dim COSINE vectors."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"[qdrant] Created collection '{collection_name}'")