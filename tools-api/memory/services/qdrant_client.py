"""
Qdrant Client

My singleton client for Qdrant. Connects lazily on first use and keeps
the connection alive. The collection is 768-dim COSINE (matches my
all-mpnet-base-v2 embeddings).

Payload fields I store:
  - user_id, user_text, facts, facts_text, source_type, source_name

Env vars:
  - QDRANT_HOST (default: localhost)
  - QDRANT_PORT (default: 6333)
  - INDEX_NAME (default: user_memory_collection)
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

_client_instance: QdrantClient | None = None


def _client() -> QdrantClient:
    """Get the singleton Qdrant client. Creates and caches the connection on first call."""
    global _client_instance
    if _client_instance is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        print(f"[qdrant_client] Connecting to Qdrant at {host}:{port}")
        _client_instance = QdrantClient(host=host, port=port)
        print("[qdrant_client] Connection established")
    return _client_instance

def _ensure_collection() -> None:
    """
    Make sure the collection exists. Creates it if missing, but leaves
    existing collections alone to avoid data loss. Idempotent and safe to
    call repeatedly.
    """
    collection_name = os.getenv("INDEX_NAME", "user_memory_collection")
    print(f"[qdrant_client] _ensure_collection() for '{collection_name}'")
    client = _client()

    try:
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists and will be used as-is")
        # Don't validate or modify - just use what exists to avoid data loss
    except Exception as e:
        # Collection doesn't exist, create it
        print(f"Collection '{collection_name}' not found, creating: {e}")
        _create_collection(client, collection_name)


def _create_collection(client: QdrantClient, collection_name: str) -> None:
    """Create a new collection with 768-dim COSINE vectors."""
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Successfully created collection '{collection_name}' with 768 dimensions, COSINE distance")
    except Exception as create_err:
        print(f"Error creating collection: {create_err}")
        raise