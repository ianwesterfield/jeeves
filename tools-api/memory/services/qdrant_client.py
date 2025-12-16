import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

_client_instance = None

def _client() -> QdrantClient:
    """Singleton Qdrant client."""
    global _client_instance
    if _client_instance is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        print(f"[qdrant_client] Connecting to Qdrant at {host}:{port}")
        _client_instance = QdrantClient(host=host, port=port)
        print("[qdrant_client] Connection established")
    return _client_instance

def _ensure_collection() -> None:
    """Create the collection if it does not already exist. Do NOT recreate existing collections."""
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
    """Helper to create collection with proper error handling. Only creates if it doesn't exist."""
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