"""
Qdrant Client Service

Provides a singleton Qdrant client and collection management utilities.
The client lazily connects on first use and maintains a persistent connection.

Collection Schema:
    - Vectors: 768 dimensions (all-mpnet-base-v2 embeddings)
    - Distance: COSINE similarity
    - Payload fields:
        - user_id: str (indexed for filtering)
        - user_text: str (original message text)
        - facts: list[dict] (extracted structured facts)
        - facts_text: str (formatted facts for display)
        - source_type: str ("document", "prompt", "url", "image")
        - source_name: str (filename, URL, or text snippet)

Environment Variables:
    QDRANT_HOST: Qdrant server hostname (default: localhost)
    QDRANT_PORT: Qdrant server port (default: 6333)
    INDEX_NAME: Collection name (default: user_memory_collection)
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

_client_instance: QdrantClient | None = None


def _client() -> QdrantClient:
    """
    Get the singleton Qdrant client instance.
    
    Creates and caches a connection on first call. Subsequent calls
    return the cached client. Thread-safe for read operations.
    
    Returns:
        QdrantClient: Connected Qdrant client instance.
        
    Raises:
        ConnectionError: If unable to connect to Qdrant server.
    """
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
    Ensure the memory collection exists in Qdrant.
    
    Checks if the collection exists and creates it if not. Existing collections
    are used as-is without modification to prevent data loss.
    
    The collection is configured for:
    - 768-dimensional vectors (all-mpnet-base-v2 output)
    - COSINE distance metric (normalized embeddings)
    
    Note:
        This function is idempotent and safe to call multiple times.
        It will NOT recreate or modify existing collections.
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
    """
    Create a new Qdrant collection with the required schema.
    
    Args:
        client: The Qdrant client instance.
        collection_name: Name for the new collection.
        
    Raises:
        Exception: If collection creation fails.
    """
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