# server.py - Uvicorn entrypoint
# Re-exports the FastAPI app from api.pragmatics
from api.pragmatics import app

__all__ = ["app"]
