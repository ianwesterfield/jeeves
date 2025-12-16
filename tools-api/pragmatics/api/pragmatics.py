# pragmatics/server.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time

# Bring classifier logic from services
from services.classifier import _is_save_request

app = FastAPI()

# logging
logger = logging.getLogger("pragmatics.api")
if not logging.getLogger().handlers:
    logging.basicConfig()
api_log_level = "DEBUG"
logger.setLevel(getattr(logging, api_log_level, logging.INFO))


class Prompt(BaseModel):
    """Payload from the caller – only plain text is needed."""
    text: str

@app.post("/api/pragmatic")
async def classify(p: Prompt):
    """
    Input:  { "text": "<raw user prompt>" }
    Output: { "is_save_request": true|false, "confidence": float }
    
    Only returns is_save_request=True when confidence is high.
    Defaults to NOT saving when uncertain (conservative approach).
    """
    start = time.time()
    try:
        is_save, confidence = _is_save_request(p.text)
        res = {"is_save_request": is_save, "confidence": round(confidence, 4)}
        duration_ms = int((time.time() - start) * 1000)
        logger.debug("Classify result=%s duration_ms=%d text=%s", res, duration_ms, p.text[:200])
        return res
    except Exception as exc:
        logger.exception("Error while classifying text: %s", p.text[:200])
        raise HTTPException(status_code=500, detail=str(exc))