"""
Pragmatics Service - Intent Classification for Memory System

Classifies user messages to determine if they should trigger memory
save/recall operations.

Endpoints:
  POST /api/pragmatic - Classify text intent (save vs. other)

Model:
  DistilBERT-based binary classifier trained on conversation intents.
  Labels: 0=other (recall, query, casual), 1=save (remember this)
  Threshold: 0.70 confidence (conservative - prefer NOT saving when uncertain)
"""

import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.classifier import classify_intent


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Pragmatics Service",
    description="Intent classification for memory system (save vs. other)",
    version="1.0.0",
)


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("pragmatics.api")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
)


# ============================================================================
# Schemas
# ============================================================================

class ClassifyRequest(BaseModel):
    """
    Request to classify user text intent.
    
    Attributes:
        text: Raw user message text
    """
    text: str = Field(..., min_length=1, max_length=5000)


class ClassifyResponse(BaseModel):
    """
    Classification result.
    
    Attributes:
        is_save_request: True if should trigger memory save
        confidence: Model confidence (0-1), only save if > threshold
    """
    is_save_request: bool
    confidence: float


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/api/pragmatic", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify user text to determine memory action.
    
    Determines whether the message should trigger:
      - Memory save: "Remember my name is Ian"
      - Memory recall: "What's my name?"
      - No action: Casual chat, queries, etc.
    
    Returns is_save_request=True only when confidence > 0.70.
    Conservative approach: default to NOT saving when uncertain.
    """
    start_time = time.time()
    text_preview = request.text[:200] if len(request.text) > 200 else request.text
    
    try:
        is_save, confidence = classify_intent(request.text)
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"[classify] result={is_save} confidence={confidence:.4f} "
            f"duration_ms={duration_ms} text_len={len(request.text)}"
        )
        
        return ClassifyResponse(
            is_save_request=is_save,
            confidence=round(confidence, 4),
        )
    
    except Exception as exc:
        logger.error(f"[classify] Error: {str(exc)} | text: {text_preview}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for orchestration."""
    return {"status": "healthy"}
