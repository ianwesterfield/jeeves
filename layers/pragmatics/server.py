"""
Pragmatics Service - Intent Classification for Jeeves

Multi-class intent classification for conversation routing.

Endpoints:
  POST /api/pragmatics/classify - Full 4-class classification (recommended)
  POST /api/pragmatic - Binary save detection (backward compatible)

Model:
  DistilBERT fine-tuned on conversation intents.
  4-class labels: casual, save, recall, task
  Confidence threshold configurable via INTENT_CONFIDENCE_THRESHOLD
"""

import logging
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.classifier import classify_intent, classify_intent_multiclass, classify_with_context


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Pragmatics Service",
    description="Multi-class intent classification for Jeeves (casual/save/recall/task)",
    version="2.0.0",
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
    """Request to classify user text intent."""
    text: str = Field(..., min_length=1, max_length=5000)


class ClassifyResponse(BaseModel):
    """Binary classification result (backward compatible)."""
    is_save_request: bool
    confidence: float


class IntentResponse(BaseModel):
    """Full 4-class intent classification result."""
    intent: str = Field(..., description="One of: casual, save, recall, task")
    confidence: float = Field(..., description="Confidence score 0-1")
    all_probs: Optional[Dict[str, float]] = Field(None, description="All class probabilities")


# ============================================================================
# Endpoints
# ============================================================================

@app.post("/api/pragmatics/classify", response_model=IntentResponse)
async def classify_multiclass(request: ClassifyRequest) -> IntentResponse:
    """
    Classify user intent into one of 4 categories.
    
    Categories:
      - casual: Greetings, general chat, questions
      - save: User providing info to remember ("My name is Ian")
      - recall: User asking for stored info ("What's my name?")
      - task: Workspace/code/execution requests ("List files")
    
    Returns the most likely intent and confidence score.
    """
    start_time = time.time()
    text_preview = request.text[:100] if len(request.text) > 100 else request.text
    
    try:
        result = classify_intent_multiclass(request.text)
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"[classify] intent={result['intent']} conf={result['confidence']:.2f} "
            f"ms={duration_ms} text='{text_preview}'"
        )
        
        return IntentResponse(
            intent=result["intent"],
            confidence=round(result["confidence"], 4),
            all_probs={k: round(v, 4) for k, v in result["all_probs"].items()},
        )
    
    except Exception as exc:
        logger.error(f"[classify] Error: {exc} | text: {text_preview}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/pragmatic", response_model=ClassifyResponse)
async def classify_binary(request: ClassifyRequest) -> ClassifyResponse:
    """
    Binary classification: is this a save request?
    
    Backward compatible endpoint for existing integrations.
    For new code, use /api/pragmatics/classify instead.
    """
    start_time = time.time()
    text_preview = request.text[:100] if len(request.text) > 100 else request.text
    
    try:
        is_save, confidence = classify_intent(request.text)
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"[classify-binary] save={is_save} conf={confidence:.2f} "
            f"ms={duration_ms}"
        )
        
        return ClassifyResponse(
            is_save_request=is_save,
            confidence=round(confidence, 4),
        )
    
    except Exception as exc:
        logger.error(f"[classify-binary] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


class ContextClassifyRequest(BaseModel):
    """Request to classify user text with conversation context."""
    text: str = Field(..., min_length=1, max_length=5000)
    context: str = Field("", max_length=2000, description="Recent conversation context (e.g., last assistant message)")


@app.post("/api/pragmatics/classify-with-context", response_model=IntentResponse)
async def classify_with_conversation_context(request: ContextClassifyRequest) -> IntentResponse:
    """
    Classify user intent with conversation context.
    
    This endpoint helps disambiguate short follow-up responses like:
    - "yes" / "ok" / "go ahead" (affirmative continuations)
    - "do it" / "make that change" (action confirmations)
    - "can you change that?" (follow-up action requests)
    
    By providing the last assistant message as context, the classifier
    can better determine if these are task continuations.
    
    Args:
        text: Current user message
        context: Recent assistant message (what the user is responding to)
    
    Returns the most likely intent and confidence score.
    """
    start_time = time.time()
    text_preview = request.text[:50] if len(request.text) > 50 else request.text
    
    try:
        result = classify_with_context(request.text, request.context)
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"[classify-ctx] intent={result['intent']} conf={result['confidence']:.2f} "
            f"ms={duration_ms} text='{text_preview}'"
        )
        
        return IntentResponse(
            intent=result["intent"],
            confidence=round(result["confidence"], 4),
            all_probs={k: round(v, 4) for k, v in result["all_probs"].items()},
        )
    
    except Exception as exc:
        logger.error(f"[classify-ctx] Error: {exc} | text: {text_preview}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for orchestration."""
    return {"status": "healthy", "model": "4-class-intent"}

