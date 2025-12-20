"""
Intent Classifier Service

Binary text classifier for determining if a message should trigger memory save.

Model:
  DistilBERT fine-tuned on conversation datasets
  Path: /app/distilbert_memory (trained checkpoint)
  Classes:
    0 = "other" (recall, query, casual chat)
    1 = "save" (remember/store this information)

Threshold: 0.70 confidence (conservative - prefer NOT saving when uncertain)
"""

import logging
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger("pragmatics.classifier")
log_level = (
    os.getenv("PRAGMATICS_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
)
logger.setLevel(getattr(logging, log_level, logging.INFO))


# ============================================================================
# Model Loading
# ============================================================================

MODEL_PATH = "/app/" + os.getenv("CLASSIFIER_MODEL", "distilbert_memory")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"[classifier] Loading model from {MODEL_PATH} on {DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    logger.info("[classifier] ✓ Model loaded successfully")
except Exception as e:
    logger.error(f"[classifier] Failed to load model: {e}")
    raise


# ============================================================================
# Classification Configuration
# ============================================================================

# Confidence threshold: only classify as save when confident
# Conservative approach: default to NOT saving when uncertain
SAVE_CONFIDENCE_THRESHOLD = float(os.getenv("SAVE_CONFIDENCE_THRESHOLD", "0.70"))
logger.info(f"[classifier] Threshold set to {SAVE_CONFIDENCE_THRESHOLD}")


# ============================================================================
# Public API
# ============================================================================

def classify_intent(text: str) -> Tuple[bool, float]:
    """
    Classify if a user message should trigger memory save.
    
    Args:
        text: User message text
    
    Returns:
        (is_save_request, confidence) tuple
        - is_save_request: True if confidence > threshold
        - confidence: Model confidence (0-1) for save class
    
    Classes:
        0 = other (recall, query, casual)
        1 = save (remember this)
    """
    text_preview = text if len(text) <= 200 else text[:197] + "..."
    logger.debug(f"[classify] Input (len={len(text)}): {text_preview}")
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    
    # Move to model device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
    
    # Extract save class probability
    save_confidence = probs[1]
    
    # Only classify as save if above threshold
    is_save = save_confidence >= SAVE_CONFIDENCE_THRESHOLD
    
    # Log result
    logger.debug(
        f"[classify] save_prob={save_confidence:.4f} threshold={SAVE_CONFIDENCE_THRESHOLD} "
        f"is_save={is_save} probs={[round(p, 4) for p in probs]}"
    )
    
    return is_save, save_confidence

