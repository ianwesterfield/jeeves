"""
Intent Classifier Service

Multi-class intent classifier for Jeeves conversation routing.

Model:
  DistilBERT fine-tuned on conversation datasets
  Path: /app/distilbert_intent (4-class) or /app/distilbert_memory (binary fallback)
  
  4-class labels:
    0 = "casual" (greetings, general chat)
    1 = "save" (remember this information)
    2 = "recall" (what do you remember about X?)
    3 = "task" (workspace/code/execution requests)
    
  Binary fallback labels:
    0 = "other" (recall, query, casual)
    1 = "save" (remember this)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Any

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
# Intent Labels (4-class)
# ============================================================================

DEFAULT_INTENT_LABELS = {
    0: "casual",
    1: "save",
    2: "recall", 
    3: "task",
}

# ============================================================================
# Model Loading
# ============================================================================

MODEL_PATH = "/app/" + os.getenv("CLASSIFIER_MODEL", "distilbert_intent")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"[classifier] Loading model from {MODEL_PATH} on {DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Determine model type (4-class vs binary)
    num_labels = model.config.num_labels
    
    # Try to load custom label mapping
    label_file = Path(MODEL_PATH) / "intent_labels.json"
    if label_file.exists():
        with open(label_file) as f:
            label_data = json.load(f)
            INTENT_LABELS = {int(k): v for k, v in label_data.get("id2label", {}).items()}
    elif num_labels == 4:
        INTENT_LABELS = DEFAULT_INTENT_LABELS
    else:
        # Binary model fallback
        INTENT_LABELS = {0: "other", 1: "save"}
    
    IS_MULTICLASS = num_labels == 4
    
    logger.info(f"[classifier] ✓ Model loaded: {num_labels} classes, multiclass={IS_MULTICLASS}")
    logger.info(f"[classifier] Labels: {INTENT_LABELS}")
    
except Exception as e:
    logger.error(f"[classifier] Failed to load model: {e}")
    raise


# ============================================================================
# Configuration
# ============================================================================

CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.50"))
logger.info(f"[classifier] Confidence threshold: {CONFIDENCE_THRESHOLD}")


# ============================================================================
# Public API - Multi-class
# ============================================================================

def classify_intent_multiclass(text: str) -> Dict[str, Any]:
    """
    Classify user intent into one of 4 categories.
    
    Args:
        text: User message text
    
    Returns:
        {
            "intent": "casual"|"save"|"recall"|"task",
            "confidence": float (0-1),
            "all_probs": {"casual": 0.1, "save": 0.2, ...}
        }
    """
    text_preview = text if len(text) <= 200 else text[:197] + "..."
    logger.debug(f"[classify] Input: {text_preview}")
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
    
    # Find best class
    best_idx = probs.index(max(probs))
    best_intent = INTENT_LABELS.get(best_idx, "casual")
    best_confidence = probs[best_idx]
    
    # Build probability dict
    all_probs = {INTENT_LABELS.get(i, f"class_{i}"): p for i, p in enumerate(probs)}
    
    logger.debug(f"[classify] Result: {best_intent} ({best_confidence:.2f}) | {all_probs}")
    
    return {
        "intent": best_intent,
        "confidence": best_confidence,
        "all_probs": all_probs,
    }


# ============================================================================
# Public API - Binary (backward compatible)
# ============================================================================

def classify_intent(text: str) -> Tuple[bool, float]:
    """
    Binary classification: is this a save request?
    
    Backward compatible with existing code.
    
    Args:
        text: User message text
    
    Returns:
        (is_save_request, confidence) tuple
    """
    if IS_MULTICLASS:
        result = classify_intent_multiclass(text)
        is_save = result["intent"] == "save" and result["confidence"] >= CONFIDENCE_THRESHOLD
        save_conf = result["all_probs"].get("save", 0.0)
        return is_save, save_conf
    else:
        # Binary model path
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
        
        save_confidence = probs[1]
        is_save = save_confidence >= CONFIDENCE_THRESHOLD
        
        return is_save, save_confidence
