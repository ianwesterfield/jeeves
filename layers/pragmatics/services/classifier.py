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
# Context-Aware Classification
# ============================================================================

# Short affirmative patterns that typically continue tasks
AFFIRMATIVE_PATTERNS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead", "do it",
    "please do", "proceed", "make it so", "sounds good", "that works",
    "perfect", "great", "go for it", "let's do it", "continue", "next",
    "all of them", "all", "both", "1", "2", "3", "commit", "save",
    "apply", "yes please", "yes go ahead", "ok do it", "sure thing",
}

# Context patterns that indicate assistant proposed an action
TASK_PROPOSAL_PATTERNS = [
    "would you like me to", "should i ", "do you want me to",
    "i can ", "i could ", "i'll ", "let me know if",
    "shall i", "want me to", "ready to", "proceed with",
    "here's the plan", "the changes would be",
    "files that need", "files to update", "files found",
    "which option", "option 1", "option 2", "choose",
    "select", "pick", "apply the", "make the change",
    "update the", "fix the", "create the", "delete the",
]


def classify_with_context(user_text: str, context: str = "") -> Dict[str, Any]:
    """
    Classify intent with conversation context.
    
    Uses a two-phase approach:
    1. Check if user message is a short affirmative AND context contains task proposal
    2. Fall back to ML classification if not
    
    Args:
        user_text: Current user message
        context: Recent conversation context (e.g., last assistant message)
    
    Returns:
        Same format as classify_intent_multiclass
    """
    # First, classify the user text alone
    standalone_result = classify_intent_multiclass(user_text)
    
    # If already high-confidence task, return immediately
    if standalone_result["intent"] == "task" and standalone_result["confidence"] >= 0.75:
        return standalone_result
    
    # If no context provided, return standalone result
    if not context or not context.strip():
        return standalone_result
    
    # Check for affirmative continuation pattern
    user_lower = user_text.lower().strip()
    context_lower = context.lower()
    
    # Is user message a short affirmative?
    is_short = len(user_text.strip()) < 50
    is_affirmative = any(
        user_lower == pattern or 
        user_lower.startswith(pattern + " ") or
        user_lower.startswith(pattern + ",") or
        user_lower.startswith(pattern + ".")
        for pattern in AFFIRMATIVE_PATTERNS
    )
    
    # Does context contain task proposal?
    has_task_proposal = any(pattern in context_lower for pattern in TASK_PROPOSAL_PATTERNS)
    
    # Upgrade to task if affirmative response to task proposal
    if is_short and is_affirmative and has_task_proposal:
        logger.info(
            f"[classify] Context upgrade: affirmative '{user_text[:30]}' + task proposal -> task"
        )
        return {
            "intent": "task",
            "confidence": 0.85,  # High confidence for clear pattern match
            "all_probs": {"task": 0.85, "casual": 0.10, "save": 0.025, "recall": 0.025},
        }
    
    # Check for follow-up action request pattern
    action_starters = ["can you", "could you", "would you", "please", "go through"]
    is_action_request = any(user_lower.startswith(starter) for starter in action_starters)
    
    if is_action_request:
        logger.info(f"[classify] Context upgrade: action request '{user_text[:30]}' -> task")
        return {
            "intent": "task",
            "confidence": 0.80,
            "all_probs": {"task": 0.80, "casual": 0.15, "save": 0.025, "recall": 0.025},
        }
    
    return standalone_result


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
