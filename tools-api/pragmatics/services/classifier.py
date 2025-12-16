import os
import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize logging
logger = logging.getLogger("pragmatics.classifier")
if not logging.getLogger().handlers:
    logging.basicConfig()
log_level = os.getenv("PRAGMATICS_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Initialize model and tokenizer from trained checkpoint
MODEL_PATH = "/app/" + os.getenv("CLASSIFIER_MODEL", "distilbert_memory")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Loading classifier from %s on device %s", MODEL_PATH, device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


# Confidence threshold: only save when model is confident it's a save request
# Default to NOT saving when uncertain (conservative approach)
SAVE_CONFIDENCE_THRESHOLD = float(os.getenv("SAVE_CONFIDENCE_THRESHOLD", "0.70"))


def _is_save_request(text: str) -> tuple[bool, float]:
    """
    Returns (is_save, confidence) tuple.
    Only returns is_save=True when confidence exceeds threshold.
    Labels:
        0 -> other (recall, query, casual)
        1 -> remember/save
    """
    short = text if len(text) <= 200 else text[:197] + "..."
    logger.debug("Classifying text (len=%d): %s", len(text), short)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    # move inputs to model device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]

    # probs[0] = probability of "other", probs[1] = probability of "save"
    save_prob = probs[1]
    
    # Only classify as save request if confidence exceeds threshold
    # Default to NOT saving when uncertain
    is_save = save_prob >= SAVE_CONFIDENCE_THRESHOLD
    
    logger.debug(
        "Prediction: save_prob=%.4f threshold=%.2f is_save=%s (probs=%s)",
        save_prob, SAVE_CONFIDENCE_THRESHOLD, is_save, [round(p, 4) for p in probs]
    )
    return is_save, save_prob

