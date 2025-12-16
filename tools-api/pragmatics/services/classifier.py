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


def _is_save_request(text: str) -> bool:
    """
    Returns True when the classifier predicts the class "remember".
    Labels:
        0 -> other
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

    pred = int(torch.argmax(logits, dim=-1).cpu().item())
    logger.debug("Prediction: %s (probs=%s)", pred, [round(p, 4) for p in probs])
    return pred == 1

