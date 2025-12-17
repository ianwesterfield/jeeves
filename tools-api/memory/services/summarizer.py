"""
Summarizer

Generates summaries of stored memories. Tries local transformers first (no
network), then HuggingFace Inference API if I have HF_TOKEN, then falls back
to just grabbing the first couple sentences.

Env vars:
  - SUMMARY_MODEL (default: sshleifer/distilbart-cnn-12-6)
  - SUMMARY_DEVICE ("cpu" or GPU index)
  - HF_TOKEN (optional, for HF Inference API)
"""

import os
import re
from typing import Optional

_hf_client = None
_pipeline = None


def _init_backends() -> None:
    """Lazy init for summarization backends. Sets up local pipeline and/or HF client."""
    global _hf_client, _pipeline
    if _hf_client is None:
        try:
            from huggingface_hub import InferenceClient
            token = os.getenv("HF_TOKEN")
            model_id = os.getenv("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
            _hf_client = InferenceClient(model=model_id, token=token) if token else None
        except Exception as e:
            print(f"HF Inference client unavailable: {e}")
            _hf_client = None
    if _pipeline is None:
        try:
            from transformers import pipeline
            model_id = os.getenv("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
            device = 0 if os.getenv("SUMMARY_DEVICE", "cpu") != "cpu" else -1
            _pipeline = pipeline("summarization", model=model_id, device=device)
        except Exception as e:
            print(f"Local summarization pipeline unavailable: {e}")
            _pipeline = None

def summarize(text: str, max_words: int = 60) -> str:
    """
    Summarize text. Tries local pipeline first, then HF API, then just grabs
    the first couple sentences if all else fails.
    """
    print(f"[summarizer] summarize() called, text length={len(text or '')}, max_words={max_words}")
    text = (text or "").strip()
    if not text:
        print("[summarizer] Empty text, returning empty string")
        return ""

    _init_backends()

    # Prefer local pipeline when available
    if _pipeline is not None:
        print("[summarizer] Using local transformers pipeline")
        try:
            # Control length roughly with word count, map to tokens
            max_tokens = max(32, min(128, int(max_words * 1.6)))
            print(f"[summarizer] Running pipeline with max_new_tokens={max_tokens}")
            result = _pipeline(text, max_new_tokens=max_tokens, do_sample=False)
            if isinstance(result, list) and result:
                out = result[0].get("summary_text", "")
                if out:
                    print(f"[summarizer] Pipeline returned summary: {out[:80]}..." if len(out) > 80 else f"[summarizer] Pipeline returned summary: {out}")
                    return out.strip()
        except Exception as e:
            print(f"[summarizer] Local summarization failed: {e}")

    # Try HF Inference API
    if _hf_client is not None:
        print("[summarizer] Using HF Inference API")
        try:
            prompt = (
                f"Summarize the key personal facts, dates, names, and preferences succinctly (<= {max_words} words).\n\n"
                f"Text:\n{text}\n\nSummary:"
            )
            out = _hf_client.text_generation(prompt=prompt, max_new_tokens=128, temperature=0.1)
            print(f"[summarizer] HF API returned summary: {(out or '')[:80]}")
            return (out or "").strip()
        except Exception as e:
            print(f"[summarizer] HF summarization failed: {e}")

    # Fallback: take first sentences and trim to max_words
    print("[summarizer] Using heuristic fallback")
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:2])
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "…"
    print(f"[summarizer] Heuristic summary: {summary[:80]}..." if len(summary) > 80 else f"[summarizer] Heuristic summary: {summary}")
    return summary