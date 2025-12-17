"""
Text Summarization Service

Provides text summarization using multiple backends with automatic fallback:
1. Local transformers pipeline (preferred, no API key needed)
2. HuggingFace Inference API (requires HF_TOKEN)
3. Heuristic fallback (simple sentence extraction)

The summarizer is used to create concise summaries of stored memories
for display in the UI and for creating more focused embeddings.

Environment Variables:
    SUMMARY_MODEL: Model ID for summarization (default: sshleifer/distilbart-cnn-12-6)
    SUMMARY_DEVICE: Device for local inference - "cpu" or GPU index (default: cpu)
    HF_TOKEN: HuggingFace API token for Inference API (optional)

Usage:
    from services.summarizer import summarize
    summary = summarize("Long text here...", max_words=60)
"""

import os
import re
from typing import Optional

_hf_client = None
_pipeline = None


def _init_backends() -> None:
    """
    Initialize summarization backends lazily.
    
    Attempts to initialize both the local transformers pipeline and
    the HuggingFace Inference API client. Either or both may fail
    gracefully if dependencies or credentials are missing.
    """
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
    Generate a concise summary of the input text.
    
    Tries multiple backends in order of preference:
    1. Local transformers pipeline (fastest, no network)
    2. HuggingFace Inference API (better quality, requires token)
    3. Heuristic fallback (first 2 sentences, trimmed)
    
    Args:
        text: The text to summarize. Can be any length.
        max_words: Target maximum words in output (approximate).
        
    Returns:
        A summarized version of the input text, or empty string if input is empty.
        
    Example:
        >>> summarize("My name is Ian. I work at Satcom Direct. I live in Florida.")
        "Personal info: Ian, works at Satcom Direct, lives in Florida."
    "
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