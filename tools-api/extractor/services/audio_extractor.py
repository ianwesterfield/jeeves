"""
Audio Extraction Service

Transcribes audio using OpenAI Whisper.
"""

import io
import tempfile
import os

# Lazy-loaded model
_model = None


def _load_model():
    """Load Whisper model on first use."""
    global _model
    
    if _model is not None:
        return
    
    print("[extractor] Loading Whisper model...")
    
    import whisper
    
    # Use 'base' for balance of speed/accuracy. Options: tiny, base, small, medium, large
    model_size = os.getenv("WHISPER_MODEL", "base")
    _model = whisper.load_model(model_size)
    
    print(f"[extractor] Whisper '{model_size}' loaded")


async def extract_from_audio(audio_data: bytes) -> str:
    """
    Transcribe audio to text.
    
    Args:
        audio_data: Raw audio bytes (wav, mp3, etc.)
    
    Returns:
        Transcribed text
    """
    _load_model()
    
    # Whisper needs a file path, so write to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_data)
        temp_path = f.name
    
    try:
        result = _model.transcribe(temp_path)
        return result["text"].strip()
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
