"""
Image Extraction Service

Generates text descriptions of images using Florence-2-large.
"""

import io
import torch
from PIL import Image
from typing import Optional

# Lazy-loaded model
_model = None
_processor = None
_device = None
_dtype = None


def _load_model():
    """Load Florence-2-large model on first use."""
    global _model, _processor, _device, _dtype
    
    if _model is not None:
        return
    
    print("[extractor] Loading Florence-2-large model...")
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.float16 if _device == "cuda" else torch.float32
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    model_id = "microsoft/Florence-2-large"
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=_dtype,
        attn_implementation="eager"  # Florence-2 doesn't support SDPA
    ).to(_device)
    
    print(f"[extractor] Florence-2-large loaded on {_device} ({_dtype})")


async def extract_from_image(image_data: bytes, prompt: Optional[str] = None) -> str:
    """
    Extract text description from image.
    
    Args:
        image_data: Raw image bytes
        prompt: Optional prompt for guided description
    
    Returns:
        Text description of the image
    """
    _load_model()
    
    # Load image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    task = "<MORE_DETAILED_CAPTION>"
    inputs = _processor(text=task, images=image, return_tensors="pt")
    
    # Move to device and cast to correct dtype
    inputs = {k: v.to(_device, dtype=_dtype) if v.dtype == torch.float32 else v.to(_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,  # Florence-2 has issues with beam search
            do_sample=False,
            early_stopping=False
        )
    
    # Decode output
    generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Use Florence-2's post_process_generation for proper decoding
    if hasattr(_processor, "post_process_generation"):
        parsed = _processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )
        if isinstance(parsed, dict) and task in parsed:
            result = parsed[task]
            if result and len(result.strip()) > 10:
                return result
    
    # Manual parsing fallback
    import re
    clean = re.sub(r'</?s>|<[A-Z_]+>', '', generated_text).strip()
    
    return clean if clean else "[No description generated]"
