"""
Image Vision Model Service

Generates detailed text descriptions of images using:
  - LLaVA-1.5-7B: 7B vision-language model (14GB full / 4GB quantized)
  - Florence-2: Microsoft vision model (4GB)

Models are selected via IMAGE_MODEL env var:
  - "llava-4bit" (default): 4-bit quantized LLaVA (~4GB VRAM, fast)
  - "llava": Full precision LLaVA (~14GB VRAM)
  - "florence": Florence-2-large fallback

Model loading is lazy (first request) and persistent (stays in memory).
"""

import io
import os
import re
from typing import Optional

import torch
from PIL import Image


# ============================================================================
# Global Model State
# ============================================================================

_model = None           # Loaded vision model
_processor = None       # Corresponding processor
_device = None          # Device (cuda or cpu)
_dtype = None           # Float precision (float16 or float32)
_model_type = None      # "llava", "llava-4bit", or "florence"


# ============================================================================
# Model Selection & Loading
# ============================================================================

def _get_model_type() -> str:
    """
    Determine model type from IMAGE_MODEL environment variable.
    
    Returns: "llava-4bit", "llava", or "florence"
    """
    model = os.environ.get("IMAGE_MODEL", "llava").lower().strip()
    
    if model in ("llava-4bit", "llava4bit", "4bit"):
        return "llava-4bit"
    elif model in ("llava", "llava-1.5", "llava-7b"):
        return "llava"
    elif model in ("florence", "florence-2", "florence2"):
        return "florence"
    
    return "llava"  # Default


def _load_model() -> None:
    """
    Load vision model on first use (lazy loading).
    
    Automatically selects appropriate loader based on IMAGE_MODEL env var.
    Sets global: _model, _processor, _device, _dtype, _model_type
    """
    global _model, _processor, _device, _dtype, _model_type
    
    if _model is not None:
        return  # Already loaded
    
    _model_type = _get_model_type()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.float16 if _device == "cuda" else torch.float32
    
    if _model_type == "llava-4bit":
        _load_llava_4bit()
    elif _model_type == "llava":
        _load_llava()
    else:
        _load_florence()


def _load_llava_4bit() -> None:
    """
    Load LLaVA-1.5-7B with 4-bit quantization.
    
    Memory: ~4GB VRAM
    Speed: ~2-3s per image
    Quality: Near full-precision
    
    Uses bitsandbytes for NF4 quantization with double quant.
    """
    global _model, _processor, _dtype
    
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    
    print("[extractor] Loading LLaVA-1.5-7B (4-bit quantized)...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )
    _dtype = torch.float16
    
    print("[extractor] ✓ LLaVA-1.5-7B (4-bit) ready on cuda:0 (~4GB VRAM)")


def _load_llava() -> None:
    """
    Load LLaVA-1.5-7B at full precision.
    
    Memory: ~14GB VRAM
    Speed: ~2-3s per image
    Quality: Full precision (slightly better than 4-bit)
    """
    global _model, _processor
    
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    
    print("[extractor] Loading LLaVA-1.5-7B (full precision)...")
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=_dtype,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    
    print(f"[extractor] ✓ LLaVA-1.5-7B ready on cuda:0 ({_dtype})")


def _load_florence() -> None:
    """
    Load Florence-2-large as fallback.
    
    Memory: ~4GB VRAM
    Speed: ~1-2s per image
    Quality: Good, compact
    """
    global _model, _processor
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    print("[extractor] Loading Florence-2-large...")
    
    model_id = "microsoft/Florence-2-large"
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=_dtype,
        attn_implementation="eager",
    ).to(_device)
    
    print(f"[extractor] ✓ Florence-2-large ready on {_device} ({_dtype})")


# ============================================================================
# Public API
# ============================================================================

async def extract_from_image(image_data: bytes, prompt: Optional[str] = None) -> str:
    """
    Generate text description of an image.
    
    Args:
        image_data: Raw image bytes (PNG, JPEG, etc.)
        prompt: Optional guided prompt for description
    
    Returns:
        Text description of the image content
    """
    _load_model()
    
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    if _model_type in ("llava", "llava-4bit"):
        return await _generate_llava(image, prompt)
    else:
        return await _generate_florence(image)


def load_model_at_startup() -> None:
    """
    Preload model at application startup.
    
    Eliminates ~8-12s latency from first image extraction request.
    Called during FastAPI startup event.
    """
    _load_model()


# ============================================================================
# Vision Model Inference
# ============================================================================

async def _generate_llava(image: Image.Image, prompt: Optional[str] = None) -> str:
    """
    Generate image description using LLaVA-1.5-7B.
    
    Args:
        image: PIL Image (RGB)
        prompt: Optional custom prompt
    
    Returns:
        Text description
    """
    
    # Build conversation in LLaVA format
    user_prompt = (
        prompt
        if prompt
        else (
            "Describe this image in detail. Include all visible objects, "
            "people, text, colors, and any notable features."
        )
    )
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    
    # Prepare inputs
    text_prompt = _processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = _processor(text=text_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
    
    # Decode and extract assistant response
    full_response = _processor.decode(output[0], skip_special_tokens=True)
    result = _extract_llava_response(full_response)
    
    return result if result else "[No description generated]"


def _extract_llava_response(full_response: str) -> str:
    """
    Extract just the assistant's response from LLaVA output.
    
    Handles various marker formats and cleans up artifacts.
    """
    
    # Try ASSISTANT: marker
    if "ASSISTANT:" in full_response:
        return full_response.split("ASSISTANT:")[-1].strip()
    
    # Try lowercase assistant marker
    lower_response = full_response.lower()
    if "assistant" in lower_response:
        idx = lower_response.rfind("assistant")
        response = full_response[idx + len("assistant") :]
        if response.startswith(":"):
            response = response[1:].strip()
        return response
    
    # No marker found, return full response
    return full_response


async def _generate_florence(image: Image.Image) -> str:
    """
    Generate image description using Florence-2.
    
    Args:
        image: PIL Image (RGB)
    
    Returns:
        Text description
    """
    
    task = "<MORE_DETAILED_CAPTION>"
    inputs = _processor(text=task, images=image, return_tensors="pt")
    
    # Move to device with correct dtypes
    inputs = {
        k: (
            v.to(_device, dtype=_dtype)
            if v.dtype == torch.float32
            else v.to(_device)
        )
        for k, v in inputs.items()
    }
    
    # Generate
    with torch.no_grad():
        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            use_cache=False,  # Florence-2 has issues with cache
        )
    
    # Decode
    generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Try post_process_generation if available
    if hasattr(_processor, "post_process_generation"):
        parsed = _processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )
        if isinstance(parsed, dict) and task in parsed:
            result = parsed[task]
            if result and len(result.strip()) > 10:
                return result
    
    # Fallback: manual cleanup
    clean = re.sub(r"</?s>|<[A-Z_]+>", "", generated_text).strip()
    return clean if clean else "[No description generated]"


def load_model_at_startup():
    """
    Preload image model at application startup.
    Eliminates latency from first image extraction request.
    """
    _load_model()
