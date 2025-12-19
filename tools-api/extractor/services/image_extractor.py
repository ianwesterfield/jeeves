"""
Image Extraction Service

Generates text descriptions of images using LLaVA-1.5-7B (or Florence-2 fallback).
Supports IMAGE_MODEL env var: "llava" (default), "llava-4bit", "florence"
"""

import io
import os
import torch
from PIL import Image
from typing import Optional

# Lazy-loaded model
_model = None
_processor = None
_device = None
_dtype = None
_model_type = None  # "llava", "llava-4bit", or "florence"


def _get_model_type() -> str:
    """Get model type from env var."""
    model = os.environ.get("IMAGE_MODEL", "llava").lower()
    if model in ("llava-4bit", "llava4bit", "4bit"):
        return "llava-4bit"
    elif model in ("llava", "llava-1.5", "llava-7b"):
        return "llava"
    elif model in ("florence", "florence-2", "florence2"):
        return "florence"
    return "llava"  # default


def _load_model():
    """Load vision model on first use."""
    global _model, _processor, _device, _dtype, _model_type
    
    if _model is not None:
        return
    
    _model_type = _get_model_type()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.float16 if _device == "cuda" else torch.float32
    
    if _model_type == "llava-4bit":
        _load_llava_4bit()
    elif _model_type == "llava":
        _load_llava()
    else:
        _load_florence()


def _load_llava_4bit():
    """Load LLaVA-1.5-7B with 4-bit quantization (uses ~4GB VRAM)."""
    global _model, _processor, _dtype
    
    print("[extractor] Loading LLaVA-1.5-7B (4-bit quantized)...")
    
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    
    # 4-bit quantization config - dramatically reduces memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda:0",  # Force to GPU
        torch_dtype=torch.float16
    )
    _dtype = torch.float16
    
    print(f"[extractor] LLaVA-1.5-7B (4-bit) loaded on cuda:0 (~4GB VRAM)")


def _load_llava():
    """Load LLaVA-1.5-7B model (full precision, ~14GB VRAM)."""
    global _model, _processor
    
    print("[extractor] Loading LLaVA-1.5-7B model...")
    
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    _processor = AutoProcessor.from_pretrained(model_id)
    _model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=_dtype,
        device_map="cuda:0",  # Force to GPU, not "auto"
        low_cpu_mem_usage=True
    )
    
    print(f"[extractor] LLaVA-1.5-7B loaded on cuda:0 ({_dtype})")


def _load_florence():
    """Load Florence-2-large model."""
    global _model, _processor
    
    print("[extractor] Loading Florence-2-large model...")
    
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    model_id = "microsoft/Florence-2-large"
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=_dtype,
        attn_implementation="eager"
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
    
    if _model_type in ("llava", "llava-4bit"):
        return await _generate_llava(image, prompt)
    else:
        return await _generate_florence(image)


async def _generate_llava(image: Image.Image, prompt: Optional[str] = None) -> str:
    """Generate description using LLaVA."""
    
    # LLaVA conversation format
    if prompt:
        user_prompt = prompt
    else:
        user_prompt = "Describe this image in detail. Include all visible objects, people, text, colors, and any notable features."
    
    # LLaVA expects this specific format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    text_prompt = _processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = _processor(text=text_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
    
    # Decode and extract only the assistant's response
    full_response = _processor.decode(output[0], skip_special_tokens=True)
    
    # Extract just the assistant response (after the prompt)
    if "ASSISTANT:" in full_response:
        result = full_response.split("ASSISTANT:")[-1].strip()
    elif "assistant" in full_response.lower():
        # Try to find assistant marker
        parts = full_response.lower().split("assistant")
        if len(parts) > 1:
            result = full_response[full_response.lower().rfind("assistant") + len("assistant"):].strip()
            # Clean up any remaining markers
            if result.startswith(":"):
                result = result[1:].strip()
        else:
            result = full_response
    else:
        result = full_response
    
    return result if result else "[No description generated]"


async def _generate_florence(image: Image.Image) -> str:
    """Generate description using Florence-2."""
    
    task = "<MORE_DETAILED_CAPTION>"
    inputs = _processor(text=task, images=image, return_tensors="pt")
    
    # Move to device and cast to correct dtype
    inputs = {k: v.to(_device, dtype=_dtype) if v.dtype == torch.float32 else v.to(_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            use_cache=False  # Florence-2 has issues with cache
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
