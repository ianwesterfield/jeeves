FROM python:3.11-slim

# Install system deps for image/audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg \
  libsndfile1 \
  git \
  && rm -rf /var/lib/apt/lists/*

# Install Python packages (including bitsandbytes for 4-bit quantization)
RUN pip install \
  "numpy<2" \
  fastapi \
  uvicorn[standard] \
  python-multipart \
  transformers \
  torch \
  torchvision \
  torchaudio \
  einops \
  timm \
  accelerate \
  bitsandbytes \
  openai-whisper \
  Pillow \
  pymupdf \
  python-magic

WORKDIR /app
COPY . /app

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
