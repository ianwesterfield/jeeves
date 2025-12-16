# ---- Small, CPU‑only image ---------------------------------
FROM python:3.11-slim

# System packages (only git & curl are needed for the layer cache)
RUN apt-get update && apt-get install -y --no-install-recommends \
  git curl && rm -rf /var/lib/apt/lists/*

# Python dependencies (FastAPI + HuggingFace + torch)
# Pin numpy<2 for torch 2.2.0 compatibility (compiled against NumPy 1.x)
RUN pip install --no-cache-dir \
  "numpy<2" \
  fastapi uvicorn[standard] pydantic \
  torch==2.2.0 \
  transformers==4.41.0 \
  sentencepiece==0.2.0 \
  scikit-learn

WORKDIR /app
COPY . /app

# If the trained model is stored under static/distilbert_memory, move it to /app/distilbert_memory
# so server can load it from MODEL_PATH="/app/distilbert_memory"
RUN if [ -d /app/static/distilbert_memory ]; then mv /app/static/distilbert_memory /app/distilbert_memory; fi

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]