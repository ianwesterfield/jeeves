# Orchestrator v1.0 – FastAPI memory service
FROM python:3.11-slim

WORKDIR /app

# Install system deps for sentence‑transformers (if used)
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc libgomp1 && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all source files from memory/ subdirectory
COPY memory/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]