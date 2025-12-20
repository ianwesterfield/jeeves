# ---- CPU-only pragmatics service ----
FROM python:3.11-slim

# Layer 1: System packages (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
  git curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 2: Copy requirements for caching
COPY requirements.txt .

# Layer 3: Install Python packages (cached if requirements.txt unchanged)
RUN pip install --no-cache-dir -r requirements.txt

# Layer 4: Copy application code (changes frequently)
COPY . /app

# Move trained model if present
RUN if [ -d /app/static/distilbert_memory ]; then mv /app/static/distilbert_memory /app/distilbert_memory; fi

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]