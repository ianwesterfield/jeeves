# Orchestrator v1.0 – FastAPI memory service
FROM python:3.11-slim

# Layer 1: System dependencies (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc libgomp1 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 2: Copy requirements for caching
COPY requirements.txt .

# Layer 3: Install Python packages (cached if requirements.txt unchanged)
RUN pip install --no-cache-dir -r requirements.txt

# Layer 4: Copy application code (changes frequently)
COPY memory/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]