# Executor Service
# Polyglot code execution with sandboxing

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for language runtimes
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  nodejs \
  npm \
  git \
  && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8005

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
