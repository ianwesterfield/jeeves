# =============================================================================
# Jeeves Pragmatics API - Application Image
# =============================================================================
# Builds FROM pre-published base image (contains system deps + pip packages)
# Only copies application code - rebuilds in ~10 seconds
#
# To rebuild base: see Dockerfile.base
# =============================================================================

ARG BASE_REGISTRY=jeeves
FROM ${BASE_REGISTRY}/jeeves-pragmatics-base:latest

WORKDIR /app

# Only application code (the fast-changing layer)
COPY . /app

# Move trained models if present
# Priority: 4-class intent model > binary memory model
RUN if [ -d /app/static/distilbert_intent ]; then \
  mv /app/static/distilbert_intent /app/distilbert_intent; \
  fi && \
  if [ -d /app/static/distilbert_memory ]; then \
  mv /app/static/distilbert_memory /app/distilbert_memory; \
  fi

EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]