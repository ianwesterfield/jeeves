# =============================================================================
# Jeeves Extractor API - Application Image
# =============================================================================
# Builds FROM pre-published base image (contains system deps + pip packages)
# Only copies application code - rebuilds in ~10 seconds
#
# To rebuild base: see Dockerfile.base
# =============================================================================

ARG BASE_REGISTRY=jeeves
FROM ${BASE_REGISTRY}/jeeves-extractor-base:latest

WORKDIR /app

# Only application code (the fast-changing layer)
COPY . /app

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
