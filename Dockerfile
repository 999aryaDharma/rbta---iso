# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

# Image provenance build arguments
ARG GIT_SHA="unknown"
ARG BUILD_DATE="unknown"

# OCI standard labels for image traceability
LABEL org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/999aryaDharma/rbta---iso" \
      org.opencontainers.image.title="RBTA + Isolation Forest Service" \
      org.opencontainers.image.description="Rule-Based Temporal Aggregation and Isolation Forest for Wazuh SIEM Security Logs" \
      org.opencontainers.image.authors="Arya Dharma"

# Set security & operational Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RBTA_HOST=0.0.0.0 \
    RBTA_PORT=8000 \
    RBTA_MODEL_REGISTRY_DIR=/app/artifacts/models \
    RBTA_STATE_FILE=/app/data/runtime/state.json

# Create non-root application user and group (UID/GID 10001:10001)
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/false --no-create-home appuser

WORKDIR /app

# Install dependencies deterministically from pyproject.toml
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir .

# Copy application source code
COPY src/ /app/src/

# Create runtime mount points with appropriate ownership
RUN mkdir -p /app/data/runtime /app/artifacts/models && \
    chown -R appuser:appgroup /app

# Switch to non-root execution
USER appuser:appgroup

# Expose service port
EXPOSE 8000

# Non-root lightweight Python stdlib healthcheck
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)' || exit 1

# Start production server
CMD ["python", "-m", "uvicorn", "src.api.server:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
