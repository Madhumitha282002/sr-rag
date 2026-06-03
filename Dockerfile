# Dockerfile
# ---------------------------------------------------------------------------
# Multi-stage build for SR-RAG.
# Stage 1 (builder): install dependencies into a venv
# Stage 2 (runtime): copy venv + source, run the app
#
# Build:  docker build -t sr-rag .
# API:    docker run -p 8000:8000 sr-rag api
# UI:     docker run -p 8501:8501 sr-rag ui
# ---------------------------------------------------------------------------

# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install into an isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/         ./src/
COPY api/         ./api/
COPY app/         ./app/
COPY configs/     ./configs/
COPY scripts/     ./scripts/
COPY .env.example ./.env.example

# Create directories that must exist at runtime
RUN mkdir -p logs data/raw_papers data/processed vector_store

# Expose both service ports
EXPOSE 8000 8501

# Entrypoint script selects which service to run
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]

# Default: run the API
CMD ["api"]