#!/usr/bin/env bash
# docker-entrypoint.sh
# Selects which service to start based on the CMD argument.
#
# Usage inside container:
#   /docker-entrypoint.sh api   → starts FastAPI on port 8000
#   /docker-entrypoint.sh ui    → starts Streamlit on port 8501

set -e

SERVICE="${1:-api}"

case "$SERVICE" in
  api)
    echo "Starting SR-RAG FastAPI on port 8000..."
    exec uvicorn api.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers 1 \
      --log-level info
    ;;
  ui)
    echo "Starting SR-RAG Streamlit on port 8501..."
    exec streamlit run app/streamlit_app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      --server.headless true \
      --server.fileWatcherType none
    ;;
  *)
    echo "Unknown service: $SERVICE"
    echo "Usage: docker run sr-rag [api|ui]"
    exit 1
    ;;
esac