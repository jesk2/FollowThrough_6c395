FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Build context is the **repo root** (see docker-compose.yml). We copy
# requirements.txt first so dependency installation is cached
# independently of source changes.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend service code.
COPY backend ./backend/

# ML package + pretrained artifacts. The artifacts (cf_model.pt,
# user_state.pt, ridge_probe.joblib, feature_stats.json) are loaded
# lazily by ml.inference.inference_api on first use; in production
# you may instead mount a volume so they can be hot-swapped without
# rebuilding the image.
COPY ml ./ml/

ENV PYTHONPATH=/workspace
EXPOSE 8000

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
