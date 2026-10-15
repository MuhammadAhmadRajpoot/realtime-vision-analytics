# CPU-only image for the API + dashboard.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RVA_HOST=0.0.0.0 \
    RVA_PORT=8103 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-deps . \
    && useradd --create-home app \
    && mkdir -p models data \
    && chown -R app:app /app
USER app

EXPOSE 8103
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8103/api/health')" || exit 1

# Fetch the demo clip (skipped if already present in the mounted volume), then serve.
CMD ["sh", "-c", "python scripts/download_demo.py && rva serve"]
