FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY configs ./configs
COPY data/manifest.yaml ./data/manifest.yaml
COPY eval ./eval

# Non-root user
RUN useradd --create-home app && mkdir -p data/raw data/processed data/cache experiments \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "regrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
