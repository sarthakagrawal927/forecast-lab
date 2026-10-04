# One image for both the one-shot DB loader and the API (see docker-compose.yml).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first so code edits don't reinstall torch. Linux resolves the
# CPU-only torch wheel (pyproject [tool.uv.sources]).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY co2tx ./co2tx
COPY deploy ./deploy
COPY artifacts ./artifacts

RUN useradd --create-home app && mkdir -p /app/data && chown app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
