# syntax=docker/dockerfile:1

# Single image used by every service (MLflow server, Streamlit dashboard and the
# optional training job). Python 3.14 matches `requires-python` in pyproject.toml.
FROM python:3.14-slim

# Bring in the uv binary (fast, lock-file based installs).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so this layer is cached when only the source changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the project (data, src, app, notebooks...).
COPY . .

# Ensure the environment is complete.
RUN uv sync --frozen --no-install-project --no-dev

EXPOSE 5000 8501

# Default to the dashboard; docker-compose overrides the command per service.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
