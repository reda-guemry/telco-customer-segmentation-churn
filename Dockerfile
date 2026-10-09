FROM python:3.13-slim

WORKDIR /usr/src/app

RUN 

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync 