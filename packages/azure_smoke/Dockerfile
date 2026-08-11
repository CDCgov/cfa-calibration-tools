FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY packages/example_model ./packages/example_model
COPY packages/azure_smoke ./packages/azure_smoke

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --locked --no-dev --package example-model-azure-smoke

ENV PATH="/app/.venv/bin:${PATH}"
