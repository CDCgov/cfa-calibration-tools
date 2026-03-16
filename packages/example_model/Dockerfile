FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/CDCgov/ocio-certificates/refs/heads/main/data/min-cdc-bundle-ca.crt \
    -o /usr/local/share/ca-certificates/min-cdc-bundle-ca.crt \
    && update-ca-certificates

WORKDIR /app

# copy the source code to the container
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
COPY packages/example_model/pyproject.toml /app/packages/example_model/pyproject.toml
COPY packages/example_model/README.md /app/packages/example_model/README.md
COPY packages/example_model/src /app/packages/example_model/src

# build the application and its dependencies into wheel files
RUN pip wheel --no-cache-dir --wheel-dir /wheels /app /app/packages/example_model


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work
# Install the application and its dependencies
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels calibrationtools example-model \
    && rm -rf /wheels

ENTRYPOINT ["example_model"]
