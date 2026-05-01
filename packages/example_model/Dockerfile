FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CDC CA bundle source. This is only needed in the builder stage so
# `curl`/`pip` can reach CDC-hosted resources while we build wheels; the
# runtime image does not depend on this extra trust anchor. Pin to an
# immutable git ref (a commit SHA is strongly preferred over a branch
# name) and verify the downloaded bundle's sha256 so a compromised
# mirror cannot silently inject a different trust anchor. Both values
# are required — the build fails fast if ``CDC_CA_BUNDLE_SHA256`` is
# empty, because an unverified bundle defeats the purpose of pinning a
# trust anchor.
ARG CDC_CA_BUNDLE_REF=2e95c173bdfaeded5decf0ed2e14c16e14b23fa6
ARG CDC_CA_BUNDLE_SHA256=32e5beca392ca54cddbc38be3bd7eb8ae9584ef0b8dd89ab7487e17efa3d8772
RUN set -eu; \
    if [ -z "${CDC_CA_BUNDLE_SHA256}" ]; then \
        echo "ERROR: CDC_CA_BUNDLE_SHA256 must be set to the sha256 of the CDC CA bundle." >&2; \
        exit 1; \
    fi; \
    bundle_url="https://raw.githubusercontent.com/CDCgov/ocio-certificates/${CDC_CA_BUNDLE_REF}/data/min-cdc-bundle-ca.crt"; \
    bundle_path=/usr/local/share/ca-certificates/min-cdc-bundle-ca.crt; \
    echo "Fetching CDC CA bundle from ${bundle_url}"; \
    curl -fsSL --retry 3 --retry-delay 2 "${bundle_url}" -o "${bundle_path}"; \
    echo "${CDC_CA_BUNDLE_SHA256}  ${bundle_path}" | sha256sum -c -; \
    [ -s "${bundle_path}" ]; \
    update-ca-certificates

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

# Intentionally start from a clean runtime image instead of copying the
# builder-stage CA bundle forward. The extra CDC trust anchor is only
# required while fetching/installing build dependencies above.

# Default to a non-root runtime user for local container execution. Azure Batch
# tasks currently override this to root in the task container options because
# BlobFuse mounts are not reliably writable by the image user.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home --home-dir /home/app app \
    && mkdir -p /work /app \
    && chown -R app:app /work /app

WORKDIR /work
# Install the application and its dependencies
COPY --from=builder /wheels /wheels
COPY --chown=app:app packages/example_model/src/example_model/example_model.mrp.toml /app/example_model.mrp.toml
COPY --chown=app:app packages/example_model/src/example_model/defaults.json /app/defaults.json
RUN pip install --no-cache-dir --no-index --find-links /wheels calibrationtools example-model \
    && rm -rf /wheels

USER app:app
CMD ["example_model"]
