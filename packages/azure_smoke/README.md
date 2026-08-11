# Example model Azure smoke test

This independent workspace package runs the bundled `example-model` through one small
Azure Batch generation. It accepts four particles in one generation, with one attempt per
particle. Its purpose is to validate the worker image, task/result Blob mount, result
download, and requested cleanup—not to produce meaningful calibration results.

It does not modify or repurpose `packages/example_model`.

## Prerequisites

Run from the repository root on the Azure-capable machine. Use the existing approved
`cfa-cloudops` credential configuration for Azure Batch, Blob Storage, and Azure Container
Registry. The command reads its `.env` through `cfa-cloudops`; in particular it uses
`AZURE_CONTAINER_REGISTRY_ACCOUNT` and the optional
`AZURE_CONTAINER_REGISTRY_DOMAIN`. Do not replace these with a placeholder registry name.

## Run the smoke test

```bash
uv sync --all-packages
uv run --package example-model-azure-smoke example-model-azure-smoke \
  --base-name example-model-smoke \
  --build-image
```

`--build-image` uses the cloud-operations helper's configured Azure CLI authentication and
the included Dockerfile; it works with a Docker-compatible Podman setup. Omit it only when
the configured ACR already contains `calibrationtools-example-smoke:smoke`. The command
removes its Azure Batch job and pool by default. Pass `--keep-job` or `--keep-pool` when
Azure-side inspection is needed. The Blob container is retained by the current executor and
should be deleted after validation if it is no longer needed.
