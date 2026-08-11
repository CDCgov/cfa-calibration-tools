# Example model Azure smoke test

This independent workspace package runs the bundled `example-model` through one small
Azure Batch generation. It accepts four particles in one generation, with one attempt per
particle. Its purpose is to validate the worker image, task/result Blob mount, result
download, and requested cleanup—not to produce meaningful calibration results.

It does not modify or repurpose `packages/example_model`.

## Prerequisites

Run from the repository root on the Azure-capable machine. Use the existing approved
`cfa-cloudops` credential configuration for Azure Batch and Blob Storage, and provide an
Azure Container Registry server through `AZURE_CONTAINER_REGISTRY_SERVER`,
`--registry-server`, or the existing `cfa-cloudops` variables
`AZURE_CONTAINER_REGISTRY_ACCOUNT` and `AZURE_CONTAINER_REGISTRY_DOMAIN`.

When your `.env` uses the latter variables, derive the server for the local container
commands without exposing any credential values:

```bash
export AZURE_CONTAINER_REGISTRY_SERVER="$(uv run --package example-model-azure-smoke \
  python -c \"from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ['AZURE_CONTAINER_REGISTRY_ACCOUNT'] + '.' + os.getenv('AZURE_CONTAINER_REGISTRY_DOMAIN', 'azurecr.io'))\")"
```

Build and push the included worker image with the normal Docker/ACR tools:

```bash
export AZURE_CONTAINER_REGISTRY_SERVER=myregistry.azurecr.io
docker build -f packages/azure_smoke/Dockerfile \
  -t "$AZURE_CONTAINER_REGISTRY_SERVER/calibrationtools-example-smoke:smoke" .
az acr login --name "${AZURE_CONTAINER_REGISTRY_SERVER%.azurecr.io}"
docker push "$AZURE_CONTAINER_REGISTRY_SERVER/calibrationtools-example-smoke:smoke"
```

Alternatively, `--build-image` asks `cfa-cloudops` to build and upload the Dockerfile. It
uses that tool's configured Azure CLI authentication, which is commonly managed identity.

## Run the smoke test

```bash
uv sync --all-packages
uv run --package example-model-azure-smoke example-model-azure-smoke \
  --registry-server "$AZURE_CONTAINER_REGISTRY_SERVER" \
  --base-name example-model-smoke
```

The command removes its Azure Batch job and pool by default. Pass `--keep-job` or
`--keep-pool` when Azure-side inspection is needed. The Blob container is retained by the
current executor and should be deleted after validation if it is no longer needed.
