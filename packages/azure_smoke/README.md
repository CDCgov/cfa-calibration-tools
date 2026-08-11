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

## Build and publish the worker image

The current cloud-operations image helper delegates authentication to Docker. On a
Docker-compatible Podman host, authenticate Podman directly with an ACR access token, then
build and push the image. This reads only the ACR account/domain settings from `.env`.

```bash
ACR_ACCOUNT="$(uv run --package example-model-azure-smoke \
  python -c 'from dotenv import load_dotenv; import os; load_dotenv(); print(os.environ["AZURE_CONTAINER_REGISTRY_ACCOUNT"])')"
ACR_DOMAIN="$(uv run --package example-model-azure-smoke \
  python -c 'from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv("AZURE_CONTAINER_REGISTRY_DOMAIN", "azurecr.io"))')"
ACR_SERVER="$ACR_ACCOUNT.$ACR_DOMAIN"
ACR_TOKEN="$(az acr login --name "$ACR_ACCOUNT" --expose-token --query accessToken --output tsv)"

podman login "$ACR_SERVER" \
  --username 00000000-0000-0000-0000-000000000000 \
  --password "$ACR_TOKEN"
podman build -f packages/azure_smoke/Dockerfile \
  -t "$ACR_SERVER/calibrationtools-example-smoke:smoke" .
podman push "$ACR_SERVER/calibrationtools-example-smoke:smoke"
```

Use `docker` in place of `podman` on a Docker host. The token is short-lived; build and
push immediately after obtaining it.

## Run the smoke test

```bash
uv sync --all-packages
uv run --package example-model-azure-smoke example-model-azure-smoke \
  --base-name example-model-smoke
```

The command removes its Azure Batch job and pool by default. Pass `--keep-job` or
`--keep-pool` when Azure-side inspection is needed. The Blob container is retained by the
current executor and should be deleted after validation if it is no longer needed.
