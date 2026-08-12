# Azure Batch operations

`AzureBatchExecutor` is an opt-in execution backend for `ABCSampler`. It sends complete
particle-acceptance tasks to Azure Batch, while sampler state and result finalization stay
local. `CalibrationStudy` coordinates several fresh samplers concurrently; it does not
introduce another distributed backend.

## Installation and configuration

Install the Azure optional dependency only on the machine that will contact Azure:

```bash
uv sync --all-packages --all-extras
```

The executor lets `cfa-cloudops` read the existing approved `.env` configuration. In
particular, it resolves the registry from `AZURE_CONTAINER_REGISTRY_ACCOUNT` and optional
`AZURE_CONTAINER_REGISTRY_DOMAIN` (or an explicitly supplied
`AZURE_CONTAINER_REGISTRY_SERVER`). Do not copy credential values into source, command
history, task payloads, or logs.

Serial, threaded-local, and existing batch execution require neither this extra nor Azure
configuration. `ABCSampler.run(execution="azure_batch", cloud_executor=...)` is the
explicit Azure opt-in.

## Model worker image

The worker image must contain:

- the same `calibrationtools` version, including `calibrationtools.azure_batch_worker`;
- the model package and every importable callable referenced by the sampler task payload;
- `git` when the locked dependency graph includes a Git dependency.

Do not set a Docker `ENTRYPOINT` for this image. Azure Batch supplies the worker command;
an entrypoint turns that command into extra application arguments and prevents the worker
from starting.

The isolated `packages/azure_smoke` workspace provides a known-small image and validation
command without modifying `packages/example_model`. On a Podman host, build and publish it
from the repository root with the existing `.env` registry settings:

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

Use `docker` instead of `podman` where Docker is the configured container runtime. The ACR
access token is short-lived, so obtain it immediately before login/build/push.

## One sampler through Azure

After publishing the image, run the small smoke test:

```bash
uv run --package example-model-azure-smoke example-model-azure-smoke \
  --base-name example-model-smoke
```

It accepts four particles in one generation, with one attempt each. By default it deletes
the Batch job and pool; use `--keep-job` and/or `--keep-pool` only when Azure-side
inspection is necessary. The Blob container is retained and should be removed when no
longer needed.

## Concurrent calibration studies and live reporting

Create `CalibrationScenario` values and a factory that returns a *new* `ABCSampler` for
each scenario. Pass one base `AzureBatchExecutor` to `CalibrationStudy`; it clones the
executor per scenario so jobs, task/result names, mutable task state, and cloud clients
remain isolated. The study prepares one shared pool and mounted Blob container before
starting scenarios, avoiding repeated pool allocation.

```python
from calibrationtools import CalibrationScenario, CalibrationStudy

study = CalibrationStudy(
    scenarios=[
        CalibrationScenario("baseline", {"target_data": 5.0}),
        CalibrationScenario("higher-target", {"target_data": 6.0}),
    ],
    sampler_factory=make_fresh_sampler,
    cloud_executor=azure_executor,
    max_concurrent_scenarios=2,
    detail_log_dir="calibration-study-logs",
)
results_by_scenario = study.run()
```

The study owns the only Rich live display, avoiding interleaved output from concurrent
samplers. Its rows show each scenario's state, generation/total, particle work,
attempts, acceptance rate, elapsed time, ETA, and a concise failure note. Detailed events,
including Azure lifecycle and poll messages, are written as JSONL files under
`detail_log_dir`, one file per scenario.

## Final Azure validation gate

On the Azure-capable machine, after this candidate is reviewed and its worker image has
been published, run exactly:

```bash
uv run --package example-model-azure-smoke example-model-azure-smoke \
  --study \
  --base-name example-model-study \
  --max-concurrent-scenarios 2 \
  --detail-log-dir azure-smoke-study-logs
```

Expected outcome: one live two-row table (the `baseline` and `higher-target` scenarios),
one shared Azure pool, separate jobs and task/result blobs per scenario-generation, one
JSONL detail log per scenario, and a final
`Azure concurrent smoke study completed` message. The command deletes the two jobs and the
shared pool by default. Inspect the logs and retained Blob container before deleting any
resources you deliberately kept. This is an external integration gate; the regular test
suite uses fakes and does not contact Azure.
