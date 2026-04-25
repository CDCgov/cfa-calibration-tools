# Cloud README

## Overview

This repo supports a cloud-backed calibration mode driven by
`example_model.mrp.cloud.toml`.

When you run cloud calibration, the workflow does all of the following:

- builds the example model Docker image locally
- tags the image with the current git short SHA
- uploads the image to Azure Container Registry (ACR)
- creates three blob containers for input, output, and logs
- creates one Azure Batch pool
- creates a reusable set of Azure Batch jobs for the full calibration run
- uploads one JSON input per particle evaluation
- submits one Azure Batch task per particle evaluation
- waits for that task to finish
- downloads `output.csv` back into the local artifact directory

The default cloud entrypoint is:

```bash
uv run --group cloudops python -m example_model.calibrate --cloud
```

## Prerequisites

- Sync the workspace with CloudOps dependencies:

```bash
uv sync --all-packages --group cloudops
```

- Make sure these tools are installed and available on `PATH`:
  `docker`, `git`, and `az`
- Make sure the Azure and CloudOps credentials are available through your
  environment, `.env`, or key vault wiring
- The default key vault in the cloud config is `cfa-predict`

Cloud mode expects these Azure settings to be configured:

- `AZURE_RESOURCE_GROUP_NAME`
- `AZURE_BATCH_ACCOUNT`
- `AZURE_SUBNET_ID`
- `AZURE_USER_ASSIGNED_IDENTITY`
- `AZURE_BLOB_STORAGE_ACCOUNT`
- `AZURE_CONTAINER_REGISTRY_ACCOUNT`

## Common Commands

### Run Cloud Calibration

```bash
uv run --group cloudops python -m example_model.calibrate --cloud
```

Useful flags:

- `--max-concurrent-simulations 25`
- `--auto-size`
- `--print-task-progress`
- `--print-task-durations`
- `--artifacts-dir ./artifacts`

Example:

```bash
uv run --group cloudops python -m example_model.calibrate --cloud \
  --auto-size \
  --max-concurrent-simulations 25 \
  --print-task-progress \
  --print-task-durations \
  --artifacts-dir ./artifacts
```


### List Cloud Resources

```bash
uv run --group cloudops python -m example_model.cloud_cleanup \
  --config example_model.mrp.cloud.toml \
  --list
```

To narrow the read-only listing to one session's Batch and Blob resources:

```bash
uv run --group cloudops python -m example_model.cloud_cleanup \
  --config example_model.mrp.cloud.toml \
  --list \
  --session-slug 20260412010101-testsha-ab12cd34ef56
```

To narrow further to one explicit image tag:

```bash
uv run --group cloudops python -m example_model.cloud_cleanup \
  --config example_model.mrp.cloud.toml \
  --list \
  --session-slug 20260412010101-testsha-ab12cd34ef56 \
  --image-tag testsha
```

### Dry-Run Cleanup Plan

```bash
uv run --group cloudops python -m example_model.cloud_cleanup \
  --config example_model.mrp.cloud.toml \
  --session-slug 20260412010101-testsha-ab12cd34ef56
```

### Delete One Cloud Session

```bash
uv run --group cloudops python -m example_model.cloud_cleanup \
  --config example_model.mrp.cloud.toml \
  --session-slug 20260412010101-testsha-ab12cd34ef56 \
  --yes
```

To also delete one specific ACR image tag for that session:

```bash
uv run --group cloudops python -m example_model.cloud_cleanup \
  --config example_model.mrp.cloud.toml \
  --session-slug 20260412010101-testsha-ab12cd34ef56 \
  --image-tag testsha \
  --yes
```

`--list` is always read-only, even if combined with `--yes`. When
`--session-slug` is provided, cleanup now reads the live Azure resource lists
and filters pools, jobs, and blob containers by that slug instead of
reconstructing exact names from the current config. ACR cleanup remains limited
to an explicit image tag instead of deleting the whole shared repository. In
list mode, each Batch, Blob, and ACR entry is annotated with the session slug
inferred for that resource so you can copy the exact value into a follow-up
cleanup command.

ACR image tags are intentionally keyed to the git short SHA, not the session
slug. That means multiple sessions from the same commit reuse the same uploaded
image tag. Use `--image-tag` cleanup only when no other live session from that
commit still needs to pull or restart tasks from that image.

## Cloud Config

The main config file is:

```text
example_model.mrp.cloud.toml
```

Important settings under `[runtime.cloud]`:

| Key | Purpose |
| --- | --- |
| `keyvault` | Key vault name used by `CloudClient` |
| `local_image` | Local Docker image name built before upload |
| `repository` | ACR repository name for the uploaded image |
| `task_mrp_config_path` | In-container path to the task-level MRP config |
| `pool_prefix` | Prefix for Azure Batch pool names |
| `job_prefix` | Prefix for Azure Batch job names |
| `input_container_prefix` | Prefix for input blob containers |
| `output_container_prefix` | Prefix for output blob containers |
| `logs_container_prefix` | Prefix for log blob containers |
| `input_mount_path` | Input blob mount path inside Batch nodes |
| `output_mount_path` | Output blob mount path inside Batch nodes |
| `logs_mount_path` | Logs blob mount path inside Batch nodes |
| `vm_size` | VM shorthand or raw Azure VM SKU |
| `jobs_per_session` | Number of reusable Azure Batch jobs shared across all generations (formerly `jobs_per_generation`, which is still accepted but deprecated) |
| `task_slots_per_node` | Number of Batch tasks each node may run concurrently |
| `pool_ready_timeout_minutes` | How long to wait for the Batch pool to become ready |
| `task_timeout_minutes` | Per-task timeout used while waiting for Batch completion |
| `dispatch_buffer` | Extra queued runs admitted beyond active submissions |
| `print_task_durations` | Prints upload, queue, run, and download timing summaries |

Current defaults in this repo:

- `keyvault = "cfa-predict"`
- `vm_size = "large"`
- `jobs_per_session = 1`
- `task_slots_per_node = 50`
- `pool_ready_timeout_minutes = 20`
- `task_timeout_minutes = 60`
- `dispatch_buffer = 1000`
- `print_task_durations = false`

In code, the example-model-specific cloud defaults now live in
`example_model.cloud_utils.DEFAULT_CLOUD_RUNTIME_SETTINGS`, layered on top of
the shared defaults built into
`calibrationtools.cloud.config.CloudRuntimeSettings`.

## VM Size Shorthands

The cloud code supports these shorthand values for `vm_size`:

| Shorthand | Azure SKU | vCPU | RAM |
| --- | --- | ---: | ---: |
| `xsmall` | `Standard_D2s_v3` | 2 | 8 GiB |
| `small` | `Standard_D4s_v3` | 4 | 16 GiB |
| `medium` | `Standard_D8s_v3` | 8 | 32 GiB |
| `large` | `Standard_D16s_v3` | 16 | 64 GiB |
| `xlarge` | `Standard_D32s_v3` | 32 | 128 GiB |

For normal cloud runs, you can also set a raw Azure SKU directly. Any `vm_size`
value outside the five shorthands is passed through unchanged.

Example:

```toml
[runtime.cloud]
vm_size = "Standard_D48s_v3"
```

## Auto-Size

Cloud calibration supports `--auto-size` as a RAM-based guardrail. The flag is
valid only with `--cloud`. Before the cloud runner is constructed, the CLI runs
one local example-model simulation in a fresh Python subprocess, measures the
child process peak RSS, and derives `task_slots_per_node` for the configured
`vm_size`.

The formula is:

```text
usable_vm_ram = vm_ram_bytes * 0.85
task_slots_per_node = floor(usable_vm_ram / measured_task_peak_rss_bytes)
```

Precedence:

- `--auto-size` overrides `[runtime.cloud].task_slots_per_node` for that run
  without editing the TOML file.
- If `--max-concurrent-simulations` is passed explicitly, it remains the total
  desired cloud capacity; auto-size changes only `task_slots_per_node`.
- If `--max-concurrent-simulations` is omitted, auto-size also sets total
  concurrency to the computed `task_slots_per_node`, preserving the default
  one-node cloud run shape.

Auto-size supports the five VM shorthands above and their documented Dsv3 SKUs
(`Standard_D2s_v3` through `Standard_D32s_v3`). Unknown raw SKUs fail fast with
a clear error because the first implementation uses the static RAM table in
this repo rather than Azure SKU discovery. If the measured task does not fit
within the 85% usable RAM budget for one node, the command fails before cloud
provisioning and asks you to choose a larger `vm_size` or disable
`--auto-size`.

## What A Cloud Run Creates

For each cloud calibration run, the runner creates:

- one local Docker image build
- one uploaded ACR image tag
- one input blob container
- one output blob container
- one logs blob container
- one Azure Batch pool
- `jobs_per_session` Azure Batch jobs

By default, the example calibration uses two generations, and the cloud config
defaults to `jobs_per_session = 1`, so a normal run creates:

- `1` pool
- `3` blob containers
- `1` Batch job

The default pool also allows `task_slots_per_node = 50`, so concurrency now
comes primarily from task slots on the pool nodes rather than from creating a
large number of shared Batch jobs.

Particle tasks are assigned to the shared job set in round-robin order using
the particle number in the `run_id`, and the same jobs are reused for later
generations.

## Cloud Runtime Notes

- Cloud mode requires a filesystem output directory
- The runner expects `run_id` values like `gen-1_particle-1`
- Each Batch task runs `mrp run` inside the container using
  `example_model.mrp.task.toml`
- `--print-task-progress` prints local generation progress as particle
  evaluations finish; in cloud mode each evaluation corresponds to one Batch task
- The progress line includes completed evaluations, accepted particles, the
  current number of in-progress tasks, and the running average task duration
  for the generation
- While tasks are still waiting on Azure Batch, the progress line can also
  include Batch task state counts plus pool node/allocation state
- During calibration, the sampler now keeps the cloud task window filled up to
  `--max-concurrent-simulations` instead of waiting for a whole batch wave to
  finish, and it best-effort cancels surplus in-flight tasks once a generation
  has enough accepted particles
- The cloud runner prints a startup summary with the created pool config and
  the number of reusable Batch jobs for the run; when `--auto-size` is enabled,
  the summary also includes measured peak RSS, VM RAM, reserve percentage,
  chosen task slots, and effective concurrency
- The cloud runner routes new tasks to the least-busy shared Batch job rather
  than assigning strictly by particle-number modulo
- The cloud executor uploads the per-particle input JSON to the input container
- The cloud executor downloads `output.csv` back to the local output directory
- Stdout and stderr logs are stored in the logs blob container
- If `--print-task-durations` is enabled, timing summaries are printed to stderr

## Operational Considerations

- Cloud runs are not lightweight. Every run rebuilds and uploads the image and
  creates fresh Azure resources.
- Cleanup is manual. The current runner marks itself closed locally, but it does
  not delete the pool, jobs, blob containers, or ACR image tag automatically.
- The cleanup command refuses to delete Batch or Blob resources unless an exact
  `--session-slug` is provided.
- `--max-concurrent-simulations` now drives the total desired task capacity for
  the fixed-size Batch pool in cloud mode. The dedicated node count is derived
  from that value and `task_slots_per_node`.
- `--auto-size` is a memory-only sizing helper. It does not account for CPU,
  IO, BlobFuse overhead, or model inputs that use more memory than the single
  local probe run.
- The async cloud controller admits up to
  `--max-concurrent-simulations + dispatch_buffer` queued runs while only
  `--max-concurrent-simulations` submissions execute at once.
- `jobs_per_session` must be at least `1`. The legacy key `jobs_per_generation` is still accepted as a deprecated alias and will be removed in a future release.
- `task_slots_per_node` must be at least `1`.
- `task_timeout_minutes` affects how long the executor waits for each Batch task.
- Only `output.csv` is explicitly downloaded back to the local output folder.
  If a task fails, the logs container is the first place to inspect.
- The pool currently uses fixed scaling and regional placement. Task slots per
  node are configurable in `example_model.mrp.cloud.toml`.
- Blobfuse-backed mounts are used for input, output, and logs, so storage and
  mount behavior matter for performance and debugging.
- The default cloud key vault, prefixes, and repository names are all project
  specific. If you copy this setup for another model, update those names first.

## Recommended Workflow

1. Sync dependencies with the CloudOps group.
2. Verify `docker`, `git`, and `az` are available.
3. Check or edit `example_model.mrp.cloud.toml`.
4. Run cloud calibration with `--cloud`.
5. Use `check_runner.py` if you need to inspect image, pool, or job state.
6. Use `example_model.cloud_cleanup` after the run to remove cloud resources.

## Using `calibrationtools` In A New Model With Cloud Execution

The `example_model` package is the reference integration. A new model that
wants to reuse the shared cloud pipeline needs to supply the same small set of
pieces. The framework (from `calibrationtools.cloud`) handles image upload,
pool creation, session slugs, Batch task dispatch, blob I/O, progress
reporting, and cleanup.

### What the framework already provides

- [CloudMRPRunner](src/calibrationtools/cloud/runner.py) — the `model_runner`
  that the `ABCSampler` drives. It exposes `simulate` and `simulate_async`,
  builds/uploads the Docker image once, creates the pool and Batch jobs, and
  routes each particle evaluation to a Batch task.
- [execute_cloud_run](src/calibrationtools/cloud/executor.py) — the in-process
  entrypoint that a per-particle `mrp run` call invokes to upload the input,
  submit a Batch task, wait for completion, and download `output.csv`.
- [CSVOutputMRPRunner](src/calibrationtools/mrp_csv_runner.py) — the shared
  local runner for MRP models that emit CSV output. It handles `mrp run`,
  staged input/output directories, stdout CSV extraction, and typed column
  parsing so model packages do not need to reimplement that logic.
- The shared cloud data models and helpers in
  [calibrationtools.cloud](src/calibrationtools/cloud/__init__.py) and its
  split modules:
  [config](src/calibrationtools/cloud/config.py),
  [session](src/calibrationtools/cloud/session.py),
  [naming](src/calibrationtools/cloud/naming.py),
  [tooling](src/calibrationtools/cloud/tooling.py), and
  [backend](src/calibrationtools/cloud/backend.py).
  Those modules own `CloudRuntimeSettings`, `CloudSession`,
  `CloudRunnerBackend`, `CloudExecutorBackend`, session-slug helpers, Azure
  tooling, and defaults for mount paths, VM sizes, and timeouts.
  [calibrationtools.cloud.utils](src/calibrationtools/cloud/utils.py) still
  exists as a compatibility facade for older imports, but it is no longer the
  primary integration surface.
- [create_cloud_mrp_runner](src/calibrationtools/cloud/runner.py) and
  [resolve_cloud_build_context](src/calibrationtools/cloud/runner.py) — shared
  construction helpers that resolve a model's Docker build context and create a
  configured `CloudMRPRunner` without forcing every model package to own a
  custom cloud-runner class.
- [cleanup module](src/calibrationtools/cloud/cleanup.py) — the shared engine
  behind `example_model.cloud_cleanup` for listing and deleting pools, jobs,
  blob containers, and ACR tags by session slug. The cleanup data models
  (`CleanupPlan`, `CleanupListing`, `CleanupResult`) also live there.

### Pieces a new model must add

For a hypothetical `my_model` workspace package, add the following files and
wire them in:

1. **A Dockerfile** that installs `calibrationtools` and your model, copies
   your `my_model.mrp.task.toml`, and sets the container entrypoint to your
   model CLI. See [packages/example_model/Dockerfile](packages/example_model/Dockerfile)
   for the two-stage wheel-based pattern.

2. **Three MRP configs at the repo root** (names are conventional):
   - `my_model.mrp.toml` — local/in-process run
   - `my_model.mrp.docker.toml` — local Docker run
   - `my_model.mrp.cloud.toml` — cloud run. It must set:
     - `[runtime]` to launch `python -m my_model.cloud_mrp_executor`
     - `[runtime.cloud]` block (see
       [example_model.mrp.cloud.toml](example_model.mrp.cloud.toml))
   - `my_model.mrp.task.toml` — config used **inside** the container to run
     one simulation. The image bakes this in at `task_mrp_config_path`.

3. **A thin `cloud_utils` module** that defines a model-specific
   `DEFAULT_CLOUD_RUNTIME_SETTINGS` object and exposes
   `load_cloud_runtime_settings` by delegating to the shared loader with those
   defaults. Mirror
   [packages/example_model/src/example_model/cloud_utils.py](packages/example_model/src/example_model/cloud_utils.py).
   Keep this module model-facing: it should own model defaults and helper
   aliases, but shared types such as `CloudRuntimeSettings`,
   `CloudRunnerBackend`, and `CloudExecutorBackend` should stay in
   `calibrationtools.cloud`.

4. **A local MRP runner binding** that either subclasses
   `CSVOutputMRPRunner` or otherwise configures the shared CSV runner with your
   output filename and value parser. See
   [packages/example_model/src/example_model/mrp_runner.py](packages/example_model/src/example_model/mrp_runner.py).

5. **A cloud-runner binding** that uses `create_cloud_mrp_runner(...)` with
   your model's default repo root, Dockerfile path, settings loader, and
   output reader. This can be a thin wrapper function; it does not need to be
   a custom subclass. See
   [packages/example_model/src/example_model/cloud_runner.py](packages/example_model/src/example_model/cloud_runner.py)
   for the example-model wrapper function.

6. **A `cloud_mrp_executor` entrypoint** inside the container image that calls
   `calibrationtools.cloud.executor.execute_cloud_run`. Mirror
   [packages/example_model/src/example_model/cloud_mrp_executor.py](packages/example_model/src/example_model/cloud_mrp_executor.py).

7. **A `cloud_cleanup` entrypoint** that wires the shared cleanup engine to
   your model's config file. Mirror
   [packages/example_model/src/example_model/cloud_cleanup.py](packages/example_model/src/example_model/cloud_cleanup.py).
   Keep it as a thin CLI wrapper; shared cleanup models such as
   `CleanupPlan`, `CleanupListing`, and `CleanupResult` should remain in
   `calibrationtools.cloud.cleanup`.

8. **A calibrate driver** that builds the `ABCSampler` with your priors,
   kernel, and distance function, and passes the cloud runner as
   `model_runner`. See
   [packages/example_model/src/example_model/calibrate.py](packages/example_model/src/example_model/calibrate.py).

9. **Package dependencies**: add `calibrationtools` (workspace source) and
   `cfa-mrp` to your package's `pyproject.toml`, and include the CloudOps
   extras (`cfa-cloudops`) in a dev group or the `cloud` optional dependency.

### Minimal calibrate wiring

```python
from pathlib import Path

from calibrationtools.perturbation_kernel import (
    IndependentKernels,
    MultivariateNormalKernel,
    SeedKernel,
)
from calibrationtools.sampler import ABCSampler
from calibrationtools.variance_adapter import AdaptMultivariateNormalVariance
from my_model import MyModelCloudRunner, DEFAULT_CLOUD_MRP_CONFIG_PATH

TOLERANCE_VALUES = [5.0, 1.0]

cloud_runner = MyModelCloudRunner(
    DEFAULT_CLOUD_MRP_CONFIG_PATH,
    generation_count=len(TOLERANCE_VALUES),
    max_concurrent_simulations=50,
    print_task_durations=False,
)

sampler = ABCSampler(
    generation_particle_count=500,
    tolerance_values=TOLERANCE_VALUES,
    priors=PRIORS,
    perturbation_kernel=IndependentKernels([
        MultivariateNormalKernel([p for p in PRIORS["priors"]]),
        SeedKernel("seed"),
    ]),
    variance_adapter=AdaptMultivariateNormalVariance(),
    particles_to_params=particles_to_params,
    outputs_to_distance=outputs_to_distance,
    target_data=target,
    model_runner=cloud_runner,
    max_concurrent_simulations=50,
    entropy=123,
    artifacts_dir=Path("."),
)
try:
    results = sampler.run(base_inputs=DEFAULT_INPUTS)
finally:
    cloud_runner.close()
```

The example package keeps a convenience wrapper named
`ExampleModelCloudRunner(...)`, but the important part is that the binding now
delegates to the shared `create_cloud_mrp_runner(...)` helper rather than
re-implementing cloud-runner construction inside the model package.

### How the pieces connect at runtime

```text
calibrate.py                        (local driver)
  └─ ABCSampler(model_runner=MyModelCloudRunner(...))
        └─ simulate_async(params, run_id=..., output_dir=...)
              └─ mrp run my_model.mrp.cloud.toml   (local mrp subprocess)
                    └─ python -m my_model.cloud_mrp_executor
                          └─ execute_cloud_run(run_json)
                                ├─ upload input JSON to input container
                                ├─ submit Azure Batch task (runs your image)
                                ├─ poll until completion
                                └─ download output.csv to local output_dir
```

The Batch task started inside Azure runs `mrp run` against the image-baked
`my_model.mrp.task.toml`, which invokes your model's CLI. Your model reads the
particle input from the mounted `input_mount_path` and writes `output.csv`
(or your `output_filename`) to `output_mount_path`. `stdout`/`stderr` land in
the logs container.

### Checklist before the first cloud run

- Image builds locally: `docker build -t my-model-python:latest -f packages/my_model/Dockerfile .`
- `my_model.mrp.task.toml` runs successfully inside the image against a local
  input JSON.
- `[runtime.cloud]` settings in `my_model.mrp.cloud.toml` use prefixes and
  repository names unique to your model (do not keep the `example-model-*`
  defaults).
- `AZURE_*` environment variables and key vault wiring are populated.
- A small smoke run (`--max-concurrent-simulations 2` with one short
  generation) completes and downloads an `output.csv` before scaling up.
