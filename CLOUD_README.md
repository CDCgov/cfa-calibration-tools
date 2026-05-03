# Cloud Calibration

Cloud orchestration lives in `calibrationtools.cloud`. Model packages supply
only the model code, Dockerfile, task MRP config, output contract, and a
model-facing cloud config.

For the bundled example model, the main cloud config is:

```toml
packages/example_model/src/example_model/example_model.cloud_config.toml
```

Start with the Makefile cloud workflow:

```bash
make setup-cloud
make cloud-run-auto
make cloud-list
```

Cleanup targets make preview/delete intent explicit:

```bash
make cloud-cleanup-preview SESSION_ID=...
make cloud-cleanup-session SESSION_ID=...
make cloud-cleanup-user-preview
make cloud-cleanup-user-delete
```

Run `make help-cloud` for cloud-specific examples and cleanup safety notes.

The equivalent raw command for cloud calibration is:

```bash
uv run --group cloudops python -m example_model.calibrate \
  --cloud \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml
```

List or clean resources with the shared cleanup CLI:

```bash
uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --list
```

Cleanup prints the deletion plan before acting. Pass `--dry-run` to preview
without deleting anything.

The preferred Makefile cleanup commands separate preview from delete:

```bash
make cloud-cleanup-preview SESSION_ID=...
make cloud-cleanup-session SESSION_ID=...
make cloud-cleanup-user-preview
make cloud-cleanup-user-delete
```

The Makefile cleanup helpers intentionally default `CLOUD_USER` to the current
shell user. `make cloud-cleanup-user-preview` previews cleanup for your own
sessions, and `make cloud-cleanup-user-delete` deletes your own sessions after
printing the plan. Pass `CLOUD_USER=other-user` only when you intend to list or
clean that user's project-scoped sessions.

Preview or delete one session:

```bash
uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --session-id 20260412010101-alice-testsha-ab12cd34ef56 \
  --dry-run

uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --session-id 20260412010101-alice-testsha-ab12cd34ef56
```

Delete all sessions for one user after previewing the plan:

```bash
uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --all-sessions-for-user \
  --user "$(id -un)" \
  --dry-run

uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --all-sessions-for-user \
  --user "$(id -un)"
```

## Config Shape

`packages/example_model/src/example_model/example_model.cloud_config.toml` uses this structure:

```toml
[cloud]
keyvault = "cfa-predict"
vm_size = "large"
jobs_per_session = 1
task_slots_per_node = 50
pool_max_nodes = 5
task_timeout_minutes = 60
pool_ready_timeout_minutes = 20
pool_auto_scale_evaluation_interval_minutes = 5
dispatch_buffer = 1000
print_task_durations = false

[cloud.image]
local_image = "cfa-calibration-tools-example-model-python"
repository = "cfa-calibration-tools-example-model"
build_context = "../../../.."
dockerfile = "packages/example_model/Dockerfile"
task_mrp_config_path = "/app/example_model.mrp.toml"

[cloud.resources]
pool_prefix = "example-model-cloud"
job_prefix = "example-model-cloud"
input_container_prefix = "example-model-cloud-input"
output_container_prefix = "example-model-cloud-output"
logs_container_prefix = "example-model-cloud-logs"
input_mount_path = "/cloud-input"
output_mount_path = "/cloud-output"
logs_mount_path = "/cloud-logs"

[cloud.output]
filename = "output.csv"
csv_value_column = "population"
csv_value_type = "int"

[cloud.auto_size]
probe = "mrp"
local_mrp_config_path = "example_model.mrp.toml"
```

The shared runner synthesizes the local MRP executor config that dispatches
simulations through `calibrationtools.cloud.executor.execute_cloud_run`.
`[cloud.image]` tells the shared runner how to build and upload the model
image, and `task_mrp_config_path` is the MRP config path used by remote Batch
tasks inside that image. `[cloud.resources]` defines the project naming scope
for Batch and Blob resources. `[cloud.output]` lets the shared runner parse the
downloaded CSV without a model-local wrapper.

`dispatch_buffer` controls how much extra work the local runner submits to
Azure Batch beyond `max_concurrent_simulations`. Those extra submitted tasks sit
pending in Batch, which gives the autoscale formula enough queued work to scale
the pool out instead of waiting for Python to submit one small wave at a time.

`build_context` is resolved relative to the cloud config file and should point
at the Docker build context. The bundled example config lives under
`packages/example_model/src/example_model`, so it uses `../../../..` to point at
the repository root; that is required because its Dockerfile path is
`packages/example_model/Dockerfile` and the build copies files from the repo
root.

## Auto Size

Use `--auto-size` to run one local MRP probe before Azure provisioning:

```bash
uv run --group cloudops python -m example_model.calibrate \
  --cloud \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --auto-size
```

The shared probe runs:

```bash
mrp run packages/example_model/src/example_model/example_model.mrp.toml
```

in a child process and uses the child peak RSS to choose
`task_slots_per_node`. Models that need a custom probe can set
`probe_module = "package.module"` in `[cloud.auto_size]`.

## New Model Checklist

To make another model cloud-capable:

1. Add a Dockerfile that installs the model package and copies the MRP config
   plus default inputs into the image.
2. Ensure that MRP config runs the model inside the container. It can be the
   same config used for local inline MRP runs when its input paths are relative
   to the config file.
3. Add a `cloud_config.toml` with `[cloud]`, `[cloud.image]`,
   `[cloud.resources]`, `[cloud.output]`, and optional `[cloud.auto_size]`.
4. In calibration code, call
   `calibrationtools.cloud.runner.create_csv_cloud_mrp_runner_from_config(...)`
   or `create_cloud_mrp_runner_from_config(...)` for non-CSV output readers.
5. Use `python -m calibrationtools.cloud.cleanup --cloud-config ...` for
   cleanup.

Cloud configs must use the top-level `[cloud]` form shown above.
