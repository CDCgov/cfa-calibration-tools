# Cloud Calibration

Cloud orchestration lives in `calibrationtools.cloud`. Model packages provide
model code, a Dockerfile, an MRP task config, an output contract, and a
model-facing cloud config.

For the bundled example model, the cloud config is:

```toml
packages/example_model/src/example_model/example_model.cloud_config.toml
```

Run the example cloud workflow with Make:

```bash
make setup-cloud
make cloud-run-auto
make cloud-list
```

The equivalent raw command is:

```bash
uv run --group cloudops python -m example_model.calibrate \
  --cloud \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml
```

## Cleanup

Cleanup prints the deletion plan before acting. Use the preview targets before
delete targets:

```bash
make cloud-cleanup-preview SESSION_ID=...
make cloud-cleanup-session SESSION_ID=...
make cloud-cleanup-user-preview
make cloud-cleanup-user-delete
```

The raw cleanup CLI is:

```bash
uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --list

uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --session-id 20260412010101-alice-testsha-ab12cd34ef56 \
  --dry-run

uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --all-sessions-for-user \
  --user "$(id -un)" \
  --dry-run
```

Makefile user cleanup defaults to the current shell user. Pass
`CLOUD_USER=other-user` only when you intend to inspect or clean that user's
project-scoped sessions.

## Config Shape

`packages/example_model/src/example_model/example_model.cloud_config.toml` uses
this structure:

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
max_parallel_output_downloads = 8

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
mode = "csv_column"
csv_value_column = "population"
csv_value_type = "int"

[cloud.auto_size]
probe = "mrp"
local_mrp_config_path = "example_model.mrp.toml"
```

`[cloud.image]` tells the shared runner how to build and upload the model
image. `task_mrp_config_path` is the MRP config path used by remote Batch tasks
inside that image. `[cloud.resources]` defines the project naming scope for
Batch and Blob resources. `[cloud.output]` lets the shared runner parse the
downloaded output without a model-local cloud wrapper.

`dispatch_buffer` controls how much extra sampler work the local runner admits
beyond `max_concurrent_simulations`. Azure Batch task submission still obeys
`max_concurrent_simulations`. `max_parallel_output_downloads` bounds concurrent
Blob downloads after successful Batch tasks complete.

`build_context` is resolved relative to the cloud config file. The bundled
example config lives under `packages/example_model/src/example_model`, so it
uses `../../../..` to point at the repository root.

## Auto Size

Use `--auto-size` to run one local probe before Azure provisioning:

```bash
uv run --group cloudops python -m example_model.calibrate \
  --cloud \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --auto-size
```

The shared MRP probe runs the configured local MRP file in a child process and
uses peak RSS to choose `task_slots_per_node`. Models that need a custom probe
can set `probe_module = "package.module"` in `[cloud.auto_size]`.

For models whose task input is rewritten by cloud payload transforms, use the
generic local-task probe:

```toml
[cloud.auto_size]
probe = "local_task"
local_mrp_config_path = "model.mrp.toml"
memory_scope = "process_tree"
```

`memory_scope = "self"` measures the probe child process. `process_tree` also
accounts for completed subprocesses via `resource.RUSAGE_CHILDREN`.

## Task Payloads And Outputs

Shared assets are uploaded once per session. Sources can be static
config-relative paths or paths selected from default inputs with an RFC 6901
JSON Pointer:

```toml
[cloud.shared_assets.reference]
source_json_pointer = "/epimodel.GlobalParams/reference_path"
blob_dir = "assets"
remote_path_var = "REFERENCE_PATH"
required = true
```

Task payload transforms are declarative JSON Pointer `set` operations:

```toml
[cloud.task_payload]
task_output_dir = "{output_mount_path}/custom-output/{run_id}"

[[cloud.task_payload.transforms]]
name = "set-reference"
op = "set"
path = "/epimodel.GlobalParams/reference_remote_path"
value = "{REFERENCE_PATH}"
on_missing = "create"
```

Scalar CSV output mode:

```toml
[cloud.output]
filename = "output.csv"
mode = "csv_column"
csv_value_column = "population"
csv_value_type = "int"
```

Structured CSV table mode:

```toml
[cloud.output]
filename = "aggregated_deaths_report.csv"
mode = "csv_table"
output_name = "aggregated_deaths_report"
orientation = "columns"
```

## New Model Checklist

1. Add a Dockerfile that installs the model package and copies the MRP config
   plus default inputs into the image.
2. Ensure the MRP config runs the model inside the container.
3. Add a `cloud_config.toml` with `[cloud]`, `[cloud.image]`,
   `[cloud.resources]`, `[cloud.output]`, and optional `[cloud.auto_size]`.
4. In calibration code, use
   `calibrationtools.cloud.runner.create_cloud_mrp_runner_from_config(...)`.
5. Use `python -m calibrationtools.cloud.cleanup --cloud-config ...` for
   cleanup.
