# Cloud Calibration

Cloud orchestration lives in `calibrationtools.cloud`. Model packages supply
only the model code, Dockerfile, task MRP config, output contract, and a
model-facing cloud config.

For the bundled example model, the main cloud config is:

```toml
packages/example_model/src/example_model/example_model.cloud_config.toml
```

Run cloud calibration with:

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

Delete one session after reviewing the plan:

```bash
uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --session-slug 20260412010101-testsha-ab12cd34ef56

uv run --group cloudops python -m calibrationtools.cloud.cleanup \
  --cloud-config packages/example_model/src/example_model/example_model.cloud_config.toml \
  --session-slug 20260412010101-testsha-ab12cd34ef56 \
  --yes
```

## Config Shape

`packages/example_model/src/example_model/example_model.cloud_config.toml` uses this structure:

```toml
[cloud]
keyvault = "cfa-predict"
simulation_mrp_config_path = "example_model.mrp.cloud.toml"
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
build_context = "../.."
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

`simulation_mrp_config_path` is used only by simulation runs, where the shared
runner still needs an MRP config to invoke the local cloud executor. Other
commands, such as cleanup and auto-size config loading, use the regular cloud
TOML directly. `[cloud.image]` tells the shared runner how to build and upload
the model image. `[cloud.resources]` defines the project naming scope for Batch
and Blob resources. `[cloud.output]` lets the shared runner parse the downloaded
CSV without a model-local wrapper.

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

Legacy MRP cloud configs with `[runtime.cloud]` are still accepted by the
shared config loader for migration, but new code should prefer the split
`cloud_config.toml` form.
