# Design Overview

Last updated: 2026-04-29

This repository is a Python framework for running Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC) calibrations against model simulations. Its main design goal is to keep the calibration algorithm independent from the way a model is executed: a model can run in process, through an MRP config, in Docker, or through Azure Batch, while the sampler still talks to one small runner interface.

The diagrams for this overview are kept as Excalidraw source in [design-overview.excalidraw](design-overview.excalidraw).

## Repository Shape

The codebase is organized around a reusable calibration package, a reusable cloud execution layer, and a bundled example model that demonstrates the intended integration style.

| Area | Purpose |
| --- | --- |
| [src/calibrationtools](../src/calibrationtools) | Core ABC-SMC package: particles, priors, kernels, sampler orchestration, model evaluation, diagnostics, and MRP CSV runner support. |
| [src/calibrationtools/cloud](../src/calibrationtools/cloud) | Shared cloud-backed MRP execution machinery for Azure Batch, Blob Storage, ACR image upload, sizing, cleanup, and resource naming. |
| [packages/example_model](../packages/example_model) | Reference model package using a binomial branching process. It shows direct, MRP, Docker, and cloud calibration modes. |
| [example_model.mrp.toml](../example_model.mrp.toml) | Inline MRP config for local model execution. |
| [example_model.mrp.docker.toml](../example_model.mrp.docker.toml) | Docker-backed MRP config for local container execution. |
| [example_model.mrp.cloud.toml](../example_model.mrp.cloud.toml) | Cloud-backed MRP config whose runtime submits one simulation as an Azure Batch task. |
| [example_model.mrp.task.toml](../example_model.mrp.task.toml) | In-container task config used by Azure Batch nodes to run the actual model. |
| [tests](../tests) and [packages/example_model/tests](../packages/example_model/tests) | Regression coverage for sampler determinism, artifact staging, MRP parsing, cloud resource setup, concurrency, cancellation, cleanup, and sizing. |

## Main Design Ideas

The core package is built around a few separations:

- The sampler owns ABC-SMC control flow, not model execution details.
- The evaluator translates particles into model inputs and model outputs into distances.
- Priors and perturbation kernels are independent pluggable probability components.
- Generation runners isolate execution strategy from sampler configuration.
- Cloud execution is an adapter behind the same `simulate` or `simulate_async` model-runner shape.
- `SeedSequence` generator slots make serial and parallel execution reproducible enough to compare in tests.

## Core ABC-SMC Architecture

The public entry point is [src/calibrationtools/sampler.py](../src/calibrationtools/sampler.py). `ABCSampler` mostly acts as a facade over specialized collaborators:

| Component | Responsibility |
| --- | --- |
| `Particle` in [src/calibrationtools/particle.py](../src/calibrationtools/particle.py) | Dict-like parameter state for one proposal. |
| `ParticlePopulation` in [src/calibrationtools/particle_population.py](../src/calibrationtools/particle_population.py) | Particle list plus weights, normalization, total weight, size, and ESS. |
| `PriorDistribution` classes in [src/calibrationtools/prior_distribution.py](../src/calibrationtools/prior_distribution.py) | Sampling from priors and evaluating prior density. Built-ins include uniform, normal, lognormal, exponential, gamma, beta, seed, and independent composites. |
| `PerturbationKernel` classes in [src/calibrationtools/perturbation_kernel.py](../src/calibrationtools/perturbation_kernel.py) | Perturbing particles and evaluating transition probability. Built-ins include normal, uniform, multivariate normal, seed, and independent composites. |
| `VarianceAdapter` classes in [src/calibrationtools/variance_adapter.py](../src/calibrationtools/variance_adapter.py) | Adapting kernel variance or covariance from the accepted population after each generation. |
| `_ParticleUpdater` in [src/calibrationtools/particle_updater.py](../src/calibrationtools/particle_updater.py) | Sampling from the current population, perturbing within prior support, computing ABC-SMC weights, normalizing/adapting on population replacement. |
| `ParticleEvaluator` in [src/calibrationtools/particle_evaluator.py](../src/calibrationtools/particle_evaluator.py) | Calling user callbacks, staging optional input/output artifacts, invoking model runner sync or async methods, and computing distances. |
| `ParticlewiseGenerationRunner` in [src/calibrationtools/particlewise_generation_runner.py](../src/calibrationtools/particlewise_generation_runner.py) | The default generation engine. It fills one deterministic proposal slot per desired particle, retrying each slot until accepted or exhausted. |
| `BatchGenerationRunner` in [src/calibrationtools/batch_generation_runner.py](../src/calibrationtools/batch_generation_runner.py) | Alternative generation engine that proposes batches, estimates future batch sizes from the observed acceptance rate, and evaluates chunks serially or in a thread pool. |
| `SamplerRunState` in [src/calibrationtools/sampler_run_state.py](../src/calibrationtools/sampler_run_state.py) | Per-run counters, deterministic generator history, and optional previous-population archive. |
| `CalibrationResults` in [src/calibrationtools/calibration_results.py](../src/calibrationtools/calibration_results.py) | Final posterior, diagnostics, acceptance rates, credible intervals, point estimates, population archive, and posterior resampling. |
| `SamplerReporter` in [src/calibrationtools/sampler_reporting.py](../src/calibrationtools/sampler_reporting.py) | Rich progress bars and generation/run summaries. |

### Sampler Flow

At a high level, a sampler run repeats this loop for each tolerance value:

1. Pick the proposal method for the generation.
2. Generation 0 samples directly from the priors.
3. Later generations sample from the previous population, perturb with the configured kernel, and reject proposals outside prior support.
4. Convert each particle to model parameters with `particles_to_params`.
5. Run the model through `model_runner.simulate(...)` or `model_runner.simulate_async(...)`.
6. Convert model output to a scalar distance with `outputs_to_distance`.
7. Accept proposals whose distance is at or below the generation tolerance.
8. Assign weight `1.0` for generation 0, or compute the ABC-SMC importance weight from prior density over the weighted transition probability from the previous population.
9. Replace the sampler population with the accepted population, normalizing weights and adapting perturbation variance.
10. Build `CalibrationResults`, then reset the sampler internals back to the original perturbation kernel for another run.

The reset behavior is deliberate: the results keep a deep copy of the fitted updater, including the adapted kernel, while the sampler instance can be re-run reproducibly from the original configuration.

### Execution Modes

`ABCSampler.run(...)` uses the particlewise generation runner. It supports:

- `execution="serial"`: one proposal slot at a time.
- `execution="parallel"`: a `ThreadPoolExecutor` when the model runner is synchronous.
- Native async collection: if the runner sets `prefer_simulate_async = True` and defines `simulate_async`, the sampler uses async tasks plus a semaphore instead of thread-pool fanout.

`ABCSampler.run_parallel_batches(...)` uses the batch generation runner. It proposes groups of particles, evaluates them in chunks, and accepts them in deterministic chunk order until the population is full.

The tests pin important invariants: repeated runs with the same entropy are deterministic, serial and parallel runs match, unpickleable runners work because the parallel path uses threads rather than processes, and async-preferred runners obey the configured concurrency limit.

## Model Runner Interface

The sampler expects three integration points from the user:

| Hook | Meaning |
| --- | --- |
| `particles_to_params(particle, **kwargs)` | Converts an accepted or proposed particle into the model input dictionary. The example model overlays particle values onto default MRP inputs. |
| `outputs_to_distance(model_output, target_data)` | Converts model output into a scalar distance used by the tolerance check. |
| `model_runner` | Object or class with `simulate(params, *, input_path=None, output_dir=None, run_id=None)` and optionally `simulate_async(...)`. Async-preferred runners can also expose `simulate_from_sync(...)` when they set `allow_simulate_from_sync_bridge = True`. |

[src/calibrationtools/particle_evaluator.py](../src/calibrationtools/particle_evaluator.py) uses signature inspection before passing `input_path`, `output_dir`, and `run_id`, so simple in-process models do not need to accept staging arguments. Runners that need files, such as MRP and cloud runners, can accept them.

When `artifacts_dir` is configured for sampler runs, the evaluator writes JSON inputs under `input/generation-N` and creates matching output directories under `output/generation-N/RUN_ID`. Direct `ParticleEvaluator` calls without sampler generation metadata use `input/direct` and `output/direct/RUN_ID` with a generated `direct-*` run ID. In both cases it writes `result.json` with the parsed model output. These artifacts are useful for Docker/cloud workflows and post-run debugging.

## Local MRP Integration

[src/calibrationtools/mrp_csv_runner.py](../src/calibrationtools/mrp_csv_runner.py) provides `CSVOutputMRPRunner`, a reusable adapter for models that are run through `mrp` and emit a CSV column. It supports two modes:

- Inline payload mode: pass JSON input overrides directly to `mrp` and parse CSV from stdout.
- Staged file mode: pass an input JSON path and an output directory, then parse the expected CSV file from disk.

The example model binds this generic adapter as `ExampleModelMRPRunner` in [packages/example_model/src/example_model/mrp_runner.py](../packages/example_model/src/example_model/mrp_runner.py), reading the `population` column from `output.csv` as integers.

## Example Model Package

The bundled example is a Galton-Watson style binomial branching process in [packages/example_model/src/example_model/example_model.py](../packages/example_model/src/example_model/example_model.py). `Binom_BP_Model.simulate(...)` returns population counts by generation. `Binom_BP_Model.run()` wraps that simulation in the MRP model interface and writes `output.csv`.

[packages/example_model/src/example_model/calibrate.py](../packages/example_model/src/example_model/calibrate.py) is the reference orchestration script. It defines:

- default model inputs (`seed`, `max_gen`, `n`, `p`, `max_infect`),
- priors over `p` and `n`, plus a generated `seed` prior,
- an `IndependentKernels` perturbation setup using a multivariate normal kernel for fitted parameters and `SeedKernel` for the model seed,
- `AdaptMultivariateNormalVariance`,
- a two-generation tolerance schedule,
- direct in-process, Docker-backed MRP, explicit MRP config, and cloud-backed runner selection.

The important design point is that all four execution modes still feed the same `ABCSampler` constructor.

## Cloud Architecture

Cloud execution is split into model-specific wrapping code in the example package and reusable infrastructure in [src/calibrationtools/cloud](../src/calibrationtools/cloud).

### Cloud Config and Defaults

`CloudRuntimeSettings` in [src/calibrationtools/cloud/config.py](../src/calibrationtools/cloud/config.py) defines the shared cloud knobs: key vault, image names, resource prefixes, mount paths, VM size, `jobs_per_session` reusable job count, task slots per node, pool node cap, task timeout, pool readiness timeout, autoscale interval, dispatch buffer, and task timing output. The legacy `jobs_per_generation` name is still accepted as a deprecated alias.

The settings are intentionally layered:

1. Shared dataclass defaults provide generic defaults for optional fields.
2. The example model supplies project-specific defaults in [packages/example_model/src/example_model/cloud_utils.py](../packages/example_model/src/example_model/cloud_utils.py).
3. [example_model.mrp.cloud.toml](../example_model.mrp.cloud.toml) can override `[runtime.cloud]` values.
4. CLI flags can override total concurrency, task duration printing, artifact location, Docker build context, and auto-size behavior.
5. `--auto-size` can override `task_slots_per_node` for the created runner without editing TOML.

### Cloud Session Lifecycle

`ExampleModelCloudRunner(...)` in [packages/example_model/src/example_model/cloud_runner.py](../packages/example_model/src/example_model/cloud_runner.py) delegates to `create_cloud_mrp_runner(...)` in [src/calibrationtools/cloud/runner.py](../src/calibrationtools/cloud/runner.py). Construction validates local Docker context and concurrency before provisioning. Then `CloudMRPRunner`:

1. Loads cloud runtime settings.
2. Creates a CloudOps client from the configured key vault.
3. Reads the git short SHA and creates a session slug.
4. Builds the local Docker image and uploads it to ACR under the git SHA tag.
5. Creates input, output, and logs blob containers.
6. Creates one Azure Batch pool with BlobFuse mounts and autoscale settings.
7. Waits for the pool to become ready.
8. Creates a reusable set of Azure Batch jobs shared across all generations.
9. Stores all names and mount paths in `CloudSession` from [src/calibrationtools/cloud/session.py](../src/calibrationtools/cloud/session.py).

If setup fails after partial provisioning, the runner rolls resources back in dependency order: jobs, then pool, then containers. Startup interruptions such as `KeyboardInterrupt` preserve their original exception type after best-effort rollback.

### Cloud Simulation Dispatch

Cloud mode has a sync path and an async path:

- `CloudMRPRunner.simulate(...)` runs the configured cloud MRP runtime locally. The MRP callable is `example_model.cloud_mrp_executor:execute_cloud_run`, which delegates to the shared executor in [src/calibrationtools/cloud/executor.py](../src/calibrationtools/cloud/executor.py).
- `CloudMRPRunner.simulate_async(...)` bypasses local `mrp_run` and directly uses the same Azure operations from the runner controller. This is the path used by `ABCSampler` because cloud runners opt in with `prefer_simulate_async = True`.

The async runner maintains:

- an active-run registry keyed by `run_id`,
- a controller event loop in a daemon thread,
- an admission semaphore sized as `max_concurrent_simulations + dispatch_buffer`,
- an in-flight semaphore sized as `max_concurrent_simulations`,
- least-busy selection across the shared job set when more than one job is configured,
- per-run futures that resolve when output is downloaded and parsed,
- a low-frequency progress cache that lists Batch tasks by job rather than polling every task on every progress tick.

For each run, the async path uploads `{run_id}.json` into the input blob container, submits an Azure Batch task with the task-level MRP command, waits for task completion with retry behavior for transient Batch API failures, downloads `output.csv` atomically into the local artifact directory, parses the result, and resolves the sampler's awaited simulation call.

The runner also supports explicit cancellation. If a local coroutine is cancelled or `close()` is called, it marks the run cancelled and best-effort terminates the remote Batch task if it has already been submitted.

### Azure Task Shape

The task command is built in both [src/calibrationtools/cloud/executor.py](../src/calibrationtools/cloud/executor.py) and [src/calibrationtools/cloud/runner.py](../src/calibrationtools/cloud/runner.py) as a quoted `bash -lc` wrapper around:

```bash
mrp run /app/example_model.mrp.task.toml --input /cloud-input/.../RUN_ID.json --output-dir /cloud-output/.../RUN_ID
```

[example_model.mrp.task.toml](../example_model.mrp.task.toml) runs the model inline inside the container and writes `output.csv` to the mounted output path. Azure task setup in [src/calibrationtools/cloud/batch.py](../src/calibrationtools/cloud/batch.py) keeps task IDs under Azure limits, attaches the configured image and mounts, saves stdout/stderr into the logs container, and runs tasks as root inside the container so BlobFuse output and log mounts are writable.

### Cloud Data Layout

For a sampler-generated `run_id` such as `gen_2_particle_5_attempt_3`, cloud naming utilities in [src/calibrationtools/cloud/naming.py](../src/calibrationtools/cloud/naming.py) parse generation and particle numbers. `CloudSession` then derives paths like:

| Artifact | Location Pattern |
| --- | --- |
| Local input | `ARTIFACTS/input/generation-2/gen_2_particle_5_attempt_3.json` |
| Local output | `ARTIFACTS/output/generation-2/gen_2_particle_5_attempt_3/output.csv` |
| Remote input blob | `input/SESSION/generation-2/gen_2_particle_5_attempt_3/gen_2_particle_5_attempt_3.json` |
| Remote output blob | `output/SESSION/generation-2/gen_2_particle_5_attempt_3/output.csv` |
| Remote logs | `logs/SESSION/JOB/RUN_ID/...` |

This layout keeps retries distinct, makes cloud task names traceable to sampler proposal slots, and lets cleanup/listing tools infer session ownership from resource names.

### Auto-Size

Auto-size is a cloud-only guardrail implemented in [src/calibrationtools/cloud/auto_size.py](../src/calibrationtools/cloud/auto_size.py) and [src/calibrationtools/cloud/sizing.py](../src/calibrationtools/cloud/sizing.py). The example CLI runs one local probe simulation in a subprocess through [packages/example_model/src/example_model/cloud_auto_size.py](../packages/example_model/src/example_model/cloud_auto_size.py), reads child peak RSS, and computes:

```text
usable_vm_ram = vm_memory_bytes * (1 - reserve)
memory_task_slots = floor(usable_vm_ram / measured_task_peak_rss_bytes)
task_slots_per_node = min(memory_task_slots, vcpu_count * 4)
```

If the user did not set `--max-concurrent-simulations`, total cloud concurrency becomes `task_slots_per_node * pool_max_nodes`. Unknown raw VM SKUs fail before the memory probe, and a task larger than the usable RAM budget fails before cloud provisioning.

### Cleanup

Cloud cleanup is shared in [src/calibrationtools/cloud/cleanup.py](../src/calibrationtools/cloud/cleanup.py) and exposed by the example package in [packages/example_model/src/example_model/cloud_cleanup.py](../packages/example_model/src/example_model/cloud_cleanup.py). It can list project-scoped Batch pools/jobs, blob containers, and ACR tags; build a dry-run deletion plan for a session slug; and delete resources when `--yes` is passed.

Cleanup intentionally reads live Azure resource lists and filters by project prefixes plus exact session slug. That avoids deleting lookalike resources from other projects. ACR tag deletion is opt-in with `--image-tag` because many sessions from the same git SHA can share one uploaded image tag.

## Testing Strategy

The tests encode the project contracts more than they test incidental implementation details:

- Sampler tests verify repeatability, serial/parallel equality, generator history, artifact staging, verbose output suppression, async-preferred runner behavior, and cleanup after runner failures.
- Population, prior, kernel, updater, variance, and results tests verify the ABC-SMC mathematical building blocks.
- MRP tests verify CSV parsing, stdout preamble handling, staged input/output directories, error propagation, and numpy scalar conversion.
- Cloud tests verify resource naming, session layout, job selection, task command construction, output download, log excerpts, progress cache behavior, dispatch and in-flight concurrency, cancellation, duplicate run rejection, setup rollback, close idempotency, auto-size math, and cleanup filtering.

## Extension Points

To add a new model, implement one runner shape and keep the sampler unchanged:

1. Define `particles_to_params` and `outputs_to_distance` for the model domain.
2. Provide a direct `simulate` method, a `CSVOutputMRPRunner`, or a cloud runner factory.
3. Choose priors, kernels, variance adapter, tolerance values, and target data.
4. Construct `ABCSampler` with the chosen runner.
5. Optionally enable `artifacts_dir` for file-backed runners and reproducible debugging.

To add a new probability component:

- subclass `PriorDistribution` and implement `sample` plus `probability_density`,
- subclass `PerturbationKernel` and implement `perturb` plus `transition_probability`,
- subclass `VarianceAdapter` if the perturbation scale should learn from accepted populations.

To adapt cloud mode for another model:

1. Supply model-specific `CloudRuntimeSettings` defaults.
2. Provide a task-level MRP config inside the Docker image.
3. Provide an output parser callback for the model's artifact format.
4. Call `create_cloud_mrp_runner(...)` with the repository root, Dockerfile, settings loader, and output parser.
5. Expose cleanup commands with the same project-specific prefixes.

## Operational Notes

- `uv sync --all-packages --all-extras` prepares the full workspace for local development.
- `make check` runs lint, format check, type check, and tests.
- `make calibrate` runs the reference local calibration.
- `make calibrate-docker` builds the example image and runs calibration through Docker-backed MRP.
- `make calibrate-cloud` runs the cloud-backed calibration using CloudOps dependencies and Azure configuration.
- `make calibrate-cloud-auto` runs cloud calibration with memory auto-size and progress output.
- `make cloud-list`, `make cloud-cleanup-plan`, and `make cloud-cleanup-delete` wrap the cleanup CLI.

## Design Boundaries

The framework deliberately avoids owning the model's scientific meaning. It provides particle proposals, execution orchestration, and diagnostics, while the model integration supplies priors, target data, parameter conversion, and distance calculation.

The cloud layer is also intentionally model-agnostic below the output parser and settings defaults. It knows how to create resources, submit MRP tasks, move artifacts, and report failures; it does not know how to interpret a model beyond reading the configured output artifact.
