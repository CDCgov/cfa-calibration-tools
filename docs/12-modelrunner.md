[← Fingerprinting and Caching Strategy](11-fingerprinting-and-caching-strategy.md) | [TOC](README.md) | [Next: Seeds and RNG Discipline →](13-seeds-and-rng-discipline.md)

---

## 12. ModelRunnerProtocol
Much of this functionality is already provided through the `cfa-mrp` crate.

`ModelRunnerProtocol` is run asynchronously to support concurrent execution without blocking threads. It takes a typed model configuration and an `output_dir` path managed by `Context`, writes any output files there, and returns generated quantities. `Context` assigns a per-particle subdirectory (`{stage_id}/{particle_id}/`) so that all artifacts for a given run are co-located and addressable for later inspection.

```rust
#[async_trait::async_trait]
pub trait ModelRunnerProtocol: Send + Sync {
    type Config: Send + Sync + 'static;
    type Output: GeneratedQuantity;
    /// Optional intermediate model state type. Runners that do not support
    /// partial execution set this to `()`.
    type State:  Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static;

    /// Run the model to completion and return generated quantities.
    /// `output_dir` is a per-particle directory owned by `Context`.
    async fn run(
        &self,
        config:     &Self::Config,
        output_dir: &std::path::Path,
    ) -> Result<Self::Output, ModelRunError>;

    /// Optional: run the model for a partial duration and return an intermediate
    /// `ModelState` snapshot. Used when a DAG stage performs a short warm-up run
    /// whose state is resumed by a later stage for a full-length run.
    /// Default implementation returns `Err(ModelRunError::StateNotSupported)`.
    async fn run_partial(
        &self,
        config:     &Self::Config,
        output_dir: &std::path::Path,
    ) -> Result<(Self::Output, Self::State), ModelRunError> {
        let _ = (config, output_dir);
        Err(ModelRunError::StateNotSupported)
    }

    /// Optional: resume a previously checkpointed `ModelState` and run to completion.
    /// `Context` retrieves the state from `ArtifactStore` using the ref stored in
    /// `StageState::model_state_refs` for the parent stage's particle.
    /// Default implementation returns `Err(ModelRunError::StateNotSupported)`.
    async fn run_from_state(
        &self,
        state:      &Self::State,
        config:     &Self::Config,
        output_dir: &std::path::Path,
    ) -> Result<Self::Output, ModelRunError> {
        let _ = (state, config, output_dir);
        Err(ModelRunError::StateNotSupported)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ModelRunError {
    ExitCode(Option<i32>),
    Io(String),
    StateNotSupported,
}
```

### ArtifactStore

**Artifact layout:** `Context` assigns a per-particle output directory of the form `{artifact_root}/{stage_id}/{particle_id}/`. Every `config.json`, model output file, and `score_provenance.json` written during a run lands under that path, making each particle's full run record directly addressable. `ArtifactRef` URIs in `StageState::model_state_refs` and `score_provenances` point into this layout. This structure also means that a failed particle's config, partial outputs, and error record are co-located, which underpins `inspect_particle` ([§15](15-abc-rejection-sampling-execution.md)).


```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub fingerprint: Fingerprint,
    /// e.g. "file:///path/to/artifact.arrow" or "s3://bucket/key"
    pub uri:         String,
    pub data_type:  String,
}

pub trait ArtifactStore: Send + Sync {
    fn put(&self, fingerprint: &Fingerprint, data: &[serde_json::Value], data_type: &str)
        -> Result<ArtifactRef, std::io::Error>;
    fn get(&self, fingerprint: &Fingerprint)
        -> Result<Vec<serde_json::Value>, std::io::Error>;
}
```

Large `ModelOutput` objects and posterior tables are stored out-of-line via `ArtifactStore` and referenced in the manifest by `ArtifactRef`.

**Model state across the DAG:** When a runner supports `run_partial`, early DAG stages can checkpoint the intermediate `State` for each accepted particle, store it via `ArtifactStore`, and record the `ArtifactRef` in `StageState::model_state_refs`. A later stage with `ModelStatePolicy::ResumeFromParent` ([§3](03-simulation-system-and-dag-stages.md)) retrieves those states and calls `run_from_state`, extending the same run rather than starting fresh. Both stages share particle identity through `ParticleId`, establishing a common language across calibration stages: the same particle fingerprint appears in both `StageState` maps, and the `PerturbationType` entry in `ExperimentManifest::realized_kernels` documents what perturbation scale values were applied at each stage of this run (see [§6](06-perturbationkernel-and-density-convention.md)).


---

[← Fingerprinting and Caching Strategy](11-fingerprinting-and-caching-strategy.md) | [TOC](README.md) | [Next: Seeds and RNG Discipline →](13-seeds-and-rng-discipline.md)
