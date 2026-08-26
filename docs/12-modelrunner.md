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

---

[← Fingerprinting and Caching Strategy](11-fingerprinting-and-caching-strategy.md) | [TOC](README.md) | [Next: Seeds and RNG Discipline →](13-seeds-and-rng-discipline.md)
