[← Seeds and RNG Discipline](13-seeds-and-rng-discipline.md) | [TOC](README.md) | [Next: ABC Rejection Sampling Execution →](15-abc-rejection-sampling-execution.md)

---

## 14. Error Propagation and Stage Resumability

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct ParticleFailure {
    pub particle_id: ParticleId,
    pub stage_id:    StageId,
    pub error:       RunStageError,
    pub attempt:     u32,
    pub retryable:   bool,
}

/// Runtime failure modes for a single particle across parse, run, and score steps.
/// Distinct from `ParticleError` (§7), which covers suffix resolution and parse errors.
#[derive(Debug, Serialize, Deserialize)]
pub enum RunStageError {
    ParseFailed(ParticleError),
    RunFailed   { exit_code: Option<i32>, message: String },
    ScoringFailed(String),
}

/// The transient result of executing a stage. Context absorbs this into
/// a StageState (see §4), which becomes the durable runtime record.
#[derive(Debug)]
pub struct CalibrationStageResult {
    pub stage_id:         StageId,
    /// Owned accepted population for this stage.
    pub accepted:         ParticlePopulation,
    /// Score values for each particle; absorbed into StageState.scores by Context.
    pub scores:           HashMap<ParticleId, HashMap<Fingerprint, ScoreValueType>>,
    /// Deduplicated failures: one entry per unique error variant,
    /// each carrying the list of particle IDs that produced it.
    pub failures:         Vec<(RunStageError, Vec<ParticleId>)>,
    pub budget_exhausted: bool,
}
```

**Behavior on budget exhaustion:** `CalibrationStageResult` is returned with `budget_exhausted = true` and whatever particles were accepted. The **next stage does not execute**; `Context` surfaces the partial result and the exhaustion diagnostics to the user.

**Behavior on particle failure:** Only unique `RunStageError` variants are reported. All particle IDs that produced the same error are grouped under a single report entry. The user is offered a re-run (same seed) or debug option. A run-level flag allows skipping all failures and presenting a summary report instead of halting.

### Checkpointing

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckpointPolicy {
    /// Write a checkpoint after every individual particle completes (highest granularity).
    Particle,
    /// Write a checkpoint after each batch of proposals (default).
    ParticleBatch,
    /// Write a checkpoint only when a full stage completes.
    Stage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StageCheckpoint {
    pub stage_id:               StageId,
    pub manifest_fingerprint:   Fingerprint,
    pub completed_particle_ids: HashSet<ParticleId>,
    pub current_population:     ParticlePopulation,
    /// Stage-owned score values at checkpoint time; mirrors StageState.scores.
    /// Keyed by particle ID and scorer fingerprint.
    pub scores:                 HashMap<ParticleId, HashMap<Fingerprint, ScoreValueType>>,
    pub failures:               Vec<ParticleFailure>,
    /// Snapshot of all four `Context` RNG states at the time this checkpoint was written.
    pub rng_snapshot:           RngSnapshot,
    pub timestamp:              chrono::DateTime<chrono::Utc>,
}

impl StageCheckpoint {
    pub fn validate_resume(&self, manifest_fp: &Fingerprint) -> Result<(), ResumeMismatch> {
        if &self.manifest_fingerprint != manifest_fp {
            return Err(ResumeMismatch {
                checkpoint: self.manifest_fingerprint.clone(),
                current:    manifest_fp.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ResumeMismatch {
    pub checkpoint: Fingerprint,
    pub current:    Fingerprint,
}

impl CheckpointExt for Context {
    pub fn resume_stage_from_checkpoint(&mut self, stage_id: &StageId, checkpoint: StageCheckpoint) -> Result<(), Error> {
        checkpoint.validate_resume(&self.manifest.id)?;

        let mut stage = context.stage_states.get_mut(stage_id)?
        stage_state.rngs = checkpoint.rng_snapshot;
        stage_state.scores.extend(checkpoint.scores.clone());
        Ok(())
        stage_state.accepted.entries.extend(checkpoint.current_population.entries.clone());
    }
}
```

### Particle Inspection and Targeted Rerun

Real-time diagnostics cover the in-flight view of a stage; post-failure inspection requires a stable, per-particle asset layout. Because `Context` writes each particle's run artifacts under `{artifact_root}/{stage_id}/{particle_id}/` ([§13](13-seeds-and-rng-discipline.md)), the full record for any particle is addressable independently of whether the stage completed.

```rust
/// A full inspection record for a single particle, assembled on demand by `Context::inspect_particle`.
/// Aggregates the particle's config, model outputs, score provenance, and any error
/// from the stored artifacts and live `StageState`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ParticleDebugBundle {
    pub particle_id:   ParticleId,
    pub stage_id:      StageId,
    pub particle:      FlatParticle,
    /// Parsed model config as a JSON value. `None` if parsing failed before the run.
    pub parsed_config: Option<serde_json::Value>,
    /// Artifact refs for model output files written during the run.
    /// `None` if the runner did not complete (e.g. exit-code failure).
    pub output_refs:   Option<Vec<ArtifactRef>>,
    /// Score provenance record, if the scorer implemented `score_with_provenance`.
    /// `None` if scoring did not reach the provenance step.
    pub provenance:    Option<ScoreProvenance>,
    /// The error that caused this particle to fail, if any.
    pub error:         Option<RunStageError>,
}

#[derive(Debug)]
pub enum InspectError {
    StageNotFound(StageId),
    ParticleNotFound(ParticleId),
    ArtifactLoadError(std::io::Error),
}
```

`Context` exposes two methods for targeted intervention after a failure:

```rust
impl Context {
    /// Assemble a `ParticleDebugBundle` for one particle from stored artifacts and
    /// the live `StageState`. Does not re-run anything.
    pub fn inspect_particle(
        &self,
        stage_id:    &StageId,
        particle_id: &ParticleId,
    ) -> Result<ParticleDebugBundle, InspectError>;

    /// Re-execute a single particle using its stored config and the current runner,
    /// bypassing the proposal and perturbation cycle. Writes new outputs into the
    /// particle's artifact directory and updates `StageState::scores` and
    /// `score_provenances` in place. Useful for rerunning a particle that failed
    /// due to a transient environment error without restarting the whole stage.
    pub async fn rerun_particle(
        &mut self,
        stage_id:    &StageId,
        particle_id: &ParticleId,
    ) -> Result<ScoreValueType, RunStageError>;
}
```

Per-particle artifact directories ([§13](13-seeds-and-rng-discipline.md)) are the load-bearing structure here. `inspect_particle` is cheap because it only reads the files already present under `{stage_id}/{particle_id}/`; it does not reload unrelated particles. `rerun_particle` writes back into the same directory so re-inspection after a rerun is consistent. Users who need to rerun a batch of failed particles can call `rerun_particle` for each `ParticleId` in `StageState::failures`, or use a convenience wrapper that iterates over the deduped failure list.

Diagnostics and predictive check stages reference `CalibrationStageResult` directly for their input data.


---

[← Seeds and RNG Discipline](13-seeds-and-rng-discipline.md) | [TOC](README.md) | [Next: ABC Rejection Sampling Execution →](15-abc-rejection-sampling-execution.md)
