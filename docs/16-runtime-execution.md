[← ABC Rejection Sampling Execution](15-abc-rejection-sampling-execution.md) | [TOC](README.md) | [Next: End-to-end example →](17-end-to-end-example.md)

---

## 16. Runtime Execution

`CalibrationManifest` is the static ruleset for a calibration — priors, stages, perturbation strategy, scorers, targets, and counterfactuals are all fixed at build time and unchanged across every run. `ExperimentManifest` owns the full runtime record for a single execution: seed, timing, stage status, checkpoints, posterior artifact refs, simulation artifact refs, and diagnostics. All behavior lives in `Context`, which builds and then reads `CalibrationManifest`, writing `ExperimentManifest` in real time.

Simulation, predictive-check, and diagnostics operations are runtime steps executed by `Context` and recorded in `ExperimentManifest`. These operations are not nodes in the `CalibrationDag`.

---

### 16.1 RunBuilder

`run_calibration()` returns a `RunBuilder`. The seed is the only input that varies between runs of the same `CalibrationManifest`; all structural parameters are fixed at build time. `.run()` executes the DAG in topological order, updating `Context::stage_states` and writing `ExperimentManifest` fields in real time.

```rust
pub trait ContextRunExt {
    fn run_calibration(&mut self) -> RunBuilder<'_>;
}

impl ContextRunExt for Context { /* ... */ }

/// Returned by `ctx.run_calibration()`. Borrows ctx mutably.
pub struct RunBuilder<'ctx> { /* opaque */ }

impl<'ctx> RunBuilder<'ctx> {
    /// Set the base RNG seed for the entire run. Supplied by Context otherwise.
    pub fn seed_override(self, seed: u64) -> Self;
    /// Restrict execution to a subset of leaf `StageId`s. Default runs all nodes.
    pub fn stages(self, ids: impl IntoIterator<Item = StageId>) -> Self;
    /// Resume from an existing `ExperimentManifest`, skipping already-completed stages.
    /// Validated via `StageCheckpoint::validate_resume` (§15) before execution.
    pub fn resume_from(self, manifest: ExperimentManifest) -> Self;
    /// Execute the calibration DAG and return the completed manifest.
    pub async fn run(self) -> Result<ExperimentManifest, RunError>;
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RunError {
    BudgetExhausted { stage_id: StageId, accepted: usize, required: usize },
    StageFailed     { stage_id: StageId, error: String },
    ManifestMismatch(ResumeMismatch),
    /// A component referenced by label was not registered on `Context`.
    MissingComponent(String),
}
```

---

### 16.2 SimulationResult

`SimulationBuilder::run()` ([§4](03-simulation-system-and-dag-stages.md)) returns a `SimulationResult`. For external simulation runs, artifact refs are appended to `ExperimentManifest::simulation_artifacts`. For internal calibration runs, `Context` absorbs the result directly into `StageState` (scores, provenances, failures, and model state refs) so that per-particle inspection ([§15](15-abc-rejection-sampling-execution.md)) and scoring provenance ([§9](09-scoreacceptancecriterion.md)) work identically for both use cases.

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct SimulationResult {
    pub population:        ParticlePopulation,
    /// Score values keyed by particle ID and scorer fingerprint.
    /// Each particle has one entry per scorer that was run.
    pub scores:            HashMap<ParticleId, HashMap<Fingerprint, ScoreValueType>>,
    /// Score provenance records keyed by particle ID and scorer fingerprint.
    /// Populated when a scorer implements `score_with_provenance` (§9).
    pub score_provenances: HashMap<ParticleId, HashMap<Fingerprint, ScoreProvenance>>,
    /// Artifact refs for all model output files written during this run,
    /// one entry per executed particle. Appended to `ExperimentManifest::simulation_artifacts`
    /// for external simulation runs; absorbed into `StageState` for calibration runs.
    pub artifacts:         Vec<ArtifactRef>,
    /// Deduplicated failures: one entry per unique error variant,
    /// each carrying the list of particle IDs that produced it.
    pub failures:          Vec<(RunStageError, Vec<ParticleId>)>,
    /// Artifact refs for intermediate runner states keyed by particle ID.
    /// Non-empty only when the active stage's `ModelStatePolicy` is `Checkpoint` or
    /// `ResumeAndCheckpoint`. Absorbed into `StageState::model_state_refs` by `Context`
    /// during calibration; always empty for external simulation runs.
    pub model_state_refs:  HashMap<ParticleId, ArtifactRef>,
    /// Variant labels for particles produced by an Iterator-mode `CounterfactualGroup`
    /// (§11), keyed by the expanded `ParticleId`.  Empty when no Iterator-mode
    /// counterfactuals were applied or when in Selector mode.  Absorbed into
    /// `StageState::counterfactual_labels` by `Context` during calibration.
    pub counterfactual_labels: HashMap<ParticleId, String>,
}
```

---

### 16.3 Failed Particle Rerun

`Context::rerun_particle` ([§14](14-error-propagation-and-stage-resumability.md)) handles single-particle targeted reruns. For batch recovery — when a stage accumulates multiple failures and the user wants to retry all of them, optionally with a different runner configuration or verbose output — `Context` exposes `rerun_failed`, which returns a `FailedParticleRerunBuilder`.

The builder reads the deduped failure list from `StageState::failures`, resolves each `ParticleId` to its stored config under `{stage_id}/{particle_id}/config.json`, and re-executes via the active (or overridden) runner. Results are written back into `StageState::scores` and `StageState::score_provenances` in place; the `StageState::failures` entry for each particle that succeeds on retry is removed.

```rust
impl Context {
    /// Construct a builder for batch-retrying all failed particles in `stage_id`.
    /// Optionally override the runner or enable verbose capture before executing.
    pub fn rerun_failed(&mut self, stage_id: &StageId) -> FailedParticleRerunBuilder<'_>;
}

/// Returned by `Context::rerun_failed`. Borrows ctx mutably for its lifetime.
pub struct FailedParticleRerunBuilder<'ctx> { /* opaque */ }

impl<'ctx> FailedParticleRerunBuilder<'ctx> {
    /// Override the model runner for this rerun batch.
    /// The replacement must have matching `Config` and `Output` associated types.
    /// Useful for switching to a patched binary or a different parallelism setting
    /// without rebuilding the full calibration.
    pub fn with_runner(self, runner: impl ModelRunnerProtocol + 'static) -> Self;

    /// Capture stdout and stderr from the runner process and write them to
    /// `{stage_id}/{particle_id}/runner.log` in the artifact store.
    /// When not set, runner output is discarded (the default, matching the original run).
    pub fn verbose(self) -> Self;

    /// Restrict the rerun to a specific subset of particle IDs instead of the full
    /// failure list. Particle IDs not present in `StageState::failures` are silently
    /// ignored.
    pub fn particles(self, ids: impl IntoIterator<Item = ParticleId>) -> Self;

    /// Re-execute the selected failed particles using the stored config for each.
    /// - Particles that succeed have their scores written into `StageState::scores`,
    ///   their provenance (if available) into `StageState::score_provenances`, and are
    ///   removed from `StageState::failures`.
    /// - Particles that fail again are retained in `StageState::failures` with an
    ///   incremented `attempt` count (see `ParticleFailure::attempt`, §15).
    ///
    /// Returns a map of `ParticleId` → result for every particle that was retried.
    pub async fn run(
        self,
    ) -> Result<HashMap<ParticleId, Result<ScoreValueType, RunStageError>>, RunError>;
}
```

**Verbose output details:** When `.verbose()` is set, `Context` attaches a pipe to the runner subprocess's stdout and stderr streams and streams the combined output into a log file at `{stage_id}/{particle_id}/runner.log`. The `ArtifactRef` for the log is appended to the particle's `output_refs` in any subsequent `inspect_particle` call ([§14](14-error-propagation-and-stage-resumability.md)), so the full execution trace is accessible alongside the config and model outputs without any separate lookup. When not set, no subprocess output is captured or stored; runner I/O behaviour is identical to the original calibration run.

**Runner override semantics:** Supplying `.with_runner(r)` replaces the runner for the duration of this builder's `.run()` call only; the runner registered on `Context` is not mutated. The replacement runner must implement `ModelRunnerProtocol` with the same `Config` and `Output` types as the original, but may have different binaries, build mechanics, or parameters. Type compatibility is checked at the call site via the trait bound; a mismatch is a compile-time error. This allows switching to a debug binary, a CPU-pinned variant, or a runner with different concurrency limits without invalidating the existing `CalibrationManifest` or requiring a new `Context`.

---

### 16.4 Calibration Comparison

`CalibrationComparisonBuilder` compares two or more accepted particle populations across any combination of DAG stages in the current calibration or stages loaded from foreign `ExperimentManifest`s. This covers three distinct use cases:

- **Stage progression within a calibration** — compare the accepted populations at stage 0, 1, and 2 of the same run to observe how the posterior tightens across ABC-SMC iterations.
- **Cross-manifest comparison** — compare the posteriors from two different experiment runs (e.g. different seeds, different priors) at the same stage.
- **Altered-features evaluation** — take the same accepted population and re-evaluate it under a revised scorer, tightened criterion, or replacement runner, to answer "how does this change affect the posterior?" without launching a new calibration.

No new perturbation or rejection-sampling loop is run in any mode. Particle identities and parameter values are fixed by the chosen source(s); only scoring, criterion evaluation, and (optionally) model execution change between sides.

#### Source abstraction

Each side of a comparison is described by a `ComparisonSource` and an optional set of evaluation overrides bundled into a `SideSpec`:

```rust
/// Identifies the accepted particle population for one side of a comparison.
#[derive(Debug, Clone)]
pub enum ComparisonSource {
    /// Accepted particles from a completed stage held in the current Context::stage_states.
    LiveStage(StageId),
    /// Accepted particles loaded from a completed stage in a foreign ExperimentManifest.
    /// Context resolves the population via the manifest's `posterior_artifacts` ArtifactRefs.
    ManifestStage {
        manifest: ExperimentManifest,
        /// The stage to load. Defaults to the manifest's leaf stage when `None`.
        stage_id: Option<StageId>,
    },
}

/// Full specification for one side of a comparison: source population and optional
/// evaluation overrides. All overrides apply only to this side; other sides are
/// unaffected.
pub struct SideSpec {
    pub label:  String,
    pub source: ComparisonSource,
}

impl SideSpec {
    pub fn new(label: impl Into<String>, source: ComparisonSource) -> Self;

    /// Override the scorer used to (re-)evaluate particles on this side.
    /// Must be registered on `Context`. When set, `Context` applies the scorer to the
    /// existing cached model outputs for each particle; no model re-execution occurs
    /// unless `.rerun_model()` is also called.
    pub fn with_scorer(self, label: &str) -> Self;

    /// Override the acceptance criterion evaluated for this side.
    /// When not set, the criterion from the original `StageNode` (or the manifest's
    /// stage node for `ManifestStage` sources) is used.
    pub fn with_criterion(self, criterion: ScoreAcceptanceCriterion) -> Self;

    /// Replace the model runner for this side's execution pass.
    /// The replacement must have matching `Config` and `Output` associated types.
    /// Implies `.rerun_model()`.
    pub fn with_runner(self, runner: impl ModelRunnerProtocol + 'static) -> Self;

    /// Re-run the model for every particle on this side rather than re-scoring from
    /// cached outputs. Required when the runner or its configuration has changed.
    /// Model outputs are written to `{stage_id}/{particle_id}/comparison/{side_label}/`
    /// so that original calibration artifacts are never overwritten.
    pub fn rerun_model(self) -> Self;
}
```

#### Builder

```rust
impl Context {
    /// Begin constructing a multi-side calibration comparison.
    pub fn compare(&mut self) -> CalibrationComparisonBuilder<'_>;
}

/// Returned by `Context::compare`. Borrows ctx mutably for its lifetime.
pub struct CalibrationComparisonBuilder<'ctx> { /* opaque */ }

impl<'ctx> CalibrationComparisonBuilder<'ctx> {
    /// Add a side to the comparison. Call at least twice before `.run()`.
    /// Sides are ordered by declaration; the first two sides are treated as
    /// baseline and candidate when computing per-particle and overlap statistics.
    pub fn side(self, spec: SideSpec) -> Self;

    /// Execute the comparison and return a `CalibrationComparison`.
    ///
    /// Per-particle records are populated when exactly two sides are declared and
    /// their accepted particle ID sets have a non-empty intersection. This occurs
    /// naturally when both sides draw from the same source population (e.g. the
    /// altered-features case) but not when comparing distinct DAG stages whose
    /// accepted sets are independent (e.g. stage-progression or cross-manifest cases).
    pub async fn run(self) -> Result<CalibrationComparison, RunError>;
}
```

#### Output types

```rust
/// Multi-side comparison result. One `ComparisonSideResult` per declared side, plus
/// aggregate statistics and optional per-particle records.
#[derive(Debug, Serialize, Deserialize)]
pub struct CalibrationComparison {
    /// One entry per side, in declaration order.
    pub sides:       Vec<ComparisonSideResult>,
    /// Per-particle records. Populated only when exactly two sides are declared and
    /// their accepted particle ID sets intersect (i.e. both sides evaluate the same
    /// particles). Each value holds the entry for side[0] first, side[1] second.
    pub per_particle: Option<HashMap<ParticleId, [ParticleComparisonEntry; 2]>>,
    /// Aggregate statistics across all sides.
    pub summary:     ComparisonSummary,
}

/// Per-side population statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonSideResult {
    pub label:      String,
    /// Identifies the experiment and stage this side was loaded from.
    pub source_id:  ComparisonSourceId,
    /// Number of particles accepted under this side's effective criterion.
    pub n_accepted: usize,
    /// Effective sample size from the log-weights of the accepted population.
    pub ess:        f64,
    /// Score distribution statistics for this side's accepted population.
    pub score_stats: ScoreDistributionStats,
}

/// Stable identifier stored in the comparison result.
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonSourceId {
    /// The ExperimentManifest::id this population was loaded from.
    /// `None` when the source is a LiveStage in the current Context.
    pub experiment_id: Option<uuid::Uuid>,
    pub stage_id:      StageId,
}

/// Score distribution summary for one side's accepted population.
#[derive(Debug, Serialize, Deserialize)]
pub struct ScoreDistributionStats {
    pub n:       usize,
    pub mean:    f64,
    pub std_dev: f64,
    pub min:     f64,
    pub max:     f64,
    /// Quantile values at the 5th, 25th, 50th, 75th, and 95th percentiles.
    pub quantiles: [(f64, ScoreValueType); 5],
}

/// Score, acceptance, and optional provenance for one particle on one side.
#[derive(Debug, Serialize, Deserialize)]
pub struct ParticleComparisonEntry {
    pub score:      ScoreValueType,
    pub accepted:   bool,
    /// Populated when the active scorer implements `score_with_provenance` (§9).
    pub provenance: Option<ScoreProvenance>,
}

/// Aggregate diff statistics.
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonSummary {
    /// Populated only when exactly two sides are declared and their particle sets
    /// intersect (i.e. `CalibrationComparison::per_particle` is `Some`).
    pub particle_overlap: Option<ParticleOverlapSummary>,
    /// Mean score delta (side[1] − side[0]) across all intersecting particles,
    /// for `Numeric` scores. `None` for `Range` scores or when `particle_overlap`
    /// is absent.
    pub mean_score_delta: Option<f64>,
}

/// Overlap statistics for two-sided comparisons sharing particle identities.
#[derive(Debug, Serialize, Deserialize)]
pub struct ParticleOverlapSummary {
    /// Number of particles accepted under both sides' criteria.
    pub jointly_accepted: usize,
    /// Accepted under side[0] but rejected under side[1].
    pub lost:             usize,
    /// Rejected under side[0] but accepted under side[1].
    pub gained:           usize,
}

impl CalibrationComparison {
    /// Print a formatted comparison table to stdout.
    ///
    /// - When `per_particle` is `Some`: prints a per-particle table (particle index,
    ///   side[0] score, side[1] score, delta, accepted flags) followed by the overlap
    ///   summary.
    /// - When `per_particle` is `None`: prints a side-by-side population statistics
    ///   table (label, n_accepted, ESS, score mean ± std, score percentiles).
    pub fn display(&self);
}
```

#### Usage patterns

**Stage progression within the same calibration:**
```rust
// Compare how the posterior tightens across all three ABC-SMC stages.
let dag_stages = ctx.calibration_manifest.stage_map.keys().cloned().collect::<Vec<_>>();

let comparison = ctx.compare()
    .side(SideSpec::new("stage_0", ComparisonSource::LiveStage(dag_stages[0].clone())))
    .side(SideSpec::new("stage_1", ComparisonSource::LiveStage(dag_stages[1].clone())))
    .side(SideSpec::new("stage_2", ComparisonSource::LiveStage(dag_stages[2].clone())))
    .run().await?;

// Particle sets are independent across stages, so per_particle is None.
// display() prints the population statistics table.
comparison.display();
```

**Cross-manifest comparison at the stage node:**
```rust
// Compare the posteriors from two experiment runs with different seeds.
let comparison = ctx.compare()
    .side(SideSpec::new("run_a", ComparisonSource::ManifestStage {
        manifest: manifest_a, stage_id: None,  // defaults to end node
    }))
    .side(SideSpec::new("run_b", ComparisonSource::ManifestStage {
        manifest: manifest_b, stage_id: None,
    }))
    .run().await?;

comparison.display();
```

**Altered-features evaluation (same population, new scorer and criterion):**
```rust
// Re-evaluate the leaf posterior under a revised scorer without rerunning the model.
ctx.register_scorer("incidence_mae_v2", RevisedIncidenceScorer);

let comparison = ctx.compare()
    .side(SideSpec::new("original", ComparisonSource::ManifestStage {
        manifest: prior_manifest.clone(), stage_id: None,
    }))
    .side(SideSpec::new("revised", ComparisonSource::ManifestStage {
        manifest: prior_manifest, stage_id: None,
    })
    .scorer("incidence_mae_v2")
    .criterion(ScoreAcceptanceCriterion::Threshold { threshold: 30.0 }))
    .run().await?;

// Both sides draw from the same population; particle IDs are identical.
// per_particle is populated. display() prints the per-particle table.
comparison.display();
// summary: jointly_accepted=312, lost=188, gained=0, mean_score_delta=-11.4
```

The `CalibrationComparison` returned by `.run()` is not written into `ExperimentManifest` automatically. Callers may serialize it independently or attach it as a diary entry ([§2.3](02-calibrator-construction.md#23-calibrationmanifest-and-experimentmanifest)) via `DiaryEntry` for persistent annotation alongside the originating experiment.

---

[← ABC Rejection Sampling Execution](15-abc-rejection-sampling-execution.md) | [TOC](README.md) | [Next: End-to-end example →](17-end-to-end-example.md)
