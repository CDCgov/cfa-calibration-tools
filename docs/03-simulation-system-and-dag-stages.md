[← Calibrator Construction](02-calibrator-construction.md) | [TOC](README.md) | [Next: Particle Type and Overlay Semantics →](04-particle-type-and-overlay-semantics.md)

---

## 3. Simulation System and DAG Stages

### StageId

`StageId` is a **content-addressed fingerprint** of the `StageNode`'s configuration, presented to users as a short integer index (e.g., `0`, `1`, `2`). The integer is a stable positional index assigned at DAG construction time by topological order.

```rust
pub type StageId = Fingerprint;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A scorer fingerprint paired with its acceptance criterion, as stored in `StageNode`
/// after criterion accumulation at build time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageScorerEntry {
    /// Fingerprint of a scorer registered on `Context`.
    pub scorer_fingerprint: Fingerprint,
    pub criterion:          ScoreAcceptanceCriterion,
}

pub struct StageNode {
    pub id:                   StageId,
    /// None only for the DAG root.
    pub parent_id:            Option<StageId>,
    /// Effective (scorer, criterion) pairs for this stage after criterion accumulation
    /// at build time. A particle is accepted only when all pairs are satisfied.
    pub scorer_criteria:      Vec<StageScorerEntry>,
    pub population_budget:    PopulationBudget,
    /// Per-stage perturbation kernel override. When `Some`, this kernel is used for
    /// this stage instead of derivations from the `CalibrationManifest::root_perturbation_kernel` (see §6).
    /// Populated by `CalibrationBuilder` when a stage-level override is declared via `CalibrationStageSpec::perturbation_override`.
    pub perturbation_override: Option<PerturbationType>,
    /// Realized kernel for this stage with scale values set by the adaptation lifecycle.
    /// Populated by `Context` at runtime; always `None` in the static `CalibrationManifest`.
    /// Mirrors `ExperimentManifest::realized_kernels` for fast in-memory access.
    pub realized_kernel:       Option<PerturbationType>,
    /// How this stage handles intermediate model states relative to its own run
    /// and relative to the model state checkpointed by its parent stage.
    /// Defaults to `ModelStatePolicy::None`.
    pub model_state_policy:  ModelStatePolicy,
}

/// Controls whether a stage checkpoints intermediate `ModelRunnerProtocol::State` values
/// and/or resumes from its parent stage's checkpointed states.
/// Enables a common particle identity across stages that run the same model at
/// different durations — the `ParticleId` is preserved across both `StageState` maps.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ModelStatePolicy {
    /// No model state is captured or inherited (default).
    #[default]
    None,
    /// Run the model via `run_partial`, checkpoint the returned `State` per particle,
    /// and store it in `ArtifactStore`. The `ArtifactRef` is recorded in
    /// `StageState::model_state_refs` keyed by `ParticleId`.
    Checkpoint,
    /// For each accepted particle in the parent stage, retrieve its checkpointed `State`
    /// from `ArtifactStore` and call `run_from_state` to extend the run.
    /// The parent stage must have `Checkpoint` or `ResumeAndCheckpoint`.
    ResumeFromParent,
    /// Both resume from the parent stage's checkpointed states and checkpoint this
    /// stage's resulting states for a potential further child stage.
    ResumeAndCheckpoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationBudget {
    pub target_accepted: usize,
    pub max_proposals:   Option<usize>,
}
```

### StageState

`StageState` is the runtime companion to `StageNode`, owned by `Context` in its `stage_states` map. It is the authoritative location for per-particle score values: scores are indexed by particle ID within the stage that produced them, not stored inside `ParticlePopulation`, although they are associated through the unique fingerprinted `ParticleId` values. `Context` passes `&stage_state.scores` into `ParticlePopulation::filter` at the call site. Unique `ParticleError`s are mapped to individual `ParticleId`s and surfaced to the user for debugging and re-running

```rust
/// Runtime state for a DAG node, owned by Context.
/// Score values live here, not in ParticlePopulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageState {
    pub node_id:          StageId,
    /// Score values keyed by particle ID and scorer fingerprint.
    /// Each particle carries one score per scorer declared in the stage's `scorer_criteria`.
    /// The inner key is `StageScorerEntry::scorer_fingerprint`.
    pub scores:           HashMap<ParticleId, HashMap<Fingerprint, ScoreValueType>>,
    /// Provenance records keyed by particle ID and scorer fingerprint.
    /// Populated when a scorer implements `score_with_provenance` (§9).
    pub score_provenances: HashMap<ParticleId, HashMap<Fingerprint, ScoreProvenance>>,
    /// Owned accepted population for this stage. Access a read-only view via `.as_slice()`.
    pub accepted:         ParticlePopulation,
    pub failures:         HashMap<RunStageError, Vec<ParticleId>>,
    pub budget_exhausted: bool,
    /// Artifact refs for intermediate `ModelRunnerProtocol::State` values keyed by particle ID.
    /// Populated when `StageNode::model_state_policy` is `Checkpoint` or `ResumeAndCheckpoint`.
    /// A child stage with `ResumeFromParent` reads these refs to call `run_from_state`.
    pub model_state_refs: HashMap<ParticleId, ArtifactRef>,
    /// Counterfactual variant label for each accepted particle.
    /// Populated when `CalibrationManifest::counterfactuals` is `Some` with `Iterator` mode.
    /// Empty for `Selector` mode (variant identity is encoded in the calibrated
    /// `ModelSelector` parameter) and when no counterfactuals are declared.
    pub counterfactual_labels: HashMap<ParticleId, String>,
    /// Stage-local RNG streams created by `Context::instantiate_stage_rng` at stage start.
    /// Keyed to this stage's identity (`base_entropy ^ StageId`); advanced throughout
    /// the rejection-sampling loop.
    pub rngs:                  RngSnapshot,
}

impl StageState {
    pub fn new(id) -> Self {
        Self {
            node_id: id,
            scores: HashMap::new(),
            score_provenances: HashMap::new(),
            accepted: ParticlePopulation::new(),
            failures: HashSet::new(),
            model_state_refs: HashMap::new(),
            counterfactual_labels: HashMap::new(),
        }
    }
}

```

### CalibrationDag

```rust
#[derive(Debug, Clone)]
pub struct CalibrationDag {
    /// Adjacency list: parent → set of children.
    pub edges: HashMap<StageId, HashSet<StageId>>,
    pub nodes: HashMap<StageId, StageNode>,
    /// Stable integer index for user-facing display, assigned in topological order.
    pub index: HashMap<StageId, usize>,
}

impl CalibrationDag {
    pub fn end_nodes(&self) -> Vec<StageId> {
        self.nodes.keys()
            .filter(|id| self.edges.get(*id).map_or(true, |ch| ch.is_empty()))
            .cloned()
            .collect()
    }

    /// Kahn's algorithm — returns a topological order or an error if a cycle exists.
    pub fn topological_order(&self) -> Result<Vec<StageId>, String> {
        // in-degree map
        let mut in_degree: HashMap<StageId, usize> = self.nodes.keys()
            .map(|id| (id.clone(), 0))
            .collect();
        for children in self.edges.values() {
            for child in children {
                *in_degree.entry(child.clone()).or_insert(0) += 1;
            }
        }
        let mut queue: std::collections::VecDeque<StageId> =
            in_degree.iter().filter(|(_, &d)| d == 0).map(|(id, _)| id.clone()).collect();
        let mut order = Vec::new();
        while let Some(id) = queue.pop_front() {
            order.push(id.clone());
            if let Some(children) = self.edges.get(&id) {
                for child in children {
                    let d = in_degree.get_mut(child).unwrap();
                    *d -= 1;
                    if *d == 0 { queue.push_back(child.clone()); }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err("CalibrationDag contains a cycle".into());
        }
        Ok(order)
    }
}
```

### Simulation builder

`SimulationBuilder` is `Context`'s unified interface for executing a particle population against the model runner and collecting generated quantities. It is used in two contexts:

- **Internally during calibration**: `Context` constructs a `SimulationBuilder` for each proposal batch in a rejection-sampling stage, supplying the batch directly via `from_population`. Results feed the stage's `ScoreCalculator` and are absorbed into `StageState`.
- **External to calibration**: Users call `ctx.simulate()` to draw from the posterior, root, or any other population, overlay parameter sweeps, and can optionally score against targets. Results are returned as a `SimulationResult` ([§16.2](16-runtime-execution.md#162-simulationresult)) whose artifacts are appended to `ExperimentManifest::simulation_artifacts`.

`SimulationBuilder` is owned by `Context` and always references the same registered runner and `ArtifactStore`.

```rust
pub trait ContextSimulateExt {
    fn simulate(&mut self) -> SimulationBuilder<'_>;
}

impl ContextSimulateExt for Context { /* ... */ }

impl SimulationBuilder<'_> {
    /// Supply a particle population directly. Used internally by `Context` for each
    /// calibration proposal batch; also available for manual population construction.
    pub fn from_population(self, population: ParticlePopulation) -> Self;
    /// Convenience: use the accepted population from the calibration leaf (posterior).
    pub fn from_posterior(self) -> Self;
    /// Convenience: use the accepted population from the DAG root (prior predictive).
    pub fn from_root(self) -> Self;
    /// Convenience: use the accepted population from a specific DAG node.
    pub fn from_stage(self, id: StageId) -> Self;
    /// Convenience: use the default parameters from root.
    pub fn from_defaults(self) -> Self
    /// Overlay a supplementary population onto every base particle before parsing.
    /// Equivalent to `ParticlePopulation::product` on the base and overlay (§6). No call limit.
    pub fn counterfactuals(self, population: ParticlePopulation, strategy: MergeStrategy) -> Self;
    /// Supply multiple (scorer, criterion) entries from a stage's `scorer_criteria`.
    /// Used internally by `Context` during calibration; each scorer is evaluated against
    /// its paired criterion and results are keyed by scorer fingerprint in `SimulationResult`.
    pub fn criteria(self, entries: &[StageScorerEntry]) -> Self;
    /// Resolve a scorer via its `ScorerRef` and run it against the population.
    /// No acceptance criterion is applied; all particles are scored and their results
    /// are available in `SimulationResult::scores` keyed by scorer fingerprint.
    /// Multiple calls chain additional scorers. Use when no target data is required
    /// (simulation-only scorers such as R₀ or peak incidence).
    pub fn scorer<GQ: GeneratedQuantity, T: Target>(self, scorer: &ScorerRef<GQ, T>) -> Self;
    /// Resolve a scorer and its target via typed refs and run against the population.
    /// Mirrors `CalibrationStageSpec::with_data_based_criterion`: the compiler enforces
    /// that the scorer's `T` matches the target's `T`. `Context` retrieves the cached
    /// target built during calibration (keyed by `target_ref.fingerprint`), guaranteeing
    /// the simulation scores against identical data. Multiple calls chain additional
    /// (scorer, target) pairs; each scorer receives only the target it was paired with.
    pub fn scorer_with_target<GQ: GeneratedQuantity, T: Target>(
        self,
        scorer:     &ScorerRef<GQ, T>,
        target_ref: &TargetRef<T>,
    ) -> Self;
    /// Assign a distinct seed to each particle by drawing from the active stage's `rngs.seed`
    /// (accessed via `ctx.stage_states[ctx.active_stage_id].rngs.seed`). Particles are
    /// visited in sorted `ParticleId` order; each draw advances the stream by one step.
    /// Particles that already carry the seed key keep their value (`PreferLeft`).
    /// No-op when no seed parameter has been registered on `Context`
    pub fn randomize_seeds(self) -> Self;
    /// Expand every particle into `n_replicates` variants. Draws `n_replicates` values from
    /// the active stage's `rngs.seed` up front
    /// No-op when no seed parameter has been registered on `Context`.
    pub fn replicate_counterfactuals(self, n_replicates: usize) -> Self;
    /// Execute all particles, collect scores and artifacts, and return the result.
    pub async fn run(self) -> Result<SimulationResult, RunError>;
}
```

`randomize_seeds` and `replicate_counterfactuals` are thin public entry points that derive the
offset from `Context` and delegate to private `_with_offset` implementations.
`Context` exposes `current_proposal_offset()`, which returns `n_proposed` during a calibration
stage and `0` for external `SimulationBuilder` calls:

```rust
// Context — internal accessor
impl Context {
    /// Returns the stage-global proposal offset used for seed derivation.
    /// Equals `n_proposed` while a calibration stage is executing;
    /// `0` for external `SimulationBuilder` calls.
    fn current_proposal_offset(&self) -> u64;
}

// Union of random new seeds for particles without seed values
pub fn randomize_seeds(self) -> Self {
    if let Some(param) = (&self.ctx.seed_param) {
        let rng = &mut self.ctx.stage_states.get_mut(self.stage_id).unwrap().rngs.seed;
        self.population = self.population.random_seeds(rng, param);
    }
}

// Cartesian product of new seed values to particles without seed values
fn replicate_counterfactuals(self, n_replicates: usize) -> Self {
    if let Some(param) = self.ctx.seed_param() {
        let rng = &mut self.ctx.stage_states.get_mut(self.stage_id).unwrap().rngs.seed;
        self.population = self.population.replicate_seeds(
            self.ctx.base_seed(), param, n_replicates, offset,
        );
    }
    self
}
```

`counterfactuals` crosses the current population with the variant population via `ParticlePopulation::product` ([§6](06-perturbationkernel-and-density-convention.md)), then strips the reserved `"__variant_label__"` key from each merged particle and records it in `counterfactual_labels` for inclusion in `SimulationResult`:

```rust
// SimulationBuilder::counterfactuals(variants, strategy)
let mut expanded = ParticlePopulation::product(&population, &variants, strategy);
for (id, wp) in &mut expanded.entries {
    if let Some(raw) = wp.particle.0.remove("__variant_label__") {
        if let Some(label) = raw.as_str() {
            counterfactual_labels.insert(id.clone(), label.to_string());
        }
    }
}
population = expanded;
```

### Diagnostics

Diagnostics are computed by `Context` after each rejection-sampling stage completes and stored in `ExperimentManifest::diagnostics` keyed by `StageId`. They are runtime records, not DAG nodes. `CalibrationStageResult` is passed directly to the diagnostic computation.

Per-stage diagnostics are accessed via `Context::diagnostics(Option<StageId>)` ([§2](02-calibrator-construction.md)); passing `None` returns the posterior leaf. Overall calibration diagnostics are accessed via `Context::calibration_diagnostics()` ([§2](02-calibrator-construction.md)).

```rust
pub struct DiagnosticsStageResult {
    pub stage_id:               StageId,
    pub effective_sample_size:  f64,
    pub acceptance_rate:        f64,
    /// KL divergence from prior to posterior marginals (per-parameter).
    pub information_gain:       HashMap<String, f64>,
    pub n_accepted:             usize,
    pub n_proposed:             usize,
    pub budget_exhausted:       bool,
}

impl DiagnosticsStageResult {
    /// Print a formatted summary table to stdout.
    pub fn display(&self);
}
```

`CalibrationDiagnosticsResult` rolls up per-stage records into a trajectory summary:

```rust
pub struct CalibrationDiagnosticsResult {
    pub effective_sample_size:   f64,
    /// Acceptance rate across all stages.
    pub overall_acceptance_rate: f64,
    /// KL divergence from prior to posterior marginals (per-parameter), taken from the leaf stage.
    pub information_gain:        HashMap<String, f64>,
    pub n_accepted_overall:      usize,
    pub n_proposed_overall:      usize,
    pub pearson_correlation: HashMap<(String, String), f64>,
    pub success:                 bool,
}

impl CalibrationDiagnosticsResult {
    /// Print a per-stage trajectory table and overall summary line to stdout.
    pub fn display(&self);
}
```

### Predictive checks
Predictive checks can be run on any stage of a calibration DAG. They can be added to root during build time of the calibration, or called on root or other stages at later time points.


**`PredictiveCheckBuilder`** — samples from the prior without rejection and returns a `PredictiveCheckReport`:

```rust
impl PredictiveCheckBuilder<'_> {
    pub fn new(self, stage: StageId, sample_size: usize) -> Self;
    pub fn target<T: Target>(self, t: T) -> Self;
    /// Whether or not to run benchmark performance diagnostics alongside the predictive check
    pub fn benchmark(self) -> Self;
    pub async fn run(self) -> Result<PredictiveCheckReport, RunError>;
}

pub struct PredictiveCheckReport {
    pub scores:    Vec<ScoreValueType>,
    pub performance: Option<BenchmarkPerformanceReport>
    pub stage_id:  StageId,
}

impl PredictiveCheckReport {
    /// Returns a formatted table of score quantiles at standard percentiles.
    pub fn score_quantile_table(&self) -> String;
    pub fn score_quantiles(&self, quantiles: &[Quantile]) -> Vec<(Quantile, ScoreValueType)>;
    /// Transform the scores of each quantile into acceptance criteria for suggestion
    pub fn limiting_criterion_quantiles(&self, quantiles: &[Quantile], criterion_type: ScoreAcceptanceCriterionType) -> Vec<(Quantile, ScoreAcceptanceCriterion)>
}
```

---

[← Calibrator Construction](02-calibrator-construction.md) | [TOC](README.md) | [Next: Particle Type and Overlay Semantics →](04-particle-type-and-overlay-semantics.md)
