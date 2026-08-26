[← ParticlePopulation, Weights, and Product](05-particlepopulation-weights-and-product.md) | [TOC](README.md) | [Next: NestedSuffixParser and ParticleError →](07-nestedsuffixparser-and-particleerror.md)

---

## 6. PerturbationKernel and Density Convention

`PerturbationType` is the single kernel type used throughout the system: as the root and per-stage override stored in `CalibrationManifest`, as the live perturbation object during sampling, and as the realized audit record written into `StageNode::realized_kernel` and `ExperimentManifest::realized_kernels`. Scale fields hold initial values at manifest-build time; between-stage adaptation updates them in place.

### `PerturbationType`

```rust
/// Single kernel type used in manifests, stage overrides, runtime sampling,
/// and as the realized audit record. Scale values are set at build time from
/// prior statistics and updated between stages by `PerturbationType::adapt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerturbationType {
    /// Continuous parameter — scale adapted as σ ← sqrt(2 · Var[param]).
    Normal             { param: String, std_dev: f64 },
    /// Continuous parameter — scale adapted as w ← 2 · sqrt(2 · Var[param]).
    Uniform            { param: String, width:   f64 },
    /// Joint continuous parameters — scale adapted as Σ ← 2 · empirical Cov.
    MultivariateNormal { params: Vec<String>, cov_matrix: Vec<Vec<f64>> },
    /// Stochastic seed parameter — retained with probability `prob_keep` or removed
    /// for re-derivation in §15, Step 2. `prob_keep` is fixed; not adapted between stages.
    /// `prob_keep = 0.0` is the default (unconditional re-derivation at each stage).
    Seed               { param: String, prob_keep: f64 },
    /// Discrete counterfactual-selector index — held fixed during perturbation.
    /// Selection occurs implicitly through the weighted resample step (§15, Step 1),
    /// which preserves the empirical distribution of selectors from the accepted
    /// population. Not adapted between stages. The prior over selector indices is
    /// uniform over `n_variants` and declared automatically by `build()`.
    ModelSelector      { param: String, n_variants: usize },
    /// Composes multiple kernels acting on disjoint parameter sets.
    /// `perturb` applies components sequentially; `log_density` is the sum.
    Independent        { kernels: Vec<PerturbationType> },
}

impl PerturbationType {
    /// Return an updated copy with scale values recomputed from a slice `population`.
    /// Adaptation rule per variant:
    ///   - `Normal`:             σ ← sqrt(2 · Var[param])
    ///   - `Uniform`:            w ← 2 · sqrt(2 · Var[param])
    ///   - `MultivariateNormal`: Σ ← 2 · empirical Cov[params]
    ///   - `Seed`:               unchanged (prob_keep is fixed unless overridden)
    ///   - `ModelSelector`:      unchanged (no scale to adapt)
    ///   - `Independent`:        each component adapted independently
    ///
    /// The factor of 2 follows the ABC-SMC optimal bandwidth heuristic
    /// (Beaumont et al. 2009). Called by the Between-Stage Adaptation Lifecycle.
    pub fn adapt(&self, population: &PopulationSlice<'_>) -> Self;
}
```

### `PerturbationKernel` Trait

```rust
use rand::RngCore;

pub trait PerturbationKernel: Send + Sync {
    /// Sample a new particle given the current one.
    fn perturb(&self, current: &FlatParticle, rng: &mut dyn RngCore) -> FlatParticle;

    /// Log probability of proposing `proposed` given current state `origin`.
    ///
    /// Convention: `log q(proposed | origin)`.
    fn log_density(&self, origin: &FlatParticle, proposed: &FlatParticle) -> f64;

    fn clone_kernel(&self) -> Box<dyn PerturbationKernel>;
}
```

`PerturbationType` is the canonical implementation. `Independent` dispatch chains components for `perturb` and sums log-densities:

```rust
impl PerturbationKernel for PerturbationType {
    fn perturb(&self, current: &FlatParticle, rng: &mut dyn RngCore) -> FlatParticle {
        match self {
            Self::Normal { .. } | Self::Uniform { .. }
            | Self::MultivariateNormal { .. } | Self::Seed { .. } => { /* variant-specific */ }
            Self::ModelSelector { .. } => {
                // Identity: selector held fixed during perturbation.
                // Model selection occurs implicitly through the weighted resample step (§15, Step 1).
                current.clone()
            }
            Self::Independent { kernels } =>
                kernels.iter().fold(current.clone(), |p, k| k.perturb(&p, rng)),
        }
    }
    fn log_density(&self, origin: &FlatParticle, proposed: &FlatParticle) -> f64 {
        match self {
            Self::Independent { kernels } =>
                kernels.iter().map(|k| k.log_density(origin, proposed)).sum(),
            _ => { /* variant-specific; see density conventions table below */ }
        }
    }
    fn clone_kernel(&self) -> Box<dyn PerturbationKernel> { Box::new(self.clone()) }
}
```

`build()` validates that parameter sets of all components within an `Independent` kernel are disjoint and together cover all calibrated prior parameters.

#### Density Conventions

| Variant | `log_density(origin, proposed)` |
|---|---|
| `Normal { std_dev }` | $\log \mathcal{N}(\text{proposed} \mid \text{origin},\, \sigma^2)$ |
| `Uniform { width }` | $\log(1/w)$ if $\text{proposed} \in [\text{origin} \pm w/2]$, else $-\infty$ |
| `Seed { prob_keep }` | $\log(p_{\text{keep}})$ if key unchanged; $\log(1 - p_{\text{keep}})$ otherwise |
| `ModelSelector` | $0$ if selector unchanged; $-\infty$ otherwise |
| `MultivariateNormal { cov_matrix }` | $\log \mathcal{N}_k(\text{proposed} \mid \text{origin},\, \Sigma)$ |
| `Independent { kernels }` | sum of component log-densities |

`Seed` perturbs by retaining the current seed key with probability `prob_keep` or removing it (triggering re-derivation in [§15](15-abc-rejection-sampling-execution.md), Step 2). `ModelSelector` returns the current particle unchanged; because selection already occurred through weighted resampling in Step 1 (which naturally preserves the empirical selector distribution from the accepted population), no further perturbation is applied, and `log_density` is $0$ when the selector is unchanged and $-\infty$ otherwise.

### `PerturbationInheritance`

```rust
/// How the root perturbation kernel propagates across DAG stages.
/// Recorded in `CalibrationManifest::perturbation_inheritance`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerturbationInheritance {
    /// Scale values are recomputed from the parent population between stages via
    /// `PerturbationType::adapt`. Per-stage `StageNode::perturbation_override` takes
    /// precedence over the root kernel for that stage.
    AdaptFromParent,
    /// Scale values are set once from the prior at run start and reused unchanged
    /// across all stages. Per-stage `StageNode::perturbation_override` is still applied.
    Fixed,
}
```

### `PerturbationStrategy` (Runtime)

```rust
/// Live perturbation kernel held by `Context` for the current calibration run.
/// Updated between stages by the adaptation lifecycle (see below). Not serializable.
pub struct PerturbationStrategy {
    pub kernel: PerturbationType,
}
```

### Perturbation Policy on `CalibrationBuilder`

`PerturbationInheritancePolicyExt` sets the root kernel policy written into `CalibrationManifest`. Per-stage overrides are declared via `CalibrationStageSpec::perturbation_override(kernel: PerturbationType)` ([§2.1](02-calibrator-construction.md#21-contextcalibrationext--calibration-declaration)) and stored in `StageNode::perturbation_override`.

`build_perturbation_kernel` and `validate_perturbation_kernel` are called automatically by `CalibrationBuilder::build()`; they are declared in the trait to make the kernel-construction contract explicit.

```rust
/// Extension trait for setting the root perturbation policy on `CalibrationBuilder`.
/// The default is `adaptive_covariance_matrix`.
pub trait PerturbationInheritancePolicyExt {
    /// Fix the root kernel to the supplied `PerturbationType` across all stages.
    /// Sets `PerturbationInheritance::Fixed`. `build_perturbation_kernel` still appends
    /// automatic `Seed` and `ModelSelector` components if applicable.
    fn fixed_perturbation(self, kernel: PerturbationType) -> Self;

    /// Independent `Normal` kernel per continuous calibrated parameter
    /// (σ ← sqrt(2 · Var[param]), re-adapted from parent population each stage).
    /// Sets `PerturbationInheritance::AdaptFromParent`.
    fn adaptive_normal_variance(self) -> Self;

    /// Independent `Uniform` kernel per continuous calibrated parameter
    /// (w ← 2 · sqrt(2 · Var[param]), re-adapted from parent population each stage).
    /// Sets `PerturbationInheritance::AdaptFromParent`.
    fn adaptive_uniform_variance(self) -> Self;

    /// Single `MultivariateNormal` kernel across all continuous calibrated parameters
    /// (Σ ← 2 · empirical Cov, re-adapted from parent population each stage). **Default.**
    /// Sets `PerturbationInheritance::AdaptFromParent`.
    fn adaptive_multivariate_covariance(self) -> Self;

    /// Construct the effective `PerturbationType` for the calibration.
    /// Called automatically by `build()`. Steps:
    ///
    /// 1. Build the continuous-parameter component from the selected `adaptive_*`
    ///    strategy (or the user-supplied `fixed_perturbation` kernel), using the
    ///    prior's parameter names.
    /// 2. Set initial scale values from the prior marginal statistics: σ = prior
    ///    std dev for `Normal`; w = prior range for `Uniform`; Σ = prior empirical
    ///    covariance for `MultivariateNormal`.
    /// 3. If a seed parameter is registered on `Context` (§1), append
    ///    `PerturbationType::Seed { param, prob_keep: 0.0 }`.
    /// 4. If `Selector`-mode counterfactuals are declared (§10), append
    ///    `PerturbationType::ModelSelector { param: "__model_selector__", n_variants }`.
    /// 5. If more than one component results from steps 1–4, wrap them in
    ///    `PerturbationType::Independent { kernels }`.
    fn build_perturbation_kernel(
        &self,
        prior_params: &[String],
        seed_param:   Option<&str>,
        n_variants:   Option<usize>,
    ) -> PerturbationType;

    /// Validate the effective kernel against the calibration's prior parameter set.
    /// Called automatically by `build()` after `build_perturbation_kernel`.
    /// Returns `CalibrationBuildError::KernelParamMismatch` if the union of parameter
    /// names across all leaf components does not exactly equal `prior_params`, or if
    /// any parameter appears in more than one component.
    fn validate_perturbation_kernel(
        kernel:       &PerturbationType,
        prior_params: &[String],
    ) -> Result<(), CalibrationBuildError>;
}

impl PerturbationInheritancePolicyExt for CalibrationBuilder<'_> { /* ... */ }
```

### Storage and Scope

| Location | Contents | Serializable |
|---|---|---|
| `CalibrationManifest::root_perturbation_kernel` | `PerturbationType` with initial scale values derived from prior | yes |
| `CalibrationManifest::perturbation_inheritance` | `PerturbationInheritance` | yes |
| `StageNode::perturbation_override` | `Option<PerturbationType>` — per-stage kernel override; `None` in static manifest | yes |
| `StageNode::realized_kernel` | `Option<PerturbationType>` — scale values after adaptation; `None` in static manifest | yes |
| `ExperimentManifest::realized_kernels` | `HashMap<StageId, PerturbationType>` — persisted realized kernels for audit and resumption | yes |
| `Context` (internal) | `PerturbationStrategy { kernel: PerturbationType }` — current live kernel | no |

`StageNode::realized_kernel` is populated by `Context` at runtime for fast in-memory lookup. `ExperimentManifest::realized_kernels` carries the same values for persistence across serialization boundaries. Neither field is set in the static `CalibrationManifest::stage_map`.

### Between-Stage Adaptation Lifecycle

After stage $t$ completes and before stage $t+1$ begins (see [§15](15-abc-rejection-sampling-execution.md), Pre-loop setup):

1. `Context` resolves the effective base kernel for stage $t+1$: `StageNode::perturbation_override` takes precedence over `CalibrationManifest::root_perturbation_kernel`.
2. For `PerturbationInheritance::AdaptFromParent`: calls `base_kernel.adapt(&stage_state_t.accepted)` to produce an updated `PerturbationType` with scale values from the parent population.
   For `PerturbationInheritance::Fixed`: the base kernel is used unchanged.
3. Writes the updated `PerturbationType` into `node.realized_kernel` (in-memory) and `experiment_manifest.realized_kernels[stage_id]` (persistence).
4. Updates `PerturbationStrategy::kernel` for use during stage $t+1$ proposals.

The DAG root ($t = 0$) has no parent population. `build_perturbation_kernel` initializes scale values from the prior at `build()` time, so `CalibrationManifest::root_perturbation_kernel` already carries valid initial scale values. The root stage's kernel is written directly to `node.realized_kernel` and `experiment_manifest.realized_kernels` without adaptation.

---

[← ParticlePopulation, Weights, and Product](05-particlepopulation-weights-and-product.md) | [TOC](README.md) | [Next: NestedSuffixParser and ParticleError →](07-nestedsuffixparser-and-particleerror.md)
