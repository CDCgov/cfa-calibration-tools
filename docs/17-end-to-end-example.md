[← Runtime Execution](16-runtime-execution.md) | [TOC](README.md)

---

## 17. End-to-end example

This example calibrates a stochastic SIR model to observed incidence data. The model
runner is an external process that reads a JSON config, writes daily incidence to a
file, and exits. An `IncidenceTargetBuilder` preprocesses the observed CSV into the
`Vec<f64>` target slice passed to each scorer; `Context` invokes the builder once
per calibration run, before the rejection-sampling loop begins, and caches the result
by `(builder_fingerprint, input_fingerprint)` so that all stages share the same built
value without reloading the file. Three sequential ABC-SMC stages are declared by
threshold alone; the DAG is constructed automatically. After calibration, three
counterfactual scenarios are projected over the posterior, and a parameter sweep from
CSV is run as a simulation stage.

```rust
use super::prelude::*;

// ---------------------------------------------------------------------------
// 1. Model runner
// ---------------------------------------------------------------------------

/// Thin wrapper around an external SIR simulator binary.
/// Receives the full model config as JSON, runs the process, returns daily incidence.
#[derive(Serialize, Deserialize)]
struct SirModelRunner {
    binary_path: std::path::PathBuf,
}

#[async_trait::async_trait]
impl ModelRunnerProtocol for SirModelRunner {
    type Config = serde_json::Value;
    type Output = Vec<f64>; // daily incidence time-series
    type State  = ();        // no partial-run support

    async fn run(
        &self,
        config:     &serde_json::Value,
        output_dir: &std::path::Path,
    ) -> Result<Vec<f64>, ModelRunError> {
        let config_path = output_dir.join("config.json");
        tokio::fs::write(&config_path, serde_json::to_vec(config)?).await?;
        let status = tokio::process::Command::new(&self.binary_path)
            .arg("--c").arg(&config_path)
            .arg("--o").arg(output_dir)
            .status().await?;
        if !status.success() {
            return Err(ModelRunError::ExitCode(status.code()));
        }
        let raw = tokio::fs::read(output_dir.join("incidence.json")).await?;
        Ok(serde_json::from_slice(&raw)?)
    }
}

// ---------------------------------------------------------------------------
// 2. Target builder: observed incidence from CSV
// ---------------------------------------------------------------------------

/// Raw input to `IncidenceTargetBuilder`. Must be `Serialize + Deserialize` so that
/// Context can store it in `CalibrationManifest::target_refs` and use it as the
/// input half of the target cache key (§8.1, §11.4).
/// Changing any field changes the input fingerprint, forcing a rebuild on the next run.
#[derive(ComponentFingerprint, Serialize, Deserialize, Clone)]
struct IncidenceBuilderConfig {
    csv_path:     std::path::PathBuf,
    /// Leading rows to trim before scoring (e.g. epidemic burn-in days).
    trim_leading: usize,
}

/// Reads daily incidence from a two-column CSV ("day", "incidence") and returns
/// the incidence column as a `Vec<f64>`, optionally trimming leading rows.
///
/// Deriving `ComponentFingerprint` ties the cache key to both the fully-qualified
/// type name and the serialized struct fields (§11.2). Renaming the type without a
/// `#[serde(rename)]` attribute, or changing a serialized field, produces a new
/// fingerprint and invalidates the target cache.
#[derive(ComponentFingerprint, Serialize, Deserialize)]
struct IncidenceTargetBuilder;

impl TargetBuilder for IncidenceTargetBuilder {
    type Input = IncidenceBuilderConfig;
    type T     = Vec<f64>;

    fn build(&self, input: &IncidenceBuilderConfig) -> Result<Vec<f64>, TargetBuildError> {
        let mut rdr = csv::Reader::from_path(&input.csv_path)
            .map_err(|e| TargetBuildError::Io(e.to_string()))?;
        let mut values: Vec<f64> = rdr
            .records()
            .map(|r| {
                r.map_err(|e| TargetBuildError::ParseFailure(e.to_string()))
                 .and_then(|rec| rec[1].parse::<f64>()
                     .map_err(|e| TargetBuildError::ParseFailure(e.to_string())))
            })
            .collect::<Result<_, _>>()?;
        values.drain(..input.trim_leading.min(values.len()));
        Ok(values)
    }
}

// ---------------------------------------------------------------------------
// 3. Score calculators: mean absolute error on daily incidence and peak incidence
// ---------------------------------------------------------------------------

// Struct-based scorers derive ComponentFingerprint automatically (§11).
// The fingerprint is computed from the fully-qualified type name plus serialized fields.
#[derive(ComponentFingerprint, Serialize, Deserialize)]
struct IncidenceScorer;

impl ScoreCalculator for IncidenceScorer {
    type GQ = Vec<f64>;
    type T  = Vec<f64>;

    fn score(
        &self,
        generated: &[Vec<f64>],
        targets:   &[Vec<f64>],
    ) -> Result<Option<ScoreValueType>, ScoringError> {
        let sim = &generated[0];
        let obs = targets.first()
            .ok_or_else(|| ScoringError::NumericalFailure("no target provided".into()))?;
        if sim.len() != obs.len() {
            return Err(ScoringError::NumericalFailure(
                format!("dimension mismatch: generated {} vs target {}", sim.len(), obs.len())
            ));
        }
        let mae = sim.iter().zip(obs).map(|(s, o)| (s - o).abs()).sum::<f64>() / sim.len() as f64;
        Ok(Some(ScoreValueType::Numeric(mae)))
    }

    fn documentation(&self) -> ScoreDocumentation {
        ScoreDocumentation::minimal("Mean absolute error on daily incidence time-series.")
    }
}

#[derive(ComponentFingerprint, Serialize, Deserialize)]
struct PeakIncidenceScorer;

impl ScoreCalculator for PeakIncidenceScorer {
    type GQ = Vec<f64>;
    type T  = Vec<f64>;

    fn score(
        &self,
        generated: &[Vec<f64>],
        targets:   &[Vec<f64>],
    ) -> Result<Option<ScoreValueType>, ScoringError> {
        let sim = &generated[0];
        let sim_peak = sim.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Ok(Some(ScoreValueType::Numeric(sim_peak)))
    }

    fn documentation(&self) -> ScoreDocumentation {
        ScoreDocumentation::minimal("Maximum incidence in the time series.")
    }
}

// ---------------------------------------------------------------------------
// 4. Build context
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {

    // The builder config is the serializable input that Context stores in
    // CalibrationManifest::target_refs. Changing any field changes the input
    // fingerprint and forces a cache miss, causing Context to call
    // IncidenceTargetBuilder::build again on the next run.
    let target_config = IncidenceBuilderConfig {
        csv_path:     "data/incidence.csv".into(),
        trim_leading: 7,
    };

    // Context: runner is the only required argument.
    // Infrastructure and component registrations are applied via trait extension methods.
    // seed_param declares the model as stochastic and makes the parameter visible to context.
    let mut ctx = Context::new(SirModelRunner { /* opaque */ })
        .artifact_store(LocalArtifactStore::new("artifacts/"))
        .checkpoint(CheckpointPolicy::ParticleBatch)
        .seed_param("seed")
        .base_entropy(12345)
        .set_defaults(sir_configuration);

    // register_scorer returns a ScorerRef<GQ, T> — the sole identity token for this scorer.
    // Hold on to it; it replaces string labels in stage specs and simulation builder calls.
    let incidence_scorer_ref: ScorerRef<Vec<f64>, Vec<f64>> = ctx.register_scorer("incidence_mae_v1", IncidenceScorer);
    let peak_scorer_ref: ScorerRef<Vec<f64>, Vec<f64>> = ctx.register_scorer("peak_v1", PeakIncidenceScorer);
    // register_target_builder returns a TargetRef<Vec<f64>> — the sole identity token
    // for this target. Hold on to it; it replaces the string label in add_target_data
    // and can pin specific scorers to this data in stage specs.
    let incidence_target_ref: TargetRef<Vec<f64>> = ctx.register_target_builder(
        "observed_incidence", IncidenceTargetBuilder, target_config.clone()
    );

    // Calibration declaration: build_calibration() returns a CalibrationBuilder that
    // borrows ctx mutably. build_and_run() validates refs, constructs the DAG and
    // manifest, commits them into ctx, and immediately executes the calibration.
    // Use .build() + ctx.run_calibration() directly when you need resume_from or
    // per-stage execution control.
    //
    // Stages 0 and 1 tighten the MAE threshold. with_data_based_criterion is used (rather
    // than the (usize, ScoreAcceptanceCriterion) shorthand) because target pinning
    // requires an explicit TargetRef — the tuple shorthand cannot carry one.
    // Scorer and target are wired via typed refs; the compiler enforces T matches.
    // Stage 2 adds a peak-incidence window. PeakIncidenceScorer ignores targets
    // entirely, so with_simulation_criterion (no TargetRef) is correct. The accumulated
    // incidence_mae_v1 criterion — including its target_fingerprint — is carried
    // forward from stages 0 and 1 by criterion accumulation (§9).
    // Stage 2's effective criteria are therefore:
    //   incidence_mae_v1 (incidence_scorer_ref): Threshold { threshold: 40.0 }  (accumulated, target pinned)
    //   peak_v1 (peak_scorer_ref):               NumericWindow { low: 120.0, high: 160.0 }  (new, no target)
    let manifest = ctx.build_calibration()
        .with_priors({ /* opaque */ })
        // adaptive_multivariate_covariance is the default; explicit here for clarity.
        // Scale values are recomputed from the parent population between stages (§6).
        .adaptive_multivariate_covariance()
        .abc_stages([
            // Stages 0 and 1: scorer and target wired via typed refs. The compiler
            // enforces incidence_scorer_ref.T == incidence_target_ref.T at the call site.
            // The shorthand tuple (usize, ScoreAcceptanceCriterion) cannot carry a
            // target_fingerprint, so with_data_based_criterion is used here.
            CalibrationStageSpec::from(500)
                .with_data_based_criterion(
                    &incidence_scorer_ref,
                    ScoreAcceptanceCriterion::threshold(80.0),
                    &incidence_target_ref,
                ),
            CalibrationStageSpec::from(500)
                .with_data_based_criterion(
                    &incidence_scorer_ref,
                    ScoreAcceptanceCriterion::threshold(40.0),
                    &incidence_target_ref,
                ),
            // Stage 2: PeakIncidenceScorer does not use target data.
            CalibrationStageSpec::from(500)
                .with_simulation_criterion(&peak_scorer_ref, ScoreAcceptanceCriterion::window(120.0, 160.0)),
        ])
        .max_proposals(50_000)
        // Counterfactuals are part of CalibrationManifest; parameter clashing with priors
        // is validated here at build time (§10). Variant labels are tracked in
        // StageState::counterfactual_labels (§3) after the run.
        .counterfactuals(CounterfactualGroup::iterator([
            ("low",    [("intervention_strength", 0.1_f64)]),
            ("medium", [("intervention_strength", 0.3_f64)]),
            ("high",   [("intervention_strength", 0.5_f64)]),
        ]))
        // add_target_data now takes a TargetRef<T> instead of a string label.
        // At build() time, Context writes into CalibrationManifest::target_refs:
        //   "observed_incidence" → {
        //     "builder_fingerprint": "<fingerprint of IncidenceTargetBuilder>",
        //     "input": { "csv_path": "data/incidence.csv", "trim_leading": 7 }
        //   }
        // At runtime (§15, Step 4, pre-loop), Context calls build(), caches the
        // Vec<f64> under incidence_target_ref.fingerprint, and dispatches it to
        // every scorer whose StageScorerSpec::target_fingerprint matches.
        .add_target_data(incidence_target_ref.clone())
        .prior_predictive_check(1_000, None)
        // All structural parameters including counterfactuals are fixed in CalibrationManifest.
        .build_and_run()
        .await?;

    // ---------------------------------------------------------------------------
    // 5. Diagnostics after calibration
    // ---------------------------------------------------------------------------

    // Prints a per-stage trajectory table and overall summary (ESS, acceptance rate, KL divergence).
    // diagnostics(None) returns the posterior leaf stage; pass Some(stage_id) for a specific stage.
    ctx.calibration_diagnostics().display();

    // ---------------------------------------------------------------------------
    // 6. Simulation: posterior crossed with a parameter sweep from CSV
    // ---------------------------------------------------------------------------

    // Each row of the CSV becomes a FlatParticle; columns are dot-delimited keys.
    // PreferRight lets sweep values override matching posterior keys.
    let sweep = ParticlePopulation::from_csv("data/intervention_sweep.csv")?;

    // .scorer_with_target() mirrors CalibrationStageSpec::with_data_based_criterion:
    // both accept a ScorerRef and a TargetRef, co-declaring the pairing via typed
    // identity tokens. The compiler enforces scorer T == target T. Context retrieves
    // the cached Vec<f64> built during calibration, guaranteeing identical target data.
    let sim = ctx.simulate()
        .from_posterior()                                          // uses the calibration DAG leaf
        .counterfactuals(sweep, MergeStrategy::PreferRight)        // overlay sweep onto each posterior particle
        .scorer_with_target(&incidence_scorer_ref, &incidence_target_ref)
        .run().await?;

    // ---------------------------------------------------------------------------
    // 7. Post-simulation scores and artifact references
    // ---------------------------------------------------------------------------

    // sim.scores: HashMap<ParticleId, HashMap<Fingerprint, ScoreValueType>>
    // sim.artifacts: Vec<ArtifactRef> retrievable via ArtifactStore using ArtifactRef::uri
    println!("Simulation ran {} particles", sim.population.entries.len());

    // manifest.posterior_artifacts: HashMap<StageId, ArtifactRef>
    // ArtifactRef::data_type carries the content-type string (e.g. "application/x-arrow").
    for (stage_id, artifact_ref) in &manifest.posterior_artifacts {
        println!("Posterior: {} ({})", artifact_ref.uri, artifact_ref.data_type);
    }

    Ok(())
}
```

---

[← Runtime Execution](16-runtime-execution.md) | [TOC](README.md)
