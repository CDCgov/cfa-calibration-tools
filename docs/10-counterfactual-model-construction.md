[← ScoreAcceptanceCriterion](09-scoreacceptancecriterion.md) | [TOC](README.md) | [Next: Fingerprinting and Caching Strategy →](11-fingerprinting-and-caching-strategy.md)

---

## 10. Counterfactual Model Construction

**Assumption** is a glossary term used in documentation and in the `Context` builder API to describe the set of values inside a model configuration. Counterfactuals are assumptions that express contradictory structural or value choices (e.g., two different transmission models, two different population input files, or two different transmission rates). An assumption is not a type in the codebase.

Counterfactuals are declared as a `CounterfactualGroup`, a named collection of `FlatParticle` overrides paired with a `CounterfactualMode` that governs how they are applied during a run.

```rust
/// A named collection of counterfactual variant particles and the mode in which
/// they are applied during a calibration or simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualGroup {
    pub mode:     CounterfactualMode,
    pub variants: Vec<(String, FlatParticle)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CounterfactualMode {
    /// Each variant is applied independently after perturbation and before parsing.
    /// Each proposed particle is cloned once per variant and merged via
    /// `FlatParticle::overlay(proposed, variant, PreferRight)`.
    /// Produces one score and (if accepted) one posterior entry per variant per particle.
    /// Use when the model is run separately under each assumption.
    /// /// Applicable to both a calibration stage and a calibration workflow, as well as a plain `simulate()` call.
    Iterator,
    /// Variants are co-calibrated: a `ModelSelector` kernel entry is injected
    /// automatically into the calibration kernel, proposing the variant index as a
    /// discrete parameter with equal weights. `Context` resolves the integer index to the appropriate
    /// variant overlay at parse time.
    /// Use when calibration should resolve which counterfactual best fits the data.
    Selector,
}

impl CounterfactualGroup {
    /// Build an iterator-mode group from `(label, overrides)` pairs.
    pub fn iterator(
        variants: impl IntoIterator<
            Item = (&'static str, impl IntoIterator<Item = (&'static str, impl Into<serde_json::Value>)>),
        >,
    ) -> Self;

    /// Build a selector-mode group.
    /// A `ModelSelector` kernel entry for `n_variants = variants.len()` is injected
    /// into the effective kernel by `CalibrationBuilder::build`.
    pub fn selector(
        variants: impl IntoIterator<
            Item = (&'static str, impl IntoIterator<Item = (&'static str, impl Into<serde_json::Value>)>),
        >,
    ) -> Self;
}
```

`CounterfactualGroup` is declared on `CalibrationBuilder::counterfactuals` ([§2.1](02-calibrator-construction.md#21-contextcalibrationext--calibration-declaration)) and stored in `CalibrationManifest::counterfactuals`. It is part of the static calibration ruleset; all variant definitions, mode selection, and parameter-clash validation occur at build time.

**Iterator vs Selector guidance:**
- **Iterator** is the common case. The calibration runs once and the posterior is projected under each assumption independently, aiming to accept the number of `target_accepted` particles for each variant. `CalibrationBuilder::build` validates that no variant key clashes with a calibrated prior parameter. Counterfactual identity is tracked in `StageState::counterfactual_labels` ([§3](03-simulation-system-and-dag-stages.md)) so per-variant posteriors can be extracted via the `counterfactual_variant` label of `Context::current_population` ([§2](02-calibrator-construction.md)) after the run.
- **Selector** is used when variants represent competing model structures that calibration should resolve. The `ModelSelector` kernel skips over the variant parameters. Weighted acceptance pressure drives the posterior toward the better-fitting variant(s). The variant count is known at build time, so the prior declaration and kernel injection ([§6](06-perturbationkernel-and-density-convention.md)) occurs during `CalibrationBuilder::build`. The calibrated variant weights are readable from `StageState::scores` after the run.

---

[← ScoreAcceptanceCriterion](09-scoreacceptancecriterion.md) | [TOC](README.md) | [Next: Fingerprinting and Caching Strategy →](11-fingerprinting-and-caching-strategy.md)
