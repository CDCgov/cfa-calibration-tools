[← ScoreCalculator and TargetBuilder](08-scorecalculator-and-target-builder.md) | [TOC](README.md) | [Next: Counterfactual Model Construction →](10-counterfactual-model-construction.md)

---

## 9. ScoreAcceptanceCriterion
`ScoreAcceptanceCriterion` owns `evaluate` methods directly and are called during population filtering.

The criterion operates on `ScoreValueType` variants:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ScoreAcceptanceCriterion {
    /// Accept if score (Numeric) <= threshold.
    Threshold { threshold: f64 },
    /// Accept if score (Numeric) >= floor.
    Floor { floor: f64 },
    /// Accept if low <= score (Numeric) <= high.
    NumericWindow { low: f64, high: f64 },
    /// Accept if the score (Range) is fully contained in [low, high].
    RangeWindow { low: f64, high: f64 },
    /// All sub-criteria must accept.
    All(Vec<ScoreAcceptanceCriterion>),
    /// Any sub-criterion must accept.
    Any(Vec<ScoreAcceptanceCriterion>),
}

impl ScoreAcceptanceCriterion {
    /// Shorthand for `Threshold { threshold }`.
    pub fn threshold(threshold: f64) -> Self { Self::Threshold { threshold } }
    /// Shorthand for `Floor { floor }`.
    pub fn floor(floor: f64) -> Self { Self::Floor { floor } }
    /// Shorthand for `NumericWindow { low, high }`.
    pub fn window(low: f64, high: f64) -> Self { Self::NumericWindow { low, high } }
    /// Shorthand for `RangeWindow { low, high }`.
    pub fn range_window(low: f64, high: f64) -> Self { Self::RangeWindow { low, high } }

    pub fn evaluate(&self, score: &ScoreValueType) -> bool {
        match (self, score) {
            (Self::Threshold { threshold }, ScoreValueType::Numeric(v))       => v <= threshold,
            (Self::Floor     { floor },     ScoreValueType::Numeric(v))       => v >= floor,
            (Self::NumericWindow { low, high }, ScoreValueType::Numeric(v))   => v >= low && v <= high,
            (Self::RangeWindow  { low, high }, ScoreValueType::Range { low: sl, high: sh }) => {
                sl >= low && sh <= high
            }
            (Self::All(criteria), score) => criteria.iter().all(|c| c.evaluate(score)),
            (Self::Any(criteria), score) => criteria.iter().any(|c| c.evaluate(score)),
            _ => false, // type mismatch between criterion and score variant
        }
    }
}
```

Score calculators that return types other than `Numeric` or `Range` are not supported by the acceptance criterion, but could be reasonably added into the evaluation system.


#### Criterion Accumulation

By default, `build()` accumulates `(scorer, criterion)` pairs across stages so that every
subsequent stage must satisfy all criteria from all preceding stages. This guarantees monotone
posterior tightening for typical worfklows.

Particles accepted at a later stage are generally expected to also have been accepted at every earlier stage.
This is not necessarily the case, and can be overriden if the user choose to skip criterion accumulation during the build. Call `skip_score_criterion_accumulation()` on the builder to store each stage's declared
`scorer_criteria` verbatim, with no constraints inherited from earlier stages.
A reasonable use case to skip accumulation would be to prime a calibration with a plausibly reduced proposal distribution from the prior.

The accumulation algorithm runs in topological order:

1. Maintain a carried set keyed by scorer fingerprint: `HashMap<Fingerprint, ScoreAcceptanceCriterion>`. The fingerprint is taken from `StageScorerSpec::scorer_fingerprint` (set by `ScorerRef`-based construction) or derived from the scorer label for string-constructed specs.
2. For each stage, merge the stage's own `scorer_criteria` into the carried set by keeping the more restrictive score criteria for the same criterion type via automatic subsumption:

     | Criterion type    | More restrictive rule                                     |
     |---|---|
     | `Threshold { t }` | lower `t`                                                 |
     | `Floor { f }`     | higher `f`                                                |
     | `NumericWindow`   | intersection: `low = max(l₁, l₂)`, `high = min(h₁, h₂)` |
     | `RangeWindow`     | intersection of both bounds                               |

     If a `NumericWindow` or `RangeWindow` intersection produces `low > high`,
     `build()` returns `CalibrationBuildError::CriterionWindowCollapse`.

   - If the score caclulation remains the same but the criterion type changes, then both are retained and the particle must satisfy
     `ScoreAcceptanceCriterion::All([carried_criterion, new_criterion])`.

3. The `scorer_criteria` written into `StageNode` is the full carried set at the end of step 2
   (all accumulated pairs, not only the stage's own declared ones).

---

[← ScoreCalculator and Target Builder](08-scorecalculator-and-target-builder.md) | [TOC](README.md) | [Next: Counterfactual Model Construction →](10-counterfactual-model-construction.md)
