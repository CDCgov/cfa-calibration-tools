[← Simulation System and DAG Stages](03-simulation-system-and-dag-stages.md) | [TOC](README.md) | [Next: ParticlePopulation, Weights, and Product →](05-particlepopulation-weights-and-product.md)

---

## 4. Particle Type and Overlay Semantics

`FlatParticle` is the primary user-facing type. It holds a flat map of otherwise nested parameter keys to values (`f64`, `i64`, `String`, or homogeneous `Vec` of these). Falt particle keys may be full dot-delimited paths (`"model.transmission.rate"`) or any unique suffix thereof (`"transmission.rate"` or `"rate"` when unambiguous). `NestedSuffixParser` resolves uniquely identifiable partial keys to full paths at parse time via a precomputed suffix index (see [§7](07-nestedsuffixparser-and-particleerror.md)). Users construct, combine, fingerprint, and serialize `FlatParticle`s directly for use in priors, perturbation overrides, and counterfactual variants.

```rust
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Primary user-facing particle type: a flat map of dot-delimited keys to values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlatParticle(pub BTreeMap<String, Value>);

/// How to resolve key conflicts when overlaying two particles.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MergeStrategy {
    PreferLeft,
    PreferRight,
}

impl FlatParticle {
    /// Flat union of two particles. Shared keys resolved by strategy.
    ///
    /// Because all keys are already flat paths, this is a plain map merge.
    /// To partially override a nested structure supply only the keys that change:
    ///
    ///   left:  {"model.rate": 0.5, "model.mean": 1.0}
    ///   right: {"model.rate": 0.8}
    ///   out (PreferRight): {"model.rate": 0.8, "model.mean": 1.0}
    pub fn overlay(left: &Self, right: &Self, strategy: MergeStrategy) -> Self {
        let mut result = left.0.clone();
        for (k, v) in &right.0 {
            match strategy {
                MergeStrategy::PreferLeft  => { result.entry(k.clone()).or_insert(v.clone()); }
                MergeStrategy::PreferRight => { result.insert(k.clone(), v.clone()); }
            }
        }
        FlatParticle(result)
    }

    /// Stable fingerprint using blake3 over the canonical flat map.
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint::of_serializable(&self.0)
    }
}

```

---

[← Simulation System and DAG Stages](03-simulation-system-and-dag-stages.md) | [TOC](README.md) | [Next: ParticlePopulation, Weights, and Product →](05-particlepopulation-weights-and-product.md)
