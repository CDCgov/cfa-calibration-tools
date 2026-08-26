[← Particle Type and Overlay Semantics](04-particle-type-and-overlay-semantics.md) | [TOC](README.md) | [Next: PerturbationKernel and Density Convention →](06-perturbationkernel-and-density-convention.md)

---

## 5. ParticlePopulation, Weights, and Product

`ParticlePopulation` holds particles and their log-weights. It does **not** hold score values, acceptance, or other metadata associated with calibration; scores are owned by the `StageState` of the DAG node that produced them (see [§3](03-simulation-system-and-dag-stages.md)) and passed in explicitly when filtering.

```rust
use std::collections::HashMap;

pub type ParticleId = Fingerprint;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedParticle {
    pub particle:    FlatParticle,
    pub log_weight:  f64, // always stored in log-space
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticlePopulation {
    pub entries: HashMap<ParticleId, WeightedParticle>,
}

impl ParticlePopulation {
    pub fn new() -> Self {
        ParticlePopulation { entries: HashMap::new() }
    }
    /// Load a population from a CSV file. All particles receive uniform log-weights.
    pub fn from_csv(path: impl AsRef<std::path::Path>) -> Result<Self, CsvLoadError>;
    /// Return a read-only view of every particle in this population.
    /// Use this when the full population is needed as a `PopulationSlice`
    /// (e.g. as the `prev` argument to `assign_weights`, or as input to
    /// `PerturbationType::adapt`).
    pub fn as_slice(&self) -> PopulationSlice<'_> {
        PopulationSlice { particles: self.entries.values().collect() }
    }
}

/// A lifetime-bound, read-only view into a ParticlePopulation.
pub struct PopulationSlice<'a> {
    pub particles: Vec<&'a WeightedParticle>,
}

impl<'a> PopulationSlice<'a> {
    /// Clone the referenced particles into a new owned ParticlePopulation.
    pub fn to_owned_population(&self) -> ParticlePopulation;
}
```

### Weight Assignment

`assign_weights` has a single signature. When `prev` is `None`, uniform log-weights are assigned. When `prev` is provided, the full ABC-SMC formula is applied in log-space:

$$w_i^{(t)} \propto \frac{\pi(\theta_i^{(t)})}{\sum_j w_j^{(t-1)} \cdot q(\theta_i^{(t)} \mid \theta_j^{(t-1)})}$$

```rust
impl ParticlePopulation {
    /// Assign normalized weights.
    /// - prev = None  → uniform weights (log(1/n) each)
    /// - prev = Some  → ABC-SMC importance weights in log-space
    pub fn assign_weights(
        &mut self,
        prior_log_prob: impl Fn(&FlatParticle) -> f64,
        perturbation_weight_calculator:           Option<(&PopulationSlice<'_>, &dyn PerturbationKernel)>,
    ) {
        if let Some((slice, kernel)) = perturbation_weight_calculator {
            for entry in self.entries.values_mut() {
                let log_prior = prior_log_prob(&entry.particle);
                let log_denom = log_sum_exp(
                    slice.particles.iter().map(|wp| {
                        wp.log_weight + kernel.log_density(&wp.particle, &entry.particle)
                    })
                );
                entry.log_weight = log_prior - log_denom;
            }
            self.normalize_log_weights();
        } else {
            let log_w = -(self.entries.len() as f64).ln();
            for entry in self.entries.values_mut() {
                entry.log_weight = log_w;
            }
        }
    }

    fn normalize_log_weights(&mut self) {
        let log_total = log_sum_exp(self.entries.values().map(|e| e.log_weight));
        for entry in self.entries.values_mut() {
            entry.log_weight -= log_total;
        }
    }

    /// Cartesian product of two populations.
    /// Each particle from `left` is overlaid with each particle from `right`.
    /// Weights are the sum of log-weights of each pair (un-normalized).
    pub fn product(
        left:     &ParticlePopulation,
        right:    &ParticlePopulation,
        strategy: MergeStrategy,
    ) -> ParticlePopulation {
        let mut entries = HashMap::new();
        for (_, lwp) in &left.entries {
            for (_, rwp) in &right.entries {
                let particle   = FlatParticle::overlay(&lwp.particle, &rwp.particle, strategy);
                let log_weight = lwp.log_weight + rwp.log_weight;
                let id         = particle.fingerprint();
                entries.insert(id, WeightedParticle { particle, log_weight });
            }
        }
        ParticlePopulation { entries }
    }

    /// One-to-one merge of two equal-length populations, matched by `ParticleId`.
    /// Both populations must contain identical key sets; panics if they differ.
    /// Each matched pair is merged via `FlatParticle::overlay(left, right, strategy)`,
    /// and the left-side `ParticleId` is preserved in the output.
    ///
    /// Unlike `product`, which produces `|left| × |right|` entries, `union` produces
    /// exactly `|left|` entries — one per particle, each merged with its uniquely paired
    /// counterpart. This is the correct primitive for assigning one distinct seed to each
    /// proposal in a single-replicate stochastic run (§16, Step 2).
    pub fn union(
        left:     &ParticlePopulation,
        right:    &ParticlePopulation,
        strategy: MergeStrategy,
    ) -> ParticlePopulation {
        assert_eq!(
            left.entries.len(), right.entries.len(),
            "union: populations must have identical key sets"
        );
        let mut entries = HashMap::new();
        for (id, lwp) in &left.entries {
            let rwp = right.entries.get(id)
                .expect("union: right population is missing a ParticleId present in left");
            let particle   = FlatParticle::overlay(&lwp.particle, &rwp.particle, strategy);
            let log_weight = lwp.log_weight;
            entries.insert(id.clone(), WeightedParticle { particle, log_weight });
        }
        ParticlePopulation { entries }
    }


    /// Filter by all (scorer, criterion) pairs in `scorer_criteria`.
    /// A particle is accepted only when every pair's criterion is satisfied by the
    /// corresponding scorer's score in `scores`. Particles missing a score for any
    /// scorer are rejected. Returns a slice of the first `limit` accepted particles.
    pub fn filter<'a>(
        &'a self,
        scores:          &'a HashMap<ParticleId, HashMap<Fingerprint, ScoreValueType>>,
        scorer_criteria: &[StageScorerEntry],
        limit:           Option<usize>,
    ) -> PopulationSlice<'a> {
        let mut accepted: Vec<&WeightedParticle> = self.entries
            .iter()
            .filter(|(id, _)| {
                let particle_scores = match scores.get(*id) {
                    Some(s) => s,
                    None    => return false,
                };
                scorer_criteria.iter().all(|entry| {
                    particle_scores
                        .get(&entry.scorer_fingerprint)
                        .map_or(false, |s| entry.criterion.evaluate(s))
                })
            })
            .map(|(_, wp)| wp)
            .collect();
        if let Some(n) = limit { accepted.truncate(n); }
        PopulationSlice { particles: accepted }
    }
}

fn log_sum_exp(iter: impl Iterator<Item = f64>) -> f64 {
    let vals: Vec<f64> = iter.collect();
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() { return f64::NEG_INFINITY; }
    max + vals.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}
```

The **final accepted `PopulationSlice`** for a completed calibration stage may be cloned into a new `ParticlePopulation` for use as stage results in subsequent stages. That population is re-weighted using a `PopulationSlice` view from the previous generation as `prev`.

Resampling between stages is **sampling from a `PopulationSlice`** of the previous step.

---

[← Particle Type and Overlay Semantics](04-particle-type-and-overlay-semantics.md) | [TOC](README.md) | [Next: PerturbationKernel and Density Convention →](06-perturbationkernel-and-density-convention.md)
