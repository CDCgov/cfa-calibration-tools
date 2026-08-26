[← ModelRunner and Python Interop](12-modelrunner.md) | [TOC](README.md) | [Next: Error Propagation and Stage Resumability →](14-error-propagation-and-stage-resumability.md)

---

## 13. Seeds and RNG Discipline

Stochastic models expose a **seed parameter name** that is registered with `Context`. The perturbation kernel uses `PerturbationType::Seed` for that parameter, which causes `Context` to inject a deterministically derived seed at parse time rather than treating it as a free continuous parameter.

The default RNG is **ChaCha20** via `rand_chacha::ChaCha20Rng`, seeded with a 32-byte value derived through blake3 keyed hashing. This may be overwritten by incorporation of methods from the `ilia` crate.

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub type Seed = [u8; 32];

/// Derive a deterministic seed for a given particle index and replicate index.
///
/// - Multiple replicates:  same seed per particle, different per replicate
///   (particles are compared under identical stochastic conditions).
/// - Single replicate:     different seed per particle.
pub fn derive_seed(base: &Seed, particle_idx: u64, replicate_idx: u64) -> Seed {
    let mut input = [0u8; 16];
    input[..8].copy_from_slice(&particle_idx.to_le_bytes());
    input[8..].copy_from_slice(&replicate_idx.to_le_bytes());
    *blake3::keyed_hash(base, &input).as_bytes()
}

pub fn make_rng(base: &Seed, particle_idx: u64, replicate_idx: u64) -> ChaCha20Rng {
    ChaCha20Rng::from_seed(derive_seed(base, particle_idx, replicate_idx))
}
```

`derive_seed` is a one-to-one mapping over $\mathbb{N} \times \mathbb{N}$: no two distinct $(i, r)$ pairs produce the same seed for a fixed $K$. Within `random_seeds` all seeds are distinct (different $i$, fixed $r = 0$). Within `replicate_seeds` all seeds are distinct across replicates (different $r$, fixed second argument $0$) and all particles within a replicate intentionally share the same seed. Across modes, `random_seeds` uses index $o + i$ and `replicate_seeds` uses index $o + r$.

### Seed Arithmetic

Let $K$ denote the 32-byte base seed and $H(K, m) = \text{BLAKE3\_keyed\_hash}(K, m)$. All seed derivation reduces to two equations over the tuple $(K, i, r)$ where $i$ is a particle index and $r$ is a replicate index.

**Distinct seeds: `ParticlePopulation::random_seeds` / `SimulationBuilder::random_particles`**

Particles are sorted lexicographically by `ParticleId`. The particle at sorted position $i$
(with context global offset $o \ge 0$) receives a unique seed

$$s_i = H\!\bigl(K,\; \mathrm{le64}(o + i) \;\|\; \mathrm{le64}(0)\bigr)$$.

The replicate index is fixed at $0$, so no two particles at distinct positions share a seed.
`Context::current_proposal_offset()` supplies $o$. The offset is adjusted based on the number of proposals or runs have already occurred, equal to $n\_proposed$ during a calibration
run (making $s_i$ globally unique across all batches) and $0$ for external `SimulationBuilder`
calls.  Callers never pass the offset directly; it is derived automatically by
`SimulationBuilder::random_particles`, which reads it from `Context` and forwards it to the
private `random_particles_with_offset`.

**Particle replicates: `ParticlePopulation::replicate_seeds` / `SimulationBuilder::replicate_counterfactuals`**

For a batch of $n$ particles, replicate count $R$, and global offset $o$, all particles
within the same replicate $r$ share an identical seed, enabling reproducible comparison
of counterfactuals under the same stochastic conditions:

$$s_r = H\!\bigl(K,\; \mathrm{le64}(o + r) \;\|\; \mathrm{le64}(0)\bigr)$$

Every particle in replicate $r$ receives the same seed, so counterfactual differences
reflect only parameter variation, not RNG variation.  `Context::current_proposal_offset()`
supplies $o$.
`SimulationBuilder::replicate_counterfactuals` reads the offset from `Context` and forwards
it to the private `replicate_counterfactuals_with_offset`. The output population of `ParticlePopulation::replicate_seeds` has
$n \times R$ entries.

Seeds for replicate runs are injected into particles via `ParticlePopulation::product` or `ParticlePopulation::union` ([§6](06-perturbationkernel-and-density-convention.md)). The seed population (one particle per seed, containing only the seed key) is joined through with the proposal population before parsing and running.

```rust
pub trait SeedRandomizationExt for ParticlePopulation {
    /// Assign a distinct derived seed to each particle, sorted lexicographically by
    /// `ParticleId`.  The particle at sorted position `i` receives:
    ///
    ///   s_i = H(K, le64(offset + i) ‖ le64(0))
    ///
    /// where `H` is BLAKE3 keyed hash, `K` is `base`, and `‖` denotes byte
    /// concatenation.  Existing values for `seed_param` are preserved
    /// (`MergeStrategy::PreferLeft`).  The output population has the same number
    /// of entries as `self`.
    ///
    /// Called by `SimulationBuilder::random_particles_with_offset`, which supplies
    /// the offset derived from `Context::current_proposal_offset` (§3).  Callers
    /// outside `SimulationBuilder` should obtain the offset from `Context` rather
    /// than hardcoding `0` or `n_proposed`.
    pub fn random_seeds(
        &self,
        base:       &Seed,
        seed_param: &str,
        offset:     u64,
    ) -> ParticlePopulation {
        let mut sorted_ids: Vec<&ParticleId> = self.entries.keys().collect();
        sorted_ids.sort();
        let seed_pop: ParticlePopulation = sorted_ids.iter()
            .enumerate()
            .map(|(i, particle_id)| {
                let seed_val = derive_seed(base, offset + i as u64, 0);
                let p = FlatParticle(BTreeMap::from([
                    (seed_param.to_string(), serde_json::json!(seed_val as i64)),
                ]));
                ((*particle_id).clone(), WeightedParticle { particle: p, log_weight: 0.0 })
            })
            .collect::<HashMap<_, _>>()
            .into();
        ParticlePopulation::union(self, &seed_pop, MergeStrategy::PreferLeft)
    }

    /// Expand every particle into `n_replicates` variants.  All particles within
    /// the same replicate `r` share an identical derived seed, enabling
    /// reproducible comparison of counterfactuals under the same stochastic
    /// conditions.  The seed for replicate `r` is:
    ///
    ///   s_r = H(K, le64(offset + r) ‖ le64(0))
    ///
    /// where `H` is BLAKE3 keyed hash, `K` is `base`, and `‖` denotes byte
    /// concatenation.  Every particle in replicate `r` receives the same seed,
    /// so counterfactual differences reflect only parameter variation.
    ///
    /// Existing values for `seed_param` are preserved (`MergeStrategy::PreferLeft`).
    /// The output population has `|self| × n_replicates` entries, one per
    /// (particle, replicate) pair.
    ///
    /// Called by `SimulationBuilder::replicate_counterfactuals_with_offset`, which
    /// supplies the offset derived from `Context::current_proposal_offset` (§3).
    /// Callers outside `SimulationBuilder` should obtain the offset from `Context`
    /// rather than hardcoding `0` or `n_proposed`.
    pub fn replicate_seeds(
        &self,
        base:         &Seed,
        seed_param:   &str,
        n_replicates: usize,
        offset:       u64,
    ) -> ParticlePopulation {
        let mut entries = HashMap::new();
        for particle_id in self.entries.keys() {
            let single = ParticlePopulation {
                entries: HashMap::from([(*particle_id, self.entries[*particle_id].clone())]),
            };
            let seed_pop: ParticlePopulation = (0..n_replicates)
                .map(|r| {
                    let seed_val = derive_seed(base, offset + r as u64, 0);
                    let p  = FlatParticle(BTreeMap::from([
                        (seed_param.to_string(), serde_json::json!(seed_val as i64)),
                    ]));
                    let id = p.fingerprint();
                    (id, WeightedParticle { particle: p, log_weight: 0.0 })
                })
                .collect::<HashMap<_, _>>()
                .into();
            // PreferLeft: existing seed keys are preserved; absent seed keys receive the derived value
            entries.extend(
                ParticlePopulation::product(&single, &seed_pop, MergeStrategy::PreferLeft).entries,
            );
        }
        ParticlePopulation { entries }
    }
}
```


---

[← ModelRunner and Python Interop](12-modelrunner.md) | [TOC](README.md) | [Next: Error Propagation and Stage Resumability →](14-error-propagation-and-stage-resumability.md)
