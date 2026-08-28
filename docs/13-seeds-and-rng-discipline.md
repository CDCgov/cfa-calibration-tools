[← ModelRunner and Python Interop](12-modelrunner.md) | [TOC](README.md) | [Next: Error Propagation and Stage Resumability →](14-error-propagation-and-stage-resumability.md)

---

## 13. Seeds and RNG Discipline

Stochastic models expose a **seed parameter name** that is registered with `Context`. The perturbation kernel uses `PerturbationType::Seed` for that parameter, which causes `Context` to inject a deterministically derived seed at parse time rather than treating it as a free continuous parameter.

`Context` holds `base_entropy` (a `u64`) as the sole RNG input. Stage-local RNG streams are created by `Context::instantiate_stage_rng` at each stage's pre-loop setup and stored in `StageState.rngs` — the runtime state for that DAG node, owned by `Context::stage_states`. `SimulationBuilder` resolves the active stage's streams via `ctx.stage_states[ctx.active_stage_id].rngs`, leaving no mutable RNG state as direct `Context` fields.

**Stage isolation.** Each stage derives its RNG streams independently, so they carry no memory of draws made by any other stage. The meaningful dependency between sequential stages is carried entirely through the accepted particle population. Stage N+1 resamples from stage N's accepted particles, not from its RNG stream. This means that changing stage 1's criterion doesn't shift stage 2's RNG streams, but would alter proposed values because of changes in the original stage's accepted popualtion. Parallel DAG branches (two stages with the same parent) each derive from their own independent streams and may execute concurrently without contention.

Each stream advances independently within its stage — prior draws never interfere with perturbation draws. The entire system is reproducible from `base_entropy` alone.

### `RngSnapshot`

`RngSnapshot` captures the intra-stage position of the four stage-local `StdRng` streams at checkpoint time. `StdRng` serializes its full state when `rand` is built with the `serde1` feature. Snapshots are written into every `StageCheckpoint` (§14) and unpacked into local variables on resume.

```rust
/// Serializable snapshot of the four stage-local RNG streams.
/// Stored in `StageState.rngs` during execution; written into `StageCheckpoint` at checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RngSnapshot {
    pub prior:    StdRng,
    pub resample: StdRng,
    pub perturb:  StdRng,
    pub seed:     StdRng,
}

impl RngExt for Context {
    pub fn get_stage_rngs(&mut self, &StageId) -> &mut RngSnapshot {
        self.stage_states.get_mut(&node.id).unwrap().rngs
    }
}
```

Context carries its own runtime RNG snapshot for simulating and replicating values outside of a given stage.

**Checkpoint resumption.** When resuming a partially-completed calibration, the stage-local RNGs are restored directly from the `RngSnapshot` stored in the latest `StageCheckpoint` (§14), overwriting the freshly-derived streams.

Restoring is O(1) and exact regardless of how many proposals were already issued.

### Seeding particles

There are two tandard methods for filling out seed parameter values in a population. `random_seeds` fills in a unique seed for each particle in a population while `replicate_seeds` makes a cartesian product of the population with a new set of seeds.

To yield random seeds, particles are sorted lexicographically by `ParticleId`. The particle at sorted position $i$ receives a consistent seed replacement from the incremented RNG.

For a batch of $n$ particles and replicate count $R$, all particles
within the same replicate $r$ share an identical seed, enabling reproducible comparison
of counterfactuals under the same stochastic conditions.

Seeds for replicate runs are injected into particles via `ParticlePopulation::product` or `ParticlePopulation::union` ([§6](06-perturbationkernel-and-density-convention.md)). The seed population (one particle per seed, containing only the seed key) is joined through with the proposal population before parsing and running.

```rust
pub trait SeedRandomizationExt for ParticlePopulation {
    /// Assign a distinct seed to each particle by drawing from `rng`, with particles
    /// visited in sorted `ParticleId` order.  Each draw advances `rng` by one step,
    /// so the caller's RNG state reflects all seeds issued when the method returns.
    ///
    /// Existing values for `seed_param` are preserved (`MergeStrategy::PreferLeft`).
    /// The output population has the same number of entries as `self`.
    ///
    /// Called by `SimulationBuilder::randomize_seeds`, which receives `rng` as an explicit
    /// `&mut StdRng` parameter from the stage execution loop.
    pub fn random_seeds(
        &self,
        rng:        &mut StdRng,
        seed_param: &str,
    ) -> ParticlePopulation {
        let mut sorted_ids: Vec<&ParticleId> = self.entries.keys().collect();
        sorted_ids.sort();
        let seed_pop: ParticlePopulation = sorted_ids.iter()
            .map(|particle_id| {
                let seed_val = rng.next_u64();
                let p = FlatParticle(BTreeMap::from([
                    (seed_param.to_string(), serde_json::json!(seed_val)),
                ]));
                ((*particle_id).clone(), WeightedParticle { particle: p, log_weight: 0.0 })
            })
            .collect::<HashMap<_, _>>()
            .into();
        ParticlePopulation::union(self, &seed_pop, MergeStrategy::PreferLeft)
    }

    /// Expand every particle into `n_replicates` variants.  Draws exactly `n_replicates`
    /// seeds from `rng` — one per replicate — then assigns the same seed to every particle
    /// in that replicate.  Each draw advances `rng` by one step.
    ///
    /// Existing values for `seed_param` are preserved (`MergeStrategy::PreferLeft`).
    /// The output population has `|self| × n_replicates` entries, one per
    /// (particle, replicate) pair.
    ///
    /// Called by `SimulationBuilder::replicate_counterfactuals`, which receives `rng` as an
    /// explicit `&mut StdRng` parameter from the stage execution loop.
    pub fn replicate_seeds(
        &self,
        rng:          &mut StdRng,
        seed_param:   &str,
        n_replicates: usize,
    ) -> ParticlePopulation {
        // Draw one seed per replicate up front so every particle in a replicate
        // receives the same value regardless of iteration order.
        let replicate_seed_vals: Vec<u64> = (0..n_replicates).map(|_| rng.next_u64()).collect();

        let mut entries = HashMap::new();
        for particle_id in self.entries.keys() {
            let single = ParticlePopulation {
                entries: HashMap::from([(*particle_id, self.entries[*particle_id].clone())]),
            };
            let seed_pop: ParticlePopulation = replicate_seed_vals.iter()
                .map(|&seed_val| {
                    let p  = FlatParticle(BTreeMap::from([
                        (seed_param.to_string(), serde_json::json!(seed_val)),
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
