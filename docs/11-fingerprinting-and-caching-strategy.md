[← Counterfactual Model Construction](10-counterfactual-model-construction.md) | [TOC](README.md) | [Next: ModelRunner and Python Interop →](12-modelrunner.md)

---

## 11. Fingerprinting and Caching Strategy

All fingerprinting is managed by `Context` ([§2](02-calibrator-construction.md)). User-supplied components (scorers, target data builders) are registered with `Context` under a **string label** for audit and display, but the cache key is the component's **content fingerprint**, not its label. This section describes the fingerprint infrastructure, how content fingerprints are derived, and the policy for components whose implementations cannot be automatically hashed.

### 11.1 `Fingerprint` Primitive

The `Fingerprint` type is a blake3 hex, 64 characters, and is stable across processes.

```rust
/// A stable opaque fingerprint (blake3 hex, 64 chars).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fingerprint(pub String);

impl Fingerprint {
    pub fn of_bytes(data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        Fingerprint(hash.to_hex().to_string())
    }

    /// Fingerprint a serializable value.
    /// Uses canonical serde_json (BTreeMap key order) as the encoding before hashing.
    pub fn of_serializable<T: Serialize>(value: &T) -> Self {
        let bytes = serde_json::to_vec(value).expect("serialization is infallible for registered types");
        Self::of_bytes(&bytes)
    }

    /// Combine multiple fingerprints into one (e.g., scorer + target + output).
    pub fn combine(parts: &[&Fingerprint]) -> Self {
        let mut hasher = blake3::Hasher::new();
        for p in parts {
            hasher.update(p.0.as_bytes());
        }
        Fingerprint(hasher.finalize().to_hex().to_string())
    }
}
```

### 11.2 `ComponentFingerprint` — Two-Track Identity

Component identity uses two tracks depending on whether the component is a serializable struct or a closure.

#### Track A: Struct-based components (primary path)

Any scorer or target builder defined as a `Serialize`-able struct derives `ComponentFingerprint` via a proc macro. The derived implementation combines the fully-qualified type name with the serialized field values:

```rust
pub trait ComponentFingerprint {
    /// A content-addressed fingerprint that changes when the type name or
    /// any serialized field value changes.
    fn component_fingerprint(&self) -> Fingerprint;
}

// Proc macro derive — requires the type to also impl Serialize.
// #[derive(ComponentFingerprint, Serialize)]
// expands to:
//
// fn component_fingerprint(&self) -> Fingerprint {
//     Fingerprint::combine(&[
//         &Fingerprint::of_bytes(std::any::type_name::<Self>().as_bytes()),
//         &Fingerprint::of_serializable(self),
//     ])
// }
```

Renaming the struct or changing a serialized field without a `#[serde(rename = "...")]` attribute changes the fingerprint and forces a cache miss, but field order does not matter. The macro documentation states this invariance explicitly so users know that structural renames require cache invalidation or a `#[serde(rename)]` guard to preserve identity.


#### Track B: Closure components (escape hatch)

Fingerprinting relies on `Versioned<F>` when a scorer or target builder cannot be expressed as a serializable struct, such as when a component has anonymous function captures, external callbacks, or dynamically composed logic. `Versioned<F>` wraps a closure with an explicit `label` and `version: u32`. The component fingerprint is derived from label + version only; the user is responsible for bumping `version` whenever the logic changes.

`Versioned<F>` is a single shared wrapper used for both scorers and target builders. It carries no phantom type parameters — only the concrete closure type `F`. The applicable trait (`ScoreCalculator` or `TargetBuilder`) is determined entirely by the bounds on `F` at the impl site.

```rust
pub struct Versioned<F> {
    pub label:       &'static str,
    /// Bump this integer whenever the function's logic changes. The cache key
    /// is derived from (label, version). Forgetting to bump causes stale cache hits.
    pub version:     u32,
    pub description: &'static str,
    f: F,
}

impl<F> Versioned<F> {
    pub fn new(label: &'static str, version: u32, description: &'static str, f: F) -> Self {
        Self { label, version, description, f }
    }
}

/// Single ComponentFingerprint impl shared by both scorer and target-builder uses.
impl<F> ComponentFingerprint for Versioned<F> {
    fn component_fingerprint(&self) -> Fingerprint {
        Fingerprint::of_serializable(&(self.label, self.version))
    }
}

/// Scorer use: F must match the ScoreCalculator signature.
impl<GQ, T, F> ScoreCalculator for Versioned<F>
where
    GQ: GeneratedQuantity,
    T: Target,
    F: Fn(&[GQ], &[T]) -> Result<Option<ScoreValueType>, ScoringError>
        + Send + Sync + 'static,
{
    // documentation() returns a ScoreDocumentation built from self.description.
}
```

The `version` field is named and required at construction to make omitting a version bump a visible code-review concern rather than a silent default. There is no compile-time enforcement of version discipline for closures; this is documented as a known limitation of Track B.

### 11.3 Registration and Label Separation

`ContextRegistrationExt` stores the **label** and the **component fingerprint** separately. The label is for audit, display, and manifest entries. The fingerprint is the cache key.

```rust
pub trait ContextRegistrationExt {
    /// Register a scorer under a human-readable label. Returns a `ScorerRef<S::GQ, S::T>`
    /// whose `fingerprint` is the scorer's `component_fingerprint()`. Pass the returned ref
    /// to `CalibrationBuilder::default_scorer` or to `StageScorerSpec` constructors for
    /// compile-time type-checked wiring (§2.1, §8).
    /// The component fingerprint is stored in `component_registry`; the label is stored
    /// separately in `component_labels` for audit and manifest display.
    /// Registering two scorers whose `component_fingerprint()` collides is a
    /// build-time error (`RegistrationError::FingerprintCollision`).
    fn register_scorer<S: ScoreCalculator + 'static>(&mut self, label: &str, scorer: S) -> ScorerRef<S::GQ, S::T>;

    /// Register target data under a label. Returns a `TargetRef<T>` whose
    /// `fingerprint` is `Fingerprint::of_serializable(&target)`. Pass the returned
    /// ref to `CalibrationBuilder::add_target_data` and, optionally, to
    /// `StageScorerSpec` to pin a scorer to this specific target (§2.1, §8.1).
    fn register_target<T: Target + ComponentFingerprint + 'static>(
        &mut self,
        label: &str,
        target: T,
    ) -> TargetRef<T>;

    /// Register a target builder and its raw input under a label. Returns a
    /// `TargetRef<B::T>` whose `fingerprint` is
    /// `Fingerprint::combine(&[&builder_fp, &input_fp])`.
    /// `TargetBuilder::build` is not called at registration time; it is deferred
    /// to the first calibration stage that requests the target, or to the first
    /// call to `Context::build_target` (§8.1). Pass the returned ref to
    /// `CalibrationBuilder::add_target_data` and optionally to `StageScorerSpec`
    /// for explicit scorer–target pairing (§2.1, §8.1).
    /// Registering two builders whose `component_fingerprint()` collides is a
    /// build-time error (`RegistrationError::FingerprintCollision`).
    fn register_target_builder<B: TargetBuilder + 'static>(
        &mut self,
        label: &str,
        builder: B,
        input: B::Input,
    ) -> TargetRef<B::T>;
}

impl ContextRegistrationExt for Context { /* ... */ }
```

`Context` stores two parallel maps:
- `component_registry: HashMap<String, Fingerprint>` — label → fingerprint (for label-based lookup during `CalibrationBuilder` stage wiring)
- `component_labels: HashMap<Fingerprint, String>` — fingerprint → label (for audit output and manifest display)


`TargetBuilder` participates in the same two-track fingerprint system as scorers ([§11.2](11-fingerprinting-and-caching-strategy.md#112-componentfingerprint--two-track-identity)). Struct-based builders derive `ComponentFingerprint` automatically (Track A). Closure-based builders use `Versioned<F>` (Track B), the same wrapper introduced for scorers. No separate type or constructor is needed — the `TargetBuilder` impl is simply an additional impl on `Versioned<F>` gated on the target-builder closure signature:

```rust
/// Target-builder use: F must match the TargetBuilder signature.
impl<I, T, F> TargetBuilder for Versioned<F>
where
    I: Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    T: Target,
    F: Fn(&I) -> Result<T, TargetBuildError> + Send + Sync + 'static,
{
    type Input = I;
    type T = T;
    fn build(&self, input: &I) -> Result<T, TargetBuildError> { (self.f)(input) }
}
```

The same `Versioned::new` constructor, `ComponentFingerprint` impl, and version-bump discipline (§11.2) apply without repetition.

### 11.4 Cache Table Summary

Cache keys use component fingerprints, not labels. Labels appear only in diagnostic and manifest display paths.

| Cache | Key | Value |
|---|---|---|
| Target cache | target builder component fingerprint + target input fingerprint | `TargetType` or artifact ref |
| Model cache | runner component fingerprint + canonical `ModelInput` fingerprint + seed + output scope | `ModelOutput` or artifact ref |
| Score cache | scorer component fingerprint + target fingerprint + model output fingerprint | `ScoreValueType` |
| Stage decision cache | criterion fingerprint + score fp | accept/reject per particle |
| Population store | calibration fingerprint + counterfactual id + stage + seed lineage | accepted particles, weights, diagnostics |
| Simulation cache | sampler spec + node population fingerprint + resolved input keys | simulation artifact refs |

The **calibration fingerprint** used in the population store key is derived from the full `CalibrationManifest` fingerprint, which transitively includes every registered scorer's `component_fingerprint()`. Any implementation change to any scorer therefore produces a new calibration fingerprint and a population cache miss, preventing cross-run contamination.


### 11.5 ArtifactStore

**Artifact layout:** `Context` assigns a per-particle output directory of the form `{artifact_root}/{stage_id}/{particle_id}/`. Every `config.json`, model output file, and `score_provenance.json` written during a run lands under that path, making each particle's full run record directly addressable. `ArtifactRef` URIs in `StageState::model_state_refs` and `score_provenances` point into this layout. This structure also means that a failed particle's config, partial outputs, and error record are co-located, which underpins `inspect_particle` ([§15](15-abc-rejection-sampling-execution.md)).


```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub fingerprint: Fingerprint,
    /// e.g. "file:///path/to/artifact.arrow" or "s3://bucket/key"
    pub uri:         String,
    pub data_type:  String,
}

pub trait ArtifactStore: Send + Sync {
    fn put(&self, fingerprint: &Fingerprint, data: &[serde_json::Value], data_type: &str)
        -> Result<ArtifactRef, std::io::Error>;
    fn get(&self, fingerprint: &Fingerprint)
        -> Result<Vec<serde_json::Value>, std::io::Error>;
}
```

Large `ModelOutput` objects and posterior tables are stored out-of-line via `ArtifactStore` and referenced in the manifest by `ArtifactRef`.

**Model state across the DAG:** When a runner supports `run_partial`, early DAG stages can checkpoint the intermediate `State` for each accepted particle, store it via `ArtifactStore`, and record the `ArtifactRef` in `StageState::model_state_refs`. A later stage with `ModelStatePolicy::ResumeFromParent` ([§3](03-simulation-system-and-dag-stages.md)) retrieves those states and calls `run_from_state`, extending the same run rather than starting fresh. Both stages share particle identity through `ParticleId`, establishing a common language across calibration stages: the same particle fingerprint appears in both `StageState` maps, and the `PerturbationType` entry in `ExperimentManifest::realized_kernels` documents what perturbation scale values were applied at each stage of this run (see [§6](06-perturbationkernel-and-density-convention.md)).


---

[← Counterfactual Model Construction](10-counterfactual-model-construction.md) | [TOC](README.md) | [Next: ModelRunner and Python Interop →](12-modelrunner.md)
