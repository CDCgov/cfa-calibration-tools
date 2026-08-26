[← PerturbationKernel and Density Convention](06-perturbationkernel-and-density-convention.md) | [TOC](README.md) | [Next: ScoreCalculator and Target Builder →](08-scorecalculator-and-target-builder.md)

---

## 7. NestedSuffixParser and ParticleError

`NestedSuffixParser` is the single parser type exposed to users. It resolves `FlatParticle` suffix keys to their full dot-delimited paths in a nested defaults object, deep-merges the particle values, and returns the merged `serde_json::Value` for the model runner.

At construction, the parser flattens all leaf paths in `defaults` and builds a **suffix index**: a `HashMap` from every period-delimited suffix of every leaf path to that full path. Only suffixes that are unique across all leaf paths are stored — if a suffix matches two or more paths it is excluded from the index. An exact full-path key always resolves unambiguously.

At parse time, each key in the `FlatParticle` is looked up in the suffix index. If the key is absent (no path ends with it) or was excluded (multiple paths ended with it), `parse` returns `ParticleError::UnresolvableKey` listing the conflicting paths for diagnosis.

An optional `pretty_names` map can be provided via the builder method. It associates suffix-style keys with human-readable display labels used in diagnostic output; it has no effect on parsing.

```rust
use std::collections::HashMap;

/// Errors arising from parsing a particle.
#[derive(Debug, Serialize, Deserialize)]
pub enum ParticleError {
    /// The particle supplies a key that cannot be resolved to exactly one full path.
    /// `matches` is empty when no path ends with the key; non-empty when multiple paths do.
    UnresolvableKey { key: String, matches: Vec<String> },
    TypeMismatch { key: String, expected: &'static str },
    OutOfBounds { key: String, value: serde_json::Value, reason: String },
}

/// Maps `FlatParticle` suffix keys to their full dot-delimited paths in a nested
/// defaults object, then deep-merges the particle values to produce a model config.
///
/// Construction is infallible: it only requires a valid `serde_json::Value` object
/// and a string label. Suffix ambiguities are detected lazily at `parse` time.
pub struct NestedSuffixParser {
    /// Nested model configuration providing default values for all parameters.
    pub defaults:     serde_json::Value,
    /// Unique suffix → full dot-delimited path. Built at construction.
    /// Contains only suffixes that map to exactly one leaf path.
    suffix_index:     HashMap<String, String>,
    /// Optional display names: suffix key → human-readable label.
    /// Used in diagnostic output; has no effect on parsing.
    pub pretty_names: Option<HashMap<String, String>>,
    /// Stable string label for cache lookup and audit (see §12).
    label:            String,
}

impl NestedSuffixParser {
    /// Build a parser from a nested defaults object and a stable label.
    ///
    /// All leaf paths in `defaults` are enumerated and indexed by every unique
    /// period-delimited suffix. Construction is infallible.
    pub fn new(defaults: serde_json::Value, label: &str) -> Self;

    /// Attach optional human-readable display labels. Keys are suffix-style names;
    /// values are the labels shown in diagnostic output.
    pub fn pretty_names(self, names: HashMap<String, String>) -> Self;

    /// Resolve each key in `particle` via the suffix index, apply the values as
    /// overrides onto a deep clone of `defaults`, and return the merged object.
    ///
    /// Returns `Err(ParticleError::UnresolvableKey)` if any particle key cannot
    /// be resolved to exactly one full path.
    pub fn parse(&self, particle: &FlatParticle) -> Result<serde_json::Value, ParticleError>;

    /// Stable fingerprint: blake3 over label + flattened defaults.
    pub fn fingerprint(&self) -> Fingerprint;
}
```

**Error behavior:** A `ParticleError` elevates to a `StageError`, which halts further stage progression. However, particles in the same batch that do not produce errors continue to run. Only **one unique `ParticleError` variant** and the list of particle IDs that triggered it are reported to the user in real time. The user is offered a re-run or debug option. Optionally, a run can be configured to skip failures and present a summary report instead.

---

[← PerturbationKernel and Density Convention](06-perturbationkernel-and-density-convention.md) | [TOC](README.md) | [Next: ScoreCalculator and Target Builder →](08-scorecalculator-and-target-builder.md)
