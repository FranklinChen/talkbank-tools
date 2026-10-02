//! Serde helpers for the `Option<u32>` / `Option<i32>` / `Option<u64>`
//! migration in Phase C of the host-facts architecture.
//!
//! Moved here from `host_facts/` on 2026-07-30. `host_facts` did not use
//! any of it: every consumer was already in `types/config`, so the module
//! only ever made `types` depend on an impure module.
//!
//! The migration changes integer config knobs that today use the
//! `0 = auto-detect` sentinel idiom (e.g.,
//! `gpu_thread_pool_size: u32` defaulting to `0` to mean "fall back
//! to the static constant") into proper `Option<T>` fields where
//! `None` means "fall through to the recommendation function." The rationale
//! is the host-facts architecture decision Q1: a sentinel value cannot express
//! "unset" without colliding with a legitimate setting of the same number.
//!
//! For the duration of the rollout, deployed `server.yaml` files will
//! continue to carry literal `0` values written before the migration
//! landed. The `zero_as_none` deserializer accepts both forms:
//!
//! - **Field absent** (canonical post-migration form) → `None`.
//! - **Field present, value `0`** (legacy form) → `None`.
//! - **Field present, value `n` ≠ 0** → `Some(n)`.
//!
//! Pair with `#[serde(default, skip_serializing_if = "Option::is_none")]`
//! on the field so re-serialization writes the canonical form
//! (field-absent for `None`).
//!
//! Once every `server.yaml` in the fleet has been re-rendered post-
//! migration, the legacy `0` form is no longer emitted by anyone.
//! The shim becomes dead code and is removed in Phase G2.

use serde::{Deserialize, Deserializer};

/// Integer types that have a meaningful "zero is the legacy auto-detect
/// sentinel" interpretation. The `is_zero` predicate is the only thing
/// `zero_as_none` needs from the underlying type.
///
/// Implemented for `u32`, `i32`, and `u64` because those are the
/// integer field types currently using the `0 = auto` idiom in
/// `ServerConfig` / `PoolConfig`. Adding a new type to the migration
/// is a one-line `impl`.
pub trait IsZero {
    /// Whether this value is the migration-equivalent of "no value set".
    fn is_zero(&self) -> bool;
}

impl IsZero for u32 {
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl IsZero for i32 {
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

impl IsZero for u64 {
    fn is_zero(&self) -> bool {
        *self == 0
    }
}

/// Newtype `MemoryMb(u64)`: the canonical migration target for the
/// `memory_gate_mb` knob (Phase C2.7). Treats `MemoryMb(0)` as the
/// legacy `0 = auto` sentinel; `MemoryMb(n)` for `n > 0` survives as
/// `Some(MemoryMb(n))`.
impl IsZero for crate::api::MemoryMb {
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

/// Serde deserializer that maps both legacy `0` and a literal `null` /
/// missing field to `None`, while preserving non-zero values as
/// `Some(n)`.
///
/// Use as `#[serde(default, deserialize_with = "zero_as_none",
/// skip_serializing_if = "Option::is_none")]` on `Option<T>` fields
/// where `T: IsZero`.
///
/// The function is generic over `T` so the same helper covers `u32`,
/// `i32`, and `u64` migration sites without per-type duplication.
/// Serde monomorphizes `T` from the field type at the use site, so
/// no turbofish is needed in the attribute.
pub fn zero_as_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + IsZero,
{
    // Accept the field as-is (Some/None), then collapse Some(0) to None.
    // This handles both `field: 0` and `field: 4` and `field: null`
    // uniformly.
    let opt = Option::<T>::deserialize(deserializer)?;
    Ok(opt.filter(|v| !v.is_zero()))
}

/// Serde deserializer for an optional operator override in seconds whose
/// legacy `0` meant "the built-in default": absent, `null` and `0` are all
/// `None`, and any other count is `Some`.
///
/// `PositiveSeconds` itself refuses `0`, so this is the one place the legacy
/// spelling is still read, for `server.yaml` files written before these fields
/// were typed. Pair with `#[serde(default, skip_serializing_if =
/// "Option::is_none")]` so re-serialization writes the canonical form.
pub fn zero_as_no_override<'de, D>(
    deserializer: D,
) -> Result<Option<crate::api::PositiveSeconds>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<u64>::deserialize(deserializer)?
        .and_then(std::num::NonZeroU64::new)
        .map(crate::api::PositiveSeconds::new))
}

/// Serde deserializer for a required interval or timeout whose legacy `0`
/// meant "the built-in default", which is `DEFAULT`. Any other count is kept.
pub fn zero_as_default<'de, D, const DEFAULT: u64>(
    deserializer: D,
) -> Result<crate::api::PositiveSeconds, D::Error>
where
    D: Deserializer<'de>,
{
    match zero_as_no_override(deserializer)? {
        Some(seconds) => Ok(seconds),
        None => Ok(crate::api::PositiveSeconds::literal::<DEFAULT>()),
    }
}

/// Serde deserializer for an optional path whose legacy empty string meant
/// "the default location": absent, `null` and `""` are all `None`.
pub fn empty_path_as_none<'de, D>(deserializer: D) -> Result<Option<std::path::PathBuf>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(Option::<std::path::PathBuf>::deserialize(deserializer)?
        .filter(|path| !path.as_os_str().is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Debug, Deserialize)]
    struct OverrideCarrier {
        #[serde(default, deserialize_with = "zero_as_no_override")]
        value: Option<crate::api::PositiveSeconds>,
    }

    #[derive(Debug, Deserialize)]
    struct DefaultedCarrier {
        #[serde(deserialize_with = "zero_as_default::<_, 30>")]
        value: crate::api::PositiveSeconds,
    }

    #[derive(Debug, Deserialize)]
    struct PathCarrier {
        #[serde(default, deserialize_with = "empty_path_as_none")]
        value: Option<std::path::PathBuf>,
    }

    /// The legacy `0` of a timeout override reads as no override, the same
    /// as leaving the field out; a count is kept.
    #[test]
    fn a_legacy_zero_override_is_no_override() {
        for yaml in ["value: 0\n", "{}", "value: null\n"] {
            let parsed: OverrideCarrier = yaml_serde::from_str(yaml).expect("parse");
            assert_eq!(parsed.value, None, "{yaml}");
        }
        let parsed: OverrideCarrier = yaml_serde::from_str("value: 3600\n").expect("parse");
        assert_eq!(
            parsed.value.map(crate::api::PositiveSeconds::get),
            Some(3600)
        );
    }

    /// A required interval's legacy `0` is the default it always meant.
    #[test]
    fn a_legacy_zero_interval_is_the_default() {
        let parsed: DefaultedCarrier = yaml_serde::from_str("value: 0\n").expect("parse");
        assert_eq!(parsed.value.get(), 30);
        let parsed: DefaultedCarrier = yaml_serde::from_str("value: 15\n").expect("parse");
        assert_eq!(parsed.value.get(), 15);
    }

    #[test]
    fn a_legacy_empty_path_is_no_path() {
        let parsed: PathCarrier = yaml_serde::from_str("value: \"\"\n").expect("parse");
        assert_eq!(parsed.value, None);
        let parsed: PathCarrier = yaml_serde::from_str("value: /tmp/w.json\n").expect("parse");
        assert_eq!(parsed.value, Some(std::path::PathBuf::from("/tmp/w.json")));
    }

    /// One round-trip carrier per integer type. The `default` plus
    /// `skip_serializing_if` attributes match the canonical migration
    /// form documented in the module-level comment.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct CarrierU32 {
        #[serde(
            default,
            deserialize_with = "zero_as_none",
            skip_serializing_if = "Option::is_none"
        )]
        value: Option<u32>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct CarrierI32 {
        #[serde(
            default,
            deserialize_with = "zero_as_none",
            skip_serializing_if = "Option::is_none"
        )]
        value: Option<i32>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    struct CarrierU64 {
        #[serde(
            default,
            deserialize_with = "zero_as_none",
            skip_serializing_if = "Option::is_none"
        )]
        value: Option<u64>,
    }

    // -------------------------------------------------------------------
    // u32: the most common migration target (gpu_thread_pool_size).
    // -------------------------------------------------------------------

    #[test]
    fn u32_legacy_zero_deserializes_to_none() {
        let parsed: CarrierU32 = yaml_serde::from_str("value: 0\n").expect("parse");
        assert_eq!(parsed.value, None);
    }

    #[test]
    fn u32_explicit_value_preserves_some() {
        let parsed: CarrierU32 = yaml_serde::from_str("value: 4\n").expect("parse");
        assert_eq!(parsed.value, Some(4));
    }

    #[test]
    fn u32_field_absent_defaults_to_none() {
        let parsed: CarrierU32 = yaml_serde::from_str("{}").expect("parse");
        assert_eq!(parsed.value, None);
    }

    #[test]
    fn u32_explicit_null_deserializes_to_none() {
        let parsed: CarrierU32 = yaml_serde::from_str("value: null\n").expect("parse");
        assert_eq!(parsed.value, None);
    }

    #[test]
    fn u32_some_value_serializes_as_bare_integer() {
        let yaml = yaml_serde::to_string(&CarrierU32 { value: Some(4) }).expect("ser");
        assert_eq!(yaml.trim(), "value: 4");
    }

    #[test]
    fn u32_none_serializes_as_omitted_field() {
        let yaml = yaml_serde::to_string(&CarrierU32 { value: None }).expect("ser");
        // Empty struct → "{}" with yaml_serde. The key is that the
        // field is omitted, not the precise empty-struct rendering.
        assert!(
            !yaml.contains("value"),
            "None must omit the field; got: {yaml:?}"
        );
    }

    /// Round-trip the canonical post-migration form. `Some(n)` writes
    /// `value: n`; reads back `Some(n)`. `None` writes nothing; reads
    /// back `None`. The combination is what makes the migration
    /// reversible during the rollout.
    #[test]
    fn u32_round_trip_canonical_form() {
        for v in [None, Some(1u32), Some(4), Some(8), Some(u32::MAX)] {
            let original = CarrierU32 { value: v };
            let yaml = yaml_serde::to_string(&original).expect("ser");
            let parsed: CarrierU32 = yaml_serde::from_str(&yaml).expect("parse");
            assert_eq!(parsed, original, "round-trip lost data for {v:?}");
        }
    }

    /// Mixed-form round-trip: legacy `0` collapses to `None`, which
    /// re-serializes as field-absent. After one round-trip the legacy
    /// form is gone.
    #[test]
    fn u32_legacy_zero_collapses_after_one_round_trip() {
        let parsed: CarrierU32 = yaml_serde::from_str("value: 0\n").expect("parse");
        assert_eq!(parsed.value, None);
        let yaml = yaml_serde::to_string(&parsed).expect("ser");
        assert!(
            !yaml.contains("value"),
            "post-round-trip form must omit `value`; got: {yaml:?}"
        );
    }

    // -------------------------------------------------------------------
    // i32: covers max_concurrent_jobs, max_workers_per_job,
    // max_workers_per_key, max_total_workers (signed sentinel).
    // -------------------------------------------------------------------

    #[test]
    fn i32_legacy_zero_deserializes_to_none() {
        let parsed: CarrierI32 = yaml_serde::from_str("value: 0\n").expect("parse");
        assert_eq!(parsed.value, None);
    }

    #[test]
    fn i32_explicit_value_preserves_some() {
        let parsed: CarrierI32 = yaml_serde::from_str("value: 8\n").expect("parse");
        assert_eq!(parsed.value, Some(8));
    }

    /// Negative values are not the legacy sentinel, only `0` is.
    /// `-1` survives as `Some(-1)` even though it's nonsensical for
    /// these fields; validation is the EffectiveConfig layer's job.
    #[test]
    fn i32_negative_value_preserves_some() {
        let parsed: CarrierI32 = yaml_serde::from_str("value: -1\n").expect("parse");
        assert_eq!(parsed.value, Some(-1));
    }

    // -------------------------------------------------------------------
    // u64: covers memory_gate_mb (already wraps in MemoryMb but the
    // serde representation is u64).
    // -------------------------------------------------------------------

    #[test]
    fn u64_legacy_zero_deserializes_to_none() {
        let parsed: CarrierU64 = yaml_serde::from_str("value: 0\n").expect("parse");
        assert_eq!(parsed.value, None);
    }

    #[test]
    fn u64_explicit_value_preserves_some() {
        let parsed: CarrierU64 = yaml_serde::from_str("value: 8000\n").expect("parse");
        assert_eq!(parsed.value, Some(8000));
    }

    #[test]
    fn u64_round_trip_huge_value() {
        let original = CarrierU64 {
            value: Some(u64::MAX),
        };
        let yaml = yaml_serde::to_string(&original).expect("ser");
        let parsed: CarrierU64 = yaml_serde::from_str(&yaml).expect("parse");
        assert_eq!(parsed, original);
    }

    // -------------------------------------------------------------------
    // The IsZero trait contract. If a future contributor adds an impl
    // for a type whose semantic "no value" isn't literal zero, these
    // tests document the expectation.
    // -------------------------------------------------------------------

    #[test]
    fn is_zero_recognizes_canonical_zeros() {
        assert!(0u32.is_zero());
        assert!(0i32.is_zero());
        assert!(0u64.is_zero());
        assert!(!1u32.is_zero());
        assert!(!(-1i32).is_zero());
        assert!(!u64::MAX.is_zero());
    }
}
