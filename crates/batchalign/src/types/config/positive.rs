//! Positive configuration scalars, admitted without filesystem access.
//!
//! A value below one is REFUSED where `server.yaml` is read, naming the field
//! and the value, the same policy as `local_lease_ttl_s` (`LeaseTtl`). Until
//! 2026-10-01 such a value was corrected to one with a warning that only the
//! callers of `ServerConfig::validate` printed, so a loader that skipped it
//! (`batchalign3 doctor`) reported a config as passing while the server ran a
//! value the operator never wrote. Refusing means every loader sees the same
//! answer, and there is no second "validated" load to forget.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::api::MachineTime;

/// A configuration scalar that must be at least one was below it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{field} must be >= 1 (got {got}); fix it in server.yaml, or leave it out for the default")]
pub struct NotPositive {
    /// The `server.yaml` key.
    pub field: &'static str,
    /// The value it held, as written.
    pub got: String,
}

macro_rules! positive_config {
    ($name:ident, $raw:ty => $prim:ty, $field:literal) => {
        #[doc = concat!("Admitted positive value for `", $field, "`: held as a `NonZero`, so no")]
        #[doc = "value below one can exist, whatever path built it, and one is refused"]
        #[doc = "where it is read."]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name(std::num::NonZero<$prim>);

        impl $name {
            /// A literal count, checked non-zero when the constant is
            /// evaluated, for defaults and tests.
            pub const fn literal<const N: $prim>() -> Self {
                const { assert!(N > 0, concat!($field, " must be at least one")) };
                Self(std::num::NonZero::<$prim>::MIN.saturating_add(N - 1))
            }

            /// The runtime value, at least one by type.
            pub const fn get(self) -> $prim {
                self.0.get()
            }
        }

        impl TryFrom<$raw> for $name {
            type Error = NotPositive;

            fn try_from(value: $raw) -> Result<Self, Self::Error> {
                <$prim>::try_from(value)
                    .ok()
                    .and_then(std::num::NonZero::<$prim>::new)
                    .map(Self)
                    .ok_or_else(|| NotPositive {
                        field: $field,
                        got: value.to_string(),
                    })
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.0.get().serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let value = <$raw>::deserialize(deserializer)?;
                Self::try_from(value).map_err(serde::de::Error::custom)
            }
        }
    };
}

positive_config!(JobTtlDays, i32 => u32, "job_ttl_days");

impl JobTtlDays {
    /// How long a job is kept.
    pub fn duration(self) -> std::time::Duration {
        std::time::Duration::from_secs(u64::from(self.get()) * 86_400)
    }

    /// The earliest submission still kept at `now`: anything submitted
    /// before it has expired.
    pub fn cutoff(self, now: MachineTime) -> MachineTime {
        now.minus(self.duration())
    }
}
positive_config!(MemoryGatePollSeconds, u64 => u64, "memory_gate_poll_s");
positive_config!(WorkerStartupLimit, u32 => usize, "max_concurrent_worker_startups");
