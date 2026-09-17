//! Positive configuration scalars, admitted without filesystem access.
//!
//! Keep legacy clamping policy and its warning, but make an invalid runtime
//! value impossible even for callers that deserialize without calling validate.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! positive_config {
    ($name:ident, $raw:ty, $field:literal) => {
        #[doc = concat!("Admitted positive value for `", $field, "`.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name {
            value: $raw,
            corrected: Option<$raw>,
        }

        impl $name {
            /// Admit the legacy scalar, preserving a correction for reporting.
            pub const fn new(value: $raw) -> Self {
                if value < 1 {
                    Self {
                        value: 1,
                        corrected: Some(value),
                    }
                } else {
                    Self {
                        value,
                        corrected: None,
                    }
                }
            }

            /// The runtime value, always at least one.
            pub const fn get(self) -> $raw {
                self.value
            }

            pub(super) fn warning(self) -> Option<String> {
                self.corrected
                    .map(|value| format!("{} must be >= 1 (got {value}), defaulting to 1", $field))
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.value.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                <$raw>::deserialize(deserializer).map(Self::new)
            }
        }
    };
}

positive_config!(JobTtlDays, i32, "job_ttl_days");
positive_config!(MemoryGatePollSeconds, u64, "memory_gate_poll_s");
positive_config!(WorkerStartupLimit, u32, "max_concurrent_worker_startups");
