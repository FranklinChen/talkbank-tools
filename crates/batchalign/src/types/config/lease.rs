//! The local queue lease's lifetime and the heartbeat that renews it.
//!
//! A live runner renews its job's lease every [`LEASE_HEARTBEAT`]. A lease
//! that lived no longer than that would lapse between renewals, and the job
//! would look orphaned while its runner was healthy. [`LeaseTtl`] is the only
//! way to name a lease lifetime, and it refuses one that short.

use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// How often a live runner renews its job's lease.
pub const LEASE_HEARTBEAT: Duration = Duration::from_secs(60);

/// How long a lease lives without renewal: always longer than
/// [`LEASE_HEARTBEAT`]. Written in config as whole seconds
/// (`local_lease_ttl_s`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseTtl(Duration);

/// A lease lifetime no longer than the heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "local_lease_ttl_s must be longer than the {}s lease heartbeat, got {got}s",
    LEASE_HEARTBEAT.as_secs()
)]
pub struct LeaseTtlTooShort {
    got: u64,
}

impl LeaseTtl {
    /// The default lifetime: five heartbeats.
    pub const DEFAULT: Self = Self(Duration::from_secs(300));

    /// A lifetime of `secs` seconds, refused unless it outlasts a heartbeat.
    pub fn from_secs(secs: u64) -> Result<Self, LeaseTtlTooShort> {
        let lifetime = Duration::from_secs(secs);
        if lifetime > LEASE_HEARTBEAT {
            Ok(Self(lifetime))
        } else {
            Err(LeaseTtlTooShort { got: secs })
        }
    }

    /// The lifetime.
    pub const fn get(self) -> Duration {
        self.0
    }
}

impl Serialize for LeaseTtl {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.as_secs().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LeaseTtl {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_secs(u64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lifetime must outlast the heartbeat; equal to it is refused too,
    /// since a renewal landing exactly at expiry would race the expiry.
    #[test]
    fn a_lease_lifetime_must_outlast_the_heartbeat() {
        assert!(LeaseTtl::from_secs(0).is_err());
        assert!(LeaseTtl::from_secs(LEASE_HEARTBEAT.as_secs()).is_err());
        assert_eq!(
            LeaseTtl::from_secs(61).unwrap().get(),
            Duration::from_secs(61)
        );
        assert_eq!(
            LeaseTtl::from_secs(30).unwrap_err().to_string(),
            "local_lease_ttl_s must be longer than the 60s lease heartbeat, got 30s"
        );
    }

    /// Config reads and writes whole seconds.
    #[test]
    fn config_form_is_whole_seconds() {
        let ttl: LeaseTtl = serde_json::from_str("300").unwrap();
        assert_eq!(ttl, LeaseTtl::DEFAULT);
        assert_eq!(serde_json::to_string(&ttl).unwrap(), "300");
        assert!(serde_json::from_str::<LeaseTtl>("10").is_err());
    }
}
