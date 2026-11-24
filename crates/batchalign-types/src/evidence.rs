//! Algorithm-specific content identities carried by worker observations.

use serde::{Deserialize, Serialize};

/// A validated lowercase SHA-256 digest.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    schemars::JsonSchema,
    utoipa::ToSchema,
)]
#[serde(transparent)]
pub struct Sha256Digest(#[schemars(regex(pattern = "^[0-9a-f]{64}$"))] String);

impl Sha256Digest {
    /// Canonical lowercase hexadecimal representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            Ok(Self(value))
        } else {
            Err("SHA-256 digest must be 64 lowercase hexadecimal characters".into())
        }
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}
