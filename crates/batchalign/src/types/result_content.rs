//! Untrusted result wire data. A binary descriptor is not permission to write:
//! destination admission and streamed identity verification happen at delivery.

use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

/// Canonical BLAKE3 identity, admitted from a digest or checked hexadecimal text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "server", schema(value_type = String))]
pub struct ArtifactDigest(blake3::Hash);

impl ArtifactDigest {
    /// Identity computed by a BLAKE3 producer, not unchecked hexadecimal text.
    pub fn from_hash(hash: blake3::Hash) -> Self {
        Self(hash)
    }
}

impl TryFrom<String> for ArtifactDigest {
    type Error = blake3::HexError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        blake3::Hash::from_hex(value).map(Self)
    }
}

impl From<ArtifactDigest> for String {
    fn from(value: ArtifactDigest) -> Self {
        value.0.to_hex().to_string()
    }
}

/// Nonempty artifact identity sent instead of binary bytes in a JSON response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct BinaryResultDescriptor {
    /// Exact byte count; a zero-length artifact is not an exported recording.
    #[cfg_attr(feature = "server", schema(value_type = u64, minimum = 1))]
    pub byte_len: NonZeroU64,
    /// Identity of the bytes, verified again by the delivery consumer.
    pub digest: ArtifactDigest,
}

impl BinaryResultDescriptor {
    /// Read a held file, with bounded memory, and rewind that same handle.
    /// This observes bytes; it does not certify acoustic quality or encoding.
    pub async fn inspect(file: &mut tokio::fs::File) -> std::io::Result<Self> {
        file.rewind().await?;
        // Heap-owned: a 64 KiB array here is held across every `.await` below
        // and so became 64 KiB of every results-request future (measured
        // 2026-10-06 with -Zprint-type-sizes: get_results 68 KB).
        let mut buffer = vec![0u8; 64 * 1024].into_boxed_slice();
        let mut hasher = blake3::Hasher::new();
        let mut count = 0u64;
        loop {
            let read = file.read(&mut buffer).await?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or_else(|| std::io::Error::other("binary result byte count overflow"))?;
            hasher.update(&buffer[..read]);
        }
        let byte_len = NonZeroU64::new(count).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "empty binary result")
        })?;
        file.rewind().await?;
        Ok(Self {
            byte_len,
            digest: ArtifactDigest::from_hash(hasher.finalize()),
        })
    }
}

/// Content on the result wire: inline text or a streamed artifact descriptor.
/// Untagged serialization retains the existing JSON string for text results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub enum ResultContent {
    /// Inline CHAT, CSV, plain text or JSON document bytes.
    Text(String),
    /// Binary bytes must be downloaded and admitted separately.
    Binary(BinaryResultDescriptor),
}

impl Default for ResultContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<String> for ResultContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for ResultContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl ResultContent {
    /// Borrow text only when the wire actually carries text. No lossy fallback.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Binary(_) => None,
        }
    }

    /// Whether a result has no content; binary descriptors are always nonempty.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.is_empty(),
            Self::Binary(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_wire_stays_a_string_and_binary_never_becomes_text() {
        let text = ResultContent::from("@Begin\n@End\n");
        assert_eq!(
            serde_json::to_value(&text).unwrap(),
            serde_json::json!("@Begin\n@End\n")
        );
        let descriptor = BinaryResultDescriptor {
            byte_len: NonZeroU64::new(4).unwrap(),
            digest: ArtifactDigest::from_hash(blake3::hash(b"\0\xff\x80\0")),
        };
        let binary = ResultContent::Binary(descriptor);
        assert!(binary.as_text().is_none());
        assert!(!binary.is_empty());
        let wire = serde_json::to_value(&binary).unwrap();
        insta::assert_json_snapshot!("binary_result_content", wire);
        assert_eq!(
            serde_json::from_value::<ResultContent>(wire).unwrap(),
            binary
        );
    }

    #[test]
    fn malformed_binary_descriptors_do_not_fall_back_to_empty_text() {
        let digest = String::from(ArtifactDigest::from_hash(blake3::hash(b"bytes")));
        for bad in [
            serde_json::json!({"byte_len": 0, "digest": digest}),
            serde_json::json!({"byte_len": 4, "digest": "not-a-digest"}),
            serde_json::json!({"byte_len": 4}),
            serde_json::json!({"byte_len": 4, "digest": digest, "content": "invented"}),
        ] {
            assert!(serde_json::from_value::<ResultContent>(bad).is_err());
        }
    }
}
