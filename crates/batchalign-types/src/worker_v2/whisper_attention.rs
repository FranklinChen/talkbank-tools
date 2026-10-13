//! Optional, file-backed Whisper attention diagnostics; never timing authority.

use super::{WorkerArtifactPathV2, WorkerRequestIdV2};
use serde::{Deserialize, Serialize};

/// A request-bound destination for a new diagnostic SQLite artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(try_from = "CaptureRequestWire")]
pub struct WhisperAttentionCaptureRequestV2 {
    destination: WorkerArtifactPathV2,
    semantic_key: String,
}

#[derive(Deserialize, schemars::JsonSchema)]
struct CaptureRequestWire {
    destination: WorkerArtifactPathV2,
    semantic_key: String,
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

impl TryFrom<CaptureRequestWire> for WhisperAttentionCaptureRequestV2 {
    type Error = String;
    fn try_from(raw: CaptureRequestWire) -> Result<Self, Self::Error> {
        Self::new(raw.destination, raw.semantic_key)
    }
}

impl WhisperAttentionCaptureRequestV2 {
    /// Admit an absolute output path and a canonical request-key digest.
    pub fn new(destination: WorkerArtifactPathV2, semantic_key: String) -> Result<Self, String> {
        if !std::path::Path::new(destination.as_ref()).is_absolute() || !is_digest(&semantic_key) {
            return Err(
                "attention capture requires an absolute path and a 64-digit request key".into(),
            );
        }
        Ok(Self {
            destination,
            semantic_key,
        })
    }
    /// Exclusive-create output destination.
    pub fn destination(&self) -> &WorkerArtifactPathV2 {
        &self.destination
    }
    /// Semantic FA cache key of the exact planned request.
    pub fn semantic_key(&self) -> &str {
        &self.semantic_key
    }
}

/// Small receipt for numeric evidence retained outside the JSON transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct WhisperAttentionCaptureReceiptV2 {
    /// Request capability, retained whole rather than independently relabelled.
    pub capture: WhisperAttentionCaptureRequestV2,
    /// Worker request that produced this artifact.
    pub request_id: WorkerRequestIdV2,
    /// BLAKE3 of the closed SQLite file, verified by the server before use.
    pub blake3: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_request_refuses_relative_paths_and_noncanonical_keys()
    -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            WhisperAttentionCaptureRequestV2::new("relative.sqlite".into(), "a".repeat(64))
                .is_err()
        );
        assert!(
            WhisperAttentionCaptureRequestV2::new("/tmp/capture.sqlite".into(), "wrong".into())
                .is_err()
        );
        let admitted =
            WhisperAttentionCaptureRequestV2::new("/tmp/capture.sqlite".into(), "a".repeat(64))?;
        let wire = serde_json::to_value(&admitted)?;
        assert_eq!(
            serde_json::from_value::<WhisperAttentionCaptureRequestV2>(wire)?,
            admitted
        );
        Ok(())
    }
}
