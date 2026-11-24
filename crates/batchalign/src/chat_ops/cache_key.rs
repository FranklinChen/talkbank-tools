//! Newtypes for cache key and task name, ensuring type safety across the
//! cache boundary.
//!
//! [`CacheKey`] wraps a BLAKE3 hash hex string. There is no constructor from
//! arbitrary strings: the only way to create one is via the task-specific
//! `cache_key()` functions in sibling modules, which compute the hash
//! internally through [`CacheKey::from_content`].
//!
//! [`CacheTaskName`] enumerates every NLP task that stores results in the
//! utterance cache, with wire strings matching the Python `CacheManager`
//! schema.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// A content-derived BLAKE3 hash used to index the utterance cache.
///
/// # Invariant
///
/// Always a 64-character lowercase hexadecimal string (256-bit BLAKE3 hash).
/// There is no constructor from arbitrary strings, the only way to create
/// a `CacheKey` is via the task-specific `cache_key()` functions, which
/// compute the hash internally.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct CacheKey(String);

impl CacheKey {
    /// Create a cache key by hashing the given content with BLAKE3.
    pub(crate) fn from_content(content: &str) -> Self {
        Self(blake3::hash(content.as_bytes()).to_hex().to_string())
    }

    /// View the hex string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for CacheKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err(serde::de::Error::custom(
                "cache key must be 64 lowercase hexadecimal characters",
            ))
        }
    }
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// CacheTaskName
// ---------------------------------------------------------------------------

/// Identifies the audio task whose result is being cached.
///
/// Only audio tasks use the utterance cache; text NLP tasks
/// (morphotag/utseg/translate) do not cache results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CacheTaskName {
    /// Forced alignment (word timings). Wire name: `"forced_alignment"`.
    ForcedAlignment,
    /// Immutable worker-protocol forced-alignment response, before local
    /// timing reconciliation. Wire name: `"forced_alignment_raw_evidence"`.
    ForcedAlignmentRawEvidence,
    /// UTR ASR result (full-file ASR for timing recovery). Wire name: `"utr_asr"`.
    UtrAsr,
    /// Immutable backend-shaped speaker evidence. Wire name:
    /// `"speaker_diarization_raw_evidence"`.
    SpeakerDiarizationRawEvidence,
    /// Locally derived normalized speaker segments. Wire name:
    /// `"speaker_diarization_segments"`.
    SpeakerDiarizationSegments,
    /// Immutable raw Rev.AI transcript evidence consumed by transcription and
    /// Rev-backed UTR during alignment.
    /// Wire name: `"rev_asr_evidence"`.
    RevAsrEvidence,
}

/// Classification of a user-supplied cache-override task name.
///
/// The CLI and server both receive the same strings, but need different
/// warning mechanisms.  Keeping the classification here gives those two
/// boundaries one exhaustive vocabulary without coupling the domain type to
/// either `eprintln!` or `tracing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheOverrideTaskName {
    /// An audio task whose evidence can be cached and selectively refreshed.
    Cacheable(CacheTaskName),
    /// A known text-NLP task, which deliberately has no media cache.
    TextNlpUnsupported,
    /// A name outside the cache-override vocabulary.
    Unknown,
}

impl CacheTaskName {
    /// Cache tasks an operator may refresh independently.
    ///
    /// Derived speaker segments are intentionally absent: they are local
    /// projections of raw evidence and refresh when that evidence does.
    pub(crate) const OVERRIDE_SELECTABLE: [Self; 4] = [
        Self::ForcedAlignment,
        Self::UtrAsr,
        Self::SpeakerDiarizationRawEvidence,
        Self::RevAsrEvidence,
    ];

    /// The wire string stored in the cache database.
    ///
    /// Changing any of these values invalidates existing cache entries.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ForcedAlignment => "forced_alignment",
            Self::ForcedAlignmentRawEvidence => "forced_alignment_raw_evidence",
            Self::UtrAsr => "utr_asr",
            Self::SpeakerDiarizationRawEvidence => "speaker_diarization_raw_evidence",
            Self::SpeakerDiarizationSegments => "speaker_diarization_segments",
            Self::RevAsrEvidence => "rev_asr_evidence",
        }
    }

    /// Classify a cache-override wire name at either process boundary.
    pub(crate) fn classify_override_name(name: &str) -> CacheOverrideTaskName {
        let trimmed = name.trim();
        if let Some(task) = Self::OVERRIDE_SELECTABLE
            .iter()
            .copied()
            .find(|task| task.as_str() == trimmed)
        {
            return CacheOverrideTaskName::Cacheable(task);
        }
        match trimmed {
            "morphosyntax" | "utterance_segmentation" | "translation" => {
                CacheOverrideTaskName::TextNlpUnsupported
            }
            _ => CacheOverrideTaskName::Unknown,
        }
    }
}

impl std::fmt::Display for CacheTaskName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_from_content_is_64_hex_chars() {
        let key = CacheKey::from_content("hello|eng|mwt");
        assert_eq!(key.as_str().len(), 64);
        assert!(key.as_str().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn override_task_name_classification_has_one_closed_vocabulary() {
        for task in CacheTaskName::OVERRIDE_SELECTABLE {
            assert_eq!(
                CacheTaskName::classify_override_name(task.as_str()),
                CacheOverrideTaskName::Cacheable(task)
            );
        }
        assert_eq!(
            CacheTaskName::classify_override_name(" rev_asr_evidence "),
            CacheOverrideTaskName::Cacheable(CacheTaskName::RevAsrEvidence)
        );
        assert_eq!(
            CacheTaskName::classify_override_name("translation"),
            CacheOverrideTaskName::TextNlpUnsupported
        );
        assert_eq!(
            CacheTaskName::classify_override_name("speaker_diarization_segments"),
            CacheOverrideTaskName::Unknown
        );
    }

    #[test]
    fn cache_key_from_content_deterministic() {
        let a = CacheKey::from_content("test input");
        let b = CacheKey::from_content("test input");
        assert_eq!(a, b);
    }

    #[test]
    fn cache_key_from_content_differs_for_different_input() {
        let a = CacheKey::from_content("input A");
        let b = CacheKey::from_content("input B");
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_display_matches_as_str() {
        let key = CacheKey::from_content("test");
        assert_eq!(format!("{key}"), key.as_str());
    }

    #[test]
    fn cache_key_serde_roundtrip() {
        let key = CacheKey::from_content("test");
        let json = serde_json::to_string(&key).unwrap();
        let deserialized: CacheKey = serde_json::from_str(&json).unwrap();
        assert_eq!(key, deserialized);
    }

    #[test]
    fn cache_key_deserialization_refuses_values_outside_its_documented_shape() {
        for json in [
            r#""short""#,
            r#""AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA""#,
            r#""gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg""#,
        ] {
            assert!(serde_json::from_str::<CacheKey>(json).is_err());
        }
    }

    #[test]
    fn cache_task_name_wire_strings_are_stable() {
        assert_eq!(CacheTaskName::ForcedAlignment.as_str(), "forced_alignment");
        assert_eq!(
            CacheTaskName::ForcedAlignmentRawEvidence.as_str(),
            "forced_alignment_raw_evidence"
        );
        assert_eq!(CacheTaskName::UtrAsr.as_str(), "utr_asr");
        assert_eq!(
            CacheTaskName::SpeakerDiarizationRawEvidence.as_str(),
            "speaker_diarization_raw_evidence"
        );
        assert_eq!(
            CacheTaskName::SpeakerDiarizationSegments.as_str(),
            "speaker_diarization_segments"
        );
        assert_eq!(CacheTaskName::RevAsrEvidence.as_str(), "rev_asr_evidence");
    }

    #[test]
    fn cache_task_name_display_matches_as_str() {
        for variant in [
            CacheTaskName::ForcedAlignment,
            CacheTaskName::ForcedAlignmentRawEvidence,
            CacheTaskName::UtrAsr,
            CacheTaskName::SpeakerDiarizationRawEvidence,
            CacheTaskName::SpeakerDiarizationSegments,
            CacheTaskName::RevAsrEvidence,
        ] {
            assert_eq!(format!("{variant}"), variant.as_str());
        }
    }
}
