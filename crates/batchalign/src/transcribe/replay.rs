//! Fingerprinted admission for legacy projected transcribe evidence.
//!
//! An old `_asr_response.json` is useful for testing newer local
//! post-processing, but it is not raw provider evidence. This boundary keeps
//! it structurally separate from the paid-evidence caches.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::api::{DurationMs, DurationSeconds};
use crate::types::worker_v2::SpeakerSegmentV2;

use super::AsrResponse;

const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// The projected-ASR producer admitted by the first replay format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LegacyProjectedAsrProducer {
    /// A BA3 `AsrResponse` projected from Rev.AI output.
    RevAi,
}

impl LegacyProjectedAsrProducer {
    pub(crate) const fn provenance_name(self) -> crate::api::StampSafeText {
        match self {
            Self::RevAi => const { crate::api::StampSafeText::from_static("rev") },
        }
    }
}

/// Exact inputs for authoring one immutable replay manifest.
pub(crate) struct LegacyReplayManifestRequest<'a> {
    pub(crate) recording_id: &'a str,
    pub(crate) media_path: &'a Path,
    pub(crate) asr_response_path: &'a Path,
    pub(crate) speaker_turns_path: Option<&'a Path>,
    pub(crate) producer: LegacyProjectedAsrProducer,
}

/// Evidence admitted for execution by the offline replay pipeline.
///
/// Construction is private to [`admit_legacy_replay_manifest`], so pipeline
/// code cannot receive an unfingerprinted response or mismatched media.
#[derive(Clone, Debug)]
pub(crate) struct AdmittedLegacyTranscribeReplay {
    recording_id: String,
    media_path: PathBuf,
    asr_response: AsrResponse,
    speaker_segments: Option<Vec<SpeakerSegmentV2>>,
    producer: LegacyProjectedAsrProducer,
    manifest_blake3: Blake3Digest,
}

impl AdmittedLegacyTranscribeReplay {
    pub(crate) fn recording_id(&self) -> &str {
        &self.recording_id
    }

    pub(crate) fn media_path(&self) -> &Path {
        &self.media_path
    }

    pub(crate) fn asr_response(&self) -> &AsrResponse {
        &self.asr_response
    }

    pub(crate) fn speaker_segments(&self) -> Option<&[SpeakerSegmentV2]> {
        self.speaker_segments.as_deref()
    }

    pub(crate) const fn producer(&self) -> LegacyProjectedAsrProducer {
        self.producer
    }

    pub(crate) fn manifest_blake3(&self) -> &str {
        self.manifest_blake3.as_str()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
struct Blake3Digest(String);

impl Blake3Digest {
    fn from_bytes(bytes: &[u8]) -> Self {
        Self(blake3::hash(bytes).to_hex().to_string())
    }

    fn from_path(path: &Path) -> Result<Self, LegacyReplayError> {
        let bytes = std::fs::read(path).map_err(|source| LegacyReplayError::Read {
            path: path.to_owned(),
            source,
        })?;
        Ok(Self::from_bytes(&bytes))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Fingerprint any file with the same digest algorithm used by replay
/// manifests. Kept at this evidence boundary so run receipts and manifests do
/// not drift to different hashing recipes.
pub(crate) fn file_blake3_hex(path: &Path) -> Result<String, LegacyReplayError> {
    Ok(Blake3Digest::from_path(path)?.0)
}

impl TryFrom<String> for Blake3Digest {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value))
        } else {
            Err("BLAKE3 digest must be 64 lowercase hexadecimal characters".into())
        }
    }
}

impl From<Blake3Digest> for String {
    fn from(value: Blake3Digest) -> Self {
        value.0
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FingerprintedPath {
    path: PathBuf,
    blake3: Blake3Digest,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectedAsrArtifact {
    #[serde(flatten)]
    artifact: FingerprintedPath,
    producer: LegacyProjectedAsrProducer,
    projection_revision: ProjectedAsrRevision,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum ProjectedAsrRevision {
    LegacyAsrResponseV1,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeakerTurnsArtifact {
    #[serde(flatten)]
    artifact: FingerprintedPath,
    format_revision: SpeakerTurnsRevision,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum SpeakerTurnsRevision {
    CanonicalTurnsV1,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyReplayManifest {
    schema_version: u32,
    recording_id: String,
    source_media: FingerprintedPath,
    projected_asr: ProjectedAsrArtifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speaker_turns: Option<SpeakerTurnsArtifact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTurnsFile {
    source: String,
    turns: Vec<CanonicalTurn>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTurn {
    start_ms: u64,
    end_ms: u64,
    track: String,
}

/// Failures while authoring or admitting projected replay evidence.
#[derive(Debug, thiserror::Error)]
pub(crate) enum LegacyReplayError {
    #[error("could not read replay artifact {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("could not write replay manifest {path}: {source}")]
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid replay JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("unsupported replay manifest schema {actual}; expected {expected}")]
    SchemaVersion { actual: u32, expected: u32 },
    #[error("replay recording_id must not be empty")]
    EmptyRecordingId,
    #[error(
        "replay recording_id {0:?} must not be '.' or '..' and may contain only ASCII letters, digits, '.', '_', and '-'"
    )]
    InvalidRecordingId(String),
    #[error("replay artifact digest mismatch for {path}: expected {expected}, observed {observed}")]
    DigestMismatch {
        path: PathBuf,
        expected: String,
        observed: String,
    },
    #[error("projected ASR token {index} has incomplete timing")]
    IncompleteAsrTiming { index: usize },
    #[error("projected ASR token {index} has invalid timing {start_s}..{end_s}")]
    InvalidAsrTiming {
        index: usize,
        start_s: f64,
        end_s: f64,
    },
    #[error("speaker turns source must not be empty")]
    EmptyTurnsSource,
    #[error("speaker turn {index} has invalid track {track:?}; expected PAR followed by digits")]
    InvalidTrack { index: usize, track: String },
    #[error("speaker turn {index} is inverted: {start_ms}..{end_ms}")]
    InvertedTurn {
        index: usize,
        start_ms: u64,
        end_ms: u64,
    },
}

/// Author a manifest binding projected artifacts to exact source-media bytes.
pub(crate) fn write_legacy_replay_manifest(
    request: LegacyReplayManifestRequest<'_>,
    output_path: &Path,
) -> Result<(), LegacyReplayError> {
    validate_recording_id(request.recording_id)?;
    let manifest = LegacyReplayManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        recording_id: request.recording_id.to_owned(),
        source_media: fingerprinted(request.media_path)?,
        projected_asr: ProjectedAsrArtifact {
            artifact: fingerprinted(request.asr_response_path)?,
            producer: request.producer,
            projection_revision: ProjectedAsrRevision::LegacyAsrResponseV1,
        },
        speaker_turns: request
            .speaker_turns_path
            .map(fingerprinted)
            .transpose()?
            .map(|artifact| SpeakerTurnsArtifact {
                artifact,
                format_revision: SpeakerTurnsRevision::CanonicalTurnsV1,
            }),
    };
    let mut bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|source| LegacyReplayError::Json {
            path: output_path.to_owned(),
            source,
        })?;
    bytes.push(b'\n');
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .map_err(|source| LegacyReplayError::Write {
            path: output_path.to_owned(),
            source,
        })?;
    output
        .write_all(&bytes)
        .map_err(|source| LegacyReplayError::Write {
            path: output_path.to_owned(),
            source,
        })
}

/// Validate every fingerprint and construct the replay-only admitted state.
pub(crate) fn admit_legacy_replay_manifest(
    manifest_path: &Path,
) -> Result<AdmittedLegacyTranscribeReplay, LegacyReplayError> {
    let manifest_bytes =
        std::fs::read(manifest_path).map_err(|source| LegacyReplayError::Read {
            path: manifest_path.to_owned(),
            source,
        })?;
    let manifest_blake3 = Blake3Digest::from_bytes(&manifest_bytes);
    let manifest: LegacyReplayManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|source| LegacyReplayError::Json {
            path: manifest_path.to_owned(),
            source,
        })?;
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        return Err(LegacyReplayError::SchemaVersion {
            actual: manifest.schema_version,
            expected: MANIFEST_SCHEMA_VERSION,
        });
    }
    validate_recording_id(&manifest.recording_id)?;
    let base = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let media_path = verified_path(base, &manifest.source_media)?;
    let asr_path = verified_path(base, &manifest.projected_asr.artifact)?;
    let asr_bytes = std::fs::read(&asr_path).map_err(|source| LegacyReplayError::Read {
        path: asr_path.clone(),
        source,
    })?;
    let asr_response: AsrResponse =
        serde_json::from_slice(&asr_bytes).map_err(|source| LegacyReplayError::Json {
            path: asr_path,
            source,
        })?;
    validate_asr_response(&asr_response)?;

    let speaker_segments = manifest
        .speaker_turns
        .as_ref()
        .map(|artifact| {
            let path = verified_path(base, &artifact.artifact)?;
            let bytes = std::fs::read(&path).map_err(|source| LegacyReplayError::Read {
                path: path.clone(),
                source,
            })?;
            let turns: CanonicalTurnsFile =
                serde_json::from_slice(&bytes).map_err(|source| LegacyReplayError::Json {
                    path: path.clone(),
                    source,
                })?;
            lower_turns(turns)
        })
        .transpose()?;

    Ok(AdmittedLegacyTranscribeReplay {
        recording_id: manifest.recording_id,
        media_path,
        asr_response,
        speaker_segments,
        producer: manifest.projected_asr.producer,
        manifest_blake3,
    })
}

fn fingerprinted(path: &Path) -> Result<FingerprintedPath, LegacyReplayError> {
    let canonical = std::fs::canonicalize(path).map_err(|source| LegacyReplayError::Read {
        path: path.to_owned(),
        source,
    })?;
    Ok(FingerprintedPath {
        blake3: Blake3Digest::from_path(&canonical)?,
        path: canonical,
    })
}

fn validate_recording_id(recording_id: &str) -> Result<(), LegacyReplayError> {
    if recording_id.is_empty() {
        return Err(LegacyReplayError::EmptyRecordingId);
    }
    if matches!(recording_id, "." | "..")
        || !recording_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(LegacyReplayError::InvalidRecordingId(
            recording_id.to_owned(),
        ));
    }
    Ok(())
}

fn verified_path(base: &Path, artifact: &FingerprintedPath) -> Result<PathBuf, LegacyReplayError> {
    let path = if artifact.path.is_absolute() {
        artifact.path.clone()
    } else {
        base.join(&artifact.path)
    };
    let observed = Blake3Digest::from_path(&path)?;
    if observed != artifact.blake3 {
        return Err(LegacyReplayError::DigestMismatch {
            path,
            expected: artifact.blake3.as_str().to_owned(),
            observed: observed.as_str().to_owned(),
        });
    }
    Ok(path)
}

fn validate_asr_response(response: &AsrResponse) -> Result<(), LegacyReplayError> {
    for (index, token) in response.tokens.iter().enumerate() {
        match (token.start_s, token.end_s) {
            (None, None) => {}
            (Some(DurationSeconds(start_s)), Some(DurationSeconds(end_s))) => {
                if !start_s.is_finite() || !end_s.is_finite() || start_s < 0.0 || end_s < start_s {
                    return Err(LegacyReplayError::InvalidAsrTiming {
                        index,
                        start_s,
                        end_s,
                    });
                }
            }
            _ => return Err(LegacyReplayError::IncompleteAsrTiming { index }),
        }
    }
    Ok(())
}

fn lower_turns(turns: CanonicalTurnsFile) -> Result<Vec<SpeakerSegmentV2>, LegacyReplayError> {
    if turns.source.trim().is_empty() {
        return Err(LegacyReplayError::EmptyTurnsSource);
    }
    turns
        .turns
        .into_iter()
        .enumerate()
        .map(|(index, turn)| {
            let suffix =
                turn.track
                    .strip_prefix("PAR")
                    .ok_or_else(|| LegacyReplayError::InvalidTrack {
                        index,
                        track: turn.track.clone(),
                    })?;
            if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(LegacyReplayError::InvalidTrack {
                    index,
                    track: turn.track,
                });
            }
            if turn.start_ms > turn.end_ms {
                return Err(LegacyReplayError::InvertedTurn {
                    index,
                    start_ms: turn.start_ms,
                    end_ms: turn.end_ms,
                });
            }
            Ok(SpeakerSegmentV2 {
                start_ms: DurationMs(turn.start_ms),
                end_ms: DurationMs(turn.end_ms),
                speaker: format!("PAR{suffix}"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;
    use crate::transcribe::AsrToken;

    fn response() -> AsrResponse {
        AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.1)),
                end_s: Some(DurationSeconds(0.4)),
                speaker: Some("0".into()),
                confidence: Some(0.9),
            }],
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        }
    }

    fn write_fixture(dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let media = dir.join("sample.wav");
        let asr = dir.join("sample_asr_response.json");
        let turns = dir.join("sample.turns.json");
        std::fs::write(&media, b"media bytes").expect("media");
        std::fs::write(
            &asr,
            serde_json::to_vec_pretty(&response()).expect("ASR JSON"),
        )
        .expect("ASR");
        std::fs::write(
            &turns,
            br#"{"source":"batchalign3:pyannote_ai:precision-2","turns":[{"track":"PAR1","start_ms":100,"end_ms":400}]}"#,
        )
        .expect("turns");
        (media, asr, turns)
    }

    fn manifest(dir: &Path, media: &Path, asr: &Path, turns: Option<&Path>) -> PathBuf {
        let path = dir.join("sample.replay.json");
        write_legacy_replay_manifest(
            LegacyReplayManifestRequest {
                recording_id: "sample",
                media_path: media,
                asr_response_path: asr,
                speaker_turns_path: turns,
                producer: LegacyProjectedAsrProducer::RevAi,
            },
            &path,
        )
        .expect("manifest");
        path
    }

    #[test]
    fn admission_requires_all_exact_artifact_fingerprints() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, turns) = write_fixture(dir.path());
        let admitted =
            admit_legacy_replay_manifest(&manifest(dir.path(), &media, &asr, Some(&turns)))
                .expect("admitted");
        assert_eq!(admitted.recording_id(), "sample");
        assert_eq!(
            admitted.media_path(),
            std::fs::canonicalize(&media).expect("canonical media")
        );
        assert_eq!(admitted.asr_response().tokens[0].text, "hello");
        assert_eq!(admitted.speaker_segments().map(<[_]>::len), Some(1));
        assert_eq!(admitted.producer().provenance_name().as_str(), "rev");
        assert_eq!(admitted.manifest_blake3().len(), 64);
    }

    #[test]
    fn media_drift_is_refused_before_replay_admission() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, turns) = write_fixture(dir.path());
        let manifest = manifest(dir.path(), &media, &asr, Some(&turns));
        std::fs::write(&media, b"different media bytes").expect("drift media");
        assert!(matches!(
            admit_legacy_replay_manifest(&manifest),
            Err(LegacyReplayError::DigestMismatch { path, .. })
                if path == std::fs::canonicalize(&media).expect("canonical media")
        ));
    }

    #[test]
    fn projected_asr_drift_is_refused_before_json_is_consumed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, turns) = write_fixture(dir.path());
        let manifest = manifest(dir.path(), &media, &asr, Some(&turns));
        std::fs::write(&asr, b"{}").expect("drift ASR");
        assert!(matches!(
            admit_legacy_replay_manifest(&manifest),
            Err(LegacyReplayError::DigestMismatch { path, .. })
                if path == std::fs::canonicalize(&asr).expect("canonical ASR")
        ));
    }

    #[test]
    fn malformed_turn_track_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, turns) = write_fixture(dir.path());
        std::fs::write(
            &turns,
            br#"{"source":"batchalign3:pyannote","turns":[{"track":"CHI","start_ms":100,"end_ms":400}]}"#,
        )
        .expect("turns");
        let manifest = manifest(dir.path(), &media, &asr, Some(&turns));
        assert!(matches!(
            admit_legacy_replay_manifest(&manifest),
            Err(LegacyReplayError::InvalidTrack { track, .. }) if track == "CHI"
        ));
    }

    #[test]
    fn incomplete_projected_word_timing_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, _turns) = write_fixture(dir.path());
        let mut bad = response();
        bad.tokens[0].end_s = None;
        std::fs::write(&asr, serde_json::to_vec_pretty(&bad).expect("JSON")).expect("ASR");
        let manifest = manifest(dir.path(), &media, &asr, None);
        assert!(matches!(
            admit_legacy_replay_manifest(&manifest),
            Err(LegacyReplayError::IncompleteAsrTiming { index: 0 })
        ));
    }

    #[test]
    fn recording_id_cannot_escape_the_future_output_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, _turns) = write_fixture(dir.path());
        let path = dir.path().join("bad.replay.json");
        assert!(matches!(
            write_legacy_replay_manifest(
                LegacyReplayManifestRequest {
                    recording_id: "../elsewhere",
                    media_path: &media,
                    asr_response_path: &asr,
                    speaker_turns_path: None,
                    producer: LegacyProjectedAsrProducer::RevAi,
                },
                &path,
            ),
            Err(LegacyReplayError::InvalidRecordingId(id)) if id == "../elsewhere"
        ));
    }

    #[test]
    fn recording_id_cannot_be_a_relative_directory_component() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, _turns) = write_fixture(dir.path());
        for recording_id in [".", ".."] {
            let path = dir.path().join(format!("{recording_id}.replay.json"));
            assert!(matches!(
                write_legacy_replay_manifest(
                    LegacyReplayManifestRequest {
                        recording_id,
                        media_path: &media,
                        asr_response_path: &asr,
                        speaker_turns_path: None,
                        producer: LegacyProjectedAsrProducer::RevAi,
                    },
                    &path,
                ),
                Err(LegacyReplayError::InvalidRecordingId(id)) if id == recording_id
            ));
        }
    }

    #[test]
    fn manifest_authoring_never_overwrites_existing_evidence_identity() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (media, asr, _turns) = write_fixture(dir.path());
        let path = dir.path().join("sample.replay.json");
        let request = || LegacyReplayManifestRequest {
            recording_id: "sample",
            media_path: &media,
            asr_response_path: &asr,
            speaker_turns_path: None,
            producer: LegacyProjectedAsrProducer::RevAi,
        };
        write_legacy_replay_manifest(request(), &path).expect("first manifest");
        assert!(matches!(
            write_legacy_replay_manifest(request(), &path),
            Err(LegacyReplayError::Write { source, .. })
                if source.kind() == std::io::ErrorKind::AlreadyExists
        ));
    }
}
