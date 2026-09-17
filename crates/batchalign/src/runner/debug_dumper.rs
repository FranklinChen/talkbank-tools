//! Pipeline debug artifact writer.
//!
//! When constructed with a directory path, writes structured CHAT/JSON
//! artifacts at each pipeline stage for offline replay and test fixture
//! generation. When constructed without a path, all methods are zero-cost
//! no-ops. This is the single testability seam for stage decomposition.

#[cfg(test)]
use crate::revai::FetchedRevAsrEvidence;

use std::{
    io::Write,
    path::{Path, PathBuf},
};

use crate::chat_ops::fa::utr::{AsrTimingToken, UtrResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::api::DurationMs;
use crate::revai::RevAsrEvidenceTrace;
use crate::runner::dispatch::diarize_turns::{
    SpeakerTurnsSource, TurnsBuildError, format_turns_json,
};
use crate::transcribe::SpeakerEvidenceTrace;
use crate::types::traces::{FaGroupTrace, FaTimelineTrace};
use crate::types::worker_v2::{SpeakerBackendV2, SpeakerSegmentV2};

/// Pipeline debug artifact writer.
///
/// When constructed with a directory path, writes structured CHAT/JSON
/// artifacts at each pipeline stage for offline replay and test fixture
/// generation. When constructed without a path, all methods are zero-cost
/// no-ops.
pub(crate) struct DebugDumper {
    dir: Option<PathBuf>,
}

/// Complete artifact bytes that can be atomically persisted.
///
/// Constructing this state before opening the destination prevents a
/// serialization failure from truncating a prior evidence artifact.
struct SerializedArtifact(Vec<u8>);

impl SerializedArtifact {
    fn json(value: &impl Serialize) -> Result<Self, serde_json::Error> {
        serde_json::to_vec_pretty(value).map(Self)
    }

    fn text(value: String) -> Self {
        Self(value.into_bytes())
    }

    fn persist(self, path: &Path) -> Result<(), std::io::Error> {
        let parent = path.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("artifact path has no parent: {}", path.display()),
            )
        })?;
        let mut temp = tempfile::NamedTempFile::new_in(parent)?;
        temp.write_all(&self.0)?;
        temp.as_file().sync_all()?;
        let persisted = temp.persist(path).map_err(|error| error.error)?;
        persisted.sync_all()?;
        #[cfg(unix)]
        std::fs::File::open(parent)?.sync_all()?;
        Ok(())
    }
}

/// Observable result of requesting a same-job speaker-turn dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpeakerTurnsDumpOutcome {
    /// Debug artifact collection was not requested for this run.
    Disabled,
    /// The canonical turns artifact was durably written at this path.
    Written(PathBuf),
}

/// Failures while retaining the exact diarization turns used by transcribe.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SpeakerTurnsDumpError {
    /// The configured debug directory could not be created.
    #[error("failed to create speaker-turn artifact directory {}: {source}", path.display())]
    CreateDirectory {
        /// Directory requested by the job.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// Worker segments could not be represented by the canonical schema.
    #[error("failed to build canonical speaker-turn artifact: {0}")]
    Build(#[from] TurnsBuildError),
    /// The completed artifact could not be written.
    #[error("failed to write speaker-turn artifact {}: {source}", path.display())]
    Write {
        /// Intended artifact path.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
}

/// Observable result of requesting a durable forced-alignment evidence dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FaEvidenceDumpOutcome {
    /// Debug artifact collection was not requested for this run.
    Disabled,
    /// The versioned evidence artifact was durably written at this path.
    Written(PathBuf),
}

/// Failures while retaining forced-alignment timing evidence.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FaEvidenceDumpError {
    /// The configured debug directory could not be created.
    #[error("failed to create FA evidence directory {}: {source}", path.display())]
    CreateDirectory {
        /// Directory requested by the job.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
    /// The typed evidence could not be serialized.
    #[error("failed to serialize FA evidence: {0}")]
    Serialize(#[from] serde_json::Error),
    /// The completed artifact could not be written.
    #[error("failed to write FA evidence {}: {source}", path.display())]
    Write {
        /// Intended artifact path.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },
}

/// Observable result of requesting a durable Rev causal-evidence dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RevEvidenceDumpOutcome {
    Disabled,
    Written(PathBuf),
}

/// Failures while retaining the Rev request/cache/projection identity.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RevEvidenceDumpError {
    #[error("failed to create Rev evidence directory {}: {source}", path.display())]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize Rev evidence trace: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("failed to write Rev evidence trace {}: {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Observable result of requesting a durable speaker causal-evidence dump.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SpeakerEvidenceDumpOutcome {
    Disabled,
    Written(PathBuf),
}

/// Failures while retaining the speaker request/cache/projection identity.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SpeakerEvidenceDumpError {
    #[error("failed to create speaker evidence directory {}: {source}", path.display())]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize speaker evidence trace: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("failed to write speaker evidence trace {}: {source}", path.display())]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Per-group FA dump data for offline replay.
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub(crate) struct FaGroupDumpData {
    /// Audio window start in milliseconds.
    pub audio_start_ms: DurationMs,
    /// Audio window end in milliseconds.
    pub audio_end_ms: DurationMs,
    /// Words in this group.
    pub words: Vec<String>,
    /// Per-word timing pairs from FA inference.
    pub timings: Vec<Option<TimingPair>>,
}

/// A start/end timing pair in milliseconds.
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub(crate) struct TimingPair {
    /// Word start time in milliseconds.
    pub start_ms: i64,
    /// Word end time in milliseconds.
    pub end_ms: i64,
}

impl DebugDumper {
    /// Create a new dumper. If `dir` is `None`, all methods are no-ops.
    pub(crate) fn new(dir: Option<&Path>) -> Self {
        Self {
            dir: dir.map(PathBuf::from),
        }
    }

    /// Create a disabled dumper (all methods are no-ops).
    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self { dir: None }
    }

    /// Whether dumping is enabled.
    pub(crate) fn is_enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// Ensure the dump directory exists, returning it. Logs and returns `None`
    /// on failure.
    fn ensure_dir(&self) -> Option<&Path> {
        let dir = self.dir.as_deref()?;
        if let Err(e) = std::fs::create_dir_all(dir) {
            debug!(%e, "failed to create debug dir");
            return None;
        }
        Some(dir)
    }

    /// Extract the file stem from a filename for use in dump file names.
    fn stem(filename: &str) -> &str {
        Path::new(filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    }

    /// Collision-resistant stem for evidence that must survive corpus runs.
    ///
    /// Keep the familiar basename for a plain filename. When the submitted
    /// identity includes a directory, append a digest of the complete identity
    /// so two corpus branches containing `sample.cha` cannot silently overwrite
    /// one another in a shared debug directory.
    fn evidence_stem(filename: &str) -> String {
        let stem = Self::stem(filename);
        if Path::new(filename).components().count() <= 1 {
            return stem.to_owned();
        }
        let digest = blake3::hash(filename.as_bytes()).to_hex();
        format!("{stem}-{}", &digest[..12])
    }

    /// Dump CHAT text before UTR injection.
    pub(crate) fn dump_utr_input(&self, filename: &str, chat_text: &str) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_utr_input.cha"));
        if let Err(e) = std::fs::write(&path, chat_text) {
            debug!(%e, "failed to write UTR debug CHAT input");
        }
    }

    /// Dump ASR timing tokens used for UTR injection.
    pub(crate) fn dump_utr_tokens(&self, filename: &str, tokens: &[AsrTimingToken]) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_utr_tokens.json"));
        match serde_json::to_string_pretty(tokens) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    debug!(%e, "failed to write UTR debug tokens");
                }
            }
            Err(e) => debug!(%e, "failed to serialize UTR tokens"),
        }

        info!(
            %filename,
            tokens = %path.display(),
            "UTR debug data dumped"
        );
    }

    /// Dump CHAT text and UtrResult after UTR injection.
    pub(crate) fn dump_utr_output(&self, filename: &str, chat_text: &str, utr_result: &UtrResult) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);

        let chat_path = dir.join(format!("{stem}_utr_output.cha"));
        if let Err(e) = std::fs::write(&chat_path, chat_text) {
            debug!(%e, "failed to write UTR debug CHAT output");
        }

        let result_path = dir.join(format!("{stem}_utr_result.json"));
        match serde_json::to_string_pretty(utr_result) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&result_path, json) {
                    debug!(%e, "failed to write UTR result JSON");
                }
            }
            Err(e) => debug!(%e, "failed to serialize UTR result"),
        }
    }

    /// Dump FA grouping plan and pre-FA CHAT text.
    #[allow(dead_code)]
    pub(crate) fn dump_fa_grouping(
        &self,
        filename: &str,
        groups: &[FaGroupTrace],
        chat_text: &str,
    ) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);

        let chat_path = dir.join(format!("{stem}_fa_input.cha"));
        if let Err(e) = std::fs::write(&chat_path, chat_text) {
            debug!(%e, "failed to write FA debug CHAT input");
        }

        let grouping_path = dir.join(format!("{stem}_fa_grouping.json"));
        match serde_json::to_string_pretty(groups) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&grouping_path, json) {
                    debug!(%e, "failed to write FA grouping JSON");
                }
            }
            Err(e) => debug!(%e, "failed to serialize FA grouping"),
        }

        info!(
            %filename,
            num_groups = groups.len(),
            "FA grouping debug data dumped"
        );
    }

    /// Dump per-group FA result (words + timings).
    #[allow(dead_code)]
    pub(crate) fn dump_fa_group_result(
        &self,
        filename: &str,
        group_idx: usize,
        data: &FaGroupDumpData,
    ) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_fa_group_{group_idx}.json"));
        match serde_json::to_string_pretty(data) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    debug!(%e, group = group_idx, "failed to write FA group result");
                }
            }
            Err(e) => debug!(%e, group = group_idx, "failed to serialize FA group result"),
        }
    }

    /// Dump final aligned CHAT text after FA.
    pub(crate) fn dump_fa_output(&self, filename: &str, chat_text: &str) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_fa_output.cha"));
        if let Err(e) = std::fs::write(&path, chat_text) {
            debug!(%e, "failed to write FA debug CHAT output");
        }
    }

    /// Durably retain the exact pre-injection FA evidence used by this run.
    ///
    /// Unlike the older best-effort dumps, an enabled request either writes a
    /// complete versioned artifact or returns a typed error. This prevents a
    /// research run from completing while silently losing the confidence and
    /// provenance that motivated `--debug-dir`.
    pub(crate) fn dump_fa_evidence(
        &self,
        filename: &str,
        evidence: &FaTimelineTrace,
    ) -> Result<FaEvidenceDumpOutcome, FaEvidenceDumpError> {
        let Some(dir) = self.dir.as_deref() else {
            return Ok(FaEvidenceDumpOutcome::Disabled);
        };
        std::fs::create_dir_all(dir).map_err(|source| FaEvidenceDumpError::CreateDirectory {
            path: dir.to_owned(),
            source,
        })?;
        let path = dir.join(format!(
            "{}_fa_evidence.json",
            Self::evidence_stem(filename)
        ));
        let artifact = SerializedArtifact::json(evidence)?;
        artifact
            .persist(&path)
            .map_err(|source| FaEvidenceDumpError::Write {
                path: path.clone(),
                source,
            })?;
        info!(%filename, evidence = %path.display(), "FA evidence dumped");
        Ok(FaEvidenceDumpOutcome::Written(path))
    }

    // -------------------------------------------------------------------
    // Transcribe pipeline debug artifacts
    // -------------------------------------------------------------------

    /// Durably retain the Rev media/request/cache/projection identity.
    pub(crate) fn dump_rev_evidence(
        &self,
        filename: &str,
        trace: &RevAsrEvidenceTrace,
    ) -> Result<RevEvidenceDumpOutcome, RevEvidenceDumpError> {
        let Some(dir) = self.dir.as_deref() else {
            return Ok(RevEvidenceDumpOutcome::Disabled);
        };
        std::fs::create_dir_all(dir).map_err(|source| RevEvidenceDumpError::CreateDirectory {
            path: dir.to_owned(),
            source,
        })?;
        let path = dir.join(format!(
            "{}_rev_evidence.json",
            Self::evidence_stem(filename)
        ));
        SerializedArtifact::json(trace)?
            .persist(&path)
            .map_err(|source| RevEvidenceDumpError::Write {
                path: path.clone(),
                source,
            })?;
        info!(%filename, evidence = %path.display(), "Rev evidence trace dumped");
        Ok(RevEvidenceDumpOutcome::Written(path))
    }

    /// Durably retain the speaker media/request/cache/projection identity.
    pub(crate) fn dump_speaker_evidence(
        &self,
        filename: &str,
        trace: &SpeakerEvidenceTrace,
    ) -> Result<SpeakerEvidenceDumpOutcome, SpeakerEvidenceDumpError> {
        let Some(dir) = self.dir.as_deref() else {
            return Ok(SpeakerEvidenceDumpOutcome::Disabled);
        };
        std::fs::create_dir_all(dir).map_err(|source| {
            SpeakerEvidenceDumpError::CreateDirectory {
                path: dir.to_owned(),
                source,
            }
        })?;
        let path = dir.join(format!(
            "{}_speaker_evidence.json",
            Self::evidence_stem(filename)
        ));
        SerializedArtifact::json(trace)?
            .persist(&path)
            .map_err(|source| SpeakerEvidenceDumpError::Write {
                path: path.clone(),
                source,
            })?;
        info!(%filename, evidence = %path.display(), "Speaker evidence trace dumped");
        Ok(SpeakerEvidenceDumpOutcome::Written(path))
    }

    /// Dump raw ASR response JSON after ASR inference.
    pub(crate) fn dump_asr_response(&self, filename: &str, response: &impl serde::Serialize) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_asr_response.json"));
        match serde_json::to_string_pretty(response) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    debug!(%e, "failed to write ASR response JSON");
                }
            }
            Err(e) => debug!(%e, "failed to serialize ASR response"),
        }
        info!(%filename, response = %path.display(), "ASR response debug data dumped");
    }

    /// Dump the words the CHAT-legality gate refused but emitted anyway.
    ///
    /// Beside the transcript, because that is where a reader asking "why is
    /// this file invalid" is standing. The warn line in
    /// `report_language_invalid_words` is the narration; this is the record.
    pub(crate) fn dump_language_invalid_words(
        &self,
        filename: &str,
        words: &impl serde::Serialize,
    ) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_language_invalid_words.json"));
        match serde_json::to_string_pretty(words) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&path, json) {
                    debug!(%e, "failed to write language-invalid word report");
                }
            }
            Err(e) => debug!(%e, "failed to serialize language-invalid word report"),
        }
        info!(%filename, report = %path.display(), "language-invalid words dumped");
    }

    /// Retain the exact same-job diarization turns used by transcribe.
    ///
    /// Unlike older debug dumps, this method does not swallow an enabled
    /// write failure. A job that explicitly requests this evidence either
    /// receives a typed `Written` outcome or fails before the turns can be
    /// discarded.
    pub(crate) fn dump_speaker_turns(
        &self,
        filename: &str,
        backend: SpeakerBackendV2,
        segments: &[SpeakerSegmentV2],
    ) -> Result<SpeakerTurnsDumpOutcome, SpeakerTurnsDumpError> {
        let Some(dir) = self.dir.as_deref() else {
            return Ok(SpeakerTurnsDumpOutcome::Disabled);
        };
        std::fs::create_dir_all(dir).map_err(|source| SpeakerTurnsDumpError::CreateDirectory {
            path: dir.to_path_buf(),
            source,
        })?;

        let artifact = SerializedArtifact::text(format_turns_json(
            SpeakerTurnsSource::from_backend(backend),
            segments,
        )?);
        let path = dir.join(format!("{}.turns.json", Self::evidence_stem(filename)));
        artifact
            .persist(&path)
            .map_err(|source| SpeakerTurnsDumpError::Write {
                path: path.clone(),
                source,
            })?;
        info!(
            %filename,
            backend = ?backend,
            turns = segments.len(),
            artifact = %path.display(),
            "Same-job speaker turns dumped"
        );
        Ok(SpeakerTurnsDumpOutcome::Written(path))
    }

    /// Dump CHAT text after CHAT assembly (post-ASR, pre-utseg).
    pub(crate) fn dump_post_asr_chat(&self, filename: &str, chat_text: &str) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_post_asr.cha"));
        if let Err(e) = std::fs::write(&path, chat_text) {
            debug!(%e, "failed to write post-ASR CHAT");
        }
    }

    /// Dump CHAT text before utterance segmentation.
    pub(crate) fn dump_pre_utseg_chat(&self, filename: &str, chat_text: &str) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_pre_utseg.cha"));
        if let Err(e) = std::fs::write(&path, chat_text) {
            debug!(%e, "failed to write pre-utseg CHAT");
        }
    }

    /// Dump CHAT text after utterance segmentation.
    pub(crate) fn dump_post_utseg_chat(&self, filename: &str, chat_text: &str) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_post_utseg.cha"));
        if let Err(e) = std::fs::write(&path, chat_text) {
            debug!(%e, "failed to write post-utseg CHAT");
        }
    }

    /// Dump CHAT text before morphosyntax.
    pub(crate) fn dump_pre_morphosyntax_chat(&self, filename: &str, chat_text: &str) {
        let Some(dir) = self.ensure_dir() else {
            return;
        };
        let stem = Self::stem(filename);
        let path = dir.join(format!("{stem}_pre_morphosyntax.cha"));
        if let Err(e) = std::fs::write(&path, chat_text) {
            debug!(%e, "failed to write pre-morphosyntax CHAT");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{LanguageCode3, NumSpeakers};
    use crate::revai::{RevAsrEvidenceRequest, RevAsrEvidenceResolution, RevAsrModelRevision};
    use crate::types::traces::{FaDecisionTrace, FaTimingDecisionTrace};
    use crate::types::worker_v2::{SpeakerBackendV2, SpeakerSegmentV2};

    fn speaker_segment(speaker: &str, start_ms: u64, end_ms: u64) -> SpeakerSegmentV2 {
        SpeakerSegmentV2 {
            start_ms: DurationMs(start_ms),
            end_ms: DurationMs(end_ms),
            speaker: speaker.to_owned(),
        }
    }

    #[test]
    fn disabled_speaker_turns_dump_reports_disabled_without_writing() {
        let dumper = DebugDumper::disabled();
        let outcome = dumper
            .dump_speaker_turns(
                "sample.wav",
                SpeakerBackendV2::PyannoteAi,
                &[speaker_segment("SPEAKER_00", 10, 20)],
            )
            .expect("disabled dumping is not an error");

        assert_eq!(outcome, SpeakerTurnsDumpOutcome::Disabled);
    }

    #[test]
    fn speaker_turns_dump_writes_backend_provenance_and_canonical_tracks() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dumper = DebugDumper::new(Some(dir.path()));
        let outcome = dumper
            .dump_speaker_turns(
                "sample.wav",
                SpeakerBackendV2::PyannoteAi,
                &[
                    speaker_segment("SPEAKER_01", 500, 900),
                    speaker_segment("SPEAKER_00", 0, 500),
                ],
            )
            .expect("valid speaker turns should be written");

        let expected_path = dir.path().join("sample.turns.json");
        assert_eq!(
            outcome,
            SpeakerTurnsDumpOutcome::Written(expected_path.clone())
        );
        let value: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(expected_path).expect("read speaker turns"),
        )
        .expect("parse speaker turns");
        assert_eq!(value["source"], "batchalign3:pyannote_ai:precision-2");
        assert_eq!(value["turns"][0]["track"], "PAR0");
        assert_eq!(value["turns"][1]["track"], "PAR1");
    }

    #[tokio::test]
    async fn rev_evidence_dump_is_durable_joined_and_collision_resistant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let audio = dir.path().join("sample.wav");
        tokio::fs::write(&audio, b"provider media")
            .await
            .expect("write audio");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &crate::types::revai_language::RevLanguage::admit(
                &crate::api::AsrLanguageRequest::One(LanguageCode3::eng()),
            )
            .expect("Rev.AI recognizes English"),
            NumSpeakers(2),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let resolution = RevAsrEvidenceResolution::replayed_for_test(
            &request,
            FetchedRevAsrEvidence {
                transcript_evidence: crate::revai::RevTranscriptEvidence::from_legacy_transcript(
                    serde_json::from_str(r#"{"monologues": []}"#).expect("valid empty transcript"),
                ),
                resolved_language: crate::api::TranscriptLanguage::One(LanguageCode3::eng()),
            },
        );
        let trace = resolution.trace(crate::revai::RevAsrProjectionRevision::AsrResponseV1);
        let dumper = DebugDumper::new(Some(dir.path()));
        let identity = "corpus-a/sample.wav";

        let outcome = dumper
            .dump_rev_evidence(identity, &trace)
            .expect("requested Rev evidence should be durable");
        let expected = dir.path().join(format!(
            "{}_rev_evidence.json",
            DebugDumper::evidence_stem(identity)
        ));
        assert_eq!(outcome, RevEvidenceDumpOutcome::Written(expected.clone()));

        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(expected).expect("read Rev evidence"))
                .expect("parse Rev evidence");
        assert_eq!(value["trace_schema_version"], 2);
        assert_eq!(value["cache_outcome"], "replayed");
        assert_eq!(value["transcript_fidelity"], "legacy_typed_projection");
        assert_eq!(
            value["projection_revision"],
            "rev-transcript-to-asr-response-v1"
        );
        assert_eq!(value["raw_evidence_key"], request.cache_key().as_str());
    }

    #[test]
    fn fa_evidence_dump_is_durable_and_versioned() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dumper = DebugDumper::new(Some(dir.path()));
        let evidence = FaTimelineTrace {
            evidence_schema_version: crate::types::traces::CURRENT_FA_EVIDENCE_SCHEMA_VERSION,
            engine: "wav2vec_fa".to_owned(),
            engine_version: "test-build".to_owned(),
            groups: Vec::new(),
            evidence_sources: Vec::new(),
            cache_keys: Vec::new(),
            pre_injection_timings: Vec::new(),
            post_injection_timings: Vec::new(),
            decisions: vec![FaDecisionTrace {
                line_idx: 7,
                speaker: "PAR0".to_owned(),
                module: "monotonicity".to_owned(),
                strategy: "timing_stripped".to_owned(),
                reason: "non_monotonic start_ms=900 previous_start_ms=1000".to_owned(),
                needs_review: true,
            }],
            timing_decisions: vec![FaTimingDecisionTrace::StartRegressionStripped {
                line_idx: 7,
                utterance_idx: 2,
                speaker: "PAR0".to_owned(),
                start_ms: 900,
                previous_start_ms: 1_000,
                previous_line_idx: 6,
                previous_utterance_idx: 1,
                previous_speaker: "PAR1".to_owned(),
            }],
            // A start regression discards no word timing, so this stays
            // empty; the artifact still carries the section.
            dropped_word_timings: Vec::new(),
            gap_healing: "Heal".to_owned(),
            violations: Vec::new(),
            fallback_events: Vec::new(),
        };

        let outcome = dumper
            .dump_fa_evidence("sample.cha", &evidence)
            .expect("requested evidence should be durable");
        let expected_path = dir.path().join("sample_fa_evidence.json");
        assert_eq!(
            outcome,
            FaEvidenceDumpOutcome::Written(expected_path.clone())
        );
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(expected_path).expect("read evidence"))
                .expect("parse evidence");
        assert_eq!(
            value["evidence_schema_version"],
            crate::types::traces::CURRENT_FA_EVIDENCE_SCHEMA_VERSION
        );
        assert_eq!(value["engine"], "wav2vec_fa");
        assert_eq!(value["decisions"][0]["line_idx"], 7);
        assert_eq!(value["decisions"][0]["module"], "monotonicity");
        assert_eq!(value["decisions"][0]["strategy"], "timing_stripped");
        assert_eq!(
            value["timing_decisions"][0]["kind"],
            "start_regression_stripped"
        );
        assert_eq!(value["timing_decisions"][0]["start_ms"], 900);
        assert_eq!(value["timing_decisions"][0]["utterance_idx"], 2);
        assert_eq!(value["timing_decisions"][0]["previous_line_idx"], 6);
        assert_eq!(value["timing_decisions"][0]["previous_utterance_idx"], 1);
        assert_eq!(value["timing_decisions"][0]["previous_speaker"], "PAR1");
    }

    #[test]
    fn fa_evidence_paths_do_not_collide_for_equal_basenames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dumper = DebugDumper::new(Some(dir.path()));
        let evidence = FaTimelineTrace {
            evidence_schema_version: 1,
            engine: "wav2vec_fa".to_owned(),
            engine_version: "test-build".to_owned(),
            groups: Vec::new(),
            evidence_sources: Vec::new(),
            cache_keys: Vec::new(),
            pre_injection_timings: Vec::new(),
            post_injection_timings: Vec::new(),
            decisions: Vec::new(),
            timing_decisions: Vec::new(),
            dropped_word_timings: Vec::new(),
            gap_healing: "Heal".to_owned(),
            violations: Vec::new(),
            fallback_events: Vec::new(),
        };

        let first = dumper
            .dump_fa_evidence("corpus-a/sample.cha", &evidence)
            .expect("first evidence should be written");
        let second = dumper
            .dump_fa_evidence("corpus-b/sample.cha", &evidence)
            .expect("second evidence should be written");

        let (FaEvidenceDumpOutcome::Written(first), FaEvidenceDumpOutcome::Written(second)) =
            (first, second)
        else {
            panic!("enabled evidence dumping must write both artifacts");
        };
        assert_ne!(first, second);
        assert!(first.exists());
        assert!(second.exists());
    }

    #[cfg(unix)]
    #[test]
    fn fa_evidence_dump_replaces_a_destination_symlink_without_following_it() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("tempdir");
        let sentinel = dir.path().join("unrelated.json");
        std::fs::write(&sentinel, "do not replace").expect("write sentinel");
        let evidence_path = dir.path().join("sample_fa_evidence.json");
        symlink(&sentinel, &evidence_path).expect("create destination symlink");
        let dumper = DebugDumper::new(Some(dir.path()));
        let evidence = FaTimelineTrace {
            evidence_schema_version: crate::types::traces::CURRENT_FA_EVIDENCE_SCHEMA_VERSION,
            engine: "wav2vec_fa".to_owned(),
            engine_version: "test-build".to_owned(),
            groups: Vec::new(),
            evidence_sources: Vec::new(),
            cache_keys: Vec::new(),
            pre_injection_timings: Vec::new(),
            post_injection_timings: Vec::new(),
            decisions: Vec::new(),
            timing_decisions: Vec::new(),
            dropped_word_timings: Vec::new(),
            gap_healing: "Heal".to_owned(),
            violations: Vec::new(),
            fallback_events: Vec::new(),
        };

        dumper
            .dump_fa_evidence("sample.cha", &evidence)
            .expect("evidence write should replace the destination entry");

        assert_eq!(
            std::fs::read_to_string(&sentinel).expect("read sentinel"),
            "do not replace"
        );
        assert!(!evidence_path.is_symlink());
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(evidence_path).expect("read evidence"))
                .expect("parse evidence");
        assert_eq!(
            written["evidence_schema_version"],
            crate::types::traces::CURRENT_FA_EVIDENCE_SCHEMA_VERSION
        );
    }

    #[test]
    fn disabled_dumper_is_noop() {
        let dumper = DebugDumper::disabled();
        assert!(!dumper.is_enabled());
        // These should all return immediately without error
        let chat = "@UTF8\n@Begin\n@End";
        dumper.dump_utr_input("test.cha", chat);
        dumper.dump_utr_tokens("test.cha", &[]);
        dumper.dump_utr_output("test.cha", chat, &UtrResult::not_run_no_untimed(0));
        dumper.dump_fa_output("test.cha", chat);
        dumper.dump_asr_response("test.wav", &serde_json::json!({"tokens": []}));
        dumper.dump_post_asr_chat("test.wav", chat);
        dumper.dump_pre_utseg_chat("test.wav", chat);
        dumper.dump_post_utseg_chat("test.wav", chat);
        dumper.dump_pre_morphosyntax_chat("test.wav", chat);
        dumper.dump_fa_group_result(
            "test.cha",
            0,
            &FaGroupDumpData {
                audio_start_ms: DurationMs(0),
                audio_end_ms: DurationMs(1000),
                words: vec!["hello".into()],
                timings: vec![Some(TimingPair {
                    start_ms: 0,
                    end_ms: 500,
                })],
            },
        );
    }

    #[test]
    fn enabled_dumper_writes_expected_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dumper = DebugDumper::new(Some(dir.path()));

        assert!(dumper.is_enabled());

        let chat = "@UTF8\n@Begin\n*CHI:\thello .\n@End";
        let tokens = vec![AsrTimingToken {
            text: "hello".into(),
            start_ms: 100,
            end_ms: 500,
        }];
        let utr_result = UtrResult::not_run_no_untimed(1);

        dumper.dump_utr_input("sample.cha", chat);
        dumper.dump_utr_tokens("sample.cha", &tokens);
        dumper.dump_utr_output("sample.cha", chat, &utr_result);
        dumper.dump_fa_output("sample.cha", chat);
        dumper.dump_asr_response(
            "sample.wav",
            &serde_json::json!({"tokens": [{"text": "hello"}]}),
        );
        dumper.dump_post_asr_chat("sample.wav", chat);
        dumper.dump_pre_utseg_chat("sample.wav", chat);
        dumper.dump_post_utseg_chat("sample.wav", chat);
        dumper.dump_pre_morphosyntax_chat("sample.wav", chat);
        dumper.dump_fa_group_result(
            "sample.cha",
            0,
            &FaGroupDumpData {
                audio_start_ms: DurationMs(0),
                audio_end_ms: DurationMs(1000),
                words: vec!["hello".into()],
                timings: vec![Some(TimingPair {
                    start_ms: 100,
                    end_ms: 500,
                })],
            },
        );

        // Verify files exist
        assert!(dir.path().join("sample_utr_input.cha").exists());
        assert!(dir.path().join("sample_utr_tokens.json").exists());
        assert!(dir.path().join("sample_utr_output.cha").exists());
        assert!(dir.path().join("sample_utr_result.json").exists());
        assert!(dir.path().join("sample_fa_output.cha").exists());
        assert!(dir.path().join("sample_fa_group_0.json").exists());
        assert!(dir.path().join("sample_asr_response.json").exists());
        assert!(dir.path().join("sample_post_asr.cha").exists());
        assert!(dir.path().join("sample_pre_utseg.cha").exists());
        assert!(dir.path().join("sample_post_utseg.cha").exists());
        assert!(dir.path().join("sample_pre_morphosyntax.cha").exists());

        // Verify tokens roundtrip
        let tokens_json = std::fs::read_to_string(dir.path().join("sample_utr_tokens.json"))
            .expect("read tokens");
        let parsed: Vec<AsrTimingToken> = serde_json::from_str(&tokens_json).expect("parse tokens");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].text, "hello");
        assert_eq!(parsed[0].start_ms, 100);

        // Verify FA group roundtrip
        let group_json =
            std::fs::read_to_string(dir.path().join("sample_fa_group_0.json")).expect("read group");
        let parsed: FaGroupDumpData = serde_json::from_str(&group_json).expect("parse group");
        assert_eq!(parsed.audio_start_ms, DurationMs(0));
        assert_eq!(parsed.audio_end_ms, DurationMs(1000));
        assert_eq!(parsed.words, vec!["hello"]);
        assert!(parsed.timings[0].is_some());
    }
}
