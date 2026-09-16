//! ASR response types, backend selection, and transcribe options.

use crate::api::{DurationSeconds, LanguageCode3, LanguageSpec};
// `SelectableEngine` is imported for its `ALL` associated constant, which is
// what makes the "engines that do work" list derived rather than restated.
use crate::types::engines::{AsrEngineName, SelectableEngine};
use crate::types::worker_v2::{AsrBackendV2, SpeakerBackendV2};
use batchalign_transform::asr_postprocess::AsrMonologue;
use serde::{Deserialize, Serialize};

/// Independent cache policy for each paid transcribe evidence boundary.
///
/// Keeping these policies in one named record prevents a Rev refresh from
/// implicitly refreshing speaker evidence (or vice versa) merely because both
/// stages happen to run inside one transcribe command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TranscribeCachePolicies {
    pub(crate) rev_asr: crate::params::CachePolicy,
    pub(crate) speaker: crate::params::CachePolicy,
}

impl TranscribeCachePolicies {
    pub(crate) fn uniform(policy: crate::params::CachePolicy) -> Self {
        Self {
            rev_asr: policy,
            speaker: policy,
        }
    }
}

// ---------------------------------------------------------------------------
// ASR response types (match Python inference/asr.py models)
// ---------------------------------------------------------------------------

/// A single raw ASR output token from the selected ASR backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrToken {
    /// Word text.
    pub text: String,
    /// Start time in seconds.
    pub start_s: Option<DurationSeconds>,
    /// End time in seconds.
    pub end_s: Option<DurationSeconds>,
    /// Speaker label (e.g. "0", "1") from diarization.
    pub speaker: Option<String>,
    /// Confidence score (0.0-1.0).
    pub confidence: Option<f64>,
}

/// Shared ASR inference response consumed by the Rust transcribe pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResponse {
    /// Raw tokens with timestamps and speaker labels.
    pub tokens: Vec<AsrToken>,
    /// Language code.
    pub lang: LanguageCode3,
    /// Optional provider-shaped monologues preserved from the ASR boundary.
    ///
    /// BA2 parity depends on not discarding provider punctuation elements and
    /// same-speaker monologue breaks before Rust post-processing runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_monologues: Option<Vec<AsrMonologue>>,
    /// The models that produced this response, as the runtime observed them.
    ///
    /// `None` only for a replayed LEGACY projection, which predates model
    /// identity and genuinely has none to report; `AsrIdentity::of_replay`
    /// records the same absence for the same reason. Every live path fills it,
    /// and the transcript's provenance stamp renders it as `asr_model=`.
    ///
    /// Recording the OBSERVED identity rather than the requested plan matters
    /// for exactly one case, and it is the case worth having: a floating model
    /// has no requested revision to name, while the worker still reports the
    /// commit it actually loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<crate::types::worker_v2::AsrModelIdentityV2>,
}

/// Which runtime boundary owns raw ASR inference for one command execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsrBackend {
    /// Use the Rust-owned Rev.AI client directly from the server.
    RustRevAi,
    /// Use the Rust-owned native Whisper path (whisper.cpp via whisper-rs),
    /// run in-process, bypassing the Python worker.
    RustWhisperRs,
    /// Use a Python worker path selected by a typed worker-mode value.
    Worker(AsrWorkerMode),
}

/// ASR backend proven not to be Rev.AI.
///
/// Raw Rev calls require a cache-miss authorization, so the generic worker
/// inference function accepts this smaller sum and cannot bypass that gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NonRevAsrBackend {
    RustWhisperRs,
    Worker(AsrWorkerMode),
}

/// Concrete Python-worker ASR execution mode selected by the Rust control
/// plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsrWorkerMode {
    /// Local Whisper via worker protocol V2 prepared-audio requests.
    LocalWhisperV2,
    /// HuggingFace Whisper fine-tune via V2 prepared-audio requests.
    /// Shares the Local Whisper request wire shape; the only difference
    /// is at worker load time, where the pool key ``whisper_hub`` makes
    /// the worker load the fine-tune instead of the stock OpenAI model.
    WhisperHubV2,
    /// Tencent ASR via worker protocol V2 provider-media requests.
    HkTencentV2,
    /// Aliyun ASR via worker protocol V2 provider-media requests.
    HkAliyunV2,
    /// FunAudio ASR via worker protocol V2 provider-media requests.
    HkFunaudioV2,
    /// Qwen3-ASR via worker protocol V2 provider-media requests.
    HkQwenV2,
}

impl AsrWorkerMode {
    /// Return the corresponding live V2 backend.
    ///
    /// Visible to the whole crate because the UTR cache namespace resolves the
    /// pinned plan for its engine and must reach the SAME mapping the request
    /// builder uses. A second copy next to the cache could disagree, and a
    /// namespace derived from a backend the run did not use is precisely the
    /// stale-reuse this pinning exists to prevent.
    pub(crate) fn as_v2_backend(self) -> AsrBackendV2 {
        match self {
            Self::LocalWhisperV2 => AsrBackendV2::LocalWhisper,
            Self::WhisperHubV2 => AsrBackendV2::WhisperHub,
            Self::HkTencentV2 => AsrBackendV2::HkTencent,
            Self::HkAliyunV2 => AsrBackendV2::HkAliyun,
            Self::HkFunaudioV2 => AsrBackendV2::HkFunaudio,
            Self::HkQwenV2 => AsrBackendV2::HkQwen,
        }
    }

    /// Stable engine name written into transcript provenance, checked as
    /// stamp-safe text at compile time.
    fn provenance_name(self) -> crate::api::StampSafeText {
        use crate::api::StampSafeText;
        match self {
            Self::LocalWhisperV2 => const { StampSafeText::from_static("whisper") },
            Self::WhisperHubV2 => const { StampSafeText::from_static("whisper_hub") },
            Self::HkTencentV2 => const { StampSafeText::from_static("tencent") },
            Self::HkAliyunV2 => const { StampSafeText::from_static("aliyun") },
            Self::HkFunaudioV2 => const { StampSafeText::from_static("funaudio") },
            Self::HkQwenV2 => const { StampSafeText::from_static("qwen") },
        }
    }
}

impl From<&crate::options::UtrEngine> for AsrBackend {
    fn from(engine: &crate::options::UtrEngine) -> Self {
        match engine {
            crate::options::UtrEngine::RevAi => Self::RustRevAi,
            crate::options::UtrEngine::Whisper => Self::Worker(AsrWorkerMode::LocalWhisperV2),
            crate::options::UtrEngine::HkTencent => Self::Worker(AsrWorkerMode::HkTencentV2),
        }
    }
}

impl AsrBackend {
    /// Select the runtime boundary for one typed ASR engine, or refuse.
    ///
    /// THE owner of "which runtime actually runs this engine". It takes the
    /// typed [`AsrEngineName`] rather than a string, and matches it
    /// EXHAUSTIVELY: there is no catch-all arm, so a new engine variant is a
    /// compile error here until somebody says what runs it.
    ///
    /// That absence is the whole point. The previous string form ended in
    /// `_ => LocalWhisperV2`, which silently mapped the two accepted-but-
    /// unimplemented names (`whisperx`, `whisper_oai`) onto stock local
    /// Whisper. Nothing in this workspace implements WhisperX or the OpenAI
    /// Whisper API, so those jobs ran an engine nobody asked for and recorded
    /// "whisper" in provenance. They are now a typed refusal instead.
    pub(crate) fn try_from_engine(
        engine: &AsrEngineName,
    ) -> Result<Self, crate::types::engines::EngineNotImplemented> {
        match engine {
            // Rust-owned runtimes: no Python worker involved.
            AsrEngineName::RevAi => Ok(Self::RustRevAi),
            AsrEngineName::WhisperRs => Ok(Self::RustWhisperRs),
            // Python-worker runtimes, one arm per live worker mode.
            AsrEngineName::Whisper => Ok(Self::Worker(AsrWorkerMode::LocalWhisperV2)),
            AsrEngineName::WhisperHub => Ok(Self::Worker(AsrWorkerMode::WhisperHubV2)),
            AsrEngineName::HkTencent => Ok(Self::Worker(AsrWorkerMode::HkTencentV2)),
            AsrEngineName::HkAliyun => Ok(Self::Worker(AsrWorkerMode::HkAliyunV2)),
            AsrEngineName::HkFunaudio => Ok(Self::Worker(AsrWorkerMode::HkFunaudioV2)),
            AsrEngineName::HkQwen => Ok(Self::Worker(AsrWorkerMode::HkQwenV2)),
            // Recognized names with nothing behind them.
            AsrEngineName::WhisperX | AsrEngineName::WhisperOai => {
                Err(crate::types::engines::EngineNotImplemented {
                    engine: engine.clone(),
                })
            }
        }
    }

    /// The wire names of every ASR engine this build can actually run.
    ///
    /// DERIVED from [`Self::try_from_engine`], so an operator-facing message
    /// listing the alternatives cannot recommend an engine that would itself
    /// be refused. This is the same discipline `validate_utr_language_support`
    /// applies to UTR engines: the remedy comes from the predicate that
    /// produced the rejection, never from a second hand-written list.
    pub(crate) fn implemented_engine_names() -> impl Iterator<Item = &'static str> {
        AsrEngineName::ALL
            .iter()
            .filter(|engine| Self::try_from_engine(engine).is_ok())
            .map(AsrEngineName::as_wire_name)
    }

    pub(crate) fn as_non_rev(self) -> Option<NonRevAsrBackend> {
        match self {
            Self::RustRevAi => None,
            Self::RustWhisperRs => Some(NonRevAsrBackend::RustWhisperRs),
            Self::Worker(mode) => Some(NonRevAsrBackend::Worker(mode)),
        }
    }

    /// Stable engine name written into transcript provenance and warnings.
    pub(crate) fn provenance_name(self) -> crate::api::StampSafeText {
        use crate::api::StampSafeText;
        match self {
            Self::RustRevAi => const { StampSafeText::from_static("rev") },
            Self::RustWhisperRs => const { StampSafeText::from_static("whisper_rs") },
            Self::Worker(mode) => mode.provenance_name(),
        }
    }

    /// The checkpoint this backend reads from its override key, admitted as
    /// stamp-safe text because provenance records it byte for byte.
    ///
    /// Only FunAudio reads one: `funaudio` spans materially different models
    /// (SenseVoice by default, Paraformer when selected). The match is
    /// exhaustive, so a new worker mode must decide. Called where options are
    /// parsed (submission validation) and where the transcribe plan is
    /// admitted, so a checkpoint provenance could not record is refused before
    /// any work runs.
    pub(crate) fn admit_checkpoint(
        self,
        engine_extras: &std::collections::BTreeMap<String, String>,
    ) -> Result<Option<crate::api::StampSafeText>, TranscribeAsrPlanError> {
        let key = match self {
            Self::Worker(AsrWorkerMode::HkFunaudioV2) => {
                crate::types::engines::FUNAUDIO_MODEL_OVERRIDE_KEY
            }
            Self::RustRevAi
            | Self::RustWhisperRs
            | Self::Worker(
                AsrWorkerMode::LocalWhisperV2
                | AsrWorkerMode::WhisperHubV2
                | AsrWorkerMode::HkTencentV2
                | AsrWorkerMode::HkAliyunV2
                | AsrWorkerMode::HkQwenV2,
            ) => return Ok(None),
        };
        engine_extras
            .get(key)
            .map(|checkpoint| {
                crate::api::StampSafeText::try_from(checkpoint.as_str())
                    .map_err(|reason| TranscribeAsrPlanError::InvalidCheckpoint { key, reason })
            })
            .transpose()
    }
}

/// The ASR identity a transcript's provenance records: the engine, and the
/// models it loaded.
///
/// The fields are private and the only constructors are
/// [`TranscribeAsrPlan::identity`] and [`AsrIdentity::of_replay`], neither of
/// which can name a model: a plan knows what it ASKED for, and a replayed
/// legacy projection predates model identity entirely. Only
/// [`AsrIdentity::with_loaded_models`] attaches models, and only a run that
/// reported them has one to pass.
///
/// The checkpoint a request selected is deliberately NOT carried here. It used
/// to be, and it fed both the stamp and the warning; a value describing what
/// was asked for, travelling beside one describing what ran, is the pair this
/// workstream exists to stop. The override is still admitted, and still
/// refused when it is not stamp-safe, by [`AsrBackend::admit_checkpoint`] at
/// submission and at plan admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AsrIdentity {
    engine: crate::api::StampSafeText,
    /// The models the run actually loaded, once one has reported them.
    ///
    /// Absent on a plan-built identity, because a plan knows what it ASKED for
    /// and not what came back, and absent for a replayed legacy projection,
    /// which predates model identity entirely.
    model: Option<crate::types::worker_v2::AsrModelIdentityV2>,
}

impl AsrIdentity {
    /// Identity of a replayed legacy projection, which records no checkpoint.
    pub(crate) fn of_replay(producer: super::replay::LegacyProjectedAsrProducer) -> Self {
        Self {
            engine: producer.provenance_name(),
            model: None,
        }
    }

    /// Attach the models a run reported loading.
    ///
    /// Consuming, so the plan-built identity is REPLACED by the one that knows
    /// what actually ran, rather than both staying in circulation where a
    /// caller could stamp the weaker of the two.
    pub(crate) fn with_loaded_models(
        mut self,
        model: crate::types::worker_v2::AsrModelIdentityV2,
    ) -> Self {
        self.model = Some(model);
        self
    }

    /// Engine name recorded as `asr=`.
    pub(crate) fn engine(&self) -> &crate::api::StampSafeText {
        &self.engine
    }

    /// The models this run loaded, each at the revision it was observed at,
    /// auxiliaries included.
    ///
    /// ONE accessor feeding BOTH lines a transcript carries about its ASR: the
    /// `asr_model=` stamp field and the human unchecked-ASR warning. Two lines
    /// describing one run that can drift apart is the defect this workstream
    /// removes, so they are not allowed to have separate sources.
    ///
    /// There is deliberately no fallback to the checkpoint the request
    /// selected. Where nothing reported its models, as when legacy evidence is
    /// replayed, this is `None` and both lines simply say less. Printing a
    /// requested checkpoint here would record what was ASKED FOR as though it
    /// had been seen, which is the exact substitution the bridge refuses.
    pub(crate) fn asr_model(&self) -> Option<crate::api::StampSafeText> {
        self.model.as_ref().map(|model| model.stamp_value())
    }
}

impl std::fmt::Display for AsrIdentity {
    /// `<engine> (<models>)`, the shape readers already parse, with the
    /// parenthetical naming what RAN rather than what was requested.
    ///
    /// Delegates to [`AsrIdentity::asr_model`] rather than formatting a second
    /// time, so the warning and the `asr_model=` stamp beside it cannot
    /// disagree. When no run reported its models the parenthetical is omitted
    /// entirely; it is never filled from the request.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.asr_model() {
            Some(model) => write!(f, "{} ({model})", self.engine),
            None => write!(f, "{}", self.engine),
        }
    }
}

/// Speaker-count policy accepted by the Rev.AI backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RevSpeakerCount {
    Automatic,
    Fixed(std::num::NonZeroU32),
}

/// Backend and speaker policy admitted together. Non-Rev inference cannot
/// carry automatic counts, and fixed counts cannot be zero or truncated.
///
/// The checkpoint a request selected is admitted here too, by
/// [`AsrBackend::admit_checkpoint`], but it is not STORED. Admission is a
/// refusal: an override that could not be written into a stamp fails the job
/// before any work runs. The value is not carried onward because provenance
/// records the models a run loaded, and a requested checkpoint travelling
/// beside an observed identity is the disagreeing pair this workstream removes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TranscribeAsrPlan {
    RevAi(RevSpeakerCount),
    NonRev {
        backend: NonRevAsrBackend,
        speakers: std::num::NonZeroU32,
    },
}

#[derive(Debug, thiserror::Error)]
/// Refusal to admit a backend, speaker-count and checkpoint request as one
/// executable plan.
pub enum TranscribeAsrPlanError {
    /// Automatic counts are not implemented by this backend.
    #[error("automatic speaker counts require the Rev.AI ASR engine")]
    UnsupportedAutomaticCount,
    /// A fixed count cannot be represented as a positive provider count.
    #[error("fixed speaker count must be in 1..=4294967295, got {0}")]
    InvalidFixedCount(usize),
    /// The checkpoint selected through the backend's override key cannot be
    /// recorded in provenance.
    #[error("ASR checkpoint override `{key}` cannot be recorded in provenance: {reason}")]
    InvalidCheckpoint {
        /// The override key the backend reads its checkpoint from.
        key: &'static str,
        /// Why the value is not stamp-safe text.
        reason: crate::api::InvalidStampSafeText,
    },
}

impl TranscribeAsrPlan {
    pub(crate) fn from_request(
        backend: AsrBackend,
        automatic: bool,
        count: usize,
        engine_extras: &std::collections::BTreeMap<String, String>,
    ) -> Result<Self, TranscribeAsrPlanError> {
        // Called for its REFUSAL, not for its value. A checkpoint override that
        // could not be written into a stamp fails the job here, before any work
        // runs, which is the whole point of admitting it at plan time. The
        // value itself is discarded: provenance names the models the run
        // loaded, never the one the request asked for.
        let _ = backend.admit_checkpoint(engine_extras)?;
        match (backend.as_non_rev(), automatic) {
            (None, true) => Ok(Self::RevAi(RevSpeakerCount::Automatic)),
            (Some(_), true) => Err(TranscribeAsrPlanError::UnsupportedAutomaticCount),
            (non_rev, false) => {
                let raw = u32::try_from(count)
                    .map_err(|_| TranscribeAsrPlanError::InvalidFixedCount(count))?;
                let count = std::num::NonZeroU32::new(raw)
                    .ok_or(TranscribeAsrPlanError::InvalidFixedCount(count))?;
                Ok(match non_rev {
                    None => Self::RevAi(RevSpeakerCount::Fixed(count)),
                    Some(backend) => Self::NonRev {
                        backend,
                        speakers: count,
                    },
                })
            }
        }
    }

    pub(crate) fn backend(&self) -> AsrBackend {
        match self {
            Self::RevAi(_) => AsrBackend::RustRevAi,
            Self::NonRev {
                backend: NonRevAsrBackend::RustWhisperRs,
                ..
            } => AsrBackend::RustWhisperRs,
            Self::NonRev {
                backend: NonRevAsrBackend::Worker(mode),
                ..
            } => AsrBackend::Worker(*mode),
        }
    }

    pub(crate) fn expected_speakers(&self) -> Option<crate::api::NumSpeakers> {
        match self {
            Self::RevAi(RevSpeakerCount::Automatic) => None,
            Self::RevAi(RevSpeakerCount::Fixed(count))
            | Self::NonRev {
                speakers: count, ..
            } => Some(crate::api::NumSpeakers(count.get())),
        }
    }

    /// The ASR identity transcript provenance records for this plan: the
    /// backend's engine name and the checkpoint admitted with the plan.
    pub(crate) fn identity(&self) -> AsrIdentity {
        match self {
            Self::RevAi(_) => AsrIdentity {
                engine: AsrBackend::RustRevAi.provenance_name(),
                model: None,
            },
            Self::NonRev { .. } => AsrIdentity {
                engine: self.backend().provenance_name(),
                model: None,
            },
        }
    }
}

/// Options controlling the transcribe pipeline after ASR-policy admission.
#[derive(Clone)]
pub struct TranscribeOptions {
    pub(crate) asr: TranscribeAsrPlan,
    /// Whether the command requested diarized speaker attribution.
    pub diarize: bool,
    /// Concrete speaker backend selected by Rust when dedicated diarization is needed.
    pub speaker_backend: Option<SpeakerBackendV2>,
    /// Language specification: `Auto` for ASR auto-detect, or a resolved code.
    ///
    /// The type system enforces that post-ASR stages (utseg, morphotag) must
    /// resolve `Auto` to a concrete language before calling NLP workers.
    pub lang: LanguageSpec,
    /// Whether to run the production utterance-segmentation topology. When
    /// enabled for supported languages this includes both the pre-CHAT word
    /// boundary pass and the post-CHAT refinement pass.
    pub with_utseg: bool,
    /// Whether to run morphosyntax after CHAT assembly.
    pub with_morphosyntax: bool,
    /// Independent inference/replay policy for Rev and speaker evidence.
    pub(crate) cache_policies: TranscribeCachePolicies,
    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback for utseg when no language-specific TalkBank BERT
    /// model is configured. Set by `--utseg-fallback-stanza` on the
    /// transcribe / transcribe-s CLI surface. Defaults to `false`.
    pub allow_stanza_fallback_utseg: bool,
    /// Whether to generate `%wor` tiers in the transcribe output.
    ///
    /// Defaults to `false` (BA2 parity: `--wor` was opt-in for transcribe).
    pub write_wor: bool,
    /// Media filename for the @Media header.
    pub media_name: Option<String>,
    /// Per-engine configuration extras drawn from
    /// `CommonOptions.engine_overrides.extras` (e.g. `qwen_model`,
    /// `qwen_device`, `funaudio_model`). Plumbed through the V2 dispatch
    /// boundary so they reach the worker spawn argv, the `backend` enum
    /// only carries WHICH engine, not its configuration.
    pub engine_extras: std::collections::BTreeMap<String, String>,
}

impl TranscribeOptions {
    /// Provider/diarizer count hint; automatic mode never uses a numeric sentinel.
    pub(crate) fn expected_speakers(&self) -> Option<crate::api::NumSpeakers> {
        self.asr.expected_speakers()
    }
}

#[cfg(test)]
mod speaker_plan_tests {
    use super::*;

    #[test]
    fn asr_response_requires_declared_language_instead_of_inventing_english() {
        let missing = serde_json::json!({"tokens": []});
        assert!(serde_json::from_value::<AsrResponse>(missing).is_err());
        let declared = serde_json::json!({"tokens": [], "lang": "eng"});
        assert_eq!(
            serde_json::from_value::<AsrResponse>(declared)
                .unwrap()
                .lang,
            LanguageCode3::eng()
        );
    }

    #[test]
    fn backend_count_admission_preserves_automatic_and_fixed_policies() {
        let none = std::collections::BTreeMap::new();
        let automatic =
            TranscribeAsrPlan::from_request(AsrBackend::RustRevAi, true, 2, &none).unwrap();
        assert_eq!(
            automatic,
            TranscribeAsrPlan::RevAi(RevSpeakerCount::Automatic)
        );
        assert_eq!(automatic.expected_speakers(), None);
        for backend in [
            AsrBackend::RustRevAi,
            AsrBackend::RustWhisperRs,
            AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2),
        ] {
            let fixed = TranscribeAsrPlan::from_request(backend, false, 3, &none).unwrap();
            assert_eq!(fixed.backend(), backend);
            assert_eq!(fixed.expected_speakers(), Some(crate::api::NumSpeakers(3)));
            assert!(matches!(
                TranscribeAsrPlan::from_request(backend, false, 0, &none),
                Err(TranscribeAsrPlanError::InvalidFixedCount(0))
            ));
            if backend != AsrBackend::RustRevAi {
                assert!(matches!(
                    TranscribeAsrPlan::from_request(backend, true, 3, &none),
                    Err(TranscribeAsrPlanError::UnsupportedAutomaticCount)
                ));
            }
        }
        if usize::BITS > 32 {
            assert!(matches!(
                TranscribeAsrPlan::from_request(AsrBackend::RustRevAi, false, usize::MAX, &none),
                Err(TranscribeAsrPlanError::InvalidFixedCount(_))
            ));
        }
    }

    /// Only the backend that reads a checkpoint admits one, and only as
    /// stamp-safe text; the same override on another backend is not attached.
    #[test]
    fn a_checkpoint_is_admitted_only_by_the_backend_that_reads_it() {
        let funaudio = AsrBackend::Worker(AsrWorkerMode::HkFunaudioV2);
        let paraformer = std::collections::BTreeMap::from([(
            crate::types::engines::FUNAUDIO_MODEL_OVERRIDE_KEY.to_owned(),
            "paraformer-zh".to_owned(),
        )]);
        // Asserted on the admission itself rather than through the identity's
        // `Display`. Provenance now names the models a run LOADED, so a
        // plan-built identity carries no parenthetical at all, and testing
        // this behaviour through `Display` would prove only that. The subject
        // here is which backend READS the override key, which is exactly what
        // `admit_checkpoint` decides.
        assert!(
            matches!(
                funaudio.admit_checkpoint(&paraformer),
                Ok(Some(ref checkpoint)) if checkpoint.as_str() == "paraformer-zh"
            ),
            "the backend that reads this override key must admit its value"
        );
        let whisper = AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2);
        assert!(
            matches!(whisper.admit_checkpoint(&paraformer), Ok(None)),
            "the same override must not attach to a backend that reads no checkpoint"
        );
        // Both are still admissible plans; the override simply does not attach.
        TranscribeAsrPlan::from_request(funaudio, false, 1, &paraformer).unwrap();
        TranscribeAsrPlan::from_request(whisper, false, 1, &paraformer).unwrap();

        let padded = std::collections::BTreeMap::from([(
            crate::types::engines::FUNAUDIO_MODEL_OVERRIDE_KEY.to_owned(),
            "paraformer-zh ".to_owned(),
        )]);
        assert!(matches!(
            TranscribeAsrPlan::from_request(funaudio, false, 1, &padded),
            Err(TranscribeAsrPlanError::InvalidCheckpoint { .. })
        ));
    }

    #[test]
    fn replay_policy_refusal_has_a_typed_usage_exit() {
        let error = crate::cli::error::CliError::from(TranscribeAsrPlanError::InvalidFixedCount(0));
        assert_eq!(error.exit_code(), crate::cli::error::CliError::EXIT_USAGE);
    }
}

#[cfg(test)]
mod tests {
    //! Worker-mode wiring for the ``whisper_hub`` engine.
    //!
    //! ``AsrWorkerMode`` is the control-plane dispatch selector; it must
    //! agree with ``AsrBackendV2`` (the wire IPC enum) and the pool-key
    //! override name in ``worker/pool/execute_v2.rs``. A new engine
    //! variant that lands in only one of those three places will
    //! mis-route at dispatch time.
    use super::*;

    #[test]
    fn whisper_hub_worker_mode_lowers_to_whisper_hub_backend() {
        assert_eq!(
            AsrWorkerMode::WhisperHubV2.as_v2_backend(),
            AsrBackendV2::WhisperHub,
        );
    }

    #[test]
    fn asr_backend_from_whisper_hub_is_worker_path_not_rev_ai() {
        // ``whisper_hub`` is not Rust-owned; it must go to the Worker
        // path just like stock Whisper, HK engines, etc.
        assert_eq!(
            AsrBackend::try_from_engine(&AsrEngineName::WhisperHub)
                .expect("whisper_hub is implemented"),
            AsrBackend::Worker(AsrWorkerMode::WhisperHubV2),
        );
    }

    #[test]
    fn asr_backend_from_whisper_rs_is_rust_native_path() {
        // ``whisper_rs`` is Rust-owned (in-process whisper.cpp), so it must
        // route to the dedicated native backend, never the Python worker.
        assert_eq!(
            AsrBackend::try_from_engine(&AsrEngineName::WhisperRs)
                .expect("whisper_rs is implemented"),
            AsrBackend::RustWhisperRs,
        );
    }

    #[test]
    fn asr_backend_from_rev_is_rust_revai_path() {
        assert_eq!(
            AsrBackend::try_from_engine(&AsrEngineName::RevAi).expect("rev is implemented"),
            AsrBackend::RustRevAi,
        );
    }

    /// The two engines this build accepts as NAMES and cannot run.
    ///
    /// Selection used to end in a catch-all that mapped both onto stock local
    /// Whisper, so the wrong engine ran and provenance said "whisper". The
    /// refusal must name the engine asked for; the caller derives the list of
    /// working alternatives from the same function.
    #[test]
    fn unimplemented_asr_engines_are_refused_rather_than_defaulted() {
        for engine in [AsrEngineName::WhisperX, AsrEngineName::WhisperOai] {
            let refusal = AsrBackend::try_from_engine(&engine).expect_err(
                "nothing in this tree implements WhisperX or the OpenAI Whisper API; \
                 selecting one must refuse, never silently run stock local Whisper",
            );
            assert_eq!(refusal.engine, engine);
            assert!(
                refusal.to_string().contains(engine.as_wire_name()),
                "the refusal must name the engine that was asked for",
            );
        }
    }

    /// The alternatives offered to an operator are exactly the engines that
    /// resolve, so the list cannot recommend an unimplemented engine.
    #[test]
    fn implemented_engine_names_exclude_the_unimplemented_ones() {
        let implemented: Vec<&str> = AsrBackend::implemented_engine_names().collect();
        assert!(implemented.contains(&"rev"));
        assert!(implemented.contains(&"whisper"));
        assert!(implemented.contains(&"whisper_rs"));
        assert!(!implemented.contains(&"whisperx"));
        assert!(!implemented.contains(&"whisper_oai"));
        assert_eq!(
            implemented.len() + 2,
            AsrEngineName::ALL.len(),
            "exactly two accepted ASR engine names are unimplemented; if this count \
             moved, either an engine was implemented or a new stub was added",
        );
    }
}
