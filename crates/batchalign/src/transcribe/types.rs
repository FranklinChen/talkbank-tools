//! ASR response types, backend selection, and transcribe options.

use crate::api::{
    AsrLanguageRequest, AudioPositionSeconds, LanguageCode3, LanguagePair, LanguageSpec,
    PerFileHasNoAsrLanguage, WorkerLanguage,
};
use crate::types::revai_language::{RevLanguage, RevLanguageRefusal};
// `SelectableEngine` is imported for its `ALL` associated constant, which is
// what makes the "engines that do work" list derived rather than restated.
use crate::types::engines::{AsrEngineName, SelectableEngine};
use crate::types::worker_v2::{AsrBackendV2, ProviderDiarizationV2, SpeakerBackendV2};
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
    /// Start of the token, as a position in the audio.
    pub start_s: Option<AudioPositionSeconds>,
    /// End of the token, as a position in the audio.
    pub end_s: Option<AudioPositionSeconds>,
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
    /// Repetitions at a Whisper window seam that the reconciler kept once
    /// (`worker::whisper_windows`), in recording order. Placed on the
    /// utterance that keeps each copy once the utterances exist
    /// (`asr_postprocess::place_seam_ambiguities`), written there as a
    /// `%com`, and reported to the job as a shortfall. Empty for every other
    /// engine and for an unambiguous stitch; absent from the serialized
    /// evidence then, so earlier `_asr_response.json` files still read.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seam_ambiguities: Vec<batchalign_transform::asr_postprocess::SeamAmbiguity>,
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

    /// Whether this mode hands the worker the media file itself (a provider
    /// reads it) rather than audio Rust prepared.
    ///
    /// Decides whether a caller needs to measure the file's length for a
    /// decode budget at all: a prepared-audio request derives its budget from
    /// the attachment it built, so measuring for it would be a wasted walk of
    /// the file. Exhaustive, so a new mode must say which it is.
    pub(crate) fn reads_provider_media(self) -> bool {
        match self {
            Self::LocalWhisperV2 | Self::WhisperHubV2 => false,
            Self::HkTencentV2 | Self::HkAliyunV2 | Self::HkFunaudioV2 | Self::HkQwenV2 => true,
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
        Self::implemented_engines().map(AsrEngineName::as_wire_name)
    }

    /// Every ASR engine this build can run, in help-display order.
    pub(crate) fn implemented_engines() -> impl Iterator<Item = &'static AsrEngineName> {
        AsrEngineName::ALL
            .iter()
            .filter(|engine| Self::try_from_engine(engine).is_ok())
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

/// How many speakers a transcribe run expects.
///
/// Decided ONCE, at plan admission, from the request's `auto_speakers` and
/// `num_speakers`, and read by every speaker engine the run involves: Rev.AI's
/// integrated diarization (through [`SpeakerCount::known`], where `None` is
/// "infer"), Tencent's (through the [`ProviderDiarizationV2`] the plan
/// derives) and a dedicated speaker engine. The recognizer path never sees
/// it: a count was a property of the ASR plan until 2026-11-03, which is why
/// `--auto-speakers` was refused for every non-Rev recognizer even when only
/// the dedicated diarizer needed the count, and that diarizer takes `Auto`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpeakerCount {
    /// Exactly this many. A `u8`: no recording has more than 255 speakers,
    /// and zero names nobody.
    Known(std::num::NonZeroU8),
    /// The speaker engine infers the count.
    Auto,
}

impl SpeakerCount {
    /// Admit the request's two fields. `auto_speakers` wins (the CLI makes
    /// the two conflict); otherwise the count must be in `1..=255`.
    pub(crate) fn admit(
        auto_speakers: bool,
        count: crate::api::NumSpeakers,
    ) -> Result<Self, SpeakerPlanError> {
        if auto_speakers {
            return Ok(Self::Auto);
        }
        u8::try_from(count.0)
            .ok()
            .and_then(std::num::NonZeroU8::new)
            .map(Self::Known)
            .ok_or(SpeakerPlanError::InvalidCount(count.0))
    }

    /// The count as a provider parameter: `None` is "infer", never a sentinel.
    pub(crate) fn known(self) -> Option<crate::api::NumSpeakers> {
        match self {
            Self::Known(count) => Some(crate::api::NumSpeakers(u32::from(count.get()))),
            Self::Auto => None,
        }
    }
}

/// Who separates the speakers of a transcribe run beyond what the recognizer
/// labels itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpeakerSeparation {
    /// Only the recognizer's own labels: Rev.AI's or Tencent's integrated
    /// diarization, or none for an engine that labels no speakers.
    RecognizerOnly,
    /// A dedicated speaker engine runs after recognition; its labels replace
    /// the recognizer's.
    Dedicated(SpeakerBackendV2),
}

impl SpeakerSeparation {
    /// From the request: `diarize` selects a dedicated engine, the override
    /// names it, and pyannoteAI Precision-2 is the default.
    pub(crate) fn from_request(
        diarize: bool,
        engine: Option<crate::types::engines::SpeakerEngineName>,
    ) -> Self {
        match diarize {
            true => Self::Dedicated(
                engine
                    .unwrap_or(crate::types::engines::SpeakerEngineName::PyannoteAi)
                    .worker_backend(),
            ),
            false => Self::RecognizerOnly,
        }
    }
}

/// The speaker plan of a transcribe run: who separates speakers and how many
/// to expect, admitted together with the recognizer, so a count an engine the
/// run involves cannot take is refused here by name and re-checked nowhere.
///
/// Built only by [`SpeakerPlan::admit`], which is called at submission (so
/// the operator hears a refusal before the job exists) and again at dispatch
/// (a persisted job predates nothing it cannot pass), both from the same
/// request fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpeakerPlan {
    count: SpeakerCount,
    separation: SpeakerSeparation,
    /// What a worker-hosted recognizer is told about speakers, derived here
    /// once from the count and the recognizer's integrated diarization. The
    /// recognizer path reads this and never the count.
    recognizer_diarization: ProviderDiarizationV2,
}

/// The run's recognizer, as the speaker plan needs it: its backend and
/// whether it separates speakers itself.
///
/// Separation is a property of the backend and, for Tencent, of the engine
/// the account chose (`16k_yue` labels no speakers, `16k_zh_en_2.0` does).
/// Two constructors, one per vantage point: submission cannot see the
/// server's account, the server's plan can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Recognizer {
    backend: AsrBackend,
    separation: RecognizerSeparation,
}

/// Whether a recognizer separates speakers itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecognizerSeparation {
    /// It attributes words to speakers when given a count.
    Separates,
    /// It labels no speakers, so it is never asked to separate them.
    LabelsNoSpeakers,
}

impl Recognizer {
    /// As submission sees it: the backend alone. Tencent counts as
    /// separating, since some of its engines do; the server's plan decides
    /// for the account's engine.
    pub(crate) fn submitted(backend: AsrBackend) -> Self {
        Self {
            backend,
            separation: Self::backend_separation(backend),
        }
    }

    /// As the server plans it, with the engine its Tencent account chose. An
    /// account with no usable engine is refused when the models are
    /// resolved, so its separation is the backend's here.
    pub(crate) fn planned(
        backend: AsrBackend,
        tencent: &crate::model_manifest::TencentEngineSetting,
    ) -> Self {
        use crate::model_manifest::TencentEngineSetting as T;
        let separation = match (backend, tencent) {
            (AsrBackend::Worker(AsrWorkerMode::HkTencentV2), T::Chosen(engine)) => {
                if engine.separates_speakers() {
                    RecognizerSeparation::Separates
                } else {
                    RecognizerSeparation::LabelsNoSpeakers
                }
            }
            (_, T::Chosen(_) | T::Unset | T::Unusable(_)) => Self::backend_separation(backend),
        };
        Self {
            backend,
            separation,
        }
    }

    fn backend_separation(backend: AsrBackend) -> RecognizerSeparation {
        match backend {
            AsrBackend::Worker(mode) if mode.as_v2_backend().separates_speakers() => {
                RecognizerSeparation::Separates
            }
            AsrBackend::Worker(_) | AsrBackend::RustRevAi | AsrBackend::RustWhisperRs => {
                RecognizerSeparation::LabelsNoSpeakers
            }
        }
    }
}

/// Refusal to admit a speaker count and separation as one plan.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpeakerPlanError {
    /// The count is not one a speaker engine can take.
    #[error(
        "expected speaker count must be positive and at most 255, got {0}; use auto_speakers for inference"
    )]
    InvalidCount(u32),
    /// Policy: a diarization speaker count is either exactly N (N at least 2)
    /// or automatic. Asking to SEPARATE speakers while asserting the
    /// recording has one is a contradiction, not a request for detection, and
    /// silently detecting would do something other than what was asked. What
    /// it used to do was obey: the count reached the diarizer, which returned
    /// one track, which is why `--diarization enabled` runs came back with a
    /// single `PAR0`.
    #[error(
        "diarization was requested with a speaker count of 1, which asks a diarizer to \
         separate speakers in a recording asserted to have one. Pass the real count \
         (2 or more); or omit diarization, for a single-speaker recording; or pass \
         auto_speakers to have the count inferred."
    )]
    SeparatingOneSpeaker,
    /// The recognizer separates speakers itself but only into a count it is
    /// given: on the worker wire there is no way to ask it to infer one. Only
    /// when it is the run's one speaker engine: with a dedicated diarizer the
    /// recognizer is asked for no separation and the diarizer infers the
    /// count.
    #[error(
        "automatic speaker counts are not supported by {engine}'s integrated diarization, \
         which must be given the count; pass --num-speakers, or --diarization enabled to \
         have a dedicated speaker engine infer it"
    )]
    AutoCountUnsupported {
        /// The recognizer's engine name, as provenance spells it.
        engine: crate::api::StampSafeText,
    },
}

impl SpeakerPlan {
    /// Admit `count` and `separation` for a run whose recognizer is
    /// `recognizer` (`None` for a replay, which runs none).
    pub(crate) fn admit(
        count: SpeakerCount,
        separation: SpeakerSeparation,
        recognizer: Option<Recognizer>,
    ) -> Result<Self, SpeakerPlanError> {
        if let (SpeakerSeparation::Dedicated(_), SpeakerCount::Known(one)) = (separation, count)
            && one.get() == 1
        {
            return Err(SpeakerPlanError::SeparatingOneSpeaker);
        }
        let recognizer_diarization = match recognizer.map(|r| (r.backend, r.separation)) {
            // No worker request carries these: a replay runs no recognizer,
            // native Whisper labels no speakers, and Rev.AI's Rust client
            // reads the count itself through `SpeakerCount::known`, taking
            // either kind.
            None | Some((AsrBackend::RustRevAi | AsrBackend::RustWhisperRs, _)) => {
                ProviderDiarizationV2::NotRequested
            }
            // A worker recognizer that labels no speakers (Whisper, or a
            // Tencent engine such as `16k_yue`) is never asked to separate
            // them, so it needs no count either.
            Some((AsrBackend::Worker(_), RecognizerSeparation::LabelsNoSpeakers)) => {
                ProviderDiarizationV2::NotRequested
            }
            Some((AsrBackend::Worker(mode), RecognizerSeparation::Separates)) => {
                let backend = mode.as_v2_backend();
                match (separation, count) {
                    // A dedicated engine's labels replace the recognizer's,
                    // so the recognizer is asked for no separation and the
                    // count, known or inferred, is the dedicated engine's.
                    // Until 2026-11-03 Tencent was asked to separate anyway,
                    // and refused `Auto` for a count it was not going to use.
                    (SpeakerSeparation::Dedicated(_), _) => ProviderDiarizationV2::NotRequested,
                    (SpeakerSeparation::RecognizerOnly, SpeakerCount::Known(known)) => {
                        ProviderDiarizationV2::for_backend(
                            backend,
                            crate::api::NumSpeakers(u32::from(known.get())),
                        )
                    }
                    // The recognizer is the run's one speaker engine, and its
                    // integrated diarization must be given the count.
                    (SpeakerSeparation::RecognizerOnly, SpeakerCount::Auto) => {
                        return Err(SpeakerPlanError::AutoCountUnsupported {
                            engine: AsrBackend::Worker(mode).provenance_name(),
                        });
                    }
                }
            }
        };
        Ok(Self {
            count,
            separation,
            recognizer_diarization,
        })
    }

    /// A plan for tests: `count` exact, a dedicated engine or none, no
    /// recognizer.
    #[cfg(test)]
    pub(crate) fn for_test(count: u32, dedicated: Option<SpeakerBackendV2>) -> Self {
        Self::admit(
            SpeakerCount::admit(false, crate::api::NumSpeakers(count)).expect("test: a count"),
            dedicated.map_or(
                SpeakerSeparation::RecognizerOnly,
                SpeakerSeparation::Dedicated,
            ),
            None,
        )
        .expect("test: an admissible speaker plan")
    }

    /// How many speakers the run expects.
    pub(crate) fn count(self) -> SpeakerCount {
        self.count
    }

    /// The dedicated speaker engine, when one runs after recognition.
    pub(crate) fn dedicated(self) -> Option<SpeakerBackendV2> {
        match self.separation {
            SpeakerSeparation::Dedicated(backend) => Some(backend),
            SpeakerSeparation::RecognizerOnly => None,
        }
    }

    /// What a worker-hosted recognizer is told about speakers.
    pub(crate) fn recognizer_diarization(self) -> ProviderDiarizationV2 {
        self.recognizer_diarization
    }
}

/// A language request for an engine that recognizes one language at a time:
/// every ASR engine but Rev.AI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SingleAsrLanguage {
    /// One known language.
    One(LanguageCode3),
    /// The engine detects the language.
    Detect,
}

impl SingleAsrLanguage {
    /// The requested language, when one was requested rather than detected.
    pub(crate) fn requested(&self) -> Option<&LanguageCode3> {
        match self {
            Self::One(code) => Some(code),
            Self::Detect => None,
        }
    }
}

impl From<&SingleAsrLanguage> for AsrLanguageRequest {
    fn from(language: &SingleAsrLanguage) -> Self {
        match language {
            SingleAsrLanguage::One(code) => Self::One(code.clone()),
            SingleAsrLanguage::Detect => Self::Detect,
        }
    }
}

impl From<&SingleAsrLanguage> for WorkerLanguage {
    /// What a worker-hosted engine is started with: the language, or `auto`.
    fn from(language: &SingleAsrLanguage) -> Self {
        match language {
            SingleAsrLanguage::One(code) => Self::Resolved(code.clone()),
            SingleAsrLanguage::Detect => Self::Auto,
        }
    }
}

/// A job's language admitted for one backend, with that backend.
///
/// The one admission of a language against an engine: plan construction uses
/// it, and submission validation calls it for a code-switched pair, so the two
/// refuse with the same message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AdmittedAsrLanguage {
    /// Rev.AI, with a language it recognizes.
    RevAi(RevLanguage),
    /// Another engine, with one language or detection.
    NonRev {
        backend: NonRevAsrBackend,
        language: SingleAsrLanguage,
    },
}

impl AdmittedAsrLanguage {
    /// Admit `language` for `backend`, or say why not.
    pub(crate) fn admit(
        backend: AsrBackend,
        language: &LanguageSpec,
    ) -> Result<Self, TranscribeAsrPlanError> {
        // A pair is withheld from transcription on every engine, here as well
        // as at submission: a job saved under an earlier build and restarted
        // after this one plans through here without being resubmitted, and
        // the withholding (`LanguagePairSupport::Withheld`, the measured
        // reason) must hold for it too. Refused before the engine is asked,
        // so neither engine arm below can see a pair.
        let request = match AsrLanguageRequest::try_from(language)? {
            AsrLanguageRequest::Pair(pair) => {
                return Err(TranscribeAsrPlanError::PairWithheld(pair));
            }
            single => single,
        };
        match backend.as_non_rev() {
            None => Ok(Self::RevAi(RevLanguage::admit(&request)?)),
            Some(backend) => Ok(Self::NonRev {
                backend,
                language: match request {
                    AsrLanguageRequest::One(code) => SingleAsrLanguage::One(code),
                    AsrLanguageRequest::Detect => SingleAsrLanguage::Detect,
                    AsrLanguageRequest::Pair(pair) => {
                        return Err(TranscribeAsrPlanError::PairWithheld(pair));
                    }
                },
            }),
        }
    }
}

/// Backend, speaker policy and language admitted together. Non-Rev inference
/// cannot carry automatic counts, fixed counts cannot be zero or truncated,
/// and only Rev.AI takes a code-switched language pair: each engine's plan
/// holds the language type that engine accepts, so a pair cannot reach an
/// engine that recognizes one language at a time.
///
/// The representation is private: [`Self::from_request`] is the only way to
/// build a live plan. Stages read it through [`Self::inference`].
///
/// The checkpoint a request selected is admitted here too, by
/// [`AsrBackend::admit_checkpoint`], but it is not STORED. Admission is a
/// refusal: an override that could not be written into a stamp fails the job
/// before any work runs. The value is not carried onward because provenance
/// records the models a run loaded, and a requested checkpoint travelling
/// beside an observed identity is the disagreeing pair this workstream removes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TranscribeAsrPlan(AsrPlanKind);

/// The recognizer and the language it was admitted with. No speaker count:
/// that is the speaker plan's ([`SpeakerPlan`]), and the recognizer path
/// cannot see or refuse it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AsrPlanKind {
    RevAi {
        language: RevLanguage,
    },
    NonRev {
        backend: NonRevAsrBackend,
        language: SingleAsrLanguage,
    },
}

/// What the ASR stage runs for a plan.
pub(crate) enum AsrInference<'p> {
    /// Rev.AI, through its evidence cache.
    RevAi { language: &'p RevLanguage },
    /// Another engine.
    NonRev {
        backend: NonRevAsrBackend,
        language: &'p SingleAsrLanguage,
    },
}

#[derive(Debug, thiserror::Error)]
/// Refusal to admit a backend, language and checkpoint request as one
/// executable plan.
pub enum TranscribeAsrPlanError {
    /// The checkpoint selected through the backend's override key cannot be
    /// recorded in provenance.
    #[error("ASR checkpoint override `{key}` cannot be recorded in provenance: {reason}")]
    InvalidCheckpoint {
        /// The override key the backend reads its checkpoint from.
        key: &'static str,
        /// Why the value is not stamp-safe text.
        reason: crate::api::InvalidStampSafeText,
    },
    /// Transcription always has a job-level language.
    #[error(transparent)]
    NoJobLanguage(#[from] PerFileHasNoAsrLanguage),
    /// Rev.AI cannot take the requested language.
    #[error(transparent)]
    RevLanguage(#[from] RevLanguageRefusal),
    /// A code-switched pair was requested from an engine that recognizes one
    /// language at a time.
    /// A pair reached engine admission: through a job saved before the
    /// withholding, since submission refuses every pair first.
    #[error("{}", crate::dispatch_language::LanguagePairSupport::withheld_message(.0))]
    PairWithheld(LanguagePair),
}

impl TranscribeAsrPlan {
    pub(crate) fn from_request(
        backend: AsrBackend,
        engine_extras: &std::collections::BTreeMap<String, String>,
        language: &LanguageSpec,
    ) -> Result<Self, TranscribeAsrPlanError> {
        // Called for its REFUSAL, not for its value. A checkpoint override that
        // could not be written into a stamp fails the job here, before any work
        // runs, which is the whole point of admitting it at plan time. The
        // value itself is discarded: provenance names the models the run
        // loaded, never the one the request asked for.
        let _ = backend.admit_checkpoint(engine_extras)?;
        Ok(Self(match AdmittedAsrLanguage::admit(backend, language)? {
            AdmittedAsrLanguage::RevAi(language) => AsrPlanKind::RevAi { language },
            AdmittedAsrLanguage::NonRev { backend, language } => {
                AsrPlanKind::NonRev { backend, language }
            }
        }))
    }

    /// What the ASR stage runs.
    pub(crate) fn inference(&self) -> AsrInference<'_> {
        match &self.0 {
            AsrPlanKind::RevAi { language } => AsrInference::RevAi { language },
            AsrPlanKind::NonRev { backend, language } => AsrInference::NonRev {
                backend: *backend,
                language,
            },
        }
    }

    /// The language this plan asks recognition for.
    pub(crate) fn language(&self) -> AsrLanguageRequest {
        match &self.0 {
            AsrPlanKind::RevAi { language } => language.request(),
            AsrPlanKind::NonRev { language, .. } => AsrLanguageRequest::from(language),
        }
    }

    pub(crate) fn backend(&self) -> AsrBackend {
        match &self.0 {
            AsrPlanKind::RevAi { .. } => AsrBackend::RustRevAi,
            AsrPlanKind::NonRev {
                backend: NonRevAsrBackend::RustWhisperRs,
                ..
            } => AsrBackend::RustWhisperRs,
            AsrPlanKind::NonRev {
                backend: NonRevAsrBackend::Worker(mode),
                ..
            } => AsrBackend::Worker(*mode),
        }
    }

    /// The ASR identity transcript provenance records for this plan: the
    /// backend's engine name. A run adds the models it loaded.
    pub(crate) fn identity(&self) -> AsrIdentity {
        AsrIdentity {
            engine: self.backend().provenance_name(),
            model: None,
        }
    }
}

/// Replay has no live inference operation and accepts transcript languages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ReplayAsrPlan {
    language: AsrLanguageRequest,
}

impl ReplayAsrPlan {
    pub(crate) fn for_legacy_replay(
        language: &LanguageSpec,
    ) -> Result<Self, TranscribeAsrPlanError> {
        Ok(Self {
            language: AsrLanguageRequest::try_from(language)?,
        })
    }
}

/// Shared post-processing policy, sealed to the two admitted execution modes.
/// `Send + Sync`: a plan travels inside the pipeline context through every
/// stage future, and stage futures are heap-owned `Send` futures
/// (`pipeline::plan::StageFuture`); the bound states what spawned execution
/// already relied on implicitly.
pub(crate) trait TranscribePlan: Clone + Send + Sync + sealed::Sealed {
    const REPLAY: bool;
    fn language(&self) -> AsrLanguageRequest;
    /// The engine recognition runs on, or `None` when the plan replays
    /// recorded recognition and runs no ASR model.
    fn recognition_backend(&self) -> Option<AsrBackend>;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::TranscribeAsrPlan {}
    impl Sealed for super::ReplayAsrPlan {}
}

impl TranscribePlan for TranscribeAsrPlan {
    const REPLAY: bool = false;
    fn language(&self) -> AsrLanguageRequest {
        self.language()
    }
    fn recognition_backend(&self) -> Option<AsrBackend> {
        Some(self.backend())
    }
}

impl TranscribePlan for ReplayAsrPlan {
    const REPLAY: bool = true;
    fn language(&self) -> AsrLanguageRequest {
        self.language.clone()
    }
    /// Replay reads recorded recognition evidence and runs no ASR model.
    fn recognition_backend(&self) -> Option<AsrBackend> {
        None
    }
}

/// ASR and segmentation admitted together, with no independently mutable
/// language or fallback policy for a consumer to substitute.
#[derive(Clone)]
pub(crate) struct AdmittedTranscribePlan<P> {
    asr: P,
    segmentation: SegmentationAdmission,
}

#[derive(Clone)]
enum SegmentationAdmission {
    Disabled,
    Known(crate::transcribe::segmentation::TranscribeSegmentation),
    Detect(crate::params::UtsegFallbackPolicy),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SegmentationResolutionError {
    #[error(transparent)]
    Unavailable(#[from] crate::utseg_route::UtsegUnavailable),
    #[error(
        "recognized language '{actual}' differs from admitted segmentation language '{expected}'"
    )]
    ChangedLanguage {
        expected: LanguageCode3,
        actual: LanguageCode3,
    },
}

impl<P: TranscribePlan> AdmittedTranscribePlan<P> {
    pub(crate) fn admit(
        asr: P,
        enabled: bool,
        fallback: crate::params::UtsegFallbackPolicy,
    ) -> Result<Self, crate::utseg_route::UtsegUnavailable> {
        use crate::transcribe::segmentation::TranscribeSegmentation;
        let segmentation = if !enabled {
            SegmentationAdmission::Disabled
        } else {
            match asr.language() {
                AsrLanguageRequest::One(code) => {
                    SegmentationAdmission::Known(TranscribeSegmentation::resolve(&code, fallback)?)
                }
                AsrLanguageRequest::Pair(pair) => SegmentationAdmission::Known(
                    TranscribeSegmentation::resolve(pair.primary(), fallback)?,
                ),
                AsrLanguageRequest::Detect => SegmentationAdmission::Detect(fallback),
            }
        };
        Ok(Self { asr, segmentation })
    }

    pub(crate) fn asr(&self) -> &P {
        &self.asr
    }

    pub(crate) fn with_utseg(&self) -> bool {
        !matches!(self.segmentation, SegmentationAdmission::Disabled)
    }

    /// Recognition finishes only the deferred route. A known route is carried
    /// forward unchanged; disagreement is refused instead of re-admitted.
    pub(crate) fn resolve_segmentation(
        &self,
        language: &crate::api::TranscriptLanguage,
    ) -> Result<
        Option<crate::transcribe::segmentation::TranscribeSegmentation>,
        SegmentationResolutionError,
    > {
        match &self.segmentation {
            SegmentationAdmission::Disabled => Ok(None),
            SegmentationAdmission::Known(route) if route.language() == language.primary() => {
                Ok(Some(route.clone()))
            }
            SegmentationAdmission::Known(route) => {
                Err(SegmentationResolutionError::ChangedLanguage {
                    expected: route.language().clone(),
                    actual: language.primary().clone(),
                })
            }
            SegmentationAdmission::Detect(fallback) => Ok(Some(
                crate::transcribe::segmentation::TranscribeSegmentation::resolve(
                    language.primary(),
                    *fallback,
                )?,
            )),
        }
    }
}

/// Options controlling the transcribe pipeline after ASR-policy admission.
#[derive(Clone)]
pub(crate) struct TranscribeOptions<P = TranscribeAsrPlan> {
    pub(crate) plan: AdmittedTranscribePlan<P>,
    /// Who separates speakers and how many to expect. Replaces a `diarize`
    /// flag beside an `Option` backend that were equal by construction.
    pub(crate) speakers: SpeakerPlan,

    /// Whether to run morphosyntax after CHAT assembly.
    pub with_morphosyntax: bool,
    /// Independent inference/replay policy for Rev and speaker evidence.
    pub(crate) cache_policies: TranscribeCachePolicies,
    /// Whether to generate `%wor` tiers in the transcribe output.
    ///
    /// Defaults to `false` (BA2 parity: `--wor` was opt-in for transcribe).
    pub write_wor: bool,
    /// Whether single letters spelling a known abbreviation are merged on
    /// the recognizer's tokens before CHAT is built (`--merge-abbrev`).
    pub merge_abbrev: crate::params::MergeAbbrevPolicy,
    /// Media filename for the @Media header.
    pub media_name: Option<talkbank_model::model::MediaFilename>,
    /// Per-engine configuration extras drawn from
    /// `CommonOptions.engine_overrides.extras` (e.g. `qwen_model`,
    /// `qwen_device`, `funaudio_model`). Plumbed through the V2 dispatch
    /// boundary so they reach the worker spawn argv, the `backend` enum
    /// only carries WHICH engine, not its configuration.
    pub engine_extras: std::collections::BTreeMap<String, String>,
}

impl<P: TranscribePlan> TranscribeOptions<P> {
    /// Provider/diarizer count hint; automatic mode never uses a numeric sentinel.
    pub(crate) fn expected_speakers(&self) -> Option<crate::api::NumSpeakers> {
        self.speakers.count().known()
    }

    /// The language recognition is asked for, admitted with the engine in
    /// [`TranscribeAsrPlan`], the only place it is held.
    pub(crate) fn language(&self) -> AsrLanguageRequest {
        self.plan.asr().language()
    }

    /// The GPU-profile engines this job's requests name: its admitted ASR
    /// backend, when a worker hosts it, with the job's engine extras, and its
    /// dedicated diarization engine, when one runs after recognition.
    ///
    /// Derived from the fields that already hold the choices, never stored
    /// beside them, so the capability probe and every request site read the
    /// same plan.
    pub(crate) fn gpu_models(&self) -> crate::worker::pool::GpuModelPlan {
        crate::worker::pool::GpuModelPlan::transcription(
            self.plan.asr().recognition_backend(),
            &self.engine_extras,
            self.speakers.dedicated(),
        )
    }
}

#[cfg(test)]
mod transcribe_segmentation_tests {
    use super::*;
    use crate::api::TranscriptLanguage;
    use crate::params::UtsegFallbackPolicy::{AllowStanza, Refuse};

    fn asr(language: LanguageSpec) -> TranscribeAsrPlan {
        TranscribeAsrPlan::from_request(AsrBackend::RustRevAi, &Default::default(), &language)
            .expect("ASR admission")
    }

    #[test]
    fn transcribe_segmentation_requires_explicit_language_admission() {
        assert!(
            AdmittedTranscribePlan::admit(asr(LanguageCode3::spa().into()), true, Refuse).is_err()
        );
        let plan =
            AdmittedTranscribePlan::admit(asr(LanguageCode3::spa().into()), true, AllowStanza)
                .expect("authorized fallback");
        let route = plan
            .resolve_segmentation(&TranscriptLanguage::One(LanguageCode3::spa()))
            .unwrap()
            .unwrap();
        assert_eq!(route.language(), &LanguageCode3::spa());
        assert_eq!(route.model().unwrap().boundary_model(), None);
        assert!(matches!(
            plan.resolve_segmentation(&TranscriptLanguage::One(LanguageCode3::eng())),
            Err(SegmentationResolutionError::ChangedLanguage { .. })
        ));
    }

    #[test]
    fn transcribe_segmentation_distinguishes_disabled_and_detected_language() {
        let disabled =
            AdmittedTranscribePlan::admit(asr(LanguageCode3::spa().into()), false, Refuse).unwrap();
        assert!(!disabled.with_utseg());
        assert!(
            disabled
                .resolve_segmentation(&TranscriptLanguage::One(LanguageCode3::spa()))
                .unwrap()
                .is_none()
        );
        let deferred =
            AdmittedTranscribePlan::admit(asr(LanguageSpec::Auto), true, Refuse).unwrap();
        assert!(deferred.with_utseg());
        assert!(
            deferred
                .resolve_segmentation(&TranscriptLanguage::One(LanguageCode3::eng()))
                .unwrap()
                .unwrap()
                .model()
                .unwrap()
                .boundary_model()
                .is_some()
        );
        assert!(matches!(
            deferred.resolve_segmentation(&TranscriptLanguage::One(LanguageCode3::spa())),
            Err(SegmentationResolutionError::Unavailable(_))
        ));
        let allowed =
            AdmittedTranscribePlan::admit(asr(LanguageSpec::Auto), true, AllowStanza).unwrap();
        assert_eq!(
            allowed
                .resolve_segmentation(&TranscriptLanguage::One(LanguageCode3::spa()))
                .unwrap()
                .unwrap()
                .model()
                .unwrap()
                .boundary_model(),
            None,
            "an authorized Spanish route runs Stanza"
        );
    }
}

#[cfg(test)]
mod speaker_plan_tests {
    use super::*;

    fn english() -> LanguageSpec {
        LanguageSpec::Resolved(LanguageCode3::eng())
    }

    fn english_spanish() -> LanguageSpec {
        LanguageSpec::Pair(
            LanguagePair::new(LanguageCode3::eng(), LanguageCode3::spa()).expect("two languages"),
        )
    }

    /// Each engine's plan holds only the language requests that engine can
    /// take: one language or detection. A pair is withheld from every engine
    /// at plan time too, which is the route a job saved under an earlier build
    /// takes when it is restarted without resubmission. Transcription is never
    /// per-file.
    #[test]
    fn the_language_is_admitted_with_the_engine() {
        let none = std::collections::BTreeMap::new();
        let whisper = AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2);

        let english_french = LanguageSpec::Pair(
            LanguagePair::new(LanguageCode3::eng(), LanguageCode3::fra()).expect("two languages"),
        );
        for (backend, pair) in [
            (AsrBackend::RustRevAi, english_spanish()),
            (whisper, english_spanish()),
            (AsrBackend::RustRevAi, english_french),
        ] {
            let refusal = TranscribeAsrPlan::from_request(backend, &none, &pair)
                .expect_err("a pair is withheld at plan time on every engine");
            assert!(
                matches!(refusal, TranscribeAsrPlanError::PairWithheld(_)),
                "{refusal}"
            );
            assert!(refusal.to_string().contains("withheld"), "{refusal}");
        }

        for backend in [AsrBackend::RustRevAi, whisper] {
            assert!(matches!(
                TranscribeAsrPlan::from_request(backend, &none, &LanguageSpec::PerFile),
                Err(TranscribeAsrPlanError::NoJobLanguage(_))
            ));
            let detect = TranscribeAsrPlan::from_request(backend, &none, &LanguageSpec::Auto)
                .expect("every engine takes detection");
            assert_eq!(detect.language(), AsrLanguageRequest::Detect);
        }
    }

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

    /// Replay calls no provider, so its language is not checked against any
    /// provider's table: a language Rev.AI does not transcribe, and a pair,
    /// both replay. Only per-file, which no transcript has, is refused.
    #[test]
    fn a_legacy_replay_plan_takes_any_transcript_language() {
        for lang in ["yue", "eng,spa", "auto"] {
            let spec = LanguageSpec::try_from(lang).expect("a spec");
            let plan = ReplayAsrPlan::for_legacy_replay(&spec).expect("replay takes the language");
            assert_eq!(plan.language().to_string(), lang);
        }
        assert!(matches!(
            ReplayAsrPlan::for_legacy_replay(&LanguageSpec::PerFile),
            Err(TranscribeAsrPlanError::NoJobLanguage(_))
        ));
    }

    /// RED FIRST: `--auto-speakers` was refused for every non-Rev recognizer,
    /// because the count was the ASR plan's. It is the speaker plan's: a
    /// dedicated diarizer takes `Auto` whatever the recognizer (Tencent
    /// included, which is then asked for no separation of its own), Rev.AI's
    /// integrated diarization takes either, a recognizer that labels no
    /// speakers has no count to refuse, and only an engine that must be given
    /// the count AND is the run's one speaker engine (Tencent's integrated
    /// diarization, on the worker wire, with no dedicated diarizer) refuses
    /// `Auto`, by name.
    #[test]
    fn the_speaker_plan_admits_auto_wherever_the_speaker_engine_takes_it() {
        let auto = SpeakerCount::admit(true, crate::api::NumSpeakers(2)).expect("auto");
        assert_eq!(auto, SpeakerCount::Auto);
        let whisper = AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2);
        let dedicated = SpeakerSeparation::Dedicated(SpeakerBackendV2::Pyannote);

        let plan = SpeakerPlan::admit(auto, dedicated, Some(Recognizer::submitted(whisper)))
            .expect("the dedicated diarizer infers the count");
        assert_eq!(plan.count().known(), None);
        assert_eq!(plan.dedicated(), Some(SpeakerBackendV2::Pyannote));
        assert_eq!(
            plan.recognizer_diarization(),
            ProviderDiarizationV2::NotRequested
        );
        for separation in [SpeakerSeparation::RecognizerOnly, dedicated] {
            SpeakerPlan::admit(
                auto,
                separation,
                Some(Recognizer::submitted(AsrBackend::RustRevAi)),
            )
            .expect("Rev.AI infers the count");
        }
        SpeakerPlan::admit(
            auto,
            SpeakerSeparation::RecognizerOnly,
            Some(Recognizer::submitted(whisper)),
        )
        .expect("no engine in this run needs a count");
        SpeakerPlan::admit(auto, dedicated, None).expect("a replay runs no recognizer");

        let tencent = AsrBackend::Worker(AsrWorkerMode::HkTencentV2);
        // The three Tencent cases. Alone, Tencent's integrated diarization
        // must be given the count: refused by name, with the dedicated
        // diarizer named as the remedy.
        match SpeakerPlan::admit(
            auto,
            SpeakerSeparation::RecognizerOnly,
            Some(Recognizer::submitted(tencent)),
        ) {
            Err(SpeakerPlanError::AutoCountUnsupported { engine }) => {
                assert_eq!(engine.as_str(), "tencent");
                let message = SpeakerPlanError::AutoCountUnsupported { engine }.to_string();
                assert!(message.contains("--diarization enabled"), "{message}");
            }
            other => panic!("Tencent alone cannot infer the count: {other:?}"),
        }
        // With a dedicated diarizer, the diarizer infers the count and
        // Tencent is asked for no separation of its own.
        let inferred = SpeakerPlan::admit(auto, dedicated, Some(Recognizer::submitted(tencent)))
            .expect("the dedicated diarizer infers the count whatever the recognizer");
        assert_eq!(inferred.count(), SpeakerCount::Auto);
        assert_eq!(inferred.dedicated(), Some(SpeakerBackendV2::Pyannote));
        assert_eq!(
            inferred.recognizer_diarization(),
            ProviderDiarizationV2::NotRequested
        );
        // And with a known count too: separation the diarizer will redo is
        // not asked of the recognizer.
        let two = SpeakerCount::admit(false, crate::api::NumSpeakers(2)).expect("two");
        assert_eq!(
            SpeakerPlan::admit(two, dedicated, Some(Recognizer::submitted(tencent)))
                .expect("a known count with a dedicated diarizer")
                .recognizer_diarization(),
            ProviderDiarizationV2::NotRequested
        );
    }

    /// Whether a Tencent recognizer separates speakers is the account's
    /// engine's property, decided when the server plans the job: on
    /// `16k_yue`, which labels no speakers, a recognizer-only run asks for no
    /// separation and needs no count, as a Whisper run does; on Large Model
    /// 2.0 it separates into the count it is given. Submission, which cannot
    /// see the account, counts Tencent as separating.
    #[test]
    fn a_tencent_recognizer_separates_speakers_only_on_an_engine_that_can() {
        use crate::api::NumSpeakers;
        use crate::model_manifest::{TencentEngine, TencentEngineSetting};
        let tencent = AsrBackend::Worker(AsrWorkerMode::HkTencentV2);
        let two = SpeakerCount::admit(false, NumSpeakers(2)).expect("two");
        let auto = SpeakerCount::Auto;
        let general =
            Recognizer::planned(tencent, &TencentEngineSetting::Chosen(TencentEngine::Yue));
        assert_eq!(
            SpeakerPlan::admit(two, SpeakerSeparation::RecognizerOnly, Some(general))
                .expect("a known count")
                .recognizer_diarization(),
            ProviderDiarizationV2::NotRequested
        );
        SpeakerPlan::admit(auto, SpeakerSeparation::RecognizerOnly, Some(general))
            .expect("an engine that labels no speakers needs no count");
        let large = Recognizer::planned(tencent, &TencentEngineSetting::LARGE_MODEL_2);
        assert!(
            SpeakerPlan::admit(two, SpeakerSeparation::RecognizerOnly, Some(large))
                .expect("a known count")
                .recognizer_diarization()
                .is_requested()
        );
        assert_eq!(
            Recognizer::submitted(tencent),
            Recognizer::planned(tencent, &TencentEngineSetting::LARGE_MODEL_2)
        );
    }

    /// A known count reaches each engine that takes one as its own parameter:
    /// Rev.AI reads it through `known`, Tencent is told to separate into it
    /// (two or more; one is a single track), and a dedicated diarizer with a
    /// count of one is the contradiction it always was. The count itself is
    /// admitted once, in `1..=255`.
    #[test]
    fn a_known_count_reaches_each_engine_as_its_own_parameter() {
        use crate::api::NumSpeakers;
        let three = SpeakerCount::admit(false, NumSpeakers(3)).expect("three");
        let rev = SpeakerPlan::admit(
            three,
            SpeakerSeparation::RecognizerOnly,
            Some(Recognizer::submitted(AsrBackend::RustRevAi)),
        )
        .expect("Rev.AI takes a count");
        assert_eq!(rev.count().known(), Some(NumSpeakers(3)));
        assert_eq!(
            rev.recognizer_diarization(),
            ProviderDiarizationV2::NotRequested
        );

        let tencent = AsrBackend::Worker(AsrWorkerMode::HkTencentV2);
        let separated = SpeakerPlan::admit(
            three,
            SpeakerSeparation::RecognizerOnly,
            Some(Recognizer::submitted(tencent)),
        )
        .expect("Tencent takes a count");
        assert_eq!(
            separated.recognizer_diarization(),
            ProviderDiarizationV2::for_backend(AsrBackendV2::HkTencent, NumSpeakers(3))
        );
        assert!(separated.recognizer_diarization().is_requested());
        let one = SpeakerCount::admit(false, NumSpeakers(1)).expect("one");
        assert_eq!(
            SpeakerPlan::admit(
                one,
                SpeakerSeparation::RecognizerOnly,
                Some(Recognizer::submitted(tencent))
            )
            .expect("one speaker is one track")
            .recognizer_diarization(),
            ProviderDiarizationV2::NotRequested
        );
        assert_eq!(
            SpeakerPlan::admit(
                one,
                SpeakerSeparation::Dedicated(SpeakerBackendV2::PyannoteAi),
                Some(Recognizer::submitted(AsrBackend::RustRevAi))
            ),
            Err(SpeakerPlanError::SeparatingOneSpeaker)
        );
        for count in [0, 256, u32::MAX] {
            assert_eq!(
                SpeakerCount::admit(false, NumSpeakers(count)),
                Err(SpeakerPlanError::InvalidCount(count))
            );
        }
        assert_eq!(
            SpeakerSeparation::from_request(true, None),
            SpeakerSeparation::Dedicated(SpeakerBackendV2::PyannoteAi)
        );
        assert_eq!(
            SpeakerSeparation::from_request(
                true,
                Some(crate::types::engines::SpeakerEngineName::Pyannote)
            ),
            SpeakerSeparation::Dedicated(SpeakerBackendV2::Pyannote)
        );
        assert_eq!(
            SpeakerSeparation::from_request(
                false,
                Some(crate::types::engines::SpeakerEngineName::Pyannote)
            ),
            SpeakerSeparation::RecognizerOnly
        );
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
        TranscribeAsrPlan::from_request(funaudio, &paraformer, &english()).unwrap();
        TranscribeAsrPlan::from_request(whisper, &paraformer, &english()).unwrap();

        let padded = std::collections::BTreeMap::from([(
            crate::types::engines::FUNAUDIO_MODEL_OVERRIDE_KEY.to_owned(),
            "paraformer-zh ".to_owned(),
        )]);
        assert!(matches!(
            TranscribeAsrPlan::from_request(funaudio, &padded, &english()),
            Err(TranscribeAsrPlanError::InvalidCheckpoint { .. })
        ));
    }

    #[test]
    fn replay_policy_refusal_has_a_typed_usage_exit() {
        let error = crate::cli::error::CliError::from(SpeakerPlanError::InvalidCount(0));
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
