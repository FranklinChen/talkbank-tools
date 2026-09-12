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
    #[serde(default = "default_lang")]
    pub lang: LanguageCode3,
    /// Optional provider-shaped monologues preserved from the ASR boundary.
    ///
    /// BA2 parity depends on not discarding provider punctuation elements and
    /// same-speaker monologue breaks before Rust post-processing runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_monologues: Option<Vec<AsrMonologue>>,
}

fn default_lang() -> LanguageCode3 {
    LanguageCode3::eng()
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
    pub(super) fn as_v2_backend(self) -> AsrBackendV2 {
        match self {
            Self::LocalWhisperV2 => AsrBackendV2::LocalWhisper,
            Self::WhisperHubV2 => AsrBackendV2::WhisperHub,
            Self::HkTencentV2 => AsrBackendV2::HkTencent,
            Self::HkAliyunV2 => AsrBackendV2::HkAliyun,
            Self::HkFunaudioV2 => AsrBackendV2::HkFunaudio,
            Self::HkQwenV2 => AsrBackendV2::HkQwen,
        }
    }

    /// Stable engine name written into transcript provenance.
    fn provenance_name(self) -> &'static str {
        match self {
            Self::LocalWhisperV2 => "whisper",
            Self::WhisperHubV2 => "whisper_hub",
            Self::HkTencentV2 => "tencent",
            Self::HkAliyunV2 => "aliyun",
            Self::HkFunaudioV2 => "funaudio",
            Self::HkQwenV2 => "qwen",
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
    pub(crate) fn provenance_name(self) -> &'static str {
        match self {
            Self::RustRevAi => "rev",
            Self::RustWhisperRs => "whisper_rs",
            Self::Worker(mode) => mode.provenance_name(),
        }
    }
}

/// Options controlling the transcribe pipeline.
#[derive(Clone)]
pub struct TranscribeOptions {
    /// Infer speaker count rather than imposing the job's numeric default.
    pub auto_speakers: bool,
    /// Which runtime boundary owns raw ASR inference.
    pub(crate) backend: AsrBackend,
    /// Whether the command requested diarized speaker attribution.
    pub diarize: bool,
    /// Concrete speaker backend selected by Rust when dedicated diarization is needed.
    pub speaker_backend: Option<SpeakerBackendV2>,
    /// Language specification: `Auto` for ASR auto-detect, or a resolved code.
    ///
    /// The type system enforces that post-ASR stages (utseg, morphotag) must
    /// resolve `Auto` to a concrete language before calling NLP workers.
    pub lang: LanguageSpec,
    /// Expected number of speakers for diarization.
    pub num_speakers: usize,
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
        (!self.auto_speakers).then_some(crate::api::NumSpeakers(self.num_speakers as u32))
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
