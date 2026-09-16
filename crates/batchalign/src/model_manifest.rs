//! The pinned model manifest: which models BA3 loads, at which revision.
//!
//! # Why a manifest
//!
//! Most local models loaded by NAME, so a hub repository that moved changed
//! what a run produced with nothing to say so, and no identity was knowable
//! until a model had loaded. This module is the ONE place that says which
//! models a stage loads and which revision of each, so the identity is known
//! before dispatch: it is what the request carries, what the worker's report is
//! checked against, and what a cache key is built from.
//!
//! # What it covers
//!
//! The ASR engines, and the utterance-boundary models. The two are here
//! together because they have one problem and one answer, not because they are
//! one subsystem: each names a model the control plane must be able to identify
//! BEFORE a worker loads it. A second manifest beside this one would recreate
//! the mirrored-table defect this file exists to remove, so a new pinned model
//! belongs in a table here rather than in a module of its own.
//!
//! # How the pins were resolved
//!
//! Hugging Face commits are the `sha` of `https://huggingface.co/api/models/<id>`,
//! read on 2026-09-15, and 2026-09-16 for the boundary models. The native
//! Whisper weights are pinned by the `lfs.oid` of their blob, which IS its
//! SHA-256, so no model cache has to be inspected to know what the file should
//! be. ModelScope models are pinned by TAG: its public API exposes no commit
//! behind a tag (checked the same day: the tag and branch endpoints do not
//! exist and the revisions endpoint returns names and timestamps only), and
//! [`RequestedRevisionV2::Tag`] records that honestly rather than inventing a
//! commit. Every pin here came from a public API; none was read out of a local
//! model cache, which would only say what this machine happens to hold.
//!
//! # What is deliberately not here
//!
//! A drift checker against the upstreams needs the network and belongs beside
//! the other drift checks in `scripts/`, not in the server binary. It lives at
//! `scripts/check_model_pin_drift.py`, which reads the pins from THIS file
//! rather than keeping a second copy of them, and reports for each one whether
//! upstream still matches, has moved, or could not be checked.

use batchalign_transform::translate::WritingSystem;

use batchalign_types::iso639_part1::Iso639Part1Lookup;

use crate::api::LanguageCode3;
use crate::types::engines::{AsrEngineName, FUNAUDIO_MODEL_OVERRIDE_KEY, PARAFORMER_CHECKPOINT};
use crate::types::worker_v2::{
    AsrBackendV2, AsrModelIdentityV2, AsrRequestedModelsV2, HubCommitV2, LoadedModelV2, ModelIdV2,
    ObservedRevisionV2, ProviderParameterV2, RequestedModelV2, RequestedRevisionV2, RevisionTagV2,
};

// ---------------------------------------------------------------------------
// The manifest entries
// ---------------------------------------------------------------------------

/// One model BA3 loads by default, with the revision it is pinned to.
///
/// A plain pair rather than a [`RequestedModelV2`] because the revision text is
/// checked once, here, when the entry is turned into a request model: the
/// constructors on the revision newtypes are fallible, and a manifest entry
/// that cannot be admitted is a defect in THIS file that must fail loudly
/// rather than silently degrade to an unpinned load.
struct ManifestEntry {
    id: &'static str,
    revision: ManifestRevision,
}

/// How one manifest entry names its revision.
enum ManifestRevision {
    /// An exact hub commit, 40 lowercase hexadecimal characters.
    Commit(&'static str),
    /// A published tag, for a hub that exposes no commit behind it.
    Tag(&'static str),
    /// A provider's own selection parameter.
    ProviderParameter(&'static str),
}

/// Stock OpenAI Whisper, which `load_whisper_asr` loads for every language it
/// is not given a fine-tune for.
const WHISPER_LARGE_V3: ManifestEntry = ManifestEntry {
    id: "openai/whisper-large-v3",
    revision: ManifestRevision::Commit("06f233fe06e710322aca913c1bc4249a0d71fce1"),
};

/// The per-language `whisper_hub` fine-tunes BA3 seeds by default.
///
/// The owner of this table, moved here from the Python resolver: Rust must
/// know the model id to pin it, and a second table on the worker side could
/// only disagree. Absent languages are not an error here; they become a
/// floating identity or a refusal at the resolver.
const WHISPER_HUB_DEFAULTS: &[(&str, ManifestEntry)] = &[(
    "mal",
    ManifestEntry {
        id: "thennal/whisper-medium-ml",
        revision: ManifestRevision::Commit("139797930d5942280c03cdeb9a780540a4f4ab0f"),
    },
)];

/// FunASR SenseVoice, the default FunAudio checkpoint.
const SENSEVOICE: ManifestEntry = ManifestEntry {
    id: "FunAudioLLM/SenseVoiceSmall",
    revision: ManifestRevision::Commit("3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"),
};

/// The voice-activity model SenseVoice loads. Resolved through FunASR's
/// Hugging Face name map (`fsmn-vad` there means `funasr/fsmn-vad`), because
/// the SenseVoice branch passes `hub="hf"`.
const SENSEVOICE_VAD: ManifestEntry = ManifestEntry {
    id: "funasr/fsmn-vad",
    revision: ManifestRevision::Commit("df20e6b30c653645fa4ff125cacfcabd1020a669"),
};

/// FunASR Paraformer. The Paraformer branch does NOT pass `hub`, so FunASR's
/// default ModelScope map applies and `paraformer-zh` resolves to the SeaCo
/// Paraformer repository, not the similarly named `speech_paraformer-large-vad-punc`.
const PARAFORMER: ManifestEntry = ManifestEntry {
    id: "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    revision: ManifestRevision::Tag("v2.0.4"),
};

/// Paraformer's voice-activity model, on ModelScope (a DIFFERENT repository
/// from the Hugging Face `funasr/fsmn-vad` SenseVoice loads).
const PARAFORMER_VAD: ManifestEntry = ManifestEntry {
    id: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    revision: ManifestRevision::Tag("v2.0.4"),
};

/// Paraformer's punctuation model.
const PARAFORMER_PUNC: ManifestEntry = ManifestEntry {
    id: "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    revision: ManifestRevision::Tag("v2.0.4"),
};

/// The Qwen3-ASR checkpoints BA3 knows, largest first. An id outside this
/// table is a user's own choice and becomes a floating identity.
const QWEN_CHECKPOINTS: &[(&str, ManifestEntry)] = &[
    (
        "Qwen/Qwen3-ASR-1.7B-hf",
        ManifestEntry {
            id: "Qwen/Qwen3-ASR-1.7B-hf",
            revision: ManifestRevision::Commit("bcd2b5b7f32b480ab5790554cfa8347f246a14f3"),
        },
    ),
    (
        "Qwen/Qwen3-ASR-0.6B-hf",
        ManifestEntry {
            id: "Qwen/Qwen3-ASR-0.6B-hf",
            revision: ManifestRevision::Commit("7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"),
        },
    ),
];

/// The default Qwen3-ASR checkpoint, matching the Python loader's default.
const QWEN_DEFAULT_ID: &str = "Qwen/Qwen3-ASR-1.7B-hf";

/// The forced aligner Qwen3-ASR loads for word timings. Not optional: the
/// engine cannot produce `%wor` without it, which is why the composition makes
/// it a required field.
const QWEN_ALIGNER: ManifestEntry = ManifestEntry {
    id: "Qwen/Qwen3-ForcedAligner-0.6B-hf",
    revision: ManifestRevision::Commit("c07281df297b9905d24a508279258cccf987a064"),
};

/// The engine-override key naming a ggml weights file for native Whisper.
///
/// Only ever reported, never resolved here: the file is chosen by the host, so
/// the manifest pins nothing for it.
const WHISPER_RS_MODEL_KEY: &str = "whisper_rs_model";

/// The default ggml weights whisper.cpp loads in process, named as repository
/// plus file.
///
/// Both halves, because a repository commit alone would not say which of that
/// repository's many weight files ran, and a file name alone would not say
/// which revision of it.
const NATIVE_WHISPER_ID: &str = "ggerganov/whisper.cpp/ggml-large-v3.bin";

/// The repository commit this build fetches those weights at.
///
/// Read by `whisper_native::config`, which owns the owner/name/file coordinates
/// its hf-hub call needs and passes this to `.revision(...)`. The coordinates
/// live with the fetch and the revision lives here, so every pinned revision in
/// this build has exactly one home.
pub(crate) const NATIVE_WHISPER_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

/// Aliyun's identity: the service, never the appkey.
///
/// An appkey identifies an ACCOUNT, not a model, so recording it would leak a
/// credential into every transcript while saying nothing about what ran.
const ALIYUN_SERVICE: ManifestEntry = ManifestEntry {
    id: "aliyun-nls",
    revision: ManifestRevision::ProviderParameter("speech-transcriber-v1"),
};

/// Rev.AI's identity: the constant its evidence cache is already namespaced by,
/// so provenance and the cache agree on what produced a transcript.
/// Rev.AI's model id, named once so the manifest entry and the loaded identity
/// the Rust-owned path reports cannot drift into two different spellings.
const REV_PROVIDER_ID: &str = "revai";

/// The Rev.AI request shape this build uses, likewise named once.
const REV_PROVIDER_PARAMETER: &str = "asynchronous-transcript-v1";

const REV_PROVIDER: ManifestEntry = ManifestEntry {
    id: REV_PROVIDER_ID,
    revision: ManifestRevision::ProviderParameter(REV_PROVIDER_PARAMETER),
};

/// Tencent's engine-model selector for every Chinese variety.
const TENCENT_CHINESE_MODEL_TYPE: &str = "16k_zh_large";

/// The engine-override key the pinned models travel to the worker under.
///
/// One key carrying one serialized composition, rather than a key per model
/// and per role: the worker parses the same typed shape Rust sent, so a new
/// auxiliary model needs no new key and no second parser.
pub(crate) const PINNED_ASR_MODELS_KEY: &str = "asr_pinned_models";

/// The worker-hosted ASR backend one engine name loads, or `None` when the
/// engine has no Python worker at all.
///
/// Exhaustive, with no catch-all, so a new ASR engine must say whether a
/// worker loads it before it can be pinned. The Rust-owned engines (Rev.AI and
/// native Whisper) and the two accepted-but-unimplemented names have no worker
/// backend: `EngineSelection` already drops them before a spawn, and their
/// identities are built by their own code paths.
pub(crate) fn worker_backend_for_engine(engine: &AsrEngineName) -> Option<AsrBackendV2> {
    match engine {
        AsrEngineName::Whisper => Some(AsrBackendV2::LocalWhisper),
        AsrEngineName::WhisperHub => Some(AsrBackendV2::WhisperHub),
        AsrEngineName::HkTencent => Some(AsrBackendV2::HkTencent),
        AsrEngineName::HkAliyun => Some(AsrBackendV2::HkAliyun),
        AsrEngineName::HkFunaudio => Some(AsrBackendV2::HkFunaudio),
        AsrEngineName::HkQwen => Some(AsrBackendV2::HkQwen),
        AsrEngineName::RevAi
        | AsrEngineName::WhisperRs
        | AsrEngineName::WhisperX
        | AsrEngineName::WhisperOai => None,
    }
}

// ---------------------------------------------------------------------------
// The utterance-boundary models
// ---------------------------------------------------------------------------

/// The TalkBank utterance-boundary model each language loads.
///
/// The owner of this table, moved here from `_RESOLVER["utterance"]` in
/// `batchalign/models/resolve.py` for the reason [`WHISPER_HUB_DEFAULTS`]
/// moved: Rust must know a model's id in order to pin it, and a second table on
/// the worker side could only disagree with this one.
///
/// It is now also the single statement of WHICH languages have a boundary
/// model. `utseg_route` reads availability from it rather than keeping a
/// parallel list, so a language this build claims to segment and a language it
/// can name a model for are the same set by construction.
///
/// `cmn` and `zho` share one model deliberately, and each names it separately:
/// they are two ISO codes for one language, and the alternative (a code that
/// redirects to another code) would put a second kind of entry in the table.
const UTSEG_BOUNDARY_MODELS: &[(&str, ManifestEntry)] = &[
    (
        "eng",
        ManifestEntry {
            id: "talkbank/CHATUtterance-en",
            revision: ManifestRevision::Commit("764ec3f762c2e24df2def8df98b5fe34940085c6"),
        },
    ),
    (
        "cmn",
        ManifestEntry {
            id: "talkbank/CHATUtterance-zh_CN",
            revision: ManifestRevision::Commit("d52d3578344d570d652e644e5da3869f25a073e4"),
        },
    ),
    (
        "zho",
        ManifestEntry {
            id: "talkbank/CHATUtterance-zh_CN",
            revision: ManifestRevision::Commit("d52d3578344d570d652e644e5da3869f25a073e4"),
        },
    ),
    (
        "yue",
        ManifestEntry {
            id: "PolyU-AngelChanLab/Cantonese-Utterance-Segmentation",
            revision: ManifestRevision::Commit("9784aeb9e11c674f55a5e70468094736f8285bb6"),
        },
    ),
];

/// The engine-override key the pinned boundary model travels to the worker
/// under, mirroring [`PINNED_ASR_MODELS_KEY`].
///
/// One model rather than a composition: the boundary model loads alone, so
/// there is no role to name and nothing for a composition enum to close over.
pub(crate) const PINNED_UTSEG_MODEL_KEY: &str = "utseg_pinned_model";

/// The boundary model this build pins for one language.
///
/// `None` means this build seeds no boundary model for that language, which is
/// a fact about the language rather than a failure: `utseg_route` turns it into
/// either the Stanza fallback or a typed refusal. `Some(Err(..))` is a defect in
/// THIS file, surfaced the same way every other malformed manifest entry is.
pub(crate) fn utseg_boundary_model(
    lang: &LanguageCode3,
) -> Option<Result<RequestedModelV2, ModelPlanError>> {
    UTSEG_BOUNDARY_MODELS
        .iter()
        .find(|(code, _)| *code == lang.as_ref())
        .map(|(_, entry)| entry.admit())
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why an ASR model plan could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ModelPlanError {
    /// A manifest entry's own text is not admissible. A defect in this file.
    #[error("ASR manifest entry for {id} is malformed: {detail}")]
    MalformedManifestEntry {
        /// The entry's model id.
        id: &'static str,
        /// What the revision constructor refused.
        detail: String,
    },
    /// An override names a model whose id cannot be recorded in provenance.
    #[error("ASR model override `{key}` value {value:?} cannot be recorded: {detail}")]
    UnrecordableOverride {
        /// The override key the value came from.
        key: &'static str,
        /// The offending value.
        value: String,
        /// Why it was refused.
        detail: String,
    },
    /// Tencent needs a concrete language to choose an engine model type.
    #[error(
        "the Tencent ASR engine needs a resolved language to choose its model; \
         re-run with an explicit `--lang <iso3>` instead of `--lang auto`"
    )]
    TencentNeedsResolvedLanguage,
    /// `whisper_hub` names a model per language, and this build seeds none for
    /// the requested one.
    ///
    /// A refusal rather than a placeholder id: the engine exists to load a
    /// SPECIFIC community fine-tune, and there is no default that could be
    /// right. Inventing an id here would turn a clear "choose a model" into a
    /// download failure for a repository that does not exist.
    #[error(
        "the whisper_hub ASR engine has no default model for {language}: it loads a \
         community fine-tune chosen per language, and this build seeds none for that one. \
         Pass one explicitly, for example \
         --engine-overrides '{{\"asr\":\"whisper_hub\",\"model_id\":\"<owner>/<model>\"}}'"
    )]
    WhisperHubHasNoDefaultModel {
        /// The language asked for, or a note that none was resolved.
        language: String,
    },
    /// Tencent has no model type for a language with no ISO 639-1 code.
    ///
    /// Previously this produced `16k_<iso3>`, a model type Tencent does not
    /// define, and the failure surfaced as a provider error mid-job.
    #[error(
        "the Tencent ASR engine has no model for language {lang}: its model types are named by \
         ISO 639-1 code and ISO 639-1 assigns {lang} none. Use a Chinese variety, a language \
         with a two-letter code, or another ASR engine."
    )]
    TencentLanguageHasNoModelType {
        /// The language that has no two-letter code.
        lang: LanguageCode3,
    },
    /// The weights on disk came from a different repository commit than the one
    /// this build pins.
    ///
    /// Refused rather than recorded: the run would produce a transcript whose
    /// stamp named a revision that did not make it, which is worse than no
    /// transcript because nothing downstream could detect the substitution.
    #[error(
        "native Whisper resolved weights from commit {observed}, but this build pins \
         {requested}. Clear the Hugging Face cache entry for those weights, or name \
         the file explicitly with BATCHALIGN_WHISPER_RS_MODEL to run it unpinned."
    )]
    NativeWhisperRevisionMismatch {
        /// The commit this build pins.
        requested: String,
        /// The commit the resolved path names.
        observed: String,
    },
    /// Timing recovery asked for a pinned composition for the in-process
    /// whisper.cpp path, which is not a worker backend and so has none.
    ///
    /// Not reachable through `From<&UtrEngine>`, which never selects that path
    /// for recovery. Named rather than folded into a catch-all so that wiring
    /// it into UTR later is a decision taken here, at compile time, instead of
    /// silently resolving to some other engine's models.
    #[error(
        "the in-process whisper.cpp path has no worker ASR backend, so timing recovery \
         cannot resolve a pinned composition for it"
    )]
    NativeWhisperHasNoWorkerBackend,
}

// ---------------------------------------------------------------------------
// Turning manifest entries into request models
// ---------------------------------------------------------------------------

impl ManifestEntry {
    /// Admit this entry as a pinned request model.
    fn admit(&self) -> Result<RequestedModelV2, ModelPlanError> {
        let malformed = |detail: String| ModelPlanError::MalformedManifestEntry {
            id: self.id,
            detail,
        };
        let revision = match &self.revision {
            ManifestRevision::Commit(commit) => RequestedRevisionV2::Commit {
                commit: HubCommitV2::try_from(*commit)
                    .map_err(|error| malformed(error.to_string()))?,
            },
            ManifestRevision::Tag(tag) => RequestedRevisionV2::Tag {
                tag: RevisionTagV2::try_from((*tag).to_owned())
                    .map_err(|error| malformed(error.to_string()))?,
            },
            ManifestRevision::ProviderParameter(parameter) => {
                RequestedRevisionV2::ProviderParameter {
                    parameter: ProviderParameterV2::try_from((*parameter).to_owned())
                        .map_err(|error| malformed(error.to_string()))?,
                }
            }
        };
        Ok(RequestedModelV2 {
            id: ModelIdV2::try_from(self.id).map_err(|error| malformed(error.to_string()))?,
            revision,
        })
    }
}

/// A model the user named that the manifest does not pin.
///
/// Typed as floating rather than refused: full-file transcription has no ASR
/// cache, so nothing users do today depends on pinning an arbitrary checkpoint.
/// The worker still has to report the commit it loaded, and a floating identity
/// can never build a cache key.
fn floating(key: &'static str, id: &str) -> Result<RequestedModelV2, ModelPlanError> {
    Ok(RequestedModelV2 {
        id: ModelIdV2::try_from(id).map_err(|error| ModelPlanError::UnrecordableOverride {
            key,
            value: id.to_owned(),
            detail: error.to_string(),
        })?,
        revision: RequestedRevisionV2::Unpinned,
    })
}

// ---------------------------------------------------------------------------
// Tencent's engine model type
// ---------------------------------------------------------------------------

/// The `EngineModelType` Tencent is asked for.
///
/// Chosen in Rust for EVERY language, through the one ISO 639-3 to 639-1
/// conversion and the one Han-script table. The Python derivation this replaces
/// used pycountry with its own five-code Chinese list that OMITTED Mandarin, so
/// a `cmn` job asked Tencent for `16k_cmn`, and any language without a
/// two-letter code asked for `16k_<iso3>`; neither is a model type Tencent
/// defines.
///
/// [`WritingSystem`] is consulted first because every Chinese variety shares
/// one large model whatever its own code, and it already owns that list.
pub(crate) fn tencent_engine_model_type(
    lang: &LanguageCode3,
) -> Result<ProviderParameterV2, ModelPlanError> {
    if WritingSystem::of_language_code(lang.as_ref()) == WritingSystem::Han {
        return Ok(const { ProviderParameterV2::from_static(TENCENT_CHINESE_MODEL_TYPE) });
    }
    match lang.to_iso_639_1() {
        Iso639Part1Lookup::Paired(code) => {
            // Total: a two-letter code is ASCII letters, so the joined value is
            // always admissible stamp-safe text.
            ProviderParameterV2::try_from(format!("16k_{code}")).map_err(|error| {
                ModelPlanError::MalformedManifestEntry {
                    id: "tencent",
                    detail: error.to_string(),
                }
            })
        }
        Iso639Part1Lookup::NoTwoLetterCode => {
            Err(ModelPlanError::TencentLanguageHasNoModelType { lang: lang.clone() })
        }
    }
}

// ---------------------------------------------------------------------------
// The resolver
// ---------------------------------------------------------------------------

/// Resolve the models one ASR request will load.
///
/// `lang` is `None` for an auto-detect job. `extras` is the user's
/// `--engine-overrides`, from which each engine reads only its own key.
pub(crate) fn resolve_asr_models(
    backend: AsrBackendV2,
    lang: Option<&LanguageCode3>,
    extras: &std::collections::BTreeMap<String, String>,
) -> Result<AsrRequestedModelsV2, ModelPlanError> {
    match backend {
        // Stock Whisper loads one checkpoint for every language.
        AsrBackendV2::LocalWhisper => Ok(AsrRequestedModelsV2::Whisper {
            asr: WHISPER_LARGE_V3.admit()?,
        }),
        AsrBackendV2::WhisperHub => Ok(AsrRequestedModelsV2::Whisper {
            asr: whisper_hub_model(lang, extras)?,
        }),
        AsrBackendV2::HkFunaudio => funaudio_models(extras),
        AsrBackendV2::HkQwen => Ok(AsrRequestedModelsV2::Qwen {
            asr: qwen_model(extras)?,
            aligner: QWEN_ALIGNER.admit()?,
        }),
        AsrBackendV2::HkTencent => {
            let lang = lang.ok_or(ModelPlanError::TencentNeedsResolvedLanguage)?;
            Ok(AsrRequestedModelsV2::Tencent {
                engine_model_type: RequestedModelV2 {
                    id: const { ModelIdV2::from_static("tencent-asr") },
                    revision: RequestedRevisionV2::ProviderParameter {
                        parameter: tencent_engine_model_type(lang)?,
                    },
                },
            })
        }
        AsrBackendV2::HkAliyun => Ok(AsrRequestedModelsV2::Aliyun {
            service: ALIYUN_SERVICE.admit()?,
        }),
        AsrBackendV2::Revai => Ok(AsrRequestedModelsV2::Rev {
            provider: REV_PROVIDER.admit()?,
        }),
    }
}

/// The composition one timing-recovery engine runs, for keying its cache.
///
/// Derived through [`crate::transcribe::AsrBackend`], the one owner of "which
/// runtime runs this engine", rather than a second UTR-specific table that
/// could disagree with the one the request builder uses.
///
/// Resolved with an EMPTY extras map, deliberately and to match reality:
/// `infer_utr_asr_response` infers with an empty map too, because there are no
/// UTR-specific knobs in `EngineOverrides.extras` today. Deriving the namespace
/// from the same plan the run executes is the point; resolving it with extras
/// the run ignores would key rows under a composition that never ran.
pub(crate) fn utr_pinned_models(
    engine: &crate::options::UtrEngine,
    lang: &LanguageCode3,
) -> Result<AsrRequestedModelsV2, ModelPlanError> {
    let backend = match crate::transcribe::AsrBackend::from(engine) {
        crate::transcribe::AsrBackend::RustRevAi => AsrBackendV2::Revai,
        crate::transcribe::AsrBackend::Worker(mode) => mode.as_v2_backend(),
        crate::transcribe::AsrBackend::RustWhisperRs => {
            return Err(ModelPlanError::NativeWhisperHasNoWorkerBackend);
        }
    };
    let extras = std::collections::BTreeMap::new();
    resolve_asr_models(backend, Some(lang), &extras)
}

/// The identity the Rust-owned Rev.AI path reports.
///
/// Infallible on purpose: both halves are compile-time constants checked by
/// `from_static`, so it cannot fail at the places that need it, which return a
/// response rather than a result and would otherwise have to invent a fallback.
///
/// The observation is `NotExposed` because that is the literal truth. A cloud
/// provider runs whatever it runs and reports no revision, so provider-side
/// drift is invisible to us; recording that absence is honest, whereas echoing
/// the request parameter back as though it had been observed would claim a
/// confirmation nobody made.
pub(crate) fn rev_loaded_identity() -> AsrModelIdentityV2 {
    AsrModelIdentityV2::Rev {
        provider: LoadedModelV2 {
            id: const { ModelIdV2::from_static(REV_PROVIDER_ID) },
            requested: RequestedRevisionV2::ProviderParameter {
                parameter: const { ProviderParameterV2::from_static(REV_PROVIDER_PARAMETER) },
            },
            observed: ObservedRevisionV2::NotExposed,
        },
    }
}

/// The models the Rust-owned native Whisper path loads.
///
/// Separate from [`resolve_asr_models`] because whisper.cpp is not a worker
/// backend: it has no [`AsrBackendV2`] variant, and its weights are a file
/// rather than a repository.
pub(crate) fn native_whisper_identity(
    model_path: &std::path::Path,
    source: crate::whisper_native::WhisperModelSource,
) -> Result<AsrModelIdentityV2, ModelPlanError> {
    let ggml = match source {
        // Fetched at a pinned repository revision, so it is named exactly as
        // every other Hugging Face model in this manifest is.
        crate::whisper_native::WhisperModelSource::PinnedDefault => {
            let malformed = |detail: String| ModelPlanError::MalformedManifestEntry {
                id: NATIVE_WHISPER_ID,
                detail,
            };
            let requested = HubCommitV2::try_from(NATIVE_WHISPER_REVISION)
                .map_err(|error| malformed(error.to_string()))?;
            let observed = observed_native_commit(model_path, &requested)?;
            LoadedModelV2 {
                id: ModelIdV2::try_from(NATIVE_WHISPER_ID)
                    .map_err(|error| malformed(error.to_string()))?,
                requested: RequestedRevisionV2::Commit { commit: requested },
                observed: ObservedRevisionV2::Commit { commit: observed },
            }
        }
        // A file the host put there. This build pins nothing for it and can
        // observe nothing about it, and saying so is honest rather than a gap:
        // an invented revision would be worse than an admitted absence.
        crate::whisper_native::WhisperModelSource::HostChosen => {
            // The file's own name, which is the only part of a host path safe
            // to record: a full path is machine-local and would carry a user's
            // directory layout into every transcript this build stamps.
            let name = model_path
                .file_name()
                .map_or_else(
                    || model_path.to_string_lossy(),
                    |name| name.to_string_lossy(),
                )
                .into_owned();
            LoadedModelV2 {
                id: ModelIdV2::try_from(name.as_str()).map_err(|error| {
                    ModelPlanError::UnrecordableOverride {
                        key: WHISPER_RS_MODEL_KEY,
                        value: name.clone(),
                        detail: error.to_string(),
                    }
                })?,
                requested: RequestedRevisionV2::Unpinned,
                observed: ObservedRevisionV2::NotExposed,
            }
        }
    };
    Ok(AsrModelIdentityV2::NativeWhisper { ggml })
}

/// What commit the weights on disk attest to, cross-checked against the pin.
///
/// hf-hub lays its cache out as `.../snapshots/<commit>/<file>`. Where that
/// segment is present it is an INDEPENDENT reading of what actually landed, so
/// it is preferred and a disagreement is refused by name. Where it is absent
/// the observation is the pinned fetch itself: `.revision(sha)` either serves
/// that revision or fails, so a returned file is evidence for that commit and
/// for no other. Neither branch invents a value, and neither echoes a request
/// that was never honoured.
fn observed_native_commit(
    model_path: &std::path::Path,
    requested: &HubCommitV2,
) -> Result<HubCommitV2, ModelPlanError> {
    let seen = model_path
        .components()
        .find_map(|component| HubCommitV2::try_from(component.as_os_str().to_str()?).ok());
    match seen {
        Some(seen) if seen != *requested => Err(ModelPlanError::NativeWhisperRevisionMismatch {
            requested: requested.as_str().to_owned(),
            observed: seen.as_str().to_owned(),
        }),
        Some(seen) => Ok(seen),
        None => Ok(requested.clone()),
    }
}

/// The `whisper_hub` fine-tune for one language.
fn whisper_hub_model(
    lang: Option<&LanguageCode3>,
    extras: &std::collections::BTreeMap<String, String>,
) -> Result<RequestedModelV2, ModelPlanError> {
    const MODEL_ID_KEY: &str = "model_id";
    if let Some(chosen) = extras.get(MODEL_ID_KEY) {
        // An explicit id wins, as it does in the Python resolver. Pinned when
        // the manifest knows it, floating otherwise.
        return match WHISPER_HUB_DEFAULTS
            .iter()
            .find(|(_, entry)| entry.id == chosen.as_str())
        {
            Some((_, entry)) => entry.admit(),
            None => floating(MODEL_ID_KEY, chosen),
        };
    }
    match lang.and_then(|lang| {
        WHISPER_HUB_DEFAULTS
            .iter()
            .find(|(code, _)| *code == lang.as_ref())
    }) {
        Some((_, entry)) => entry.admit(),
        // No seeded default. Refused by name rather than pinned to an
        // invented id: nothing could load, and a placeholder would surface as
        // a missing-repository download error instead of the remedy.
        None => Err(ModelPlanError::WhisperHubHasNoDefaultModel {
            language: lang.map_or_else(
                || "an unresolved language".to_owned(),
                |lang| lang.to_string(),
            ),
        }),
    }
}

/// The FunAudio composition, which depends on the checkpoint selected.
fn funaudio_models(
    extras: &std::collections::BTreeMap<String, String>,
) -> Result<AsrRequestedModelsV2, ModelPlanError> {
    match extras.get(FUNAUDIO_MODEL_OVERRIDE_KEY) {
        // `--asr-engine paraformer` sets this, and so may a user by hand.
        Some(checkpoint) if checkpoint == PARAFORMER_CHECKPOINT => {
            Ok(AsrRequestedModelsV2::Paraformer {
                asr: PARAFORMER.admit()?,
                vad: PARAFORMER_VAD.admit()?,
                punc: PARAFORMER_PUNC.admit()?,
            })
        }
        // Another FunASR checkpoint entirely. Its composition follows the
        // loader's own branch, which keys on whether the name says paraformer.
        Some(checkpoint) if checkpoint.contains("paraformer") => {
            Ok(AsrRequestedModelsV2::Paraformer {
                asr: floating(FUNAUDIO_MODEL_OVERRIDE_KEY, checkpoint)?,
                vad: PARAFORMER_VAD.admit()?,
                punc: PARAFORMER_PUNC.admit()?,
            })
        }
        Some(checkpoint) => Ok(AsrRequestedModelsV2::SenseVoice {
            asr: floating(FUNAUDIO_MODEL_OVERRIDE_KEY, checkpoint)?,
            vad: SENSEVOICE_VAD.admit()?,
        }),
        None => Ok(AsrRequestedModelsV2::SenseVoice {
            asr: SENSEVOICE.admit()?,
            vad: SENSEVOICE_VAD.admit()?,
        }),
    }
}

/// The Qwen3-ASR checkpoint selected.
fn qwen_model(
    extras: &std::collections::BTreeMap<String, String>,
) -> Result<RequestedModelV2, ModelPlanError> {
    const QWEN_MODEL_KEY: &str = "qwen_model";
    let chosen = extras
        .get(QWEN_MODEL_KEY)
        .map_or(QWEN_DEFAULT_ID, String::as_str);
    match QWEN_CHECKPOINTS.iter().find(|(name, _)| *name == chosen) {
        Some((_, entry)) => entry.admit(),
        None => floating(QWEN_MODEL_KEY, chosen),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extras(pairs: &[(&str, &str)]) -> std::collections::BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    fn lang(code: &str) -> LanguageCode3 {
        LanguageCode3::try_new(code).expect("test language code is valid")
    }

    /// Every manifest entry admits. A malformed revision in this file is a
    /// defect that must fail here rather than at dispatch on a live job.
    #[test]
    fn every_manifest_entry_admits() {
        let none = extras(&[]);
        for backend in AsrBackendV2::ALL {
            let resolved = match backend {
                // Tencent needs a language; covered by its own tests.
                AsrBackendV2::HkTencent => resolve_asr_models(backend, Some(&lang("eng")), &none),
                _ => resolve_asr_models(backend, Some(&lang("mal")), &none),
            };
            resolved.unwrap_or_else(|error| panic!("{backend:?} must resolve: {error}"));
        }
        native_whisper_identity(
            std::path::Path::new("/models/ggml-large-v3.bin"),
            crate::whisper_native::WhisperModelSource::HostChosen,
        )
        .expect("a host-chosen ggml file must admit");
        native_whisper_identity(
            std::path::Path::new(
                "/cache/models--ggerganov--whisper.cpp/snapshots/\
                 5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3.bin",
            ),
            crate::whisper_native::WhisperModelSource::PinnedDefault,
        )
        .expect("the pinned default must admit");
    }

    /// Every boundary-model entry admits, and pins an exact commit.
    ///
    /// A malformed revision here would otherwise surface at spawn time on a
    /// live job, which is the failure `every_manifest_entry_admits` exists to
    /// prevent for the ASR side.
    #[test]
    fn every_boundary_model_entry_admits_with_a_pinned_commit() {
        for (code, _) in UTSEG_BOUNDARY_MODELS {
            let model = utseg_boundary_model(&lang(code))
                .unwrap_or_else(|| panic!("{code} is named in the table"))
                .unwrap_or_else(|error| panic!("{code} must admit: {error}"));
            assert!(
                matches!(model.revision, RequestedRevisionV2::Commit { .. }),
                "{code} must pin an exact commit, got {:?}",
                model.revision
            );
        }
    }

    /// A language the table does not name has no boundary model. That absence
    /// is what `utseg_route` turns into the Stanza fallback or a typed
    /// refusal, so it must never be filled in with another language's model.
    #[test]
    fn a_language_with_no_boundary_model_is_absent_rather_than_defaulted() {
        assert!(utseg_boundary_model(&lang("spa")).is_none());
        assert!(utseg_boundary_model(&lang("fra")).is_none());
    }

    /// The two Mandarin codes load ONE model at ONE commit. Drift between them
    /// would silently segment `cmn` and `zho` with different weights.
    #[test]
    fn both_mandarin_codes_pin_the_same_model_and_commit() {
        let cmn = utseg_boundary_model(&lang("cmn"))
            .expect("cmn is named")
            .expect("cmn admits");
        let zho = utseg_boundary_model(&lang("zho"))
            .expect("zho is named")
            .expect("zho admits");
        assert_eq!(cmn, zho);
    }

    /// The Rust-owned Rev.AI path reports an identity its own plan admits, and
    /// it observes NOTHING.
    ///
    /// Both halves matter. A cloud service reports no revision, so the honest
    /// record is an explicit absence; echoing the request parameter back as an
    /// observation would claim a confirmation the provider never gave. And the
    /// pairing has to be one `admit` accepts, because this identity travels on
    /// every Rev transcript.
    #[test]
    fn the_rev_identity_is_admissible_and_observes_nothing() {
        let loaded = rev_loaded_identity();
        let requested = resolve_asr_models(AsrBackendV2::Revai, None, &extras(&[]))
            .expect("Rev resolves a pinned composition");
        loaded
            .admit(&requested, AsrBackendV2::Revai)
            .expect("the Rev identity must admit against its own plan");

        let (_, provider) = loaded.primary();
        assert_eq!(
            provider.observed,
            ObservedRevisionV2::NotExposed,
            "a cloud provider exposes no revision, and the record must say so"
        );
    }

    /// THE FIX. Mandarin asks for the Chinese model, like every other Chinese
    /// variety. It used to ask for `16k_cmn`, which Tencent does not define,
    /// because the Python list of Chinese codes omitted `cmn`.
    #[test]
    fn every_chinese_variety_including_mandarin_asks_for_the_chinese_model() {
        for code in ["cmn", "zho", "yue", "wuu", "nan", "hak"] {
            assert_eq!(
                tencent_engine_model_type(&lang(code))
                    .expect("a Chinese variety has a model type")
                    .as_str(),
                "16k_zh_large",
                "{code} is a Chinese variety"
            );
        }
    }

    /// A language with a two-letter code is unchanged from the Python
    /// derivation, which is what makes this a safe substitution.
    #[test]
    fn languages_with_a_two_letter_code_keep_their_model_type() {
        for (code, expected) in [
            ("eng", "16k_en"),
            ("fra", "16k_fr"),
            ("jpn", "16k_ja"),
            ("spa", "16k_es"),
            ("kor", "16k_ko"),
        ] {
            assert_eq!(
                tencent_engine_model_type(&lang(code))
                    .expect("a two-letter language has a model type")
                    .as_str(),
                expected
            );
        }
    }

    /// A language with no two-letter code is refused at PLAN time, naming the
    /// language. It used to become `16k_<iso3>`, a model type Tencent does not
    /// define, and failed mid-job as a provider error.
    #[test]
    fn a_language_with_no_two_letter_code_is_refused_by_name() {
        let refusal = tencent_engine_model_type(&lang("ceb"))
            .expect_err("Cebuano has no ISO 639-1 code, so Tencent has no model for it");
        assert!(
            matches!(
                refusal,
                ModelPlanError::TencentLanguageHasNoModelType { .. }
            ),
            "{refusal:?}"
        );
        assert!(refusal.to_string().contains("ceb"), "{refusal}");
    }

    /// An auto-detect job cannot choose a Tencent model, and says so instead of
    /// asking for `16k_auto`.
    #[test]
    fn tencent_refuses_an_unresolved_language() {
        assert_eq!(
            resolve_asr_models(AsrBackendV2::HkTencent, None, &extras(&[])),
            Err(ModelPlanError::TencentNeedsResolvedLanguage)
        );
    }

    /// FunAudio's composition follows its checkpoint: the default is
    /// SenseVoice with its VAD, and `paraformer-zh` is Paraformer with the VAD
    /// and punctuation models it additionally loads.
    #[test]
    fn the_funaudio_checkpoint_decides_the_composition() {
        let default = resolve_asr_models(AsrBackendV2::HkFunaudio, None, &extras(&[]))
            .expect("the default resolves");
        assert!(matches!(default, AsrRequestedModelsV2::SenseVoice { .. }));
        assert!(default.is_fully_pinned());

        let paraformer = resolve_asr_models(
            AsrBackendV2::HkFunaudio,
            None,
            &extras(&[(FUNAUDIO_MODEL_OVERRIDE_KEY, PARAFORMER_CHECKPOINT)]),
        )
        .expect("paraformer resolves");
        assert!(matches!(
            paraformer,
            AsrRequestedModelsV2::Paraformer { .. }
        ));
        assert!(paraformer.is_fully_pinned());
    }

    /// The two FunASR VAD models are DIFFERENT repositories: SenseVoice loads
    /// the Hugging Face one, Paraformer the ModelScope one, because only the
    /// SenseVoice branch passes `hub="hf"`. Pinning both to one id would load
    /// the wrong weights for one of them.
    #[test]
    fn the_two_funasr_vad_models_are_not_the_same_repository() {
        let sensevoice = SENSEVOICE_VAD.admit().expect("admits");
        let paraformer = PARAFORMER_VAD.admit().expect("admits");
        assert_ne!(sensevoice.id, paraformer.id);
        assert_eq!(sensevoice.id.as_str(), "funasr/fsmn-vad");
        assert!(paraformer.id.as_str().starts_with("iic/"));
    }

    /// A checkpoint the manifest does not know is floating, not refused: full
    /// file transcription has no ASR cache, so this refuses nothing users do.
    /// Its aligner stays pinned, because the engine always loads that one.
    #[test]
    fn an_unknown_checkpoint_is_floating_and_uncacheable() {
        let resolved = resolve_asr_models(
            AsrBackendV2::HkQwen,
            None,
            &extras(&[("qwen_model", "someone/a-qwen-fine-tune")]),
        )
        .expect("an unknown Qwen checkpoint still resolves");
        assert!(
            !resolved.is_fully_pinned(),
            "an unpinned member makes the whole composition uncacheable"
        );
        let AsrRequestedModelsV2::Qwen { asr, aligner } = &resolved else {
            panic!("expected a Qwen composition");
        };
        assert_eq!(asr.revision, RequestedRevisionV2::Unpinned);
        assert!(matches!(
            aligner.revision,
            RequestedRevisionV2::Commit { .. }
        ));
    }

    /// The known Qwen checkpoints are pinned, including the non-default one.
    #[test]
    fn known_qwen_checkpoints_are_pinned() {
        for id in ["Qwen/Qwen3-ASR-1.7B-hf", "Qwen/Qwen3-ASR-0.6B-hf"] {
            let resolved =
                resolve_asr_models(AsrBackendV2::HkQwen, None, &extras(&[("qwen_model", id)]))
                    .expect("resolves");
            assert!(resolved.is_fully_pinned(), "{id} should be pinned");
        }
    }

    /// `whisper_hub` exists to load a SPECIFIC community fine-tune, so a
    /// language this build seeds none for is refused by name with the remedy.
    ///
    /// It used to resolve to a floating member carrying an invented id, which
    /// would have surfaced as a download failure for a repository that does
    /// not exist rather than as "choose a model".
    #[test]
    fn whisper_hub_without_a_seeded_default_is_refused_with_the_remedy() {
        let refusal =
            resolve_asr_models(AsrBackendV2::WhisperHub, Some(&lang("eng")), &extras(&[]))
                .expect_err("this build seeds no English whisper_hub fine-tune");
        assert!(
            matches!(refusal, ModelPlanError::WhisperHubHasNoDefaultModel { .. }),
            "{refusal:?}"
        );
        let message = refusal.to_string();
        assert!(message.contains("eng"), "{message}");
        assert!(
            message.contains("model_id"),
            "the refusal must name the remedy: {message}"
        );

        // An explicit id is exactly the remedy the message names, and an id
        // this build does not pin is floating rather than refused.
        let chosen = resolve_asr_models(
            AsrBackendV2::WhisperHub,
            Some(&lang("eng")),
            &extras(&[("model_id", "someone/a-whisper-fine-tune")]),
        )
        .expect("an explicit model_id resolves");
        assert!(!chosen.is_fully_pinned());
    }

    /// Aliyun records its service, never the appkey that identifies an account.
    #[test]
    fn aliyun_records_a_service_not_a_credential() {
        let resolved = resolve_asr_models(AsrBackendV2::HkAliyun, None, &extras(&[]))
            .expect("aliyun resolves");
        let AsrRequestedModelsV2::Aliyun { service } = &resolved else {
            panic!("expected an Aliyun composition");
        };
        assert_eq!(service.id.as_str(), "aliyun-nls");
    }

    /// WIRE FORMAT: the exact JSON a worker receives under
    /// [`PINNED_ASR_MODELS_KEY`].
    ///
    /// The Python bootstrap parses this shape to know which revision to load,
    /// and two hand-written parsers in different languages agree only while
    /// something pins the bytes. This is that pin: the tags (`engine`, `kind`),
    /// the role field names and the revision payload names are all contract.
    #[test]
    fn the_pinned_composition_serializes_in_the_shape_python_parses() {
        let sensevoice = resolve_asr_models(AsrBackendV2::HkFunaudio, None, &extras(&[]))
            .expect("the default FunAudio composition resolves");
        let value = serde_json::to_value(&sensevoice).expect("the composition serializes");
        assert_eq!(value["engine"], "sense_voice");
        assert_eq!(value["asr"]["id"], "FunAudioLLM/SenseVoiceSmall");
        assert_eq!(value["asr"]["revision"]["kind"], "commit");
        assert_eq!(
            value["asr"]["revision"]["commit"],
            "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
        );
        assert_eq!(value["vad"]["id"], "funasr/fsmn-vad");

        // A ModelScope composition carries a TAG, and a cloud one a provider
        // parameter, so both non-commit revision kinds are pinned here too.
        let paraformer = resolve_asr_models(
            AsrBackendV2::HkFunaudio,
            None,
            &extras(&[(FUNAUDIO_MODEL_OVERRIDE_KEY, PARAFORMER_CHECKPOINT)]),
        )
        .expect("the Paraformer composition resolves");
        let value = serde_json::to_value(&paraformer).expect("the composition serializes");
        assert_eq!(value["engine"], "paraformer");
        assert_eq!(value["asr"]["revision"]["kind"], "tag");
        assert_eq!(value["asr"]["revision"]["tag"], "v2.0.4");
        assert_eq!(value["punc"]["revision"]["tag"], "v2.0.4");

        let tencent = resolve_asr_models(AsrBackendV2::HkTencent, Some(&lang("yue")), &extras(&[]))
            .expect("Tencent resolves for Cantonese");
        let value = serde_json::to_value(&tencent).expect("the composition serializes");
        assert_eq!(value["engine"], "tencent");
        assert_eq!(
            value["engine_model_type"]["revision"]["kind"],
            "provider_parameter"
        );
        assert_eq!(
            value["engine_model_type"]["revision"]["parameter"],
            "16k_zh_large"
        );

        // An unpinned member is a tagged variant with no payload, not a null.
        let floating = resolve_asr_models(
            AsrBackendV2::HkQwen,
            None,
            &extras(&[("qwen_model", "someone/a-qwen-fine-tune")]),
        )
        .expect("an unknown checkpoint still resolves");
        let value = serde_json::to_value(&floating).expect("the composition serializes");
        assert_eq!(value["asr"]["revision"]["kind"], "unpinned");
    }

    /// Every composition a backend resolves to is one that backend is allowed
    /// to report, so the plan can never fail its own admission.
    #[test]
    fn a_resolved_composition_is_one_its_backend_may_report() {
        let none = extras(&[]);
        for backend in AsrBackendV2::ALL {
            // `whisper_hub` loads a fine-tune chosen per language, so it
            // resolves only for a language this build seeds one for; every
            // other engine loads the same models whatever the language.
            let language = match backend {
                AsrBackendV2::WhisperHub => lang("mal"),
                _ => lang("eng"),
            };
            let resolved = resolve_asr_models(backend, Some(&language), &none)
                .expect("every backend resolves for a language it supports");
            assert!(
                resolved.shape().admits_backend(backend),
                "{backend:?} resolved to a composition it cannot report"
            );
        }
    }
}
