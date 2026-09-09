//! REST API request models, `POST /jobs` submission types.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use serde::{Deserialize, Serialize};

use super::revai_language::{revai_known_broken, try_revai_language_hint};
use crate::options::{AsrEngineName, CommandOptions, FaEngineName, UtrEngine};
use crate::types::engines::SelectableEngine;
use crate::types::params::FaParams;

use super::domain::{DisplayPath, LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand};

// ---------------------------------------------------------------------------
// Request models
// ---------------------------------------------------------------------------

/// A single CHAT file submitted by the client.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FilePayload {
    /// Original filename (e.g. "01DM_18.cha").
    pub filename: DisplayPath,
    /// Full CHAT file text.
    pub content: String,
}

/// `POST /jobs` request body.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobSubmission {
    /// Batchalign command (align, morphotag, etc.).
    pub command: ReleasedCommand,
    /// Language specification: a 3-letter ISO code or `"auto"` for
    /// ASR-driven detection.
    #[serde(default = "default_lang")]
    pub lang: LanguageSpec,
    /// Number of speakers.
    #[serde(default = "default_num_speakers")]
    pub num_speakers: NumSpeakers,
    /// CHAT files to process.
    #[serde(default)]
    pub files: Vec<FilePayload>,
    /// Media filenames for the server to resolve from media_roots (transcribe only).
    #[serde(default)]
    pub media_files: Vec<String>,
    /// Key into server's media_mappings config (e.g. "childes-data").
    #[serde(default)]
    pub media_mapping: batchalign_types::paths::MediaMappingKey,
    /// Subdirectory under the mapped root (e.g. "Eng-NA/MacWhinney/0young-ASR").
    #[serde(default)]
    pub media_subdir: batchalign_types::paths::RepoRelativePath,
    /// Client's input directory path (for dashboard display).
    #[serde(default)]
    pub source_dir: batchalign_types::paths::ClientPath,
    /// Typed command options (engine selections, processing flags, etc.).
    #[cfg_attr(feature = "server", schema(value_type = serde_json::Value))]
    pub options: CommandOptions,

    // Paths mode: local daemon sends filesystem paths instead of content.
    /// When true, server reads/writes files directly via source_paths/output_paths.
    #[serde(default)]
    pub paths_mode: bool,
    /// Absolute paths to read input files from (paths_mode only).
    #[serde(default)]
    pub source_paths: Vec<batchalign_types::paths::ClientPath>,
    /// Absolute paths to write output files to (paths_mode only).
    #[serde(default)]
    pub output_paths: Vec<batchalign_types::paths::ClientPath>,
    /// Human-readable filenames for display (paths_mode only, optional).
    #[serde(default)]
    pub display_names: Vec<String>,

    /// When true, the server collects detailed algorithm traces for
    /// visualization (DP alignment matrices, ASR pipeline stages, FA
    /// timelines, retokenization mappings). Defaults to false, zero
    /// overhead when off.
    #[serde(default)]
    pub debug_traces: bool,

    /// Absolute paths to "before" files for incremental processing
    /// (paths_mode only). When non-empty, the diff engine compares each
    /// before file against its corresponding source_path and only
    /// reprocesses changed utterances.
    ///
    /// Must be the same length as `source_paths` when non-empty.
    #[serde(default)]
    pub before_paths: Vec<batchalign_types::paths::ClientPath>,
}

pub(crate) fn default_lang() -> LanguageSpec {
    LanguageSpec::Resolved(LanguageCode3::eng())
}

pub(crate) fn default_num_speakers() -> NumSpeakers {
    NumSpeakers(1)
}

impl JobSubmission {
    /// Validate submission constraints (paths_mode, command consistency).
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate options command tag matches the command field.
        if self.command != self.options.command_name() {
            return Err(ValidationError(format!(
                "options command tag '{}' does not match submission command '{}'",
                self.options.command_name(),
                self.command
            )));
        }

        self.validate_cache_policy()?;

        // Refuse an engine this build cannot run, before any language or path
        // check: an unimplemented engine is wrong for every language, and a
        // language-shaped message would send the operator hunting the wrong
        // thing.
        self.validate_asr_engine_implemented()?;

        // Validate the (command, lang) pairing is a legal one. Morphotag,
        // translate, and coref MUST submit `LanguageSpec::PerFile`; every
        // other processing command MUST NOT. This boundary check keeps
        // pipeline code from ever observing an invalid combination, and
        // makes the dashboard / job record show honest values.
        self.validate_lang_command_pairing()?;

        // Validate language support for engines the command will use.
        self.validate_language_support()?;

        if self.paths_mode {
            if self.source_paths.is_empty() || self.output_paths.is_empty() {
                return Err(ValidationError(
                    "paths_mode requires non-empty source_paths and output_paths".into(),
                ));
            }
            if self.source_paths.len() != self.output_paths.len() {
                return Err(ValidationError(
                    "source_paths and output_paths must have equal length".into(),
                ));
            }
            if !self.before_paths.is_empty() && self.before_paths.len() != self.source_paths.len() {
                return Err(ValidationError(
                    "before_paths must have the same length as source_paths when non-empty".into(),
                ));
            }
            if !self.files.is_empty() || !self.media_files.is_empty() {
                return Err(ValidationError(
                    "paths_mode is mutually exclusive with files/media_files".into(),
                ));
            }
        }
        Ok(())
    }

    /// The ASR engine this submission will actually use, if it uses one.
    ///
    /// THE owner of that question for validation. It was written out twice,
    /// once per check that needed it; a third check would have made a third
    /// copy, and the copies are what drift when a command joins the family.
    fn selected_asr_engine(&self) -> Option<AsrEngineName> {
        match &self.options {
            CommandOptions::Transcribe(opts) | CommandOptions::TranscribeS(opts) => {
                Some(opts.effective_asr_engine())
            }
            CommandOptions::Benchmark(opts) => Some(opts.effective_asr_engine()),
            _ => None,
        }
    }

    /// Refuse an ASR engine whose NAME this build accepts and whose runtime it
    /// does not have.
    ///
    /// `whisperx` and `whisper_oai` parse, appear in `--help`, and round-trip
    /// through the wire format, and nothing in this workspace implements
    /// either WhisperX or the OpenAI Whisper API. Backend selection used to
    /// end in a catch-all arm that ran stock local Whisper for both, so such a
    /// job SUCCEEDED while running an engine nobody asked for and writing that
    /// other engine's name into provenance.
    ///
    /// The verdict comes from `AsrBackend::try_from_engine`, the same total
    /// function dispatch uses, and the alternatives are derived from it as
    /// well, so this message and the runtime cannot disagree about what works.
    fn validate_asr_engine_implemented(&self) -> Result<(), ValidationError> {
        use crate::transcribe::AsrBackend;

        let Some(engine) = self.selected_asr_engine() else {
            return Ok(());
        };
        let Err(refusal) = AsrBackend::try_from_engine(&engine) else {
            return Ok(());
        };

        let alternatives: Vec<&str> = AsrBackend::implemented_engine_names().collect();
        Err(ValidationError(format!(
            "{refusal}. Use --asr-engine with one of: {}.",
            alternatives.join(", ")
        )))
    }

    /// Reject cache flag combinations that would ask one task to both reuse
    /// and replace evidence. CLI parsing already prevents these shapes; this
    /// check protects direct HTTP clients at the same admission boundary.
    fn validate_cache_policy(&self) -> Result<(), ValidationError> {
        use crate::chat_ops::CacheTaskName;
        use crate::chat_ops::cache_key::CacheOverrideTaskName;

        let common = self.options.common();
        if common.require_media_cache
            && (common.override_media_cache || !common.override_media_cache_tasks.is_empty())
        {
            return Err(ValidationError(
                "require_media_cache is mutually exclusive with media-cache refresh options"
                    .to_owned(),
            ));
        }
        if common.override_media_cache && !common.override_media_cache_tasks.is_empty() {
            return Err(ValidationError(
                "override_media_cache is mutually exclusive with override_media_cache_tasks"
                    .to_owned(),
            ));
        }
        if let Some(unknown) = common.override_media_cache_tasks.iter().find(|name| {
            matches!(
                CacheTaskName::classify_override_name(name),
                CacheOverrideTaskName::Unknown
            )
        }) {
            return Err(ValidationError(format!(
                "unknown media-cache override task {unknown:?}"
            )));
        }
        Ok(())
    }

    /// Check that the job's language is supported by all engines the command
    /// will use.
    ///
    /// Called at job submission time to fail fast with a clear diagnostic
    /// rather than letting errors surface deep in the pipeline (Rev.AI HTTP
    /// 400, Whisper wrong-language transcription, Stanza model-not-found).
    /// Reject (command, lang) combinations that would be lies.
    ///
    /// `LanguageSpec::PerFile` is the unique correct shape for morphotag,
    /// translate, and coref; they have no `--lang` CLI flag and read
    /// language per-file from each CHAT file's `@Languages:` header. Any
    /// other shape on those commands means a placeholder is sneaking
    /// through (the 2026-05-03 morphotag incident).
    ///
    /// Conversely, every other processing command (transcribe,
    /// transcribe_s, benchmark, align, compare, utseg, opensmile, avqi)
    /// takes a concrete `--lang` (or `--lang auto` for ASR-detect). Those
    /// must never carry `PerFile`; if they do, something downstream of
    /// the CLI built a malformed `JobSubmission`.
    fn validate_lang_command_pairing(&self) -> Result<(), ValidationError> {
        use crate::api::ReleasedCommand;
        use crate::types::domain::LanguageSpec;

        let is_per_file_command = matches!(
            self.command,
            ReleasedCommand::Morphotag | ReleasedCommand::Translate | ReleasedCommand::Coref,
        );

        match (&self.lang, is_per_file_command) {
            (LanguageSpec::PerFile, true) => Ok(()),
            (LanguageSpec::PerFile, false) => Err(ValidationError(format!(
                "command '{}' does not accept LanguageSpec::PerFile; pass --lang or --lang auto",
                self.command
            ))),
            (LanguageSpec::Auto | LanguageSpec::Resolved(_), true) => {
                Err(ValidationError(format!(
                    "command '{}' has no --lang; submission must use LanguageSpec::PerFile (per-file \
                 resolution from @Languages: header). Job-level lang sentinels are banned for \
                 this command: see the 2026-05-03 morphotag incident.",
                    self.command
                )))
            }
            (LanguageSpec::Auto | LanguageSpec::Resolved(_), false) => Ok(()),
        }
    }

    fn validate_language_support(&self) -> Result<(), ValidationError> {
        // Auto-detect and per-file resolution both defer language to a later
        // stage, so submission-time engine-support checks can't run here. For
        // PerFile commands (morphotag/translate/coref) the per-file
        // `@Languages:` header is the authority; they don't use Rev.AI or
        // engine-tied processing this validator covers.
        let lang = match &self.lang {
            LanguageSpec::Auto | LanguageSpec::PerFile => return Ok(()),
            LanguageSpec::Resolved(code) => code,
        };

        // Commands that use eager request-level ASR validation: transcribe,
        // transcribe_s, benchmark.
        //
        // Align is intentionally excluded here. Whether align needs a UTR/ASR
        // stage depends on the parsed file's timing state, so align-specific
        // backend validation is deferred to the runtime after parsing.
        let asr_engine = self.selected_asr_engine();

        // Check Rev.AI language support
        if let Some(AsrEngineName::RevAi) = &asr_engine
            && try_revai_language_hint(lang).is_none()
        {
            // Not derived like the UTR message below: ASR has no per-engine
            // language predicate to filter on, since Rev.AI is the only engine
            // whose language support is modelled at all. Deriving it means
            // modelling that first.
            return Err(ValidationError(format!(
                "Language '{}' is not supported by Rev.AI ASR. Alternatives:\n\
                 - Use --asr-engine whisper for local Whisper ASR (supports most languages)\n\
                 - Use --asr-engine tencent for Chinese/Hakka via Tencent\n\
                 - Run `batchalign3 transcribe --help` for every engine name\n\
                 - Check supported languages: book/src/batchalign/reference/language-code-resolution.md",
                lang
            )));
        }

        // Rev.AI known-broken (engine, language) deny-list.
        //
        // Rev.AI advertises support for a language but has been observed to
        // return output unusable for CHAT construction, see
        // `REVAI_KNOWN_BROKEN` in `types/revai_language.rs` for current entries
        // with dated provenance. Rejecting at preflight turns a late-stage
        // per-token validation failure into a clear up-front message that
        // names a working alternative.
        //
        // Rationale and escalation path (when the deny-list stops being
        // enough): book/src/batchalign/reference/revai-language-quality-strategy.md.
        if let Some(AsrEngineName::RevAi) = &asr_engine
            && let Some(entry) = revai_known_broken(lang)
        {
            return Err(ValidationError(format!(
                "Language '{}' is known to produce unusable quality on Rev.AI ASR: {}. \
                 Use --asr-engine {} instead. \
                 See book/src/batchalign/reference/revai-language-quality-strategy.md \
                 for the rationale and the list of known-broken pairs.",
                lang, entry.reason, entry.recommended_engine
            )));
        }

        // Commands that use Stanza: morphotag, utseg, coref, compare
        let uses_stanza = matches!(
            &self.options,
            CommandOptions::Morphotag(_)
                | CommandOptions::Utseg(_)
                | CommandOptions::Coref(_)
                | CommandOptions::Compare(_)
        );
        if uses_stanza && !is_stanza_supported_language(lang)? {
            return Err(ValidationError(format!(
                "Language '{}' is not supported by Stanza. Supported languages:\n\
                 {}",
                lang,
                stanza_supported_languages_help()
            )));
        }

        // Check HK ASR engine language constraints
        if let Some(engine) = &asr_engine {
            let chinese_codes = ["zho", "yue", "wuu", "nan", "hak", "cmn"];
            match engine {
                AsrEngineName::HkTencent if !chinese_codes.contains(&lang.as_ref()) => {
                    return Err(ValidationError(format!(
                        "Language '{}' is not supported by Tencent ASR (Chinese variants only: {}). \
                         Use --asr-engine whisper or --asr-engine rev instead.",
                        lang,
                        chinese_codes.join(", ")
                    )));
                }
                AsrEngineName::HkAliyun if lang.as_ref() != "yue" => {
                    return Err(ValidationError(format!(
                        "Language '{}' is not supported by Aliyun ASR (Cantonese 'yue' only). \
                         Use --asr-engine whisper or --asr-engine rev instead.",
                        lang
                    )));
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Validate that one selected UTR backend can support a resolved language.
///
/// This is intended for stage-aware runtime checks inside `align`, after the
/// file has been parsed and the runtime knows whether UTR is actually needed.
pub(crate) fn validate_utr_language_support(
    lang: &LanguageCode3,
    engine: &UtrEngine,
) -> Result<(), ValidationError> {
    if utr_engine_supports_language(engine, lang) {
        return Ok(());
    }

    // The alternatives are DERIVED from the same predicate that produced the
    // rejection, not written out as prose, so the message cannot recommend an
    // engine that does not work or name a flag that has been renamed. Why this
    // matters: see `SelectableEngine`.
    let alternatives: Vec<&str> = UtrEngine::ALL
        .iter()
        .filter(|candidate| utr_engine_supports_language(candidate, lang))
        .map(UtrEngine::selection_name)
        .collect();

    let remedy = if alternatives.is_empty() {
        "No UTR engine supports this language; run with --no-utr to skip \
         utterance timing recovery."
            .to_owned()
    } else {
        format!("Use --utr-engine with one of: {}.", alternatives.join(", "))
    };

    Err(ValidationError(format!(
        "This file requires utterance timing recovery, but the selected UTR \
         engine '{}' does not support language '{lang}'. {remedy}",
        engine.selection_name()
    )))
}

/// Whether one FA engine can handle one language.
///
/// A field lookup on the engine's own row in `FA_ENGINES`, not a per-engine
/// match. This used to be three things: an ISO-code list, a `FaLanguageScope`
/// enum naming the "does this engine read the declaration at all" question,
/// and a roster match mapping every variant onto one of them, whose second
/// level carried an arm its own comment called unreachable by construction.
/// The scope WAS the shape of the list, so `LanguageSupport` is now one field
/// and the unreachable arm has nowhere to be written.
///
/// The same shape as [`utr_engine_supports_language`]: the rejection message
/// below derives its list of alternatives from this predicate, so a message
/// cannot recommend an engine that would also refuse.
fn fa_engine_supports_language(engine: FaEngineName, lang: &LanguageCode3) -> bool {
    engine.language_support().covers(lang.as_ref())
}

/// One SECONDARY `@Languages:` entry, as far as we could read it.
///
/// A sum type rather than an `Option<LanguageCode3>` because the two cases are
/// different operator actions: a supported code needs nothing, an unsupported
/// one needs a different engine, and one we cannot parse at all needs the
/// header fixed. Only a `General` engine may ignore either.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DeclaredSecondary {
    /// A parseable ISO-639-3 code.
    Code(LanguageCode3),
    /// Declared, but not a code we can look up in any engine's support list.
    Unreadable(String),
}

/// Every language a file's `@Languages:` header declares, in header order.
///
/// The first is the PRIMARY; the rest are SECONDARY, and they are not
/// decoration. An utterance switches to a secondary language with a `[- deu]`
/// precode and a single word with `word@s:deu`, so a file declaring
/// `eng, deu` really does contain German words, and they travel to the aligner
/// in the same groups as the English ones under the same single label.
///
/// A distinct type from `LanguageCode3` because admission used to read
/// `languages.first()` and so ADMITTED `@Languages: eng, deu` for the Qwen3
/// aligner while REFUSING `@Languages: deu, eng` over identical content. The
/// primary alone cannot answer the question, so it is no longer what gets
/// asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeclaredLanguages {
    primary: LanguageCode3,
    secondary: Vec<DeclaredSecondary>,
}

/// Why a `@Languages:` header plus a `--lang` fallback yields no primary.
///
/// Two cases and not one, because they are different operator actions: a file
/// with no header needs one written or a `--lang` passed, while a file whose
/// first entry is unreadable needs that entry fixed. Rendered by the align
/// dispatch, which is the only thing that knows the filename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HeaderLanguageError {
    /// The header's first entry is not a parseable code and there is no
    /// `--lang` to stand in for it.
    UnreadablePrimary {
        /// The entry exactly as the file spells it.
        raw: String,
    },
    /// There is no `@Languages:` header at all and no `--lang` to stand in.
    NoHeader,
}

impl DeclaredLanguages {
    /// Resolve a whole `@Languages:` header against the job's `--lang`.
    ///
    /// THE only constructor, and it owns the resolution the align dispatch
    /// used to do at its own call site. That split is what made header ORDER
    /// decide admission: the caller resolved the primary (falling back to
    /// `--lang` when the first entry would not parse) and then passed
    /// `skip(1)` of the header as the secondaries, so an unreadable FIRST
    /// entry was dropped on the floor while an unreadable SECOND one refused
    /// the file.
    ///
    /// Here the fallback replaces only the unreadable entry's ROLE as primary;
    /// the entry itself stays in the declaration as `Unreadable`, because it
    /// is still something the file declares and still something the engine
    /// must be shown to support.
    pub(crate) fn from_header<'a>(
        entries: impl IntoIterator<Item = &'a str>,
        fallback: Option<&LanguageCode3>,
    ) -> Result<Self, HeaderLanguageError> {
        let entries: Vec<&'a str> = entries.into_iter().collect();
        let Some((first, rest)) = entries.split_first() else {
            return match fallback {
                Some(code) => Ok(Self {
                    primary: code.clone(),
                    secondary: Vec::new(),
                }),
                None => Err(HeaderLanguageError::NoHeader),
            };
        };

        let (primary, displaced) = match LanguageCode3::try_new(first) {
            Ok(code) => (code, None),
            Err(_) => match fallback {
                Some(code) => (
                    code.clone(),
                    Some(DeclaredSecondary::Unreadable((*first).to_owned())),
                ),
                None => {
                    return Err(HeaderLanguageError::UnreadablePrimary {
                        raw: (*first).to_owned(),
                    });
                }
            },
        };

        Ok(Self {
            primary,
            secondary: displaced
                .into_iter()
                .chain(rest.iter().map(|raw| match LanguageCode3::try_new(raw) {
                    Ok(code) => DeclaredSecondary::Code(code),
                    Err(_) => DeclaredSecondary::Unreadable((*raw).to_owned()),
                }))
                .collect(),
        })
    }

    /// The primary language, for a caller that also needs it on its own.
    ///
    /// Read from the declaration rather than resolved a second time, so the
    /// language a file is processed under is necessarily the one that was
    /// validated.
    pub(crate) fn primary(&self) -> &LanguageCode3 {
        &self.primary
    }
}

/// Proof that a run's forced-alignment engine was checked against every
/// language its file declares.
///
/// # Every way to obtain one, enumerated
///
/// [`validate_fa_language_support`], and nothing else. No other constructor,
/// no `Default`, no `#[cfg(test)]` shortcut, no public field: a caller holding
/// a [`FaParams`] and a [`DeclaredLanguages`] cannot assert the relation
/// between them, it has to be checked.
///
/// # What actually stops the unproven pair being used
///
/// NOT the move. [`DeclaredLanguages`] is genuinely consumed, but [`FaParams`]
/// is `Copy`, so the caller keeps a perfectly usable unvalidated copy of it and
/// always did. What protects the run is downstream: `AlignAudioTask` holds this
/// proof and NOT a `FaParams`, so the align path has no unvalidated engine to
/// read. Say what the type does, since a reader who believes the move is the
/// guarantee will not look for the thing that is.
///
/// # Why it exists
///
/// The validator returned `Result<(), _>`, so the check left no value behind
/// and the permission drifted from the run. The engine that aligned the file
/// and the language stamped into its provenance were both read from variables
/// the caller happened to still be holding, and one of them was read BEFORE
/// the check: reordering two statements would have un-validated the run with
/// nothing to notice. Reading them off this type instead makes the pair that
/// reaches the dispatch the pair that was admitted.
///
/// # What this proof does NOT cover
///
/// The engine that ends up running one GROUP. A group whose engine reports a
/// recoverable constraint is re-dispatched on that engine's declared fallback
/// target (`FaFallbackPolicy::RetryGroupOn`), and that re-dispatch does not
/// come back through this validator. It is sound because every fallback target
/// is checked at COMPILE TIME to be [`LanguageSupport::Any`], in the `const`
/// block beside `FA_ENGINES`, so no declared language can make it wrong. Until
/// 2026-09-07 it was sound only by accident: the target was the literal
/// `Whisper` at the call site, and nothing said a language-restricted engine
/// could not be named there.
///
/// [`LanguageSupport::Any`]: crate::types::engines::LanguageSupport::Any
#[derive(Debug)]
pub(crate) struct AdmittedFaParams {
    params: FaParams,
    declared: DeclaredLanguages,
}

impl AdmittedFaParams {
    /// The run parameters, whose engine is the one that was admitted.
    pub(crate) fn params(&self) -> &FaParams {
        &self.params
    }

    /// The primary declared language, read off the declaration that was
    /// CHECKED rather than resolved a second time from the header.
    pub(crate) fn primary_language(&self) -> &LanguageCode3 {
        self.declared.primary()
    }
}

/// Validate that the selected forced-alignment engine can support EVERY
/// language a file declares, refusing by name rather than substituting another
/// engine.
///
/// Called per file in the align dispatch, once `@Languages:` has resolved,
/// which is the first moment both facts are known. The alternative would be a
/// silent fallback to a language-general engine, and a run that quietly used a
/// different aligner than the one asked for is a run whose numbers mean
/// something other than what the operator thinks.
///
/// Returns the [`AdmittedFaParams`] proof rather than `()`: a validator that
/// leaves no value behind is a check anything downstream may forget was made.
pub(crate) fn validate_fa_language_support(
    declared: DeclaredLanguages,
    params: FaParams,
) -> Result<AdmittedFaParams, ValidationError> {
    let engine = params.engine;
    // A language-general engine never reads the declaration, so nothing in it
    // can make the engine wrong, including an entry we cannot parse.
    if engine.language_support().is_language_general() {
        return Ok(AdmittedFaParams { params, declared });
    }

    if !fa_engine_supports_language(engine, &declared.primary) {
        return Err(fa_language_refusal(engine, &declared.primary, None));
    }
    for entry in &declared.secondary {
        match entry {
            DeclaredSecondary::Code(code) if fa_engine_supports_language(engine, code) => {}
            DeclaredSecondary::Code(code) => {
                return Err(fa_language_refusal(engine, code, Some(SECONDARY_NOTE)));
            }
            DeclaredSecondary::Unreadable(raw) => {
                return Err(ValidationError(format!(
                    "The file declares `@Languages:` entry '{raw}', which is not a parseable \
                     ISO 639-3 code, so we cannot show that the selected forced-alignment \
                     engine '{}' supports it. Fix the header, or use --fa-engine with one of: \
                     {}.",
                    engine.selection_name(),
                    language_general_engine_names().join(", ")
                )));
            }
        }
    }
    Ok(AdmittedFaParams { params, declared })
}

/// Why a SECONDARY language still decides admission, said once.
const SECONDARY_NOTE: &str = " It is a secondary language in `@Languages:`, and its \
                              utterances reach the aligner in the same groups, under the \
                              same single language label, as the primary language's.";

/// The engines that would accept any language, for a remedy line.
fn language_general_engine_names() -> Vec<&'static str> {
    FaEngineName::ALL
        .iter()
        .filter(|candidate| candidate.language_support().is_language_general())
        .map(FaEngineName::selection_name)
        .collect()
}

/// The one refusal message, so the primary and secondary cases cannot drift.
fn fa_language_refusal(
    engine: FaEngineName,
    lang: &LanguageCode3,
    note: Option<&str>,
) -> ValidationError {
    let alternatives: Vec<&str> = FaEngineName::ALL
        .iter()
        .filter(|candidate| fa_engine_supports_language(**candidate, lang))
        .map(FaEngineName::selection_name)
        .collect();

    let remedy = if alternatives.is_empty() {
        "No forced-alignment engine supports this language.".to_owned()
    } else {
        format!("Use --fa-engine with one of: {}.", alternatives.join(", "))
    };

    ValidationError(format!(
        "The selected forced-alignment engine '{}' does not support language '{lang}'.{} {remedy}",
        engine.selection_name(),
        note.unwrap_or("")
    ))
}

/// The Chinese variants Tencent UTR covers.
const TENCENT_UTR_LANGUAGES: [&str; 6] = ["zho", "yue", "wuu", "nan", "hak", "cmn"];

/// Whether one UTR engine can handle one language.
///
/// THE owner of that question. It was previously inlined into the validation
/// match, which meant the only way to ask "what else would work here" was to
/// restate the answer in an error string.
fn utr_engine_supports_language(engine: &UtrEngine, lang: &LanguageCode3) -> bool {
    match engine {
        UtrEngine::RevAi => try_revai_language_hint(lang).is_some(),
        UtrEngine::HkTencent => TENCENT_UTR_LANGUAGES.contains(&lang.as_ref()),
        // Local Whisper is language-general.
        UtrEngine::Whisper => true,
    }
}

/// Validation error for request models.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{0}")]
pub struct ValidationError(pub String);

// ---------------------------------------------------------------------------
// Stanza language support: hardcoded fallback table
// ---------------------------------------------------------------------------

/// Hardcoded fallback table of ISO 639-3 codes supported by Stanza.
///
/// **DEPRECATED as the primary check.** The authoritative source is now
/// the `StanzaRegistry` built from Stanza's `resources.json` at worker
/// startup. This table is ONLY used as a pre-validation safety net when
/// the registry hasn't been populated yet (before first worker spawn).
///
/// Check whether an ISO 639-3 language code is supported by Stanza.
///
/// Single Rust source of truth: delegates to
/// `crate::chat_ops::morphosyntax_ops::is_stanza_supported`.
/// A previous hardcoded `STANZA_SUPPORTED_ISO3` list duplicated that data
/// and silently drifted from it (see the 2026-04-24 Malayalam crash
/// audit at `stanza_languages.rs` module docs). The authoritative
/// truth ultimately lives in the Python capability table built from
/// Stanza's installed `resources.json`; this Rust function is a fast
/// preflight that uses the most-recently-audited approximation.
///
/// Fallible only because `LanguageCode` construction became fallible in
/// chatter 0.3.0 (rejects empty input); `LanguageCode3` guarantees three
/// ASCII letters, so the error arm is unreachable in practice but
/// propagated rather than panicked on. The error is stringified because
/// chatter v0.3.0 does not re-export `LanguageCodeError` (upstream
/// defect, reported).
fn is_stanza_supported_language(lang: &LanguageCode3) -> Result<bool, ValidationError> {
    let code = crate::chat_ops::LanguageCode::new(lang.as_ref()).map_err(|e| {
        ValidationError(format!(
            "Language '{lang}' is not a usable language code: {e}"
        ))
    })?;
    Ok(crate::chat_ops::morphosyntax_ops::is_stanza_supported(
        &code,
    ))
}

/// Format a help string listing supported Stanza languages for error messages.
fn stanza_supported_languages_help() -> String {
    crate::chat_ops::morphosyntax_ops::supported_iso3_codes()
        .chunks(10)
        .map(|chunk| chunk.join(", "))
        .collect::<Vec<_>>()
        .join(",\n  ")
}

/// Validate a job's language support using the runtime Stanza registry.
///
/// This is the **authoritative** language validation, called from
/// `materialize_submission_job()` where the registry is available.
/// It supersedes the hardcoded `is_stanza_supported_language()` check
/// in `validate_language_support()`, which acts as a conservative
/// pre-filter only.
///
/// Returns `Ok(())` when:
/// - The command doesn't use Stanza
/// - The language is auto-detect
/// - The registry confirms the language has required processors
/// - The registry is not populated (fallback to hardcoded table)
pub fn validate_language_with_registry(
    submission: &JobSubmission,
    registry: Option<&crate::stanza_registry::StanzaRegistry>,
) -> Result<(), ValidationError> {
    // Auto: can't validate until ASR resolves the language.
    // PerFile: morphotag/translate/coref resolve per-file from @Languages:;
    // the registry validation happens per-file in stage_parse.
    let lang = match &submission.lang {
        LanguageSpec::Auto | LanguageSpec::PerFile => return Ok(()),
        LanguageSpec::Resolved(code) => code,
    };

    let uses_stanza = matches!(
        &submission.options,
        CommandOptions::Morphotag(_)
            | CommandOptions::Utseg(_)
            | CommandOptions::Coref(_)
            | CommandOptions::Compare(_)
    );

    if !uses_stanza {
        return Ok(());
    }

    let Some(reg) = registry else {
        // Registry not populated: the hardcoded table in validate() already
        // caught obviously unsupported languages.
        return Ok(());
    };

    if !reg.supports_morphosyntax(lang.as_ref()) {
        let supported = reg.supported_languages().join(", ");
        return Err(ValidationError(format!(
            "Language '{}' is not supported by Stanza on this server. \
             Supported languages: {}",
            lang, supported
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{
        AlignOptions, CommonOptions, MorphotagOptions, TranscribeOptions, UtsegOptions,
    };

    /// Build a minimal morphotag `JobSubmission` for testing validation.
    ///
    /// Morphotag has no `--lang` flag, every legal submission carries
    /// `LanguageSpec::PerFile`. The `lang_spec` parameter exists only so
    /// regression tests can construct *invalid* submissions (e.g. with
    /// `Resolved(eng)`) and assert that `validate()` rejects them.
    fn morphotag_submission_with_lang(lang_spec: LanguageSpec) -> JobSubmission {
        JobSubmission {
            command: ReleasedCommand::Morphotag,
            lang: lang_spec,
            num_speakers: NumSpeakers(1),
            files: vec![],
            media_files: vec![],
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),

                ..Default::default()
            }),
            paths_mode: false,
            source_paths: vec![],
            output_paths: vec![],
            display_names: vec![],
            debug_traces: false,
            before_paths: vec![],
        }
    }

    /// Convenience for the common case: a legal morphotag submission.
    fn morphotag_submission() -> JobSubmission {
        morphotag_submission_with_lang(LanguageSpec::PerFile)
    }

    fn utseg_submission(lang: &str) -> JobSubmission {
        JobSubmission {
            command: ReleasedCommand::Utseg,
            lang: LanguageSpec::Resolved(LanguageCode3::try_new(lang).expect("test lang")),
            num_speakers: NumSpeakers(1),
            files: vec![],
            media_files: vec![],
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
            options: CommandOptions::Utseg(UtsegOptions {
                common: CommonOptions::default(),
                merge_abbrev: Default::default(),
                utseg_fallback: Default::default(),
            }),
            paths_mode: false,
            source_paths: vec![],
            output_paths: vec![],
            display_names: vec![],
            debug_traces: false,
            before_paths: vec![],
        }
    }

    fn align_submission(lang: &str, utr_engine: Option<UtrEngine>) -> JobSubmission {
        JobSubmission {
            command: ReleasedCommand::Align,
            lang: LanguageSpec::Resolved(LanguageCode3::try_new(lang).expect("test lang")),
            num_speakers: NumSpeakers(1),
            files: vec![],
            media_files: vec![],
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
            options: CommandOptions::Align(AlignOptions {
                common: CommonOptions::default(),
                utr: crate::options::AlignUtrOptions {
                    engine: utr_engine,
                    ..Default::default()
                },
                ..AlignOptions::default()
            }),
            paths_mode: true,
            source_paths: vec!["/tmp/test.cha".into()],
            output_paths: vec!["/tmp/out.cha".into()],
            display_names: vec![],
            debug_traces: false,
            before_paths: vec![],
        }
    }

    /// Build a minimal transcribe submission parameterized by language and
    /// ASR engine. Used by the deny-list tests below to exercise the
    /// validation path without spinning up a real server.
    fn transcribe_submission(lang: &str, asr_engine: AsrEngineName) -> JobSubmission {
        JobSubmission {
            command: ReleasedCommand::Transcribe,
            lang: LanguageSpec::Resolved(LanguageCode3::try_new(lang).expect("test lang")),
            num_speakers: NumSpeakers(1),
            files: vec![],
            media_files: vec![],
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: Default::default(),
            options: CommandOptions::Transcribe(TranscribeOptions {
                common: CommonOptions::default(),
                asr_engine,
                diarize: false,
                wor: false.into(),
                merge_abbrev: false.into(),
                utseg_fallback: false.into(),
                batch_size: 8,
            }),
            paths_mode: true,
            source_paths: vec!["/tmp/test.mp3".into()],
            output_paths: vec!["/tmp/out.cha".into()],
            display_names: vec![],
            debug_traces: false,
            before_paths: vec![],
        }
    }

    // --- RED: known-broken (engine, language) pair deny-list -----------------
    //
    // As of 2026-04-22, Rev.AI's Malayalam (lang=ml / iso3=mal) ASR is
    // unusable in practice. A ~1-minute Malayalam sample re-submitted
    // directly to Rev.AI with language=ml returned 55 text elements
    // comprising Korean Hangul mixed with Malayalam vowel signs
    // ('모두െ'), stray Latin words ('occurrence', 'Moo', 'Take', 'Me',
    // 'ganhar', 'segueiasm'), bare Gurmukhi/Punjabi tokens in the final
    // third of the transcript, U+FFFD replacement characters inside
    // tokens (');�', 'philan�ുടഖ഻ിറ്'), and semicolon+paren punctuation
    // as "words" (');�'). Evidence is kept in an operational workspace
    // outside this repo; see the strategy doc for the procedure.
    //
    // `try_revai_language_hint("mal")` maps to "ml" which Rev.AI accepts
    // but the result is cross-script garbage that no CHAT validator can
    // accept. Propagating that output produces confusing late-stage E220 /
    // E330 validation errors on arbitrary tokens; users have no way to tell
    // it was the ASR that failed, not their transcript.
    //
    // The fix is to extend `validate_language_support()` with a known-broken
    // deny-list that rejects unusable (engine, language) pairs at submission
    // time and points the user at a working alternative.
    #[test]
    fn transcribe_rev_on_malayalam_is_rejected_as_known_broken() {
        let submission = transcribe_submission("mal", AsrEngineName::RevAi);
        let err = submission
            .validate()
            .expect_err("Rev.AI + Malayalam must be rejected at preflight");
        let msg = err.to_string();
        assert!(
            msg.contains("mal") || msg.contains("Malayalam"),
            "error must name the offending language; got: {msg}"
        );
        assert!(
            msg.contains("whisper_hub"),
            "error must recommend --asr-engine whisper_hub specifically \
             (stock whisper is also empirically broken for mal; see the \
             strategy doc); got: {msg}"
        );
        // The message must explain *why* (quality / known-broken), so the
        // user understands this isn't their file.
        let low = msg.to_lowercase();
        assert!(
            low.contains("known") || low.contains("quality") || low.contains("unusable"),
            "error must explain quality / known-broken reason; got: {msg}"
        );
    }

    /// Guard rail: the deny-list must not over-reject. `eng` + Rev.AI is the
    /// default path used by every English-language job on the fleet and
    /// must keep passing validation.
    #[test]
    fn transcribe_rev_on_english_remains_valid() {
        let submission = transcribe_submission("eng", AsrEngineName::RevAi);
        submission
            .validate()
            .expect("eng + Rev.AI must remain valid; deny-list must not over-reject");
    }

    /// `whisperx` and `whisper_oai` are accepted engine NAMES with no
    /// implementation anywhere in the tree. Selecting one used to fall through
    /// a catch-all arm to stock local Whisper, so the job ran a different
    /// engine than the one asked for and said so nowhere. Submission is the
    /// boundary where that must be refused, because it is the one the operator
    /// sees before any work starts.
    #[test]
    fn submission_refuses_unimplemented_asr_engines() {
        for engine in [AsrEngineName::WhisperX, AsrEngineName::WhisperOai] {
            let wire_name = engine.as_wire_name();
            let error = transcribe_submission("eng", engine)
                .validate()
                .expect_err("an unimplemented engine must be refused at submission");
            let message = error.to_string();
            assert!(
                message.contains(wire_name),
                "the refusal must name the engine that was asked for; got: {message}"
            );
            assert!(
                message.contains("not implemented"),
                "the refusal must say the engine is missing, not merely unsupported; \
                 got: {message}"
            );
            assert!(
                message.contains("whisper_rs"),
                "the refusal must name engines that DO work, derived from the same \
                 function that refused this one; got: {message}"
            );
        }
    }

    #[test]
    fn submission_rejects_required_cache_combined_with_refresh() {
        let mut submission = transcribe_submission("eng", AsrEngineName::RevAi);
        let common = match &mut submission.options {
            CommandOptions::Transcribe(options) => &mut options.common,
            _ => panic!("helper must build transcribe options"),
        };
        common.require_media_cache = true;
        common.override_media_cache = true;

        let error = submission
            .validate()
            .expect_err("one job cannot both require and refresh media evidence");

        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn submission_rejects_unknown_selective_cache_task() {
        let mut submission = transcribe_submission("eng", AsrEngineName::RevAi);
        let common = match &mut submission.options {
            CommandOptions::Transcribe(options) => &mut options.common,
            _ => panic!("helper must build transcribe options"),
        };
        common.override_media_cache_tasks = vec!["rev_asr_evidnce".to_owned()];

        let error = submission
            .validate()
            .expect_err("an unknown experiment cache domain must fail closed");

        assert!(error.to_string().contains("rev_asr_evidnce"));
    }

    /// Whisper is a fallback alternative for languages where the stock
    /// model still produces usable output. It must itself pass validation
    /// for Malayalam submissions so any downstream escalation from Rev.AI
    /// (or any other engine) doesn't hit a second validation failure.
    ///
    /// Caveat: empirical evaluation (2026-04-22) showed stock
    /// Whisper's Malayalam output is *also* broken, it collapses into
    /// Khmer/Gurmukhi loops and hallucinates "Thank you for watching."
    /// That is why the Rev.AI deny-list for ``mal`` now recommends
    /// ``whisper_hub``, not ``whisper``. This test remains as a *validation*
    /// guard rail (API-supported), not a quality claim.
    #[test]
    fn transcribe_whisper_on_malayalam_passes_validation() {
        let submission = transcribe_submission("mal", AsrEngineName::Whisper);
        submission
            .validate()
            .expect("whisper + mal must pass validation; quality caveats live in the book");
    }

    /// ``whisper_hub`` is the recommended alternative in the updated
    /// Rev.AI deny-list error message (see
    /// ``book/src/batchalign/reference/revai-language-quality-strategy.md``). For
    /// the recommendation to be viable, ``whisper_hub`` + ``mal`` must
    /// pass validation itself.
    #[test]
    fn transcribe_whisper_hub_on_malayalam_passes_validation() {
        let submission = transcribe_submission("mal", AsrEngineName::WhisperHub);
        submission.validate().expect(
            "whisper_hub + mal must pass validation so the deny-list \
             recommendation is viable",
        );
    }

    #[test]
    fn morphotag_per_file_passes_validation() {
        // The only legal morphotag submission shape: PerFile lang.
        let submission = morphotag_submission();
        submission
            .validate()
            .expect("morphotag with LanguageSpec::PerFile must pass validation");
    }

    /// 2026-05-03 incident regression test. A morphotag job-level
    /// `Resolved(eng)` was the historical placeholder; submission-time
    /// validation must reject it so it never appears in job records or
    /// leaks into worker pre-warming.
    #[test]
    fn morphotag_resolved_lang_is_rejected() {
        let submission = morphotag_submission_with_lang(LanguageSpec::Resolved(
            LanguageCode3::try_new("eng").unwrap(),
        ));
        let err = submission
            .validate()
            .expect_err("morphotag with Resolved(eng) must be rejected");
        assert!(
            err.to_string().contains("LanguageSpec::PerFile"),
            "rejection message must point at PerFile remedy: {err}"
        );
    }

    #[test]
    fn morphotag_auto_lang_is_rejected() {
        let submission = morphotag_submission_with_lang(LanguageSpec::Auto);
        let err = submission
            .validate()
            .expect_err("morphotag with Auto must be rejected (Auto is an ASR-engine signal)");
        assert!(err.to_string().contains("LanguageSpec::PerFile"));
    }

    /// Mirror coverage: utseg DOES take an explicit `--lang`, so it must
    /// continue to reject `PerFile` (the per-file variant is reserved for
    /// morphotag, translate, coref).
    #[test]
    fn utseg_per_file_lang_is_rejected() {
        let mut submission = utseg_submission("eng");
        submission.lang = LanguageSpec::PerFile;
        let err = submission
            .validate()
            .expect_err("utseg with PerFile must be rejected");
        assert!(
            err.to_string()
                .contains("does not accept LanguageSpec::PerFile")
        );
    }

    #[test]
    fn utseg_with_unsupported_language_fails() {
        let submission = utseg_submission("xyz");
        let err = submission.validate().unwrap_err();
        assert!(
            err.to_string().contains("not supported by Stanza"),
            "expected Stanza error, got: {err}"
        );
    }

    #[test]
    fn align_with_unsupported_rev_language_does_not_fail_request_validation() {
        let submission = align_submission("yue", Some(UtrEngine::RevAi));
        assert!(
            submission.validate().is_ok(),
            "align should defer UTR language checks until file timing state is known"
        );
    }

    /// Rev.AI cannot do Cantonese, and the user is told what can.
    ///
    /// This used to pin the exact sentence `"Use --utr-engine whisper"`, which
    /// made the prose the contract: the remedy could not gain the Cantonese
    /// engine, or lose the flag name it had outlived, without a test failing
    /// for the wrong reason. It now asserts that a working engine is offered.
    /// Which engines those are, and that the list cannot disagree with the
    /// verdict, is covered in `utr_language_support_tests`.
    #[test]
    fn utr_runtime_validation_rejects_rev_for_unsupported_language() {
        let err =
            validate_utr_language_support(&LanguageCode3::yue(), &UtrEngine::RevAi).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("requires utterance timing recovery"),
            "expected stage-aware UTR error, got: {err}"
        );
        assert!(
            message.contains(UtrEngine::Whisper.selection_name()),
            "local Whisper does support yue and must be offered: {err}"
        );
    }

    #[test]
    fn utr_runtime_validation_allows_whisper_for_yue() {
        assert!(validate_utr_language_support(&LanguageCode3::yue(), &UtrEngine::Whisper).is_ok());
    }
}

#[cfg(test)]
mod fa_language_support_tests {
    use super::*;
    use crate::types::engines::LanguageSupport;

    /// Only the primary declared, the ordinary case.
    fn only(code: &str) -> DeclaredLanguages {
        DeclaredLanguages::from_header([code], None).expect("a readable header")
    }

    /// `--lang eng`, the fallback the align dispatch passes.
    fn eng_fallback() -> LanguageCode3 {
        LanguageCode3::try_new("eng").expect("a valid ISO-639-3 code")
    }

    /// Run the admission for one engine.
    ///
    /// The tests care about the engine and the declaration; the rest of
    /// [`FaParams`] is whatever the align dispatch would have built, and none
    /// of it reaches the check. Written out rather than reached for through a
    /// `Default`, which `FaParams` deliberately does not have: an engine is
    /// exactly the kind of fact a default would fabricate.
    fn admit(
        declared: DeclaredLanguages,
        engine: FaEngineName,
    ) -> Result<AdmittedFaParams, ValidationError> {
        validate_fa_language_support(
            declared,
            FaParams {
                gap_healing: crate::chat_ops::fa::WordGapHealing::Heal,
                existing_wor_boundaries: crate::chat_ops::fa::ExistingWorBoundaryPolicy::Preserve,
                end_overlap_policy: crate::chat_ops::fa::EndOverlapPolicy::ClampAllAdjacent,
                engine,
                cache_policy: crate::types::params::CachePolicy::UseCache,
                wor_tier: crate::types::params::WorTierPolicy::Include,
                bullet_repair: true,
                review_level: crate::chat_ops::fa::ReviewLevel::None,
            },
        )
    }

    /// An unparseable FIRST entry still decides admission.
    ///
    /// THE CALLER'S resolution, which is why it is exercised through
    /// `from_header` rather than through a pre-resolved primary: the align
    /// dispatch used to resolve the primary itself, falling back to `--lang`
    /// when the first header entry would not parse, and then hand `skip(1)` of
    /// the header over as the secondaries. The entry it could not read
    /// vanished. `@Languages: not-a-code, eng` with `--lang eng` was ADMITTED
    /// while `@Languages: eng, not-a-code` was refused, for identical content,
    /// which is the order-decides-admission defect this type exists to remove.
    #[test]
    fn an_unreadable_first_entry_still_decides_admission() {
        let fallback = eng_fallback();
        let unreadable_first =
            DeclaredLanguages::from_header(["not-a-code", "eng"], Some(&fallback))
                .expect("--lang supplies the primary");
        let unreadable_second =
            DeclaredLanguages::from_header(["eng", "not-a-code"], Some(&fallback))
                .expect("the header supplies the primary");
        let first = admit(unreadable_first, FaEngineName::Qwen3)
            .expect_err("an unreadable entry cannot be shown supported wherever it sits");
        let second = admit(unreadable_second, FaEngineName::Qwen3)
            .expect_err("an unreadable entry cannot be shown supported wherever it sits");
        assert_eq!(first.to_string(), second.to_string());
        assert!(
            first.to_string().contains("not-a-code"),
            "the refusal must quote the entry it could not read: {first}"
        );
    }

    /// The primary the caller goes on to use is the one the declaration holds.
    ///
    /// There is no second resolution to disagree with: `file_lang` is read off
    /// the same value that was validated, so a fallback used for the primary
    /// cannot silently differ from the primary that was checked.
    #[test]
    fn the_resolved_primary_is_the_one_the_declaration_carries() {
        let fallback = eng_fallback();
        let from_header =
            DeclaredLanguages::from_header(["yue", "eng"], Some(&fallback)).expect("readable");
        assert_eq!(
            from_header.primary(),
            &LanguageCode3::try_new("yue").unwrap()
        );
        let from_fallback =
            DeclaredLanguages::from_header(Vec::<&str>::new(), Some(&fallback)).expect("readable");
        assert_eq!(from_fallback.primary(), &fallback);
    }

    /// With no header and no `--lang`, there is nothing to resolve from.
    #[test]
    fn a_header_and_fallback_that_are_both_absent_are_refused_apart() {
        assert!(matches!(
            DeclaredLanguages::from_header(Vec::<&str>::new(), None),
            Err(HeaderLanguageError::NoHeader)
        ));
        assert!(matches!(
            DeclaredLanguages::from_header(["not-a-code"], None),
            Err(HeaderLanguageError::UnreadablePrimary { .. })
        ));
    }

    /// The Qwen3 aligner is refused for a language it has no label for, by
    /// name, and never by quietly aligning with something else.
    ///
    /// POLICY, so it stays a test: falling back to a language-general engine
    /// would produce a perfectly plausible `%wor` tier from a model the
    /// operator did not choose.
    #[test]
    fn qwen3_fa_refuses_an_unmapped_language_by_name() {
        let error = admit(only("deu"), FaEngineName::Qwen3)
            .expect_err("the Qwen3 aligner has no German label wired");
        let message = error.to_string();
        assert!(
            message.contains(FaEngineName::Qwen3.selection_name()),
            "the refusal must name the engine that refused: {message}"
        );
        assert!(
            message.contains("deu"),
            "the refusal must name the language: {message}"
        );
    }

    /// The refusal RENDERS exactly this, whitespace included.
    ///
    /// A WIRE-ish format: it is what an operator reads in a failed file's
    /// status, so it is a measurement of the rendered string rather than an
    /// invariant a type can hold. Pinned exactly because every other test here
    /// used `contains`, which is how the message shipped carrying a ten-space
    /// run before `'{lang}'`.
    #[test]
    fn the_unsupported_language_refusal_renders_exactly() {
        let message = admit(only("deu"), FaEngineName::Qwen3)
            .expect_err("German is unsupported")
            .to_string();
        assert_eq!(
            message,
            "The selected forced-alignment engine 'qwen3_fa' does not support language 'deu'. \
             Use --fa-engine with one of: wav2vec, whisper, cantonese."
        );
    }

    /// The remedy names engines that actually work for THIS language, derived
    /// from the same predicate that produced the rejection.
    #[test]
    fn a_rejection_suggests_only_engines_that_support_the_language() {
        let message = admit(only("deu"), FaEngineName::Qwen3)
            .expect_err("German is unsupported")
            .to_string();
        for engine in FaEngineName::ALL.iter().copied() {
            if engine == FaEngineName::Qwen3 {
                continue;
            }
            assert!(
                message.contains(engine.selection_name()),
                "{} supports deu and must be offered: {message}",
                engine.selection_name()
            );
        }
    }

    /// Every language the Python label map knows is admitted here.
    ///
    /// The two lists live in different languages and cannot be one value, so
    /// this pins our half; the Python half raises loudly on anything it cannot
    /// map, which is what catches a drift in the other direction.
    #[test]
    fn qwen3_fa_admits_every_language_its_label_map_covers() {
        let codes = match FaEngineName::Qwen3.language_support() {
            LanguageSupport::Only(codes) => codes,
            // A language-general aligner admits everything, so the loop below
            // would pass vacuously rather than pinning the label map.
            LanguageSupport::Any => &[][..],
        };
        assert!(
            !codes.is_empty(),
            "the Qwen3 row must name the codes its Python label map covers"
        );
        for code in codes {
            assert!(
                admit(only(code), FaEngineName::Qwen3).is_ok(),
                "{code} is in the label map and must be admitted"
            );
        }
    }

    /// A SECONDARY declared language the engine cannot handle is refused too.
    ///
    /// POLICY, and the one this admission got wrong: `@Languages: eng, deu`
    /// used to be admitted because only the first entry was read, so every
    /// `[- deu]` utterance was aligned under the English label; the same
    /// content written `deu, eng` was refused.
    #[test]
    fn a_secondary_declared_language_is_admitted_on_its_own_merits() {
        let declared = DeclaredLanguages::from_header(["eng", "deu"], None).expect("readable");
        let message = admit(declared, FaEngineName::Qwen3)
            .expect_err("German is unsupported wherever it is declared")
            .to_string();
        assert!(
            message.contains("deu") && message.contains("secondary"),
            "the refusal must name the secondary language and say it is one: {message}"
        );
    }

    /// The PROOF carries the same primary the declaration resolved, whichever
    /// order the header declares its languages in.
    ///
    /// The proof is what the align dispatch now reads `file_lang` off, so this
    /// is the order question asked of the new value rather than of the old
    /// one: an admission that returned, say, the first SUPPORTED entry would
    /// silently stamp `@Languages:` with a language the file does not lead
    /// with. Both codes here are in the Qwen3 label map, so admission succeeds
    /// either way and only the primary can differ.
    #[test]
    fn the_proof_carries_the_headers_own_primary_in_either_order() {
        let yue_first = admit(
            DeclaredLanguages::from_header(["yue", "eng"], None).expect("readable"),
            FaEngineName::Qwen3,
        )
        .expect("both codes are mapped");
        assert_eq!(yue_first.primary_language().as_ref(), "yue");

        let eng_first = admit(
            DeclaredLanguages::from_header(["eng", "yue"], None).expect("readable"),
            FaEngineName::Qwen3,
        )
        .expect("both codes are mapped");
        assert_eq!(eng_first.primary_language().as_ref(), "eng");
    }

    /// The proof carries the engine that was actually checked.
    ///
    /// `params()` is what the align dispatch dispatches on, so an admission
    /// that checked one engine and handed back another would be the exact
    /// drift the proof exists to prevent, and no signature says the two are
    /// the same value.
    #[test]
    fn the_proof_carries_the_engine_that_was_checked() {
        let admitted =
            admit(only("yue"), FaEngineName::Qwen3).expect("Cantonese is in the label map");
        assert_eq!(admitted.params().engine, FaEngineName::Qwen3);

        let general = admit(only("deu"), FaEngineName::Wave2Vec)
            .expect("a language-general engine admits anything");
        assert_eq!(general.params().engine, FaEngineName::Wave2Vec);
    }

    /// Header ORDER is not what decides admission.
    #[test]
    fn header_order_does_not_decide_admission() {
        let eng_first = DeclaredLanguages::from_header(["eng", "deu"], None).expect("readable");
        let deu_first = DeclaredLanguages::from_header(["deu", "eng"], None).expect("readable");
        assert!(admit(eng_first, FaEngineName::Qwen3).is_err());
        assert!(admit(deu_first, FaEngineName::Qwen3).is_err());
    }

    /// A language-general engine reads no part of the declaration, so a
    /// secondary it cannot parse cannot make it wrong.
    #[test]
    fn a_language_general_engine_ignores_an_unparseable_secondary() {
        let declared =
            DeclaredLanguages::from_header(["eng", "not-a-code"], None).expect("readable");
        assert!(admit(declared.clone(), FaEngineName::Wave2Vec).is_ok());
        let message = admit(declared, FaEngineName::Qwen3)
            .expect_err("Qwen3 needs a label for every declared language")
            .to_string();
        assert!(
            message.contains("not-a-code"),
            "the refusal must quote the entry it could not read: {message}"
        );
    }
}

#[cfg(test)]
mod utr_language_support_tests {
    use super::*;

    /// The remedy names engines that actually work for THIS language, derived
    /// from the same predicate that rejected the selected one.
    ///
    /// Behaviour a signature cannot describe, so it stays a test: the point is
    /// that the message and the verdict cannot disagree, which is a property of
    /// how they are computed rather than of their types.
    #[test]
    fn a_rejection_suggests_only_engines_that_support_the_language() {
        let error = validate_utr_language_support(&LanguageCode3::yue(), &UtrEngine::RevAi)
            .expect_err("Rev.AI does not support Cantonese");
        let message = error.to_string();
        for engine in UtrEngine::ALL.iter().cloned() {
            if engine == UtrEngine::RevAi {
                continue;
            }
            let named = message.contains(engine.selection_name());
            assert_eq!(
                named,
                utr_engine_supports_language(&engine, &LanguageCode3::yue()),
                "{} should be suggested iff it supports yue: {message}",
                engine.selection_name()
            );
        }
    }

    /// The message must not ADVERTISE the deprecated flag.
    ///
    /// `--utr-engine-custom` still exists and still works; it is hidden from
    /// help on purpose. The defect was that an error message sent users to it,
    /// which is how it stayed the documented route long after `--utr-engine`
    /// could do the job.
    #[test]
    fn a_rejection_does_not_advertise_the_deprecated_flag() {
        let error = validate_utr_language_support(&LanguageCode3::yue(), &UtrEngine::RevAi)
            .expect_err("Rev.AI does not support Cantonese");
        let message = error.to_string();
        assert!(
            !message.contains("--utr-engine-custom"),
            "the deprecated flag must not be advertised: {message}"
        );
        assert!(
            message.contains("--utr-engine"),
            "the surviving flag must be named: {message}"
        );
    }
}
