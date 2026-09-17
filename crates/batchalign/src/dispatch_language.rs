//! Where a command's processing language comes from, decided once.
//!
//! Two dispatch shapes exist and they are not interchangeable:
//!
//! - **Per-file.** The command has no `--lang` at all. Each input file's
//!   processing language is read from its own `@Languages:` header (morphotag,
//!   translate) or is a constant the command owns (coref, which is
//!   English-only). A job-level language would be a placeholder, and the
//!   2026-05-03 incident is what a placeholder costs.
//! - **Job-level.** The command carries one resolved ISO code for every file
//!   (utseg, compare).
//!
//! Before this module the two shapes were decided by a runtime check at
//! dispatch: each dispatcher asked `job.dispatch.lang.as_resolved()` and
//! returned a typed error when the answer was `None`. For coref that check
//! could never pass. Coref is a per-file command, so submission validation
//! REQUIRES it to arrive as [`LanguageSpec::PerFile`], and `as_resolved()` on
//! `PerFile` is always `None`: every coref job was refused at dispatch with a
//! message about a `--lang` flag coref does not have, and the language was
//! then ignored anyway because coref hardcodes English.
//!
//! [`DispatchLanguage`] makes the shape a fact about the command rather than a
//! question asked of a value. A dispatcher that needs a code takes
//! [`JobLanguage`] and cannot be handed a per-file job; a per-file dispatcher
//! takes no language at all and cannot demand one.

use crate::api::{LanguageCode3, LanguageSpec, ReleasedCommand};

/// Where one command's processing language comes from.
///
/// Declared per command below, with no catch-all arm, so a new released
/// command cannot compile until it states which shape it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandLanguageSource {
    /// No job-level language: the file, or a constant the command owns.
    PerFile,
    /// A job-level language: one resolved code, `auto`, or, for transcription
    /// only, a code-switched pair (see [`language_pair_support`]). Commands
    /// dispatched through [`DispatchLanguage`] need exactly one code.
    JobLevel,
}

/// The one owner of the per-file/job-level question.
///
/// This predicate used to be written out twice: once in submission validation
/// (`types::request::JobSubmission::validate_lang_command_pairing`) as a
/// `matches!` over three command names, and once implicitly in each
/// dispatcher's `as_resolved()` check. Two spellings of one rule is how the
/// rule came to disagree with itself, which is what made coref undispatchable.
pub(crate) const fn language_source(command: ReleasedCommand) -> CommandLanguageSource {
    match command {
        // No `--lang` on the CLI. Morphotag and translate resolve per file
        // from `@Languages:`; coref is English-only and owns the constant.
        ReleasedCommand::Morphotag | ReleasedCommand::Translate | ReleasedCommand::Coref => {
            CommandLanguageSource::PerFile
        }
        // Everything else carries a real job-level language (or `auto`, which
        // ASR resolves before any language-bearing stage runs, or, for
        // transcription, a pair; see `language_pair_support`).
        ReleasedCommand::Align
        | ReleasedCommand::Transcribe
        | ReleasedCommand::TranscribeS
        | ReleasedCommand::Utseg
        | ReleasedCommand::Benchmark
        | ReleasedCommand::Opensmile
        | ReleasedCommand::Compare
        | ReleasedCommand::Avqi
        | ReleasedCommand::Diarize
        | ReleasedCommand::SpeakerIdentify => CommandLanguageSource::JobLevel,
    }
}

/// Whether one command accepts a code-switched language pair such as `eng,spa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LanguagePairSupport {
    /// The command is the one that could recognize both languages of a pair,
    /// and withholds it: the only pair model available, Rev.AI's `en/es`, was
    /// measured on 2026-09-17 writing fluent English where Spanish was spoken,
    /// with real timestamps, on one bilingual corpus recording (several stretches; real
    /// Spanish in others of the same run) and on one public clip on
    /// 2026-09-16. Translation presented as transcription is a value the file
    /// cannot reveal as fabricated, so no transcript is produced from the pair
    /// until a route exists whose Spanish words are recognized by a Spanish
    /// model (two single-language passes merged by stretch, the next build).
    /// Engine admission refuses a pair for the same reason, so a job saved
    /// under an earlier build cannot plan one on restart. The pair's request
    /// type, header writing and evidence-cache keys stay: the routed merge
    /// declares the pair in its header, and evidence already recorded under
    /// pair keys must stay readable. Nothing measures the pair today;
    /// unwithholding needs a measurement route that does not yet exist.
    Withheld,
    /// The command runs under one language, or none.
    Refused,
}

impl LanguagePairSupport {
    /// The refusal a withheld pair gets at submission: the measurement, and
    /// what to run instead. One text, so the CLI and the API say the same.
    pub(crate) fn withheld_message(pair: &crate::api::LanguagePair) -> String {
        format!(
            "the language pair '{pair}' is withheld from transcription: the only pair model, \
             Rev.AI's en/es, was measured writing fluent English where Spanish was spoken \
             (translation presented as transcription, with real timestamps, which nothing \
             downstream can tell from speech; 2026-09-17, one recording, and a public clip \
             on 2026-09-16). Run the recording under each language alone (`--lang eng`; \
             `--lang spa` with `--utseg-fallback-stanza`) until the routed two-pass merge \
             lands; `--lang auto` tags utterances by language but recognizes with one model."
        )
    }
}

/// The one owner of which commands take a language pair.
///
/// Only transcription could: a pair describes a recording, and transcription is
/// what recognizes one; it withholds the pair today, for the measured reason on
/// the variant. No catch-all arm, so a new command states its answer.
pub(crate) const fn language_pair_support(command: ReleasedCommand) -> LanguagePairSupport {
    match command {
        ReleasedCommand::Transcribe | ReleasedCommand::TranscribeS => LanguagePairSupport::Withheld,
        ReleasedCommand::Morphotag
        | ReleasedCommand::Translate
        | ReleasedCommand::Coref
        | ReleasedCommand::Align
        | ReleasedCommand::Utseg
        | ReleasedCommand::Benchmark
        | ReleasedCommand::Opensmile
        | ReleasedCommand::Compare
        | ReleasedCommand::Avqi
        | ReleasedCommand::Diarize
        | ReleasedCommand::SpeakerIdentify => LanguagePairSupport::Refused,
    }
}

/// One job's resolved processing language.
///
/// The inner code is private to this module, so the ONLY way to obtain a
/// `JobLanguage` is [`DispatchLanguage::resolve`], which refuses to mint one
/// for a per-file command. A dispatcher taking this type therefore has a
/// compiler-checked proof that its command has a job-level language, rather
/// than a runtime check it could forget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JobLanguage(LanguageCode3);

impl JobLanguage {
    /// Borrow the resolved ISO 639-3 code.
    pub(crate) fn code(&self) -> &LanguageCode3 {
        &self.0
    }
}

/// The dispatch shape for one job, derived from its command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DispatchLanguage {
    /// The command has no job-level language; the file or the command decides.
    PerFile,
    /// The command carries one resolved language for every file.
    Job(JobLanguage),
}

/// Why a submission's language could not be read as a dispatch shape.
///
/// Both variants mean the submission is malformed for its command, not that
/// the user asked for something unsupported: submission validation refuses
/// each shape at the wire boundary, so reaching one of these means something
/// downstream of the CLI built a job by hand.
#[derive(Debug, Clone, thiserror::Error)]
pub(crate) enum DispatchLanguageError {
    /// A per-file command arrived carrying a job-level language.
    #[error(
        "command '{command}' has no --lang and must dispatch per file, but the job carries \
         language '{lang}'"
    )]
    UnexpectedJobLanguage {
        /// The per-file command.
        command: ReleasedCommand,
        /// What the job carried instead of `per-file`.
        lang: LanguageSpec,
    },
    /// A job-level command arrived without a resolved language.
    #[error(
        "command '{command}' requires a resolved `--lang <iso3>`; got '{lang}'. Re-submit with \
         an explicit language (for example `--lang eng`)."
    )]
    MissingJobLanguage {
        /// The job-level command.
        command: ReleasedCommand,
        /// The unusable language specification.
        lang: LanguageSpec,
    },
}

impl DispatchLanguage {
    /// Derive the dispatch shape for one job.
    ///
    /// The ONE constructor. It reads the shape from the command and then
    /// checks the submitted `LanguageSpec` against it, so the two can never be
    /// paired the wrong way round downstream.
    pub(crate) fn resolve(
        command: ReleasedCommand,
        lang: &LanguageSpec,
    ) -> Result<Self, DispatchLanguageError> {
        match language_source(command) {
            CommandLanguageSource::PerFile => match lang {
                LanguageSpec::PerFile => Ok(Self::PerFile),
                LanguageSpec::Auto | LanguageSpec::Resolved(_) | LanguageSpec::Pair(_) => {
                    Err(DispatchLanguageError::UnexpectedJobLanguage {
                        command,
                        lang: lang.clone(),
                    })
                }
            },
            CommandLanguageSource::JobLevel => match lang {
                LanguageSpec::Resolved(code) => Ok(Self::Job(JobLanguage(code.clone()))),
                LanguageSpec::Auto | LanguageSpec::Pair(_) | LanguageSpec::PerFile => {
                    Err(DispatchLanguageError::MissingJobLanguage {
                        command,
                        lang: lang.clone(),
                    })
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this module exists to remove: coref is a per-file command,
    /// so it resolves to the per-file shape rather than being refused for
    /// lacking a `--lang` it does not have.
    #[test]
    fn coref_resolves_to_the_per_file_shape() {
        assert_eq!(
            DispatchLanguage::resolve(ReleasedCommand::Coref, &LanguageSpec::PerFile)
                .expect("coref submits PerFile and must dispatch"),
            DispatchLanguage::PerFile
        );
    }

    /// A job-level command still gets its code, and the code is reachable only
    /// through the proof type.
    #[test]
    fn utseg_resolves_to_its_job_language() {
        let resolved = DispatchLanguage::resolve(
            ReleasedCommand::Utseg,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
        )
        .expect("utseg submits a resolved language");
        let DispatchLanguage::Job(job_language) = resolved else {
            panic!("utseg is a job-level command");
        };
        assert_eq!(job_language.code(), &LanguageCode3::eng());
    }

    /// A job-level command with no resolved language is refused, and the
    /// message names the remedy that command actually has.
    #[test]
    fn a_job_level_command_without_a_language_is_refused() {
        let error = DispatchLanguage::resolve(ReleasedCommand::Utseg, &LanguageSpec::PerFile)
            .expect_err("utseg has no per-file shape");
        assert!(error.to_string().contains("--lang"), "{error}");
    }

    /// A per-file command carrying a job-level language is refused rather than
    /// silently using it: that value is the placeholder the 2026-05-03
    /// incident was about.
    #[test]
    fn a_per_file_command_carrying_a_job_language_is_refused() {
        let error = DispatchLanguage::resolve(
            ReleasedCommand::Morphotag,
            &LanguageSpec::Resolved(LanguageCode3::eng()),
        )
        .expect_err("morphotag has no job-level language");
        assert!(error.to_string().contains("per file"), "{error}");
    }

    /// Every released command states its shape, and the three per-file
    /// commands are exactly the three with no `--lang` on the CLI.
    #[test]
    fn exactly_the_no_lang_commands_are_per_file() {
        for command in ReleasedCommand::ALL {
            let expected = match command {
                ReleasedCommand::Morphotag
                | ReleasedCommand::Translate
                | ReleasedCommand::Coref => CommandLanguageSource::PerFile,
                ReleasedCommand::Align
                | ReleasedCommand::Transcribe
                | ReleasedCommand::TranscribeS
                | ReleasedCommand::Utseg
                | ReleasedCommand::Benchmark
                | ReleasedCommand::Opensmile
                | ReleasedCommand::Compare
                | ReleasedCommand::Avqi
                | ReleasedCommand::Diarize
                | ReleasedCommand::SpeakerIdentify => CommandLanguageSource::JobLevel,
            };
            assert_eq!(language_source(command), expected, "{command}");
        }
    }
}
