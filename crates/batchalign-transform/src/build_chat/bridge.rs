use talkbank_model::model::{ChatFile, LanguageCode};

use crate::asr_postprocess;

use super::{ParticipantDesc, TranscriptDescription, UtteranceDesc, WordDesc};

/// Build a CHAT file from a JSON transcript description string.
///
/// This is the entry point used by the PyO3 bridge (`build_chat_inner`).
pub fn build_chat_from_json(json: &str) -> Result<ChatFile, String> {
    let desc: TranscriptDescription =
        serde_json::from_str(json).map_err(|e| format!("Invalid JSON: {e}"))?;
    // The PyO3 edge speaks strings; the typed error is flattened HERE, at the
    // boundary, rather than by everything upstream of it.
    super::build_chat(&desc).map_err(|e| e.to_string())
}

/// Domain errors from building a `TranscriptDescription`.
///
/// Exposes structured failure information, the offending word's
/// position, text, declared language, and the full
/// `Vec<talkbank_model::ParseError>` from `ChatWordText::try_from_lang`
///, so upstream callers can render diagnostics or branch on failure
/// class without re-parsing a string.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptBuildError {
    /// Explicitly requested diagnostic evidence could not be persisted.
    #[error(transparent)]
    Diagnostic(#[from] AsrDiagnosticError),
    /// Explicit participant names do not cover an observed speaker.
    #[error("no participant code supplied for ASR speaker {0:?}")]
    MissingParticipantCode(asr_postprocess::SpeakerIndex),
    /// No primary language was supplied for word admission.
    #[error("ASR transcript requires a declared primary language")]
    MissingPrimaryLanguage,
    /// A word failed CHAT-legality validation under its utterance's
    /// language. Normalization upstream in `process_raw_asr` should
    /// have rewritten reporter-class tokens (`%`, digit-hyphen compounds)
    /// before this gate; any failure surfacing here is a residual case
    /// the normalizer hasn't been taught yet.
    #[error(
        "word {word_idx} ({word_text:?}) in utterance {utt_idx} \
         (speaker *{speaker_id}:, lang {lang}) failed CHAT validation: {}",
        parse_errors.iter()
            .map(|e| format!("[{}] {}", e.code.as_str(), e.message))
            .collect::<Vec<_>>()
            .join("; ")
    )]
    WordFailedValidation {
        /// Zero-based index of the utterance containing the bad word.
        utt_idx: usize,
        /// Zero-based index of the word within its utterance.
        word_idx: usize,
        /// Speaker code for the enclosing utterance (e.g. `"PAR0"`).
        speaker_id: String,
        /// Original ASR token text (before any attempted normalization).
        word_text: String,
        /// ISO 639-3 language code the word was validated under.
        lang: String,
        /// Structured parse/validation errors from
        /// [`ChatWordText::try_from_lang`].
        parse_errors: Vec<talkbank_model::ParseError>,
    },

    /// A language code supplied to the bridge (transcript-level or
    /// per-utterance) is not a valid CHAT language code. chatter 0.3.x
    /// made [`LanguageCode`] construction fallible; the bridge parses
    /// every code at this boundary so downstream code only sees typed
    /// values. (chatter v0.3.1 re-exports the error type, so the source
    /// is typed.)
    #[error("invalid language code {lang:?}")]
    InvalidLanguageCode {
        /// The offending raw code as supplied by the caller.
        lang: String,
        /// The upstream construction error.
        #[source]
        source: talkbank_model::model::LanguageCodeError,
    },
}

/// Failure to persist explicitly requested ASR diagnostic evidence.
#[derive(Debug, thiserror::Error)]
pub enum AsrDiagnosticError {
    /// The source utterances could not be encoded.
    #[error("could not encode ASR utterance diagnostics: {0}")]
    Encode(#[from] serde_json::Error),
    /// The requested destination could not be written.
    #[error("could not write ASR utterance diagnostics to {path:?}: {source}")]
    Write {
        /// Requested destination, preserved without lossy path conversion.
        path: std::path::PathBuf,
        /// Original filesystem failure.
        #[source]
        source: std::io::Error,
    },
}

fn write_utterance_dump(
    path: &std::path::Path,
    utterances: &[asr_postprocess::Utterance],
) -> Result<(), AsrDiagnosticError> {
    let json = serde_json::to_vec_pretty(utterances)?;
    std::fs::write(path, json).map_err(|source| AsrDiagnosticError::Write {
        path: path.to_owned(),
        source,
    })?;
    tracing::warn!(path = %path.display(), "BA3_DUMP_UTTERANCES wrote post-processed utterances");
    Ok(())
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn diagnostic_write_reports_real_filesystem_failure() {
        let path = std::path::Path::new("");
        assert!(
            matches!(write_utterance_dump(path, &[]), Err(AsrDiagnosticError::Write { path: failed, .. }) if failed == path)
        );
    }

    #[test]
    fn diagnostic_write_success_has_readable_bytes() {
        let destination = tempfile::NamedTempFile::new().unwrap();
        write_utterance_dump(destination.path(), &[]).unwrap();
        assert_eq!(std::fs::read(destination.path()).unwrap(), b"[]");
    }
}

/// A word the language gate refused, emitted anyway for human review.
///
/// This type exists because the refusal used to go to `tracing::warn!` and
/// nowhere else. Shape C in the workspace's list, in its commonest disguise: a
/// return type too weak to hold what the stage learned sends the fact to a log
/// line, where it looks handled. `WordDesc` cannot say "this one is unproved",
/// so the fact travels beside the description instead.
///
/// Emitting the token verbatim is a deliberate, unchanged POLICY: the surface
/// is the provider's observation and what it should have been is a human's
/// call, not the pipeline's. Reporting it is the other half of that policy,
/// which was never built.
#[derive(Debug, Clone)]
pub struct LanguageInvalidWord {
    /// Zero-based index of the utterance the word is in.
    pub utt_idx: usize,
    /// Zero-based index of the word within its utterance.
    pub word_idx: usize,
    /// Speaker code for the enclosing utterance (e.g. `"PAR1"`).
    pub speaker_id: String,
    /// The provider's surface, verbatim, exactly as emitted.
    pub text: String,
    /// ISO 639-3 language the word was judged under.
    pub lang: String,
    /// Why the language gate refused it (E220 digits, E241 reserved
    /// untranscribed marker, and anything else `Word::validate` reports).
    pub parse_errors: Vec<talkbank_model::ParseError>,
}

/// A transcript description together with every word that reached it unproved.
///
/// `#[must_use]`, and the two fields are deliberately not collapsed into one:
/// a caller that wants only the description has to say so, in a line a reader
/// can see, rather than by never being offered the rest.
#[must_use]
#[derive(Debug, Clone)]
pub struct AsrTranscript {
    /// The pre-serialization transcript, ready for `build_chat`.
    pub description: TranscriptDescription,
    /// Words emitted despite failing language-level validation, in emission
    /// order. Empty means every word carried a full language-level proof.
    pub language_invalid: Vec<LanguageInvalidWord>,
}

struct NamedAsrUtterance<'a> {
    utterance: &'a asr_postprocess::Utterance,
    speaker_id: String,
}

/// Speaker names bound to their immutable source utterances. Numbered naming
/// allocates only for observed utterances, never for a requested count or the
/// largest numeric speaker label. Explicit names must cover every speaker.
pub struct NamedAsrUtterances<'a> {
    source: &'a [asr_postprocess::Utterance],
    named: Vec<NamedAsrUtterance<'a>>,
}

impl<'a> NamedAsrUtterances<'a> {
    /// Assign neutral PAR codes while preserving provider speaker indices.
    pub fn numbered(source: &'a [asr_postprocess::Utterance]) -> Self {
        Self {
            source,
            named: source
                .iter()
                .map(|utterance| NamedAsrUtterance {
                    utterance,
                    speaker_id: format!("PAR{}", utterance.speaker.as_usize()),
                })
                .collect(),
        }
    }

    /// Admit explicit codes for this source; missing codes are not invented.
    pub fn with_participant_ids(
        source: &'a [asr_postprocess::Utterance],
        ids: &[String],
    ) -> Result<Self, TranscriptBuildError> {
        let named = source
            .iter()
            .map(|utterance| {
                let speaker_id = ids
                    .get(utterance.speaker.as_usize())
                    .ok_or(TranscriptBuildError::MissingParticipantCode(
                        utterance.speaker,
                    ))?
                    .clone();
                Ok(NamedAsrUtterance {
                    utterance,
                    speaker_id,
                })
            })
            .collect::<Result<_, TranscriptBuildError>>()?;
        Ok(Self { source, named })
    }

    /// Build from the exact utterances admitted with these names.
    pub fn into_transcript(
        self,
        langs: &[String],
        media_name: Option<&str>,
        write_wor: bool,
    ) -> Result<AsrTranscript, TranscriptBuildError> {
        build_named_asr_transcript(self, langs, media_name, write_wor)
    }
}

/// Convert post-processed ASR utterances into a pre-serialization
/// `TranscriptDescription`.
///
/// Each word's text is validated at construction via
/// [`ChatWordText::try_from_lang`][try_lang] under the utterance's declared
/// language (or the required primary `langs[0]`). Fails
/// with [`TranscriptBuildError`] at the first offending word. This is
/// the "loud guard" half of strategy 4c: normalization runs upstream
/// in `process_raw_asr`'s stages; this gate is the belt after the
/// braces.
///
/// A word that is structurally legal but fails the LANGUAGE-level rules is
/// still emitted verbatim, deliberately, and is now also returned in
/// [`AsrTranscript::language_invalid`] so a caller can act on it.
///
/// [try_lang]: asr_postprocess::ChatWordText::try_from_lang
pub fn transcript_from_asr_utterances(
    utterances: &[asr_postprocess::Utterance],
    participant_ids: &[String],
    langs: &[String],
    media_name: Option<&str>,
    write_wor: bool,
) -> Result<AsrTranscript, TranscriptBuildError> {
    NamedAsrUtterances::with_participant_ids(utterances, participant_ids)?
        .into_transcript(langs, media_name, write_wor)
}

fn build_named_asr_transcript(
    input: NamedAsrUtterances<'_>,
    langs: &[String],
    media_name: Option<&str>,
    write_wor: bool,
) -> Result<AsrTranscript, TranscriptBuildError> {
    let utterances = input.source;
    if let Some(path) = std::env::var_os("BA3_DUMP_UTTERANCES") {
        write_utterance_dump(std::path::Path::new(&path), utterances)?;
    }

    let participants = build_asr_participants(&input.named);
    let primary_lang_raw = langs
        .first()
        .ok_or(TranscriptBuildError::MissingPrimaryLanguage)?;
    let primary_lang_code = LanguageCode::new(primary_lang_raw).map_err(|source| {
        TranscriptBuildError::InvalidLanguageCode {
            lang: primary_lang_raw.to_string(),
            source,
        }
    })?;

    let mut utterance_descs = Vec::with_capacity(utterances.len());
    let mut language_invalid: Vec<LanguageInvalidWord> = Vec::new();
    for (utt_idx, named) in input.named.into_iter().enumerate() {
        let utterance = named.utterance;
        let speaker_id = named.speaker_id;
        let utterance_lang = match utterance.lang.as_deref() {
            Some(raw) => LanguageCode::new(raw).map_err(|source| {
                TranscriptBuildError::InvalidLanguageCode {
                    lang: raw.to_string(),
                    source,
                }
            })?,
            None => primary_lang_code.clone(),
        };

        let mut words = Vec::with_capacity(utterance.words.len());
        for (word_idx, word) in utterance.words.iter().enumerate() {
            let admitted =
                validate_asr_word(word, &speaker_id, &utterance_lang, utt_idx, word_idx)?;
            // Exhaustive: a future admission outcome must state what the
            // caller owes it rather than falling through this match.
            match admitted {
                WordAdmission::LanguageProved(desc) => words.push(desc),
                WordAdmission::StructuralOnly { desc, refusal } => {
                    words.push(desc);
                    language_invalid.push(refusal);
                }
            }
        }

        utterance_descs.push(UtteranceDesc {
            speaker: speaker_id,
            words: Some(words),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: utterance.lang.clone(),
        });
    }

    Ok(AsrTranscript {
        description: TranscriptDescription {
            langs: if langs.is_empty() {
                vec!["eng".to_string()]
            } else {
                langs.to_vec()
            },
            participants,
            media_name: media_name.map(String::from),
            media_type: Some("audio".to_string()),
            media_status: None,
            utterances: utterance_descs,
            write_wor,
        },
        language_invalid,
    })
}

fn build_asr_participants(utterances: &[NamedAsrUtterance<'_>]) -> Vec<ParticipantDesc> {
    let mut seen_speakers = std::collections::BTreeMap::new();
    for named in utterances {
        seen_speakers
            .entry(named.utterance.speaker)
            .or_insert(&named.speaker_id);
    }

    seen_speakers
        .into_values()
        .map(|id| {
            let (_name, role) = role_for_speaker_code(&id);
            ParticipantDesc {
                id: id.clone(),
                name: None,
                role,
                corpus: String::new(),
            }
        })
        .collect()
}

/// How much proof one ASR token carries into the transcript.
///
/// Two admissions, and they are NOT the same value: the first is a word the
/// language rules accept, the second is a word they refuse that we emit
/// anyway. Before 2026-09-03 both arrived at the caller as a bare
/// `ChatWordText` and the difference lived only in a log line, which is why
/// two invalid tokens could reach a corpus with nothing recording that the
/// gate had already caught them.
enum WordAdmission {
    /// Passed [`ChatWordText::try_from_lang`][try_lang] under the utterance's
    /// language: structurally a word AND legal in that language.
    ///
    /// [try_lang]: asr_postprocess::ChatWordText::try_from_lang
    LanguageProved(WordDesc),
    /// Parses as a word but breaks a language-level rule (E220 digits, E241
    /// reserved untranscribed marker, ...). Emitted verbatim by policy, with
    /// the refusal attached so the caller has to decide what to do about it.
    StructuralOnly {
        /// The word as it will be emitted: the provider's surface, unchanged.
        desc: WordDesc,
        /// What the language gate refused, and why.
        refusal: LanguageInvalidWord,
    },
}

fn validate_asr_word(
    word: &asr_postprocess::AsrWord,
    speaker_id: &str,
    utterance_lang: &LanguageCode,
    utt_idx: usize,
    word_idx: usize,
) -> Result<WordAdmission, TranscriptBuildError> {
    let describe = |text: asr_postprocess::ChatWordText| WordDesc {
        text,
        start_ms: word.start_ms.map(|ms| ms as u64),
        end_ms: word.end_ms.map(|ms| ms as u64),
        kind: word.kind,
    };

    let lang_errors =
        match asr_postprocess::ChatWordText::try_from_lang(word.text.as_str(), utterance_lang) {
            Ok(text) => return Ok(WordAdmission::LanguageProved(describe(text))),
            Err(lang_errors) => lang_errors,
        };

    // Structural legality is a genuinely weaker claim, so it produces a
    // genuinely different admission rather than the same `ChatWordText` the
    // proved path returns.
    match asr_postprocess::ChatWordText::try_from(word.text.as_str()) {
        Ok(structural) => {
            // Logged AND returned. Narration for whoever is watching the run
            // is fine; it stopped being the only record of the fact.
            tracing::warn!(
                utt_idx,
                word_idx,
                speaker_id = %speaker_id,
                word_text = %word.text.as_str(),
                lang = %utterance_lang.as_str(),
                lang_errors = ?lang_errors,
                "ASR token fails language-level validation \
                 (structurally legal CHAT); emitting verbatim \
                 for downstream validator + CHECK to surface",
            );
            Ok(WordAdmission::StructuralOnly {
                desc: describe(structural),
                refusal: LanguageInvalidWord {
                    utt_idx,
                    word_idx,
                    speaker_id: speaker_id.to_owned(),
                    text: word.text.as_str().to_owned(),
                    lang: utterance_lang.as_str().to_owned(),
                    parse_errors: lang_errors,
                },
            })
        }
        Err(parse_errors) => Err(TranscriptBuildError::WordFailedValidation {
            utt_idx,
            word_idx,
            speaker_id: speaker_id.to_owned(),
            word_text: word.text.as_str().to_owned(),
            lang: utterance_lang.as_str().to_owned(),
            parse_errors,
        }),
    }
}

fn role_for_speaker_code(code: &str) -> (String, String) {
    match code {
        "INV" => ("Investigator".into(), "Investigator".into()),
        "CHI" => ("Target_Child".into(), "Target_Child".into()),
        "MOT" => ("Mother".into(), "Mother".into()),
        "FAT" => ("Father".into(), "Father".into()),
        "EXP" => ("Experimenter".into(), "Experimenter".into()),
        "OBS" => ("Observer".into(), "Observer".into()),
        "TEA" => ("Teacher".into(), "Teacher".into()),
        _ => ("Participant".into(), "Participant".into()),
    }
}
