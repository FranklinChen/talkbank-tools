//! Build a CHAT file from a structured transcript description.
//!
//! This module constructs a [`ChatFile`] AST from structured input — either
//! a JSON transcript description (for PyO3 bridge compatibility) or typed
//! Rust structs (for the Rust server's transcribe orchestrator).
//!
//! # Two entry points
//!
//! - [`build_chat`] — takes a typed [`TranscriptDescription`] struct
//! - [`build_chat_from_json`] — deserializes JSON into `TranscriptDescription`,
//!   then calls `build_chat`. Used by the PyO3 bridge to delegate here.
//!
//! # Convenience
//!
//! - [`transcript_from_asr_utterances`] — converts post-processed ASR
//!   utterances into a `TranscriptDescription` for CHAT assembly.

use std::path::Path;

use serde::Deserialize;
use talkbank_model::Span;
use talkbank_model::model::{
    BracketedContent, BracketedItem, Bullet, ChatFile, DependentTier, Header, IDHeader,
    LanguageCode, LanguageCodes, Line, MediaHeader, MediaType, ParticipantEntries,
    ParticipantEntry, ParticipantName, ParticipantRole, Retrace, RetraceKind, Separator,
    SpeakerCode, Terminator, Utterance, UtteranceContent, Word,
};

use crate::asr_postprocess;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Structured description of a transcript to be assembled into CHAT format.
///
/// Fields mirror the JSON format accepted by the PyO3 `build_chat()` function.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptDescription {
    /// ISO 639-3 language codes (e.g. `["eng"]`). Defaults to `["eng"]` if empty.
    #[serde(default)]
    pub langs: Vec<String>,
    /// Participant entries. At least one is required.
    pub participants: Vec<ParticipantDesc>,
    /// Optional media filename (e.g. `"recording.mp3"`).
    pub media_name: Option<String>,
    /// Optional media type (`"audio"` or `"video"`). Defaults to `"audio"`.
    pub media_type: Option<String>,
    /// Utterances to include in the transcript.
    #[serde(default)]
    pub utterances: Vec<UtteranceDesc>,
    /// Whether to generate `%wor` tiers when word-level timing is available.
    ///
    /// Defaults to `false` (BA2 parity: transcribe omits `%wor` unless
    /// explicitly requested via `--wor`). The JSON bridge (PyO3) defaults to
    /// `false` via serde; callers that want `%wor` must set this to `true`.
    #[serde(default)]
    pub write_wor: bool,
}

/// A participant in the transcript.
#[derive(Debug, Clone, Deserialize)]
pub struct ParticipantDesc {
    /// Speaker code (e.g. `"PAR"`, `"INV"`, `"CHI"`).
    pub id: String,
    /// Participant name for `@Participants` header. `None` omits the name
    /// field (output: `CODE Role`). `Some("...")` adds it (output: `CODE Name Role`).
    pub name: Option<String>,
    /// Participant role (e.g. `"Participant"`, `"Investigator"`, `"Target_Child"`).
    /// Callers should always set this — derive from speaker code via
    /// `role_for_speaker_code` if unknown. Defaults to `"Participant"` only
    /// for JSON backward compatibility.
    #[serde(default = "default_participant_role")]
    pub role: String,
    /// Corpus name for `@ID` header. Empty string if unknown.
    #[serde(default)]
    pub corpus: String,
}

/// An utterance in the transcript.
///
/// Either `words` (word-level with individual timings) or `text` (parse as
/// a single CHAT utterance line) should be provided. If both are present,
/// `words` takes precedence (when non-empty).
#[derive(Debug, Clone, Deserialize)]
pub struct UtteranceDesc {
    /// Speaker code for this utterance.
    pub speaker: String,
    /// Word-level tokens with optional per-word timing.
    pub words: Option<Vec<WordDesc>>,
    /// Full utterance text (alternative to word-level). Parsed via tree-sitter.
    ///
    /// This is a public API surface for callers who want to pass pre-formatted
    /// CHAT text rather than individual word tokens. The text is wrapped in a
    /// mini CHAT document and parsed by `build_text_utterance()`. Currently
    /// unused by the ASR pipeline (which always provides `words`), but
    /// preserved for external JSON API consumers.
    pub text: Option<String>,
    /// Utterance-level start time in ms (used with `text` mode).
    pub start_ms: Option<u64>,
    /// Utterance-level end time in ms (used with `text` mode).
    pub end_ms: Option<u64>,
    /// Detected language for this utterance (ISO 639-3). When set and different
    /// from the primary language (`langs[0]`), a `[- lang]` precode is prepended.
    #[serde(default)]
    pub lang: Option<String>,
}

/// A single word token with optional timing.
#[derive(Debug, Clone, Deserialize)]
pub struct WordDesc {
    /// Word text (ready for CHAT assembly via TreeSitterParser).
    pub text: asr_postprocess::ChatWordText,
    /// Start time in milliseconds.
    pub start_ms: Option<u64>,
    /// End time in milliseconds.
    pub end_ms: Option<u64>,
    /// What role this word plays (regular, retrace, etc.).
    #[serde(default)]
    pub kind: asr_postprocess::WordKind,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Terminator recognition lives on `Terminator::is_chat_terminator` in
// talkbank-model — see callers below.

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a CHAT file from a JSON transcript description string.
///
/// This is the entry point used by the PyO3 bridge (`build_chat_inner`).
pub fn build_chat_from_json(json: &str) -> Result<ChatFile, String> {
    let desc: TranscriptDescription =
        serde_json::from_str(json).map_err(|e| format!("Invalid JSON: {e}"))?;
    build_chat(&desc)
}

/// Build a CHAT file from a typed transcript description.
pub fn build_chat(desc: &TranscriptDescription) -> Result<ChatFile, String> {
    let parser = talkbank_parser::TreeSitterParser::new()
        .map_err(|e| format!("Failed to create parser: {e}"))?;
    let langs: Vec<String> = if desc.langs.is_empty() {
        vec!["eng".to_string()]
    } else {
        desc.langs.clone()
    };

    if desc.participants.is_empty() {
        return Err("At least one participant is required".to_string());
    }

    // --- Build participant entries and @ID headers ---
    let mut participant_entries: Vec<ParticipantEntry> = Vec::new();
    let mut id_headers: Vec<IDHeader> = Vec::new();

    for p in &desc.participants {
        let role = p.role.as_str();
        let corpus = if p.corpus.is_empty() {
            "corpus_name"
        } else {
            p.corpus.as_str()
        };

        let entry = ParticipantEntry {
            speaker_code: SpeakerCode::new(p.id.as_str()),
            name: p.name.as_ref().map(ParticipantName::new),
            role: ParticipantRole::new(role),
        };
        participant_entries.push(entry);

        let lang_code = langs.first().map(String::as_str).unwrap_or("eng");
        let id = IDHeader::new(lang_code, p.id.as_str(), role).with_corpus(corpus);
        id_headers.push(id);
    }

    // --- Build header lines ---
    let mut lines: Vec<Line> = vec![
        Line::header(Header::Utf8),
        Line::header(Header::Begin),
        Line::header(Header::Languages {
            codes: LanguageCodes::new(langs.iter().map(LanguageCode::new).collect()),
        }),
        Line::header(Header::Participants {
            entries: ParticipantEntries::new(participant_entries),
        }),
    ];
    for id in id_headers {
        lines.push(Line::header(Header::ID(id)));
    }

    // --- Optional @Media header ---
    if let Some(ref media_name) = desc.media_name {
        let normalized_media_name = normalize_media_name(media_name);
        let media_type = match desc.media_type.as_deref() {
            Some("video") => MediaType::Video,
            Some("audio") | None => MediaType::Audio,
            other => {
                tracing::warn!(media_type = ?other, "unrecognized media_type, defaulting to audio");
                MediaType::Audio
            }
        };
        lines.push(Line::header(Header::Media(MediaHeader::new(
            normalized_media_name.as_str(),
            media_type,
        ))));
    }

    // --- Build utterances ---
    let primary_lang = langs.first().map(String::as_str).unwrap_or("eng");
    for utt_desc in &desc.utterances {
        let words = utt_desc.words.as_deref().unwrap_or(&[]);

        if words.is_empty() {
            // Text-level utterance: parse via tree-sitter
            if let Some(ref text) = utt_desc.text
                && let Some(utt_line) = build_text_utterance(
                    &parser,
                    &utt_desc.speaker,
                    text,
                    utt_desc.start_ms,
                    utt_desc.end_ms,
                    &langs,
                )?
            {
                lines.push(utt_line);
            }
            continue;
        }

        // Word-level utterance
        if let Some(mut utt_line) =
            build_word_utterance(&parser, &utt_desc.speaker, words, desc.write_wor)?
        {
            // Set [- lang] precode when utterance language differs from primary
            if let Some(ref utt_lang) = utt_desc.lang
                && utt_lang != primary_lang
                && let Line::Utterance(ref mut utt) = utt_line
            {
                utt.main.content.language_code = Some(LanguageCode::new(utt_lang.as_str()));
            }
            lines.push(utt_line);
        }
    }

    lines.push(Line::header(Header::End));

    Ok(ChatFile::new(lines))
}

/// Derive participant name and role from the CHAT speaker code.
///
/// Standard CHAT speaker codes have conventional roles. This replaces the
/// silent `unwrap_or("Participant")` default that gave every speaker the
/// same role regardless of their code.
fn default_participant_role() -> String {
    "Participant".to_string()
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
        // PAR and anything else: Participant
        _ => ("Participant".into(), "Participant".into()),
    }
}

fn normalize_media_name(raw: &str) -> String {
    let candidate = Path::new(raw);
    candidate
        .file_stem()
        .filter(|stem| !stem.is_empty())
        .or_else(|| candidate.file_name())
        .filter(|name| !name.is_empty())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| raw.to_string())
}

/// Convert post-processed ASR utterances into a [`TranscriptDescription`].
///
/// Speaker indices (0-based) are mapped to `participant_ids`. If a speaker
/// index exceeds the participant list, a generated ID like `"SP1"` is used.
///
/// `write_wor` controls whether the resulting CHAT will include `%wor` tiers
/// when word-level timing is present.
/// Domain errors from building a `TranscriptDescription`.
///
/// Exposes structured failure information — the offending word's
/// position, text, declared language, and the full
/// `Vec<talkbank_model::ParseError>` from `ChatWordText::try_from_lang`
/// — so upstream callers can render diagnostics or branch on failure
/// class without re-parsing a string.
#[derive(Debug, thiserror::Error)]
pub enum TranscriptBuildError {
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
}

/// Convert post-processed ASR utterances into a pre-serialization
/// `TranscriptDescription`.
///
/// Each word's text is validated at construction via
/// [`ChatWordText::try_from_lang`][try_lang] under the utterance's declared
/// language (falling back to the primary `langs[0]` or `"eng"`). Fails
/// with [`TranscriptBuildError`] at the first offending word. This is
/// the "loud guard" half of strategy 4c: normalization runs upstream
/// in `process_raw_asr`'s stages; this gate is the belt after the
/// braces.
///
/// [try_lang]: asr_postprocess::ChatWordText::try_from_lang
pub fn transcript_from_asr_utterances(
    utterances: &[asr_postprocess::Utterance],
    participant_ids: &[String],
    langs: &[String],
    media_name: Option<&str>,
    write_wor: bool,
) -> Result<TranscriptDescription, TranscriptBuildError> {
    // Optional diagnostic: when `BA3_DUMP_UTTERANCES=/path/to/file.json`
    // is set, write the full post-processed `Vec<Utterance>` to disk
    // before validation. Useful for grep/jq exploration of pipeline
    // output without rerunning the whole ASR job. Documented under
    // "Debugging and Per-Stage Inspection" in `CLAUDE.md`.
    if let Ok(path) = std::env::var("BA3_DUMP_UTTERANCES")
        && let Ok(json) = serde_json::to_string_pretty(utterances)
    {
        let _ = std::fs::write(&path, json);
        tracing::warn!(path = %path, "BA3_DUMP_UTTERANCES wrote post-processed utterances");
    }

    // Collect unique speaker indices to build participant list
    let mut seen_speakers: Vec<asr_postprocess::SpeakerIndex> = Vec::new();
    for utt in utterances {
        if !seen_speakers.contains(&utt.speaker) {
            seen_speakers.push(utt.speaker);
        }
    }
    seen_speakers.sort_unstable();

    let participants: Vec<ParticipantDesc> = seen_speakers
        .iter()
        .map(|&idx| {
            let i = idx.as_usize();
            let id = if i < participant_ids.len() {
                participant_ids[i].clone()
            } else {
                format!("SP{i}")
            };
            let (_name, role) = role_for_speaker_code(&id);
            ParticipantDesc {
                id,
                name: None, // omit name — @Participants writes "CODE Role", not "CODE Name Role"
                role,
                corpus: String::new(),
            }
        })
        .collect();

    // Primary language for utterances that don't carry an override. Falls
    // back to "eng" to match `TranscriptDescription`'s default below.
    let primary_lang_str = langs.first().map(|s| s.as_str()).unwrap_or("eng");
    let primary_lang_code = talkbank_model::model::LanguageCode::from(primary_lang_str);

    let mut utt_descs: Vec<UtteranceDesc> = Vec::with_capacity(utterances.len());
    for (utt_idx, utt) in utterances.iter().enumerate() {
        let i = utt.speaker.as_usize();
        let speaker_id = if i < participant_ids.len() {
            participant_ids[i].clone()
        } else {
            format!("SP{i}")
        };

        // Resolve the language for this utterance's word validation.
        // Utterance-level override (code-switching segment) wins over the
        // primary language.
        let utt_lang_code = utt
            .lang
            .as_deref()
            .map(talkbank_model::model::LanguageCode::from)
            .unwrap_or_else(|| primary_lang_code.clone());

        let mut words: Vec<WordDesc> = Vec::with_capacity(utt.words.len());
        for (word_idx, w) in utt.words.iter().enumerate() {
            let text =
                match asr_postprocess::ChatWordText::try_from_lang(w.text.as_str(), &utt_lang_code)
                {
                    Ok(t) => t,
                    Err(lang_errors) => {
                        // Two distinct failure classes need distinct handling:
                        //
                        // 1. **Validation failure** (E220 numeric digits and
                        //    other Word::validate language rules) — the
                        //    token is structurally legal CHAT (tree-sitter
                        //    accepts e.g. `C-3PO`, `13th`, `abc123`) but
                        //    violates a language-level policy. Emit the
                        //    token verbatim. The full-file Rust validator
                        //    and CHECK will fire the same E220 downstream
                        //    and the file shows up in the human review
                        //    queue. The transcriber listens, decides what
                        //    was actually said, and fixes the transcript.
                        //    ASR has no business inventing semantics here,
                        //    and substituting `xxx` would corrupt that
                        //    marker's "transcriber listened and could not
                        //    make it out" meaning across the whole corpus.
                        //    See `untranscribed-markers.md` in the
                        //    talkbank-tools book for the canonical
                        //    `xxx`/`yyy`/`www` semantics.
                        //
                        // 2. **Structural (parse) failure** — tree-sitter
                        //    rejected the token (e.g., embedded literal `"`
                        //    that Stage 2c boundary-strip didn't catch).
                        //    There is no way to serialize this as legal
                        //    CHAT, so the gate fails loud and the file
                        //    aborts. That residual "loud guard" matters:
                        //    emitting malformed CHAT silently would corrupt
                        //    the file beyond CHECK's ability to flag it.
                        //
                        // The discriminator is `try_from` (structural-only,
                        // skips Word::validate): if it succeeds, the token
                        // is class 1; if it also fails, class 2.
                        match asr_postprocess::ChatWordText::try_from(w.text.as_str()) {
                            Ok(structural) => {
                                tracing::warn!(
                                    utt_idx,
                                    word_idx,
                                    speaker_id = %speaker_id,
                                    word_text = %w.text.as_str(),
                                    lang = %utt_lang_code.as_str(),
                                    lang_errors = ?lang_errors,
                                    "ASR token fails language-level validation \
                                     (structurally legal CHAT); emitting verbatim \
                                     for downstream validator + CHECK to surface",
                                );
                                structural
                            }
                            Err(parse_errors) => {
                                return Err(TranscriptBuildError::WordFailedValidation {
                                    utt_idx,
                                    word_idx,
                                    speaker_id: speaker_id.clone(),
                                    word_text: w.text.as_str().to_owned(),
                                    lang: utt_lang_code.as_str().to_owned(),
                                    parse_errors,
                                });
                            }
                        }
                    }
                };
            words.push(WordDesc {
                text,
                start_ms: w.start_ms.map(|ms| ms as u64),
                end_ms: w.end_ms.map(|ms| ms as u64),
                kind: w.kind,
            });
        }

        utt_descs.push(UtteranceDesc {
            speaker: speaker_id,
            words: Some(words),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: utt.lang.clone(),
        });
    }

    Ok(TranscriptDescription {
        langs: if langs.is_empty() {
            vec!["eng".to_string()]
        } else {
            langs.to_vec()
        },
        participants,
        media_name: media_name.map(String::from),
        media_type: Some("audio".to_string()),
        utterances: utt_descs,
        write_wor,
    })
}

/// If `text` is a tag-marker separator (comma, tag marker, vocative marker),
/// return the corresponding [`Separator`] model type. Otherwise return `None`.
pub fn tag_marker_separator(text: &str) -> Option<Separator> {
    match text {
        "," => Some(Separator::Comma { span: Span::DUMMY }),
        "\u{201E}" => Some(Separator::Tag { span: Span::DUMMY }),
        "\u{2021}" => Some(Separator::Vocative { span: Span::DUMMY }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Build a text-level utterance by parsing through tree-sitter.
///
/// This path constructs a minimal valid CHAT document around the input text
/// and parses it with `parse_strict()`. The mini-document hack is necessary
/// because tree-sitter requires complete document context (headers, `@Begin`,
/// `@End`) to parse a single utterance correctly.
///
/// **Callers:** This function is used by the `UtteranceDesc.text` API path —
/// when a caller provides a pre-formatted CHAT utterance string instead of
/// word-level tokens. It has zero production callers in the current codebase
/// (the ASR pipeline always uses word-level `WordDesc` tokens), but it
/// preserves the JSON API contract for external callers who construct
/// `TranscriptDescription` directly. The PyO3 bridge tests exercise this path.
fn build_text_utterance(
    parser: &talkbank_parser::TreeSitterParser,
    speaker: &str,
    text: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    langs: &[String],
) -> Result<Option<Line>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }

    let bullet_str = match (start_ms, end_ms) {
        (Some(s), Some(e)) => format!(" \x15{}_{}\x15", s, e),
        _ => String::new(),
    };

    let lang_code = langs.first().map(String::as_str).unwrap_or("eng");
    let mini_chat = format!(
        "@UTF8\n@Begin\n@Languages:\t{lang}\n@Participants:\t{spk} Participant Participant\n\
         @ID:\t{lang}|corpus_name|{spk}|||||Participant|||\n*{spk}:\t{text}{bullet}\n@End\n",
        lang = lang_code,
        spk = speaker,
        text = text,
        bullet = bullet_str,
    );

    let parsed = crate::parse::parse_strict(parser, &mini_chat)
        .map_err(|e| format!("Failed to parse text utterance for speaker {speaker}: {e}"))?;

    for parsed_line in parsed.lines.into_iter() {
        if let Line::Utterance(utt) = parsed_line {
            return Ok(Some(Line::Utterance(utt)));
        }
    }

    Ok(None)
}

/// Parse a single word, falling back to unchecked for ASR tokens.
fn parse_asr_word(parser: &talkbank_parser::TreeSitterParser, text: &str) -> Word {
    let errors = talkbank_model::NullErrorSink;
    match parser.parse_word_fragment(text, 0, &errors).into_option() {
        Some(parsed) => parsed,
        None => {
            tracing::warn!(
                word = text,
                "ASR word is not valid CHAT syntax; using unchecked fallback"
            );
            Word::new_unchecked(text, text)
        }
    }
}

/// Parse a word and attach inline bullet timing, updating utterance-level
/// timing bookkeeping. Returns the parsed `Word` and whether timing was present.
fn parse_and_time_word(
    parser: &talkbank_parser::TreeSitterParser,
    text: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    utt_start_ms: &mut Option<u64>,
    utt_end_ms: &mut Option<u64>,
    has_timing: &mut bool,
) -> Word {
    let mut word = parse_asr_word(parser, text);
    if let (Some(s), Some(e)) = (start_ms, end_ms) {
        word.inline_bullet = Some(Bullet::new(s, e));
        *has_timing = true;
        if utt_start_ms.is_none() {
            *utt_start_ms = Some(s);
        }
        *utt_end_ms = Some(e);
    }
    word
}

/// Build a word-level utterance from individual word tokens.
///
/// When `write_wor` is `true` and word-level timing is present, a `%wor`
/// dependent tier is generated. When `false`, the `%wor` tier is omitted
/// regardless of timing (BA2 default for transcribe).
///
/// Words marked with `WordKind::Retrace` are grouped into consecutive runs
/// and wrapped in proper CHAT retrace AST nodes:
/// - A single retrace word → one `[/]` annotated-word node (`word [/]`).
/// - A run of N > 1 Retrace words that are all the **same** word (unigram
///   run, e.g. `"a a a"` where the first two `a`s are marked Retrace) →
///   N separate `[/]` annotated-word nodes (`a [/] a [/]`…). In CHAT
///   convention the bracket form `<w1 w2> [/]` means a repeated *phrase*
///   (multi-word unit); a string of identical unigrams is semantically
///   N separate repetitions of the same word.
/// - A run of N > 1 Retrace words with differing text → one bracketed
///   annotated-group node (`<I want> [/] I want cookie`).
fn build_word_utterance(
    parser: &talkbank_parser::TreeSitterParser,
    speaker: &str,
    words: &[WordDesc],
    write_wor: bool,
) -> Result<Option<Line>, String> {
    let mut content: Vec<UtteranceContent> = Vec::new();
    let mut utt_start_ms: Option<u64> = None;
    let mut utt_end_ms: Option<u64> = None;
    let mut has_timing = false;

    // Determine terminator from the last word. Unrecognized input (and the
    // empty-words case) defaults to Period, preserving prior behavior.
    let last_text = words.last().map(|w| w.text.as_str()).unwrap_or(".");
    let terminator = Terminator::try_from_chat_str(last_text)
        .unwrap_or(Terminator::Period { span: Span::DUMMY });

    let mut i = 0;
    while i < words.len() {
        let w = &words[i];
        let text = w.text.as_str().trim();

        if text.is_empty() {
            i += 1;
            continue;
        }

        // Skip ending punctuation (it's captured in the terminator)
        if Terminator::is_chat_terminator(text) {
            i += 1;
            continue;
        }

        // Tag-marker separators are not words
        if let Some(sep) = tag_marker_separator(text) {
            content.push(UtteranceContent::Separator(sep));
            i += 1;
            continue;
        }

        if w.kind == asr_postprocess::WordKind::Retrace {
            // Collect consecutive retrace words.
            let group_start = i;
            while i < words.len() && words[i].kind == asr_postprocess::WordKind::Retrace {
                i += 1;
            }
            let retrace_words = &words[group_start..i];

            // Parse each retrace word with timing.
            let mut parsed: Vec<Word> = Vec::new();
            for rw in retrace_words {
                let t = rw.text.as_str().trim();
                if t.is_empty() {
                    continue;
                }
                let word = parse_and_time_word(
                    parser,
                    t,
                    rw.start_ms,
                    rw.end_ms,
                    &mut utt_start_ms,
                    &mut utt_end_ms,
                    &mut has_timing,
                );
                parsed.push(word);
            }

            if parsed.is_empty() {
                continue;
            }

            // CHAT reserves `<w1 w2> [/]` for a repeated *phrase* (multi-word
            // unit). A run of identical unigrams (`"a a a"`) is N single-word
            // repetitions and emits as N separate `w [/]` nodes.
            let first_text = parsed[0].cleaned_text();
            let all_same_text = parsed.len() > 1
                && parsed
                    .iter()
                    .skip(1)
                    .all(|w| w.cleaned_text().eq_ignore_ascii_case(first_text));

            if parsed.len() == 1 || all_same_text {
                for word in parsed {
                    let bracketed =
                        BracketedContent::new(vec![BracketedItem::Word(Box::new(word))]);
                    let retrace = Retrace::new(bracketed, RetraceKind::Partial);
                    content.push(UtteranceContent::Retrace(Box::new(retrace)));
                }
            } else {
                let items: Vec<BracketedItem> = parsed
                    .into_iter()
                    .map(|w| BracketedItem::Word(Box::new(w)))
                    .collect();
                let bracketed = BracketedContent::new(items);
                let retrace = Retrace::new(bracketed, RetraceKind::Partial).as_group();
                content.push(UtteranceContent::Retrace(Box::new(retrace)));
            }
            continue;
        }

        // Regular word
        let word = parse_and_time_word(
            parser,
            text,
            w.start_ms,
            w.end_ms,
            &mut utt_start_ms,
            &mut utt_end_ms,
            &mut has_timing,
        );
        content.push(UtteranceContent::Word(Box::new(word)));
        i += 1;
    }

    if content.is_empty() {
        return Ok(None);
    }

    let mut main = talkbank_model::model::MainTier::new(speaker, content, terminator);
    if let (Some(start), Some(end)) = (utt_start_ms, utt_end_ms) {
        main = main.with_bullet(Bullet::new(start, end));
    }

    let mut utt = Utterance::new(main);
    if write_wor && has_timing {
        let wor_tier = utt.main.generate_wor_tier();
        utt.dependent_tiers.push(DependentTier::Wor(wor_tier));
    }

    Ok(Some(Line::utterance(utt)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{TreeSitterParser, parse_lenient};
    use crate::serialize::to_chat_string;

    /// Helper: create a regular WordDesc with validated text and explicit kind.
    fn wd(text: &str, start_ms: Option<u64>, end_ms: Option<u64>) -> WordDesc {
        WordDesc {
            text: asr_postprocess::ChatWordText::try_from(text)
                .expect("test: word text must be CHAT-legal"),
            start_ms,
            end_ms,
            kind: asr_postprocess::WordKind::Regular,
        }
    }

    #[test]
    fn test_build_chat_minimal() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("hello", None, None),
                    wd("world", None, None),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("@Languages:\teng"));
        assert!(output.contains("*PAR:\thello world ."));
    }

    #[test]
    fn test_build_chat_with_timing() {
        let parser = TreeSitterParser::new().unwrap();
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: Some("test.mp3".to_string()),
            media_type: Some("audio".to_string()),
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("hello", Some(0), Some(500)),
                    wd("world", Some(500), Some(1000)),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: true,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("@Media:\ttest, audio"), "got: {output}");
        assert!(output.contains("%wor:"));
        let (_parsed, errors) = parse_lenient(&parser, &output);
        assert!(
            errors.is_empty(),
            "serialized CHAT should reparse cleanly: {errors:?}"
        );
    }

    #[test]
    fn test_build_chat_from_json() {
        let json = r#"{
            "langs": ["eng"],
            "participants": [{"id": "PAR"}],
            "utterances": [
                {"speaker": "PAR", "words": [
                    {"text": "hello"},
                    {"text": "."}
                ]}
            ]
        }"#;

        let chat_file = build_chat_from_json(json).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:\thello ."));
    }

    #[test]
    fn test_build_chat_text_utterance() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: None,
                text: Some("hello world .".to_string()),
                start_ms: Some(0),
                end_ms: Some(1000),
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:\thello world ."));
    }

    #[test]
    fn test_build_chat_question_terminator() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![wd("how", None, None), wd("?", None, None)]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:\thow ?"));
    }

    #[test]
    fn test_write_wor_false_suppresses_wor_tier() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: Some("test.mp3".to_string()),
            media_type: Some("audio".to_string()),
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("hello", Some(0), Some(500)),
                    wd("world", Some(500), Some(1000)),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };

        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(
            !output.contains("%wor:"),
            "write_wor=false should suppress %wor tier, got: {output}"
        );
        // Inline word bullets should still be present
        assert!(
            output.contains("\u{15}"),
            "word-level bullets should still appear on the main tier"
        );
    }

    #[test]
    fn test_transcript_from_asr_utterances() {
        let utterances = vec![
            asr_postprocess::Utterance {
                speaker: asr_postprocess::SpeakerIndex(0),
                words: vec![
                    asr_postprocess::AsrWord::new("hello", Some(0), Some(500)),
                    asr_postprocess::AsrWord::new(".", None, None),
                ],
                lang: None,
            },
            asr_postprocess::Utterance {
                speaker: asr_postprocess::SpeakerIndex(1),
                words: vec![
                    asr_postprocess::AsrWord::new("world", Some(500), Some(1000)),
                    asr_postprocess::AsrWord::new(".", None, None),
                ],
                lang: None,
            },
        ];

        let ids = vec!["PAR".to_string(), "INV".to_string()];
        let desc = transcript_from_asr_utterances(
            &utterances,
            &ids,
            &["eng".to_string()],
            Some("test.mp3"),
            false,
        )
        .expect("test: transcript_from_asr_utterances should succeed");

        assert_eq!(desc.participants.len(), 2);
        assert_eq!(desc.participants[0].id, "PAR");
        assert_eq!(desc.participants[1].id, "INV");
        assert_eq!(desc.utterances.len(), 2);
        assert_eq!(desc.utterances[0].speaker, "PAR");
        assert_eq!(desc.utterances[1].speaker, "INV");

        // Should build a valid CHAT file
        let chat_file = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat_file);
        assert!(output.contains("*PAR:"));
        assert!(output.contains("*INV:"));
    }

    #[test]
    fn test_transcript_from_asr_auto_generates_speaker_ids() {
        let utterances = vec![asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(5),
            words: vec![asr_postprocess::AsrWord::new("hello", None, None)],
            lang: None,
        }];

        let desc =
            transcript_from_asr_utterances(&utterances, &[], &["eng".to_string()], None, false)
                .expect("test: transcript_from_asr_utterances should succeed");
        assert_eq!(desc.participants[0].id, "SP5");
    }

    // ── ASR-to-CHAT validation gap regression tests ──────────────────────
    //
    // Two ASR token classes used to tank the entire transcribe job at the
    // `transcript_from_asr_utterances` gate:
    //
    //   1. Boundary quote marks (e.g. `"My`) — Whisper transcribes quoted
    //      speech verbatim, leaving stray `"` characters glued to the next
    //      word. Tree-sitter rejects the `"` and the file aborts.
    //   2. Digit-bearing alphanumeric tokens (e.g. `C-3PO` under English) —
    //      structurally legal CHAT, but `Word::validate` fires E220
    //      ("numeric digits not allowed") in languages outside the
    //      digit-permitting set.
    //
    // Design: maximize information preservation, but never abuse the
    // reserved `xxx` / `yyy` / `www` markers (those mean specific things
    // about transcriber experience — see `untranscribed-markers.md` in
    // the talkbank-tools book).
    //
    //   • Boundary quotes get a silent orthographic strip (Stage 2c) —
    //     no information is lost when `"` becomes a no-op character.
    //   • Validation-only failures (digit policy etc.) fall back to the
    //     structural-only `ChatWordText::try_from` path, shipping the
    //     token verbatim. The downstream full-file validator and CHECK
    //     fire the same E220 and the file ends up in the human review
    //     queue. The transcriber listens, decides what was actually
    //     said, fixes the transcript. ASR doesn't pretend to know.
    //   • Genuine structural (parse) failures still fail loud — emitting
    //     malformed CHAT silently would corrupt the file beyond CHECK's
    //     ability to flag it.

    #[test]
    fn quote_mark_in_asr_token_is_silently_stripped() {
        // `"My` arrives at the pipeline; Stage 2c boundary-quote strip
        // (`cleanup::strip_boundary_quotes`) removes the leading `"`
        // before validation. The whole pipeline succeeds and the first
        // content word is `My`, not `"My`.
        let result = run_transcribe_to_description(
            &[("\"My", 0.0, 0.5), ("cake", 0.5, 1.0), (".", 1.0, 1.0)],
            "eng",
        )
        .expect("Stage 2c should silently strip the leading quote and pass validation");

        let first_word = result
            .utterances
            .first()
            .and_then(|u| u.words.as_deref()?.first())
            .expect("first utterance has at least one word");
        assert_eq!(
            first_word.text.as_str(),
            "My",
            "boundary quote should be stripped",
        );
    }

    #[test]
    fn alphanumeric_token_under_eng_is_emitted_verbatim_for_review() {
        // Whisper transcribes proper nouns like `C-3PO` verbatim. Under
        // eng, digits are illegal (E220). The ASR pipeline does NOT
        // invent semantics for digit-bearing alphanumerics (digit-by-
        // digit? cardinal? ordinal? — unknowable from surface form).
        // The gate falls back to the structural-only `try_from` path,
        // shipping `C-3PO` verbatim. Tree-sitter accepts the token,
        // the file builds, and the downstream validator + CHECK fire
        // E220 for the human reviewer to listen and decide.
        //
        // Substituting `xxx` is BANNED — it would corrupt that marker's
        // "transcriber listened and could not make it out" meaning
        // across the whole corpus.
        let result = run_transcribe_to_description(
            &[
                ("the", 0.0, 0.1),
                ("C-3PO", 0.1, 0.8),
                ("droid", 0.8, 1.2),
                (".", 1.2, 1.2),
            ],
            "eng",
        )
        .expect(
            "gate should fall back to structural-only path so `C-3PO` ships \
             verbatim for downstream validator + CHECK to flag",
        );

        let words: Vec<String> = result
            .utterances
            .iter()
            .flat_map(|u| u.words.as_deref().unwrap_or(&[]))
            .map(|w| w.text.as_str().to_owned())
            .collect();
        assert!(
            words.iter().any(|w| w == "C-3PO"),
            "expected `C-3PO` preserved verbatim in {words:?}",
        );
        assert!(
            !words.iter().any(|w| w == "xxx"),
            "must not substitute `xxx` — it has the reserved meaning \
             \"transcriber listened and could not make it out\". Found: {words:?}",
        );
    }

    #[test]
    fn alphanumeric_token_passes_zho_validation_gate() {
        // Counterpart to the eng test: numeric digits ARE legal in some
        // languages, so the same `C-3PO` token must pass the gate under
        // those languages. Confirms the rule is language-specific and
        // bounds the scope of any fix — the normalizer must consult the
        // utterance's language before deciding whether to act.
        //
        // NOTE on the language code: the talkbank-tools word validator's
        // current digit-allow set is `{zho, cym, vie, tha, nan, yue, min,
        // hak}`. `cmn` (Mandarin, spoken) is NOT in that set even though
        // it's the more linguistically precise code; `zho` (Chinese, the
        // macrolanguage / written) is. That asymmetry is a separate
        // talkbank-tools concern — not this crate's bug — but documenting
        // it here so a future ASR-output-with-cmn test failure traces to
        // the right place.
        let utterances = vec![asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(0),
            words: vec![
                asr_postprocess::AsrWord::new("C-3PO", Some(0), Some(500)),
                asr_postprocess::AsrWord::new("。", None, None),
            ],
            lang: None,
        }];
        let result = transcript_from_asr_utterances(
            &utterances,
            &["PAR".to_string()],
            &["zho".to_string()],
            None,
            false,
        );
        assert!(
            result.is_ok(),
            "zho should accept digit-containing tokens; got {result:?}"
        );
    }

    #[test]
    fn test_tag_marker_separator() {
        assert!(tag_marker_separator(",").is_some());
        assert!(tag_marker_separator("\u{201E}").is_some());
        assert!(tag_marker_separator("\u{2021}").is_some());
        assert!(tag_marker_separator("hello").is_none());
    }

    #[test]
    fn test_empty_participants_error() {
        let desc = TranscriptDescription {
            langs: vec![],
            participants: vec![],
            media_name: None,
            media_type: None,
            utterances: vec![],
            write_wor: false,
        };
        assert!(build_chat(&desc).is_err());
    }

    // -- Retrace AST construction tests --

    /// Helper: create a retrace WordDesc.
    fn wd_retrace(text: &str, start_ms: Option<u64>, end_ms: Option<u64>) -> WordDesc {
        WordDesc {
            text: asr_postprocess::ChatWordText::try_from(text)
                .expect("test: word text must be CHAT-legal"),
            start_ms,
            end_ms,
            kind: asr_postprocess::WordKind::Retrace,
        }
    }

    /// Helper: build a single-utterance CHAT file and return serialized output.
    fn build_single_utterance(words: Vec<WordDesc>) -> String {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(words),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };
        let chat = build_chat(&desc).unwrap();
        to_chat_string(&chat)
    }

    #[test]
    fn single_word_retrace_produces_annotated_word() {
        // "I [/] I went ." → AnnotatedWord with PartialRetracing
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd("I", None, None),
            wd("went", None, None),
            wd(".", None, None),
        ]);
        assert!(
            output.contains("I [/] I went ."),
            "expected single-word retrace: {output}"
        );
    }

    #[test]
    fn multi_word_retrace_produces_annotated_group() {
        // "<I want> [/] I want cookie ."
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd_retrace("want", None, None),
            wd("I", None, None),
            wd("want", None, None),
            wd("cookie", None, None),
            wd(".", None, None),
        ]);
        assert!(
            output.contains("<I want> [/] I want cookie ."),
            "expected multi-word retrace: {output}"
        );
    }

    #[test]
    fn retrace_preserves_per_word_timing() {
        let output = build_single_utterance(vec![
            wd_retrace("go", Some(0), Some(200)),
            wd("go", Some(200), Some(400)),
            wd("home", Some(400), Some(600)),
            wd(".", None, None),
        ]);
        // The retrace word should have an inline bullet.
        assert!(
            output.contains("\u{15}"),
            "retrace word should preserve timing bullets: {output}"
        );
        assert!(output.contains("[/]"), "expected retrace marker: {output}");
    }

    #[test]
    fn retrace_output_reparses_cleanly() {
        let parser = TreeSitterParser::new().unwrap();
        // Single-word retrace
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd("I", None, None),
            wd("went", None, None),
            wd(".", None, None),
        ]);
        let (_parsed, errors) = parse_lenient(&parser, &output);
        assert!(
            errors.is_empty(),
            "single-word retrace should reparse: {errors:?}\noutput: {output}"
        );

        // Multi-word retrace
        let output = build_single_utterance(vec![
            wd_retrace("I", None, None),
            wd_retrace("want", None, None),
            wd("I", None, None),
            wd("want", None, None),
            wd("cookie", None, None),
            wd(".", None, None),
        ]);
        let (_parsed, errors) = parse_lenient(&parser, &output);
        assert!(
            errors.is_empty(),
            "multi-word retrace should reparse: {errors:?}\noutput: {output}"
        );
    }

    #[test]
    fn disfluency_and_retrace_end_to_end() {
        let parser = TreeSitterParser::new().unwrap();
        // Full pipeline: raw ASR → process_raw_asr (includes disfluency + retrace)
        // → transcript_from_asr_utterances → build_chat.
        //
        // Input "um um I I went" exercises BOTH pipeline stages:
        //   - disfluency: "um um" → "&-um &-um" (filled pauses stay as fillers,
        //     BA2 parity: fillers do NOT emit [/])
        //   - retrace: "I I" → "<I> [/] I" (genuine word repetition still marks)
        let output = asr_postprocess::AsrOutput {
            monologues: vec![asr_postprocess::AsrMonologue {
                speaker: asr_postprocess::SpeakerIndex(0),
                elements: vec![
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("um"),
                        ts: asr_postprocess::AsrTimestampSecs(0.0),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.2),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("um"),
                        ts: asr_postprocess::AsrTimestampSecs(0.2),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.4),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("I"),
                        ts: asr_postprocess::AsrTimestampSecs(0.4),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.5),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("I"),
                        ts: asr_postprocess::AsrTimestampSecs(0.5),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.6),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                    asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new("went"),
                        ts: asr_postprocess::AsrTimestampSecs(0.6),
                        end_ts: asr_postprocess::AsrTimestampSecs(0.8),
                        kind: asr_postprocess::AsrElementKind::Text,
                    },
                ],
            }],
        };
        let utts = asr_postprocess::process_raw_asr(&output, "eng");

        let desc = transcript_from_asr_utterances(
            &utts,
            &["PAR".to_string()],
            &["eng".to_string()],
            None,
            false,
        )
        .expect("test: transcript_from_asr_utterances should succeed");
        let chat = build_chat(&desc).unwrap();
        let serialized = to_chat_string(&chat);

        // Should contain filled pause marker and retrace
        assert!(
            serialized.contains("&-um"),
            "expected filled pause: {serialized}"
        );
        assert!(
            serialized.contains("[/]"),
            "expected retrace marker: {serialized}"
        );

        let (_parsed, errors) = parse_lenient(&parser, &serialized);
        assert!(
            errors.is_empty(),
            "disfluency+retrace should reparse cleanly: {errors:?}\noutput: {serialized}"
        );
    }

    // ── a user 2026-04-02 bug reports ────────────────────────────────

    /// Bug 1: @Media header must have comma+space between name and type.
    /// a user saw: `@Media: 279home-2audio`
    /// Expected:  `@Media: 279home-2, audio`
    #[test]
    fn media_header_has_comma_separator() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: Some("279home-2.mp3".to_string()),
            media_type: Some("audio".to_string()),
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![wd("hello", None, None), wd(".", None, None)]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };
        let chat = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat);
        assert!(
            output.contains("@Media:\t279home-2, audio"),
            "@Media must have 'name, type' format with comma+space, got: {}",
            output
                .lines()
                .find(|l| l.contains("@Media"))
                .unwrap_or("(no @Media)")
        );
    }

    /// Bug 2: When transcript_from_asr_utterances generates participants
    /// from speaker IDs, it must set name and role from the speaker code.
    /// PAR → Participant, INV → Investigator. No silent defaults.
    ///
    /// a user saw: `PAR Participant Participant, INV Participant Participant`
    /// Expected:   `PAR Participant , INV Investigator`
    ///
    /// The root cause is transcript_from_asr_utterances setting name=None,
    /// role=None and build_chat silently defaulting both to "Participant".
    #[test]
    fn asr_pipeline_sets_correct_participant_roles() {
        use asr_postprocess::{AsrWord, SpeakerIndex, Utterance};

        let utterances = vec![
            Utterance {
                speaker: SpeakerIndex(0),
                words: vec![
                    AsrWord::new("hello", Some(0), Some(500)),
                    AsrWord::new(".", None, None),
                ],
                lang: None,
            },
            Utterance {
                speaker: SpeakerIndex(1),
                words: vec![
                    AsrWord::new("world", Some(600), Some(1000)),
                    AsrWord::new(".", None, None),
                ],
                lang: None,
            },
        ];
        let participant_ids = vec!["PAR0".to_string(), "PAR1".to_string()];
        let desc = transcript_from_asr_utterances(
            &utterances,
            &participant_ids,
            &["eng".to_string()],
            None,
            false,
        )
        .expect("test: transcript_from_asr_utterances should succeed");

        // Both speakers get generic codes and "Participant" role
        let par0 = desc
            .participants
            .iter()
            .find(|p| p.id == "PAR0")
            .expect("must have PAR0 participant");
        assert_eq!(par0.role, "Participant");
        assert_eq!(par0.name, None, "name should be None (not doubled as role)");

        let par1 = desc
            .participants
            .iter()
            .find(|p| p.id == "PAR1")
            .expect("must have PAR1 participant");
        assert_eq!(par1.role, "Participant");

        // Verify serialization: "PAR0 Participant, PAR1 Participant"
        // No doubled role words
        let chat = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat);
        let participants_line = output
            .lines()
            .find(|l| l.starts_with("@Participants:"))
            .expect("must have @Participants");
        assert!(
            participants_line.contains("PAR0 Participant"),
            "got: {participants_line}"
        );
        assert!(
            participants_line.contains("PAR1 Participant"),
            "got: {participants_line}"
        );
        assert!(
            !participants_line.contains("Participant Participant"),
            "@Participants must NOT double the role word, got: {participants_line}"
        );
    }

    /// Bug 4: @Comment must include "DO NOT USE" and use commit hash, not version.
    #[test]
    fn transcribe_comment_includes_do_not_use() {
        let desc = TranscriptDescription {
            langs: vec!["eng".to_string()],
            participants: vec![ParticipantDesc {
                id: "PAR".to_string(),
                name: None,
                role: "Participant".to_string(),
                corpus: String::new(),
            }],
            media_name: None,
            media_type: None,
            utterances: vec![UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![wd("hello", None, None), wd(".", None, None)]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            }],
            write_wor: false,
        };
        let chat = build_chat(&desc).unwrap();
        let output = to_chat_string(&chat);
        // Find the "Unchecked output" comment
        if let Some(comment) = output.lines().find(|l| l.contains("Unchecked output")) {
            assert!(
                comment.contains("DO NOT USE"),
                "@Comment with 'Unchecked output' must include 'DO NOT USE', got: {comment}"
            );
        }
        // Version should not be a semver like "0.1.0" — should be commit hash or omitted
        for line in output.lines() {
            if line.starts_with("@Comment:") && line.contains("Batchalign") {
                assert!(
                    !line.contains("0.1.0") && !line.contains("1.0.0"),
                    "@Comment must not contain hardcoded semver, got: {line}"
                );
            }
        }
    }

    // ── End-to-end transcribe-pipeline invariants ────────────────────────────
    //
    // These tests drive the full pipeline (`process_raw_asr` →
    // `transcript_from_asr_utterances` → `build_chat` → `to_chat_string`)
    // and pin user-visible casing and retrace-shape invariants in the
    // serialized CHAT output — the boundary that no earlier unit test
    // reaches.

    /// Build a single-speaker `AsrOutput` from `(text, start_secs, end_secs)`
    /// tuples, treating `.`/`?`/`!` as punctuation elements.
    fn single_speaker_asr_output(tokens: &[(&str, f64, f64)]) -> asr_postprocess::AsrOutput {
        let elements: Vec<asr_postprocess::AsrElement> = tokens
            .iter()
            .map(|(text, start, end)| {
                let kind = if matches!(*text, "." | "?" | "!") {
                    asr_postprocess::AsrElementKind::Punctuation
                } else {
                    asr_postprocess::AsrElementKind::Text
                };
                asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new(*text),
                    ts: asr_postprocess::AsrTimestampSecs(*start),
                    end_ts: asr_postprocess::AsrTimestampSecs(*end),
                    kind,
                }
            })
            .collect();
        asr_postprocess::AsrOutput {
            monologues: vec![asr_postprocess::AsrMonologue {
                speaker: asr_postprocess::SpeakerIndex(0),
                elements,
            }],
        }
    }

    /// Drive the full ASR → CHAT pipeline for a single-speaker input and
    /// return the typed `TranscriptDescription` (pre-serialization).
    fn run_transcribe_to_description(
        tokens: &[(&str, f64, f64)],
        lang: &str,
    ) -> Result<TranscriptDescription, TranscriptBuildError> {
        let output = single_speaker_asr_output(tokens);
        let utts = asr_postprocess::process_raw_asr(&output, lang);
        transcript_from_asr_utterances(
            &utts,
            &["PAR1".to_string()],
            &[lang.to_string()],
            None,
            false,
        )
    }

    /// Drive the full ASR → CHAT pipeline for a single-speaker input and
    /// return the serialized CHAT output.
    fn run_transcribe_pipeline(tokens: &[(&str, f64, f64)], lang: &str) -> String {
        let desc = run_transcribe_to_description(tokens, lang)
            .expect("test: transcript_from_asr_utterances should succeed");
        let chat = build_chat(&desc).unwrap();
        to_chat_string(&chat)
    }

    /// The English pronoun "I" and its contractions ("I'm", "I'd") must be
    /// preserved uppercase in the serialized output.
    #[test]
    fn user_report_english_pronoun_i_preserved_from_rev_ai() {
        let output = run_transcribe_pipeline(
            &[
                ("this", 0.0, 0.2),
                ("is", 0.2, 0.4),
                ("not", 0.4, 0.6),
                ("where", 0.6, 0.8),
                ("I", 0.8, 0.9),
                ("grew", 0.9, 1.1),
                ("up", 1.1, 1.3),
                (".", 1.3, 1.4),
                ("I'm", 1.5, 1.7),
                ("sure", 1.7, 1.9),
                ("I'd", 1.9, 2.1),
                ("know", 2.1, 2.3),
                (".", 2.3, 2.4),
            ],
            "eng",
        );
        assert!(
            output.contains(" I "),
            "standalone English pronoun 'I' must be preserved uppercase: {output}"
        );
        assert!(
            output.contains("I'm"),
            "English contraction 'I'm' must be preserved uppercase: {output}"
        );
        assert!(
            output.contains("I'd"),
            "English contraction 'I'd' must be preserved uppercase: {output}"
        );
        assert!(
            !output.contains(" i "),
            "lowercase standalone 'i' must NOT appear in English CHAT output: {output}"
        );
    }

    /// Three adjacent identical unigrams serialize as `w [/] w [/] w`,
    /// not as the phrase form `<w w> [/] w`.
    ///
    /// The utterance-initial-cap rule (2026-04-23) uppercases the
    /// first non-retrace word in an English utterance, so the
    /// emitted form is `a [/] a [/] A` — retrace copies preserve the
    /// speaker's original lowercase `a`s, and the final "real"
    /// word gets the sentence-initial capitalization.
    #[test]
    fn user_report_single_word_triple_repetition_emits_separate_retraces() {
        let output = run_transcribe_pipeline(
            &[
                ("a", 0.0, 0.2),
                ("a", 0.2, 0.4),
                ("a", 0.4, 0.6),
                (".", 0.6, 0.7),
            ],
            "eng",
        );
        assert!(
            output.contains("a [/] a [/] A"),
            "triple single-word repetition must emit 'a [/] a [/] A': {output}"
        );
        assert!(
            !output.contains("<a a>"),
            "must NOT emit the multi-word group form '<a a>' for a unigram repetition: {output}"
        );
    }

    /// Proper nouns supplied uppercase by the ASR provider are preserved
    /// through the pipeline. (Promoting sentence-initial function words
    /// like "well"/"and"/"of" is out of scope — handled elsewhere.)
    #[test]
    fn user_report_proper_nouns_preserved_from_rev_ai() {
        let output = run_transcribe_pipeline(
            &[
                ("well", 0.0, 0.2),
                ("I", 0.2, 0.3),
                ("hate", 0.3, 0.5),
                ("to", 0.5, 0.6),
                ("give", 0.6, 0.8),
                ("away", 0.8, 1.0),
                ("my", 1.0, 1.1),
                ("age", 1.1, 1.3),
                ("Sarah", 1.3, 1.6),
                (".", 1.6, 1.7),
                ("I", 1.8, 1.9),
                ("live", 1.9, 2.1),
                ("in", 2.1, 2.2),
                ("Cincinnati", 2.2, 2.6),
                (".", 2.6, 2.7),
            ],
            "eng",
        );
        assert!(
            output.contains("Sarah"),
            "proper noun 'Sarah' must be preserved uppercase: {output}"
        );
        assert!(
            output.contains("Cincinnati"),
            "proper noun 'Cincinnati' must be preserved uppercase: {output}"
        );
        assert!(
            !output.contains("sarah"),
            "lowercase 'sarah' must NOT appear (proper noun): {output}"
        );
        assert!(
            !output.contains("cincinnati"),
            "lowercase 'cincinnati' must NOT appear (proper noun): {output}"
        );
    }

    // -------------------------------------------------------------------
    // RED — Fundamental B witness + the reporter end-to-end canary
    //
    // Fundamental A (enforce validation at `ChatWordText` construction)
    // is expressed in `asr_postprocess/asr_types.rs::tests`. Once A
    // goes green, most symptom-level tests that constructed
    // Rev.AI-shaped `AsrOutput`s containing `%` and expected parse-clean
    // CHAT become redundant — `ChatWordText::try_from` will refuse
    // them before `build_chat` is even reached.
    //
    // Two tests remain at this layer because they exercise properties
    // A alone does not cover:
    //
    //   * `red_fund_b_digit_hyphenated_eng_emits_no_bare_digits` — the
    //     end-to-end witness that `process_raw_asr` (Fundamental B)
    //     respects the language-aware variant of the `ChatWordText`
    //     invariant. The digit-hyphenated token `17-year-old` is
    //     structurally legal (tree-sitter accepts it) but fails E220
    //     for eng. This is the cleanest forcing function for the
    //     language-aware construction policy and acts as B's witness.
    //
    //   * `red_reporter_c465e6e8_97c_repro_fixture` — the
    //     "The reporter's bug stays fixed" regression canary. Three authentic
    //     Rev.AI tokens with real timestamps; any future regression in
    //     either A (structural) or B (pipeline postcondition) will
    //     reopen this test.
    //
    // Four `%`-only symptom tests that previously lived here were
    // deleted in the 2026-04-22 RED-suite sharpening pass; they
    // duplicated the invariant Fundamental A expresses cleanly. See
    // §4 for the triage rationale.
    // -------------------------------------------------------------------

    /// Build a one-speaker AsrOutput from (value, start_s, end_s) triples.
    /// Small helper kept for the user canary; re-used if additional
    /// end-to-end B-witnesses are added later.
    fn asr_single_speaker(elements: &[(&str, f64, f64)]) -> asr_postprocess::AsrOutput {
        asr_postprocess::AsrOutput {
            monologues: vec![asr_postprocess::AsrMonologue {
                speaker: asr_postprocess::SpeakerIndex(0),
                elements: elements
                    .iter()
                    .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new(*v),
                        ts: asr_postprocess::AsrTimestampSecs(*ts),
                        end_ts: asr_postprocess::AsrTimestampSecs(*end_ts),
                        kind: asr_postprocess::AsrElementKind::Text,
                    })
                    .collect(),
            }],
        }
    }

    /// Run the full Rev.AI-facing chain: process_raw_asr → transcript builder
    /// → build_chat → serialize → re-parse. Returns the serialized CHAT and
    /// the parse errors the lenient parser emits on the re-parse.
    fn asr_to_chat_roundtrip(
        output: &asr_postprocess::AsrOutput,
        lang: &str,
    ) -> (String, Vec<talkbank_model::ParseError>) {
        let utts = asr_postprocess::process_raw_asr(output, lang);
        let desc = transcript_from_asr_utterances(
            &utts,
            &["PAR".to_string()],
            &[lang.to_string()],
            None,
            false,
        )
        .expect("test: transcript_from_asr_utterances should succeed");
        let chat = build_chat(&desc).expect("build_chat should not fail on normalized input");
        let serialized = to_chat_string(&chat);
        let parser = TreeSitterParser::new().expect("grammar loads");
        let (_chat, errors) = crate::parse::parse_lenient(&parser, &serialized);
        (serialized, errors)
    }

    #[test]
    fn red_fund_b_digit_hyphenated_eng_emits_no_bare_digits() {
        // Reproduces c465e6e8-97c line 669:
        //   *PAR1: And my 17-year-old he wants to go to Harvard .
        // tree-sitter accepts `17-year-old` as a single hyphenated word, so
        // this does NOT fail L0 parse. It fails E220 at talkbank-model
        // validation ("numeric digits not allowed" for eng). The
        // sanitization contract here is: digit-bearing tokens must be
        // normalized (spelled out or segmented) before emission to CHAT
        // for languages where E220 applies.
        let output = asr_single_speaker(&[
            ("my", 0.0, 0.2),
            ("17-year-old", 1341.565, 1342.245),
            ("son", 0.5, 0.7),
            (".", 0.7, 0.7),
        ]);
        let (serialized, parse_errors) = asr_to_chat_roundtrip(&output, "eng");
        assert!(
            parse_errors.is_empty(),
            "re-parse must produce no structural errors; got:\n{parse_errors:#?}\n\
             serialized:\n{serialized}"
        );

        // Re-parse and run the full validator under an eng context. This is
        // the true E220 check — it reads the ChatFile's word AST, applies
        // language-aware digit rules per word, and collects any violations.
        // Using the real machinery instead of string-matching digits makes
        // the test robust against surface-level artifacts such as utterance
        // timing suffixes (`0_700`) that are part of the CHAT line but not
        // part of word content.
        let parser = TreeSitterParser::new().expect("grammar loads");
        let (chat, _) = crate::parse::parse_lenient(&parser, &serialized);
        let validation_errors = talkbank_model::ErrorCollector::new();
        chat.validate(&validation_errors, None);
        let validation_errs = validation_errors.into_vec();
        let e220s: Vec<_> = validation_errs
            .iter()
            .filter(|e| e.code.as_str() == "E220")
            .collect();
        assert!(
            e220s.is_empty(),
            "emitted CHAT must not fire E220 (numeric digits not allowed in eng) \
             on any word. Found {} E220 error(s):\n{e220s:#?}\nserialized:\n{serialized}",
            e220s.len()
        );
    }

    #[test]
    fn red_reporter_c465e6e8_97c_end_to_end_canary() {
        // Full regression fixture: the exact three offending Rev.AI tokens
        // from a reporter's failing job c465e6e8-97c (file 545.mp4), run through
        // the same Rev.AI-facing chain that `transcribe` uses. Timings are
        // the authentic Rev.AI values captured at
        //   the private bug-repro fixture
        //     offending_asr_tokens.json
        // Context tokens surrounding each offender are synthesized so each
        // utterance has a terminator; only the three problematic tokens are
        // from the Rev.AI wire response.
        let output = asr_postprocess::AsrOutput {
            monologues: vec![
                asr_postprocess::AsrMonologue {
                    // Speaker 1: "And my 17-year-old son ."
                    speaker: asr_postprocess::SpeakerIndex(1),
                    elements: [
                        ("and", 1340.4, 1340.6),
                        ("my", 1340.6, 1340.8),
                        ("17-year-old", 1341.565, 1342.245),
                        ("son", 1342.3, 1342.5),
                        (".", 1343.685, 1343.685),
                    ]
                    .iter()
                    .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new(*v),
                        ts: asr_postprocess::AsrTimestampSecs(*ts),
                        end_ts: asr_postprocess::AsrTimestampSecs(*end_ts),
                        kind: asr_postprocess::AsrElementKind::Text,
                    })
                    .collect(),
                },
                asr_postprocess::AsrMonologue {
                    // Speaker 0: "remember 80% of it ."
                    speaker: asr_postprocess::SpeakerIndex(0),
                    elements: [
                        ("remember", 1774.1, 1774.5),
                        ("80%", 1774.765, 1775.405),
                        ("of", 1775.405, 1775.5),
                        ("it", 1775.5, 1775.645),
                        (".", 1775.645, 1775.645),
                    ]
                    .iter()
                    .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new(*v),
                        ts: asr_postprocess::AsrTimestampSecs(*ts),
                        end_ts: asr_postprocess::AsrTimestampSecs(*end_ts),
                        kind: asr_postprocess::AsrElementKind::Text,
                    })
                    .collect(),
                },
                asr_postprocess::AsrMonologue {
                    // Speaker 0: "that other 20% ."
                    speaker: asr_postprocess::SpeakerIndex(0),
                    elements: [
                        ("that", 1775.9, 1776.3),
                        ("other", 1776.3, 1776.8),
                        ("20%", 1776.825, 1777.685),
                        (".", 1777.685, 1777.685),
                    ]
                    .iter()
                    .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                        value: asr_postprocess::AsrRawText::new(*v),
                        ts: asr_postprocess::AsrTimestampSecs(*ts),
                        end_ts: asr_postprocess::AsrTimestampSecs(*end_ts),
                        kind: asr_postprocess::AsrElementKind::Text,
                    })
                    .collect(),
                },
            ],
        };

        let utts = asr_postprocess::process_raw_asr(&output, "eng");
        let desc = transcript_from_asr_utterances(
            &utts,
            &["PAR0".to_string(), "PAR1".to_string()],
            &["eng".to_string()],
            Some("545"),
            false,
        )
        .expect("test: transcript_from_asr_utterances should succeed");
        let chat = build_chat(&desc).expect("build_chat should not fail");
        let serialized = to_chat_string(&chat);
        let parser = TreeSitterParser::new().expect("grammar loads");
        let (_chat, errors) = crate::parse::parse_lenient(&parser, &serialized);
        assert!(
            errors.is_empty(),
            "c465e6e8-97c fixture must reparse with zero parse errors \
             (currently fails with E316 on `80%`/`20%`); got {} error(s):\n{:#?}\n\
             serialized:\n{serialized}",
            errors.len(),
            errors
        );
        assert!(
            !serialized.contains('%'),
            "a reporter fixture: emitted CHAT must not contain bare `%`; \
             serialized:\n{serialized}"
        );
    }

    // ── 2026-04-23 transcribe-pipeline corrections end-to-end ──

    /// All three 2026-04-23 English transcribe rules fire in the
    /// full in-process transcribe pipeline (`process_raw_asr` →
    /// `build_chat`): bare `i` uppercases, `Dr.` loses its period,
    /// utterance-initial words get capitalized. Evidence that the
    /// rules are wired into `finalize_utterances` and not just
    /// unit-tested in isolation.
    #[test]
    fn english_transcribe_rules_fire_end_to_end() {
        let output = run_transcribe_pipeline(
            &[
                ("hello", 0.0, 0.3),
                (".", 0.3, 0.4),
                ("i", 0.5, 0.6),
                ("said", 0.6, 0.9),
                ("Dr.", 0.9, 1.2),
                ("Smith", 1.2, 1.5),
                ("arrived", 1.5, 1.9),
                (".", 1.9, 2.0),
                ("i'll", 2.1, 2.4),
                ("see", 2.4, 2.6),
                ("him", 2.6, 2.8),
                (".", 2.8, 2.9),
            ],
            "eng",
        );
        // Utterance 1: `Hello .` (utterance-initial cap).
        assert!(
            output.contains("Hello ."),
            "first utterance must be capitalized `Hello .`: {output}"
        );
        // Utterance 2: `I said Dr Smith arrived .` (I-cap on `i`,
        // period-strip on `Dr.`). `I` is already capitalized by
        // I-cap, so utterance-initial cap is a no-op there.
        assert!(
            output.contains("I said Dr Smith arrived ."),
            "second utterance must show I-cap + period-strip: {output}"
        );
        // Utterance 3: `I'll see him .` (I-cap on contraction).
        assert!(
            output.contains("I'll see him ."),
            "third utterance must show I-cap on contraction: {output}"
        );
        // Negative assertions — the rules must NOT have fired on
        // unrelated material.
        assert!(
            !output.contains(" i "),
            "bare `i` must have been uppercased: {output}"
        );
        assert!(
            !output.contains("Dr."),
            "title `Dr.` must have lost its period: {output}"
        );
    }

    /// Non-English input is untouched by the 2026-04-23 English
    /// rules. Language gate is the sole guard.
    #[test]
    fn english_transcribe_rules_skip_other_languages() {
        let output = run_transcribe_pipeline(
            &[
                ("ho", 0.0, 0.2),
                ("visto", 0.2, 0.5),
                ("i", 0.5, 0.6),
                ("bambini", 0.6, 1.0),
                (".", 1.0, 1.1),
            ],
            "ita",
        );
        // Italian `ho` must NOT be capitalized; Italian `i` (plural
        // masculine article) must NOT be uppercased.
        assert!(
            output.contains("ho visto i bambini ."),
            "Italian output must be untouched by English rules: {output}"
        );
        assert!(
            !output.contains(" I "),
            "Italian `i` must not be uppercased: {output}"
        );
    }
}
