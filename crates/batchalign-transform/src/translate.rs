//! Translation helpers for the server-side translate orchestrator.
//!
//! Owns both ends of a translation: [`TranslationSource`], what one utterance
//! produced, as a typed model rather than a joined string, and
//! [`TranslationText`], a translation with something to apply, injected as a
//! `%xtra` dependent tier.
//!
//! # What is translated
//!
//! What was spoken: every word the speaker produced, in transcript order,
//! retraced words and filled pauses included, followed by the utterance's
//! terminator when that terminator is readable punctuation. Batchalign 2 sent
//! the same words: both its engines called
//! `utterance.strip(join_with_spaces=False, include_retrace=True,
//! include_fp=True)` (`batchalign/pipelines/translate/gtrans.py` and
//! `seamless.py`), which detokenizes rather than joining with spaces.
//!
//! Which words those are, which punctuation travels, and how a terminator is
//! written are batchalign3's own rules, stated below as closed matches over
//! chatter's word categories, separators and terminators. The text an engine
//! receives is produced in exactly one place,
//! [`TranslationSource::render`], so no call site joins words itself.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::Span;
use talkbank_model::model::content::{Separator, Terminator};
use talkbank_model::model::{
    ChatFile, DependentTier, LanguageCode, Line, NonEmptyString, UserDefinedDependentTier,
    Utterance, Word, WordCategory,
};

// ---------------------------------------------------------------------------
// Wire type (matches Python's TranslateBatchItem)
// ---------------------------------------------------------------------------

/// Input payload for a single translation request.
///
/// The text is what [`TranslationSource::render`] wrote: the words the speaker
/// produced, in order, and the utterance's terminator. It is never assembled
/// at a call site. Matches the Python `TranslateBatchItem` wire format.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TranslateBatchItem {
    /// Source-language text to translate: the rendered translation source.
    pub text: String,
}

// ---------------------------------------------------------------------------
// The translation source: what one utterance produced
// ---------------------------------------------------------------------------

/// One word as a translation engine should see it.
///
/// The only route in is [`TranslatableWordText::of_produced_word`], so which
/// words reach an engine is decided once rather than at each call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslatableWordText(String);

impl TranslatableWordText {
    /// The cleaned text of a word the speaker produced, or `None` when there
    /// is nothing for a translator to read.
    ///
    /// Batchalign 3's rule, written as a match over chatter's closed
    /// [`WordCategory`] so a category added later has to state its own answer
    /// instead of inheriting one: ordinary words and filled pauses are read
    /// aloud and are sent (a filled pause without its `&-` prefix), while a
    /// word recorded as not said, a rendering of a noise, and the untranscribed
    /// markers `xxx` / `yyy` / `www` are not language a translator can use.
    /// That filled pauses and retraces travel at all is batchalign2's
    /// behaviour, from the `include_retrace` and `include_fp` arguments its
    /// translate engines passed to `strip`.
    fn of_produced_word(word: &Word) -> Option<Self> {
        // `xxx` / `yyy` / `www`: something was said but not transcribed, so
        // there are no words to translate.
        if word.untranscribed().is_some() {
            return None;
        }
        match &word.category {
            // Ordinary orthography, and a filler, which is a produced sound.
            None | Some(WordCategory::Filler) => {}
            // Recorded as NOT said: there is nothing spoken to translate.
            Some(WordCategory::Omission | WordCategory::CAOmission) => return None,
            // A rendering of a noise rather than a spelling of a word.
            Some(WordCategory::Nonword | WordCategory::PhonologicalFragment) => return None,
        }
        let cleaned = word.cleaned_text();
        (!cleaned.is_empty()).then(|| Self(cleaned.to_owned()))
    }

    /// The text an engine receives for this word.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A separator that is readable punctuation, and so travels with the words.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslatableSeparator(&'static str);

impl TranslatableSeparator {
    /// `Some` for a separator that is ordinary punctuation in written
    /// language, `None` for CHAT-only marks.
    ///
    /// The comma is punctuation any translator reads. The tag marker and the
    /// vocative are written `„` and `‡`, and the CA marks (`[^c]`, the
    /// intonation arrows, `≡`, `≈`) describe delivery: each would arrive at
    /// the engine as a symbol to translate or echo, so none is sent. Closed,
    /// like the per-word and terminator rules, so a separator added later has
    /// to state its own answer.
    fn of_separator(separator: &Separator) -> Option<Self> {
        match separator {
            Separator::Comma { .. } => Some(Self(",")),
            Separator::Semicolon { .. }
            | Separator::Colon { .. }
            | Separator::Tag { .. }
            | Separator::Vocative { .. }
            | Separator::CaContinuation { .. }
            | Separator::UnmarkedEnding { .. }
            | Separator::Uptake { .. }
            | Separator::CaNoBreak { .. }
            | Separator::CaTechnicalBreak { .. }
            | Separator::RisingToHigh { .. }
            | Separator::RisingToMid { .. }
            | Separator::Level { .. }
            | Separator::FallingToMid { .. }
            | Separator::FallingToLow { .. } => None,
        }
    }

    /// The punctuation an engine receives for this separator.
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

/// One item of a [`TranslationSource`], in transcript order.
#[derive(Debug, Clone, PartialEq)]
pub enum TranslationUnit {
    /// A word that was produced.
    Word(TranslatableWordText),
    /// A separator written between words.
    Separator(TranslatableSeparator),
}

/// Whether the utterance ends with a terminator, and which one.
///
/// A variant rather than an `Option`, because the renderer must state what it
/// does in both cases: a CHAT file written under `@Options: CA` may have no
/// terminator, and that is a state, not a missing value.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceTerminator {
    /// The utterance's terminator, which is sent with the words.
    Terminated(Terminator),
    /// The utterance has no terminator, so none is sent.
    Unterminated,
}

/// U+3002, the full stop of Han script.
const IDEOGRAPHIC_FULL_STOP: &str = "\u{3002}";

/// Every string [`render_terminator`] can emit.
///
/// One list read from both ends: the renderer emits only these, and
/// [`TranslationText::admit`] refuses a translation that is nothing but one of
/// them, so an engine echoing the punctuation we sent cannot be written to a
/// `%xtra` tier.
const RENDERED_TERMINATORS: [&str; 4] = [".", "?", "!", IDEOGRAPHIC_FULL_STOP];

/// The languages batchalign3's language table records as written in Han script.
///
/// The same varieties the request layer lists for the Chinese-capable ASR and
/// timing-recovery engines (`crates/batchalign/src/types/request.rs`), which
/// answers a different question (engine coverage) about the same set. Kept
/// separate for that reason: engine coverage can change without the writing
/// system changing.
const HAN_SCRIPT_LANGUAGES: [&str; 6] = ["zho", "cmn", "yue", "wuu", "nan", "hak"];

/// The writing system a source language uses, which is what decides how the
/// units of a [`TranslationSource`] are joined.
///
/// A property of the language rather than a list of special cases at the
/// renderer: Han script has no word spaces and its own full stop, whoever is
/// writing it, so every Chinese variety in the table behaves the same way
/// instead of the two codes that happened to be named first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritingSystem {
    /// Han script (Chinese varieties): no spaces between words, and a period
    /// is written as the ideographic full stop.
    Han,
    /// A script whose words are separated by spaces.
    Alphabetic,
}

impl WritingSystem {
    /// The writing system `lang` is written in.
    pub fn of_language(lang: &LanguageCode) -> Self {
        Self::of_language_code(lang.as_str())
    }

    /// The writing system an ISO 639-3 code is written in.
    ///
    /// The same answer as [`Self::of_language`], for a caller holding a
    /// different language newtype. Both delegate here so
    /// [`HAN_SCRIPT_LANGUAGES`] stays the ONE list: the ASR control plane
    /// needs this question answered about a `LanguageCode3` when it chooses a
    /// provider's model, and a second copy of the Chinese varieties there is
    /// exactly how the Python side came to carry a five-code list that omitted
    /// Mandarin.
    pub fn of_language_code(lang: &str) -> Self {
        if HAN_SCRIPT_LANGUAGES.contains(&lang) {
            Self::Han
        } else {
            Self::Alphabetic
        }
    }
}

/// What one utterance produced, in the form a translation engine receives it.
///
/// Built only by [`TranslationSource::of_utterance`], which walks the
/// utterance with chatter's word walk over no tier domain, so retraced words
/// (`<I like> [/]`) and filled pauses are included, as they were in
/// batchalign2's `strip` call. A source always holds at least one word.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationSource {
    units: Vec<TranslationUnit>,
    terminator: SourceTerminator,
}

impl TranslationSource {
    /// The source for one utterance, or `None` when it produced no words and
    /// so has nothing to translate.
    ///
    /// A word the transcriber replaced (`hafta [: have to]`) contributes its
    /// REPLACEMENT: the replacement is the standard-language form of what was
    /// said, and it is what a translator can actually read. That choice is
    /// batchalign3's own; the recorded batchalign2 comparison does not cover
    /// replacements.
    fn of_utterance(utterance: &Utterance) -> Option<Self> {
        use talkbank_model::alignment::helpers::{WordItem, walk_words};

        let mut units = Vec::new();
        let push_word = |word: &Word, units: &mut Vec<TranslationUnit>| {
            if let Some(text) = TranslatableWordText::of_produced_word(word) {
                units.push(TranslationUnit::Word(text));
            }
        };
        walk_words(
            &utterance.main.content.content,
            None,
            &mut |item| match item {
                WordItem::Word(word) => push_word(word, &mut units),
                WordItem::ReplacedWord(replaced) => {
                    for word in replaced.replacement.words.iter() {
                        push_word(word, &mut units);
                    }
                }
                WordItem::Separator(separator) => {
                    if let Some(separator) = TranslatableSeparator::of_separator(separator) {
                        units.push(TranslationUnit::Separator(separator));
                    }
                }
            },
        );

        units
            .iter()
            .any(|unit| matches!(unit, TranslationUnit::Word(_)))
            .then(|| Self {
                terminator: match &utterance.main.content.terminator {
                    Some(terminator) => SourceTerminator::Terminated(terminator.clone()),
                    None => SourceTerminator::Unterminated,
                },
                units,
            })
    }

    /// Render the payload one engine receives.
    ///
    /// THE one place source text is built. Nothing that is CHAT notation
    /// rather than readable text reaches the engine: a separator or terminator
    /// with no ordinary-language spelling contributes nothing.
    pub fn render(&self, script: WritingSystem) -> TranslateBatchItem {
        let mut text = String::new();
        for unit in &self.units {
            match unit {
                TranslationUnit::Word(word) => {
                    if !text.is_empty() && script == WritingSystem::Alphabetic {
                        text.push(' ');
                    }
                    text.push_str(word.as_str());
                }
                // Punctuation attaches to the word before it, as it is written.
                TranslationUnit::Separator(separator) => {
                    text.push_str(separator.as_str());
                }
            }
        }
        match &self.terminator {
            SourceTerminator::Terminated(terminator) => {
                if let Some(rendered) = render_terminator(terminator, script) {
                    text.push_str(rendered);
                }
            }
            SourceTerminator::Unterminated => {}
        }
        TranslateBatchItem { text }
    }
}

/// How a terminator is written for an engine, or `None` when it has no
/// readable spelling.
///
/// A closed mapping, in the same shape as the per-word rule: only the three
/// terminators that ARE ordinary punctuation are sent, the question-bearing
/// CHAT variants (`+/?`, `+!?`, `+//?`, `+..?`) as a question mark. The rest
/// (`+...`, `+/.`, `+//.`, `+"/.`, `+".`, `+.`) are CHAT notation, not
/// language: sending them gave the engine a token to translate or echo, and an
/// echo of one used to be written straight into `%xtra`. Han script writes the
/// period as the ideographic full stop, which post-processing maps back to `.`;
/// the question and exclamation marks are sent as written, which is what
/// batchalign2's Chinese branch did with everything but the period (its Google
/// engine replaced spaces and `.` only).
fn render_terminator(terminator: &Terminator, script: WritingSystem) -> Option<&'static str> {
    match terminator {
        Terminator::Period { .. } => Some(match script {
            WritingSystem::Han => IDEOGRAPHIC_FULL_STOP,
            WritingSystem::Alphabetic => ".",
        }),
        Terminator::Question { .. }
        | Terminator::InterruptedQuestion { .. }
        | Terminator::BrokenQuestion { .. }
        | Terminator::SelfInterruptedQuestion { .. }
        | Terminator::TrailingOffQuestion { .. } => Some("?"),
        Terminator::Exclamation { .. } => Some("!"),
        Terminator::TrailingOff { .. }
        | Terminator::Interruption { .. }
        | Terminator::SelfInterruption { .. }
        | Terminator::QuotedNewLine { .. }
        | Terminator::QuotedPeriodSimple { .. }
        | Terminator::BreakForCoding { .. } => None,
    }
}

// ---------------------------------------------------------------------------
// Payload collection
// ---------------------------------------------------------------------------

/// Collect translate payloads from all utterances in a ChatFile.
///
/// Returns `(line_idx, TranslationSource)` pairs. Utterances that produced no
/// words are skipped. `line_idx` is the index into `chat_file.lines` (needed
/// for injection).
pub fn collect_translate_payloads(chat_file: &ChatFile) -> Vec<(usize, TranslationSource)> {
    chat_file
        .lines
        .iter()
        .enumerate()
        .filter_map(|(line_idx, line)| {
            let Line::Utterance(utterance) = line else {
                return None;
            };
            TranslationSource::of_utterance(utterance).map(|source| (line_idx, source))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Injection
// ---------------------------------------------------------------------------

/// A translation that has something to apply, as the text that will be
/// written: trimmed, so the value and the rule that admitted it describe the
/// same string.
///
/// Its existence is the proof: a translation with no content cannot be built,
/// so no injection site has to check for one and none can forget to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationText(NonEmptyString);

/// An engine returned a translation with nothing to apply.
///
/// Batchalign 2 dropped these silently at injection: its CHAT serializer
/// (`batchalign/formats/chat/generator.py`) wrote a `%xtra` tier only when the
/// translation's text was not `""`, `"."`, `"!"` or `"?"`. Batchalign 3
/// refuses them instead, so the run says what happened rather than writing a
/// file with a tier missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the engine returned a translation with no content")]
pub struct EmptyTranslation;

impl TranslationText {
    /// Admit one engine translation, refusing text with nothing to apply:
    /// blank, or nothing but punctuation this crate itself sent
    /// ([`RENDERED_TERMINATORS`]). An engine that echoes the terminator, or
    /// answers an utterance it made nothing of with a bare `.`, produces no
    /// tier and a named failure instead.
    pub fn admit(translation: &str) -> Result<Self, EmptyTranslation> {
        let trimmed = translation.trim();
        if trimmed.is_empty() || RENDERED_TERMINATORS.contains(&trimmed) {
            return Err(EmptyTranslation);
        }
        NonEmptyString::new(trimmed)
            .map(Self)
            .map_err(|_| EmptyTranslation)
    }

    /// The text written to the `%xtra` tier.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Inject a translation as a `%xtra` dependent tier on an utterance.
///
/// Creates a `DependentTier::UserDefined` with label "xtra" and uses
/// `replace_or_add_tier` to inject it (replacing any existing `%xtra`).
/// Infallible: the content is an admitted [`TranslationText`].
pub fn inject_translation(utterance: &mut Utterance, translation: &TranslationText) {
    // A compile-time literal, non-empty where it is written, so there is no
    // runtime failure to report and no error path for a caller to mishandle.
    let label = NonEmptyString::new_unchecked("xtra");

    let new_tier = DependentTier::UserDefined(UserDefinedDependentTier {
        label,
        content: Some(translation.0.clone()),
        span: Span::DUMMY,
    });

    crate::inject::replace_or_add_tier(&mut utterance.dependent_tiers, new_tier);
}

// ---------------------------------------------------------------------------
// Result application
// ---------------------------------------------------------------------------

/// Apply translation results to a ChatFile.
///
/// `results` maps `line_idx` to an admitted translation. Lines whose indices
/// are not in the map are left unchanged. There is no failure to report: every
/// value is a [`TranslationText`], so injection cannot fail, and the
/// "failed to inject" warning this loop used to emit is gone with the states
/// that produced it.
pub fn apply_translate_results(
    chat_file: &mut ChatFile,
    results: &HashMap<usize, TranslationText>,
) {
    if results.is_empty() {
        return;
    }

    for (&line_idx, translation) in results {
        if let Some(Line::Utterance(utt)) = chat_file.lines.as_mut_slice().get_mut(line_idx) {
            inject_translation(utt, translation);
        }
    }
}

// ---------------------------------------------------------------------------
// Extraction (for caching after injection)
// ---------------------------------------------------------------------------

/// Entry for extracting `%xtra` tier content from a processed utterance.
pub struct TranslationStringsEntry {
    /// Index into `ChatFile.lines`.
    pub line_idx: usize,
    /// Extracted `%xtra` tier translation text.
    pub translation: String,
}

/// Extract `%xtra` tier content from specified utterances for caching.
pub fn extract_translation_strings(
    chat_file: &ChatFile,
    line_indices: &[usize],
) -> Vec<TranslationStringsEntry> {
    let mut results = Vec::with_capacity(line_indices.len());

    for &line_idx in line_indices {
        let Some(line) = chat_file.lines.get(line_idx) else {
            continue;
        };
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        for tier in &utt.dependent_tiers {
            if let DependentTier::UserDefined(ud) = &tier.tier
                && ud.label.as_ref() == "xtra"
            {
                results.push(TranslationStringsEntry {
                    line_idx,
                    translation: ud.content.as_deref().unwrap_or_default().to_string(),
                });
                break;
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Pre/post-processing (moved from Python translate.py)
// ---------------------------------------------------------------------------

/// Returns the CHAT punctuation characters used for translation spacing.
///
/// Includes both MOR separators (vocative ‡, tag „, comma ,) and terminators
/// (. ? ! +... +/. +/? etc.).
pub fn chat_punct_chars() -> Vec<String> {
    use talkbank_model::Span;
    use talkbank_model::model::content::{Separator, Terminator};

    let separators: Vec<String> = vec![
        Separator::Vocative { span: Span::DUMMY },
        Separator::Tag { span: Span::DUMMY },
        Separator::Comma { span: Span::DUMMY },
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();

    let terminators: Vec<String> = vec![
        Terminator::Period { span: Span::DUMMY },
        Terminator::Question { span: Span::DUMMY },
        Terminator::Exclamation { span: Span::DUMMY },
        Terminator::TrailingOff { span: Span::DUMMY },
        Terminator::Interruption { span: Span::DUMMY },
        Terminator::SelfInterruption { span: Span::DUMMY },
        Terminator::InterruptedQuestion { span: Span::DUMMY },
        Terminator::BrokenQuestion { span: Span::DUMMY },
        Terminator::QuotedNewLine { span: Span::DUMMY },
        Terminator::QuotedPeriodSimple { span: Span::DUMMY },
        Terminator::SelfInterruptedQuestion { span: Span::DUMMY },
        Terminator::TrailingOffQuestion { span: Span::DUMMY },
        Terminator::BreakForCoding { span: Span::DUMMY },
    ]
    .into_iter()
    .map(|t| t.to_string())
    .collect();

    [separators, terminators].concat()
}

/// Post-process raw translation output from the API.
///
/// Applies normalization:
/// - Ideographic full stop → period
/// - Curly quotes → straight quotes
/// - Zero-width spaces removed
/// - Tab → space
/// - Punctuation spacing (add space before each punct char)
pub fn postprocess_translation(raw: &str, punct_chars: &[&str]) -> String {
    // Single-pass normalization: char mapping + zero-width space removal
    let mut result: String = raw
        .chars()
        .filter(|&c| c != '\u{200b}') // zero-width space
        .map(|c| match c {
            '\u{3002}' => '.',               // ideographic full stop → period
            '\u{2018}' | '\u{2019}' => '\'', // curly quotes → straight
            '\t' => ' ',
            c => c,
        })
        .collect();

    for p in punct_chars {
        result = result.replace(p, &format!(" {p}"));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::WriteChat;
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).expect_built()
    }

    fn get_utterance_mut(chat: &mut ChatFile, idx: usize) -> &mut talkbank_model::model::Utterance {
        let mut utt_idx = 0;
        for line in &mut chat.lines {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt;
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    /// Build a CHAT file from `utterances`, one `*PAR:` line each.
    fn chat_with(lang: &str, utterances: &[&str]) -> ChatFile {
        let mut text = format!(
            "@UTF8\n@Begin\n@Languages:\t{lang}\n@Participants:\tPAR Participant\n\
             @ID:\t{lang}|test|PAR|||||Participant|||\n"
        );
        for utterance in utterances {
            text.push_str("*PAR:\t");
            text.push_str(utterance);
            text.push('\n');
        }
        text.push_str("@End\n");
        parse_chat(&text)
    }

    /// The text one utterance is sent as, in the writing system of `lang`.
    fn rendered(lang: &str, utterance: &str) -> Option<String> {
        let chat = chat_with(lang, &[utterance]);
        let payloads = collect_translate_payloads(&chat);
        payloads.first().map(|(_, source)| {
            source
                .render(WritingSystem::of_language(&language(lang)))
                .text
        })
    }

    fn language(code: &str) -> LanguageCode {
        LanguageCode::new(code).expect("valid test language code")
    }

    fn admitted(text: &str) -> TranslationText {
        TranslationText::admit(text).expect("test translation has content")
    }

    /// What was spoken is what is sent: every produced word in order, and the
    /// terminator when it is readable punctuation. Retraces and filled pauses
    /// travelling is the recorded batchalign2 behaviour, and the reason the
    /// translate parity goldens read `I like I like beans .` rather than
    /// `I like beans`.
    #[test]
    fn collected_source_sends_the_spoken_words_and_the_terminator() {
        assert_eq!(
            rendered("eng", "I like <I like> [/] beans .").as_deref(),
            Some("I like I like beans.")
        );
        assert_eq!(
            rendered("eng", "so I was like &-um yeah .").as_deref(),
            Some("so I was like um yeah.")
        );
        // A question reaches the engine as a question.
        assert_eq!(
            rendered("eng", "do you like beans ?").as_deref(),
            Some("do you like beans?")
        );
    }

    /// The per-word rule: omissions, nonwords, fragments and untranscribed
    /// markers are not things a translator can read, so none of them is sent.
    #[test]
    fn collected_source_leaves_out_what_was_not_produced_as_words() {
        assert_eq!(
            rendered("eng", "the 0det dog &~gaga &+fr xxx ran .").as_deref(),
            Some("the dog ran.")
        );
        // Nothing produced: no payload at all, so nothing is sent.
        assert_eq!(rendered("eng", "xxx ."), None);
    }

    /// A transcriber's replacement is the form an engine can read, so it is
    /// what travels.
    #[test]
    fn collected_source_sends_a_replacement_rather_than_the_replaced_form() {
        assert_eq!(
            rendered("eng", "I hafta [: have to] go .").as_deref(),
            Some("I have to go.")
        );
    }

    /// Readable punctuation travels with the words, attached the way it is
    /// written. CHAT-only marks do not: the tag marker and the vocative would
    /// arrive as `„` and `‡`.
    #[test]
    fn collected_source_keeps_readable_punctuation_only() {
        assert_eq!(
            rendered("eng", "hello , world .").as_deref(),
            Some("hello, world.")
        );
        assert_eq!(
            rendered("eng", "hello „ world .").as_deref(),
            Some("hello world.")
        );
    }

    /// Han script, whatever the variety: no spaces between words, and the
    /// period written as the ideographic full stop. `cmn` is Han script like
    /// `yue` and `zho`; it used to be left alphabetic by a two-code match.
    #[test]
    fn han_script_is_rendered_without_spaces_and_with_a_full_stop() {
        for han in ["yue", "zho", "cmn"] {
            assert_eq!(
                rendered(han, "你 好 .").as_deref(),
                Some("你好。"),
                "{han} is written in Han script"
            );
        }
        // Other writing systems keep their spaces and their own punctuation.
        assert_eq!(rendered("spa", "el gato .").as_deref(), Some("el gato."));
    }

    /// RED FIRST (review item 1): a terminator with no ordinary spelling sends
    /// nothing. It used to send its CHAT token, so `+...` and `+/.` reached
    /// the engine as text to translate.
    #[test]
    fn a_chat_only_terminator_sends_nothing() {
        assert_eq!(
            rendered("eng", "and then +...").as_deref(),
            Some("and then")
        );
        assert_eq!(
            rendered("eng", "I was going to +/.").as_deref(),
            Some("I was going to")
        );
        // A question-bearing CHAT terminator is still a question.
        assert_eq!(
            rendered("eng", "you were going to +/?").as_deref(),
            Some("you were going to?")
        );
        assert_eq!(
            rendered("eng", "stop that !").as_deref(),
            Some("stop that!")
        );
    }

    /// An engine result with nothing to apply has no representation, so no
    /// injection site can silently drop one. The refusal covers every string
    /// the renderer itself can emit, so an engine echoing our punctuation
    /// cannot become a tier.
    #[test]
    fn a_translation_with_no_content_is_refused() {
        for empty in ["", "   ", "\t", ".", "!", "?", "  .  ", "。"] {
            assert_eq!(
                TranslationText::admit(empty),
                Err(EmptyTranslation),
                "{empty:?} has nothing to apply"
            );
        }
        // Admitted text is the text that gets written: trimmed.
        assert_eq!(admitted("  hola .  ").as_str(), "hola .");
    }

    #[test]
    fn test_inject_translation() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_female.cha");
        let mut chat = parse_chat(chat_text);
        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, &admitted("hola"));

        let output = chat.to_chat_string();
        assert!(output.contains("%xtra:\thola"), "Output: {output}");
    }

    #[test]
    fn test_inject_translation_replaces_existing() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_with_xtra.cha");
        let mut chat = parse_chat(chat_text);

        let output_before = chat.to_chat_string();
        assert!(
            output_before.contains("old translation"),
            "Before: {output_before}"
        );

        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, &admitted("new translation"));

        let output = chat.to_chat_string();
        assert!(output.contains("new translation"), "After: {output}");
        assert!(
            !output.contains("old translation"),
            "Old should be gone: {output}"
        );
    }

    #[test]
    fn test_apply_translate_results() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_goodbye.cha");
        let mut chat = parse_chat(chat_text);

        let payloads = collect_translate_payloads(&chat);
        assert_eq!(payloads.len(), 2);
        let line_idx_0 = payloads[0].0;
        let line_idx_1 = payloads[1].0;

        let mut results = HashMap::new();
        results.insert(line_idx_0, admitted("hola"));
        results.insert(line_idx_1, admitted("adiós"));

        apply_translate_results(&mut chat, &results);

        let output = chat.to_chat_string();
        assert!(output.contains("%xtra:\thola"), "Output: {output}");
        assert!(output.contains("%xtra:\tadiós"), "Output: {output}");
    }

    #[test]
    fn test_extract_translation_strings() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_female.cha");
        let mut chat = parse_chat(chat_text);

        let payloads = collect_translate_payloads(&chat);
        let line_idx = payloads[0].0;
        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, &admitted("hola"));

        let entries = extract_translation_strings(&chat, &[line_idx]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].line_idx, line_idx);
        assert_eq!(entries[0].translation, "hola");
    }

    #[test]
    fn snapshot_translate_batch_item() {
        let item = TranslateBatchItem {
            text: "I eat cookies".into(),
        };
        insta::assert_json_snapshot!(item, @r#"
        {
          "text": "I eat cookies"
        }
        "#);
    }

    #[test]
    fn test_postprocess_basic() {
        let raw = "Hello\u{3002} World\u{2019}s";
        let punct = vec![".", "?"];
        let result = postprocess_translation(raw, &punct);
        assert_eq!(result, "Hello . World's");
    }

    #[test]
    fn test_postprocess_zero_width_space() {
        let raw = "hello\u{200b}world";
        let result = postprocess_translation(raw, &[]);
        assert_eq!(result, "helloworld");
    }
}
