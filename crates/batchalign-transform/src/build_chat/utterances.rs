use super::BuildChatError;
use talkbank_model::Location;
use talkbank_model::model::{
    BracketedContent, BracketedItem, DependentTier, LanguageCode, Line, Retrace, RetraceKind,
    Separator, Utterance, UtteranceContent, Word,
};
use talkbank_parser::TreeSitterParser;

use crate::asr_postprocess;

use super::{DescribedTiming, SynthesizedEnd, TranscriptDescription, UtteranceBody, WordDesc};

/// The built utterance lines, with every one whose terminator the
/// description's producer synthesized.
pub(super) struct BuiltUtterances {
    pub(super) lines: Vec<Line>,
    pub(super) synthesized_ends: Vec<SynthesizedEnd>,
}

pub(super) fn build_utterance_lines(
    desc: &TranscriptDescription,
    parser: &TreeSitterParser,
) -> Result<BuiltUtterances, BuildChatError> {
    let primary_lang = desc.langs.primary();
    let mut lines = Vec::with_capacity(desc.utterances.len());
    let mut synthesized_ends = Vec::new();

    for utterance in &desc.utterances {
        let built = match &utterance.body {
            UtteranceBody::Text { text, timing } => {
                build_text_utterance(parser, &utterance.speaker, text, *timing, primary_lang)?
            }
            UtteranceBody::Words { words, end } => {
                let built = build_word_utterance(&utterance.speaker, words, end, desc.write_wor);
                match built {
                    Some(mut line) => {
                        apply_utterance_language_override(
                            &mut line,
                            utterance.lang.as_ref(),
                            primary_lang,
                        );
                        if let asr_postprocess::UtteranceEnd::Synthesized(_) = end {
                            // Its position among the lines written, which is
                            // what a reader of the CHAT file counts.
                            synthesized_ends.push(SynthesizedEnd {
                                utterance: lines.len() + 1,
                                speaker: utterance.speaker.clone(),
                            });
                        }
                        Some(line)
                    }
                    None => None,
                }
            }
        };
        lines.extend(built);
    }

    Ok(BuiltUtterances {
        lines,
        synthesized_ends,
    })
}

fn apply_utterance_language_override(
    line: &mut Line,
    utterance_lang: Option<&LanguageCode>,
    primary_lang: &LanguageCode,
) {
    if let Some(utterance_lang) = utterance_lang
        && utterance_lang != primary_lang
        && let Line::Utterance(utterance) = line
    {
        utterance.main.content.language_code = Some(talkbank_model::model::LanguagePrecode::new(
            Location::Unlocated,
            utterance_lang.clone(),
        ));
    }
}

/// If `text` is a tag-marker separator (comma, tag marker, vocative marker),
/// return the corresponding [`Separator`] model type. Otherwise return `None`.
pub fn tag_marker_separator(text: &str) -> Option<Separator> {
    match text {
        "," => Some(Separator::Comma {
            location: Location::Unlocated,
        }),
        "\u{201E}" => Some(Separator::Tag {
            location: Location::Unlocated,
        }),
        "\u{2021}" => Some(Separator::Vocative {
            location: Location::Unlocated,
        }),
        _ => None,
    }
}

/// Build a text-level utterance by parsing through tree-sitter.
///
/// This path constructs a minimal valid CHAT document around the input text
/// and parses it with `parse_strict()`. The mini-document hack is necessary
/// because tree-sitter requires complete document context (headers, `@Begin`,
/// `@End`) to parse a single utterance correctly.
///
/// **Callers:** This function is used by the `UtteranceDesc.text` API path
/// when a caller provides a pre-formatted CHAT utterance string instead of
/// word-level tokens. It has zero production callers in the current codebase
/// (the ASR pipeline always uses word-level `WordDesc` tokens), but it
/// preserves the JSON API contract for external callers who construct
/// `TranscriptDescription` directly. The PyO3 bridge tests exercise this path.
fn build_text_utterance(
    parser: &TreeSitterParser,
    speaker: &str,
    text: &str,
    timing: DescribedTiming,
    primary_lang: &LanguageCode,
) -> Result<Option<Line>, BuildChatError> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }

    // Only a positive interval is written; the parser then admits it.
    let bullet_str = match timing {
        DescribedTiming::Positive(interval) => {
            format!(" \x15{}_{}\x15", interval.start_ms(), interval.end_ms())
        }
        DescribedTiming::Untimed(_) => String::new(),
    };

    // The transcript's own primary language, parsed where the description was
    // built: the mini-document states the language the caller declared,
    // never an invented one.
    let lang_code = primary_lang.as_str();
    let mini_chat = format!(
        "@UTF8\n@Begin\n@Languages:\t{lang}\n@Participants:\t{speaker} Participant Participant\n\
         @ID:\t{lang}|corpus_name|{speaker}|||||Participant|||\n*{speaker}:\t{text}{bullet}\n@End\n",
        lang = lang_code,
        speaker = speaker,
        text = text,
        bullet = bullet_str,
    );

    let parsed = crate::parse::parse_strict(parser, &mini_chat).map_err(|error| {
        BuildChatError::Utterance {
            speaker: speaker.to_string(),
            message: error.to_string(),
        }
    })?;

    for parsed_line in parsed.lines.into_iter() {
        if let Line::Utterance(utterance) = parsed_line {
            return Ok(Some(Line::Utterance(utterance)));
        }
    }

    Ok(None)
}

/// The span an utterance's timed words cover so far: none, or the positive
/// interval covering every one of them. Absence is the only "untimed" state,
/// so there is no separate has-timing flag to keep in step with it.
#[derive(Debug, Clone, Copy, Default)]
struct UtteranceSpan(Option<asr_postprocess::PositiveInterval>);

impl UtteranceSpan {
    /// Extend the span over one timed word.
    fn cover(&mut self, word: asr_postprocess::PositiveInterval) {
        self.0 = Some(match self.0 {
            Some(span) => span.covering(word),
            None => word,
        });
    }
}

/// A parsed word with its inline bullet timing, updating utterance-level
/// timing bookkeeping.
fn timed_word(word: &Word, timing: DescribedTiming, span: &mut UtteranceSpan) -> Word {
    let mut word = word.clone();
    if let Some(interval) = timing.positive() {
        word.inline_bullet = Some(interval.bullet());
        span.cover(interval);
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
    speaker: &str,
    words: &[WordDesc],
    end: &asr_postprocess::UtteranceEnd,
    write_wor: bool,
) -> Option<Line> {
    let mut content: Vec<UtteranceContent> = Vec::new();
    let mut span = UtteranceSpan::default();

    // The description states how the utterance ends, as a field: the
    // terminator the recognizer observed, or the period a synthesized end is
    // written with. Every word is content.
    let terminator = end.terminator();

    let mut index = 0;
    while index < words.len() {
        let word = &words[index];
        match word.text.content() {
            asr_postprocess::ChatWordContent::Separator(separator) => {
                content.push(UtteranceContent::Separator(separator.clone()));
                index += 1;
            }
            asr_postprocess::ChatWordContent::Word(_)
                if word.kind == asr_postprocess::WordKind::Retrace =>
            {
                index = push_retrace_run(words, index, &mut content, &mut span);
            }
            asr_postprocess::ChatWordContent::Word(parsed) => {
                content.push(UtteranceContent::Word(Box::new(timed_word(
                    parsed,
                    word.timing,
                    &mut span,
                ))));
                index += 1;
            }
        }
    }

    if content.is_empty() {
        return None;
    }

    let mut main = talkbank_model::model::MainTier::new(
        talkbank_model::model::MainTierLocations::UNLOCATED,
        speaker,
        content,
        terminator,
    );
    // The utterance bullet covers every timed word: positive by construction.
    if let UtteranceSpan(Some(interval)) = span {
        main = main.with_bullet(interval.bullet());
    }

    let mut utterance = Utterance::new(main);
    if write_wor && span.0.is_some() {
        let wor_tier = utterance.main.generate_wor_tier();
        utterance
            .dependent_tiers
            .push(DependentTier::Wor(wor_tier).into());
    }

    Some(Line::utterance(utterance))
}

/// Push the run of retrace words starting at `start_index` (a separator
/// ends the run: it is not a word, so it cannot be retraced) and return the
/// index after it.
fn push_retrace_run(
    words: &[WordDesc],
    start_index: usize,
    content: &mut Vec<UtteranceContent>,
    span: &mut UtteranceSpan,
) -> usize {
    let mut parsed: Vec<Word> = Vec::new();
    let mut end_index = start_index;
    while let Some(retrace_word) = words.get(end_index)
        && retrace_word.kind == asr_postprocess::WordKind::Retrace
        && let asr_postprocess::ChatWordContent::Word(word) = retrace_word.text.content()
    {
        parsed.push(timed_word(word, retrace_word.timing, span));
        end_index += 1;
    }

    push_retrace_content(parsed, content);
    end_index
}

fn push_retrace_content(parsed: Vec<Word>, content: &mut Vec<UtteranceContent>) {
    if parsed.is_empty() {
        return;
    }

    let first_text = parsed[0].cleaned_text();
    let all_same_text = parsed.len() > 1
        && parsed
            .iter()
            .skip(1)
            .all(|word| word.cleaned_text().eq_ignore_ascii_case(first_text));

    if parsed.len() == 1 || all_same_text {
        for word in parsed {
            let bracketed = BracketedContent::new(vec![BracketedItem::Word(Box::new(word))]);
            let retrace = Retrace::new(
                Location::Unlocated,
                Location::Unlocated,
                bracketed,
                RetraceKind::Partial,
            );
            content.push(UtteranceContent::Retrace(Box::new(retrace)));
        }
        return;
    }

    let items: Vec<BracketedItem> = parsed
        .into_iter()
        .map(|word| BracketedItem::Word(Box::new(word)))
        .collect();
    let bracketed = BracketedContent::new(items);
    let retrace = Retrace::new(
        Location::Unlocated,
        Location::Unlocated,
        bracketed,
        RetraceKind::Partial,
    )
    .as_group();
    content.push(UtteranceContent::Retrace(Box::new(retrace)));
}
