use super::BuildChatError;
use talkbank_model::Span;
use talkbank_model::model::{
    BracketedContent, BracketedItem, DependentTier, LanguageCode, Line, Retrace, RetraceKind,
    Separator, Terminator, Utterance, UtteranceContent, Word,
};
use talkbank_parser::TreeSitterParser;

use crate::asr_postprocess;

use super::{DescribedTiming, TranscriptDescription, WordDesc};

pub(super) fn build_utterance_lines(
    desc: &TranscriptDescription,
    parser: &TreeSitterParser,
    langs: &[LanguageCode],
    primary_lang: &LanguageCode,
) -> Result<Vec<Line>, BuildChatError> {
    let mut lines = Vec::with_capacity(desc.utterances.len());

    for utterance in &desc.utterances {
        let words = utterance.words.as_deref().unwrap_or(&[]);
        let should_apply_language_override = !words.is_empty();

        let built = if words.is_empty() {
            utterance.text.as_ref().map_or(Ok(None), |text| {
                build_text_utterance(parser, &utterance.speaker, text, utterance.timing, langs)
            })?
        } else {
            build_word_utterance(parser, &utterance.speaker, words, desc.write_wor)?
        };

        if let Some(mut line) = built {
            if should_apply_language_override {
                apply_utterance_language_override(
                    &mut line,
                    utterance.lang.as_deref(),
                    primary_lang,
                )?;
            }
            lines.push(line);
        }
    }

    Ok(lines)
}

fn apply_utterance_language_override(
    line: &mut Line,
    utterance_lang: Option<&str>,
    primary_lang: &LanguageCode,
) -> Result<(), BuildChatError> {
    if let Some(utterance_lang) = utterance_lang
        && utterance_lang != primary_lang.as_str()
        && let Line::Utterance(utterance) = line
    {
        let code =
            LanguageCode::new(utterance_lang).map_err(|source| BuildChatError::LanguageCode {
                code: utterance_lang.to_string(),
                source,
            })?;
        utterance.main.content.language_code = Some(code);
    }
    Ok(())
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
    langs: &[LanguageCode],
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

    let lang_code = langs.first().map(LanguageCode::as_str).unwrap_or("eng");
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

/// Admit a word only after a clean fragment parse, including for JSON input.
fn parse_asr_word(parser: &TreeSitterParser, text: &str) -> Result<Word, BuildChatError> {
    let errors = talkbank_model::ErrorCollector::new();
    let outcome = parser.parse_word_fragment(text, 0, &errors);
    let diagnostics = errors.into_vec();
    match outcome {
        talkbank_model::ParseOutcome::Parsed(word) if diagnostics.is_empty() => Ok(word),
        talkbank_model::ParseOutcome::Parsed(_) | talkbank_model::ParseOutcome::Rejected => {
            Err(BuildChatError::Word {
                text: text.to_owned(),
                diagnostics,
            })
        }
    }
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

/// Parse a word and attach inline bullet timing, updating utterance-level
/// timing bookkeeping only after successful word admission.
fn parse_and_time_word(
    parser: &TreeSitterParser,
    text: &str,
    timing: DescribedTiming,
    span: &mut UtteranceSpan,
) -> Result<Word, BuildChatError> {
    let mut word = parse_asr_word(parser, text)?;
    if let Some(interval) = timing.positive() {
        word.inline_bullet = Some(interval.bullet());
        span.cover(interval);
    }
    Ok(word)
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
    parser: &TreeSitterParser,
    speaker: &str,
    words: &[WordDesc],
    write_wor: bool,
) -> Result<Option<Line>, BuildChatError> {
    let mut content: Vec<UtteranceContent> = Vec::new();
    let mut span = UtteranceSpan::default();

    let last_text = words.last().map(|word| word.text.as_str()).unwrap_or(".");
    let terminator = Terminator::try_from_chat_str(last_text)
        .unwrap_or(Terminator::Period { span: Span::DUMMY });

    let mut index = 0;
    while index < words.len() {
        let word = &words[index];
        let text = word.text.as_str().trim();

        if text.is_empty() {
            index += 1;
            continue;
        }

        if Terminator::is_chat_terminator(text) {
            index += 1;
            continue;
        }

        if let Some(separator) = tag_marker_separator(text) {
            content.push(UtteranceContent::Separator(separator));
            index += 1;
            continue;
        }

        if word.kind == asr_postprocess::WordKind::Retrace {
            index = push_retrace_run(parser, words, index, &mut content, &mut span)?;
            continue;
        }

        let parsed = parse_and_time_word(parser, text, word.timing, &mut span)?;
        content.push(UtteranceContent::Word(Box::new(parsed)));
        index += 1;
    }

    if content.is_empty() {
        return Ok(None);
    }

    let mut main = talkbank_model::model::MainTier::new(speaker, content, terminator);
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

    Ok(Some(Line::utterance(utterance)))
}

fn push_retrace_run(
    parser: &TreeSitterParser,
    words: &[WordDesc],
    start_index: usize,
    content: &mut Vec<UtteranceContent>,
    span: &mut UtteranceSpan,
) -> Result<usize, BuildChatError> {
    let mut end_index = start_index;
    while end_index < words.len() && words[end_index].kind == asr_postprocess::WordKind::Retrace {
        end_index += 1;
    }

    let mut parsed: Vec<Word> = Vec::new();
    for retrace_word in &words[start_index..end_index] {
        let text = retrace_word.text.as_str().trim();
        if text.is_empty() {
            continue;
        }
        let word = parse_and_time_word(parser, text, retrace_word.timing, span)?;
        parsed.push(word);
    }

    push_retrace_content(parsed, content);
    Ok(end_index)
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
            let retrace = Retrace::new(bracketed, RetraceKind::Partial);
            content.push(UtteranceContent::Retrace(Box::new(retrace)));
        }
        return;
    }

    let items: Vec<BracketedItem> = parsed
        .into_iter()
        .map(|word| BracketedItem::Word(Box::new(word)))
        .collect();
    let bracketed = BracketedContent::new(items);
    let retrace = Retrace::new(bracketed, RetraceKind::Partial).as_group();
    content.push(UtteranceContent::Retrace(Box::new(retrace)));
}
