//! Merge consecutive single-letter words that form known abbreviations.
//!
//! ASR engines typically emit abbreviations as individual letters (for example
//! `*CHI: F B I .`). This transform collapses those letter sequences back into
//! one word when the concatenation matches a known abbreviation list.

use std::collections::HashSet;
use std::sync::LazyLock;

use talkbank_model::model::TierContentItems;
use talkbank_model::model::{ChatFile, Line, UtteranceContent, Word};
use talkbank_parser::TreeSitterParser;

/// Known abbreviations loaded from the embedded JSON list.
#[allow(clippy::expect_used)]
static ABBREV: LazyLock<HashSet<String>> = LazyLock::new(|| {
    let data: Vec<String> = serde_json::from_str(include_str!("../data/abbrev.json"))
        .expect("embedded abbrev.json is valid");
    data.into_iter().map(|s| s.to_uppercase()).collect()
});

/// Merge consecutive single-letter words matching known abbreviations.
pub fn merge_abbreviations(chat_file: &mut ChatFile) {
    for line in &mut chat_file.lines {
        if let Line::Utterance(utt) = line {
            let mut items = utt.main.content.content.take();
            merge_in_content_items(&mut items);
            utt.main.content.content = TierContentItems::new(items);
        }
    }
}

/// Merge abbreviations across a whole CHAT document supplied as text.
///
/// The text-to-text form every production caller needs: parse leniently,
/// merge, re-serialize.
///
/// Two deliberate properties, both inherited from the four call sites this was
/// collapsed out of (`runner/dispatch/audio_output.rs`,
/// `runner/dispatch/benchmark_pipeline.rs`, `execution/text_io.rs`,
/// `execution/kernel.rs`), which each carried their own byte-identical copy:
///
/// - **Parse diagnostics are discarded.** A document with parse errors
///   round-trips through the lenient model's best effort rather than failing
///   the job. That is why this takes the lenient path and not
///   `parse_chat_file`.
/// - **The parser is a parameter, not a local.** Constructing a
///   `TreeSitterParser` can fail, so the fallible step stays at the caller's
///   boundary instead of putting a panic inside a pure transform.
pub fn merge_abbreviations_in_chat_text(parser: &TreeSitterParser, chat_text: &str) -> String {
    let (mut chat_file, _parse_diagnostics) =
        talkbank_transform::parse::parse_lenient(parser, chat_text);
    merge_abbreviations(&mut chat_file);
    talkbank_transform::serialize::to_chat_string(&chat_file)
}

fn merge_in_content_items(items: &mut Vec<UtteranceContent>) {
    if items.len() < 2 {
        return;
    }

    let mut result: Vec<UtteranceContent> = Vec::with_capacity(items.len());
    let mut i = 0;

    while i < items.len() {
        let run_start = i;
        let mut letters: Vec<String> = Vec::new();

        while i < items.len() {
            if let Some(letter) = single_letter_word(&items[i]) {
                letters.push(letter);
                i += 1;
            } else {
                break;
            }
        }

        if letters.len() < 2 {
            if !letters.is_empty() {
                result.push(items[run_start].clone());
            } else {
                result.push(items[i].clone());
                i += 1;
            }
            continue;
        }

        let mut j = 0;
        while j < letters.len() {
            let mut matched = false;
            let max_len = letters.len() - j;
            for len in (2..=max_len).rev() {
                let candidate: String = letters[j..j + len]
                    .iter()
                    .map(|s| s.to_uppercase())
                    .collect();
                if ABBREV.contains(&candidate) {
                    let merged_text: String = letters[j..j + len].concat();
                    result.push(UtteranceContent::Word(Box::new(Word::simple(
                        talkbank_model::Location::Unlocated,
                        merged_text,
                    ))));
                    j += len;
                    matched = true;
                    break;
                }
            }

            if !matched {
                result.push(items[run_start + j].clone());
                j += 1;
            }
        }
    }

    *items = result;
}

fn single_letter_word(item: &UtteranceContent) -> Option<String> {
    match item {
        UtteranceContent::Word(w) => {
            let text = w.cleaned_text();
            let mut chars = text.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_alphabetic() => Some(c.to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{merge_abbreviations, merge_abbreviations_in_chat_text};
    use talkbank_model::model::{Terminator, WriteChat};
    use talkbank_parser::TreeSitterParser;

    fn merge_and_serialize(chat: &str) -> String {
        let parser = TreeSitterParser::new().unwrap();
        let mut file = parser.parse_chat_file(chat).expect_built();
        merge_abbreviations(&mut file);
        file.to_chat_string()
    }

    fn main_tier_words(chat_output: &str) -> Vec<String> {
        chat_output
            .lines()
            .filter(|l| l.starts_with('*'))
            .flat_map(|l| {
                let after_colon = l.split_once(':').map(|x| x.1).unwrap_or("");
                after_colon
                    .split_whitespace()
                    .filter(|w| !w.starts_with('\u{15}') && !Terminator::is_chat_terminator(w))
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn minimal_chat(utterance: &str) -> String {
        format!(
            "@UTF8\n\
             @Begin\n\
             @Languages:\teng\n\
             @Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI|||||Target_Child|||\n\
             *CHI:\t{utterance}\n\
             @End\n"
        )
    }

    #[test]
    fn merge_fbi() {
        let chat = minimal_chat("the F B I is here .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(words.contains(&"FBI".to_string()));
        assert!(!words.contains(&"F".to_string()));
    }

    #[test]
    fn no_merge_unknown_tail() {
        let chat = minimal_chat("X Y Z Q W .");
        let out = merge_and_serialize(&chat);
        let words = main_tier_words(&out);
        assert!(words.contains(&"XYZ".to_string()));
        assert!(words.contains(&"Q".to_string()));
        assert!(words.contains(&"W".to_string()));
    }

    /// The text-to-text form is what every production caller actually invokes,
    /// and it existed as three byte-identical private copies before being
    /// collapsed here. Pin it directly so the single remaining copy cannot
    /// drift without a test noticing.
    #[test]
    fn text_form_merges_over_a_lenient_parse() -> Result<(), Box<dyn std::error::Error>> {
        let parser = TreeSitterParser::new()?;
        let out = merge_abbreviations_in_chat_text(&parser, &minimal_chat("the F B I is here ."));
        let words = main_tier_words(&out);
        assert!(words.contains(&"FBI".to_string()), "got {words:?}");
        assert!(!words.contains(&"F".to_string()), "got {words:?}");
        Ok(())
    }

    /// The lenient parse is the whole reason the text form differs from
    /// `merge_and_serialize` above, which parses strictly. A document with
    /// strict error diagnostics must still come back as text with its content
    /// intact, because that is what all four production call sites relied on:
    /// this transform runs over freshly generated ASR output, so "the parser
    /// reported an error" must not become "the job produced nothing".
    ///
    /// The fixture is an unmatched `[`, a main-tier malformation, chosen
    /// because the strict parser reports an `Error`-severity diagnostic
    /// while lenient recovery keeps every word. Note what does NOT qualify: a
    /// missing `@End`, an undeclared speaker, and a missing `@Participants`
    /// are all accepted by `parse_chat_file`, since they are validation
    /// concerns rather than parse errors.
    #[test]
    fn text_form_survives_input_the_strict_parser_rejects() -> Result<(), Box<dyn std::error::Error>>
    {
        let parser = TreeSitterParser::new()?;
        let malformed = minimal_chat("the F B I is here [ .");
        assert!(
            parser.parse_chat_file(&malformed).has_error_diagnostics(),
            "fixture must produce a strict parse error, or this pins nothing"
        );

        let words = main_tier_words(&merge_abbreviations_in_chat_text(&parser, &malformed));
        assert_eq!(words, vec!["the", "FBI", "is", "here"], "got {words:?}");
        Ok(())
    }
}
