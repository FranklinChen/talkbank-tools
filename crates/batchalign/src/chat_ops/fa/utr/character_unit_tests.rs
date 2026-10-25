//! Cantonese timing recovery in character space: both sides projected the
//! same way, word anchors only where one provider token is exactly one word.
//!
//! The corpora write Cantonese both ways (one character per word, and
//! multi-character words), and recognizers return both phrases and words, so
//! each combination is exercised here.

use talkbank_model::UtteranceIdx;
use talkbank_model::model::{Line, TranscriptName};

use super::{AnchorLookup, AsrTimingToken, GlobalUtr, PreparedUtrTokens, UtrStrategy};
use crate::api::LanguageCode3;

/// A one-utterance Cantonese transcript whose main tier is `words`.
fn cantonese(words: &str) -> talkbank_model::model::ChatFile {
    batchalign_transform::parse_source_with_parser(
        &crate::chat_parser(),
        &format!(
            "@UTF8\n@Begin\n@Languages:\tyue\n@Participants:\tPAR Participant\n@ID:\tyue|test|PAR|||||Participant|||\n@Media:\taudio, audio, unlinked\n*PAR:\t{words} .\n@End\n"
        ),
    )
    .admit(TranscriptName::Anonymous, &talkbank_model::NullErrorSink)
    .expect("valid synthetic CHAT")
    .into_valid_file()
    .into_unchecked()
}

fn prepared(tokens: &[(&str, u64, u64)]) -> PreparedUtrTokens {
    PreparedUtrTokens::prepare(
        tokens
            .iter()
            .map(|&(text, start_ms, end_ms)| AsrTimingToken {
                text: text.to_owned(),
                start_ms,
                end_ms,
            })
            .collect(),
        &LanguageCode3::yue(),
    )
    .expect("these characters keep their count under the Cantonese table")
}

fn bullet(document: &talkbank_model::model::ChatFile) -> Option<(u64, u64)> {
    document.lines.iter().find_map(|line| match line {
        Line::Utterance(utterance) => utterance
            .main
            .content
            .bullet
            .as_ref()
            .and_then(talkbank_model::BulletState::as_parsed)
            .map(|bullet| (bullet.timing.start_ms, bullet.timing.end_ms)),
        _ => None,
    })
}

/// `(word, start, end)` of each anchor of the first utterance, or `None`
/// when recovery had no reliable match there.
fn anchors(result: &mut super::UtrResult) -> Option<Vec<(usize, u64, u64)>> {
    let index = result.take_anchors();
    match index.lookup(UtteranceIdx::new(0)) {
        AnchorLookup::Anchored(anchors) => Some(
            anchors
                .anchors()
                .iter()
                .map(|anchor| {
                    (
                        anchor.word().raw(),
                        anchor.start().get(),
                        anchor.end().get(),
                    )
                })
                .collect(),
        ),
        AnchorLookup::NoReliableMatch => None,
        other => panic!("recovery observed this utterance: {other:?}"),
    }
}

/// RED FIRST: a transcript written in words (`佢哋 食 嘢`) against a
/// recognizer that returns words. Projecting only the recognizer side into
/// characters left the transcript word `佢哋` facing `佢` and `哋`, which it
/// never equals, so the first word was never matched and no anchor was
/// possible. In one character space every word matches, and a token that is
/// exactly one transcript word anchors it.
#[test]
fn a_word_segmented_transcript_meets_word_tokens_and_anchors_each_word() {
    let mut document = cantonese("佢哋 食 嘢");
    let mut result = GlobalUtr.inject(
        &mut document,
        &prepared(&[("佢哋", 100, 300), ("食", 300, 400), ("嘢", 400, 500)]),
    );
    assert_eq!((result.injected(), result.unmatched()), (1, 0));
    assert_eq!(bullet(&document), Some((100, 500)));
    assert_eq!(
        anchors(&mut result),
        Some(vec![(0, 100, 300), (1, 300, 400), (2, 400, 500)])
    );
}

/// A word-segmented transcript against a recognizer phrase: the characters
/// match, the phrase bounds the utterance, and no word is anchored, because
/// the phrase does not say when any one word ended.
#[test]
fn a_word_segmented_transcript_meets_a_phrase_without_anchors() {
    let mut document = cantonese("佢哋 食 嘢");
    let mut result = GlobalUtr.inject(&mut document, &prepared(&[("佢哋食嘢。", 100, 900)]));
    assert_eq!((result.injected(), result.unmatched()), (1, 0));
    assert_eq!(bullet(&document), Some((100, 900)));
    assert_eq!(anchors(&mut result), None);
}

/// The shape of transcripts written one character per word, against a
/// recognizer phrase: recovered as an envelope, with no fabricated
/// per-character timing.
#[test]
fn a_per_character_transcript_meets_a_phrase_as_an_envelope() {
    let mut document = cantonese("佢 哋 食 嘢");
    let mut result = GlobalUtr.inject(&mut document, &prepared(&[("佢哋食嘢", 100, 900)]));
    assert_eq!((result.injected(), result.unmatched()), (1, 0));
    assert_eq!(bullet(&document), Some((100, 900)));
    assert_eq!(anchors(&mut result), None);
}

/// A per-character transcript against word tokens: a single-character token
/// anchors its word; a two-character token holds two transcript words and
/// anchors neither.
#[test]
fn a_per_character_transcript_anchors_only_single_character_tokens() {
    let mut document = cantonese("佢 哋 食 嘢");
    let mut result = GlobalUtr.inject(
        &mut document,
        &prepared(&[("佢哋", 100, 300), ("食", 300, 400), ("嘢", 400, 500)]),
    );
    assert_eq!((result.injected(), result.unmatched()), (1, 0));
    assert_eq!(
        anchors(&mut result),
        Some(vec![(2, 300, 400), (3, 400, 500)])
    );
}

/// A word whose characters are split across tokens, or matched only in
/// part, is not anchored: no single token says when it began and ended.
#[test]
fn a_word_split_across_tokens_is_not_anchored() {
    let mut document = cantonese("佢哋 食");
    let mut result = GlobalUtr.inject(
        &mut document,
        &prepared(&[("佢", 100, 200), ("哋", 200, 300), ("食", 300, 400)]),
    );
    assert_eq!((result.injected(), result.unmatched()), (1, 0));
    assert_eq!(anchors(&mut result), Some(vec![(1, 300, 400)]));
}

/// A non-Han word inside a Cantonese transcript stays one unit on both
/// sides, so English code-switches still match word for word.
#[test]
fn latin_words_in_a_cantonese_transcript_stay_whole() {
    let mut document = cantonese("OK 啦");
    let mut result = GlobalUtr.inject(
        &mut document,
        &prepared(&[("OK", 100, 300), ("啦", 300, 400)]),
    );
    assert_eq!((result.injected(), result.unmatched()), (1, 0));
    assert_eq!(
        anchors(&mut result),
        Some(vec![(0, 100, 300), (1, 300, 400)])
    );
}

/// A token's trailing ASCII punctuation is not a unit of it. `嘢?` used to
/// count two units (`嘢` and the punctuation-only run `?`), so although the
/// `?` was then dropped from matching, the token held "two units" and never
/// equalled the one-unit word `嘢`: the word was matched but not anchored.
/// The same held for an ellipsis, and for a run of punctuation before the
/// character, which also shifted the character's place in the token.
#[test]
fn a_tokens_punctuation_is_not_a_unit_of_it() {
    for token in ["嘢?", "嘢…", "嘢...", "?嘢", "嘢 ."] {
        let mut document = cantonese("食 嘢");
        let mut result = GlobalUtr.inject(
            &mut document,
            &prepared(&[("食", 100, 300), (token, 300, 500)]),
        );
        assert_eq!((result.injected(), result.unmatched()), (1, 0), "{token}");
        assert_eq!(
            anchors(&mut result),
            Some(vec![(0, 100, 300), (1, 300, 500)]),
            "{token}"
        );
    }
}
