//! ASR post-processing: compound merging, number expansion, retokenization,
//! disfluency marking, and retrace detection.
//!
//! This module ports the Python ASR post-processing pipeline to Rust. After
//! the Python worker returns raw ASR tokens (via `batch_infer` with task
//! `"asr"`), the Rust server applies these transformations before utterance
//! segmentation and CHAT assembly.
//!
//! # Pipeline stages
//!
//! 1. **Compound merging**: merge adjacent words that form known compounds
//! 2. **Cantonese normalization**: simplified→HK traditional + domain
//!    replacements (lang=yue only), once per monologue, before any splitting
//! 3. **Multi-word splitting**: split tokens containing spaces, interpolate timestamps
//! 4. **Number expansion**: convert digit strings to word form
//! 5. **Long turn splitting**, chunk monologues >300 words
//! 6. **Retokenization**: split into utterances by punctuation
//! 7. **Disfluency replacement**: mark filled pauses ("um" → "&-um") and orthographic
//!    replacements ("'cause" → "(be)cause") from per-language wordlists
//! 8. **N-gram retrace detection**, detect repeated n-grams and wrap in `<...> [/]`
//!
//! The implementation is split by stage so callers can find preparation,
//! number-expansion, chunking, and utterance-finalization logic quickly.

mod asr_types;
pub mod cantonese;
mod chunking;
mod cleanup;
mod compounds;
mod english_caps;
mod expand;
pub mod lang_detect;
mod num2chinese;
mod num2text;
#[cfg(test)]
mod num2text_baseline;
mod ordinal_por;
mod ordinal_year_eng;
mod prepare;
mod retrace;
mod snapshot;
#[cfg(test)]
mod tests;
mod timing;
mod utterance;

use serde::{Deserialize, Serialize};

pub use asr_types::{AsrNormalizedText, AsrRawText, AsrTimestampSecs, ChatWordText, SpeakerIndex};
pub use cantonese::{AlignedNormalization, NormalizationChangedLength};
pub use chunking::{
    finalize_words_to_chunks, finalize_words_to_chunks_with_snapshot,
    split_prepared_chunk_by_assignments,
};
pub use compounds::merge_compounds;
pub use expand::split_words_with_whitespace;
pub use num2text::{NumberExpansionMode, detect_expansion, expand_number};
pub use snapshot::AsrPipelineSnapshot;
pub use timing::{AdmittedInterval, IntervalBound, IntervalRefusal, UntimedCause, WordTiming};
pub use utterance::{
    finalize_utterances, prepare_asr_chunks, process_raw_asr, utterances_from_prepared_chunks,
};

use expand::expand_numbers_in_words;
use prepare::trim_word_boundaries;
pub use prepare::{
    is_cjk_ideograph, prepare_words_pre_expansion, prepare_words_pre_expansion_with_snapshot,
};
pub use retrace::{ExactRetraceAnalysis, analyze_exact_retraces};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The language or languages of the ASR text being post-processed.
///
/// Most steps mark structure (compound merges, fillers, retraces,
/// capitalization) and run under [`Self::rules`]. Two steps WRITE words in a
/// language: number expansion (`25` to `twenty five`) and the percent split
/// (`80%` to `80` `percent`). For code-switched text nothing says which
/// language a numeral was spoken in, and expanding it in either would put
/// words in the transcript the speaker may not have said, so those two steps
/// keep what was recognized: digits stay digits, and `%`, which cannot reach
/// the main tier, is dropped as it is for a language with no percent word. The
/// digits then fail CHAT's word rules and are reported for human review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsrTextLanguage<'a> {
    /// All of the text is in this one language.
    One(&'a str),
    /// Two languages, with no word's language known.
    CodeSwitched {
        /// The language structural rules run under.
        primary: &'a str,
    },
}

impl<'a> AsrTextLanguage<'a> {
    /// The language structural rules run under.
    pub fn rules(self) -> &'a str {
        match self {
            Self::One(lang) | Self::CodeSwitched { primary: lang } => lang,
        }
    }
}

impl<'a> From<&'a str> for AsrTextLanguage<'a> {
    /// A bare language code is text in that one language.
    fn from(lang: &'a str) -> Self {
        Self::One(lang)
    }
}

/// What role a word plays in the CHAT output.
///
/// The `build_chat` module reads this to decide how to represent the word
/// in the AST. Regular words become `UtteranceContent::Word`; retrace words
/// get wrapped in `<...> [/]` bracketed groups; filled pauses are already
/// encoded in the text as `&-um` etc. and parse normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WordKind {
    /// Normal content word (or filled pause already in `&-um` form).
    #[default]
    Regular,
    /// This word is part of a retrace group, a repeated n-gram that
    /// should be wrapped in `<...> [/]` annotation in the CHAT output.
    Retrace,
}

/// A single token from ASR output, with timing information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AsrWord {
    /// The word text (normalized through the ASR pipeline).
    pub text: AsrNormalizedText,
    /// Start time in milliseconds (None if unknown).
    pub start_ms: Option<i64>,
    /// End time in milliseconds (None if unknown).
    pub end_ms: Option<i64>,
    /// What kind of word this is (regular, retrace, etc.).
    #[serde(default)]
    pub kind: WordKind,
}

impl AsrWord {
    /// Create a regular (non-retrace) word with timing.
    pub fn new(text: impl Into<String>, start_ms: Option<i64>, end_ms: Option<i64>) -> Self {
        Self {
            text: AsrNormalizedText::new(text),
            start_ms,
            end_ms,
            kind: WordKind::default(),
        }
    }
}

/// A speaker-attributed utterance after retokenization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Utterance {
    /// Speaker index (0-based).
    pub speaker: SpeakerIndex,
    /// Words in the utterance (last word is a terminator like ".").
    pub words: Vec<AsrWord>,
    /// Detected language for this utterance (ISO 639-3), if different from
    /// the primary language. Used for `[- lang]` code-switching precodes in CHAT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lang: Option<String>,
}

/// One prepared pre-CHAT chunk after ASR normalization but before utterance
/// segmentation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreparedMonologueChunk {
    /// Speaker index (0-based).
    pub speaker: SpeakerIndex,
    /// Normalized ASR words for this chunk.
    pub words: Vec<AsrWord>,
}

/// Raw monologue from ASR output (before post-processing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrMonologue {
    /// Speaker index (0-based).
    pub speaker: SpeakerIndex,
    /// Raw ASR elements.
    pub elements: Vec<AsrElement>,
}

/// What kind of raw ASR element this is.
///
/// Currently only `Text` and `Punctuation` are emitted by providers.
/// Defaults to `Text` when not specified (e.g. omitted from JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AsrElementKind {
    /// A word token.
    #[default]
    Text,
    /// A punctuation token (period, question mark, etc.).
    Punctuation,
}

/// A single element from raw ASR output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrElement {
    /// Token text (raw from the ASR provider).
    pub value: AsrRawText,
    /// Start time in seconds.
    #[serde(default)]
    pub ts: AsrTimestampSecs,
    /// End time in seconds.
    #[serde(default)]
    pub end_ts: AsrTimestampSecs,
    /// Element kind: text or punctuation.
    #[serde(default)]
    pub kind: AsrElementKind,
}

/// Raw ASR output structure (matches Rev.AI format).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrOutput {
    /// Speaker monologues.
    pub monologues: Vec<AsrMonologue>,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// CHAT-legal sentence terminators.
pub(super) const ENDING_PUNCT: &[&str] = &[
    ".", "?", "!", "+...", "+/.", "+//.", "+/?", "+!?", "+\"/.", "+\".", "+//?", "+..?", "+.",
    "...", "(.)",
];

/// CHAT morphological punctuation markers.
///
/// Main-tier-legal separators that are NOT words, the tree-sitter
/// word fragment parser rejects them as such. `ChatWordText`'s
/// structural_check uses this list as a second short-circuit
/// alongside `Terminator::is_chat_terminator` so the ASR pipeline
/// can emit separator tokens (comma at clause boundaries, vocative
/// ‡, tag „) as regular `AsrWord` entries without tripping the
/// "must be a word" gate.
pub(super) const MOR_PUNCT: &[&str] = &["‡", "„", ","];

/// RTL punctuation that needs ASCII normalization.
pub(super) const RTL_PUNCT: &[(&str, &str)] = &[("؟", "?"), ("۔", "."), ("،", ","), ("؛", ";")];

/// Maximum words per turn before splitting.
pub(super) const MAX_TURN_LEN: usize = 300;

/// Long silence threshold used as a fallback boundary when ASR omits sentence
/// punctuation but timing gaps strongly suggest a new utterance.
pub(super) const LONG_PAUSE_SPLIT_MS: i64 = 800;

/// Common English sentence starters worth treating as utterance starts after a
/// long pause in otherwise unpunctuated ASR output.
pub(super) const LONG_PAUSE_SENTENCE_STARTERS: &[&str] = &[
    "and", "but", "did", "do", "does", "go", "have", "has", "had", "he", "how", "i", "is", "it",
    "no", "now", "okay", "so", "then", "they", "we", "well", "what", "when", "where", "who", "why",
    "yes", "you",
];

#[cfg(test)]
mod integration_tests {
    use std::collections::BTreeSet;

    use super::{
        AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs,
        ChatWordText, SpeakerIndex, Utterance, expand_number, prepare_words_pre_expansion,
        process_raw_asr,
    };

    /// One monologue holding one ASR element timed in seconds.
    fn single_element(value: &str, start_s: f64, end_s: f64) -> AsrOutput {
        AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![AsrElement {
                    value: AsrRawText::new(value),
                    ts: AsrTimestampSecs::Observed(start_s),
                    end_ts: AsrTimestampSecs::Observed(end_s),
                    kind: AsrElementKind::Text,
                }],
            }],
        }
    }

    /// Build CHAT from pipeline utterances through the real transcript gate
    /// (which validates every word for `lang`), serialize it, and assert it
    /// reparses cleanly. Returns the serialized CHAT.
    fn build_and_reparse(utterances: &[Utterance], lang: &str) -> String {
        let parser = talkbank_parser::TreeSitterParser::new().unwrap();
        let desc = crate::build_chat::transcript_from_asr_utterances(
            utterances,
            &["PAR".to_string()],
            &[lang.to_string()],
            Some("clip"),
            true,
        )
        .expect("test: transcript_from_asr_utterances should succeed")
        .description;
        let chat = crate::build_chat::build_chat(&desc).expect("build chat");
        let serialized = crate::serialize::to_chat_string(&chat);
        let (_parsed, errors) = crate::parse::parse_lenient(&parser, &serialized);
        assert!(
            errors.is_empty(),
            "generated CHAT should reparse cleanly: {errors:?}\n{serialized}"
        );
        serialized
    }

    fn texts(utterance: &Utterance) -> Vec<&str> {
        utterance.words.iter().map(|w| w.text.as_str()).collect()
    }

    /// A Cantonese monologue of nothing but CJK punctuation reaches CHAT
    /// assembly with utterances that hold no content, and assembly refuses
    /// instead of writing a headers-only file.
    ///
    /// This is the DOWNSTREAM route to an empty Cantonese transcript, and it
    /// is downstream of normalization: the words survive normalization intact
    /// (`。` is not a Han character and the tables do not touch it) and are
    /// lost at tokenization, where every CJK sentence mark becomes a bare
    /// terminator, and then at utterance building, where a terminator is not
    /// content. Post-processing reports two utterances, so nothing upstream
    /// sees an emptiness to report; only the built lines do.
    #[test]
    fn a_cantonese_transcript_of_only_punctuation_is_refused_not_written_empty() {
        let output = single_element("。。", 0.0, 1.0);
        let utterances = process_raw_asr(&output, "yue")
            .expect("test: ASR post-processing must not refuse this input");

        assert_eq!(
            utterances.len(),
            2,
            "post-processing still reports utterances: {utterances:?}"
        );
        assert!(
            utterances.iter().all(|utterance| texts(utterance) == ["."]),
            "every token is a bare terminator: {utterances:?}"
        );

        let desc = crate::build_chat::transcript_from_asr_utterances(
            &utterances,
            &["PAR".to_string()],
            &["yue".to_string()],
            Some("clip"),
            true,
        )
        .expect("test: transcript_from_asr_utterances should succeed")
        .description;

        let refusal = crate::build_chat::build_chat(&desc)
            .expect_err("a transcript with no utterance content must be refused");
        assert!(
            matches!(
                refusal,
                crate::build_chat::BuildChatError::NoUtteranceSurvivedBuild { described }
                    if described == utterances.len()
            ),
            "the refusal must name what produced nothing, got: {refusal}"
        );
    }

    #[test]
    fn pipeline_output_still_roundtrips_through_build_chat() {
        let output = single_element(
            "這麼搞笑?我還清了啊!我還覺得奇怪為什麼在一個三次頭的電話打工呢?",
            0.0,
            0.0,
        );
        let utterances = process_raw_asr(&output, "yue")
            .expect("test: ASR post-processing must not refuse this input");
        build_and_reparse(&utterances, "yue");
    }

    /// Real-data shape: a timed-word ASR provider (Rev-style) emitted the
    /// Portuguese ordinal `54ª` as one token spanning 3160-3800 ms. It must
    /// survive tokenization as one timed word, expand to gendered words that
    /// share its span, and pass the CHAT gate. The dotted spelling `54.ª`
    /// must behave identically rather than split at its abbreviation period.
    #[test]
    fn portuguese_indicator_ordinal_keeps_timing_through_prepare_and_expansion() {
        for ordinal in ["54ª", "54.ª"] {
            let output = single_element(ordinal, 3.16, 3.8);

            let prepared = prepare_words_pre_expansion(&output.monologues[0].elements, "por")
                .expect("test: ASR post-processing must not refuse this input");
            assert_eq!(prepared.len(), 1, "{ordinal}: {prepared:?}");
            assert_eq!(prepared[0].text.as_str(), ordinal);
            assert_eq!(
                (prepared[0].start_ms, prepared[0].end_ms),
                (Some(3160), Some(3800))
            );

            let utterances = process_raw_asr(&output, "por")
                .expect("test: ASR post-processing must not refuse this input");
            assert_eq!(utterances.len(), 1, "{ordinal}: {utterances:?}");
            assert_eq!(texts(&utterances[0]), ["quinquagésima", "quarta", "."]);
            let words = &utterances[0].words;
            assert_eq!(words[0].start_ms, Some(3160));
            assert_eq!(words[1].end_ms, Some(3800));
            assert_eq!(words[0].end_ms, words[1].start_ms);
            assert!(words[0].start_ms < words[0].end_ms);
            assert!(words[1].start_ms < words[1].end_ms);

            let chat = build_and_reparse(&utterances, "por");
            assert!(chat.contains("quinquagésima quarta ."), "{chat}");
            assert!(!chat.contains(ordinal), "unexpanded ordinal: {chat}");
        }
    }

    /// Segment-level shape: the ordinal arrives inside running text followed
    /// by a genuine sentence period. The abbreviation period stays inside the
    /// ordinal; the following period still ends the utterance.
    #[test]
    fn portuguese_ordinal_abbreviation_period_is_not_a_sentence_end() {
        let output = single_element("54.ª. então", 3.16, 3.8);
        let utterances = process_raw_asr(&output, "por")
            .expect("test: ASR post-processing must not refuse this input");
        let all: Vec<Vec<&str>> = utterances.iter().map(texts).collect();
        assert_eq!(
            all,
            [vec!["quinquagésima", "quarta", "."], vec!["então", "."]]
        );
        let chat = build_and_reparse(&utterances, "por");
        assert!(chat.contains("quinquagésima quarta ."), "{chat}");
    }

    /// A recognized ordinal with no rendering keeps its abbreviation period
    /// and is left unexpanded, so its digits stay visible to validation.
    #[test]
    fn portuguese_ordinal_beyond_rendered_range_passes_through_whole() {
        let output = single_element("1001.º.", 0.0, 1.0);
        let prepared = prepare_words_pre_expansion(&output.monologues[0].elements, "por")
            .expect("test: ASR post-processing must not refuse this input");
        let prepared: Vec<&str> = prepared.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(prepared, ["1001.º", "."]);
        assert_eq!(expand_number("1001.º", "por"), "1001.º");
    }

    /// Every rendered ordinal word, in both genders and numbers, is a legal
    /// Portuguese CHAT word (no digits, parses as a word).
    #[test]
    fn every_rendered_portuguese_ordinal_is_a_legal_portuguese_chat_word() {
        let lang = talkbank_model::model::LanguageCode::new("por").expect("valid language code");
        let parser = talkbank_parser::TreeSitterParser::new().unwrap();
        let mut words = BTreeSet::new();
        for rank in 1..=1000 {
            for form in ["º", "ª", ".ºs", ".ªs"] {
                let input = format!("{rank}{form}");
                let expanded = expand_number(&input, "por");
                assert_ne!(expanded, input, "{input} must expand");
                words.extend(expanded.split_whitespace().map(str::to_owned));
            }
        }
        for word in &words {
            assert!(
                ChatWordText::try_from_lang_with_parser(word, &parser, &lang).is_ok(),
                "{word} is not a legal Portuguese CHAT word"
            );
        }
    }
}
