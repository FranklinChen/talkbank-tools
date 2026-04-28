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
//! 1. **Compound merging** — merge adjacent words that form known compounds
//! 2. **Multi-word splitting** — split tokens containing spaces, interpolate timestamps
//! 3. **Number expansion** — convert digit strings to word form
//! 4. **Cantonese normalization** — simplified→HK traditional + domain replacements (lang=yue only)
//! 5. **Long turn splitting** — chunk monologues >300 words
//! 6. **Retokenization** — split into utterances by punctuation
//! 7. **Disfluency replacement** — mark filled pauses ("um" → "&-um") and orthographic
//!    replacements ("'cause" → "(be)cause") from per-language wordlists
//! 8. **N-gram retrace detection** — detect repeated n-grams and wrap in `<...> [/]`

mod asr_types;
pub mod cantonese;
mod cleanup;
mod compounds;
pub mod lang_detect;
mod num2chinese;
mod num2text;
mod ordinal_year_eng;
pub mod registry;
mod snapshot;

pub use asr_types::{AsrNormalizedText, AsrRawText, AsrTimestampSecs, ChatWordText, SpeakerIndex};
pub use snapshot::AsrPipelineSnapshot;

use serde::{Deserialize, Serialize};

pub use compounds::merge_compounds;
pub use num2text::{NumberExpansionMode, detect_expansion, expand_number};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
    /// This word is part of a retrace group — a repeated n-gram that
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
const ENDING_PUNCT: &[&str] = &[
    ".", "?", "!", "+...", "+/.", "+//.", "+/?", "+!?", "+\"/.", "+\".", "+//?", "+..?", "+.",
    "...", "(.)",
];

/// CHAT morphological punctuation markers.
///
/// Main-tier-legal separators that are NOT words — the tree-sitter
/// word fragment parser rejects them as such. `ChatWordText`'s
/// structural_check uses this list as a second short-circuit
/// alongside `Terminator::is_chat_terminator` so the ASR pipeline
/// can emit separator tokens (comma at clause boundaries, vocative
/// ‡, tag „) as regular `AsrWord` entries without tripping the
/// "must be a word" gate.
pub(super) const MOR_PUNCT: &[&str] = &["‡", "„", ","];

/// RTL punctuation that needs ASCII normalization.
const RTL_PUNCT: &[(&str, &str)] = &[("؟", "?"), ("۔", "."), ("،", ","), ("؛", ";")];

/// Maximum words per turn before splitting.
const MAX_TURN_LEN: usize = 300;

/// Long silence threshold used as a fallback boundary when ASR omits sentence
/// punctuation but timing gaps strongly suggest a new utterance.
const LONG_PAUSE_SPLIT_MS: i64 = 800;

/// Common English sentence starters worth treating as utterance starts after a
/// long pause in otherwise unpunctuated ASR output.
const LONG_PAUSE_SENTENCE_STARTERS: &[&str] = &[
    "and", "but", "did", "do", "does", "go", "have", "has", "had", "he", "how", "i", "is", "it",
    "no", "now", "okay", "so", "then", "they", "we", "well", "what", "when", "where", "who", "why",
    "yes", "you",
];

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Run the full ASR post-processing pipeline on raw ASR output.
///
/// Applies compound merging, timing conversion, multi-word splitting,
/// number expansion, long turn splitting, punctuation-based retokenization,
/// disfluency replacement, and n-gram retrace detection. Returns
/// speaker-attributed utterances ready for CHAT assembly via `build_chat()`.
pub fn process_raw_asr(output: &AsrOutput, lang: &str) -> Vec<Utterance> {
    let mut all_utterances = utterances_from_prepared_chunks(prepare_asr_chunks(output, lang));
    finalize_utterances(&mut all_utterances, lang);
    all_utterances
}

/// Stages 1-3: compound merging, timed word extraction with separator strip,
/// and multi-word token splitting.
///
/// Returns words ready for number expansion. The caller is responsible for
/// expanding numbers (either via the Rust fallback tables or Python IPC)
/// before passing the words to [`finalize_words_to_chunks`].
pub fn prepare_words_pre_expansion(elements: &[AsrElement], lang: &str) -> Vec<AsrWord> {
    prepare_words_pre_expansion_with_snapshot(elements, lang, None)
}

/// Snapshot-aware variant of [`prepare_words_pre_expansion`].
///
/// When `snapshot` is `Some`, intermediate stage outputs
/// (`after_compound_merge`, `after_timing_extract`,
/// `after_multiword_split`) are populated for downstream trace
/// rendering. When `None`, behavior is identical to the bare variant
/// at zero capture cost.
///
/// Callers who want stage 4 (number expansion) captured must record
/// it themselves — this function returns BEFORE expansion runs.
pub fn prepare_words_pre_expansion_with_snapshot(
    elements: &[AsrElement],
    lang: &str,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Vec<AsrWord> {
    // Stage 1: compound merging
    let merged = merge_compounds(elements);
    if let Some(ref mut s) = snapshot {
        s.after_compound_merge = merged.clone();
    }

    // 2026-04-23 English title-period strip MUST fire here, before
    // `extract_timed_words` and `split_multiword_tokens` — the
    // latter splits on `.` (`normalized_split_separator`) and
    // would slice `Dr.` into `Dr` + `.` before our rule sees the
    // intact allowlisted surface. Operates on raw element text
    // (pre-tokenization). English-gated.
    let merged = cleanup::strip_english_title_periods_on_elements(merged, lang);

    // Stage 2: extract words with ms timings, filter pauses
    let mut words = extract_timed_words(&merged);
    if let Some(ref mut s) = snapshot {
        s.after_timing_extract = words.clone();
    }

    // Stage 2b: strip MOR_PUNCT separators from the boundaries of each word.
    // Case is preserved — downstream CHAT consumers need uppercase "I" and
    // proper nouns intact; disfluency and retrace matching are case-insensitive
    // internally.
    words = strip_separator_words(words);

    // Stage 2c: strip stray quote marks at word boundaries. ASR providers
    // emit tokens like `"My` when transcribing quoted speech verbatim;
    // tree-sitter rejects the literal `"` so these tokens otherwise
    // tank the whole transcribe job. Silent strip rather than a CHAT
    // annotation: pure orthographic noise, no information lost.
    words = cleanup::strip_boundary_quotes(words);

    // Stage 3: split multi-word tokens with timestamp interpolation
    let words = split_multiword_tokens(words, lang);

    // Stage 3c: re-run boundary-quote strip AFTER Stage 3. Stage 3 splits
    // on whitespace and `.`/`?`/`!`/`,` — a single ASR element like
    // `Ross." said.` (period+quote glued, internal whitespace) splits
    // into `["Ross", ".", "\"", "said", "."]`. The standalone `"` part
    // bypassed Stage 2c (which ran before the split). Stripping again
    // here also catches symmetric `"hello` / `world"` shapes that
    // Stage 3 produces from `He said "hello world." said.`-style
    // multi-token ASR values.
    let words = cleanup::strip_boundary_quotes(words);

    // Stage 3b: split percent-suffix tokens (`80%` → `80`, `percent`).
    // `%` is the CHAT dep-tier sigil and cannot reach the main tier in
    // any language; this stage guarantees that property for every
    // downstream consumer, including the Python-routed number-expansion
    // path used by `transcribe`.
    let result = split_percent_suffix_words(words, lang);
    if let Some(ref mut s) = snapshot {
        s.after_multiword_split = result.clone();
    }
    result
}

/// Stages 4b-5b: Cantonese normalization, long turn splitting, and
/// pause-based splitting.
///
/// Takes words that have already been through number expansion and
/// produces speaker-attributed chunks ready for retokenization.
pub fn finalize_words_to_chunks(
    words: Vec<AsrWord>,
    speaker: SpeakerIndex,
    lang: &str,
) -> Vec<PreparedMonologueChunk> {
    finalize_words_to_chunks_with_snapshot(words, speaker, lang, None)
}

/// Snapshot-aware variant of [`finalize_words_to_chunks`].
///
/// Populates `after_cantonese_norm` (only for `yue`) and
/// `after_long_turn_split` when `snapshot` is `Some`. `None` is the
/// zero-overhead default.
pub fn finalize_words_to_chunks_with_snapshot(
    words: Vec<AsrWord>,
    speaker: SpeakerIndex,
    lang: &str,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Vec<PreparedMonologueChunk> {
    // Stage 4b: Cantonese normalization (simplified→HK traditional + domain replacements)
    let mut words = if lang == "yue" {
        let normalized = normalize_cantonese_words(words);
        if let Some(ref mut s) = snapshot {
            s.after_cantonese_norm = Some(normalized.clone());
        }
        normalized
    } else {
        words
    };

    // 2026-04-23 English transcribe-pipeline corrections — the two
    // **per-word** rules (I-cap, title-period strip) must run
    // BEFORE stage 6 retokenize, because retokenize splits on
    // trailing `.` and would slice `Dr.` in half if the period
    // weren't stripped first. English-gated; no-op for other
    // languages.
    cleanup::apply_english_transcribe_rules_pre_retokenize(&mut words, lang);

    // Stage 5: long turn splitting
    let chunks = split_long_turns(words);
    if let Some(ref mut s) = snapshot {
        s.after_long_turn_split = chunks.clone();
    }

    // Stage 5b: add timing-gap boundaries for long unpunctuated runs.
    let chunks = split_on_long_pauses(chunks);

    chunks
        .into_iter()
        .filter(|chunk| !chunk.is_empty())
        .map(|words| PreparedMonologueChunk { speaker, words })
        .collect()
}

/// Normalize raw ASR monologues into pre-CHAT chunks while preserving speaker
/// boundaries.
///
/// This is the monolithic pipeline that applies all stages in sequence.
/// For pipelines that need to intercept the number expansion step (e.g. to
/// route through Python IPC), use [`prepare_words_pre_expansion`] and
/// [`finalize_words_to_chunks`] separately.
pub fn prepare_asr_chunks(output: &AsrOutput, lang: &str) -> Vec<PreparedMonologueChunk> {
    let mut prepared = Vec::new();

    for monologue in &output.monologues {
        let words = prepare_words_pre_expansion(&monologue.elements, lang);
        // Stage 4: number expansion (Rust fallback tables + CJK + currency)
        let words = expand_numbers_in_words(words, lang);
        prepared.extend(finalize_words_to_chunks(words, monologue.speaker, lang));
    }

    prepared
}

/// Retokenize prepared chunks into utterances using punctuation boundaries.
pub fn utterances_from_prepared_chunks(chunks: Vec<PreparedMonologueChunk>) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    for chunk in chunks {
        utterances.extend(retokenize(chunk.speaker, chunk.words));
    }
    utterances
}

/// Apply the post-retokenization cleanup passes shared by all ASR paths.
pub fn finalize_utterances(utterances: &mut [Utterance], lang: &str) {
    // Matches BA2's DisfluencyReplacementEngine which ran after ASR on all utterances.
    cleanup::apply_disfluency_replacements(utterances, lang);

    // Matches BA2's NgramRetraceEngine which ran after disfluency on all utterances.
    cleanup::apply_retrace_detection(utterances, lang);

    // 2026-04-23 transcribe-pipeline correction: utterance-initial
    // cap. The two per-word rules (I-cap, title-period strip) have
    // already run pre-retokenize in `finalize_words_to_chunks`;
    // this post-retokenize hook handles the per-utterance rule that
    // needs to see utterance boundaries. English-gated.
    cleanup::apply_english_transcribe_rules_post_retokenize(utterances, lang);
}

/// Split one prepared chunk into smaller prepared chunks according to word-level
/// utterance assignments.
pub fn split_prepared_chunk_by_assignments(
    chunk: &PreparedMonologueChunk,
    assignments: &[usize],
) -> Vec<PreparedMonologueChunk> {
    if chunk.words.len() <= 1 || assignments.is_empty() || assignments.len() != chunk.words.len() {
        return vec![chunk.clone()];
    }

    let mut split_chunks = Vec::new();
    let mut current_group = assignments[0];
    let mut current_words = Vec::new();

    for (word, group) in chunk.words.iter().cloned().zip(assignments.iter().copied()) {
        if !current_words.is_empty() && group != current_group {
            split_chunks.push(PreparedMonologueChunk {
                speaker: chunk.speaker,
                words: std::mem::take(&mut current_words),
            });
            current_group = group;
        }
        current_words.push(word);
    }

    if !current_words.is_empty() {
        split_chunks.push(PreparedMonologueChunk {
            speaker: chunk.speaker,
            words: current_words,
        });
    }

    if split_chunks.is_empty() {
        vec![chunk.clone()]
    } else {
        split_chunks
    }
}

/// Extract timed words from ASR elements, converting seconds to milliseconds.
///
/// Filters out pause markers (like `<pause>`) and blank values.
fn extract_timed_words(elements: &[AsrElement]) -> Vec<AsrWord> {
    let mut words = Vec::new();
    for elem in elements {
        let value = elem.value.as_str().trim();
        if value.is_empty() {
            continue;
        }
        // Filter pause markers like <pause>, <inaudible>, etc.
        if value.starts_with('<') && value.ends_with('>') {
            continue;
        }
        let (start_ms, end_ms) = normalized_timing_range(elem.ts.as_f64(), elem.end_ts.as_f64());
        words.push(AsrWord::new(value, start_ms, end_ms));
    }
    words
}

/// Strip MOR_PUNCT separators from the boundaries of each word token, and
/// drop words that become empty after stripping.
///
/// ENDING_PUNCT (`.` `?` `!` etc.) is **not** stripped — retokenize needs
/// them for sentence boundary detection. Only MOR_PUNCT (comma, tag `„`,
/// vocative `‡`) and RTL separators (`،` `؛`) are removed.
///
/// Case is preserved: uppercase tokens from the ASR provider (pronoun
/// "I", proper nouns like "Sarah" / "Cincinnati", sentence-initial
/// capitals) survive unchanged. Downstream components that need
/// case-insensitive comparison (disfluency lookup, retrace detection)
/// lowercase at comparison time.
fn strip_separator_words(words: Vec<AsrWord>) -> Vec<AsrWord> {
    /// MOR_PUNCT separators (comma, tag, vocative) plus the Arabic /
    /// RTL comma (U+060C) and semicolon (U+061B). Hardcoded as chars
    /// rather than string slices because `trim_matches` takes
    /// `FnMut(char) -> bool`; see [`super::MOR_PUNCT`] for the string-
    /// slice equivalent used elsewhere.
    fn is_strippable(c: char) -> bool {
        matches!(
            c,
            ',' | talkbank_model::chars::TAG_MARKER
            | talkbank_model::chars::VOCATIVE_MARKER
            | '\u{060C}' // ARABIC COMMA
            | '\u{061B}' // ARABIC SEMICOLON
        )
    }

    trim_word_boundaries(words, is_strippable)
}

/// Trim characters matching `is_strip` from the boundaries of each word, and
/// drop any word whose text becomes empty.
///
/// Shared primitive behind every boundary-trim pass in
/// `prepare_words_pre_expansion` (Stage 2b separators, Stage 2c boundary
/// quotes). The text payload is rewritten only when trimming actually
/// changed the string, keeping the common no-op path allocation-free.
pub(super) fn trim_word_boundaries(
    words: Vec<AsrWord>,
    is_strip: impl Fn(char) -> bool,
) -> Vec<AsrWord> {
    words
        .into_iter()
        .filter_map(|mut word| {
            let stripped: &str = word.text.as_str().trim_matches(&is_strip);
            if stripped.is_empty() {
                return None;
            }
            if stripped.len() != word.text.as_str().len() {
                word.text = AsrNormalizedText::new(stripped);
            }
            Some(word)
        })
        .collect()
}

/// Split tokens containing spaces into multiple words with interpolated timestamps.
///
/// Also handles hyphen-prefixed words by joining them with the previous word.
fn split_multiword_tokens(words: Vec<AsrWord>, lang: &str) -> Vec<AsrWord> {
    let mut result: Vec<AsrWord> = Vec::new();

    for word in words {
        // Join hyphen-prefixed words with previous
        if word.text.starts_with('-') && !result.is_empty() {
            // SAFETY: `!result.is_empty()` guard above ensures last_mut() succeeds.
            #[allow(clippy::unwrap_used)]
            let prev = result.last_mut().unwrap();
            prev.text.push_str(word.text.as_str());
            prev.end_ms = word.end_ms;
            continue;
        }

        result.extend(split_chunk_word(word, lang));
    }

    result
}

fn normalized_timing_range(start_s: f64, end_s: f64) -> (Option<i64>, Option<i64>) {
    if !start_s.is_finite() || !end_s.is_finite() {
        return (None, None);
    }

    let start_ms = (start_s * 1000.0).round() as i64;
    let end_ms = (end_s * 1000.0).round() as i64;
    if end_ms <= start_ms {
        (None, None)
    } else {
        (Some(start_ms), Some(end_ms))
    }
}

fn split_chunk_word(word: AsrWord, lang: &str) -> Vec<AsrWord> {
    let mut parts: Vec<(String, bool)> = Vec::new();
    let mut current = String::new();

    let flush_current = |parts: &mut Vec<(String, bool)>, current: &mut String| {
        if !current.is_empty() {
            parts.push((std::mem::take(current), false));
        }
    };

    for ch in word.text.as_str().chars() {
        if ch.is_whitespace() {
            flush_current(&mut parts, &mut current);
            continue;
        }

        if let Some(separator) = normalized_split_separator(ch) {
            flush_current(&mut parts, &mut current);
            if let Some(text) = separator {
                parts.push((text.to_string(), true));
            }
            continue;
        }

        current.push(ch);
    }
    flush_current(&mut parts, &mut current);

    let mut expanded_parts: Vec<(String, bool)> = Vec::new();
    for (text, is_separator) in parts {
        if is_separator {
            expanded_parts.push((text, true));
            continue;
        }
        expanded_parts.extend(expand_language_part(text, lang));
    }
    let parts = expanded_parts;

    if parts.len() == 1 && !parts[0].1 && parts[0].0 == word.text.as_str() {
        return vec![word];
    }

    let total_text_chars: usize = parts
        .iter()
        .filter(|(_, is_separator)| !*is_separator)
        .map(|(text, _)| text.chars().count())
        .sum();

    let mut consumed_chars = 0usize;
    let total_span = match (word.start_ms, word.end_ms) {
        (Some(start), Some(end)) if end > start && total_text_chars > 0 => {
            Some((start, end - start))
        }
        _ => None,
    };

    parts
        .into_iter()
        .map(|(text, is_separator)| {
            if is_separator {
                return AsrWord::new(text, None, None);
            }

            let timings = total_span.map(|(start, span)| {
                let part_chars = text.chars().count();
                let part_start = start + (span * consumed_chars as i64 / total_text_chars as i64);
                consumed_chars += part_chars;
                let part_end = start + (span * consumed_chars as i64 / total_text_chars as i64);
                (Some(part_start), Some(part_end))
            });

            let (start_ms, end_ms) = timings.unwrap_or((None, None));
            AsrWord::new(text, start_ms, end_ms)
        })
        .collect()
}

fn expand_language_part(text: String, lang: &str) -> Vec<(String, bool)> {
    if lang != "yue" || !should_split_cantonese_chars(&text) {
        return vec![(text, false)];
    }

    let tokens = cantonese::cantonese_char_tokens(&text);
    if tokens.len() <= 1 {
        return vec![(text, false)];
    }
    tokens.into_iter().map(|token| (token, false)).collect()
}

fn should_split_cantonese_chars(text: &str) -> bool {
    let mut has_cjk = false;
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch.is_ascii_digit() {
            return false;
        }
        if is_cjk_ideograph(ch) {
            has_cjk = true;
        }
    }
    has_cjk
}

fn is_cjk_ideograph(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2F800..=0x2FA1F
    )
}

fn normalized_split_separator(ch: char) -> Option<Option<&'static str>> {
    match ch {
        '.' => Some(Some(".")),
        '?' | '？' | '؟' => Some(Some("?")),
        '!' | '！' => Some(Some("!")),
        ',' | '，' | '、' | '،' => Some(Some(",")),
        '¿' | '¡' => Some(None),
        '。' => Some(Some(".")),
        _ => None,
    }
}

/// Normalize Cantonese text in all words (simplified→HK traditional + domain replacements).
fn normalize_cantonese_words(words: Vec<AsrWord>) -> Vec<AsrWord> {
    words
        .into_iter()
        .map(|w| AsrWord {
            text: w.text.map(cantonese::normalize_cantonese),
            ..w
        })
        .collect()
}

/// Split tokens of the form `{digits}%` into two words.
///
/// ASR providers (notably Rev.AI) sometimes emit `"80%"` as a single
/// word token. `%` is the CHAT dependent-tier sigil and cannot appear
/// as main-tier word content in *any* language, so the literal token
/// must never reach `build_chat`. This stage splits the offender into
/// two `AsrWord`s:
///
/// 1. The digit group (still a single token, timing = first portion
///    of the original span).
/// 2. The language-specific percent word (timing = remaining portion).
///
/// When the language has no mapped percent word (uncommon), the `%`
/// is stripped and the digit group is emitted alone — better than
/// producing malformed CHAT. When the language permits ASCII digits
/// in word content (yue/zho/cmn/nan/hak/min/cym/vie/tha), the digit
/// group is still emitted; the language-aware `ChatWordText`
/// validation will accept it.
///
/// Purely structural: this runs before number expansion so the digit
/// group can be expanded by the existing pipeline if the language
/// supports it.
fn split_percent_suffix_words(words: Vec<AsrWord>, lang: &str) -> Vec<AsrWord> {
    let mut result: Vec<AsrWord> = Vec::with_capacity(words.len());
    for word in words {
        let Some(digit_prefix) = word.text.as_str().strip_suffix('%') else {
            result.push(word);
            continue;
        };
        if digit_prefix.is_empty() || !digit_prefix.chars().all(|c| c.is_ascii_digit()) {
            // `%` not following a pure-digit prefix — not our case.
            result.push(word);
            continue;
        }

        let (digit_start, digit_end, percent_start, percent_end) =
            match (word.start_ms, word.end_ms) {
                (Some(start), Some(end)) if end > start => {
                    // Distribute timing proportionally by text length.
                    // The digit group already has 1-N characters; the
                    // percent suffix is the final 1 character (`%`).
                    // Roughly split the total span in that ratio so
                    // downstream FA can realign if the timing matters.
                    let total_chars = word.text.as_str().chars().count() as i64;
                    let digit_chars = digit_prefix.chars().count() as i64;
                    // Guard against div-by-zero even though total_chars
                    // is non-zero when we reach here (word.text was
                    // non-empty — the `%` alone is caught above).
                    let split = start + ((end - start) * digit_chars) / total_chars.max(1);
                    (Some(start), Some(split), Some(split), Some(end))
                }
                _ => (word.start_ms, word.end_ms, None, None),
            };

        let digit_word = AsrWord::new(digit_prefix, digit_start, digit_end);
        result.push(digit_word);

        if let Some(percent_word) = num2text::language_percent_word(lang) {
            result.push(AsrWord::new(percent_word, percent_start, percent_end));
        }
        // If no mapped percent word: we've already dropped the `%` by
        // constructing digit_word without the suffix. The main-tier
        // invariant is preserved; we just lose the "percent" semantic
        // marker until the table is extended for this language.
    }
    result
}

/// Expand digit strings to word form in all words.
///
/// Some expansions turn one input word into multi-word text (e.g.
/// `"100"` → `"one hundred"` via num2words, `"2001-2002"` →
/// `"two thousand one two thousand two"` via the dash-split branch).
/// Such outputs must be re-split into separate `AsrWord`s — a
/// `ChatWordText` holds one token on the main tier, and whitespace
/// inside a single word makes the fragment parser reject it as two
/// tokens glued together. Timing is distributed proportionally by
/// text length so downstream FA can realign if needed.
fn expand_numbers_in_words(words: Vec<AsrWord>, lang: &str) -> Vec<AsrWord> {
    words
        .into_iter()
        .flat_map(|w| {
            let expanded = expand_number(w.text.as_str(), lang);
            if !expanded.contains(char::is_whitespace) {
                return vec![AsrWord {
                    text: AsrNormalizedText::new(expanded),
                    ..w
                }];
            }
            split_expanded_text_into_words(&expanded, w.start_ms, w.end_ms, w.kind)
        })
        .collect()
}

/// Replace every `AsrWord` whose text contains whitespace with a
/// sequence of single-token `AsrWord`s, preserving overall timing.
///
/// Number expansion (`expand_number`) can turn one input token into multi-word text
/// (`"100"` → `"one hundred"`, `"2001-2002"` → `"two thousand one two
/// thousand two"`). A single `AsrWord` cannot carry whitespace —
/// `ChatWordText` holds one token on the main tier, and the fragment
/// parser rejects whitespace inside a word. This pass normalises such
/// outputs after expansion, distributing timing proportionally by
/// character count.
pub fn split_words_with_whitespace(words: &mut Vec<AsrWord>) {
    if !words
        .iter()
        .any(|w| w.text.as_str().contains(char::is_whitespace))
    {
        return;
    }
    let taken = std::mem::take(words);
    words.reserve(taken.len());
    for w in taken {
        if w.text.as_str().contains(char::is_whitespace) {
            words.extend(split_expanded_text_into_words(
                w.text.as_str(),
                w.start_ms,
                w.end_ms,
                w.kind,
            ));
        } else {
            words.push(w);
        }
    }
}

/// Distribute a whitespace-separated expansion across several
/// `AsrWord`s, proportioning timing by text length.
///
/// Called only on the post-expansion path — input comes from
/// `expand_number`, which is deterministic given the original token,
/// so no need to re-run any normalization on the split parts.
fn split_expanded_text_into_words(
    expanded: &str,
    start_ms: Option<i64>,
    end_ms: Option<i64>,
    kind: WordKind,
) -> Vec<AsrWord> {
    let parts: Vec<&str> = expanded.split_whitespace().collect();
    if parts.is_empty() {
        return Vec::new();
    }

    let total_chars: i64 = parts.iter().map(|p| p.chars().count() as i64).sum();
    let span = match (start_ms, end_ms) {
        (Some(s), Some(e)) if e > s && total_chars > 0 => Some((s, e - s)),
        _ => None,
    };

    let mut consumed: i64 = 0;
    parts
        .into_iter()
        .map(|part| {
            let part_chars = part.chars().count() as i64;
            let (ps, pe) = match span {
                Some((s, dur)) => {
                    let start = s + (dur * consumed) / total_chars.max(1);
                    consumed += part_chars;
                    let end = s + (dur * consumed) / total_chars.max(1);
                    (Some(start), Some(end))
                }
                None => (None, None),
            };
            AsrWord {
                text: AsrNormalizedText::new(part),
                start_ms: ps,
                end_ms: pe,
                kind,
            }
        })
        .collect()
}

/// Split a word list into chunks of at most [`MAX_TURN_LEN`].
fn split_long_turns(words: Vec<AsrWord>) -> Vec<Vec<AsrWord>> {
    if words.len() <= MAX_TURN_LEN {
        return vec![words];
    }
    words.chunks(MAX_TURN_LEN).map(|c| c.to_vec()).collect()
}

fn split_on_long_pauses(chunks: Vec<Vec<AsrWord>>) -> Vec<Vec<AsrWord>> {
    let mut result = Vec::new();

    for chunk in chunks {
        if chunk.len() <= 1 {
            result.push(chunk);
            continue;
        }

        let mut current = Vec::new();
        let mut previous_timed: Option<AsrWord> = None;

        for word in chunk {
            if previous_timed
                .as_ref()
                .is_some_and(|prev| long_pause_starts_new_utterance(prev, &word, current.len()))
                && !current.is_empty()
            {
                result.push(std::mem::take(&mut current));
            }

            if word.start_ms.is_some() && word.end_ms.is_some() {
                previous_timed = Some(word.clone());
            }
            current.push(word);
        }

        if !current.is_empty() {
            result.push(current);
        }
    }

    result
}

fn long_pause_starts_new_utterance(prev: &AsrWord, next: &AsrWord, current_len: usize) -> bool {
    if current_len < 2 {
        return false;
    }

    let (Some(prev_end), Some(next_start)) = (prev.end_ms, next.start_ms) else {
        return false;
    };
    if next_start - prev_end < LONG_PAUSE_SPLIT_MS {
        return false;
    }

    let starter = next
        .text
        .as_str()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_ascii_lowercase();
    LONG_PAUSE_SENTENCE_STARTERS.contains(&starter.as_str())
}

/// Check if a word is or ends with a sentence-ending punctuation mark.
fn is_ending_punct(word: &str) -> bool {
    if ENDING_PUNCT.contains(&word) {
        return true;
    }
    // Check RTL punctuation
    for (rtl, _) in RTL_PUNCT {
        if word == *rtl {
            return true;
        }
    }
    false
}

/// Check if a word ends with ending punctuation (last char).
fn ends_with_ending_punct(word: &str) -> bool {
    match word.chars().last() {
        Some(c) => {
            let mut buf = [0u8; 4];
            is_ending_punct(c.encode_utf8(&mut buf))
        }
        None => false,
    }
}

/// Normalize RTL punctuation to ASCII equivalent.
fn normalize_punct(word: &str) -> String {
    for (rtl, ascii) in RTL_PUNCT {
        if word == *rtl {
            return ascii.to_string();
        }
    }
    word.to_string()
}

/// Split a word list into utterances based on punctuation boundaries.
///
/// This is the simple punctuation-based retokenizer (no BERT model).
///
/// # Panic safety
///
/// The `unwrap()` calls on `buf.last()`, `buf.last_mut()`, and `buf.pop()` are
/// all immediately preceded by `buf.push(word)` within the same loop iteration,
/// so `buf` is guaranteed non-empty at each call site.
#[allow(clippy::unwrap_used)]
fn retokenize(speaker: SpeakerIndex, words: Vec<AsrWord>) -> Vec<Utterance> {
    let mut utterances = Vec::new();
    let mut buf: Vec<AsrWord> = Vec::new();

    for word in words {
        // Normalize Japanese period and remove inverted punctuation
        let word = AsrWord {
            text: word
                .text
                .map(|t| t.replace('。', ".").replace(['¿', '¡'], " ")),
            ..word
        };

        buf.push(word);

        let last_text = buf.last().unwrap().text.as_str();

        if is_ending_punct(last_text) {
            // Whole word is ending punct — flush utterance
            let punct = normalize_punct(last_text);
            buf.last_mut().unwrap().text = AsrNormalizedText::new(punct);
            utterances.push(Utterance {
                speaker,
                words: std::mem::take(&mut buf),
                lang: None,
            });
        } else if ends_with_ending_punct(last_text) {
            // Last character is ending punct — split the word
            let text = buf.pop().unwrap();
            let s = text.text.as_str();
            let last_char_boundary = s.char_indices().next_back().map(|(i, _)| i).unwrap_or(0);
            let word_part = &s[..last_char_boundary];
            let punct_part = &s[last_char_boundary..];

            if !word_part.is_empty() {
                buf.push(AsrWord::new(word_part, text.start_ms, text.end_ms));
            }
            buf.push(AsrWord::new(normalize_punct(punct_part), None, None));
            utterances.push(Utterance {
                speaker,
                words: std::mem::take(&mut buf),
                lang: None,
            });
        }
    }

    // Flush remaining words
    if !buf.is_empty() {
        // Remove trailing MOR_PUNCT
        while buf
            .last()
            .is_some_and(|w| MOR_PUNCT.contains(&w.text.as_str()))
        {
            buf.pop();
        }
        if !buf.is_empty() {
            // Append terminator
            buf.push(AsrWord::new(".", None, None));
            utterances.push(Utterance {
                speaker,
                words: buf,
                lang: None,
            });
        }
    }

    utterances
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(value: &str, ts: f64, end_ts: f64) -> AsrElement {
        AsrElement {
            value: AsrRawText::new(value),
            ts: AsrTimestampSecs(ts),
            end_ts: AsrTimestampSecs(end_ts),
            kind: AsrElementKind::Text,
        }
    }

    #[test]
    fn bare_quote_element_is_stripped_at_stage_2c() {
        // A standalone `"` ASR element has no semantic content. Stage 2c
        // (boundary-quote strip) must drop it before it reaches the gate.
        let elements = vec![
            elem("\"", 0.0, 0.137),
            elem("Ross", 0.137, 0.685),
            elem("said", 0.685, 1.0),
        ];
        let words = prepare_words_pre_expansion(&elements, "eng");
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert!(
            !texts.iter().any(|t| *t == "\""),
            "bare `\"` element should not survive Stage 2c, got: {texts:?}"
        );
    }

    #[test]
    fn embedded_quote_in_multi_word_element_is_stripped_at_stage_3c() {
        // ASR sometimes emits one element whose value glues a `"` to
        // adjacent punctuation: `Ross." said.`. Stage 3 splits on `.`
        // and whitespace, producing a standalone `"` part that bypasses
        // Stage 2c (which ran before the split). Stage 3c re-runs the
        // boundary-quote strip after the split to drop it.
        let elements = vec![elem("Ross.\" said.", 0.0, 1.0)];
        let words = prepare_words_pre_expansion(&elements, "eng");
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert!(
            !texts.iter().any(|t| *t == "\""),
            "post-split `\"` part should not survive Stage 3c, got: {texts:?}"
        );
    }

    #[test]
    fn full_transcribe_pipeline_drops_isolated_quote_element() {
        // End-to-end check via `process_raw_asr`: when ASR emits a bare
        // `"` element between sentences (Whisper's verbatim quoted-speech
        // rendering), the resulting utterances contain no `"` token.
        let asr_output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("He's", 0.0, 0.3),
                    elem("a", 0.3, 0.4),
                    elem("droid", 0.4, 0.7),
                    elem(".", 0.7, 0.7),
                    elem("\"", 0.8, 0.95),
                    elem("Ross", 0.95, 1.5),
                    elem("said", 1.5, 2.0),
                    elem(".", 2.0, 2.0),
                ],
            }],
        };
        let utterances = process_raw_asr(&asr_output, "eng");
        let bare_quote_utt = utterances
            .iter()
            .position(|u| u.words.iter().any(|w| w.text.as_str() == "\""));
        assert!(
            bare_quote_utt.is_none(),
            "bare `\"` should not survive the full pipeline; found in utt {bare_quote_utt:?}"
        );
    }

    #[test]
    fn test_extract_timed_words_filters_pauses() {
        let elems = vec![
            elem("hello", 0.0, 0.5),
            elem("<pause>", 0.5, 1.0),
            elem("world", 1.0, 1.5),
        ];
        let words = extract_timed_words(&elems);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "hello");
        assert_eq!(words[1].text, "world");
    }

    #[test]
    fn test_extract_timed_words_converts_to_ms() {
        let elems = vec![elem("hello", 1.234, 2.567)];
        let words = extract_timed_words(&elems);
        assert_eq!(words[0].start_ms, Some(1234));
        assert_eq!(words[0].end_ms, Some(2567));
    }

    #[test]
    fn test_extract_timed_words_treats_zero_duration_as_untimed() {
        let elems = vec![elem("hello", 0.0, 0.0)];
        let words = extract_timed_words(&elems);
        assert_eq!(words[0].start_ms, None);
        assert_eq!(words[0].end_ms, None);
    }

    #[test]
    fn test_split_multiword_tokens() {
        let words = vec![AsrWord::new("hello world", Some(0), Some(1000))];
        let result = split_multiword_tokens(words, "eng");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello");
        assert_eq!(result[0].start_ms, Some(0));
        assert_eq!(result[0].end_ms, Some(500));
        assert_eq!(result[1].text, "world");
        assert_eq!(result[1].start_ms, Some(500));
        assert_eq!(result[1].end_ms, Some(1000));
    }

    #[test]
    fn test_split_multiword_tokens_splits_embedded_sentence_punctuation() {
        let words = vec![AsrWord::new("hello?world!", None, None)];
        let result = split_multiword_tokens(words, "eng");
        let texts: Vec<&str> = result.iter().map(|word| word.text.as_str()).collect();
        assert_eq!(texts, vec!["hello", "?", "world", "!"]);
        assert!(
            result
                .iter()
                .all(|word| word.start_ms.is_none() && word.end_ms.is_none())
        );
    }

    #[test]
    fn test_hyphen_joining() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("-world", Some(500), Some(1000)),
        ];
        let result = split_multiword_tokens(words, "eng");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello-world");
        assert_eq!(result[0].start_ms, Some(0));
        assert_eq!(result[0].end_ms, Some(1000));
    }

    #[test]
    fn test_split_long_turns() {
        let words: Vec<AsrWord> = (0..650)
            .map(|i| AsrWord::new(format!("word{i}"), Some(i as i64), Some(i as i64 + 1)))
            .collect();
        let chunks = split_long_turns(words);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 300);
        assert_eq!(chunks[1].len(), 300);
        assert_eq!(chunks[2].len(), 50);
    }

    #[test]
    fn test_split_on_long_pauses_uses_sentence_starters() {
        let chunks = split_on_long_pauses(vec![vec![
            AsrWord::new("On", Some(65), Some(285)),
            AsrWord::new("television", Some(285), Some(765)),
            AsrWord::new("Have", Some(1595), Some(1885)),
            AsrWord::new("you", Some(1885), Some(1965)),
            AsrWord::new("ever", Some(1965), Some(2085)),
            AsrWord::new("been", Some(2085), Some(2285)),
            AsrWord::new("on", Some(2285), Some(2445)),
            AsrWord::new("television", Some(2445), Some(2925)),
            AsrWord::new("Well", Some(4845), Some(5135)),
            AsrWord::new("you", Some(5135), Some(5295)),
            AsrWord::new("know", Some(5295), Some(5375)),
            AsrWord::new("we", Some(5375), Some(5535)),
            AsrWord::new("bring", Some(5535), Some(5735)),
            AsrWord::new("some", Some(5735), Some(5935)),
            AsrWord::new("kids", Some(5935), Some(6135)),
            AsrWord::new("Do", Some(7875), Some(8095)),
            AsrWord::new("you", Some(8095), Some(8175)),
            AsrWord::new("like", Some(8175), Some(8375)),
            AsrWord::new("to", Some(8375), Some(8495)),
            AsrWord::new("play", Some(8495), Some(8695)),
        ]]);

        let texts: Vec<String> = chunks
            .into_iter()
            .map(|chunk| {
                chunk
                    .into_iter()
                    .map(|word| word.text.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        assert_eq!(
            texts,
            vec![
                "On television",
                "Have you ever been on television",
                "Well you know we bring some kids",
                "Do you like to play",
            ]
        );
    }

    #[test]
    fn test_retokenize_simple() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("world", Some(500), Some(1000)),
            AsrWord::new(".", None, None),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].speaker, SpeakerIndex(0));
        assert_eq!(utts[0].words.len(), 3);
        assert_eq!(utts[0].words[2].text, ".");
    }

    #[test]
    fn test_retokenize_splits_on_period() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new(".", Some(500), Some(600)),
            AsrWord::new("world", Some(600), Some(1000)),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 2);
        assert_eq!(utts[0].words.len(), 2); // hello .
        assert_eq!(utts[0].words[0].text, "hello");
        assert_eq!(utts[0].words[1].text, ".");
        assert_eq!(utts[1].words.len(), 2); // world .
        assert_eq!(utts[1].words[0].text, "world");
        assert_eq!(utts[1].words[1].text, "."); // auto-appended
    }

    #[test]
    fn test_retokenize_trailing_no_terminator() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("world", Some(500), Some(1000)),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words.last().unwrap().text, "."); // auto-appended
    }

    #[test]
    fn test_retokenize_rtl_punct() {
        let words = vec![
            AsrWord::new("hello", Some(0), Some(500)),
            AsrWord::new("؟", None, None),
        ];
        let utts = retokenize(SpeakerIndex(0), words);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words[1].text, "?"); // normalized
    }

    /// Golden test: matches Python `_process_raw_asr` output for simple input.
    #[test]
    fn test_process_raw_asr_golden_simple() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem("world", 0.5, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                    elem("how", 1.5, 2.0),
                    elem("are", 2.0, 2.3),
                    elem("you", 2.3, 2.5),
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts.len(), 2);

        // First utterance: Hello world .
        // (Utterance-initial cap fires on English — rule landed 2026-04-23.)
        assert_eq!(utts[0].words[0].text, "Hello");
        assert_eq!(utts[0].words[0].start_ms, Some(0));
        assert_eq!(utts[0].words[1].text, "world");
        assert_eq!(utts[0].words[1].start_ms, Some(500));
        assert_eq!(utts[0].words[2].text, ".");

        // Second utterance: How are you .
        assert_eq!(utts[1].words[0].text, "How");
        assert_eq!(utts[1].words[0].start_ms, Some(1500));
        assert_eq!(utts[1].words.last().unwrap().text, ".");
    }

    /// Golden test: compound merging in pipeline.
    #[test]
    fn test_process_raw_asr_golden_compound() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("the", 0.0, 0.3),
                    elem("air", 0.3, 0.6),
                    elem("plane", 0.6, 0.9),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(0.9),
                        end_ts: AsrTimestampSecs(1.0),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts.len(), 1);
        // Utterance-initial cap (2026-04-23 rule) uppercases `The`.
        assert_eq!(utts[0].words[0].text, "The");
        assert_eq!(utts[0].words[1].text, "airplane");
    }

    /// Golden test: number expansion in pipeline.
    #[test]
    fn test_process_raw_asr_golden_number() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("I", 0.0, 0.3),
                    elem("have", 0.3, 0.6),
                    elem("5", 0.6, 0.9),
                    elem("cats", 0.9, 1.2),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.2),
                        end_ts: AsrTimestampSecs(1.3),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].words[2].text, "five");
    }

    #[test]
    fn test_process_raw_asr_splits_unpunctuated_turn_on_long_pause_starters() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("On", 0.065, 0.285),
                    elem("television", 0.285, 0.765),
                    elem("Have", 1.595, 1.885),
                    elem("you", 1.885, 1.965),
                    elem("ever", 1.965, 2.085),
                    elem("been", 2.085, 2.285),
                    elem("on", 2.285, 2.445),
                    elem("television", 2.445, 2.925),
                    elem("Well", 4.845, 5.135),
                    elem("you", 5.135, 5.295),
                    elem("know", 5.295, 5.375),
                    elem("we", 5.375, 5.535),
                    elem("bring", 5.535, 5.735),
                    elem("some", 5.735, 5.935),
                    elem("kids", 5.935, 6.135),
                    elem("in", 6.135, 6.335),
                    elem("here", 6.335, 6.455),
                    elem("and", 6.455, 6.575),
                    elem("we'd", 6.575, 6.815),
                    elem("be", 6.815, 6.895),
                    elem("playing", 6.895, 7.135),
                    elem("all", 7.275, 7.495),
                    elem("the", 7.495, 7.615),
                    elem("time", 7.615, 7.775),
                    elem("Do", 7.875, 8.095),
                    elem("you", 8.095, 8.175),
                    elem("like", 8.175, 8.375),
                    elem("to", 8.375, 8.495),
                    elem("play", 8.495, 8.695),
                    elem("What", 10.405, 10.695),
                    elem("do", 10.695, 10.815),
                    elem("you", 10.815, 10.935),
                ],
            }],
        };

        let utts = process_raw_asr(&output, "eng");
        let texts: Vec<String> = utts
            .iter()
            .map(|utt| {
                utt.words
                    .iter()
                    .map(|word| word.text.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        assert_eq!(utts.len(), 4);
        assert_eq!(
            texts,
            vec![
                "On television .",
                "Have you ever been on television .",
                "Well you know we bring some kids in here and we'd be playing all the time Do you like to play .",
                "What do you .",
            ]
        );
    }

    #[test]
    fn test_process_raw_asr_preserves_same_speaker_monologue_boundaries() {
        let output = AsrOutput {
            monologues: vec![
                AsrMonologue {
                    speaker: SpeakerIndex(0),
                    elements: vec![elem("on", 0.065, 0.285), elem("television", 0.285, 0.765)],
                },
                AsrMonologue {
                    speaker: SpeakerIndex(0),
                    elements: vec![
                        elem("have", 1.595, 1.885),
                        elem("you", 1.885, 1.965),
                        elem("ever", 1.965, 2.085),
                        elem("been", 2.085, 2.285),
                        elem("on", 2.285, 2.445),
                        elem("television", 2.445, 2.925),
                    ],
                },
            ],
        };

        let utts = process_raw_asr(&output, "eng");
        let texts: Vec<String> = utts
            .iter()
            .map(|utt| {
                utt.words
                    .iter()
                    .map(|word| word.text.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();

        assert_eq!(
            texts,
            vec!["On television .", "Have you ever been on television ."]
        );
    }

    #[test]
    fn test_split_prepared_chunk_by_assignments_preserves_speaker_and_groups() {
        let chunk = PreparedMonologueChunk {
            speaker: SpeakerIndex(2),
            words: vec![
                AsrWord::new("on", Some(0), Some(100)),
                AsrWord::new("television", Some(100), Some(200)),
                AsrWord::new("have", Some(300), Some(400)),
                AsrWord::new("you", Some(400), Some(500)),
            ],
        };

        let split = split_prepared_chunk_by_assignments(&chunk, &[0, 0, 1, 1]);
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].speaker, SpeakerIndex(2));
        assert_eq!(
            split[0]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["on", "television"]
        );
        assert_eq!(
            split[1]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["have", "you"]
        );
    }

    /// Golden test: Cantonese normalization in pipeline.
    #[test]
    fn test_process_raw_asr_golden_cantonese() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("你", 0.0, 0.3),
                    elem("真系", 0.3, 0.6),
                    elem("好", 0.6, 0.9),
                    elem("吵", 0.9, 1.2),
                    elem("呀", 1.2, 1.5),
                ],
            }],
        };
        let utts = process_raw_asr(&output, "yue");
        assert_eq!(utts.len(), 1);
        let tokens: Vec<&str> = utts[0]
            .words
            .iter()
            .map(|word| word.text.as_str())
            .collect();
        assert_eq!(tokens, vec!["你", "真", "係", "好", "嘈", "啊", "."]);
    }

    /// Cantonese normalization should NOT activate for non-yue languages.
    #[test]
    fn test_process_raw_asr_no_cantonese_for_eng() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem("系", 0.0, 0.5)],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        assert_eq!(utts[0].words[0].text, "系"); // NOT normalized
    }

    #[test]
    fn test_process_raw_asr_handles_single_chunk_cantonese_whisper_output() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem(
                    "這麼搞笑?我還清了啊!我還覺得奇怪為什麼在一個三次頭的電話打工呢?",
                    0.0,
                    0.0,
                )],
            }],
        };

        let utts = process_raw_asr(&output, "yue");
        assert_eq!(utts.len(), 3);
        assert_eq!(utts[0].words.last().unwrap().text, "?");
        assert_eq!(utts[1].words.last().unwrap().text, "!");
        assert_eq!(utts[2].words.last().unwrap().text, "?");
        assert_eq!(
            utts[0]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["這", "麼", "搞", "笑", "?"]
        );
        assert!(
            utts.iter()
                .flat_map(|utt| utt.words.iter())
                .filter(|word| !matches!(word.text.as_str(), "." | "!" | "?"))
                .count()
                > 10
        );
        assert!(
            utts.iter()
                .flat_map(|utt| utt.words.iter())
                .all(|word| !(word.start_ms == Some(0) && word.end_ms == Some(0)))
        );
    }

    #[test]
    fn test_process_raw_asr_keeps_ascii_words_intact_for_yue() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem("hello", 0.0, 0.5)],
            }],
        };

        let utts = process_raw_asr(&output, "yue");
        assert_eq!(utts.len(), 1);
        assert_eq!(
            utts[0]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["hello", "."]
        );
    }

    // ── MOR_PUNCT / separator stripping ─────────────────────────────
    //
    // BA2 stripped ALL MOR_PUNCT (comma `,`, tag `„`, vocative `‡`) and
    // RTL punctuation from ASR word tokens BEFORE utseg/CHAT building:
    //
    //   for j in MOR_PUNCT + ENDING_PUNCT + ["؟", "۔", "،", "؛"]:
    //       i[0] = i[0].strip(j).lower()
    //   utterance = [i for i in utterance if i[0].strip() != ""]
    //
    // This prevents separators from surviving into CHAT as misplaced
    // Separator nodes or invariant-violating Word nodes.

    /// A standalone comma word token must be stripped from ASR output.
    /// BA2 stripped it in utils.py:108. Without stripping, it survives
    /// into CHAT and can land at utterance boundaries after utseg split.
    #[test]
    fn mor_punct_comma_stripped_from_asr_words() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem(",", 0.5, 0.6), // comma as a word token, not Punctuation
                    elem("world", 0.6, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        let words: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert!(
            !words.contains(&","),
            "standalone comma word should be stripped, got: {words:?}"
        );
    }

    /// A word with trailing comma ("dishes,") must have the comma stripped.
    #[test]
    fn mor_punct_trailing_comma_stripped() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("dishes,", 0.0, 0.5),
                    elem("or", 0.6, 0.8),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(0.8),
                        end_ts: AsrTimestampSecs(0.9),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        let words: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        // Utterance-initial cap (2026-04-23) uppercases the first
        // word; the assertion here is about comma-stripping, which
        // continues to work regardless of case.
        assert_eq!(
            words[0], "Dishes",
            "trailing comma should be stripped from 'dishes,', got: {words:?}"
        );
    }

    /// Tag marker `„` must be stripped from ASR word tokens.
    #[test]
    fn mor_punct_tag_marker_stripped() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem("\u{201E}", 0.5, 0.6), // „ tag marker
                    elem("world", 0.6, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        let words: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert!(
            !words.contains(&"\u{201E}"),
            "tag marker „ should be stripped, got: {words:?}"
        );
    }

    /// Vocative marker `‡` must be stripped from ASR word tokens.
    #[test]
    fn mor_punct_vocative_marker_stripped() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem("\u{2021}", 0.5, 0.6), // ‡ vocative
                    elem("world", 0.6, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        let words: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert!(
            !words.contains(&"\u{2021}"),
            "vocative marker ‡ should be stripped, got: {words:?}"
        );
    }

    /// RTL comma `،` must be stripped (BA2: in the explicit RTL punct list).
    #[test]
    fn rtl_comma_stripped() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem("،", 0.5, 0.6), // Arabic comma
                    elem("world", 0.6, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        let words: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert!(
            !words.contains(&"،"),
            "RTL comma ، should be stripped, got: {words:?}"
        );
    }

    /// After stripping, words that become empty must be removed entirely.
    /// BA2: `utterance = [i for i in utterance if i[0].strip() != ""]`
    #[test]
    fn stripped_empty_words_removed() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("hello", 0.0, 0.5),
                    elem(",", 0.5, 0.6),  // becomes empty after strip
                    elem(",,", 0.6, 0.7), // also becomes empty
                    elem("world", 0.7, 1.0),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.0),
                        end_ts: AsrTimestampSecs(1.1),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };
        let utts = process_raw_asr(&output, "eng");
        let words: Vec<&str> = utts[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(
            words,
            vec!["Hello", "world", "."],
            "empty words after stripping should be removed, got: {words:?}"
        );
    }

    // ── Split pipeline equivalence tests ────────────────────────────────

    /// Verify that the split pipeline (pre_expansion → expand → finalize)
    /// produces identical output to the monolithic `prepare_asr_chunks`.
    #[test]
    fn split_pipeline_matches_monolithic_simple() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("I", 0.0, 0.3),
                    elem("have", 0.3, 0.6),
                    elem("5", 0.6, 0.9),
                    elem("cats", 0.9, 1.2),
                    AsrElement {
                        value: AsrRawText::new("."),
                        ts: AsrTimestampSecs(1.2),
                        end_ts: AsrTimestampSecs(1.3),
                        kind: AsrElementKind::Punctuation,
                    },
                ],
            }],
        };

        let monolithic = prepare_asr_chunks(&output, "eng");

        // Split path: pre-expand → expand → finalize per monologue.
        let mut split_result = Vec::new();
        for monologue in &output.monologues {
            let words = prepare_words_pre_expansion(&monologue.elements, "eng");
            let words = expand_numbers_in_words(words, "eng");
            split_result.extend(finalize_words_to_chunks(words, monologue.speaker, "eng"));
        }

        assert_eq!(monolithic.len(), split_result.len());
        for (m, s) in monolithic.iter().zip(split_result.iter()) {
            assert_eq!(m.speaker, s.speaker);
            let m_texts: Vec<&str> = m.words.iter().map(|w| w.text.as_str()).collect();
            let s_texts: Vec<&str> = s.words.iter().map(|w| w.text.as_str()).collect();
            assert_eq!(m_texts, s_texts, "chunk word texts differ");
        }
    }

    /// Split pipeline with Cantonese normalization.
    #[test]
    fn split_pipeline_matches_monolithic_cantonese() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    elem("你", 0.0, 0.3),
                    elem("真系", 0.3, 0.6),
                    elem("好", 0.6, 0.9),
                ],
            }],
        };

        let monolithic = prepare_asr_chunks(&output, "yue");

        let mut split_result = Vec::new();
        for monologue in &output.monologues {
            let words = prepare_words_pre_expansion(&monologue.elements, "yue");
            let words = expand_numbers_in_words(words, "yue");
            split_result.extend(finalize_words_to_chunks(words, monologue.speaker, "yue"));
        }

        assert_eq!(monolithic.len(), split_result.len());
        for (m, s) in monolithic.iter().zip(split_result.iter()) {
            let m_texts: Vec<&str> = m.words.iter().map(|w| w.text.as_str()).collect();
            let s_texts: Vec<&str> = s.words.iter().map(|w| w.text.as_str()).collect();
            assert_eq!(m_texts, s_texts);
        }
    }

    /// Split pipeline with multiple monologues.
    #[test]
    fn split_pipeline_matches_monolithic_multi_monologue() {
        let output = AsrOutput {
            monologues: vec![
                AsrMonologue {
                    speaker: SpeakerIndex(0),
                    elements: vec![elem("hello", 0.0, 0.5), elem("world", 0.5, 1.0)],
                },
                AsrMonologue {
                    speaker: SpeakerIndex(1),
                    elements: vec![elem("42", 2.0, 2.5), elem("things", 2.5, 3.0)],
                },
            ],
        };

        let monolithic = prepare_asr_chunks(&output, "eng");

        let mut split_result = Vec::new();
        for monologue in &output.monologues {
            let words = prepare_words_pre_expansion(&monologue.elements, "eng");
            let words = expand_numbers_in_words(words, "eng");
            split_result.extend(finalize_words_to_chunks(words, monologue.speaker, "eng"));
        }

        assert_eq!(monolithic.len(), split_result.len());
        for (m, s) in monolithic.iter().zip(split_result.iter()) {
            assert_eq!(m.speaker, s.speaker);
            let m_texts: Vec<&str> = m.words.iter().map(|w| w.text.as_str()).collect();
            let s_texts: Vec<&str> = s.words.iter().map(|w| w.text.as_str()).collect();
            assert_eq!(m_texts, s_texts);
        }
    }

    /// Currency tokens must survive the split pipeline. Regression test for
    /// a bug where the Rust fallback pass had a digits-only guard that
    /// silently dropped "$12" because it starts with '$'.
    #[test]
    fn split_pipeline_expands_currency_tokens() {
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![elem("costs", 0.0, 0.5), elem("$12", 0.5, 1.0)],
            }],
        };

        // The monolithic path calls expand_number on every word, which
        // handles currency via try_expand_currency.
        let monolithic = prepare_asr_chunks(&output, "eng");
        let m_texts: Vec<&str> = monolithic[0]
            .words
            .iter()
            .map(|w| w.text.as_str())
            .collect();
        assert!(
            m_texts.iter().any(|w| w.contains("dollars")),
            "monolithic pipeline should expand $12: {m_texts:?}"
        );

        // The split path must also expand currency via the Rust residual pass.
        let mut split_result = Vec::new();
        for monologue in &output.monologues {
            let mut words = prepare_words_pre_expansion(&monologue.elements, "eng");
            // No Python expansion — currency is Rust-only.
            // Simulate what the pipeline does: call expand_number on each word.
            for word in &mut words {
                let text = word.text.as_str();
                let expanded = expand_number(text, "eng");
                if expanded != text {
                    word.text = AsrNormalizedText::new(&expanded);
                }
            }
            split_result.extend(finalize_words_to_chunks(words, monologue.speaker, "eng"));
        }
        let s_texts: Vec<&str> = split_result[0]
            .words
            .iter()
            .map(|w| w.text.as_str())
            .collect();
        assert!(
            s_texts.iter().any(|w| w.contains("dollars")),
            "split pipeline should expand $12 via Rust residual: {s_texts:?}"
        );
    }
}
