//! WER word conforming for benchmark evaluation.
//!
//! Normalizes words before Word Error Rate (WER) comparison by applying
//! deterministic transformations: compound splitting, contraction expansion,
//! filler normalization, name replacement, abbreviation expansion, and special
//! word handling.
//!
//! This is the Rust replacement for the Python `_conform()` function from
//! `inference/benchmark.py`. Which rules apply is a [`WerNormalization`],
//! chosen once per comparison and applied to both transcripts.
//!
//! # Data files
//!
//! The module loads four embedded JSON data files at first use via [`LazyLock`]:
//!
//! - `compounds.json`: 3,584 compound word pairs (shared with [`asr_postprocess`])
//! - `names.json`: ~6,700 proper names (lowercased)
//! - `abbrev.json`: ~400 abbreviations (original case)
//!
//! [`asr_postprocess`]: crate::asr_postprocess

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Compound word lookup: maps joined compound → (part_a, part_b) for O(1) splitting.
///
/// Reuses the same data file as [`asr_postprocess::compounds`](crate::asr_postprocess).
// Data is compile-time-constant: `include_str!` embeds the JSON at build time.
#[allow(clippy::expect_used)]
static COMPOUND_MAP: LazyLock<HashMap<String, (String, String)>> = LazyLock::new(|| {
    let data: Vec<[String; 2]> = serde_json::from_str(include_str!("../data/compounds.json"))
        .expect("embedded compounds.json is valid");
    data.into_iter()
        .map(|[a, b]| {
            let joined = format!("{a}{b}");
            (joined, (a, b))
        })
        .collect()
});

/// Known proper names (lowercased). Replaced with `"name"` during WER evaluation
/// to avoid penalizing name recognition errors.
// Data is compile-time-constant: `include_str!` embeds the JSON at build time.
#[allow(clippy::expect_used)]
static NAMES: LazyLock<HashSet<String>> = LazyLock::new(|| {
    let data: Vec<String> = serde_json::from_str(include_str!("../data/names.json"))
        .expect("embedded names.json is valid");
    data.into_iter().collect()
});

/// Known abbreviations (original case). Letter-expanded during WER evaluation
/// (e.g., `"FBI"` → `["F", "B", "I"]`).
///
/// Python checks abbreviations in original case (`i.strip() in abbrev`), not
/// lowercased, so we preserve that behavior here.
// Data is compile-time-constant: `include_str!` embeds the JSON at build time.
#[allow(clippy::expect_used)]
static ABBREV: LazyLock<HashSet<String>> = LazyLock::new(|| {
    let data: Vec<String> = serde_json::from_str(include_str!("../data/abbrev.json"))
        .expect("embedded abbrev.json is valid");
    data.into_iter().collect()
});

/// Common speech fillers, all normalized to `"um"` during WER evaluation.
static FILLERS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    ["um", "uhm", "em", "mhm", "uhhm", "eh", "uh", "hm"]
        .into_iter()
        .collect()
});

/// Which of the normalizer's rules a comparison applies.
///
/// Chosen ONCE per comparison and applied to both transcripts. It must not be
/// chosen per word or per side from either transcript's language labels: a
/// hypothesis word recognized correctly but labeled with the wrong language
/// would then be normalized differently from the identical gold word and
/// charged as an error, and a benchmark of code-switched speech exists to
/// measure those labels separately from recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WerNormalization {
    /// Every rule, including replacing known proper names with `name`.
    English,
    /// Every rule except name replacement.
    ///
    /// The name list holds everyday words of other languages (Spanish `linda`,
    /// `clara`, `flor`, `luz`), so on material that is not English it turns
    /// different words into the same token and scores them as matches. The
    /// other rules only fire on English forms and are harmless elsewhere.
    KeepNames,
}

impl WerNormalization {
    /// The normalization for a transcript that declares `languages`.
    ///
    /// A transcript declaring English alone gets every rule. One declaring any
    /// other language keeps names. A transcript declaring nothing is normalized
    /// as English, which is how every such comparison was normalized before
    /// this choice existed, so their numbers do not move.
    pub fn for_declared_languages(languages: &[talkbank_model::model::LanguageCode]) -> Self {
        match languages {
            [] => Self::English,
            [only] if only.as_str() == "eng" => Self::English,
            [_] | [_, _, ..] => Self::KeepNames,
        }
    }

    /// The normalization two transcripts can share: English only when both
    /// would be normalized as English.
    pub fn shared(self, other: Self) -> Self {
        match (self, other) {
            (Self::English, Self::English) => Self::English,
            (Self::KeepNames, _) | (_, Self::KeepNames) => Self::KeepNames,
        }
    }

    /// Normalize a word list, one word at a time.
    pub fn conform_words(self, words: &[String]) -> Vec<String> {
        let mut result = Vec::with_capacity(words.len());
        for word in words {
            self.conform_word_into(word, &mut result);
        }
        result
    }

    /// Normalize one word, appending its token(s) to `out`.
    ///
    /// The word is lowercased and checked against a priority-ordered rule
    /// chain; the first matching rule produces the output token(s). A word can
    /// produce more than one token (compound splits, contraction expansions).
    /// Appending to the caller's buffer rather than returning a vector is what
    /// lets a whole transcript be conformed without an allocation per word.
    ///
    /// # Transformation rules (in priority order)
    ///
    /// 1. **Compound splitting**: known compound words are split into their
    ///    constituent parts (e.g., `"airplane"` → `["air", "plane"]`).
    /// 2. **Abbreviation letter expansion**, known abbreviations are expanded to
    ///    individual letters in original case (e.g., `"FBI"` → `["F", "B", "I"]`).
    /// 3. **Contraction expansion**: English contractions are split and expanded
    ///    (`'s` → `is`, `'ve` → `have`, `'d` → `had`, `'m` → `am`).
    /// 4. **Filler normalization**: common fillers (`um`, `uhm`, `eh`, `mhm`,
    ///    etc.) are all normalized to `"um"`.
    /// 5. **Hyphen splitting**: hyphenated words are split at hyphens.
    /// 6. **Special word expansions**, colloquial forms are expanded
    ///    (`gimme` → `give me`, `wanna` → `want to`, `gonna` → `going to`, etc.).
    /// 7. **Name replacement**: known proper names are replaced with `"name"`.
    /// 8. **Specific acronym expansion**, selected acronyms are letter-expanded
    ///    (`mba`, `tli`, `bbc`, `ai`, `aa`, `ii`).
    /// 9. **Underscore splitting**: underscore-joined words are split.
    /// 10. **Passthrough**: unrecognized words pass through lowercased.
    pub fn conform_word_into(self, word: &str, result: &mut Vec<String>) {
        let trimmed = word.trim();
        let w = trimmed.to_lowercase();

        if let Some((a, b)) = COMPOUND_MAP.get(&w) {
            push_token(result, a.clone());
            push_token(result, b.clone());
        } else if ABBREV.contains(trimmed) {
            // Python checks abbreviations in original case
            for ch in trimmed.chars() {
                push_token(result, ch.to_string());
            }
        } else if w.contains("'s") {
            push_token(result, w.split('\'').next().unwrap_or("").to_string());
            push_token(result, "is".to_string());
        } else if w.contains("'ve") {
            push_token(result, w.split('\'').next().unwrap_or("").to_string());
            push_token(result, "have".to_string());
        } else if w.contains("'d") {
            push_token(result, w.split('\'').next().unwrap_or("").to_string());
            push_token(result, "had".to_string());
        } else if w.contains("'m") {
            push_token(result, w.split('\'').next().unwrap_or("").to_string());
            push_token(result, "am".to_string());
        } else if FILLERS.contains(w.as_str()) {
            push_token(result, "um".to_string());
        } else if w.contains('-') {
            for part in w.split('-') {
                push_token(result, part.trim().to_string());
            }
        } else if w == "ok" {
            push_token(result, "okay".to_string());
        } else if w == "gimme" {
            result.extend(["give", "me"].map(String::from));
        } else if w == "hafta" || w == "havta" {
            result.extend(["have", "to"].map(String::from));
        } else if self == Self::English && NAMES.contains(&w) {
            push_token(result, "name".to_string());
        } else if w == "dunno" {
            result.extend(["don't", "know"].map(String::from));
        } else if w == "wanna" {
            result.extend(["want", "to"].map(String::from));
        } else if w == "gonna" {
            result.extend(["going", "to"].map(String::from));
        } else if w == "gotta" {
            result.extend(["got", "to"].map(String::from));
        } else if w == "kinda" {
            result.extend(["kind", "of"].map(String::from));
        } else if w == "sorta" {
            result.extend(["sort", "of"].map(String::from));
        } else if w == "shoulda" {
            result.extend(["should", "have"].map(String::from));
        } else if w == "sposta" {
            result.extend(["supposed", "to"].map(String::from));
        } else if w == "hadta" {
            result.extend(["had", "to"].map(String::from));
        } else if w == "alright" || w == "alrightie" {
            result.extend(["all", "right"].map(String::from));
        } else if w == "i'd" {
            result.extend(["i", "had"].map(String::from));
        } else if w == "this'll" {
            result.extend(["this", "will"].map(String::from));
        } else if w == "farmhouse" {
            result.extend(["farm", "house"].map(String::from));
        } else if w == "mm" || w == "hmm" {
            push_token(result, "hm".to_string());
        } else if w == "em" {
            push_token(result, "them".to_string());
        } else if w == "eh" {
            push_token(result, "uh".to_string());
        } else if w == "til" {
            push_token(result, "until".to_string());
        } else if w == "ed" {
            push_token(result, "education".to_string());
        } else if matches!(w.as_str(), "mba" | "tli" | "bbc" | "ai" | "aa" | "ii") {
            for ch in w.chars() {
                push_token(result, ch.to_string());
            }
        } else if w.contains('_') {
            for part in w.split('_') {
                push_token(result, part.to_string());
            }
        } else {
            push_token(result, w);
        }
    }
}

/// Append `token` unless it is empty.
///
/// Every branch of `conform_word_into` appends through here. A split can
/// leave nothing on one side of a hyphen, an apostrophe or an underscore
/// (`-`, `'s`, `a_`), and a whitespace-only word lowercases to nothing.
/// BA2's `_conform()` passed those empties on as tokens; here a word conforms
/// to zero tokens rather than to an empty one, which the compare serializer
/// refuses (`EmptyXsrepToken`) and which no alignment could ever match.
fn push_token(result: &mut Vec<String>, token: String) {
    if !token.is_empty() {
        result.push(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn separator_only_words_conform_to_nothing() {
        for word in ["-", "--", "_", " ", ""] {
            assert!(
                WerNormalization::English
                    .conform_words(&s(&[word]))
                    .is_empty(),
                "{word:?} must conform to no token"
            );
        }
        // A bare contraction keeps its expansion and loses only the empty stem.
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["'s"])),
            s(&["is"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["a-", "-b"])),
            s(&["a", "b"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["x_"])),
            s(&["x"])
        );
    }

    #[test]
    fn test_data_loaded() {
        assert!(NAMES.len() > 5000);
        assert!(ABBREV.len() > 100);
        assert!(COMPOUND_MAP.len() > 3000);
    }

    #[test]
    fn test_compound_split() {
        let result = WerNormalization::English.conform_words(&s(&["airplane"]));
        assert_eq!(result, s(&["air", "plane"]));
    }

    #[test]
    fn test_contraction_expansion() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["he's"])),
            s(&["he", "is"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["I've"])),
            s(&["i", "have"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["she'd"])),
            s(&["she", "had"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["I'm"])),
            s(&["i", "am"])
        );
    }

    #[test]
    fn test_filler_normalization() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["uhm"])),
            s(&["um"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["mhm"])),
            s(&["um"])
        );
    }

    /// Names are kept whole under `KeepNames`, which is what stops two
    /// different Spanish words on the name list from matching.
    #[test]
    fn keep_names_does_not_collapse_names() {
        assert_eq!(
            WerNormalization::KeepNames.conform_words(&s(&["linda", "he's"])),
            s(&["linda", "he", "is"])
        );
    }

    /// The choice is made from what a transcript declares, and a pair shares
    /// English only when both would.
    #[test]
    fn normalization_follows_declared_languages() {
        let code = |c: &str| talkbank_model::model::LanguageCode::new(c).expect("code");
        assert_eq!(
            WerNormalization::for_declared_languages(&[]),
            WerNormalization::English
        );
        assert_eq!(
            WerNormalization::for_declared_languages(&[code("eng")]),
            WerNormalization::English
        );
        assert_eq!(
            WerNormalization::for_declared_languages(&[code("spa")]),
            WerNormalization::KeepNames
        );
        assert_eq!(
            WerNormalization::for_declared_languages(&[code("eng"), code("spa")]),
            WerNormalization::KeepNames
        );
        assert_eq!(
            WerNormalization::English.shared(WerNormalization::KeepNames),
            WerNormalization::KeepNames
        );
    }

    #[test]
    fn test_name_replacement() {
        // "aaron" is a common name that should be in the list
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["Aaron"])),
            s(&["name"])
        );
    }

    #[test]
    fn test_special_words() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["ok"])),
            s(&["okay"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["gimme"])),
            s(&["give", "me"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["wanna"])),
            s(&["want", "to"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["gonna"])),
            s(&["going", "to"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["dunno"])),
            s(&["don't", "know"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["alright"])),
            s(&["all", "right"])
        );
    }

    #[test]
    fn test_hyphen_split() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["well-known"])),
            s(&["well", "known"])
        );
    }

    #[test]
    fn test_underscore_split() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["ice_cream"])),
            s(&["ice", "cream"])
        );
    }

    #[test]
    fn test_abbreviation_expansion() {
        // Python iterates original-case chars, so "FBI" → "F", "B", "I"
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["FBI"])),
            s(&["F", "B", "I"])
        );
    }

    #[test]
    fn test_acronym_expansion() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["mba"])),
            s(&["m", "b", "a"])
        );
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["ai"])),
            s(&["a", "i"])
        );
    }

    #[test]
    fn test_passthrough() {
        assert_eq!(
            WerNormalization::English.conform_words(&s(&["hello", "world"])),
            s(&["hello", "world"])
        );
    }

    #[test]
    fn test_empty() {
        let result = WerNormalization::English.conform_words(&s(&[]));
        assert!(result.is_empty());
    }

    #[test]
    fn test_mixed() {
        let result = WerNormalization::English.conform_words(&s(&["Aaron", "he's", "gonna", "ok"]));
        assert_eq!(result, s(&["name", "he", "is", "going", "to", "okay"]));
    }

    // --- property tests ---

    use proptest::prelude::*;

    fn word_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            // Common words (passthrough)
            Just("hello".to_string()),
            Just("world".to_string()),
            Just("the".to_string()),
            Just("cat".to_string()),
            // Special words (expansion)
            Just("gonna".to_string()),
            Just("wanna".to_string()),
            Just("gimme".to_string()),
            Just("ok".to_string()),
            Just("dunno".to_string()),
            // Contractions
            Just("he's".to_string()),
            Just("I've".to_string()),
            Just("she'd".to_string()),
            // Fillers
            Just("um".to_string()),
            Just("uhm".to_string()),
            Just("mhm".to_string()),
            // Hyphenated
            Just("well-known".to_string()),
            // Random lowercase words
            "[a-z]{1,6}".prop_map(|s| s),
        ]
    }

    /// Words whose separators leave nothing on one side, or nothing at all:
    /// the shapes that used to conform to empty tokens.
    fn separator_word_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("-".to_string()),
            Just("--".to_string()),
            Just("a-".to_string()),
            Just("-a".to_string()),
            Just("'s".to_string()),
            Just("_".to_string()),
            Just("a_".to_string()),
            Just(" ".to_string()),
            "[a-z-_']{1,6}".prop_map(|s| s),
        ]
    }

    fn word_vec(max_len: usize) -> impl Strategy<Value = Vec<String>> {
        prop::collection::vec(word_strategy(), 0..=max_len)
    }

    proptest! {
        /// Output length is always >= input length (transforms expand, never reduce).
        #[test]
        fn output_never_shrinks(words in word_vec(10)) {
            let result = WerNormalization::English.conform_words(&words);
            prop_assert!(
                result.len() >= words.len(),
                "output {} < input {}", result.len(), words.len()
            );
        }

        /// No empty strings in output when input has no empty strings.
        #[test]
        fn no_empty_output_tokens(words in word_vec(10)) {
            let non_empty: Vec<String> = words.into_iter()
                .filter(|w| !w.trim().is_empty())
                .collect();
            let result = WerNormalization::English.conform_words(&non_empty);
            for (i, token) in result.iter().enumerate() {
                prop_assert!(
                    !token.is_empty(),
                    "Empty token at index {} from input {:?}", i, non_empty
                );
            }
        }

        /// Applying conform twice is a fixed point: conform(conform(x)) == conform(conform(conform(x))).
        /// The first application may change case (abbreviation expansion preserves original case),
        /// but the second application normalizes to lowercase, which is stable thereafter.
        #[test]
        fn double_application_is_fixed_point(words in word_vec(8)) {
            let once = WerNormalization::English.conform_words(&words);
            let twice = WerNormalization::English.conform_words(&once);
            let thrice = WerNormalization::English.conform_words(&twice);
            prop_assert_eq!(
                &twice, &thrice,
                "Not a fixed point at depth 2: {:?} -> {:?} -> {:?}",
                once, twice, thrice
            );
        }

        /// Empty input always produces empty output.
        #[test]
        fn empty_input_empty_output(_dummy in 0..1u8) {
            let result = WerNormalization::English.conform_words(&[]);
            prop_assert!(result.is_empty());
        }

        /// A separator-only word conforms to nothing, never to an empty
        /// token: the compare serializer refuses an empty token and no
        /// alignment could match one. Found 2026-09-22 when a Whisper
        /// Cantonese transcript reached compare with such a word.
        #[test]
        fn separators_never_yield_an_empty_token(words in prop::collection::vec(separator_word_strategy(), 0..=10)) {
            let result = WerNormalization::English.conform_words(&words);
            for (i, token) in result.iter().enumerate() {
                prop_assert!(!token.is_empty(), "Empty token at index {} from input {:?}", i, words);
            }
        }

        /// Each input word produces at least one output word.
        /// This verifies no words are silently dropped.
        #[test]
        fn every_word_produces_output(words in word_vec(10)) {
            let non_empty: Vec<String> = words.into_iter()
                .filter(|w| !w.trim().is_empty())
                .collect();
            let result = WerNormalization::English.conform_words(&non_empty);
            prop_assert!(
                result.len() >= non_empty.len(),
                "Some words were dropped: {} input, {} output",
                non_empty.len(), result.len()
            );
        }
    }
}
