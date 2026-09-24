//! Number-to-text expansion for ASR post-processing.
//!
//! Converts digit-bearing ASR tokens to spoken word forms so the transcript
//! satisfies E220 (numbers must be written out as pronounced).
//!
//! # Ownership
//!
//! The general generator belongs to chatter:
//! [`talkbank_transform::num_words::expand_number`] owns the per-language
//! cardinal tables, English short-scale composition, Chinese/Japanese/Cantonese
//! numerals, English suffix ordinals and decades, currency symbols, dash ranges
//! and digit-leading hyphen compounds. Batchalign calls it rather than keeping
//! a copy, so a correction there (for example chatter 0.26.0's English scale
//! composition, `2024` as "two thousand twenty-four") reaches ASR output by
//! bumping the pin.
//!
//! Batchalign keeps only what chatter does not provide:
//!
//! 1. Portuguese indicator ordinals (`54ª`, `1.º`, `54.ºs`), rendered with the
//!    gender and number the indicator marks, by `ordinal_por`. These are
//!    settled before the generic generator, which would otherwise see a
//!    non-digit token and leave it alone.
//! 2. The per-language percent word used when an ASR provider emits a
//!    `%`-suffixed token ([`language_percent_word`]).

use super::ordinal_por::OrdinalTokenExpansion;

/// Per-language word for "percent", used when an ASR provider emits a bare
/// `%`-suffixed numeric token (e.g. Rev.AI returning `"80%"`). The Rust
/// post-processor strips `%`, expands the digit part, and appends the
/// language-specific percent word so the output reaches the CHAT tier as
/// legal main-tier word content (`%` is the CHAT dep-tier sigil and cannot
/// appear on the main tier in any language).
///
/// Tracked by ISO 639-3 code. Languages not listed here fall back to the
/// English word; a future extension can delete that fallback once the
/// remaining coverage gaps are audited. (Decision N1 from the
/// 2026-04-22 ASR-normalization design, operator-local.)
const PERCENT_WORD_BY_LANG: &[(&str, &str)] = &[
    ("eng", "percent"),
    ("fra", "pour_cent"),
    ("spa", "por_ciento"),
    ("deu", "Prozent"),
    ("ita", "per_cento"),
    ("por", "por_cento"),
    ("nld", "procent"),
    ("jpn", "パーセント"),
    ("zho", "百分"),
    ("cmn", "百分"),
    ("yue", "百分"),
];

/// Language-specific CHAT word for the percent symbol.
///
/// Returns the per-language word to substitute when `%` is stripped from an
/// ASR token, or `None` if no mapping is known. Callers fall back to a
/// reasonable default (typically eng) rather than panicking on unmapped
/// languages: the goal is that the CHAT output is never worse than it
/// would have been without the normalizer.
pub fn language_percent_word(lang: &str) -> Option<&'static str> {
    let lower = lang.to_lowercase();
    PERCENT_WORD_BY_LANG
        .iter()
        .find(|(l, _)| *l == lower.as_str())
        .map(|(_, w)| *w)
}

/// Expand a digit-bearing ASR token to its spoken form in `lang`.
///
/// Portuguese indicator ordinals are rendered here; every other form goes to
/// chatter's generator. A token neither can expand is returned exactly as
/// written, so CHAT validation reports its digits (E220) instead of a guessed
/// word.
///
/// # Arguments
/// * `word` - The word to potentially expand.
/// * `lang` - ISO 639-3 language code.
pub fn expand_number(word: &str, lang: &str) -> String {
    // Portuguese indicator ordinals carry grammatical gender and number in
    // the token itself, which no cardinal table can express, so they are
    // settled first. A recognized ordinal with no rendering stays exactly as
    // written, the same policy the generic generator applies.
    match super::ordinal_por::expand_ordinal_token(word, lang) {
        OrdinalTokenExpansion::Expanded(words) => words,
        OrdinalTokenExpansion::OutOfRange(_) => word.to_owned(),
        OrdinalTokenExpansion::NotOrdinal => {
            talkbank_transform::num_words::expand_number(word, lang)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Seam tests: they prove Batchalign reaches chatter's generator and keeps
    //! its own Portuguese ordinals in front of it. The generator's behaviour
    //! per language is chatter's to test; the frozen cross-language baseline
    //! (`num2text_baseline`) records what Batchalign actually emits.

    use super::*;

    /// English scales compose from chatter's short-scale units. Before
    /// chatter 0.26.0 (and in Batchalign's former copy of the generator)
    /// these multiplied whole table phrases: "two one thousand twenty-four".
    #[test]
    fn english_scales_come_from_chatter_generator() {
        assert_eq!(expand_number("2024", "eng"), "two thousand twenty-four");
        assert_eq!(expand_number("1000000", "eng"), "one million");
        assert_eq!(
            expand_number("123456789", "eng"),
            "one hundred twenty-three million four hundred fifty-six thousand \
             seven hundred eighty-nine"
        );
    }

    /// Chinese numerals skip an empty four-digit group with a single zero.
    /// Batchalign's former copy wrote the zero twice (`一亿零零一`).
    #[test]
    fn chinese_skipped_group_writes_one_zero() {
        assert_eq!(expand_number("100000001", "zho"), "一亿零一");
        assert_eq!(expand_number("100000001", "yue"), "一億零一");
    }

    /// Chatter's other forms (currency, ranges, suffix ordinals, decades) are
    /// reached through the same call.
    #[test]
    fn other_generic_forms_reach_chatter() {
        assert_eq!(expand_number("$12", "spa"), "doce dollars");
        assert_eq!(expand_number("21-22", "eng"), "twenty-one twenty-two");
        assert_eq!(expand_number("3rd", "eng"), "third");
        assert_eq!(expand_number("1950s", "eng"), "nineteen fifties");
        assert_eq!(expand_number("17-year-old", "eng"), "seventeen-year-old");
    }

    /// Portuguese indicator ordinals are Batchalign's own and run first.
    #[test]
    fn portuguese_indicator_ordinals_precede_the_generic_generator() {
        assert_eq!(expand_number("54ª", "por"), "quinquagésima quarta");
        // Recognized but out of the rendered range: kept exactly as written.
        assert_eq!(expand_number("1001.º", "por"), "1001.º");
        // Plain cardinals in Portuguese still reach chatter.
        assert_eq!(expand_number("5", "por"), "cinco");
    }

    /// A token nothing can expand is returned unchanged, never guessed.
    #[test]
    fn unexpandable_tokens_pass_through() {
        assert_eq!(expand_number("42", "xxx"), "42");
        assert_eq!(expand_number("hello", "eng"), "hello");
        assert_eq!(expand_number("", "eng"), "");
    }

    #[test]
    fn percent_word_is_per_language() {
        assert_eq!(language_percent_word("eng"), Some("percent"));
        assert_eq!(language_percent_word("SPA"), Some("por_ciento"));
        assert_eq!(language_percent_word("xxx"), None);
    }
}
