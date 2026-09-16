//! Cantonese text normalization for ASR post-processing.
//!
//! Two steps, in this order:
//!
//! 1. Simplified to HK Traditional Chinese conversion via embedded OpenCC rules
//! 2. A domain-specific replacement table (31 entries) for Cantonese-specific
//!    character corrections
//!
//! # One owner, and why the type is the owner
//!
//! Until 2026-09-16 three places ran this normalization: the pyo3 provider
//! bridge (per token for Tencent and Aliyun, per joined segment for FunASR),
//! the server's stage 4b (per word, after number expansion), and
//! [`cantonese_char_tokens`] itself. The normalization is NOT idempotent, so
//! running it twice is not a harmless repetition: the table maps `繫` to `係`,
//! so a second pass turns an already-converted `聯繫` back into `聯係`. Running
//! it per character is the same defect from the other side, because a
//! multi-character replacement cannot match inside a single character.
//!
//! [`AlignedNormalization`] is now the ONLY route to normalized Cantonese text.
//! It takes a RUN of units (the words, or the provider's own timestamped
//! units), normalizes their concatenation once so multi-character replacements
//! still match across unit boundaries, and hands each unit back exactly as many
//! characters as it contributed. Handing them back is only sound when the
//! conversion preserved the character count, so the constructor proves that
//! first and refuses with both counts otherwise: a value of the type IS that
//! proof. This matters because the ASR pipeline pairs units with per-unit
//! timings, and a normalization that changed the count would shift every later
//! unit's timing while the text still looked plausible.

use std::sync::LazyLock;

use ferrous_opencc::{OpenCC, config::BuiltinConfig};

// ---------------------------------------------------------------------------
// Replacement table
// ---------------------------------------------------------------------------

/// Domain-specific Cantonese replacements applied AFTER zh-HK conversion.
///
/// Multi-character entries come first to prevent partial matches (e.g.,
/// "系" → "係" firing before "聯係" → "聯繫").
///
/// Ordered longest-first so that `replace_all` via sequential scan handles
/// overlapping patterns correctly.
///
/// Every entry maps N characters to N characters. That is not a coincidence to
/// be rediscovered later: `every_replacement_entry_preserves_character_count`
/// holds it, because an entry that changed the count would make
/// [`AlignedNormalization`] refuse every run containing it.
const REPLACEMENTS: &[(&str, &str)] = &[
    // Multi-character (longest first)
    ("聯繫", "聯繫"),
    ("聯係", "聯繫"),
    ("系啊", "係啊"),
    ("真系", "真係"),
    ("唔系", "唔係"),
    ("中意", "鍾意"),
    ("遊水", "游水"),
    ("羣組", "群組"),
    ("古仔", "故仔"),
    ("較剪", "鉸剪"),
    ("衝涼", "沖涼"),
    ("分鍾", "分鐘"),
    ("重復", "重複"),
    // Single-character
    ("系", "係"),
    ("繫", "係"),
    ("呀", "啊"),
    ("噶", "㗎"),
    ("咧", "呢"),
    ("嗬", "喎"),
    ("只", "隻"),
    ("咯", "囉"),
    ("嚇", "吓"),
    ("啫", "咋"),
    ("哇", "嘩"),
    ("着", "著"),
    ("嘞", "喇"),
    ("啵", "噃"),
    ("甕", "㧬"),
    ("牀", "床"),
    ("松", "鬆"),
    ("吵", "嘈"),
];

/// CJK punctuation and whitespace to strip during char tokenization.
///
/// Matches: fullwidth space, ideographic comma, ideographic period,
/// fullwidth comma, fullwidth exclamation, fullwidth question mark,
/// left/right corner brackets, fullwidth colon, fullwidth semicolon,
/// and ASCII whitespace.
fn is_cjk_punct_or_space(c: char) -> bool {
    c == '\u{3000}' // ideographic space
        || is_cjk_punctuation(c)
        || c.is_ascii_whitespace()
}

/// CJK punctuation that ASR providers emit and that is never a word:
/// ideographic comma and period, fullwidth comma, exclamation mark, question
/// mark, colon and semicolon, and the left and right corner brackets.
///
/// The one owner of this set; FunASR unit admission in batchalign-pyo3 uses
/// it too.
pub fn is_cjk_punctuation(c: char) -> bool {
    matches!(
        c,
        '\u{3001}' // ideographic comma
        | '\u{3002}' // ideographic period
        | '\u{FF0C}' // fullwidth comma
        | '\u{FF01}' // fullwidth exclamation
        | '\u{FF1F}' // fullwidth question mark
        | '\u{300C}' // left corner bracket
        | '\u{300D}' // right corner bracket
        | '\u{FF1A}' // fullwidth colon
        | '\u{FF1B}' // fullwidth semicolon
    )
}

// ---------------------------------------------------------------------------
// Aho-Corasick replacement engine
// ---------------------------------------------------------------------------

#[allow(clippy::expect_used)]
static HK_OPENCC: LazyLock<OpenCC> = LazyLock::new(|| {
    OpenCC::from_config(BuiltinConfig::S2hk)
        .expect("embedded S2hk conversion tables should be available")
});

/// Pre-built Aho-Corasick automaton for the replacement table.
///
/// Uses leftmost-longest matching to handle overlapping patterns correctly
/// (multi-char entries like "聯係" match before single-char "系").
// Data is compile-time-constant: patterns are static string literals defined above.
#[allow(clippy::expect_used)]
static REPLACER: LazyLock<aho_corasick::AhoCorasick> = LazyLock::new(|| {
    let patterns: Vec<&str> = REPLACEMENTS.iter().map(|(from, _)| *from).collect();
    aho_corasick::AhoCorasick::builder()
        .match_kind(aho_corasick::MatchKind::LeftmostLongest)
        .build(&patterns)
        .expect("cantonese replacement patterns are valid")
});

/// Apply the domain-specific replacement table using Aho-Corasick.
fn apply_replacements(text: &str) -> String {
    let replacements: Vec<&str> = REPLACEMENTS.iter().map(|(_, to)| *to).collect();
    REPLACER.replace_all(text, &replacements)
}

/// Normalize Cantonese text: simplified to HK traditional, then the domain
/// replacement table.
///
/// PRIVATE on purpose. A caller that could reach this directly could normalize
/// an already-normalized string (the table is not idempotent) or normalize one
/// character at a time (which loses every multi-character replacement), and
/// both were live defects before 2026-09-16. [`AlignedNormalization::admit`] is
/// the only way in.
fn normalize_cantonese(text: &str) -> String {
    let converted = HK_OPENCC.convert(text);
    apply_replacements(&converted)
}

// ---------------------------------------------------------------------------
// The one route to normalized Cantonese text
// ---------------------------------------------------------------------------

/// Cantonese normalization changed a run's character count, so its units
/// cannot be handed back their own characters.
///
/// Carries both counts and the number of units, because the interesting
/// question when this ever fires is which run it was and by how much it moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "Cantonese normalization changed a run of {units} unit(s) from {before} to {after} \
     characters, so each unit cannot keep its own characters"
)]
pub struct NormalizationChangedLength {
    /// How many units the run held.
    pub units: usize,
    /// Characters in the run before normalization.
    pub before: usize,
    /// Characters in the run after normalization.
    pub after: usize,
}

/// A run of ASR units normalized as ONE string, each unit holding exactly the
/// characters it contributed.
///
/// The field is private and [`AlignedNormalization::admit`] is the only
/// constructor, so holding a value is a proof that the normalization preserved
/// the run's character count. `units()` is therefore parallel to the units the
/// run was built from: same count, same order, same per-unit character widths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedNormalization {
    /// One normalized surface per source unit, in the source order.
    units: Vec<String>,
}

impl AlignedNormalization {
    /// Normalize a run of unit surfaces together.
    ///
    /// The units are concatenated in order, normalized once (so `真系` still
    /// becomes `真係` when `真` and `系` arrived as two units), and the result
    /// is split back at the source widths. A conversion that changed the
    /// character count is refused with both counts rather than silently
    /// re-cutting the units, which is what would move timings.
    pub fn admit<'a>(
        units: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, NormalizationChangedLength> {
        Self::admit_with(units, normalize_cantonese)
    }

    /// The whole of [`admit`], with the normalization injected.
    ///
    /// Only the tests pass anything but [`normalize_cantonese`]: the refusal is
    /// unreachable through the real tables (see
    /// `examples/s2hk_length_audit.rs`), and a refusal no test can reach is a
    /// refusal nobody has checked.
    ///
    /// [`admit`]: AlignedNormalization::admit
    fn admit_with<'a>(
        units: impl IntoIterator<Item = &'a str>,
        normalize: impl Fn(&str) -> String,
    ) -> Result<Self, NormalizationChangedLength> {
        let mut joined = String::new();
        let mut widths: Vec<usize> = Vec::new();
        for unit in units {
            widths.push(unit.chars().count());
            joined.push_str(unit);
        }

        let normalized = normalize(&joined);
        let before: usize = widths.iter().sum();
        let after = normalized.chars().count();
        if before != after {
            return Err(NormalizationChangedLength {
                units: widths.len(),
                before,
                after,
            });
        }

        // Total by construction: the widths sum to the normalized character
        // count, which was just proven.
        let mut characters = normalized.chars();
        let units = widths
            .into_iter()
            .map(|width| characters.by_ref().take(width).collect())
            .collect();
        Ok(Self { units })
    }

    /// The normalized surfaces, one per source unit, in the source order.
    pub fn units(&self) -> &[String] {
        &self.units
    }

    /// Take the normalized surfaces, one per source unit, in the source order.
    pub fn into_units(self) -> Vec<String> {
        self.units
    }
}

// ---------------------------------------------------------------------------
// Character tokenization
// ---------------------------------------------------------------------------

/// Split text into per-character tokens, dropping CJK punctuation and spaces.
///
/// This does NOT normalize. It used to, which made it a second owner of a
/// non-idempotent transformation and normalized each character on its own, so
/// `聯係` reached CHAT as `聯係` rather than `聯繫`. Normalization happens once
/// per run, before this stage, through [`AlignedNormalization`].
pub fn cantonese_char_tokens(text: &str) -> Vec<String> {
    text.chars()
        .filter(|c| !is_cjk_punct_or_space(*c))
        .map(|c| c.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Normalize a single string through the one route.
    fn normalized(text: &str) -> String {
        AlignedNormalization::admit([text])
            .expect("test input normalizes without changing its character count")
            .into_units()
            .concat()
    }

    #[test]
    fn test_single_char_replacement() {
        assert_eq!(normalized("系"), "係");
        assert_eq!(normalized("呀"), "啊");
        assert_eq!(normalized("松"), "鬆");
        assert_eq!(normalized("吵"), "嘈");
    }

    #[test]
    fn test_multi_char_replacement() {
        assert_eq!(normalized("真系"), "真係");
        assert_eq!(normalized("中意"), "鍾意");
        assert_eq!(normalized("較剪"), "鉸剪");
    }

    #[test]
    fn test_multi_char_priority_over_single() {
        // "聯係" should become "聯繫", not "聯" + "係"
        assert_eq!(normalized("聯係"), "聯繫");
        // "系啊" should match as a unit, not "系" + "啊"
        assert_eq!(normalized("系啊"), "係啊");
    }

    #[test]
    fn test_opencc_simplified_to_hk() {
        // Embedded OpenCC rules handle standard simplified to HK traditional.
        assert_eq!(normalized("联系"), "聯繫");
    }

    #[test]
    fn test_full_sentence() {
        assert_eq!(normalized("你真系好吵呀"), "你真係好嘈啊");
    }

    #[test]
    fn test_idempotent_on_hk_text() {
        assert_eq!(normalized("你好"), "你好");
    }

    /// The reason the type takes a RUN rather than a string: the providers hand
    /// us one unit per character, and a two-character replacement has to see
    /// both of them.
    #[test]
    fn a_run_is_normalized_together_and_each_unit_keeps_its_width() {
        let aligned = AlignedNormalization::admit(["聯", "係", "好"]).expect("length preserved");
        assert_eq!(aligned.units(), ["聯", "繫", "好"]);

        let aligned = AlignedNormalization::admit(["真", "系", "好", "吵", "呀"])
            .expect("length preserved");
        assert_eq!(aligned.units(), ["真", "係", "好", "嘈", "啊"]);
    }

    /// A unit wider than one character keeps exactly its own characters.
    #[test]
    fn a_multi_character_unit_gets_back_its_own_characters() {
        let aligned =
            AlignedNormalization::admit(["你", "真系", "好吵", "呀"]).expect("length preserved");
        assert_eq!(aligned.units(), ["你", "真係", "好嘈", "啊"]);
    }

    /// Normalizing each unit ALONE is what the run exists to prevent: it loses
    /// the phrase entry and, on already-converted text, walks it backwards.
    #[test]
    fn unit_by_unit_normalization_would_lose_the_phrase() {
        let per_unit: String = ["聯", "係"]
            .iter()
            .map(|unit| normalize_cantonese(unit))
            .collect();
        assert_eq!(per_unit, "聯係", "the phrase entry cannot match one character at a time");
        assert_eq!(
            AlignedNormalization::admit(["聯", "係"])
                .expect("length preserved")
                .into_units()
                .concat(),
            "聯繫"
        );
    }

    /// The constructor's proof, exercised. No real input reaches it (see
    /// `examples/s2hk_length_audit.rs`), so the seam is how the refusal gets
    /// tested at all.
    #[test]
    fn a_length_changing_normalization_is_refused_with_both_counts() {
        let refusal = AlignedNormalization::admit_with(["你", "好"], |text| format!("{text}了"))
            .expect_err("a normalization that adds a character must be refused");
        assert_eq!(
            refusal,
            NormalizationChangedLength {
                units: 2,
                before: 2,
                after: 3
            }
        );
        let message = refusal.to_string();
        assert!(message.contains("2 to 3 characters"), "{message}");
    }

    /// The empty run is a run: no units in, no units out, no refusal.
    #[test]
    fn an_empty_run_is_admitted() {
        let aligned = AlignedNormalization::admit(std::iter::empty()).expect("empty run");
        assert!(aligned.units().is_empty());
    }

    /// The table is data, and this is the gate on editing it: an entry whose
    /// sides differ in character count would make every run containing it
    /// refuse.
    #[test]
    fn every_replacement_entry_preserves_character_count() {
        for (from, to) in REPLACEMENTS {
            assert_eq!(
                from.chars().count(),
                to.chars().count(),
                "replacement {from} to {to} changes the character count"
            );
        }
    }

    /// Every Han character converts to exactly one character.
    ///
    /// This is the single-character half of the length question, settled
    /// exhaustively over the code points the pipeline calls ideographs rather
    /// than argued from the dictionary's shape. The phrase half is measured by
    /// `examples/s2hk_length_audit.rs` over OpenCC's own dictionaries.
    #[test]
    fn normalization_preserves_character_count_for_every_han_code_point() {
        let mut checked = 0usize;
        for code_point in 0x3400u32..=0x2FA1Fu32 {
            let Some(character) = char::from_u32(code_point) else {
                continue;
            };
            if !super::super::is_cjk_ideograph(character) {
                continue;
            }
            checked += 1;
            let source = character.to_string();
            let normalized = normalize_cantonese(&source);
            assert_eq!(
                normalized.chars().count(),
                1,
                "{source} (U+{code_point:04X}) normalized to {normalized}"
            );
        }
        // 81,520 code points at the time of writing: the eight ranges
        // `is_cjk_ideograph` names, with the gaps between them left out.
        assert!(
            checked > 80_000,
            "the sweep must actually cover the ideograph ranges, checked {checked}"
        );
    }

    #[test]
    fn test_char_tokens_basic() {
        let tokens = cantonese_char_tokens("真係呀，");
        assert_eq!(tokens, vec!["真", "係", "呀"]);
    }

    /// Tokenization is now split-only. Whatever normalization is owed happened
    /// before this stage, over the whole run.
    #[test]
    fn char_tokens_do_not_normalize() {
        assert_eq!(cantonese_char_tokens("真系"), vec!["真", "系"]);
    }

    #[test]
    fn test_char_tokens_strips_all_cjk_punct() {
        let tokens = cantonese_char_tokens("「你好」！");
        assert_eq!(tokens, vec!["你", "好"]);
    }

    #[test]
    fn test_char_tokens_empty() {
        let tokens = cantonese_char_tokens("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_char_tokens_only_punct() {
        let tokens = cantonese_char_tokens("，。！？");
        assert!(tokens.is_empty());
    }
}
