//! Han-script normalization of ASR text: one owner for every language that
//! has one.
//!
//! Two normalizations exist, chosen once from the language code by
//! [`HanScriptNormalization::for_language`]:
//!
//! - Cantonese (`yue`): simplified to Hong Kong traditional via embedded
//!   OpenCC rules, then a domain-specific replacement table for
//!   Cantonese-specific characters.
//! - Taiwanese (`nan`): simplified to Taiwan-standard traditional. TalkBank's
//!   Taiwanese transcripts are written in traditional script, while a
//!   Taiwanese recognizer labelled as Chinese emits simplified characters.
//!
//! # One owner, and why the type is the owner
//!
//! Until 2026-09-16 three places ran the Cantonese normalization: the pyo3
//! provider bridge (per token for Tencent and Aliyun, per joined segment for
//! FunASR), the server's stage 4b (per word, after number expansion), and the
//! character tokenizer itself. The normalization is NOT idempotent, so running
//! it twice is not a harmless repetition: the table maps `繫` to `係`, so a
//! second pass turns an already-converted `聯繫` back into `聯係`. Running it
//! per character is the same defect from the other side, because a
//! multi-character replacement cannot match inside a single character.
//!
//! [`HanScriptNormalization::admit`] is the ONLY route to normalized text. It
//! takes a RUN of units (the words, or the provider's own timestamped units),
//! normalizes their concatenation once so multi-character entries still match
//! across unit boundaries, and hands each unit back exactly as many characters
//! as it contributed, as an [`AlignedNormalization`]. Handing them back is only
//! sound when the conversion preserved the character count, so the constructor
//! proves that first and refuses with both counts otherwise: a value of the
//! type IS that proof. This matters because the ASR pipeline pairs units with
//! per-unit timings, and a normalization that changed the count would shift
//! every later unit's timing while the text still looked plausible.

use std::sync::LazyLock;

use ferrous_opencc::{OpenCC, config::BuiltinConfig};

// ---------------------------------------------------------------------------
// Replacement table
// ---------------------------------------------------------------------------

/// Domain-specific Cantonese replacements applied AFTER zh-HK conversion.
///
/// Matching is leftmost-longest (see [`REPLACER`]), so a multi-character entry
/// wins over a single-character one inside it: "聯係" becomes "聯繫" rather than
/// firing "系" to "係". The identity entry `聯繫` exists for the same reason:
/// it keeps the single-character `繫` entry from firing inside `聯繫`.
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

// ---------------------------------------------------------------------------
// Aho-Corasick replacement engine
// ---------------------------------------------------------------------------

#[allow(clippy::expect_used)]
static HK_OPENCC: LazyLock<OpenCC> = LazyLock::new(|| {
    OpenCC::from_config(BuiltinConfig::S2hk)
        .expect("embedded S2hk conversion tables should be available")
});

/// Simplified to Taiwan-standard traditional. Character conversion only
/// (`STPhrases`, `STCharacters`, `TWVariants`): not `s2twp`, which also
/// substitutes Taiwan Mandarin vocabulary and would put words in a Taiwanese
/// transcript that the speaker did not say.
#[allow(clippy::expect_used)]
static TW_OPENCC: LazyLock<OpenCC> = LazyLock::new(|| {
    OpenCC::from_config(BuiltinConfig::S2tw)
        .expect("embedded S2tw conversion tables should be available")
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
///
/// Each match indexes straight into [`REPLACEMENTS`], so no list of
/// replacement strings is rebuilt per call.
fn apply_replacements(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    REPLACER.replace_all_with(text, &mut out, |found, _, out| {
        out.push_str(REPLACEMENTS[found.pattern().as_usize()].1);
        true
    });
    out
}

/// Normalize Cantonese text: simplified to HK traditional, then the domain
/// replacement table.
///
/// PRIVATE on purpose. A caller that could reach this directly could normalize
/// an already-normalized string (the table is not idempotent) or normalize one
/// character at a time (which loses every multi-character replacement), and
/// both were live defects before 2026-09-16. [`HanScriptNormalization::admit`] is
/// the only way in.
fn normalize_cantonese(text: &str) -> String {
    let converted = HK_OPENCC.convert(text);
    apply_replacements(&converted)
}

// ---------------------------------------------------------------------------
// The closed set of normalizations
// ---------------------------------------------------------------------------

/// The Han-script normalization a language's ASR text receives.
///
/// A closed set, chosen once from the language code by [`Self::for_language`].
/// Every caller that used to compare the code against `"yue"` asks this type
/// instead, so a language either has a variant here or receives no
/// normalization anywhere: there is no second list to keep in step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HanScriptNormalization {
    /// `yue`: simplified to Hong Kong traditional, then the Cantonese domain
    /// replacement table.
    HongKongCantonese,
    /// `nan`: simplified to Taiwan-standard traditional, and nothing else. No
    /// domain table: written Taiwanese in TalkBank is traditional script, and
    /// a Taiwanese recognizer labelled as Chinese emits simplified characters,
    /// so the conversion restores the corpus script only.
    TaiwanTraditional,
}

impl HanScriptNormalization {
    /// Every normalization, for exhaustive audits and tests.
    pub const ALL: [Self; 2] = [Self::HongKongCantonese, Self::TaiwanTraditional];

    /// The normalization a language's ASR text receives, if any.
    pub fn for_language(language: &str) -> Option<Self> {
        match language {
            "yue" => Some(Self::HongKongCantonese),
            "nan" => Some(Self::TaiwanTraditional),
            _ => None,
        }
    }

    /// Normalize a run of unit surfaces together; see [`AlignedNormalization`].
    ///
    /// The ONLY public route to normalized text.
    pub fn admit<'a>(
        self,
        units: impl IntoIterator<Item = &'a str>,
    ) -> Result<AlignedNormalization, NormalizationChangedLength> {
        AlignedNormalization::admit_with(units, |text| self.convert(text))
            .map_err(|change| change.under(self))
    }

    /// The conversion itself, private for the reasons given on
    /// [`normalize_cantonese`].
    fn convert(self, text: &str) -> String {
        match self {
            Self::HongKongCantonese => normalize_cantonese(text),
            Self::TaiwanTraditional => TW_OPENCC.convert(text),
        }
    }
}

impl std::fmt::Display for HanScriptNormalization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::HongKongCantonese => "Cantonese (Hong Kong traditional)",
            Self::TaiwanTraditional => "Taiwanese (Taiwan traditional)",
        })
    }
}

// ---------------------------------------------------------------------------
// The one route to normalized text
// ---------------------------------------------------------------------------

/// A Han-script normalization changed a run's character count, so its units
/// cannot be handed back their own characters.
///
/// Carries both counts and the number of units, because the interesting
/// question when this ever fires is which run it was and by how much it moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "{normalization} normalization changed a run of {units} unit(s) from {before} to {after} \
     characters, so each unit cannot keep its own characters"
)]
pub struct NormalizationChangedLength {
    /// Which normalization ran.
    pub normalization: HanScriptNormalization,
    /// How many units the run held.
    pub units: usize,
    /// Characters in the run before normalization.
    pub before: usize,
    /// Characters in the run after normalization.
    pub after: usize,
}

/// The counts a refused conversion measured, before the caller names which
/// normalization it was. Kept apart so the test seam, which injects a
/// conversion, cannot pair one table's name with another table's behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CountChanged {
    units: usize,
    before: usize,
    after: usize,
}

impl CountChanged {
    /// Name the normalization that produced these counts.
    fn under(self, normalization: HanScriptNormalization) -> NormalizationChangedLength {
        NormalizationChangedLength {
            normalization,
            units: self.units,
            before: self.before,
            after: self.after,
        }
    }
}

/// A run of ASR units normalized as ONE string, each unit holding exactly the
/// characters it contributed.
///
/// The field is private and [`HanScriptNormalization::admit`] is the only
/// constructor, so holding a value is a proof that the normalization preserved
/// the run's character count. `units()` is therefore parallel to the units the
/// run was built from: same count, same order, same per-unit character widths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedNormalization {
    /// One normalized surface per source unit, in the source order.
    units: Vec<String>,
}

impl AlignedNormalization {
    /// The units are concatenated in order, normalized once (so `真系` still
    /// becomes `真係` when `真` and `系` arrived as two units), and the result
    /// is split back at the source widths. A conversion that changed the
    /// character count is refused with both counts rather than silently
    /// re-cutting the units, which is what would move timings.
    ///
    /// Reached through [`HanScriptNormalization::admit`], which supplies the
    /// real conversion. Only the tests inject anything else: the refusal is
    /// unreachable through the real tables (see
    /// `examples/han_script_length_audit.rs`), and a refusal no test can reach
    /// is a refusal nobody has checked.
    fn admit_with<'a>(
        units: impl IntoIterator<Item = &'a str>,
        normalize: impl Fn(&str) -> String,
    ) -> Result<Self, CountChanged> {
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
            return Err(CountChanged {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(text: &str) -> String {
        HanScriptNormalization::HongKongCantonese
            .admit([text])
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
        let aligned = HanScriptNormalization::HongKongCantonese
            .admit(["聯", "係", "好"])
            .expect("length preserved");
        assert_eq!(aligned.units(), ["聯", "繫", "好"]);

        let aligned = HanScriptNormalization::HongKongCantonese
            .admit(["真", "系", "好", "吵", "呀"])
            .expect("length preserved");
        assert_eq!(aligned.units(), ["真", "係", "好", "嘈", "啊"]);
    }

    /// A unit wider than one character keeps exactly its own characters.
    #[test]
    fn a_multi_character_unit_gets_back_its_own_characters() {
        let aligned = HanScriptNormalization::HongKongCantonese
            .admit(["你", "真系", "好吵", "呀"])
            .expect("length preserved");
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
        assert_eq!(
            per_unit, "聯係",
            "the phrase entry cannot match one character at a time"
        );
        assert_eq!(
            HanScriptNormalization::HongKongCantonese
                .admit(["聯", "係"])
                .expect("length preserved")
                .into_units()
                .concat(),
            "聯繫"
        );
    }

    /// The constructor's proof, exercised. No real input reaches it (see
    /// `examples/han_script_length_audit.rs`), so the seam is how the refusal gets
    /// tested at all.
    #[test]
    fn a_length_changing_normalization_is_refused_with_both_counts() {
        let refusal = AlignedNormalization::admit_with(["你", "好"], |text| format!("{text}了"))
            .expect_err("a normalization that adds a character must be refused")
            .under(HanScriptNormalization::TaiwanTraditional);
        assert_eq!(
            refusal,
            NormalizationChangedLength {
                normalization: HanScriptNormalization::TaiwanTraditional,
                units: 2,
                before: 2,
                after: 3
            }
        );
        let message = refusal.to_string();
        assert!(message.contains("2 to 3 characters"), "{message}");
        assert!(message.starts_with("Taiwanese"), "{message}");
    }

    /// The empty run is a run: no units in, no units out, no refusal.
    #[test]
    fn an_empty_run_is_admitted() {
        let aligned = HanScriptNormalization::HongKongCantonese
            .admit(std::iter::empty())
            .expect("empty run");
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

    /// Every Han character converts to exactly one character, under every
    /// normalization.
    ///
    /// This is the single-character half of the length question, settled
    /// exhaustively over the code points the pipeline calls ideographs rather
    /// than argued from the dictionary's shape. The phrase half is measured by
    /// `examples/han_script_length_audit.rs` over OpenCC's own dictionaries.
    #[test]
    fn normalization_preserves_character_count_for_every_han_code_point() {
        let ideographs: Vec<char> = (0x3400u32..=0x2FA1Fu32)
            .filter_map(char::from_u32)
            .filter(|&c| super::super::is_cjk_ideograph(c))
            .collect();
        // One conversion per normalization over every ideograph, each on its
        // own line: no dictionary entry spans a newline, so every character is
        // converted in isolation, exactly as it would be one call at a time.
        let lines: String = ideographs.iter().flat_map(|&c| [c, '\n']).collect();
        for normalization in HanScriptNormalization::ALL {
            let converted = normalization.convert(&lines);
            let pieces: Vec<&str> = converted.split_terminator('\n').collect();
            assert_eq!(
                pieces.len(),
                ideographs.len(),
                "{normalization}: line count"
            );
            for (source, normalized) in ideographs.iter().zip(&pieces) {
                assert_eq!(
                    normalized.chars().count(),
                    1,
                    "{normalization}: {source} (U+{:04X}) normalized to {normalized}",
                    u32::from(*source)
                );
            }
        }
        // 81,520 code points at the time of writing: the eight ranges
        // `is_cjk_ideograph` names, with the gaps between them left out.
        assert!(
            ideographs.len() > 80_000,
            "the sweep must actually cover the ideograph ranges, checked {}",
            ideographs.len()
        );
    }

    /// Taiwanese recognition labelled as Chinese arrives simplified; the
    /// corpus script is traditional, character for character.
    #[test]
    fn taiwanese_converts_to_taiwan_traditional_without_changing_words() {
        let aligned = HanScriptNormalization::TaiwanTraditional
            .admit(["这", "个", "故", "事", "来", "唱", "这", "首", "歌"])
            .expect("length preserved");
        assert_eq!(aligned.into_units().concat(), "這個故事來唱這首歌",);
    }

    /// The Cantonese domain table is Cantonese only: Taiwanese `系` and `呀`
    /// keep their own characters.
    #[test]
    fn taiwanese_does_not_receive_the_cantonese_table() {
        let units = HanScriptNormalization::TaiwanTraditional
            .admit(["系", "呀"])
            .expect("length preserved")
            .into_units();
        assert_eq!(units.concat(), "系呀");
    }

    /// Only the two languages with a normalization have one.
    #[test]
    fn languages_map_to_their_normalization() {
        assert_eq!(
            HanScriptNormalization::for_language("yue"),
            Some(HanScriptNormalization::HongKongCantonese)
        );
        assert_eq!(
            HanScriptNormalization::for_language("nan"),
            Some(HanScriptNormalization::TaiwanTraditional)
        );
        for other in ["zho", "cmn", "hak", "eng"] {
            assert_eq!(HanScriptNormalization::for_language(other), None, "{other}");
        }
    }
}
