//! Normalized word text for use as frequency-counting map keys.
//!
//! The transformation `word.cleaned_text().to_lowercase()` is the canonical
//! form used as map keys across all analysis commands. [`NormalizedWord`]
//! encapsulates this transformation so it is always applied consistently
//! across FREQ, MAXWD, DIST, COOCCUR, KWAL, COMBO, and all other commands
//! that need word-level deduplication or matching.
//!
//! [`clan_display_form()`] provides an alternative form that preserves `+` in
//! compound words (`ice+cream`) for CLAN-compatible output rendering.
//!
//! # Forward-compatibility
//!
//! As the grammar moves toward a looser `grammar.js` that pushes more checks
//! from parsing into validation, words that previously would not reach the AST
//! may start arriving as less-classified `Word` nodes. Centralizing the
//! normalization here means only [`NormalizedWord::from_word()`] needs updating
//! when that happens -- no command code changes are required.

use std::borrow::Borrow;
use std::fmt;

use serde::Serialize;
use talkbank_model::Word;

/// Lowercased, cleaned word text suitable for frequency counting.
///
/// Encapsulates `word.cleaned_text().to_lowercase()` — the canonical form
/// used as map keys across all analysis commands:
/// - `freq.rs` word counts
/// - `maxwd.rs` unique-word deduplication
/// - `dist.rs` per-word distribution tracking
/// - `cooccur.rs` word-pair keys
/// - `kwal.rs` / `combo.rs` keyword matching
///
/// # Using as a map key
///
/// ```
/// use std::collections::HashMap;
/// use talkbank_clan::framework::NormalizedWord;
/// let mut map: HashMap<NormalizedWord, u64> = HashMap::new();
/// // `map.get("hello")` works because NormalizedWord: Borrow<str>
/// ```
///
/// # Invariant
///
/// The inner `String` is always `Word::cleaned_text()` (CHAT markup stripped),
/// optionally lower-cased. Case-folding is controlled by the caller via
/// [`Self::from_word`] (always lowercased — CLAN default) versus
/// [`Self::from_word_cased`] (lowercased iff `case_sensitive` is `false`,
/// preserved otherwise — controls CLAN `+k`).
///
/// Never construct with `NormalizedWord(raw_string)` directly outside of this
/// module; always go through one of the constructors so the invariant stays
/// in one place.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct NormalizedWord(pub(crate) String);

impl NormalizedWord {
    /// Construct the canonical lowercased form of a word — CLAN's default.
    ///
    /// Applies `word.cleaned_text()` (strips CHAT markup / trailing punctuation)
    /// then `to_lowercase()`. Equivalent to `from_word_cased(word, false)`.
    ///
    /// # Precondition
    ///
    /// `word` must pass [`crate::framework::word_filter::is_countable_word`].
    /// Results are unspecified (but safe) for non-countable words.
    pub fn from_word(word: &Word) -> Self {
        Self::from_word_cased(word, false)
    }

    /// Construct the word key honouring CLAN's `+k` flag. When
    /// `case_sensitive` is `false` (CLAN default), the result is lower-cased
    /// `cleaned_text`; when `true`, original case is preserved.
    ///
    /// Used by every analysis command's `process_utterance` so the `+k`
    /// branch lives in one place rather than being inlined per command.
    ///
    /// # Precondition
    ///
    /// `word` must pass [`crate::framework::word_filter::is_countable_word`].
    pub fn from_word_cased(word: &Word, case_sensitive: bool) -> Self {
        Self::from_text_cased(&word.cleaned_text(), case_sensitive)
    }

    /// Construct the word key from an already-cleaned `&str` (e.g. `%mor`
    /// serializations), honouring `+k`. Sibling of [`Self::from_word_cased`]
    /// for callers that don't have a `&Word`.
    pub fn from_text_cased(text: &str, case_sensitive: bool) -> Self {
        let key = if case_sensitive {
            text.to_owned()
        } else {
            text.to_lowercase()
        };
        NormalizedWord(key)
    }

    /// Return the normalized text as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Return the CLAN display form of a word (lowercased, overlap markers stripped).
///
/// CLAN preserves `+` in compound words (`ice+cream`, `choo+choo's`) and
/// uses lowercased raw text. Our `NormalizedWord` uses `cleaned_text()`
/// which strips `+`, so multiple commands need this alternative form for
/// CLAN-compatible output.
///
/// Note: CLAN `freq` is an exception — it preserves original case. Use
/// [`clan_display_form_preserve_case()`] for freq-style output.
pub fn clan_display_form(word: &Word) -> String {
    strip_overlap_markers(&word.raw_text().to_lowercase())
}

/// Return the CLAN display form of a word preserving original case.
///
/// Used by FREQ which displays words in their original casing.
pub fn clan_display_form_preserve_case(word: &Word) -> String {
    strip_overlap_markers(word.raw_text())
}

/// Strip CA overlap markers (⌈⌉⌊⌋ and indexed variants) from text.
fn strip_overlap_markers(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '⌈' | '⌉' | '⌊' | '⌋'))
        .collect()
}

impl fmt::Display for NormalizedWord {
    /// Print the normalized token text without additional formatting.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for NormalizedWord {
    /// Expose the normalized token as `&str` for generic APIs.
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Enables `map.get("hello")` on `HashMap<NormalizedWord, _>` — no temporary
/// `NormalizedWord` allocation required at lookup sites.
impl Borrow<str> for NormalizedWord {
    /// Enable zero-allocation `&str` lookup against `NormalizedWord` map keys.
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::Word;

    /// Construction from `Word` should lowercase and clean CHAT markup.
    #[test]
    fn normalized_word_lowercases() {
        let word = Word::simple("HELLO");
        let nw = NormalizedWord::from_word(&word);
        assert_eq!(nw.as_str(), "hello");
    }

    /// `Ord` should provide deterministic lexical ordering for map iteration.
    #[test]
    fn normalized_word_ord_for_map_ordering() {
        let a = NormalizedWord(String::from("apple"));
        let b = NormalizedWord(String::from("banana"));
        assert!(a < b);
    }

    /// `Borrow<str>` should allow `HashMap::get` with plain `&str` keys.
    #[test]
    fn borrow_enables_str_lookup() {
        use std::collections::HashMap;
        let mut map: HashMap<NormalizedWord, u64> = HashMap::new();
        map.insert(NormalizedWord(String::from("hello")), 42);
        // Lookup with &str — works via Borrow<str>
        assert_eq!(map.get("hello"), Some(&42));
    }
}
