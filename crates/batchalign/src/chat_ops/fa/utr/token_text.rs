//! The lexical units UTR matches in, chosen once per transcript language and
//! applied to BOTH sides: the recognizer's tokens and the transcript's words.
//!
//! Cantonese is the reason this exists. A recognizer returns Cantonese as
//! phrases (`佢哋食嘢`), or as words (`佢哋`), while a transcript may write one
//! character per word (`佢 哋 食 嘢`) or multi-character words (`佢哋 食 嘢`):
//! the census of the corpora finds both. Comparing words to phrases, or
//! characters on one side to words on the other, finds almost nothing. So for
//! Cantonese both sides are projected into the same character space, where
//! any segmentation of the same speech meets any other.
//!
//! The projection never manufactures timing. A provider token stays ONE timed
//! observation whatever number of units it holds, and a transcript word keeps
//! its own identity across its units: a word is anchored at a time only when
//! one provider token's units are exactly that word's units (see
//! `anchors::WordAnchor`).

use super::AsrTimingToken;
use super::evidence::{
    UtrUnitOfWord, UtrUnitOrdinal, UtrUtteranceOrdinal, UtrWordAddress, UtrWordOrdinal,
};
use crate::api::LanguageCode3;
use batchalign_transform::asr_postprocess::cantonese::is_cjk_punctuation;
use batchalign_transform::asr_postprocess::{
    HanScriptNormalization, NormalizationChangedLength, is_cjk_ideograph,
};

/// How text is cut into the units UTR matches, on both sides.
///
/// Chosen from the transcript language's Han-script handling
/// ([`Self::for_script`]), the one typed decision about that language, never
/// from comparing a code with a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UtrLexicalUnits {
    /// Words delimited by whitespace; a transcript word is one unit.
    SpaceDelimitedWords,
    /// Every Han character is its own unit; a run of other characters
    /// (Latin, digits) between whitespace or CJK punctuation is one unit, and
    /// CJK punctuation is no unit at all.
    HanCharacters,
}

impl UtrLexicalUnits {
    /// The units for a language with this Han-script handling.
    ///
    /// Cantonese is matched in characters. Taiwanese keeps the word matching
    /// it has always had; whether it would gain from characters has not been
    /// measured. A language with no Han-script handling is space-delimited.
    pub fn for_script(script: Option<HanScriptNormalization>) -> Self {
        match script {
            Some(HanScriptNormalization::HongKongCantonese) => Self::HanCharacters,
            Some(HanScriptNormalization::TaiwanTraditional) | None => Self::SpaceDelimitedWords,
        }
    }

    /// The units for a transcript whose primary language is `language`.
    pub fn for_language(language: &LanguageCode3) -> Self {
        Self::for_script(HanScriptNormalization::for_language(language.as_ref()))
    }

    /// The units of `text`, as slices of it, in order. A piece with no
    /// letter or digit (punctuation only: `?`, `...`, an ellipsis) is no
    /// unit, so a token's unit count is the count of what can be matched.
    pub(super) fn split(self, text: &str) -> Vec<&str> {
        match self {
            Self::SpaceDelimitedWords => text
                .split_whitespace()
                .filter(|unit| is_lexical(unit))
                .collect(),
            Self::HanCharacters => han_character_units(text),
        }
    }
}

impl UtrLexicalUnits {
    /// The transcript side: each alignable word's units, in order, each
    /// knowing its word. The only constructor of [`UtrChatUnits`] and of
    /// [`UtrUnitOfWord`]. A word is one unit under
    /// [`Self::SpaceDelimitedWords`]; a word with no unit under
    /// [`Self::HanCharacters`] (punctuation only) contributes none.
    pub(super) fn project_words(self, words: Vec<String>) -> UtrChatUnits {
        let mut units = UtrChatUnits {
            words,
            texts: Vec::new(),
            of_word: Vec::new(),
        };
        for (word_index, word) in units.words.iter().enumerate() {
            let parts: Vec<&str> = match self {
                Self::SpaceDelimitedWords => vec![word.as_str()],
                Self::HanCharacters => han_character_units(word),
            };
            // A word with no unit contributes none.
            let Some(count) = std::num::NonZeroUsize::new(parts.len()) else {
                continue;
            };
            for (text, of_word) in parts
                .into_iter()
                .zip(UtrUnitOfWord::units_of(UtrWordOrdinal(word_index), count))
            {
                units.texts.push(text.to_owned());
                units.of_word.push(of_word);
            }
        }
        units
    }
}

/// One transcript utterance's words and their lexical units, owned together.
/// Projection consumes the words; consumers cannot pair units with another
/// word population or mutate the words after their units were derived.
///
/// Unit texts and their word addresses are equal-length sequences, private
/// and built together by [`UtrLexicalUnits::project_words`]. Aligners borrow
/// the unit texts while every position still names its retained source word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct UtrChatUnits {
    words: Vec<String>,
    texts: Vec<String>,
    of_word: Vec<UtrUnitOfWord>,
}

impl UtrChatUnits {
    /// The exact word population consumed by the unit projection.
    pub(super) fn words(&self) -> &[String] {
        &self.words
    }

    /// The unit texts, in order.
    pub(super) fn texts(&self) -> &[String] {
        &self.texts
    }

    /// How many units the utterance has.
    pub(super) fn len(&self) -> usize {
        self.texts.len()
    }

    /// Whether the utterance has no unit.
    pub(super) fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Every unit's address, in order, as utterance `utterance`.
    pub(super) fn addresses(&self, utterance: UtrUtteranceOrdinal) -> Vec<UtrWordAddress> {
        self.iter()
            .map(|(unit_index, _, of_word)| UtrWordAddress {
                utterance_index: utterance,
                unit_index,
                of_word,
            })
            .collect()
    }

    /// Every unit with its utterance ordinal and its word.
    pub(super) fn iter(&self) -> impl Iterator<Item = (UtrUnitOrdinal, &str, UtrUnitOfWord)> {
        self.texts
            .iter()
            .zip(&self.of_word)
            .enumerate()
            .map(|(index, (text, of_word))| (UtrUnitOrdinal(index), text.as_str(), *of_word))
    }
}

/// Whether a piece of text can be matched: it holds a letter or a digit (a
/// Han character is a letter). Punctuation alone is not a unit on either side.
fn is_lexical(unit: &str) -> bool {
    unit.chars().any(char::is_alphanumeric)
}

/// Han characters one by one; other runs whole, unless punctuation only; CJK
/// punctuation dropped.
fn han_character_units(text: &str) -> Vec<&str> {
    let mut units = Vec::new();
    let mut run: Option<usize> = None;
    fn close<'t>(units: &mut Vec<&'t str>, run: &'t str) {
        if is_lexical(run) {
            units.push(run);
        }
    }
    for (at, ch) in text.char_indices() {
        let ends_run = ch.is_whitespace() || is_cjk_punctuation(ch) || is_cjk_ideograph(ch);
        if ends_run && let Some(start) = run.take() {
            close(&mut units, &text[start..at]);
        }
        if is_cjk_ideograph(ch) {
            units.push(&text[at..at + ch.len_utf8()]);
        } else if !ends_run && run.is_none() {
            run = Some(at);
        }
    }
    if let Some(start) = run {
        close(&mut units, &text[start..]);
    }
    units
}

/// Recognizer tokens ready for UTR: in the transcript's script, carrying the
/// unit projection both sides are matched in.
///
/// The tokens keep the provider's own text (after script normalization) and
/// timing; the projection is applied where UTR reads them, so what is dumped
/// or cached is what the provider said. The units travel with the tokens, so
/// the transcript side cannot be projected differently.
#[derive(Debug, Clone)]
pub struct PreparedUtrTokens {
    tokens: Vec<AsrTimingToken>,
    units: UtrLexicalUnits,
}

impl PreparedUtrTokens {
    /// Prepare recognizer tokens for a transcript in `language`: normalize
    /// the run's script once (a normalization that changed the character
    /// count refuses rather than re-cutting tokens away from their timings)
    /// and fix the unit projection, both from the same typed decision.
    pub fn prepare(
        mut tokens: Vec<AsrTimingToken>,
        language: &LanguageCode3,
    ) -> Result<Self, NormalizationChangedLength> {
        let script = HanScriptNormalization::for_language(language.as_ref());
        if let Some(normalization) = script {
            let aligned = normalization.admit(tokens.iter().map(|token| token.text.as_str()))?;
            for (token, text) in tokens.iter_mut().zip(aligned.into_units()) {
                token.text = text;
            }
        }
        Ok(Self {
            tokens,
            units: UtrLexicalUnits::for_script(script),
        })
    }

    /// Tokens retained from an earlier preparation for a transcript in
    /// `language`: already in its script, so not normalized again, and
    /// matched in the units that language takes, never in units a caller
    /// chooses.
    pub fn retained(tokens: Vec<AsrTimingToken>, language: &LanguageCode3) -> Self {
        Self {
            tokens,
            units: UtrLexicalUnits::for_language(language),
        }
    }

    /// The tokens as the provider gave them (script-normalized).
    pub fn tokens(&self) -> &[AsrTimingToken] {
        &self.tokens
    }

    /// The projection both sides are matched in.
    pub fn units(&self) -> UtrLexicalUnits {
        self.units
    }
}

#[cfg(test)]
impl PreparedUtrTokens {
    /// Space-delimited tokens, the projection of every language without
    /// Han-script handling.
    pub(crate) fn spaced(tokens: &[AsrTimingToken]) -> Self {
        Self {
            tokens: tokens.to_vec(),
            units: UtrLexicalUnits::SpaceDelimitedWords,
        }
    }
}

#[cfg(test)]
impl UtrChatUnits {
    /// Space-delimited units of `words`, for tests of local recovery.
    pub(crate) fn spaced(words: &[String]) -> Self {
        UtrLexicalUnits::SpaceDelimitedWords.project_words(words.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn han_characters_split_ideographs_and_keep_other_runs_whole() {
        assert_eq!(
            UtrLexicalUnits::HanCharacters.split("A12小狗跑！can't 3.5"),
            ["A12", "小", "狗", "跑", "can't", "3.5"]
        );
        assert_eq!(
            UtrLexicalUnits::SpaceDelimitedWords.split("A12小狗跑！can't 3.5"),
            ["A12小狗跑！can't", "3.5"]
        );
    }

    /// The tokens keep the provider's own text: the projection is a reading,
    /// not a rewrite, so a dump or a cache row shows what the provider said.
    #[test]
    fn preparation_keeps_the_provider_text_and_interval() {
        let prepared = PreparedUtrTokens::prepare(
            vec![AsrTimingToken {
                text: "佢哋食嘢。".into(),
                start_ms: 100,
                end_ms: 900,
            }],
            &LanguageCode3::yue(),
        )
        .expect("the Cantonese table preserves character counts");
        assert_eq!(prepared.units(), UtrLexicalUnits::HanCharacters);
        let token = &prepared.tokens()[0];
        assert_eq!(
            (&*token.text, token.start_ms, token.end_ms),
            ("佢哋食嘢。", 100, 900)
        );
        let english = PreparedUtrTokens::prepare(prepared.tokens().to_vec(), &LanguageCode3::eng())
            .expect("no normalization to refuse");
        assert_eq!(english.units(), UtrLexicalUnits::SpaceDelimitedWords);
    }
}
