//! Portuguese ordinal numerals written with an ordinal indicator.
//!
//! ASR providers transcribing Portuguese emit ordinals the way they are
//! printed: ASCII digits followed by the masculine indicator `º` or the
//! feminine indicator `ª`, optionally with an abbreviation period before the
//! indicator (`54.ª`) and a plural `s` after it (`54.ªs`). The digits are
//! illegal on a Portuguese CHAT main tier (E220), and the generic tokenizer
//! would split `54.ª` at its period, inventing a sentence end and leaving a
//! stray indicator. This module owns both halves: recognizing the written form
//! at a token boundary, and rendering it to words carrying the gender and
//! number the form marks.
//!
//! # Types, in the order a token travels
//!
//! 1. `PortugueseOrdinalSyntax`: the written shape (digit run, gender from the
//!    indicator, number from the plural mark), borrowed from the source text.
//!    Built only by `PortugueseOrdinalSyntax::lex_prefix`, the single parser of
//!    the form, which also demands a token boundary right after it.
//! 2. `PortugueseOrdinal`: a syntax whose rank lies in the rendered range
//!    1 through 1000. Built only by `TryFrom<PortugueseOrdinalSyntax>`.
//! 3. `PortugueseOrdinal::to_words`: the rendering.
//!
//! Recognition and range are deliberately separate. `1001.º` is still one
//! abbreviation token (its period is not a sentence end) although there is no
//! rendering for it; `expand_number` then keeps it exactly as written, which is
//! the number-expansion module's policy for any expansion it cannot perform, so
//! CHAT validation reports the digits instead of a guessed word.
//!
//! Rendering composes the hundreds, tens and units ordinals
//! (`655º` is `sexcentésimo quinquagésimo quinto`), with `milésimo` for 1000.
//! Each component has one fixed spelling here; spelling variants are not
//! produced.

use super::prepare::normalized_split_separator;

/// U+00BA MASCULINE ORDINAL INDICATOR. Written as an escape because the
/// degree sign U+00B0 looks the same and must not match.
const MASCULINE_INDICATOR: char = '\u{00BA}';
/// U+00AA FEMININE ORDINAL INDICATOR.
const FEMININE_INDICATOR: char = '\u{00AA}';
/// Optional abbreviation period between the digits and the indicator.
const ABBREVIATION_PERIOD: char = '.';
/// Optional plural mark after the indicator.
const PLURAL_MARK: char = 's';

/// Highest rank the renderer covers.
const RANK_MAX: u16 = 1000;

/// Ordinal stems without their final gender vowel, indexed by digit. Zero has
/// no word of its own, so it is `None` rather than an empty-string sentinel.
const UNIT_STEMS: [Option<&str>; 10] = [
    None,
    Some("primeir"),
    Some("segund"),
    Some("terceir"),
    Some("quart"),
    Some("quint"),
    Some("sext"),
    Some("sétim"),
    Some("oitav"),
    Some("non"),
];
const TEN_STEMS: [Option<&str>; 10] = [
    None,
    Some("décim"),
    Some("vigésim"),
    Some("trigésim"),
    Some("quadragésim"),
    Some("quinquagésim"),
    Some("sexagésim"),
    Some("septuagésim"),
    Some("octogésim"),
    Some("nonagésim"),
];
const HUNDRED_STEMS: [Option<&str>; 10] = [
    None,
    Some("centésim"),
    Some("ducentésim"),
    Some("tricentésim"),
    Some("quadringentésim"),
    Some("quingentésim"),
    Some("sexcentésim"),
    Some("septingentésim"),
    Some("octingentésim"),
    Some("noningentésim"),
];
const THOUSAND_STEM: &str = "milésim";

/// Indicator ordinals are a Portuguese convention; every entry point is
/// gated on the ISO 639-3 code.
fn is_portuguese(lang: &str) -> bool {
    lang.eq_ignore_ascii_case("por")
}

/// Grammatical gender marked by the ordinal indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrdinalGender {
    /// `º`: `primeiro`.
    Masculine,
    /// `ª`: `primeira`.
    Feminine,
}

impl OrdinalGender {
    fn from_indicator(ch: char) -> Option<Self> {
        match ch {
            MASCULINE_INDICATOR => Some(Self::Masculine),
            FEMININE_INDICATOR => Some(Self::Feminine),
            _ => None,
        }
    }

    /// The final vowel every ordinal component takes in this gender.
    fn vowel(self) -> char {
        match self {
            Self::Masculine => 'o',
            Self::Feminine => 'a',
        }
    }
}

/// Grammatical number marked by the optional plural `s`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrammaticalNumber {
    Singular,
    Plural,
}

impl GrammaticalNumber {
    /// Suffix every ordinal component takes in this number.
    fn suffix(self) -> &'static str {
        match self {
            Self::Singular => "",
            Self::Plural => "s",
        }
    }
}

/// The written shape of an indicator ordinal, borrowed from the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PortugueseOrdinalSyntax<'a> {
    /// The whole written form, for example `54.ªs`.
    written: &'a str,
    /// The ASCII digit run, for example `54`.
    digits: &'a str,
    gender: OrdinalGender,
    number: GrammaticalNumber,
}

impl<'a> PortugueseOrdinalSyntax<'a> {
    /// Lex an indicator ordinal at the start of `text`.
    ///
    /// Grammar: `DIGIT+ "."? INDICATOR "s"?`, and the next character must
    /// end the token: end of text, whitespace, or a character the tokenizer
    /// splits on. That boundary rule is what keeps `54ªabc`, `54.ª-feira` and
    /// the degree-sign lookalike `54°` out, while `54.ª.` still yields `54.ª`
    /// with its final period left over as a sentence terminator.
    fn lex_prefix(text: &'a str) -> Option<Self> {
        let digits_len = text.bytes().take_while(u8::is_ascii_digit).count();
        if digits_len == 0 {
            return None;
        }
        // ASCII digits are one byte each, so this is a char boundary.
        let (digits, after_digits) = text.split_at(digits_len);
        let after_period = after_digits
            .strip_prefix(ABBREVIATION_PERIOD)
            .unwrap_or(after_digits);
        let mut chars = after_period.chars();
        let gender = OrdinalGender::from_indicator(chars.next()?)?;
        let after_indicator = chars.as_str();
        let (number, after_ordinal) = match after_indicator.strip_prefix(PLURAL_MARK) {
            Some(after_plural) => (GrammaticalNumber::Plural, after_plural),
            None => (GrammaticalNumber::Singular, after_indicator),
        };
        let at_token_boundary = after_ordinal
            .chars()
            .next()
            .is_none_or(|next| next.is_whitespace() || normalized_split_separator(next).is_some());
        at_token_boundary.then(|| Self {
            written: &text[..text.len() - after_ordinal.len()],
            digits,
            gender,
            number,
        })
    }

    /// Recognize `token` as exactly one indicator ordinal, nothing more.
    fn parse_token(token: &'a str) -> Option<Self> {
        Self::lex_prefix(token).filter(|syntax| syntax.written.len() == token.len())
    }
}

/// A recognized indicator ordinal whose rank has no rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("Portuguese ordinal `{written}` is outside the rendered range 1..=1000")]
pub(super) struct OrdinalOutOfRange<'a> {
    /// The token as written.
    written: &'a str,
}

/// An ordinal rank the renderer covers: 1 through [`RANK_MAX`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrdinalRank(u16);

/// A Portuguese ordinal ready to render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PortugueseOrdinal {
    rank: OrdinalRank,
    gender: OrdinalGender,
    number: GrammaticalNumber,
}

impl<'a> TryFrom<PortugueseOrdinalSyntax<'a>> for PortugueseOrdinal {
    type Error = OrdinalOutOfRange<'a>;

    fn try_from(syntax: PortugueseOrdinalSyntax<'a>) -> Result<Self, Self::Error> {
        match syntax.digits.parse::<u16>() {
            Ok(value @ 1..=RANK_MAX) => Ok(Self {
                rank: OrdinalRank(value),
                gender: syntax.gender,
                number: syntax.number,
            }),
            // Zero, above the covered range, or too large for u16 (the digit
            // run is all ASCII digits, so overflow is the only parse failure).
            Ok(_) | Err(_) => Err(OrdinalOutOfRange {
                written: syntax.written,
            }),
        }
    }
}

impl PortugueseOrdinal {
    /// Render as space-separated words, each inflected for gender and number:
    /// `54ª` is `quinquagésima quarta`, `54.ºs` is `quinquagésimos quartos`.
    fn to_words(self) -> String {
        let rank = usize::from(self.rank.0);
        let stems: [Option<&str>; 3] = if rank == usize::from(RANK_MAX) {
            [Some(THOUSAND_STEM), None, None]
        } else {
            // rank is 1..=999 here, so every index is a single digit.
            [
                HUNDRED_STEMS[rank / 100],
                TEN_STEMS[rank / 10 % 10],
                UNIT_STEMS[rank % 10],
            ]
        };
        let mut words = String::new();
        // A rank of at least 1 always has at least one non-zero digit, so
        // the result is never empty.
        for stem in stems.into_iter().flatten() {
            if !words.is_empty() {
                words.push(' ');
            }
            words.push_str(stem);
            words.push(self.gender.vowel());
            words.push_str(self.number.suffix());
        }
        words
    }
}

/// Outcome of checking one whole token for a Portuguese indicator ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum OrdinalTokenExpansion<'a> {
    /// Not an indicator ordinal in this language; other expanders may apply.
    NotOrdinal,
    /// Rendered words.
    Expanded(String),
    /// Recognized, but the rank has no rendering.
    OutOfRange(OrdinalOutOfRange<'a>),
}

/// Expand `word` if it is exactly one Portuguese indicator ordinal.
pub(super) fn expand_ordinal_token<'a>(word: &'a str, lang: &str) -> OrdinalTokenExpansion<'a> {
    if !is_portuguese(lang) {
        return OrdinalTokenExpansion::NotOrdinal;
    }
    let Some(syntax) = PortugueseOrdinalSyntax::parse_token(word) else {
        return OrdinalTokenExpansion::NotOrdinal;
    };
    match PortugueseOrdinal::try_from(syntax) {
        Ok(ordinal) => OrdinalTokenExpansion::Expanded(ordinal.to_words()),
        Err(out_of_range) => OrdinalTokenExpansion::OutOfRange(out_of_range),
    }
}

/// The indicator ordinal at the start of `text`, if the tokenizer must keep
/// it whole instead of splitting at its abbreviation period.
///
/// Covers out-of-range ranks too: their period is just as much part of the
/// abbreviation.
pub(super) fn protected_ordinal_prefix<'a>(text: &'a str, lang: &str) -> Option<&'a str> {
    if !is_portuguese(lang) {
        return None;
    }
    PortugueseOrdinalSyntax::lex_prefix(text).map(|syntax| syntax.written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(words: &str) -> OrdinalTokenExpansion<'static> {
        OrdinalTokenExpansion::Expanded(words.to_owned())
    }

    #[test]
    fn indicator_ordinals_render_gender_and_number() {
        for (input, words) in [
            ("54ª", "quinquagésima quarta"),
            ("54.ª", "quinquagésima quarta"),
            ("54º", "quinquagésimo quarto"),
            ("54.º", "quinquagésimo quarto"),
            ("1ª", "primeira"),
            ("1º", "primeiro"),
            ("3.ª", "terceira"),
            ("21ª", "vigésima primeira"),
            ("21º", "vigésimo primeiro"),
            ("1ªs", "primeiras"),
            ("54.ªs", "quinquagésimas quartas"),
            ("54ºs", "quinquagésimos quartos"),
            ("100ª", "centésima"),
            ("111º", "centésimo décimo primeiro"),
            ("500º", "quingentésimo"),
            ("655º", "sexcentésimo quinquagésimo quinto"),
            ("700ª", "septingentésima"),
            ("999º", "noningentésimo nonagésimo nono"),
            ("1000ª", "milésima"),
            // Leading zeros do not change the rank.
            ("054º", "quinquagésimo quarto"),
        ] {
            assert_eq!(
                expand_ordinal_token(input, "por"),
                expanded(words),
                "{input}"
            );
            assert_eq!(
                expand_ordinal_token(input, "POR"),
                expanded(words),
                "language code is case-insensitive: {input}"
            );
            assert_eq!(
                expand_ordinal_token(words, "por"),
                OrdinalTokenExpansion::NotOrdinal,
                "rendered words are not re-expanded: {words}"
            );
        }
    }

    #[test]
    fn recognition_rejects_lookalikes_and_partial_tokens() {
        for input in [
            "54\u{00B0}", // degree sign, not the masculine indicator
            "54a",
            "54o",
            "abc54ª",
            "54ªabc",
            "54.ªabc",
            "n.º",
            "3.14",
            "-54ª",
            "54.ª-feira",
            "54..ª",
            "ª",
            ".ª",
            "54",
            "54.",
        ] {
            assert_eq!(
                expand_ordinal_token(input, "por"),
                OrdinalTokenExpansion::NotOrdinal,
                "{input}"
            );
            assert_eq!(protected_ordinal_prefix(input, "por"), None, "{input}");
        }
    }

    #[test]
    fn other_languages_are_never_recognized() {
        for lang in ["eng", "spa", "ita", "xxx"] {
            assert_eq!(
                expand_ordinal_token("54ª", lang),
                OrdinalTokenExpansion::NotOrdinal
            );
            assert_eq!(protected_ordinal_prefix("54.ª", lang), None);
        }
    }

    #[test]
    fn out_of_range_ordinals_are_recognized_but_not_rendered() {
        for input in ["0ª", "1001º", "99999999999999999999999.ª"] {
            assert_eq!(
                expand_ordinal_token(input, "por"),
                OrdinalTokenExpansion::OutOfRange(OrdinalOutOfRange { written: input }),
                "{input}"
            );
            assert_eq!(
                protected_ordinal_prefix(input, "por"),
                Some(input),
                "{input} keeps its abbreviation period"
            );
        }
    }

    #[test]
    fn protected_prefix_leaves_a_following_separator_outside() {
        assert_eq!(protected_ordinal_prefix("54.ª. então", "por"), Some("54.ª"));
        assert_eq!(protected_ordinal_prefix("54.ª?", "por"), Some("54.ª"));
        assert_eq!(protected_ordinal_prefix("54.ªs!", "por"), Some("54.ªs"));
        assert_eq!(protected_ordinal_prefix("1º, 2º", "por"), Some("1º"));
        assert_eq!(protected_ordinal_prefix("54ª então", "por"), Some("54ª"));
    }
}
