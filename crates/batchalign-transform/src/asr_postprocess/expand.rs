use super::{AsrNormalizedText, AsrTextLanguage, AsrWord, WordKind, expand_number};

/// Stage 4: write every ASR token in a form CHAT can hold.
///
/// Three kinds of token cannot be written as recognized; each is rewritten
/// by its shape, as the CHAT manual says to write it, and nothing else is
/// touched ([`WrittenForm`]):
///
/// - Digits. "Numbers should be written out in words" (CHAT manual 8.8.4):
///   `expand_number` spells a numeral, ordinal, decade or currency token.
///   Some expansions are several words (`"100"` to `"one hundred"`); they
///   are re-split into separate `AsrWord`s, timing distributed by length,
///   because a `ChatWordText` holds one token.
/// - A letter-led token whose digits stand alone (`b2`, `mp3`, `R2D2`).
///   Words "should not contain numbers" (manual, section 2), and "acronyms
///   or ones with numbers may require underscores, as in C_three_PO and
///   R_two_D_two" (8.8.3): each digit is spelled and joined to the letter
///   runs with underscores, letters kept as recognized (`b_two`,
///   `mp_three`, `R_two_D_two`), in the Latin script only. One token stays
///   one word. A run of two or
///   more digits (`abc123`) is not rewritten: the manual writes numbers
///   "depending on how it was" said ("two five six", "two fifty six", "two
///   hundred fifty six"), and the surface does not say which, so the token
///   stays as recognized and is reported for review.
/// - A token spelled like a CHAT reserved marker (`www`, `xxx`, `yyy`, in
///   any case). Those mark material the transcriber did NOT transcribe; a
///   recognizer never emits that judgement, so the token is something
///   spoken, a string of letters ("W W W"). It is marked as a letter string
///   (`www@k`, manual 8.8.1: "strings of letters ... use the @k symbol"),
///   lowercase as letters are written, one word as recognized. Written
///   bare it would become CHAT's untranscribed marker (`www`) or be refused
///   as a mis-cased one (`Www`).
///
/// Code-switched text keeps its digits (see [`AsrTextLanguage`]): nothing
/// says which language they were spoken in. A digit token whose language has
/// no expander also stays as recognized. Both then fail CHAT's word rules
/// and are reported for review; the reserved-marker rule writes no word in
/// a language, so it applies to every text.
pub fn write_word_forms(words: Vec<AsrWord>, language: AsrTextLanguage<'_>) -> Vec<AsrWord> {
    let mut written = Vec::with_capacity(words.len());
    for word in words {
        match WrittenForm::of(word.text.as_str(), language) {
            WrittenForm::AsRecognized => written.push(word),
            WrittenForm::Rewritten { text, rule } => {
                tracing::debug!(from = %word.text.as_str(), to = %text, ?rule, "ASR token rewritten to a CHAT word form");
                if text.contains(char::is_whitespace) {
                    written.extend(split_expanded_text_into_words(
                        &text,
                        word.start_ms,
                        word.end_ms,
                        word.kind,
                    ));
                } else {
                    written.push(AsrWord {
                        text: AsrNormalizedText::new(text),
                        ..word
                    });
                }
            }
        }
    }
    written
}

/// How one ASR token is written in CHAT.
#[derive(Debug, Clone, PartialEq, Eq)]
enum WrittenForm {
    /// As recognized: CHAT can hold it, or no rule here knows how to write it.
    AsRecognized,
    /// Rewritten by one rule.
    Rewritten { text: String, rule: WordFormRule },
}

/// Which rule rewrote a token, for the trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordFormRule {
    /// Digits spelled out as number words.
    NumberSpelled,
    /// Letters and spelled digits joined by underscores.
    AlphanumericLinkage,
    /// A reserved-marker spelling marked as a letter string.
    LetterString,
}

/// CHAT's markers for material not transcribed: unintelligible (`xxx`),
/// phonological coding only (`yyy`), untranscribed (`www`).
const RESERVED_MARKERS: [&str; 3] = ["xxx", "yyy", "www"];

impl WrittenForm {
    fn of(token: &str, language: AsrTextLanguage<'_>) -> Self {
        if let Some(marker) = RESERVED_MARKERS
            .iter()
            .find(|marker| token.eq_ignore_ascii_case(marker))
        {
            return Self::Rewritten {
                text: format!("{marker}@k"),
                rule: WordFormRule::LetterString,
            };
        }
        // Every expander and the linkage below need a digit.
        if !token.bytes().any(|b| b.is_ascii_digit()) {
            return Self::AsRecognized;
        }
        let lang = match language {
            AsrTextLanguage::One(lang) => lang,
            AsrTextLanguage::CodeSwitched { .. } => return Self::AsRecognized,
        };
        let expanded = expand_number(token, lang);
        if expanded != token {
            return Self::Rewritten {
                text: expanded,
                rule: WordFormRule::NumberSpelled,
            };
        }
        match alphanumeric_linkage(token, lang) {
            Some(text) => Self::Rewritten {
                text,
                rule: WordFormRule::AlphanumericLinkage,
            },
            None => Self::AsRecognized,
        }
    }
}

/// `b2` to `b_two`, `R2D2` to `R_two_D_two`: a token of letters and ASCII
/// digits that starts with a letter and whose digits stand alone, each digit
/// spelled in `lang` and joined to the letter runs with underscores. `None`
/// for any other shape: a digit-led token (a number with a suffix, which the
/// expander owns), a run of several digits (whose reading the surface does
/// not give), or a digit `lang` cannot spell.
///
/// Latin script only, letters and spelled digits alike, as the manual's
/// examples are: joining `A` to a Han or Cyrillic numeral would mix scripts in
/// one word, which nothing in the manual writes, so such a token stays as
/// recognized.
fn alphanumeric_linkage(token: &str, lang: &str) -> Option<String> {
    let letter_led = token
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic());
    if !letter_led || !token.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    let mut parts = Vec::new();
    let mut rest = token;
    while let Some(first) = rest.chars().next() {
        let digits = first.is_ascii_digit();
        let end = rest
            .find(|c: char| c.is_ascii_digit() != digits)
            .unwrap_or(rest.len());
        let (run, tail) = rest.split_at(end);
        if digits {
            // One digit has one reading; several do not.
            if run.len() != 1 {
                return None;
            }
            let spelled = expand_number(run, lang);
            if spelled == run || !spelled.chars().all(is_latin_letter) {
                return None;
            }
            parts.push(spelled);
        } else {
            parts.push(run.to_owned());
        }
        rest = tail;
    }
    Some(parts.join("_"))
}

/// A letter of the Latin script: ASCII, or Latin-1 Supplement and Latin
/// Extended-A/B (`é`, `ñ`, `ş`), which the spelled digits of Latin-script
/// languages use.
fn is_latin_letter(c: char) -> bool {
    c.is_ascii_alphabetic() || (('\u{C0}'..='\u{24F}').contains(&c) && c.is_alphabetic())
}

/// Distribute a whitespace-separated expansion across several
/// `AsrWord`s, proportioning timing by text length.
///
/// Called only by [`write_word_forms`] on an `expand_number` result, which
/// is deterministic given the original token, so no need to re-run any
/// normalization on the split parts.
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
