//! Han-character tokenization helpers for the Chinese provider bridges.
//!
//! Splitting provider text one character per unit, and recognizing the CJK
//! punctuation providers emit. Neither step converts script: Han-script
//! normalization (Hong Kong traditional for Cantonese, Taiwan traditional for
//! Taiwanese) has one owner, [`super::han_script`], and runs once per run of
//! units there.

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
// Character tokenization
// ---------------------------------------------------------------------------

/// Split text into per-character tokens, dropping CJK punctuation and spaces.
///
/// This does NOT normalize. It used to, which made it a second owner of a
/// non-idempotent transformation and normalized each character on its own, so
/// `聯係` reached CHAT as `聯係` rather than `聯繫`. Normalization happens once
/// per run, before this stage, through
/// [`super::han_script::HanScriptNormalization::admit`].
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
