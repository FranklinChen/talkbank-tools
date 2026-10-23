//! Tokenizer realignment: merge Stanza tokens back to original CHAT words.
//!
//! When Stanza's neural tokenizer runs, it may re-split compound words
//! (e.g. "ice-cream" → `["ice", "-", "cream"]`). This module merges those
//! spurious splits back, preserving the 1-to-1 mapping between CHAT words
//! and Stanza tokens.
//!
//! # Architecture
//!
//! This module is the shared implementation used by both:
//! - The PyO3 bridge (`batchalign-core`), called from `align_tokens` pyfunction
//! - The standalone Rust server (`batchalign-server`), called directly
//!
//! The PyO3 crate provides only a thin wrapper that converts `Vec<PatchedToken>`
//! to Stanza's string, MWT-hint, or explicit-component representations.
//!
//! # MWT Hint Convention
//!
//! Stanza's `tokenize_postprocessor` uses a tuple convention:
//!
//! - `(text, True)`: MWT: let the MWT processor expand (e.g. "don't" → do + n't)
//! - `(text, False)`, NOT an MWT: suppress expansion (e.g. merged "ice-cream")
//! - plain string: retain the token (Python restores its unchanged native hint)
//!
//! [`PatchedToken`] encodes this convention at the Rust↔Python boundary.
//! Exact ordered grouping admits one patched token per authoritative word.
//! Merges retain checked French components or carry an explicit MWT hint.
//!
//! French elision retains native tokenizer components under one authoritative
//! word. Joining their spelling is not permission to erase their analysis.

/// Native French components checked against the word they jointly spell.
///
/// Only the realigner constructs this capability. It cannot name unrelated
/// components, introduce spelling, or split across authoritative word boundaries.
#[derive(Debug, Clone, PartialEq)]
pub struct FrenchElision {
    surface: String,
    components: Vec<String>,
}

impl FrenchElision {
    fn admit(surface: &str, components: Vec<String>) -> Option<Self> {
        let (last, prefixes) = components.split_last()?;
        if prefixes.is_empty()
            || last.is_empty()
            || components
                .iter()
                .any(|part| part.chars().any(char::is_whitespace))
            || prefixes.iter().any(|part| {
                !(part.ends_with('\'') || part.ends_with('’')) || part.chars().count() < 2
            })
            || components.concat() != surface
        {
            return None;
        }
        Some(Self {
            surface: surface.to_owned(),
            components,
        })
    }

    /// The single authoritative word.
    pub fn surface(&self) -> &str {
        &self.surface
    }

    /// Exactly the native components that spell that word.
    pub fn components(&self) -> &[String] {
        &self.components
    }
}

/// Token produced by [`align_tokens`] at the Rust↔Python boundary.
///
/// Encodes Stanza's tokenize-postprocessor MWT hint convention:
/// - `Plain(text)`, unchanged token whose native hint Python can restore
/// - `Hint(text, true)`: force MWT expansion
/// - `Hint(text, false)`: suppress MWT expansion
#[derive(Debug, Clone, PartialEq)]
pub enum PatchedToken {
    /// Plain string, no MWT hint.
    Plain(String),
    /// MWT hint tuple: `(text, should_expand)`.
    Hint(String, bool),
    /// Stanza's explicit expansion, retaining checked native French components.
    FrenchElision(FrenchElision),
}

impl PatchedToken {
    /// Extract the text content regardless of variant.
    pub fn text(&self) -> &str {
        match self {
            PatchedToken::Plain(s) | PatchedToken::Hint(s, _) => s,
            PatchedToken::FrenchElision(group) => group.surface(),
        }
    }
}

// ─── Realignment language ───────────────────────────────────────────────────

/// The languages whose realignment differs, read from a pipeline's Stanza
/// code once, where the pipeline is built.
///
/// Replaces an `alpha2: &str` that every rule compared against `"en"` or
/// `"fr"`, and that Python defaulted to `""`: an English pipeline built
/// without its code silently lost contraction expansion, and nothing could
/// tell that empty string from a language with no rule. `Other` is every
/// language without a rule of its own; an empty code is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealignLanguage {
    /// Stanza `en`: a merged token with an apostrophe is a contraction for
    /// the MWT processor to expand.
    English,
    /// Stanza `fr`: native elision components are retained.
    French,
    /// Stanza `it`: no realignment rule here, but the Python postprocessor
    /// applies its multi-word-token policy.
    Italian,
    /// Every other language: merges suppress expansion.
    Other,
}

/// A Stanza language code that names no language.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("token realignment needs a Stanza language code; got an empty one")]
pub struct EmptyStanzaCode;

impl RealignLanguage {
    /// The realignment language of a Stanza language code (`"en"`, `"fr"`,
    /// `"it"`, `"zh-hans"`, ...).
    ///
    /// # Errors
    ///
    /// [`EmptyStanzaCode`] for an empty code.
    pub fn of_stanza_code(code: &str) -> Result<Self, EmptyStanzaCode> {
        Ok(match code {
            "" => return Err(EmptyStanzaCode),
            "en" => Self::English,
            "fr" => Self::French,
            "it" => Self::Italian,
            _ => Self::Other,
        })
    }

    /// Whether a merged token is an MWT contraction for this language.
    ///
    /// Replicates batchalign2's `ud.py` lines 680-685 for English: the token
    /// contains `'` and its prefix before the first `'` is not `"o"`
    /// (o'clock, o'er). No other language marks one.
    fn marks_contraction(self, text: &str) -> bool {
        match self {
            Self::English => {
                text.contains('\'')
                    && text
                        .split('\'')
                        .next()
                        .is_none_or(|prefix| prefix.trim().to_lowercase() != "o")
            }
            Self::French | Self::Italian | Self::Other => false,
        }
    }
}

// ─── Core alignment algorithm ───────────────────────────────────────────────

/// Align Stanza tokenizer output back to original CHAT words.
///
/// Native tokens must exactly tile each authoritative cleaned word in order.
/// A token crossing a word boundary, missing content or different spelling
/// refuses admission; it never falls back to an unbound token sequence.
///
/// French native elision components are retained as an explicit expansion;
/// English contraction merges request model expansion. Other merges suppress it.
///
/// # Arguments
///
/// - `original_words`: cleaned words derived from the typed CHAT structure
/// - `stanza_tokens`: tokens from Stanza's neural tokenizer
/// - `language`: the pipeline's realignment language
pub fn align_tokens(
    original_words: &[String],
    stanza_tokens: &[String],
    language: RealignLanguage,
) -> Result<AlignedTokens, TokenAlignmentError> {
    let mut tokens = Vec::with_capacity(original_words.len());
    let mut native_index = 0;
    for (word_index, word) in original_words.iter().enumerate() {
        if word.is_empty() {
            return Err(TokenAlignmentError::EmptyWord { word_index });
        }
        let start = native_index;
        let mut remaining = word.as_str();
        while !remaining.is_empty() {
            let native = stanza_tokens
                .get(native_index)
                .ok_or(TokenAlignmentError::MissingToken { word_index })?;
            if native.is_empty() {
                return Err(TokenAlignmentError::EmptyToken { native_index });
            }
            remaining = remaining.strip_prefix(native.as_str()).ok_or(
                TokenAlignmentError::WordBoundaryMismatch {
                    word_index,
                    native_index,
                },
            )?;
            native_index += 1;
        }
        let components = &stanza_tokens[start..native_index];
        if components.len() == 1 {
            tokens.push(PatchedToken::Plain(word.clone()));
        } else {
            let elision = match language {
                RealignLanguage::French => FrenchElision::admit(word, components.to_vec()),
                RealignLanguage::English | RealignLanguage::Italian | RealignLanguage::Other => {
                    None
                }
            };
            tokens.push(match elision {
                Some(group) => PatchedToken::FrenchElision(group),
                None => PatchedToken::Hint(word.clone(), language.marks_contraction(word)),
            });
        }
    }
    if native_index != stanza_tokens.len() {
        return Err(TokenAlignmentError::ExtraTokens { native_index });
    }
    Ok(AlignedTokens { tokens })
}

/// A complete one-to-one realignment, admitted only by [`align_tokens`].
#[derive(Debug)]
pub struct AlignedTokens {
    tokens: Vec<PatchedToken>,
}

impl AlignedTokens {
    /// Checked tokens, one per authoritative word in input order.
    pub fn tokens(&self) -> &[PatchedToken] {
        &self.tokens
    }
}

/// Native tokenization could not establish the authoritative word binding.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenAlignmentError {
    /// The typed input contained an empty word.
    #[error("token realignment: authoritative word {word_index} is empty")]
    EmptyWord {
        /// Zero-based authoritative word position.
        word_index: usize,
    },
    /// The native token sequence contained an empty token.
    #[error("token realignment: native token {native_index} is empty")]
    EmptyToken {
        /// Zero-based native token position.
        native_index: usize,
    },
    /// The native sequence ended before its word was complete.
    #[error("token realignment: missing native content for word {word_index}")]
    MissingToken {
        /// Zero-based authoritative word position.
        word_index: usize,
    },
    /// Native spelling differed or a token crossed an authoritative boundary.
    #[error("token realignment: native token {native_index} does not fit word {word_index}")]
    WordBoundaryMismatch {
        /// Zero-based authoritative word position.
        word_index: usize,
        /// Zero-based native token position.
        native_index: usize,
    },
    /// Native content remained after every word was bound.
    #[error("token realignment: extra native content at token {native_index}")]
    ExtraTokens {
        /// First unbound native token position.
        native_index: usize,
    },
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EN: RealignLanguage = RealignLanguage::English;
    const FR: RealignLanguage = RealignLanguage::French;

    fn aligned(words: &[String], native: &[String], lang: RealignLanguage) -> Vec<PatchedToken> {
        align_tokens(words, native, lang).unwrap().tokens().to_vec()
    }

    // ── RealignLanguage ───────────────────────────────────────────────

    #[test]
    fn a_stanza_code_names_its_realignment_language() {
        assert_eq!(RealignLanguage::of_stanza_code("en"), Ok(EN));
        assert_eq!(RealignLanguage::of_stanza_code("fr"), Ok(FR));
        assert_eq!(
            RealignLanguage::of_stanza_code("it"),
            Ok(RealignLanguage::Italian)
        );
        for code in ["de", "nl", "zh-hans", "pt"] {
            assert_eq!(
                RealignLanguage::of_stanza_code(code),
                Ok(RealignLanguage::Other)
            );
        }
    }

    /// The empty code used to be a default that read as "no rule": an
    /// English pipeline built without its code lost contraction expansion.
    #[test]
    fn an_empty_stanza_code_is_refused() {
        assert_eq!(RealignLanguage::of_stanza_code(""), Err(EmptyStanzaCode));
    }

    #[test]
    fn test_english_contraction_detected() {
        for text in ["don't", "can't", "won't", "it's", "I'm", "Claus'", "John's"] {
            assert!(EN.marks_contraction(text), "{text}");
        }
    }

    #[test]
    fn test_english_oclock_not_contraction() {
        assert!(!EN.marks_contraction("o'clock"));
        assert!(!EN.marks_contraction("O'er"));
    }

    #[test]
    fn test_non_english_not_contraction() {
        assert!(!FR.marks_contraction("l'homme"));
        assert!(!RealignLanguage::Italian.marks_contraction("dell'acqua"));
        assert!(!RealignLanguage::Other.marks_contraction("d'água"));
        assert!(!RealignLanguage::Other.marks_contraction("'s"));
    }

    #[test]
    fn test_no_apostrophe_not_contraction() {
        assert!(!EN.marks_contraction("hello"));
        assert!(!EN.marks_contraction("ice-cream"));
    }

    // ── align_tokens ─────────────────────────────────────────────────

    #[test]
    fn test_empty_inputs() {
        let result = aligned(&[], &[], EN);
        assert!(result.is_empty());
    }

    #[test]
    fn test_one_to_one_mapping() {
        let words = vec!["hello".into(), "world".into()];
        let tokens = vec!["hello".into(), "world".into()];
        let result = aligned(&words, &tokens, EN);
        assert_eq!(
            result,
            vec![
                PatchedToken::Plain("hello".into()),
                PatchedToken::Plain("world".into()),
            ]
        );
    }

    #[test]
    fn test_merge_compound() {
        // Stanza splits "ice-cream" into ["ice", "-", "cream"]
        let words = vec!["ice-cream".into()];
        let tokens = vec!["ice".into(), "-".into(), "cream".into()];
        let result = aligned(&words, &tokens, EN);
        assert_eq!(result, vec![PatchedToken::Hint("ice-cream".into(), false)]);
    }

    #[test]
    fn test_english_contraction_merge() {
        // Stanza splits "don't" into ["do", "n't"]
        let words = vec!["don't".into()];
        let tokens = vec!["do".into(), "n't".into()];
        let result = aligned(&words, &tokens, EN);
        assert_eq!(result, vec![PatchedToken::Hint("don't".into(), true)]);
    }

    #[test]
    fn french_native_elision_keeps_components_and_word_identity() {
        let words = vec!["l'escargot".into(), "dort".into()];
        let native = vec!["l'".into(), "escargot".into(), "dort".into()];
        let result = aligned(&words, &native, FR);
        let PatchedToken::FrenchElision(group) = &result[0] else {
            panic!("native French components must survive realignment");
        };
        assert_eq!(group.surface(), "l'escargot");
        assert_eq!(group.components(), &["l'", "escargot"]);
        assert_eq!(result[1], PatchedToken::Plain("dort".into()));
        assert_eq!(
            result.iter().map(PatchedToken::text).collect::<Vec<_>>(),
            words
        );
    }

    #[test]
    fn french_elision_admission_refuses_unrelated_or_empty_components() {
        for (surface, components) in [
            ("l'escargot", vec!["l'", "chat"]),
            ("l'", vec!["l'", ""]),
            ("ice-cream", vec!["ice", "-cream"]),
            ("l'escargot", vec!["l'escargot"]),
            ("l' escargot", vec!["l'", " escargot"]),
        ] {
            assert!(
                FrenchElision::admit(surface, components.into_iter().map(str::to_owned).collect())
                    .is_none()
            );
        }
        assert!(FrenchElision::admit("l’escargot", vec!["l’".into(), "escargot".into()]).is_some());
    }

    #[test]
    fn character_mismatch_refuses_unbound_tokens() {
        let words = vec!["hello".into()];
        let tokens = vec!["goodbye".into()];
        assert!(matches!(
            align_tokens(&words, &tokens, EN),
            Err(TokenAlignmentError::WordBoundaryMismatch { .. })
        ));
    }

    #[test]
    fn raw_shortening_notation_is_not_reparsed_by_the_realigner() {
        let words = vec!["(be)cause".into()];
        let tokens = vec!["because".into()];
        assert!(align_tokens(&words, &tokens, EN).is_err());
        assert_eq!(
            aligned(&["because".into()], &tokens, EN),
            vec![PatchedToken::Plain("because".into())]
        );
    }

    // ── align_tokens: cross-language passthrough (no per-language rules) ──

    #[test]
    fn test_english_passthrough_no_patches() {
        let words = vec!["the".into(), "dog".into()];
        let tokens = vec!["the".into(), "dog".into()];
        let result = aligned(&words, &tokens, EN);
        assert_eq!(
            result,
            vec![
                PatchedToken::Plain("the".into()),
                PatchedToken::Plain("dog".into()),
            ]
        );
    }

    #[test]
    fn native_tokens_cannot_cross_authoritative_word_boundaries() {
        assert!(align_tokens(&["foo".into(), "bar".into()], &["foobar".into()], EN).is_err());
        assert!(align_tokens(&["foo".into()], &["foo".into(), "bar".into()], EN).is_err());
        assert!(align_tokens(&["foo".into()], &[], EN).is_err());
        assert!(align_tokens(&[], &["foo".into()], EN).is_err());
        assert!(align_tokens(&["".into()], &[], EN).is_err());
        assert!(align_tokens(&["foo".into()], &["".into(), "foo".into()], EN).is_err());
    }

    #[test]
    fn unicode_native_components_tile_words_without_character_maps() {
        let result = aligned(
            &["école".into(), "中文".into()],
            &["é".into(), "cole".into(), "中文".into()],
            FR,
        );
        assert_eq!(
            result.iter().map(PatchedToken::text).collect::<Vec<_>>(),
            ["école", "中文"]
        );
    }
}
