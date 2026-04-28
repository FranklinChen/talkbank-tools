//! Provenance-encoding newtype wrappers for text flowing through CHAT pipelines.
//!
//! These wrappers make it explicit whether a string still contains CHAT surface
//! markup or has already been cleaned for downstream linguistic processing.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Raw text as it appears on a CHAT main tier, before any cleaning.
///
/// This can still include CHAT markers, bullets, or other transcript-level
/// surface notation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ChatRawText(String);

impl ChatRawText {
    /// Wraps text that should be treated as raw CHAT surface content.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrows the wrapped raw text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ChatRawText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ChatRawText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Lexical content extracted from CHAT after markup stripping.
///
/// This is the cleaned text suitable for NLP, alignment, comparison, and cache
/// key generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ChatCleanedText(String);

impl ChatCleanedText {
    /// Wraps text that has already been cleaned of CHAT surface markup.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrows the wrapped cleaned text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns an iterator over the characters of the cleaned text.
    pub fn chars(&self) -> std::str::Chars<'_> {
        self.0.chars()
    }

    /// Lowercases the cleaned text.
    pub fn to_lowercase(&self) -> String {
        self.0.to_lowercase()
    }
}

impl fmt::Display for ChatCleanedText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ChatCleanedText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatCleanedText, ChatRawText};

    #[test]
    fn cleaned_and_raw_text_remain_distinct() {
        let raw = ChatRawText::new("hello@c");
        let cleaned = ChatCleanedText::new("hello");

        assert_eq!(raw.as_str(), "hello@c");
        assert_eq!(cleaned.as_str(), "hello");
    }

    #[test]
    fn cleaned_text_serializes_transparently() {
        let text = ChatCleanedText::new("test");
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, "\"test\"");

        let decoded: ChatCleanedText = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, text);
    }
}
