//! CHAT parsing helpers for transform-oriented callers.
//!
//! These are thin convenience wrappers over `talkbank-parser` that keep the
//! common Batchalign call pattern available from the canonical shared crate.

use talkbank_model::model::ChatFile;

pub use talkbank_parser::TreeSitterParser;

/// Parse CHAT text leniently (tree-sitter with error recovery).
///
/// Always returns a `ChatFile` (best-effort), plus any parse warnings/errors.
pub fn parse_lenient(
    parser: &TreeSitterParser,
    chat_text: &str,
) -> (ChatFile, Vec<talkbank_model::ParseError>) {
    let errors = talkbank_model::ErrorCollector::new();
    let chat_file = parser.parse_chat_file_streaming(chat_text, &errors);
    let error_vec = errors.into_vec();
    (chat_file, error_vec)
}

/// Parse CHAT text strictly (tree-sitter, no error recovery).
pub fn parse_strict(
    parser: &TreeSitterParser,
    chat_text: &str,
) -> Result<ChatFile, talkbank_model::ParseErrors> {
    parser.parse_chat_file(chat_text)
}

/// Check whether a parsed CHAT file has `@Options: dummy`.
///
/// Dummy files are pass-through placeholders that should not be processed by
/// any NLP pipeline.
pub fn is_dummy(chat_file: &ChatFile) -> bool {
    use talkbank_model::model::ChatOptionFlag;

    chat_file
        .options
        .iter()
        .any(|f| matches!(f, ChatOptionFlag::Unsupported(s) if s == "dummy"))
}

/// Check whether a parsed CHAT file has `@Options: NoAlign`.
///
/// Files with `NoAlign` should skip forced alignment and be output unchanged by
/// alignment-oriented commands.
pub fn is_no_align(chat_file: &ChatFile) -> bool {
    chat_file.options.iter().any(|f| f.skips_alignment())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI||female|||Target_Child|||\n*CHI:\thello world .\n@End\n";

    const DUMMY_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Options:\tdummy\n\
@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI||female|||Target_Child|||\n\
*CHI:\thello world .\n@End\n";

    const NOALIGN_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Options:\tNoAlign\n\
@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI||female|||Target_Child|||\n\
*CHI:\thello world .\n@End\n";

    const BULLETS_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Options:\tbullets\n\
@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI||female|||Target_Child|||\n\
*CHI:\thello world .\n@End\n";

    #[test]
    fn test_parse_strict_round_trip_basics() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let chat_file = parse_strict(&parser, MINIMAL_CHAT).expect("strict parse");
        assert_eq!(chat_file.utterances().count(), 1);
    }

    #[test]
    fn test_is_dummy_with_dummy_option() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let (chat_file, _) = parse_lenient(&parser, DUMMY_CHAT);
        assert!(is_dummy(&chat_file));
    }

    #[test]
    fn test_is_dummy_without_options() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let (chat_file, _) = parse_lenient(&parser, MINIMAL_CHAT);
        assert!(!is_dummy(&chat_file));
    }

    #[test]
    fn test_is_dummy_with_other_options() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let (chat_file, _) = parse_lenient(&parser, BULLETS_CHAT);
        assert!(!is_dummy(&chat_file));
    }

    #[test]
    fn test_is_no_align_with_noalign_option() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let (chat_file, _) = parse_lenient(&parser, NOALIGN_CHAT);
        assert!(is_no_align(&chat_file));
    }

    #[test]
    fn test_is_no_align_without_options() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let (chat_file, _) = parse_lenient(&parser, MINIMAL_CHAT);
        assert!(!is_no_align(&chat_file));
    }

    #[test]
    fn test_is_no_align_with_dummy_option() {
        let parser = TreeSitterParser::new().expect("tree-sitter parser");
        let (chat_file, _) = parse_lenient(&parser, DUMMY_CHAT);
        assert!(!is_no_align(&chat_file));
    }
}
