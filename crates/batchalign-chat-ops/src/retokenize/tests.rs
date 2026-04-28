use super::*;
use crate::extract;
use mapping::{build_word_token_mapping, try_deterministic_word_token_mapping};
use parse_helpers::{handle_ending_punct_skip, try_parse_token_as_word};
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{ChatFile, GrammaticalRelation, Line, Mor, Utterance, WriteChat};
use talkbank_parser::TreeSitterParser;

fn parse_chat(text: &str) -> ChatFile {
    let parser = TreeSitterParser::new().unwrap();
    parser.parse_chat_file(text).unwrap()
}

fn get_utterance(chat: &mut ChatFile, idx: usize) -> &mut Utterance {
    let mut utt_idx = 0;
    for line in &mut chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt;
            }
            utt_idx += 1;
        }
    }
    panic!("Utterance {idx} not found");
}

fn extract_words(utt: &Utterance) -> Vec<extract::ExtractedWord> {
    let mut words = Vec::new();
    extract::collect_utterance_content(&utt.main.content.content, TierDomain::Mor, &mut words);
    words
}

/// Construct a `Mor` item from a compact `"POS|lemma"` or `"POS:sub|lemma-Feat"` string.
///
/// Supports:
/// - `"v|eat"` → POS=v, lemma=eat
/// - `"pro|I"` → POS=pro:sub, lemma=I
/// - `"n|cookie-PL"` → POS=n, lemma=cookie, feature=PL
/// - `"aux|go&PRESP"` → POS=aux, lemma=go, feature=PRESP
/// - `"v|do-3S"` → POS=v, lemma=do, feature=3S
///
/// batchalign3 constructs `Mor` values from structured Stanza output (not CHAT text),
/// so this helper mirrors that direct-construction path rather than parsing CHAT syntax.
fn mor(spec: &str) -> Mor {
    use talkbank_model::model::{MorFeature, MorStem, MorWord, PosCategory};

    let (pos_str, rest) = spec.split_once('|').unwrap_or((spec, ""));

    // Find the first feature separator ('-' or '&') to split lemma from features.
    let sep_pos = rest.find(['-', '&']);
    let (lemma, features_str) = match sep_pos {
        Some(pos) => (&rest[..pos], &rest[pos + 1..]),
        None => (rest, ""),
    };

    let mut word = MorWord::new(PosCategory::new(pos_str), MorStem::new(lemma));
    if !features_str.is_empty() {
        // Features are separated by '-' (suffixes) or '&' (fusion)
        for feat in features_str.split(['-', '&']) {
            if !feat.is_empty() {
                word = word.with_feature(MorFeature::new(feat));
            }
        }
    }
    Mor::new(word)
}

/// Construct a `GrammaticalRelation` from `"index|head|RELATION"`.
fn gra(spec: &str) -> GrammaticalRelation {
    let parts: Vec<&str> = spec.split('|').collect();
    assert_eq!(
        parts.len(),
        3,
        "GRA spec must be 'index|head|RELATION': {spec}"
    );
    GrammaticalRelation::new(
        parts[0].parse::<usize>().unwrap(),
        parts[1].parse::<usize>().unwrap(),
        parts[2],
    )
}

/// Build a Mor vector and optional terminator from a compact spec string.
///
/// Format: `"POS|lemma POS|lemma ."` — words separated by spaces, terminator is
/// the trailing punctuation character.
fn build_mor(spec: &str) -> (Vec<Mor>, Option<String>) {
    let tokens: Vec<&str> = spec.split_whitespace().collect();
    let mut mors = Vec::new();
    let mut terminator = None;
    for tok in tokens {
        if tok == "." || tok == "?" || tok == "!" {
            terminator = Some(tok.to_string());
        } else {
            mors.push(mor(tok));
        }
    }
    (mors, terminator)
}

/// Build a GrammaticalRelation vector from a compact spec string.
///
/// Format: `"1|2|SUBJ 2|0|ROOT 3|2|PUNCT"` — relations separated by spaces.
fn build_gra(spec: &str) -> Vec<GrammaticalRelation> {
    spec.split_whitespace().map(gra).collect()
}

#[test]
fn test_try_deterministic_word_token_mapping_exact() {
    let chat_text = include_str!("../../../../test-fixtures/retok_i_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["i".to_string(), "eat".to_string()];

    let mapping = try_deterministic_word_token_mapping(&original_words, &stanza_tokens)
        .expect("deterministic mapping should succeed");
    assert_eq!(mapping.len(), 2);
    assert_eq!(mapping[0].as_slice(), &[0]);
    assert_eq!(mapping[1].as_slice(), &[1]);
}

#[test]
fn test_try_deterministic_word_token_mapping_split_and_merge() {
    let chat_text = include_str!("../../../../test-fixtures/retok_gon_na_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["gonna".to_string(), "eat".to_string()];

    let mapping = try_deterministic_word_token_mapping(&original_words, &stanza_tokens)
        .expect("deterministic mapping should succeed");
    assert_eq!(mapping.len(), 3);
    assert_eq!(mapping[0].as_slice(), &[0]);
    assert_eq!(mapping[1].as_slice(), &[0]);
    assert_eq!(mapping[2].as_slice(), &[1]);
}

#[test]
fn test_try_deterministic_word_token_mapping_none_on_diverged_text() {
    let chat_text = include_str!("../../../../test-fixtures/retok_i_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["you".to_string(), "eat".to_string()];

    let mapping = try_deterministic_word_token_mapping(&original_words, &stanza_tokens);
    assert!(
        mapping.is_none(),
        "diverged text should require non-deterministic fallback"
    );
}

#[test]
fn test_build_word_token_mapping_diverged_equal_lengths_indexes_by_position() {
    let chat_text = include_str!("../../../../test-fixtures/retok_gonna_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["going".to_string(), "eat".to_string()];

    let mapping = build_word_token_mapping(&original_words, &stanza_tokens);
    assert_eq!(mapping.word_count(), 2);
    assert_eq!(mapping.tokens_for_word(0), &[0]);
    assert_eq!(mapping.tokens_for_word(1), &[1]);
}

#[test]
fn test_build_word_token_mapping_diverged_length_mismatch_uses_monotonic_bins() {
    let chat_text = include_str!("../../../../test-fixtures/retok_i_dont_know.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["alpha".to_string(), "beta".to_string()];

    let mapping = build_word_token_mapping(&original_words, &stanza_tokens);
    assert_eq!(mapping.word_count(), 3);
    assert_eq!(mapping.tokens_for_word(0), &[0]);
    assert_eq!(mapping.tokens_for_word(1), &[0]);
    assert_eq!(mapping.tokens_for_word(2), &[1]);
}

#[test]
fn test_same_tokenization() {
    // 1:1 same text -- words unchanged, mor injected
    let chat_text = include_str!("../../../../test-fixtures/retok_i_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["I".to_string(), "eat".to_string()];

    let (mors, term) = build_mor("pro|I v|eat .");
    let gra_rels = build_gra("1|2|SUBJ 2|0|ROOT 3|2|PUNCT");

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    assert!(
        output.contains("I eat"),
        "Main tier should be unchanged: {output}"
    );
    assert!(
        output.contains("%mor:\tpro|I v|eat ."),
        "Should have %mor tier: {output}"
    );
    assert!(output.contains("%gra:"), "Should have %gra tier: {output}");
}

#[test]
fn test_different_text_1_to_1() {
    // 1:1 different text -- word text updated
    let chat_text = include_str!("../../../../test-fixtures/retok_gonna_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let stanza_tokens = vec!["going".to_string(), "eat".to_string()];

    let (mors, term) = build_mor("aux|go&PRESP v|eat .");
    let gra_rels = build_gra("1|2|AUX 2|0|ROOT 3|2|PUNCT");

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    assert!(
        output.contains("going eat"),
        "Word text should be updated: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_clitic_split_1_to_n() {
    // 1:N split -- single word "don't" becomes "do" and "n't"
    let chat_text = include_str!("../../../../test-fixtures/retok_i_dont_know.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    assert_eq!(original_words.len(), 3); // I, don't, know

    let stanza_tokens = vec![
        "I".to_string(),
        "do".to_string(),
        "n't".to_string(),
        "know".to_string(),
    ];

    let (mors, term) = build_mor("pro|I v|do neg|not v|know .");
    let gra_rels = build_gra("1|2|SUBJ 2|0|ROOT 3|2|NEG 4|2|XCOMP 5|4|PUNCT");

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    // After retokenization, "don't" should be split into "do" and "n't"
    assert!(
        output.contains("do n't") || output.contains("do n't"),
        "Should contain split tokens: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_preserves_non_word_content() {
    // Retrace group should be untouched, only non-retraced words retokenized
    let chat_text = include_str!("../../../../test-fixtures/retok_retrace_i_need_cookie.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    // In mor domain, retrace is skipped: words are [I, need, cookie]
    assert_eq!(original_words.len(), 3);

    let stanza_tokens = vec!["I".to_string(), "need".to_string(), "cookie".to_string()];

    let (mors, term) = build_mor("pro|I v|need n|cookie .");
    let gra_rels = build_gra("1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT");

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    // Retrace group still present
    assert!(
        output.contains("[/]"),
        "Retrace group should be preserved: {output}"
    );
    assert!(
        output.contains("%mor:\tpro|I v|need n|cookie ."),
        "Should have correct %mor: {output}"
    );
}

#[test]
fn test_xbxxx_restoration() {
    // Special form word: Stanza returns "xbxxx", should be restored to original
    let chat_text = include_str!("../../../../test-fixtures/retok_gumma_special_form.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    assert_eq!(original_words[0].text.as_str(), "gumma");
    assert_eq!(
        original_words[0].form_type,
        Some(talkbank_model::model::FormType::C)
    );

    let stanza_tokens = vec!["xbxxx".to_string(), "is".to_string(), "yummy".to_string()];

    let (mors, term) = build_mor("c|gumma v|be&3S adj|yummy .");
    let gra_rels = build_gra("1|2|FLAT 2|0|ROOT 3|2|XCOMP 4|2|PUNCT");

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    // xbxxx should NOT appear in output -- restored to "gumma"
    assert!(
        !output.contains("xbxxx"),
        "xbxxx should be restored: {output}"
    );
    assert!(
        output.contains("gumma"),
        "Original text should be present: {output}"
    );
}

#[test]
fn test_round_trip_serialization() {
    // Parse -> retokenize -> serialize -> verify output
    let chat_text = include_str!("../../../../test-fixtures/retok_i_dont_know.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    let stanza_tokens = vec![
        "I".to_string(),
        "do".to_string(),
        "n't".to_string(),
        "know".to_string(),
    ];

    let mor_str = "pro|I v|do neg|not v|know .";
    let gra_str = "1|2|SUBJ 2|0|ROOT 3|2|NEG 4|2|XCOMP 5|4|PUNCT";

    let (mors, term) = build_mor(mor_str);
    let gra_rels = build_gra(gra_str);

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    // The output should be valid CHAT -- verify it re-parses
    let reparsed = parse_chat(&output);
    let line_count = reparsed
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();
    assert_eq!(line_count, 1, "Should still have 1 utterance");

    // Verify tiers are present
    assert!(output.contains("%mor:"), "Missing %mor tier: {output}");
    assert!(output.contains("%gra:"), "Missing %gra tier: {output}");
}

#[test]
fn test_comma_separator_preserves_words() {
    // Regression test: comma separators are NLP words (cm|cm) but are
    // Separator nodes in the AST. The retokenize walk must increment
    // word_counter for them so subsequent words map correctly.
    let chat_text = include_str!("../../../../test-fixtures/retok_mot_okay_comma.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // Extract should include the comma as an NLP word
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(
        texts,
        vec!["okay", ",", "push", "the", "start", "button"],
        "Comma should be in extracted words"
    );

    // Stanza tokens: same words, no comma (Stanza doesn't tokenize commas as words)
    let stanza_tokens = vec![
        "okay".to_string(),
        "push".to_string(),
        "the".to_string(),
        "start".to_string(),
        "button".to_string(),
    ];

    let mor_str = "intj|okay cm|cm v|push det|the n|start n|button .";
    let gra_str = "1|3|DISCOURSE 2|1|PUNCT 3|6|ROOT 4|6|DET 5|6|COMPOUND 6|3|OBJ 7|3|PUNCT";

    let (mors, term) = build_mor(mor_str);
    let gra_rels = build_gra(gra_str);

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();

    // The main tier must preserve all words -- "button" must NOT be dropped
    assert!(
        output.contains("button"),
        "Word 'button' should not be dropped after comma: {output}"
    );
    assert!(
        output.contains("push"),
        "Word 'push' should be preserved: {output}"
    );
    assert!(
        output.contains("start"),
        "Word 'start' should be preserved: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_non_nlp_words_preserved() {
    // Regression test: non-NLP words (xxx, &~uh) must be preserved as-is.
    // They are not extracted by collect_utterance_content, so retokenize
    // must skip them to keep word_counter in sync with the mapping.
    let chat_text = include_str!("../../../../test-fixtures/retok_mot_filler_popcorn.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // &~uh is NOT extracted (non-NLP), only "popcorn" is
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, vec!["popcorn"], "Only NLP words should be extracted");

    let stanza_tokens = vec!["popcorn".to_string()];

    let mor_str = "n|popcorn !";
    let gra_str = "1|0|ROOT 2|1|PUNCT";

    let (mors, term) = build_mor(mor_str);
    let gra_rels = build_gra(gra_str);

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();

    // &~uh must still be present -- NOT dropped
    assert!(
        output.contains("&~uh"),
        "Non-NLP word &~uh should be preserved: {output}"
    );
    // popcorn must NOT be duplicated
    let popcorn_count = output.matches("popcorn").count();
    assert!(
        popcorn_count <= 2,
        "popcorn should appear in main tier and %mor only, not duplicated: {output}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_xxx_preserved_in_utterance() {
    // Regression test: xxx (unintelligible marker) must not be dropped or
    // cause adjacent words to duplicate.
    let chat_text = include_str!("../../../../test-fixtures/retok_mot_xxx_put_it_here.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // xxx is NOT extracted (non-NLP)
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert!(
        !texts.contains(&"xxx"),
        "xxx should not be in extracted words: {texts:?}"
    );

    // Stanza tokens = the NLP words (without xxx and comma)
    let stanza_tokens: Vec<String> = texts
        .iter()
        .filter(|t| **t != ",")
        .map(|t| t.to_string())
        .collect();

    // Build a simple mor string for the NLP words
    let mor_str = "intj|okay cm|cm co|I~aux|will adv|just v|put pro|it adv|here .";
    let gra_str =
        "1|6|DISCOURSE 2|1|PUNCT 3|6|SUBJ 4|6|AUX 5|6|ADVMOD 6|0|ROOT 7|6|OBJ 8|6|ADVMOD 9|6|PUNCT";

    let (mors, term) = build_mor(mor_str);
    let gra_rels = build_gra(gra_str);

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();

    // xxx must still be present
    assert!(output.contains("xxx"), "xxx should be preserved: {output}");
    // "here" must NOT be duplicated (the old bug pattern)
    let main_line = output
        .lines()
        .find(|l| l.starts_with("*MOT:"))
        .unwrap_or("");
    let here_count = main_line.matches("here").count();
    assert_eq!(
        here_count, 1,
        "'here' should appear once in main tier, not duplicated: {main_line}"
    );
}

#[test]
fn test_ending_punct_mismatch_emits_diagnostic() {
    let mut diagnostics = Vec::new();
    let skipped = handle_ending_punct_skip("?", Some("."), &mut diagnostics);
    assert!(skipped, "ending punctuation should be skipped");
    assert_eq!(diagnostics.len(), 1, "mismatch should be diagnosed");
}

#[test]
fn test_try_parse_token_as_word_returns_none_for_invalid() {
    let parser = TreeSitterParser::new().unwrap();
    let mut diagnostics = Vec::new();
    let result = try_parse_token_as_word(&parser, "two words", &mut diagnostics);
    assert!(
        result.is_none(),
        "invalid CHAT token should return None, not an unchecked Word"
    );
    assert!(
        !diagnostics.is_empty(),
        "invalid token should emit diagnostics"
    );
}

#[test]
fn test_whitespace_in_stanza_token_stripped() {
    // Regression: Stanza's Japanese tokenizer sometimes merges adjacent CHAT
    // words into a single token with an embedded space (e.g. "ふ す").
    // The whitespace-stripping in lib.rs normalizes this before calling
    // retokenize_utterance. Here we verify that a token WITHOUT space works
    // correctly as the merged form.
    let chat_text = include_str!("../../../../test-fixtures/retok_jpn_fu_su.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, vec!["ふ", "す"]);

    // Simulate: Stanza merged "ふ" and "す" into one token "ふす"
    // (whitespace already stripped by lib.rs normalization)
    let stanza_tokens = vec!["ふす".to_string()];

    let mor_str = "n|ふす .";
    let gra_str = "1|0|ROOT 2|1|PUNCT";

    let (mors, term) = build_mor(mor_str);
    let gra_rels = build_gra(gra_str);

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    // The merged word "ふす" should appear exactly once on the main tier
    let main_line = output
        .lines()
        .find(|l| l.starts_with("*CHI:"))
        .unwrap_or("");
    assert!(
        main_line.contains("ふす"),
        "Merged token should appear on main tier: {main_line}"
    );
    // "ふす" should NOT be duplicated (N:1 merge dedup)
    let count = main_line.matches("ふす").count();
    assert_eq!(
        count, 1,
        "Merged token should appear exactly once, not duplicated: {main_line}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

#[test]
fn test_n_to_1_merge_no_duplication() {
    // Regression: when multiple original words map to the same Stanza token
    // (N:1 merge), only the first original word should emit the merged word.
    // Subsequent original words that map to the same token should be skipped.
    let chat_text = include_str!("../../../../test-fixtures/retok_gon_na_eat.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, vec!["gon", "na", "eat"]);

    // Stanza merged "gon" + "na" -> "gonna", "eat" unchanged
    let stanza_tokens = vec!["gonna".to_string(), "eat".to_string()];

    let mor_str = "aux|go&PRESP v|eat .";
    let gra_str = "1|2|AUX 2|0|ROOT 3|2|PUNCT";

    let (mors, term) = build_mor(mor_str);
    let gra_rels = build_gra(gra_str);

    let parser = TreeSitterParser::new().unwrap();
    retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    )
    .unwrap();

    let output = chat.to_chat_string();
    let main_line = output
        .lines()
        .find(|l| l.starts_with("*CHI:"))
        .unwrap_or("");

    // "gonna" should appear exactly once -- NOT duplicated
    let gonna_count = main_line.matches("gonna").count();
    assert_eq!(
        gonna_count, 1,
        "'gonna' should appear once on main tier (N:1 merge dedup): {main_line}"
    );
    // "eat" should still be present
    assert!(
        main_line.contains("eat"),
        "'eat' should be present: {main_line}"
    );
    assert!(output.contains("%mor:"), "Should have %mor tier: {output}");
}

/// CJK retokenize with retrace marker: `<下 次> [/]`.
///
/// Retraces are excluded from MOR domain word extraction. When retokenize
/// is active, the word count sent to Stanza (non-retraced only) must match
/// the MOR items returned. The retrace content in the AST should be
/// preserved but not counted for MOR alignment.
///
/// Regression test for MOST corpus failure:
/// "MOR item count (5) does not match alignable word count (6)"
/// Source: data/childes-other-data/Chinese/Cantonese/MOST/10002/40415b.cha
#[test]
fn test_cjk_retokenize_with_retrace() {
    let chat_text = include_str!("../../../../test-fixtures/retok_yue_retrace.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // MOR domain extraction should skip the retraced <下 次> [/]
    // Non-retraced words only
    let non_retrace_count = original_words.len();
    assert!(non_retrace_count > 0, "Should have non-retraced words");

    // Stanza tokens match non-retraced words (1:1, no merging for this test)
    let stanza_tokens: Vec<String> = original_words
        .iter()
        .map(|w| w.text.as_ref().to_string())
        .collect();

    // Build matching MOR/GRA items
    let mor_specs: Vec<&str> = (0..non_retrace_count).map(|_| "noun|x").collect();
    let mor_str = format!("{} .", mor_specs.join(" "));
    let gra_specs: Vec<String> = (1..=non_retrace_count)
        .map(|i| format!("{i}|0|ROOT"))
        .collect();
    let gra_str = gra_specs.join(" ");

    let (mors, term) = build_mor(&mor_str);
    let gra_rels = build_gra(&gra_str);

    let parser = TreeSitterParser::new().unwrap();
    let result = retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    );
    assert!(
        result.is_ok(),
        "Retokenize should handle retrace markers: {:?}",
        result.err()
    );

    let output = chat.to_chat_string();
    assert!(output.contains("[/]"), "Retrace preserved: {output}");
    assert!(output.contains("%mor:"), "Has %mor tier: {output}");
}

/// Exact reproduction of MOST corpus retrace bug with N:1 merges.
///
/// Debug dump showed:
/// - Extracted: 7 words ["呢", "度", "下", "次", "食", "飯", "啦"]
/// - Stanza:    5 tokens ["呢", "度", "下次", "食飯", "啦"]
/// - Two N:1 merges: 下+次→下次, 食+飯→食飯
/// - Bug: rebuild_content produces 6 alignable words instead of 5
///
/// The retrace <下 次> [/] in the AST means "下 次" is repeated.
/// MOR extraction skips the retrace, giving 7 non-retrace words.
/// PyCantonese segments 7→5 (merging 下+次 and 食+飯).
/// After retokenize, the AST should have 5 alignable words.
#[test]
fn test_cjk_retokenize_retrace_with_n1_merges() {
    let chat_text = include_str!("../../../../test-fixtures/retok_yue_retrace.cha");
    let mut chat = parse_chat(chat_text);
    let utt = get_utterance(&mut chat, 0);
    let original_words = extract_words(utt);

    // Debug dump confirmed: 7 extracted words
    let texts: Vec<&str> = original_words.iter().map(|w| w.text.as_ref()).collect();
    eprintln!("Extracted words: {:?} (count: {})", texts, texts.len());
    assert_eq!(texts.len(), 7, "Should extract 7 MOR-domain words");

    // Exact Stanza tokens from debug dump (after PyCantonese segmentation)
    let stanza_tokens = vec![
        "呢".to_string(),
        "度".to_string(),
        "下次".to_string(), // N:1 merge: 下+次
        "食飯".to_string(), // N:1 merge: 食+飯
        "啦".to_string(),
    ];

    // 5 MOR items matching 5 Stanza tokens
    let (mors, term) = build_mor("noun|呢 noun|度 noun|下次 verb|食飯 part|啦 .");
    let gra_rels = build_gra("1|5|CASE 2|5|NMOD 3|5|NMOD 4|5|COMPOUND 5|0|ROOT");

    eprintln!(
        "Stanza tokens: {:?} (count: {})",
        stanza_tokens,
        stanza_tokens.len()
    );
    eprintln!("MOR count: {}, GRA count: {}", mors.len(), gra_rels.len());

    // Debug: show the word-token mapping
    let mapping = mapping::build_word_token_mapping(&original_words, &stanza_tokens);
    for (i, word) in original_words.iter().enumerate() {
        let tokens = mapping.tokens_for_word(i);
        eprintln!("  word[{i}] '{}' → tokens {:?}", word.text.as_ref(), tokens);
    }

    let parser = TreeSitterParser::new().unwrap();
    let result = retokenize_utterance(
        &parser,
        utt,
        &original_words,
        &stanza_tokens,
        mors,
        term,
        gra_rels,
    );

    if let Err(ref e) = result {
        // Print the rewritten AST for debugging
        let output = chat.to_chat_string();
        eprintln!("ERROR: {e}");
        eprintln!("Rewritten AST:\n{output}");
    }

    assert!(
        result.is_ok(),
        "Retokenize with N:1 merges should succeed: {:?}",
        result.err()
    );

    let output = chat.to_chat_string();
    assert!(output.contains("[/]"), "Retrace preserved: {output}");
    assert!(
        output.contains("下次"),
        "Merged 下次 should appear: {output}"
    );
    assert!(
        output.contains("食飯"),
        "Merged 食飯 should appear: {output}"
    );
    assert!(output.contains("%mor:"), "Has %mor tier: {output}");
}
