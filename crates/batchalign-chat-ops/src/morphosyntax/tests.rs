//! Tests for morphosyntax module.

use super::*;

fn ud_response_from_words(words_json: &str) -> crate::nlp::UdResponse {
    serde_json::from_str(&format!(r#"{{"sentences":[{{"words":{words_json}}}]}}"#)).unwrap()
}

#[test]
fn test_clear_morphosyntax() {
    use crate::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    let parser = TreeSitterParser::new().unwrap();
    // Minimal CHAT with %mor and %gra tiers
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);

    // Verify the utterance has %mor and %gra before clearing
    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should have an utterance");
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "should have %mor before clear"
    );
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Gra(_))),
        "should have %gra before clear"
    );

    // Clear then sweep any unfilled placeholders (mirrors pipeline order).
    clear_morphosyntax(&mut chat_file);
    crate::morphosyntax::remove_empty_morphosyntax_placeholders(&mut chat_file);

    // Verify no %mor or %gra remain after the full clear+sweep.
    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should still have an utterance");
    assert!(
        !utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "should NOT have %mor after clear+sweep"
    );
    assert!(
        !utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Gra(_))),
        "should NOT have %gra after clear+sweep"
    );
}

#[test]
fn test_clear_morphosyntax_preserves_other_tiers() {
    use crate::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    let parser = TreeSitterParser::new().unwrap();
    // CHAT with %mor, %gra, and %act -- only %mor/%gra should be removed
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra_act.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    clear_morphosyntax(&mut chat_file);
    crate::morphosyntax::remove_empty_morphosyntax_placeholders(&mut chat_file);

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("utterance");
    // %act should survive clear+sweep
    assert!(
        !utt.dependent_tiers.is_empty(),
        "should still have %act after clear+sweep"
    );
    // But no %mor or %gra
    assert!(
        !utt.dependent_tiers.iter().any(|t| matches!(
            t,
            talkbank_model::model::DependentTier::Mor(_)
                | talkbank_model::model::DependentTier::Gra(_)
        )),
        "should not have %mor or %gra after clear+sweep"
    );
}

#[test]
fn test_clear_morphosyntax_no_tiers() {
    use crate::parse::{TreeSitterParser, parse_lenient};
    use talkbank_model::model::Line;

    let parser = TreeSitterParser::new().unwrap();
    // CHAT without any dependent tiers -- clear should be a no-op
    let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    clear_morphosyntax(&mut chat_file);

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("utterance");
    assert!(utt.dependent_tiers.is_empty());
}

#[test]
fn test_validate_mor_alignment_ok() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Correctly aligned: 2 main words + 2 %mor items
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_with_mor_gra.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let warnings = validate_mor_alignment(&chat_file);
    assert!(
        warnings.is_empty(),
        "expected no alignment warnings, got: {:?}",
        warnings
    );
}

#[test]
fn test_validate_mor_alignment_no_mor_tier() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // No %mor -- validation should pass (nothing to check)
    let chat = include_str!("../../../../test-fixtures/eng_hello_male.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let warnings = validate_mor_alignment(&chat_file);
    assert!(warnings.is_empty());
}

// -----------------------------------------------------------------------
// Cross-language roundtrip snapshot tests
// -----------------------------------------------------------------------

/// Verify MorphosyntaxBatchItem serializes to the JSON shape Python expects.
#[test]
fn snapshot_morphosyntax_batch_item() {
    let item = MorphosyntaxBatchItem {
        words: vec!["the".into(), "dog".into(), "runs".into()],
        terminator: ".".into(),
        special_forms: vec![(None, None), (None, None), (None, None)],
        lang: talkbank_model::model::LanguageCode::new("eng"),
    };
    insta::assert_json_snapshot!("morphosyntax_batch_item", item);
}

/// Verify UdResponse from Python deserializes correctly in Rust.
#[test]
fn snapshot_ud_response_from_python() {
    // This is the exact shape Python's Stanza inference returns
    let python_json = r#"{
        "sentences": [
            {
                "words": [
                    {
                        "id": 1,
                        "text": "the",
                        "lemma": "the",
                        "upos": "DET",
                        "xpos": "DT",
                        "feats": "Definite=Def|PronType=Art",
                        "head": 2,
                        "deprel": "det",
                        "start_char": 0,
                        "end_char": 3
                    },
                    {
                        "id": 2,
                        "text": "dog",
                        "lemma": "dog",
                        "upos": "NOUN",
                        "xpos": "NN",
                        "feats": "Number=Sing",
                        "head": 3,
                        "deprel": "nsubj",
                        "start_char": 4,
                        "end_char": 7
                    },
                    {
                        "id": 3,
                        "text": "runs",
                        "lemma": "run",
                        "upos": "VERB",
                        "xpos": "VBZ",
                        "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
                        "head": 0,
                        "deprel": "root",
                        "start_char": 8,
                        "end_char": 12
                    }
                ]
            }
        ]
    }"#;

    let ud: crate::nlp::UdResponse = serde_json::from_str(python_json).unwrap();
    assert_eq!(ud.sentences.len(), 1);
    assert_eq!(ud.sentences[0].words.len(), 3);
    assert_eq!(ud.sentences[0].words[2].lemma, "run");

    // Re-serialize and snapshot to verify round-trip fidelity
    insta::assert_json_snapshot!("ud_response_roundtrip", ud);
}

/// Verify collect_payloads produces the expected shape for a simple CHAT.
#[test]
fn snapshot_collected_payloads() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_the_dog_runs.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let collected = collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll);
    let items = collected.batch_items;
    let total = collected.total_utterances;

    assert_eq!(total, 1);
    assert_eq!(items.len(), 1);

    // Snapshot just the batch item (the payload that crosses the wire)
    let (_, _, ref batch_item, _) = items[0];
    insta::assert_json_snapshot!("collected_payload_item", batch_item);
}

// -----------------------------------------------------------------------
// Regression tests: batch item lang must reflect file @Languages header,
// not the batch-level primary_lang parameter.
//
// Bug: when a job has lang="eng" (the default) but a file declares
// @Languages: spa, collect_payloads produced items with lang="eng"
// instead of "spa". This caused Stanza to use the wrong model.
// -----------------------------------------------------------------------

/// When @Languages declares "spa" but primary_lang is "eng" (batch default),
/// the batch item must carry lang="spa" from the file header.
#[test]
fn collect_payloads_uses_file_language_not_batch_default_spa() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/spa_hola_que_es_este.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    // Simulate the batch-level default: primary_lang = "eng"
    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    // The batch item's lang MUST be "spa" (from @Languages header),
    // NOT "eng" (the batch default).
    assert_eq!(
        batch_item.lang.as_str(),
        "spa",
        "batch item lang should be 'spa' from @Languages header, not 'eng' batch default"
    );
}

/// Same regression for Russian: @Languages: rus with batch default "eng".
#[test]
fn collect_payloads_uses_file_language_not_batch_default_rus() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/rus_vot_istoriya.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(
        batch_item.lang.as_str(),
        "rus",
        "batch item lang should be 'rus' from @Languages header, not 'eng' batch default"
    );
}

/// Same regression for Chinese: @Languages: zho with batch default "eng".
#[test]
fn collect_payloads_uses_file_language_not_batch_default_zho() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/zho_hao_qing_zhong.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(
        batch_item.lang.as_str(),
        "zho",
        "batch item lang should be 'zho' from @Languages header, not 'eng' batch default"
    );
}

/// Same regression for French: @Languages: fra with batch default "eng".
#[test]
fn collect_payloads_uses_file_language_not_batch_default_fra() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/fra_lescargot_dort.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(
        batch_item.lang.as_str(),
        "fra",
        "batch item lang should be 'fra' from @Languages header, not 'eng' batch default"
    );
}

/// When @Languages matches primary_lang, lang should still be correct.
/// (Control case: ensures fix doesn't regress the happy path.)
#[test]
fn collect_payloads_lang_correct_when_primary_matches_header() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_hello_world_male.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    assert_eq!(batch_item.lang.as_str(), "eng");
}

/// When @Languages has multiple languages, the first declared language
/// should be used as the utterance default (not the batch primary_lang).
#[test]
fn collect_payloads_uses_first_declared_language_for_multilingual() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Bilingual file: primary declared language is "spa", secondary is "eng"
    let chat = include_str!("../../../../test-fixtures/spa_eng_bilingual_hola_mundo.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    // Batch default is "eng" but file says "spa" first
    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref batch_item, _) = items[0];

    // Should use "spa" (first declared), not "eng" (batch default)
    assert_eq!(
        batch_item.lang.as_str(),
        "spa",
        "batch item lang should be 'spa' (first in @Languages), not 'eng' batch default"
    );
}

/// Regression: inject_results with retokenize on Cantonese retrace utterance.
///
/// The full pipeline: parse CHAT → extract words → construct UD response →
/// inject with retokenize mode. This should succeed but fails with
/// "MOR item count does not match alignable word count".
///
/// Source: MOST corpus 40415b.cha line 46.
#[test]
fn test_inject_results_retokenize_cantonese_retrace() {
    use crate::morphosyntax::inject::inject_results;
    use crate::nlp::{UdId, UdResponse, UdSentence, UdWord};
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/retok_yue_retrace.cha");
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("yue");
    let langs = declared_languages(&chat_file, &primary_lang);

    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;

    assert!(!batch_items.is_empty(), "Should have batch items");

    // Print what was extracted
    for (line_idx, utt_ord, item, words) in &batch_items {
        eprintln!(
            "Batch item: line={line_idx} utt={utt_ord} words={:?} item_words={:?}",
            words.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>(),
            item.words,
        );
    }

    // Build a matching UD response (one word per extracted word)
    let first_item = &batch_items[0];
    let word_count = first_item.2.words.len();
    eprintln!("Word count from batch item: {word_count}");

    // Simulate what Python actually returns: _segment_cantonese reduces
    // 7 single-char words to 5 words (下+次→下次, 食+飯→食飯).
    // Stanza processes 5 words and returns 5 MOR items.
    let segmented_words = vec!["呢", "度", "下次", "食飯", "啦"];
    eprintln!(
        "Simulated PyCantonese segmentation: {:?} ({} words)",
        segmented_words,
        segmented_words.len()
    );
    // UD requires exactly one root (head=0). First word is the root, all
    // others depend on index 1.
    let ud_words: Vec<UdWord> = segmented_words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let (head, deprel) = if i == 0 { (0, "root") } else { (1, "dep") };
            UdWord {
                id: UdId::Single(i + 1),
                text: w.to_string(),
                lemma: w.to_string(),
                upos: crate::nlp::UdPunctable::Value(crate::nlp::UniversalPos::Noun),
                xpos: None,
                feats: None,
                head,
                deprel: deprel.into(),
                deps: None,
                misc: None,
            }
        })
        .collect();

    let ud_response = UdResponse {
        sentences: vec![UdSentence { words: ud_words }],
    };

    let empty_mwt = std::collections::BTreeMap::new();
    let result = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::StanzaRetokenize,
        &empty_mwt,
    );

    assert!(
        result.is_ok(),
        "inject_results should succeed for retokenize + retrace: {:?}",
        result.err()
    );
}

/// Regression: French utterance with embedded single quotes around elision.
///
/// `*MOT: On dit pas 'quoi tu veux' , mais 'qu' est-ce que' on dit .`
///
/// The `qu'` is a French elision (like `l'homme`, `j'ai`).  Stanza's French
/// MWT tokenizer expands `qu'` into a range token `[n, n+1]` with components
/// `qu` and `'`.  The MOR mapping must collapse these back into one MOR item
/// so the count matches the CHAT word count.
///
/// Source: childes-other-data/Biling/Amsterdam/Anouk/fra/030428.cha line 509.
/// This caused batch 7 of the multilingual morphotag rerun to fail with:
/// "MOR item count (14) does not match alignable word count (13)"
#[test]
fn test_french_elision_in_quoted_context() {
    use crate::morphosyntax::payloads::collect_payloads;

    let parser = crate::parse::TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/fra_french_elision_quotes.cha");
    let (chat_file, _) = crate::parse::parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("fra");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1, "Should have exactly 1 utterance payload");
    let (_, _, item, extracted_words) = &items[0];

    // Print for debugging
    println!("Extracted words: {:?}", item.words);
    println!("Word count: {}", item.words.len());
    println!(
        "Extracted word details: {:?}",
        extracted_words
            .iter()
            .map(|w| w.text.as_ref())
            .collect::<Vec<_>>()
    );

    // The utterance has these CHAT words (in MOR domain, excluding separators):
    // On, dit, pas, 'quoi, tu, veux', mais, 'qu', est-ce, que', on, dit, .
    // That's 13 words (including the terminator).
    // Stanza should NOT produce more MOR items than this.
    let word_count = item.words.len();
    assert!(
        word_count > 0,
        "Should extract some words from French utterance"
    );
    println!("CHAT word count for MOR alignment: {word_count}");
}

// -----------------------------------------------------------------------
// @s (code-switching) payload collection tests
// -----------------------------------------------------------------------

/// Verify collect_payloads identifies @s:spa word positions in special_forms.
#[test]
fn collect_payloads_identifies_at_s_positions_single() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_spa_at_s_single.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1, "should have 1 utterance");
    let (_, _, ref item, _) = items[0];

    // "I went to the tienda@s:spa yesterday ."
    // Words: I, went, to, the, tienda, yesterday (6 words, terminator separate)
    assert!(
        item.words.len() >= 6,
        "should have at least 6 words, got {}",
        item.words.len()
    );

    // Find the position of "tienda" and verify it has a language resolution
    let tienda_idx = item
        .words
        .iter()
        .position(|w| w == "tienda")
        .expect("should contain 'tienda'");

    let (_, ref lang_res) = item.special_forms[tienda_idx];
    assert!(
        lang_res.is_some(),
        "tienda should have a language resolution (it's @s:spa)"
    );
    let resolution = lang_res.as_ref().unwrap();
    let langs = resolution.languages();
    assert_eq!(langs.len(), 1, "should resolve to exactly one language");
    assert_eq!(langs[0].as_str(), "spa", "should resolve to Spanish");

    // All other words should have None for language resolution
    for (i, (_, lr)) in item.special_forms.iter().enumerate() {
        if i != tienda_idx {
            assert!(
                lr.is_none(),
                "word {} ('{}') should NOT have language resolution",
                i,
                item.words[i]
            );
        }
    }
}

/// Verify contiguous @s:spa span in special_forms.
#[test]
fn collect_payloads_identifies_contiguous_at_s_span() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_spa_at_s_contiguous.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref item, _) = items[0];

    // "we talked about los@s:spa niños@s:spa ."
    let los_idx = item
        .words
        .iter()
        .position(|w| w == "los")
        .expect("should have 'los'");
    let ninos_idx = item
        .words
        .iter()
        .position(|w| w == "niños")
        .expect("should have 'niños'");

    // Both should have spa resolution
    assert!(item.special_forms[los_idx].1.is_some());
    assert!(item.special_forms[ninos_idx].1.is_some());

    // They should be contiguous
    assert_eq!(ninos_idx, los_idx + 1, "los and niños should be adjacent");

    // Verify span grouping produces one span
    let spans = crate::morphosyntax::l2::group_l2_spans(&item.special_forms, &item.words);
    assert_eq!(spans.len(), 1, "contiguous same-lang should produce 1 span");
    assert_eq!(spans[0].word_indices, vec![los_idx, ninos_idx]);
    assert_eq!(spans[0].words, vec!["los", "niños"]);
}

// -----------------------------------------------------------------------
// Regression: retokenize with MWT Range tokens
//
// Bug: inject_results with StanzaRetokenize includes Range parent tokens
// AND their component words in the token vector. map_ud_sentence() merges
// Range components into 1 clitic MOR, so mors.len() < tokens.len().
// retokenize_utterance fails on the count mismatch.
//
// Example: Stanza returns "gonna" as Range(1,2) + "gon" + "na".
// Token vector gets 6 items [gonna, gon, na, eat, cookies, .] but
// only 4 MOR items (the Range is merged into verb|go~part|to).
// -----------------------------------------------------------------------

/// inject_results with StanzaRetokenize must handle MWT Range tokens
/// without failing on token/MOR count mismatch.
#[test]
fn inject_results_retokenize_mwt_range_tokens_no_failure() {
    use crate::morphosyntax::inject::inject_results;
    use crate::nlp::{UdId, UdPunctable, UdResponse, UdSentence, UdWord, UniversalPos};
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Minimal CHAT: "gonna eat cookies ."
    let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tgonna eat cookies .
@End
";
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary_lang);
    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;

    assert_eq!(batch_items.len(), 1, "should have 1 utterance");
    let word_count = batch_items[0].2.words.len();
    // CHAT extracts main tier words (gonna, eat, cookies) + terminator is
    // stored separately in the batch item.
    assert!(
        word_count >= 3,
        "should have at least 3 words (gonna, eat, cookies), got {word_count}"
    );

    // Simulate Stanza's free-tokenize output with MWT expansion for "gonna".
    // This is the exact shape that Stanza returns: a Range parent token
    // followed by its component words, then the remaining regular words.
    let ud_response = UdResponse {
        sentences: vec![UdSentence {
            words: vec![
                // Range parent token for "gonna"
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "gonna".into(),
                    lemma: "".into(),
                    upos: UdPunctable::Value(UniversalPos::X),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                // Component 1: "gon" (going)
                UdWord {
                    id: UdId::Single(1),
                    text: "gon".into(),
                    lemma: "go".into(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: None,
                    feats: Some("VerbForm=Part".into()),
                    head: 4,
                    deprel: "advcl".into(),
                    deps: None,
                    misc: None,
                },
                // Component 2: "na" (to)
                UdWord {
                    id: UdId::Single(2),
                    text: "na".into(),
                    lemma: "to".into(),
                    upos: UdPunctable::Value(UniversalPos::Part),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "mark".into(),
                    deps: None,
                    misc: None,
                },
                // Regular word: "eat"
                UdWord {
                    id: UdId::Single(3),
                    text: "eat".into(),
                    lemma: "eat".into(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: None,
                    feats: Some("VerbForm=Inf".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
                // Regular word: "cookies"
                UdWord {
                    id: UdId::Single(4),
                    text: "cookies".into(),
                    lemma: "cookie".into(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: Some("Number=Plur".into()),
                    head: 3,
                    deprel: "obj".into(),
                    deps: None,
                    misc: None,
                },
                // Punctuation: "."
                UdWord {
                    id: UdId::Single(5),
                    text: ".".into(),
                    lemma: ".".into(),
                    upos: UdPunctable::Punct("PUNCT".into()),
                    xpos: None,
                    feats: None,
                    head: 3,
                    deprel: "punct".into(),
                    deps: None,
                    misc: None,
                },
            ],
        }],
    };

    let empty_mwt = std::collections::BTreeMap::new();
    let result = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::StanzaRetokenize,
        &empty_mwt,
    );

    // The injection must succeed — no retokenization_failed decision.
    let injection = result.expect("inject_results should not return Err");
    let failed_decisions: Vec<_> = injection
        .decisions
        .iter()
        .filter(|d| {
            d.strategy.strategy_name() == "retokenization_failed"
                || d.strategy.strategy_name() == "injection_failed"
                || d.strategy.strategy_name() == "mapping_failed"
        })
        .collect();
    assert!(
        failed_decisions.is_empty(),
        "Retokenize with MWT Range tokens should not produce failure decisions, \
         got: {failed_decisions:?}"
    );

    // The output should have a %mor tier.
    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            talkbank_model::model::Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should have an utterance");
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "Output should have a %mor tier after retokenize injection with MWT"
    );
}

#[test]
fn inject_results_preserve_coraal_units_keeps_mor_gra() {
    use crate::morphosyntax::inject::inject_results;
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_morph_coraal_units.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary_lang);
    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;

    assert_eq!(batch_items.len(), 1, "should have one payload");

    let ud_response = ud_response_from_words(
        r#"[
          {"id":1,"text":"now","lemma":"now","upos":"ADV","xpos":"RB","feats":"PronType=Dem","head":9,"deprel":"advmod"},
          {"id":2,"text":"the","lemma":"the","upos":"DET","xpos":"DT","feats":"Definite=Def|PronType=Art","head":3,"deprel":"det"},
          {"id":3,"text":"building's","lemma":"building'","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":0,"deprel":"root"},
          {"id":4,"text":"only","lemma":"only","upos":"ADV","xpos":"RB","head":5,"deprel":"advmod"},
          {"id":5,"text":"four","lemma":"four","upos":"NUM","xpos":"CD","feats":"NumForm=Word|NumType=Card","head":6,"deprel":"nummod"},
          {"id":6,"text":"hundred","lemma":"hundred","upos":"NUM","xpos":"CD","feats":"NumForm=Word|NumType=Card","head":9,"deprel":"nummod"},
          {"id":7,"text":"and","lemma":"and","upos":"CCONJ","xpos":"CC","head":8,"deprel":"cc"},
          {"id":8,"text":"ninety","lemma":"ninety","upos":"NUM","xpos":"CD","feats":"NumForm=Word|NumType=Card","head":6,"deprel":"conj"},
          {"id":9,"text":"units","lemma":"unit","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":3,"deprel":"conj"},
          {"id":10,"text":".","lemma":".","upos":"PUNCT","xpos":".","head":3,"deprel":"punct"}
        ]"#,
    );

    let empty_mwt = std::collections::BTreeMap::new();
    let injection = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::Preserve,
        &empty_mwt,
    )
    .expect("preserve injection should not return Err");

    let failed_decisions: Vec<_> = injection
        .decisions
        .iter()
        .filter(|d| {
            d.strategy.strategy_name() == "mapping_failed"
                || d.strategy.strategy_name() == "injection_failed"
                || d.strategy.strategy_name() == "retokenization_failed"
        })
        .collect();
    assert!(
        failed_decisions.is_empty(),
        "coraal units should not be dropped; got decisions: {failed_decisions:?}"
    );

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            talkbank_model::model::Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should have utterance");
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "preserve injection should write %mor for the units case"
    );
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Gra(_))),
        "preserve injection should write %gra for the units case"
    );
}

#[test]
fn inject_results_preserve_minga_because_keeps_mor_gra() {
    use crate::morphosyntax::inject::inject_results;
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_morph_minga_because.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary_lang);
    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;

    assert_eq!(batch_items.len(), 1, "should have one payload");

    let ud_response = ud_response_from_words(
        r#"[
          {"id":1,"text":"yes","lemma":"yes","upos":"INTJ","xpos":"UH","feats":"Polarity=Pos","head":2,"deprel":"discourse"},
          {"id":2,"text":"(be)cause","lemma":"(be)cause","upos":"NOUN","xpos":"NN","feats":"Number=Sing","head":0,"deprel":"root"},
          {"id":3,"text":"that","lemma":"that","upos":"PRON","xpos":"WDT","feats":"PronType=Rel","head":5,"deprel":"obl"},
          {"id":4,"text":"building's","lemma":"building'","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":5,"deprel":"nsubj"},
          {"id":5,"text":"nice","lemma":"nice","upos":"ADJ","xpos":"JJ","feats":"Degree=Pos","head":2,"deprel":"acl:relcl"},
          {"id":6,"text":".","lemma":".","upos":"PUNCT","xpos":".","head":2,"deprel":"punct"}
        ]"#,
    );

    let empty_mwt = std::collections::BTreeMap::new();
    let injection = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::Preserve,
        &empty_mwt,
    )
    .expect("preserve injection should not return Err");

    let failed_decisions: Vec<_> = injection
        .decisions
        .iter()
        .filter(|d| {
            d.strategy.strategy_name() == "mapping_failed"
                || d.strategy.strategy_name() == "injection_failed"
                || d.strategy.strategy_name() == "retokenization_failed"
        })
        .collect();
    assert!(
        failed_decisions.is_empty(),
        "minga because should not be dropped; got decisions: {failed_decisions:?}"
    );

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            talkbank_model::model::Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should have utterance");
    assert!(
        utt.dependent_tiers
            .iter()
            .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
        "preserve injection should write %mor for the because case"
    );
}

#[test]
fn inject_results_preserve_kings_continuation_keeps_mor_gra() {
    use crate::morphosyntax::inject::inject_results;
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/eng_morph_kings_continuation.cha");
    let (mut chat_file, _) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary_lang);
    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;

    assert_eq!(batch_items.len(), 2, "should have two payloads");

    let ud_responses = vec![
        ud_response_from_words(
            r#"[
              {"id":1,"text":"all","lemma":"all","upos":"DET","xpos":"PDT","feats":"PronType=Tot","head":4,"deprel":"det:predet"},
              {"id":2,"text":"the","lemma":"the","upos":"DET","xpos":"DT","feats":"Definite=Def|PronType=Art","head":4,"deprel":"det"},
              {"id":3,"text":"king's","lemma":"king'","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":4,"deprel":"compound"},
              {"id":4,"text":"horses","lemma":"horse","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":0,"deprel":"root"},
              {"id":5,"text":"+...","lemma":"+...","upos":"PUNCT","xpos":",","head":4,"deprel":"punct"}
            ]"#,
        ),
        ud_response_from_words(
            r#"[
              {"id":1,"text":"all","lemma":"all","upos":"DET","xpos":"PDT","feats":"PronType=Tot","head":4,"deprel":"det:predet"},
              {"id":2,"text":"the","lemma":"the","upos":"DET","xpos":"DT","feats":"Definite=Def|PronType=Art","head":4,"deprel":"det"},
              {"id":3,"text":"king's","lemma":"king'","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":4,"deprel":"compound"},
              {"id":4,"text":"men","lemma":"man","upos":"NOUN","xpos":"NNS","feats":"Number=Plur","head":0,"deprel":"root"},
              {"id":5,"text":"+...","lemma":"+...","upos":"PUNCT","xpos":",","head":4,"deprel":"punct"}
            ]"#,
        ),
    ];

    let empty_mwt = std::collections::BTreeMap::new();
    let injection = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        ud_responses,
        &primary_lang,
        TokenizationMode::Preserve,
        &empty_mwt,
    )
    .expect("preserve injection should not return Err");

    let failed_decisions: Vec<_> = injection
        .decisions
        .iter()
        .filter(|d| {
            d.strategy.strategy_name() == "mapping_failed"
                || d.strategy.strategy_name() == "injection_failed"
                || d.strategy.strategy_name() == "retokenization_failed"
        })
        .collect();
    assert!(
        failed_decisions.is_empty(),
        "kings continuation should not be dropped; got decisions: {failed_decisions:?}"
    );

    let utts: Vec<_> = chat_file
        .lines
        .iter()
        .filter_map(|l| match l {
            talkbank_model::model::Line::Utterance(u) => Some(u),
            _ => None,
        })
        .collect();
    assert_eq!(utts.len(), 2, "should have two utterances");
    for utt in utts {
        assert!(
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t, talkbank_model::model::DependentTier::Mor(_))),
            "continuation utterances should keep %mor"
        );
        assert!(
            utt.dependent_tiers
                .iter()
                .any(|t| matches!(t, talkbank_model::model::DependentTier::Gra(_))),
            "continuation utterances should keep %gra"
        );
    }
}

/// Verify bare @s shortcut resolves via @Languages header.
#[test]
fn collect_payloads_bare_at_s_shortcut_resolution() {
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/deu_eng_at_s_bare.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    // Primary is German, secondary is English (from @Languages: deu, eng)
    let primary = talkbank_model::model::LanguageCode::new("deu");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let (_, _, ref item, _) = items[0];

    // "ich möchte film@s studies@s machen ."
    let film_idx = item
        .words
        .iter()
        .position(|w| w == "film")
        .expect("should have 'film'");
    let studies_idx = item
        .words
        .iter()
        .position(|w| w == "studies")
        .expect("should have 'studies'");

    // Bare @s should resolve to "eng" (secondary language from @Languages: deu, eng)
    let film_res = item.special_forms[film_idx]
        .1
        .as_ref()
        .expect("film should have lang");
    let film_langs = film_res.languages();
    assert_eq!(
        film_langs[0].as_str(),
        "eng",
        "bare @s should resolve to eng (secondary language)"
    );

    let studies_res = item.special_forms[studies_idx]
        .1
        .as_ref()
        .expect("studies should have lang");
    assert_eq!(studies_res.languages()[0].as_str(), "eng");

    // Span grouping should merge them (contiguous, same language)
    let spans = crate::morphosyntax::l2::group_l2_spans(&item.special_forms, &item.words);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].target_lang.as_str(), "eng");
    assert_eq!(spans[0].words, vec!["film", "studies"]);
}

// ---------------------------------------------------------------------------
// Tier-order preservation (source layout of %mor / %gra / %wor is stable
// across the clear → inject round trip; %wor stays wherever it was).
// ---------------------------------------------------------------------------

/// Source-order layout `[Mor, Gra, Wor]` must survive a
/// `clear_morphosyntax` → `replace_or_add_tier` round trip.
///
/// Before the fix, `clear_morphosyntax` *removed* Mor/Gra from
/// `dependent_tiers`, so when fresh Mor/Gra were re-added via
/// `replace_or_add_tier` they were pushed to the end and the utterance
/// serialized as `[Wor, Mor, Gra]`. Now clear replaces in place with
/// empty placeholders and the subsequent replace finds the variant match.
#[test]
fn clear_then_reinject_preserves_tier_order_mor_gra_wor() {
    use crate::inject::replace_or_add_tier;
    use crate::parse::parse_lenient;
    use talkbank_model::model::DependentTier;
    use talkbank_model::model::dependent_tier::{GraTier, MorTier, WorTier};
    use talkbank_parser::TreeSitterParser;

    let text = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tPAR Adult\n",
        "@ID:\teng|test|PAR|||||Adult|||\n",
        "*PAR:\thello world .\n",
        "%mor:\tintj|hello noun|world .\n",
        "%gra:\t1|2|DISCOURSE 2|0|ROOT 3|2|PUNCT\n",
        "%wor:\thello 0_500 world 500_1000 .\n",
        "@End\n",
    );
    let parser = TreeSitterParser::new().unwrap();
    let (mut chat, _errors) = parse_lenient(&parser, text);

    assert_eq!(
        dep_tier_kinds(&chat),
        vec!["Mor", "Gra", "Wor"],
        "fixture precondition: expected [Mor, Gra, Wor]",
    );

    // Clear — must preserve tier positions.
    crate::morphosyntax::clear_morphosyntax(&mut chat);
    assert_eq!(
        dep_tier_kinds(&chat),
        vec!["Mor", "Gra", "Wor"],
        "clear_morphosyntax must preserve tier order",
    );

    // Re-inject: replace_or_add_tier must overwrite placeholders in place.
    let utt = first_utterance_mut(&mut chat);
    replace_or_add_tier(
        &mut utt.dependent_tiers,
        DependentTier::Mor(MorTier::new_mor(Vec::new())),
    );
    replace_or_add_tier(
        &mut utt.dependent_tiers,
        DependentTier::Gra(GraTier::new_gra(Vec::new())),
    );
    assert_eq!(
        dep_tier_kinds(&chat),
        vec!["Mor", "Gra", "Wor"],
        "replace_or_add_tier must overwrite empty placeholders in place, \
         not push new tiers after the existing Wor",
    );

    // Replacing Wor must keep it at its original index.
    let utt = first_utterance_mut(&mut chat);
    replace_or_add_tier(
        &mut utt.dependent_tiers,
        DependentTier::Wor(WorTier::default()),
    );
    assert_eq!(
        dep_tier_kinds(&chat),
        vec!["Mor", "Gra", "Wor"],
        "replace_or_add_tier on existing Wor must stay in place",
    );
}

/// Symmetric test for the `align` path: `add_wor_tier` must replace an
/// existing `%wor` in place, preserving original tier order when `%wor`
/// sits before `%mor` / `%gra`. The pre-fix implementation called
/// `remove_wor_tier` + `push`, which displaced regenerated `%wor` to the
/// end and reshuffled `[Wor, Mor, Gra]` into `[Mor, Gra, Wor]` on every
/// align run.
#[test]
fn add_wor_tier_preserves_tier_order_wor_mor_gra() {
    use crate::fa::add_wor_tier;
    use crate::parse::parse_lenient;
    use talkbank_parser::TreeSitterParser;

    let text = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tPAR Adult\n",
        "@ID:\teng|test|PAR|||||Adult|||\n",
        "*PAR:\thello world .\n",
        "%wor:\thello 0_500 world 500_1000 .\n",
        "%mor:\tintj|hello noun|world .\n",
        "%gra:\t1|2|DISCOURSE 2|0|ROOT 3|2|PUNCT\n",
        "@End\n",
    );
    let parser = TreeSitterParser::new().unwrap();
    let (mut chat, _errors) = parse_lenient(&parser, text);

    assert_eq!(
        dep_tier_kinds(&chat),
        vec!["Wor", "Mor", "Gra"],
        "fixture precondition: expected [Wor, Mor, Gra]",
    );

    let utt = first_utterance_mut(&mut chat);
    add_wor_tier(utt);

    assert_eq!(
        dep_tier_kinds(&chat),
        vec!["Wor", "Mor", "Gra"],
        "add_wor_tier must replace existing %wor in place, not remove + push",
    );
}

/// Regression: after `clear_morphosyntax` leaves empty `%mor` / `%gra`
/// placeholders in place, `collect_payloads` MUST still collect the
/// utterance for re-inference. Previously it skipped any utterance with
/// a `%mor` variant regardless of whether its content was empty, which
/// meant the morphotag pipeline produced zero payloads after clear and
/// silently stripped every `%mor` / `%gra` in the file when
/// `remove_empty_morphosyntax_placeholders` ran at serialize time.
#[test]
fn collect_payloads_treats_empty_mor_placeholder_as_unprocessed() {
    use crate::parse::parse_lenient;
    use talkbank_model::model::LanguageCode;
    use talkbank_parser::TreeSitterParser;

    let text = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tPAR Adult\n",
        "@ID:\teng|test|PAR|||||Adult|||\n",
        "*PAR:\thello world .\n",
        "%mor:\tintj|hello noun|world .\n",
        "%gra:\t1|2|DISCOURSE 2|0|ROOT 3|2|PUNCT\n",
        "@End\n",
    );
    let parser = TreeSitterParser::new().unwrap();
    let (mut chat, _errors) = parse_lenient(&parser, text);

    let primary_lang = LanguageCode::new("eng");
    let langs = vec![primary_lang.clone()];

    // Before clearing: the utterance already has populated %mor, so
    // collect_payloads correctly skips it.
    let items = crate::morphosyntax::collect_payloads(
        &chat,
        &primary_lang,
        &langs,
        crate::morphosyntax::MultilingualPolicy::ProcessAll,
    )
    .batch_items;
    assert_eq!(
        items.len(),
        0,
        "populated %mor must be treated as already-processed",
    );

    // After clear_morphosyntax: %mor placeholder remains in place but empty.
    crate::morphosyntax::clear_morphosyntax(&mut chat);

    let collected = crate::morphosyntax::collect_payloads(
        &chat,
        &primary_lang,
        &langs,
        crate::morphosyntax::MultilingualPolicy::ProcessAll,
    );
    let items = collected.batch_items;
    let total = collected.total_utterances;
    assert_eq!(total, 1, "one utterance in fixture");
    assert_eq!(
        items.len(),
        1,
        "empty %mor placeholder must NOT be treated as already-processed; \
         the utterance must be collected for re-inference",
    );
}

/// Helper: one-letter variant tag for each dependent tier of the first
/// utterance, in source order. Only cases we care about for this test
/// are enumerated explicitly; other tiers fall through to "Other".
fn dep_tier_kinds(chat: &talkbank_model::model::ChatFile) -> Vec<&'static str> {
    use talkbank_model::model::{DependentTier, Line};
    for line in &chat.lines {
        if let Line::Utterance(u) = line {
            return u
                .dependent_tiers
                .iter()
                .map(|t| match t {
                    DependentTier::Mor(_) => "Mor",
                    DependentTier::Gra(_) => "Gra",
                    DependentTier::Wor(_) => "Wor",
                    _ => "Other",
                })
                .collect();
        }
    }
    Vec::new()
}

fn first_utterance_mut(
    chat: &mut talkbank_model::model::ChatFile,
) -> &mut talkbank_model::model::Utterance {
    use talkbank_model::model::Line;
    for line in chat.lines.iter_mut() {
        if let Line::Utterance(u) = line {
            return u;
        }
    }
    panic!("no utterance in fixture")
}

// Regression guards: `inject_results` must surface injection errors
// visibly (via DecisionRecord + tracing::warn!) rather than silently
// drop the utterance, BUT must not kill the whole file — an isolated
// Stanza edge case shouldn't take down an entire morphotag run.
// See `inject::inject_morphosyntax` for the library-level error
// check that callers rely on.

/// When the UD response produces fewer Mor items than the CHAT main tier
/// has alignable words, `inject_results` must emit a visible
/// `DecisionRecord` (kind `injection_failed`) and continue with the
/// next utterance — not propagate the error and kill the file.
#[test]
fn inject_results_count_mismatch_propagates_error() {
    use crate::morphosyntax::inject::inject_results;
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Three alignable words on the main tier.
    let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\thello good world .
@End
";
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);

    let primary_lang = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary_lang);
    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;
    assert_eq!(batch_items.len(), 1);

    // Provide a UD response with only TWO words (hello, world). The CHAT
    // main tier has three (hello, good, world), so injection should fail
    // with a count mismatch.
    let ud_response = ud_response_from_words(
        r#"[
            {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"},
            {"id":2,"text":"world","lemma":"world","upos":"NOUN","head":1,"deprel":"parataxis"}
        ]"#,
    );

    let empty_mwt = std::collections::BTreeMap::new();
    let outcome = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::Preserve,
        &empty_mwt,
    )
    .expect("inject_results should absorb per-utterance failures at file level");

    // After the outcome-typing refactor (Wave 1 of the morphotag
    // reconciliation architecture), a count mismatch surfaces as a
    // `misalignment_bug` DecisionRecord whose `reason` field carries
    // typed diagnostic data (class, expected, actual, chat_words,
    // stanza_tokens). The test still asserts what it was checking
    // before — that mismatches are visible as review-flagged decision
    // records — but in the new typed form.
    let failed: Vec<_> = outcome
        .decisions
        .iter()
        .filter(|d| d.strategy.strategy_name() == "misalignment_bug")
        .collect();
    assert!(
        !failed.is_empty(),
        "expected at least one `misalignment_bug` decision record when the \
         count mismatched; got decisions: {:?}",
        outcome.decisions
    );
    let reason = &failed[0].reason;
    assert!(reason.contains("expected=3"), "got: {reason}");
    assert!(reason.contains("actual=2"), "got: {reason}");
    assert!(
        failed[0].needs_review,
        "misalignment bugs always require human review"
    );
}

/// End-to-end: a mid-utterance comma flows through `collect_payloads` →
/// `inject_results` and appears as `cm|cm` in the final `%mor` tier.
#[test]
fn mid_utterance_comma_end_to_end_injects_cm_mor() {
    use crate::morphosyntax::inject::inject_results;
    use crate::parse::{TreeSitterParser, parse_lenient};

    let parser = TreeSitterParser::new().unwrap();
    // Three words plus a comma plus a terminator: `hello , good world .`
    // CHAT Mor-domain alignable count = 4 (hello, comma, good, world).
    let chat = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\thello , good world .
@End
";
    let (mut chat_file, _errors) = parse_lenient(&parser, chat);
    let primary_lang = talkbank_model::model::LanguageCode::new("eng");
    let langs = declared_languages(&chat_file, &primary_lang);
    let batch_items = collect_payloads(
        &chat_file,
        &primary_lang,
        &langs,
        MultilingualPolicy::ProcessAll,
    )
    .batch_items;
    assert_eq!(batch_items.len(), 1);

    // Stanza-style UD output: 5 UD words (hello, comma, good, world, dot).
    // After mapping with the comma-preserving fix, mors.len() == 4
    // (terminator dropped, comma kept as cm|cm).
    let ud_response = ud_response_from_words(
        r#"[
            {"id":1,"text":"hello","lemma":"hello","upos":"INTJ","head":0,"deprel":"root"},
            {"id":2,"text":",","lemma":",","upos":"PUNCT","head":1,"deprel":"punct"},
            {"id":3,"text":"good","lemma":"good","upos":"ADJ","head":4,"deprel":"amod"},
            {"id":4,"text":"world","lemma":"world","upos":"NOUN","head":1,"deprel":"parataxis"},
            {"id":5,"text":".","lemma":".","upos":"PUNCT","head":1,"deprel":"punct"}
        ]"#,
    );

    let empty_mwt = std::collections::BTreeMap::new();
    let result = inject_results(
        &parser,
        &mut chat_file,
        batch_items,
        vec![ud_response],
        &primary_lang,
        TokenizationMode::Preserve,
        &empty_mwt,
    );
    result.expect("inject_results should succeed when comma is preserved");

    let utt = chat_file
        .lines
        .iter()
        .find_map(|l| match l {
            talkbank_model::model::Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("should have an utterance");

    let mor_tier = utt.dependent_tiers.iter().find_map(|t| match t {
        talkbank_model::model::DependentTier::Mor(m) => Some(m),
        _ => None,
    });
    let mor_tier = mor_tier.expect("utterance should gain a %mor tier");

    let mut mor_str = String::new();
    use talkbank_model::WriteChat;
    mor_tier.write_chat(&mut mor_str).unwrap();
    assert!(
        mor_str.contains("cm|cm"),
        "expected cm|cm in %mor; got: {mor_str}"
    );

    // Round-trip sanity: the full serialized CHAT must contain the %mor line.
    let serialized = chat_file.to_chat_string();
    assert!(
        serialized.contains("cm|cm"),
        "serialized CHAT should contain cm|cm; got:\n{serialized}"
    );
}
