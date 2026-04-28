//! Compatibility re-exports for canonical coreference helpers now defined in
//! `talkbank-transform`.

pub use talkbank_transform::coref::{
    ChainRef, CorefAnnotation, CorefBatchItem, CorefOutcome, CorefOutcomeKind,
    CorefPayloadCollection, CorefRawAnnotation, CorefRawResponse, CorefResponse,
    apply_coref_results, apply_coref_results_with_outcomes, build_bracket_annotation, clear_coref,
    collect_coref_payloads, inject_coref, raw_to_bracket_response,
};

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use talkbank_model::SpeakerCode;
#[cfg(test)]
use talkbank_model::model::{ChatFile, Line};

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::WriteChat;
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).unwrap()
    }

    fn get_utterance_mut(chat: &mut ChatFile, idx: usize) -> &mut talkbank_model::model::Utterance {
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

    #[test]
    fn test_collect_coref_payloads() {
        let chat_text = include_str!("../../../test-fixtures/eng_three_sentences_coref.cha");
        let chat = parse_chat(chat_text);
        let collected = collect_coref_payloads(&chat);
        let item = collected.batch_item;
        let line_indices = collected.line_indices;

        assert_eq!(item.sentences.len(), 3);
        assert_eq!(line_indices.len(), 3);
        assert_eq!(item.sentences[0], vec!["the", "dog", "ran"]);
        assert_eq!(item.sentences[1], vec!["it", "was", "fast"]);
        assert_eq!(item.sentences[2], vec!["the", "cat", "slept"]);
    }

    #[test]
    fn test_inject_coref() {
        let chat_text = include_str!("../../../test-fixtures/eng_the_dog_ran.cha");
        let mut chat = parse_chat(chat_text);
        let utt = get_utterance_mut(&mut chat, 0);
        inject_coref(utt, "(0, -, 0)").unwrap();

        let output = chat.to_chat_string();
        assert!(output.contains("%xcoref:\t(0, -, 0)"), "Output: {output}");
    }

    #[test]
    fn test_inject_coref_replaces_existing() {
        let chat_text = include_str!("../../../test-fixtures/eng_the_dog_ran_with_xcoref.cha");
        let mut chat = parse_chat(chat_text);

        let output_before = chat.to_chat_string();
        assert!(
            output_before.contains("old annotation"),
            "Before: {output_before}"
        );

        let utt = get_utterance_mut(&mut chat, 0);
        inject_coref(utt, "(1, -, 1)").unwrap();

        let output = chat.to_chat_string();
        assert!(output.contains("(1, -, 1)"), "After: {output}");
        assert!(
            !output.contains("old annotation"),
            "Old should be gone: {output}"
        );
    }

    #[test]
    fn test_apply_coref_results_sparse() {
        let chat_text = include_str!("../../../test-fixtures/eng_three_sentences_coref.cha");
        let mut chat = parse_chat(chat_text);

        let line_indices = collect_coref_payloads(&chat).line_indices;
        assert_eq!(line_indices.len(), 3);

        // Only annotate utterances 0 and 2 (sparse)
        let mut results = HashMap::new();
        results.insert(line_indices[0], "(0, -, 0)".to_string());
        results.insert(line_indices[2], "(1, -, 1)".to_string());

        apply_coref_results(&mut chat, &results);

        let output = chat.to_chat_string();
        assert!(output.contains("%xcoref:\t(0, -, 0)"), "Output: {output}");
        assert!(output.contains("%xcoref:\t(1, -, 1)"), "Output: {output}");

        // Utterance 1 should NOT have %xcoref
        let lines: Vec<&str> = output.lines().collect();
        let utt1_line = lines
            .iter()
            .position(|l| l.contains("it was fast"))
            .unwrap();
        // Check that the next line is not %xcoref
        if utt1_line + 1 < lines.len() {
            assert!(
                !lines[utt1_line + 1].starts_with("%xcoref"),
                "Utterance 1 should not have xcoref: {}",
                lines[utt1_line + 1]
            );
        }
    }

    #[test]
    fn test_clear_coref() {
        let chat_text = include_str!("../../../test-fixtures/eng_the_dog_ran.cha");
        let mut chat = parse_chat(chat_text);

        // Inject %xcoref
        let utt = get_utterance_mut(&mut chat, 0);
        inject_coref(utt, "(0, -, 0)").unwrap();
        let output = chat.to_chat_string();
        assert!(output.contains("%xcoref"), "Should have xcoref: {output}");

        // Clear
        clear_coref(&mut chat);
        let output = chat.to_chat_string();
        assert!(!output.contains("%xcoref"), "Should be gone: {output}");
    }

    #[test]
    fn test_clear_coref_preserves_other_tiers() {
        let chat_text = include_str!("../../../test-fixtures/eng_the_dog_ran_with_xtra.cha");
        let mut chat = parse_chat(chat_text);

        // Inject %xcoref alongside existing %xtra
        let utt = get_utterance_mut(&mut chat, 0);
        inject_coref(utt, "(0, -, 0)").unwrap();
        let output = chat.to_chat_string();
        assert!(output.contains("%xcoref"), "Should have xcoref: {output}");
        assert!(output.contains("%xtra"), "Should have xtra: {output}");

        // Clear only %xcoref
        clear_coref(&mut chat);
        let output = chat.to_chat_string();
        assert!(!output.contains("%xcoref"), "xcoref gone: {output}");
        assert!(output.contains("%xtra"), "xtra preserved: {output}");
    }

    #[test]
    fn test_inject_coref_empty_is_noop() {
        let chat_text = include_str!("../../../test-fixtures/eng_the_dog_ran.cha");
        let mut chat = parse_chat(chat_text);
        let output_before = chat.to_chat_string();

        let utt = get_utterance_mut(&mut chat, 0);
        inject_coref(utt, "").unwrap();

        let output_after = chat.to_chat_string();
        assert_eq!(output_before, output_after);
    }

    // -----------------------------------------------------------------------
    // Snapshot tests
    // -----------------------------------------------------------------------

    #[test]
    fn snapshot_coref_batch_item() {
        let item = CorefBatchItem {
            sentences: vec![
                vec!["the".into(), "dog".into(), "ran".into()],
                vec!["it".into(), "was".into(), "fast".into()],
            ],
        };
        insta::assert_json_snapshot!(item, @r#"
        {
          "sentences": [
            [
              "the",
              "dog",
              "ran"
            ],
            [
              "it",
              "was",
              "fast"
            ]
          ]
        }
        "#);
    }

    #[test]
    fn snapshot_coref_response() {
        let resp = CorefResponse {
            annotations: vec![
                CorefAnnotation {
                    sentence_idx: 0,
                    annotation: "(0, -, 0)".into(),
                },
                CorefAnnotation {
                    sentence_idx: 1,
                    annotation: "0), -, -".into(),
                },
            ],
        };
        insta::assert_json_snapshot!(resp, @r#"
        {
          "annotations": [
            {
              "sentence_idx": 0,
              "annotation": "(0, -, 0)"
            },
            {
              "sentence_idx": 1,
              "annotation": "0), -, -"
            }
          ]
        }
        "#);
    }

    #[test]
    fn snapshot_coref_annotation() {
        let ann = CorefAnnotation {
            sentence_idx: 2,
            annotation: "(1 2, -, 1) 2)".into(),
        };
        insta::assert_json_snapshot!(ann, @r#"
        {
          "sentence_idx": 2,
          "annotation": "(1 2, -, 1) 2)"
        }
        "#);
    }

    // -----------------------------------------------------------------------
    // Structured coref data model tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_bracket_simple_chain() {
        // "the dog ran" — dog starts chain 0, ran ends chain 0
        let words = vec![
            vec![],
            vec![ChainRef {
                chain_id: 0,
                is_start: true,
                is_end: false,
            }],
            vec![ChainRef {
                chain_id: 0,
                is_start: false,
                is_end: true,
            }],
        ];
        assert_eq!(build_bracket_annotation(&words), "-, (0, 0)");
    }

    #[test]
    fn test_build_bracket_start_and_end_same_word() {
        // Single-word mention: starts and ends on same word
        let words = vec![
            vec![ChainRef {
                chain_id: 0,
                is_start: true,
                is_end: true,
            }],
            vec![],
            vec![],
        ];
        assert_eq!(build_bracket_annotation(&words), "(0), -, -");
    }

    #[test]
    fn test_build_bracket_multi_chain() {
        // Word participates in two chains simultaneously
        let words = vec![
            vec![
                ChainRef {
                    chain_id: 1,
                    is_start: true,
                    is_end: false,
                },
                ChainRef {
                    chain_id: 2,
                    is_start: true,
                    is_end: false,
                },
            ],
            vec![],
            vec![
                ChainRef {
                    chain_id: 1,
                    is_start: false,
                    is_end: true,
                },
                ChainRef {
                    chain_id: 2,
                    is_start: false,
                    is_end: true,
                },
            ],
        ];
        assert_eq!(build_bracket_annotation(&words), "(1 (2, -, 1) 2)");
    }

    #[test]
    fn test_build_bracket_continuation() {
        // Middle of a chain: neither start nor end
        let words = vec![
            vec![ChainRef {
                chain_id: 0,
                is_start: true,
                is_end: false,
            }],
            vec![ChainRef {
                chain_id: 0,
                is_start: false,
                is_end: false,
            }],
            vec![ChainRef {
                chain_id: 0,
                is_start: false,
                is_end: true,
            }],
        ];
        assert_eq!(build_bracket_annotation(&words), "(0, 0, 0)");
    }

    #[test]
    fn test_build_bracket_empty_words() {
        assert_eq!(build_bracket_annotation(&[]), "");
    }

    #[test]
    fn test_raw_to_bracket_response() {
        let raw = CorefRawResponse {
            annotations: vec![CorefRawAnnotation {
                sentence_idx: 0,
                words: vec![
                    vec![],
                    vec![ChainRef {
                        chain_id: 0,
                        is_start: true,
                        is_end: false,
                    }],
                    vec![ChainRef {
                        chain_id: 0,
                        is_start: false,
                        is_end: true,
                    }],
                ],
            }],
        };
        let bracket = raw_to_bracket_response(&raw);
        assert_eq!(bracket.annotations.len(), 1);
        assert_eq!(bracket.annotations[0].sentence_idx, 0);
        assert_eq!(bracket.annotations[0].annotation, "-, (0, 0)");
    }

    #[test]
    fn snapshot_chain_ref() {
        let cr = ChainRef {
            chain_id: 0,
            is_start: true,
            is_end: false,
        };
        insta::assert_json_snapshot!(cr, @r#"
        {
          "chain_id": 0,
          "is_start": true,
          "is_end": false
        }
        "#);
    }

    #[test]
    fn snapshot_coref_raw_annotation() {
        let ann = CorefRawAnnotation {
            sentence_idx: 0,
            words: vec![
                vec![],
                vec![ChainRef {
                    chain_id: 0,
                    is_start: true,
                    is_end: true,
                }],
                vec![],
            ],
        };
        insta::assert_json_snapshot!(ann, @r#"
        {
          "sentence_idx": 0,
          "words": [
            [],
            [
              {
                "chain_id": 0,
                "is_start": true,
                "is_end": true
              }
            ],
            []
          ]
        }
        "#);
    }

    // ---------------------------------------------------------------------
    // Wave 5 outcome-classification tests
    // ---------------------------------------------------------------------

    fn parse_chat_for_outcome(text: &str) -> ChatFile {
        let parser = crate::parse::TreeSitterParser::new().expect("parser init");
        parser.parse_chat_file(text).expect("parse")
    }

    fn three_utt_chat() -> String {
        "@UTF8\n\
         @Begin\n\
         @Languages:\teng\n\
         @Participants:\tCHI Target_Child\n\
         @ID:\teng|test|CHI||female|||Target_Child|||\n\
         *CHI:\thello world .\n\
         *CHI:\t&-hmm .\n\
         *CHI:\tI see the cat .\n\
         @End\n"
            .into()
    }

    #[test]
    fn collect_emits_not_applicable_for_filler_only() {
        let chat = parse_chat_for_outcome(&three_utt_chat());
        let collected = collect_coref_payloads(&chat);

        // 2 of 3 utterances are dispatched; the filler-only one
        // produces a NotApplicable outcome.
        assert_eq!(collected.batch_item.sentences.len(), 2);
        assert_eq!(collected.line_indices.len(), 2);
        assert_eq!(
            collected.not_applicable.len(),
            1,
            "expected the filler-only utterance to be classified NotApplicable",
        );
        match &collected.not_applicable[0].kind {
            CorefOutcomeKind::NotApplicable => {}
            other => panic!("expected NotApplicable, got {other:?}"),
        }
    }

    #[test]
    fn apply_with_outcomes_marks_dispatched_without_annotation_as_no_chains() {
        let mut chat = parse_chat_for_outcome(&three_utt_chat());
        let collected = collect_coref_payloads(&chat);
        let dispatched = collected.line_indices.clone();

        // Worker returned chains for only the FIRST dispatched utterance.
        let mut results = HashMap::new();
        results.insert(dispatched[0], "(0".to_string());

        let outcomes = apply_coref_results_with_outcomes(&mut chat, &results, &dispatched);

        assert_eq!(outcomes.len(), 2, "two dispatched utterances");
        // The first one got an annotation.
        match &outcomes[0].kind {
            CorefOutcomeKind::ChainsInjected { annotation } => {
                assert_eq!(annotation, "(0");
            }
            other => panic!("expected ChainsInjected on first dispatched, got {other:?}"),
        }
        // The second got none — that's not an anomaly.
        match &outcomes[1].kind {
            CorefOutcomeKind::NoChainsForSentence => {}
            other => panic!("expected NoChainsForSentence on second, got {other:?}"),
        }
    }

    #[test]
    fn apply_with_outcomes_flags_worker_annotation_for_undispatched_line() {
        let mut chat = parse_chat_for_outcome(&three_utt_chat());
        let collected = collect_coref_payloads(&chat);
        let dispatched = collected.line_indices.clone();

        // Worker annotated a line that was NOT dispatched (e.g. the
        // filler-only utterance's line_idx). Contract violation.
        let filler_line_idx = collected.not_applicable[0].line_idx;
        let mut results = HashMap::new();
        results.insert(filler_line_idx, "(0".to_string());

        let outcomes = apply_coref_results_with_outcomes(&mut chat, &results, &dispatched);

        // Every dispatched utterance should be NoChainsForSentence (nothing
        // in results for them), PLUS one ChainsInjected for the undispatched
        // line (the legacy behavior injected; we record the anomaly in the
        // outcomes stream). The exact shape depends on iteration order —
        // we just check both properties exist.
        let dispatched_outcomes = &outcomes[..dispatched.len()];
        for o in dispatched_outcomes {
            assert!(matches!(o.kind, CorefOutcomeKind::NoChainsForSentence));
        }
        assert!(
            outcomes.len() > dispatched.len(),
            "undispatched line produced an extra outcome"
        );
    }

    #[test]
    fn coref_outcome_to_decision_record_happy_paths_are_none() {
        let happy = [
            CorefOutcomeKind::NotApplicable,
            CorefOutcomeKind::NoChainsForSentence,
            CorefOutcomeKind::ChainsInjected {
                annotation: "(0".into(),
            },
        ];
        for kind in happy {
            let outcome = CorefOutcome {
                line_idx: 5,
                speaker: SpeakerCode::new("CHI"),
                kind,
            };
            assert!(
                outcome.to_decision_record().is_none(),
                "expected None for non-anomaly variant"
            );
        }
    }

    #[test]
    fn coref_outcome_to_decision_record_anomalies_need_review() {
        let outcome = CorefOutcome {
            line_idx: 5,
            speaker: SpeakerCode::new("CHI"),
            kind: CorefOutcomeKind::SentenceIndexOutOfBounds {
                sentence_idx: 99,
                resolved_line_idx: 42,
            },
        };
        let record = outcome.to_decision_record().unwrap();
        assert_eq!(
            record.strategy.module(),
            crate::decisions::DecisionModule::Coref
        );
        assert_eq!(
            record.strategy.strategy_name(),
            "sentence_index_out_of_bounds"
        );
        assert!(record.needs_review);
    }
}
