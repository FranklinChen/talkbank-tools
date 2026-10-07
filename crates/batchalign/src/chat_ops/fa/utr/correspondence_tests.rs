//! Typed-AST boundary controls: certainty and recovery are different measures.

use super::*;
use batchalign_transform::parse_source_with_parser;
use talkbank_model::{
    NullErrorSink,
    model::{FileStem, TranscriptName},
};
use talkbank_parser::TreeSitterParser;

fn admitted_chat(words: &str) -> ChatFile {
    let text = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\t{words} .\n@End\n"
    );
    let parser = TreeSitterParser::new().expect("parser");
    parse_source_with_parser(&parser, &text)
        .admit(
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            &NullErrorSink,
        )
        .expect("CHAT source admission")
        .into_valid_file()
        .into_unchecked()
}

fn tokens(text: &str) -> Vec<AsrTimingToken> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, text)| AsrTimingToken {
            text: text.to_owned(),
            start_ms: 100 + i as u64 * 100,
            end_ms: 180 + i as u64 * 100,
        })
        .collect()
}

fn admitted_dialogue(turns: &str) -> ChatFile {
    let text = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n@ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n{turns}\n@End\n"
    );
    let parser = TreeSitterParser::new().expect("parser");
    parse_source_with_parser(&parser, &text)
        .admit(
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            &NullErrorSink,
        )
        .expect("CHAT dialogue admission")
        .into_valid_file()
        .into_unchecked()
}

#[test]
fn joint_source_composition_recovers_unmarked_backchannel_inside_host() {
    let mut chat = admitted_dialogue("*PAR:\tone two three .\n*INV:\tyes .");
    let result = inject_utr_timing(&mut chat, &tokens("one yes two three"));
    assert_eq!(result.injected, 2);
    let bullets = super::super::utterances_indexed(&chat)
        .map(|(_, _, utterance)| {
            let timing = &utterance
                .main
                .content
                .bullet
                .as_ref()
                .expect("recovered")
                .timing;
            (timing.start_ms, timing.end_ms)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        bullets,
        [(100, 480), (200, 280)],
        "cross-speaker end is not the following turn's floor"
    );
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    assert_eq!(plan.strategy, UtrOrderModel::Interleaved);
    assert!(
        plan.regions
            .iter()
            .all(|region| region.strategy == UtrAlignmentStrategy::LocalInterleaving)
    );
}

#[test]
fn whole_source_composition_reserves_the_following_turns_repeat() {
    let mut chat = admitted_dialogue(
        "*PAR:\tanchor .\n*PAR:\tone two three .\n*INV:\tyes .\n*PAR:\ttail yes .",
    );
    let result = inject_utr_timing(&mut chat, &tokens("anchor one yes two three tail yes"));
    assert_eq!(result.injected, 4);
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    let UtrUtteranceAlignmentEvidence::Matched {
        admitted_matches, ..
    } = &plan.utterances[2]
    else {
        panic!("backchannel endpoints");
    };
    assert_eq!(
        admitted_matches
            .iter()
            .next()
            .expect("word")
            .matched()
            .token
            .token_index(),
        2,
        "later yes belongs to the following source turn"
    );
}

#[test]
fn missing_words_keep_a_source_order_search_corridor_without_timing_authority() {
    let mut chat =
        admitted_dialogue("*PAR:\tbefore .\n*PAR:\thost missing .\n*INV:\tyes .\n*INV:\tafter .");
    let observed = tokens("before host yes after");
    let result = inject_utr_timing(&mut chat, &observed);
    assert_eq!(
        result.injected(),
        3,
        "missing endpoint cannot become a timing hint"
    );
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global proof")
    };
    insta::assert_json_snapshot!(plan.search_envelopes[1], @r#"
    {
      "words": [
        "host",
        "missing"
      ],
      "floor_ms": 100,
      "ceiling_ms": 480,
      "scope": {
        "kind": "order_corridor"
      }
    }
    "#);
    let recording =
        super::super::coordinates::Recording::of_duration(super::super::coordinates::Ms(1000))
            .unwrap();
    let words: Vec<_> = ["host", "missing"]
        .into_iter()
        .enumerate()
        .map(|(word, text)| super::super::FaWord {
            utterance_index: talkbank_model::UtteranceIdx::new(1),
            utterance_word_index: talkbank_model::WordIdx::new(word),
            text: text.to_owned(),
        })
        .collect();
    let search = result
        .anchors
        .search_window(
            talkbank_model::UtteranceIdx::new(1),
            &words,
            super::super::TimeSpan::new(280, 300),
            &recording,
        )
        .unwrap();
    assert_eq!(
        (search.start_ms, search.end_ms),
        (100, 480),
        "the adjacent backchannel is not a boundary"
    );
    let mut wrong_source = words;
    wrong_source[1].text = "different".to_owned();
    assert!(
        result
            .anchors
            .search_window(
                talkbank_model::UtteranceIdx::new(1),
                &wrong_source,
                super::super::TimeSpan::new(280, 300),
                &recording
            )
            .is_none()
    );
}

#[test]
fn missing_endpoint_search_uses_the_actual_recording_edge_not_a_synthetic_time() {
    let mut chat = admitted_dialogue("*PAR:\thost missing .\n*INV:\tyes .");
    let result = inject_utr_timing(&mut chat, &tokens("host yes"));
    assert_eq!(result.injected(), 1);
    let words: Vec<_> = ["host", "missing"]
        .into_iter()
        .enumerate()
        .map(|(word, text)| super::super::FaWord {
            utterance_index: talkbank_model::UtteranceIdx::new(0),
            utterance_word_index: talkbank_model::WordIdx::new(word),
            text: text.to_owned(),
        })
        .collect();
    for end in [1000, 2000] {
        let recording =
            super::super::coordinates::Recording::of_duration(super::super::coordinates::Ms(end))
                .unwrap();
        let search = result
            .anchors
            .search_window(
                talkbank_model::UtteranceIdx::new(0),
                &words,
                super::super::TimeSpan::new(100, 180),
                &recording,
            )
            .unwrap();
        assert_eq!((search.start_ms, search.end_ms), (0, end));
        let grouping =
            super::super::grouping::group_utterances(&chat, 500, &recording, &result.anchors);
        assert!(
            !grouping
                .groups
                .iter()
                .flat_map(|group| group.words())
                .any(|word| word.text == "missing"),
            "an over-budget obligation must not fall back to a narrow convenient crop"
        );
        assert!(grouping.decisions.iter().any(|decision| matches!(
            decision.strategy,
            batchalign_transform::decisions::DecisionStrategy::Fa(
                batchalign_transform::decisions::FaStrategy::WindowRefused(_)
            )
        )));
    }
}

#[test]
fn obligatory_ambiguous_boundaries_keep_every_candidate_and_nested_provider_end() {
    let chat =
        admitted_dialogue("*PAR:\trepeat .\n*PAR:\thost missing .\n*INV:\tyes .\n*INV:\tafter .");
    let mut provider = tokens("repeat repeat host yes after");
    // A provider word inside another word's interval is not a reversed onset.
    provider[2].end_ms = 460;
    let plan = plan_global_utr_alignment(
        &chat,
        &provider,
        MatchMode::Exact,
        GlobalUtrParticipation::AllUtterances,
    );
    insta::assert_json_snapshot!(plan.search_envelopes[1], @r#"
    {
      "words": [
        "host",
        "missing"
      ],
      "floor_ms": 100,
      "ceiling_ms": 580,
      "scope": {
        "kind": "order_corridor"
      }
    }
    "#);
    provider[3].start_ms = 50;
    let reversed = plan_global_utr_alignment(
        &chat,
        &provider,
        MatchMode::Exact,
        GlobalUtrParticipation::AllUtterances,
    );
    assert!(
        reversed.search_envelopes[1].is_none(),
        "reversed provider onsets have no corridor authority"
    );
}

#[test]
fn ambiguous_interleaving_search_keeps_all_candidates_across_host_crop() {
    let mut chat =
        admitted_dialogue("*PAR:\twho are paving the way for all of you .\n*INV:\tall of .");
    let spans = [
        (0, 120),
        (120, 500),
        (500, 1140),
        (1140, 1340),
        (1340, 1520),
        (1520, 1740),
        (1740, 1820),
        (1740, 1980),
        (1820, 1900),
        (1980, 2140),
        (2140, 2520),
        (3000, 3200),
        (3200, 3400),
    ];
    let source = "who are paving the way for all all of of you all of"
        .split_whitespace()
        .zip(spans)
        .map(|(text, (start_ms, end_ms))| AsrTimingToken {
            text: text.into(),
            start_ms,
            end_ms,
        })
        .collect::<Vec<_>>();
    let result = inject_utr_timing(&mut chat, &source);
    assert_eq!(
        result.injected, 1,
        "ambiguous backchannel receives no selected-repeat hint"
    );
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global plan");
    };
    insta::assert_json_snapshot!(&plan.search_envelopes[1], @r#"
    {
      "words": [
        "all",
        "of"
      ],
      "start_ms": 1740,
      "end_ms": 3400,
      "scope": {
        "kind": "interleaved",
        "floor_ms": 0,
        "ceiling_ms": null
      }
    }
    "#);
    let mut texts = Vec::new();
    super::super::extraction::collect_fa_words(
        &super::super::utterances_indexed(&chat)
            .nth(1)
            .expect("backchannel")
            .2
            .main
            .content
            .content,
        &mut texts,
    );
    let words = texts
        .into_iter()
        .enumerate()
        .map(|(word, text)| super::super::FaWord {
            utterance_index: talkbank_model::UtteranceIdx::new(1),
            utterance_word_index: talkbank_model::WordIdx::new(word),
            text,
        })
        .collect::<Vec<_>>();
    let span = result
        .anchors
        .search_window(
            talkbank_model::UtteranceIdx::new(1),
            &words,
            super::super::TimeSpan::new(2520, 4000),
            &super::super::coordinates::Recording::of_duration(super::super::coordinates::Ms(4000))
                .expect("recording"),
        )
        .expect("other-speaker crop cannot discard a candidate");
    assert_eq!((span.start_ms, span.end_ms), (1740, 3400));
    let recording =
        super::super::coordinates::Recording::of_duration(super::super::coordinates::Ms(4000))
            .expect("recording");
    let groups = super::super::grouping::group_utterances(&chat, 2000, &recording, &result.anchors);
    assert!(
        groups
            .groups
            .iter()
            .any(|group| group.audio_start_ms() <= 1740 && group.audio_end_ms() >= 3400),
        "end-to-end grouping retains both early and late repeat locations"
    );
}

#[test]
fn interleaved_search_refuses_candidate_hulls_outside_retained_same_speaker_bounds() {
    let mut chat = admitted_dialogue(
        "@Media:\tinput, audio\n*INV:\tbefore . \u{15}100_200\u{15}\n*PAR:\tone two three .\n*INV:\tyes .\n*INV:\tafter . \u{15}1000_1200\u{15}",
    );
    let source = tokens("before one yes two three after yes");
    let originals = super::super::utterances_indexed(&chat)
        .filter_map(|(_, index, utterance)| {
            utterance
                .main
                .content
                .bullet
                .clone()
                .map(|bullet| (index, bullet))
        })
        .collect::<Vec<_>>();
    let unmodified = chat.clone();
    let result = inject_utr_timing(&mut chat, &source);
    assert_eq!(result.skipped, 2);
    for (index, original) in originals {
        let retained = super::super::utterances_indexed(&chat)
            .find(|(_, i, _)| *i == index)
            .expect("same census")
            .2;
        assert_eq!(retained.main.content.bullet.as_ref(), Some(&original));
    }
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    let envelope = serde_json::to_value(&plan.search_envelopes[2]).expect("bound envelope");
    assert_eq!(envelope["scope"]["floor_ms"], 200);
    assert_eq!(envelope["scope"]["ceiling_ms"], 1000);
    assert_eq!(envelope["start_ms"], 300);
    assert_eq!(
        envelope["end_ms"], 380,
        "later repeat is reserved by document order"
    );
    let mut conflicting = source;
    conflicting[2].start_ms = 1300;
    conflicting[2].end_ms = 1400;
    let plan = plan_global_utr_alignment(
        &unmodified,
        &conflicting,
        MatchMode::CaseInsensitive,
        GlobalUtrParticipation::AllUtterances,
    );
    // The moved repeat lies after the following anchor's bullet, so it is
    // outside the anchored region's onset window and is never a candidate.
    // What remains is the retained same-speaker corridor itself: an order
    // obligation between the bullets, not a candidate hull narrowed by it.
    insta::assert_json_snapshot!(plan.search_envelopes[2], @r#"
    {
      "words": [
        "yes"
      ],
      "floor_ms": 200,
      "ceiling_ms": 1000,
      "scope": {
        "kind": "order_corridor"
      }
    }
    "#);
}

#[test]
fn joint_budget_refusal_grants_no_fallback_timing_or_absence_claim() {
    let words = vec!["hello"; 1000].join(" ");
    let mut chat = admitted_dialogue(&format!("*PAR:\t{words} .\n*INV:\t{words} ."));
    let result = inject_utr_timing(&mut chat, &tokens("hello"));
    assert_eq!((result.injected, result.unmatched), (0, 2));
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    assert!(plan.utterances.iter().all(|evidence| matches!(
        evidence,
        UtrUtteranceAlignmentEvidence::Refused {
            reason: UtrCorrespondenceRefusal::BudgetExhausted(UtrBudgetRefusal {
                budget: UtrSearchBudget::InterleavingMemory { .. },
                ..
            }),
            ..
        }
    )));
    assert!(plan.search_envelopes.iter().all(Option::is_none));
    assert_eq!(result.anchors.anchored_utterances(), 0);
}

#[test]
fn complete_ambiguous_embeddings_bound_search_without_selecting_timings() {
    let mut chat = admitted_chat("hello now");
    let source = tokens("hello now noise now tail");
    let result = inject_utr_timing(&mut chat, &source);
    assert_eq!(result.injected, 0, "ambiguous end remains unhinted");
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global plan");
    };
    insta::assert_json_snapshot!(&plan.search_envelopes, @r#"
    [
      {
        "words": [
          "hello",
          "now"
        ],
        "start_ms": 100,
        "end_ms": 480
      }
    ]
    "#);
    let recording =
        super::super::coordinates::Recording::of_duration(super::super::coordinates::Ms(4000))
            .expect("recording");
    let groups = super::super::grouping::group_utterances(&chat, 1000, &recording, &result.anchors);
    assert_eq!(
        groups.groups.len(),
        1,
        "complete candidate envelope fits engine budget"
    );
    assert_eq!(groups.groups[0].word_count(), 2);
    assert_eq!(groups.groups[0].audio_start_ms(), 100);
    assert!(
        groups.groups[0].audio_end_ms() >= 480,
        "last candidate retained"
    );
    assert!(
        groups.groups[0].audio_end_ms() <= 1100,
        "padding also fits budget"
    );
    let words = vec![
        super::super::FaWord {
            utterance_index: talkbank_model::UtteranceIdx::new(0),
            utterance_word_index: talkbank_model::WordIdx::new(0),
            text: "hello".into(),
        },
        super::super::FaWord {
            utterance_index: talkbank_model::UtteranceIdx::new(0),
            utterance_word_index: talkbank_model::WordIdx::new(1),
            text: "now".into(),
        },
    ];
    let index = talkbank_model::UtteranceIdx::new(0);
    let corridor = super::super::TimeSpan::new(0, 4000);
    assert!(
        result
            .anchors
            .search_window(index, &words, corridor, &recording)
            .is_some()
    );
    let mut changed = words.clone();
    changed[1].text = "later".into();
    assert!(
        result
            .anchors
            .search_window(index, &changed, corridor, &recording)
            .is_none(),
        "same length is not source binding"
    );
    changed = words.clone();
    changed[0].utterance_index = talkbank_model::UtteranceIdx::new(1);
    assert!(
        result
            .anchors
            .search_window(index, &changed, corridor, &recording)
            .is_none(),
        "different source turn"
    );
    assert!(
        result
            .anchors
            .search_window(
                index,
                &words,
                super::super::TimeSpan::new(0, 400),
                &recording
            )
            .is_none(),
        "cannot discard last candidate at retained neighbor"
    );
    assert!(
        result
            .anchors
            .search_window(
                index,
                &words,
                super::super::TimeSpan::new(200, 4000),
                &recording
            )
            .is_none(),
        "cannot discard earliest candidate"
    );
}

#[test]
fn incomplete_or_disordered_provider_evidence_cannot_narrow_search() {
    for (words, mut source) in [
        ("missing hello now", tokens("hello now now")),
        ("hello missing now", tokens("hello now now")),
        ("hello now missing", tokens("hello now now")),
        ("hello now", tokens("hello now now")),
    ] {
        if words == "hello now" {
            source[1].start_ms = 50;
        }
        let plan = plan_global_utr_alignment(
            &admitted_chat(words),
            &source,
            MatchMode::CaseInsensitive,
            GlobalUtrParticipation::AllUtterances,
        );
        assert!(plan.search_envelopes.iter().all(Option::is_none), "{words}");
    }
}

#[test]
fn provider_segments_bound_search_but_do_not_become_word_anchors() {
    let mut chat = admitted_chat("hello now");
    let source = vec![AsrTimingToken {
        text: "hello now noise now".into(),
        start_ms: 100,
        end_ms: 900,
    }];
    let result = inject_utr_timing(&mut chat, &source);
    assert_eq!(result.injected, 0);
    assert!(matches!(
        result.anchors.lookup(talkbank_model::UtteranceIdx::new(0)),
        AnchorLookup::NoReliableMatch
    ));
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    assert!(plan.search_envelopes[0].is_some());
}

#[test]
fn search_envelopes_use_global_order_and_do_not_replace_original_bullets() {
    let mut chat = admitted_chat("hello .\n*PAR:\tnow");
    let source = tokens("now hello now now");
    let plan = plan_global_utr_alignment(
        &chat,
        &source,
        MatchMode::CaseInsensitive,
        GlobalUtrParticipation::AllUtterances,
    );
    let envelope = serde_json::to_value(&plan.search_envelopes[1]).expect("search evidence");
    assert_eq!(
        envelope["start_ms"], 300,
        "earlier now cannot follow hello in a complete embedding"
    );
    assert_eq!(envelope["end_ms"], 480);
    let _ = inject_utr_timing(&mut chat, &tokens("hello now"));
    let plan = plan_global_utr_alignment(
        &chat,
        &source,
        MatchMode::CaseInsensitive,
        GlobalUtrParticipation::AllUtterances,
    );
    assert!(
        plan.search_envelopes.iter().all(Option::is_none),
        "timed source remains authoritative"
    );
}

#[test]
fn terminal_asr_punctuation_keeps_source_evidence_and_word_anchors() {
    let mut chat = admitted_chat("who are paving the way for all of you");
    let result = inject_utr_timing(
        &mut chat,
        &tokens("who are paving the way for all of you. tail"),
    );
    assert_eq!(result.injected, 1);
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global plan");
    };
    let UtrUtteranceAlignmentEvidence::Matched {
        admitted_matches, ..
    } = &plan.utterances[0]
    else {
        panic!("matched endpoints");
    };
    let last = admitted_matches.iter().last().expect("last word").matched();
    assert_eq!(last.asr_text, "you.");
    assert_eq!(last.token.token_index(), 8);
    assert_eq!(last.relation, UtrLexicalRelation::TerminalPunctuation);
    insta::assert_json_snapshot!(last, @r#"
    {
      "word": {
        "utterance_index": 0,
        "word_index": 8
      },
      "token": {
        "token_index": 8,
        "word_index": 0
      },
      "chat_text": "you",
      "asr_text": "you.",
      "relation": {
        "kind": "terminal_punctuation"
      }
    }
    "#);
    let AnchorLookup::Anchored(words) = result.anchors.lookup(talkbank_model::UtteranceIdx::new(0))
    else {
        panic!("word anchors");
    };
    assert_eq!(words.anchors().len(), 9);
}

#[test]
fn punctuation_normalization_cannot_resolve_repeated_word_ambiguity() {
    let mut chat = admitted_chat("hello");
    let result = inject_utr_timing(&mut chat, &tokens("hello hello."));
    assert_eq!(result.injected, 0);
    assert!(matches!(
        result.anchors.lookup(talkbank_model::UtteranceIdx::new(0)),
        AnchorLookup::NoReliableMatch
    ));
}

#[test]
fn provider_lexical_admission_preserves_internal_punctuation_and_addresses() {
    let source = tokens(". it's well-known. U.S. !");
    let lexical = lexical::UtrLexicalStream::from_tokens(&source);
    assert_eq!(lexical.texts(), vec!["it's", "well-known", "U.S"]);
    let selected = lexical.matched_word(
        1,
        UtrWordAddress {
            utterance_index: evidence::UtrUtteranceOrdinal(0),
            word_index: evidence::UtrWordOrdinal(0),
        },
        "well-known",
    );
    assert_eq!(selected.token.token_index(), 2);
    assert_eq!(selected.asr_text, "well-known.");
    assert_eq!(selected.relation, UtrLexicalRelation::TerminalPunctuation);
}

#[test]
fn correspondence_ambiguous_repeated_word_keeps_selection_but_no_hint_or_anchor() {
    let mut chat = admitted_chat("hello");
    let result = inject_utr_timing(&mut chat, &tokens("hello hello"));
    assert_eq!((result.injected, result.unmatched), (0, 1));
    assert!(matches!(
        result.anchors.lookup(talkbank_model::UtteranceIdx::new(0)),
        AnchorLookup::NoReliableMatch
    ));
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    assert_eq!(plan.token_extents(), vec![Some((0, 0))]);
    let wire = serde_json::to_value(&plan.utterances[0]).expect("wire evidence");
    assert_eq!(wire["status"], "selected_only");
    assert_eq!(wire["reason"], "ambiguous");
    assert!(wire.get("proposal").is_none());
    assert_eq!(wire["matches"]["first"]["token"]["token_index"], 0);
}

#[test]
fn correspondence_partial_certainty_retains_anchor_and_complete_candidate_range() {
    let mut chat = admitted_chat("hello now");
    let mut source = tokens("hello hello now");
    let result = inject_utr_timing(&mut chat, &source);
    assert_eq!((result.injected, result.unmatched), (0, 1));
    // A later independently changed stream cannot redirect the proof's timing.
    source[2].start_ms = 9000;
    source[2].end_ms = 9999;
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    let UtrUtteranceAlignmentEvidence::InteriorOnly {
        admitted_matches,
        matches,
        missing_endpoints,
        ..
    } = &plan.utterances[0]
    else {
        panic!("common now must retain interior evidence");
    };
    assert_eq!(*missing_endpoints, MissingUtrEndpoints::First);
    assert_eq!(
        matches.rest.len(),
        1,
        "selected path remains complete for inspection"
    );
    assert_eq!(admitted_matches.iter().count(), 1);
    let anchors = AnchorIndex::from_plan(&plan);
    let AnchorLookup::Anchored(words) = anchors.lookup(talkbank_model::UtteranceIdx::new(0)) else {
        panic!("common exact word anchors");
    };
    assert_eq!(words.anchors().len(), 1);
    assert_eq!(words.anchors()[0].word(), talkbank_model::WordIdx::new(1));
    let recording =
        super::super::coordinates::Recording::of_duration(super::super::coordinates::Ms(1000))
            .expect("positive recording");
    let groups = super::super::grouping::group_utterances(&chat, 1000, &recording, &anchors);
    assert_eq!(
        groups.groups.len(),
        1,
        "unhinted FA recovery remains available"
    );
    assert_eq!(
        groups.groups[0].word_count(),
        2,
        "unresolved prefix is not discarded"
    );
    assert_eq!(
        groups.groups[0].audio_start_ms(),
        100,
        "earliest hello retained by full embedding proof, not interior anchor"
    );
    assert!(
        groups.groups[0].audio_end_ms() >= 380,
        "latest full embedding retained"
    );
    assert!(groups.groups[0].audio_end_ms() <= 1000);
}

#[test]
fn correspondence_endpoint_matrix_keeps_unhinted_recovery_and_unique_controls() {
    for (words, asr, missing) in [
        (
            "hello now",
            "hello hello now",
            Some(MissingUtrEndpoints::First),
        ),
        (
            "hello now",
            "hello now now",
            Some(MissingUtrEndpoints::Last),
        ),
        (
            "hello middle now",
            "hello hello middle now now",
            Some(MissingUtrEndpoints::Both),
        ),
        ("hello missing now", "hello now", None),
        ("hello", "hello", None),
    ] {
        let mut chat = admitted_chat(words);
        let source = tokens(asr);
        let result = inject_utr_timing(&mut chat, &source);
        assert_eq!(
            result.injected,
            usize::from(missing.is_none()),
            "{words} / {asr}"
        );
        if let Some(expected) = missing {
            let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
                panic!("global plan");
            };
            assert!(
                matches!(&plan.utterances[0], UtrUtteranceAlignmentEvidence::InteriorOnly { missing_endpoints, .. } if *missing_endpoints == expected)
            );
            assert_eq!(
                two_pass::recover_overlap_timing(
                    &words
                        .split_whitespace()
                        .map(str::to_owned)
                        .collect::<Vec<_>>(),
                    &source,
                    0,
                    1000,
                    MatchMode::CaseInsensitive
                ),
                None
            );
        } else {
            assert!(
                two_pass::recover_overlap_timing(
                    &words
                        .split_whitespace()
                        .map(str::to_owned)
                        .collect::<Vec<_>>(),
                    &source,
                    0,
                    1000,
                    MatchMode::CaseInsensitive
                )
                .is_some(),
                "unique endpoints retain local recovery"
            );
        }
    }
}

#[test]
fn correspondence_local_overlap_abstains_on_repetition_and_retains_unique_control() {
    let words = vec!["hello".to_owned()];
    assert_eq!(
        two_pass::recover_overlap_timing(
            &words,
            &tokens("hello hello"),
            0,
            1000,
            MatchMode::CaseInsensitive
        ),
        None
    );
    let recovered = two_pass::recover_overlap_timing(
        &words,
        &tokens("noise hello tail"),
        0,
        1000,
        MatchMode::CaseInsensitive,
    )
    .expect("unique local recovery");
    assert_eq!((recovered.start_ms(), recovered.end_ms()), (200, 280));
}

#[test]
fn correspondence_missing_word_does_not_destroy_unique_remaining_recovery() {
    let mut chat = admitted_chat("hello missing now");
    let result = inject_utr_timing(&mut chat, &tokens("hello now"));
    assert_eq!((result.injected, result.unmatched), (1, 0));
    let UtrAlignmentEvidence::Global { plan } = result.alignment else {
        panic!("global plan");
    };
    let UtrUtteranceAlignmentEvidence::Matched {
        admitted_matches, ..
    } = &plan.utterances[0]
    else {
        panic!("unique remaining words recover");
    };
    assert_eq!(admitted_matches.iter().count(), 2);
}

/// An endpoint matched in every optimum, though not to one token, still
/// bounds its utterance: by the extent of every token it may match. Here the
/// two turns' boundary words are the same word, and two adjacent tokens can
/// go either way, so neither endpoint has a common correspondence. Each turn
/// still gets a hint, and each hint contains both tokens its endpoint may
/// match, so whichever reading is true none of its speech is cropped.
#[test]
fn an_endpoint_matched_in_every_optimum_bounds_its_hint_by_every_candidate() {
    let mut chat = admitted_dialogue("*PAR:\tthe had .\n*INV:\thad was .");
    // the 100-180, had 200-280, had 300-380, was 400-480.
    let result = inject_utr_timing(&mut chat, &tokens("the had had was"));
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global proof")
    };
    assert!(
        plan.utterances
            .iter()
            .all(|evidence| matches!(evidence, UtrUtteranceAlignmentEvidence::Matched { .. })),
        "both turns are bounded: {:?}",
        plan.utterances
    );
    let bullets: Vec<_> = chat
        .lines
        .iter()
        .filter_map(|line| match line {
            Line::Utterance(utterance) => utterance
                .main
                .content
                .bullet
                .as_ref()
                .map(|bullet| (bullet.timing.start_ms, bullet.timing.end_ms)),
            _ => None,
        })
        .collect();
    assert_eq!(bullets, vec![(100, 380), (200, 480)]);
}
