use super::BuildChatError;
use super::*;
use crate::asr_postprocess;
use crate::parse::{TreeSitterParser, parse_lenient};
use crate::serialize::to_chat_string;
use talkbank_model::model::TranscriptName;

/// Helper: create a regular WordDesc with validated text and explicit kind.
fn wd(text: &str, start_ms: Option<u64>, end_ms: Option<u64>) -> WordDesc {
    WordDesc {
        text: asr_postprocess::ChatWordText::try_from(text)
            .expect("test: word text must be CHAT-legal"),
        start_ms,
        end_ms,
        kind: asr_postprocess::WordKind::Regular,
    }
}

#[test]
fn test_build_chat_minimal() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![
                wd("hello", None, None),
                wd("world", None, None),
                wd(".", None, None),
            ]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };

    let chat_file = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(output.contains("@Languages:\teng"));
    assert!(output.contains("*PAR:\thello world ."));
}

#[test]
fn test_build_chat_with_timing() {
    let parser = TreeSitterParser::new().unwrap();
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: Some("test.mp3".to_string()),
        media_type: Some("audio".to_string()),
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![
                wd("hello", Some(0), Some(500)),
                wd("world", Some(500), Some(1000)),
                wd(".", None, None),
            ]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: true,
    };

    let chat_file = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(output.contains("@Media:\ttest, audio"), "got: {output}");
    assert!(output.contains("%wor:"));
    let (_parsed, errors) = parse_lenient(&parser, &output);
    assert!(
        errors.is_empty(),
        "serialized CHAT should reparse cleanly: {errors:?}"
    );
}

/// Regression for the archived 3039 replay: a segmented utterance had timed
/// `%wor` words but no main-tier bullet, while the next main-tier bullet kept
/// an enclosing pre-segmentation start. Word-mode CHAT construction owns both
/// tiers, so each main bullet must be the hull of that utterance's own words.
#[test]
fn word_utterance_main_bullets_are_derived_from_own_word_hulls() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: Some("3039.mp3".to_string()),
        media_type: Some("audio".to_string()),
        media_status: None,
        utterances: vec![
            UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("Department", Some(24_275), Some(24_945)),
                    wd("store", Some(24_945), Some(25_265)),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            },
            UtteranceDesc {
                speaker: "PAR".to_string(),
                words: Some(vec![
                    wd("almost", Some(25_265), Some(25_585)),
                    wd("anniversary", Some(27_455), Some(28_145)),
                    wd(".", None, None),
                ]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
            },
        ],
        write_wor: true,
    };

    let output = to_chat_string(&build_chat(&desc).expect("word-timed CHAT should build"));

    assert!(
        output.contains("*PAR:\tDepartment store . \u{15}24275_25265\u{15}"),
        "first main-tier hull must come from its own words: {output}"
    );
    assert!(
        output.contains("*PAR:\talmost anniversary . \u{15}25265_28145\u{15}"),
        "second main-tier hull must not retain an enclosing start: {output}"
    );
}

#[test]
fn test_build_chat_from_json() {
    let json = r#"{
        "langs": ["eng"],
        "participants": [{"id": "PAR"}],
        "utterances": [
            {"speaker": "PAR", "words": [
                {"text": "hello"},
                {"text": "."}
            ]}
        ]
    }"#;

    let chat_file = build_chat_from_json(json).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(output.contains("*PAR:\thello ."));
}

#[test]
fn test_build_chat_text_utterance() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: None,
            text: Some("hello world .".to_string()),
            start_ms: Some(0),
            end_ms: Some(1000),
            lang: None,
        }],
        write_wor: false,
    };

    let chat_file = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(output.contains("*PAR:\thello world ."));
}

#[test]
fn test_build_chat_question_terminator() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![wd("how", None, None), wd("?", None, None)]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };

    let chat_file = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(output.contains("*PAR:\thow ?"));
}

#[test]
fn test_write_wor_false_suppresses_wor_tier() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: Some("test.mp3".to_string()),
        media_type: Some("audio".to_string()),
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![
                wd("hello", Some(0), Some(500)),
                wd("world", Some(500), Some(1000)),
                wd(".", None, None),
            ]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };

    let chat_file = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(
        !output.contains("%wor:"),
        "write_wor=false should suppress %wor tier, got: {output}"
    );
    // Inline word bullets should still be present
    assert!(
        output.contains("\u{15}"),
        "word-level bullets should still appear on the main tier"
    );
}

#[test]
fn test_transcript_from_asr_utterances() {
    let utterances = vec![
        asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(0),
            words: vec![
                asr_postprocess::AsrWord::new("hello", Some(0), Some(500)),
                asr_postprocess::AsrWord::new(".", None, None),
            ],
            lang: None,
        },
        asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(1),
            words: vec![
                asr_postprocess::AsrWord::new("world", Some(500), Some(1000)),
                asr_postprocess::AsrWord::new(".", None, None),
            ],
            lang: None,
        },
    ];

    let ids = vec!["PAR".to_string(), "INV".to_string()];
    let desc = transcript_from_asr_utterances(
        &utterances,
        &ids,
        &["eng".to_string()],
        Some("test.mp3"),
        false,
    )
    .expect("test: transcript_from_asr_utterances should succeed")
    .description;

    assert_eq!(desc.participants.len(), 2);
    assert_eq!(desc.participants[0].id, "PAR");
    assert_eq!(desc.participants[1].id, "INV");
    assert_eq!(desc.utterances.len(), 2);
    assert_eq!(desc.utterances[0].speaker, "PAR");
    assert_eq!(desc.utterances[1].speaker, "INV");

    // Should build a valid CHAT file
    let chat_file = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat_file);
    assert!(output.contains("*PAR:"));
    assert!(output.contains("*INV:"));
}

#[test]
fn test_numbered_asr_names_require_no_dense_speaker_population() {
    let utterances = vec![asr_postprocess::Utterance {
        speaker: asr_postprocess::SpeakerIndex(usize::MAX),
        words: vec![asr_postprocess::AsrWord::new("hello", None, None)],
        lang: None,
    }];

    assert!(matches!(
        transcript_from_asr_utterances(&utterances, &[], &["eng".to_string()], None, false),
        Err(TranscriptBuildError::MissingParticipantCode(_))
    ));
    let desc = NamedAsrUtterances::numbered(&utterances)
        .into_transcript(&["eng".to_string()], None, false)
        .expect("test: transcript_from_asr_utterances should succeed")
        .description;
    assert_eq!(desc.participants.len(), 1);
    assert_eq!(desc.participants[0].id, format!("PAR{}", usize::MAX));
    assert_eq!(desc.utterances[0].speaker, desc.participants[0].id);
}

#[test]
fn test_numbered_asr_empty_population_and_missing_language() {
    let empty = NamedAsrUtterances::numbered(&[])
        .into_transcript(&["eng".to_string()], None, false)
        .unwrap();
    assert!(empty.description.participants.is_empty());
    assert!(empty.description.utterances.is_empty());
    assert!(matches!(
        NamedAsrUtterances::numbered(&[]).into_transcript(&[], None, false),
        Err(TranscriptBuildError::MissingPrimaryLanguage)
    ));
}

// ── ASR-to-CHAT validation gap regression tests ──────────────────────
//
// Two ASR token classes used to tank the entire transcribe job at the
// `transcript_from_asr_utterances` gate:
//
//   1. Boundary quote marks (e.g. `"My`), Whisper transcribes quoted
//      speech verbatim, leaving stray `"` characters glued to the next
//      word. Tree-sitter rejects the `"` and the file aborts.
//   2. Digit-bearing alphanumeric tokens (e.g. `C-3PO` under English)
//      structurally legal CHAT, but `Word::validate` fires E220
//      ("numeric digits not allowed") in languages outside the
//      digit-permitting set.
//
// Design: maximize information preservation, but never abuse the
// reserved `xxx` / `yyy` / `www` markers (those mean specific things
// about transcriber experience: see `untranscribed-markers.md` in
// the talkbank-tools book).
//
//   • Boundary quotes get a silent orthographic strip (Stage 2c)
//     no information is lost when `"` becomes a no-op character.
//   • Validation-only failures (digit policy etc.) fall back to the
//     structural-only `ChatWordText::try_from` path, shipping the
//     token verbatim. The downstream full-file validator and CHECK
//     fire the same E220 and the file ends up in the human review
//     queue. The transcriber listens, decides what was actually
//     said, fixes the transcript. ASR doesn't pretend to know.
//   • Genuine structural (parse) failures still fail loud, emitting
//     malformed CHAT silently would corrupt the file beyond CHECK's
//     ability to flag it.

#[test]
fn quote_mark_in_asr_token_is_silently_stripped() {
    // `"My` arrives at the pipeline; Stage 2c boundary-quote strip
    // (`cleanup::strip_boundary_quotes`) removes the leading `"`
    // before validation. The whole pipeline succeeds and the first
    // content word is `My`, not `"My`.
    let result = run_transcribe_to_description(
        &[("\"My", 0.0, 0.5), ("cake", 0.5, 1.0), (".", 1.0, 1.0)],
        "eng",
    )
    .expect("Stage 2c should silently strip the leading quote and pass validation");

    let first_word = result
        .description
        .utterances
        .first()
        .and_then(|u| u.words.as_deref()?.first())
        .expect("first utterance has at least one word");
    assert_eq!(
        first_word.text.as_str(),
        "My",
        "boundary quote should be stripped",
    );
}

#[test]
fn alphanumeric_token_under_eng_is_emitted_verbatim_for_review() {
    // Whisper transcribes proper nouns like `C-3PO` verbatim. Under
    // eng, digits are illegal (E220). The ASR pipeline does NOT
    // invent semantics for digit-bearing alphanumerics (digit-by-
    // digit? cardinal? ordinal?: unknowable from surface form).
    // The gate falls back to the structural-only `try_from` path,
    // shipping `C-3PO` verbatim. Tree-sitter accepts the token,
    // the file builds, and the downstream validator + CHECK fire
    // E220 for the human reviewer to listen and decide.
    //
    // Substituting `xxx` is BANNED; it would corrupt that marker's
    // "transcriber listened and could not make it out" meaning
    // across the whole corpus.
    let result = run_transcribe_to_description(
        &[
            ("the", 0.0, 0.1),
            ("C-3PO", 0.1, 0.8),
            ("droid", 0.8, 1.2),
            (".", 1.2, 1.2),
        ],
        "eng",
    )
    .expect(
        "gate should fall back to structural-only path so `C-3PO` ships \
         verbatim for downstream validator + CHECK to flag",
    );

    let words: Vec<String> = result
        .description
        .utterances
        .iter()
        .flat_map(|u| u.words.as_deref().unwrap_or(&[]))
        .map(|w| w.text.as_str().to_owned())
        .collect();
    assert!(
        words.iter().any(|w| w == "C-3PO"),
        "expected `C-3PO` preserved verbatim in {words:?}",
    );
    assert!(
        !words.iter().any(|w| w == "xxx"),
        "must not substitute `xxx`; it has the reserved meaning \
         \"transcriber listened and could not make it out\". Found: {words:?}",
    );
}

#[test]
fn alphanumeric_token_passes_zho_validation_gate() {
    // Counterpart to the eng test: numeric digits ARE legal in some
    // languages, so the same `C-3PO` token must pass the gate under
    // those languages. Confirms the rule is language-specific and
    // bounds the scope of any fix, the normalizer must consult the
    // utterance's language before deciding whether to act.
    //
    // NOTE on the language code: the talkbank-tools word validator's
    // current digit-allow set is `{zho, cym, vie, tha, nan, yue, min,
    // hak}`. `cmn` (Mandarin, spoken) is NOT in that set even though
    // it's the more linguistically precise code; `zho` (Chinese, the
    // macrolanguage / written) is. That asymmetry is a separate
    // talkbank-tools concern, not this crate's bug, but documenting
    // it here so a future ASR-output-with-cmn test failure traces to
    // the right place.
    let utterances = vec![asr_postprocess::Utterance {
        speaker: asr_postprocess::SpeakerIndex(0),
        words: vec![
            asr_postprocess::AsrWord::new("C-3PO", Some(0), Some(500)),
            asr_postprocess::AsrWord::new("。", None, None),
        ],
        lang: None,
    }];
    let result = transcript_from_asr_utterances(
        &utterances,
        &["PAR".to_string()],
        &["zho".to_string()],
        None,
        false,
    );
    assert!(
        result.is_ok(),
        "zho should accept digit-containing tokens; got {result:?}"
    );
}

#[test]
fn test_tag_marker_separator() {
    assert!(tag_marker_separator(",").is_some());
    assert!(tag_marker_separator("\u{201E}").is_some());
    assert!(tag_marker_separator("\u{2021}").is_some());
    assert!(tag_marker_separator("hello").is_none());
}

#[test]
fn test_empty_participants_error() {
    let desc = TranscriptDescription {
        langs: vec![],
        participants: vec![],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![],
        write_wor: false,
    };
    assert!(build_chat(&desc).is_err());
}

// -- Retrace AST construction tests --

/// Helper: create a retrace WordDesc.
fn wd_retrace(text: &str, start_ms: Option<u64>, end_ms: Option<u64>) -> WordDesc {
    WordDesc {
        text: asr_postprocess::ChatWordText::try_from(text)
            .expect("test: word text must be CHAT-legal"),
        start_ms,
        end_ms,
        kind: asr_postprocess::WordKind::Retrace,
    }
}

/// Helper: build a single-utterance CHAT file and return serialized output.
fn build_single_utterance(words: Vec<WordDesc>) -> String {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(words),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };
    let chat = build_chat(&desc).unwrap();
    to_chat_string(&chat)
}

#[test]
fn single_word_retrace_produces_annotated_word() {
    // "I [/] I went ." → AnnotatedWord with PartialRetracing
    let output = build_single_utterance(vec![
        wd_retrace("I", None, None),
        wd("I", None, None),
        wd("went", None, None),
        wd(".", None, None),
    ]);
    assert!(
        output.contains("I [/] I went ."),
        "expected single-word retrace: {output}"
    );
}

#[test]
fn multi_word_retrace_produces_annotated_group() {
    // "<I want> [/] I want cookie ."
    let output = build_single_utterance(vec![
        wd_retrace("I", None, None),
        wd_retrace("want", None, None),
        wd("I", None, None),
        wd("want", None, None),
        wd("cookie", None, None),
        wd(".", None, None),
    ]);
    assert!(
        output.contains("<I want> [/] I want cookie ."),
        "expected multi-word retrace: {output}"
    );
}

#[test]
fn retrace_preserves_per_word_timing() {
    let output = build_single_utterance(vec![
        wd_retrace("go", Some(0), Some(200)),
        wd("go", Some(200), Some(400)),
        wd("home", Some(400), Some(600)),
        wd(".", None, None),
    ]);
    // The retrace word should have an inline bullet.
    assert!(
        output.contains("\u{15}"),
        "retrace word should preserve timing bullets: {output}"
    );
    assert!(output.contains("[/]"), "expected retrace marker: {output}");
}

#[test]
fn retrace_output_reparses_cleanly() {
    let parser = TreeSitterParser::new().unwrap();
    // Single-word retrace
    let output = build_single_utterance(vec![
        wd_retrace("I", None, None),
        wd("I", None, None),
        wd("went", None, None),
        wd(".", None, None),
    ]);
    let (_parsed, errors) = parse_lenient(&parser, &output);
    assert!(
        errors.is_empty(),
        "single-word retrace should reparse: {errors:?}\noutput: {output}"
    );

    // Multi-word retrace
    let output = build_single_utterance(vec![
        wd_retrace("I", None, None),
        wd_retrace("want", None, None),
        wd("I", None, None),
        wd("want", None, None),
        wd("cookie", None, None),
        wd(".", None, None),
    ]);
    let (_parsed, errors) = parse_lenient(&parser, &output);
    assert!(
        errors.is_empty(),
        "multi-word retrace should reparse: {errors:?}\noutput: {output}"
    );
}

#[test]
fn disfluency_and_retrace_end_to_end() {
    let parser = TreeSitterParser::new().unwrap();
    // Full pipeline: raw ASR → process_raw_asr (includes disfluency + retrace).expect("test: ASR post-processing must not refuse this input")
    // → transcript_from_asr_utterances → build_chat.
    //
    // Input "um um I I went" exercises BOTH pipeline stages:
    //   - disfluency: "um um" → "&-um &-um" (filled pauses stay as fillers,
    //     BA2 parity: fillers do NOT emit [/])
    //   - retrace: "I I" → "<I> [/] I" (genuine word repetition still marks)
    let output = asr_postprocess::AsrOutput {
        monologues: vec![asr_postprocess::AsrMonologue {
            speaker: asr_postprocess::SpeakerIndex(0),
            elements: vec![
                asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new("um"),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(0.0),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(0.2),
                    kind: asr_postprocess::AsrElementKind::Text,
                },
                asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new("um"),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(0.2),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(0.4),
                    kind: asr_postprocess::AsrElementKind::Text,
                },
                asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new("I"),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(0.4),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(0.5),
                    kind: asr_postprocess::AsrElementKind::Text,
                },
                asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new("I"),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(0.5),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(0.6),
                    kind: asr_postprocess::AsrElementKind::Text,
                },
                asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new("went"),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(0.6),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(0.8),
                    kind: asr_postprocess::AsrElementKind::Text,
                },
            ],
        }],
    };
    let utts = asr_postprocess::process_raw_asr(&output, "eng")
        .expect("test: ASR post-processing must not refuse this input");

    let desc = transcript_from_asr_utterances(
        &utts,
        &["PAR".to_string()],
        &["eng".to_string()],
        None,
        false,
    )
    .expect("test: transcript_from_asr_utterances should succeed")
    .description;
    let chat = build_chat(&desc).unwrap();
    let serialized = to_chat_string(&chat);

    // Should contain filled pause marker and retrace
    assert!(
        serialized.contains("&-um"),
        "expected filled pause: {serialized}"
    );
    assert!(
        serialized.contains("[/]"),
        "expected retrace marker: {serialized}"
    );

    let (_parsed, errors) = parse_lenient(&parser, &serialized);
    assert!(
        errors.is_empty(),
        "disfluency+retrace should reparse cleanly: {errors:?}\noutput: {serialized}"
    );
}

// ── a user 2026-04-02 bug reports ────────────────────────────────

/// Bug 1: @Media header must have comma+space between name and type.
/// a user saw: `@Media: 279home-2audio`
/// Expected:  `@Media: 279home-2, audio`
#[test]
fn media_header_has_comma_separator() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: Some("279home-2.mp3".to_string()),
        media_type: Some("audio".to_string()),
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![wd("hello", None, None), wd(".", None, None)]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };
    let chat = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat);
    assert!(
        output.contains("@Media:\t279home-2, audio"),
        "@Media must have 'name, type' format with comma+space, got: {}",
        output
            .lines()
            .find(|l| l.contains("@Media"))
            .unwrap_or("(no @Media)")
    );
}

/// Bug 2: When transcript_from_asr_utterances generates participants
/// from speaker IDs, it must set name and role from the speaker code.
/// PAR → Participant, INV → Investigator. No silent defaults.
///
/// a user saw: `PAR Participant Participant, INV Participant Participant`
/// Expected:   `PAR Participant , INV Investigator`
///
/// The root cause is transcript_from_asr_utterances setting name=None,
/// role=None and build_chat silently defaulting both to "Participant".
#[test]
fn asr_pipeline_sets_correct_participant_roles() {
    use asr_postprocess::{AsrWord, SpeakerIndex, Utterance};

    let utterances = vec![
        Utterance {
            speaker: SpeakerIndex(0),
            words: vec![
                AsrWord::new("hello", Some(0), Some(500)),
                AsrWord::new(".", None, None),
            ],
            lang: None,
        },
        Utterance {
            speaker: SpeakerIndex(1),
            words: vec![
                AsrWord::new("world", Some(600), Some(1000)),
                AsrWord::new(".", None, None),
            ],
            lang: None,
        },
    ];
    let participant_ids = vec!["PAR0".to_string(), "PAR1".to_string()];
    let desc = transcript_from_asr_utterances(
        &utterances,
        &participant_ids,
        &["eng".to_string()],
        None,
        false,
    )
    .expect("test: transcript_from_asr_utterances should succeed")
    .description;

    // Both speakers get generic codes and "Participant" role
    let par0 = desc
        .participants
        .iter()
        .find(|p| p.id == "PAR0")
        .expect("must have PAR0 participant");
    assert_eq!(par0.role, "Participant");
    assert_eq!(par0.name, None, "name should be None (not doubled as role)");

    let par1 = desc
        .participants
        .iter()
        .find(|p| p.id == "PAR1")
        .expect("must have PAR1 participant");
    assert_eq!(par1.role, "Participant");

    // Verify serialization: "PAR0 Participant, PAR1 Participant"
    // No doubled role words
    let chat = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat);
    let participants_line = output
        .lines()
        .find(|l| l.starts_with("@Participants:"))
        .expect("must have @Participants");
    assert!(
        participants_line.contains("PAR0 Participant"),
        "got: {participants_line}"
    );
    assert!(
        participants_line.contains("PAR1 Participant"),
        "got: {participants_line}"
    );
    assert!(
        !participants_line.contains("Participant Participant"),
        "@Participants must NOT double the role word, got: {participants_line}"
    );
}

/// Bug 4: @Comment must include "DO NOT USE" and use commit hash, not version.
#[test]
fn transcribe_comment_includes_do_not_use() {
    let desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![wd("hello", None, None), wd(".", None, None)]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };
    let chat = build_chat(&desc).unwrap();
    let output = to_chat_string(&chat);
    // Find the "Unchecked output" comment
    if let Some(comment) = output.lines().find(|l| l.contains("Unchecked output")) {
        assert!(
            comment.contains("DO NOT USE"),
            "@Comment with 'Unchecked output' must include 'DO NOT USE', got: {comment}"
        );
    }
    // Version should not be a semver like "0.1.0", should be commit hash or omitted
    for line in output.lines() {
        if line.starts_with("@Comment:") && line.contains("Batchalign") {
            assert!(
                !line.contains("0.1.0") && !line.contains("1.0.0"),
                "@Comment must not contain hardcoded semver, got: {line}"
            );
        }
    }
}

// ── End-to-end transcribe-pipeline invariants ────────────────────────────
//
// These tests drive the full pipeline (`process_raw_asr` →
// `transcript_from_asr_utterances` → `build_chat` → `to_chat_string`)
// and pin user-visible casing and retrace-shape invariants in the
// serialized CHAT output: the boundary that no earlier unit test
// reaches.

/// Build a single-speaker `AsrOutput` from `(text, start_secs, end_secs)`
/// tuples, treating `.`/`?`/`!` as punctuation elements.
fn single_speaker_asr_output(tokens: &[(&str, f64, f64)]) -> asr_postprocess::AsrOutput {
    let elements: Vec<asr_postprocess::AsrElement> = tokens
        .iter()
        .map(|(text, start, end)| {
            let kind = if matches!(*text, "." | "?" | "!") {
                asr_postprocess::AsrElementKind::Punctuation
            } else {
                asr_postprocess::AsrElementKind::Text
            };
            asr_postprocess::AsrElement {
                value: asr_postprocess::AsrRawText::new(*text),
                ts: asr_postprocess::AsrTimestampSecs::Observed(*start),
                end_ts: asr_postprocess::AsrTimestampSecs::Observed(*end),
                kind,
            }
        })
        .collect();
    asr_postprocess::AsrOutput {
        monologues: vec![asr_postprocess::AsrMonologue {
            speaker: asr_postprocess::SpeakerIndex(0),
            elements,
        }],
    }
}

/// Drive the full ASR → CHAT pipeline for a single-speaker input and
/// return the typed `TranscriptDescription` (pre-serialization).
fn run_transcribe_to_description(
    tokens: &[(&str, f64, f64)],
    lang: &str,
) -> Result<AsrTranscript, TranscriptBuildError> {
    let output = single_speaker_asr_output(tokens);
    let utts = asr_postprocess::process_raw_asr(&output, lang)
        .expect("test: ASR post-processing must not refuse this input");
    transcript_from_asr_utterances(
        &utts,
        &["PAR1".to_string()],
        &[lang.to_string()],
        None,
        false,
    )
}

/// Drive the full ASR → CHAT pipeline for a single-speaker input and
/// return the serialized CHAT output.
fn run_transcribe_pipeline(tokens: &[(&str, f64, f64)], lang: &str) -> String {
    let desc = run_transcribe_to_description(tokens, lang)
        .expect("test: transcript_from_asr_utterances should succeed")
        .description;
    let chat = build_chat(&desc).unwrap();
    to_chat_string(&chat)
}

/// The English pronoun "I" and its contractions ("I'm", "I'd") must be
/// preserved uppercase in the serialized output.
#[test]
fn user_report_english_pronoun_i_preserved_from_rev_ai() {
    let output = run_transcribe_pipeline(
        &[
            ("this", 0.0, 0.2),
            ("is", 0.2, 0.4),
            ("not", 0.4, 0.6),
            ("where", 0.6, 0.8),
            ("I", 0.8, 0.9),
            ("grew", 0.9, 1.1),
            ("up", 1.1, 1.3),
            (".", 1.3, 1.4),
            ("I'm", 1.5, 1.7),
            ("sure", 1.7, 1.9),
            ("I'd", 1.9, 2.1),
            ("know", 2.1, 2.3),
            (".", 2.3, 2.4),
        ],
        "eng",
    );
    assert!(
        output.contains(" I "),
        "standalone English pronoun 'I' must be preserved uppercase: {output}"
    );
    assert!(
        output.contains("I'm"),
        "English contraction 'I'm' must be preserved uppercase: {output}"
    );
    assert!(
        output.contains("I'd"),
        "English contraction 'I'd' must be preserved uppercase: {output}"
    );
    assert!(
        !output.contains(" i "),
        "lowercase standalone 'i' must NOT appear in English CHAT output: {output}"
    );
}

/// Three adjacent identical unigrams serialize as `w [/] w [/] w`,
/// not as the phrase form `<w w> [/] w`.
///
/// The utterance-initial-cap rule (2026-04-23) uppercases the
/// first non-retrace word in an English utterance, so the
/// emitted form is `a [/] a [/] A`, retrace copies preserve the
/// speaker's original lowercase `a`s, and the final "real"
/// word gets the sentence-initial capitalization.
#[test]
fn user_report_single_word_triple_repetition_emits_separate_retraces() {
    let output = run_transcribe_pipeline(
        &[
            ("a", 0.0, 0.2),
            ("a", 0.2, 0.4),
            ("a", 0.4, 0.6),
            (".", 0.6, 0.7),
        ],
        "eng",
    );
    assert!(
        output.contains("a [/] a [/] A"),
        "triple single-word repetition must emit 'a [/] a [/] A': {output}"
    );
    assert!(
        !output.contains("<a a>"),
        "must NOT emit the multi-word group form '<a a>' for a unigram repetition: {output}"
    );
}

/// Proper nouns supplied uppercase by the ASR provider are preserved
/// through the pipeline. (Promoting sentence-initial function words
/// like "well"/"and"/"of" is out of scope, handled elsewhere.)
#[test]
fn user_report_proper_nouns_preserved_from_rev_ai() {
    let output = run_transcribe_pipeline(
        &[
            ("well", 0.0, 0.2),
            ("I", 0.2, 0.3),
            ("hate", 0.3, 0.5),
            ("to", 0.5, 0.6),
            ("give", 0.6, 0.8),
            ("away", 0.8, 1.0),
            ("my", 1.0, 1.1),
            ("age", 1.1, 1.3),
            ("Sarah", 1.3, 1.6),
            (".", 1.6, 1.7),
            ("I", 1.8, 1.9),
            ("live", 1.9, 2.1),
            ("in", 2.1, 2.2),
            ("Cincinnati", 2.2, 2.6),
            (".", 2.6, 2.7),
        ],
        "eng",
    );
    assert!(
        output.contains("Sarah"),
        "proper noun 'Sarah' must be preserved uppercase: {output}"
    );
    assert!(
        output.contains("Cincinnati"),
        "proper noun 'Cincinnati' must be preserved uppercase: {output}"
    );
    assert!(
        !output.contains("sarah"),
        "lowercase 'sarah' must NOT appear (proper noun): {output}"
    );
    assert!(
        !output.contains("cincinnati"),
        "lowercase 'cincinnati' must NOT appear (proper noun): {output}"
    );
}

// -------------------------------------------------------------------
// RED: Fundamental B witness + the reporter end-to-end canary
//
// Fundamental A (enforce validation at `ChatWordText` construction)
// is expressed in `asr_postprocess/asr_types.rs::tests`. Once A
// goes green, most symptom-level tests that constructed
// Rev.AI-shaped `AsrOutput`s containing `%` and expected parse-clean
// CHAT become redundant: `ChatWordText::try_from` will refuse
// them before `build_chat` is even reached.
//
// Two tests remain at this layer because they exercise properties
// A alone does not cover:
//
//   * `red_fund_b_digit_hyphenated_eng_emits_no_bare_digits`: the
//     end-to-end witness that `process_raw_asr` (Fundamental B)
//     respects the language-aware variant of the `ChatWordText`
//     invariant. The digit-hyphenated token `17-year-old` is
//     structurally legal (tree-sitter accepts it) but fails E220
//     for eng. This is the cleanest forcing function for the
//     language-aware construction policy and acts as B's witness.
//
//   * `red_reporter_c465e6e8_97c_repro_fixture`: the
//     "The reporter's bug stays fixed" regression canary. Three authentic
//     Rev.AI tokens with real timestamps; any future regression in
//     either A (structural) or B (pipeline postcondition) will
//     reopen this test.
//
// Four `%`-only symptom tests that previously lived here were
// deleted in the 2026-04-22 RED-suite sharpening pass; they
// duplicated the invariant Fundamental A expresses cleanly. See
// §4 for the triage rationale.
// -------------------------------------------------------------------

/// Build a one-speaker AsrOutput from (value, start_s, end_s) triples.
/// Small helper kept for the user canary; re-used if additional
/// end-to-end B-witnesses are added later.
fn asr_single_speaker(elements: &[(&str, f64, f64)]) -> asr_postprocess::AsrOutput {
    asr_postprocess::AsrOutput {
        monologues: vec![asr_postprocess::AsrMonologue {
            speaker: asr_postprocess::SpeakerIndex(0),
            elements: elements
                .iter()
                .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new(*v),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(*ts),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(*end_ts),
                    kind: asr_postprocess::AsrElementKind::Text,
                })
                .collect(),
        }],
    }
}

/// Run the full Rev.AI-facing chain: process_raw_asr → transcript builder
/// → build_chat → serialize → re-parse. Returns the serialized CHAT and
/// the parse errors the lenient parser emits on the re-parse.
fn asr_to_chat_roundtrip(
    output: &asr_postprocess::AsrOutput,
    lang: &str,
) -> (String, Vec<talkbank_model::ParseError>) {
    let utts = asr_postprocess::process_raw_asr(output, lang)
        .expect("test: ASR post-processing must not refuse this input");
    let desc = transcript_from_asr_utterances(
        &utts,
        &["PAR".to_string()],
        &[lang.to_string()],
        None,
        false,
    )
    .expect("test: transcript_from_asr_utterances should succeed")
    .description;
    let chat = build_chat(&desc).expect("build_chat should not fail on normalized input");
    let serialized = to_chat_string(&chat);
    let parser = TreeSitterParser::new().expect("grammar loads");
    let (_chat, errors) = crate::parse::parse_lenient(&parser, &serialized);
    (serialized, errors)
}

#[test]
fn red_fund_b_digit_hyphenated_eng_emits_no_bare_digits() {
    // Reproduces c465e6e8-97c line 669:
    //   *PAR1: And my 17-year-old he wants to go to Harvard .
    // tree-sitter accepts `17-year-old` as a single hyphenated word, so
    // this does NOT fail L0 parse. It fails E220 at talkbank-model
    // validation ("numeric digits not allowed" for eng). The
    // sanitization contract here is: digit-bearing tokens must be
    // normalized (spelled out or segmented) before emission to CHAT
    // for languages where E220 applies.
    let output = asr_single_speaker(&[
        ("my", 0.0, 0.2),
        ("17-year-old", 1341.565, 1342.245),
        ("son", 0.5, 0.7),
        (".", 0.7, 0.7),
    ]);
    let (serialized, parse_errors) = asr_to_chat_roundtrip(&output, "eng");
    assert!(
        parse_errors.is_empty(),
        "re-parse must produce no structural errors; got:\n{parse_errors:#?}\n\
         serialized:\n{serialized}"
    );

    // Re-parse and run the full validator under an eng context. This is
    // the true E220 check, it reads the ChatFile's word AST, applies
    // language-aware digit rules per word, and collects any violations.
    // Using the real machinery instead of string-matching digits makes
    // the test robust against surface-level artifacts such as utterance
    // timing suffixes (`0_700`) that are part of the CHAT line but not
    // part of word content.
    let parser = TreeSitterParser::new().expect("grammar loads");
    let (chat, _) = crate::parse::parse_lenient(&parser, &serialized);
    let validation_errors = talkbank_model::ErrorCollector::new();
    chat.validate(&validation_errors, TranscriptName::Anonymous);
    let validation_errs = validation_errors.into_vec();
    let e220s: Vec<_> = validation_errs
        .iter()
        .filter(|e| e.code.as_str() == "E220")
        .collect();
    assert!(
        e220s.is_empty(),
        "emitted CHAT must not fire E220 (numeric digits not allowed in eng) \
         on any word. Found {} E220 error(s):\n{e220s:#?}\nserialized:\n{serialized}",
        e220s.len()
    );
}

#[test]
fn red_reporter_c465e6e8_97c_end_to_end_canary() {
    // Full regression fixture: the exact three offending Rev.AI tokens
    // from a reporter's failing job c465e6e8-97c (file 545.mp4), run through
    // the same Rev.AI-facing chain that `transcribe` uses. Timings are
    // the authentic Rev.AI values captured at
    //   the private bug-repro fixture
    //     offending_asr_tokens.json
    // Context tokens surrounding each offender are synthesized so each
    // utterance has a terminator; only the three problematic tokens are
    // from the Rev.AI wire response.
    let output = asr_postprocess::AsrOutput {
        monologues: vec![
            asr_postprocess::AsrMonologue {
                // Speaker 1: "And my 17-year-old son ."
                speaker: asr_postprocess::SpeakerIndex(1),
                elements: [
                    ("and", 1340.4, 1340.6),
                    ("my", 1340.6, 1340.8),
                    ("17-year-old", 1341.565, 1342.245),
                    ("son", 1342.3, 1342.5),
                    (".", 1343.685, 1343.685),
                ]
                .iter()
                .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new(*v),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(*ts),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(*end_ts),
                    kind: asr_postprocess::AsrElementKind::Text,
                })
                .collect(),
            },
            asr_postprocess::AsrMonologue {
                // Speaker 0: "remember 80% of it ."
                speaker: asr_postprocess::SpeakerIndex(0),
                elements: [
                    ("remember", 1774.1, 1774.5),
                    ("80%", 1774.765, 1775.405),
                    ("of", 1775.405, 1775.5),
                    ("it", 1775.5, 1775.645),
                    (".", 1775.645, 1775.645),
                ]
                .iter()
                .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new(*v),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(*ts),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(*end_ts),
                    kind: asr_postprocess::AsrElementKind::Text,
                })
                .collect(),
            },
            asr_postprocess::AsrMonologue {
                // Speaker 0: "that other 20% ."
                speaker: asr_postprocess::SpeakerIndex(0),
                elements: [
                    ("that", 1775.9, 1776.3),
                    ("other", 1776.3, 1776.8),
                    ("20%", 1776.825, 1777.685),
                    (".", 1777.685, 1777.685),
                ]
                .iter()
                .map(|(v, ts, end_ts)| asr_postprocess::AsrElement {
                    value: asr_postprocess::AsrRawText::new(*v),
                    ts: asr_postprocess::AsrTimestampSecs::Observed(*ts),
                    end_ts: asr_postprocess::AsrTimestampSecs::Observed(*end_ts),
                    kind: asr_postprocess::AsrElementKind::Text,
                })
                .collect(),
            },
        ],
    };

    let utts = asr_postprocess::process_raw_asr(&output, "eng")
        .expect("test: ASR post-processing must not refuse this input");
    let desc = transcript_from_asr_utterances(
        &utts,
        &["PAR0".to_string(), "PAR1".to_string()],
        &["eng".to_string()],
        Some("545"),
        false,
    )
    .expect("test: transcript_from_asr_utterances should succeed")
    .description;
    let chat = build_chat(&desc).expect("build_chat should not fail");
    let serialized = to_chat_string(&chat);
    let parser = TreeSitterParser::new().expect("grammar loads");
    let (_chat, errors) = crate::parse::parse_lenient(&parser, &serialized);
    assert!(
        errors.is_empty(),
        "c465e6e8-97c fixture must reparse with zero parse errors \
         (currently fails with E316 on `80%`/`20%`); got {} error(s):\n{:#?}\n\
         serialized:\n{serialized}",
        errors.len(),
        errors
    );
    assert!(
        !serialized.contains('%'),
        "a reporter fixture: emitted CHAT must not contain bare `%`; \
         serialized:\n{serialized}"
    );
}

// ── 2026-04-23 transcribe-pipeline corrections end-to-end ──

/// All three 2026-04-23 English transcribe rules fire in the
/// full in-process transcribe pipeline (`process_raw_asr` →
/// `build_chat`): bare `i` uppercases, `Dr.` loses its period,
/// utterance-initial words get capitalized. Evidence that the
/// rules are wired into `finalize_utterances` and not just
/// unit-tested in isolation.
#[test]
fn english_transcribe_rules_fire_end_to_end() {
    let output = run_transcribe_pipeline(
        &[
            ("hello", 0.0, 0.3),
            (".", 0.3, 0.4),
            ("i", 0.5, 0.6),
            ("said", 0.6, 0.9),
            ("Dr.", 0.9, 1.2),
            ("Smith", 1.2, 1.5),
            ("arrived", 1.5, 1.9),
            (".", 1.9, 2.0),
            ("i'll", 2.1, 2.4),
            ("see", 2.4, 2.6),
            ("him", 2.6, 2.8),
            (".", 2.8, 2.9),
        ],
        "eng",
    );
    // Utterance 1: `Hello .` (utterance-initial cap).
    assert!(
        output.contains("Hello ."),
        "first utterance must be capitalized `Hello .`: {output}"
    );
    // Utterance 2: `I said Dr Smith arrived .` (I-cap on `i`,
    // period-strip on `Dr.`). `I` is already capitalized by
    // I-cap, so utterance-initial cap is a no-op there.
    assert!(
        output.contains("I said Dr Smith arrived ."),
        "second utterance must show I-cap + period-strip: {output}"
    );
    // Utterance 3: `I'll see him .` (I-cap on contraction).
    assert!(
        output.contains("I'll see him ."),
        "third utterance must show I-cap on contraction: {output}"
    );
    // Negative assertions: the rules must NOT have fired on
    // unrelated material.
    assert!(
        !output.contains(" i "),
        "bare `i` must have been uppercased: {output}"
    );
    assert!(
        !output.contains("Dr."),
        "title `Dr.` must have lost its period: {output}"
    );
}

// ASR-emitted CHAT-illegal characters must be sanitized at the
// ASR-postprocess layer rather than tanking the transcript at the
// build gate. Drill-down tests for the sanitization helper live in
// `asr_postprocess/tests.rs`.

#[test]
fn whisper_colon_token_survives_pipeline_via_sanitization() {
    // Whisper occasionally emits a bare `:` post-segment leak.
    let result = run_transcribe_to_description(
        &[
            ("hello", 0.0, 0.3),
            (":", 0.3, 0.4),
            ("world", 0.4, 0.7),
            (".", 0.7, 0.8),
        ],
        "eng",
    );
    assert!(
        result.is_ok(),
        "bare `:` ASR token must be sanitized at ASR-postprocess, \
         not error the whole transcribe run. got: {result:?}"
    );
}

#[test]
fn tencent_tilde_token_survives_pipeline_via_sanitization() {
    // Tencent occasionally emits bare `~` (a CHAT structural
    // separator).
    let result = run_transcribe_to_description(
        &[
            ("好", 0.0, 0.3),
            ("~", 0.3, 0.35),
            ("耐", 0.35, 0.7),
            ("。", 0.7, 0.8),
        ],
        "yue",
    );
    assert!(
        result.is_ok(),
        "bare `~` ASR token must be sanitized at ASR-postprocess, \
         not error the whole transcribe run. got: {result:?}"
    );
}

#[test]
fn exotic_unicode_glued_to_word_survives_pipeline_via_sanitization() {
    // Whisper occasionally glues exotic Unicode (Tibetan + Greek +
    // math) to real word content; the rest of the utterance should
    // survive even when the bad chars are stripped (or the token
    // dropped if it becomes empty).
    let result = run_transcribe_to_description(
        &[
            ("hello", 0.0, 0.3),
            ("ཌྷᾱ≡ᾱworld", 0.3, 0.6),
            (".", 0.6, 0.7),
        ],
        "eng",
    );
    assert!(
        result.is_ok(),
        "ASR token with embedded CHAT-illegal Unicode must be \
         sanitized at ASR-postprocess, not error the whole \
         transcribe run. got: {result:?}"
    );
}

/// Non-English input is untouched by the 2026-04-23 English
/// rules. Language gate is the sole guard.
#[test]
fn english_transcribe_rules_skip_other_languages() {
    let output = run_transcribe_pipeline(
        &[
            ("ho", 0.0, 0.2),
            ("visto", 0.2, 0.5),
            ("i", 0.5, 0.6),
            ("bambini", 0.6, 1.0),
            (".", 1.0, 1.1),
        ],
        "ita",
    );
    // Italian `ho` must NOT be capitalized; Italian `i` (plural
    // masculine article) must NOT be uppercased.
    assert!(
        output.contains("ho visto i bambini ."),
        "Italian output must be untouched by English rules: {output}"
    );
    assert!(
        !output.contains(" I "),
        "Italian `i` must not be uppercased: {output}"
    );
}

/// An invalid transcript-level language code must surface as a build
/// error naming the offending code, never a panic and never a silently
/// wrong @Languages header. (chatter 0.3.0 made LanguageCode
/// construction fallible; this pins our boundary handling of it.)
#[test]
fn test_build_chat_invalid_language_code_is_a_build_error() {
    let desc = TranscriptDescription {
        langs: vec![String::new()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: None,
        media_type: None,
        media_status: None,
        utterances: vec![],
        write_wor: false,
    };
    let err = build_chat(&desc).expect_err("empty language code must be rejected");
    // Match the VARIANT, not the prose: that is what the typed error buys.
    assert!(
        matches!(err, BuildChatError::LanguageCode { .. }),
        "error must name the language problem, got: {err}"
    );
}

/// Programmatically assembled CHAT must carry the same typed participant
/// metadata as parsed CHAT: `ChatFile::new` deliberately leaves the
/// derived participant map empty (parser-intermediate semantics), so
/// `build_chat` must populate it itself. Downstream consumers that
/// consult `chat_file.participants` instead of re-parsing header lines
/// would otherwise see an empty roster on ASR-built files.
#[test]
fn built_chat_file_carries_typed_participant_map() {
    let utterances = vec![
        asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(0),
            words: vec![
                asr_postprocess::AsrWord::new("hello", Some(100), Some(400)),
                asr_postprocess::AsrWord::new(".", None, None),
            ],
            lang: None,
        },
        asr_postprocess::Utterance {
            speaker: asr_postprocess::SpeakerIndex(1),
            words: vec![
                asr_postprocess::AsrWord::new("hi", Some(500), Some(800)),
                asr_postprocess::AsrWord::new(".", None, None),
            ],
            lang: None,
        },
    ];
    let ids = vec!["PAR".to_string(), "INV".to_string()];
    let desc = transcript_from_asr_utterances(
        &utterances,
        &ids,
        &["eng".to_string()],
        Some("test.mp3"),
        false,
    )
    .expect("test: transcript_from_asr_utterances should succeed")
    .description;

    let chat_file = build_chat(&desc).unwrap();

    assert_eq!(
        chat_file.participants.len(),
        2,
        "built ChatFile must populate its typed participant map"
    );
    let par = chat_file
        .participants
        .values()
        .find(|p| p.code.as_str() == "PAR")
        .expect("PAR participant present");
    assert_eq!(par.role.as_str(), "Participant");
    assert!(
        chat_file
            .participants
            .values()
            .any(|p| p.code.as_str() == "INV"),
        "INV participant present"
    );
}

/// A media name that `@Media` cannot represent must FAIL, not vanish.
///
/// `@Media` delimits the filename with a comma, so `interview,part2` is not a
/// writable name. When chatter made `MediaHeader::new` take a parsed
/// `MediaFilename`, the migration absorbed the new failure with `.ok()`, which
/// turned an unusable name into `None` -- the same value that means "the
/// caller named no media". The pipeline then reported success and wrote a
/// transcript with NO `@Media` line and no diagnostic, silently losing the
/// audio link. Losing a media link without saying so is the exact failure this
/// whole area was being repaired for.
#[test]
fn an_unrepresentable_media_name_is_an_error_not_a_missing_header() {
    let mut desc = TranscriptDescription {
        langs: vec!["eng".to_string()],
        participants: vec![ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: "Participant".to_string(),
            corpus: String::new(),
        }],
        media_name: Some("interview,part2".to_string()),
        media_type: Some("audio".to_string()),
        media_status: None,
        utterances: vec![UtteranceDesc {
            speaker: "PAR".to_string(),
            words: Some(vec![wd("hello", None, None), wd(".", None, None)]),
            text: None,
            start_ms: None,
            end_ms: None,
            lang: None,
        }],
        write_wor: false,
    };

    let err = build_chat(&desc).expect_err("a comma-bearing media name must be rejected");
    assert!(
        matches!(err, BuildChatError::MediaFilename(_)),
        "the error must name the header it could not write: {err}"
    );

    // And the ordinary name still works, so the check is not simply refusing.
    desc.media_name = Some("interview_part2".to_string());
    let chat = build_chat(&desc).expect("a representable media name still builds");
    assert!(to_chat_string(&chat).contains("@Media:\tinterview_part2, audio"));
}

// ---------------------------------------------------------------------------
// A word the language gate refuses is REPORTED, never only logged
// ---------------------------------------------------------------------------
//
// The verbatim-for-review policy above is unchanged and deliberately so: an
// ASR surface is the provider's observation, and the pipeline does not invent
// what a human is meant to adjudicate. What was missing is the other half.
// The gate knew the token was language-invalid, said so to `tracing::warn!`,
// and returned a value indistinguishable from a token that had PASSED. A log
// line is not a return value: nothing downstream, no operator report, no test
// double could tell the two apart, so 2026-09-03 shipped `Www` (E241) and
// `b2` (E220) into a corpus with no record anywhere that the pipeline had
// already spotted them.

/// Collect the codes reported for one token's admission failure.
fn language_invalid_codes(result: &AsrTranscript, text: &str) -> Vec<String> {
    result
        .language_invalid
        .iter()
        .filter(|word| word.text == text)
        .flat_map(|word| word.parse_errors.iter())
        .map(|error| error.code.as_str().to_owned())
        .collect()
}

/// E241. Rev.AI returned the surface `Www`, whose case-folded form is a
/// reserved untranscribed marker, so CHAT reads it as `www` spelled wrongly.
#[test]
fn a_reserved_marker_collision_is_reported_by_the_gate() {
    let result = run_transcribe_to_description(&[("Www", 0.0, 0.5), (".", 0.5, 0.5)], "eng")
        .expect("test: the token is structurally legal, so the file still builds");

    assert_eq!(
        language_invalid_codes(&result, "Www"),
        vec!["E241"],
        "the gate must hand back what it refused: {:?}",
        result.language_invalid,
    );

    let words: Vec<&str> = result
        .description
        .utterances
        .iter()
        .flat_map(|u| u.words.as_deref().unwrap_or(&[]))
        .map(|w| w.text.as_str())
        .collect();
    assert!(
        words.contains(&"Www"),
        "the provider's surface still ships verbatim for review: {words:?}",
    );
}

/// E220. Rev.AI returned `b2`; English words may not contain digits.
#[test]
fn a_digit_bearing_token_is_reported_by_the_gate() {
    let result = run_transcribe_to_description(
        &[
            ("we", 0.0, 0.2),
            ("took", 0.2, 0.4),
            ("b2", 0.4, 0.7),
            (".", 0.7, 0.7),
        ],
        "eng",
    )
    .expect("test: the token is structurally legal, so the file still builds");

    assert_eq!(
        language_invalid_codes(&result, "b2"),
        vec!["E220"],
        "the gate must hand back what it refused: {:?}",
        result.language_invalid,
    );
}

/// A clean transcript reports nothing, so a non-empty report always means
/// something really did fail the language gate.
#[test]
fn a_clean_transcript_reports_no_language_invalid_words() {
    let result = run_transcribe_to_description(
        &[("the", 0.0, 0.2), ("dog", 0.2, 0.4), (".", 0.4, 0.4)],
        "eng",
    )
    .expect("test: a clean transcript builds");
    assert!(
        result.language_invalid.is_empty(),
        "unexpected report: {:?}",
        result.language_invalid,
    );
}
