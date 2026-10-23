use super::*;
use crate::chat_ops::fa::WordGapHealing;
use crate::engine_reports::FaCacheNamespace;
use crate::fa::{FaInputDocument, input::FaAdmission};
use talkbank_model::Location;
use talkbank_model::alignment::helpers::{WordItemMut, walk_words_mut};
use talkbank_model::model::{Bullet, FileStem, TranscriptName};

fn attempt(words: &str) -> (talkbank_model::ChatFile, FaAdmission) {
    attempt_with_options(words, "")
}

fn attempt_with_options(words: &str, options: &str) -> (talkbank_model::ChatFile, FaAdmission) {
    let source = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n{options}@ID:\teng|test|PAR|||||Participant|||\n@Media:\tinput, audio, unlinked\n*PAR:\t{words} .\n@End\n"
    );
    let read = crate::fa::read_fa_source_named(
        &source,
        TranscriptName::Named(
            FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
        ),
        crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
    )
    .expect("complete named source admission");
    match read.attempt() {
        FaInputDocument::Active(active) => (active.chat_file, active.admission),
        FaInputDocument::Preserved(_) => panic!("ordinary source must align"),
    }
}

fn time_words(file: &mut talkbank_model::ChatFile, only_last: bool) -> usize {
    for line in &mut file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        let words = lexical_shape(utterance).len();
        let mut ordinal = 0;
        walk_words_mut(
            utterance.main.content.content.as_mut_slice(),
            None,
            &mut |leaf| {
                let word = match leaf {
                    WordItemMut::Word(word) => word,
                    WordItemMut::ReplacedWord(replaced) => &mut replaced.word,
                    WordItemMut::Separator(_) => return,
                };
                if counts_for_tier(word, TierDomain::Wor) {
                    if !only_last || ordinal + 1 == words {
                        word.inline_bullet = Some(if only_last {
                            Bullet::new(Location::Unlocated, 19_980, 20_000)
                        } else {
                            Bullet::new(
                                Location::Unlocated,
                                100 + ordinal as u64 * 100,
                                180 + ordinal as u64 * 100,
                            )
                        });
                    }
                    ordinal += 1;
                }
            },
        );
        utterance.main.content.bullet = Some(Bullet::new(Location::Unlocated, 0, 20_000));
        crate::chat_ops::fa::add_wor_tier(utterance);
        return words;
    }
    panic!("fixture must have an utterance");
}

fn result(file: talkbank_model::ChatFile) -> FaResult<talkbank_model::ChatFile> {
    FaResult::without_groups(
        file,
        WordGapHealing::PreserveMeasured,
        "test_engine",
        &FaCacheNamespace::for_test("test-build"),
    )
}

/// One timed word out of nine is a partial result: written, with the
/// account of the eight untimed words as the file's shortfall, never a
/// refusal and never a clean success.
///
/// RED FIRST for the cause: this result carries no grouping decision and no
/// request, so nothing the run recorded explains the eight words. The cause
/// was `no_usable_timing`, a claim about an aligner that was never asked; it
/// is now `unrecorded`.
#[test]
fn eight_missing_timings_are_written_with_their_account() {
    let (mut file, admission) = attempt("one two three four five six seven eight nine");
    assert_eq!(time_words(&mut file, true), 9);
    let admitted = admission
        .finish(result(file))
        .expect("partial timing is written, not refused");
    let shortfalls = admitted.into_document().shortfalls;
    insta::assert_json_snapshot!(shortfalls, @r#"
    [
      {
        "kind": "timing_incomplete",
        "required_words": 9,
        "untimed_words": 8,
        "untimed_utterances": 1,
        "first_untimed": [
          {
            "utterance": 1,
            "words": 9,
            "untimed_words": 8,
            "cause": {
              "kind": "unrecorded"
            }
          }
        ]
      }
    ]
    "#);
    insta::assert_snapshot!(shortfalls[0].to_string(), @"timing incomplete: 8 of 9 words in 1 utterance(s) have no timing and were written without it (first: utterance 1, 8 of 9 words, no cause recorded)");
}

/// A request that covered the utterance is what explains its untimed words,
/// and the account counts what the requests yielded instead of naming a
/// cause: here one answered request and one left unaligned.
#[test]
fn requested_untimed_words_count_what_their_requests_yielded() {
    use crate::types::traces::{FaEvidenceSourceTrace, FaGroupSpanTrace};
    let (mut file, admission) = attempt("one two three");
    assert_eq!(time_words(&mut file, true), 3);
    let mut output = result(file);
    for source in [
        FaEvidenceSourceTrace::Inference,
        FaEvidenceSourceTrace::Unaligned,
    ] {
        output
            .group_evidence
            .push(crate::types::results::FaGroupEvidence {
                group: crate::types::results::FaGroupContents {
                    audio_start_ms: crate::api::DurationMs(0),
                    audio_end_ms: crate::api::DurationMs(20_000),
                    utterances: vec![talkbank_model::UtteranceIdx::new(0)],
                    words: Vec::new(),
                    word_ids: Vec::new(),
                    span: FaGroupSpanTrace::Single {
                        source,
                        cache_key: "key".to_owned(),
                    },
                },
                pre_injection_timings: Vec::new(),
            });
    }
    let shortfalls = admission
        .finish(output)
        .expect("partial timing is written")
        .into_document()
        .shortfalls;
    insta::assert_json_snapshot!(shortfalls, @r#"
    [
      {
        "kind": "timing_incomplete",
        "required_words": 3,
        "untimed_words": 2,
        "untimed_utterances": 1,
        "first_untimed": [
          {
            "utterance": 1,
            "words": 3,
            "untimed_words": 2,
            "cause": {
              "kind": "requested",
              "answered": 1,
              "unaligned": 1,
              "unrequested": 0
            }
          }
        ]
      }
    ]
    "#);
}

/// No timed word at all is still a partial result, written with its
/// account: the words and the source structure are the output's value.
#[test]
fn a_document_with_no_word_timing_is_written_with_its_account() {
    let (file, admission) = attempt("hello world");
    let admitted = admission
        .finish(result(file))
        .expect("untimed output is written, not refused");
    let shortfalls = admitted.into_document().shortfalls;
    let [
        crate::api::OutputShortfallRecord::TimingIncomplete {
            required_words: 2,
            untimed_words: 2,
            untimed_utterances: 1,
            first_untimed,
        },
    ] = shortfalls.as_slice()
    else {
        panic!("expected one timing shortfall, got {shortfalls:?}");
    };
    assert_eq!(first_untimed.len(), 1);
}

#[test]
fn completed_timing_retains_its_payload_through_output_admission() {
    let (mut file, admission) = attempt("hello world");
    assert_eq!(time_words(&mut file, false), 2);
    let admitted = admission
        .finish(result(file))
        .expect("complete timing admission");
    let (output, _timeline) = admitted.into_document_and_timeline();
    assert!(output.shortfalls.is_empty(), "complete timing is clean");
    assert!(output.exclusions.is_empty(), "and nothing was left out");
}

/// The actual CHAT boundary, not the richer in-memory timing model.
#[test]
fn unprinted_main_word_timings_cannot_certify_written_completion() {
    // This describes existing omitted-tier output, not a policy endorsement.
    for options in ["", "@Options:\tCA\n"] {
        let (mut file, admission) = attempt_with_options("hello world", options);
        time_words(&mut file, false);
        for line in &mut file.lines {
            if let Line::Utterance(utterance) = line {
                utterance.dependent_tiers.retain(|entry| {
                    !matches!(entry.tier, talkbank_model::model::DependentTier::Wor(_))
                });
            }
        }
        let output = admission
            .finish(result(file))
            .expect("partial output")
            .into_document();
        let (reread, errors) = batchalign_transform::parse::parse_lenient(
            &crate::chat_parser(),
            output.document.as_str(),
        );
        assert!(errors.is_empty(), "written CHAT parses: {errors:?}");
        let utterance = reread.utterances().next().expect("utterance");
        assert!(utterance.wor_tier().is_none());
        assert!(
            collect_existing_fa_word_timings(utterance)
                .iter()
                .all(Option::is_none)
        );
        assert!(
            matches!(
                output.shortfalls.as_slice(),
                [crate::api::OutputShortfallRecord::TimingIncomplete {
                    required_words: 2,
                    untimed_words: 2,
                    ..
                }]
            ),
            "diagnostics must describe the written CHAT: {:?}",
            output.shortfalls
        );
    }
}

#[derive(Debug, PartialEq, serde::Serialize)]
struct WrittenWord {
    text: String,
    interval: Option<WrittenInterval>,
}

#[derive(Debug, PartialEq, serde::Serialize)]
struct WrittenInterval {
    start_ms: u64,
    end_ms: u64,
}

fn written_words(file: &talkbank_model::ChatFile) -> Vec<WrittenWord> {
    file.utterances()
        .flat_map(|utterance| {
            utterance
                .wor_tier()
                .into_iter()
                .flat_map(|tier| tier.words())
        })
        .map(|word| WrittenWord {
            text: word.cleaned_text().to_owned(),
            interval: word.inline_bullet.as_ref().map(|bullet| WrittenInterval {
                start_ms: bullet.timing.start_ms,
                end_ms: bullet.timing.end_ms,
            }),
        })
        .collect()
}

#[test]
fn real_injection_retains_exact_word_intervals_through_written_chat() {
    use crate::chat_ops::fa::{WordTiming, inject_timings_for_utterance};
    let (mut file, admission) = attempt("hello [: hi] &-uh_huh");
    let utterance = (&mut file.lines)
        .into_iter()
        .find_map(|line| match line {
            Line::Utterance(utterance) => Some(utterance),
            _ => None,
        })
        .expect("utterance");
    let timings = [
        WordTiming::fixture(100, 180),
        WordTiming::fixture(200, 280),
        WordTiming::fixture(300, 380),
    ];
    let mut cursor = 0;
    let _injected = inject_timings_for_utterance(utterance, &timings, &mut cursor);
    assert_eq!(cursor, 3, "compound filler consumes two model units");
    utterance.main.content.bullet = Some(Bullet::new(Location::Unlocated, 100, 380));
    crate::chat_ops::fa::add_wor_tier(utterance);
    let expected = written_words(&file);
    let output = admission
        .finish(result(file))
        .expect("complete written timing")
        .into_document();
    assert!(output.shortfalls.is_empty());
    // Exercise the writer's optional final transform as well as serialization.
    let document = output
        .document
        .with_provenance_injected(&crate::provenance::align_provenance(
            &crate::api::LanguageCode3::eng(),
            &FaCacheNamespace::for_test("test-build"),
            &crate::provenance::UtrContribution::NotRun,
            true,
            false,
            crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
        ))
        .expect("provenance preserves the timing census")
        .with_abbreviations_merged()
        .expect("no abbreviation change");
    let (reread, errors) =
        batchalign_transform::parse::parse_lenient(&crate::chat_parser(), document.as_str());
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(written_words(&reread), expected);
    insta::assert_json_snapshot!(expected, @r#"
    [
      {
        "text": "hello",
        "interval": {
          "start_ms": 100,
          "end_ms": 180
        }
      },
      {
        "text": "uh_huh",
        "interval": {
          "start_ms": 200,
          "end_ms": 380
        }
      }
    ]
    "#);
}

#[test]
fn durable_partial_tier_is_authoritative_over_working_word_timings() {
    let (mut file, admission) = attempt("hello world");
    time_words(&mut file, true);
    for line in &mut file.lines {
        if let Line::Utterance(utterance) = line {
            // Add working timing AFTER the partial durable tier was generated.
            walk_words_mut(
                utterance.main.content.content.as_mut_slice(),
                None,
                &mut |leaf| {
                    if let WordItemMut::Word(word) = leaf
                        && word.inline_bullet.is_none()
                    {
                        word.inline_bullet = Some(Bullet::new(Location::Unlocated, 100, 180));
                    }
                },
            );
        }
    }
    let output = admission
        .finish(result(file))
        .expect("partial output")
        .into_document();
    let (reread, errors) =
        batchalign_transform::parse::parse_lenient(&crate::chat_parser(), output.document.as_str());
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(
        written_words(&reread)
            .iter()
            .filter(|word| word.interval.is_none())
            .count(),
        1
    );
    assert!(matches!(
        output.shortfalls.as_slice(),
        [crate::api::OutputShortfallRecord::TimingIncomplete {
            required_words: 2,
            untimed_words: 1,
            ..
        }]
    ));
}

#[test]
fn later_abbreviation_merge_cannot_invalidate_admitted_timing_census() {
    for include_wor in [true, false] {
        let (mut file, admission) = attempt("F B I");
        time_words(&mut file, false);
        if !include_wor {
            for line in &mut file.lines {
                if let Line::Utterance(utterance) = line {
                    utterance.dependent_tiers.retain(|entry| {
                        !matches!(entry.tier, talkbank_model::model::DependentTier::Wor(_))
                    });
                }
            }
        }
        let output = admission
            .finish(result(file))
            .expect("admitted output")
            .into_document();
        assert!(
            output.document.with_abbreviations_merged().is_err(),
            "a post-completion lexical change cannot retain its old census (wor={include_wor})"
        );
    }
}

#[test]
fn a_different_valid_lexical_output_cannot_discharge_source_obligations() {
    let (_, admission) = attempt("hello world");
    for words in ["hello", "hello there", "hello world now"] {
        let (mut file, _) = attempt(words);
        time_words(&mut file, false);
        let Err(error) = admission.clone().finish(result(file)) else {
            panic!("source drift: {words}");
        };
        assert!(matches!(
            error,
            ServerError::AlignmentCompletion(AlignmentCompletionFailure::SourceChanged { .. })
        ));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
    }
}

#[test]
fn typed_compound_fillers_and_replacements_use_chat_word_obligations() {
    let (mut file, admission) = attempt("&-uh_huh hello [: hi] &-uh");
    assert_eq!(
        time_words(&mut file, false),
        3,
        "compound filler parts are model labels, not separate source words"
    );
    assert!(
        admission
            .finish(result(file))
            .expect("complete")
            .into_document()
            .shortfalls
            .is_empty()
    );
}

/// Admit a two-utterance source: spoken words with a bullet, then a
/// `[+ diary]` note, with the bullet `diary_bullet` if given.
fn attempt_with_diary(
    diary_bullet: &str,
    policy: crate::chat_ops::fa::MainBulletPolicy,
) -> (talkbank_model::ChatFile, FaAdmission) {
    let source = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n@ID:\teng|test|MOT|||||Mother|||\n@Media:\tinput, audio\n*MOT:\tgo there now . \u{15}0_20000\u{15}\n*MOT:\twent to the park . [+ diary]{diary_bullet}\n@End\n"
    );
    let read = crate::fa::read_fa_source_named(
        &source,
        TranscriptName::Named(
            FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
        ),
        policy,
    )
    .expect("a diary note is admitted");
    match read.attempt() {
        FaInputDocument::Active(active) => (active.chat_file, active.admission),
        FaInputDocument::Preserved(_) => panic!("ordinary source must align"),
    }
}

fn diary_utterance(file: &mut talkbank_model::ChatFile) -> &mut Utterance {
    (&mut file.lines)
        .into_iter()
        .filter_map(|line| match line {
            Line::Utterance(utterance) => Some(utterance.as_mut()),
            _ => None,
        })
        .nth(1)
        .expect("the diary note is the second utterance")
}

/// A `[+ diary]` note is written untimed and listed as an exclusion,
/// never as a shortfall (ruling of 2026-10-07: an intentional exclusion
/// must not count against a file). With all speech timed, the file is
/// clean: no shortfall, and the note in the separate exclusion channel.
#[test]
fn a_diary_note_is_an_exclusion_and_does_not_diagnose_the_file() {
    let (mut file, admission) =
        attempt_with_diary("", crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY);
    assert_eq!(time_words(&mut file, false), 3);
    let crate::fa::AlignedOutput {
        document,
        shortfalls,
        exclusions,
    } = admission
        .finish(result(file))
        .expect("the note is written untimed")
        .into_document();
    assert!(shortfalls.is_empty(), "{shortfalls:?}");
    assert_eq!(
        crate::pipeline::post_validate::OutputReport::of(&document.into(), &shortfalls),
        crate::pipeline::post_validate::OutputReport::Clean,
        "a file whose only untimed utterance is a note is clean"
    );
    insta::assert_json_snapshot!(exclusions, @r#"
    [
      {
        "kind": "not_in_recording",
        "excluded_utterances": 1,
        "excluded_words": 4,
        "first_excluded": [
          {
            "utterance": 2,
            "words": 4,
            "postcode": "diary"
          }
        ]
      }
    ]
    "#);
    insta::assert_snapshot!(exclusions[0].to_string(), @"left out of alignment by design: 1 utterance(s) (4 words) marked as not in the recording were not looked for in it (first: utterance 2, [+ diary])");
}

/// Untimed speech beside a note: the speech is the shortfall, counted
/// over the words owed timing only, and the note is still an exclusion,
/// not a cause in the untimed account.
#[test]
fn untimed_speech_beside_a_note_is_a_shortfall_that_does_not_count_the_note() {
    let (file, admission) = attempt_with_diary("", crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY);
    let crate::fa::AlignedOutput {
        shortfalls,
        exclusions,
        ..
    } = admission
        .finish(result(file))
        .expect("untimed speech is written with its account")
        .into_document();
    let [
        crate::api::OutputShortfallRecord::TimingIncomplete {
            required_words: 3,
            untimed_words: 3,
            untimed_utterances: 1,
            first_untimed,
        },
    ] = shortfalls.as_slice()
    else {
        panic!("expected the speech's shortfall alone, got {shortfalls:?}");
    };
    assert_eq!(first_untimed[0].utterance, 1);
    assert!(matches!(
        exclusions.as_slice(),
        [crate::api::OutputExclusionRecord::NotInRecording {
            excluded_utterances: 1,
            ..
        }]
    ));
}

/// The input's bullet on a diary note: admission removes it from the
/// working model under every policy, so no stage reads it as evidence.
/// Under `derive` the output may not carry one; under `keep` it may carry
/// exactly the given one (which imposition restores) and nothing else.
#[test]
fn a_diary_notes_given_bullet_is_kept_only_where_the_policy_keeps_given_bullets() {
    use crate::chat_ops::fa::MainBulletPolicy;
    let given = " \u{15}20000_21000\u{15}";
    for policy in [
        MainBulletPolicy::DeriveFromWords,
        MainBulletPolicy::KeepGiven,
    ] {
        let (mut file, _) = attempt_with_diary(given, policy);
        assert!(
            diary_utterance(&mut file).main.content.bullet.is_none(),
            "{policy:?}: the working model never holds it"
        );
    }

    let (mut file, admission) = attempt_with_diary(given, MainBulletPolicy::KeepGiven);
    time_words(&mut file, false);
    diary_utterance(&mut file).main.content.bullet =
        Some(Bullet::new(Location::Unlocated, 20_000, 21_000));
    assert!(
        admission.finish(result(file)).is_ok(),
        "keep: the given bullet, restored, is the input's own"
    );

    for (policy, bullet) in [
        (
            MainBulletPolicy::DeriveFromWords,
            Bullet::new(Location::Unlocated, 20_000, 21_000),
        ),
        (
            MainBulletPolicy::KeepGiven,
            Bullet::new(Location::Unlocated, 20_000, 22_000),
        ),
    ] {
        let (mut file, admission) = attempt_with_diary(given, policy);
        time_words(&mut file, false);
        diary_utterance(&mut file).main.content.bullet = Some(bullet);
        let Err(error) = admission.finish(result(file)) else {
            panic!("{policy:?}: a bullet the policy does not keep is invented timing");
        };
        assert!(matches!(
            error,
            ServerError::AlignmentCompletion(
                AlignmentCompletionFailure::OffRecordUtteranceTimed { .. }
            )
        ));
    }
}

/// Under `keep`, a diary note's given bullet is restored exactly but
/// orders nothing, so timing aligned around it can conflict with it (here
/// the same speaker's aligned utterance overlaps it). The output cannot
/// be valid; the refusal names the note and the remedy, as the input and
/// policy conflict it is, not an internal fault.
#[test]
fn a_kept_diary_bullet_that_conflicts_with_aligned_timing_is_named() {
    let source = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n@ID:\teng|test|MOT|||||Mother|||\n@Media:\tinput, audio\n*MOT:\tgo there now .\n*MOT:\twent to the park . [+ diary] \u{15}1000_2000\u{15}\n@End\n";
    let read = crate::fa::read_fa_source_named(
        source,
        TranscriptName::Named(
            FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
        ),
        crate::chat_ops::fa::MainBulletPolicy::KeepGiven,
    )
    .expect("the note's bullet is the input's only timing, and admitted");
    let FaInputDocument::Active(active) = read.attempt() else {
        panic!("ordinary source must align");
    };
    let (mut file, admission) = (active.chat_file, active.admission);
    // The spoken utterance aligned across the note's given extent.
    time_words(&mut file, false);
    diary_utterance(&mut file).main.content.bullet =
        Some(Bullet::new(Location::Unlocated, 1_000, 2_000));
    let Err(error) = admission.finish(result(file)) else {
        panic!("an overlapping same-speaker bullet cannot be written");
    };
    assert!(
        matches!(
            &error,
            ServerError::KeptOffRecordBulletConflict { utterances } if utterances == &[2]
        ),
        "{error}"
    );
    assert_eq!(
        crate::runner::util::classify_server_error(&error),
        crate::scheduling::FailureCategory::Validation
    );
}

/// A timed word on a diary note is a placement nothing in the recording
/// supports: an internal fault, never output.
#[test]
fn a_timed_word_on_a_diary_note_is_a_producer_fault() {
    let (mut file, admission) =
        attempt_with_diary("", crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY);
    time_words(&mut file, false);
    let mut first = true;
    walk_words_mut(
        diary_utterance(&mut file)
            .main
            .content
            .content
            .as_mut_slice(),
        None,
        &mut |leaf| {
            if let WordItemMut::Word(word) = leaf
                && std::mem::take(&mut first)
            {
                word.inline_bullet = Some(Bullet::new(Location::Unlocated, 20_100, 20_400));
            }
        },
    );
    let Err(error) = admission.finish(result(file)) else {
        panic!("a timed diary word must not be written");
    };
    assert!(matches!(
        error,
        ServerError::AlignmentCompletion(
            AlignmentCompletionFailure::OffRecordUtteranceTimed { .. }
        )
    ));
    assert_eq!(
        crate::runner::util::classify_server_error(&error),
        crate::scheduling::FailureCategory::System
    );
}

#[test]
fn an_untranscribed_turn_has_no_lexical_timing_obligation() {
    let (file, admission) = attempt("xxx");
    assert!(
        admission
            .finish(result(file))
            .expect("nothing to time")
            .into_document()
            .shortfalls
            .is_empty()
    );
}
