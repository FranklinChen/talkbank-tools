#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::chat_ops::fa::*;
use talkbank_model::{UtteranceIdx, WordIdx};
use talkbank_parser::TreeSitterParser;
use talkbank_parser_tests::test_error::ParseProductTestExt;

fn timing(start: u64, end: u64) -> Option<WordTiming> {
    Some(
        WordTiming::new(
            start,
            end,
            origin::Origin::TranscriptBullet,
            origin::Origin::TranscriptBullet,
        )
        .unwrap(),
    )
}

#[test]
fn exact_source_retry_recovers_missing_word_without_overwriting_primary() {
    let mut chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\tyes okay . \u{15}100_200\u{15}\n@End\n"
    ).expect_built();
    let authority = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).unwrap();
    let recording = Recording::of_duration(Ms(400)).unwrap();
    let sources = authority.supplied_retry_sources(&chat, &recording).unwrap();
    let primary = FaGroup::test_fixture(
        TimeSpan::new(100, 400),
        ["yes", "okay"]
            .into_iter()
            .enumerate()
            .map(|(i, text)| FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(i),
                text: text.into(),
            })
            .collect(),
        vec![UtteranceIdx::new(0)],
    );
    let applied = apply_fa_results_with_projection_policy(
        &mut chat,
        &[primary],
        &[vec![timing(110, 150), timing(258, 318)]],
        FaProjection::new(
            FaProjectionPolicy::new(
                WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
                ExistingWorBoundaryPolicy::Preserve,
                EndOverlapPolicy::ClampAllAdjacent,
            ),
            authority,
        ),
        true,
    )
    .unwrap();
    let imposed = applied.then_impose(&mut chat).unwrap();
    let SuppliedRetryStep::Pending(pending) = SuppliedRetrySession::new(
        imposed,
        &mut chat,
        sources
            .into_sources()
            .into_iter()
            .map(|source| (source, ())),
    )
    .unwrap()
    .next() else {
        panic!("missing source-fit word must issue one obligation")
    };
    let (pending, ()) = pending.start();
    assert_eq!(
        (
            pending.source().window().audio_start().get(),
            pending.source().window().end().get()
        ),
        (100, 200)
    );
    assert_eq!(
        pending.source().words().len(),
        2,
        "complete utterance is requested"
    );
    let SuppliedRetryStep::Complete(imposed) = pending
        .complete(SuppliedRetryAnswer::Timings {
            timings: vec![timing(140, 160), timing(170, 190)],
            source: "controlled exact source answer".into(),
        })
        .next()
    else {
        panic!("no repeated retry transition")
    };
    let _finalized = imposed.then_finalize(BulletRepairPolicy::Disabled).unwrap();
    let mut spans = Vec::new();
    walk_words(
        &chat.utterances().next().unwrap().main.content.content,
        None,
        &mut |word| {
            if let WordItem::Word(word) = word {
                spans.push(get_word_timing(word));
            }
        },
    );
    assert_eq!(
        spans,
        vec![Some(TimeSpan::new(110, 150)), Some(TimeSpan::new(170, 190))]
    );
}

fn projected(
    main: &str,
    primary: Vec<Option<WordTiming>>,
) -> (ChatFile, FaApplied, SuppliedRetrySources) {
    let mut chat = TreeSitterParser::new().unwrap().parse_chat_file(&format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\t{main} . \u{15}100_200\u{15}\n@End\n"
    )).expect_built();
    let authority = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).unwrap();
    let sources = authority
        .supplied_retry_sources(&chat, &Recording::of_duration(Ms(400)).unwrap())
        .unwrap();
    let group = FaGroup::test_fixture(
        TimeSpan::new(100, 400),
        sources.sources()[0].words().to_vec(),
        vec![UtteranceIdx::new(0)],
    );
    let applied = apply_fa_results_with_projection_policy(
        &mut chat,
        &[group],
        &[primary],
        FaProjection::new(
            FaProjectionPolicy::new(
                WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
                ExistingWorBoundaryPolicy::Preserve,
                EndOverlapPolicy::ClampAllAdjacent,
            ),
            authority,
        ),
        true,
    )
    .unwrap();
    (chat, applied, sources)
}

fn spans(chat: &ChatFile) -> Vec<Option<TimeSpan>> {
    let mut spans = Vec::new();
    walk_words(
        &chat.utterances().next().unwrap().main.content.content,
        None,
        &mut |item| {
            let word = match item {
                WordItem::Word(word) => word,
                WordItem::ReplacedWord(word) => &word.word,
                WordItem::Separator(_) => return,
            };
            spans.push(get_word_timing(word));
        },
    );
    spans
}

#[test]
fn retry_session_refuses_a_source_missing_from_the_working_document() {
    let other = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\tyes . \u{15}100_200\u{15}\n*SPK:\tno . \u{15}200_300\u{15}\n@End\n"
    ).expect_built();
    let sources = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &other)
        .unwrap()
        .supplied_retry_sources(&other, &Recording::of_duration(Ms(400)).unwrap())
        .unwrap();
    let (mut chat, applied, _) = projected("yes", vec![timing(110, 150)]);
    let imposed = applied.then_impose(&mut chat).unwrap();
    let result = SuppliedRetrySession::new(
        imposed,
        &mut chat,
        sources
            .into_sources()
            .into_iter()
            .skip(1)
            .map(|source| (source, ())),
    );
    assert!(matches!(
        result,
        Err(KeptBulletError::RetrySourceMissing { utterance }) if utterance == UtteranceIdx::new(1)
    ));
}

#[test]
fn all_retry_answers_are_retained_before_any_retry_is_applied() {
    let mut chat = TreeSitterParser::new().unwrap().parse_chat_file(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n@ID:\teng|test|SPK|||||Speaker|||\n@Media:\tsample, audio\n*SPK:\tyes okay . \u{15}100_200\u{15}\n*SPK:\twell done . \u{15}200_300\u{15}\n@End\n"
    ).expect_built();
    let authority = MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).unwrap();
    let recording = Recording::of_duration(Ms(400)).unwrap();
    let sources = authority.supplied_retry_sources(&chat, &recording).unwrap();
    let words = sources
        .sources()
        .iter()
        .flat_map(|source| source.words().iter().cloned())
        .collect();
    let group = FaGroup::test_fixture(
        TimeSpan::new(100, 300),
        words,
        vec![UtteranceIdx::new(0), UtteranceIdx::new(1)],
    );
    let applied = apply_fa_results_with_projection_policy(
        &mut chat,
        &[group],
        &[vec![timing(110, 140), None, timing(210, 240), None]],
        FaProjection::new(
            FaProjectionPolicy::new(
                WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
                ExistingWorBoundaryPolicy::Preserve,
                EndOverlapPolicy::ClampAllAdjacent,
            ),
            authority,
        ),
        true,
    )
    .unwrap();
    let imposed = applied.then_impose(&mut chat).unwrap();
    let SuppliedRetryStep::Pending(first) = SuppliedRetrySession::new(
        imposed,
        &mut chat,
        sources
            .into_sources()
            .into_iter()
            .map(|source| (source, ())),
    )
    .unwrap()
    .next() else {
        panic!("first retry")
    };
    let (first, ()) = first.start();
    let session = first.complete(SuppliedRetryAnswer::Timings {
        timings: vec![timing(110, 140), timing(150, 180)],
        source: "first".into(),
    });
    let mut first_spans = Vec::new();
    walk_words(
        &session
            .chat
            .utterances()
            .next()
            .unwrap()
            .main
            .content
            .content,
        None,
        &mut |item| {
            if let WordItem::Word(word) = item {
                first_spans.push(get_word_timing(word));
            }
        },
    );
    assert_eq!(first_spans, vec![Some(TimeSpan::new(110, 140)), None]);
    let SuppliedRetryStep::Pending(second) = session.next() else {
        panic!("second retry")
    };
    let (second, ()) = second.start();
    let SuppliedRetryStep::Complete(completed) = second
        .complete(SuppliedRetryAnswer::Timings {
            timings: vec![timing(210, 240), timing(250, 280)],
            source: "second".into(),
        })
        .next()
    else {
        panic!("all retries resolved")
    };
    assert_eq!(completed.resolved.len(), 2);
    let finalized = completed
        .then_finalize(BulletRepairPolicy::Disabled)
        .unwrap();
    assert!(finalized.monotonicity().effects().is_empty());
    let spans: Vec<_> = chat
        .utterances()
        .map(|utterance| {
            let mut timings = Vec::new();
            walk_words(&utterance.main.content.content, None, &mut |item| {
                if let WordItem::Word(word) = item {
                    timings.push(get_word_timing(word));
                }
            });
            timings
        })
        .collect();
    assert_eq!(
        spans,
        vec![
            vec![Some(TimeSpan::new(110, 140)), Some(TimeSpan::new(150, 180))],
            vec![Some(TimeSpan::new(210, 240)), Some(TimeSpan::new(250, 280))],
        ]
    );
}

#[test]
fn retry_preserves_compound_excluded_and_replaced_word_slot_correspondence() {
    let (mut chat, applied, sources) = projected(
        "yes &-you_know xxx go [: going] home",
        vec![timing(110, 120), None, None, None, timing(180, 190)],
    );
    let imposed = applied.then_impose(&mut chat).unwrap();
    let SuppliedRetryStep::Pending(pending) = SuppliedRetrySession::new(
        imposed,
        &mut chat,
        sources
            .into_sources()
            .into_iter()
            .map(|source| (source, ())),
    )
    .unwrap()
    .next() else {
        panic!("missing compound and replaced original")
    };
    let (pending, ()) = pending.start();
    assert_eq!(
        pending
            .source()
            .words()
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>(),
        ["yes", "you", "know", "go", "home"]
    );
    let SuppliedRetryStep::Complete(imposed) = pending
        .complete(SuppliedRetryAnswer::Timings {
            timings: vec![
                timing(105, 115),
                timing(130, 140),
                timing(145, 155),
                timing(160, 170),
                timing(185, 195),
            ],
            source: "controlled source answer".into(),
        })
        .next()
    else {
        panic!("one terminal attempt")
    };
    let _finalized = imposed.then_finalize(BulletRepairPolicy::Disabled).unwrap();
    assert_eq!(
        spans(&chat),
        vec![
            Some(TimeSpan::new(110, 120)),
            Some(TimeSpan::new(130, 155)),
            None,
            Some(TimeSpan::new(160, 170)),
            Some(TimeSpan::new(180, 190))
        ]
    );
}

#[test]
fn retry_rejects_answers_that_conflict_with_retained_neighbours() {
    for candidate in [timing(125, 145), timing(150, 185), timing(90, 105), None] {
        let (mut chat, applied, sources) = projected(
            "yes okay home",
            vec![timing(110, 140), None, timing(180, 190)],
        );
        let imposed = applied.then_impose(&mut chat).unwrap();
        let SuppliedRetryStep::Pending(pending) = SuppliedRetrySession::new(
            imposed,
            &mut chat,
            sources
                .into_sources()
                .into_iter()
                .map(|source| (source, ())),
        )
        .unwrap()
        .next() else {
            panic!("one missing slot")
        };
        let (pending, ()) = pending.start();
        let SuppliedRetryStep::Complete(imposed) = pending
            .complete(SuppliedRetryAnswer::Timings {
                timings: vec![timing(105, 115), candidate, timing(185, 195)],
                source: "controlled conflicting answer".into(),
            })
            .next()
        else {
            panic!("unsuccessful retry is terminal too")
        };
        let _finalized = imposed.then_finalize(BulletRepairPolicy::Disabled).unwrap();
        assert_eq!(
            spans(&chat),
            vec![
                Some(TimeSpan::new(110, 140)),
                None,
                Some(TimeSpan::new(180, 190))
            ]
        );
    }
}
