//! Source bytes, application semantics and externally returned response populations.
use super::*;
use crate::api::DisplayPath;
use crate::scheduling::FailureCategory;
use crate::worker::pool::PoolConfig;
use batchalign_types::segmentation::UtteranceSegmentation;
const VALID: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello world .\n@End\n";

fn collect_one(_file: &ChatFile) -> Vec<(usize, ())> {
    vec![(0, ())]
}

fn apply_dropping_terminators(
    file: &mut ChatFile,
    _items: &[(usize, ())],
    _r: &[()],
) -> Result<(), ServerError> {
    use talkbank_model::model::Line;
    for line in &mut file.lines {
        if let Line::Utterance(utt) = line {
            utt.main.content.terminator = None;
        }
    }
    Ok(())
}

fn apply_nothing(
    _file: &mut ChatFile,
    _items: &[(usize, ())],
    _r: &[()],
) -> Result<(), ServerError> {
    Ok(())
}

fn collect_nothing(_file: &ChatFile) -> Vec<(usize, ())> {
    Vec::new()
}

fn test_stamp(
    lang: &LanguageCode3,
    responses: &[()],
) -> Result<TextStamp, crate::api::InvalidStampSafeText> {
    let engine = crate::api::ReportedEngineName::try_from("test-engine")?;
    Ok(crate::provenance::result_named_provenance(
        crate::provenance::ResultNamedCommand::Translate,
        lang,
        responses.iter().map(|()| &engine),
    ))
}

fn localized_b2() -> crate::pipeline::post_validate::LocalizedDiagnosis {
    use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};
    const GENERATED: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tPAR0 Participant, PAR1 Participant\n\
@ID:\teng|test|PAR0|||||Participant|||\n@ID:\teng|test|PAR1|||||Participant|||\n\
*PAR0:\tthe red ball rolled far away .\n\
*PAR1:\twe took it there .\n\
*PAR0:\tthen we went home again today .\n@End\n";
    let parser = crate::chat_parser();
    let (mut file, errors) = batchalign_transform::parse::parse_lenient(&parser, GENERATED);
    assert!(errors.is_empty(), "{errors:?}");
    // The generated word CHAT cannot hold, as a producer would have
    // built it before word forms were written (an invented utterance).
    let mut ordinal = 0;
    for line in &mut file.lines {
        if let Line::Utterance(utterance) = line {
            if ordinal == 1 {
                utterance.main.content.content = TierContentItems::new(
                    ["we", "took", "b2", "there"]
                        .into_iter()
                        .map(|word| {
                            UtteranceContent::Word(Box::new(Word::new(
                                talkbank_model::Location::Unlocated,
                                talkbank_model::WordText::new(word).expect("nonempty fixture word"),
                            )))
                        })
                        .collect(),
                );
            }
            ordinal += 1;
        }
    }
    let crate::pipeline::post_validate::ProducedOutput::Diagnosed(diagnosed) =
        PostValidated::produced(file, ReleasedCommand::Transcribe)
    else {
        panic!("the invalid word must diagnose the transcript");
    };
    let localized = diagnosed
        .localize()
        .expect("the finding is the second utterance's");
    assert_eq!(localized.held_out().ordinals().collect::<Vec<_>>(), [1]);
    localized
}

fn split_after_third_word(
    items: &[(usize, batchalign_transform::utseg::UtsegBatchItem)],
) -> Vec<Result<Vec<usize>, crate::text_batch::EngineItemFailure>> {
    items
        .iter()
        .map(|(_, item)| {
            Ok((0..item.words.len())
                .map(|word| usize::from(word >= 3))
                .collect())
        })
        .collect()
}

/// Guards: malformed retained CHAT is refused before pending work exists, and
/// a valid no-work source retains its original bytes through both admission routes.
/// Kind: boundary
/// Not a type: actual CHAT bytes must earn source admission through the parser
/// and validator before collection can hold a valid document.
/// State: pure admitted source observations own synthetic CHAT and no worker resources.
/// Concurrency: none
#[tokio::test]
async fn collected_sources_preserve_refusals_and_no_work_bytes() {
    let retrace = VALID.replace("hello world .", "<the dog> [///] .");
    let corrupt_mor = VALID.replace("@End", "%mor:\tnoun| .\n@End");
    let bad_header = "@UTF8\n@Begin\n@Options:\tdummy\n*CHI:\thello .\n@End\n";
    let files = [retrace.as_str(), corrupt_mor.as_str(), bad_header, VALID]
        .into_iter()
        .enumerate()
        .map(|(index, source)| {
            TextBatchFileInput::new(DisplayPath::from(format!("{index}.cha")), source.to_owned())
        })
        .collect::<Vec<_>>();
    let hooks = TextBatchHooks {
        command: ReleasedCommand::Translate,
        collect: collect_nothing,
        apply: apply_nothing,
        provenance: test_stamp,
    };
    let batch::CollectedTextBatch::Complete(results) =
        batch::CollectedTextBatch::collect(&files, &hooks)
    else {
        panic!("these sources yield completed admissions without inference authority");
    };
    assert_eq!(results.len(), files.len());
    for (input, result) in files.iter().zip(results) {
        assert_eq!(input.filename, result.filename);
        if input.chat_text == VALID {
            assert_eq!(result.result.unwrap().as_str(), VALID);
        } else {
            assert_eq!(
                result.result.unwrap_err().category(),
                FailureCategory::Validation
            );
        }
    }
    let original = VALID.replace("*CHI:", "@Comment:\tstable spacing   \n*CHI:");
    let named = TextBatchFileInput::new("same.cha", original.clone());
    let admitted = AdmittedTextSource::admit(&crate::chat_parser(), &named).unwrap();
    let output = run_admitted_text_source(
        admitted,
        &LanguageCode3::eng(),
        hooks,
        async |_| -> Result<Vec<Result<(), ItemFailure<std::convert::Infallible>>>, ServerError> {
            panic!("a no-work source cannot reach inference")
        },
    )
    .await;
    let [result]: [crate::text_batch::TextBatchFileResult; 1] =
        output.try_into().expect("one result");
    assert_eq!(result.filename, named.filename);
    assert_eq!(result.result.unwrap().as_str(), original);
}

/// Guards: applied output carries its actual engine provenance and refuses a
/// mutation that removes a required terminator.
/// Kind: contract
/// Not a type: an application may produce structurally expressible invalid CHAT,
/// and the engine-name bytes must survive stamping and serialization.
/// State: a pure response double and collected source owners exercise actual
/// application, provenance and post-validation without a pool or cache.
/// Concurrency: none
#[tokio::test]
async fn applied_output_preserves_provenance_or_reports_its_new_finding() {
    enum Application {
        Preserve,
        DropTerminators,
    }
    for application in [Application::Preserve, Application::DropTerminators] {
        let apply = match application {
            Application::Preserve => apply_nothing,
            Application::DropTerminators => apply_dropping_terminators,
        };
        let hooks = TextBatchHooks {
            command: ReleasedCommand::Translate,
            collect: collect_one,
            apply,
            provenance: test_stamp,
        };
        let files = [TextBatchFileInput::new("a.cha", VALID)];
        let mut results = batch::CollectedTextBatch::collect(&files, &hooks)
            .run_with(
                &LanguageCode3::eng(),
                &hooks,
                async |items: &[(usize, ())]| -> Result<
                    Vec<Result<(), crate::text_batch::EngineItemFailure>>,
                    ServerError,
                > { Ok(items.iter().map(|_| Ok(())).collect()) },
            )
            .await;
        assert_eq!(results.len(), 1);
        match application {
            Application::Preserve => {
                let output = results.remove(0).result.unwrap();
                assert!(output.as_str().contains("*CHI:"));
                assert!(output.as_str().contains(&format!(
                    "{}engine=test-engine ; lang=eng | ",
                    crate::provenance::written_stamp_opening("translate")
                )));
            }
            Application::DropTerminators => {
                let failure = results.remove(0).result.unwrap_err();
                assert_eq!(failure.category(), FailureCategory::System);
                assert!(
                    failure.to_string().contains("lost its terminator"),
                    "{failure}"
                );
            }
        }
    }
}

/// Guards: a new invalid generated word outside the held-out utterance is the sole reported output finding.
/// Kind: contract
/// Not a type: the actual segmented CHAT content and relocated finding ownership require semantic validation.
/// State: synthetic generated CHAT, a local unused pool and noop cache satisfy
/// the normal entry API, while a typed response double replaces model inference.
/// Concurrency: the runtime owns all fixtures, with no model, process or shared state.
#[tokio::test]
async fn segmentation_that_adds_a_finding_outside_the_held_out_utterance_is_refused() {
    let pool = WorkerPool::new(PoolConfig::for_tests());
    let cache = crate::cache::UtteranceCache::noop();
    let infer = async |_pool: &WorkerPool,
                       items: &[(usize, batchalign_transform::utseg::UtsegBatchItem)],
                       _lang: &LanguageCode3|
           -> Result<
        Vec<Result<Vec<usize>, crate::text_batch::EngineItemFailure>>,
        ServerError,
    > { Ok(split_after_third_word(items)) };
    let error = run_localized_text_pipeline(
        localized_b2(),
        &LanguageCode3::eng(),
        PipelineServices::new(&pool, &cache),
        TextPipelineHooks {
            command: ReleasedCommand::Utseg,
            collect: crate::utseg::collect_utseg_batch_items,
            integrate: |state: &mut HashMap<usize, UtteranceSegmentation>,
                        items,
                        responses: &[Vec<usize>]| {
                for ((utterance, _), assignment) in items.iter().zip(responses) {
                    state.insert(
                        *utterance,
                        UtteranceSegmentation::unstated(assignment.clone()),
                    );
                }
            },
            apply: |file, state| {
                use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};
                let layout = crate::utseg::apply_utseg_document(file, state)?;
                // A defective application: the first child of the first
                // split carries a word CHAT cannot hold.
                for line in &mut file.lines {
                    if let Line::Utterance(utterance) = line {
                        utterance.main.content.content = TierContentItems::new(
                            ["the", "c3", "ball"]
                                .into_iter()
                                .map(|word| {
                                    UtteranceContent::Word(Box::new(Word::new(
                                        talkbank_model::Location::Unlocated,
                                        talkbank_model::WordText::new(word)
                                            .expect("nonempty fixture word"),
                                    )))
                                })
                                .collect(),
                        );
                        break;
                    }
                }
                Ok(crate::pipeline::post_validate::AppliedLayout::Segmented(
                    layout,
                ))
            },
            provenance: |_lang, _responses| {
                Ok(TextStamp::NotStamped(
                    crate::provenance::NoStampReason::NothingApplied,
                ))
            },
        },
        infer,
        |_items, _responses| Ok(()),
    )
    .await
    .expect_err("a stage that breaks an utterance it was given is refused");
    let ServerError::OutputAdmission {
        details: crate::error::OutputAdmissionRefusal::Judged { first, rest, .. },
        ..
    } = &error
    else {
        panic!("refused as the stage's judged output, got {error}");
    };
    let findings: Vec<&str> = std::iter::once(first)
        .chain(rest)
        .map(|finding| finding.message.as_str())
        .collect();
    // Never empty: the first finding is a field of its own.
    assert!(
        findings.iter().all(|message| message.contains("c3")),
        "only the stage's own finding: {findings:?}"
    );
}

/// Guards: only sound utterances reach inference and actual segmentation preserves the held-out generated word.
/// Kind: contract
/// Not a type: actual utterance splitting changes positions and must preserve the source word bytes and finding attribution.
/// State: synthetic generated CHAT, a local unused pool and noop cache satisfy
/// the normal entry API, while a typed response double replaces model inference.
/// Concurrency: the runtime owns all fixtures, with no model, process or shared state.
#[tokio::test]
async fn a_diagnosed_transcript_is_segmented_outside_its_faulty_utterance() {
    let localized = localized_b2();

    let pool = WorkerPool::new(PoolConfig::for_tests());
    let cache = crate::cache::UtteranceCache::noop();
    let mut inferred = Vec::new();
    let infer = async |_pool: &WorkerPool,
                       items: &[(usize, batchalign_transform::utseg::UtsegBatchItem)],
                       _lang: &LanguageCode3|
           -> Result<
        Vec<Result<Vec<usize>, crate::text_batch::EngineItemFailure>>,
        ServerError,
    > {
        inferred.extend(items.iter().map(|(utterance, _)| *utterance));
        Ok(split_after_third_word(items))
    };
    let produced = run_localized_text_pipeline(
        localized,
        &LanguageCode3::eng(),
        PipelineServices::new(&pool, &cache),
        TextPipelineHooks {
            command: ReleasedCommand::Utseg,
            collect: crate::utseg::collect_utseg_batch_items,
            integrate: |state: &mut HashMap<usize, UtteranceSegmentation>,
                        items,
                        responses: &[Vec<usize>]| {
                for ((utterance, _), assignment) in items.iter().zip(responses) {
                    state.insert(
                        *utterance,
                        UtteranceSegmentation::unstated(assignment.clone()),
                    );
                }
            },
            apply: |file, state| {
                crate::utseg::apply_utseg_document(file, state)
                    .map(crate::pipeline::post_validate::AppliedLayout::Segmented)
                    .map_err(ServerError::from)
            },
            provenance: |_lang, _responses| {
                Ok(TextStamp::NotStamped(
                    crate::provenance::NoStampReason::NothingApplied,
                ))
            },
        },
        infer,
        |_items, _responses| Ok(()),
    )
    .await
    .expect("the localized stage runs");
    assert_eq!(inferred, [0, 2], "the faulty utterance is never inferred");
    let crate::pipeline::post_validate::ProducedOutput::Diagnosed(still) = &produced else {
        panic!("the invalid word is still there, so the output is still diagnosed");
    };
    assert!(
        still
            .findings()
            .errors()
            .all(|finding| finding.message.contains("b2")),
        "{still:?}"
    );
    let text = produced.as_str();
    assert!(text.contains("*PAR0:\tthe red ball .\n"), "{text}");
    assert!(text.contains("*PAR1:\twe took b2 there .\n"), "{text}");
    assert!(text.contains("*PAR0:\tthen we went .\n"), "{text}");
}

fn admitted_input() -> PostValidated {
    let parser = crate::chat_parser();
    let source = admit_retained_text(&parser, VALID, TranscriptName::Anonymous).unwrap();
    PostValidated::pass_through(source, ReleasedCommand::Translate)
}
/// Guards: missing and extra response populations are refused by both normal
/// single-file and pooled entry paths before application, while exact counts pass.
/// Kind: contract
/// Not a type: the external inference vector has an observed runtime length,
/// admitted once against the owned collected items before issuing matched proofs.
/// State: synthetic CHAT, a local unused pool and noop cache enter the actual
/// APIs with a count-producing effect double and no model or filesystem effects.
/// Concurrency: the test runtime exclusively owns its inputs and doubles.
#[tokio::test]
async fn batch_response_population_is_admitted_before_application() {
    for count in [0, 1, 2] {
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let files = [TextBatchFileInput::new("a.cha", VALID)];
        let results = run_text_batch_pipeline(
            &files,
            &LanguageCode3::eng(),
            &pool,
            TextBatchHooks {
                command: ReleasedCommand::Translate,
                collect: collect_one,
                apply: apply_nothing,
                provenance: test_stamp,
            },
            async |_pool: &WorkerPool,
                   _items: &[(usize, ())],
                   _lang: &LanguageCode3|
                   -> Result<
                Vec<Result<(), crate::text_batch::EngineItemFailure>>,
                ServerError,
            > { Ok((0..count).map(|_| Ok(())).collect()) },
        )
        .await;
        let cache = crate::cache::UtteranceCache::noop();
        let single = run_admitted_text_pipeline(
            admitted_input(),
            &LanguageCode3::eng(),
            PipelineServices::new(&pool, &cache),
            TextPipelineHooks {
                command: ReleasedCommand::Translate,
                collect: collect_one,
                integrate: |_state: &mut HashMap<usize, ()>, _items, _responses| {},
                apply: |_file, _state| Ok(crate::pipeline::post_validate::AppliedLayout::InPlace),
                provenance: test_stamp,
            },
            async |_pool: &WorkerPool,
                   _items: &[(usize, ())],
                   _lang: &LanguageCode3|
                   -> Result<
                Vec<Result<(), crate::text_batch::EngineItemFailure>>,
                ServerError,
            > { Ok((0..count).map(|_| Ok(())).collect()) },
            |_items, _responses| Ok(()),
        )
        .await;
        if count == 1 {
            assert!(single.is_ok());
        } else {
            let error = single.expect_err("single-file correspondence must be admitted");
            assert!(error.to_string().contains("response population"), "{error}");
        }
        assert_eq!(results.len(), 1);
        if count == 1 {
            assert!(results[0].result.is_ok());
        } else {
            let failure = results[0]
                .result
                .as_ref()
                .expect_err("cardinality mismatch must refuse application");
            assert_eq!(failure.category(), FailureCategory::WorkerProtocol);
            assert!(
                failure.to_string().contains("response population"),
                "{failure}"
            );
        }
    }
}
