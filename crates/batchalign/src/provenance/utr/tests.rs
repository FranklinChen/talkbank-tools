use super::*;
use crate::chat_ops::fa::utr::{AsrTimingToken, inject_spaced};
use crate::worker::error::WorkerError;

fn failure() -> ServerError {
    ServerError::Worker(WorkerError::WorkerResponse(
        "controlled provider refusal".into(),
    ))
}

/// A pass that matched one of two utterances, from responses that named no
/// models (so the stamp writes `utr=` alone).
fn partial_pass() -> UtrPassOutcome {
    UtrPassOutcome::Recovered {
        result: partial_result(),
        models: UtrModels::Unreported,
    }
}

fn partial_result() -> UtrResult {
    let parser = crate::chat_parser();
    let mut file = batchalign_transform::parse::parse_lenient(&parser,
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\taudio, audio\n*PAR:\thello .\n*PAR:\tgoodbye .\n@End\n").0;
    inject_spaced(
        &mut file,
        &[AsrTimingToken {
            text: "hello".into(),
            start_ms: 100,
            end_ms: 400,
        }],
    )
}

#[test]
fn utr_attempt_history_wire_and_stamp_are_independent_of_successful_timing() {
    let mut report = UtrContribution::NotRun;
    let mut retry = report
        .record_initial(&UtrEngine::RevAi, Err(failure()))
        .unwrap();
    let permit = retry.take().unwrap();
    assert!(retry.take().is_none(), "the retry permission is consumed");
    report.record_fallback(permit, Err(failure())).unwrap();
    let shortfalls = report.shortfalls();
    insta::assert_json_snapshot!(shortfalls, @r#"
    [
      {
        "kind": "utr_unavailable",
        "engine": "rev_utr",
        "pass": "initial",
        "category": "provider_transient",
        "detail": "worker error: worker returned error: controlled provider refusal"
      },
      {
        "kind": "utr_unavailable",
        "engine": "rev_utr",
        "pass": "fallback",
        "category": "provider_transient",
        "detail": "worker error: worker returned error: controlled provider refusal"
      }
    ]
    "#);
    let wire = serde_json::to_value(&shortfalls).unwrap();
    assert_eq!(
        serde_json::from_value::<Vec<OutputShortfallRecord>>(wire).unwrap(),
        shortfalls
    );
    assert!(
        report
            .completed()
            .map(crate::provenance::UtrProducer::engine)
            .is_none()
    );
    insta::assert_snapshot!(report.unavailable_stamp().unwrap().as_str(), @"initial:rev:provider_transient+fallback:rev:provider_transient");

    let diagnostics = crate::api::FileOutputDiagnostics::of_findings(Vec::new(), shortfalls);
    assert!(
        diagnostics
            .summary()
            .contains("UTR initial pass unavailable")
    );

    let mut earlier_success = UtrContribution::NotRun;
    let mut fallback = earlier_success
        .record_initial(&UtrEngine::Whisper, Ok(partial_pass()))
        .unwrap();
    assert!(
        earlier_success
            .completed()
            .map(crate::provenance::UtrProducer::engine)
            .is_some()
    );
    earlier_success
        .record_fallback(fallback.take().unwrap(), Err(failure()))
        .unwrap();
    assert_eq!(
        earlier_success
            .completed()
            .map(crate::provenance::UtrProducer::engine),
        Some(&UtrEngine::Whisper)
    );
    assert_eq!(earlier_success.shortfalls().len(), 1);

    let mut later_success = UtrContribution::NotRun;
    let mut fallback = later_success
        .record_initial(&UtrEngine::Whisper, Err(failure()))
        .unwrap();
    later_success
        .record_fallback(fallback.take().unwrap(), Ok(partial_pass()))
        .unwrap();
    assert_eq!(
        later_success
            .completed()
            .map(crate::provenance::UtrProducer::engine),
        Some(&UtrEngine::Whisper)
    );
    assert_eq!(
        later_success.shortfalls().len(),
        1,
        "success cannot erase earlier failure evidence"
    );

    let stamp = super::super::align_provenance(
        &crate::api::LanguageCode3::eng(),
        &crate::engine_reports::FaCacheNamespace::for_test("test-fa"),
        &earlier_success,
        false,
        false,
        crate::chat_ops::fa::MainBulletPolicy::DeriveFromWords,
        crate::provenance::AlignTiming::Aligned,
    )
    .format();
    let text = format!("@Comment:\t{stamp}\n");
    let entries = crate::provenance::extract_provenance(&text).unwrap();
    let mut fields = entries[0].fields.clone();
    fields.remove("build");
    insta::assert_json_snapshot!(fields, @r#"
    {
      "fa": "test-fa",
      "lang": "eng",
      "main_bullets": "derive",
      "utr": "whisper",
      "utr_unavailable": "fallback:whisper:provider_transient"
    }
    "#);
}

#[test]
fn omitted_no_work_and_cancelled_passes_are_not_provider_shortfalls() {
    let mut report = UtrContribution::NotRun;
    let mut no_work = report
        .record_initial(
            &UtrEngine::RevAi,
            Ok(UtrPassOutcome::NothingUntimed(
                UtrResult::not_run_no_untimed(2),
            )),
        )
        .unwrap();
    assert!(no_work.take().is_none());
    assert_eq!(report, UtrContribution::NotRun);
    assert!(report.shortfalls().is_empty());
    assert!(matches!(
        report.record_initial(&UtrEngine::RevAi, Err(ServerError::Cancelled)),
        Err(UtrCancelled)
    ));
    assert_eq!(report, UtrContribution::NotRun);
    assert!(report.unavailable_stamp().is_none());

    let _ = report
        .record_initial(
            &UtrEngine::RevAi,
            Err(ServerError::Worker(WorkerError::WorkerResponse(
                "x; | ]\n".repeat(200),
            ))),
        )
        .unwrap();
    let shortfalls = report.shortfalls();
    let OutputShortfallRecord::UtrUnavailable { detail, .. } = &shortfalls[0] else {
        panic!("failure report");
    };
    assert_eq!(detail.chars().count(), 500);
    assert_eq!(
        report.unavailable_stamp().unwrap().as_str(),
        "initial:rev:provider_transient"
    );
}

/// RED FIRST: a denied model and a worker that cannot boot kept the
/// same-engine fallback, which asks the same engine for the same model on
/// the same worker key and can only fail the same way. They now withhold it,
/// as an exhausted quota does; a failure of the request keeps it.
#[test]
fn an_engine_level_failure_withholds_the_same_engine_fallback() {
    let withheld = [
        ServerError::ModelAccessDenied("gated repository requires accepted terms".into()),
        ServerError::Worker(WorkerError::Bootstrap {
            message: "model weights could not be loaded".into(),
            worker: crate::worker::error::WorkerAfterError::Exits,
        }),
        ServerError::ProviderBillingRefused(crate::error::ProviderBillingRefusal::new(
            crate::error::BilledProduct::NotNamed,
            "resource pack exhausted".into(),
        )),
    ];
    for error in withheld {
        let rendered = error.to_string();
        let mut report = UtrContribution::NotRun;
        let mut fallback = report
            .record_initial(&UtrEngine::Whisper, Err(error))
            .unwrap();
        assert!(fallback.take().is_none(), "withheld after: {rendered}");
        assert_eq!(report.shortfalls().len(), 1, "still recorded: {rendered}");
    }

    let mut report = UtrContribution::NotRun;
    let mut fallback = report
        .record_initial(&UtrEngine::Whisper, Err(failure()))
        .unwrap();
    assert!(
        fallback.take().is_some(),
        "a failure of this request keeps the fallback"
    );
}

/// RED FIRST: a Rev.AI account refusal (an unaccepted key, a forbidden
/// account, no credit) was a terminal failure of the request and kept the
/// same-engine fallback, sent from the same refused account; and a model
/// download that failed in transit was a deterministic bootstrap failure and
/// withheld a fallback a later attempt could have served. Each now goes by
/// what it belongs to: the account withholds it, the transit keeps it.
#[test]
fn an_account_refusal_withholds_and_a_failed_download_keeps_the_fallback() {
    let mut report = UtrContribution::NotRun;
    let mut fallback = report
        .record_initial(
            &UtrEngine::RevAi,
            Err(ServerError::AsrProvider {
                disposition: crate::error::AsrProviderDisposition::AccountRefused,
                message: "Rev.AI returned HTTP 401: unauthorized".into(),
            }),
        )
        .unwrap();
    assert!(
        fallback.take().is_none(),
        "the account refuses the fallback too"
    );

    let mut report = UtrContribution::NotRun;
    let mut fallback = report
        .record_initial(
            &UtrEngine::Whisper,
            Err(ServerError::Worker(WorkerError::ModelDownload {
                message: "ensure_task failed: connection reset".into(),
                worker: crate::worker::error::WorkerAfterError::Exits,
            })),
        )
        .unwrap();
    assert!(
        fallback.take().is_some(),
        "a download that failed in transit may get through on the fallback"
    );
}

/// A completed pass names the models its responses reported beside the
/// engine, as transcribe's `asr_model=` does; a pass whose responses could
/// not all name the same models writes `utr=` alone.
#[test]
fn the_align_stamp_names_the_models_the_recovery_responses_reported() {
    let stamp_of = |models: UtrModels| {
        let mut report = UtrContribution::NotRun;
        report
            .record_initial(
                &UtrEngine::RevAi,
                Ok(UtrPassOutcome::Recovered {
                    result: partial_result(),
                    models,
                }),
            )
            .unwrap();
        super::super::align_provenance(
            &crate::api::LanguageCode3::eng(),
            &crate::engine_reports::FaCacheNamespace::for_test("test-fa"),
            &report,
            false,
            false,
            crate::chat_ops::fa::MainBulletPolicy::DeriveFromWords,
            crate::provenance::AlignTiming::Aligned,
        )
        .format()
    };
    let rev = crate::model_manifest::rev_loaded_identity();
    let named = stamp_of(UtrModels::Reported(Box::new(rev.clone())));
    assert!(
        named.contains(&format!("utr=rev ; utr_model={}", rev.stamp_value())),
        "{named}"
    );
    for unnamed in [UtrModels::Unreported, UtrModels::Disagreeing] {
        let stamp = stamp_of(unnamed);
        assert!(
            stamp.contains("utr=rev ;") || stamp.contains("utr=rev |"),
            "{stamp}"
        );
        assert!(!stamp.contains("utr_model="), "{stamp}");
    }
}

/// The fold over a pass's responses: equal reports stay named, a different
/// report makes the pass disagree, and a response with no report makes it
/// unreported.
#[test]
fn the_models_of_a_pass_fold_over_its_responses() {
    let response = |model| crate::transcribe::AsrResponse {
        tokens: Vec::new(),
        lang: crate::api::LanguageCode3::eng(),
        seam_ambiguities: Vec::new(),
        source_monologues: None,
        model,
    };
    let rev = crate::model_manifest::rev_loaded_identity();
    let whisper = crate::worker::asr_result_v2::test_fixtures::test_identity();
    let reported = response(Some(rev.clone()));
    assert_eq!(
        UtrModels::of(&reported).and(&reported),
        UtrModels::Reported(Box::new(rev))
    );
    assert_eq!(
        UtrModels::of(&reported).and(&response(Some(whisper))),
        UtrModels::Disagreeing
    );
    assert_eq!(
        UtrModels::of(&reported).and(&response(None)),
        UtrModels::Unreported
    );
    assert_eq!(
        UtrModels::Disagreeing.and(&reported),
        UtrModels::Disagreeing
    );
}
