use crate::ml_golden::golden::fixtures::{ENG_COREF, SPA_SIMPLE};
use crate::ml_golden::golden::helpers::{
    assert_golden_snapshot, has_user_defined_tier, parse_output, require_managed_session,
    require_managed_session_warmed,
};
use batchalign::api::ReleasedCommand;
use batchalign::options::{CommandOptions, CommonOptions, CorefOptions};
use batchalign::worker::InferTask;

/// Guards: the English dog/it fixture producing its observed xcoref chain and provenance bytes.
/// Kind: journey
/// Not a type: The managed session owns transport and staging, but its types cannot prove the English dog/it fixture producing its observed xcoref chain and provenance bytes from the loaded model weights and serialized response.
/// State: A managed HTTP session, isolated job staging and real model workers
/// exercise serialization, inference and injection together.
/// Concurrency: The shared backend serializes sessions and retains warmed models;
/// job-local staging keeps submitted sources and outputs isolated.
#[tokio::test]
async fn golden_coref_eng() {
    let jobs = require_managed_session_warmed(
        InferTask::Coref,
        ReleasedCommand::Coref,
        "eng",
        "Managed session does not support coref infer",
    )
    .await;

    let options = CommandOptions::Coref(CorefOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
    });

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Coref,
            "eng",
            "eng_coref.cha",
            ENG_COREF,
            options,
        )
        .await;

    assert_eq!(
        info.status,
        batchalign::api::JobStatus::Completed,
        "coref_eng: job should complete; error={:?}",
        info.error
    );
    let file = parse_output(
        results[0].content.as_text().expect("text command result"),
        "coref_eng",
    );
    if has_user_defined_tier(&file, "xcoref") {
        eprintln!("Coref model detected chains: snapshotting with %xcoref");
    } else {
        eprintln!("Coref model found no chains (valid for short input)");
    }
    assert_golden_snapshot!(
        "coref_eng",
        results[0].content.as_text().expect("text command result")
    );
}

/// Guards: Spanish @Languages source bytes select pass-through and preserve the emitted CHAT without xcoref.
/// Kind: contract
/// Not a type: Managed session ownership cannot prove that the Spanish coreference header bytes choose the pass-through route or preserve the serialized file exactly.
/// State: An unwarmed managed HTTP backend stages and returns the supplied CHAT source through real admission and routing.
/// Concurrency: The shared session semaphore and private job staging isolate the source and output paths.
#[tokio::test]
async fn golden_coref_spa_passthrough() {
    let jobs = require_managed_session(
        InferTask::Coref,
        "Managed session does not support coref infer",
    )
    .await;

    let options = CommandOptions::Coref(CorefOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
    });

    let (info, results) = jobs
        .submit_content_job(
            ReleasedCommand::Coref,
            "eng",
            "spa_simple.cha",
            SPA_SIMPLE,
            options,
        )
        .await;

    assert_eq!(
        info.status,
        batchalign::api::JobStatus::Completed,
        "coref_spa_passthrough: job should complete; error={:?}",
        info.error
    );
    let file = parse_output(
        results[0].content.as_text().expect("text command result"),
        "coref_spa_passthrough",
    );
    assert!(
        !has_user_defined_tier(&file, "xcoref"),
        "non-English coref input should pass through without %xcoref"
    );
    assert_golden_snapshot!(
        "coref_spa_passthrough",
        results[0].content.as_text().expect("text command result")
    );
}
