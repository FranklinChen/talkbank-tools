use crate::ml_golden::parity_helpers::{Ba2Baseline, run_parity_test};
use batchalign::api::ReleasedCommand;
use batchalign::options::{CommandOptions, CommonOptions, TranslateOptions};
use batchalign::worker::InferTask;

fn translate_opts() -> CommandOptions {
    CommandOptions::Translate(TranslateOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
        ..TranslateOptions::default()
    })
}

#[tokio::test]
async fn parity_translate_eng() {
    run_parity_test(
        ReleasedCommand::Translate,
        InferTask::Translate,
        "eng_disfluency",
        "eng",
        translate_opts(),
        Ba2Baseline::Golden,
    )
    .await;
}

#[tokio::test]
async fn parity_translate_spa() {
    run_parity_test(
        ReleasedCommand::Translate,
        InferTask::Translate,
        "spa_simple",
        "spa",
        translate_opts(),
        Ba2Baseline::Golden,
    )
    .await;
}
