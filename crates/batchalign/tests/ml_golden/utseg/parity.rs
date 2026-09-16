use crate::ml_golden::parity_helpers::run_parity_test;
use batchalign::api::ReleasedCommand;
use batchalign::options::{CommandOptions, CommonOptions, UtsegOptions};
use batchalign::worker::InferTask;

fn utseg_opts() -> CommandOptions {
    utseg_opts_with_fallback(false)
}

/// Utseg options stating this fixture's segmenter policy; see the golden
/// suite's copy for why a language without a boundary model must ask for the
/// Stanza fallback explicitly.
fn utseg_opts_with_fallback(allow_stanza_fallback: bool) -> CommandOptions {
    CommandOptions::Utseg(UtsegOptions {
        common: CommonOptions {
            override_media_cache: true,
            ..CommonOptions::default()
        },
        merge_abbrev: false.into(),
        utseg_fallback: allow_stanza_fallback.into(),
    })
}

#[tokio::test]
async fn parity_utseg_eng_multi() {
    run_parity_test(
        ReleasedCommand::Utseg,
        InferTask::Utseg,
        "eng_multi_speaker",
        "eng",
        utseg_opts(),
    )
    .await;
}

#[tokio::test]
async fn parity_utseg_spa() {
    run_parity_test(
        ReleasedCommand::Utseg,
        InferTask::Utseg,
        "spa_simple",
        "spa",
        // Spanish has no TalkBank boundary model; Stanza is its only
        // segmenter and must be authorized.
        utseg_opts_with_fallback(true),
    )
    .await;
}

#[tokio::test]
async fn parity_utseg_eng_disfluency() {
    run_parity_test(
        ReleasedCommand::Utseg,
        InferTask::Utseg,
        "eng_disfluency",
        "eng",
        utseg_opts(),
    )
    .await;
}
