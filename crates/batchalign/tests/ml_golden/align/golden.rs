use crate::common::{LiveServerJobClient, assert_completed_without_errors, require_live_server};
use crate::ml_golden::align::helpers::{align_options, prepare_align_fixture_job};
use batchalign::api::ReleasedCommand;
use batchalign::options::{FaEngineName, WorTierPolicy};
use batchalign::worker::InferTask;

use crate::ml_golden::audio_helpers::{
    assert_all_utterances_timed, assert_valid_chat_structure, count_wor_tiers,
};

/// Guards: Wav2Vec alignment through managed HTTP emits timed utterances and word tiers.
/// Kind: journey.
/// Not a type: Wav2Vec weights and decoded fixture audio determine the CTC timing results.
/// State: A borrowed LiveServerSession owns HTTP state and managed workers;
/// the fixture owns real CHAT and adjacent audio files in that session directory.
/// Concurrency: The shared live-fixture semaphore serializes model sessions;
/// each session has isolated job, output and cache directories.
#[tokio::test]
async fn golden_align_eng_wav2vec() {
    let session =
        require_live_server(InferTask::Fa, "Managed session does not support FA infer").await;
    let jobs = LiveServerJobClient::from_session(&session);

    let fixture = prepare_align_fixture_job(session.state_dir(), "align_wav2vec");

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Align,
            "eng",
            vec![fixture.source_path],
            vec![fixture.output_path],
            align_options(FaEngineName::Wave2Vec, WorTierPolicy::Include),
        )
        .await;

    assert_completed_without_errors("align_eng_wav2vec", &info, &[]);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "align_eng_wav2vec");
    assert!(
        count_wor_tiers(output) > 0,
        "align_eng_wav2vec: %wor tier should be present"
    );
    assert_valid_chat_structure(output, "align_eng_wav2vec");
}

/// Guards: Whisper alignment through managed HTTP emits timed utterances and word tiers.
/// Kind: journey.
/// Not a type: Whisper attention tensors and decoded fixture audio determine lexical timing results.
/// State: A borrowed LiveServerSession owns HTTP state and managed workers;
/// the fixture owns real CHAT and adjacent audio files in that session directory.
/// Concurrency: The shared live-fixture semaphore serializes model sessions;
/// each session has isolated job, output and cache directories.
#[tokio::test]
async fn golden_align_eng_whisper_fa() {
    let session =
        require_live_server(InferTask::Fa, "Managed session does not support FA infer").await;
    let jobs = LiveServerJobClient::from_session(&session);

    let fixture = prepare_align_fixture_job(session.state_dir(), "align_whisper");

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Align,
            "eng",
            vec![fixture.source_path],
            vec![fixture.output_path],
            align_options(FaEngineName::Whisper, WorTierPolicy::Include),
        )
        .await;

    assert_completed_without_errors("align_eng_whisper_fa", &info, &[]);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "align_eng_whisper_fa");
    assert!(
        count_wor_tiers(output) > 0,
        "align_eng_whisper_fa: %wor tier should be present"
    );
    assert_valid_chat_structure(output, "align_eng_whisper_fa");
}

/// Guards: Managed alignment omits word tiers when the output policy requests omission.
/// Kind: journey.
/// Not a type: Serialized CHAT bytes must reflect the chosen word-tier policy after real inference.
/// State: A borrowed LiveServerSession owns HTTP state and managed workers;
/// the fixture owns real CHAT and adjacent audio files in that session directory.
/// Concurrency: The shared live-fixture semaphore serializes model sessions;
/// each session has isolated job, output and cache directories.
#[tokio::test]
async fn golden_align_eng_no_wor() {
    let session =
        require_live_server(InferTask::Fa, "Managed session does not support FA infer").await;
    let jobs = LiveServerJobClient::from_session(&session);

    let fixture = prepare_align_fixture_job(session.state_dir(), "align_no_wor");

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Align,
            "eng",
            vec![fixture.source_path],
            vec![fixture.output_path],
            align_options(FaEngineName::Wave2Vec, WorTierPolicy::Omit),
        )
        .await;

    assert_completed_without_errors("align_eng_no_wor", &info, &[]);
    assert_eq!(outputs.len(), 1);

    let output = &outputs[0];
    assert_all_utterances_timed(output, "align_eng_no_wor");
    assert_eq!(
        count_wor_tiers(output),
        0,
        "align_eng_no_wor: %wor tier should be absent when wor=Omit"
    );
    assert_valid_chat_structure(output, "align_eng_no_wor");
}
