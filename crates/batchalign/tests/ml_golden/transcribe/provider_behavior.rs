use crate::common::{LiveServerJobClient, require_live_server};
use crate::ml_golden::audio_helpers::{assert_valid_chat_structure, require_revai};
use crate::ml_golden::transcribe::helpers::{prepare_transcribe_fixture_job, transcribe_options};
use batchalign::api::{JobStatus, ReleasedCommand};
use batchalign::options::{AsrEngineName, WorTierPolicy};
use batchalign::worker::InferTask;

fn media_header(chat: &str, label: &str) -> String {
    chat.lines()
        .find(|line| line.starts_with("@Media:\t"))
        .unwrap_or_else(|| panic!("{label}: expected @Media header"))
        .to_string()
}

#[tokio::test]
async fn transcribe_server_eng_revai_completes_when_configured() {
    require_revai();

    let server =
        require_live_server(InferTask::Asr, "live server does not support ASR infer").await;
    let jobs = LiveServerJobClient::from_session(&server);

    let fixture = prepare_transcribe_fixture_job(
        server.state_dir(),
        "transcribe_server_eng_revai_completes_when_configured",
    );

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Transcribe,
            "eng",
            vec![fixture.source_path],
            vec![fixture.output_path],
            transcribe_options(AsrEngineName::RevAi, false, WorTierPolicy::Omit),
        )
        .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "transcribe_server_eng_revai_completes_when_configured: job should complete; error={:?}",
        info.error
    );
    assert_eq!(outputs.len(), 1);
    assert_valid_chat_structure(
        &outputs[0],
        "transcribe_server_eng_revai_completes_when_configured",
    );
}

#[tokio::test]
async fn transcribe_server_eng_revai_preserves_source_media_basename() {
    require_revai();

    let server =
        require_live_server(InferTask::Asr, "live server does not support ASR infer").await;
    let jobs = LiveServerJobClient::from_session(&server);

    let fixture = prepare_transcribe_fixture_job(
        server.state_dir(),
        "transcribe_server_eng_revai_preserves_source_media_basename",
    );

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Transcribe,
            "eng",
            vec![fixture.source_path],
            vec![fixture.output_path],
            transcribe_options(AsrEngineName::RevAi, false, WorTierPolicy::Omit),
        )
        .await;

    assert_eq!(
        info.status,
        JobStatus::Completed,
        "transcribe_server_eng_revai_preserves_source_media_basename: job should complete; error={:?}",
        info.error
    );
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        media_header(
            &outputs[0],
            "transcribe_server_eng_revai_preserves_source_media_basename"
        ),
        "@Media:\ttest, audio",
        "server transcribe Rev.AI path should preserve the original source media basename"
    );
}
