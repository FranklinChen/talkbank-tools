use crate::common::{LiveServerJobClient, require_live_server};
use crate::ml_golden::audio_helpers::{parse_output, transcribe_audio_clip};
use crate::ml_golden::transcribe::helpers::prepare_named_transcribe_fixture_job;
use batchalign::api::ReleasedCommand;
use batchalign::options::{
    AsrEngineName, CommandOptions, CommonOptions, TranscribeOptions, WorTierPolicy,
};
use batchalign::worker::InferTask;

/// Do not authorize the Stanza fallback.
///
/// Passed for a language that HAS a TalkBank utterance-boundary model (eng,
/// cmn, zho, yue), so the test exercises that model rather than passing against
/// a substitute segmenter.
///
/// Named for the value it supplies, not for the property of the language: an
/// earlier spelling called this `HAS_BOUNDARY_MODEL`, which asserted something
/// true while supplying `false`, so a reader had to know the parameter's name
/// to see that the constant meant its own negation.
const NO_STANZA_FALLBACK: bool = false;

/// Authorize the Stanza fallback.
///
/// Passed for a language with no TalkBank boundary model. Transcribe runs
/// utterance segmentation by default, and such a language has no segmenter
/// without this opt-in, so the job is refused when it is planned, before any
/// ASR runs. These fixtures used to pass no policy at all and failed at the far
/// end of the pipeline, after a full ASR pass, with a message naming the wire
/// format.
const ALLOW_STANZA_FALLBACK: bool = true;

#[tokio::test]
async fn transcribe_spa_whisper() {
    transcribe_audio_clip(
        "spa_marrero_clip",
        "spa",
        "transcribe_spa",
        ALLOW_STANZA_FALLBACK,
    )
    .await;
}

#[tokio::test]
async fn transcribe_fra_whisper() {
    transcribe_audio_clip(
        "fra_geneva_clip",
        "fra",
        "transcribe_fra",
        ALLOW_STANZA_FALLBACK,
    )
    .await;
}

#[tokio::test]
async fn transcribe_jpn_whisper() {
    transcribe_audio_clip(
        "jpn_tyo_clip",
        "jpn",
        "transcribe_jpn",
        ALLOW_STANZA_FALLBACK,
    )
    .await;
}

#[tokio::test]
async fn transcribe_yue_whisper() {
    transcribe_audio_clip("yue_hku_clip", "yue", "transcribe_yue", NO_STANZA_FALLBACK).await;
}

#[tokio::test]
async fn transcribe_biling_cat_spa_whisper() {
    transcribe_audio_clip(
        "biling_cat_spa_clip",
        "cat",
        "transcribe_biling_cat_spa",
        ALLOW_STANZA_FALLBACK,
    )
    .await;
}

#[tokio::test]
async fn transcribe_eng_multi_speaker_whisper() {
    transcribe_audio_clip(
        "eng_multi_speaker",
        "eng",
        "transcribe_eng_multi_speaker",
        NO_STANZA_FALLBACK,
    )
    .await;
}

async fn transcribe_audio_clip_server(
    audio_name: &str,
    lang: &str,
    label: &str,
    allow_stanza_fallback: bool,
) {
    let Some(server) =
        require_live_server(InferTask::Asr, "live server does not support ASR infer").await
    else {
        return;
    };
    let jobs = LiveServerJobClient::from_session(&server);

    let Some(fixture) = prepare_named_transcribe_fixture_job(server.state_dir(), label, audio_name)
    else {
        return;
    };

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Transcribe,
            lang,
            vec![fixture.source_path],
            vec![fixture.output_path],
            // Built here rather than through `transcribe_options`, which
            // carries the default no-fallback policy correct for the English
            // fixtures that share it. These clips are not English, and a
            // language without a boundary model must state its segmenter.
            CommandOptions::Transcribe(TranscribeOptions {
                auto_speakers: false,
                common: CommonOptions {
                    override_media_cache: true,
                    ..CommonOptions::default()
                },
                asr_engine: AsrEngineName::Whisper,
                diarize: false,
                wor: WorTierPolicy::Omit,
                merge_abbrev: false.into(),
                batch_size: 8,
                utseg_fallback: allow_stanza_fallback.into(),
            }),
        )
        .await;

    if info.status != batchalign::api::JobStatus::Completed {
        let results = jobs.job_results(&info.job_id).await;
        panic!("{label}: server job failed; info={info:?}; results={results:#?}");
    }
    assert_eq!(
        info.status,
        batchalign::api::JobStatus::Completed,
        "{label}: server job should complete; error={:?}",
        info.error
    );
    assert_eq!(outputs.len(), 1);
    let file = parse_output(&outputs[0], label);
    assert!(
        file.utterance_count() >= 1,
        "{label}: expected at least 1 utterance, got {}",
        file.utterance_count()
    );
}

#[tokio::test]
async fn transcribe_server_spa_whisper() {
    transcribe_audio_clip_server(
        "spa_marrero_clip",
        "spa",
        "transcribe_server_spa",
        ALLOW_STANZA_FALLBACK,
    )
    .await;
}

#[tokio::test]
async fn transcribe_server_yue_whisper() {
    transcribe_audio_clip_server(
        "yue_hku_clip",
        "yue",
        "transcribe_server_yue",
        NO_STANZA_FALLBACK,
    )
    .await;
}

#[tokio::test]
async fn transcribe_server_biling_cat_spa_whisper() {
    transcribe_audio_clip_server(
        "biling_cat_spa_clip",
        "cat",
        "transcribe_server_biling_cat_spa",
        ALLOW_STANZA_FALLBACK,
    )
    .await;
}
