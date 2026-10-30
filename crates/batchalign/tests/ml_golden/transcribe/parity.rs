use crate::common::{LiveDirectJobClient, prepare_audio_fixtures, require_live_direct};
use batchalign::api::{JobStatus, ReleasedCommand};
use batchalign::options::{
    AsrEngineName, CommandOptions, CommonOptions, TranscribeOptions, WorTierPolicy,
};
use batchalign::worker::InferTask;

#[tokio::test]
async fn parity_transcribe_disfluency_markup() {
    let session =
        require_live_direct(InferTask::Asr, "Direct session does not support ASR infer").await;
    let jobs = LiveDirectJobClient::new(&session);

    let fixtures = prepare_audio_fixtures(jobs.state_dir());

    let out_dir = jobs.state_dir().join("out_disfluency");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
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
        utseg_fallback: false.into(),
    });

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Transcribe,
            "eng",
            vec![fixtures.audio.to_string_lossy().into()],
            vec![output_path.to_string_lossy().into()],
            options,
        )
        .await;

    if info.status != JobStatus::Completed {
        panic!("PREREQUISITE MISSING: transcribe failed")
    }

    assert!(
        outputs[0].contains("&-um") || outputs[0].contains("&-uh"),
        "D1 PARITY GAP: transcribe output should contain filled pause markers (&-um/&-uh). \
         BA2 runs DisfluencyReplacementEngine after ASR; BA3 does not yet implement this stage."
    );
}

#[tokio::test]
async fn parity_transcribe_retrace_markup() {
    let session =
        require_live_direct(InferTask::Asr, "Direct session does not support ASR infer").await;
    let jobs = LiveDirectJobClient::new(&session);

    let fixtures = prepare_audio_fixtures(jobs.state_dir());

    let out_dir = jobs.state_dir().join("out_retrace");
    std::fs::create_dir_all(&out_dir).expect("mkdir");
    let output_path = out_dir.join("test.cha");

    let options = CommandOptions::Transcribe(TranscribeOptions {
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
        utseg_fallback: false.into(),
    });

    let (info, outputs) = jobs
        .submit_paths_job(
            ReleasedCommand::Transcribe,
            "eng",
            vec![fixtures.audio.to_string_lossy().into()],
            vec![output_path.to_string_lossy().into()],
            options,
        )
        .await;

    if info.status != JobStatus::Completed {
        panic!("PREREQUISITE MISSING: transcribe failed")
    }

    assert!(
        outputs[0].contains("[/]"),
        "D1b PARITY GAP: transcribe output should contain retrace markers ([/]). \
         BA2 runs NgramRetraceEngine after ASR; BA3 does not yet implement this stage."
    );
}
