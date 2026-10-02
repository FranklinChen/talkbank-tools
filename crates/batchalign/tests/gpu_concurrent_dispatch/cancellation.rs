//! Cancelling a job stops its in-flight file work, through the daemon's HTTP
//! surface and the real job runner.
//!
//! Field failure: an `align --utr-engine whisper` job was cancelled with
//! `POST /jobs/{id}/cancel`, answered "cancelled", and its in-flight files kept
//! running for hours: each finished its Whisper pass, dumped UTR debug data
//! into the cancelled job's directory, ran forced alignment, and occupied the
//! only worker a live job also needed.
//!
//! Two facts were lost where the runner spawns a file task: the job id (so no
//! dispatch registered against the job and the cancel's worker kill found
//! nothing) and the job's cancellation token (so nothing told the task to
//! stop). This test drives a real transcribe job whose one file is held in a
//! worker dispatch by a slow echo worker, cancels it over HTTP, and requires
//! the file to be recorded as cancelled long before the dispatch could finish
//! on its own.
//!
//! The post-cancel wait reads `GET /jobs/{id}` under a hard bound: the job's
//! SSE stream closes at the cancel itself, so there is no later event to await.
//! The bound is a failure line, not a synchronization: it sits at a third of
//! the time the dispatch takes to finish by itself.

use super::*;
use batchalign::api::{
    FilePayload, FileProgressStage, FileStatusEntry, FileStatusKind, JobInfo, JobSubmission,
    LanguageSpec, NumSpeakers,
};
use batchalign::config::{RuntimeLayout, ServerConfig};
use batchalign::options::{AsrEngineName, CommandOptions, CommonOptions, TranscribeOptions};
use batchalign::scheduling::FailureCategory;

/// How long the echo worker holds each request. The cancelled file can only
/// finish by itself after this long.
const ECHO_DISPATCH_MS: u64 = 90_000;

/// The cancelled file must reach its terminal state within this long of the
/// cancel: a third of the echo dispatch, so meeting it cannot be the
/// dispatch finishing naturally.
const CANCEL_STOPS_FILE_WITHIN: Duration = Duration::from_secs(30);

/// How long the job may take to reach the worker dispatch at all (a cold echo
/// worker spawn plus audio preparation).
const DISPATCH_REACHED_WITHIN: Duration = Duration::from_secs(60);

/// A one-second 16 kHz mono PCM WAV of silence.
fn silent_wav() -> Vec<u8> {
    const RATE: u32 = 16_000;
    let samples = RATE; // one second
    let data_len = samples * 2;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&RATE.to_le_bytes());
    wav.extend_from_slice(&(RATE * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(44 + data_len as usize, 0);
    wav
}

async fn get_job(client: &reqwest::Client, base_url: &str, job_id: &str) -> JobInfo {
    client
        .get(format!("{base_url}/jobs/{job_id}"))
        .send()
        .await
        .expect("GET /jobs/{id}")
        .json()
        .await
        .expect("parse JobInfo")
}

/// Read the job until `accept` holds for its one file, or `bound` passes.
async fn await_file(
    client: &reqwest::Client,
    base_url: &str,
    job_id: &str,
    bound: Duration,
    accept: impl Fn(&FileStatusEntry) -> bool,
) -> Result<FileStatusEntry, Box<JobInfo>> {
    let started = std::time::Instant::now();
    loop {
        let info = get_job(client, base_url, job_id).await;
        if let Some(file) = info.file_statuses.iter().find(|file| accept(file)) {
            return Ok(file.clone());
        }
        if started.elapsed() > bound {
            return Err(Box::new(info));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_over_http_stops_a_file_held_in_a_worker_dispatch() {
    let python = require_python!();
    let scratch = tempfile::tempdir().expect("scratch");
    let state_dir = scratch.path().join("state");
    std::fs::create_dir_all(&state_dir).expect("state dir");
    let media = scratch.path().join("held.wav");
    std::fs::write(&media, silent_wav()).expect("write wav");
    let output = scratch.path().join("out").join("held.cha");

    let pool_config = PoolConfig {
        python_path: python,
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<600>(),
        ready_timeout_s: batchalign::api::PositiveSeconds::literal::<60>(),
        test_echo: true,
        test_delay_ms: ECHO_DISPATCH_MS,
        max_workers_per_key: PerProfile::uniform(1),
        verbose: 0,
        task_timeouts: batchalign::types::worker_v2::TaskTimeoutOverrides {
            audio: Some(batchalign::api::PositiveSeconds::literal::<600>()),
            analysis: None,
        },
        worker_registry_path: Some(state_dir.join("workers.json")),
        runtime: WorkerRuntimeConfig {
            state_dir: Some(state_dir.clone()),
            ..Default::default()
        },
        ..Default::default()
    };
    let workers = batchalign::worker_setup::prepare_echo_workers_behind_live_runner(pool_config)
        .await
        .expect("echo workers behind the live runner");
    let (router, _state) = batchalign::create_test_app_with_prepared_workers(
        ServerConfig::default(),
        RuntimeLayout::from_state_dir(state_dir.clone()),
        batchalign::AppStorageOverrides {
            jobs_dir: Some(scratch.path().join("jobs").display().to_string()),
            db_dir: Some(scratch.path().join("db")),
            cache_dir: Some(scratch.path().join("cache")),
        },
        Some("cancel-in-flight-test".into()),
        workers,
        std::sync::Arc::new(batchalign::clock::SystemClock),
    )
    .await
    .expect("test app");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let base_url = format!("http://{}", listener.local_addr().expect("addr"));
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });
    let client = reqwest::Client::new();

    let submission = JobSubmission {
        command: ReleasedCommand::Transcribe,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files: Vec::<FilePayload>::new(),
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options: CommandOptions::Transcribe(TranscribeOptions {
            auto_speakers: false,
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::Whisper,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        paths_mode: true,
        source_paths: vec![media.display().to_string().as_str().into()],
        output_paths: vec![output.display().to_string().as_str().into()],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };
    let submitted: JobInfo = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs")
        .json()
        .await
        .expect("parse submitted job");
    let job_id = submitted.job_id.to_string();

    // The file is in its ASR dispatch, which the echo worker holds.
    if let Err(info) = await_file(
        &client,
        &base_url,
        &job_id,
        DISPATCH_REACHED_WITHIN,
        |file| {
            file.status == FileStatusKind::Processing
                && file.progress_stage == Some(FileProgressStage::Transcribing)
        },
    )
    .await
    {
        panic!("the file never reached its worker dispatch: {info:#?}");
    }

    let cancelled_at = std::time::Instant::now();
    let response = client
        .post(format!("{base_url}/jobs/{job_id}/cancel"))
        .send()
        .await
        .expect("POST cancel");
    assert!(
        response.status().is_success(),
        "cancel refused: {response:?}"
    );

    let stopped = await_file(
        &client,
        &base_url,
        &job_id,
        CANCEL_STOPS_FILE_WITHIN,
        |file| file.status.is_terminal(),
    )
    .await;
    let file = match stopped {
        Ok(file) => file,
        Err(info) => panic!(
            "{:?} after the cancel the file is still not terminal: the in-flight \
             dispatch is running to completion for a cancelled job. {info:#?}",
            cancelled_at.elapsed()
        ),
    };
    assert_eq!(file.status, FileStatusKind::Error, "{file:#?}");
    assert_eq!(
        file.error_category,
        Some(FailureCategory::Cancelled),
        "the file must be recorded as stopped by the cancel: {file:#?}"
    );
    assert!(
        !output.exists(),
        "a cancelled job must write no output, found {}",
        output.display()
    );
}
