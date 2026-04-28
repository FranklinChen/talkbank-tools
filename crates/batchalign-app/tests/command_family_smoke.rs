//! Smoke tests that exercise every released command family through the full
//! HTTP job lifecycle: submit → poll → verify results.
//!
//! Each test submits a job for one command family using test-echo workers (no
//! ML models), verifies the job completes, and checks that result files are
//! present. This proves the dispatch routing, job store persistence, and
//! result retrieval work end-to-end for every command.
//!
//! These are Tier 1 (fast) tests — no real inference, safe to run on any machine.

mod common;

use batchalign_app::api::{
    FilePayload, JobInfo, JobResultResponse, JobStatus, LanguageCode3, LanguageSpec, MemoryMb,
    NumSpeakers, ReleasedCommand,
};
use batchalign_app::config::ServerConfig;
use batchalign_app::create_test_app;
use batchalign_app::options::{
    AlignOptions, AsrEngineName, CommandOptions, CommonOptions, CorefOptions, MorphotagOptions,
    TranscribeOptions, TranslateOptions, UtsegOptions,
};
use batchalign_app::worker::pool::PoolConfig;
use common::resolve_python;
use tokio::sync::{Semaphore, SemaphorePermit};

use batchalign_app::api::JobSubmission;

/// Serialize command-family smoke tests to avoid resource contention.
static SMOKE_SLOTS: Semaphore = Semaphore::const_new(1);

macro_rules! require_python {
    () => {
        match resolve_python() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: Python 3 with batchalign not available");
                return;
            }
        }
    };
}

/// Start a test server on a random port and return the base URL.
async fn start_smoke_server(
    python_path: &str,
) -> (
    String,
    tempfile::TempDir,
    std::sync::Arc<batchalign_app::AppState>,
    SemaphorePermit<'static>,
) {
    let permit = SMOKE_SLOTS
        .acquire()
        .await
        .expect("smoke test semaphore should stay open");
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let jobs_dir = tmp.path().join("jobs");
    std::fs::create_dir_all(&jobs_dir).expect("mkdir jobs");

    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        job_ttl_days: 7,
        warmup_commands: vec![],
        memory_gate_mb: Some(MemoryMb(0)),
        ..Default::default()
    };

    let pool_config = PoolConfig {
        python_path: python_path.into(),
        test_echo: true,
        health_check_interval_s: 600,
        idle_timeout_s: 600,
        ready_timeout_s: 30,
        max_workers_per_key: 8,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    };

    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(&db_dir).expect("mkdir db");

    let (router, state) = create_test_app(
        config,
        pool_config,
        Some(jobs_dir.to_string_lossy().into()),
        Some(db_dir),
        Some("smoke-test-build-hash".into()),
    )
    .await
    .expect("create_test_app");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    let base_url = format!("http://127.0.0.1:{port}");

    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    (base_url, tmp, state, permit)
}

/// Poll until a job reaches a terminal state (completed, failed, cancelled).
async fn poll_job_done(client: &reqwest::Client, base_url: &str, job_id: &str) -> JobInfo {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(60);
    loop {
        let resp = client
            .get(format!("{base_url}/jobs/{job_id}"))
            .send()
            .await
            .expect("GET job");
        let body = resp.text().await.expect("read body");
        let info: JobInfo = serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("parse job failed: {e}\nbody: {body}"));

        if matches!(
            info.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        ) {
            return info;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "Job {job_id} did not finish within 60s (status: {:?})",
            info.status
        );
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
}

fn minimal_chat() -> String {
    "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n".into()
}

fn make_submission(command: ReleasedCommand, options: CommandOptions) -> JobSubmission {
    JobSubmission {
        command,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files: vec![FilePayload {
            filename: "smoke.cha".into(),
            content: minimal_chat(),
        }],
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options,
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    }
}

/// Submit a job with the given command and options, verify it completes,
/// and return the result response.
async fn submit_and_verify(
    client: &reqwest::Client,
    base_url: &str,
    command: ReleasedCommand,
    options: CommandOptions,
) -> JobResultResponse {
    let submission = make_submission(command, options);
    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(
        resp.status(),
        200,
        "submit {command:?} failed: {}",
        resp.text().await.unwrap_or_default()
    );

    let info: JobInfo = {
        let body = resp.text().await.expect("read submit body");
        serde_json::from_str(&body)
            .unwrap_or_else(|e| panic!("parse submit response for {command:?}: {e}\n{body}"))
    };
    assert_eq!(info.command, command);

    let final_info = poll_job_done(client, base_url, &info.job_id).await;
    assert!(
        matches!(final_info.status, JobStatus::Completed),
        "{command:?} job did not complete: {:?}",
        final_info.status
    );

    let resp = client
        .get(format!("{base_url}/jobs/{}/results", info.job_id))
        .send()
        .await
        .expect("GET results");
    assert_eq!(resp.status(), 200);
    let results: JobResultResponse = resp.json().await.expect("parse results");
    assert!(
        !results.files.is_empty(),
        "{command:?} produced no result files"
    );
    results
}

// ---------------------------------------------------------------------------
// Command family smoke tests
// ---------------------------------------------------------------------------

/// Morphotag: batched text inference (Stanza profile).
#[tokio::test]
async fn smoke_morphotag() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_smoke_server(&python).await;
    let client = reqwest::Client::new();

    let results = submit_and_verify(
        &client,
        &base_url,
        ReleasedCommand::Morphotag,
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            merge_abbrev: Default::default(),

            ..Default::default()
        }),
    )
    .await;

    assert!(
        results.files[0].error.is_none(),
        "morphotag smoke file should have no error"
    );
}

/// Utseg: batched text inference (Stanza constituency profile).
#[tokio::test]
async fn smoke_utseg() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_smoke_server(&python).await;
    let client = reqwest::Client::new();

    let results = submit_and_verify(
        &client,
        &base_url,
        ReleasedCommand::Utseg,
        CommandOptions::Utseg(UtsegOptions {
            common: CommonOptions::default(),
            merge_abbrev: Default::default(),
        }),
    )
    .await;

    assert!(
        results.files[0].error.is_none(),
        "utseg smoke file should have no error"
    );
}

/// Translate: batched text inference (IO profile).
#[tokio::test]
async fn smoke_translate() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_smoke_server(&python).await;
    let client = reqwest::Client::new();

    let results = submit_and_verify(
        &client,
        &base_url,
        ReleasedCommand::Translate,
        CommandOptions::Translate(TranslateOptions {
            common: CommonOptions::default(),
            merge_abbrev: Default::default(),
        }),
    )
    .await;

    assert!(
        results.files[0].error.is_none(),
        "translate smoke file should have no error"
    );
}

/// Coref: batched text inference (Stanza coref profile).
#[tokio::test]
async fn smoke_coref() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_smoke_server(&python).await;
    let client = reqwest::Client::new();

    let results = submit_and_verify(
        &client,
        &base_url,
        ReleasedCommand::Coref,
        CommandOptions::Coref(CorefOptions {
            common: CommonOptions::default(),
            merge_abbrev: Default::default(),
        }),
    )
    .await;

    assert!(
        results.files[0].error.is_none(),
        "coref smoke file should have no error"
    );
}

/// Transcribe: per-file audio pipeline (GPU profile for ASR + speaker).
/// Test-echo workers process this without real audio.
#[tokio::test]
async fn smoke_transcribe() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_smoke_server(&python).await;
    let client = reqwest::Client::new();

    let results = submit_and_verify(
        &client,
        &base_url,
        ReleasedCommand::Transcribe,
        CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: Default::default(),
            merge_abbrev: Default::default(),
            batch_size: 8,
        }),
    )
    .await;

    // Transcribe returns content (test-echo echoes the input).
    assert!(
        !results.files[0].content.is_empty(),
        "transcribe smoke should return content"
    );
}

/// Align: per-file audio pipeline (GPU profile for FA).
/// Test-echo workers process this without real audio.
#[tokio::test]
async fn smoke_align() {
    let python = require_python!();
    let (base_url, _tmp, _state, _permit) = start_smoke_server(&python).await;
    let client = reqwest::Client::new();

    let results = submit_and_verify(
        &client,
        &base_url,
        ReleasedCommand::Align,
        CommandOptions::Align(AlignOptions::default()),
    )
    .await;

    assert!(
        !results.files[0].content.is_empty(),
        "align smoke should return content"
    );
}
