//! Utility functions, constants, and auto-tuning for the runner module.

mod auto_tune;
pub(crate) mod batch_progress;
mod error_classification;
mod file_status;

/// Shared recording event sink for tests in other modules.
#[cfg(test)]
pub(crate) use file_status::test_sink;
mod media;

// Re-export everything at the same paths callers already use.
pub(super) use auto_tune::compute_job_workers;

pub(crate) use error_classification::classify_server_error;
pub(super) use error_classification::user_facing_error;
pub(crate) use error_classification::{classify_worker_error, is_retryable_worker_failure};

pub(crate) use file_status::{
    FileRunTracker, FileStage, FileTaskOutcome, ProgressSender, ProgressUpdate, RunnerEventSink,
    StoreRunnerEventSink, set_file_progress,
};
pub(super) use file_status::{
    drain_supervised_file_tasks, force_terminal_file_states, record_file_cancelled_before_dispatch,
    spawn_progress_forwarder, spawn_supervised_file_task,
};

#[cfg(test)]
pub(super) use media::apply_result_filename;
pub(super) use media::{compute_audio_identity, get_audio_duration_ms, preflight_validate_media};

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::api::{DisplayPath, NumWorkers, ReleasedCommand};
    use crate::config::ServerConfig;
    use crate::error::{MissingForcedAlignmentEvidence, MissingRequiredEvidence, ServerError};
    use crate::host_facts::EffectiveConfig;
    use crate::runtime;

    /// Resolve an [`EffectiveConfig`] for a test against the live
    /// host's detected facts, same wiring as production
    /// `DispatchHostContext::from_store`. Tests on Apple Silicon vs
    /// CUDA hosts will see different `gpu_thread_pool_size`
    /// recommendations, so the `compute_workers_*` assertions use
    /// inequality bounds (`<=`) where the cap matters.
    fn effective_for_test(config: &ServerConfig) -> EffectiveConfig {
        EffectiveConfig::resolve_from_server_config(config)
    }
    use crate::scheduling::FailureCategory;
    use crate::store::PendingJobFile;
    use crate::worker::error::WorkerError;

    #[tokio::test]
    async fn audio_identity_resolves_media_aliases() {
        let directory = tempfile::tempdir().expect("temporary media directory");
        let media = directory.path().join("audio.wav");
        std::fs::write(&media, b"media identity fixture").expect("write media");
        let direct = compute_audio_identity(media.to_str().expect("UTF-8 path"))
            .await
            .expect("media identity");
        let dotted = directory.path().join(".").join("audio.wav");
        assert_eq!(
            Some(direct.clone()),
            compute_audio_identity(dotted.to_str().expect("UTF-8 alias")).await,
            "path spelling must not trigger another ASR run"
        );
        #[cfg(unix)]
        {
            let alias = directory.path().join("staged.wav");
            std::os::unix::fs::symlink(&media, &alias).expect("media symlink");
            assert_eq!(
                Some(direct),
                compute_audio_identity(alias.to_str().expect("UTF-8 alias")).await,
                "staging symlinks must share the original media identity"
            );
        }
    }

    #[test]
    fn apply_result_filename_basic() {
        use std::path::PathBuf;
        assert_eq!(
            apply_result_filename(std::path::Path::new("/out/dir/audio.wav"), "audio.cha"),
            PathBuf::from("/out/dir/audio.cha")
        );
    }

    #[test]
    fn apply_result_filename_preserves_dir() {
        use std::path::PathBuf;
        assert_eq!(
            apply_result_filename(
                std::path::Path::new("/data/corpus/output/test.mp3"),
                "test.cha"
            ),
            PathBuf::from("/data/corpus/output/test.cha")
        );
    }

    #[test]
    fn compute_workers_single_file() {
        let config = ServerConfig::default();
        let effective = effective_for_test(&config);
        assert_eq!(
            compute_job_workers(ReleasedCommand::Morphotag, 1, &effective, &config),
            NumWorkers(1)
        );
    }

    #[test]
    fn compute_workers_explicit_config() {
        // The explicit cap is a CEILING. `compute_job_workers` also
        // clamps by `available_parallelism()` and per-category caps,
        // both of which are host-derived and shrink on small CI
        // runners. The right invariant is `1 <= result <= explicit`
        //, same shape as `compute_workers_auto_tune_caps_at_max`
        // below. Asserting `==` overspecified and only worked on
        // many-core dev machines.
        let config = ServerConfig {
            max_workers_per_job: Some(3),
            ..Default::default()
        };
        let effective = effective_for_test(&config);
        let result = compute_job_workers(ReleasedCommand::Morphotag, 10, &effective, &config);
        assert!(*result >= 1 && *result <= 3, "got {result:?}");
    }

    #[test]
    fn compute_workers_explicit_caps_at_file_count() {
        // Same reasoning as `compute_workers_explicit_config`:
        // ceiling, not equality. The intent is "explicit cap of 8 is
        // honored AND further clamped to file count of 2".
        let config = ServerConfig {
            max_workers_per_job: Some(8),
            ..Default::default()
        };
        let effective = effective_for_test(&config);
        let result = compute_job_workers(ReleasedCommand::Morphotag, 2, &effective, &config);
        assert!(*result >= 1 && *result <= 2, "got {result:?}");
    }

    #[test]
    fn compute_workers_auto_tune_caps_at_max() {
        let config = ServerConfig::default();
        let effective = effective_for_test(&config);
        // Auto-tune should never exceed runtime::max_thread_workers()
        let result = compute_job_workers(ReleasedCommand::Opensmile, 100, &effective, &config);
        assert!(*result <= runtime::max_thread_workers());
        assert!(*result >= 1);
    }

    #[test]
    fn compute_workers_gpu_commands_use_thread_pool_cap() {
        let config = ServerConfig {
            gpu_thread_pool_size: Some(2),
            ..Default::default()
        };
        let effective = effective_for_test(&config);
        let result = compute_job_workers(ReleasedCommand::Transcribe, 47, &effective, &config);
        assert!(
            *result <= 2,
            "GPU worker count must respect gpu_thread_pool_size, got {result}"
        );
    }

    #[test]
    fn compute_workers_force_cpu_treats_gpu_commands_as_cpu_bound() {
        // When the host runs with `force_cpu=true` (Apple Silicon with
        // MPS excluded, hosts without CUDA, operator-set CPU mode),
        // GPU-tagged commands like Transcribe execute on CPU workers.
        // There is no shared GPU model to contend on, so
        // `gpu_thread_pool_size` must NOT clamp the worker count. The
        // integration test `tests/serve_start_workers_persisted.rs` pins
        // the same invariant at the CLI subprocess seam; this unit test
        // guards it at the planner level.
        //
        // The absolute worker count is host-derived (`compute_job_workers`
        // clamps by `available_parallelism()` and the resolved memory
        // tier, both small on CI runners), so asserting `>= 2` baked in a
        // many-core dev-box assumption that fails on a small runner.
        // Assert the host-INDEPENDENT invariant instead: under force_cpu,
        // `gpu_thread_pool_size` has no effect at all. With the force_cpu
        // guard removed, a GPU command with a tiny pool would clamp to the
        // pool size while a large pool would not, so the two results would
        // diverge; this equality catches that regression on any core count.
        let small_pool = ServerConfig {
            max_workers_per_job: Some(4),
            gpu_thread_pool_size: Some(1),
            force_cpu: Some(true),
            ..Default::default()
        };
        let large_pool = ServerConfig {
            max_workers_per_job: Some(4),
            gpu_thread_pool_size: Some(64),
            force_cpu: Some(true),
            ..Default::default()
        };
        let small = compute_job_workers(
            ReleasedCommand::Transcribe,
            6,
            &effective_for_test(&small_pool),
            &small_pool,
        );
        let large = compute_job_workers(
            ReleasedCommand::Transcribe,
            6,
            &effective_for_test(&large_pool),
            &large_pool,
        );
        assert_eq!(
            small, large,
            "under force_cpu, gpu_thread_pool_size must not affect the worker \
             count (gpu_pool=1 gave {small:?}, gpu_pool=64 gave {large:?})"
        );
    }

    #[test]
    fn compute_workers_gpu_commands_respect_medium_tier_cap() {
        let config = ServerConfig {
            memory_tier: Some(crate::types::runtime::MemoryTierKind::Medium),
            gpu_thread_pool_size: Some(4),
            ..Default::default()
        };
        let effective = effective_for_test(&config);
        let result = compute_job_workers(ReleasedCommand::Align, 7, &effective, &config);
        assert_eq!(
            result,
            NumWorkers(1),
            "Medium tier GPU jobs should not exceed the tier worker cap (1 for LazyProfile)"
        );
    }

    #[test]
    fn compute_workers_cpu_commands_respect_medium_tier_cap() {
        let config = ServerConfig {
            memory_tier: Some(crate::types::runtime::MemoryTierKind::Medium),
            ..Default::default()
        };
        let effective = effective_for_test(&config);
        let result = compute_job_workers(ReleasedCommand::Opensmile, 100, &effective, &config);
        assert_eq!(
            result,
            NumWorkers(1),
            "Medium tier auto-tune should not exceed the tier worker cap (1 for LazyProfile)"
        );
    }

    #[test]
    fn command_execution_budgets_follow_runtime_constants() {
        use batchalign_types::api::ReleasedCommand;
        use batchalign_types::command_spec::command_spec_for;

        // In test environment PYTHON_GIL is not set to "0", so the process
        // (conservative) budget table is used by default (not free-threaded).
        let free_threaded = false;

        let morphotag = command_spec_for(ReleasedCommand::Morphotag);
        assert_eq!(
            (morphotag.base_mb_for_runtime(free_threaded).0 as f64 * morphotag.loading_overhead.0)
                as u64,
            (8_000f64 * 1.5) as u64,
        );

        let align = command_spec_for(ReleasedCommand::Align);
        assert_eq!(
            (align.base_mb_for_runtime(free_threaded).0 as f64 * align.loading_overhead.0) as u64,
            (4_000f64 * 1.5) as u64,
        );

        let opensmile = command_spec_for(ReleasedCommand::Opensmile);
        assert_eq!(
            (opensmile.base_mb_for_runtime(free_threaded).0 as f64 * opensmile.loading_overhead.0)
                as u64,
            (500f64 * 1.5) as u64,
        );
    }

    #[test]
    fn worker_error_classification_is_stable() {
        assert_eq!(
            classify_worker_error(&WorkerError::ProcessExited {
                code: Some(9),
                stderr: None
            }),
            FailureCategory::WorkerCrash
        );
        assert_eq!(
            classify_worker_error(&WorkerError::ReadyTimeout { timeout_s: 30 }),
            FailureCategory::WorkerTimeout
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Protocol(
                "timeout waiting for infer response".into()
            )),
            FailureCategory::WorkerTimeout
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Protocol("bad frame".into())),
            FailureCategory::WorkerProtocol
        );
        assert_eq!(
            classify_worker_error(&WorkerError::WorkerResponse("temporary".into())),
            FailureCategory::ProviderTransient
        );
        assert!(is_retryable_worker_failure(FailureCategory::WorkerCrash));
        assert!(is_retryable_worker_failure(FailureCategory::WorkerTimeout));
        assert!(is_retryable_worker_failure(
            FailureCategory::ProviderTransient
        ));
        assert!(!is_retryable_worker_failure(
            FailureCategory::WorkerProtocol
        ));
        assert!(!is_retryable_worker_failure(FailureCategory::Validation));
    }

    #[test]
    fn required_evidence_refusal_is_actionable_and_not_a_system_error() {
        let error = ServerError::RequiredEvidenceUnavailable(
            MissingRequiredEvidence::ForcedAlignment(MissingForcedAlignmentEvidence::new(1, &[3])),
        );
        let category = classify_server_error(&error);
        assert_eq!(category, FailureCategory::EvidenceUnavailable);

        let message = user_facing_error(
            category,
            "Alignment",
            "test.cha",
            &format!("Alignment failed: {error}"),
        );
        assert!(message.contains("--require-media-cache"));
        assert!(message.contains("groups [1, 3]"));
        assert!(!message.contains("internal error"));
    }

    /// Regression: bootstrap-class errors must NOT be retried.
    ///
    /// Historical context: a Stanza-catalog-missing failure during worker
    /// bootstrap was classified as a transient `WorkerCrash` and retried
    /// 3× with a full Python traceback per attempt. On a long-running
    /// daemon this produced multi-GB log explosions from a deterministic
    /// config error. The fix: the Python worker surfaces typed bootstrap
    /// failures via the `{"op":"error", "kind":"bootstrap"}` wire shape;
    /// the Rust side classifies these as terminal so the orchestrator
    /// stops retrying.
    #[test]
    fn bootstrap_errors_are_non_retryable() {
        let category = classify_worker_error(&WorkerError::Bootstrap(
            "Failed to download Stanza resource catalog from \
             https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main: \
             connection refused"
                .into(),
        ));
        assert!(
            !is_retryable_worker_failure(category),
            "Bootstrap-class errors must be terminal at the worker layer; \
             retrying a deterministic config error 3× is the bug shape this \
             test is pinning. Got category: {category:?}"
        );
    }

    #[test]
    fn process_exited_display_includes_stderr_when_present() {
        let error = WorkerError::ProcessExited {
            code: Some(1),
            stderr: Some("Traceback (most recent call last):\n  File \"worker.py\", line 42\ntorch.cuda.OutOfMemoryError: CUDA out of memory".into()),
        };
        let msg = error.to_string();
        assert!(
            msg.contains("exit code: Some(1)"),
            "should show exit code: {msg}"
        );
        assert!(
            msg.contains("OutOfMemoryError"),
            "should include stderr tail: {msg}"
        );
        assert!(
            msg.contains("--- worker stderr ---"),
            "should have stderr header: {msg}"
        );
    }

    #[test]
    fn process_exited_display_without_stderr_is_clean() {
        let error = WorkerError::ProcessExited {
            code: Some(9),
            stderr: None,
        };
        let msg = error.to_string();
        assert!(
            msg.contains("exit code: Some(9)"),
            "should show exit code: {msg}"
        );
        assert!(!msg.contains("stderr"), "should not mention stderr: {msg}");
    }

    #[test]
    fn worker_crash_user_facing_error_includes_raw_detail() {
        let raw = "FA processing failed: worker process exited unexpectedly (exit code: Some(-9))\n--- worker stderr ---\nKilled: 9";
        let msg = user_facing_error(FailureCategory::WorkerCrash, "Alignment", "test.cha", raw);
        assert!(
            msg.contains("Killed: 9"),
            "should include stderr detail: {msg}"
        );
        assert!(msg.contains("test.cha"), "should include filename: {msg}");
    }

    #[tokio::test]
    async fn media_validate_not_paths_mode() {
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "audio.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![batchalign_types::paths::ClientPath::new(
            "/nonexistent/audio.wav",
        )];
        let failures = preflight_validate_media(&file_list, &source_paths, false).await;
        assert!(
            failures.is_empty(),
            "Should skip validation when not paths_mode"
        );
    }

    #[tokio::test]
    async fn media_validate_skips_chat_files() {
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.cha".into(),
            has_chat: true,
        }];
        let source_paths = vec![batchalign_types::paths::ClientPath::new(
            "/nonexistent/test.cha",
        )];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert!(failures.is_empty(), "Should skip CHAT files");
    }

    #[tokio::test]
    async fn media_validate_missing_file() {
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "missing.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![batchalign_types::paths::ClientPath::new(
            "/nonexistent/path/missing.wav",
        )];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert_eq!(failures.len(), 1);
        assert!(failures[&0].contains("not found"));
    }

    #[tokio::test]
    async fn media_validate_empty_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // File is empty (0 bytes)
        let path = tmp.path().to_string_lossy().to_string();
        // Rename to .wav for extension check
        let wav_path = format!("{path}.wav");
        std::fs::rename(&path, &wav_path).unwrap();
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![batchalign_types::paths::ClientPath::new(&wav_path)];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert_eq!(failures.len(), 1);
        assert!(failures[&0].contains("empty"));
        let _ = std::fs::remove_file(&wav_path);
    }

    #[tokio::test]
    async fn media_validate_unknown_extension() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".xyz").unwrap();
        std::io::Write::write_all(&mut tmp, b"data").unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.xyz".into(),
            has_chat: false,
        }];
        let source_paths = vec![batchalign_types::paths::ClientPath::new(&path)];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert_eq!(failures.len(), 1);
        assert!(failures[&0].contains("Unknown media format"));
    }

    #[tokio::test]
    async fn media_validate_valid_file() {
        let mut tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        std::io::Write::write_all(&mut tmp, b"RIFF....").unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let file_list = vec![PendingJobFile {
            file_index: 0,
            filename: "test.wav".into(),
            has_chat: false,
        }];
        let source_paths = vec![batchalign_types::paths::ClientPath::new(&path)];
        let failures = preflight_validate_media(&file_list, &source_paths, true).await;
        assert!(failures.is_empty(), "Valid .wav file should pass");
    }

    #[tokio::test]
    async fn media_validate_all_known_extensions() {
        // Enumerated through the same verb production uses, so a format added
        // to `MediaExtensions` is covered here without anyone remembering to
        // widen a second list. That was the previous arrangement, and the two
        // lists it created disagreed by two formats.
        for name in crate::media::MediaExtensions::candidates("test") {
            let extension = Path::new(&name)
                .extension()
                .expect("candidates always carry one")
                .to_string_lossy()
                .to_string();
            let mut tmp = tempfile::NamedTempFile::with_suffix(format!(".{extension}")).unwrap();
            std::io::Write::write_all(&mut tmp, b"data").unwrap();
            let path = tmp.path().to_string_lossy().to_string();
            let file_list = vec![PendingJobFile {
                file_index: 0,
                filename: DisplayPath::from(name.clone()),
                has_chat: false,
            }];
            let source_paths = vec![batchalign_types::paths::ClientPath::new(&path)];
            let failures = preflight_validate_media(&file_list, &source_paths, true).await;
            assert!(failures.is_empty(), "{name} should be accepted");
        }
    }

    /// Verifies media_roots fallback: audio not alongside cha or in source_dir,
    /// but present in a configured media_root.
    #[tokio::test]
    async fn resolve_audio_media_roots_fallback() {
        let media_root = tempfile::tempdir().unwrap();
        let wav = media_root.path().join("ACWT01a.wav");
        std::fs::write(&wav, b"RIFF").unwrap();

        // Calls the real resolution verb. This test used to re-implement the
        // production loop and then assert its own copy worked, so it could not
        // have failed for any change to production code.
        let found = crate::media::MediaExtensions::find_in(media_root.path(), "ACWT01a").await;
        assert!(found.is_some(), "Should find audio in media_root");
        assert!(found.unwrap().ends_with("ACWT01a.wav"));
    }
}
