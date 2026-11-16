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
pub(crate) use error_classification::classify_worker_error;
pub(super) use error_classification::user_facing_error;

pub(crate) use file_status::{
    AfterFailure, FileAttempts, FileRunTracker, FileStage, FileTaskOutcome, ProgressSender,
    ProgressUpdate, RunnerEventSink, StoreRunnerEventSink, TerminalRecorded, set_file_progress,
};
pub(crate) use file_status::{
    drain_supervised_file_tasks, observing_file_waits, record_file_cancelled_before_dispatch,
    spawn_observed_progress_forwarder, spawn_supervised_file_task,
};

#[cfg(test)]
pub(super) use media::apply_result_filename;
pub(super) use media::compute_audio_identity;
pub(crate) use media::probe_audio_duration;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::api::{NumWorkers, ReleasedCommand};
    use crate::config::ServerConfig;
    use crate::error::{MissingForcedAlignmentEvidence, MissingRequiredEvidence, ServerError};
    use crate::scheduling::FailureCategory;
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

    /// Guards: File, CPU, operator, GPU and tier ceilings are resolved from the declared host even when the running machine has other resources.
    /// Kind: policy
    /// Not a type: HostSnapshot binds the facts together, while the chosen numerical allocation among valid policy alternatives requires checking the resulting worker counts.
    /// State: None. Each row resolves the standard declared laptop with explicit policy overrides. No ledger lease or host observation is taken.
    /// Concurrency: None. Snapshots and configuration values are owned by each row.
    #[test]
    fn worker_planning_uses_one_declared_host() {
        use crate::types::runtime::MemoryTierKind;
        let base = ServerConfig {
            max_workers_per_job: Some(64),
            memory_tier: Some(MemoryTierKind::Fleet),
            force_cpu: Some(true),
            ..Default::default()
        };
        let cases = [
            (ReleasedCommand::Morphotag, 1, base.clone(), 1),
            (ReleasedCommand::Morphotag, 2, base.clone(), 2),
            (ReleasedCommand::Morphotag, 40, base.clone(), 4),
            (
                ReleasedCommand::Morphotag,
                40,
                ServerConfig {
                    max_workers_per_job: Some(3),
                    ..base.clone()
                },
                3,
            ),
            (
                ReleasedCommand::Transcribe,
                40,
                ServerConfig {
                    force_cpu: Some(false),
                    gpu_thread_pool_size: Some(2),
                    ..base.clone()
                },
                2,
            ),
            (
                ReleasedCommand::Transcribe,
                40,
                ServerConfig {
                    force_cpu: Some(false),
                    gpu_thread_pool_size: Some(1),
                    ..base.clone()
                },
                1,
            ),
            (
                ReleasedCommand::Transcribe,
                40,
                ServerConfig {
                    gpu_thread_pool_size: Some(1),
                    ..base.clone()
                },
                4,
            ),
            (
                ReleasedCommand::Transcribe,
                40,
                ServerConfig {
                    gpu_thread_pool_size: Some(64),
                    ..base.clone()
                },
                4,
            ),
            (
                ReleasedCommand::Opensmile,
                40,
                ServerConfig {
                    memory_tier: Some(MemoryTierKind::Medium),
                    ..base.clone()
                },
                1,
            ),
            (
                ReleasedCommand::Align,
                40,
                ServerConfig {
                    memory_tier: Some(MemoryTierKind::Medium),
                    force_cpu: Some(false),
                    ..base
                },
                1,
            ),
        ];
        for (command, files, config, expected) in cases {
            let host = crate::host_facts::HostSnapshot::for_tests(&config);
            assert_eq!(
                compute_job_workers(command, files, &host),
                NumWorkers(expected),
                "{command} with {files} files and {config:?}"
            );
        }
    }

    #[test]
    fn command_execution_budgets_follow_runtime_constants() {
        use batchalign_types::api::ReleasedCommand;
        use batchalign_types::command_spec::command_spec_for;

        // The GIL-enabled (process) budgets.
        let free_threaded = batchalign_types::command_spec::PythonRuntime::Gil;

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
        let native = classify_worker_error(&WorkerError::NativeCommand {
            command: crate::api::ReleasedCommand::Align,
        });
        assert_eq!(native, FailureCategory::System);
        assert!(!((native).retryability() == crate::types::scheduling::Retryability::Retryable));
        assert_eq!(
            classify_worker_error(&WorkerError::ProcessExited {
                code: Some(9),
                stderr: None
            }),
            FailureCategory::WorkerCrash
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Stalled {
                waited_for: crate::worker::LoadingWait::Ready,
                stall: crate::worker::LoadingStall::new(
                    crate::api::PositiveSeconds::literal::<30>()
                ),
            }),
            FailureCategory::WorkerTimeout
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Timeout {
                waited_for: crate::worker::error::WorkerWait::Infer,
                limit: crate::api::PositiveSeconds::literal::<120>(),
            }),
            FailureCategory::WorkerTimeout
        );
        // A protocol message that merely mentions a timeout (worker text the
        // reader quotes) is a protocol failure, never a retryable timeout.
        assert_eq!(
            classify_worker_error(&WorkerError::Protocol(
                "failed to decode execute_v2 response: field timeout_s".into()
            )),
            FailureCategory::WorkerProtocol
        );
        assert_eq!(
            classify_worker_error(&WorkerError::Protocol("bad frame".into())),
            FailureCategory::WorkerProtocol
        );
        assert_eq!(
            classify_worker_error(&WorkerError::WorkerResponse("temporary".into())),
            FailureCategory::ProviderTransient
        );
        assert!(
            ((FailureCategory::WorkerCrash).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
        assert!(
            ((FailureCategory::WorkerTimeout).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
        assert!(
            ((FailureCategory::ProviderTransient).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
        // A refused spawn and a device shortfall clear when memory frees.
        assert!(
            ((FailureCategory::MemoryPressure).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
        assert!(
            !((FailureCategory::WorkerProtocol).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
        assert!(
            !((FailureCategory::Validation).retryability()
                == crate::types::scheduling::Retryability::Retryable)
        );
    }

    #[test]
    fn required_evidence_refusal_is_actionable_and_not_a_system_error() {
        let error = ServerError::RequiredEvidenceUnavailable(
            MissingRequiredEvidence::ForcedAlignment(MissingForcedAlignmentEvidence::new(1, [3])),
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
        assert!(message.contains("requests [1, 3]"));
        assert!(!message.contains("internal error"));
        assert!(message.contains("could not complete"));
        assert!(!message.contains("did not run"));
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
        let category = classify_worker_error(&WorkerError::Bootstrap {
            message: "Stanza has no pipeline for 'que': its catalog lacks the language".into(),
            worker: crate::worker::error::WorkerAfterError::Exits,
        });
        assert!(
            !((category).retryability() == crate::types::scheduling::Retryability::Retryable),
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
        let found =
            crate::media::MediaExtensions::find_under(media_root.path(), Path::new(""), "ACWT01a");
        assert_eq!(found, crate::media::MediaLookup::Found(wav));
    }
}
