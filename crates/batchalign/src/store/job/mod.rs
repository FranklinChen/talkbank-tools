//! In-memory job model and lifecycle methods.
//!
//! Split into focused submodules by responsibility:
//!
//! - [`types`]: struct definitions (`Job`, `JobIdentity`, runner snapshots, etc.)
//! - [`lease`]: queue lease management (claim, renew, release, expiry)
//! - [`file_status`]: per-file status mutations (processing, done, error, retry)
//! - [`lifecycle`]: job-level state transitions (running, failed, cancelled, restart, recovery)
//! - [`projections`]: API response projections (`JobInfo`, `JobListItem`, `RunnerJobSnapshot`)
//! - [`conflict`]: file-level conflict detection between incoming and active jobs

mod conflict;
mod file_status;
mod lease;
mod lifecycle;
mod projections;
mod types;

#[cfg(test)]
pub(crate) mod test_support;

pub use conflict::*;
pub(crate) use projections::{JobStatusColumns, Stop};
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{
        ContentType, CorrelationId, DisplayPath, FileProgressStage, FileStatusKind, JobId,
        JobStatus, LanguageSpec, MachineTime, NodeId, NumSpeakers, ReleasedCommand,
    };
    use crate::options::CommandOptions;
    use crate::scheduling::LeaseRecord;
    use crate::store::{FileResultEntry, FileStatus};
    use std::collections::{BTreeMap, HashMap};
    use tokio_util::sync::CancellationToken;

    /// Build a small queued job for projection and conflict tests.
    fn sample_job(job_id: &str, filenames: &[&str]) -> Job {
        let file_statuses = filenames
            .iter()
            .map(|filename| {
                let name = DisplayPath::from(*filename);
                (String::from(name.clone()), FileStatus::new(name))
            })
            .collect();
        let has_chat = filenames.iter().map(|_| true).collect();

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: CorrelationId::from(format!("corr-{job_id}")),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Morphotag,
                lang: LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Morphotag(crate::options::MorphotagOptions {
                    common: crate::options::CommonOptions::default(),

                    ..Default::default()
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitter: Some(crate::store::Submitter::client(
                    std::net::Ipv4Addr::LOCALHOST.into(),
                    "localhost".into(),
                )),
                source_dir: "/corpus".into(),
            },
            filesystem: JobFilesystemConfig {
                filenames: filenames
                    .iter()
                    .map(|filename| DisplayPath::from(*filename))
                    .collect(),
                has_chat,
                staging_dir: "/tmp/job".into(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: Default::default(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
            },
            schedule: JobScheduleState {
                submitted_at: crate::unix_time(100.0),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: None,
                last_cancel: None,
            },
            runtime: JobRuntimeControl {
                cancel_token: CancellationToken::new(),
                runner_active: false,
                run_generation: crate::store::RunGeneration::FIRST,
            },
            execution_plan: None,
        }
    }

    /// Pending files exclude terminal file states.
    #[test]
    fn pending_files_skip_terminal_entries() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.file_statuses.get_mut("a.cha").unwrap().phase =
            crate::store::FilePhase::Done {
                started_at: None,
                finished_at: Some(crate::unix_time(5.0)),
            };

        let pending = job.pending_files();

        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].filename.as_ref(), "b.cha");
    }

    /// Conflict detection keys on submitter and source-scoped filename.
    #[test]
    fn find_conflicts_uses_submitter_and_source_scope() {
        let mut active = sample_job("active", &["a.cha"]);
        active.execution.status = JobStatus::Running;

        let incoming = sample_job("incoming", &["a.cha"]);
        let jobs = HashMap::from([(active.identity.job_id.clone(), active)]);

        let conflicts = find_conflicts(&jobs, &incoming);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].filename, "a.cha");
        assert_eq!(conflicts[0].job_id.as_ref(), "active");
    }

    /// `find_conflicts` trusts the store HashMap it receives. This layer is
    /// intentionally pure over current store state: any non-terminal entry is
    /// treated as a conflict until normal shutdown/recovery transitions move it
    /// forward.
    #[test]
    fn find_conflicts_trusts_store_state_and_does_not_reconcile() {
        let mut abandoned = sample_job("abandoned-job", &["020724a.mp3"]);
        abandoned.execution.status = JobStatus::Queued;
        abandoned.schedule.submitted_at =
            MachineTime::now().minus(std::time::Duration::from_secs(3 * 3600));
        abandoned.runtime.runner_active = false;

        let incoming = sample_job("resubmit", &["020724a.mp3"]);
        let jobs = HashMap::from([(abandoned.identity.job_id.clone(), abandoned)]);

        let conflicts = find_conflicts(&jobs, &incoming);

        assert_eq!(
            conflicts.len(),
            1,
            "find_conflicts is pure over the current store HashMap. Cleanup \
             of stale jobs happens through normal shutdown/recovery paths, not \
             inside conflict detection."
        );
    }

    /// Restart preparation keeps successful work and resets unfinished state.
    #[test]
    fn prepare_for_restart_resets_unfinished_state() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.status = JobStatus::Failed;
        job.execution.error = Some("failed".into());
        job.execution.file_statuses.get_mut("a.cha").unwrap().phase =
            crate::store::FilePhase::Done {
                started_at: None,
                finished_at: Some(crate::unix_time(5.0)),
            };
        let retry_file = job.execution.file_statuses.get_mut("b.cha").unwrap();
        retry_file.phase = crate::store::FilePhase::Error {
            started_at: Some(crate::unix_time(10.0)),
            finished_at: Some(crate::unix_time(12.0)),
            failure: crate::store::FileFailure::of_failed_row(Some("boom".into()), None),
        };
        retry_file.progress.stage = Some(FileProgressStage::Aligning);
        job.execution.results.push(FileResultEntry {
            filename: DisplayPath::from("a.cha"),
            content_type: ContentType::Chat,
            error: None,
        });
        job.execution.results.push(FileResultEntry {
            filename: DisplayPath::from("b.cha"),
            content_type: ContentType::Chat,
            error: Some("boom".into()),
        });
        job.schedule.completed_at = Some(crate::unix_time(20.0));
        job.schedule.next_eligible_at = Some(crate::unix_time(25.0));
        job.schedule.lease = Some(
            LeaseRecord::new(
                NodeId::from("node-1"),
                crate::unix_time(28.0),
                crate::unix_time(30.0),
            )
            .expect("an ordered fixture lease"),
        );
        job.runtime.runner_active = true;

        job.prepare_for_restart();

        assert_eq!(job.execution.status, JobStatus::Queued);
        assert_eq!(job.execution.error, None);
        assert_eq!(job.execution.completed_files, 1);
        assert_eq!(job.execution.results.len(), 1);
        assert_eq!(
            job.execution.file_statuses["a.cha"].status(),
            FileStatusKind::Done
        );
        assert_eq!(
            job.execution.file_statuses["b.cha"].status(),
            FileStatusKind::Queued
        );
        assert!(job.schedule.completed_at.is_none());
        assert!(job.schedule.next_eligible_at.is_none());
        assert!(job.schedule.lease.is_none());
        // Restart must NOT pretend the old runner is gone: the claim is
        // released only by the runner itself, and the restarted runner
        // waits on it (begin_runner). What restart does do is move the
        // job to a new run generation, making the old runner stale.
        assert!(job.runtime.runner_active);
        assert_eq!(
            job.runtime.run_generation,
            crate::store::RunGeneration::FIRST.next()
        );
    }

    /// Recovery re-queues interrupted jobs when resumable file work remains.
    #[test]
    fn reconcile_recovered_runtime_state_requeues_resumable_files() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.status = JobStatus::Running;
        job.execution.file_statuses.get_mut("a.cha").unwrap().phase =
            crate::store::FilePhase::Done {
                started_at: None,
                finished_at: Some(crate::unix_time(5.0)),
            };
        let resumable = job.execution.file_statuses.get_mut("b.cha").unwrap();
        resumable.phase = crate::store::FilePhase::Interrupted {
            started_at: Some(crate::unix_time(10.0)),
            last_failure: None,
        };
        job.execution.error = Some("an earlier run failed".into());
        job.schedule.completed_at = Some(crate::unix_time(20.0));
        job.schedule.next_eligible_at = Some(crate::unix_time(21.0));
        job.schedule.lease = Some(
            LeaseRecord::new(
                NodeId::from("node-1"),
                crate::unix_time(28.0),
                crate::unix_time(30.0),
            )
            .expect("an ordered fixture lease"),
        );

        let disposition = job.reconcile_recovered_runtime_state();

        assert_eq!(disposition, RecoveryDisposition::Requeued);
        assert_eq!(job.execution.status, JobStatus::Queued);
        assert_eq!(
            job.status_columns().error,
            None,
            "a requeued job carries no earlier error into its row"
        );
        assert_eq!(
            job.execution.file_statuses["b.cha"].status(),
            FileStatusKind::Queued
        );
        assert_eq!(
            job.execution.file_statuses["b.cha"].phase,
            crate::store::FilePhase::Queued,
            "a requeued file keeps no times"
        );
        assert!(job.schedule.completed_at.is_none());
        assert!(job.schedule.lease.is_none());
    }

    /// Recovery promotes all-terminal interrupted jobs to a lease-free final state.
    #[test]
    fn reconcile_recovered_runtime_state_promotes_terminal_jobs() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        job.execution.status = JobStatus::Interrupted;
        job.execution.file_statuses.get_mut("a.cha").unwrap().phase =
            crate::store::FilePhase::Done {
                started_at: None,
                finished_at: Some(crate::unix_time(5.0)),
            };
        let failed = job.execution.file_statuses.get_mut("b.cha").unwrap();
        failed.phase = crate::store::FilePhase::Error {
            started_at: None,
            finished_at: None,
            failure: crate::store::FileFailure::of_failed_row(Some("boom".into()), None),
        };
        job.schedule.completed_at = Some(crate::unix_time(20.0));
        job.schedule.next_eligible_at = Some(crate::unix_time(21.0));
        job.schedule.lease = Some(
            LeaseRecord::new(
                NodeId::from("node-1"),
                crate::unix_time(28.0),
                crate::unix_time(30.0),
            )
            .expect("an ordered fixture lease"),
        );

        let disposition = job.reconcile_recovered_runtime_state();

        assert_eq!(disposition, RecoveryDisposition::Completed);
        assert_eq!(job.execution.status, JobStatus::Completed);
        assert_eq!(job.execution.completed_files, 2);
        assert!(job.schedule.next_eligible_at.is_none());
        assert!(job.schedule.lease.is_none());
    }

    /// Local queue claims and renewals stay on the job boundary.
    #[test]
    fn local_dispatch_claim_and_renew_roundtrip() {
        let mut job = sample_job("job-1", &["a.cha"]);
        let node_id = NodeId::from("node-a");
        let claimed = job
            .claim_for_local_dispatch(
                &node_id,
                crate::unix_time(10.0),
                crate::config::LeaseTtl::from_secs(90).unwrap(),
            )
            .expect("claim");

        assert_eq!(claimed.leased_by_node(), &node_id);
        assert_eq!(claimed.heartbeat_at(), crate::unix_time(10.0));
        assert_eq!(claimed.expires_at(), crate::unix_time(100.0));
        assert!(job.runtime.runner_active);

        let renewed = job
            .renew_local_dispatch_lease(
                &node_id,
                crate::unix_time(20.0),
                crate::config::LeaseTtl::from_secs(90).unwrap(),
            )
            .expect("renew");
        assert_eq!(renewed.heartbeat_at(), crate::unix_time(20.0));
        assert_eq!(renewed.expires_at(), crate::unix_time(110.0));

        job.release_local_dispatch_claim();
        assert!(!job.runtime.runner_active);
        assert!(job.schedule.lease.is_none());
    }

    /// Jobs with live leases or deferrals do not report ready for dispatch.
    #[test]
    fn ready_for_local_dispatch_respects_leases_and_deferrals() {
        let mut job = sample_job("job-1", &["a.cha"]);
        let now = crate::unix_time(10.0);
        assert!(job.ready_for_local_dispatch(now));

        job.schedule.next_eligible_at = Some(crate::unix_time(20.0));
        assert!(!job.ready_for_local_dispatch(now));
        assert_eq!(
            job.next_local_dispatch_wake_at(now),
            Some(crate::unix_time(20.0))
        );

        job.schedule.next_eligible_at = None;
        job.schedule.lease = Some(
            LeaseRecord::new(
                NodeId::from("node-a"),
                crate::unix_time(10.0),
                crate::unix_time(30.0),
            )
            .expect("an ordered fixture lease"),
        );
        assert!(!job.ready_for_local_dispatch(now));
        assert_eq!(
            job.next_local_dispatch_wake_at(now),
            Some(crate::unix_time(30.0))
        );
    }

    /// A diagnosed file wrote its output: a restart of the job (after another
    /// file failed) keeps it rather than requeueing it, so it is never
    /// retried, and it counts as completed, not failed.
    #[test]
    fn a_restart_keeps_a_diagnosed_file_and_requeues_only_the_failure() {
        let mut job = sample_job("job-1", &["a.cha", "b.cha"]);
        let t = crate::unix_time;
        assert!(job.mark_file_processing("a.cha", t(1.0)));
        assert!(job.mark_file_done(
            "a.cha",
            t(2.0),
            FileCompletion::Diagnosed {
                result: CompletedFileOutput {
                    filename: DisplayPath::from("a.cha"),
                    content_type: ContentType::Chat,
                    stamp: crate::api::FileStampOutcome::Unrecorded,
                },
                diagnostics: crate::api::FileOutputDiagnostics::of_findings(
                    vec![crate::api::FileOutputDiagnostics::coded_finding(
                        "E241",
                        "reserved marker",
                    )],
                    Vec::new(),
                ),
            },
        ));
        assert!(job.mark_file_processing("b.cha", t(1.0)));
        assert!(job.mark_file_error(
            "b.cha",
            &FileFailureRecord {
                message: "worker crashed".into(),
                category: crate::scheduling::FailureCategory::WorkerCrash,
                finished_at: crate::store::EventTime::fixed(t(3.0)),
            },
        ));
        assert!(job.any_terminal_files_failed(), "the error is a failure");
        assert!(
            !job.all_terminal_files_failed(),
            "the diagnosed file is not"
        );

        job.execution.status = crate::api::JobStatus::Failed;
        job.prepare_for_restart();

        let a = job.execution.file_statuses["a.cha"].to_entry();
        assert_eq!(a.status, FileStatusKind::Diagnosed, "never retried");
        assert_eq!(
            job.execution.file_statuses["b.cha"].status(),
            FileStatusKind::Queued
        );
        assert_eq!(job.execution.completed_files, 1);
    }

    /// File completion mutates file state and appends a success result.
    #[test]
    fn mark_file_done_updates_file_state() {
        let mut job = sample_job("job-1", &["a.cha"]);
        assert!(job.mark_file_retry_pending(
            "a.cha",
            &FileRetryRecord {
                message: "stale".into(),
                category: crate::scheduling::FailureCategory::WorkerTimeout,
                finished_at: crate::store::EventTime::fixed(crate::unix_time(11.0)),
                retry_at: crate::unix_time(11.5),
            }
        ));

        assert!(job.mark_file_done(
            "a.cha",
            crate::unix_time(12.0),
            FileCompletion::Clean(CompletedFileOutput {
                filename: DisplayPath::from("a.cha"),
                content_type: ContentType::Chat,
                stamp: crate::api::FileStampOutcome::Unrecorded,
            })
        ));

        let entry = job.execution.file_statuses["a.cha"].to_entry();
        assert_eq!(entry.status, FileStatusKind::Done);
        assert_eq!(entry.finished_at, Some(crate::unix_time(12.0)));
        assert!(entry.error.is_none(), "the retry's error is gone");
        assert!(entry.error_category.is_none());
        assert_eq!(job.execution.completed_files, 1);
        assert_eq!(job.execution.results.len(), 1);
    }

    /// Retry scheduling stays on the job boundary.
    #[test]
    fn mark_file_retry_pending_sets_retry_metadata() {
        let mut job = sample_job("job-1", &["a.cha"]);

        assert!(job.mark_file_retry_pending(
            "a.cha",
            &FileRetryRecord {
                message: "retry".into(),
                category: crate::scheduling::FailureCategory::WorkerTimeout,
                finished_at: crate::store::EventTime::fixed(crate::unix_time(11.0)),
                retry_at: crate::unix_time(20.0),
            }
        ));

        let entry = job.execution.file_statuses["a.cha"].to_entry();
        assert_eq!(entry.status, FileStatusKind::Processing);
        assert_eq!(entry.next_eligible_at, Some(crate::unix_time(20.0)));
        assert_eq!(
            entry.progress_stage,
            Some(FileProgressStage::RetryScheduled)
        );
        // Still in flight: the failed attempt's end is not the file's finish.
        assert_eq!(entry.finished_at, None);
        assert_eq!(entry.duration_s, None);
    }

    /// Clearing retry state also clears stale retry errors before a new attempt.
    #[test]
    fn clear_file_retry_state_clears_retry_error_metadata() {
        let mut job = sample_job("job-1", &["a.cha"]);
        assert!(job.mark_file_retry_pending(
            "a.cha",
            &FileRetryRecord {
                message: "retry".into(),
                category: crate::scheduling::FailureCategory::WorkerTimeout,
                finished_at: crate::store::EventTime::fixed(crate::unix_time(11.0)),
                retry_at: crate::unix_time(20.0),
            }
        ));

        assert!(job.clear_file_retry_state("a.cha"));
        let entry = job.execution.file_statuses["a.cha"].to_entry();
        assert!(entry.error.is_none());
        assert!(entry.error_category.is_none());
        assert!(entry.finished_at.is_none());
        assert!(entry.next_eligible_at.is_none());
    }

    /// A row image: the columns a phase owns, as `from_row` reads them.
    fn row(
        status: FileStatusKind,
        error: Option<&str>,
        started_at: Option<crate::api::MachineTime>,
        finished_at: Option<crate::api::MachineTime>,
        next_eligible_at: Option<crate::api::MachineTime>,
    ) -> crate::store::FilePhaseColumns<'_> {
        crate::store::FilePhaseColumns {
            status,
            error,
            error_category: error.map(|_| crate::scheduling::FailureCategory::WorkerTimeout),
            diagnostics: None,
            started_at,
            finished_at,
            next_eligible_at,
        }
    }

    /// The database boundary rebuilds the one phase a row describes: a
    /// processing row with a deadline is a pending retry whose failed-attempt
    /// end is not a finish time, and an error column a phase cannot hold is
    /// reported rather than kept.
    #[test]
    fn rows_become_the_phase_they_describe() {
        use crate::store::FilePhase;
        let t = crate::unix_time;

        let (phase, dropped) = FilePhase::from_row(row(
            FileStatusKind::Processing,
            Some("timeout"),
            Some(t(10.0)),
            Some(t(11.0)),
            Some(t(20.0)),
        ));
        assert_eq!(dropped, None);
        assert_eq!(phase.kind(), FileStatusKind::Processing);
        assert_eq!(phase.finished_at(), None);
        assert_eq!(phase.next_eligible_at(), Some(t(20.0)));
        assert_eq!(phase.last_activity_at(), Some(t(11.0)));

        let (phase, dropped) = FilePhase::from_row(row(
            FileStatusKind::Queued,
            Some("timeout"),
            Some(t(10.0)),
            Some(t(11.0)),
            None,
        ));
        assert_eq!(phase, FilePhase::Queued);
        assert!(dropped.is_some(), "the stray error is reported");
    }

    /// `FilePhase::columns` and `FilePhase::from_row` are inverses: every
    /// phase written as its column image reads back as itself, with nothing
    /// reported dropped. Two functions that must agree, which no type pins.
    #[test]
    fn every_phase_roundtrips_through_its_columns() {
        use crate::store::{FileFailure, FilePhase};
        let failure = || {
            FileFailure::recorded(
                "timeout".into(),
                crate::scheduling::FailureCategory::WorkerTimeout,
            )
        };
        let t = crate::unix_time;
        let phases = [
            FilePhase::Queued,
            FilePhase::Processing {
                started_at: Some(t(1.0)),
            },
            FilePhase::RetryPending {
                started_at: Some(t(1.0)),
                failed_at: Some(t(2.0)),
                retry_at: t(3.0),
                failure: failure(),
            },
            FilePhase::Done {
                started_at: Some(t(1.0)),
                finished_at: Some(t(2.0)),
            },
            FilePhase::Diagnosed {
                started_at: Some(t(1.0)),
                finished_at: Some(t(2.0)),
                diagnostics: Some(crate::api::FileOutputDiagnostics::of_findings(
                    vec![crate::api::FileOutputDiagnostics::coded_finding(
                        "E220",
                        "digits inside a word",
                    )],
                    vec![crate::api::OutputShortfallRecord::StageSkipped {
                        stage: crate::api::OptionalStage::Morphosyntax,
                    }],
                )),
            },
            FilePhase::Diagnosed {
                started_at: None,
                finished_at: None,
                diagnostics: None,
            },
            FilePhase::Error {
                started_at: None,
                finished_at: Some(t(2.0)),
                failure: failure(),
            },
            FilePhase::Error {
                started_at: None,
                finished_at: None,
                failure: FileFailure::of_failed_row(None, None),
            },
            FilePhase::Interrupted {
                started_at: Some(t(1.0)),
                last_failure: Some(failure()),
            },
            FilePhase::Interrupted {
                started_at: None,
                last_failure: None,
            },
        ];
        for phase in phases {
            let (back, dropped) = FilePhase::from_row(phase.columns());
            assert_eq!(back, phase);
            assert_eq!(dropped, None, "{phase:?} wrote a column it does not own");
        }
    }

    /// A failure is never empty: stored columns holding neither a message nor
    /// a category are no failure, so an interrupted file cannot carry a
    /// failure that writes nothing and reads back as none.
    #[test]
    fn an_empty_failure_is_no_failure() {
        use crate::store::FileFailure;
        assert_eq!(FileFailure::from_columns(None, None), None);
        let unrecorded = FileFailure::of_failed_row(None, None);
        assert_eq!((unrecorded.message(), unrecorded.category()), (None, None));
        let category_only =
            FileFailure::from_columns(None, Some(crate::scheduling::FailureCategory::WorkerCrash))
                .expect("a category alone is a failure");
        assert_eq!(category_only.message(), None);
    }

    /// Interrupting moves only in-flight phases, keeping the start and the
    /// failure a pending retry was retrying.
    #[test]
    fn interrupting_keeps_the_start_and_the_retried_failure() {
        use crate::store::{FileFailure, FilePhase};
        let t = crate::unix_time;
        let failure = FileFailure::recorded(
            "crash".into(),
            crate::scheduling::FailureCategory::WorkerCrash,
        );
        let retry = FilePhase::RetryPending {
            started_at: Some(t(1.0)),
            failed_at: Some(t(2.0)),
            retry_at: t(3.0),
            failure: failure.clone(),
        };
        assert_eq!(
            retry.interrupted(),
            FilePhase::Interrupted {
                started_at: Some(t(1.0)),
                last_failure: Some(failure),
            }
        );
        let done = FilePhase::Done {
            started_at: Some(t(1.0)),
            finished_at: Some(t(2.0)),
        };
        assert_eq!(done.clone().interrupted(), done);
    }

    /// Every phase reports the columns it cannot hold, by one rule.
    #[test]
    fn a_column_a_phase_does_not_own_is_reported_for_every_phase() {
        use crate::store::FilePhase;
        let t = crate::unix_time;

        let (phase, dropped) = FilePhase::from_row(row(
            FileStatusKind::Processing,
            None,
            Some(t(1.0)),
            Some(t(2.0)),
            None,
        ));
        assert_eq!(
            phase,
            FilePhase::Processing {
                started_at: Some(t(1.0))
            }
        );
        assert_eq!(
            dropped.as_deref(),
            Some("the processing row held finished_at, which was not kept")
        );

        let (_, dropped) = FilePhase::from_row(row(
            FileStatusKind::Error,
            None,
            None,
            Some(t(2.0)),
            Some(t(3.0)),
        ));
        assert_eq!(
            dropped.as_deref(),
            Some("the error row held next_eligible_at, which was not kept")
        );

        let (_, dropped) = FilePhase::from_row(row(
            FileStatusKind::Interrupted,
            None,
            Some(t(1.0)),
            Some(t(2.0)),
            Some(t(3.0)),
        ));
        assert_eq!(
            dropped.as_deref(),
            Some("the interrupted row held finished_at, next_eligible_at, which were not kept")
        );
    }
}
